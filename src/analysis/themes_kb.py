"""Puzzle 战术标签知识库访问层，集中封装 puzzle_themes.json 的加载、分类、查询。"""

from src.common import Logger
from typing import List, Tuple, Optional
import json
import os

_KB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "puzzle_themes.json",
)

# A 类：需深度讲解 → effective
_A_CLASS_BUCKETS = ("motifs", "advanced", "mate_patterns", "other", "position_eval")
# B 类：可选讲解（背景注入）→ auxiliary
_B_CLASS_BUCKETS = ("phase_sub",)

_THEMES_KB: dict = {}
_ALL_KEYS: set = set()


def load_kb() -> dict:
    """加载并缓存 puzzle_themes.json。校验不变量，失败抛出明确异常。"""
    global _THEMES_KB, _ALL_KEYS
    if _THEMES_KB:
        return _THEMES_KB

    with open(_KB_PATH, "r", encoding="utf-8") as f:
        _THEMES_KB = json.load(f)

    # 校验桶
    expected_buckets = {"motifs", "advanced", "mate_patterns", "phase_sub", "other", "position_eval"}
    actual_buckets = set(_THEMES_KB.keys())
    missing = expected_buckets - actual_buckets
    if missing:
        raise RuntimeError(f"puzzle_themes.json 缺少顶层桶: {missing}")

    # 收集所有标签
    for bucket_name, entries in _THEMES_KB.items():
        for entry in entries:
            key = entry.get("key", "")
            if key:
                _ALL_KEYS.add(key)

    # 校验每个标签含 4 个新字段
    required_fields = ("prerequisite", "common_mistakes", "related_themes", "difficulty_level")
    valid_levels = {"basic", "intermediate", "advanced"}
    for bucket_name, entries in _THEMES_KB.items():
        for entry in entries:
            key = entry.get("key", "?")
            for field in required_fields:
                if field not in entry:
                    raise RuntimeError(f"标签 '{key}' 缺少字段 '{field}'")
            if entry["difficulty_level"] not in valid_levels:
                raise RuntimeError(
                    f"标签 '{key}' 的 difficulty_level='{entry['difficulty_level']}' 不合法，"
                    f"应为 {valid_levels} 之一")

    # 校验 related_themes 无悬空引用
    for bucket_name, entries in _THEMES_KB.items():
        for entry in entries:
            key = entry.get("key", "?")
            for ref in entry.get("related_themes", []):
                if ref not in _ALL_KEYS:
                    raise RuntimeError(
                        f"标签 '{key}' 的 related_themes 引用了不存在的 '{ref}'")

    Logger.info(f"战术标签知识库加载完成: {len(_ALL_KEYS)} 个标签")
    return _THEMES_KB


def get_theme(key: str) -> Optional[dict]:
    """按 key 取单个标签定义（跨所有桶查找），未命中返回 None。"""
    kb = load_kb()
    for entries in kb.values():
        for entry in entries:
            if entry.get("key") == key:
                return entry
    return None


def filter_themes(raw_themes: List[str]) -> Tuple[List[str], List[str]]:
    """把原始 Themes 拆成 (effective_A类有效标签, auxiliary_B类辅助标签)。
    不在 KB 中的标签（阶段/长度/来源等 C 类）直接丢弃并记录。
    保持 raw_themes 中的出现顺序——effective[0] 即主标签。
    """
    kb = load_kb()
    a_keys = set()
    for bucket in _A_CLASS_BUCKETS:
        for entry in kb.get(bucket, []):
            a_keys.add(entry["key"])
    b_keys = set()
    for bucket in _B_CLASS_BUCKETS:
        for entry in kb.get(bucket, []):
            b_keys.add(entry["key"])

    effective = []
    auxiliary = []
    discarded = []
    for t in raw_themes:
        t = t.strip()
        if not t:
            continue
        if t in a_keys:
            if t not in effective:
                effective.append(t)
        elif t in b_keys:
            if t not in auxiliary:
                auxiliary.append(t)
        else:
            if t not in discarded:
                discarded.append(t)

    if discarded:
        Logger.info(f"已丢弃 C 类标签（不在知识库中）: {discarded}")
    return effective, auxiliary


# 讲解核心选取的桶分层（tier 越小越优先作为讲解核心）：
#   tier 0 战术机理：motifs / advanced —— fork/pin/sacrifice 等，最值得讲透
#   tier 1 收束目标：mate_patterns —— 杀型，告诉观众结局形态
#   tier 2 评估背景：position_eval / other —— crushing/advantage 等，只描述
#                    优势程度，讲不出机理，仅在没有更优标签时兜底
_CORE_BUCKETS = ("motifs", "advanced")
_OUTCOME_BUCKETS = ("mate_patterns",)

# 根因3修复：叙事基调标签集合。这些标签描述的是"谁占优、局势基调是什么"，
# 与 fork/pin/sacrifice 等"用了什么手段"的机理标签是两个正交维度，不应该
# 混在同一个 tier 系统里排序——否则会出现 defensiveMove(曾被误放进机理桶)
# 压过 crushing(评估桶) 的语义冲突：Lichess 常给同一题同时打上
# ["crushing", "defensiveMove", ...]（描述整条 PV 里出现过的各种元素，不
# 代表单一叙事主导），若靠桶优先级选核心标签，会把"碾压式主动进攻"误判成
# "防守化解危机"，两者叙事方向完全相反。
# 修复方案：defensiveMove 与 position_eval 桶的 crushing/advantage/equality
# 一起，单独抽出来做"叙事基调"判断，不参与 select_core_theme 的机理选择；
# select_core_theme 排除这组标签后只在纯机理/杀型标签中选教学核心。
_STANCE_KEYS = frozenset({"crushing", "advantage", "equality", "defensiveMove"})
# 基调标签内部优先级：进攻性评估标签（crushing/advantage，均表示解题方在
# 赢子/占优）必须排在 defensiveMove 之前——Lichess 常把"碾压/占优"题同时打上
# defensiveMove（因为整条 PV 里对方某步有过精确防守），若 defensiveMove 抢先，
# 会把明明是解题方主动进攻的题误判成"防守化解危机"（见 004Ys）。defensiveMove
# 只在没有任何进攻性评估标签时才主导基调（此时才是真正的防守型题）；equality
# 兜底（防守成和）。storyboard 侧还有子力事实兜底核验作为第二道保险。
_STANCE_PRIORITY = ("crushing", "advantage", "defensiveMove", "equality")

_KEY_TO_BUCKET: dict = {}


def _key_to_bucket() -> dict:
    """构建并缓存 标签 key → 所属桶名 的映射。"""
    global _KEY_TO_BUCKET
    if _KEY_TO_BUCKET:
        return _KEY_TO_BUCKET
    kb = load_kb()
    for bucket, entries in kb.items():
        for entry in entries:
            k = entry.get("key")
            if k:
                _KEY_TO_BUCKET[k] = bucket
    return _KEY_TO_BUCKET


def _theme_tier(key: str) -> int:
    """标签讲解优先级 tier：0 机理 < 1 杀型 < 2 评估/其它。

    叙事基调标签（_STANCE_KEYS）永远归为 tier 2，不参与机理优先级竞争——
    即使 defensiveMove 物理上仍登记在 advanced 桶里，也不再享有 tier 0
    优先级，避免压过真正的机理标签或制造攻守错判。
    """
    if key in _STANCE_KEYS:
        return 2
    bucket = _key_to_bucket().get(key, "")
    if bucket in _CORE_BUCKETS:
        return 0
    if bucket in _OUTCOME_BUCKETS:
        return 1
    return 2


def select_core_theme(effective: List[str]) -> str:
    """从 effective 选最适合作为讲解核心的标签（纯"机理/杀型"教学标签）。

    规则：tier 升序（机理 > 杀型 > 评估兜底），tier 相同时保持 lichess 原顺序。
    这样 ['crushing', 'sacrifice'] 会选中 sacrifice（机理）而非 crushing（评估）。
    叙事基调标签（crushing/advantage/equality/defensiveMove）已被 _theme_tier
    统一降到 tier 2，只在没有任何机理/杀型标签时才会被选为核心（作为兜底）。
    空列表返回 ''。
    """
    if not effective:
        return ""
    return min(effective, key=lambda k: (_theme_tier(k), effective.index(k)))


def select_narrative_stance(effective: List[str]) -> str:
    """从 effective 中选出决定"叙事基调"的标签，与核心教学标签正交、独立选择。

    只在 _STANCE_KEYS（crushing/advantage/equality/defensiveMove）范围内挑选，
    按 _STANCE_PRIORITY 的严重程度顺序取共存标签中优先级最高的一个；
    没有任何基调标签时返回 ''（由调用方决定默认基调）。
    """
    present = [k for k in effective if k in _STANCE_KEYS]
    if not present:
        return ""
    for k in _STANCE_PRIORITY:
        if k in present:
            return k
    return present[0]


def related_intersection(core_key: str, others: List[str]) -> List[str]:
    """与核心标签存在联动关系的次要标签（按 others 顺序）。

    related_themes 是单向声明，故做双向判定：core 指向 other，或 other 指向 core，
    任一成立即视为联动。例如 sacrifice 未声明 crushing，但 crushing 声明了
    sacrifice，二者仍应识别为联动。
    """
    core = get_theme(core_key)
    if not core:
        return []
    core_related = set(core.get("related_themes", []))
    result = []
    for k in others:
        if k in core_related:
            result.append(k)
            continue
        other = get_theme(k)
        if other and core_key in other.get("related_themes", []):
            result.append(k)
    return result


def get_theme_definitions_text(themes: List[str], include_en: bool = True) -> str:
    """将标签列表转为可注入 prompt 的定义文本块。

    include_en=False 时不输出英文标签名，避免英文混入小模型的中文口播输出。
    """
    lines = []
    for key in themes:
        t = get_theme(key)
        if t is None:
            continue
        if include_en:
            lines.append(f"【{t['cn']}】（{t['en']}）")
        else:
            lines.append(f"【{t['cn']}】")
        lines.append(f"  定义: {t['definition']}")
        if t.get("prerequisite"):
            lines.append(f"  前提: {t['prerequisite']}")
        if t.get("recognition"):
            lines.append(f"  识别: {t['recognition']}")
        if t.get("key_move_signal"):
            lines.append(f"  关键手信号: {t['key_move_signal']}")
        if t.get("teaching_focus"):
            lines.append(f"  教学重点: {t['teaching_focus']}")
        if t.get("typical_consequence"):
            lines.append(f"  典型后果: {t['typical_consequence']}")
        if t.get("defense_reference"):
            lines.append(f"  防守思路: {t['defense_reference']}")
        if t.get("common_mistakes"):
            lines.append(f"  常见错误: {'；'.join(t['common_mistakes'])}")
        lines.append("")
    return "\n".join(lines)
