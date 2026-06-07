"""Puzzle 战术标签知识库访问层，集中封装 puzzle_themes.json 的加载、分类、查询。"""

from src.common import Logger
from typing import List, Tuple, Optional
import json
import os

_KB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
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
    """标签讲解优先级 tier：0 机理 < 1 杀型 < 2 评估/其它。"""
    bucket = _key_to_bucket().get(key, "")
    if bucket in _CORE_BUCKETS:
        return 0
    if bucket in _OUTCOME_BUCKETS:
        return 1
    return 2


def select_core_theme(effective: List[str]) -> str:
    """从 effective 选最适合作为讲解核心的标签。

    规则：tier 升序（机理 > 杀型 > 评估兜底），tier 相同时保持 lichess 原顺序。
    这样 ['crushing', 'sacrifice'] 会选中 sacrifice（机理）而非 crushing（评估）。
    空列表返回 ''。
    """
    if not effective:
        return ""
    return min(effective, key=lambda k: (_theme_tier(k), effective.index(k)))


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
