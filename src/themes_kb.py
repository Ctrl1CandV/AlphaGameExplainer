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


def primary_theme(effective: List[str]) -> str:
    """主标签 = effective[0]，空列表返回 ''。"""
    return effective[0] if effective else ""


def get_theme_definitions_text(themes: List[str]) -> str:
    """将标签列表转为可注入 prompt 的定义文本块。"""
    lines = []
    for key in themes:
        t = get_theme(key)
        if t is None:
            continue
        lines.append(f"【{t['cn']}】（{t['en']}）")
        lines.append(f"  定义: {t['definition']}")
        if t.get("prerequisite"):
            lines.append(f"  前提: {t['prerequisite']}")
        if t.get("recognition"):
            lines.append(f"  识别: {t['recognition']}")
        if t.get("teaching_focus"):
            lines.append(f"  教学重点: {t['teaching_focus']}")
        if t.get("common_mistakes"):
            lines.append(f"  常见错误: {'；'.join(t['common_mistakes'])}")
        lines.append("")
    return "\n".join(lines)
