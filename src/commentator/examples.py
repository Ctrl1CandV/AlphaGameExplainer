"""经审核解说范例的精确匹配与实验开关。"""

import json
import os


_EXAMPLES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "commentary_examples.json",
)
_EXAMPLES = None


def _load_examples() -> list:
    global _EXAMPLES
    if _EXAMPLES is None:
        try:
            with open(_EXAMPLES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            _EXAMPLES = [item for item in data
                         if isinstance(item, dict) and item.get("reviewed") is True]
        except (OSError, ValueError, TypeError):
            _EXAMPLES = []
    return _EXAMPLES


def get_commentary_example(pipeline: str, category_or_theme: str,
                           narrative_role: str) -> str:
    """按 pipeline + category/theme + role 精确取一个已审核范例。"""
    for item in _load_examples():
        if (item.get("pipeline") == pipeline
                and item.get("category_or_theme") == category_or_theme
                and item.get("narrative_role") == narrative_role):
            return item.get("text", "")
    return ""


def commentary_example_mode() -> str:
    """实验分组：none / fixed / matched；默认关闭，待消融通过后再启用。"""
    mode = os.getenv("COMMENTARY_EXAMPLE_MODE", "none").strip().lower()
    return mode if mode in ("none", "fixed", "matched") else "none"
