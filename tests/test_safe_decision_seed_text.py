"""text_filters.safe_decision_seed_text 单测（PLAN-010 阶段0 步骤3，真纯函数）。

断言口径：对 KB 六原型文本表层校验全过、保住语义（PLAN-009 记录的转换示例：
c4+e4 → 后翼和中心，b4-b5 → 后翼推进，d5/b5 → 中心或后翼）。
预期输出均先用探针对当前代码实测得出，不凭正则规则在脑中模拟推导
（正则链顺序敏感，凭空预测容易与实现细节脱节）。
"""
import json
import os

import pytest

from src.commentator.text_filters import safe_decision_seed_text
from src.commentator.validators import validate_puzzle_voiceover_surface

_KB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "structure_kb.json")


def test_pawn_advance_sequence_to_region_word():
    """b4-b5（推进序列）→ 后翼推进（取末格 file 定区域）。"""
    assert safe_decision_seed_text("b4-b5 兵推进") == "后翼推进 兵推进"


def test_pawn_combo_joined_by_and():
    """c4+e4（并列组合）→ 后翼和中心。"""
    assert safe_decision_seed_text("c4+e4 双兵锁住中心") == "后翼和中心 双兵锁住中心"


def test_pawn_alternative_joined_by_or():
    """d5/b5（择一）→ 中心或后翼。"""
    assert safe_decision_seed_text("寻求 d5/b5 解放") == "寻求 中心或后翼 解放"


def test_capture_notation_to_zone_capture_word():
    """d5xd4（吃子记号）→ 目标格区域 + 兑子。"""
    assert safe_decision_seed_text("d5xd4 消除孤兵") == "中心兑子 消除孤兵"


def test_bare_file_line_reference():
    """裸纵线字母「c 线」→ 区域词 + 这条线。"""
    assert safe_decision_seed_text("c 线兵未推进") == "后翼这条线兵未推进"


def test_bare_file_pawn_reference():
    """裸纵线字母「d 兵」→ 区域词 + 兵。"""
    assert safe_decision_seed_text("d 兵难推进") == "中心兵难推进"


def test_empty_string_returns_empty():
    assert safe_decision_seed_text("") == ""


def test_no_coordinates_passthrough_pure_chinese():
    """纯中文无坐标文本原样返回（不误伤）。"""
    text = "利用轻子灵活性换取长期施压优势"
    assert safe_decision_seed_text(text) == text


def test_kb_all_theory_and_mechanism_pass_surface_validator():
    """KB 六原型的 theory + 全部 plans[].mechanism 逐条清洗后，
    表层硬闸（validate_puzzle_voiceover_surface）全部通过——
    对齐 PLAN-009 实测「KB 全部 20 处文本表层校验 0 失败」的结论。
    """
    with open(_KB_PATH, encoding="utf-8") as f:
        kb = json.load(f)

    checked = 0
    failures = []
    for arch_id, entry in kb.items():
        theory = entry.get("theory", "")
        if theory:
            cleaned = safe_decision_seed_text(theory)
            ok, issues = validate_puzzle_voiceover_surface(cleaned)
            checked += 1
            if not ok:
                failures.append((arch_id, "theory", cleaned, issues))
        for plan in entry.get("plans", []):
            mech = plan.get("mechanism", "")
            if mech:
                cleaned = safe_decision_seed_text(mech)
                ok, issues = validate_puzzle_voiceover_surface(cleaned)
                checked += 1
                if not ok:
                    failures.append((arch_id, plan.get("name"), cleaned, issues))

    assert checked >= 14, f"KB 文本条目数异常低（{checked}），KB 结构可能变了"
    assert not failures, f"以下条目清洗后未过表层硬闸: {failures}"


def test_semantic_preservation_carlsbad_mechanism():
    """卡尔斯巴德少数派攻击 mechanism（实测样本，含 b4-b5 序列）保留区域语义。"""
    raw = "用 b4-b5 兵推进冲击对方 c 线兵，制造孤立/后退弱兵后长期施压"
    cleaned = safe_decision_seed_text(raw)
    assert "b4" not in cleaned and "b5" not in cleaned
    assert "后翼推进" in cleaned
    assert "后翼这条线" in cleaned
