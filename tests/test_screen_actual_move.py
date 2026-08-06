"""decision_pipeline._screen_actual_move 单测（PLAN-010 阶段0 步骤4，0b 引擎依赖）。

同 test_assess_actual_move.py 的 mock 策略：不接真实 Stockfish，
mock 的是 `_screen_actual_move` 内部调用的 `assess_actual_move`
（局部 import，patch 源模块属性即可在调用时生效）。

断言口径：UCI 与 SAN 双格式都解析为同一手；开局名/乱码被拒；
未过评估筛返回 None；sf_path 只是透传，不关心其真实性。

FEN 与着法选择说明：改用 `d4d5`/SAN "d5"（悬兵推进），避免此局面下
`Rc1` 因两车都可达而产生的 AmbiguousMoveError（已用 python-chess 实测
`board.parse_san("d5") == chess.Move.from_uci("d4d5")` 且唯一）。
"""
from unittest.mock import patch

import chess

from src.pipeline.decision_pipeline import _screen_actual_move

# 悬兵局面：白走 d4d5（推进悬兵）合法且 SAN 无歧义
BOARD_FEN = "2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/1R2R1K1 w - - 1 21"


def _board():
    return chess.Board(BOARD_FEN)


def test_san_input_resolved_and_passes():
    """SAN 格式输入（人工填写常见形态），过评估筛后返回其 SAN。"""
    board = _board()
    with patch("src.solver.branch_explorer.assess_actual_move",
               return_value=(True, 0)) as mock_assess:
        result = _screen_actual_move(board, "d5", "fake_sf_path")
    assert result == "d5"
    # 断言真正调用了 assess_actual_move，且传入的 move 已被正确解析
    assert mock_assess.call_count == 1
    passed_board, passed_move, passed_sf = mock_assess.call_args[0]
    assert passed_move == board.parse_san("d5")
    assert passed_sf == "fake_sf_path"


def test_uci_input_resolved_and_passes():
    """UCI 格式输入（挖掘器 continuation 格式），同样解析并返回 SAN。"""
    board = _board()
    uci_move = chess.Move.from_uci("d4d5")  # 与 SAN "d5" 是同一手
    assert board.parse_san("d5") == uci_move  # 前置校验：确认两种写法指向同一手
    with patch("src.solver.branch_explorer.assess_actual_move",
               return_value=(True, 5)):
        result = _screen_actual_move(board, "d4d5", "fake_sf_path")
    assert result == "d5"  # 返回值统一是 SAN，不管输入格式


def test_opening_name_rejected_as_illegal_move():
    """开局名（如 'Sicilian Defense: ...'）不是合法 SAN/UCI，安全返回 None。

    这是阶段 9 记录的真实历史 bug 复现——runner 曾把 pick['opening'] 误传为
    provenance，5 个局面全部静默返回 None。本用例锁死这个已修复的行为。
    """
    board = _board()
    with patch("src.solver.branch_explorer.assess_actual_move") as mock_assess:
        result = _screen_actual_move(
            board, "Sicilian Defense: O'Kelly Variation", "fake_sf_path")
    assert result is None
    # 非法输入应在解析阶段就被拒绝，根本不应该走到引擎评估
    mock_assess.assert_not_called()


def test_garbage_input_rejected_safely():
    """乱码/空字符串输入不抛异常，安全返回 None。"""
    board = _board()
    with patch("src.solver.branch_explorer.assess_actual_move") as mock_assess:
        assert _screen_actual_move(board, "not a move at all!!", "fake_sf") is None
        assert _screen_actual_move(board, "", "fake_sf") is None
    mock_assess.assert_not_called()


def test_move_fails_evaluation_screen_returns_none():
    """着法能被正确解析，但未过评估筛（净损失过大）——返回 None，不注入。"""
    board = _board()
    with patch("src.solver.branch_explorer.assess_actual_move",
               return_value=(False, 612)):  # 对齐 branch_explorer 文档里的送后样本
        result = _screen_actual_move(board, "d5", "fake_sf_path")
    assert result is None


def test_legal_uci_but_not_actually_legal_in_position_rejected():
    """格式合法的 UCI，但在当前局面中不是合法着——安全拒绝，不误传给引擎。"""
    board = _board()
    # a8d8 在这个局面中不合法（historically 触发过 push() 断言错误的着）
    with patch("src.solver.branch_explorer.assess_actual_move") as mock_assess:
        result = _screen_actual_move(board, "a8d8", "fake_sf_path")
    assert result is None
    mock_assess.assert_not_called()
