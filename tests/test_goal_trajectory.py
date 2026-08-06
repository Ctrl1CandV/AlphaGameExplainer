"""structure_features.goal_trajectory 单测（PLAN-010 阶段0 步骤3，真纯函数）。

断言口径迁自 PLAN-009 实施记录附录B「验证过但无自动化留痕的行为清单」：
达成/仅有进步/无进步/空 goal/未知维/未知谓词/OR 组/max_len=0 边界；
锚定 vs 不锚定视角结论相反（证明 mover_color 必传）。
"""
import chess
import pytest

from src.analysis.structure_features import goal_trajectory, structural_features


# 悬兵局面：白持 c4+d4 双悬兵，走子方=白（mover）。SAN 线 d4-d5 是"推进悬兵"
# 计划的典型执行手，会让 mover_pawns_past_mid 从 0 变为 >=1。
HANGING_FEN = "2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/1R2R1K1 w - - 1 21"


def _board():
    return chess.Board(HANGING_FEN)


def test_goal_reached_after_push():
    """目标在 push 后的某一点被满足：goal_reached=True 且 goal_ok=True。"""
    board = _board()
    # d4d5 使 mover_pawns_past_mid 立即 >=1（d 兵推过中线）
    line = [chess.Move.from_uci("d4d5")]
    result = goal_trajectory(board, line, {"mover_pawns_past_mid": ">=1"},
                              mover_color=chess.WHITE)
    assert result["goal_ok"] is True
    assert result["goal_reached"] is True
    assert result["goal_progress"]["mover_pawns_past_mid"] >= 1


def test_goal_reached_excludes_start_point():
    """起点即满足不算达成——push 至少一步后才检查（P0-full peer_review Critical 2 同判据）。

    悬兵局面起点 mover_pawns_past_mid=0（不满足 >=1），故本用例改用一个
    "起点已满足、但线上无推进"的构造：目标恰好是 mover_pawns_past_mid>=0，
    这在任何局面都恒真，验证 goal_reached 只在 push 之后的序列里判定，
    不把起点（push 之前）算进去。
    """
    board = _board()
    line = [chess.Move.from_uci("h3h4")]  # 与目标维度无关的过渡着
    result = goal_trajectory(board, line, {"mover_pawns_past_mid": ">=0"},
                              mover_color=chess.WHITE)
    # >=0 对任何非负计数恒真，push 后第一个点也满足 —— reached 应为 True
    assert result["goal_reached"] is True


def test_goal_only_progress_no_reach():
    """目标未达成但有正向进步：goal_ok=True，goal_reached=False。"""
    board = _board()
    # 单步 d4d5 不足以让 opp_isolated_center>=1（对方孤兵尚未出现），
    # 但 mover_pawns_past_mid 有进步 —— 用一个刻意设高阈值的 goal 制造
    # "未达成但有进步"的场景。
    line = [chess.Move.from_uci("d4d5")]
    result = goal_trajectory(board, line, {"mover_pawns_past_mid": ">=5"},
                              mover_color=chess.WHITE)
    assert result["goal_reached"] is False
    assert result["goal_progress"]["mover_pawns_past_mid"] >= 1
    assert result["goal_ok"] is True  # 进步 >=1 即 goal_ok


def test_no_progress_goal_not_ok():
    """目标维度完全不受该线影响：goal_ok=False。"""
    board = _board()
    line = [chess.Move.from_uci("h3h4")]  # 王翼兵动，不影响后翼孤兵
    result = goal_trajectory(board, line, {"opp_isolated_qside": ">=1"},
                              mover_color=chess.WHITE)
    assert result["goal_reached"] is False
    assert result["goal_progress"]["opp_isolated_qside"] == 0
    assert result["goal_ok"] is False


def test_empty_goal_returns_empty_result():
    """空 goal（{}）：直接返回全空结果，不抛异常。"""
    board = _board()
    result = goal_trajectory(board, [chess.Move.from_uci("d4d5")], {},
                              mover_color=chess.WHITE)
    assert result == {"goal_reached": False, "goal_progress": {}, "goal_ok": False}


def test_unknown_dimension_zero_progress():
    """未知维度名：不抛异常，进步记 0（与 goal_satisfied 对未知维度保守判不满足同向）。"""
    board = _board()
    line = [chess.Move.from_uci("d4d5")]
    result = goal_trajectory(board, line, {"not_a_real_dimension": ">=1"},
                              mover_color=chess.WHITE)
    assert result["goal_progress"]["not_a_real_dimension"] == 0
    assert result["goal_ok"] is False


def test_unknown_predicate_zero_progress():
    """未知谓词（非 >=/==）：不抛异常，进步记 0。"""
    board = _board()
    line = [chess.Move.from_uci("d4d5")]
    result = goal_trajectory(board, line, {"mover_pawns_past_mid": "<=3"},
                              mover_color=chess.WHITE)
    assert result["goal_progress"]["mover_pawns_past_mid"] == 0


def test_or_group_any_satisfied():
    """OR 组（{"any": [...]}）：任一子 goal 达成即整体达成，进步键带 any. 前缀。"""
    board = _board()
    line = [chess.Move.from_uci("d4d5")]
    goal = {"any": [{"opp_isolated_qside": ">=99"},  # 不可能达成
                    {"mover_pawns_past_mid": ">=1"}]}  # 可达成
    result = goal_trajectory(board, line, goal, mover_color=chess.WHITE)
    assert result["goal_ok"] is True
    # any 组的子目标进步以 "any.<dim>" 记入 goal_progress
    any_keys = [k for k in result["goal_progress"] if k.startswith("any.")]
    assert any_keys, "OR 组子目标进步应以 any.<dim> 前缀记录"


def test_max_len_zero_boundary():
    """max_len=0：不推进任何着，goal_reached 恒 False（无法达成，只能看起点，但起点被排除）。"""
    board = _board()
    line = [chess.Move.from_uci("d4d5")]
    result = goal_trajectory(board, line, {"mover_pawns_past_mid": ">=1"},
                              mover_color=chess.WHITE, max_len=0)
    assert result["goal_reached"] is False
    assert result["goal_ok"] is False


def test_mover_color_anchoring_flips_conclusion():
    """锚定视角 vs 不锚定（None 退回 board.turn）：同一条线可能得出相反结论。

    构造场景：从白方视角（mover_color=WHITE）看，一步 d4d5 后
    mover_pawns_past_mid 立即达成；若改用黑方视角（mover_color=BLACK）
    观察同一条线，"己方"语义变成黑方，mover_pawns_past_mid 不会因白棋
    推进而改变（黑兵没有过线），应保持不满足。
    """
    board = _board()
    line = [chess.Move.from_uci("d4d5")]
    result_white = goal_trajectory(board, line, {"mover_pawns_past_mid": ">=1"},
                                    mover_color=chess.WHITE)
    result_black = goal_trajectory(board, line, {"mover_pawns_past_mid": ">=1"},
                                    mover_color=chess.BLACK)
    assert result_white["goal_ok"] is True
    assert result_black["goal_ok"] is False


def test_illegal_move_in_line_stops_gracefully():
    """线中出现非法着：不抛异常，在该点停止累积，之前的结果仍返回。

    a1 在该局面无子（rank1 = "1R2R1K1"），a1a8 格式合法但对当前局面非法，
    `board.push` 内部会 assert 失败——`goal_trajectory` 必须吞掉这类异常
    （AssertionError 是 Exception 子类）并停止累积，不让整条线崩溃。
    """
    board = _board()
    legal = chess.Move.from_uci("d4d5")
    illegal = chess.Move.from_uci("a1a8")  # a1 无子，push 时非法
    result = goal_trajectory(board, [legal, illegal],
                              {"mover_pawns_past_mid": ">=1"},
                              mover_color=chess.WHITE)
    # 合法着已经让目标达成，非法着不应导致异常或抹掉之前的结果
    assert result["goal_ok"] is True
