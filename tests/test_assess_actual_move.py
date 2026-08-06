"""branch_explorer.assess_actual_move 单测（PLAN-010 阶段0 步骤4，引擎依赖函数）。

**不接真实 Stockfish**：mock `_open_engine`，返回一个假引擎，其 `.analyse()`
按调用参数（有无 root_moves）返回预设 cp，模拟 python-chess 的 info dict
形状（`{"score": PovScore-like}`，`score.relative.score()/.mate()`）。

断言口径迁自 PLAN-009 实施记录附录B：好着 0~20cp 通过、送后 612cp 拦下、
非法着失败安全、边界值（等于阈值）判定。
"""
from unittest.mock import patch

import chess

from src.solver.branch_explorer import DEFAULT_ACTUAL_LOSS_CP, assess_actual_move


class _FakeRelativeScore:
    """模拟 python-chess Score 对象：只需 .mate() 和 .score() 两个方法。"""

    def __init__(self, cp):
        self._cp = cp

    def mate(self):
        return None

    def score(self):
        return self._cp


class _FakePovScore:
    """模拟 python-chess PovScore：只需 .relative 属性。"""

    def __init__(self, cp):
        self.relative = _FakeRelativeScore(cp)


class _FakeEngine:
    """假引擎：按是否传 root_moves 区分"最优着评估"与"指定着评估"。

    第一次 analyse()（无 root_moves）返回 best_cp；
    第二次 analyse()（有 root_moves=[move]）返回 move_cp。
    与 assess_actual_move 的调用顺序（先查最优、再查该着）严格对应。
    """

    def __init__(self, best_cp, move_cp):
        self.best_cp = best_cp
        self.move_cp = move_cp
        self.analyse_calls = []

    def analyse(self, board, limit, root_moves=None):
        self.analyse_calls.append(root_moves)
        cp = self.move_cp if root_moves is not None else self.best_cp
        return {"score": _FakePovScore(cp)}

    def quit(self):
        pass


def _board():
    # 任意合法中局局面，move 用其中一个合法着即可（cp 由 mock 控制，
    # 不依赖真实局面强弱）
    return chess.Board(
        "2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/1R2R1K1 w - - 1 21")


def test_good_move_within_threshold_passes():
    """好着：与最优着差距在阈值内（0~20cp），通过。"""
    board = _board()
    move = chess.Move.from_uci("d4d5")  # 悬兵推进，合法着
    fake = _FakeEngine(best_cp=140, move_cp=120)  # 净损失 20cp
    with patch("src.solver.branch_explorer._open_engine", return_value=fake):
        passed, loss = assess_actual_move(board, move, "unused_sf_path")
    assert passed is True
    assert loss == 20


def test_blunder_large_loss_rejected():
    """送后类大损失（612cp）：拦下。"""
    board = _board()
    move = chess.Move.from_uci("d4d5")
    fake = _FakeEngine(best_cp=140, move_cp=140 - 612)
    with patch("src.solver.branch_explorer._open_engine", return_value=fake):
        passed, loss = assess_actual_move(board, move, "unused_sf_path")
    assert passed is False
    assert loss == 612


def test_illegal_move_fails_safe_without_opening_engine():
    """非法着：直接返回 (False, None)，不启动引擎（失败安全 + 省资源）。"""
    board = _board()
    illegal = chess.Move.from_uci("a1a2")  # a1 车走到 a2（无子/非法目标）
    with patch("src.solver.branch_explorer._open_engine") as mock_open:
        passed, loss = assess_actual_move(board, illegal, "unused_sf_path")
    assert (passed, loss) == (False, None)
    mock_open.assert_not_called()


def test_loss_exactly_at_threshold_passes():
    """边界值：净损失恰好等于 max_loss_cp（默认 30），应通过（<=，非 <）。"""
    board = _board()
    move = chess.Move.from_uci("d4d5")
    fake = _FakeEngine(best_cp=100, move_cp=100 - DEFAULT_ACTUAL_LOSS_CP)
    with patch("src.solver.branch_explorer._open_engine", return_value=fake):
        passed, loss = assess_actual_move(board, move, "unused_sf_path")
    assert loss == DEFAULT_ACTUAL_LOSS_CP
    assert passed is True


def test_loss_one_above_threshold_fails():
    """边界值：净损失比阈值多 1cp，应拒绝。"""
    board = _board()
    move = chess.Move.from_uci("d4d5")
    fake = _FakeEngine(best_cp=100, move_cp=100 - DEFAULT_ACTUAL_LOSS_CP - 1)
    with patch("src.solver.branch_explorer._open_engine", return_value=fake):
        passed, loss = assess_actual_move(board, move, "unused_sf_path")
    assert loss == DEFAULT_ACTUAL_LOSS_CP + 1
    assert passed is False


def test_move_better_than_best_clamped_to_zero_loss():
    """该着 cp 反超"最优着"评估（引擎评估非严格单调、mock 场景）：
    loss 用 max(0, best-move) 防止出现负损失。"""
    board = _board()
    move = chess.Move.from_uci("d4d5")
    fake = _FakeEngine(best_cp=100, move_cp=150)  # move 比 best 还高
    with patch("src.solver.branch_explorer._open_engine", return_value=fake):
        passed, loss = assess_actual_move(board, move, "unused_sf_path")
    assert loss == 0
    assert passed is True


def test_score_none_returns_failure_safe():
    """引擎返回空 score（如异常/未完成分析）：(False, None) 失败安全。"""
    board = _board()
    move = chess.Move.from_uci("d4d5")

    class _NoScoreEngine:
        def analyse(self, board, limit, root_moves=None):
            return {"score": None}

        def quit(self):
            pass

    with patch("src.solver.branch_explorer._open_engine",
               return_value=_NoScoreEngine()):
        passed, loss = assess_actual_move(board, move, "unused_sf_path")
    assert (passed, loss) == (False, None)


def test_engine_analyse_raises_exception_fails_safe():
    """analyse() 抛异常（真实场景：引擎进程崩溃/超时）：不传播异常，失败安全。"""
    board = _board()
    move = chess.Move.from_uci("d4d5")

    class _CrashingEngine:
        def analyse(self, board, limit, root_moves=None):
            raise RuntimeError("engine crashed")

        def quit(self):
            pass

    with patch("src.solver.branch_explorer._open_engine",
               return_value=_CrashingEngine()):
        passed, loss = assess_actual_move(board, move, "unused_sf_path")
    assert (passed, loss) == (False, None)
