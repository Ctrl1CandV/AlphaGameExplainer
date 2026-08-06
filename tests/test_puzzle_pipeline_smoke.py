"""puzzle 管线端到端回归（PLAN-010 阶段0 步骤5，HANDOFF T2）。

用固化的 tests/fixtures/backrank_puzzle.json（底线杀，已用引擎验证 #+1）
与 fork_puzzle.json（马叉后车，已验证攻击 d8+h8 双目标）驱动完整 4 步文本
管线，断言关键手定位/解说段数/字数量级。

标记 slow：真实调 LLM API，日常回归用 `pytest -m "not slow"` 跳过。
"""
import os

import pytest

from src.pipeline.puzzle_pipeline import _run_puzzle_pipeline

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _read_fixture(name: str) -> str:
    with open(os.path.join(FIXTURES_DIR, name), encoding="utf-8") as f:
        return f.read().strip()


@pytest.mark.slow
def test_backrank_puzzle_full_pipeline_produces_valid_commentary():
    """底线杀 puzzle（Qd8#）：4 步管线跑通，关键手定位正确。

    该 puzzle 的 fen 是预备着前局面，moves[0]=a7a5（黑方预备步），
    moves[1]=d1d8（白方后走底线将杀，是本题唯一正解）。
    """
    input_text = _read_fixture("backrank_puzzle.json")
    result = _run_puzzle_pipeline(input_text)
    assert result is not None, "底线杀已用引擎验证为真实将杀，管线不应返回 None"

    commentary, board, puzzle, storyboard, prelude_san, pre_fen, prelude_narration = result

    # 预备着应被正确抽取（Lichess 约定 moves[0] 为预备步）
    assert prelude_san == "a5"
    # 正解步数：原 moves 2 步，减去预备步后应剩 1 步（Qd8#）
    assert len(puzzle.moves) == 1

    assert not getattr(commentary, "aborted", False)
    assert commentary.raw_text and len(commentary.raw_text) > 0
    assert len(commentary.segments) >= 1


@pytest.mark.slow
def test_fork_puzzle_full_pipeline_produces_valid_commentary():
    """马叉后车 puzzle（Nf7 攻击 d8 后 + h8 车）：无预备步单步题。"""
    input_text = _read_fixture("fork_puzzle.json")
    result = _run_puzzle_pipeline(input_text)
    assert result is not None

    commentary, board, puzzle, storyboard, prelude_san, pre_fen, prelude_narration = result

    # 单步题（moves 只有 1 步）不应拆出预备步
    assert prelude_san == ""
    assert len(puzzle.moves) == 1

    assert not getattr(commentary, "aborted", False)
    assert commentary.raw_text and len(commentary.raw_text) > 0


@pytest.mark.slow
def test_backrank_puzzle_text_mode_smoke():
    """`run_puzzle()`（--text 等价）不抛异常。"""
    from src.pipeline.puzzle_pipeline import run_puzzle

    input_text = _read_fixture("backrank_puzzle.json")
    run_puzzle(input_text)
