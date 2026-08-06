"""endgame 管线端到端回归（PLAN-010 阶段0 步骤5，HANDOFF T2）。

用固化的 tests/fixtures/krk.fen（KRvK 表库残局）驱动完整 5 步文本管线，
断言表库步数/节点压缩/解说段数/字数量级——防止 TTS/composer/text_filters
等共享模块的改动悄悄破坏老管线行为。

标记 slow：真实调 Stockfish + Syzygy 表库 + LLM API，日常回归用
`pytest -m "not slow"` 跳过；CI/手动验证时单独 `pytest -m slow` 跑。
"""
import os

import chess
import pytest

from src.pipeline.endgame_pipeline import _run_pipeline

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _read_fixture(name: str) -> str:
    with open(os.path.join(FIXTURES_DIR, name), encoding="utf-8") as f:
        return f.read().strip()


@pytest.mark.slow
def test_krk_endgame_full_pipeline_produces_valid_commentary():
    """KRvK 表库残局：5 步管线跑通，产出结构合理的解说。

    断言点对齐 PLAN-010 步骤5 要求：表库步数、节点数、解说段数、字数量级。
    不断言具体文案内容（LLM 输出非确定性），只断言结构性不变量。
    """
    fen = _read_fixture("krk.fen")
    result = _run_pipeline(fen)
    assert result is not None, "KRvK 是已知可解的表库残局，管线不应返回 None"

    commentary, board, game_data, analyzed_moves, storyboard, compressed, winner_color = result

    # 表库步数：KRvK 是简单必胜残局，解法应在个位数到十余步量级（非 0、非离谱大）
    assert 1 <= len(analyzed_moves) <= 30, \
        f"KRvK 表库解法步数异常: {len(analyzed_moves)}"

    # 节点压缩：压缩后节点数不应超过原始步数（压缩语义是合并，不应增加）
    assert 1 <= len(compressed) <= len(analyzed_moves)

    # 胜方应可判定（KRvK 白方必胜，走子方=白）
    assert winner_color == chess.WHITE

    # 解说完整性：非 aborted，含正文，段数与压缩节点数量级相当
    assert not getattr(commentary, "aborted", False)
    assert commentary.raw_text and len(commentary.raw_text) > 0
    assert len(commentary.segments) >= 1
    # 段数不应远超节点数（允许因分块/开场总结略有浮动，但不应是数量级差异）
    assert len(commentary.segments) <= len(compressed) + 2


@pytest.mark.slow
def test_krk_endgame_text_mode_smoke():
    """`run()`（--text 等价）不抛异常，能跑到底（老管线 print 路径）。"""
    from src.pipeline.endgame_pipeline import run

    fen = _read_fixture("krk.fen")
    # run() 只 print，不返回值——这里只验证不崩溃即视为通过
    run(fen)
