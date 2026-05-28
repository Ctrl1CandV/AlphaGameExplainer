from src.stockfish_analyzer import get_solution
from src.common import Logger, resolve_path, GeneratedCommentary
from src.storyboard import compress, build
from src.commentator import generate_structured, generate
from src.llm_backend import release_backend
from src.tablebase import TablebaseSolver
from dotenv import load_dotenv
from src.parser import parse
import chess
import time
import os

load_dotenv()

def run(input_text: str) -> str:
    Logger.info("=" * 20 + "AlphaGameExplainer 开始运行" + "=" * 20)

    stockfish_path = resolve_path(os.getenv("STOCKFISH_PATH", "stockfish-windows-x86-64-avx2.exe"))
    syzygy_path = os.getenv("SYZYGY_PATH", "")
    gaviota_path = os.getenv("GAVIOTA_PATH", "")

    tablebase_solver = None
    if syzygy_path or gaviota_path:
        tablebase_solver = TablebaseSolver(
            syzygy_dir=syzygy_path,
            gaviota_dir=gaviota_path,
        )
        Logger.info(f"表库配置: Syzygy={syzygy_path or '未设置'}, Gaviota={gaviota_path or '未设置'}")

    Logger.info("[1/5] 解析对局...")
    game_data = parse(input_text)
    board = chess.Board(game_data.initial_fen)

    if not board.is_valid():
        Logger.error(f"非法初始局面: FEN不合法 (status={board.status()})，无法生成解说")
        return ""

    Logger.info("[2/5] 查询最优解法...")
    analyzed_moves = get_solution(board, stockfish_path, tablebase_solver, syzygy_path)
    if not analyzed_moves:
        Logger.warn("未能找到解法")
        return ""

    draw_error = _check_draw(board, analyzed_moves, tablebase_solver)
    if draw_error:
        Logger.error(draw_error)
        Logger.error("当前版本仅支持必胜残局解说，和棋局面暂不处理。")
        return ""

    Logger.info("[3/5] 节点压缩...")
    compressed = compress(board, analyzed_moves)

    Logger.info("[4/5] 构建叙事分镜...")
    storyboard = build(board, compressed)

    Logger.info("[5/5] 生成中文解说...")
    try:
        commentary = generate_structured(board, storyboard)
    except Exception as e:
        Logger.warn(f"结构化生成失败，回退纯文本: {e}")
        text = generate(board, storyboard)
        commentary = GeneratedCommentary(raw_text=text, fallback_used=True)

    if commentary.segments:
        Logger.info(f"  {len(commentary.segments)} 段 - pacing分布: " +
                    ", ".join(f"{p}={sum(1 for s in commentary.segments if s.pacing == p)}"
                              for p in ["slow", "normal", "fast", "pause_before", "pause_after"]
                              if any(s.pacing == p for s in commentary.segments)))

    print(commentary.raw_text)

    try:
        release_backend()
    except Exception:
        pass

    if tablebase_solver:
        try:
            tablebase_solver.close()
        except Exception:
            pass

    return commentary.raw_text


def _check_draw(board, analyzed_moves, tablebase_solver) -> str:
    if tablebase_solver is not None:
        try:
            tablebase_solver.open()
        except Exception:
            pass
        is_draw = tablebase_solver.is_draw(board)
        if is_draw is True:
            return "此局面为理论上的和棋（或超出50步规则无法兑现的必胜），无法生成必胜解说。"
        if is_draw is False:
            return ""

    temp = board.copy()
    for am in analyzed_moves:
        if temp.is_game_over():
            break
        temp.push(am.move)

    if not temp.is_game_over():
        return ""

    outcome = temp.outcome()
    if outcome is None:
        return ""
    if outcome.winner is None:
        return "该残局最终局面为逼和/和棋，无法生成必胜解说。"

    return ""