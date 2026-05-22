from src.stockfish_analyzer import get_solution
from src.common import Logger, resolve_path
from src.storyboard import compress, build
from src.commentator import generate
from dotenv import load_dotenv
from src.parser import parse
import chess
import time
import os

load_dotenv()

def run(input_text: str) -> str:
    """ Tablebase 最优解 → SF 标注压缩 → 知识库分镜 → LLM 解说 → 视频合成 """
    Logger.info("=" * 20 + "AlphaGameExplainer 开始运行" + "=" * 20)

    stockfish_path = resolve_path(os.getenv("STOCKFISH_PATH", "stockfish-windows-x86-64-avx2.exe"))
    
    Logger.info("[1/5] 解析对局...")
    game_data = parse(input_text)
    board = chess.Board(game_data.initial_fen)

    Logger.info("[2/5] 查询最优解法...")
    analyzed_moves = get_solution(board, stockfish_path)
    if not analyzed_moves:
        Logger.warn("未能找到解法")
        return ""

    Logger.info("[3/5] SF标注 & 节点压缩...")
    compressed = compress(board, analyzed_moves)

    Logger.info("[4/5] 构建叙事分镜...")
    storyboard = build(board, compressed)

    Logger.info("[5/5] 生成中文解说...")
    commentary = generate(board, storyboard)
    print(commentary)