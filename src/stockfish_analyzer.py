from src.common import Logger
from typing import List, Optional
import chess.engine
import urllib.parse
import requests
import chess
import time
import re

CDB_URL = "http://www.chessdb.cn/cdb.php"
SEARCH_TIME = 30.0
MATE_TIME = 8.0
CDB_CACHE: dict = {}

MOVE_RE = re.compile(r"move:(\S+),score:(-?\d+),rank:(\d+),note:(.*)")


def _query_cdb(fen: str) -> Optional[str]:
    if fen in CDB_CACHE:
        return CDB_CACHE[fen]
    url = f"{CDB_URL}?action=queryall&board={urllib.parse.quote(fen)}&egtbmetric=dtm"
    try:
        resp = requests.get(url, timeout=15)
        body = resp.text
        if body in ("unknown", "nobestmove", "invalid board", ""):
            CDB_CACHE[fen] = None
            return None
        CDB_CACHE[fen] = body
        time.sleep(0.3)
        return body
    except Exception:
        CDB_CACHE[fen] = None
        return None


def _pick_best(cdb_body: str) -> Optional[str]:
    best_uci = None
    best_score = -9999999
    best_rank = 0
    for part in cdb_body.split("|"):
        m = MOVE_RE.match(part)
        if not m:
            continue
        uci, score, rank = m.group(1), int(m.group(2)), int(m.group(3))
        if score > best_score or (score == best_score and rank > best_rank):
            best_uci, best_score, best_rank = uci, score, rank
    return best_uci


def _sf_solve(board: chess.Board, stockfish_path: str) -> List[chess.Move]:
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        info = engine.analyse(board, chess.engine.Limit(mate=30, time=MATE_TIME))
        pv = info.get("pv", [])
        score = info.get("score")
        mate = score.relative.mate() if score else None

        if mate is not None and mate > 0 and len(pv) >= 1:
            Logger.info(f"SF mate搜索: M{mate}, {len(pv)} 着/{MATE_TIME:.0f}s")
            return list(pv)

        Logger.info(f"SF 深度搜索 ({SEARCH_TIME:.0f}s)...")
        info = engine.analyse(board, chess.engine.Limit(time=SEARCH_TIME))
        pv = info.get("pv", [])
        if pv:
            return list(pv)
        return []
    finally:
        engine.quit()


def get_solution(board: chess.Board, stockfish_path: str) -> List[chess.Move]:
    piece_count = len(board.piece_map())
    temp = board.copy()
    moves = []

    if piece_count <= 7:
        Logger.info("从 ChessDB 查询最优解法...")
        for _ in range(60):
            if temp.is_game_over():
                break
            body = _query_cdb(temp.fen())
            if not body:
                break
            uci = _pick_best(body)
            if not uci:
                break
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                break
            moves.append(move)
            temp.push(move)

        if temp.is_game_over():
            Logger.success(f"ChessDB 完整解法: {len(moves)} 步")
            return moves

        if moves:
            Logger.info(f"ChessDB 部分数据 ({len(moves)}步)，SF 补全...")

    if not temp.is_game_over():
        Logger.info("Stockfish 搜索解法...")
        pv = _sf_solve(temp, stockfish_path)
        for move in pv:
            if temp.is_game_over():
                break
            moves.append(move)
            temp.push(move)

    Logger.success(f"解法: {len(moves)} 步")
    return moves
