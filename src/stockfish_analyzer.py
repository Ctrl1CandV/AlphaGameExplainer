from typing import List, Optional
from src.common import Logger, AnalyzedMove
import chess.engine
import urllib.parse
import requests
import asyncio
import chess
import time
import re

CDB_URL = "http://www.chessdb.cn/cdb.php"
MATE_TIME = 2.0                           # 将杀搜索的时间限制
PER_STEP_FAST = 0.3                       # 快速模式每步搜索时间
PER_STEP_HEAVY = 1.5                      # 精标注模式每步搜索时间
ONLY_MOVE_THRESHOLD = 150                 # 次优与最优分差阈值，超过视为唯一好着
TRAP_THRESHOLD = 100                      # 次优/第三优与最优分差阈值，超过视为陷阱
HEAVY_INTERVAL = 5                        # 每隔N步做一次精标注

def _cdb_query(fen: str, action: str) -> Optional[str]:
    url = f"{CDB_URL}?action={action}&board={urllib.parse.quote(fen)}"
    try:
        body = requests.get(url, timeout=15).text.strip()
        if not body or body in ("unknown", "nobestmove", "invalid board"):
            return None

        time.sleep(0.15)
        return body
    except Exception:
        return None

def _query_cdb(fen: str) -> Optional[str]:
    """ 先用action=query获取最佳着法，失败则回退到action=queryall """
    body = _cdb_query(fen, "query")
    if body:
        return body
    body = _cdb_query(fen, "queryall")
    if body:
        return body
    return None

def _pick_best(cdb_body: str) -> Optional[str]:
    """ 从query或queryall返回体中提取最佳着法 """
    trimmed = cdb_body.strip()
    if not trimmed:
        return None
    if not trimmed.startswith("move:"):
        if re.match(r"^[a-h][1-8][a-h][1-8][qrbn]?$", trimmed):
            return trimmed
        parts = trimmed.split("|")
        if parts and re.match(r"^[a-h][1-8][a-h][1-8][qrbn]?$", parts[0]):
            return parts[0]
        return None

    best_uci, best_score, best_rank = None, -9999999, 9999999
    for part in cdb_body.split("|"):
        MOVE_RE = re.compile(r"move:(\S+),score:(-?\d+),rank:(\d+),note:(.*)")
        move = MOVE_RE.match(part.strip())
        if not move:
            continue

        uci, score_str, rank_str = move.group(1), move.group(2), move.group(3)
        try:
            score = int(score_str)
            rank = int(rank_str)
        except ValueError:
            continue

        # score最高者优先，score相同时rank更小者优先（rank:1表示最佳）
        if score > best_score or (score == best_score and rank < best_rank):
            best_uci, best_score, best_rank = uci, score, rank
    return best_uci

def _cdb_step(board: chess.Board) -> Optional[AnalyzedMove]:
    """ 对当前局面查询ChessDB，返回含评分的AnalyzedMove """
    body = _query_cdb(board.fen())
    if not body:
        return None
    uci = _pick_best(body)
    if not uci:
        return None
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None

    # 从原始响应中提取该走法的评分
    score = None
    for part in body.split("|"):
        MOVE_RE = re.compile(r"move:(\S+),score:(-?\d+),rank:(\d+),note:(.*)")
        m = MOVE_RE.match(part.strip())
        if m and m.group(1) == uci:
            try:
                score = int(m.group(2))
            except ValueError:
                pass
            break

    return AnalyzedMove(
        move=move,
        score=score,
        candidates=[],
        is_only_move=False,
        trap_san=None,
        source="chessdb",
    )

def _sf_mate_try(engine, board: chess.Board) -> Optional[List[AnalyzedMove]]:
    """ 尝试将杀搜索，命中则返回完整AnalyzedMove列表 """
    info = engine.analyse(board, chess.engine.Limit(mate=30, time=MATE_TIME))
    pv, score = info.get("pv", []), info.get("score")
    mate = score.relative.mate() if score else None

    if mate is None or mate <= 0 or len(pv) == 0:
        return None

    Logger.info(f"SF mate搜索: M{mate}, {len(pv)} 着/{MATE_TIME:.0f}s")
    result = []
    temp = board.copy()
    for move in pv:
        if temp.is_game_over():
            break
        result.append(AnalyzedMove(
            move=move, score=None,
            candidates=[], is_only_move=True,
            trap_san=None, source="sf",
        ))
        temp.push(move)
    return result

def _sf_step_fast(engine, board: chess.Board) -> Optional[AnalyzedMove]:
    """ 快速单步分析：multipv=1，低时限 """
    info = engine.analyse(board, chess.engine.Limit(time=PER_STEP_FAST))
    pv = info.get("pv", [])
    if not pv:
        return None
    s = info.get("score")
    cp = s.relative.score(mate_score=10000) if s else None
    return AnalyzedMove(
        move=pv[0], score=cp,
        candidates=[], is_only_move=False,
        trap_san=None, source="sf",
    )

def _sf_step_heavy(engine, board: chess.Board) -> AnalyzedMove:
    """ 精标注单步：multipv=3，含候选走法和陷阱检测 """
    best_move, best_cp = None, None
    candidates: List[str] = []
    is_only = False
    trap_san = None

    with engine.analysis(board, chess.engine.Limit(time=PER_STEP_HEAVY), multipv=3) as analysis:
        for info in analysis:
            pv = info.get("pv", [])
            if not pv:
                continue
            s = info.get("score")
            cp = s.relative.score(mate_score=10000) if s else None
            mpv = info.get("multipv", 1)

            if mpv == 1:
                best_move, best_cp = pv[0], cp
            elif mpv == 2:
                if cp is not None and best_cp is not None:
                    if (best_cp - cp) > ONLY_MOVE_THRESHOLD:
                        is_only = True
                try:
                    candidates.append(board.san(pv[0]))
                except Exception:
                    pass
            elif mpv == 3:
                if cp is not None and best_cp is not None:
                    if (best_cp - cp) > TRAP_THRESHOLD:
                        try:
                            trap_san = board.san(pv[0])
                        except Exception:
                            pass

    if best_move is None:
        info = engine.analyse(board, chess.engine.Limit(time=PER_STEP_HEAVY))
        pv = info.get("pv", [])
        if pv:
            best_move = pv[0]
            s = info.get("score")
            best_cp = s.relative.score(mate_score=10000) if s else None
        else:
            return AnalyzedMove(
                move=chess.Move.null(), score=None,
                candidates=[], is_only_move=False,
                trap_san=None, source="sf",
            )

    return AnalyzedMove(
        move=best_move, score=best_cp,
        candidates=candidates, is_only_move=is_only,
        trap_san=trap_san, source="sf",
    )

def _sf_solve(board: chess.Board, stockfish_path: str, syzygy_path: str = "") -> List[AnalyzedMove]:
    """ 两阶段求解：先快速拿到走法序列，再对关键位置精标注 """
    def _quiet_handler(_loop, context):
        exc = context.get('exception')
        if isinstance(exc, asyncio.InvalidStateError):
            return
        if old_handler:
            old_handler(_loop, context)
        else:
            _loop.default_exception_handler(context)

    loop, engine = asyncio.get_event_loop(), None
    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(_quiet_handler)
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        if syzygy_path:
            try:
                engine.configure({"SyzygyPath": syzygy_path, "SyzygyProbeDepth": 1})
                Logger.info(f"SF 已接入 Syzygy: {syzygy_path}")
            except Exception as e:
                Logger.warn(f"SF 配置 Syzygy 失败: {e}")
    except Exception as e:
        Logger.error(f"无法启动Stockfish引擎: {e}")
        return []
    try:
        temp = board.copy()

        mate_result = _sf_mate_try(engine, temp)
        if mate_result is not None:
            return mate_result

        # 阶段1：快速求解
        Logger.info(f"SF 快速求解 ...")
        fast_moves: List[AnalyzedMove] = []
        fast_boards: List[chess.Board] = []
        for _step in range(60):
            if temp.is_game_over():
                break
            am = _sf_step_fast(engine, temp)
            if am is None:
                break
            fast_boards.append(temp.copy())
            fast_moves.append(am)
            temp.push(am.move)

        if not fast_moves:
            return []

        total = len(fast_moves)
        # 阶段2：确定需要精标注的位置
        key_indices: set = {0, total - 1}
        for i in range(total):
            if i % HEAVY_INTERVAL == 0:
                key_indices.add(i)
            board_i = fast_boards[i]
            move = fast_moves[i].move
            if board_i.gives_check(move) or board_i.is_capture(move):
                key_indices.add(i)
        if total >= 3:
            key_indices.add(total - 2)

        # 阶段3：精标注关键位置
        if len(key_indices) > 0:
            Logger.info(f"SF 精标注 {len(key_indices)}/{total} 个关键位置 (每步{PER_STEP_HEAVY}s)...")
            replay = board.copy()
            for i in range(total):
                if i in key_indices:
                    heavy = _sf_step_heavy(engine, replay)
                    fast_moves[i] = AnalyzedMove(
                        move=fast_moves[i].move,
                        score=heavy.score,
                        candidates=heavy.candidates,
                        is_only_move=heavy.is_only_move,
                        trap_san=heavy.trap_san,
                        source="sf",
                    )
                replay.push(fast_moves[i].move)

        return fast_moves
    except Exception as e:
        Logger.warn(f"SF搜索异常: {e}")
        return []
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass
        loop.set_exception_handler(old_handler)

def get_solution(board: chess.Board, stockfish_path: str, tablebase_solver=None, syzygy_path: str = "") -> List[AnalyzedMove]:
    """
    获取从当前局面到终局的最优解法，优先级：本地表库 > ChessDB > Stockfish

    - 本地表库 (Syzygy/Gaviota): 穷举真值，≤7子且命中的局面可直接走到终局
    - ChessDB: ≤7子未命中表库时的在线查询，可能不完整
    - Stockfish: 兜底搜索，可接入Syzygy辅助评估
    返回包含评分、候选走法、陷阱等元数据的AnalyzedMove列表
    """
    piece_count = len(board.piece_map())
    temp, result = board.copy(), []

    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        Logger.error("非法局面：缺少白王或黑王，无法分析")
        return []
    if board.is_game_over():
        Logger.info("局面已结束，无需分析")
        return []

    if tablebase_solver is not None:
        try:
            tablebase_solver.open()
        except Exception as e:
            Logger.warn(f"表库打开失败: {e}")

        if tablebase_solver.is_hit(temp):
            Logger.info("从本地表库查询最优解法...")
            tb_result = tablebase_solver.solve(temp)
            if tb_result:
                for am in tb_result:
                    if temp.is_game_over():
                        break
                    if am.move not in temp.legal_moves:
                        Logger.warn(f"表库返回非法走法 {am.move.uci()}，回退到SF")
                        break
                    result.append(am)
                    temp.push(am.move)
                if temp.is_game_over():
                    Logger.success(f"本地表库完整解法: {len(result)} 步 (来源:{tb_result[0].source})")
                    return result
                if result:
                    Logger.info(f"本地表库部分数据 ({len(result)}步)，SF 补全...")

    if not temp.is_game_over() and piece_count <= 7:
        Logger.info("从 ChessDB 查询最优解法...")
        for _ in range(60):
            am = _cdb_step(temp)
            if am is None:
                break

            if am.move not in temp.legal_moves:
                Logger.error(f"ChessDB 返回非法走法 {am.move.uci()}，中断查询")
                break

            result.append(am), temp.push(am.move)

        if temp.is_game_over():
            Logger.success(f"ChessDB 完整解法: {len(result)} 步")
            return result

        if result and len(result) > 0 and result[-1].source == "chessdb":
            Logger.info(f"ChessDB 部分数据 ({len(result)}步)，SF 补全...")

    if not temp.is_game_over():
        Logger.info("Stockfish 搜索解法...")
        sf_result = _sf_solve(temp, stockfish_path, syzygy_path)
        for am in sf_result:
            if temp.is_game_over():
                break
            result.append(am), temp.push(am.move)

    Logger.success(f"解法: {len(result)} 步")
    return result
