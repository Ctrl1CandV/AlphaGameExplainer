from typing import List, Optional
from src.common import Logger, AnalyzedMove
import chess.engine
import logging
logging.getLogger("chess.engine").setLevel(logging.CRITICAL)
import chess
import time
import os

MATE_TIME = 2.0
PER_STEP_FAST = 0.3
PER_STEP_FAST_6P = 0.6
PER_STEP_FAST_NON_TB = 1.0
PER_STEP_HEAVY = 1.5

# ============================================================
# Stockfish 搜索函数
# ============================================================

def _sf_mate_try(engine, board: chess.Board) -> Optional[List[AnalyzedMove]]:
    try:
        info = engine.analyse(board, chess.engine.Limit(time=MATE_TIME, depth=25))
    except Exception:
        return None
    pv, score = info.get("pv", []), info.get("score")
    if score is None:
        return None
    relative_score = score.relative
    mate = relative_score.mate()
    if mate is None or mate <= 0 or len(pv) == 0:
        return None
    Logger.info(f"SF mate搜索: M{mate}, {len(pv)} 着/{MATE_TIME:.0f}s")
    result = []
    temp = board.copy()
    for move in pv:
        if temp.is_game_over():
            break
        if move not in temp.legal_moves:
            Logger.warn(f"mate搜索PV含非法走法 {move.uci()}，截断")
            break
        result.append(AnalyzedMove(
            move=move, score=None, candidates=[], is_only_move=True,
            trap_san=None, source="sf",
        ))
        temp.push(move)
    return result if result else None

def _extract_mate_score(info: dict) -> Optional[int]:
    """从 SF 分析结果中提取将杀步数（全着），返回正数=距将杀的全着数，None=非将杀局面"""
    score_obj = info.get("score")
    if score_obj is None:
        return None
    try:
        relative = score_obj.relative
        mate = relative.mate()
        if mate is None or mate <= 0:
            return None
        return mate
    except Exception:
        return None


def _sf_step_fast(engine, board: chess.Board, step_time: float = PER_STEP_FAST) -> tuple:
    """返回 (AnalyzedMove, mate_score_or_None)"""
    try:
        info = engine.analyse(board, chess.engine.Limit(time=step_time))
        pv = info.get("pv", [])
        mate_score = _extract_mate_score(info)
        if pv and pv[0] in board.legal_moves:
            return AnalyzedMove(
                move=pv[0], score=None, candidates=[],
                is_only_move=False, trap_san=None, source="sf",
            ), mate_score
    except chess.engine.EngineTerminatedError:
        raise
    except Exception:
        pass
    return None, None

def _sf_step_heavy(engine, board: chess.Board) -> Optional[AnalyzedMove]:
    try:
        info = engine.analyse(board, chess.engine.Limit(time=PER_STEP_HEAVY))
        pv = info.get("pv", [])
        if pv and pv[0] in board.legal_moves:
            score_cp = None
            try:
                score_obj = info.get("score")
                if score_obj is not None:
                    cp = score_obj.relative.score()
                    score_cp = cp if board.turn == chess.WHITE else -cp
            except Exception:
                pass
            return AnalyzedMove(
                move=pv[0], score=score_cp, candidates=[],
                is_only_move=False, trap_san=None, source="sf",
            )
    except chess.engine.EngineTerminatedError:
        raise
    except Exception:
        pass
    return None

def _sf_fallback_move(board: chess.Board) -> AnalyzedMove:
    legal = list(board.legal_moves)
    if not legal:
        return AnalyzedMove(
            move=chess.Move.null(), score=None, candidates=[],
            is_only_move=False, trap_san=None, source="sf_fallback",
        )
    best_move = legal[0]
    best_see = -99999
    for move in legal:
        try:
            see_val = board.see(move)
        except Exception:
            see_val = 0
        if see_val is not None and see_val > best_see:
            best_see = see_val
            best_move = move
    return AnalyzedMove(
        move=best_move, score=None, candidates=[],
        is_only_move=False, trap_san=None, source="sf_fallback",
    )

def _sf_solve(board: chess.Board, stockfish_path: str, syzygy_path: str = "",
              piece_count: int = 0, max_steps: int = 80,
              tablebase_solver=None) -> List[AnalyzedMove]:
    max_restarts = 2

    def _open_engine():
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        cfg = {"Hash": 128, "Threads": 1}
        if syzygy_path and os.path.isdir(syzygy_path):
            cfg["SyzygyPath"] = os.path.abspath(syzygy_path)
            cfg["SyzygyProbeLimit"] = 6
        engine.configure(cfg)
        return engine

    try:
        engine = _open_engine()
    except Exception as e:
        Logger.error(f"无法启动Stockfish引擎: {e}")
        return []

    restart_count = 0
    tablebase_hit = tablebase_solver is not None and tablebase_solver.is_hit(board)
    if piece_count >= 6:
        fast_step_time = PER_STEP_FAST_6P
    elif tablebase_hit:
        fast_step_time = PER_STEP_FAST
    else:
        fast_step_time = PER_STEP_FAST_NON_TB

    try:
        temp = board.copy()
        current_piece_count = piece_count

        # ---- 阶段 1：mate 优先 ----
        mate_result = _sf_mate_try(engine, temp)
        if mate_result:
            Logger.success(f"SF mate搜索命中: {len(mate_result)} 步")
            return mate_result

        # ---- 阶段 2：逐步求解（阶段化时间 + 将杀进度追踪） ----
        fast_moves: List[AnalyzedMove] = []
        fast_boards: List[chess.Board] = []
        fast_piece_counts: List[int] = []
        engine_died = False

        current_mate: Optional[int] = None
        mate_stagnation = 0
        position_hashes: set = set()

        for _step in range(max_steps):
            if temp.is_game_over():
                break
            fast_boards.append(temp.copy())
            fast_piece_counts.append(current_piece_count)

            board_hash = (
                temp.board_fen(),
                temp.turn,
                temp.castling_xfen(),
                temp.ep_square,
            )
            if board_hash in position_hashes:
                step_time = max(fast_step_time, PER_STEP_HEAVY)
            else:
                position_hashes.add(board_hash)
                step_time = fast_step_time

            try:
                am, mate_score = _sf_step_fast(engine, temp, step_time=step_time)
            except chess.engine.EngineTerminatedError:
                if restart_count >= max_restarts:
                    Logger.warn("  SF 引擎多次崩溃，搜索中止，结果标记为不完整")
                    engine_died = True
                    break
                restart_count += 1
                try:
                    engine.quit()
                except Exception:
                    pass
                time.sleep(0.3)
                try:
                    engine = _open_engine()
                except Exception as e:
                    Logger.error(f"  SF 重启失败: {e}")
                    engine_died = True
                    break
                try:
                    am, mate_score = _sf_step_fast(engine, temp, step_time=step_time)
                except Exception:
                    am, mate_score = None, None
            except Exception:
                am, mate_score = None, None

            if am is None:
                am = _sf_fallback_move(temp)
                am.source = "sf_degraded"

            if am.move == chess.Move.null():
                break
            fast_moves.append(am)
            temp.push(am.move)

            # 将杀进度追踪与时间自适应
            if mate_score is not None:
                if current_mate is None:
                    fast_step_time = max(fast_step_time, 2.0)
                    current_mate = mate_score + 1
                    mate_stagnation = 0
                if mate_score < current_mate:
                    current_mate = mate_score
                    mate_stagnation = 0
                else:
                    mate_stagnation += 1
                    if mate_stagnation >= 6:
                        fast_step_time = max(fast_step_time, 2.5)
                        mate_stagnation = 0

            current_piece_count = len(temp.piece_map())
            if current_piece_count < piece_count and current_piece_count <= 5 and tablebase_solver is not None:
                if tablebase_solver.is_hit(temp):
                    remaining = _sf_continue_with_tablebase(
                        temp, tablebase_solver, max_steps - _step - 1)
                    fast_moves.extend(remaining)
                    break

        if engine_died and fast_moves:
            last = fast_moves[-1]
            fast_moves[-1] = AnalyzedMove(
                move=last.move, score=last.score, candidates=last.candidates,
                is_only_move=last.is_only_move, trap_san=last.trap_san,
                source="sf_incomplete",
            )

        if not fast_moves:
            return []

        # ---- 阶段 3：精简标注（仅首3 + 尾3 + 吃子/将军步） ----
        total = len(fast_moves)
        key_indices: set = set()
        for i in range(min(3, total)):
            key_indices.add(i)
        for i in range(max(0, total - 3), total):
            key_indices.add(i)
        for i in range(min(total, len(fast_boards))):
            board_i = fast_boards[i]
            move = fast_moves[i].move
            if board_i.gives_check(move) or board_i.is_capture(move):
                key_indices.add(i)

        if key_indices and not engine_died:
            replay = board.copy()
            for i in range(total):
                if i >= len(fast_boards):
                    break
                if i in key_indices:
                    try:
                        heavy = _sf_step_heavy(engine, replay)
                    except chess.engine.EngineTerminatedError:
                        try:
                            engine.quit()
                        except Exception:
                            pass
                        time.sleep(0.3)
                        try:
                            engine = _open_engine()
                        except Exception:
                            pass
                    except Exception:
                        pass
                    else:
                        if heavy is not None and heavy.move != chess.Move.null():
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
        try:
            engine.quit()
        except Exception:
            pass


def _sf_continue_with_tablebase(board: chess.Board, tablebase_solver,
                                 max_remaining: int) -> List[AnalyzedMove]:
    temp = board.copy()
    result = []
    for _ in range(min(max_remaining, 100)):
        if temp.is_game_over():
            break
        am = tablebase_solver.solve_step(temp)
        if am is None:
            break
        result.append(am)
        temp.push(am.move)
    return result


# ============================================================
# 主入口 get_solution
# ============================================================

def get_solution(board: chess.Board, stockfish_path: str,
                 tablebase_solver=None, syzygy_path: str = "",
                 use_stockfish: bool = True) -> List[AnalyzedMove]:
    """
    最优解法求解，优先级：本地表库 > Stockfish
      - 3-7 子表库命中 → 表库直解
      - 表库未命中或不完备 → Stockfish 兜底（含 SyzygyPath 导入）
    返回 AnalyzedMove 列表，调用方根据列表终局情况自行判定和棋/必胜
    """
    temp, result = board.copy(), []

    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        Logger.error("非法局面：缺少白王或黑王，无法分析")
        return []
    if not board.is_valid():
        Logger.error(f"非法局面: FEN不合法 (status={board.status()})，拒绝分析")
        return []
    if board.is_game_over():
        Logger.info("局面已结束，无需分析")
        return []

    has_tablebase_moves = False
    if tablebase_solver is not None:
        try:
            tablebase_solver.open()
        except Exception as e:
            Logger.warn(f"表库打开失败: {e}")

        if tablebase_solver.is_hit(temp):
            Logger.info("表库命中，查询最优解法...")
            tb_result = tablebase_solver.solve(temp)
            if tb_result:
                for am in tb_result:
                    if temp.is_game_over():
                        break
                    if am.move not in temp.legal_moves:
                        Logger.warn(f"表库返回非法走法 {am.move.uci()}，回退SF")
                        break
                    result.append(am)
                    temp.push(am.move)
                    has_tablebase_moves = True
                if temp.is_game_over():
                    Logger.success(f"表库完整解法: {len(result)} 步")
                    return result
                if has_tablebase_moves:
                    Logger.info(f"表库部分解法 ({len(result)} 步)，残余交SF续解...")
            else:
                Logger.info("表库命中但无法求解完整路线，交SF处理")
        else:
            Logger.info("表库未命中，交SF搜索")

    if not temp.is_game_over() and use_stockfish:
        current_pc = len(temp.piece_map())
        sf_max_steps = 80
        if current_pc <= 5 and not has_tablebase_moves:
            sf_max_steps = 120
        Logger.info(f"SF 搜索 (剩余{current_pc}子, 上限{sf_max_steps}步)...")
        sf_result = _sf_solve(temp, stockfish_path, syzygy_path, current_pc,
                              max_steps=sf_max_steps, tablebase_solver=tablebase_solver)
        for am in sf_result:
            if temp.is_game_over():
                break
            if am.move not in temp.legal_moves:
                break
            result.append(am)
            temp.push(am.move)

    if (not temp.is_game_over()) and result and (not has_tablebase_moves) and len(result) >= 80:
        Logger.warn(f"SF 在 {len(result)} 步内未完成收官，解法质量不足，跳过本次解说。")
        return []

    if result:
        Logger.success(f"解法就绪: {len(result)} 步")
    return result
