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

def _sf_step_fast(engine, board: chess.Board, step_time: float = PER_STEP_FAST) -> Optional[AnalyzedMove]:
    try:
        info = engine.analyse(board, chess.engine.Limit(time=step_time))
        pv = info.get("pv", [])
        if pv and pv[0] in board.legal_moves:
            return AnalyzedMove(
                move=pv[0], score=None, candidates=[],
                is_only_move=False, trap_san=None, source="sf",
            )
    except chess.engine.EngineTerminatedError:
        raise
    except Exception:
        pass
    return None

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
              piece_count: int = 0, max_steps: int = 60,
              tablebase_solver=None) -> List[AnalyzedMove]:
    max_restarts = 2

    def _open_engine():
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        cfg = {"Hash": 128, "Threads": 1}
        if syzygy_path and os.path.isdir(syzygy_path):
            cfg["SyzygyPath"] = os.path.abspath(syzygy_path)
        engine.configure(cfg)
        return engine

    try:
        engine = _open_engine()
    except Exception as e:
        Logger.error(f"无法启动Stockfish引擎: {e}")
        return []

    Logger.info(f"SF 自主搜索 (子力{piece_count})")
    restart_count = 0
    step_log_count = 0
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

        # ---- 阶段 2：快速求解 ----
        fast_moves: List[AnalyzedMove] = []
        fast_boards: List[chess.Board] = []
        fast_piece_counts: List[int] = []
        engine_died = False
        for _step in range(max_steps):
            if temp.is_game_over():
                break
            fast_boards.append(temp.copy())
            fast_piece_counts.append(current_piece_count)
            try:
                am = _sf_step_fast(engine, temp, step_time=fast_step_time)
            except chess.engine.EngineTerminatedError:
                if restart_count >= max_restarts:
                    Logger.warn("  SF 引擎多次崩溃，搜索中止，结果标记为不完整")
                    engine_died = True
                    break
                restart_count += 1
                Logger.warn(f"  SF 引擎崩溃，重启中... ({restart_count}/{max_restarts})")
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
                    am = _sf_step_fast(engine, temp, step_time=fast_step_time)
                except Exception:
                    am = None
            except Exception:
                am = None

            if am is None:
                am = _sf_fallback_move(temp)
                am.source = "sf_degraded"

            if am.move == chess.Move.null():
                break
            fast_moves.append(am)
            temp.push(am.move)
            step_log_count += 1
            if step_log_count <= 2 or step_log_count % 10 == 0:
                Logger.info(f"  SF 第{_step + 1}步: {am.move.uci()}")
            current_piece_count = len(temp.piece_map())
            if current_piece_count < piece_count and current_piece_count <= 5 and tablebase_solver is not None:
                if tablebase_solver.is_hit(temp):
                    Logger.info(f"  SF 吃到子，已进入{tablebase_solver.syzygy_path or '表库'}覆盖范围，后续切换表库求解")
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
            Logger.info(f"SF 精标注 {len(key_indices)}/{total} 个关键位置 ...")
            replay = board.copy()
            for i in range(total):
                if i >= len(fast_boards):
                    break
                if i in key_indices:
                    try:
                        heavy = _sf_step_heavy(engine, replay)
                    except chess.engine.EngineTerminatedError:
                        Logger.warn(f"  精标注第{i + 1}步引擎崩溃，跳过此步精标注")
                        try:
                            engine.quit()
                        except Exception:
                            pass
                        time.sleep(0.3)
                        try:
                            engine = _open_engine()
                        except Exception:
                            pass
                    except Exception as e:
                        Logger.warn(f"  精标注第{i + 1}步异常: {e}")
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
                    Logger.info(f"本地表库部分数据 ({len(result)}步)，继续 Stockfish...")

    if not temp.is_game_over() and use_stockfish:
        Logger.info("Stockfish 搜索解法...")
        sf_max_steps = 60
        current_pc = len(temp.piece_map())
        if current_pc <= 5 and tablebase_solver is not None and not tablebase_solver.is_hit(temp):
            sf_max_steps = 100
            Logger.info(f"  表库未命中但子力≤5（如KBNvK），增加搜索步数至{sf_max_steps}")
        sf_result = _sf_solve(temp, stockfish_path, syzygy_path, current_pc,
                              max_steps=sf_max_steps, tablebase_solver=tablebase_solver)
        for am in sf_result:
            if temp.is_game_over():
                break
            if am.move not in temp.legal_moves:
                Logger.warn(f"SF 返回非法走法 {am.move.uci()}，跳过")
                break
            result.append(am)
            temp.push(am.move)

    Logger.success(f"解法: {len(result)} 步")
    return result
