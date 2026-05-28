from typing import List, Optional, Tuple
import chess
import chess.syzygy
import chess.gaviota
from src.common import Logger, AnalyzedMove

class TablebaseSolver:
    def __init__(self, syzygy_dir: str = "", gaviota_dir: str = ""):
        self.syzygy_dir = syzygy_dir
        self.gaviota_dir = gaviota_dir
        self._syzygy = None
        self._gaviota = None
        self._syzygy_available = False
        self._gaviota_available = False
        self._opened = False

    def open(self):
        if self._opened:
            return
        if self.syzygy_dir:
            try:
                self._syzygy = chess.syzygy.open_tablebase(self.syzygy_dir)
                Logger.info(f"Syzygy 表库已加载: {self.syzygy_dir}")
                self._syzygy_available = True
            except Exception as e:
                Logger.warn(f"Syzygy 加载失败: {e}")
        if self.gaviota_dir:
            try:
                self._gaviota = chess.gaviota.open_tablebase(self.gaviota_dir)
                Logger.info(f"Gaviota 表库已加载: {self.gaviota_dir}")
                self._gaviota_available = True
            except Exception as e:
                Logger.warn(f"Gaviota 加载失败: {e}")
        self._opened = True

    def close(self):
        if self._syzygy:
            try:
                self._syzygy.close()
            except Exception:
                pass
            self._syzygy = None
        if self._gaviota:
            try:
                self._gaviota.close()
            except Exception:
                pass
            self._gaviota = None
        self._syzygy_available = False
        self._gaviota_available = False
        self._opened = False

    def is_available(self, board: chess.Board) -> bool:
        if not self._opened:
            return False
        if self._gaviota_available and len(board.piece_map()) <= 5:
            return self._try_probe_any(board) is not None
        return False

    def probe_wdl(self, board: chess.Board) -> Optional[int]:
        if not self._opened:
            return None
        if self._syzygy_available:
            try:
                return self._syzygy.probe_wdl(board)
            except (KeyError, chess.syzygy.MissingTableError):
                pass
        if self._gaviota_available:
            try:
                result = self._gaviota.probe_wdl(board)
                if result is not None:
                    return 2 if result > 0 else (-2 if result < 0 else 0)
            except Exception:
                pass
        return None

    def probe_dtm(self, board: chess.Board) -> Optional[int]:
        if not self._opened:
            return None
        if self._gaviota_available:
            try:
                return self._gaviota.probe_dtm(board)
            except Exception:
                pass
        return None

    def probe_dtz(self, board: chess.Board) -> Optional[int]:
        if not self._opened:
            return None
        if self._syzygy_available:
            try:
                return self._syzygy.probe_dtz(board)
            except (KeyError, chess.syzygy.MissingTableError):
                pass
        return None

    def _try_probe_any(self, board: chess.Board) -> Optional[int]:
        result = self.probe_wdl(board)
        if result is not None:
            return result
        return None

    def _best_move_syzygy(self, board: chess.Board) -> Optional[Tuple[chess.Move, int, int]]:
        current_wdl = self.probe_wdl(board)
        if current_wdl is None and not self._syzygy_available:
            return None

        best_move = None
        best_wdl = None
        best_dtz = None
        best_dtz_abs = None
        fallback_mode = current_wdl is None
        is_winning = current_wdl > 0 if not fallback_mode else True
        for move in board.legal_moves:
            temp = board.copy()
            temp.push(move)
            try:
                wdl = self._syzygy.probe_wdl(temp)
                if wdl is None:
                    continue
                dtz = None
                try:
                    dtz = self._syzygy.probe_dtz(temp)
                except Exception:
                    pass
                dtz_abs = abs(dtz) if dtz is not None else None
                if best_wdl is None:
                    best_wdl, best_dtz, best_dtz_abs, best_move = wdl, dtz, dtz_abs, move
                    continue
                if fallback_mode:
                    if wdl < best_wdl or (wdl == best_wdl and dtz_abs is not None and (best_dtz_abs is None or dtz_abs < best_dtz_abs)):
                        best_wdl, best_dtz, best_dtz_abs, best_move = wdl, dtz, dtz_abs, move
                else:
                    if wdl < best_wdl:
                        best_wdl, best_dtz, best_dtz_abs, best_move = wdl, dtz, dtz_abs, move
                    elif wdl == best_wdl and dtz_abs is not None:
                        if best_dtz_abs is None:
                            best_wdl, best_dtz, best_dtz_abs, best_move = wdl, dtz, dtz_abs, move
                        elif is_winning:
                            if dtz_abs < best_dtz_abs:
                                best_wdl, best_dtz, best_dtz_abs, best_move = wdl, dtz, dtz_abs, move
                        else:
                            if dtz_abs > best_dtz_abs:
                                best_wdl, best_dtz, best_dtz_abs, best_move = wdl, dtz, dtz_abs, move
            except Exception:
                continue
        if best_move is not None:
            return (best_move, best_wdl, best_dtz if best_dtz is not None else 0)
        return None

    def _best_move_gaviota(self, board: chess.Board) -> Optional[Tuple[chess.Move, int, int]]:
        best_move = None
        best_wdl = -999
        best_dtm = 9999
        for move in board.legal_moves:
            temp = board.copy()
            temp.push(move)
            try:
                wdl_raw = self._gaviota.probe_wdl(temp)
                if wdl_raw is None:
                    continue
                wdl = 2 if wdl_raw > 0 else (-2 if wdl_raw < 0 else 0)
                dtm = self._gaviota.probe_dtm(temp)
                if dtm is None:
                    dtm = 9999
                dtm_abs = abs(dtm) if dtm != 0 else 9999
                if wdl > best_wdl or (wdl == best_wdl and dtm_abs < best_dtm):
                    best_wdl, best_dtm, best_move = wdl, dtm_abs, move
            except Exception:
                continue
        if best_move is not None:
            return (best_move, best_wdl, -best_dtm)
        return None

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        if not self._opened:
            return None
        if self._gaviota_available and len(board.piece_map()) <= 5:
            result = self._best_move_gaviota(board)
            if result:
                return result[0]
        if self._syzygy_available:
            result = self._best_move_syzygy(board)
            if result:
                return result[0]
        return None

    def solve(self, board: chess.Board) -> List[AnalyzedMove]:
        if not self._opened:
            return []
        piece_count = len(board.piece_map())
        if piece_count > 7:
            return []
        if piece_count <= 5 and self._gaviota_available:
            return self._solve_with(board, use="gaviota")
        if self._syzygy_available:
            return self._solve_with(board, use="syzygy")
        return []

    def solve_step(self, board: chess.Board) -> Optional[AnalyzedMove]:
        if not self._opened:
            return None
        wdl = self.probe_wdl(board)
        if wdl is None:
            return None
        if self._gaviota_available and len(board.piece_map()) <= 5:
            pair = self._best_move_gaviota(board)
        elif self._syzygy_available:
            pair = self._best_move_syzygy(board)
        else:
            return None
        if pair is None:
            return None
        move, move_wdl, _ = pair
        return AnalyzedMove(
            move=move,
            score=None,
            candidates=[],
            is_only_move=(wdl == 2 and move_wdl < 2) or (wdl == -2),
            trap_san=None,
            source="gaviota" if self._gaviota_available else "syzygy",
        )

    def _solve_with(self, board: chess.Board, use: str) -> List[AnalyzedMove]:
        temp = board.copy()
        result = []
        max_steps = 100
        for _ in range(max_steps):
            if temp.is_game_over():
                break
            wdl = self.probe_wdl(temp)
            if use == "gaviota":
                pair = self._best_move_gaviota(temp)
            else:
                pair = self._best_move_syzygy(temp)
            if pair is None:
                break
            move, move_wdl, info = pair
            if wdl is None:
                wdl = 2 if move_wdl is not None and move_wdl < 0 else (move_wdl if move_wdl is not None else 0)
            dtm = None
            if use == "gaviota":
                dtm = info
            result.append(AnalyzedMove(
                move=move,
                score=None,
                candidates=[],
                is_only_move=(wdl == 2 and move_wdl < 2) or (wdl == -2),
                trap_san=None,
                source="gaviota" if use == "gaviota" else "syzygy",
                dtm=dtm,
            ))
            temp.push(move)
        return result

    def is_hit(self, board: chess.Board) -> bool:
        if self.probe_wdl(board) is not None:
            return True
        if self._syzygy_available:
            return self._best_move_syzygy(board) is not None
        return False

    def is_draw(self, board: chess.Board) -> Optional[bool]:
        """
        检测局面在五十步规则约束下是否为和棋。
        返回:
            True  - 和棋（理论必和/子力不足/50步内无法兑现）
            False - 存在一方在50步内可兑现的必胜
            None  - 表库未命中，无法判定
        """
        wdl = self.probe_wdl(board)
        if wdl is None:
            return None
        if abs(wdl) <= 1:
            return True
        if self._syzygy_available:
            try:
                dtz = self._syzygy.probe_dtz(board)
            except (KeyError, chess.syzygy.MissingTableError):
                dtz = None
            if dtz is not None and dtz != 0:
                remaining = 100 - board.halfmove_clock
                if abs(dtz) > remaining:
                    return True
        return False

    @property
    def syzygy_path(self) -> str:
        return self.syzygy_dir

    @property
    def gaviota_path(self) -> str:
        return self.gaviota_dir
