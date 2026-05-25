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
        best_move = None
        best_wdl = -999
        best_dtz = 0
        for move in board.legal_moves:
            temp = board.copy()
            temp.push(move)
            try:
                wdl = self._syzygy.probe_wdl(temp)
                if wdl is None:
                    continue
                dtz = 0
                try:
                    dtz = self._syzygy.probe_dtz(temp) or 0
                except Exception:
                    pass
                if wdl > best_wdl or (wdl == best_wdl and abs(dtz) < abs(best_dtz)):
                    best_wdl, best_dtz, best_move = wdl, dtz, move
            except Exception:
                continue
        if best_move is not None:
            return (best_move, best_wdl, best_dtz)
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

    def _solve_with(self, board: chess.Board, use: str) -> List[AnalyzedMove]:
        temp = board.copy()
        result = []
        max_steps = 100
        for _ in range(max_steps):
            if temp.is_game_over():
                break
            wdl = self.probe_wdl(temp)
            if wdl is None:
                break
            if use == "gaviota":
                pair = self._best_move_gaviota(temp)
            else:
                pair = self._best_move_syzygy(temp)
            if pair is None:
                break
            move, move_wdl, info = pair
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
            if use == "gaviota":
                wdl_before = self.probe_wdl(temp)
                temp.push(move)
                wdl_after = self.probe_wdl(temp)
                if wdl_before == wdl_after:
                    continue
            temp.push(move)
        return result

    def is_hit(self, board: chess.Board) -> bool:
        return self.probe_wdl(board) is not None

    def is_draw(self, board: chess.Board) -> Optional[bool]:
        """
        检测局面是否为和棋（含理论必和与 cursed win / blessed loss）。
        返回:
            True  - 和棋（WDL=0 理论必和, 或 WDL=±1 超50步无法兑现）
            False - 存在一方必胜（WDL=±2）
            None  - 表库未命中，无法判定
        """
        wdl = self.probe_wdl(board)
        if wdl is None:
            return None
        return abs(wdl) <= 1

    @property
    def syzygy_path(self) -> str:
        return self.syzygy_dir

    @property
    def gaviota_path(self) -> str:
        return self.gaviota_dir
