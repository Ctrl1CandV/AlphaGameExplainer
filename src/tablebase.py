from typing import List, Optional, Tuple
import chess
import chess.syzygy
import chess.gaviota
from src.common import Logger, AnalyzedMove


# ============================================================
# DTZ 平局打破启发式（纯装饰，不影响胜负/DTZ最优性/50步规则）
#
# 背景：KBNvK 这类无兵杀法没有零化着，强方全程靠「子局面 DTZ 绝对值最小」
# 贪心选着。多个着 DTZ 相等时，旧实现取 legal_moves 的第一个（按格子序号），
# 完全任意，导致弱方王活动格数在最优线里反复涨回（实测 KBNvK 出现 1→4 的
# 回弹），观众看是「王被赶来赶去、没真正被压」，这些低进展节点没有棋理事实
# 可讲，LLM 便填套话甚至编出不存在的吃子。
#
# 修法：仅在 DTZ 已经相等的着之间做二次排序，优先选「让弱方王活动格更少、
# 更贴近角落、两王更靠近」的着。因为只在等-DTZ 的着里重排，杀法长度与胜负
# 完全不变，只是把同样最优的「丑着」换成「收紧感强的好着」。
# ============================================================

def _king_mobility(board: chess.Board, color: bool) -> int:
    """color 方王的安全活动格数（相邻空格或可吃子、且不被对方攻击）。越小=越被压缩。"""
    ksq = board.king(color)
    if ksq is None:
        return 0
    cnt = 0
    for sq in chess.SQUARES:
        if chess.square_distance(ksq, sq) != 1:
            continue
        if board.piece_at(sq) is not None and board.color_at(sq) == color:
            continue
        if board.is_attacked_by(not color, sq):
            continue
        cnt += 1
    return cnt


def _corner_distance(square: int) -> int:
    """到最近角落的切比雪夫距离（0=已在角落）。"""
    f = chess.square_file(square)
    r = chess.square_rank(square)
    return min(max(f, r), max(7 - f, r), max(f, 7 - r), max(7 - f, 7 - r))


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
        loaded = []
        if self.syzygy_dir:
            try:
                self._syzygy = chess.syzygy.open_tablebase(self.syzygy_dir)
                self._syzygy_available = True
                loaded.append("Syzygy")
            except Exception:
                pass
        if self.gaviota_dir:
            try:
                self._gaviota = chess.gaviota.open_tablebase(self.gaviota_dir)
                self._gaviota_available = True
                loaded.append("Gaviota")
            except Exception:
                pass
        if loaded:
            Logger.info(f"表库已加载: {'+'.join(loaded)}")
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

    def _best_move_syzygy(self, board: chess.Board) -> Optional[Tuple[chess.Move, int, int]]:
        """标准 DTZ 收敛选着。

        旧实现按「子局面 DTZ 绝对值最小」选着，但推兵/升变是零化着会重置 DTZ
        变大，导致获胜方永远嫌推兵更差、死活不升变、两王无限循环最终和棋。
        正确做法：
          获胜方(current_wdl>0)：只在保住胜势(子局面对手 wdl<0)的着里选；
              优先零化着(推兵/升变/吃子，board.is_zeroing)，再按子 DTZ 绝对值最小(推进最快)。
          防守方(current_wdl<0)：拖最久——避免零化、子 DTZ 绝对值最大。
          和棋(current_wdl==0)：维持和棋(子局面对手 wdl>=0)，DTZ 绝对值最大。
        返回 (move, child_wdl, child_dtz)，child_wdl 为「子局面对手视角」的 WDL，
        与旧签名一致（调用方据此判 is_only_move）。逐着 probe 失败跳过；
        全部失败则返回 None，由上层回退 Stockfish。
        """
        current_wdl = self.probe_wdl(board)
        if not self._syzygy_available:
            return None

        # 收集所有可 probe 的候选着
        # cands 元素：(move, zeroing, child_wdl, child_dtz, tiebreak)
        # tiebreak 是「弱方王活动格, 弱方王到角距离, 两王距离」三元组，仅用于
        # 获胜方在 DTZ 相等的着之间二次排序，值越小=收紧感越强（王更被压、更近角、
        # 两王更贴）。两王距离取正值，让强方王在等优着里主动贴近弱方王逼角。
        opp = not board.turn  # 弱方（被将杀方）颜色
        cands = []
        for move in board.legal_moves:
            zeroing = board.is_zeroing(move)
            temp = board.copy()
            temp.push(move)
            try:
                child_wdl = self._syzygy.probe_wdl(temp)
            except Exception:
                continue
            if child_wdl is None:
                continue
            try:
                child_dtz = self._syzygy.probe_dtz(temp)
            except Exception:
                child_dtz = None
            opp_ksq = temp.king(opp)
            my_ksq = temp.king(board.turn)
            if opp_ksq is not None and my_ksq is not None:
                tiebreak = (
                    _king_mobility(temp, opp),
                    _corner_distance(opp_ksq),
                    chess.square_distance(my_ksq, opp_ksq),
                )
            else:
                tiebreak = (99, 99, 0)
            cands.append((move, zeroing, child_wdl, child_dtz, tiebreak))

        if not cands:
            return None

        def dtz_abs(c):
            return abs(c[3]) if c[3] is not None else 9999

        # current_wdl 缺失时按「获胜方」保守处理（尽量推进）
        winning = current_wdl is None or current_wdl > 0
        losing = current_wdl is not None and current_wdl < 0

        if winning:
            # 保住胜势：我方走完后，对手视角应为负(child_wdl<0)
            keep = [c for c in cands if c[2] < 0]
            pool = keep or cands
            # 零化着优先(0)，再按推进最快(子 DTZ 绝对值小)，
            # DTZ 相等时用 tiebreak 选收紧感最强的着（消除原地游走，不改最优性）
            best = min(pool, key=lambda c: (0 if c[1] else 1, dtz_abs(c), c[4]))
        elif losing:
            # 防守：避免零化、拖最久(子 DTZ 绝对值大)
            best = min(cands, key=lambda c: (1 if c[1] else 0, -dtz_abs(c)))
        else:
            # 和棋：维持不输(child_wdl>=0)，拖最久
            keep = [c for c in cands if c[2] >= 0]
            pool = keep or cands
            best = min(pool, key=lambda c: (1 if c[1] else 0, -dtz_abs(c)))

        move, _z, child_wdl, child_dtz, _tb = best
        return (move, child_wdl, child_dtz if child_dtz is not None else 0)

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

    def solve(self, board: chess.Board) -> List[AnalyzedMove]:
        if not self._opened:
            return []
        # 根局面本身必须可探，否则不认命中（避免 6 子根局面被
        # _best_move_syzygy「只剩能降子的吃子着」逼着弃子走进有表可查的败局）
        if self.probe_wdl(board) is None:
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
        # 只认「根局面本身可探」。旧实现在根局面探不到时回退
        # _best_move_syzygy，而后者只保留「走完能落进 5 子表」的着——
        # 6 子局面里唯一能降子的就是吃子，于是把必胜局面的弃子着误当成解，
        # 逼着强方弃子走进有表可查的败局。表库只在能精确探到时才接管，
        # 否则交给 Stockfish(+SyzygyPath) 求解。
        return self.probe_wdl(board) is not None

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
