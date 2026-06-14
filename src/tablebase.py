from src.common import Logger, AnalyzedMove
from typing import List, Optional, Tuple
import chess.syzygy
import chess

def _king_mobility(board: chess.Board, color: bool) -> int:
    """ color方王的安全活动格数，即相邻空格或可吃子、且不被对方攻击，越小=越被压缩 """
    ksq = board.king(color)
    if ksq is None:
        return 0
    count = 0
    for sq in chess.SQUARES:
        if chess.square_distance(ksq, sq) != 1:
            continue
        if board.piece_at(sq) is not None and board.color_at(sq) == color:
            continue
        if board.is_attacked_by(not color, sq):
            continue
        count += 1
    return count

def _corner_distance(square: int) -> int:
    """ 到最近角落的切比雪夫距离，0代表已在角落 """
    f,r  = chess.square_file(square), chess.square_rank(square)
    return min(max(f, r), max(7 - f, r), max(f, 7 - r), max(7 - f, 7 - r))

class TablebaseSolver:
    def __init__(self, syzygy_dir: str = ""):
        self.syzygy_dir = syzygy_dir
        self._syzygy: chess.syzygy.Tablebase = None
        self._syzygy_available = False
        self._opened = False

    def open(self):
        if self._opened:
            return
        if self.syzygy_dir:
            try:
                self._syzygy = chess.syzygy.open_tablebase(self.syzygy_dir)
                self._syzygy_available = True
                Logger.info("Syzygy表库已加载")
            except Exception as e:
                Logger.error(f"表库打开失败：{e}")
        self._opened = True

    def close(self):
        if self._syzygy:
            try:
                self._syzygy.close()
            except Exception as e:
                Logger.error(f"表库关闭失败：{e}")
            self._syzygy = None
        self._syzygy_available = False
        self._opened = False

    def probe_wdl(self, board: chess.Board) -> Optional[int]:
        if not self._opened or not self._syzygy_available:
            return None
        try:
            return self._syzygy.probe_wdl(board)
        except (KeyError, chess.syzygy.MissingTableError):
            return None

    def _best_move_syzygy(self, board: chess.Board) -> Optional[Tuple[chess.Move, int, int]]:
        """
        获胜方只在保住胜势的着里选；优先零化着(推兵/升变/吃子，board.is_zeroing)，再按子DTZ绝对值最小0
        防守方：选择拖最久的且避免零化、子DTZ绝对值最大的着
        和棋：维持和棋，DTZ绝对值最大
        返回(move, child_wdl, child_dtz)，child_wdl为子局面对手视角的WDL
        全部失败则返回None，由上层回退到Stockfish查找
        """
        current_wdl = self.probe_wdl(board)
        if not self._syzygy_available:
            return None

        # 弱方颜色
        opp = not board.turn
        cands = []
        for move in board.legal_moves:
            zeroing, temp = board.is_zeroing(move), board.copy()
            temp.push(move)
            try:
                # push之后获取的wdl，为行棋后对手视角的WDL
                child_wdl = self._syzygy.probe_wdl(temp)
            except Exception:
                continue
            if child_wdl is None:
                continue
            try:
                child_dtz = self._syzygy.probe_dtz(temp)
            except Exception:
                child_dtz = None

            opp_ksq, my_ksq = temp.king(opp), temp.king(board.turn)
            if opp_ksq is not None and my_ksq is not None:
                tiebreak = (
                    _king_mobility(temp, opp),                  # 弱方王活动格数
                    _corner_distance(opp_ksq),                  # 弱方王到最近角的距离
                    chess.square_distance(my_ksq, opp_ksq),     # 两王的距离
                )
            else:
                tiebreak = (99, 99, 0)
            cands.append((move, zeroing, child_wdl, child_dtz, tiebreak))
        if not cands:
            return None

        def dtz_abs(c):
            return abs(c[3]) if c[3] is not None else 9999

        # current_wdl缺失时按获胜方保守处理
        winning = current_wdl is None or current_wdl > 0
        losing = current_wdl is not None and current_wdl < 0
        if winning:
            # c[2]是child_wdl，<0意味着对手仍然处于输势，即我走完这步后，对手仍然必败
            keep = [cand for cand in cands if c[2] < 0]
            pool = keep or cands
            # 筛选策略，根据三元组的优先级选出最优move
            best = min(pool, key=lambda c: (0 if c[1] else 1, dtz_abs(c), c[4]))
        elif losing:
            # 首先避免零化，再是拖最久
            best = min(cands, key=lambda c: (1 if c[1] else 0, -dtz_abs(c)))
        else:
            # 维持不输，再者拖最久
            keep = [c for c in cands if c[2] >= 0]
            pool = keep or cands
            best = min(pool, key=lambda c: (1 if c[1] else 0, -dtz_abs(c)))

        move, _z, child_wdl, child_dtz, _tb = best
        return (move, child_wdl, child_dtz if child_dtz is not None else 0)

    def solve(self, board: chess.Board) -> List[AnalyzedMove]:
        if not self._opened:
            return []
        # 根局面本身必须可探，否则不认命中
        if self.probe_wdl(board) is None:
            return []
        
        piece_count = len(board.piece_map())
        if piece_count > 7:
            return []
        if self._syzygy_available:
            return self._solve_with(board, use="syzygy")
        return []

    def solve_step(self, board: chess.Board) -> Optional[AnalyzedMove]:
        if not self._opened:
            return None
        wdl = self.probe_wdl(board)
        if wdl is None:
            return None
        if self._syzygy_available:
            pair = self._best_move_syzygy(board)
        else:
            return None
        if pair is None:
            return None
        move, move_wdl, _ = pair
        return AnalyzedMove(
            move=move,
            score=None,
            is_only_move=(wdl == 2 and move_wdl < 2) or (wdl == -2),
            source="syzygy",
        )

    def _solve_with(self, board: chess.Board, use: str = "syzygy") -> List[AnalyzedMove]:
        temp, result, max_steps = board.copy(), [], 100
        for _ in range(max_steps):
            if temp.is_game_over():
                break
            wdl, pair = self.probe_wdl(temp), self._best_move_syzygy(temp)
            if pair is None:
                break
            move, move_wdl, _info = pair
            if wdl is None:
                wdl = 2 if move_wdl is not None and move_wdl < 0 else (move_wdl if move_wdl is not None else 0)
            result.append(AnalyzedMove(
                move=move,
                score=None,
                is_only_move=(wdl == 2 and move_wdl < 2) or (wdl == -2),
                source="syzygy",
            ))
            temp.push(move)
        return result

    def is_hit(self, board: chess.Board) -> bool:
        """ 表库里面是否存在这个局面 """
        return self.probe_wdl(board) is not None

    def is_draw(self, board: chess.Board) -> Optional[bool]:
        """
        检测局面在五十步规则约束下是否为和棋
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
                # board.halfmove_clock为上一次零着后走的半步数
                remaining = 100 - board.halfmove_clock
                if abs(dtz) > remaining:
                    return True
        return False