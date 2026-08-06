"""引擎探针与「等强」单一定义。

ADR-020 §决策 7（单一事实来源）：「等强」在全链路只有一个定义。本模块提供
`equivalence_gap()`，M5/M8/可行性闸/A3 全部调它，不得各写一套——否则会出现
FINDINGS-002 P7 那类「挖矿口径 Go 通过、运行时口径 80% 降级」的脱节。

进程管理对齐 `src/solver/stockfish_analyzer.py` 的既有模式：popen_uci → configure
→ finally quit。不复用该模块的函数，因为它是单线求解语义（返回一条 PV），
而这里要的是 MultiPV 分布。
"""
from dataclasses import dataclass, field
from typing import List, Optional
import logging
import os

import chess
import chess.engine
import sys

# 单一事实来源正式化（2026-08-03）：方向与等强迁移到 src/analysis/direction.py，
# 本模块只保留引擎封装与探针逻辑，禁止本地重复实现（FINDINGS-002 §3.8）。
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.analysis.direction import (  # noqa: E402
    DEFAULT_EQUIV_CP,
    equivalence_gap,
    is_equivalent,
    direction_zone,
    direction_score,
    direction_candidates,
)

logging.getLogger("chess.engine").setLevel(logging.CRITICAL)

# 项目根（本文件位于 tools/decision_probe/ 下，上溯两级）。
# 用于把 .env 里的相对 STOCKFISH_PATH 解析成绝对路径，语义对齐
# src/common.py:resolve_path。
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 将杀分的 cp 折算上限。python-chess 要求给 mate_score 才能把 Mate(n) 转成 int。
# 取 10000 远大于任何实战 cp 差，保证「有杀」在数值比较中永远压过「无杀」。
MATE_CP = 10000

# M5 判据阈值（FINDINGS-002 P18：这是本探针要校准的核心参数，不是定论）。
# 语义：top-1 比 top-2 好出这么多，就认为局面有「唯一好着」，属战术计算局面而非战略选择局面。
DEFAULT_STANDOUT_CP = 150


# MultiPV 候选数（ADR-020 阶段 4 的 k=4~6，取中）。
DEFAULT_MULTIPV = 5


@dataclass
class PvLine:
    """MultiPV 的一条候选线。"""
    rank: int                      # 1-based，1 = 引擎首选
    move: chess.Move               # 根着
    cp: int                        # 走子方视角的 cp（将杀已折算为 ±MATE_CP 量级）
    is_mate: bool                  # 是否将杀分
    mate_in: Optional[int] = None  # 正数=走子方将杀对手，负数=被将杀
    pv: List[chess.Move] = field(default_factory=list)


@dataclass
class MultiPvResult:
    """一次 MultiPV 分析的完整结果。"""
    fen: str
    lines: List[PvLine]
    depth: int

    @property
    def best(self) -> Optional[PvLine]:
        return self.lines[0] if self.lines else None

    @property
    def second(self) -> Optional[PvLine]:
        return self.lines[1] if len(self.lines) > 1 else None


class EngineProbe:
    """Stockfish MultiPV 探针（上下文管理器，保证进程回收）。

    `threads` 默认 1（PLAN-010 阶段 1 对齐，HANDOFF-001 §5.4 记录的遗留问题）。
    多线程 Stockfish 的搜索结果本质不可复现——各线程共享置换表，写入顺序
    随 OS 调度而变，同一局面同一深度每次跑出的 PV 都可能不同（实测见
    `src/solver/branch_explorer._open_engine` docstring，maroczy 决策点
    depth=14：Threads=2 连跑 3 次得到 3 条不同的线）。本探针被 M5 系列冒烟
    脚本（m5_smoke/opening_m5_probe/pgn_m5_probe）复用，这些脚本产出的
    存活率数据同样建立在线的内容之上，不对齐则结果不可复现、不可比较。
    产品链路（`branch_explorer._open_engine`/`stockfish_analyzer.py`）
    早已统一 `Threads: 1`，这里是对齐既有约定，不是新发明。
    """

    def __init__(self, sf_path: str, threads: int = 1, hash_mb: int = 256):
        self.sf_path = sf_path
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def __enter__(self) -> "EngineProbe":
        if not self.sf_path or not os.path.isfile(self.sf_path):
            raise FileNotFoundError(f"Stockfish 不可用: {self.sf_path!r}")
        self._engine = chess.engine.SimpleEngine.popen_uci(self.sf_path)
        self._engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
        return self

    def __exit__(self, *exc) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def multipv(self, board: chess.Board, k: int = 5, depth: int = 16) -> MultiPvResult:
        """MultiPV 分析，返回走子方视角的候选线列表（按 rank 升序）。

        k 会被合法着数截断——残局末端可能只有 1~2 个合法着，请求 5 条会拿到更少，
        调用方须按实际条数判断而非假定拿满 k 条。
        """
        if self._engine is None:
            raise RuntimeError("EngineProbe 未进入上下文")

        legal_count = board.legal_moves.count()
        if legal_count == 0:
            return MultiPvResult(fen=board.fen(), lines=[], depth=depth)
        effective_k = min(k, legal_count)

        infos = self._engine.analyse(
            board, chess.engine.Limit(depth=depth), multipv=effective_k
        )
        # multipv=1 时 python-chess 可能返回单个 dict 而非 list，统一成 list
        if isinstance(infos, dict):
            infos = [infos]

        lines: List[PvLine] = []
        for idx, info in enumerate(infos):
            score_obj = info.get("score")
            pv = list(info.get("pv") or [])
            if score_obj is None or not pv:
                continue
            rel = score_obj.relative
            mate_in = rel.mate()
            lines.append(PvLine(
                rank=info.get("multipv", idx + 1),
                move=pv[0],
                cp=rel.score(mate_score=MATE_CP),
                is_mate=mate_in is not None,
                mate_in=mate_in,
                pv=pv,
            ))

        lines.sort(key=lambda ln: ln.rank)
        return MultiPvResult(fen=board.fen(), lines=lines, depth=depth)


# ---------------------------------------------------------------------------
# M5：无强制战术判定
# ---------------------------------------------------------------------------

@dataclass
class M5Verdict:
    """M5 判定结果。`passed=True` 表示「无强制战术」，即适合做战略讲解。"""
    passed: bool
    reason: str
    top1_cp: Optional[int] = None
    top2_cp: Optional[int] = None
    gap_cp: Optional[int] = None
    legal_count: int = 0
    is_mate_line: bool = False
    in_check: bool = False
    # 本次判定所用的 MultiPV 结果。带出来让调用方能直接做 M8/方向统计，
    # 不必为拿候选线再跑一次引擎（引擎调用是本链路最贵的一步）。
    # 前三条判据（终局/被将军/唯一着）在调用引擎前返回，此字段为 None。
    multipv: Optional[MultiPvResult] = None


def assess_m5(
    board: chess.Board,
    probe: EngineProbe,
    k: int = 5,
    depth: int = 16,
    standout_cp: int = DEFAULT_STANDOUT_CP,
) -> M5Verdict:
    """判定局面是否「无强制战术」（战略选择局面而非战术计算局面）。

    判定链（任一条命中即 M5 不通过）：
      1. 已终局 / 无合法着 —— 没有可选择的余地；
      2. 正在被将军 —— 应招高度受限，是战术处理而非战略选择；
      3. top-1 是将杀分 —— 有强制取胜序列；
      4. 只有 1 个合法着 —— 唯一着，无选择；
      5. top-1 比 top-2 好出 standout_cp 以上 —— 存在「唯一好着」，属战术计算。

    第 5 条是核心判据，也是本探针要用数据校准的参数。它与 M8（是否存在多个
    近等强首着）测的是同一个量的两侧：M5 要求「没有一手独好」，M8 要求
    「至少两手够好」。二者共用同一次 MultiPV 调用，也共用 `equivalence_gap`。
    """
    if board.is_game_over():
        return M5Verdict(False, "已终局", legal_count=0)

    legal_count = board.legal_moves.count()
    in_check = board.is_check()
    if in_check:
        return M5Verdict(False, "走子方正被将军", legal_count=legal_count, in_check=True)
    if legal_count <= 1:
        return M5Verdict(False, f"合法着仅 {legal_count} 个", legal_count=legal_count)

    res = probe.multipv(board, k=k, depth=depth)
    best, second = res.best, res.second
    if best is None:
        return M5Verdict(False, "引擎无有效返回", legal_count=legal_count)

    if best.is_mate:
        return M5Verdict(
            False, f"存在将杀线（M{best.mate_in}）",
            top1_cp=best.cp, legal_count=legal_count, is_mate_line=True,
            multipv=res,
        )
    if second is None:
        return M5Verdict(
            False, "MultiPV 只返回 1 条线",
            top1_cp=best.cp, legal_count=legal_count, multipv=res,
        )

    gap = equivalence_gap(best.cp, second.cp)
    if gap >= standout_cp:
        return M5Verdict(
            False, f"存在唯一好着（首选优于次选 {gap}cp）",
            top1_cp=best.cp, top2_cp=second.cp, gap_cp=gap, legal_count=legal_count,
            multipv=res,
        )

    return M5Verdict(
        True, f"无唯一好着（首选与次选相差 {gap}cp）",
        top1_cp=best.cp, top2_cp=second.cp, gap_cp=gap, legal_count=legal_count,
        multipv=res,
    )


# ---------------------------------------------------------------------------
# 「方向」的唯一定义（ADR-020 §决策 7 单一事实来源）
#
# 三分区按 file 切：后翼 a-c、中心 d-e、王翼 f-h。KB 的 `direction.target_zone`
# 与 M8 的「方向不同」判定必须共用它——否则出现 FINDINGS-002 P0-B3 那类
# 「M8 说有选择、KB 候选全在同一方向」的两套逻辑各说各话。
# 更细的 pawn_files 打分属将来的 direction_score()，建在本函数之上。
# ---------------------------------------------------------------------------
def count_material(board: chess.Board) -> int:
    """盘面棋子总数（含王与兵）。M3 子力窗口用。

    语义是「棋子个数」不是「子力价值」——ADR-020 漏斗 M3（≥18）指个数口径。
    与 `src/chess_utils/material.py` 的价值计分是两个不同的量，不要混用。
    口径与已执行的 M5 冒烟一致（`len(piece_map())`），改动会使历史数据不可比。
    """
    return len(board.piece_map())


def resolve_stockfish() -> str:
    """解析 Stockfish 路径：env `STOCKFISH_PATH` 优先，回退项目根同名 exe。

    相对路径语义对齐 `src/common.py:resolve_path`（相对项目根）。
    调用方需先 `load_dotenv()`，否则读不到 env。
    """
    raw = os.getenv("STOCKFISH_PATH", "").strip().strip('"')
    if raw:
        path = raw if os.path.isabs(raw) else os.path.normpath(
            os.path.join(_PROJECT_ROOT, raw))
        if os.path.isfile(path):
            return path
    fallback = os.path.join(_PROJECT_ROOT, "stockfish-windows-x86-64-avx2.exe")
    return fallback if os.path.isfile(fallback) else ""


def equivalent_first_moves(
    res: MultiPvResult,
    equiv_cp: int = DEFAULT_EQUIV_CP,
) -> tuple:
    """M8：与首选近等强的首着有几个。返回 `(是否 ≥2, 个数)`。

    通过 `equivalence_gap` 换算，与 M5 / 可行性闸 / A3 共用同一「等强」定义。
    """
    if res.best is None:
        return False, 0
    base = res.best.cp
    equiv = [ln for ln in res.lines if equivalence_gap(base, ln.cp) <= equiv_cp]
    return len(equiv) >= 2, len(equiv)


def distinct_direction_count(
    res: MultiPvResult,
    equiv_cp: int = DEFAULT_EQUIV_CP,
) -> int:
    """近等强首着覆盖了几个不同的战略方向。

    这是 M8 的**方向版**判据，比单纯计数更贴近「真有战略选择」的语义：
    两个近等强首着若都指向王翼，那只是同一方向的两种执行，不是战略分歧。
    FINDINGS-002 P0-B3 要求 M8 与 KB `direction` 同源，本函数即该接点。
    """
    if res.best is None:
        return 0
    base = res.best.cp
    zones = {
        direction_zone(ln.move)
        for ln in res.lines
        if equivalence_gap(base, ln.cp) <= equiv_cp
    }
    return len(zones)


def solving_position(fen: str, uci_moves: List[str]) -> Optional[chess.Board]:
    """把 Lichess puzzle 的 FEN 推进到「解题局面」。

    Lichess 约定（与 `src/parser.py:_build_puzzle_data` 一致）：FEN 是对手预备着
    **之前**的局面，`moves[0]` 是对手的预备着，`moves[1:]` 才是解答。所以要谈
    「解题方面对什么选择」，必须先推进预备着。

    这是 FINDINGS-002 P15 的技术前提：不推预备着就测 M5，测的是对手的选择，
    整个冒烟结论都会错。
    """
    try:
        board = chess.Board(fen)
    except ValueError:
        return None
    if not uci_moves:
        return board
    try:
        prelude = chess.Move.from_uci(uci_moves[0])
    except ValueError:
        return None
    if prelude not in board.legal_moves:
        return None
    board.push(prelude)
    return board
