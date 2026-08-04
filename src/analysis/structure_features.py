"""结构特征向量与结构目标求值（决策管线，ADR-020）。

P16 定稿（2026-08-03，12 维固定向量）的宿主模块。职责：

- `structural_features(board)`：P16 12 维特征向量（走子方视角，0-1 有界）
- `line_features(line)`：一条线的特征序列（阶段 5 趋势采样 / P8 分歧深度用）
- `goal_satisfied(board, structural_goal)`：A2 结构目标达成判定
- `feature_distance(fv_a, fv_b)`：A3 可分离性 / P8 分歧深度的距离度量

**为什么必须跨原型固定同一组维度（P16）**：推进型计划会把局面从原型 X
变成原型 Y（如 IQP 推 d5 兑掉后孤兵消失），若特征随原型变，两条线之间的距离
就没有定义，A2/A3/P8 三项判据全部塌掉。维度集定稿后**不可边做边加**。

颜色归一化（P22）：**本模块自行归一化，调用方不需要预先 mirror**。
原设计把归一化责任推给调用方，但三个调用方全都没做，黑方走子时 12 维中
2 维静默算错（详见 `_raw_features` docstring）。现收口于 `_raw_features`。

**沿线采样必须传 `mover_color` 锚定视角**：`line_features` / 趋势采样这类
逐着推演的场景，`board.turn` 每着交替。若不显式锚定决策点的走子方，
特征的「我方/对方」语义会每着翻转，序列变成两方视角交替的锯齿，
趋势单调性与分歧深度全部失真。整条线必须共用同一个 `mover_color`。

失败安全：任何异常返回全零向量（距离语义：与任何局面距离最大，保守）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import chess

# ---------------------------------------------------------------- 维度元数据
# P16 定稿：每维的 (名称, 含义, 上界)。上界用于归一化到 0-1。
# 顺序即向量索引，**定稿后不可增删改序**（A2/A3/P8/阶段 5 全部依赖）。

DIMS: List[Tuple[str, str, float]] = [
    ("opp_isolated_qside",  "对方后翼（a-c 线）孤立兵数", 3.0),
    ("opp_isolated_center", "对方中心（d-e 线）孤立兵数", 2.0),
    ("opp_isolated_kside",  "对方王翼（f-h 线）孤立兵数", 3.0),
    ("opp_backward",         "对方后退兵总数",           4.0),
    ("passed_diff",          "通路兵数差（己-彼）",      4.0),
    ("mover_pawns_past_mid", "己方越过中线兵数",         5.0),
    ("pawn_islands_diff",    "兵岛数差（己-彼）",        4.0),
    ("open_files",           "开放线数（无兵线）",       8.0),
    ("half_open_own",        "己方半开放线数（线上仅对方兵）", 8.0),
    ("outposts",             "己方前哨轻子数",           4.0),
    ("knight_bishop_diff",   "轻子对比（己方马-象数差）", 4.0),
    ("opp_king_exposure",    "对方王暴露度（王周围格中对方兵不控制的格数）", 8.0),
]

DIM_NAMES: List[str] = [d[0] for d in DIMS]
DIM_COUNT = len(DIMS)

_MOVER = chess.WHITE
_OPPONENT = chess.BLACK
_MID_RANK = 4          # 白方越中线：rank >= 4（0-based，第 5 横线起）
_OPP_HALF = 4          # 白方轻子前哨：rank >= 4


def _pawn_files(board: chess.Board, color: int) -> set:
    return {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}


def _isolated_pawns(board: chess.Board, color: int,
                    files: Tuple[int, ...]) -> int:
    """指定线范围内该方的孤立兵数（相邻线无己方兵）。"""
    pawn_files = _pawn_files(board, color)
    cnt = 0
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        if f not in files:
            continue
        if (f - 1) not in pawn_files and (f + 1) not in pawn_files:
            cnt += 1
    return cnt


def _backward_pawns(board: chess.Board, color: int) -> int:
    """后退兵数。

    定义（经典）：兵无法前进（前方格被对方兵攻击或占据），且推进格
    得不到相邻线己方兵的掩护（无法通过推进获得保护）。
    """
    step = 8 if color == chess.WHITE else -8
    opp = not color
    cnt = 0
    for sq in board.pieces(chess.PAWN, color):
        ahead = sq + step
        if not 0 <= ahead < 64:
            continue
        # 前方格被对方兵攻击 → 无法推进
        if not (board.attackers_mask(opp, ahead)
                & board.pieces_mask(chess.PAWN, opp)):
            continue
        # 推进格是否有相邻线己方兵掩护（己方兵攻击该格）
        protected = (board.attackers_mask(color, ahead)
                     & board.pieces_mask(chess.PAWN, color))
        if not protected:
            cnt += 1
    return cnt


def _passed_pawns(board: chess.Board, color: int) -> int:
    """通路兵数：同 file 及相邻 file 无对方兵在其前方。"""
    opp = not color
    cnt = 0
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        blocked = False
        for o_sq in board.pieces(chess.PAWN, opp):
            of = chess.square_file(o_sq)
            if abs(of - f) > 1:
                continue
            orank = chess.square_rank(o_sq)
            if color == chess.WHITE and orank > r:
                blocked = True
                break
            if color == chess.BLACK and orank < r:
                blocked = True
                break
        if not blocked:
            cnt += 1
    return cnt


def _pawn_islands(board: chess.Board, color: int) -> int:
    """兵岛数：相邻 file 有兵的连续兵组算一个岛。"""
    files = sorted(_pawn_files(board, color))
    if not files:
        return 0
    islands = 1
    for i in range(1, len(files)):
        if files[i] != files[i - 1] + 1:
            islands += 1
    return islands


def _outposts(board: chess.Board, color: int) -> int:
    """前哨轻子数：轻子在对方半场、被己方兵保护、且对方兵不能攻击。"""
    opp = not color
    cnt = 0
    for pt in (chess.KNIGHT, chess.BISHOP):
        for sq in board.pieces(pt, color):
            if chess.square_rank(sq) < _OPP_HALF:
                continue
            if not (board.attackers_mask(color, sq)
                    & board.pieces_mask(chess.PAWN, color)):
                continue  # 无己方兵保护
            if board.attackers_mask(opp, sq) & board.pieces_mask(
                    chess.PAWN, opp):
                continue  # 可被对方兵攻击
            cnt += 1
    return cnt


def _king_exposure(board: chess.Board, opp_king_sq: Optional[int]) -> int:
    """对方王暴露度：王周围格中对方兵不控制的格数。

    王周围 8 邻格（棋盘内），统计其中不被对方兵攻击的格数——王翼兵墙
    缺失程度（王前兵推进/被兑后的暴露面）。
    """
    if opp_king_sq is None:
        return 0
    cnt = 0
    for sq in chess.scan_forward(chess.BB_KING_ATTACKS[opp_king_sq]):
        if not (board.attackers_mask(_OPPONENT, sq)
                & board.pieces_mask(chess.PAWN, _OPPONENT)):
            cnt += 1
    return cnt


def _mirror_normalize(board: chess.Board) -> chess.Board:
    """颜色归一化（P22）：黑方走子局面镜像为白方走子。

    与 `structure_id._mirror_normalize` 同一语义（`board.mirror()` =
    上下翻转 + 颜色互换），两处刻意保持一致实现，不跨模块 import
    以免 analysis 层内部互相依赖。
    """
    if board.turn == chess.BLACK:
        return board.mirror()
    return board


def _raw_features(board: chess.Board,
                  mover_color: Optional[bool] = None) -> Dict[str, int]:
    """P16 12 维的**原始计数**（单一事实来源）。

    `structural_features`（归一化）与 `goal_satisfied`（谓词求值）都从这里取数，
    禁止两处各自实现——goal 用真实计数（"opp_isolated_qside >= 1"），
    向量用归一化值，两者是同一组数字的两种投影。

    **颜色归一化在此处收口（P22）**：本函数用 `_MOVER = WHITE` /
    `_OPPONENT = BLACK` 硬编码走子方视角，故必须先把非白方视角的局面镜像。
    原实现把归一化责任推给调用方（docstring 写"调用方保证已归一化"），
    但三个调用方（decision_pipeline / consequence_projector /
    decision_builder）**全都没做**，导致黑方走子时 12 维中 2 维静默算错：
    `opp_isolated_center` 把走子方自己的孤兵记成对手的、`pawn_islands_diff`
    符号反向。这类错误不抛异常、不留日志，A2/A3/P8 三项判据全部建在错数上。

    归一化收口在这里而不是各调用方，是因为这是两个消费者的唯一共同取数点——
    在此处做一次，`structural_features` 与 `goal_satisfied` 自动都正确。

    **`mover_color` 必须显式锚定视角，不能一律用 `board.turn`**：
    `line_features` 沿线逐着 push，走子方每着交替。若按 `board.turn` 归一化，
    特征向量的"我方/对方"语义会**每着翻转一次**，得到的序列不是一条线的结构
    演化而是两方视角交替的锯齿——趋势单调性（阶段 5）与分歧深度（P8）全部失真。
    视角必须锚定在**决策点的走子方**，整条线共用同一个 `mover_color`。
    `None` 时退回 `board.turn`（单个局面的自然语义）。
    """
    if mover_color is None:
        mover_color = board.turn
    if mover_color == chess.BLACK:
        board = board.mirror()
    mover_pawn_files = _pawn_files(board, _MOVER)
    opp_pawn_files = _pawn_files(board, _OPPONENT)
    return {
        "opp_isolated_qside": _isolated_pawns(board, _OPPONENT, (0, 1, 2)),
        "opp_isolated_center": _isolated_pawns(board, _OPPONENT, (3, 4)),
        "opp_isolated_kside": _isolated_pawns(board, _OPPONENT, (5, 6, 7)),
        "opp_backward": _backward_pawns(board, _OPPONENT),
        "passed_diff": (_passed_pawns(board, _MOVER)
                        - _passed_pawns(board, _OPPONENT)),
        "mover_pawns_past_mid": sum(
            1 for sq in board.pieces(chess.PAWN, _MOVER)
            if chess.square_rank(sq) >= _MID_RANK),
        "pawn_islands_diff": (_pawn_islands(board, _MOVER)
                              - _pawn_islands(board, _OPPONENT)),
        "open_files": sum(
            1 for f in range(8)
            if f not in mover_pawn_files and f not in opp_pawn_files),
        "half_open_own": sum(
            1 for f in range(8)
            if f not in mover_pawn_files and f in opp_pawn_files),
        "outposts": _outposts(board, _MOVER),
        "knight_bishop_diff": (len(board.pieces(chess.KNIGHT, _MOVER))
                               - len(board.pieces(chess.BISHOP, _MOVER))),
        "opp_king_exposure": _king_exposure(board, board.king(_OPPONENT)),
    }


def structural_features(board: chess.Board,
                        mover_color: Optional[bool] = None) -> List[float]:
    """P16 12 维结构特征向量（走子方视角，每维 0-1 有界）。

    返回长度 12 的 float 列表，索引与 DIM_NAMES 对应。颜色归一化在
    `_raw_features` 内部完成，调用方无需自行 mirror。

    `mover_color`：视角锚点。单个局面可省略（用 `board.turn`）；**沿一条线
    采样时必须显式传决策点的走子方**，否则视角每着翻转（见 `_raw_features`）。
    失败安全：异常返回全零。
    """
    try:
        raw = _raw_features(board, mover_color)
        return [max(0.0, min(1.0, raw[name] / bound))
                for name, _, bound in DIMS]
    except Exception:
        return [0.0] * DIM_COUNT


def line_features(line: List[chess.Move], initial_board: chess.Board,
                  ) -> List[List[float]]:
    """一条线的特征序列：从 initial_board 逐步推演，每个节点采一次特征。

    **视角锚定在 initial_board 的走子方**（决策点的一方），整条序列共用。
    这是必须的：沿线 push 会让 `board.turn` 每着交替，若各节点按自身 turn
    归一化，"我方/对方"语义每着翻转，序列变成两方视角交替的锯齿，
    趋势单调性（阶段 5）与分歧深度（P8）都会失真。
    """
    mover = initial_board.turn
    out = []
    b = initial_board.copy()
    out.append(structural_features(b, mover))
    for mv in line:
        try:
            b.push(mv)
        except Exception:
            break
        out.append(structural_features(b, mover))
    return out


def goal_satisfied(board: chess.Board, goal: Dict[str, str],
                   mover_color: Optional[bool] = None) -> bool:
    """结构目标达成判定（A2）：goal 形如 {"dim": ">=1"} / {"dim": "==0"}。

    谓词在 `_raw_features` 的**原始计数**上求值（goal 用真实计数，如
    "opp_isolated_qside >= 1" 表示对方后翼出现至少 1 个孤立兵）。
    未知维度或未知谓词：保守判不满足（失败安全）。

    `mover_color`：视角锚点。A2 判「计划是否达成结构目标」时应传**决策点**
    走子方——线末局面的 `turn` 可能是对手，按它归一化会把目标判到对方头上。

    **OR 组表达**：goal 可含特殊键 `"any"`（列表，元素为普通 goal dict），
    任一子 goal 满足即整体满足——FINDINGS A2 示例「对方 c 线或 d 线出现
    孤立/后退兵」是 OR 语义，dict 的 AND 表达不了（如
    `{"any": [{"opp_isolated_qside": ">=1"}, {"opp_backward": ">=1"}]}`）。
    """
    try:
        if "any" in goal:
            return any(goal_satisfied(board, g, mover_color)
                       for g in goal["any"])
        raw = _raw_features(board, mover_color)
        for dim, pred in goal.items():
            if dim not in raw:
                return False
            val = raw[dim]
            if pred.startswith(">="):
                if val < int(pred[2:]):
                    return False
            elif pred.startswith("=="):
                if val != int(pred[2:]):
                    return False
            else:
                return False  # 未知谓词：保守判不满足
        return True
    except Exception:
        return False


def feature_distance(fv_a: List[float], fv_b: List[float]) -> float:
    """特征向量距离（加权曼哈顿，权重初版全 1；A3 自校准时可调权重）。"""
    if len(fv_a) != DIM_COUNT or len(fv_b) != DIM_COUNT:
        return float(DIM_COUNT)  # 维度不符：最大距离（保守）
    return sum(abs(a - b) for a, b in zip(fv_a, fv_b))


if __name__ == "__main__":
    # 自检：卡尔斯巴德 / IQP 局面特征与 goal 谓词
    import json
    sys_path = None
    cases = [
        ("卡尔斯巴德", "r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/2NBPN2/PPQ2PPP/R4RK1 w - - 8 11"),
        ("IQP 施压", "r1bq1rk1/pp2bppp/2n2n2/3p4/N7/5NP1/PP2PPBP/R1BQ1RK1 w - - 2 11"),
        ("初始局面", chess.STARTING_FEN),
    ]
    for name, fen in cases:
        b = chess.Board(fen)
        fv = structural_features(b)
        print(f"{name}: {[round(x, 2) for x in fv]}")
    # goal 谓词自检
    b = chess.Board("r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/2NBPN2/PPQ2PPP/R4RK1 w - - 8 11")
    print("carlsbad 少数派 goal(对方后翼孤兵>=1):",
          goal_satisfied(b, {"opp_isolated_qside": ">=1"}))
    print("carlsbad 中心突破 goal(己方兵过中线>=1):",
          goal_satisfied(b, {"mover_pawns_past_mid": ">=1"}))
    b2 = chess.Board("r1bq1rk1/pp2bppp/2n2n2/3p4/N7/5NP1/PP2PPBP/R1BQ1RK1 w - - 2 11")
    print("iqp 施压 goal(对方中心孤兵>=1):",
          goal_satisfied(b2, {"opp_isolated_center": ">=1"}))
    print("距离(同局面):", feature_distance(
        structural_features(b), structural_features(b)))
