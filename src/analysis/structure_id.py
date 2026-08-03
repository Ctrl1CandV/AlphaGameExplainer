"""兵形原型识别器（决策管线，ADR-020）。

纯函数、零引擎依赖，对齐 `insight_extractor.py` 的失败安全设计。
职责：从 FEN 确定性识别兵形原型（structure_kb 第一级），供决策管线
（挖掘器 G7 / P0-B1 / 阶段 2）使用。

颜色归一化（P22）：入口统一 `if not board.turn: board = board.mirror()`——
KB 只写「走子方视角」（mover/opponent 语义），识别在归一化棋盘上进行。
原型 id 本身与颜色无关，无需反变换。

失败安全：任何异常返回 (None, 0.0, {})，不抛错。

v0 覆盖 2 原型（最小路径）：carlsbad（卡尔斯巴德）、iqp（孤后兵）。
识别不到返回 None——PGN 源矿脉无限，宁可漏识别不可误识别（丢弃成本为零）。
"""
from __future__ import annotations

from typing import Optional, Tuple

import chess

# 走子方视角常量（归一化后 mover=白方）
_MOVER = chess.WHITE
_OPPONENT = chess.BLACK
_FD, _FC, _FE = 3, 2, 4  # d/c/e 线索引（python-chess 无 FILE_D 常量）


def _mirror_normalize(board: chess.Board) -> chess.Board:
    """颜色归一化（P22）：黑方走子局面镜像为白方走子。"""
    if board.turn == chess.BLACK:
        return board.mirror()
    return board


def _pawn_files(board: chess.Board, color: int) -> set:
    """某方兵占据的线集合（file → 该方兵存在）。"""
    return {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}


def _isolated_d_pawn_squares(board: chess.Board, color: int) -> list:
    """某方 d 线孤立兵所在格（相邻 c/e 线无该方兵）。

    限定 rank 4/5（孤后兵的中盘定义域）——马洛齐的 d6 孤立兵不是 IQP，
    那是「孤立的中心兵」而非孤后兵结构。
    """
    out = []
    d_file = _FD
    files = _pawn_files(board, color)
    for sq in board.pieces(chess.PAWN, color):
        if chess.square_file(sq) != d_file:
            continue
        rank = chess.square_rank(sq)
        if rank not in (3, 4):  # rank 4/5（rank 0-based）→ d4/d5
            continue
        if (d_file - 1) not in files and (d_file + 1) not in files:
            out.append(sq)
    return out


def _carlsbad_check(board: chess.Board) -> bool:
    """卡尔斯巴德：mover 有 d4 兵且 c 线无兵；对方有 d5 兵、e 线无兵，
    且对方 c 线兵未推进到 c5（推进到 c5 是塔拉什/卡尔斯巴德 c5 反击，
    结构语义已变，宁可漏识别）。
    """
    mover_d4 = any(
        chess.square_file(sq) == _FD
        and chess.square_rank(sq) == 3
        for sq in board.pieces(chess.PAWN, _MOVER)
    )
    if not mover_d4:
        return False
    if _FC in _pawn_files(board, _MOVER):
        return False
    opp_d5 = any(
        chess.square_file(sq) == _FD
        and chess.square_rank(sq) == 4
        for sq in board.pieces(chess.PAWN, _OPPONENT)
    )
    if not opp_d5:
        return False
    if _FE in _pawn_files(board, _OPPONENT):
        return False
    # 对方 c 线兵只允许在 c6/c7（rank 5/6），推进到 c5（rank 4）判为塔拉什类
    for sq in board.pieces(chess.PAWN, _OPPONENT):
        if chess.square_file(sq) == _FC:
            if chess.square_rank(sq) < 5:
                return False
    return True


def _iqp_check(board: chess.Board) -> bool:
    """孤后兵：任一方的 d4/d5 孤立兵（mover 持孤兵 或 对方持孤兵）。"""
    if _isolated_d_pawn_squares(board, _MOVER):
        return True
    if _isolated_d_pawn_squares(board, _OPPONENT):
        return True
    return False


def detect_pawn_structure(board: chess.Board) -> Tuple[Optional[str], float, dict]:
    """识别兵形原型。

    返回 (archetype_id | None, confidence, pawn_features)。
    - archetype_id：structure_kb 的键（"carlsbad" / "iqp"），未命中 None；
    - confidence：v0 纯几何命中记 1.0（后续接入 opening_hints 交叉时细化）；
    - pawn_features：识别用兵形事实（供诊断/交叉验证，非 P16 结构特征向量）。
    """
    try:
        b = _mirror_normalize(board.copy())
        features = {
            "mover_pawn_files": sorted(_pawn_files(b, _MOVER)),
            "opponent_pawn_files": sorted(_pawn_files(b, _OPPONENT)),
            "mover_isolated_d": [chess.square_name(s)
                                 for s in _isolated_d_pawn_squares(b, _MOVER)],
            "opponent_isolated_d": [chess.square_name(s)
                                    for s in _isolated_d_pawn_squares(b, _OPPONENT)],
        }
        if _carlsbad_check(b):
            return "carlsbad", 1.0, features
        if _iqp_check(b):
            return "iqp", 1.0, features
        return None, 0.0, features
    except Exception:
        return None, 0.0, {}


if __name__ == "__main__":
    # 自检：A1 探针实测 FEN + 镜像 + 负样本
    cases = [
        ("卡尔斯巴德", "carlsbad",
         "r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/2NBPN2/PPQ2PPP/R4RK1 w - - 8 11"),
        ("卡尔斯巴德镜像(黑方走子)", "carlsbad",
         chess.Board("r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/2NBPN2/PPQ2PPP/R4RK1 w - - 8 11").mirror().fen()),
        ("IQP(对方持孤兵)", "iqp",
         "r1bq1rk1/pp2bppp/2n2n2/3p4/N7/5NP1/PP2PPBP/R1BQ1RK1 w - - 2 11"),
        ("马洛齐(黑d6孤立兵,非IQP)", None,
         "r2q1rk1/pp2ppbp/3pbnp1/8/2P1P3/2N1B3/PP1QBPPP/R3K2R w KQ - 5 11"),
        ("荷兰石墙", None,
         "rn3rk1/pb2q1pp/1ppbpn2/3pNp2/2PP4/1P4P1/PB1NPPBP/R2Q1RK1 w - - 2 11"),
        ("意大利开局", None,
         "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ]
    import sys
    sys.path.insert(0, ".")
    ok = True
    for name, expect, fen in cases:
        got, conf, _ = detect_pawn_structure(chess.Board(fen))
        status = "PASS" if got == expect else "FAIL"
        if got != expect:
            ok = False
        print(f"[{status}] {name}: 期望 {expect} 实得 {got} (conf={conf})")
    print("全部通过" if ok else "存在失败")
    sys.exit(0 if ok else 1)
