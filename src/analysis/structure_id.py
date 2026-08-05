"""兵形原型识别器（决策管线，ADR-020）。

纯函数、零引擎依赖，对齐 `insight_extractor.py` 的失败安全设计。
职责：从 FEN 确定性识别兵形原型（structure_kb 第一级），供决策管线
（挖掘器 G7 / P0-B1 / 阶段 2）使用。

颜色归一化（P22）：入口统一 `if not board.turn: board = board.mirror()`——
KB 只写「走子方视角」（mover/opponent 语义），识别在归一化棋盘上进行。
原型 id 本身与颜色无关，无需反变换。

失败安全：任何异常返回 (None, 0.0, {})，不抛错。

v1 覆盖 6 原型：carlsbad（卡尔斯巴德）、iqp（孤后兵）、hanging（悬兵）、
maroczy（马洛齐束缚）、stonewall（石墙）、majority（通路兵/多数兵）。
识别优先级「具体 → 泛」（中心兵形类优先于翼侧兵数类）：
carlsbad → iqp → hanging → maroczy → stonewall → majority。
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


def _pawn_on(board: chess.Board, color: int, sq_name: str) -> bool:
    """指定格是否有该方兵（sq_name 如 "d4"）。"""
    sq = chess.parse_square(sq_name)
    p = board.piece_at(sq)
    return p is not None and p.piece_type == chess.PAWN and p.color == color


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


def _hanging_check(board: chess.Board) -> bool:
    """悬兵：mover c4+d4 双兵并列且 b3/e3 无 mover 兵（双兵无支撑）；
    或对方 c5+d5 双兵且 b6/e6 无对方兵（黑方悬兵——归一化后仍显对方
    颜色，KB identify 的「或对方 c5+d5」）。

    守卫查**格**（b3/e3）而非线：b2 兵不保护 c4（兵保护是斜前方），
    悬兵仍成立（Berlin 实测——首版按线检查漏识别）。
    """
    for color, sq_pair, guard_sqs in (
        (_MOVER, ("c4", "d4"), ("b3", "e3")),
        (_OPPONENT, ("c5", "d5"), ("b6", "e6")),
    ):
        if _pawn_on(board, color, sq_pair[0]) and _pawn_on(board, color, sq_pair[1]):
            if all(not _pawn_on(board, color, g) for g in guard_sqs):
                return True
    return False


def _maroczy_check(board: chess.Board) -> bool:
    """马洛齐束缚：mover c4+e4 双中心兵控制 d5 + 对方 d 线兵在 d6
    （rank 5，未过 d5——d6 兵被 c4/e4 束缚）。"""
    if not (_pawn_on(board, _MOVER, "c4") and _pawn_on(board, _MOVER, "e4")):
        return False
    return any(
        chess.square_file(sq) == _FD and chess.square_rank(sq) == 5
        for sq in board.pieces(chess.PAWN, _OPPONENT)
    )


def _stonewall_check(board: chess.Board) -> bool:
    """石墙：对方 d5+f5 双前伸兵（e6 支撑）；或 mover d4+f4（e3 支撑）。
    归一化后主要查对方版（荷兰石墙黑方常见）——mover 版（伦敦/白方
    石墙）同样识别。
    """
    if (_pawn_on(board, _OPPONENT, "d5") and _pawn_on(board, _OPPONENT, "f5")
            and _pawn_on(board, _OPPONENT, "e6")):
        return True
    if (_pawn_on(board, _MOVER, "d4") and _pawn_on(board, _MOVER, "f4")
            and _pawn_on(board, _MOVER, "e3")):
        return True
    return False


def _majority_check(board: chess.Board) -> bool:
    """多数兵：mover 在翼侧有兵多数（后翼 a-c 线或王翼 f-h 线
    己方兵数 ≥3 且对方同翼 ≤2），且多数翼至少一兵未过中线（可推进）。
    翼侧兵数类是六原型中最泛的判据——必须排最后（具体原型优先），
    且要求「可推进」收紧，控制误报。
    """
    def wing_count(color: int, files: tuple) -> int:
        return sum(1 for sq in board.pieces(chess.PAWN, color)
                   if chess.square_file(sq) in files)

    for files, past_rank in (((0, 1, 2), 4), ((5, 6, 7), 4)):
        # (后翼 a-c, 王翼 f-h) 两翼；past_rank = 该翼过中线的 rank
        # （0-based，对齐 structure_features 的 _MID_RANK=4）
        wq = wing_count(_MOVER, files)
        bq = wing_count(_OPPONENT, files)
        if wq < 3 or bq >= wq:
            continue
        # 可推进：多数翼至少一兵未过中线（rank < past_rank 0-based）
        movable = any(
            chess.square_rank(sq) < past_rank
            for sq in board.pieces(chess.PAWN, _MOVER)
            if chess.square_file(sq) in files
        )
        if movable:
            return True
    return False


def applicable_mover_side(board: chess.Board, archetype: str) -> Optional[str]:
    """决策点走子方在该原型里扮演哪一方角色，用于筛选适用计划。

    返回 `"mover"` / `"opponent"` / None：
    - `"mover"`：走子方是**结构特征的持有方**（如持孤后兵、持悬兵、
      摆石墙的一方），适用 KB 中 `mover_side == "mover"` 的计划；
    - `"opponent"`：走子方是**面对该结构的一方**（如围攻对方孤兵），
      适用 `mover_side == "opponent"` 的计划；
    - None：该原型的角色不可判或无需区分（判据只从走子方视角成立，
      如 carlsbad / maroczy / majority——KB 里这些原型的全部计划都是
      `mover_side == "mover"`，不存在错选可能）。

    为什么必须有这个函数（08.04 补，Critical）：
    `_iqp_check` / `_hanging_check` / `_stonewall_check` 都是「**任一方**
    持有该结构即命中」，而 KB 的 iqp 条目里 4 条计划分属两种角色
    （施压方 2 条 + 持有方 2 条）。管线此前不读 `mover_side`，把**双方的
    计划**一起当成「走子方的几条路」讲——实测 iqp demo（黑方走子、白持
    d4 孤兵）四条全讲，其中「保持孤兵」「推进兑掉孤兵」是**白方**的选择，
    黑方根本执行不了。这是会教错棋的硬事实错（SPEC §8 零容忍），不是
    表达问题：观众照着讲解去走，走的是对手的计划。

    判据与 P0-full 探针的「计划角色 ↔ 局面角色」匹配约定同源
    （见 `p0_full_probe.run_a2_goals` docstring）：谁的 d 线兵孤立，
    谁就是 iqp 的持有方；悬兵/石墙同理按兵形归属判。
    """
    try:
        b = _mirror_normalize(board.copy())
        if archetype == "iqp":
            if _isolated_d_pawn_squares(b, _MOVER):
                return "mover"          # 走子方自己持孤兵
            if _isolated_d_pawn_squares(b, _OPPONENT):
                return "opponent"       # 对方持孤兵，走子方是施压方
            return None
        if archetype == "hanging":
            if _pawn_on(b, _MOVER, "c4") and _pawn_on(b, _MOVER, "d4"):
                return "mover"
            if _pawn_on(b, _OPPONENT, "c5") and _pawn_on(b, _OPPONENT, "d5"):
                return "opponent"
            return None
        if archetype == "stonewall":
            if (_pawn_on(b, _MOVER, "d4") and _pawn_on(b, _MOVER, "f4")
                    and _pawn_on(b, _MOVER, "e3")):
                return "mover"
            if (_pawn_on(b, _OPPONENT, "d5") and _pawn_on(b, _OPPONENT, "f5")
                    and _pawn_on(b, _OPPONENT, "e6")):
                return "opponent"
            return None
        # carlsbad / maroczy / majority：判据本身锚定走子方，无角色歧义
        return None
    except Exception:  # noqa: BLE001
        return None


def detect_pawn_structure(board: chess.Board) -> Tuple[Optional[str], float, dict]:
    """识别兵形原型。

    返回 (archetype_id | None, confidence, pawn_features)。
    - archetype_id：structure_kb 的键（"carlsbad"/"iqp"/"hanging"/
      "maroczy"/"stonewall"/"majority"），未命中 None；
    - confidence：纯几何命中记 1.0（后续接入 opening_hints 交叉时细化）；
    - pawn_features：识别用兵形事实（供诊断/交叉验证，非 P16 结构特征向量）。

    优先级「具体 → 泛」：中心兵形类（carlsbad/iqp/hanging/maroczy/
    stonewall）先于翼侧兵数类（majority）——多数兵是任何局面的泛化
    属性，具体结构优先标注。
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
        if _hanging_check(b):
            return "hanging", 1.0, features
        if _maroczy_check(b):
            return "maroczy", 1.0, features
        if _stonewall_check(b):
            return "stonewall", 1.0, features
        if _majority_check(b):
            return "majority", 1.0, features
        return None, 0.0, features
    except Exception:
        return None, 0.0, {}


if __name__ == "__main__":
    # 自检：A1 探针实测 FEN + 镜像 + 负样本（v1 覆盖 6 原型）
    cases = [
        ("卡尔斯巴德", "carlsbad",
         "r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/2NBPN2/PPQ2PPP/R4RK1 w - - 8 11"),
        ("卡尔斯巴德镜像(黑方走子)", "carlsbad",
         chess.Board("r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/2NBPN2/PPQ2PPP/R4RK1 w - - 8 11").mirror().fen()),
        ("卡尔斯巴德 Grünfeld", "carlsbad",
         "4r3/pp2r1k1/2p2ppn/3pP2p/3P3P/2NRP2K/PP3P2/2R5 w - - 0 31"),
        ("IQP(对方持孤兵)", "iqp",
         "r1bq1rk1/pp2bppp/2n2n2/3p4/N7/5NP1/PP2PPBP/R1BQ1RK1 w - - 2 11"),
        ("IQP(走子方持孤兵,镜像)", "iqp",
         "r1bq1rk1/pp2ppbp/5np1/n7/3P4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 2 11"),
        ("悬兵 Alapin", "hanging",
         "2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/1R2R1K1 w - - 1 21"),
        ("悬兵 Berlin(白b2兵存在)", "hanging",
         "4r1k1/pp1q1ppp/2p5/3pQ3/2PPb3/8/PP3PPP/4RBK1 w - - 1 22"),
        ("马洛齐束缚", "maroczy",
         "r2q1rk1/pp2ppbp/3pbnp1/8/2P1P3/2N1B3/PP1QBPPP/R3K2R w KQ - 5 11"),
        ("荷兰石墙", "stonewall",
         "rn3rk1/pb2q1pp/1ppbpn2/3pNp2/2PP4/1P4P1/PB1NPPBP/R2Q1RK1 w - - 2 11"),
        ("多数兵 Dragon", "majority",
         "r2qr3/3bRpk1/p2p2p1/3P2Qp/1p6/1N3P2/PPP3PP/1K1R4 w - - 1 21"),
        ("意大利开局", None,
         "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("初始局面", None, chess.STARTING_FEN),
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
