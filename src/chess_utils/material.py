"""子力相关工具函数——签名/计分/差值/中文描述。

合并自 endgame_knowledge._piece_signature / _sig_name、
insight_extractor._material_signature / _sig_name / _material_balance、
storyboard._material_score / _color_name / _side_material_desc，
消除 12+ 处函数级重复。

定义 PIECE_VALUES / PIECE_CN 两个常量为本包权威来源，
common.py 通过 re-export 保持现有调用方不受影响。
"""
import chess
from typing import List

# 子力点值（不含王，王点值无意义）
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

# 棋子类型 → 中文名
PIECE_CN = {
    chess.KING: "王", chess.QUEEN: "后", chess.ROOK: "车",
    chess.BISHOP: "象", chess.KNIGHT: "马", chess.PAWN: "兵",
}


def piece_cn(piece_type) -> str:
    """棋子类型 → 中文名，未知类型返回「子」。"""
    return PIECE_CN.get(piece_type, "子")


def piece_signature(board: chess.Board, color: chess.Color) -> tuple:
    """某一方除王外的子力组成签名，如 ((chess.ROOK, 1), (chess.PAWN, 1))。"""
    counts = {}
    for piece in board.piece_map().values():
        if piece.color == color and piece.piece_type != chess.KING:
            counts[piece.piece_type] = counts.get(piece.piece_type, 0) + 1
    return tuple(sorted(counts.items()))


def signature_name(sig: tuple) -> str:
    """子力签名 → 中文简称，如「一车一兵」「单王」。"""
    if not sig:
        return "单王"
    parts = []
    cn_num = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八"}
    for pt, cnt in sig:
        name = PIECE_CN.get(pt, "?")
        parts.append(f"{cn_num.get(cnt, str(cnt))}{name}" if cnt > 1 else name)
    return "".join(parts)


def material_score(board: chess.Board, color: chess.Color) -> int:
    """某一方子力总点值（不含王）。"""
    total = 0
    for piece in board.piece_map().values():
        if piece.color == color and piece.piece_type != chess.KING:
            total += PIECE_VALUES.get(piece.piece_type, 0)
    return total


def material_balance(board: chess.Board, perspective_color: chess.Color) -> int:
    """perspective 方相对对方的子力点值差（不含王）。

    正数=占优，负数=落后。
    """
    mine = other = 0
    for p in board.piece_map().values():
        if p.piece_type == chess.KING:
            continue
        v = PIECE_VALUES.get(p.piece_type, 0)
        if p.color == perspective_color:
            mine += v
        else:
            other += v
    return mine - other


def color_name(color: chess.Color) -> str:
    """颜色 → 「白方」/「黑方」。"""
    return "白方" if color == chess.WHITE else "黑方"


def side_material_desc(board: chess.Board, color: chess.Color) -> str:
    """某一方除王外的子力中文描述（如「一车一兵」「单王」），永远可得。

    用于开场白介绍双方子力对比，在 KB 未命中的残局下仍能产出准确描述。
    """
    order = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    cn_num = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八"}
    counts = {}
    for piece in board.piece_map().values():
        if piece.color == color and piece.piece_type != chess.KING:
            counts[piece.piece_type] = counts.get(piece.piece_type, 0) + 1
    parts = []
    for pt in order:
        c = counts.get(pt, 0)
        if c > 0:
            parts.append(f"{cn_num.get(c, str(c))}{piece_cn(pt)}")
    return "".join(parts) if parts else "单王"


# 大子（后/车/象/马）类型，用于 absent_pieces 计算。兵/王不标 absent。
MAJOR_PIECE_TYPES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)


def absent_major_pieces(board: chess.Board, sans=None, previously_captured_types=None) -> List[str]:
    """本局面上不存在的大子中文名列表（不分颜色），供 prompt 负面事实注入。

    PLAN-004 阶段 B：消除"提后但局面无后"这类真幻觉。模型在 prompt 中缺少
    "本节点无某子"的显式负面事实时会凭空捏造（KPvK 把兵讲成后、KBNvK 造后等）。
    这里从 board 程序化解析盘面真实拥有的棋子种类，返回不存在的大子中文名。

    升变放行：若本节点 sans 含 =Q/=R/=B/=N，对应新棋子种类视为"存在"
    （即使升变前盘面无该子，本节点内会产生），不计入 absent。

    前序被吃放行（PLAN-004 B peer_review O1 修复，对齐 validator B+）：若某大子类型
    在前序节点被吃过（previously_captured_types），后续节点允许回顾该子的战术成果，
    不计入 absent——否则 prompt 禁令会让模型回避合法的历史叙述（如"白方已吃掉黑后"），
    与 validator 的 B+ 放行逻辑语义错位。

    兵不标（存在性普遍，标注无信息量）。王不标（双方永远有王）。
    """
    present = {p.piece_type for p in board.piece_map().values()}

    promoted = set()
    if sans:
        for san in sans:
            if isinstance(san, str) and "=" in san:
                letter = san.rsplit("=", 1)[-1][:1]
                promo = {"Q": chess.QUEEN, "R": chess.ROOK,
                         "B": chess.BISHOP, "N": chess.KNIGHT}
                if letter in promo:
                    promoted.add(promo[letter])

    captured_before = set(previously_captured_types or [])

    absent = []
    for pt in MAJOR_PIECE_TYPES:
        if pt not in present and pt not in promoted and pt not in captured_before:
            absent.append(piece_cn(pt))
    return absent
