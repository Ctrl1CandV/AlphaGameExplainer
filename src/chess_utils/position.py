"""棋盘位置工具——对王检测/区域判定/王安全格。

合并自 insight_extractor._king_safe_squares / _square_region / _detect_opposition、
storyboard._detect_opposition / _tag_position / _detect_zugzwang_hint /
_piece_square / _piece_squares，消除位置相关函数的重复定义。

对王检测的设计：拆为 detect_opposition_type（返回枚举）+ opposition_text（返回中文），
避免两个调用方因返回格式不同而各自维护一份实现。
"""
import chess
from typing import List, Optional

# 棋盘区域中文名
REGION_CN = {
    "corner": "角落", "edge": "边线",
    "center": "中心", "near_center": "中心一带",
}

# 对王类型枚举 → insight_extractor 使用的中文叙述（含括注）
_OPPOSITION_TEXT = {
    "direct_vertical": "正对王（竖向，逼对方让路）",
    "direct_horizontal": "正对王（横向，逼对方让路）",
    "diagonal": "斜向对王",
    "close": "近距离对峙",
}

# 对王类型枚举 → storyboard._tag_position 使用的简短中文标签
_OPPOSITION_TAG = {
    "direct_vertical": "对王(竖排)",
    "direct_horizontal": "对王(横排)",
    "diagonal": "斜线对王",
    "close": "近距离对峙",
}


def king_safe_squares(board: chess.Board, color: chess.Color) -> set:
    """color 方王在当前局面下能安全去的相邻格集合（近似王活动度）。

    判定：相邻格中，非己方占用、不与对方王相邻、不被对方攻击。
    不依赖轮到谁走（legal_moves 只算轮走方，残局里对方王常常不是轮走方）。
    """
    ksq = board.king(color)
    if ksq is None:
        return set()
    enemy = not color
    enemy_king = board.king(enemy)
    enemy_king_zone = (
        set(chess.SquareSet(chess.BB_KING_ATTACKS[enemy_king]))
        if enemy_king is not None else set()
    )
    out = set()
    for sq in chess.SquareSet(chess.BB_KING_ATTACKS[ksq]):
        piece = board.piece_at(sq)
        if piece is not None and piece.color == color:
            continue
        if sq in enemy_king_zone:
            continue
        if board.is_attacked_by(enemy, sq):
            continue
        out.add(sq)
    return out


def square_region(sq: int) -> str:
    """把一个格子归到棋盘区域：corner / edge / center / near_center。"""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    if f in (0, 7) and r in (0, 7):
        return "corner"
    if f in (0, 7) or r in (0, 7):
        return "edge"
    if f in (3, 4) and r in (3, 4):
        return "center"
    return "near_center"


def detect_opposition_type(board: chess.Board) -> str:
    """两王相对态势类型，返回枚举字符串。

    返回 'direct_vertical' / 'direct_horizontal' / 'diagonal' / 'close' / ''（无对王）。
    只返回类型枚举，不含中文叙述——调用方各自映射成需要的格式。
    """
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return ""
    df = abs(chess.square_file(wk) - chess.square_file(bk))
    dr = abs(chess.square_rank(wk) - chess.square_rank(bk))
    if df == 0 and dr == 2:
        return "direct_vertical"
    if df == 2 and dr == 0:
        return "direct_horizontal"
    if df == 2 and dr == 2:
        return "diagonal"
    if df == 0 and dr == 0:
        return ""
    if df <= 1 and dr <= 1:
        return "close"
    return ""


def opposition_text(opp_type: str) -> str:
    """对王类型 → insight_extractor 使用的含括注中文叙述。"""
    return _OPPOSITION_TEXT.get(opp_type, "")


def opposition_tag(opp_type: str) -> str:
    """对王类型 → storyboard._tag_position 使用的简短中文标签。"""
    return _OPPOSITION_TAG.get(opp_type, "")


def tag_position(board: chess.Board, move: chess.Move) -> List[str]:
    """为一步走法生成位置标签列表（将军/吃子/对王/困王提示）。

    合并自 storyboard._tag_position，内部调用 detect_opposition_type +
    opposition_tag 替代原 _detect_opposition。
    """
    tags = []
    if board.gives_check(move):
        tags.append("将军")
    if board.is_capture(move):
        tags.append("吃子")
    opp = detect_opposition(board)
    if opp:
        tags.append(opp)
    zug = detect_zugzwang_hint(board)
    if zug:
        tags.append(zug)
    return tags


def detect_opposition(board: chess.Board) -> str:
    """两王相对态势 → storyboard 风格的简短中文标签（对王(竖排) 等）。

    供 tag_position 使用，等价于原 storyboard._detect_opposition 的返回格式。
    """
    return opposition_tag(detect_opposition_type(board))


def detect_zugzwang_hint(board: chess.Board) -> str:
    """检测困王提示（仅一安全格/王被困）。"""
    if board.is_check() or board.is_game_over():
        return ""
    opponent = chess.WHITE if board.turn == chess.BLACK else chess.BLACK
    threat_sqs = list(board.attackers(opponent, board.king(board.turn)))
    if len(threat_sqs) == 1:
        return "仅一安全格"
    if len(threat_sqs) >= 3:
        return "王被困"
    return ""


def piece_square(board: chess.Board, color: chess.Color,
                 piece_type: chess.PieceType) -> Optional[int]:
    """找某一方某类棋子的格子（取第一个），无则 None。"""
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type == piece_type:
            return sq
    return None


def piece_squares(board: chess.Board, color: chess.Color,
                  piece_type: chess.PieceType) -> List[int]:
    """找某一方某类棋子的所有格子，排序返回。"""
    squares = []
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type == piece_type:
            squares.append(sq)
    return sorted(squares)
