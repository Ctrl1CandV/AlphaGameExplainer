"""Puzzle 预备步旁白生成（纯模板，不依赖 LLM）。

从 storyboard.py 提取。交代对方走法 + 子力对比 + 优势 + 轮到谁。
"""
from src.chess_utils.material import PIECE_VALUES, side_material_desc
import chess


def _prelude_san_piece_cn(san: str) -> str:
    """从SAN走法提取棋子中文名。"""
    _map = {"N": "马", "B": "象", "R": "车", "Q": "后", "K": "王"}
    for letter, cn in _map.items():
        if san.startswith(letter):
            return cn
    return "兵"


def _advantage_desc(board: chess.Board) -> str:
    """根据子力价值判断当前局面优势方。"""
    white_val = sum(
        PIECE_VALUES[p.piece_type] for p in board.piece_map().values()
        if p.color == chess.WHITE and p.piece_type != chess.KING
    )
    black_val = sum(
        PIECE_VALUES[p.piece_type] for p in board.piece_map().values()
        if p.color == chess.BLACK and p.piece_type != chess.KING
    )
    diff = white_val - black_val
    if diff >= 5:
        return "白方子力大幅领先"
    if diff >= 2:
        return "白方子力占优"
    if diff <= -5:
        return "黑方子力大幅领先"
    if diff <= -2:
        return "黑方子力占优"
    return "双方子力均势"


def build_prelude_narration(prelude_san: str, board_after: chess.Board, puzzle_side: str) -> str:
    """纯模板生成预备着开场旁白（一到两句），交代对方走法 + 子力对比 + 优势 + 轮到谁。"""
    opponent = "白方" if puzzle_side == "黑方" else "黑方"
    piece_cn_name = _prelude_san_piece_cn(prelude_san)
    is_capture = "x" in prelude_san
    is_check = board_after.is_check()

    white_mat = side_material_desc(board_after, chess.WHITE)
    black_mat = side_material_desc(board_after, chess.BLACK)
    advantage = _advantage_desc(board_after)

    if is_capture and is_check:
        move_part = f"{opponent}{piece_cn_name}直接吃子并将军，撕开对方防线"
    elif is_capture:
        move_part = f"{opponent}{piece_cn_name}果断吃子，直接获取子力"
    elif is_check:
        move_part = f"{opponent}{piece_cn_name}走子并将军，施加压力"
    else:
        move_part = f"{opponent}{piece_cn_name}调整位置，为后续战术做铺垫"

    material_part = (
        f"目前白方有{white_mat}，黑方有{black_mat}，{advantage}。"
        f"现在轮到{puzzle_side}，需要找出最强的战术手段。"
    )

    return f"{move_part}。{material_part}"
