import chess
import os
from PIL import Image, ImageDraw
from typing import List, Tuple
from src.common import Segment, Logger

SQUARE = 64
BOARD_SIZE = SQUARE * 8
MARGIN = 20
IMG_W = BOARD_SIZE + MARGIN * 2
PIECES_DIR = os.path.join("assets", "pieces")
FRAMES_DIR = os.path.join("output", "frames")

COLOR_LIGHT = (240, 217, 181)
COLOR_DARK = (181, 136, 99)
COLOR_HIGHLIGHT_FROM = (255, 255, 0, 120)
COLOR_HIGHLIGHT_TO = (255, 200, 0, 120)
COLOR_ARROW = (255, 50, 50)

PIECE_MAP = {
    "K": "king-w.png", "Q": "queen-w.png", "R": "rook-w.png",
    "B": "bishop-w.png", "N": "knight-w.png", "P": "pawn-w.png",
    "k": "king-b.png", "q": "queen-b.png", "r": "rook-b.png",
    "b": "bishop-b.png", "n": "knight-b.png", "p": "pawn-b.png",
}

_piece_cache: dict = {}


def _load_piece(char: str) -> Image.Image:
    if char not in _piece_cache:
        path = os.path.join(PIECES_DIR, PIECE_MAP[char])
        img = Image.open(path).convert("RGBA").resize((SQUARE, SQUARE))
        _piece_cache[char] = img
    return _piece_cache[char]


def _square_center(sq: int) -> Tuple[int, int]:
    col = chess.square_file(sq)
    row = 7 - chess.square_rank(sq)
    return MARGIN + col * SQUARE + SQUARE // 2, MARGIN + row * SQUARE + SQUARE // 2


def _draw_board(draw: ImageDraw.ImageDraw):
    for r in range(8):
        for c in range(8):
            x, y = MARGIN + c * SQUARE, MARGIN + r * SQUARE
            color = COLOR_LIGHT if (r + c) % 2 == 0 else COLOR_DARK
            draw.rectangle([x, y, x + SQUARE - 1, y + SQUARE - 1], fill=color)


def _draw_highlight(draw: ImageDraw.ImageDraw, sq: int, color):
    col = chess.square_file(sq)
    row = 7 - chess.square_rank(sq)
    x, y = MARGIN + col * SQUARE, MARGIN + row * SQUARE
    overlay = Image.new("RGBA", (SQUARE, SQUARE), color)
    draw._image.paste(overlay, (x, y), overlay)


def _draw_arrow(draw: ImageDraw.ImageDraw, from_sq: int, to_sq: int):
    fx, fy = _square_center(from_sq)
    tx, ty = _square_center(to_sq)
    draw.line([(fx, fy), (tx, ty)], fill=COLOR_ARROW, width=4)
    import math
    angle = math.atan2(ty - fy, tx - fx)
    al = 15
    aa = math.pi / 6
    p1 = (tx - al * math.cos(angle - aa), ty - al * math.sin(angle - aa))
    p2 = (tx - al * math.cos(angle + aa), ty - al * math.sin(angle + aa))
    draw.polygon([(tx, ty), p1, p2], fill=COLOR_ARROW)


def _draw_pieces(img: Image.Image, board: chess.Board):
    for sq, piece in board.piece_map().items():
        col = chess.square_file(sq)
        row = 7 - chess.square_rank(sq)
        x, y = MARGIN + col * SQUARE, MARGIN + row * SQUARE
        img.paste(_load_piece(str(piece)), (x, y), _load_piece(str(piece)))


def render_frame(board: chess.Board, from_sq=None, to_sq=None) -> Image.Image:
    """渲染单帧棋盘，支持高亮来源格、目标格和走法箭头"""
    img = Image.new("RGBA", (IMG_W, IMG_W), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    _draw_board(draw)
    if from_sq is not None:
        _draw_highlight(draw, from_sq, COLOR_HIGHLIGHT_FROM)
    if to_sq is not None:
        _draw_highlight(draw, to_sq, COLOR_HIGHLIGHT_TO)
    if from_sq is not None and to_sq is not None:
        _draw_arrow(draw, from_sq, to_sq)
    _draw_pieces(img, board)
    return img


def render_frames(game_data, segments: List[Segment]) -> Tuple[List[str], List[float]]:
    """
    根据 Segment 时长渲染帧序列，每步棋 3 阶段展示。
    返回: (frame_paths, frame_durations)
    """
    os.makedirs(FRAMES_DIR, exist_ok=True)
    board = chess.Board(game_data.initial_fen)
    moves = game_data.moves
    frame_paths: List[str] = []
    durations: List[float] = []
    fnum = 0

    # 初始局面
    path = os.path.join(FRAMES_DIR, f"frame_{fnum:05d}.png")
    render_frame(board).save(path)
    frame_paths.append(path)
    durations.append(2.0)
    fnum += 1

    for i, move in enumerate(moves):
        seg = segments[i] if i < len(segments) else None
        seg_dur = seg.duration_s if seg else 3.0
        from_sq, to_sq = move.from_square, move.to_square
        board_before = board.copy()

        # 阶段1: 高亮+箭头（30%）
        img = render_frame(board_before, from_sq=from_sq, to_sq=to_sq)
        path = os.path.join(FRAMES_DIR, f"frame_{fnum:05d}.png")
        img.save(path)
        frame_paths.append(path)
        durations.append(seg_dur * 0.3)
        fnum += 1

        # 阶段2: 走棋后+高亮（30%）
        board.push(move)
        img = render_frame(board, from_sq=from_sq, to_sq=to_sq)
        path = os.path.join(FRAMES_DIR, f"frame_{fnum:05d}.png")
        img.save(path)
        frame_paths.append(path)
        durations.append(seg_dur * 0.3)
        fnum += 1

        # 阶段3: 干净新局面（40%）
        img = render_frame(board)
        path = os.path.join(FRAMES_DIR, f"frame_{fnum:05d}.png")
        img.save(path)
        frame_paths.append(path)
        durations.append(seg_dur * 0.4)
        fnum += 1

    Logger.success(f"渲染完成: {len(frame_paths)} 帧")
    return frame_paths, durations
