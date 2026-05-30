import chess
import os
import math
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
from src.common import Segment, Logger

SQUARE = 64
BOARD_SIZE = SQUARE * 8
MARGIN_LEFT = 32
MARGIN_TOP = 12
LABEL_SIZE = 16
PANEL_WIDTH = 190
PANEL_GAP = 16
IMG_W = BOARD_SIZE + MARGIN_LEFT + 12  # 556 (without panel)
IMG_H = BOARD_SIZE + MARGIN_TOP + LABEL_SIZE + 108  # 648 (含底部100px字幕区)
IMG_W_FULL = IMG_W + PANEL_GAP + PANEL_WIDTH  # 762 (with panel)
PIECES_DIR = os.path.join("assets", "pieces")
FRAMES_DIR = os.path.join("output", "frames")

# ---- 计时模型（与 video_composer 输出 fps 对齐，保证滑动不丢帧）----
FPS = 30                 # 渲染与最终视频统一帧率
SLIDE_SEC = 0.45         # 棋子滑动时长（与解说时长解耦，固定快速平滑）
GLOW_SEC = 0.30          # 落子后的高光脉冲时长
INTRO_SEC = 1.5          # 开场静态展示初始局面的时长
MIN_HOLD_SEC = 0.4       # 滑动+高光后最短定格时长

COLOR_LIGHT = (240, 217, 181)
COLOR_DARK = (181, 136, 99)
COLOR_HIGHLIGHT_FROM = (255, 255, 0, 90)
COLOR_HIGHLIGHT_TO = (255, 165, 0, 110)
COLOR_HIGHLIGHT_CHECK = (255, 50, 50, 130)
COLOR_BG = (30, 30, 30)
COLOR_GLOW = (255, 215, 0)        # 落子高光（金）
COLOR_CHECK_GLOW = (255, 60, 60)  # 将军高光（红）
COLOR_CAPTURE_GLOW = (255, 140, 0)  # 吃子高光（橙）

PIECE_MAP = {
    "K": "king-w.png", "Q": "queen-w.png", "R": "rook-w.png",
    "B": "bishop-w.png", "N": "knight-w.png", "P": "pawn-w.png",
    "k": "king-b.png", "q": "queen-b.png", "r": "rook-b.png",
    "b": "bishop-b.png", "n": "knight-b.png", "p": "pawn-b.png",
}

_piece_cache: dict = {}
_font_cache: dict = {}


def _load_piece(char: str) -> Image.Image:
    if char not in _piece_cache:
        path = os.path.join(PIECES_DIR, PIECE_MAP[char])
        _piece_cache[char] = Image.open(path).convert("RGBA").resize((SQUARE, SQUARE))
    return _piece_cache[char]


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype("simhei.ttf", size)
        except Exception:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def _sq_xy(sq: int) -> Tuple[int, int]:
    col = chess.square_file(sq)
    row = 7 - chess.square_rank(sq)
    return MARGIN_LEFT + col * SQUARE, MARGIN_TOP + row * SQUARE


def _sq_center(sq: int) -> Tuple[int, int]:
    x, y = _sq_xy(sq)
    return x + SQUARE // 2, y + SQUARE // 2


def ease_in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4 * t * t * t
    return 1 - (-2 * t + 2) ** 3 / 2


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


_bg_cache: dict = {}


def _get_background(width: int, height: int) -> Image.Image:
    """预渲染的渐变背景图（缓存复用，避免逐像素重绘）"""
    key = (width, height)
    if key not in _bg_cache:
        img = Image.new("RGBA", key, COLOR_BG)
        draw = ImageDraw.Draw(img)
        for y in range(height):
            r = int(25 + (y / height) * 15)
            g = int(25 + (y / height) * 12)
            b = int(30 + (y / height) * 20)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
        _bg_cache[key] = img
    return _bg_cache[key].copy()


def _draw_board(draw: ImageDraw.ImageDraw):
    # 棋盘边框阴影
    shadow_offset = 3
    board_rect = [MARGIN_LEFT - 2, MARGIN_TOP - 2, 
                  MARGIN_LEFT + BOARD_SIZE + 1, MARGIN_TOP + BOARD_SIZE + 1]
    draw.rectangle([board_rect[0] + shadow_offset, board_rect[1] + shadow_offset,
                    board_rect[2] + shadow_offset, board_rect[3] + shadow_offset], 
                   fill=(15, 15, 15))
    
    # 棋盘边框
    draw.rectangle(board_rect, outline=(80, 80, 80), width=2)
    
    for r in range(8):
        for c in range(8):
            x = MARGIN_LEFT + c * SQUARE
            y = MARGIN_TOP + r * SQUARE
            color = COLOR_LIGHT if (r + c) % 2 == 0 else COLOR_DARK
            draw.rectangle([x, y, x + SQUARE - 1, y + SQUARE - 1], fill=color)


def _draw_coordinates(draw: ImageDraw.ImageDraw):
    font = _get_font(11)
    for i in range(8):
        # column labels a-h
        x = MARGIN_LEFT + i * SQUARE + SQUARE // 2
        draw.text((x, MARGIN_TOP + BOARD_SIZE + 2),
                  chr(ord("a") + i), fill=(180, 180, 180), font=font, anchor="mt")
        # row labels 1-8
        y = MARGIN_TOP + i * SQUARE + SQUARE // 2
        draw.text((MARGIN_LEFT - 10, y),
                  str(8 - i), fill=(180, 180, 180), font=font, anchor="rm")


def _draw_highlight(img: Image.Image, sq: int, color: tuple):
    x, y = _sq_xy(sq)
    overlay = Image.new("RGBA", (SQUARE, SQUARE), color)
    img.paste(overlay, (x, y), overlay)


def _draw_glow(img: Image.Image, sq: int, color: tuple, intensity: float):
    """在格子上叠加一层发光边框，intensity 0~1 控制亮度，用于落子脉冲/呼吸效果"""
    if intensity <= 0:
        return
    intensity = max(0.0, min(1.0, intensity))
    x, y = _sq_xy(sq)
    overlay = Image.new("RGBA", (SQUARE, SQUARE), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    # 多层递减边框模拟辉光
    layers = 5
    for i in range(layers):
        a = int(150 * intensity * (1 - i / layers))
        if a <= 0:
            continue
        od.rectangle([i, i, SQUARE - 1 - i, SQUARE - 1 - i],
                     outline=color + (a,), width=2)
    img.paste(overlay, (x, y), overlay)


def _draw_arrow(draw: ImageDraw.ImageDraw, from_sq: int, to_sq: int, color=(255, 80, 80)):
    fx, fy = _sq_center(from_sq)
    tx, ty = _sq_center(to_sq)
    draw.line([(fx, fy), (tx, ty)], fill=color, width=5)
    angle = math.atan2(ty - fy, tx - fx)
    al, aa = 14, math.pi / 6
    p1 = (int(tx - al * math.cos(angle - aa)), int(ty - al * math.sin(angle - aa)))
    p2 = (int(tx - al * math.cos(angle + aa)), int(ty - al * math.sin(angle + aa)))
    draw.polygon([(int(tx), int(ty)), p1, p2], fill=color)


def _draw_eval_bar(img: Image.Image, x: int, y: int, score: float,
                   width: int = 18, height: int = 120):
    """绘制评估条。score 正值=白优, 负值=黑优"""
    draw = ImageDraw.Draw(img)
    draw.rectangle([x - 1, y - 1, x + width + 1, y + height + 1],
                   outline=(100, 100, 100), width=1)

    clamped = max(-10, min(10, score))
    white_ratio = (clamped + 10) / 20
    white_h = int(height * white_ratio)
    black_h = height - white_h

    draw.rectangle([x, y, x + width, y + black_h], fill=(40, 40, 40))
    draw.rectangle([x, y + black_h, x + width, y + height], fill=(240, 240, 240))

    font = _get_font(10)
    label = f"{score:+.1f}" if abs(score) < 100 else "M"
    draw.text((x + width // 2, y + height + 5), label, fill=(180, 180, 180),
              font=font, anchor="mt")


def _draw_info_panel(img: Image.Image, info: dict):
    """在棋盘右侧绘制信息面板"""
    px = IMG_W + PANEL_GAP
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("simhei.ttf", 18)
        font_body = ImageFont.truetype("simhei.ttf", 13)
    except Exception:
        font_title = _get_font(16)
        font_body = _get_font(12)

    # 面板背景
    panel_rect = [px - 8, 4, px + PANEL_WIDTH + 8, IMG_H - 4]
    draw.rounded_rectangle(panel_rect, radius=8, fill=(40, 40, 45), outline=(70, 70, 75))

    y = 16

    # 标题（带装饰线）
    endgame_name = info.get("endgame_name", "残局")
    draw.text((px, y), endgame_name, fill=(255, 215, 0), font=font_title)
    y += 28
    draw.line([(px, y), (px + PANEL_WIDTH - 16, y)], fill=(60, 60, 65), width=1)
    y += 12

    # 步数
    move_num = info.get("move_num", 0)
    total = info.get("total_moves", 0)
    draw.text((px, y), f"第 {move_num}/{total} 步", fill=(220, 220, 220), font=font_body)
    y += 24

    # 评估条
    score = info.get("score")
    if score is not None:
        _draw_eval_bar(img, px + 10, y, score)
        y += 140

    # 走法记录
    draw.text((px, y), "走法记录:", fill=(150, 150, 150), font=font_body)
    y += 20
    history = info.get("history", [])
    for move_san in history[-8:]:
        draw.text((px + 8, y), move_san, fill=(200, 200, 200), font=font_body)
        y += 18


def _draw_pieces_static(img: Image.Image, board: chess.Board, skip_sq: Optional[int] = None):
    for sq, piece in board.piece_map().items():
        if sq == skip_sq:
            continue
        x, y = _sq_xy(sq)
        piece_img = _load_piece(str(piece))
        img.paste(piece_img, (x, y), piece_img)


def render_frame(board: chess.Board, from_sq=None, to_sq=None,
                 arrow_color=(255, 80, 80), is_check: bool = False,
                 info: Optional[dict] = None) -> Image.Image:
    """渲染单帧棋盘，可选右侧信息面板"""
    w = IMG_W_FULL if info else IMG_W
    img = _get_background(w, IMG_H)

    draw = ImageDraw.Draw(img)
    _draw_board(draw)
    _draw_coordinates(draw)
    if from_sq is not None:
        _draw_highlight(img, from_sq, COLOR_HIGHLIGHT_FROM)
    if to_sq is not None:
        hl_color = COLOR_HIGHLIGHT_CHECK if is_check else COLOR_HIGHLIGHT_TO
        _draw_highlight(img, to_sq, hl_color)
    if from_sq is not None and to_sq is not None:
        _draw_arrow(draw, from_sq, to_sq, arrow_color)
    _draw_pieces_static(img, board)
    if info:
        _draw_info_panel(img, info)
    return img


def _render_move_sequence(board_before: chess.Board, move: chess.Move,
                          board_after: chess.Board, seg_dur: float,
                          is_check: bool = False,
                          info: Optional[dict] = None) -> List[Tuple[Image.Image, float]]:
    """
    为单步走法生成 (帧, 时长) 序列，三阶段：
      1. 滑动：固定 SLIDE_SEC 的快速平滑移动（与解说时长解耦）
      2. 落子高光脉冲：GLOW_SEC 的辉光呼吸（将军红/吃子橙/普通金）
      3. 定格保持：剩余时长用单帧撑满，让解说继续播
    """
    from_sq = move.from_square
    to_sq = move.to_square
    piece = board_before.piece_at(from_sq)
    w = IMG_W_FULL if info else IMG_W
    frame_dur = 1.0 / FPS
    is_capture = board_before.is_capture(move)
    out: List[Tuple[Image.Image, float]] = []

    if piece is None:
        img = render_frame(board_after, from_sq, to_sq, is_check=is_check, info=info)
        out.append((img, max(MIN_HOLD_SEC, seg_dur)))
        return out

    captured = board_before.piece_at(to_sq)
    piece_img = _load_piece(str(piece))
    from_x, from_y = _sq_xy(from_sq)
    to_x, to_y = _sq_xy(to_sq)

    # ---- 阶段1：滑动 ----
    slide_n = max(2, round(SLIDE_SEC * FPS))
    for i in range(slide_n):
        t = ease_in_out_cubic(i / (slide_n - 1))
        img = _get_background(w, IMG_H)
        draw = ImageDraw.Draw(img)
        _draw_board(draw)
        _draw_coordinates(draw)
        _draw_highlight(img, from_sq, COLOR_HIGHLIGHT_FROM)
        _draw_pieces_static(img, board_before, skip_sq=from_sq)
        # 被吃子在滑动过程中淡出
        if captured is not None:
            cap_img = _load_piece(str(captured)).copy()
            cap_img.putalpha(int(255 * max(0.0, 1.0 - t)))
            img.paste(cap_img, (to_x, to_y), cap_img)
        cur_x = int(lerp(from_x, to_x, t))
        cur_y = int(lerp(from_y, to_y, t))
        img.paste(piece_img, (cur_x, cur_y), piece_img)
        if info:
            _draw_info_panel(img, info)
        out.append((img, frame_dur))

    # ---- 阶段2：落子高光脉冲 ----
    glow_color = COLOR_CHECK_GLOW if is_check else (COLOR_CAPTURE_GLOW if is_capture else COLOR_GLOW)
    glow_n = max(2, round(GLOW_SEC * FPS))
    for i in range(glow_n):
        intensity = math.sin((i / (glow_n - 1)) * math.pi)  # 0→1→0 脉冲
        img = render_frame(board_after, from_sq=from_sq, to_sq=to_sq,
                           is_check=is_check, info=info)
        _draw_glow(img, to_sq, glow_color, intensity)
        out.append((img, frame_dur))

    # ---- 阶段3：定格保持（用实际帧数计算，保证每步总时长精确等于 seg_dur）----
    used = (slide_n + glow_n) / FPS
    hold = max(MIN_HOLD_SEC, seg_dur - used)
    hold_img = render_frame(board_after, from_sq=from_sq, to_sq=to_sq,
                            is_check=is_check, info=info)
    out.append((hold_img, hold))
    return out


def render_animated_frames(moves: list, initial_fen: str, segments: List[Segment],
                           panel_info: Optional[dict] = None) -> Tuple[List[str], List[float]]:
    """
    根据走法列表生成平滑动画帧序列。
    panel_info 可选: {"endgame_name": str, "scores": [...]}
    返回: (frame_paths, frame_durations)
    每步时长 = INTRO_SEC(首帧) 或 该段音频时长(seg_dur)，与音频严格对齐。
    """
    os.makedirs(FRAMES_DIR, exist_ok=True)
    board = chess.Board(initial_fen)
    total = len(moves)
    history = []

    frame_paths = []
    durations = []
    fnum = 0

    def _save(img: Image.Image, dur: float):
        nonlocal fnum
        if img.mode != "RGB":
            img = img.convert("RGB")
        fpath = os.path.join(FRAMES_DIR, f"frame_{fnum:05d}.png")
        img.save(fpath)
        frame_paths.append(fpath)
        durations.append(dur)
        fnum += 1

    # 初始局面（静态展示 INTRO_SEC，音频侧由 composer 补等长静音对齐）
    init_info = _make_frame_info(panel_info, 0, total, history, 0.0)
    _save(render_frame(board, info=init_info), INTRO_SEC)

    for i, move in enumerate(moves):
        seg = segments[i] if i < len(segments) else None
        seg_dur = seg.duration_s if seg else 3.0
        is_check = board.gives_check(move)
        history.append(board.san(move))

        score = panel_info.get("scores", [None] * total)[i] if panel_info else None
        frame_info = _make_frame_info(panel_info, i + 1, total, history, score)

        board_before = board.copy()
        board.push(move)

        for img, dur in _render_move_sequence(board_before, move, board, seg_dur,
                                              is_check, info=frame_info):
            _save(img, dur)

    Logger.success(f"动画渲染完成: {len(frame_paths)} 帧, {sum(durations):.1f}s")
    return frame_paths, durations


def _make_frame_info(panel_info: Optional[dict], move_num: int, total: int,
                     history: list, score: Optional[float]) -> Optional[dict]:
    if panel_info is None:
        return None
    return {
        "endgame_name": panel_info.get("endgame_name", "残局"),
        "move_num": move_num,
        "total_moves": total,
        "history": list(history),
        "score": score,
    }
