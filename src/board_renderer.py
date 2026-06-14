from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional
from src.common import Segment, Logger
from dotenv import load_dotenv
import chess
import math
import os

load_dotenv()
_video_layout = os.getenv("VIDEO_LAYOUT", "horizontal").strip().lower()
IS_VERTICAL = _video_layout == "vertical"

# 画布布局：4:3 偏方正比例，棋盘主导，避免 16:9 两侧留白
CANVAS_W = 720 if IS_VERTICAL else 960
CANVAS_H = 1280 if IS_VERTICAL else 720
SQUARE = 75
BOARD_SIZE = 600                # SQUARE * 8
BOARD_LEFT = (CANVAS_W - BOARD_SIZE) // 2 if IS_VERTICAL else 28  # 竖版居中
BOARD_TOP = 262 if IS_VERTICAL else 20

# 右侧信息面板：紧凑宽度，与棋盘整体视觉平衡
PANEL_GAP = 24
PANEL_LEFT = BOARD_LEFT + BOARD_SIZE + PANEL_GAP     # 652
PANEL_WIDTH = CANVAS_W - PANEL_LEFT - BOARD_LEFT     # 280

PIECES_DIR = os.path.join("assets", "pieces")
FRAMES_DIR = os.path.join("output", "frames")

# 计时模型
FPS = 30
SLIDE_SEC = 0.45
GLOW_SEC = 0.30
INTRO_SEC = 1.5
MIN_STEP_HOLD = 0.35

# 颜色
COLOR_LIGHT = (240, 217, 181)
COLOR_DARK = (181, 136, 99)
COLOR_HIGHLIGHT_FROM = (255, 255, 0, 90)
COLOR_HIGHLIGHT_TO = (255, 165, 0, 110)
COLOR_HIGHLIGHT_CHECK = (255, 50, 50, 130)
COLOR_BG = (30, 30, 30)
COLOR_GLOW = (255, 215, 0)
COLOR_CHECK_GLOW = (255, 60, 60)
COLOR_CAPTURE_GLOW = (255, 140, 0)

# 压缩块内子步颜色轮换（from_hl, to_hl, arrow），区分连续多步
_SUBSTEP_COLORS = [
    ((255, 255, 0, 90),   (255, 165, 0, 110),  (255, 80, 80)),
    ((100, 200, 255, 90), (70, 130, 255, 110),  (70, 130, 255)),
    ((150, 255, 150, 90), (50, 200, 50, 110),   (50, 180, 50)),
    ((255, 180, 255, 90), (200, 100, 255, 110), (180, 80, 255)),
    ((255, 220, 100, 90), (255, 180, 50, 110),  (255, 160, 40)),
    ((150, 255, 220, 90), (50, 220, 200, 110),  (50, 200, 200)),
]

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
    col, row = chess.square_file(sq), 7 - chess.square_rank(sq)
    return BOARD_LEFT + col * SQUARE, BOARD_TOP + row * SQUARE

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
    """ 预渲染的渐变背景图 """
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

#  绘制函数

def _draw_board(draw: ImageDraw.ImageDraw):
    """ 绘制棋盘（双层暖色描边，轻微浮起感） """
    # 外层暖色描边
    outer_rect = [BOARD_LEFT - 4, BOARD_TOP - 4,
                  BOARD_LEFT + BOARD_SIZE + 5, BOARD_TOP + BOARD_SIZE + 5]
    draw.rounded_rectangle(outer_rect, radius=8, outline=(120, 95, 60), width=3)
    # 内层细描边
    board_rect = [BOARD_LEFT - 2, BOARD_TOP - 2,
                  BOARD_LEFT + BOARD_SIZE + 3, BOARD_TOP + BOARD_SIZE + 3]
    draw.rounded_rectangle(board_rect, radius=6, outline=(90, 90, 92), width=2)

    for r in range(8):
        for c in range(8):
            x = BOARD_LEFT + c * SQUARE
            y = BOARD_TOP + r * SQUARE
            color = COLOR_LIGHT if (r + c) % 2 == 0 else COLOR_DARK
            draw.rectangle([x, y, x + SQUARE - 1, y + SQUARE - 1], fill=color)

def _draw_coordinates(draw: ImageDraw.ImageDraw):
    """ 坐标标注 """
    font = _get_font(11)
    for i in range(8):
        # 列标 a-h（棋盘下方居中）
        x = BOARD_LEFT + i * SQUARE + SQUARE // 2
        draw.text(
            (x, BOARD_TOP + BOARD_SIZE + 2), chr(ord("a") + i),
            fill=(180, 180, 180), font=font, anchor="mt"
        )
        # 行标 1-8（棋盘左侧居中）
        y = BOARD_TOP + i * SQUARE + SQUARE // 2
        draw.text(
            (BOARD_LEFT - 10, y), str(8 - i),
            fill=(180, 180, 180), font=font, anchor="rm"
        )

def _draw_highlight(img: Image.Image, sq: int, color: tuple):
    """ 格子高亮 """
    x, y = _sq_xy(sq)
    overlay = Image.new("RGBA", (SQUARE, SQUARE), color)
    img.paste(overlay, (x, y), overlay)

def _draw_glow(img: Image.Image, sq: int, color: tuple, intensity: float):
    """ 落子后辉光脉冲 """
    if intensity <= 0:
        return
    intensity = max(0.0, min(1.0, intensity))
    x, y = _sq_xy(sq)
    overlay = Image.new("RGBA", (SQUARE, SQUARE), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    layers = 5
    for i in range(layers):
        a = int(150 * intensity * (1 - i / layers))
        if a <= 0:
            continue
        od.rectangle([i, i, SQUARE - 1 - i, SQUARE - 1 - i],
                     outline=color + (a,), width=2)
    img.paste(overlay, (x, y), overlay)

def _draw_arrow(
    draw: ImageDraw.ImageDraw, from_sq: int, to_sq: int,
    color=(255, 80, 80), progress: Optional[float] = None
    ):
    """ 绘制战术箭头,progress非None时在箭头上叠加移动指示圆点（滑动阶段使用） """
    fx, fy = _sq_center(from_sq)
    tx, ty = _sq_center(to_sq)

    # 主线（半透明）
    draw.line([(fx, fy), (tx, ty)], fill=color + (150,), width=5)

    # 箭头尖
    angle = math.atan2(ty - fy, tx - fx)
    al, aa = 14, math.pi / 6
    p1 = (int(tx - al * math.cos(angle - aa)), int(ty - al * math.sin(angle - aa)))
    p2 = (int(tx - al * math.cos(angle + aa)), int(ty - al * math.sin(angle + aa)))
    draw.polygon([(int(tx), int(ty)), p1, p2], fill=color + (150,))

    # 移动指示圆点（仅滑动阶段）
    if progress is not None:
        dot_x = int(lerp(fx, tx, progress))
        dot_y = int(lerp(fy, ty, progress))
        r = 5
        bright = tuple(min(255, c + 80) for c in color[:3])
        draw.ellipse(
            [dot_x - r, dot_y - r, dot_x + r, dot_y + r],
            fill=bright,
        )

def _draw_pieces_static(
    img: Image.Image, board: chess.Board, skip_sq: Optional[int] = None
    ):
    """ 绘制静止棋子 """
    for sq, piece in board.piece_map().items():
        if sq == skip_sq:
            continue
        x, y = _sq_xy(sq)
        piece_img = _load_piece(str(piece))
        img.paste(piece_img, (x, y), piece_img)


#  HUD 叠加层

def _draw_vertical_eval_bar(img: Image.Image, cx: int, top: int, bar_w: int,
                            bar_h: int, score: float):
    """纵向评估条：白方在下方填充（优势越大白区越高），黑方在上"""
    draw = ImageDraw.Draw(img)
    x0 = cx - bar_w // 2
    x1 = cx + bar_w // 2

    clamped = max(-10.0, min(10.0, float(score)))
    white_ratio = (clamped + 10) / 20          # 0=黑优, 1=白优
    white_h = int(bar_h * white_ratio)

    # 背景（黑方区域）
    draw.rectangle([x0, top, x1, top + bar_h], fill=(48, 48, 54),
                   outline=(80, 80, 88), width=1)
    # 白方区域（自下而上填充）
    if white_h > 0:
        draw.rectangle([x0 + 1, top + bar_h - white_h, x1 - 1, top + bar_h - 1],
                       fill=(222, 222, 226))
    # 中心标记线
    mid_y = top + bar_h // 2
    draw.line([(x0 - 3, mid_y), (x1 + 3, mid_y)], fill=(120, 120, 128), width=1)

    # 顶部「黑优」、底部「白优」标签
    font_s = _get_font(11)
    draw.text((x1 + 8, top), "黑优", fill=(150, 150, 158), font=font_s, anchor="lm")
    draw.text((x1 + 8, top + bar_h), "白优", fill=(190, 190, 198), font=font_s, anchor="lm")


def _draw_captured_row(img: Image.Image, x: int, y: int, max_w: int,
                       cap_black: list, cap_white: list):
    """面板内已吃子图标行（黑方丢子 / 白方丢子，超宽自动换行）"""
    draw = ImageDraw.Draw(img)
    CAP_SIZE = 22
    GAP = 2
    cur_x = x
    cur_y = y

    def _paste_list(pieces: list, start_x: int, start_y: int):
        nonlocal_x = start_x
        nonlocal_y = start_y
        for pchar in pieces:
            if nonlocal_x + CAP_SIZE > x + max_w:
                nonlocal_x = x
                nonlocal_y += CAP_SIZE + 2
            try:
                icon = _load_piece(pchar).resize((CAP_SIZE, CAP_SIZE))
                img.paste(icon, (nonlocal_x, nonlocal_y), icon)
            except Exception:
                pass
            nonlocal_x += CAP_SIZE + GAP
        return nonlocal_x, nonlocal_y

    if cap_black:
        cur_x, cur_y = _paste_list(cap_black, cur_x, cur_y)
    if cap_black and cap_white:
        draw.line([(cur_x + 2, cur_y + CAP_SIZE // 2),
                   (cur_x + 8, cur_y + CAP_SIZE // 2)],
                  fill=(120, 120, 120), width=1)
        cur_x += 12
    if cap_white:
        _paste_list(cap_white, cur_x, cur_y)


def _draw_panel_card(draw: ImageDraw.ImageDraw, x: int, y: int, w: int,
                     h: int, alpha: int = 170):
    """面板内单张卡片：半透明深色圆角矩形。"""
    draw.rounded_rectangle([x, y, x + w, y + h], radius=10,
                           fill=(22, 22, 32, alpha))


def _draw_side_panel(img: Image.Image, info: dict):
    """右侧信息面板：卡片式分区，每块独立圆角矩形，避免大面积黑块。

    所有区域在数据缺失时优雅降级（谜题链路无评分数据时不画评估条）。
    """
    px = PANEL_LEFT
    pw = PANEL_WIDTH
    panel_top = BOARD_TOP
    inner_x = px + 16
    inner_w = pw - 32

    draw = ImageDraw.Draw(img)
    y = panel_top

    # ---- 卡片 1：标题 + 进度 ----
    card1_h = 80
    _draw_panel_card(draw, px, y, pw, card1_h)
    endgame_name = info.get("endgame_name", "残局")
    winner = info.get("winner_color")
    title_color = (180, 200, 230) if winner == chess.BLACK else (255, 215, 0)
    draw.text((px + pw // 2, y + 24), endgame_name,
              fill=title_color, font=_get_font(20), anchor="ma")
    move_num = info.get("move_num", 0)
    total = info.get("total_moves", 0)
    draw.text((px + pw // 2, y + 54), f"第 {move_num} / {total} 步",
              fill=(200, 200, 200), font=_get_font(16), anchor="ma")
    y += card1_h + 12

    # ---- 卡片 2：纵向评估条（仅在有评分数据时绘制）----
    score = info.get("score")
    if score is not None:
        card2_h = 210
        _draw_panel_card(draw, px, y, pw, card2_h)
        _draw_vertical_eval_bar(img, px + pw // 2 - 12, y + 14, 14, 180, score)
        y += card2_h + 12

    # ---- 卡片 3：已吃子 ----
    cap_white = info.get("captured_white", [])
    cap_black = info.get("captured_black", [])
    if cap_white or cap_black:
        card3_h = 60
        _draw_panel_card(draw, px, y, pw, card3_h)
        draw.text((inner_x, y + 6), "已吃子",
                  fill=(150, 150, 158), font=_get_font(12), anchor="lt")
        _draw_captured_row(img, inner_x, y + 24, inner_w, cap_black, cap_white)
        y += card3_h + 12

    # ---- 卡片 4：当前阶段 ----
    phase = info.get("current_phase", "")
    if phase:
        card4_h = 42
        _draw_panel_card(draw, px, y, pw, card4_h)
        draw.text((px + pw // 2, y + 22), phase,
                  fill=(255, 215, 0), font=_get_font(16), anchor="ma")
        y += card4_h + 12

    # ---- 卡片 5：走法历史（最近 5 步）----
    history = info.get("history", [])
    if history:
        card5_h = 60
        _draw_panel_card(draw, px, y, pw, card5_h)
        draw.text((inner_x, y + 6), "近期走法",
                  fill=(150, 150, 158), font=_get_font(12), anchor="lt")
        font_hist = _get_font(14)
        recent = history[-5:]
        line = ""
        row_y = y + 24
        for token in recent:
            trial = (line + " " + token).strip()
            if draw.textlength(trial, font=font_hist) > inner_w and line:
                draw.text((inner_x, row_y), line, fill=(190, 190, 198),
                          font=font_hist, anchor="lt")
                row_y += 18
                line = token
            else:
                line = trial
        if line:
            draw.text((inner_x, row_y), line, fill=(190, 190, 198),
                      font=font_hist, anchor="lt")

def _draw_vertical_info_bar(img: Image.Image, info: dict):
    """竖版：棋盘下方的紧凑信息条，适配手机窄屏。

    仅保留战术名 + 步数 + 已吃子图标，不展示评估条和走法历史。
    """
    draw = ImageDraw.Draw(img)
    bar_top = BOARD_TOP + BOARD_SIZE + 16
    bar_h = 44
    bar_w = BOARD_SIZE
    bar_x = BOARD_LEFT

    # 半透明圆角背景条
    draw.rounded_rectangle(
        [bar_x, bar_top, bar_x + bar_w, bar_top + bar_h],
        radius=10, fill=(22, 22, 32, 190),
    )

    # 左侧：战术名（金色）
    endgame_name = info.get("endgame_name", "残局")
    draw.text((bar_x + 16, bar_top + bar_h // 2), endgame_name,
              fill=(255, 215, 0), font=_get_font(18), anchor="lm")

    # 右侧：步数
    move_num = info.get("move_num", 0)
    total = info.get("total_moves", 0)
    step_text = f"第{move_num}/{total}步" if total else ""
    if step_text:
        draw.text((bar_x + bar_w - 16, bar_top + bar_h // 2), step_text,
                  fill=(200, 200, 200), font=_get_font(14), anchor="rm")

    # 已吃子小图标（紧贴战术名右侧）
    cap_black = info.get("captured_black", []) or []
    cap_white = info.get("captured_white", []) or []
    cap_icons: list = []
    if cap_black:
        cap_icons.extend(cap_black)
    if cap_white:
        cap_icons.extend(cap_white)
    if cap_icons:
        CAP_SIZE = 20
        icon_x = bar_x + 16 + draw.textlength(endgame_name, font=_get_font(18)) + 12
        for pchar in cap_icons[-5:]:  # 最多 5 枚，避免溢出
            try:
                icon = _load_piece(pchar).resize((CAP_SIZE, CAP_SIZE))
                img.paste(icon, (int(icon_x), bar_top + (bar_h - CAP_SIZE) // 2), icon)
            except Exception:
                pass
            icon_x += CAP_SIZE + 2

def _draw_phase_label(img: Image.Image, phase_name: str, alpha: int = 200):
    """ 阶段切换标记——棋盘右上角半透明标签 """
    if not phase_name or alpha <= 0:
        return
    draw = ImageDraw.Draw(img)
    font = _get_font(20)

    # 计算文字尺寸
    bbox = draw.textbbox((0, 0), phase_name, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    px, py = 16, 8
    bw, bh = tw + px * 2, th + py * 2

    # 标签位置：棋盘右上角，稍微内缩
    bx = BOARD_LEFT + BOARD_SIZE - bw - 8
    by_ = BOARD_TOP + 8

    alpha = max(0, min(255, alpha))
    draw.rounded_rectangle(
        [bx, by_, bx + bw, by_ + bh],
        radius=14,
        fill=(15, 15, 25, alpha),
    )
    draw.text((bx + px, by_ + py), phase_name, fill=(255, 215, 0, alpha), font=font)

#  帧渲染

def render_frame(
    board: chess.Board, from_sq=None, to_sq=None, arrow_color=(255, 80, 80), is_check: bool = False,
    info: Optional[dict] = None, from_hl_color=None, to_hl_color=None, is_mate: bool = False,
    arrow_progress: Optional[float] = None, phase_label_name: str = "", phase_label_alpha: int = 0
    ) -> Image.Image:
    """ 渲染单帧棋盘，所有叠加层可选，统一走此函数 """
    img = _get_background(CANVAS_W, CANVAS_H)
    draw = ImageDraw.Draw(img)

    # 棋盘层
    _draw_board(draw)
    _draw_coordinates(draw)

    # 高亮层
    if from_sq is not None:
        _draw_highlight(img, from_sq, from_hl_color or COLOR_HIGHLIGHT_FROM)
    if to_sq is not None:
        hl_color = COLOR_HIGHLIGHT_CHECK if is_check else (to_hl_color or COLOR_HIGHLIGHT_TO)
        _draw_highlight(img, to_sq, hl_color)

    # 箭头层（progress 非 None 时带移动圆点）
    if from_sq is not None and to_sq is not None:
        _draw_arrow(draw, from_sq, to_sq, arrow_color, progress=arrow_progress)

    # 棋子层
    _draw_pieces_static(img, board)

    # HUD 层
    if info:
        if IS_VERTICAL:
            _draw_vertical_info_bar(img, info)
        else:
            _draw_side_panel(img, info)

    # 阶段标签层
    if phase_label_name:
        _draw_phase_label(img, phase_label_name, alpha=phase_label_alpha)

    # 将杀特效层
    if is_mate:
        mate_rect = [BOARD_LEFT - 5, BOARD_TOP - 5,
                     BOARD_LEFT + BOARD_SIZE + 6, BOARD_TOP + BOARD_SIZE + 6]
        draw.rounded_rectangle(mate_rect, radius=8, outline=(255, 215, 0, 220), width=4)
        mate_font = _get_font(36)
        draw.text((BOARD_LEFT + BOARD_SIZE - 16, BOARD_TOP + BOARD_SIZE - 12),
                  "将杀", fill=(255, 215, 0), font=mate_font, anchor="rb")

    return img

#  走法动画序列

def _render_move_sequence(
    board_before: chess.Board, move: chess.Move, board_after: chess.Board,
    hold_sec: float, is_check: bool = False, info: Optional[dict] = None,
    sub_colors=None, is_mate: bool = False, phase_label_name: str = "",
    phase_label_fade_frames: int = 0, phase_label_start_frame: int = 0,
    _sub_frame_idx: int = 0
    ) -> List[Tuple[Image.Image, float]]:
    """
    为单步走法生成 (帧, 时长) 序列
    三阶段：滑动(0.45s) → 落子高光脉冲(0.30s) → 定格保持(hold_sec)
    滑动阶段箭头带移动指示圆点。定格保持为单帧长时长。
    phase_label_* 参数用于在首帧叠加阶段标记。
    """
    from_sq = move.from_square
    to_sq = move.to_square
    piece = board_before.piece_at(from_sq)
    frame_dur = 1.0 / FPS
    is_capture = board_before.is_capture(move)
    out: List[Tuple[Image.Image, float]] = []

    from_hl = sub_colors[0] if sub_colors else None
    to_hl = sub_colors[1] if sub_colors else None
    arrow_col = sub_colors[2] if sub_colors else (255, 80, 80)

    hold = max(MIN_STEP_HOLD, hold_sec)

    # ---- 阶段1：滑动 ----
    if piece is not None:
        captured = board_before.piece_at(to_sq)
        piece_img = _load_piece(str(piece))
        from_x, from_y = _sq_xy(from_sq)
        to_x, to_y = _sq_xy(to_sq)

        slide_n = max(2, round(SLIDE_SEC * FPS))
        for i in range(slide_n):
            t = ease_in_out_cubic(i / (slide_n - 1))
            img = _get_background(CANVAS_W, CANVAS_H)
            draw = ImageDraw.Draw(img)

            _draw_board(draw)
            _draw_coordinates(draw)
            _draw_highlight(img, from_sq, from_hl or COLOR_HIGHLIGHT_FROM)
            _draw_pieces_static(img, board_before, skip_sq=from_sq)

            # 被吃棋子渐隐
            if captured is not None:
                cap_img = _load_piece(str(captured)).copy()
                cap_img.putalpha(int(255 * max(0.0, 1.0 - t)))
                img.paste(cap_img, (to_x, to_y), cap_img)

            # 移动棋子
            cur_x = int(lerp(from_x, to_x, t))
            cur_y = int(lerp(from_y, to_y, t))
            img.paste(piece_img, (cur_x, cur_y - 2), piece_img)

            # 箭头 + 移动指示圆点（progress 与棋子同步）
            _draw_arrow(draw, from_sq, to_sq, arrow_col, progress=t)

            # HUD
            if info:
                if IS_VERTICAL:
                    _draw_vertical_info_bar(img, info)
                else:
                    _draw_side_panel(img, info)

            # 阶段标签（仅在首帧序列前 N 帧叠加）
            frame_idx_global = _sub_frame_idx + i
            if phase_label_name and frame_idx_global < phase_label_start_frame + phase_label_fade_frames:
                rel = frame_idx_global - phase_label_start_frame
                alpha = _phase_label_alpha(rel, phase_label_fade_frames)
                _draw_phase_label(img, phase_label_name, alpha=alpha)

            out.append((img, frame_dur))
    else:
        # piece is None（罕见：不合法的走法）→ 直接渲染最终帧
        slide_n = 0

    # ---- 阶段2：落子高光脉冲 ----
    glow_color = (COLOR_CHECK_GLOW if is_check
                  else COLOR_CAPTURE_GLOW if is_capture
                  else COLOR_GLOW)
    glow_n = max(2, round(GLOW_SEC * FPS))
    for i in range(glow_n):
        intensity = math.sin((i / (glow_n - 1)) * math.pi)
        img = render_frame(board_after, from_sq=from_sq, to_sq=to_sq,
                           arrow_color=arrow_col, is_check=is_check, info=info,
                           from_hl_color=from_hl, to_hl_color=to_hl,
                           is_mate=is_mate)
        _draw_glow(img, to_sq, glow_color, intensity)

        # 阶段标签（可能在 glow 帧上也叠加）
        frame_idx_global = _sub_frame_idx + slide_n + i
        if phase_label_name and frame_idx_global < phase_label_start_frame + phase_label_fade_frames:
            rel = frame_idx_global - phase_label_start_frame
            alpha = _phase_label_alpha(rel, phase_label_fade_frames)
            _draw_phase_label(img, phase_label_name, alpha=alpha)

        out.append((img, frame_dur))

    # ---- 阶段3：定格保持（单帧长时长）----
    hold_img = render_frame(board_after, from_sq=from_sq, to_sq=to_sq,
                             arrow_color=arrow_col, is_check=is_check, info=info,
                             from_hl_color=from_hl, to_hl_color=to_hl,
                             is_mate=is_mate)
    out.append((hold_img, hold))
    return out

def _phase_label_alpha(frame_rel: int, total_frames: int) -> int:
    """阶段标签 alpha 曲线：fade in → 保持 → fade out"""
    if total_frames <= 0:
        return 0
    t = frame_rel / total_frames
    if t < 0.2:
        return int(255 * t / 0.2)         # 0 → 255
    elif t < 0.8:
        return 255                         # 保持
    else:
        return int(255 * (1 - (t - 0.8) / 0.2))  # 255 → 0

def _step_overhead_sec() -> float:
    """单个子步「滑动+高光」的固定开销（秒），不含定格"""
    slide_n = max(2, round(SLIDE_SEC * FPS))
    glow_n = max(2, round(GLOW_SEC * FPS))
    return (slide_n + glow_n) / FPS

#  主渲染入口

def render_animated_frames(
        segments: List[Segment], initial_fen: str, panel_info: Optional[dict] = None
    ) -> Tuple[List[str], List[float]]:
    """
    节点级动画渲染
    每个segment在其解说音频时长内顺序播放本节点的全部子步
    时长对齐、音画同步逻辑不变，新增阶段切换时的标签叠加
    panel_info可选:{"endgame_name": str, "scores": [...], "winner_color": ...}
    返回: (frame_paths, frame_durations)
    """
    os.makedirs(FRAMES_DIR, exist_ok=True)
    board = chess.Board(initial_fen)

    total = sum(len(getattr(seg, "moves", []) or []) for seg in segments)
    history: list = []

    white_captured = 0
    black_captured = 0
    captured_white_list: list = []
    captured_black_list: list = []

    frame_paths: List[str] = []
    durations: List[float] = []
    fnum = 0
    move_num = 0
    step_overhead = _step_overhead_sec()

    # 阶段追踪（用于阶段切换标签）
    prev_phase = ""
    global_frame_idx = 0       # 累计帧序号（跨 segment，用于标签计时）
    active_phase_label = ""    # 当前活跃的阶段标签文本
    phase_label_start = 0      # 标签起始帧序号（累计值）
    PHASE_FADE_FRAMES = 30     # 标签持续 ~1s (30fps)

    def _save(img: Image.Image, dur: float):
        nonlocal fnum
        if img.mode != "RGB":
            img = img.convert("RGB")
        fpath = os.path.join(FRAMES_DIR, f"frame_{fnum:05d}.png")
        img.save(fpath)
        frame_paths.append(fpath)
        durations.append(dur)
        fnum += 1

    # 初始静态展示帧
    init_info = _make_frame_info(
        panel_info, 0, total, history, 0.0,
        white_captured, black_captured,
        captured_white_list, captured_black_list,
        current_phase=""
    )
    _save(render_frame(board, info=init_info), INTRO_SEC)
    global_frame_idx += 1

    time_cursor = 0.0

    for seg in segments:
        node_moves = list(getattr(seg, "moves", []) or [])
        seg_target = seg.duration_s if seg.duration_s and seg.duration_s > 0 else 3.0
        seg_start_cursor = time_cursor

        # 检测阶段切换 → 启动阶段标签
        seg_phase = getattr(seg, "phase", "") or ""
        if seg_phase and seg_phase != prev_phase:
            active_phase_label = seg_phase
            phase_label_start = global_frame_idx
        prev_phase = seg_phase

        # ---- 无走法节点（开场白/总结段）—— 静态定格 ----
        if not node_moves:
            is_mate = board.is_checkmate()
            score = _safe_score(panel_info, move_num, total)
            info = _make_frame_info(
                panel_info, move_num, total, history, score,
                white_captured, black_captured,
                captured_white_list, captured_black_list,
                current_phase=seg_phase
            )
            # 无走法段不叠加阶段标签（开场白/总结词的 phase 为空或不变）
            img = render_frame(board, info=info, is_mate=is_mate)
            _save(img, seg_target)
            seg.start_time = seg_start_cursor
            seg.duration_s = seg_target
            time_cursor += seg_target
            global_frame_idx += 1
            continue

        # ---- 有走法节点 ----
        # 子步重要性加权分配定格时长：吃子/将军/将杀步获得更多定格，
        # 重复驱赶步快速带过。总时长不变，但观众能在重要步上"看清楚"。
        n = len(node_moves)
        budget_hold = seg_target - n * step_overhead

        # 预计算每步权重（在副本上推进，保证每步在正确局面上评估）
        weights = []
        temp_board = board.copy()
        for sub_move in node_moves:
            w = 1.0
            if temp_board.is_capture(sub_move):
                w = 3.0
            if temp_board.gives_check(sub_move):
                w = max(w, 2.0)
            weights.append(w)
            temp_board.push(sub_move)

        if budget_hold >= n * MIN_STEP_HOLD and n > 0:
            # 充足预算：标准步 MIN_STEP_HOLD，重要步按权重分配剩余
            std_hold = MIN_STEP_HOLD
            extra_budget = budget_hold - n * std_hold
            total_extra_w = sum(max(0, w - 1.0) for w in weights)
            if total_extra_w > 0:
                hold_per_extra = extra_budget / total_extra_w
                holds = [std_hold + max(0, w - 1.0) * hold_per_extra for w in weights]
            else:
                holds = [budget_hold / n] * n
        else:
            # 紧张预算：均摊（保底 MIN_STEP_HOLD）
            per_hold = max(MIN_STEP_HOLD, budget_hold / n if n > 0 else 0.0)
            holds = [per_hold] * n

        seg_rendered = 0.0
        for sub_idx, move in enumerate(node_moves):
            is_check = board.gives_check(move)
            history.append(board.san(move))

            # 跟踪吃子
            if board.is_capture(move):
                cap_sq = move.to_square
                cap_piece = board.piece_at(cap_sq)
                if cap_piece is None:
                    ep_sq = chess.square(chess.square_file(move.to_square),
                                         chess.square_rank(move.from_square))
                    cap_piece = board.piece_at(ep_sq)
                cap_char = str(cap_piece) if cap_piece else "p"
                if board.turn == chess.WHITE:
                    black_captured += 1
                    captured_black_list.append(cap_char)
                else:
                    white_captured += 1
                    captured_white_list.append(cap_char)

            move_num += 1
            score = _safe_score(panel_info, move_num - 1, total)
            frame_info = _make_frame_info(
                panel_info, move_num, total, history, score,
                white_captured, black_captured,
                captured_white_list, captured_black_list,
                current_phase=seg_phase
            )

            sub_colors = _SUBSTEP_COLORS[sub_idx % len(_SUBSTEP_COLORS)]

            board_before = board.copy()
            board.push(move)
            is_mate = board.is_checkmate()

            # 阶段标签参数
            pl_name = ""
            pl_start = 0
            pl_fade = 0
            if active_phase_label and global_frame_idx < phase_label_start + PHASE_FADE_FRAMES:
                pl_name = active_phase_label
                pl_start = phase_label_start
                pl_fade = PHASE_FADE_FRAMES

            for img, dur in _render_move_sequence(
                    board_before, move, board, holds[sub_idx],
                    is_check, info=frame_info,
                    sub_colors=sub_colors,
                    is_mate=is_mate,
                    phase_label_name=pl_name,
                    phase_label_fade_frames=pl_fade,
                    phase_label_start_frame=pl_start,
                    _sub_frame_idx=global_frame_idx):
                _save(img, dur)
                seg_rendered += dur
                global_frame_idx += 1

        seg.start_time = seg_start_cursor
        seg.duration_s = seg_rendered
        time_cursor += seg_rendered

    Logger.success(f"动画渲染完成: {len(frame_paths)} 帧, {sum(durations):.1f}s")
    return frame_paths, durations

# ---- 辅助函数 ----

def _safe_score(panel_info, idx, total) -> Optional[float]:
    if not panel_info:
        return None
    scores = panel_info.get("scores")
    if not scores:
        return None
    if idx < len(scores):
        return scores[idx]
    return None

def _make_frame_info(
    panel_info: Optional[dict], move_num: int, total: int,
    history: list, score: Optional[float],
    white_captured: int = 0, black_captured: int = 0,
    captured_white: Optional[list] = None,
    captured_black: Optional[list] = None,
    current_phase: str = ""
    ) -> Optional[dict]:
    """构建帧级信息字典（供 HUD 叠加层使用）"""
    if panel_info is None:
        return None
    return {
        "endgame_name": panel_info.get("endgame_name", "残局"),
        "move_num": move_num,
        "total_moves": total,
        "history": list(history),
        "score": score,
        "white_captured": white_captured,
        "black_captured": black_captured,
        "winner_color": panel_info.get("winner_color"),
        "captured_white": captured_white or [],
        "captured_black": captured_black or [],
        "current_phase": current_phase,
    }