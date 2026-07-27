from PIL import Image, ImageDraw, ImageFont, ImageFilter
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
CANVAS_H = 960 if IS_VERTICAL else 720
SQUARE = 75
BOARD_SIZE = 600                # SQUARE * 8
BOARD_LEFT = (CANVAS_W - BOARD_SIZE) // 2 if IS_VERTICAL else 28  # 竖版居中
BOARD_TOP = 40 if IS_VERTICAL else 20  # 竖版棋盘贴顶，减少中间黑色空隙

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

# PLAN-006 视觉分化：emphasis_level → 画面效果参数
# 阶段 E 二轮反馈修正：用户明确全局效果（整屏明暗/缩放/字幕变体）观感差，
# 只保留局部效果（落子辉光增强、面板着法着色），详略改由局部辉光 + 时间节奏承担。
GLOW_SEC_PIVOTAL = 0.45          # pivotal 辉光延长（默认 0.30），局部作用于落点格

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
_piece_shadow_cache: dict = {}
_font_cache: dict = {}

def _load_piece(char: str) -> Image.Image:
    if char not in _piece_cache:
        path = os.path.join(PIECES_DIR, PIECE_MAP[char])
        _piece_cache[char] = Image.open(path).convert("RGBA").resize((SQUARE, SQUARE))
    return _piece_cache[char]

def _get_piece_shadow(char: str) -> Optional[Image.Image]:
    """棋子投影精灵缓存：取棋子 alpha 剪影填深色后高斯模糊，一次生成、每帧粘贴。
    与棋盘投影共用右下偏移的单光源约定。"""
    if not ENABLE_PIECE_SHADOW:
        return None
    if char in _piece_shadow_cache:
        return _piece_shadow_cache[char]
    piece = _load_piece(char)
    alpha = piece.split()[-1]
    shadow = Image.new("RGBA", piece.size, (0, 0, 0, 0))
    shadow.putalpha(alpha.point(lambda a: int(a * 0.45)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=3))
    _piece_shadow_cache[char] = shadow
    return shadow

def _paste_piece(img: Image.Image, piece_img: Image.Image, x: int, y: int,
                 shadow_char: Optional[str] = None):
    """粘贴棋子；若提供 shadow_char 则先在右下粘贴投影（光源左上）。"""
    if shadow_char is not None:
        sh = _get_piece_shadow(shadow_char)
        if sh is not None:
            img.paste(sh, (x + 4, y + 6), sh)
    img.paste(piece_img, (x, y), piece_img)

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

# ============================================================
# PLAN-005 Core 2 视觉质量开关与缓存（B1~B5）
# 每项可独立关闭以做前后对照；棋盘/棋子投影一次生成、每帧仅粘贴。
# ============================================================
ENABLE_BOARD_SHADOW = True      # B1: 棋盘投影
ENABLE_AA_ARROWS = True         # B2: 箭头 2x overlay 抗锯齿
ENABLE_SOFT_GLOW = True         # B3: glow 高斯柔光（替代同心矩形）
ENABLE_SOFT_HIGHLIGHT = True    # B4: 格子高亮柔和圆角内嵌环
ENABLE_PIECE_SHADOW = True      # B5: 棋子投影精灵缓存

_board_shadow_cache: dict = {}

def _get_board_shadow() -> Optional[Image.Image]:
    """棋盘投影缓存：在棋盘矩形下方画一块模糊的深色阴影，让棋盘浮起。
    一次生成、每帧粘贴；阴影略大于棋盘并向右下偏移，模拟单一光源。"""
    if not ENABLE_BOARD_SHADOW:
        return None
    key = ("board_shadow",)
    if key in _board_shadow_cache:
        return _board_shadow_cache[key]
    # 投影画布比棋盘大一圈以容纳模糊扩散
    pad = 24
    w, h = BOARD_SIZE + pad * 2, BOARD_SIZE + pad * 2
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rounded_rectangle([pad - 6, pad - 6, pad + BOARD_SIZE + 6, pad + BOARD_SIZE + 6],
                         radius=12, fill=(0, 0, 0, 150))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=12))
    _board_shadow_cache[key] = shadow
    return shadow

def _paste_board_shadow(img: Image.Image):
    """把棋盘投影粘贴到画布上（背景之上、棋盘格之下）。向右下偏移模拟光源。"""
    shadow = _get_board_shadow()
    if shadow is None:
        return
    # 投影中心对齐棋盘中心，整体向右下偏移（光源在左上）
    offset_x = BOARD_LEFT - 24 + 8
    offset_y = BOARD_TOP - 24 + 8
    img.paste(shadow, (offset_x, offset_y), shadow)

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
    """ 格子高亮：柔和圆角内填 + 内嵌描边环（B4），可关闭回退整格平铺 alpha。"""
    x, y = _sq_xy(sq)
    if not ENABLE_SOFT_HIGHLIGHT:
        overlay = Image.new("RGBA", (SQUARE, SQUARE), color)
        img.paste(overlay, (x, y), overlay)
        return
    # 圆角内填：alpha 取 color 第 4 分量，略降到 70% 避免压住棋子辨识
    base_a = color[3] if len(color) >= 4 else 110
    fill_a = int(base_a * 0.7)
    rgb = color[:3]
    inset = 6
    overlay = Image.new("RGBA", (SQUARE, SQUARE), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle(
        [inset, inset, SQUARE - inset, SQUARE - inset], radius=10,
        fill=rgb + (fill_a,), outline=rgb + (min(255, base_a + 40),), width=2,
    )
    img.paste(overlay, (x, y), overlay)

def _draw_glow(img: Image.Image, sq: int, color: tuple, intensity: float):
    """ 落子后辉光脉冲：高斯柔光（B3）替代 5 层同心矩形；可关闭回退原阶梯式。"""
    if intensity <= 0:
        return
    intensity = max(0.0, min(1.0, intensity))
    x, y = _sq_xy(sq)
    rgb = color[:3]
    if not ENABLE_SOFT_GLOW:
        # 原阶梯式（保留作前后对照）
        overlay = Image.new("RGBA", (SQUARE, SQUARE), (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        layers = 5
        for i in range(layers):
            a = int(150 * intensity * (1 - i / layers))
            if a <= 0:
                continue
            od.rectangle([i, i, SQUARE - 1 - i, SQUARE - 1 - i],
                         outline=rgb + (a,), width=2)
        img.paste(overlay, (x, y), overlay)
        return
    # 柔光：亮色圆角块经高斯模糊得连续衰减光晕，亮度随 intensity 脉冲
    glow = Image.new("RGBA", (SQUARE, SQUARE), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.rounded_rectangle([4, 4, SQUARE - 5, SQUARE - 5], radius=8,
                         fill=rgb + (int(180 * intensity),))
    glow = glow.filter(ImageFilter.GaussianBlur(radius=5))
    img.paste(glow, (x, y), glow)

def _draw_arrow(
    img: Image.Image, from_sq: int, to_sq: int,
    color=(255, 80, 80), progress: Optional[float] = None
    ):
    """绘制战术箭头，progress 非 None 时叠加移动指示圆点（滑动阶段）。

    B2：画到 2x 放大的局部 overlay，LANCZOS 缩回 1x 再合成，消除锯齿；
    关闭 ENABLE_AA_ARROWS 时回退 1x 直接绘制。overlay 范围取起终点外扩一格，
    避免箭头尖/圆点被裁。
    """
    fx, fy = _sq_center(from_sq)
    tx, ty = _sq_center(to_sq)
    rgb = color[:3]

    if not ENABLE_AA_ARROWS:
        draw = ImageDraw.Draw(img)
        draw.line([(fx, fy), (tx, ty)], fill=rgb + (150,), width=5)
        angle = math.atan2(ty - fy, tx - fx)
        al, aa = 14, math.pi / 6
        p1 = (int(tx - al * math.cos(angle - aa)), int(ty - al * math.sin(angle - aa)))
        p2 = (int(tx - al * math.cos(angle + aa)), int(ty - al * math.sin(angle + aa)))
        draw.polygon([(int(tx), int(ty)), p1, p2], fill=rgb + (150,))
        if progress is not None:
            dot_x = int(lerp(fx, tx, progress))
            dot_y = int(lerp(fy, ty, progress))
            r = 5
            bright = tuple(min(255, c + 80) for c in rgb)
            draw.ellipse([dot_x - r, dot_y - r, dot_x + r, dot_y + r], fill=bright)
        return

    # ---- 2x overlay 抗锯齿 ----
    pad = SQUARE  # 外扩一格，保证箭头尖/圆点不被裁
    min_x = min(fx, tx) - pad
    min_y = min(fy, ty) - pad
    max_x = max(fx, tx) + pad
    max_y = max(fy, ty) + pad
    ow, oh = max_x - min_x, max_y - min_y
    SCALE = 2
    overlay = Image.new("RGBA", (ow * SCALE, oh * SCALE), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    sfx, sfy = (fx - min_x) * SCALE, (fy - min_y) * SCALE
    stx, sty = (tx - min_x) * SCALE, (ty - min_y) * SCALE
    # 主线（2x 线宽）
    od.line([(sfx, sfy), (stx, sty)], fill=rgb + (150,), width=5 * SCALE)
    # 箭头尖（2x 尺寸）
    angle = math.atan2(sty - sfy, stx - sfx)
    al, aa = 14 * SCALE, math.pi / 6
    p1 = (stx - al * math.cos(angle - aa), sty - al * math.sin(angle - aa))
    p2 = (stx - al * math.cos(angle + aa), sty - al * math.sin(angle + aa))
    od.polygon([(stx, sty), p1, p2], fill=rgb + (150,))
    # 移动指示圆点
    if progress is not None:
        dot_x = int(lerp(sfx, stx, progress))
        dot_y = int(lerp(sfy, sty, progress))
        r = 5 * SCALE
        bright = tuple(min(255, c + 80) for c in rgb)
        od.ellipse([dot_x - r, dot_y - r, dot_x + r, dot_y + r], fill=bright)
    # 缩回 1x 并以 alpha 合成到画布
    down = overlay.resize((ow, oh), Image.LANCZOS)
    img.paste(down, (min_x, min_y), down)

def _draw_pieces_static(
    img: Image.Image, board: chess.Board, skip_sq: Optional[int] = None
    ):
    """ 绘制静止棋子（每枚棋子先粘投影再粘本体） """
    for sq, piece in board.piece_map().items():
        if sq == skip_sq:
            continue
        x, y = _sq_xy(sq)
        piece_img = _load_piece(str(piece))
        _paste_piece(img, piece_img, x, y, shadow_char=str(piece))


# ============================================================
# PLAN-005 Core 3 落子结果澄清（仅 python-chess 确定性事件）
# - 将军攻击线：从将军子到被将王画红色虚线，区别于实线走子箭头
# - 将杀无逃生格：在被将王周围被对方控制的格标低干扰红点
# 均只在 glow/hold 帧出现，由 board 状态确定性计算，不做战术推断。
# ============================================================
COLOR_ATTACK_LINE = (235, 70, 70)
COLOR_ESCAPE_BLOCKED = (235, 70, 70, 120)   # 半透明，低干扰

def _draw_check_attack_line(img: Image.Image, board: chess.Board):
    """将军时从每个将军子画一条红色虚线到被将王（board.checkers() 确定性）。"""
    if not board.is_check():
        return
    king_sq = board.king(board.turn)
    if king_sq is None:
        return
    kx, ky = _sq_center(king_sq)
    for attacker_sq in board.checkers():
        ax, ay = _sq_center(attacker_sq)
        _draw_dashed_line(img, (ax, ay), (kx, ky), COLOR_ATTACK_LINE,
                          dash=10, gap=7, width=3)

def _draw_dashed_line(img: Image.Image, p0: tuple, p1: tuple,
                      color: tuple, dash: int = 10, gap: int = 7, width: int = 3):
    """沿 p0->p1 画半透明虚线。

    PIL 的 ImageDraw.line 在 RGBA 主图上是像素替换而非 alpha 合成，
    且 _save 会把主图 convert('RGB') 丢弃 alpha——直接画无法实现半透明。
    故在整图大小的透明 overlay 上聚合所有虚线段，再一次 paste 走真正的 alpha 合成。
    overlay 一次构造、段数少（每条线约 6 段），开销可忽略。
    """
    x0, y0 = p0
    x1, y1 = p1
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1:
        return
    ux, uy = (x1 - x0) / dist, (y1 - y0) / dist
    rgb = color[:3]
    alpha = color[3] if len(color) >= 4 else 180
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    covered = 0.0
    while covered < dist:
        seg_end = min(covered + dash, dist)
        sx, sy = x0 + ux * covered, y0 + uy * covered
        ex, ey = x0 + ux * seg_end, y0 + uy * seg_end
        od.line([(sx, sy), (ex, ey)], fill=rgb + (alpha,), width=width)
        covered = seg_end + gap
    img.paste(overlay, (0, 0), overlay)

def _draw_mate_escape_blocks(img: Image.Image, board: chess.Board):
    """将杀时在被将王周围被对方控制的空格标低干扰半透明红点，配合现有金框。

    仅标记王邻接 8 格中无己方棋子且被对方攻击的格，避免满屏脏。
    攻击关系由 board.attacks() 对剩余对方棋子聚合得到（确定性）。
    """
    if not board.is_checkmate():
        return
    king_sq = board.king(board.turn)
    if king_sq is None:
        return
    attacker_color = not board.turn
    # 聚合对方所有攻击格（棋子类型无关，仅用于标记"被控制"）
    attacked: set = set()
    temp = board.copy()
    # 移除被将王，避免其自占格影响攻击判定
    temp.remove_piece_at(king_sq)
    for sq, piece in temp.piece_map().items():
        if piece.color == attacker_color:
            attacked.update(temp.attacks(sq))
    draw = ImageDraw.Draw(img)
    for nb in chess.SQUARES:
        if chess.square_distance(king_sq, nb) != 1:
            continue
        piece = board.piece_at(nb)
        if piece is not None and piece.color == board.turn:
            continue  # 己方棋子占的格不标
        if nb in attacked:
            _draw_escape_dot(img, draw, nb)


def _draw_escape_dot(img: Image.Image, draw: ImageDraw.ImageDraw, sq: int):
    """在格中心画一个低干扰半透明红点（小尺寸，避免压住其他视觉元素）。"""
    cx, cy = _sq_center(sq)
    r = 8
    # 外晕：半透明红圆
    halo = Image.new("RGBA", (r * 4, r * 4), (0, 0, 0, 0))
    hd = ImageDraw.Draw(halo)
    hd.ellipse([r, r, r * 3, r * 3], fill=COLOR_ESCAPE_BLOCKED)
    img.paste(halo, (cx - r * 2, cy - r * 2), halo)
    # 中心实心小点
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(235, 70, 70, 220))


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
    # PLAN-006：当前着法按 emphasis 着色
    _emphasis = info.get("emphasis_level", "important")
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
        # PLAN-006：最新着法按 emphasis 着色，其余保持灰色
        _HIST_COLORS = {"pivotal": (255, 235, 150), "important": (230, 230, 238),
                        "routine": (140, 140, 148)}
        line = ""
        row_y = y + 24
        for token in recent[:-1]:
            trial = (line + " " + token).strip()
            if draw.textlength(trial, font=font_hist) > inner_w and line:
                draw.text((inner_x, row_y), line, fill=(190, 190, 198),
                          font=font_hist, anchor="lt")
                row_y += 18
                line = token
            else:
                line = trial
        # 最后一枚 token（当前着法）用 emphasis 色绘制
        last_color = _HIST_COLORS.get(_emphasis, (190, 190, 198))
        if recent:
            trial = (line + " " + recent[-1]).strip()
            if draw.textlength(trial, font=font_hist) > inner_w and line:
                draw.text((inner_x, row_y), line, fill=(190, 190, 198),
                          font=font_hist, anchor="lt")
                row_y += 18
                line = recent[-1]
            else:
                line = trial
        if line:
            draw.text((inner_x, row_y), line, fill=last_color,
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

    # 棋盘投影层（背景之上、棋盘格之下）
    _paste_board_shadow(img)
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
        _draw_arrow(img, from_sq, to_sq, arrow_color, progress=arrow_progress)

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

    # Core 3：落子结果澄清（仅 glow/hold 帧，确定性计算，普通步不触发）
    if is_check:
        _draw_check_attack_line(img, board)
    if is_mate:
        _draw_mate_escape_blocks(img, board)

    return img

#  走法动画序列

def _render_move_sequence(
    board_before: chess.Board, move: chess.Move, board_after: chess.Board,
    hold_sec: float, is_check: bool = False, info: Optional[dict] = None,
    sub_colors=None, is_mate: bool = False, phase_label_name: str = "",
    phase_label_fade_frames: int = 0, phase_label_start_frame: int = 0,
    _sub_frame_idx: int = 0, slide_sec: float = SLIDE_SEC,
    emphasis: str = "important"
    ) -> List[Tuple[Image.Image, float]]:
    """
    为单步走法生成 (帧, 时长) 序列
    三阶段：滑动(slide_sec) → 落子高光脉冲(glow) → 定格保持(hold_sec)
    滑动阶段箭头带移动指示圆点。定格保持为单帧长时长。
    phase_label_* 参数用于在首帧叠加阶段标记。
    PLAN-006：pivotal 落子辉光更强更长（局部效果，作用于落点格）。
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

        slide_n = max(2, round(slide_sec * FPS))
        for i in range(slide_n):
            t = ease_in_out_cubic(i / (slide_n - 1))
            img = _get_background(CANVAS_W, CANVAS_H)
            draw = ImageDraw.Draw(img)

            # 棋盘投影层（背景之上、棋盘格之下）
            _paste_board_shadow(img)
            _draw_board(draw)
            _draw_coordinates(draw)
            _draw_highlight(img, from_sq, from_hl or COLOR_HIGHLIGHT_FROM)
            _draw_pieces_static(img, board_before, skip_sq=from_sq)

            # 被吃棋子渐隐
            if captured is not None:
                cap_img = _load_piece(str(captured)).copy()
                cap_img.putalpha(int(255 * max(0.0, 1.0 - t)))
                img.paste(cap_img, (to_x, to_y), cap_img)

            # 移动棋子（带投影；y-2 维持原有轻微上浮感）
            cur_x = int(lerp(from_x, to_x, t))
            cur_y = int(lerp(from_y, to_y, t))
            _paste_piece(img, piece_img, cur_x, cur_y - 2, shadow_char=str(piece))

            # 箭头 + 移动指示圆点（progress 与棋子同步）
            _draw_arrow(img, from_sq, to_sq, arrow_col, progress=t)

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

    # ---- 阶段2：落子高光脉冲（PLAN-006：pivotal 辉光更强更长）----
    glow_color = (COLOR_CHECK_GLOW if is_check
                  else COLOR_CAPTURE_GLOW if is_capture
                  else COLOR_GLOW)
    _glow_sec = GLOW_SEC_PIVOTAL if emphasis == "pivotal" else GLOW_SEC
    _glow_boost = 1.4 if emphasis == "pivotal" else 1.0
    glow_n = max(2, round(_glow_sec * FPS))
    for i in range(glow_n):
        intensity = math.sin((i / (glow_n - 1)) * math.pi) * _glow_boost
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

def _step_overhead_sec(slide_sec: float = SLIDE_SEC, emphasis: str = "important") -> float:
    """单个子步「滑动+高光」的固定开销（秒），不含定格。PLAN-006：pivotal 辉光更长。"""
    slide_n = max(2, round(slide_sec * FPS))
    _glow_sec = GLOW_SEC_PIVOTAL if emphasis == "pivotal" else GLOW_SEC
    glow_n = max(2, round(_glow_sec * FPS))
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
        seg_emphasis = getattr(seg, "emphasis_level", "important") or "important"

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
                current_phase=seg_phase, emphasis=seg_emphasis
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
        step_overhead = _step_overhead_sec(getattr(seg, "slide_sec", SLIDE_SEC), seg_emphasis)
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
                current_phase=seg_phase, emphasis=seg_emphasis
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
                    _sub_frame_idx=global_frame_idx,
                    slide_sec=getattr(seg, "slide_sec", SLIDE_SEC),
                    emphasis=seg_emphasis):
                _save(img, dur)
                seg_rendered += dur
                global_frame_idx += 1

        seg.start_time = seg_start_cursor
        seg.duration_s = seg_rendered   # 仅写画面占用时长（含动画最低预算，可能略长于语音）
        # speech_duration_s（真实语音截止）由 TTS 写入，渲染器不得改写——字幕据此分配 cue 避免落入尾静音
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
    current_phase: str = "", emphasis: str = "important"
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
        "emphasis_level": emphasis,
    }