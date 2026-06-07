import os
from typing import List
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    ImageSequenceClip, AudioFileClip, CompositeVideoClip,
    concatenate_audioclips, TextClip, ImageClip,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from src.common import Logger
from src.board_renderer import (
    CANVAS_W, CANVAS_H, BOARD_LEFT, BOARD_TOP, BOARD_SIZE,
    COLOR_BG, INTRO_SEC, FPS as RENDER_FPS, render_frame,
    IS_VERTICAL,
)
import chess

TITLE_SEC = 3.5                 # 片头动画总时长
FPS = RENDER_FPS                # 与渲染帧率统一
# 竖版字幕稍大、位置稍高，适配手机屏阅读
SUBTITLE_HEIGHT = 80 if IS_VERTICAL else 62
SUBTITLE_MARGIN = 16 if IS_VERTICAL else 10
# 视频开头静音 = 片头标题卡 + 初始局面静态展示
LEAD_SILENCE = TITLE_SEC + INTRO_SEC


# ============================================================
#  片头动画
# ============================================================

def _make_title_frames(endgame_name: str, width: int, height: int,
                        initial_fen: str = "") -> List[Image.Image]:
    """生成片头动画帧序列（~3.5s × FPS 帧）。

    动画时间轴：
      0.0-0.8s:  背景从纯黑渐变显现
      0.5-1.2s:  棋盘缩略图 105% → 100%（微小 Ken Burns 效果）
      1.0-2.0s:  标题从下方 30px 滑入
      1.5-2.5s:  副标题从下方 20px 滑入
      2.0-3.0s:  装饰线从左到右画出
      3.0-3.5s:  底部文字 fade in
    """
    total = round(TITLE_SEC * FPS)
    frames: List[Image.Image] = []

    # 预渲染棋盘缩略图
    thumb = None
    thumb_size = 0
    if initial_fen:
        try:
            b = chess.Board(initial_fen)
            raw = render_frame(b)
            cropped = raw.crop(
                (BOARD_LEFT, BOARD_TOP, BOARD_LEFT + BOARD_SIZE, BOARD_TOP + BOARD_SIZE))
            thumb_size = min(height - 80, 280)
            thumb = cropped.resize((thumb_size, thumb_size))
        except Exception:
            pass

    # 预加载字体
    try:
        font_big = ImageFont.truetype("simhei.ttf", 48)
        font_small = ImageFont.truetype("simhei.ttf", 24)
        font_tiny = ImageFont.truetype("simhei.ttf", 16)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = font_big
        font_tiny = font_big

    # 静态布局计算
    text_x = width // 2
    if thumb:
        text_x = width // 2 + thumb_size // 2

    for i in range(total):
        t = i / (total - 1) if total > 1 else 0.0
        img = Image.new("RGB", (width, height), (10, 10, 10))
        draw = ImageDraw.Draw(img)

        # 背景渐变（0.0-0.8s fade in）
        bg_alpha = min(1.0, t / 0.25)
        for y in range(height):
            r = int(30 * bg_alpha + (y / height) * 15 * bg_alpha)
            g = int(30 * bg_alpha + (y / height) * 10 * bg_alpha)
            b = int(30 * bg_alpha + (y / height) * 20 * bg_alpha)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # 棋盘缩略图（0.5-1.2s: 105% → 100%）
        if thumb:
            thumb_t_start = 0.15
            thumb_t_end = 0.35
            if t >= thumb_t_start:
                tt = min(1.0, (t - thumb_t_start) / (thumb_t_end - thumb_t_start))
                scale = 1.05 - 0.05 * tt  # 105% → 100%
                sw = int(thumb_size * scale)
                sh = int(thumb_size * scale)
                scaled_thumb = thumb.resize((sw, sh))
                tx = (width // 2 - thumb_size) // 2 + (thumb_size - sw) // 2
                ty = (height - thumb_size) // 2 + (thumb_size - sh) // 2
                # 缩略图投影
                shadow = Image.new("RGBA", (sw + 10, sh + 10), (0, 0, 0, 80))
                img.paste(shadow, (tx + 4, ty + 4), shadow)
                img.paste(scaled_thumb.convert("RGB"), (tx, ty))

        # 标题（1.0-2.0s: 从下方滑入）
        title_t = (t - 0.28) / 0.28
        title_y_base = height // 2 - 50
        if title_t < 0:
            title_y = title_y_base + 40
            title_alpha = 0
        elif title_t < 1.0:
            title_y = title_y_base + int(40 * (1 - title_t))
            title_alpha = min(255, int(255 * title_t))
        else:
            title_y = title_y_base
            title_alpha = 255

        if title_alpha > 0:
            # 文字阴影
            draw.text((text_x + 1, title_y + 1), "AI 讲棋",
                      fill=(0, 0, 0, title_alpha), font=font_big, anchor="mm")
            draw.text((text_x, title_y), "AI 讲棋",
                      fill=(255, 215, 0, title_alpha), font=font_big, anchor="mm")

        # 副标题（1.5-2.5s: 从下方滑入）
        sub_t = (t - 0.42) / 0.28
        sub_y_base = height // 2 + 6
        if sub_t < 0:
            sub_y = sub_y_base + 30
            sub_alpha = 0
        elif sub_t < 1.0:
            sub_y = sub_y_base + int(30 * (1 - sub_t))
            sub_alpha = min(255, int(255 * sub_t))
        else:
            sub_y = sub_y_base
            sub_alpha = 255

        if sub_alpha > 0:
            draw.text((text_x, sub_y), endgame_name,
                      fill=(220, 220, 220, sub_alpha), font=font_small, anchor="mm")

        # 装饰线（2.0-3.0s: 从左到右画出）
        line_t = (t - 0.57) / 0.28
        if 0 < line_t <= 1.0:
            line_y = height // 2 + 36
            line_w = 200
            line_x0 = text_x - line_w // 2
            line_x1 = int(line_x0 + line_w * line_t)
            draw.line([(line_x0, line_y), (line_x1, line_y)],
                      fill=(100, 100, 100), width=2)

        # 底部文字（3.0-3.5s: fade in）
        foot_t = (t - 0.85) / 0.15
        foot_alpha = max(0, min(255, int(255 * foot_t)))
        if foot_alpha > 0:
            draw.text((width // 2, height - 40), "国际象棋残局教学",
                      fill=(150, 150, 150, foot_alpha), font=font_tiny, anchor="mm")

        frames.append(img)

    return frames


# ============================================================
#  片尾画面
# ============================================================

def _make_outro_frames(last_frame: Image.Image, width: int, height: int) -> List[Image.Image]:
    """生成片尾帧序列（~2.5s）：画面渐暗 + "感谢观看" fade in"""
    outro_sec = 2.5
    total = round(outro_sec * FPS)
    frames: List[Image.Image] = []

    try:
        font = ImageFont.truetype("simhei.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    for i in range(total):
        t = i / (total - 1) if total > 1 else 0.0
        frame = last_frame.copy()

        # 画面渐暗（0 → 0.85 半透明黑叠加，模拟暗角效果）
        dark_alpha = min(0.85, t * 2.0)
        if dark_alpha > 0.01:
            overlay = Image.new("RGBA", (width, height),
                                (0, 0, 0, int(255 * dark_alpha)))
            frame.paste(overlay, (0, 0), overlay)

        # "感谢观看"（1.0-2.0s fade in）
        text_t = (t - 0.35) / 0.25
        text_alpha = max(0, min(255, int(255 * text_t)))
        if text_alpha > 0:
            draw = ImageDraw.Draw(frame)
            draw.text((width // 2 + 1, height // 2 + 1), "感谢观看",
                      fill=(0, 0, 0, text_alpha), font=font, anchor="mm")
            draw.text((width // 2, height // 2), "感谢观看",
                      fill=(255, 255, 255, text_alpha), font=font, anchor="mm")

        frames.append(frame)

    return frames


# ============================================================
#  音频 & 字幕 辅助
# ============================================================

def _make_silence(duration: float, output_path: str) -> str:
    """生成静音片段"""
    from pydub import AudioSegment
    silence = AudioSegment.silent(duration=int(duration * 1000))
    silence.export(output_path, format="wav")
    return output_path


def _create_subtitle_background(width: int, height: int) -> str:
    """创建居中圆角字幕卡片背景——不覆盖全宽，仅占画面中央区域。"""
    card_w = min(width - 80, 800)
    card_x = (width - card_w) // 2
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # 居中圆角矩形，半透明深色
    draw.rounded_rectangle(
        [card_x, 0, card_x + card_w, height],
        radius=12, fill=(12, 12, 18, 210),
    )
    path = os.path.join("output", "frames", "subtitle_bg.png")
    img.save(path)
    return path


# ============================================================
#  主合成函数
# ============================================================

def compose(frame_paths: List[str], frame_durations: List[float],
            segments, srt_path: str, endgame_name: str = "",
            fps: int = FPS, cues=None, initial_fen: str = "",
            skip_title: bool = False, skip_outro: bool = False) -> str:
    """合成最终视频。包含片头动画 → 渲染帧 → 片尾，并叠加字幕。

    initial_fen: 初始局面 FEN，用于片头棋盘缩略图
    skip_title: True 时跳过片头标题卡动画（puzzle 链路使用）
    skip_outro: True 时跳过片尾动画（puzzle 链路使用）
    """
    Logger.info("合成视频...")

    frames_dir = os.path.join("output", "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_w, frame_h = CANVAS_W, CANVAS_H

    # ---- 片头动画帧 ----
    title_frames: List[Image.Image] = []
    if not skip_title:
        title_frames = _make_title_frames(
            endgame_name or "残局讲解", frame_w, frame_h, initial_fen)
        Logger.info(f"片头动画: {len(title_frames)} 帧, {TITLE_SEC:.1f}s")
    title_durations = [1.0 / fps] * len(title_frames)

    # ---- 片尾帧 ----
    outro_frames: List[Image.Image] = []
    if not skip_outro and frame_paths:
        try:
            last_frame = Image.open(frame_paths[-1]).convert("RGB")
            outro_frames = _make_outro_frames(last_frame, frame_w, frame_h)
        except Exception as e:
            Logger.warn(f"片尾生成失败: {e}")
    outro_durations = [1.0 / fps] * len(outro_frames)
    if outro_frames:
        Logger.info(f"片尾: {len(outro_frames)} 帧, {len(outro_frames)/fps:.1f}s")

    # ---- 组装帧序列 ----
    title_paths = []
    for idx, f in enumerate(title_frames):
        p = os.path.join(frames_dir, f"_title_{idx:04d}.png")
        f.save(p)
        title_paths.append(p)

    outro_paths = []
    for idx, f in enumerate(outro_frames):
        p = os.path.join(frames_dir, f"_outro_{idx:04d}.png")
        f.save(p)
        outro_paths.append(p)

    all_frames = title_paths + frame_paths + outro_paths
    all_durations = title_durations + frame_durations + outro_durations

    video = ImageSequenceClip(all_frames, durations=all_durations)

    # ---- 组装音频 ----
    # 跳过片头时前导静音仅含初始局面展示，否则含片头动画 + 初始局面展示
    lead_silence = INTRO_SEC if skip_title else LEAD_SILENCE
    silence_path = os.path.join(frames_dir, "_silence.wav")
    _make_silence(lead_silence, silence_path)
    audio_clips = [AudioFileClip(silence_path)]
    pad_paths = []
    for idx, seg in enumerate(segments):
        clip = None
        if seg.audio_path and os.path.exists(seg.audio_path):
            try:
                clip = AudioFileClip(seg.audio_path)
                audio_clips.append(clip)
            except Exception as e:
                Logger.warn(f"跳过音频 {seg.audio_path}: {e}")
        target = max(0.0, float(getattr(seg, "duration_s", 0.0)))
        played = clip.duration if clip is not None else 0.0
        gap = target - played
        if gap > 0.01:
            pad_path = os.path.join(frames_dir, f"_pad_{idx:03d}.wav")
            _make_silence(gap, pad_path)
            audio_clips.append(AudioFileClip(pad_path))
            pad_paths.append(pad_path)
    if len(audio_clips) > 1:
        video = video.with_audio(concatenate_audioclips(audio_clips))

    # ---- 字幕 ----
    band_top = frame_h - SUBTITLE_HEIGHT - SUBTITLE_MARGIN
    card_w = min(frame_w - 80, 800)
    card_x = (frame_w - card_w) // 2

    # 字体查找
    _FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
    if not os.path.exists(_FONT_PATH):
        _FONT_PATH = "C:/Windows/Fonts/msyh.ttf"
    if not os.path.exists(_FONT_PATH):
        _FONT_PATH = "C:/Windows/Fonts/arial.ttf"

    def _mk_sub(txt):
        sub_font_size = 28 if IS_VERTICAL else 26
        return TextClip(
            text=txt, font=_FONT_PATH, font_size=sub_font_size, color="#F0EDE5",
            stroke_color="#1a1a1a", stroke_width=1.5,
            method="caption", size=(card_w, SUBTITLE_HEIGHT),
            text_align="center",
        )

    if cues is None:
        try:
            from moviepy.video.tools.subtitles import file_to_subtitles
            cues = file_to_subtitles(srt_path, encoding="utf-8")
            cues = [c for c in cues if c[0] is not None and len(c[0]) == 2]
        except Exception as e:
            Logger.warn(f"读取字幕文件失败，将不渲染字幕: {e}")
            cues = []

    layers = [video]
    if cues:
        sub_bg_path = _create_subtitle_background(frame_w, SUBTITLE_HEIGHT)
        sub_bg_clip = (ImageClip(sub_bg_path)
                       .with_duration(video.duration)
                       .with_position((0, band_top)))
        subs = SubtitlesClip(cues, make_textclip=_mk_sub)
        subs = subs.with_position((card_x, band_top))
        layers.extend([sub_bg_clip, subs])
    else:
        Logger.warn("无有效字幕，跳过字幕层")

    final = CompositeVideoClip(layers)

    output = os.path.join("output", "analysis.mp4")
    final.write_videofile(output, codec="libx264", audio_codec="aac", fps=fps)
    final.close()
    video.close()

    # 清理临时帧文件
    for p in title_paths + outro_paths:
        try:
            os.remove(p)
        except Exception:
            pass

    Logger.success(f"视频已生成: {output}")
    return output
