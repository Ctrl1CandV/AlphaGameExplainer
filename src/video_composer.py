import os
from typing import List
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    ImageSequenceClip, AudioFileClip, CompositeVideoClip,
    concatenate_audioclips, TextClip, ImageClip,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from src.common import Logger
from src.board_renderer import IMG_W, IMG_H, COLOR_BG, INTRO_SEC, FPS as RENDER_FPS, render_frame, MARGIN_LEFT, MARGIN_TOP, BOARD_SIZE
import chess

TITLE_SEC = 2.5
FPS = RENDER_FPS              # 与渲染帧率统一，避免重采样导致的卡顿
SUBTITLE_HEIGHT = 84
SUBTITLE_MARGIN = 18
# 视频开头静音 = 片头标题卡 + 初始局面静态展示，二者都没有解说音频
LEAD_SILENCE = TITLE_SEC + INTRO_SEC


def _make_title_card(endgame_name: str, width: int, height: int, initial_fen: str = "") -> str:
    """生成片头标题卡，可附带棋盘缩略图"""
    img = Image.new("RGB", (width, height), COLOR_BG)
    draw = ImageDraw.Draw(img)

    # 渐变背景
    for y in range(height):
        r = int(30 + (y / height) * 20)
        g = int(30 + (y / height) * 15)
        b = int(30 + (y / height) * 25)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    try:
        font_big = ImageFont.truetype("simhei.ttf", 48)
        font_small = ImageFont.truetype("simhei.ttf", 24)
        font_tiny = ImageFont.truetype("simhei.ttf", 16)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_tiny = ImageFont.load_default()

    # 左侧棋盘缩略图
    text_x = width // 2
    if initial_fen:
        try:
            b = chess.Board(initial_fen)
            thumb = render_frame(b)
            # 裁剪到纯棋盘区域
            thumb = thumb.crop((
                MARGIN_LEFT, MARGIN_TOP,
                MARGIN_LEFT + BOARD_SIZE, MARGIN_TOP + BOARD_SIZE,
            ))
            thumb_size = min(height - 80, 300)
            thumb = thumb.resize((thumb_size, thumb_size))
            thumb_x = (width // 2 - thumb_size) // 2
            thumb_y = (height - thumb_size) // 2
            # 缩略图投影
            shadow = Image.new("RGBA", (thumb_size + 12, thumb_size + 12), (0, 0, 0, 80))
            img.paste(shadow, (thumb_x + 5, thumb_y + 5), shadow)
            img.paste(thumb.convert("RGB"), (thumb_x, thumb_y))
            text_x = width // 2 + thumb_size // 2
        except Exception:
            pass

    # 标题
    title_y = height // 2 - 50
    draw.text((text_x, title_y), "AI 讲棋",
              fill=(255, 215, 0), font=font_big, anchor="mm")

    # 副标题
    draw.text((text_x, title_y + 50), endgame_name,
              fill=(220, 220, 220), font=font_small, anchor="mm")

    # 装饰线
    line_y = title_y + 80
    line_width = 200
    draw.line([(text_x - line_width // 2, line_y),
               (text_x + line_width // 2, line_y)],
              fill=(100, 100, 100), width=2)

    # 底部信息
    draw.text((width // 2, height - 40), "国际象棋残局教学",
              fill=(150, 150, 150), font=font_tiny, anchor="mm")

    path = os.path.join("output", "frames", "title_card.png")
    img.save(path)
    return path


def _make_silence(duration: float, output_path: str) -> str:
    """生成静音片段"""
    from pydub import AudioSegment
    silence = AudioSegment.silent(duration=int(duration * 1000))
    silence.export(output_path, format="wav")
    return output_path


def _create_subtitle_background(width: int, height: int) -> str:
    """创建半透明字幕背景"""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 180))
    path = os.path.join("output", "frames", "subtitle_bg.png")
    img.save(path)
    return path


def compose(frame_paths: List[str], frame_durations: List[float],
            segments, srt_path: str, endgame_name: str = "",
            fps: int = FPS, cues=None, initial_fen: str = "") -> str:
    """
    合成最终视频。
    initial_fen: 初始局面FEN，用于片头棋盘缩略图
    """
    Logger.info("合成视频...")

    frames_dir = os.path.join("output", "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # 检测帧尺寸
    frame_w, frame_h = IMG_W, IMG_H
    if frame_paths:
        try:
            with Image.open(frame_paths[0]) as test:
                frame_w, frame_h = test.size
        except Exception:
            pass

    # 片头
    title_path = _make_title_card(endgame_name or "残局讲解", frame_w, frame_h, initial_fen)

    # 组装帧序列：片头标题卡 + 渲染帧
    all_frames = [title_path] + frame_paths
    all_durations = [TITLE_SEC] + frame_durations

    video = ImageSequenceClip(all_frames, durations=all_durations)

    # 组装音频: 开头静音(标题卡+初始局面，无解说) + 逐段 TTS 音频
    # 关键: 每段音频后补足静音，使该段音频块时长精确等于 seg.duration_s，
    # 与帧/字幕时间轴对齐（否则 concat 无间隙会逐段累积漂移，字幕落后于音频）。
    silence_path = os.path.join(frames_dir, "_silence.wav")
    _make_silence(LEAD_SILENCE, silence_path)
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
        # 该段目标时长（与帧/字幕一致）减去音频实际时长，差额补静音
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

    # 字幕带：固定在底部留白区（棋盘下方，不覆盖棋盘）
    band_top = frame_h - SUBTITLE_HEIGHT - SUBTITLE_MARGIN

    # 字体查找（回退链：simhei → msyh → 系统默认）
    _FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
    if not os.path.exists(_FONT_PATH):
        _FONT_PATH = "C:/Windows/Fonts/msyh.ttf"
    if not os.path.exists(_FONT_PATH):
        _FONT_PATH = "C:/Windows/Fonts/arial.ttf"

    def _mk_sub(txt):
        return TextClip(
            text=txt, font=_FONT_PATH, font_size=22, color="white",
            stroke_color="black", stroke_width=1,
            method="caption", size=(frame_w - 40, SUBTITLE_HEIGHT),
            text_align="center",
        )

    # 优先用 cue 列表构造字幕，绕开 moviepy 脆弱的 SRT 文件解析；
    # 列表为空（无任何字幕）时跳过字幕层，避免 SubtitlesClip 对空列表崩溃。
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
        subs = subs.with_position((20, band_top))
        layers.extend([sub_bg_clip, subs])
    else:
        Logger.warn("无有效字幕，跳过字幕层")

    # 合成：视频 + 字幕背景 + 字幕
    final = CompositeVideoClip(layers)

    output = os.path.join("output", "analysis.mp4")
    final.write_videofile(output, codec="libx264", audio_codec="aac", fps=fps)
    final.close()
    video.close()

    Logger.success(f"视频已生成: {output}")
    return output
