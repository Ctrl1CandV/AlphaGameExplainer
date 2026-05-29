import os
from typing import List
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    ImageSequenceClip, AudioFileClip, CompositeVideoClip,
    concatenate_audioclips, TextClip, ColorClip,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from src.common import Logger
from src.board_renderer import IMG_W, IMG_H, COLOR_BG

TITLE_SEC = 2.5
FPS = 24
SUBTITLE_HEIGHT = 80
SUBTITLE_MARGIN = 20


def _make_title_card(endgame_name: str, width: int, height: int) -> str:
    """生成片头标题卡"""
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

    # 标题
    title_y = height // 2 - 50
    draw.text((width // 2, title_y), "AI 讲棋",
              fill=(255, 215, 0), font=font_big, anchor="mm")
    
    # 副标题
    draw.text((width // 2, title_y + 50), endgame_name,
              fill=(220, 220, 220), font=font_small, anchor="mm")
    
    # 装饰线
    line_y = title_y + 80
    line_width = 200
    draw.line([(width // 2 - line_width // 2, line_y), 
               (width // 2 + line_width // 2, line_y)], 
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
            fps: int = FPS) -> str:
    """
    合成最终视频。
    frame_paths: 帧图片路径
    frame_durations: 每帧显示秒数
    segments: 含 audio_path 的段列表
    srt_path: SRT 字幕文件
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
    title_path = _make_title_card(endgame_name or "残局讲解", frame_w, frame_h)

    # 组装帧序列
    all_frames = [title_path] + frame_paths
    all_durations = [TITLE_SEC] + frame_durations

    video = ImageSequenceClip(all_frames, durations=all_durations)

    # 组装音频: 片头静音 + TTS 音频
    silence_path = os.path.join(frames_dir, "_silence.wav")
    _make_silence(TITLE_SEC, silence_path)
    audio_clips = [AudioFileClip(silence_path)]
    for seg in segments:
        if seg.audio_path and os.path.exists(seg.audio_path):
            try:
                audio_clips.append(AudioFileClip(seg.audio_path))
            except Exception as e:
                Logger.warn(f"跳过音频 {seg.audio_path}: {e}")
    if len(audio_clips) > 1:
        video = video.with_audio(concatenate_audioclips(audio_clips))

    # 字幕背景
    sub_bg_path = _create_subtitle_background(frame_w, SUBTITLE_HEIGHT)
    sub_bg_clip = (ImageClip(sub_bg_path)
                   .with_duration(video.duration)
                   .with_position(("center", frame_h - SUBTITLE_HEIGHT - SUBTITLE_MARGIN)))

    # 字幕
    _FONT_PATH = "C:/Windows/Fonts/simhei.ttf"

    def _mk_sub(txt):
        return TextClip(
            text=txt, font=_FONT_PATH, font_size=22, color="white",
            stroke_color="black", stroke_width=1,
            method="caption", size=(frame_w - 60, None),
        )
    
    subs = SubtitlesClip(srt_path, encoding="utf-8", make_textclip=_mk_sub)
    subs = subs.with_position(("center", frame_h - SUBTITLE_HEIGHT - SUBTITLE_MARGIN + 10))
    
    # 合成：视频 + 字幕背景 + 字幕
    final = CompositeVideoClip([video, sub_bg_clip, subs])

    output = os.path.join("output", "analysis.mp4")
    final.write_videofile(output, codec="libx264", audio_codec="aac", fps=fps)
    final.close()
    video.close()

    Logger.success(f"视频已生成: {output}")
    return output
