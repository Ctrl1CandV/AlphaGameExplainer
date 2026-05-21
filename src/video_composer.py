import os
from typing import List
from moviepy import (
    ImageSequenceClip, AudioFileClip, CompositeVideoClip,
    concatenate_audioclips, TextClip,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from src.common import Logger
from src.board_renderer import IMG_W

IMG_H = IMG_W


def compose(
    frame_paths: List[str],
    frame_durations: List[float],
    segments,
    srt_path: str,
    fps: int = 24,
) -> str:
    """
    合成最终视频：
      - frame_paths: 帧图片路径列表
      - frame_durations: 每帧显示秒数
      - segments: 含 audio_path 的段列表
      - srt_path: 字幕文件路径
    返回: 输出 mp4 路径
    """
    Logger.info("合成视频中...")

    video = ImageSequenceClip(frame_paths, durations=frame_durations)

    audio_clips = []
    for seg in segments:
        if seg.audio_path and os.path.exists(seg.audio_path):
            audio_clips.append(AudioFileClip(seg.audio_path))
    if audio_clips:
        video = video.with_audio(concatenate_audioclips(audio_clips))

    def _mk_sub(txt):
        return TextClip(
            text=txt, font_size=28, color="white",
            stroke_color="black", stroke_width=2,
            method="caption", size=(600, None),
        )

    subs = SubtitlesClip(srt_path, encoding="utf-8", make_textclip=_mk_sub)
    subs = subs.with_position(("center", IMG_H - 80))

    final = CompositeVideoClip([video, subs])

    output = os.path.join("output", "analysis.mp4")
    final.write_videofile(output, codec="libx264", audio_codec="aac", fps=fps)
    final.close()
    video.close()

    Logger.success(f"视频已生成: {output}")
    return output
