import os
from typing import List
from src.common import Segment


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate(segments: List[Segment]) -> str:
    """基于 Segments 生成 SRT 字幕文件"""
    lines = []
    num = 1
    for seg in segments:
        if not seg.text.strip():
            continue
        start = seg.start_time
        end = start + seg.duration_s
        lines.append(str(num))
        lines.append(f"{_fmt_time(start)} --> {_fmt_time(end)}")
        lines.append(seg.text)
        lines.append("")
        num += 1

    srt_path = os.path.join("output", "subtitles.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return srt_path
