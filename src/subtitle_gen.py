import os
from typing import List
from src.common import Segment

MAX_CHARS_PER_LINE = 16
MAX_LINES = 2
MAX_TOTAL_CHARS = 60


def _split_text(text: str) -> str:
    """长文本按标点智能分行，限制最大行数和字符数"""
    text = text.strip()
    
    # 截断过长文本
    if len(text) > MAX_TOTAL_CHARS:
        # 找最后一个标点截断
        for i in range(min(MAX_TOTAL_CHARS, len(text) - 1), MAX_TOTAL_CHARS - 20, -1):
            if text[i] in "。！？；":
                text = text[:i + 1]
                break
        else:
            text = text[:MAX_TOTAL_CHARS] + "..."
    
    if len(text) <= MAX_CHARS_PER_LINE:
        return text

    # 智能分行
    lines = []
    remaining = text
    
    while remaining and len(lines) < MAX_LINES:
        if len(remaining) <= MAX_CHARS_PER_LINE:
            lines.append(remaining)
            break
        
        # 在合适位置断行
        best_pos = -1
        for pos in range(min(MAX_CHARS_PER_LINE, len(remaining)), max(0, MAX_CHARS_PER_LINE - 8), -1):
            if remaining[pos] in "，。、；！？：":
                best_pos = pos + 1
                break
        
        if best_pos == -1:
            # 没有标点，在空格或任意位置断
            best_pos = MAX_CHARS_PER_LINE
        
        lines.append(remaining[:best_pos])
        remaining = remaining[best_pos:]
    
    # 如果还有剩余且达到行数限制，最后一行加省略号
    if remaining and len(lines) >= MAX_LINES:
        lines[-1] = lines[-1][:MAX_CHARS_PER_LINE - 3] + "..."
    
    return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate(segments: List[Segment], offset_s: float = 0.0) -> str:
    """基于 Segments 生成 SRT 字幕文件"""
    lines = []
    num = 1
    for seg in segments:
        if not seg.text.strip():
            continue
        start = offset_s + seg.start_time
        end = start + seg.duration_s
        lines.append(str(num))
        lines.append(f"{_fmt_time(start)} --> {_fmt_time(end)}")
        lines.append(_split_text(seg.text))
        lines.append("")
        num += 1

    srt_path = os.path.join("output", "subtitles.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return srt_path
