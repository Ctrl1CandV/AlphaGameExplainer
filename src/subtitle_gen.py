import os
import re
from typing import List
from src.common import Segment

MAX_CHARS_PER_LINE = 15   # 每行最多字符（中文）
MAX_LINES = 2             # 单条字幕最多行数
MIN_CUE_SEC = 1.0         # 单句字幕最短停留
MAX_CUE_CHARS = 30        # 单条字幕（≤2行）承载的最多字符，超过则继续切句


def _split_sentences(text: str) -> List[str]:
    """按句末标点把整段解说切成短句，保留标点；过长的句子再按逗号/顿号细分"""
    text = text.strip()
    if not text:
        return []

    # 先按强标点断句（句号/问号/感叹号/分号），保留标点
    rough = re.split(r"(?<=[。！？；])", text)
    rough = [s.strip() for s in rough if s.strip()]

    cues: List[str] = []
    for sent in rough:
        if len(sent) <= MAX_CUE_CHARS:
            cues.append(sent)
            continue
        # 长句按次级标点（逗号/顿号/冒号）继续切，累积到接近上限就出一条
        parts = re.split(r"(?<=[，、：,])", sent)
        buf = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if buf and len(buf) + len(p) > MAX_CUE_CHARS:
                cues.append(buf)
                buf = p
            else:
                buf += p
        if buf:
            # 仍超长（无标点的长串）则按字数硬切
            while len(buf) > MAX_CUE_CHARS:
                cues.append(buf[:MAX_CUE_CHARS])
                buf = buf[MAX_CUE_CHARS:]
            if buf:
                cues.append(buf)
    return cues


def _wrap_lines(text: str) -> str:
    """把一条字幕按标点优先折成 ≤MAX_LINES 行，不截断内容"""
    if len(text) <= MAX_CHARS_PER_LINE:
        return text
    lines: List[str] = []
    remaining = text
    while remaining and len(lines) < MAX_LINES:
        if len(remaining) <= MAX_CHARS_PER_LINE:
            lines.append(remaining)
            remaining = ""
            break
        # 在行宽附近找标点断行
        best = -1
        for pos in range(min(MAX_CHARS_PER_LINE, len(remaining) - 1),
                         max(0, MAX_CHARS_PER_LINE - 6), -1):
            if remaining[pos] in "，。、；！？：,":
                best = pos + 1
                break
        if best == -1:
            best = MAX_CHARS_PER_LINE
        lines.append(remaining[:best])
        remaining = remaining[best:]
    if remaining:
        # 剩余内容并入最后一行（宁可略长也不丢内容）
        lines[-1] = lines[-1] + remaining
    return "\n".join(lines)


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _allocate_cues(text: str, start: float, duration: float) -> List[tuple]:
    """把一段解说切成多句，按字数比例分配到 [start, start+duration] 内逐句推进"""
    cues = _split_sentences(text)
    if not cues:
        return []
    total_chars = sum(len(c) for c in cues) or 1
    result = []
    t = start
    for c in cues:
        share = duration * (len(c) / total_chars)
        cue_dur = max(MIN_CUE_SEC, share)
        result.append((t, t + cue_dur, _wrap_lines(c)))
        t += cue_dur
    # 末句对齐到该段结束，避免逐句 max 累积溢出本段
    if result:
        last_start = result[-1][0]
        end = max(last_start + MIN_CUE_SEC, start + duration)
        result[-1] = (last_start, end, result[-1][2])
    return result


def generate(segments: List[Segment], offset_s: float = 0.0) -> str:
    """基于 Segments 生成逐句滚动的 SRT 字幕文件"""
    lines = []
    num = 1
    for seg in segments:
        if not seg.text.strip():
            continue
        seg_start = offset_s + seg.start_time
        for start, end, text in _allocate_cues(seg.text, seg_start, seg.duration_s):
            lines.append(str(num))
            lines.append(f"{_fmt_time(start)} --> {_fmt_time(end)}")
            lines.append(text)
            lines.append("")
            num += 1

    srt_path = os.path.join("output", "subtitles.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return srt_path
