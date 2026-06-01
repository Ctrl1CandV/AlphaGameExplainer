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
    # 先算出各 cue 的理想时长
    raw_durs = [duration * (len(c) / total_chars) for c in cues]
    # 若总时长超出 duration，按比例压缩回区间内
    raw_total = sum(raw_durs)
    scale = duration / raw_total if raw_total > duration else 1.0
    result = []
    t = start
    end = start + duration
    for i, c in enumerate(cues):
        cue_dur = max(MIN_CUE_SEC, raw_durs[i] * scale)
        # 守卫：不允许最后一个 cue 超出段结束时间
        if i == len(cues) - 1:
            cue_dur = max(MIN_CUE_SEC, end - t)
        result.append((t, t + cue_dur, _wrap_lines(c)))
        t += cue_dur
    return result


def build_cues(segments: List[Segment], offset_s: float = 0.0) -> List[tuple]:
    """构建字幕 cue 列表，格式 [((start_s, end_s), text), ...]。

    直接供 moviepy 的 SubtitlesClip(list) 使用，绕开其脆弱的 SRT 文本解析
    （file_to_subtitles 对多行文本/空行/数字冒号易误判，产生 None 时间戳后崩溃）。
    过滤空文本 cue，确保下游不会拿到非法条目。
    """
    cues: List[tuple] = []
    for seg in segments:
        if not seg.text.strip():
            continue
        seg_start = offset_s + seg.start_time
        for start, end, text in _allocate_cues(seg.text, seg_start, seg.duration_s):
            t = (text or "").strip()
            if not t:
                continue
            cues.append(((float(start), float(end)), t))
    return cues


def generate(segments: List[Segment], offset_s: float = 0.0) -> str:
    """基于 Segments 生成 SRT 字幕文件（留档/调试用），返回路径。

    视频合成不再依赖此文件解析，改用 build_cues() 的列表直接构造字幕。
    """
    cues = build_cues(segments, offset_s)
    lines = []
    for num, ((start, end), text) in enumerate(cues, 1):
        lines.append(str(num))
        lines.append(f"{_fmt_time(start)} --> {_fmt_time(end)}")
        lines.append(text)
        lines.append("")

    srt_path = os.path.join("output", "subtitles.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        # 结尾补换行，确保最后一条 cue 后也有空行分隔（标准 SRT 要求）。
        f.write("\n".join(lines) + "\n")
    return srt_path
