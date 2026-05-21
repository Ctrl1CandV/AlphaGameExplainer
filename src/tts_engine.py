import os
import pyttsx3
from pydub import AudioSegment
from typing import List
from src.common import Segment, Logger

AUDIO_DIR = os.path.join("output", "audio")


def _find_chinese_voice(engine) -> str:
    for v in engine.getProperty("voices"):
        for lang in getattr(v, "languages", []):
            if isinstance(lang, bytes):
                lang = lang.decode()
            if "zh" in str(lang).lower():
                return v.id
    return None


def synthesize(segments: List[Segment], rate: int = 180) -> List[Segment]:
    """
    逐段合成中文语音，将音频时长回填到 Segment。
    返回: 补全了 audio_path 和 duration_s 的 Segment 列表。
    """
    os.makedirs(AUDIO_DIR, exist_ok=True)
    engine = pyttsx3.init("sapi5")

    voice_id = _find_chinese_voice(engine)
    if voice_id:
        engine.setProperty("voice", voice_id)
    engine.setProperty("rate", rate)

    Logger.info("合成语音中...")
    time_cursor = 0.0

    for seg in segments:
        if not seg.text.strip():
            seg.duration_s = 1.0
            seg.start_time = time_cursor
            time_cursor += seg.duration_s
            continue

        path = os.path.join(AUDIO_DIR, f"seg_{seg.move_idx:03d}.wav")
        engine.save_to_file(seg.text, path)
        engine.runAndWait()

        try:
            audio = AudioSegment.from_wav(path)
            seg.duration_s = max(1.0, audio.duration_seconds + 0.3)
        except Exception:
            seg.duration_s = max(1.0, len(seg.text) * 0.1)

        seg.audio_path = path
        seg.start_time = time_cursor
        time_cursor += seg.duration_s

    engine.stop()
    Logger.success(f"语音合成完成: {len(segments)} 段, 总时长 {time_cursor:.1f}s")
    return segments
