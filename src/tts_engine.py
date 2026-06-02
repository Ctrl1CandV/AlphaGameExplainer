import os
import re
import time
from typing import List, Optional
from pydub import AudioSegment
from src.common import Segment, Logger

AUDIO_DIR = os.path.join("output", "audio")
DEFAULT_VOICE = os.path.join("assets", "voices", "ref_male_pro.wav")
# 持久化的 ChatTTS 说话人向量，保证跨运行音色一致（首次随机采样后写盘复用）
SPEAKER_FILE = os.path.join("assets", "voices", "chattts_speaker.txt")
# 全片统一目标响度（dBFS），消除 ChatTTS 逐段幅度不一致导致的忽轻忽响
TARGET_DBFS = -20.0

PACING_MAP = {
    "slow":          ("calm", 0.85),
    "normal":        ("default", 1.0),
    "fast":          ("default", 1.15),
    "pause_before":  ("excited", 0.9),
    "pause_after":   ("calm", 0.9),
}

# ChatTTS 模型缓存
_chattts: Optional[object] = None
_chattts_spk_emb: Optional[str] = None
_CHATTTS_SAMPLE_RATE = 24000


def _free_gpu_before_tts():
    """ChatTTS 按"加载时空闲显存"选设备(阈值约2GB)，加载前主动清一次 torch 缓存，
    避免上游 LLM 残留显存导致 ChatTTS 静默回退 CPU（变慢且音质不稳）。"""
    try:
        import torch
        import gc
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            free_mb = free / (1024 * 1024)
            if free_mb < 2200:
                Logger.warn("GPU 显存不足，ChatTTS 可能回退 CPU")
    except Exception:
        pass


def _init_chattts():
    """加载 ChatTTS 模型（首次调用时加载，后续复用）"""
    global _chattts, _chattts_spk_emb
    if _chattts is not None:
        return True

    try:
        from ChatTTS import Chat
        _free_gpu_before_tts()
        Logger.info("加载 ChatTTS 模型...")
        chat = Chat()
        ok = chat.load(compile=False, source="huggingface")
        if not ok:
            Logger.warn("ChatTTS 模型加载失败")
            return False
        _chattts = chat
        _chattts_spk_emb = _load_or_create_speaker(chat)
        Logger.success("ChatTTS 模型就绪")
        return True
    except Exception as e:
        Logger.warn(f"ChatTTS 初始化失败: {e}")
        return False


def _log_chattts_device(chat):
    """打印 ChatTTS 实际运行设备，便于确认是否在 GPU。"""
    try:
        dev = getattr(chat, "device", None)
        Logger.info(f"ChatTTS 运行设备: {dev}")
        if dev is not None and "cuda" not in str(dev):
            Logger.warn("ChatTTS 未运行在 GPU 上，合成会较慢。检查 torch 是否为 CUDA 版及显存占用。")
    except Exception:
        pass


def _load_or_create_speaker(chat) -> str:
    """加载持久化的说话人向量；不存在则随机采样一次并写盘，保证跨运行音色稳定。"""
    try:
        if os.path.exists(SPEAKER_FILE):
            with open(SPEAKER_FILE, "r", encoding="utf-8") as f:
                spk = f.read().strip()
            if spk:
                return spk
    except Exception:
        pass

    spk = chat.sample_random_speaker()
    try:
        os.makedirs(os.path.dirname(SPEAKER_FILE), exist_ok=True)
        with open(SPEAKER_FILE, "w", encoding="utf-8") as f:
            f.write(spk)
        pass
    except Exception:
        pass
    return spk


def _normalize_audio(path: str, target_dbfs: float = TARGET_DBFS):
    """将音频响度归一化到统一 dBFS，并做峰值保护避免削顶。原地覆盖写回。"""
    try:
        audio = AudioSegment.from_file(path)
        if audio.dBFS == float("-inf"):
            return  # 纯静音，跳过
        gain = target_dbfs - audio.dBFS
        adjusted = audio.apply_gain(gain)
        # 峰值保护：留 1dB 余量，防止增益后削顶
        if adjusted.max_dBFS > -1.0:
            adjusted = adjusted.apply_gain(-1.0 - adjusted.max_dBFS)
        adjusted.export(path, format="wav")
    except Exception as e:
        Logger.warn(f"音量归一化失败 {os.path.basename(path)}: {e}")


def _clean_text_for_speech(text: str) -> str:
    """把解说文本里 ChatTTS 念不出的棋盘记号转成中文/剔除，仅用于喂 TTS。

    不修改 seg.text（字幕仍保留 h7/g5 等坐标，屏幕上更精确、与棋盘高亮一致）。
    根因：分步解说 voiceover 满是 h7/f4/g1=Q 这类坐标，ChatTTS 词表里没有
    a-h/0-9 这些 token，行为未定义——会跳读、发糊，甚至即兴生成填充音
    （女声里夹进来的男声「嗯」就来自这里）。日志中反复出现的
    `found invalid characters: {'7'}` 即此问题。

    处理顺序（先长后短，避免误伤）：
      升变  e8=Q / g1=Q+  → 「升变」
      坐标  h7 / a1       → 「该格」（保留语义又可发音）
      纵线  h线 / a-h线   → 「这一线」
      残余 ASCII 字母数字、算式符号 → 删除
    """
    t = text
    # 升变（带可选将军/将杀号）：字母+数字 = 棋子字母
    t = re.sub(r"[a-h][1-8]=[QRBN][+#]?", "升变", t)
    # 纵线表述：h线 / a线
    t = re.sub(r"[a-h]\s*线", "这一线", t)
    # 单独坐标 a1-h8（前后非字母，避免切到中文里夹的拼音）
    t = re.sub(r"(?<![A-Za-z])[a-h][1-8](?![0-9])", "该格", t)
    # 残余棋子字母+将军号、孤立 ASCII 字母/数字、算式符号
    t = re.sub(r"[A-Za-z0-9=+#×*/\\_^<>\[\]{}]", "", t)
    # 替换后产生的「该格格」「该格与该格」等重复收敛为自然中文
    t = t.replace("该格格", "该格")
    t = re.sub(r"该格(与|和|、)该格", r"两个关键格", t)
    t = re.sub(r"该格(该格)+", "这些格", t)
    t = re.sub(r"[，,]{2,}", "，", t)
    t = re.sub(r"。{2,}", "。", t)
    t = re.sub(r"[，、：]+。", "。", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def _preprocess_text_for_chattts(text: str, pacing: str) -> str:
    """根据 pacing 为 ChatTTS 添加语速与韵律标记。

    只用 [speed_N] 控制语速 + 句中 [uv_break] 控制停顿。
    不再使用 [oral_N]：它会让模型注入口语填充词（嗯/啊/那个），
    且填充音音色常与主音色不同，听上去像另一个人在旁边"嗯"。
    """
    import re

    speed_map = {
        "slow": "[speed_3]",
        "normal": "[speed_5]",
        "fast": "[speed_6]",
        "pause_before": "[speed_4]",
        "pause_after": "[speed_4]",
    }
    speed_tag = speed_map.get(pacing, "[speed_5]")

    # 句中韵律：句末标点必停；逗号/顿号/冒号仅在距上次停顿已积累足够内容时才停，避免碎读
    body = text.strip()
    out = []
    since_break = 0
    for ch in body:
        out.append(ch)
        since_break += 1
        if ch in "。！？；":
            out.append("[uv_break]")
            since_break = 0
        elif ch in "，、：" and since_break >= 8:
            out.append("[uv_break]")
            since_break = 0
    body = "".join(out)
    # 收敛重复/收尾多余的 break
    body = re.sub(r"(\[uv_break\])+", "[uv_break]", body)
    body = re.sub(r"\[uv_break\]\s*$", "", body)

    return f"{speed_tag}{body}"


def _synthesize_chattts(segments: List[Segment], speed: float = 1.0) -> bool:
    """用 ChatTTS 逐段合成，成功返回 True"""
    global _chattts, _chattts_spk_emb
    if _chattts is None:
        return False

    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        Logger.warn("soundfile 未安装，ChatTTS 不可用")
        return False

    os.makedirs(AUDIO_DIR, exist_ok=True)
    chat = _chattts

    # 按 pacing 分组：相同 emotion 的段落可使用同一 speaker，但可微调参数
    batch_texts = []
    batch_segments = []
    for seg in segments:
        if seg.text.strip():
            batch_texts.append(seg.text.strip())
            batch_segments.append(seg)

    if not batch_texts:
        return True

    Logger.info(f"语音合成中 ({len(batch_texts)} 段)...")
    t_start = time.time()

    success_count = 0
    for i, (text, seg) in enumerate(zip(batch_texts, batch_segments)):
        path = os.path.abspath(os.path.join(AUDIO_DIR, f"seg_{seg.move_idx:03d}.wav"))
        seg.audio_path = path

        pacing_params = {
            "slow":     {"temperature": 0.1, "top_P": 0.5, "top_K": 15},
            "normal":   {"temperature": 0.2, "top_P": 0.6, "top_K": 18},
            "fast":     {"temperature": 0.3, "top_P": 0.7, "top_K": 20},
            "pause_before": {"temperature": 0.1, "top_P": 0.5, "top_K": 15},
            "pause_after":  {"temperature": 0.15, "top_P": 0.5, "top_K": 15},
        }
        pp = pacing_params.get(seg.pacing, pacing_params["normal"])
        
        speech_text = _clean_text_for_speech(text)
        processed_text = _preprocess_text_for_chattts(speech_text, seg.pacing)

        try:
            params = chat.InferCodeParams(
                spk_emb=_chattts_spk_emb,
                temperature=pp["temperature"],
                top_P=pp["top_P"],
                top_K=pp["top_K"],
            )
            wavs = chat.infer([processed_text], params_infer_code=params, skip_refine_text=True)
            if not wavs or len(wavs) == 0 or len(wavs[0]) == 0:
                continue

            wav = np.array(wavs[0])
            sf.write(path, wav, _CHATTTS_SAMPLE_RATE)
            _normalize_audio(path)

            audio = AudioSegment.from_wav(path)
            seg.duration_s = audio.duration_seconds + 0.3
            success_count += 1

        except Exception:
            seg.audio_path = ""

    elapsed = time.time() - t_start
    Logger.success(f"语音合成完成: {success_count}/{len(batch_texts)} 段, {elapsed:.1f}s")
    return success_count > 0


def synthesize(segments: List[Segment], voice_prompt: str = None,
               emotion: str = "default", speed: float = 1.0) -> List[Segment]:
    """
    合成语音，音频路径和时长回填到 Segment。
    优先级: ChatTTS > pyttsx3
    """
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # 确保空段有时间戳
    time_cursor = 0.0
    for seg in segments:
        if not seg.text.strip():
            seg.audio_path = ""
            seg.duration_s = 1.0
            seg.start_time = time_cursor
            time_cursor += 1.0

    # ---- 优先: ChatTTS ----
    if _init_chattts():
        chattts_ok = _synthesize_chattts(segments, speed)
        # 回填时间戳并检查是否有失败的段
        all_good = chattts_ok
        if chattts_ok:
            time_cursor = 0.0
            fallback_needed = []
            for seg in segments:
                if not seg.text.strip():
                    seg.start_time = time_cursor
                    time_cursor += seg.duration_s
                    continue
                if seg.audio_path and os.path.exists(seg.audio_path):
                    seg.start_time = time_cursor
                    time_cursor += seg.duration_s
                else:
                    fallback_needed.append(seg)

            if not fallback_needed:
                return segments
            fb_engine = _init_fallback_engine()
            if fb_engine:
                for seg in fallback_needed:
                    seg.audio_path = os.path.abspath(
                        os.path.join(AUDIO_DIR, f"seg_{seg.move_idx:03d}.wav"))
                    seg.duration_s = _fallback_pyttsx3(seg.text, seg.audio_path, fb_engine)
                try:
                    fb_engine.stop()
                except Exception:
                    pass
            # 重新计算时间戳
            time_cursor = 0.0
            for seg in segments:
                seg.start_time = time_cursor
                time_cursor += seg.duration_s
            return segments

    fallback_engine = _init_fallback_engine()

    for seg in segments:
        if not seg.text.strip():
            continue
        path = os.path.abspath(os.path.join(AUDIO_DIR, f"seg_{seg.move_idx:03d}.wav"))
        seg.audio_path = path
        seg.duration_s = _fallback_pyttsx3(seg.text, seg.audio_path, fallback_engine)

    if fallback_engine:
        try:
            fallback_engine.stop()
        except Exception:
            pass

    time_cursor = 0.0
    for seg in segments:
        seg.start_time = time_cursor
        time_cursor += seg.duration_s

    return segments


def _init_fallback_engine():
    try:
        import pyttsx3
        engine = pyttsx3.init("sapi5")
        return engine
    except Exception as e:
        Logger.warn(f"pyttsx3 初始化失败: {e}")
        return None


def _fallback_pyttsx3(text: str, output_path: str, engine) -> float:
    """pyttsx3 回退，返回时长"""
    if engine is None:
        return max(1.0, len(text) * 0.1)
    try:
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        _normalize_audio(output_path)
        audio = AudioSegment.from_wav(output_path)
        return audio.duration_seconds + 0.3
    except Exception as e:
        Logger.error(f"pyttsx3 合成失败: {e}")
        return max(1.0, len(text) * 0.1)
