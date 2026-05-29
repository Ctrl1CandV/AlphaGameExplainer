import os
import time
from typing import List, Optional
from pydub import AudioSegment
from src.common import Segment, Logger

AUDIO_DIR = os.path.join("output", "audio")
DEFAULT_VOICE = os.path.join("assets", "voices", "ref_male_pro.wav")

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


def _init_chattts():
    """加载 ChatTTS 模型（首次调用时加载，后续复用）"""
    global _chattts, _chattts_spk_emb
    if _chattts is not None:
        return True

    try:
        from ChatTTS import Chat
        Logger.info("加载 ChatTTS 模型...")
        chat = Chat()
        ok = chat.load(compile=False, source="huggingface")
        if not ok:
            Logger.warn("ChatTTS 模型加载失败")
            return False
        _chattts = chat
        _chattts_spk_emb = chat.sample_random_speaker()
        Logger.success("ChatTTS 模型就绪")
        return True
    except Exception as e:
        Logger.warn(f"ChatTTS 初始化失败: {e}")
        return False


def _preprocess_text_for_chattts(text: str, pacing: str) -> str:
    """根据 pacing 和内容为 ChatTTS 添加情感标记"""
    processed = text
    
    # 根据 pacing 添加语速标记
    speed_map = {
        "slow": "[speed_3]",
        "normal": "[speed_5]",
        "fast": "[speed_7]",
        "pause_before": "[speed_4][break_2]",
        "pause_after": "[speed_4]",
    }
    speed_tag = speed_map.get(pacing, "[speed_5]")
    
    # 检测关键内容并添加情感标记
    check_keywords = ["将军", "叫杀", "将杀", "绝杀", "杀王"]
    capture_keywords = ["吃掉", "吃子", "兑子", "捕获"]
    important_keywords = ["关键", "重要", "转折", "突破", "妙手"]
    
    has_check = any(kw in text for kw in check_keywords)
    has_capture = any(kw in text for kw in capture_keywords)
    has_important = any(kw in text for kw in important_keywords)
    
    # 根据内容调整语调
    if has_check or has_capture:
        # 将军/吃子时稍微激动
        processed = f"[oral_3]{speed_tag}[break_1]{processed}"
    elif has_important:
        # 关键时刻强调
        processed = f"[oral_2]{speed_tag}[break_1]{processed}"
    else:
        # 正常讲解
        processed = f"[oral_1]{speed_tag}{processed}"
    
    return processed


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

    Logger.info(f"ChatTTS 合成 ({len(batch_texts)} 段)...")
    t_start = time.time()

    success_count = 0
    for i, (text, seg) in enumerate(zip(batch_texts, batch_segments)):
        path = os.path.abspath(os.path.join(AUDIO_DIR, f"seg_{seg.move_idx:03d}.wav"))
        seg.audio_path = path

        # pacing → 参数映射（更细致的情感控制）
        pacing_params = {
            "slow":     {"temperature": 0.2, "top_P": 0.5, "top_K": 15},
            "normal":   {"temperature": 0.3, "top_P": 0.7, "top_K": 20},
            "fast":     {"temperature": 0.4, "top_P": 0.8, "top_K": 30},
            "pause_before": {"temperature": 0.2, "top_P": 0.5, "top_K": 15},
            "pause_after":  {"temperature": 0.25, "top_P": 0.6, "top_K": 18},
        }
        pp = pacing_params.get(seg.pacing, pacing_params["normal"])
        
        # 预处理文本，添加情感标记
        processed_text = _preprocess_text_for_chattts(text, seg.pacing)

        try:
            params = chat.InferCodeParams(
                spk_emb=_chattts_spk_emb,
                temperature=pp["temperature"],
                top_P=pp["top_P"],
                top_K=pp["top_K"],
            )
            wavs = chat.infer([processed_text], params_infer_code=params, skip_refine_text=True)
            if not wavs or len(wavs) == 0 or len(wavs[0]) == 0:
                Logger.warn(f"  ChatTTS seg_{seg.move_idx:03d}: 空输出")
                continue

            wav = np.array(wavs[0])
            sf.write(path, wav, _CHATTTS_SAMPLE_RATE)

            # 回填时长
            audio = AudioSegment.from_wav(path)
            seg.duration_s = audio.duration_seconds + 0.3
            success_count += 1
            if (i + 1) <= 2 or (i + 1) % 5 == 0:
                Logger.info(f"  ChatTTS [{i+1}/{len(batch_texts)}] seg_{seg.move_idx:03d} ({seg.duration_s:.1f}s)")

        except Exception as e:
            Logger.warn(f"  ChatTTS seg_{seg.move_idx:03d} 失败: {e}")
            # 标记此段需要回退
            seg.audio_path = ""

    elapsed = time.time() - t_start
    Logger.success(f"ChatTTS 完成: {success_count}/{len(batch_texts)} 段, {elapsed:.1f}s")
    return success_count > 0


def synthesize(segments: List[Segment], voice_prompt: str = None,
               emotion: str = "default", speed: float = 1.0) -> List[Segment]:
    """
    合成语音，音频路径和时长回填到 Segment。
    优先级: ChatTTS > IndexTTS2 > pyttsx3
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
                Logger.success(f"语音合成完成: {len(segments)} 段 (ChatTTS), 总时长 {time_cursor:.1f}s")
                return segments

            # 有部分段失败，回退处理失败的段
            Logger.info(f"ChatTTS 部分失败，回退 {len(fallback_needed)} 段到 pyttsx3...")
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
            Logger.success(f"语音合成完成: {len(segments)} 段 (ChatTTS+pyttsx3), 总时长 {time_cursor:.1f}s")
            return segments

    # ---- 现有逻辑: IndexTTS2 / pyttsx3（保持不变）----
    Logger.info("ChatTTS 不可用，回退到原有 TTS 方案...")

    voice = voice_prompt or os.path.abspath(DEFAULT_VOICE)
    use_indextts = True

    if not os.path.exists(voice):
        Logger.warn(f"参考语音文件不存在: {voice}")
        use_indextts = False
    elif os.path.getsize(voice) < 1024:
        Logger.warn(f"参考语音文件无效 (大小仅 {os.path.getsize(voice)} 字节): {voice}")
        use_indextts = False
    if not use_indextts:
        Logger.warn("请将 5-15 秒的 WAV 参考音频放入 assets/voices/ 目录")
        Logger.warn("当前将使用 pyttsx3 回退引擎生成语音")

    batch_items = []
    for seg in segments:
        if not seg.text.strip():
            continue
        path = os.path.abspath(os.path.join(AUDIO_DIR, f"seg_{seg.move_idx:03d}.wav"))
        seg.audio_path = path
        emo, spd = PACING_MAP.get(seg.pacing, ("default", 1.0))
        batch_items.append({
            "text": seg.text.strip(),
            "output_path": path,
            "emotion": emo,
            "speed": spd * speed,
        })

    if not batch_items:
        return segments

    Logger.info(f"合成语音 ({len(batch_items)} 段)...")

    results = _try_indextts(batch_items, voice) if use_indextts else None
    fallback_engine = _init_fallback_engine() if not results else None

    time_cursor = 0.0
    for seg in segments:
        if not seg.text.strip():
            seg.duration_s = 1.0
            seg.start_time = time_cursor
            time_cursor += seg.duration_s
            continue

        dur = _get_result_duration(seg.audio_path, results) if results else None
        if dur is not None:
            seg.duration_s = dur
        else:
            seg.duration_s = _fallback_pyttsx3(seg.text, seg.audio_path, fallback_engine)

        seg.start_time = time_cursor
        time_cursor += seg.duration_s

    if fallback_engine:
        try:
            fallback_engine.stop()
        except Exception:
            pass

    Logger.success(f"语音合成完成: {len(segments)} 段, 总时长 {time_cursor:.1f}s")
    return segments


def _init_fallback_engine():
    try:
        import pyttsx3
        engine = pyttsx3.init("sapi5")
        return engine
    except Exception as e:
        Logger.warn(f"pyttsx3 初始化失败: {e}")
        return None


def _try_indextts(batch_items: list, voice: str) -> dict:
    """尝试 IndexTTS2 批量合成，返回 {output_path: duration} 或 None"""
    try:
        from src.tts_bridge import synthesize_batch
        raw_results = synthesize_batch(
            [{"text": b["text"], "output_path": b["output_path"],
              "emotion": b["emotion"], "speed": b["speed"]}
             for b in batch_items],
            voice_prompt=voice,
            output_dir=AUDIO_DIR,
        )
        if raw_results:
            return {r["output_path"]: r.get("duration", 2.0)
                    for r in raw_results if r.get("status") == "ok"}
    except Exception as e:
        Logger.warn(f"IndexTTS2 batch error: {e}")
    return None


def _get_result_duration(path: str, results: dict) -> float:
    """从批量结果中获取音频时长，并验证文件存在"""
    dur = results.get(path)
    if dur and os.path.exists(path):
        try:
            audio = AudioSegment.from_wav(path)
            return audio.duration_seconds + 0.3
        except Exception:
            return dur + 0.3
    return None


def _fallback_pyttsx3(text: str, output_path: str, engine) -> float:
    """pyttsx3 回退，返回时长"""
    if engine is None:
        return max(1.0, len(text) * 0.1)
    try:
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        audio = AudioSegment.from_wav(output_path)
        return audio.duration_seconds + 0.3
    except Exception as e:
        Logger.error(f"pyttsx3 合成失败: {e}")
        return max(1.0, len(text) * 0.1)
