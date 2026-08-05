from src.common import Segment, Logger, GeneratedCommentary, CompressedStep
from typing import List, Optional, Tuple
from pydub import AudioSegment
import chess
import torch
import time
import os
import re
import gc

AUDIO_DIR = os.path.join("output", "audio")
# 持久化的ChatTTS说话人向量，保证跨运行音色一致
SPEAKER_FILE = os.path.join("assets", "voices", "chattts_speaker.txt")
# 全片统一目标响度，消除ChatTTS逐段幅度不一致导致的忽轻忽响
TARGET_DBFS = -20.0

# ChatTTS 模型缓存
_chattts: Optional[object] = None
_chattts_spk_emb: Optional[str] = None
_CHATTTS_SAMPLE_RATE = 24000

def _free_gpu_before_tts():
    """
    ChatTTS按"加载时空闲显存"选设备(阈值约2GB)，加载前主动清一次torch缓存
    避免上游LLM残留显存导致ChatTTS静默回退CPU
    """
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        free_mb = free / (1024 * 1024)
        if free_mb < 2200:
            Logger.warn("GPU显存不足，ChatTTS回退CPU")


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
    # 全角分号/冒号归一为逗号（08.04 修）。ChatTTS 词表只含逗号句号等基本
    # 标点，不含「；」「：」，日志会报 `found invalid characters: {'；'}` /
    # `{'：'}`，并在该处行为未定义（跳读或即兴填充音）。决策管线的对比段
    # 天然爱用这两个标点做并列与提示（「甲…；乙…」「代价是：…」），命中率
    # 比两条老管线高得多，故必须在此归一。语义等价（都是句中停顿），
    # 不改 seg.text——字幕保留原标点，屏幕上更清晰。
    t = t.replace("；", "，").replace(";", "，")
    t = t.replace("：", "，").replace(":", "，")
    # 破折号/问号/全角括号（08.04 补）。同属「ChatTTS 词表没有、日志报
    # `found invalid characters` 且行为未定义」这一类，实测本轮批量日志里
    # 出现 `{'—'}` 3 次、`{'？'}` 1 次、`{'（', '）'}` 1 次。
    #   —  破折号是解说词最爱的转折符（「这条路稳——但空间受限」），
    #      转逗号即可，语义是同一个停顿；
    #   ？ 决策管线的开场爱用设问（「该先动哪一翼？」），转句号：口播的
    #      疑问语气靠语调而非字符，ChatTTS 读不出问号但能读句末停顿；
    #   （）括注内容本身要念（不能删），只去掉括号本体，前后补逗号让停顿自然。
    t = re.sub(r"[—–―]+", "，", t)
    t = t.replace("？", "。").replace("?", "。")
    t = re.sub(r"[（(]\s*", "，", t)
    t = re.sub(r"\s*[）)]", "，", t)
    t = re.sub(r"[，,]{2,}", "，", t)
    t = re.sub(r"。{2,}", "。", t)
    # 收敛替换后产生的相邻标点：「，。」→「。」、句首逗号、句末逗号。
    # 括号转逗号后极易在句末留下「…，。」或段尾孤立逗号，必须收干净——
    # 否则 TTS 会在结尾多停一拍，听上去像没说完。
    t = re.sub(r"[，、：]+。", "。", t)
    t = re.sub(r"^[，、]+", "", t)
    t = re.sub(r"[，、]+$", "。", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def _preprocess_text_for_chattts(text: str, pacing: str, speed_override: int = 0) -> str:
    """根据 pacing 为 ChatTTS 添加语速与韵律标记。

    只用 [speed_N] 控制语速 + 句中 [uv_break] 控制停顿。
    不再使用 [oral_N]：它会让模型注入口语填充词（嗯/啊/那个），
    且填充音音色常与主音色不同，听上去像另一个人在旁边“嗯”。

    speed_override: PLAN-006 阶段 C，从二维参数表查出的 speed 值（0 表示用旧逻辑）。
    """
    import re

    if speed_override > 0:
        speed_tag = f"[speed_{speed_override}]"
    else:
        speed_map = {
            "slow": "[speed_3]",
            "normal": "[speed_5]",
            "fast": "[speed_6]",
            "pause_before": "[speed_4]",
            "pause_after": "[speed_4]",
        }
        speed_tag = speed_map.get(pacing, "[speed_5]")

    # 句中韵律：句末标点仅在本句已积累足够内容时才插停顿；逗号/顿号/冒号同理。
    # 短句（如开场白「这是一个X残局。」）不再每句都插 [uv_break]——过密的停顿
    # 叠加低温采样会让 ChatTTS 在停顿处坍缩出「嗯/呃」衬词（开头几段尤甚）。
    # 短句靠标点本身的自然停顿即可，不强行加 break。
    body = text.strip()
    out = []
    since_break = 0
    for ch in body:
        out.append(ch)
        since_break += 1
        if ch in "。！？；" and since_break >= 6:
            out.append("[uv_break]")
            since_break = 0
        elif ch in "，、：" and since_break >= 10:
            out.append("[uv_break]")
            since_break = 0
    body = "".join(out)
    # 收敛重复/收尾多余的 break
    body = re.sub(r"(\[uv_break\])+", "[uv_break]", body)
    body = re.sub(r"\[uv_break\]\s*$", "", body)

    return f"{speed_tag}{body}"


# 句末标点 → 句后静音时长（毫秒）。营造自然换气与语气停顿。
_PAUSE_AFTER_PUNCT = {
    "。": 300, "！": 400, "？": 350, "；": 250,
}
_DEFAULT_PAUSE_MS = 250
# 短句合并阈值：少于该字符数的句子并入相邻句，避免短句+低温触发 ChatTTS 衬词
_MIN_SENTENCE_CHARS = 10
# 句音频头尾淡入淡出毫秒，避免硬接静音产生的爆音/断线感
_SENTENCE_FADE_MS = 60

# PLAN-006 阶段 C：pacing × emphasis 二维 TTS 参数表
# 阶段 0 冒烟结论：temp 0.5 长句不可用，上限 0.4；温度差异感知微弱，主要靠 speed + silence 分化。
# 阶段 E 反馈修正：routine speed 6-7 过快导致发音模糊，降速保清晰；pivotal 放慢+加长停顿拉大情绪对比。
_EMPHASIS_TTS_PARAMS = {
    # (pacing, emphasis): {temp, top_P, top_K, speed, post_ms, pre_s}
    ("slow", "pivotal"):     {"temp": 0.40, "top_P": 0.7, "top_K": 20, "speed": 3, "post_ms": 550, "pre_s": 0.4},
    ("slow", "important"):   {"temp": 0.30, "top_P": 0.7, "top_K": 20, "speed": 5, "post_ms": 400, "pre_s": 0.2},
    ("slow", "routine"):     {"temp": 0.25, "top_P": 0.6, "top_K": 18, "speed": 5, "post_ms": 300, "pre_s": 0.0},
    ("normal", "pivotal"):   {"temp": 0.40, "top_P": 0.7, "top_K": 20, "speed": 4, "post_ms": 500, "pre_s": 0.4},
    # A3（PLAN-006 REVIEW-002 修）：important 与 routine 此前 speed/pre_s 全同，只差
    # temp+post_ms，短段（单句、无句间停顿）几乎无差异，三档塌成两档。routine 提速
    # 会发音模糊（阶段 E 一轮已验证，禁止），故靠给 important 段前微停顿 pre_s 0.15
    # + 拉大句间停顿（420 vs 190）+ 温度韵律区分，让 important 更从容、routine 更紧凑。
    # important pre_s 依赖 A1 字幕偏移补偿（seg.pre_silence_s 写回），二者必须同批落地。
    ("normal", "important"): {"temp": 0.32, "top_P": 0.6, "top_K": 18, "speed": 5, "post_ms": 420, "pre_s": 0.15},
    ("normal", "routine"):   {"temp": 0.20, "top_P": 0.5, "top_K": 15, "speed": 5, "post_ms": 190, "pre_s": 0.0},
    ("fast", "pivotal"):     {"temp": 0.35, "top_P": 0.7, "top_K": 20, "speed": 4, "post_ms": 450, "pre_s": 0.3},
    ("fast", "important"):   {"temp": 0.25, "top_P": 0.6, "top_K": 18, "speed": 6, "post_ms": 300, "pre_s": 0.0},
    ("fast", "routine"):     {"temp": 0.20, "top_P": 0.5, "top_K": 15, "speed": 6, "post_ms": 220, "pre_s": 0.0},
}
# pause_before/pause_after 归入 slow/normal 对应行
_PACING_ALIAS = {"pause_before": "slow", "pause_after": "normal"}


def _lookup_tts_params(pacing: str, emphasis: str) -> dict:
    """查二维参数表，未定义组合回退 emphasis='important' 行。"""
    p = _PACING_ALIAS.get(pacing, pacing)
    if p not in ("slow", "normal", "fast"):
        p = "normal"
    key = (p, emphasis)
    if key not in _EMPHASIS_TTS_PARAMS:
        key = (p, "important")
    return _EMPHASIS_TTS_PARAMS[key]


def _split_sentences(text: str) -> List[tuple]:
    """把整段文本按句末标点切分为 [(句子文本, 句后静音毫秒), ...]。

    过短的句子并入相邻句，避免短句独立合成触发 ChatTTS 衬词。
    句子文本保留原标点，句后静音时长由句末标点决定。
    """
    if not text:
        return []
    # 按句末标点切分，保留标点（最后一段可能无句末标点）
    parts = re.findall(r"[^。！？；]*[。！？；]|[^。！？；]+", text)
    sentences = [p.strip() for p in parts if p.strip()]
    if not sentences:
        return []

    # 合并过短句：累积到达阈值再独立成句；末尾残留并入上一句
    merged: List[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = buffer + sentence
        if len(candidate) < _MIN_SENTENCE_CHARS:
            buffer = candidate
        else:
            merged.append(candidate)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] = merged[-1] + buffer
        else:
            merged.append(buffer)

    # 为每句计算句后静音
    result = []
    for sentence in merged:
        last_char = sentence[-1] if sentence else ""
        pause_ms = _PAUSE_AFTER_PUNCT.get(last_char, _DEFAULT_PAUSE_MS)
        result.append((sentence, pause_ms))
    return result



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
    prev_emphasis = ""  # PLAN-006：跟踪前一段 emphasis，连续 pivotal 时 pre_silence 减半
    for i, (text, seg) in enumerate(zip(batch_texts, batch_segments)):
        path = os.path.abspath(os.path.join(AUDIO_DIR, f"seg_{seg.move_idx:03d}.wav"))
        seg.audio_path = path

        # PLAN-006 阶段 C：从 pacing×emphasis 二维表查参数
        pp = _lookup_tts_params(seg.pacing, seg.emphasis_level)

        speech_text = _clean_text_for_speech(text)

        # 分句合成：把整段按句末标点切成单句逐句合成，句间插入真实静音，
        # 让解说有自然的换气与停顿，而非整段匀速念稿。
        sentences = _split_sentences(speech_text)
        if not sentences:
            seg.audio_path = ""
            continue

        try:
            combined = AudioSegment.empty()

            # PLAN-006：段前静音（pivotal 停顿感，连续 pivotal 减半）
            pre_s = pp["pre_s"]
            if pre_s > 0 and prev_emphasis == "pivotal" and seg.emphasis_level == "pivotal":
                pre_s = pre_s / 2
            if pre_s > 0:
                combined += AudioSegment.silent(duration=int(pre_s * 1000), frame_rate=_CHATTTS_SAMPLE_RATE)

            sent_ok = False
            for sent_idx, (sentence, pause_ms) in enumerate(sentences):
                processed_text = _preprocess_text_for_chattts(sentence, seg.pacing, speed_override=pp["speed"])
                # 段首句用更低温度稳定音色，后续句略升温减少累积漂移
                temperature = pp["temp"] if sent_idx == 0 else min(pp["temp"] + 0.1, 0.4)
                params = chat.InferCodeParams(
                    spk_emb=_chattts_spk_emb,
                    temperature=temperature,
                    top_P=pp["top_P"],
                    top_K=pp["top_K"],
                )
                wavs = chat.infer([processed_text], params_infer_code=params, skip_refine_text=True)
                if not wavs or len(wavs) == 0 or len(wavs[0]) == 0:
                    continue

                # 单句先落临时文件做响度归一化，再读回拼接
                sent_path = os.path.abspath(
                    os.path.join(AUDIO_DIR, f"_sent_{seg.move_idx:03d}_{sent_idx:02d}.wav"))
                sf.write(sent_path, np.array(wavs[0]), _CHATTTS_SAMPLE_RATE)
                _normalize_audio(sent_path)

                sent_audio = AudioSegment.from_wav(sent_path)
                # 头尾淡入淡出，避免句与静音硬接产生爆音/断线感
                sent_audio = sent_audio.fade_in(_SENTENCE_FADE_MS).fade_out(_SENTENCE_FADE_MS)
                combined += sent_audio
                # 句后插入真实静音（最后一句不加，段间静音由 composer 处理）
                # PLAN-006：句后静音用查表值（体现 emphasis 分化），不再纯由标点决定
                if sent_idx < len(sentences) - 1:
                    combined += AudioSegment.silent(duration=pp["post_ms"])
                sent_ok = True

                try:
                    os.remove(sent_path)
                except Exception:
                    pass

            if not sent_ok or len(combined) == 0:
                seg.audio_path = ""
                seg.speech_duration_s = 0.0  # 防御：段复用时清残留，避免误导字幕
                seg.duration_s = 0.0
                continue

            combined.export(path, format="wav")
            # speech_duration_s = 真实语音截止（不含前置静音和 0.3 尾静音），字幕据此分配 cue，
            # 避免末条字幕落入尾部静音；duration_s 仍含尾静音供画面/音频对齐用。
            seg.speech_duration_s = combined.duration_seconds - pre_s
            seg.duration_s = combined.duration_seconds + 0.3
            # A1（PLAN-006 REVIEW-002 修）：写回段前静音长度（此前是死字段）。
            # 语音实际从 start_time + pre_s 才开始，字幕 cue 起点须据此后移，
            # 否则 pivotal/important 段（pre_s>0）字幕整体早出 0.15~0.4s。
            seg.pre_silence_s = pre_s
            success_count += 1
            prev_emphasis = seg.emphasis_level

        except Exception:
            seg.audio_path = ""


    elapsed = time.time() - t_start
    Logger.success(f"语音合成完成: {success_count}/{len(batch_texts)} 段, {elapsed:.1f}s")
    return success_count > 0


# ============================================================
# Segment 构造（commentary + moves → List[Segment]）
# 原位置：pipeline.py。Segment 是 TTS 的输入类型，转换逻辑应与 TTS 同域。
# pipeline 只负责调度，不应理解 voice_map / node_moves 的内部细节。
# ============================================================

def build_node_segments(
        commentary: GeneratedCommentary, moves: List[chess.Move],
        compressed: Optional[List[CompressedStep]] = None
    ) -> List[Segment]:
    """
    按压缩节点分段：一个节点 = 一段解说 + 该节点的全部子步走法。
    这是解决音画粒度错位的核心改动：旧实现把节点整段解说塞给第一个子步，
    其余子步置空文本，导致首步静止十几秒念完、后续子步无声飞闪、解说视角与画面错位。

    现在改为节点级分段：每段携带本节点的全部 moves，由 board_renderer 在该段音频
    时长内顺序播放这些子步并均摊定格，解说推进时棋子也在持续走。
    一段一段音频，不再有空文本段，从根上消除空段累积漂移。
    无压缩信息时退化为逐步分段。
    """
    # 节点 id → (voiceover, pacing) 查找表
    voice_map: dict = {}
    if commentary.segments:
        for seg in commentary.segments:
            voice_map[seg.id] = (seg.voiceover, seg.pacing)

    result: List[Segment] = []
    if compressed:
        move_cursor = 0
        for cs in compressed:
            n = len(cs.sans)
            node_moves = moves[move_cursor:move_cursor + n]
            move_cursor += n
            if not node_moves:
                continue
            vo, pac = voice_map.get(cs.idx, (None, "normal"))
            text = vo if vo else ""
            result.append(Segment(
                move_idx=cs.idx,
                text=text,
                pacing=pac or "normal",
                moves=list(node_moves),
                phase=getattr(cs, "phase", ""),
            ))
        # 解法被截断、moves 比 compressed 覆盖的还多时，剩余走法兜底成一段静默节点
        if move_cursor < len(moves):
            result.append(Segment(
                move_idx=(compressed[-1].idx if compressed else 0) + 1,
                text="",
                pacing="normal",
                moves=list(moves[move_cursor:]),
            ))
        return result

    # 无压缩信息：逐步分段
    for i, move in enumerate(moves):
        result.append(Segment(move_idx=i + 1, text="", pacing="normal", moves=[move]))
    return result


def build_puzzle_segments(
        commentary: GeneratedCommentary, moves: List[chess.Move], nodes: list
    ) -> List[Segment]:
    """按节点构造 Segment 列表（puzzle 版：无开场白/总结段，phase 为空）。"""
    voice_map: dict = {}
    if commentary.segments:
        for seg in commentary.segments:
            voice_map[seg.id] = (seg.voiceover, seg.pacing)

    result: List[Segment] = []
    # PLAN-006 阶段 D（REVIEW-002 V2）：与 endgame_pipeline 对齐，0.60/0.30 放大节奏差异
    _SLIDE_BY_EMPHASIS = {"pivotal": 0.60, "important": 0.45, "routine": 0.30}
    for node in nodes:
        nid = node["id"]
        vo, pac = voice_map.get(nid, (None, "normal"))
        text = vo if vo else ""
        # 每节点一个 move（puzzle 不压缩）
        node_moves: List[chess.Move] = []
        if nid <= len(moves):
            node_moves = [moves[nid - 1]]
        emph = node.get("emphasis_level", "important")
        result.append(Segment(
            move_idx=nid,
            text=text,
            pacing=pac or "normal",
            moves=node_moves,
            phase="",
            emphasis_level=emph,
            slide_sec=_SLIDE_BY_EMPHASIS.get(emph, 0.45),
        ))
    return result


def synthesize(
        segments: List[Segment], voice_prompt: str = None,
        emotion: str = "default", speed: float = 1.0
    ) -> List[Segment]:
    """ 合成语音，音频路径和时长回填到Segment """
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # 确保空段有时间戳
    time_cursor = 0.0
    for seg in segments:
        if not seg.text.strip():
            seg.audio_path = ""
            seg.duration_s = 1.0
            seg.speech_duration_s = 0.0  # 空段无语音，字幕跳过
            seg.start_time = time_cursor
            time_cursor += 1.0

    # ---- 优先: ChatTTS ----
    if _init_chattts():
        chattts_ok = _synthesize_chattts(segments, speed)
        # 回填时间戳并检查是否有失败的段
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
                    seg.speech_duration_s, seg.duration_s = _fallback_pyttsx3(
                        seg.text, seg.audio_path, fb_engine)
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
        seg.speech_duration_s, seg.duration_s = _fallback_pyttsx3(
            seg.text, path, fallback_engine)

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


def _fallback_pyttsx3(text: str, output_path: str, engine) -> Tuple[float, float]:
    """pyttsx3 回退，返回 (speech_duration_s, duration_s)。

    speech_duration_s 为真实语音时长（不含尾静音），duration_s 含 0.3 尾静音，
    与 ChatTTS 路径保持同一契约，供字幕按真实语音分配 cue。
    """
    if engine is None:
        d = max(1.0, len(text) * 0.1)
        return d, d + 0.3
    try:
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        _normalize_audio(output_path)
        audio = AudioSegment.from_wav(output_path)
        speech = audio.duration_seconds
        return speech, speech + 0.3
    except Exception as e:
        Logger.error(f"pyttsx3 合成失败: {e}")
        d = max(1.0, len(text) * 0.1)
        return d, d + 0.3
