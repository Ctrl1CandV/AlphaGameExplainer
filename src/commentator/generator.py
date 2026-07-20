"""
通用解说生成框架
从generate_structured和generate_puzzle_structured中提取的
公共控制流：分块 → 重试 → JSON解析 → auto-fix → 校验 → 修复 → fallback → 后处理
双链路的差异通过CommentaryConfig回调注入，不用继承
"""
from src.common import GeneratedCommentary, StoryboardSegment, Logger, normalize_pacing
from src.commentator.json_utils import parse_storyboard_json, INVALID_JSON_SENTINEL
from src.commentator.text_filters import strip_thinking
from src.commentator.grammar import build_retry_prompt
from typing import Callable, Optional
from dataclasses import dataclass
import re

CHUNK_SIZE = 4
MAX_CHARS = 1800
MAX_RETRIES = 1
MIN_VOICEOVER_LEN = 48

@dataclass
class CommentaryConfig:
    """
    双链路的生成配置——由endgame_commentary / puzzle_commentary各自构造
    build_chunk_prompt签名: (header, chunk_nodes, chunk_idx, total_chunks, all_nodes) -> str
        残局版用all_nodes计算prev_context；Puzzle版忽略all_nodes
    build_fallback_voiceover 签名: (chunk_nodes, json_prompt) -> list[StoryboardSegment]
    post_process签名: (commentary, all_segments, nodes, storyboard, backend) -> None
    """
    build_header: Callable[[dict], str]
    build_chunk_prompt: Callable
    build_grammar: Callable[[int], str]
    validate_chunk: Callable[[dict, list], tuple]
    auto_fix_voiceover: Callable[[str, dict], str]
    repair_failed_segments: Optional[Callable]
    build_fallback_voiceover: Callable
    post_process: Callable

def dict_to_storyboard_segments(data: dict, chunk_nodes: list) -> list:
    """ 把LLM输出的JSONsegments转为StoryboardSegment列表 """
    result = []
    node_by_id = {node["id"]: node for node in chunk_nodes}
    for seg in data.get("segments", []):
        node = node_by_id.get(int(seg.get("id", 0)), {})
        result.append(StoryboardSegment(
            id=int(seg.get("id", 0)),
            sub_endgame=str(seg.get("sub_endgame", "")),
            voiceover=str(seg.get("voiceover", "")),
            pacing=normalize_pacing(str(seg.get("pacing", "normal"))),
        ))
    return result

def finalize_chunk_segments(data_or_segments, chunk_nodes: list):
    """ 统一的segment定稿：接受dict或segments list，输出StoryboardSegment list """
    data = data_or_segments if isinstance(data_or_segments, dict) else {"segments": data_or_segments}
    return dict_to_storyboard_segments(data, chunk_nodes)

def generate_commentary(storyboard: dict, backend, config: CommentaryConfig) -> GeneratedCommentary:
    """
    通用生成框架：分块 → 重试 → JSON解析 → auto-fix → 校验 → 修复 → fallback → 后处理
    双链路的控制流完全一致，差异通过config回调注入
    逐行对照原始generate_structured / generate_puzzle_structured的控制流
    """
    nodes = storyboard.get("nodes", [])
    commentary = GeneratedCommentary()

    if not nodes:
        Logger.warn("分镜数据为空，无法生成解说")
        return commentary

    node_count = len(nodes)
    total_chunks = max(1, (node_count + CHUNK_SIZE - 1) // CHUNK_SIZE)

    json_header = config.build_header(storyboard)
    all_segments = []
    commentary.chunks_total = total_chunks

    for chunk_idx in range(total_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, node_count)
        chunk_nodes = nodes[start:end]

        json_prompt = config.build_chunk_prompt(json_header, chunk_nodes, chunk_idx, total_chunks, nodes)
        chunk_grammar = config.build_grammar(len(chunk_nodes))

        success = False
        err_msg = "首次尝试失败"
        # 内容重试预算：API模式backend声明content_retry_limit（默认2，含首试共3次）
        # 本地模式无该属性时沿用模块常量MAX_RETRIES（=1，含首试共2次）
        retry_budget = getattr(backend, "content_retry_limit", MAX_RETRIES)
        for attempt in range(retry_budget + 1):
            if attempt == 0:
                prompt = json_prompt
            else:
                prompt = build_retry_prompt(json_prompt, err_msg, attempt)
                commentary.retries_total += 1

            raw_text = strip_thinking(backend.generate(prompt, grammar=chunk_grammar))
            if not raw_text:
                err_msg = "生成空结果"
                continue

            data = parse_storyboard_json(raw_text)
            if data is INVALID_JSON_SENTINEL:
                err_msg = "输出不是合法JSON"
                continue

            # 校验前预处理：对所有segment先做auto-fix清洗
            segments = data.get("segments")
            if isinstance(segments, list) and len(segments) == len(chunk_nodes):
                for si, seg in enumerate(segments):
                    seg["voiceover"] = config.auto_fix_voiceover(
                        seg.get("voiceover", ""), chunk_nodes[si])

            ok, err_msg = config.validate_chunk(data, chunk_nodes)
            if ok:
                chunk_segments = finalize_chunk_segments(data, chunk_nodes)
                all_segments.extend(chunk_segments)
                commentary.chunks_succeeded += 1
                success = True
                break

            # 残局版用_repair_failed_segments，puzzle版用内联逐段修复
            if config.repair_failed_segments is not None:
                if isinstance(segments, list) and len(segments) == len(chunk_nodes):
                    repaired = config.repair_failed_segments(backend, segments, chunk_nodes)
                    if repaired is not None:
                        repaired_ok, _ = config.validate_chunk(repaired, chunk_nodes)
                        if repaired_ok:
                            chunk_segments = finalize_chunk_segments(repaired, chunk_nodes)
                            all_segments.extend(chunk_segments)
                            commentary.chunks_succeeded += 1
                            success = True
                            break
            else:
                # Puzzle内联修复：再跑一遍auto_fix
                if isinstance(segments, list) and len(segments) == len(chunk_nodes):
                    for si, seg in enumerate(segments):
                        node = chunk_nodes[si]
                        original_vo = seg.get("voiceover", "")
                        fixed_vo = config.auto_fix_voiceover(original_vo, node)
                        if fixed_vo != original_vo:
                            seg["voiceover"] = fixed_vo
                    data["segments"] = segments
                    repaired_ok, _ = config.validate_chunk(data, chunk_nodes)
                    if repaired_ok:
                        chunk_segments = finalize_chunk_segments(data, chunk_nodes)
                        all_segments.extend(chunk_segments)
                        commentary.chunks_succeeded += 1
                        success = True
                        break

        if not success:
            # 走到这里说明无论API还是本地，内容经重试仍不合格——按用户意图舍弃，不硬给答案
            commentary.aborted = True
            commentary.aborted_chunk = chunk_idx + 1
            commentary.aborted_reason = err_msg or "内容级失败"
            Logger.warn(
                f"  块{chunk_idx + 1}内容级失败（{err_msg}），重试耗尽仍不合格，"
                f"按SPEC §8放弃本片生成"
            )
            return commentary

    commentary.segments = all_segments

    # 后处理（去重/将杀追加/双关键点/opening/summary等）
    config.post_process(commentary, all_segments, nodes, storyboard, backend)

    status = "正常" if not commentary.fallback_used else f"部分回退({commentary.chunks_succeeded}/{total_chunks})"
    Logger.success(f"解说生成完成: {len(all_segments)} 段, {status}")
    return commentary