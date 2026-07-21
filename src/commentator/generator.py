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
# 与 validators.MIN_VOICEOVER_LEN 同步（PLAN-003 B2 宽松化 48→44）。本模块未直接
# 使用该值做长度判定，仅为避免外部 from generator import MIN_VOICEOVER_LEN 出现
# 不一致而保留同步；真值源在 validators.py。
MIN_VOICEOVER_LEN = 44

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


def _align_segments_to_nodes(segments: list, chunk_nodes: list):
    """PLAN-003 C1a：segment 数量多于节点时的本地重组。

    API 模型（尤其推理模型）在多 chunk 场景常不遵守边界，一次吐出全部节点的
    segment（如 chunk2 只 1 个节点却吐 5 个）。这里按 chunk_nodes 的 id 顺序，
    从 segments 里挑出 id 匹配的那些，丢弃多余的。

    **安全原则（REVIEW-003 C1a bug 修复）**：只接受 id 全部精确匹配的情况——
    按 chunk_nodes 的 id 顺序取对应 segment，保证位置与 node 一一对应，后续
    auto_fix/validate 按位置用对的 node 校验（ADR-015 真值边界）。id 不全匹配
    时**直接返回 None 放弃抢救**，让原数量校验失败走重试→§8 舍弃。

    **不做"按位置取前 N / 后 N"退化**：REVIEW-003 指出，触发样本多为末尾 chunk
    （chunk2 只 1 节点），模型多吐时正确内容在输出末尾；按位置取前 N 会把前序
    节点内容错配给当前 chunk 节点，validator 还强制改写 id，产出"讲错棋却盖对 id"
    的视频——比整片舍弃更危险，违反 SPEC §3「抢救不得伪造/错配内容」。宁可舍弃
    不可静默错配。

    重组只挑选、不编造、不复制内容；选出的 segment 仍需走完整 auto-fix + validator。
    返回对齐后的 segments list（长度 == len(chunk_nodes)）；无法精确对齐返回 None。
    """
    if not segments or not chunk_nodes:
        return None
    need = len(chunk_nodes)
    if len(segments) <= need:
        return None  # 数量不多于节点，不归 C1a 处理

    # 收集 chunk_nodes 期望的 id 顺序（None 表示该 node 无 id，无法精确匹配）。
    # 注意用 `isinstance(x, int) and not isinstance(x, bool)`：Python 中 bool 是
    # int 子类，isinstance(True, int)==True，模型若把 id 写成 true/false 会误匹配。
    def _is_int_id(v):
        return isinstance(v, int) and not isinstance(v, bool)

    expected_ids = []
    for n in chunk_nodes:
        nid = n.get("id")
        expected_ids.append(nid if _is_int_id(nid) else None)
    # 期望 id 自身不能重复（同 chunk 两节点同 id 是数据异常），重复则放弃
    real_ids = [eid for eid in expected_ids if eid is not None]
    if len(real_ids) != len(set(real_ids)):
        return None

    # 建立 model segment 的 id -> seg 索引（首次命中保留，天然去重重复 id）
    by_id = {}
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        sid = seg.get("id")
        if _is_int_id(sid) and sid not in by_id:
            by_id[sid] = seg

    # 仅当所有期望 id 都精确命中时，按期望顺序取——位置与 node 一一对应
    if all(eid is not None and eid in by_id for eid in expected_ids):
        return [by_id[eid] for eid in expected_ids]

    return None


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

            # PLAN-003 C1a：segment 数量对齐。API 模型（尤其推理模型）常在多 chunk
            # 场景下不遵守 chunk 边界，一次吐出全部节点的 segment（如 chunk2 只有
            # 1 个节点却吐 5 个）。此前数量不匹配时直接跳过所有 auto-fix/repair
            # 走重试，导致非确定性的整片舍弃。这里先做本地重组：
            #   - 数量多于节点：仅当所有期望 id 都精确匹配时，按 id 顺序取对应
            #     segment（保证位置与 node 一一对应）；id 不全匹配则放弃抢救
            #     （REVIEW-003：不取前N/后N退化，避免末尾chunk错配违反 SPEC §3）。
            #   - 数量少于节点：不补生成（PLAN-003 C1b 降级为直接走 §8 舍弃，
            #     避免引入新的 API 调用与泄漏面）。
            # 重组后仍走完整 auto-fix + validator，不放宽任何内容校验。
            segments = data.get("segments")
            if isinstance(segments, list) and chunk_nodes and len(segments) != len(chunk_nodes):
                if len(segments) > len(chunk_nodes):
                    aligned = _align_segments_to_nodes(segments, chunk_nodes)
                    if aligned is not None:
                        data["segments"] = aligned
                        segments = aligned
                        # 不算重试成功，仍走下面 auto-fix + validate 确认内容合法
                    else:
                        # 放弃抢救：多吐的 segment 无法按 id 精确匹配，宁可走 §8
                        # 舍弃也不静默错配（SPEC §3）。记日志便于排查模型输出模式。
                        Logger.warn(
                            f"  C1a 放弃抢救：模型吐 {len(segments)} 段但本 chunk "
                            f"{len(chunk_nodes)} 节点，id 无法精确匹配"
                        )
                # 数量少于节点：不处理，让 validator 的数量校验照常拦

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