"""
PLAN-007：解说词全局润色 Pass

在 post_process 之后以全量上下文调用 LLM，修正「语气与局面进度不匹配」的段落。
核心价值：信息不对称——生成时模型只看 4 节点/chunk，润色时看全部节点+全部解说词+全局骨架。

三档开关（.env ENABLE_POLISH）：
  off    → 完全跳过（默认）
  detect → 调用 LLM 检测但不修改，日志输出建议（验证精度用）
  true   → 实际修改 + 安全网

安全网管线：id存在性 → 保护段排除 → auto_fix清洗 → 长度守恒(±20%)
           → claim_level词汇检查 → 硬截断(30%) → 接受/回退
"""
from src.common import GeneratedCommentary, Logger
from src.commentator.text_filters import strip_thinking
from typing import Optional
import os
import re
import json
import math


def _get_polish_mode() -> str:
    """读取 ENABLE_POLISH 环境变量，返回 off/detect/true。"""
    mode = os.getenv("ENABLE_POLISH", "off").strip().lower()
    if mode in ("off", "detect", "true"):
        return mode
    return "off"


def _should_skip(backend) -> bool:
    """本地后端 n_ctx=4096 token 预算不足，跳过润色。"""
    # DeepSeekBackend 有 model 属性且无 n_ctx；LlamaCppBackend 有 n_ctx
    n_ctx = getattr(backend, "n_ctx", None)
    if n_ctx is not None and n_ctx < 8192:
        return True

    # P2（REVIEW-002）：API 不可用即不润色。润色是 API 独占能力——全量 prompt
    # 约 5600 token，本地 n_ctx=4096 必然超窗。原实现只递归查 n_ctx，而
    # FallbackBackend/DeepSeekBackend 都无该属性 → 恒 False，降级本地时仍会
    # 白发一次超窗请求（异常被吞、静默跳过）。改为显式判断：主后端永久失败
    # （缺 key / 401 / 403）或熔断已开 → 直接跳过，不再尝试。
    if getattr(backend, "is_permanently_broken", False):
        return True
    if getattr(backend, "_permanent_failure", False) or getattr(backend, "_circuit_open", False):
        return True

    # FallbackBackend 包装时检查内部
    primary = getattr(backend, "_primary", None)
    if primary is not None:
        return _should_skip(primary)
    return False


def _is_protected_segment(seg, nodes: list, storyboard: dict) -> bool:
    """判断段是否受保护（不接受 polish 编辑）。

    保护规则：
    - 末段且 claim_level=="terminal"（保护将杀结论句）
    - puzzle 链路中 is_core_theme_key_move==True 的节点（保护双关键点）
    """
    seg_id = getattr(seg, "id", 0)
    # 找对应 node
    node = None
    for n in nodes:
        if n.get("id") == seg_id:
            node = n
            break
    if node is None:
        return False

    # 末段 + terminal
    if node.get("claim_level") == "terminal" and seg_id == nodes[-1].get("id"):
        return True

    # Puzzle 双关键点保护
    if node.get("is_core_theme_key_move"):
        return True

    return False


# claim_level 词汇约束：不同权限级别禁止出现的词
_CLAIM_FORBIDDEN = {
    "positioning": ("将杀", "绝杀", "胜负已定", "胜势", "终局"),
    "constraining": ("将杀", "绝杀", "胜负已定", "胜势"),
    "forcing": ("将杀", "绝杀", "胜负已定"),
    "terminal": (),  # terminal 无限制
}


def _check_claim_level(new_text: str, node: dict) -> bool:
    """检查编辑后文本是否违反 claim_level 词汇约束。返回 True=合法。"""
    claim = node.get("claim_level", "positioning")
    forbidden = _CLAIM_FORBIDDEN.get(claim, ())
    for word in forbidden:
        if word in new_text:
            return False
    return True


def _length_ok(original: str, new: str, tolerance: float = 0.2) -> bool:
    """长度守恒检查：新文本字数不超过原文 ±tolerance。"""
    orig_len = len(original)
    new_len = len(new)
    if orig_len == 0:
        return new_len == 0
    ratio = new_len / orig_len
    return (1 - tolerance) <= ratio <= (1 + tolerance)


def _build_polish_prompt(segments, nodes: list, storyboard: dict) -> str:
    """构造润色 prompt：全量上下文 + 铁律约束 + few-shot 示例。"""
    endgame_name = storyboard.get("endgame_name", storyboard.get("tactic_name", "残局"))
    role_summary = storyboard.get("role_summary", "")
    winning_side = storyboard.get("winning_side", "")

    parts = [
        "你是持有完整棋谱的编辑。你的唯一任务是修正「语气与局面进度不匹配」的段落。",
        "你不改写事实、不增删内容、不调整结构——只微调语气词和判断措辞。",
        "",
        f"【全局上下文】残局类型: {endgame_name}",
    ]
    if winning_side:
        parts.append(f"取胜方: {winning_side}")
    if role_summary:
        parts.append(f"攻守角色: {role_summary}")

    # 取胜骨架
    phases = storyboard.get("phases", []) or []
    phase_names = [p[0] for p in phases if isinstance(p, (list, tuple)) and p]
    if phase_names:
        parts.append(f"取胜阶段次序: {' → '.join(phase_names)}")

    # 每节点标注
    parts.extend(["", "【各段标注】"])
    node_by_id = {n.get("id"): n for n in nodes}
    for seg in segments:
        seg_id = getattr(seg, "id", 0)
        node = node_by_id.get(seg_id, {})
        emphasis = node.get("emphasis_level", "important")
        claim = node.get("claim_level", "positioning")
        phase = node.get("phase", "")
        goal = node.get("position_goal", "")
        parts.append(
            f"  id={seg_id} | emphasis={emphasis} | claim={claim} | "
            f"phase={phase} | goal={goal}"
        )

    # 全部 voiceover 原文
    parts.extend(["", "【全部解说词原文】"])
    for seg in segments:
        seg_id = getattr(seg, "id", 0)
        vo = getattr(seg, "voiceover", "")
        parts.append(f"[id={seg_id}] {vo}")

    # 铁律约束
    parts.extend([
        "",
        "【铁律（违反任何一条则该编辑无效）】",
        "1. 只改语气/判断措辞，不改事实描述（棋子动作、局面状态、战术名称）",
        "2. 只能修改形容词/副词/语气词/判断句，不得增删事实主语与动词",
        "3. 不得改变句子数量（原文3句改后仍3句）",
        "4. 不得增删棋学术语（对王、关键格、菲利多防线等）",
        "5. 长度守恒：修改后字数不超过原文±20%",
        "6. 禁止引入坐标/英文字母/数字",
        "7. claim_level权限不升级（positioning段不能出现将杀/绝杀/胜势）",
        "8. 保守原则：拿不准的不动，宁可漏改不可错改",
        "9. 如果所有段都没有明显问题，输出空edits数组",
        "10. 不得引入以下已去重的套话：看似平淡实则、胜利的天平、囊中之物、不可阻挡、暗藏杀机、步步为营、致命一击、大局已定、完全掌握主动权、稳获子力优势",
        "",
        "【示例】",
        '原文: "关键一手！白车横移封住黑王退路" (id=3, emphasis=routine)',
        '修改: "白车横移封住黑王退路" (reason: routine段不应用"关键一手"的强语气)',
        "",
        '原文: "白王稳步逼近，胜券在握" (id=7, phase=压缩, claim=constraining)',
        '修改: "白王稳步逼近，继续压缩黑王空间" (reason: constraining阶段不应说"胜券在握")',
        "",
        '原文: "黑王被逼入角落，至此形成将杀" (id=12, claim=terminal)',
        "不改（terminal段有权说将杀，且语气匹配）",
        "",
        '【输出格式】只输出JSON，不加markdown标记：',
        '{"edits": [{"id": int, "voiceover": "修正后全文", "reason": "一句话理由"}]}',
        "如果无需修改，输出：{\"edits\": []}",
    ])

    return "\n".join(parts)


def _parse_polish_response(raw_text: str) -> list:
    """解析 LLM 润色响应为 edits 列表。失败返回空列表。"""
    if not raw_text:
        return []
    # 尝试提取 JSON
    text = raw_text.strip()
    # 去除可能的 markdown 代码块标记
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        edits = data.get("edits", [])
        if isinstance(edits, list):
            return edits
    except (json.JSONDecodeError, AttributeError):
        pass
    # 尝试从文本中提取 JSON 对象（支持嵌套大括号）
    match = re.search(r'\{.*?"edits"\s*:\s*\[.*?\]\s*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            edits = data.get("edits", [])
            if isinstance(edits, list):
                return edits
        except (json.JSONDecodeError, AttributeError):
            pass
    return []


def polish_commentary(commentary: GeneratedCommentary, nodes: list,
                      storyboard: dict, backend, config=None) -> None:
    """PLAN-007 全局润色主入口。

    在 post_process 之后调用。根据 ENABLE_POLISH 环境变量决定行为：
    - off: 直接返回
    - detect: 检测并日志输出，不修改
    - true: 实际修改 + 安全网
    """
    mode = _get_polish_mode()
    if mode == "off":
        return

    segments = commentary.segments
    if not segments or len(segments) < 2:
        return

    # 本地后端 token 预算不足，跳过
    if _should_skip(backend):
        Logger.info("[Polish] 本地后端 token 预算不足，跳过润色")
        return

    # 构造 prompt 并调用 LLM
    prompt = _build_polish_prompt(segments, nodes, storyboard)
    try:
        raw = backend.generate(prompt, grammar=None)
        raw = strip_thinking(raw)
    except Exception as e:
        Logger.warn(f"[Polish] LLM 调用异常，跳过: {e}")
        return

    edits = _parse_polish_response(raw)
    if not edits:
        Logger.info("[Polish] LLM 未建议任何修改")
        return

    # 硬截断：edits > 30% 段数时只取前 N 条（按 id 升序）
    max_edits = math.ceil(len(segments) * 0.3)
    if len(edits) > max_edits:
        edits = sorted(edits, key=lambda e: e.get("id", 0))[:max_edits]
        Logger.info(f"[Polish] 硬截断：{len(edits)} 条（上限 {max_edits}）")

    # 建立 seg id -> (index, seg) 映射
    seg_by_id = {}
    for i, seg in enumerate(segments):
        seg_by_id[getattr(seg, "id", 0)] = (i, seg)

    # 建立 node id -> node 映射
    node_by_id = {n.get("id"): n for n in nodes}

    if mode == "detect":
        # Detect 模式：只日志输出，不修改
        Logger.info(f"[Polish·Detect] 检测到 {len(edits)} 段建议修改：")
        for edit in edits:
            eid = edit.get("id", "?")
            reason = edit.get("reason", "无理由")
            Logger.info(f"  id={eid}: {reason}")
        return

    # === true 模式：应用编辑 + 安全网 ===
    accepted = 0
    rejected = 0
    reject_reasons = []

    # 去重：同一 id 只取第一条
    seen_ids = set()
    unique_edits = []
    for edit in edits:
        eid = edit.get("id")
        if eid not in seen_ids:
            seen_ids.add(eid)
            unique_edits.append(edit)

    for edit in unique_edits:
        eid = edit.get("id")
        new_vo = edit.get("voiceover", "")
        if not eid or not new_vo:
            rejected += 1
            reject_reasons.append(f"id={eid} 无效编辑")
            continue

        # id 存在性检查
        if eid not in seg_by_id:
            rejected += 1
            reject_reasons.append(f"id={eid} 不存在")
            continue

        idx, seg = seg_by_id[eid]
        node = node_by_id.get(eid, {})
        original_vo = getattr(seg, "voiceover", "")

        # node 缺失（edit 的 id 对不上任何 node）：无法核查，保守拒绝。
        # validate_single_segment 内部会访问 node["id"]，空 node 会抛异常，故前置拦截。
        if not node:
            rejected += 1
            reject_reasons.append(f"id={eid} 无对应node无法核查")
            continue

        # 保护段排除
        if _is_protected_segment(seg, nodes, storyboard):
            rejected += 1
            reject_reasons.append(f"id={eid} 保护段")
            continue

        # auto_fix 清洗（I-1：确保 polish 输出也经过事实纠错）
        if config and hasattr(config, "auto_fix_voiceover"):
            new_vo = config.auto_fix_voiceover(new_vo, node)

        # P1（PLAN-007 REVIEW-002 修）：复用主生成的 validator 重验，而非轻量表层闸。
        # 润色跑在链路最后、作用于成片文本，却是唯一无 §8 舍弃通道的一环——写坏直接进片。
        # 主生成必须过 validator，润色没有理由用更弱的尺子。这里对单条 edit 用其对应
        # node 重跑 config.validate_chunk（endgame→validate_storyboard_chunk /
        # puzzle→validate_puzzle_chunk，内部逐段跑完整 validate_single_segment /
        # validate_puzzle_segment：surface 泄漏闸、material_existence 造子、假将军、
        # 多着提前宣称将杀等全部规则）。不通过则保留原文——原文本已过 validator，
        # 回退永远安全，润色退化为"改不动就不改"，纯装饰、零风险。
        # 注意：validate_chunk 会原地改写 seg dict 的 id，故传入临时 dict 而非污染真实对象。
        if config and getattr(config, "validate_chunk", None):
            probe_seg = dict(seg_by_id[eid][1].__dict__) if hasattr(seg_by_id[eid][1], "__dict__") else {}
            probe_seg["id"] = node.get("id", eid)
            probe_seg["voiceover"] = new_vo
            probe_seg["pacing"] = getattr(seg, "pacing", "normal")
            v_ok, v_err = config.validate_chunk({"segments": [probe_seg]}, [node])
            if not v_ok:
                rejected += 1
                reject_reasons.append(f"id={eid} validator拒绝({v_err})")
                continue
        else:
            # config 缺失（防御，正常链路不会走到）：保守拒绝，宁可不润色也不放行未核查文本
            rejected += 1
            reject_reasons.append(f"id={eid} 无validator无法核查")
            continue

        # 长度守恒检查（润色专属：validator 只管下限，这里防止语气微调时大幅改写）
        if not _length_ok(original_vo, new_vo):
            rejected += 1
            reject_reasons.append(f"id={eid} 长度超限")
            continue

        # claim_level 词汇检查（润色专属：跨段进度语气约束，validator 不覆盖）
        if not _check_claim_level(new_vo, node):
            rejected += 1
            reject_reasons.append(f"id={eid} claim_level升级")
            continue

        # 通过所有安全网，接受编辑
        seg.voiceover = new_vo
        accepted += 1

    Logger.info(
        f"[Polish] 输入 {len(segments)} 段，LLM 建议 {len(edits)} 条编辑，"
        f"接受 {accepted} 条，回退 {rejected} 条"
    )
    if reject_reasons:
        Logger.info(f"[Polish] 回退原因: {'; '.join(reject_reasons[:5])}")
