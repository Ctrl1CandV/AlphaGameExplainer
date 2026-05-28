from src.common import Logger, StoryboardSegment, StoryboardVisuals, StoryboardArrow
from src.common import GeneratedCommentary, ALLOWED_PACING, ALLOWED_ARROW_COLORS
from src.common import is_valid_square_name, normalize_pacing
from src.llm_backend import create_backend_from_env, release_backend
from dotenv import load_dotenv
from typing import Optional
import chess
import json
import os
import re

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
CHUNK_SIZE = 4
MAX_CHARS = 1800
MAX_RETRIES = 1
MIN_VOICEOVER_LEN = 48

_EXAMPLE_BY_ENDGAME = {
    "单车杀王": (
        "第1步：Ra5+。白车移至a5将军，画出了第一条控制线——黑王被限制在第5排以上，活动空间开始缩小。\n"
        "第2步：Kd4→Kc4→Kd3→Ke3（4着）。白王稳步向中心推进，与车形成配合之势，逐步压缩黑王活动范围。\n"
        "第3步：Rh4。关键一手！车横向移至h4，与白王形成横排对王——黑王被锁定在棋盘边缘，再无法回到中心。\n"
        "总结：单车杀王的核心是盒子法和对王配合——用车画控制线→王车合围压缩→在边线将杀。"
    ),
    "车兵对车": (
        "第1步：Rc1。白车退至c1建立菲利多防线——白王守住兵前的关键格e3，车在后方控制c线，黑方无法正面突破。\n"
        "第2步：Kf6→Ke6→Kd5（3着）。黑王向白兵逼近，试图将白王挤出防线。\n"
        "第3步：Ra1+。黑车突然从侧翼将军！白王被迫离开d线，菲利多防线被正面打破，局面出现转折。\n"
        "总结：车兵对车的关键是弱方建立菲利多防线。防线一旦被侧翼突破，兵失去保护则必败。"
    ),
    "单兵残局": (
        "第1步：Kd4。白王抢占关键格d4——正对黑王形成对王，黑王被迫后退，为兵推进扫清障碍。\n"
        "第2步：Kd6→Ke5（2着）。黑王被迫退至d6防守，白王牢牢占据关键格，兵可以安全前进。\n"
        "第3步：e4。白兵在王的保护下开始推进，向升变格e8迈出第一步。\n"
        "总结：单兵残局的核心是对王与关键格。占据关键格→保护兵推进→升变取胜。"
    ),
    "单后杀王": (
        "第1步：Qf6。白后从远处控制黑王逃跑路线，将黑王限制在棋盘右下角区域。\n"
        "第2步：Kc6→Kd5→Ke4（3着）。白王向黑王稳步靠近，准备配合后完成合围。\n"
        "第3步：Qg7。关键一手！后将黑王锁定在边线，保持安全距离避免逼和。\n"
        "总结：单后杀王的关键是保持后的安全距离避免逼和，用己方王配合后逐步将对方王逼至边线角落将杀。"
    ),
}

_EXAMPLE_FALLBACK = (
    "第1步：Nf3。白马跳至f3控制中心d4和e5格。\n"
    "第2步：e5→d6→Nc6（3着）。黑方在中心展开反击，用兵和马争夺中心空间。\n"
    "第3步：Bg5。关键一手！象牵制黑方f6马，削弱黑方对d5格的控制。\n"
    "总结：先控制中心，再展开子力，最后集中火力发动攻击完成将杀。"
)

_JSON_EXAMPLE = """{"segments":[{"id":1,"sub_endgame":"车兵对车","voiceover":"白车退至c1建立菲利多防线，核心是让白王继续守住兵前关键格e3，同时用车在后方稳定控制c线。这样黑方即使想从正面逼近，也暂时找不到直接突破的入口。","pacing":"slow"},{"id":2,"sub_endgame":"车兵对车","voiceover":"承接前一步已经搭好的防线，黑王继续向白兵逼近，意图把白王从关键格一带挤开。这里的重点不是立刻制造战术，而是通过王位前压不断测试防线是否会出现松动。","pacing":"normal"},{"id":3,"sub_endgame":"车兵对车","voiceover":"顺着前面对防线的持续施压，黑车突然从侧翼发力，对白方王位形成更直接的骚扰。白王一旦被迫离开d线附近，原本稳定的菲利多防线就会出现裂缝，局面也会随之进入真正的转折阶段。","pacing":"slow"}]}"""


def _get_example(endgame_name: str) -> str:
    return _EXAMPLE_BY_ENDGAME.get(endgame_name, _EXAMPLE_FALLBACK)


def _build_header(storyboard: dict) -> str:
    endgame_name = storyboard.get("endgame_name", "残局")
    role_summary = storyboard.get("role_summary", "")
    hard_constraints = storyboard.get("hard_constraints", [])
    has_switch = storyboard.get("has_sub_endgame_switch", False)

    node_count = len(storyboard.get("nodes", []))

    parts = [
        "你是专业的国际象棋残局教练。请输出专业、严谨、以棋理为核心的中文解说，纯文本输出。",
        "要求：只依据给定走法和局面信息解说，禁止虚构剧情，禁止空泛比喻。你的任务是解释局面变化，而不是自由创作。",
        "",
        f"【起始残局类型】{endgame_name}",
    ]

    if role_summary:
        parts.append(f"【攻守角色】{role_summary}")
    if hard_constraints:
        parts.append(f"【全局约束】{'；'.join(hard_constraints)}")

    if has_switch:
        parts.extend([
            "",
            "【重要】中途可能发生残局类型转换（如车兵对车→升变→车对车→单车杀王）。",
            "每个节点都标注了当前的子残局类型，必须按当前类型使用对应的概念和术语。",
            "残局类型转换后，禁止继续沿用旧类型的理论框架。",
        ])

    parts.extend([
        "",
        "【解说规则】",
        f"- 你只能输出第1步到第{node_count}步这{node_count}个节点，绝不能额外扩写成更多步",
        "- 每一步都按这个顺序组织：走法是什么 → 局面变化是什么 → 为什么这样走有用 → 这一段的残局教学点是什么",
        "- 必须严格遵循每个节点标注的【当前残局】【允许概念】【禁止概念】",
        "- 关键步骤：详细分析局面发生了什么实质变化",
        "- 多着调整步骤：概括整段机动的目的，但仍要点出关键轨迹和为什么有用",
        "- 如果节点起止局面相同，只能解释为反复试探、等招或调车，不得写成突破",
        "- 禁止使用引擎术语：评估值、分数、厘兵、半着、DTM、mate in N",
        "- 不得虚构或假设走法。如数据中未提供某步的精确走法，描述为「经过N着调整」而非编造格子",
        "- 禁止使用「假设」「可能」「如果」等猜测性语言描述已发生的走法",
        "- 最后一个节点允许总结，但只总结当前节点的子残局类型规律，不要机械地套用起始残局的框架",
        "",
    ])
    return "\n".join(parts)


def _build_json_header(storyboard: dict) -> str:
    endgame_name = storyboard.get("endgame_name", "残局")
    role_summary = storyboard.get("role_summary", "")
    hard_constraints = storyboard.get("hard_constraints", [])
    winning_side = storyboard.get("winning_side", "")
    losing_side = storyboard.get("losing_side", "")
    node_count = len(storyboard.get("nodes", []))

    parts = [
        "你是国际象棋赛事解说员，负责为残局教学视频配解说词。只输出合法JSON，不加任何解释或markdown标记。",
        "风格要求：像专业赛事解说员那样——既有技术深度，又有叙事感染力。把每一段残局讲成一个有推进感的故事。",
        "",
        f"【残局类型】{endgame_name}",
    ]

    if winning_side and losing_side:
        parts.extend([
            f"【叙事立场】从{winning_side}（主动推进方）视角讲述。聚焦于{winning_side}如何逐步建立优势、压缩对手空间。",
        ])

    if role_summary:
        parts.append(f"【攻守角色】{role_summary}")
    if hard_constraints:
        parts.append(f"【全局约束】{'；'.join(hard_constraints)}")

    parts.extend([
        "",
        "每个节点有 claim_level 控制可用的结论深度：",
        "positioning→只能讲站位和控制  constraining→可以讲空间压缩",
        "forcing→可以讲强制/被迫  terminal→才能说将杀/绝杀",
        "",
        "节点可能标注「将军驱赶」或「反复试探等待」→ 这种节点是多着合并的叙事块，你要用流畅的段落描述这段过程，而不是逐步数着。",
        "",
        "【JSON格式】",
        '{"segments":[{"id":int,"sub_endgame":"string","voiceover":"string","pacing":"slow|normal|fast|pause_before|pause_after"},...]}',
        "segments数量必须等于本块节点数。不输出visuals字段。",
        "",
        "【解说要求】",
        f"- 正好{node_count}个segment，不增不减",
        "- 用自然的中文解说，避免引擎术语（如评估值、DTM、mate in N）",
        "- 每段50-200字，summary_only的用1句话概括（≤80字）",
        "- 各段之间连续推进，后一段承接前一段已建立的局面",
        "- 最后一段若是terminal权限，以「至此形成将杀」或「至此胜负已定」收束",
        "- 王的描述侧重于「逼近」「封住逃格」「配合主力子压缩空间」等位置性语言",
        "",
    ])
    return "\n".join(parts)


def _goal_to_narrative_phrase(goal: str) -> str:
    mapping = {
        "improve_piece_coordination": "改善站位与子力协调",
        "hold_net": "维持既有控制网",
        "shrink_space": "继续压缩对方王的活动空间",
        "drive_to_edge": "把对方王继续逼向边线",
        "drive_to_corner": "把对方王进一步赶向角落",
        "convert_to_mate": "把优势转入最后收网",
    }
    return mapping.get(goal, "继续推进优势")


def _build_chunk_outline(chunk_nodes: list) -> str:
    parts = []
    for node in chunk_nodes:
        parts.append(f"第{node['id']}步{_goal_to_narrative_phrase(node.get('position_goal', ''))}")
    return "；".join(parts)


def _build_prev_context(prev_node: dict) -> str:
    if not prev_node:
        return ""
    phase = prev_node.get("phase", "") or "推进阶段"
    goal_text = _goal_to_narrative_phrase(prev_node.get("position_goal", ""))
    teaching = prev_node.get("teaching_focus", "")
    parts = [
        f"上一段落点：第{prev_node.get('id')}步结束后，局面已经进入「{phase}」，当前主线是{goal_text}。",
    ]
    if teaching:
        parts.append(f"上一段留下的教学重点：{teaching}")
    return "\n".join(parts)


def _build_chunk_prompt(header: str, chunk_nodes: list, chunk_idx: int, total_chunks: int, example: str, prev_context: str = "") -> str:
    is_last = (chunk_idx == total_chunks - 1)
    parts = [header]
    if chunk_idx == 0 and example:
        parts.extend(["【输出示例】", example, ""])

    chunk_rule = ""
    if total_chunks > 1:
        chunk_rule += "本段只解说这些步骤，禁止提前总结。" if not is_last else "本段包含最后几步，允许总结。"
    parts.append(f"--- 第{chunk_idx + 1}/{total_chunks}段节点 {'(最后)' if is_last else ''} ---")
    if chunk_rule:
        parts.append(chunk_rule)
    if prev_context:
        parts.extend(["【上一段承接】", prev_context])
    parts.extend([
        "【本段推进主线】",
        _build_chunk_outline(chunk_nodes),
        "写作要求：第一个segment先承接上一段落点；后续segment承接本块上一节点已经形成的局面结果。",
    ])
    parts.append("")

    for node in chunk_nodes:
        node_id = node["id"]
        sub_name = node.get("sub_endgame_name", "")
        goal = _goal_to_narrative_phrase(node.get("position_goal", ""))
        claim = node.get("claim_level", "positioning")
        summary_only = node.get("summary_only", False)
        tags = node.get("tags", [])

        parts.append(f"--- 节点{node_id} ---")
        parts.append(f"走法: {node['moves']}（{node['move_count']}着, {node.get('turn','')}）")
        parts.append(f"状态: {'已将军' if node.get('is_check_after') else '含将军走法' if node.get('has_check_in_node') else '非将军'}"
                     f" | {'已将杀' if node.get('is_checkmate_after') else '未将杀'}"
                     f" | {'含吃子' if node.get('is_capture_node') else '未吃子'}")
        parts.append(f"目标: {goal} | 权限: {claim}{' (禁止将杀/绝杀)' if claim != 'terminal' else ' (可宣告胜负)'}")

        drive_tag = next((t for t in tags if t in ("将军驱赶", "连续将军驱赶", "反复试探等待")), "")
        if drive_tag:
            parts.append(f"类型: 「{drive_tag}」叙事块 — 这是多着合并，描述整体过程，不要逐步数着")

        if summary_only:
            parts.append("概括模式: 只1句话概括（≤80字）")

        if node.get("endgame_changed"):
            parts.append(f"残局切换: {sub_name}")
        if sub_name:
            parts.append(f"当前残局: {sub_name}")
        if node.get("transition_summary") and node["is_critical"]:
            parts.append(f"局面变化: {node['transition_summary']}")

        parts.append("")

    return "\n".join(parts)


def _strip_thinking(text: str) -> str:
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    return text.strip()


def _extract_json_text(text: str) -> str:
    t = text.strip()
    if not t:
        return ""
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    brace_start = t.find("{")
    brace_end = t.rfind("}")
    if brace_start == -1 or brace_end <= brace_start:
        return ""
    return t[brace_start:brace_end + 1]


def _repair_common_json_issues(text: str) -> str:
    fixed = text.strip()
    fixed = re.sub(r"^```(?:json)?\s*", "", fixed)
    fixed = re.sub(r"\s*```$", "", fixed)
    fixed = fixed.replace("“", "\"").replace("”", "\"")
    fixed = fixed.replace("‘", "'").replace("’", "'")
    fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)
    return fixed


def _parse_storyboard_json(text: str) -> dict:
    candidates = []
    extracted = _extract_json_text(text)
    if extracted:
        candidates.append(extracted)
        candidates.append(_repair_common_json_issues(extracted))
    repaired_full = _repair_common_json_issues(text)
    if repaired_full and repaired_full not in candidates:
        candidates.append(repaired_full)

    for json_text in candidates:
        if not json_text:
            continue
        try:
            data = json.loads(json_text)
            if isinstance(data, dict) and "segments" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            continue
    return _INVALID_JSON_SENTINEL

_INVALID_JSON_SENTINEL = object()


def _auto_fix_voiceover(text: str, node: dict) -> str:
    fixed = text

    allows_check = node.get("is_check_after") or node.get("has_check_in_node")

    if node.get("king_moved"):
        for w in ("王将军", "王形成杀", "王绝杀", "王直接", "致命将军"):
            fixed = fixed.replace(w, "王步步紧逼")

    if not allows_check:
        fixed = fixed.replace("连续将军驱赶", "连续追击")
        fixed = fixed.replace("将军驱赶", "追击驱赶")
        fixed = fixed.replace("将军追击", "追击")
        fixed = fixed.replace("连续将军", "连续进攻")
        fixed = fixed.replace("发起将军", "发起进攻")
        fixed = fixed.replace("开始将军", "开始进攻")
        fixed = fixed.replace("实施将军", "施加压力")
        fixed = re.sub(r"(?<=[a-h][1-8])\s*将军", " 叫杀", fixed)
        fixed = fixed.replace("将军", "施压")

    _CHECKMATE_WORDS = ("将杀", "绝杀", "杀王", "终局已定", "锁定胜局")
    if node.get("is_checkmate_after") is not True:
        fixed = fixed.replace("死局已定", "败局已定")
        fixed = fixed.replace("无路可走", "陷入绝境")
        fixed = fixed.replace("无路可逃", "陷入绝境")
        fixed = fixed.replace("死局", "败势已现")
        for w in _CHECKMATE_WORDS:
            fixed = fixed.replace(w, "胜势")

    _NEUTRALITY_WORDS = ("双方等待", "局势平衡", "互相试探", "积蓄力量", "均势", "双方都在")
    for w in _NEUTRALITY_WORDS:
        if w in fixed:
            fixed = fixed.replace("双方等待", "周旋")
            fixed = fixed.replace("局势平衡", "局面明朗")
            fixed = fixed.replace("互相试探", "相互牵制")
            fixed = fixed.replace("积蓄力量", "蓄势待发")
            fixed = fixed.replace("均势", "局面向好")
            fixed = fixed.replace("双方都在", "双方")

    if not node.get("is_capture_node"):
        fixed = fixed.replace("吃掉", "控制").replace("吃子", "控制")
        fixed = fixed.replace("兑掉", "交换").replace("兑子", "交换子力")
        fixed = fixed.replace("吞掉", "占据")
        fixed = fixed.replace("吃", "控制")
        fixed = fixed.replace("控制控制", "控制")

    if node.get("is_game_over_after") and node.get("legal_reply_count_after", 1) == 0:
        for w in ("黑方应将", "白方应将", "黑方应对", "白方应对"):
            fixed = fixed.replace(w, "")
        fixed = re.sub(r"(?:黑方|白方)应[，,]?\s*", "", fixed)
        fixed = re.sub(r"(?:下一步|随后再)[^，。,]{0,8}(?:，|,|\s*)", "", fixed)

    fixed = re.sub(r"[，,]{2,}", "，", fixed)
    fixed = re.sub(r"。{2,}", "。", fixed)
    fixed = re.sub(r"\s{2,}", " ", fixed)
    fixed = fixed.strip()

    return fixed


def _validate_storyboard_chunk(data: dict, chunk_nodes: list) -> tuple:
    segments = data.get("segments")
    if not isinstance(segments, list):
        return False, "顶层缺少segments数组"
    if len(segments) != len(chunk_nodes):
        return False, f"segments数量{len(segments)}与节点数{len(chunk_nodes)}不一致"

    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            return False, f"第{i+1}个segment不是对象"
        node = chunk_nodes[i]

        seg_id = seg.get("id")
        if not isinstance(seg_id, int) or seg_id != node["id"]:
            return False, f"segment[{i}]的id={seg_id}，应为{node['id']}"

        voiceover = seg.get("voiceover")
        if not isinstance(voiceover, str) or not voiceover.strip():
            return False, f"segment[{i}]的voiceover为空"
        min_len = 28 if node.get("summary_only") else MIN_VOICEOVER_LEN
        if len(voiceover.strip()) < min_len:
            return False, f"segment[{i}]的voiceover过短({len(voiceover.strip())}<{min_len})"

        pacing = seg.get("pacing", "normal")
        pacing = str(pacing).strip().lower()
        if pacing not in ALLOWED_PACING:
            return False, f"segment[{i}]的pacing='{pacing}'不合法"

        sub_endgame = seg.get("sub_endgame")
        if not isinstance(sub_endgame, str) or not sub_endgame.strip():
            return False, f"segment[{i}]的sub_endgame为空"

        text = voiceover.strip()
        _CHECKMATE_BANNED = ("将杀", "绝杀", "杀王", "无路可走", "无路可逃",
                             "死局", "终局已定", "锁定胜局")
        if node.get("is_checkmate_after") is False and any(word in text for word in _CHECKMATE_BANNED):
            return False, f"segment[{i}]错误宣称将杀"

        allows_check_word = node.get("is_check_after") or node.get("has_check_in_node")
        if not allows_check_word and "将军" in text:
            return False, f"segment[{i}]错误宣称将军"

        king_moved = node.get("king_moved", False)
        checking_types = node.get("checking_piece_types", [])
        king_claims_check = king_moved and chess.KING not in checking_types
        if king_claims_check and any(word in text for word in ("王将军", "王形成杀", "王绝杀", "王直接", "致命将军")):
            return False, f"segment[{i}]错误宣称王将军——国际象棋中王不能直接将军"

        if not node.get("is_capture_node") and any(word in text for word in ("吃掉", "兑掉", "吞掉")):
            return False, f"segment[{i}]错误宣称吃子"
        if node.get("is_game_over_after") and node.get("legal_reply_count_after", 1) == 0:
            if any(word in text for word in ("黑方应", "白方应", "下一步", "随后再")):
                return False, f"segment[{i}]在终局后继续虚构后续走法"

        _NEUTRALITY_BANNED = ("双方等待", "局势平衡", "互相试探", "积蓄力量", "均势", "双方都在")
        if any(word in text for word in _NEUTRALITY_BANNED):
            return False, f"segment[{i}]含有均势叙事词——这是必胜残局变现，必须从强方主导推进角度写"

        if node.get("claim_level", "positioning") != "terminal" and node.get("is_last_node"):
            if not any(word in text for word in ("胜负已定", "将杀", "形成将杀", "完成转化", "胜势兑现", "终局形成")):
                pass

        if node.get("summary_only"):
            if len(text) > 120:
                return False, f"segment[{i}]概括模式节点过长({len(text)}>120)，应只用1句话"

    return True, ""


def _safe_phase_label(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", "", text)[:12]


def _build_visuals_from_node(node: dict) -> StoryboardVisuals:
    highlights = []
    arrows = []

    try:
        temp = chess.Board(node.get("fen_before", ""))
    except Exception:
        return StoryboardVisuals(phase_label=_safe_phase_label(node.get("suggested_phase_label", "")))

    sans = node.get("sans") or [part.strip() for part in str(node.get("moves", "")).split("→") if part.strip()]
    for san in sans[:2]:
        try:
            move = temp.parse_san(san)
        except ValueError:
            continue

        from_sq = chess.square_name(move.from_square)
        to_sq = chess.square_name(move.to_square)
        color = "blue"
        label = ""

        if temp.gives_check(move):
            color = "red"
            label = "将军"
        elif temp.is_capture(move):
            color = "yellow"
            label = "吃子"
        else:
            piece = temp.piece_at(move.from_square)
            if piece and piece.piece_type == chess.KING:
                color = "green"
                label = "王位推进"

        if is_valid_square_name(to_sq) and to_sq not in highlights:
            highlights.append(to_sq)
        if is_valid_square_name(from_sq) and is_valid_square_name(to_sq) and color in ALLOWED_ARROW_COLORS:
            arrows.append(StoryboardArrow(
                from_sq=from_sq,
                to_sq=to_sq,
                color=color,
                label=label,
            ))
        temp.push(move)

    if node.get("is_checkmate_after") or node.get("is_check_after"):
        try:
            board_after = chess.Board(node.get("fen_after", ""))
            king_sq = board_after.king(board_after.turn)
            if king_sq is not None:
                sq_name = chess.square_name(king_sq)
                if sq_name not in highlights:
                    highlights.append(sq_name)
        except Exception:
            pass

    return StoryboardVisuals(
        extra_highlights=highlights[:3],
        arrows=arrows[:2],
        phase_label=_safe_phase_label(node.get("suggested_phase_label", "") or node.get("phase", "")),
    )


def _build_bridge_prefix(prev_node: dict, is_chunk_start: bool) -> str:
    if not prev_node:
        return ""
    goal = prev_node.get("position_goal", "")
    chunk_mapping = {
        "improve_piece_coordination": "承接前一段已经改善的站位，",
        "hold_net": "承接前一段已经搭起的控制网，",
        "shrink_space": "承接前一段已经完成的空间压缩，",
        "drive_to_edge": "顺着前一段把对方王逼向边线的路线，",
        "drive_to_corner": "顺着前一段把对方王赶向角落的思路，",
        "convert_to_mate": "承接前一段已经形成的收网态势，",
    }
    inner_mapping = {
        "improve_piece_coordination": "在前一步改善站位的基础上，",
        "hold_net": "承接前一步已经搭起的控制网，",
        "shrink_space": "顺着前一步的空间压缩，",
        "drive_to_edge": "顺着前一步把对方王逼向边线的思路，",
        "drive_to_corner": "顺着前一步把对方王赶向角落的路线，",
        "convert_to_mate": "承接前一步已经形成的收网态势，",
    }
    mapping = chunk_mapping if is_chunk_start else inner_mapping
    return mapping.get(goal, "承接前面的推进思路，")


def _cleanup_voiceover_text(text: str) -> str:
    fixed = text
    fixed = re.sub(r"[，,]{2,}", "，", fixed)
    fixed = re.sub(r"。{2,}", "。", fixed)
    fixed = re.sub(r"\s{2,}", " ", fixed)

    bridge_parts = (
        r"(?:承接前一步|顺着前一步|在前一步|承接前一段|顺着前一段|沿着前一步)"
        r"[^，。,]*?[，,]"
    )
    fixed = re.sub(rf"({bridge_parts})\s*{bridge_parts}", r"\1", fixed)

    return fixed.strip()


def _apply_chunk_bridges(chunk_segments: list, chunk_nodes: list, prev_tail_node: dict = None) -> list:
    connective = re.compile(r"^(承接|顺着|沿着|在前一步|在前面|继续|随后|接着|紧接着)")
    move_notation = re.compile(r"[a-h][1-8]\s*→|[KQRBNP][a-h]?[1-8]?x?[a-h][1-8]")
    for idx, seg in enumerate(chunk_segments):
        prev_node = prev_tail_node if idx == 0 else chunk_nodes[idx - 1]
        prefix = _build_bridge_prefix(prev_node, idx == 0)
        text = seg.voiceover.strip()
        if prefix and text and not connective.match(text) and not move_notation.search(text):
            seg.voiceover = prefix + text
        seg.voiceover = _cleanup_voiceover_text(seg.voiceover)
    return chunk_segments


def _dict_to_storyboard_segments(data: dict, chunk_nodes: list) -> list:
    result = []
    node_by_id = {node["id"]: node for node in chunk_nodes}
    for seg in data.get("segments", []):
        node = node_by_id.get(int(seg.get("id", 0)), {})
        visuals = _build_visuals_from_node(node)
        result.append(StoryboardSegment(
            id=int(seg.get("id", 0)),
            sub_endgame=str(seg.get("sub_endgame", "")),
            voiceover=str(seg.get("voiceover", "")),
            pacing=normalize_pacing(str(seg.get("pacing", "normal"))),
            visuals=visuals,
        ))
    return result


def _polish_voiceover(text: str, backend) -> str:
    if len(text) < 50:
        return text
    prompt = (
        "你是中文编辑。使以下象棋解说更流畅自然，不改走法和结论。只输出结果。\n\n"
        f"{text}"
    )
    result = backend.generate(prompt)
    if not result:
        return text
    polished = _strip_thinking(result).strip()
    if len(polished) < 20:
        return text
    changed = polished != text
    if changed:
        Logger.debug(f"      polish: {len(text)}→{len(polished)}字")
    return polished


def _finalize_chunk_segments(backend, data_or_segments, chunk_nodes: list, prev_tail_node: dict):
    data = data_or_segments if isinstance(data_or_segments, dict) else {"segments": data_or_segments}
    segments = _dict_to_storyboard_segments(data, chunk_nodes)
    segments = _apply_chunk_bridges(segments, chunk_nodes, prev_tail_node)
    polished_count = 0
    for seg in segments:
        before = seg.voiceover
        seg.voiceover = _polish_voiceover(seg.voiceover, backend)
        if seg.voiceover != before:
            polished_count += 1
    if polished_count:
        Logger.info(f"    polish: {polished_count}/{len(segments)} 段润色")
    return segments


def _build_chunk_grammar(n_segments: int) -> str:
    if n_segments <= 0:
        return ""
    seg_repeat = "segment" + "".join(' ws "," ws segment' for _ in range(n_segments - 1))
    return (
        'root ::= "{" ws "\\"segments\\"" ws ":" ws "[" ws ' + seg_repeat + ' ws "]" ws "}"\n'
        'segment ::= "{" ws "\\"id\\"" ws ":" ws integer ws "," ws '
        '"\\"sub_endgame\\"" ws ":" ws string ws "," ws '
        '"\\"voiceover\\"" ws ":" ws string ws "," ws '
        '"\\"pacing\\"" ws ":" ws pacing ws "}"\n'
        'pacing ::= "\\"slow\\"" | "\\"normal\\"" | "\\"fast\\"" | "\\"pause_before\\"" | "\\"pause_after\\""\n'
        'integer ::= [0-9]+\n'
        'string ::= "\\"" [^"\\\\x00-\\x1F]* "\\""\n'
        'ws ::= [ \\t\\n]*'
    )


def _build_retry_prompt(prompt: str, error_msg: str, attempt: int = 1) -> str:
    if "JSON" in error_msg or "不是合法" in error_msg:
        hint = (
            "请只输出一个合法JSON对象，不要加 ```json 代码块、"
            "markdown标记、或任何解释性文字。输出体必须以 { 开头、以 } 结尾。"
        )
    elif "宣称" in error_msg:
        if "将杀" in error_msg:
            hint = (
                f"上一轮输出包含不准确的终结性描述。{error_msg}。"
                "请检查每个节点的走后真值：只有明确写「已形成将杀」的节点才能写将杀/绝杀。"
                "其他节点请改用「压缩空间」「封住逃格」「确立胜势」等描述。只输出合法JSON。"
            )
        else:
            hint = (
                f"上一轮输出包含不准确的战术描述。{error_msg}。"
                "请根据节点信息中的实际走法和状态来调整用词。只输出合法JSON。"
            )
    elif "过短" in error_msg:
        hint = "请补足解说信息量：普通节点至少55字，summary_only节点也要用一句完整地交代机动目的。"
    else:
        hint = "请修改输出以通过校验，只输出合法JSON对象；不要输出visuals字段。"

    return prompt + f"\n\n上一轮输出校验失败: {error_msg}。{hint}"


_SEGMENT_GRAMMAR = (
    'root ::= "{" ws "\\"id\\"" ws ":" ws integer ws "," ws '
    '"\\"sub_endgame\\"" ws ":" ws string ws "," ws '
    '"\\"voiceover\\"" ws ":" ws string ws "," ws '
    '"\\"pacing\\"" ws ":" ws pacing ws "}"\n'
    'pacing ::= "\\"slow\\"" | "\\"normal\\"" | "\\"fast\\"" | "\\"pause_before\\"" | "\\"pause_after\\""\n'
    'integer ::= [0-9]+\n'
    'string ::= "\\"" [^"\\\\x00-\\x1F]* "\\""\n'
    'ws ::= [ \\t\\n]*'
)


def _validate_single_segment(seg: dict, node: dict) -> tuple:
    seg_id = seg.get("id")
    if not isinstance(seg_id, int) or seg_id != node["id"]:
        return False, f"id={seg_id}应为{node['id']}"

    voiceover = seg.get("voiceover")
    if not isinstance(voiceover, str) or not voiceover.strip():
        return False, f"voiceover为空"
    min_len = 28 if node.get("summary_only") else MIN_VOICEOVER_LEN
    if len(voiceover.strip()) < min_len:
        return False, f"voiceover过短({len(voiceover.strip())}<{min_len})"

    pacing = seg.get("pacing", "normal")
    pacing = str(pacing).strip().lower()
    if pacing not in ALLOWED_PACING:
        return False, f"pacing='{pacing}'不合法"

    sub_endgame = seg.get("sub_endgame")
    if not isinstance(sub_endgame, str) or not sub_endgame.strip():
        return False, f"sub_endgame为空"

    text = voiceover.strip()
    _CHECKMATE_BANNED = ("将杀", "绝杀", "杀王", "无路可走", "无路可逃",
                         "死局", "终局已定", "锁定胜局")
    if node.get("is_checkmate_after") is False and any(word in text for word in _CHECKMATE_BANNED):
        return False, "错误宣称将杀"

    allows_check_word = node.get("is_check_after") or node.get("has_check_in_node")
    if not allows_check_word and "将军" in text:
        return False, "错误宣称将军"

    king_moved = node.get("king_moved", False)
    checking_types = node.get("checking_piece_types", [])
    king_claims_check = king_moved and chess.KING not in checking_types
    if king_claims_check and any(word in text for word in ("王将军", "王形成杀", "王绝杀", "王直接", "致命将军")):
        return False, "错误宣称王将军——国际象棋中王不能直接将军"

    if not node.get("is_capture_node") and any(word in text for word in ("吃掉", "兑掉", "吞掉")):
        return False, "错误宣称吃子"

    if node.get("is_game_over_after") and node.get("legal_reply_count_after", 1) == 0:
        if any(word in text for word in ("黑方应", "白方应", "下一步", "随后再")):
            return False, "在终局后继续虚构后续走法"

    _NEUTRALITY_BANNED = ("双方等待", "局势平衡", "互相试探", "积蓄力量", "均势", "双方都在")
    if any(word in text for word in _NEUTRALITY_BANNED):
        return False, "含有均势叙事词——这是必胜残局变现，必须从强方主导推进角度写"

    if node.get("summary_only") and len(text) > 120:
        return False, f"概括模式节点过长({len(text)}>120)"

    return True, ""


def _build_segment_repair_prompt(node: dict, error_msg: str) -> str:
    parts = [
        "你是专业的国际象棋残局教练。只输出一个合法JSON对象，不要任何解释。",
        "",
        f"节点id={node['id']}，需要修复。",
        f"【当前残局】{node.get('sub_endgame_name', '残局')}",
        f"走法: {node.get('moves', '?')} ({node.get('move_count', 0)}着)",
        f"走后状态: {'已将杀' if node.get('is_checkmate_after') else '将军' if node.get('is_check_after') else '非将军'}",
        f"含吃子: {'是' if node.get('is_capture_node') else '否'}",
        f"叙事权限: {node.get('claim_level', 'positioning')}",
    ]
    if node.get("summary_only"):
        parts.append("概括模式: 是 ← 必须只用1句话概括")
    parts.extend([
        "",
        f"校验失败原因: {error_msg}",
        "请修复后只输出: {\"id\": int, \"sub_endgame\": \"string\", \"voiceover\": \"string\", \"pacing\": \"normal|slow|fast|pause_before|pause_after\"}",
    ])
    return "\n".join(parts)


def _split_fallback_text(text: str, chunk_nodes: list) -> dict:
    parts = re.split(r"第\s*(\d+)\s*步[：:\s]*", text)
    result = {}
    for i in range(1, len(parts), 2):
        try:
            step_id = int(parts[i])
        except ValueError:
            continue
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        result[step_id] = content
    return result


def _generate_chunk_fallback(prompt: str) -> str:
    try:
        backend = create_backend_from_env()
        return _strip_thinking(backend.generate(prompt))
    except Exception as e:
        Logger.warn(f"  fallback generate 异常: {type(e).__name__}: {e}")
    return ""


def _repair_failed_segments(backend, segments: list, chunk_nodes: list) -> Optional[dict]:
    node_by_id = {node["id"]: node for node in chunk_nodes}
    repaired_any = False
    for i, seg in enumerate(segments):
        node = chunk_nodes[i]
        ok, err = _validate_single_segment(seg, node)
        if ok:
            continue

        original_vo = seg.get("voiceover", "")
        fixed_vo = _auto_fix_voiceover(original_vo, node)
        if fixed_vo != original_vo:
            seg["voiceover"] = fixed_vo
            if _validate_single_segment(seg, node)[0]:
                repaired_any = True
                continue

        prompt = _build_segment_repair_prompt(node, err)
        raw = backend.generate(prompt, grammar=_SEGMENT_GRAMMAR)
        if not raw:
            Logger.warn(f"    单段修复 id={node['id']} 生成空结果")
            continue
        repaired_seg = _parse_single_segment(raw)
        if repaired_seg is None:
            continue
        repaired_vo = repaired_seg.get("voiceover", "")
        repaired_seg["voiceover"] = _auto_fix_voiceover(repaired_vo, node)
        repaired_ok, _ = _validate_single_segment(repaired_seg, node)
        if repaired_ok:
            segments[i] = repaired_seg
            repaired_any = True

    if repaired_any:
        return {"segments": segments}
    return None


def _parse_single_segment(raw_text: str) -> Optional[dict]:
    t = raw_text.strip()
    brace_start = t.find("{")
    brace_end = t.rfind("}")
    if brace_start == -1 or brace_end <= brace_start:
        return None
    try:
        obj = json.loads(t[brace_start:brace_end + 1])
        if isinstance(obj, dict) and "id" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def generate_structured(board: chess.Board, storyboard: dict) -> GeneratedCommentary:
    nodes = storyboard.get("nodes", [])
    commentary = GeneratedCommentary()

    if not nodes:
        Logger.warn("分镜数据为空，无法生成解说")
        return commentary

    backend = create_backend_from_env()
    commentary.backend = backend.name

    node_count = len(nodes)
    total_chunks = max(1, (node_count + CHUNK_SIZE - 1) // CHUNK_SIZE)
    Logger.info(f"结构化生成: {node_count} 节点 → {total_chunks} 块 (后端:{backend.name})")

    json_header = _build_json_header(storyboard)
    text_header = _build_header(storyboard)
    text_example = _get_example(storyboard.get("endgame_name", "残局"))

    all_segments = []
    commentary.chunks_total = total_chunks

    for chunk_idx in range(total_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, node_count)
        chunk_nodes = nodes[start:end]
        prev_context = _build_prev_context(nodes[start - 1]) if start > 0 else ""
        prev_tail_node = nodes[start - 1] if start > 0 else None

        json_prompt = _build_chunk_prompt(
            json_header, chunk_nodes, chunk_idx, total_chunks,
            _JSON_EXAMPLE if chunk_idx == 0 else "",
            prev_context=prev_context,
        )
        chunk_grammar = _build_chunk_grammar(len(chunk_nodes))
        Logger.info(f"  [{chunk_idx + 1}/{total_chunks}] 节点{chunk_nodes[0]['id']}-{chunk_nodes[-1]['id']}"
                    f"{' [grammar]' if chunk_grammar else ''}")

        success = False
        err_msg = "首次尝试失败"
        for attempt in range(MAX_RETRIES + 1):
            if attempt == 0:
                prompt = json_prompt
            else:
                prompt = _build_retry_prompt(json_prompt, err_msg, attempt)
                commentary.retries_total += 1

            raw_text = _strip_thinking(backend.generate(prompt, grammar=chunk_grammar))
            if not raw_text:
                err_msg = "生成空结果"
                Logger.warn(f"  块{chunk_idx + 1}生成空结果")
                continue

            data = _parse_storyboard_json(raw_text)
            if data is _INVALID_JSON_SENTINEL:
                err_msg = "输出不是合法JSON"
                Logger.warn(f"  块{chunk_idx + 1}尝试{attempt} JSON解析失败")
                continue

            ok, err_msg = _validate_storyboard_chunk(data, chunk_nodes)
            if ok:
                chunk_segments = _finalize_chunk_segments(backend, data, chunk_nodes, prev_tail_node)
                all_segments.extend(chunk_segments)
                commentary.chunks_succeeded += 1
                Logger.info(f"  块{chunk_idx + 1}: {len(chunk_segments)} 段")
                success = True
                break

            Logger.warn(f"  块{chunk_idx + 1}尝试{attempt} 校验失败: {err_msg}")

            segments = data.get("segments")
            if isinstance(segments, list) and len(segments) == len(chunk_nodes):
                auto_fixed_any = False
                for si, seg in enumerate(segments):
                    node = chunk_nodes[si]
                    seg_ok, _ = _validate_single_segment(seg, node)
                    if seg_ok:
                        continue
                    original_vo = seg.get("voiceover", "")
                    fixed_vo = _auto_fix_voiceover(original_vo, node)
                    if fixed_vo != original_vo:
                        seg["voiceover"] = fixed_vo
                        auto_fixed_any = True

                if auto_fixed_any:
                    auto_ok, auto_err = _validate_storyboard_chunk({"segments": segments}, chunk_nodes)
                    if auto_ok:
                        chunk_segments = _finalize_chunk_segments(backend, segments, chunk_nodes, prev_tail_node)
                        all_segments.extend(chunk_segments)
                        commentary.chunks_succeeded += 1
                        Logger.info(f"  块{chunk_idx + 1}: {len(chunk_segments)} 段 (自动修复)")
                        success = True
                        break
                    Logger.warn(f"  块{chunk_idx + 1}自动修复后仍不通过: {auto_err}")

                repaired = _repair_failed_segments(backend, segments, chunk_nodes)
                if repaired is not None:
                    repaired_ok, repaired_err = _validate_storyboard_chunk(repaired, chunk_nodes)
                    if repaired_ok:
                        chunk_segments = _finalize_chunk_segments(backend, repaired, chunk_nodes, prev_tail_node)
                        all_segments.extend(chunk_segments)
                        commentary.chunks_succeeded += 1
                        Logger.info(f"  块{chunk_idx + 1}: {len(chunk_segments)} 段 (单段修复)")
                        success = True
                        break
                    else:
                        Logger.warn(f"  块{chunk_idx + 1}单段修复仍失败: {repaired_err}")

        if not success:
            Logger.warn(f"  块{chunk_idx + 1}结构化生成失败，回退文本模式")
            commentary.fallback_used = True
            text_prompt = _build_chunk_prompt(
                text_header, chunk_nodes, chunk_idx, total_chunks,
                text_example if chunk_idx == 0 else "",
                prev_context=prev_context,
            )
            text_output = _generate_chunk_fallback(text_prompt)
            fallback_parts = _split_fallback_text(text_output, chunk_nodes) if text_output else {}

            chunk_segments = []
            for node in chunk_nodes:
                nid = node["id"]
                if nid in fallback_parts:
                    voice = fallback_parts[nid]
                elif text_output:
                    voice = text_output[:MAX_CHARS] if nid == chunk_nodes[0]["id"] else node.get("transition_summary", f"第{nid}步")
                else:
                    voice = node.get("transition_summary", f"第{nid}步（解说生成失败）")
                chunk_segments.append(StoryboardSegment(
                    id=nid,
                    sub_endgame=node.get("sub_endgame_name", ""),
                    voiceover=voice,
                    pacing=normalize_pacing(node.get("suggested_pacing", "normal")),
                    visuals=_build_visuals_from_node(node),
                ))
            chunk_segments = _apply_chunk_bridges(chunk_segments, chunk_nodes, prev_tail_node)
            all_segments.extend(chunk_segments)

    if all_segments and not commentary.fallback_used:
        last_node = nodes[-1] if nodes else {}
        last_seg = all_segments[-1]
        is_terminal = last_node.get("is_checkmate_after") or last_node.get("claim_level") == "terminal"
        has_conclusion = any(w in last_seg.voiceover for w in ("将杀", "绝杀", "胜负已定", "胜势兑现", "终局", "结束"))
        if is_terminal and not has_conclusion:
            winner = storyboard.get("winning_side", "白方")
            conclusion = f"。至此{winner}形成将杀，胜负已定。"
            last_seg.voiceover = last_seg.voiceover.rstrip("。") + conclusion

    commentary.segments = all_segments
    commentary.raw_text = "\n".join(
        f"第{seg.id}步：{seg.voiceover}" for seg in all_segments
    )

    status = "结构化完成" if not commentary.fallback_used else f"部分回退(成功{commentary.chunks_succeeded}/{total_chunks})"
    Logger.success(f"解说生成: {len(all_segments)} 段 ({status}, 重试{commentary.retries_total}次)")
    return commentary


def generate(board: chess.Board, storyboard: dict) -> str:
    """兼容包装：优先使用结构化生成，失败则走旧纯文本链路"""
    try:
        structured = generate_structured(board, storyboard)
        if structured.segments:
            return structured.raw_text
    except Exception as e:
        Logger.warn(f"结构化生成异常，回退旧文本链路: {e}")

    nodes = storyboard.get("nodes", [])
    if not nodes:
        Logger.warn("分镜数据为空，无法生成解说")
        return ""

    node_count = len(nodes)
    total_chunks = max(1, (node_count + CHUNK_SIZE - 1) // CHUNK_SIZE)
    backend = create_backend_from_env()
    Logger.info(f"生成解说: {node_count} 节点 → {total_chunks} 块 (后端:{backend.name})")

    header = _build_header(storyboard)
    example = _get_example(storyboard.get("endgame_name", "残局"))

    all_parts = []
    for chunk_idx in range(total_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, node_count)
        chunk_nodes = nodes[start:end]

        prompt = _build_chunk_prompt(header, chunk_nodes, chunk_idx, total_chunks, example if chunk_idx == 0 else "")
        Logger.info(f"  [{chunk_idx + 1}/{total_chunks}] 节点{chunk_nodes[0]['id']}-{chunk_nodes[-1]['id']} (提示词{len(prompt)}字)")

        result = _generate_chunk_fallback(prompt)
        if not result:
            Logger.warn(f"  块{chunk_idx + 1}失败，跳过")
            continue

        if len(result) > MAX_CHARS:
            Logger.warn(f"  块{chunk_idx + 1}过长({len(result)}>{MAX_CHARS})，限长重试")
            prompt_short = prompt + f"\n\n每步尽量写成2-4句，但总共不超过{MAX_CHARS//2}字。重新输出。"
            retry = _generate_chunk_fallback(prompt_short)
            if retry:
                result = retry
            else:
                result = result[:MAX_CHARS]

        all_parts.append(result)
        Logger.info(f"  块{chunk_idx + 1}: {len(result)} 字")

    if not all_parts:
        Logger.error("所有块生成均失败")
        return ""

    final = "\n".join(all_parts).strip()
    Logger.success(f"解说生成完成 ({len(final)} 字符)")
    return final
