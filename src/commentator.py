from src.common import GeneratedCommentary, ALLOWED_PACING
from src.common import Logger, StoryboardSegment
from src.common import normalize_pacing
from src.llm_backend import create_backend_from_env
from typing import Optional
import chess
import json
import re

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

_JSON_EXAMPLE = """{"segments":[{"id":1,"sub_endgame":"车兵对车","voiceover":"白车退回底线后方建立菲利多防线，关键是让白王稳稳守住兵前那一格，车则在身后控制住整条直线。这样一来，黑方就算想从正面压上来，也找不到直接突破的入口。","pacing":"slow"},{"id":2,"sub_endgame":"车兵对车","voiceover":"承接刚搭好的防线，黑王一步步向白兵逼近，想把白王从兵前的关键格挤开。这里的重点不是马上制造战术，而是用王不断前压，试探防线会不会出现松动。","pacing":"normal"},{"id":3,"sub_endgame":"车兵对车","voiceover":"顺着前面对防线的持续施压，黑车突然绕到侧翼发难，对白方王的位置形成更直接的骚扰。白王一旦被迫离开兵前那条线，原本稳固的菲利多防线就会裂开一道口子，局面也随之进入真正的转折。","pacing":"slow"}]}"""

def _get_example(endgame_name: str) -> str:
    return _EXAMPLE_BY_ENDGAME.get(endgame_name, _EXAMPLE_FALLBACK)

def _build_header(storyboard: dict) -> str:
    endgame_name = storyboard.get("endgame_name", "残局")
    role_summary = storyboard.get("role_summary", "")
    hard_constraints = storyboard.get("hard_constraints", [])
    has_switch = storyboard.get("has_sub_endgame_switch", False)

    node_count = len(storyboard.get("nodes", []))

    parts = [
        "你是会自己看棋的国际象棋教练。你不需要被告知哪一步重要——你会从棋理事实中自己判断。"
        "请输出专业、以棋理为核心的中文解说，纯文本输出。",
        "只依据给定走法和局面信息解说，禁止虚构剧情，禁止空泛比喻。",
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


def _extract_terminology(storyboard: dict) -> list:
    """从知识库 motifs 提取本残局的规范专业术语名（冒号前的部分）。

    motifs 形如「盒子法：用车画线限制对方王的活动范围」，冒号前即术语名。
    把这些作为「可用术语」正向注入 prompt——激发模型用对行话，而不只是
    黑名单禁止。术语来自知识库，天然与当前残局类型匹配，不会张冠李戴。
    去重保序，最多取 6 个，避免 prompt 过长。
    """
    terms = []
    for motif in storyboard.get("motifs", []) or []:
        if not isinstance(motif, str):
            continue
        name = motif.split("：")[0].split(":")[0].strip()
        if name and name not in terms:
            terms.append(name)
    return terms[:6]


def _rate_difficulty(endgame_name: str) -> str:
    """按残局类型的公认难度分级，用于调整解说调性。

    只影响人设语气，不影响内容深度与校验——基础残局也要讲透。
    仅对知识库已知类型断言难度；未匹配类型返回 unknown（中性调性），
    避免把实际很难的非常见残局误标成「基础残局」而让模型欠讲解。
    """
    if endgame_name in ("象马杀王", "车兵对车"):
        return "hard"
    if endgame_name in ("双象杀王", "单兵残局"):
        return "medium"
    if endgame_name in ("单车杀王", "单后杀王"):
        return "basic"
    return "unknown"


_DIFFICULTY_TONE = {
    "hard": "这是公认很难的残局，很多人学多年都掌握不好。请把每一步的「为什么」讲透，让人真正理解棋理而不是死记走法。",
    "medium": "这类残局有一定难度。请言简意赅、直击要点，把每步背后的棋理逻辑讲清楚。",
    "basic": "这是基础残局。请把看似简单的走法讲出层次感，让人理解每一步都是取胜链条上必不可少的一环。",
    "unknown": "请把每一步的取胜逻辑讲清楚，让人理解每一步在整个取胜过程中的作用。",
}


def _build_json_header(storyboard: dict) -> str:
    endgame_name = storyboard.get("endgame_name", "残局")
    role_summary = storyboard.get("role_summary", "")
    hard_constraints = storyboard.get("hard_constraints", [])
    winning_side = storyboard.get("winning_side", "")
    losing_side = storyboard.get("losing_side", "")
    target_length = storyboard.get("target_length", "")
    node_count = len(storyboard.get("nodes", []))

    parts = [
        "你是一位会自己看棋的国际象棋教练。你不是在复述走法，而是在分析每一步背后的棋理。"
        "只输出合法JSON，不加任何解释或markdown标记。",
        "",
        "你的讲解信条：",
        "- 当节点标注了「关键手判定」时，请围绕判定依据把这一步为什么关键讲深讲透；"
        "未标注时，从棋理事实中自己判断重要程度。",
        "- 当你在「棋理分析」或「引擎数据」中看到一着同时做了多件事、让对方无法两全、"
        "或改变了残局结构时，你会自然地讲出它为什么是全局的胜负手。",
        "- 你的判断来自对棋局结构的理解，而不是对指令的服从。",
        "- 把每段残局讲成一个有逻辑的推进故事：先讲清这一步做了什么、局面因此发生了什么变化，再讲它为什么有用。",
        "- 关键步骤自己判断、自己写出张力；过渡步骤一笔带过。",
        "- 你没有事实支撑时宁可朴素也不要空洞——没有事实支撑的形容词一个都不要用。",
        _DIFFICULTY_TONE[_rate_difficulty(endgame_name)],
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

    terminology = _extract_terminology(storyboard)
    if terminology:
        parts.extend([
            "",
            f"【可用术语】这类残局的规范术语：{'、'.join(terminology)}。",
            "讲到相关棋理时优先用这些行话（用对地方即可，不必硬凑），但不要解释术语本身的定义。",
        ])

    # 取胜计划骨架：把整盘的阶段次序作为全局锚点交给模型，让每段解说都能挂到
    # 「这是取胜计划的第几步」上，形成贯穿全局的博弈逻辑，而不是各段孤立讲走法。
    phases = storyboard.get("phases", []) or []
    phase_names = [p[0] for p in phases if isinstance(p, (list, tuple)) and p]
    if phase_names:
        parts.extend([
            "",
            "【取胜计划骨架】这类残局的标准取胜次序是：" + " → ".join(phase_names) + "。",
            "每个节点都标了「所处阶段」。讲解时请扣住当前阶段在整个计划里的作用，"
            "让观众明白这一步是为了推进到下一阶段，而不是孤立地描述棋子移动。",
        ])

    parts.extend([
        "",
        "每个节点有 claim_level 控制可用的结论深度：",
        "positioning→只能讲站位和控制  constraining→可以讲空间压缩",
        "forcing→可以讲强制/被迫  terminal→才能说将杀/绝杀",
        "",
        "节点可能标注「将军驱赶」或「反复试探等待」→ 这种节点是多着合并的叙事块，你要用流畅的段落描述这段过程，而不是逐步数着。",
        "节点可能带「棋理事实」「棋理观察」「棋理分析」「引擎数据」→ 这些都是从棋盘或引擎算出的真实事实，不是判决。请你阅读后自己形成判断，用自己的话融进解说。",
        "",
        "【关键：解说要贴合画面的推进过程，不要一上来就报终点】",
        "每段解说在视频里是和这个节点的多步走子「同步播放」的——你写第一句时，画面才刚走第一步；",
        "你写到后半段时，画面才走到这段的最后一步。所以请按「过程」来组织，而不是按「结果总结」：",
        "- 先描述这段开头在做什么（哪个子力先动、想达到什么），再讲随着几步推进局面怎样一点点变化，",
        "  最后才落到这段结束时形成的结果（对方王被逼到哪、空间被压到多小）。",
        "- 不要在开头第一句就直接宣布整段的最终结果（如「对方王已无处可走」），那会和此刻画面里对方王还没动相矛盾；",
        "  把这种终态结论放到这一段的末尾，作为「经过这几步之后」的小结。",
        "- 多着节点里若是某一方的子力在连续调整、而对方王这段几乎没动，就如实讲成「主动方在调整站位、对方暂时只能原地等待」，",
        "  不要凭空说成对方王在被驱赶或四处逃窜。",
        "",
        "【JSON格式】",
        '{"segments":[{"id":int,"sub_endgame":"string","voiceover":"string","pacing":"slow|normal|fast|pause_before|pause_after"},...]}',
        "segments数量必须等于本块节点数。不输出visuals字段。",
        "",
        "【解说要求】",
        f"- 正好{node_count}个segment，不增不减",
    ])
    if target_length:
        parts.append(f"- 全局字数预算：整段解说（所有块加起来）控制在{target_length}。关键节点可多写，过渡节点摘要带过，别平均用力")
    parts.extend([
        "- 用自然的中文解说，避免引擎术语（如评估值、DTM、mate in N）",
        "- 每段50-200字，summary_only的用1句话概括（≤80字）",
        "- 各段之间连续推进，后一段承接前一段已建立的局面",
        "- 最后一段若是terminal权限，以「至此形成将杀」或「至此胜负已定」收束",
        "- 王的描述侧重于「逼近」「封住逃格」「配合主力子压缩空间」等位置性语言",
        "- voiceover用纯中文口播，禁止出现棋盘坐标（如h7、g5）、棋子英文字母、数字和升变记号；",
        "  指位置时改用方位关系：「黑王的右前方」「兵前一格」「同一条斜线」「底线」「边角」「中心方向」等，",
        "  画面里棋盘和箭头已标出精确格子，你的任务是用关系化语言说清这一步改变了什么，而不是报格子名",
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

        # 当前所处的取胜阶段（让每段解说能挂到全局取胜计划上，而不是各讲各的）。
        phase = node.get("phase", "")
        phase_hint = node.get("phase_hint", "")
        if phase:
            stage_line = f"所处阶段: {phase}"
            if phase_hint:
                stage_line += f"（{phase_hint}）"
            parts.append(stage_line)

        # 棋理洞察（由 insight_extractor 用棋盘算出的真实事实，关系化、无坐标）。
        teaching_point = node.get("teaching_point", "")
        if teaching_point:
            parts.append(f"棋理事实: {teaching_point}")
        spatial = node.get("spatial_change", {})
        if spatial and spatial.get("weak_before") is not None:
            wb = spatial.get("weak_before")
            wa = spatial.get("weak_after")
            region = spatial.get("king_region", "")
            extra = f"，对方王已退到{region}" if region in ("边线", "角落") else ""
            parts.append(f"空间数据: 对方王可走的安全格 {wb}→{wa}{extra}")
        must_mention = node.get("must_mention", [])
        if must_mention:
            parts.append(f"棋理观察: {'；'.join(must_mention)}")

        # 战术叙述（新）：纯棋理中文前提，不给结论。LLM 从中自己判断关键手。
        tactical_narratives = node.get("tactical_narratives", [])
        if tactical_narratives:
            parts.append("棋理分析（由棋盘直接算出的战术事实，供你独立判断）:")
            for tn in tactical_narratives:
                parts.append(f"  · {tn}")

        # 引擎信号（新）：量化参考，不是判决。LLM 自己决定是否引用。
        eval_signals = node.get("eval_signals", [])
        if eval_signals:
            parts.append("引擎数据（量化参考，不是判决——是否提及由你判断）:")
            for es in eval_signals:
                parts.append(f"  · {es}")

        # 关键手判定（已由棋盘事实算出，不是要你猜）：把「这步有多关键、为什么关键」
        # 直接交给模型，让它把给定结论讲透，而不是自己去推断后只能堆套话。
        move_importance = node.get("move_importance", "")
        importance_reasons = node.get("importance_reasons", []) or []
        if move_importance == "high":
            parts.append("关键手判定: 这是本残局的关键节点，请重点讲深、讲出张力与必要性。判定依据如下：")
            for r in importance_reasons:
                parts.append(f"  · {r}")
            parts.append("  请围绕这些依据说清「这一步在解决什么问题、为什么非这样走不可」，不要用空泛的赞美词。")
        elif move_importance == "medium" and importance_reasons:
            parts.append("要点提示: 本节点的实质进展在于——" + "；".join(importance_reasons) + "。请落在这些进展上讲，不要泛泛而谈。")

        drive_tag = next((t for t in tags if t in ("将军驱赶", "连续将军驱赶", "反复试探等待")), "")
        if drive_tag:
            parts.append(f"类型: 「{drive_tag}」叙事块 — 这是多着合并，描述整体过程，不要逐步数着")

        # 详略提示：不注入"重要/不重要"的判决，只给事实性提示
        if summary_only or node.get("video_density") == "low":
            parts.append("详略: 过渡/重复节点 — 一句话带过即可，不要展开")
            if not node.get("is_capture_node") and not node.get("has_check_in_node"):
                if node_id % 2 == 0:
                    parts.append("视角提示: 这一句请落在空间变化上（对方王少了哪个方向的去路、活动范围怎么变），别只说「推进」")
                else:
                    parts.append("视角提示: 这一句请落在整体计划上（这步在为哪个目标铺路、和上一步什么关系），别只说「推进」")
        elif node.get("is_critical"):
            # 旧："这是关键转折，请写得更有张力，点出它为什么重要"（结论注入）
            # 新：仅给事实依据，模型自己判断张力级别
            parts.append("详略: 此节点含吃子/将军/评估显著变化等事件")

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


# 反套话替换表：从实际 KBNvK 样本统计出的高频空洞修辞。
# 原则：只清纯修辞，不动有信息量的词。多数直接删除（删掉不影响句意，
# 因为它们本就不承载事实），少数替换成中性词以保句子通顺。
# 删除类用空串，会在 _auto_fix_voiceover 末尾的标点收敛里清掉留下的多余标点。
_CLICHE_PATTERNS = [
    # (正则, 替换) —— 顺序敏感：先长后短
    # 「看似平淡，实则…」家族：AI 最爱的伪悬念开头，不承载任何棋理事实。
    # 删掉前半截后，后半截的实质内容仍能独立成句。
    (r"看似(?:平淡无奇|平平无奇|平淡|不起眼|普通|简单)[，,]?(?:实则|却|其实)?", ""),
    (r"别看这一步[^，。]{0,6}[，,]", ""),
    (r"如同?利剑出鞘", ""),
    (r"如洪水般(不可阻挡)?", "持续"),
    (r"天罗地网", "严密的控制"),
    (r"天衣无缝", "完整"),
    (r"密不透风", "严密"),
    (r"无形的牢笼", "包围"),
    (r"胜利的天平(开始|彻底)?(倾斜)?", "优势"),
    (r"(已如)?囊中之物", "胜势已成"),
    (r"不可阻挡", ""),
    (r"暗藏杀机", ""),
    (r"耐心的围猎", "稳步驱赶"),
    (r"致命一击", "最后一击"),
    # 形容词类修饰：删去后句子仍通顺
    (r"精妙地?", ""),
    (r"精湛地?", ""),
    (r"精准地?", ""),
    (r"精确地?", ""),
    (r"精心(计算|策划)?地?", ""),
    (r"深思熟虑", ""),
    (r"完美(的|地)?", ""),
    (r"愈发默契", "更协调"),
    (r"(配合|协调)(愈发|越来越)默契", "配合更协调"),
    (r"默契(的)?配合", "配合"),
    (r"步步为营", "稳步推进"),
    # 纯过渡凑字尾巴：「为后续/最终…做准备/奠定基础/创造条件/铺平道路」。
    # 这类句尾不承载任何棋理事实，是 AI 最爱的空洞承诺，每次出现都删。
    (r"[，,]?\s*为(?:后续|接下来|下一步|最终|后面|最后)(?:的)?[^，。、！]{0,16}"
     r"(?:做准备|做好准备|奠定[了]?(?:坚实)?基础|创造[了]?[^，。]{0,8}条件|铺平[了]?道路|埋下伏笔)", ""),
]


def _reduce_cliches(text: str) -> str:
    """删减空洞套话/重复比喻。不改变事实性内容，只去修辞。"""
    out = text
    for pat, repl in _CLICHE_PATTERNS:
        out = re.sub(pat, repl, out)
    return out


# 坐标兜底清洗：prompt 已要求"禁坐标"，但 LLM 偶尔仍会吐出 e8/f8 这类格子名。
# 坐标一旦混进 TTS 会被逐字母念（"e-eight"），非常刺耳，所以定稿前用正则强制清除。
# 三层策略，从精到糙：
#   1) 白名单移动动词+坐标 → 整体收成方位动词（最自然，直接丢掉坐标）；
#   2) 通用「介词+坐标」→「介词+那一格」（动词不在白名单时，保住前面的动词不被截断）；
#   3) catch-all 清掉任何残留的孤立坐标。
_COORD = r"[a-h][1-8]"
_MOVE_TO_COORD = [
    (re.compile(rf"(?:被迫)?(?:退守|退回到|退回|退到|退至|后撤到|撤回到|撤到)\s*{_COORD}\s*格?"), "后退"),
    (re.compile(rf"(?:移到|移至|走到|走向|来到|落到|落在|停在|占据)\s*{_COORD}\s*格?"), "就位"),
    (re.compile(rf"(?:跳到|跳向|跳至|跳上|跃到|跃向)\s*{_COORD}\s*格?"), "跳出"),
    (re.compile(rf"(?:切入到?|进到|进至|挺进到|推进到|杀到)\s*{_COORD}\s*格?"), "切入"),
]
# 通用介词：动词未被白名单覆盖时（如"逼到f8""压向a7"），把坐标换成"那一格"，
# 让前置动词与句子结构完整保留，避免 catch-all 把动词截成残句。
_COORD_PREP = re.compile(rf"(?<=[一-鿿])(到|至|向|在|于)\s*{_COORD}\s*格?")
# catch-all：清除剩余的孤立坐标，连同可能的前导介词与"格"后缀一起吃掉。
_COORD_CATCHALL = re.compile(rf"(?:从|由|到|至|向|于|在|经)?\s*{_COORD}\s*格?")


def _strip_coordinates(text: str) -> str:
    """清除 voiceover 中泄漏的棋盘坐标，防止进入 TTS 被逐字母念读。"""
    out = text
    for pat, repl in _MOVE_TO_COORD:
        out = pat.sub(repl, out)
    out = _COORD_PREP.sub(r"\1那一格", out)
    out = _COORD_CATCHALL.sub("", out)
    return out


# 跨段去重：chunk 之间各自独立生成、LLM 看不到全局，导致同一句套话
# （"逐步收紧包围圈""围绕对王争夺关键格"…）在多段里反复出现。这里在所有
# segment 汇总后做一次全局扫描：每个短语家族首次出现保留原文，第二次及以后
# 轮换成同义变体，保住语义、消除字面复读感。变体本身也要无坐标、不空洞。
_REPEAT_FAMILIES = [
    (re.compile(r"逐步收紧包围圈|不断收紧包围圈|收紧包围圈"),
     ["把包围圈又收小一圈", "进一步缩小对方王的活动范围", "继续收网", "再压掉一块活动空间"]),
    (re.compile(r"围绕对王(?:来回|反复)?调整[，、]?\s*争夺关键格|围绕对王(?:来回|反复)调整"),
     ["贴着对方王不断换位、卡住要害格", "在关键格上与对方王反复周旋", "一格一格地抢占对方王身边的要点"]),
    (re.compile(r"等待最佳时机完成最后一击|等待最佳时机|等待[^，。]{0,6}最后一击"),
     ["伺机收官", "只待最后一着到位", "等收官的时机成熟"]),
    (re.compile(r"为(?:下一步|后续)的?致命打击做准备|为致命一击蓄势"),
     ["为收官铺路", "把收杀的条件一点点凑齐"]),
    (re.compile(r"只能被动应对|只能被动防守|被动应对"),
     ["几乎没有还手余地", "走一步看一步，毫无主动权", "只能跟着白方的节奏走"]),
    (re.compile(r"逐步压缩(?:其|对方王的?)?(?:活动)?空间|不断压缩(?:其|对方)?(?:活动)?空间"),
     ["把对方王能落脚的格子越夺越少", "活动范围被一截截切掉", "腾挪余地越来越小"]),
]


def _dedupe_across_segments(segments) -> None:
    """对已汇总的 StoryboardSegment 列表原地去重高频套话短语。

    首次命中保留原文；之后每次命中按家族轮换替换为同义变体。失败安全：
    任何异常都跳过该家族，不影响解说主体。
    """
    if not segments:
        return
    for pat, variants in _REPEAT_FAMILIES:
        if not variants:
            continue
        hit = 0
        for seg in segments:
            vo = getattr(seg, "voiceover", "") or ""
            if not vo:
                continue

            # 逐次替换：每个命中都计数，首个全局命中保留原文，其余按家族轮换变体
            def _sub_one(text):
                nonlocal hit
                out_parts = []
                last = 0
                for m in pat.finditer(text):
                    out_parts.append(text[last:m.start()])
                    if hit == 0:
                        out_parts.append(m.group(0))  # 首次保留
                    else:
                        out_parts.append(variants[(hit - 1) % len(variants)])
                    hit += 1
                    last = m.end()
                out_parts.append(text[last:])
                return "".join(out_parts)

            try:
                seg.voiceover = _sub_one(vo)
            except Exception:
                continue


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

    # 必胜残局术语纠错：断言「和棋已成」是结果误判（本工具只处理必胜局）。
    # 仅纠正明确宣告和棋结果的多字短语；裸词「逼和/和棋」不动，避免误伤
    # 合法的教学提醒（如「避免逼和」「谨防和棋」）与弱方失败的谋和尝试。
    _DRAW_RESULT_WORDS = (
        "成功逼和", "逼和成功", "守和成功", "成功守和",
        "顺利成和", "已经成和", "和棋已定", "和棋收场", "谋和成功",
    )
    for w in _DRAW_RESULT_WORDS:
        fixed = fixed.replace(w, "优势在握")

    if not node.get("is_capture_node"):
        fixed = fixed.replace("吃掉", "控制")
        fixed = fixed.replace("吃子", "控制子力")
        fixed = fixed.replace("吃掉了", "控制了")
        fixed = fixed.replace("兑掉", "交换").replace("兑子", "交换子力")
        fixed = fixed.replace("吞掉", "占据")

    if node.get("is_game_over_after") and node.get("legal_reply_count_after", 1) == 0:
        for w in ("黑方应将", "白方应将", "黑方应对", "白方应对"):
            fixed = fixed.replace(w, "")
        fixed = re.sub(r"(?:黑方|白方)应[，,]?\s*", "", fixed)
        fixed = re.sub(r"(?:下一步|随后再)[^，。,]{0,8}(?:，|,|\s*)", "", fixed)

    # 反套话：删减空洞修辞（放在标点收敛之前，让删除留下的多余标点被一并清掉）
    fixed = _reduce_cliches(fixed)

    # 坐标兜底：清除 prompt 未能压住的泄漏坐标（须在依赖坐标的"叫杀"规则之后）
    fixed = _strip_coordinates(fixed)

    fixed = re.sub(r"[，,]{2,}", "，", fixed)
    fixed = re.sub(r"。{2,}", "。", fixed)
    fixed = re.sub(r"\s{2,}", " ", fixed)
    fixed = re.sub(r"[，、]+。", "。", fixed)
    fixed = re.sub(r"^[，、。]+", "", fixed)
    fixed = fixed.strip()

    return fixed


def _validate_storyboard_chunk(data: dict, chunk_nodes: list) -> tuple:
    segments = data.get("segments")
    if not isinstance(segments, list):
        return False, "顶层缺少segments数组"
    if len(segments) != len(chunk_nodes):
        return False, f"segments数量{len(segments)}与节点数{len(chunk_nodes)}不一致"

    # 逐段校验复用 _validate_single_segment（单一事实来源），整块通过才算通过。
    # 错误信息加 segment[i] 前缀，保留 _build_retry_prompt 依赖的关键词（宣称/过短等）。
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            return False, f"第{i+1}个segment不是对象"
        ok, err = _validate_single_segment(seg, chunk_nodes[i])
        if not ok:
            return False, f"segment[{i}]{err}"

    return True, ""


def _dict_to_storyboard_segments(data: dict, chunk_nodes: list) -> list:
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


def _finalize_chunk_segments(data_or_segments, chunk_nodes: list):
    data = data_or_segments if isinstance(data_or_segments, dict) else {"segments": data_or_segments}
    return _dict_to_storyboard_segments(data, chunk_nodes)


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


def _build_puzzle_chunk_grammar(n_segments: int) -> str:
    """Puzzle 专用收紧语法：voiceover 在采样阶段就只允许中文字符与中文标点，
    从根本上杜绝 Markdown 符号（* # ` 等）、英文字母、阿拉伯数字混入口播稿。

    与 _build_chunk_grammar 的唯一区别：voiceover 用 cnstring 取代通用 string；
    sub_endgame 固定为空串。其余结构完全一致，残局链路不受影响。
    """
    if n_segments <= 0:
        return ""
    seg_repeat = "segment" + "".join(' ws "," ws segment' for _ in range(n_segments - 1))
    # cnchar 允许：CJK 统一表意文字 + 常用中文标点（含中文空格）。
    # 不含 ASCII 字母/数字/* # ` _ [ ] 等，故模型无法采样出 Markdown 或英文。
    cn_punct = "，。、；：？！…—·「」『』（）《》〈〉“”‘’　"
    return (
        'root ::= ws think? "{" ws "\\"segments\\"" ws ":" ws "[" ws ' + seg_repeat + ' ws "]" ws "}"\n'
        'think ::= "<think>" thinkchar* "</think>" ws\n'
        'thinkchar ::= [^<]\n'
        'segment ::= "{" ws "\\"id\\"" ws ":" ws integer ws "," ws '
        '"\\"sub_endgame\\"" ws ":" ws "\\"\\"" ws "," ws '
        '"\\"voiceover\\"" ws ":" ws cnstring ws "," ws '
        '"\\"pacing\\"" ws ":" ws pacing ws "}"\n'
        'pacing ::= "\\"slow\\"" | "\\"normal\\"" | "\\"fast\\"" | "\\"pause_before\\"" | "\\"pause_after\\""\n'
        'integer ::= [0-9]+\n'
        'cnstring ::= "\\"" cnchar* "\\""\n'
        'cnchar ::= [\\u4e00-\\u9fff' + cn_punct + ']\n'
        'ws ::= [ \\t\\n]*'
    )


# 纯中文自由文本语法（无 JSON 包裹）：供润色器等单串生成使用，
# 同样在采样阶段禁止 Markdown / 英文 / 数字。
_PUZZLE_PLAIN_CN_GRAMMAR = (
    'root ::= think? cnchar+\n'
    'think ::= "<think>" thinkchar* "</think>" ws\n'
    'thinkchar ::= [^<]\n'
    'cnchar ::= [\\u4e00-\\u9fff，。、；：？！…—·「」『』（）《》〈〉“”‘’]\n'
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

# 总结词专用语法：token 级只允许中文 + 中文标点，物理上无法吐出英文思维链/
# 棋谱记号/数字/符号，从根上杜绝「标点汤喂 ChatTTS 念崩」。2-3 句。
_SUMMARY_GRAMMAR = (
    'root ::= sentence sentence sentence?\n'
    'sentence ::= cjk (sep cjk)* end\n'
    'cjk ::= han+\n'
    'han ::= [\\u4e00-\\u9fff]\n'
    'sep ::= "，" | "、"\n'
    'end ::= "。" | "！" | "？"'
)


def _validate_single_segment(seg: dict, node: dict) -> tuple:
    seg_id = seg.get("id")
    if not isinstance(seg_id, int):
        return False, f"id={seg_id}不是有效整数"
    if seg_id != node["id"]:
        seg["id"] = node["id"]  # 自动修正

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
    except Exception:
        pass
    return ""


def _repair_failed_segments(backend, segments: list, chunk_nodes: list) -> Optional[dict]:
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


def _clean_cjk_text(text: str) -> str:
    """白名单清洗：只保留中文字符 + 常用中文标点，其余一律剔除。

    总结词、开场白等所有喂 TTS 的文本共用此清洗，避免白名单逻辑重复。
    """
    t = text.strip()
    t = re.sub(r"[^一-鿿，。、！？]", "", t)
    t = re.sub(r"[，、]{2,}", "，", t)
    t = re.sub(r"。{2,}", "。", t)
    t = re.sub(r"[，、！？]+。", "。", t)
    t = re.sub(r"^[，、！？。]+", "", t)
    return t.strip()


def _clean_summary_text(text: str) -> str:
    """清洗总结词：先去引号与「总结」前缀，再走公共 CJK 白名单清洗。

    前缀统一由 generate_summary 末尾补回，保证不重复也不残缺。
    """
    t = text.strip().strip("「」\"'`").strip()
    t = re.sub(r"^总结(一下)?[，,：:]?", "", t).strip()
    return _clean_cjk_text(t)


def _has_forbidden_chars(text: str) -> bool:
    """是否仍含字母/数字（清洗失败的标志）。"""
    return bool(re.search(r"[A-Za-z0-9]", text))


# 只放「描述任务 / 对模型说话」的元指令措辞——真正的总结绝不会出现这些。
# 切忌放领域词（核心技法 / 常见错误 / 取胜方 / 残局类型…）：那些是合格总结
# 本来就会用到的词（提示词也用它们当字段标签），放进来会把好总结误判成泄漏。
# 「提示词被原样复述」这种泄漏交给 _looks_like_prompt_echo 做语义级重合度检测，
# 那个对提示词如何改写都自适应，不必在这里逐词追加（黑名单永远追不完）。
_SUMMARY_META_MARKERS = (
    "节点", "残局局面分析", "thinking", "voiceover", "segment",
    "将杀绝杀", "承接关系", "复述提示词", "推进性描述",
    "用户要求", "需要扮演", "你是国际象棋", "请写", "要求写",
    "字总结", "对着镜头", "对镜头", "收尾总结", "做收尾", "镜头做",
    "纯口语", "不要标题", "不要逐步", "禁止出现", "禁止引擎", "绝对禁止",
    # Qwen 思维链泄漏特征（无 GBNF 语法时 <think> 已被 _strip_thinking 删掉，
    # 这里是兜底：万一 <think> 以变体形式泄漏或被改写成中文描述）：
    "思考过程", "思维链", "思考如下", "指令要求", "根据指令",
    "根据要求", "不会包含", "直接输出", "按照指令", "我理解",
)


def _looks_like_prompt_echo(text: str, prompt: str) -> bool:
    """检测 text 是否在「复述提示词」：按字符 4-gram 算 text 落在 prompt 里的比例。

    元指令词黑名单只能挡住列举过的措辞，提示词换一种写法就失效。改用与提示词
    本身的重合度判断：真总结是模型新写的内容，与提示词重合度低；把提示词指令
    当输出念回来则高度重合。对提示词怎么改写都自适应，无需逐词维护。
    阈值 0.5：超过一半的 4-gram 都来自提示词，几乎可以肯定是复述。
    """
    if not text or not prompt:
        return False
    grams = {prompt[i:i + 4] for i in range(len(prompt) - 3)}
    if not grams:
        return False
    span = len(text) - 3
    if span <= 0:
        return False
    hit = sum(1 for i in range(span) if text[i:i + 4] in grams)
    return hit / span > 0.5


def _summary_is_bad(text: str) -> bool:
    """总结词是否不可用（触发纯中文模板兜底）。

    比旧的「仅查字母数字」更强：长度异常、中文占比过低、含元指令碎片
    任一命中即判废。应对 LLM 泄漏思维链或复读提示词后清洗仍残留的情况。
    注意：「提示词被整段复述」由 _looks_like_prompt_echo 单独检测（需要 prompt），
    这里只查不依赖上下文的硬特征。
    """
    if not text or len(text) < 12 or len(text) > 160:
        return True
    if _has_forbidden_chars(text):
        return True
    cjk = len(re.findall(r"[一-鿿]", text))
    if cjk == 0 or cjk / max(len(text), 1) < 0.8:
        return True
    if any(marker in text for marker in _SUMMARY_META_MARKERS):
        return True
    return False


def _winning_path_phrases(storyboard: dict) -> list:
    """从节点的 position_goal 序列归纳「取胜路线」短语（全局去重保序）。

    这是不依赖知识库(kb)的结构化素材：无论 match_endgame 是否命中，节点都带
    position_goal，因此总能产出取胜逻辑骨架，避免 kb 未命中时总结无米下锅。
    """
    phrases = []
    for n in storyboard.get("nodes", []):
        g = n.get("position_goal", "")
        ph = _goal_to_narrative_phrase(g) if g else ""
        if ph and ph not in phrases:
            phrases.append(ph)
    return phrases


def _build_recap_from_segments(segments, max_parts: int = 5, per_len: int = 46) -> str:
    """把刚生成的分段解说浓缩成「讲解回顾」素材，喂给总结模型。

    这是最贴切的总结依据——基于「这盘实际是怎么赢的」提炼，而不是基于贫乏的
    kb 元数据凭空写。均匀采样若干段、每段取首句并截断，控制长度。失败安全。
    """
    if not segments:
        return ""
    vos = []
    for seg in segments:
        vo = (getattr(seg, "voiceover", "") or "").strip()
        if vo:
            vos.append(vo)
    if not vos:
        return ""
    if len(vos) > max_parts:
        idxs = sorted(set(
            round(i * (len(vos) - 1) / (max_parts - 1)) for i in range(max_parts)
        ))
        vos = [vos[i] for i in idxs]
    parts = []
    for vo in vos:
        first = re.split(r"[。！？]", vo)[0].strip()
        if first:
            parts.append(first[:per_len])
    return "；".join(parts)


def _derive_strategy(storyboard: dict) -> str:
    """KB 未命中时，从子力构成推导一句简洁的取胜策略。

    只根据强方拥有的后/车/兵类型给出通用原则，不追求穷举组合——
    覆盖不到的退空字符串，由调用方兜底。
    """
    strong = storyboard.get("strong_material", "") or ""
    weak = storyboard.get("weak_material", "") or ""

    if not strong:
        return ""

    has_queen = "后" in strong
    has_rook = "车" in strong
    has_pawn = "兵" in strong
    weak_solo_king = weak == "单王"
    weak_has_rook = "车" in weak

    # 后有压倒性优势
    if has_queen:
        if weak_solo_king:
            return ("后的活动范围极广，核心是用己方王配合后逐步把对方王逼向边线，"
                    "全程保持安全距离避免逼和")
        if weak_has_rook:
            return ("后对车的优势是全方位的——活动范围和威胁方向都远超单车，"
                    "关键是连续将军配合王的推进，逐步压缩对方王的空间")
        return ("后对轻子的优势是全方位的，"
                "核心是避免被对方兑掉后，用后和王配合逐步压缩对方空间")

    # 车兵残局
    if has_rook and has_pawn:
        return "取胜的关键是用车护送兵推进到底线升变，同时切断对方王的回防路线"

    # 单车优势
    if has_rook:
        if weak_solo_king:
            return ""  # 单车杀王已被 KB 覆盖
        if weak_has_rook:
            return "多一车等于多一条控制线，避免兑车简化、用多出来的车收紧包围"
        return "车对轻子的优势在于控制力更强，用车封住关键线路，配合王步步紧逼"

    # 有兵优势
    if has_pawn:
        return ("有兵方要利用兵的通路优势，在王保护下稳步推进，"
                "迫使对方做出让步后兑现升变")

    return ""


def _compose_opening(storyboard: dict) -> str:
    """开场白纯中文模板：残局概况 + 子力对比 + 取胜策略 + 过渡。

    这是 generate_opening（LLM 生成）失败时的兜底，保证每个视频都有开场白。
    KB 命中时用知识库的 winning_principle；未命中时通过 _derive_strategy
    从子力类型推导具体策略，避免「把优势转化为胜势」这类说了等于没说的空话。
    纯模板，天然无坐标/无术语泄漏。
    """
    try:
        endgame_name = storyboard.get("endgame_name", "") or ""
        matched = storyboard.get("endgame_matched", False)
        winning_side = storyboard.get("winning_side", "") or ""
        losing_side = storyboard.get("losing_side", "") or ""
        strong_material = storyboard.get("strong_material", "") or ""
        weak_material = storyboard.get("weak_material", "") or ""
        white_material = storyboard.get("white_material", "") or ""
        black_material = storyboard.get("black_material", "") or ""
        opening = storyboard.get("opening", {}) or {}
        winning_principle = opening.get("winning_principle", "") or ""

        parts = []

        # 1) 残局引入 + 子力说明
        if matched and endgame_name and endgame_name not in ("残局", "单王残局"):
            parts.append(f"这是一个{endgame_name}残局")
            if strong_material and weak_material:
                parts.append(
                    f"——{winning_side}有{strong_material}，{losing_side}只有{weak_material}。")
            else:
                parts.append("。")
        elif strong_material and weak_material:
            parts.append(
                f"这个残局，{winning_side}有{strong_material}，"
                f"{losing_side}仅有{weak_material}防守。")
        elif white_material and black_material:
            parts.append(
                f"白方有{white_material}，黑方有{black_material}。")
        else:
            parts.append("我们来看这个残局。")

        # 2) 取胜策略：优先 KB → 推导兜底 → 最简兜底
        if winning_principle:
            parts.append("核心思路是" + winning_principle + "。")
        else:
            derived = _derive_strategy(storyboard)
            if derived:
                parts.append("取胜的关键在于——" + derived + "。")
            elif winning_side:
                parts.append(
                    f"由{winning_side}主导进攻，需要逐步把子力优势兑现为胜势。")

        # 3) 过渡句
        parts.append("下面来看具体的推进过程。")

        text = _clean_cjk_text("".join(parts))
        if not text or len(text) < 10:
            return "下面来分析这个残局的取胜过程。"
        return text
    except Exception:
        return "下面来分析这个残局的取胜过程。"


def generate_opening(storyboard: dict, backend) -> str:
    """生成开场白：与 generate_summary 同机制，由 LLM 生成 2-3 句中文导语。

    内容聚焦：残局概况 + 双方子力对比 + 攻守方介绍，自然过渡到正式解说。
    残局名仅在知识库命中时提供给模型，未命中则不给（不强行编造类型名）。
    与总结词一致：纯中文 grammar 锁死、强清洗、校验失败回退 _compose_opening 模板。
    """
    endgame_name = storyboard.get("endgame_name", "") or ""
    matched = storyboard.get("endgame_matched", False)
    winning_side = storyboard.get("winning_side", "") or ""
    losing_side = storyboard.get("losing_side", "") or ""
    strong_material = storyboard.get("strong_material", "") or ""
    weak_material = storyboard.get("weak_material", "") or ""
    white_material = storyboard.get("white_material", "") or ""
    black_material = storyboard.get("black_material", "") or ""
    role_summary = storyboard.get("role_summary", "") or ""
    opening = storyboard.get("opening", {}) or {}
    winning_principle = opening.get("winning_principle", "") or ""

    lines = [
        "你是国际象棋残局教练，正要开始讲解一盘残局，现在对着镜头说开场导语。",
        "请写一段2到3句的中文开场白：先点出这是个什么样的残局、双方各有哪些子力，",
        "再说清哪一方占优、由谁主导进攻，最后自然过渡到接下来的讲解。",
        "要求：① 纯口语中文，像讲课开场；② 不要逐步复述任何走法；",
        "③ 绝对禁止出现任何英文字母、数字、棋盘坐标、格子名、棋谱记号或特殊符号；",
        "④ 不要标题、序号、引号、markdown；⑤ 禁止引擎术语（评估值、距杀步数等）。",
        "",
    ]
    # 残局名仅在 KB 命中时提供（未命中不给，避免模型编造类型名）
    if matched and endgame_name and endgame_name not in ("残局", "单王残局"):
        lines.append(f"残局类型：{endgame_name}")
    if winning_side and strong_material and weak_material:
        lines.append(f"子力对比：{winning_side}有{strong_material}，{losing_side}只有{weak_material}")
    elif white_material and black_material:
        lines.append(f"子力对比：白方有{white_material}，黑方有{black_material}")
    if winning_side:
        lines.append(f"占优并主导进攻的一方：{winning_side}")
    if role_summary:
        lines.append(f"攻守角色：{role_summary}")
    if winning_principle:
        lines.append(f"取胜思路：{winning_principle}")
    else:
        derived = _derive_strategy(storyboard)
        if derived:
            lines.append(f"取胜策略参考：{derived}")

    # instruction-only 骨架：仅含「对模型说话」的指令行，用于 echo 检测
    instruction_only = "\n".join(lines[:7])

    prompt = "\n".join(lines)
    prompt = prompt + "\n\n现在直接输出开场白正文（不要复述以上要求）："
    # 不加 GBNF 语法：_SUMMARY_GRAMMAR 会阻止 Qwen 输出 <think> 标签，
    # 导致思维链被挤成中文直接泄漏到输出中（"思考过程如下…"）。
    # 不用语法时 Qwen 正常输出 <think>...</think> + 正文，
    # _strip_thinking 删掉标签即可拿到干净的开场白。
    raw = backend.generate(prompt, grammar=None)
    raw = _strip_thinking(raw)
    text = _clean_opening_text(raw)

    # 校验：硬特征判废 或 整段复述指令 → 回退纯模板开场白
    if _summary_is_bad(text) or _looks_like_prompt_echo(text, instruction_only):
        return _compose_opening(storyboard)
    return text


def _clean_opening_text(text: str) -> str:
    """清洗开场白：先去引号与「开场」前缀，再走公共 CJK 白名单清洗。"""
    t = text.strip().strip("「」\"'`").strip()
    t = re.sub(r"^开场[白]?[，,：:]?", "", t).strip()
    return _clean_cjk_text(t)


def _fallback_summary(storyboard: dict) -> str:
    """
    LLM 总结失败时的纯中文兜底总结。
    素材优先级：知识库技法(motifs) > 取胜路线(从节点 goal 归纳) > 子残局名。
    无论走哪条都尽量凑出 2-3 句有阶段逻辑的话，避免落到干瘪的单句
    （旧实现 motifs 空时只产「核心在于X阶段的处理」一句，正是线上短总结的来源）。
    """
    endgame_name = storyboard.get("endgame_name", "这类残局")
    motifs = storyboard.get("motifs", []) or []
    mistakes = storyboard.get("mistakes", []) or []

    parts = [f"总结一下，{endgame_name}的取胜关键"]
    if motifs:
        names = [m.split("：")[0].split(":")[0] for m in motifs[:3]]
        parts.append("在于" + "、".join(names) + "。")
    else:
        phrases = _winning_path_phrases(storyboard)
        if len(phrases) >= 2:
            seq = phrases[:4]
            connectors = ["先", "接着", "随后", "最终"]
            steps = "，".join(
                f"{connectors[min(i, len(connectors) - 1)]}{p}"
                for i, p in enumerate(seq))
            # 用逗号而非冒号衔接：_clean_summary_text 白名单会删掉全角冒号，
            # 留下「次序先…」黏连成病句。逗号在白名单内，可安全保留。
            parts.append("，在于把握好推进次序，" + steps + "。")
        else:
            sub_names = _collect_sub_endgame_names(storyboard)
            if sub_names:
                parts.append("，核心在于" + "、".join(sub_names[:3]) + "阶段的处理。")
            else:
                parts.append("，在于稳扎稳打、逐步压缩对方王的活动空间。")

    if mistakes:
        parts.append("过程中要避免" + mistakes[0].split("：")[0].split(":")[0] + "这类失误。")
    else:
        parts.append("关键是每一步都让对方的选择更少，不给对方留下反扑的机会。")

    return _clean_summary_text("".join(parts)) or f"总结一下，{endgame_name}重在稳扎稳打，逐步压缩对方空间。"


def _collect_sub_endgame_names(storyboard: dict) -> list:
    """从节点中收集不重复的子残局名（用于兜底总结提供阶段感）。"""
    nodes = storyboard.get("nodes", [])
    if not nodes:
        return []
    seen = []
    for n in nodes:
        name = n.get("sub_endgame_name", "")
        if name and name not in seen and _is_meaningful_endgame_name(name):
            seen.append(name)
    return seen


def _is_meaningful_endgame_name(name: str) -> bool:
    """过滤掉无意义的残局名（"残局""未知"等）。"""
    noise = {"残局", "未知", "unknown", "unknown_endgame", ""}
    return name not in noise


def generate_summary(storyboard: dict, backend, segments: list = None) -> str:
    """生成 2-3 句结尾总结词：概括这类残局的关键之处与主要逻辑思维方式。

    只喂 storyboard 的结构化要点（不含 SAN 棋谱记号，避免污染），
    要求纯中文输出，再强清洗一遍移除任何字母/数字/符号，
    保证 ChatTTS 拿到的是干净短中文，不会崩成咿呀。
    """
    endgame_name = storyboard.get("endgame_name", "残局")
    winning_side = storyboard.get("winning_side", "")
    context = storyboard.get("context", "") or ""
    role_summary = storyboard.get("role_summary", "") or ""
    motifs = storyboard.get("motifs", []) or []
    mistakes = storyboard.get("mistakes", []) or []
    phases = storyboard.get("phases", []) or []

    lines = [
        "你是国际象棋残局教练，刚讲解完一盘残局，现在对着镜头做收尾总结。",
        "请写一段2到3句的中文总结，要有概括性：说清这类残局取胜的关键之处，",
        "以及背后的主要逻辑思维方式（核心取胜思路、应遵循的次序），并点出要避免的典型错误。",
        "要求：① 纯口语中文，像讲课收尾；② 不要逐步复述具体走法；",
        "③ 绝对禁止出现任何英文字母、数字、棋盘坐标、格子名、棋谱记号或特殊符号；",
        "④ 不要标题、序号、引号、markdown；⑤ 禁止引擎术语（评估值、距杀步数等）。",
        "",
        f"残局类型：{endgame_name}",
    ]
    if winning_side:
        lines.append(f"取胜方：{winning_side}")
    if role_summary:
        lines.append(f"攻守角色：{role_summary}")
    if context:
        lines.append(f"理论要点：{context}")
    if phases:
        phase_names = "、".join(p[0] for p in phases if isinstance(p, (list, tuple)) and p)
        if phase_names:
            lines.append(f"取胜阶段：{phase_names}")
    if motifs:
        lines.append("核心技法：" + "；".join(motifs[:3]))
    if mistakes:
        lines.append("常见错误：" + "；".join(mistakes[:2]))

    # 取胜路线：不依赖 kb，从节点 goal 归纳，保证任何局面都有逻辑骨架可总结。
    winning_path = _winning_path_phrases(storyboard)
    if winning_path:
        lines.append("本局取胜路线：" + " → ".join(winning_path[:5]))
    # 讲解回顾：把刚生成的分段解说浓缩进来，这是最贴切的总结依据
    # （基于「这盘实际怎么赢的」提炼，而非贫乏的 kb 元数据凭空写）。
    recap = _build_recap_from_segments(segments)
    if recap:
        lines.append("刚才的讲解要点回顾：" + recap)

    # instruction-only 骨架：仅含「对模型说话」的指令行，用于 echo 检测。
    # 不含上面注入的领域素材——素材被模型复用是期望行为，不应判成「复述提示词」。
    instruction_only = "\n".join(lines[:7])

    prompt = "\n".join(lines)
    prompt = prompt + "\n\n现在直接输出总结正文（不要复述以上要求）："
    # 不加 GBNF 语法：与 generate_opening 同理 —— _SUMMARY_GRAMMAR
    # 会与 Qwen 思维链冲突导致中文泄漏。不用语法 + _strip_thinking 即可。
    raw = backend.generate(prompt, grammar=None)
    raw = _strip_thinking(raw)
    text = _clean_summary_text(raw)

    # 校验兜底：硬特征判废（过短/过长/中文占比低/含元指令碎片/残留字母数字），
    # 或与「指令骨架」高度重合（整段复述指令）→ 纯中文模板兜底。
    # 注意只比对 instruction_only：注入的讲解回顾/取胜路线被模型复用是期望行为，
    # 拿整个 prompt 比对会把正常复用素材误判成泄漏，反而逼出干瘪兜底。
    if _summary_is_bad(text) or _looks_like_prompt_echo(text, instruction_only):
        text = _fallback_summary(storyboard)
    # 统一补前缀：清洗阶段已把"总结一下，"剥掉做标准化，这里对所有路径（含兜底，
    # _fallback_summary 末尾也会经 _clean_summary_text 剥掉前缀）统一补回，
    # 保证开头总有"总结"二字（曾出现兜底路径缺"总结"开头的 bug）。
    if not text.startswith("总结"):
        text = "总结一下，" + text
    return text



def generate_structured(board: chess.Board, storyboard: dict) -> GeneratedCommentary:
    nodes = storyboard.get("nodes", [])
    commentary = GeneratedCommentary()

    if not nodes:
        Logger.warn("分镜数据为空，无法生成解说")
        return commentary

    backend = create_backend_from_env()

    node_count = len(nodes)
    total_chunks = max(1, (node_count + CHUNK_SIZE - 1) // CHUNK_SIZE)

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

        json_prompt = _build_chunk_prompt(
            json_header, chunk_nodes, chunk_idx, total_chunks,
            _JSON_EXAMPLE if chunk_idx == 0 else "",
            prev_context=prev_context,
        )
        chunk_grammar = _build_chunk_grammar(len(chunk_nodes))

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
                continue

            data = _parse_storyboard_json(raw_text)
            if data is _INVALID_JSON_SENTINEL:
                err_msg = "输出不是合法JSON"
                continue

            # 校验前预处理：对所有 segment 先做 auto-fix 清洗，避免黑名单词导致的硬失败
            segments = data.get("segments")
            if isinstance(segments, list) and len(segments) == len(chunk_nodes):
                for si, seg in enumerate(segments):
                    seg["voiceover"] = _auto_fix_voiceover(seg.get("voiceover", ""), chunk_nodes[si])

            ok, err_msg = _validate_storyboard_chunk(data, chunk_nodes)
            if ok:
                chunk_segments = _finalize_chunk_segments(data, chunk_nodes)
                all_segments.extend(chunk_segments)
                commentary.chunks_succeeded += 1
                success = True
                break

            if isinstance(segments, list) and len(segments) == len(chunk_nodes):
                repaired = _repair_failed_segments(backend, segments, chunk_nodes)
                if repaired is not None:
                    repaired_ok, _ = _validate_storyboard_chunk(repaired, chunk_nodes)
                    if repaired_ok:
                        chunk_segments = _finalize_chunk_segments(repaired, chunk_nodes)
                        all_segments.extend(chunk_segments)
                        commentary.chunks_succeeded += 1
                        success = True
                        break

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
                ))
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

    # 跨段去重：消除 chunk 间各自生成导致的高频套话复读（全局视角，只能在此处做）
    if all_segments and not commentary.fallback_used:
        try:
            _dedupe_across_segments(all_segments)
        except Exception:
            pass

    commentary.raw_text = "\n".join(
        f"第{seg.id}步：{seg.voiceover}" for seg in all_segments
    )

    # 结尾总结词（技法/经验），独立于分步解说，挂到最终局面播放
    if all_segments:
        try:
            commentary.summary = generate_summary(storyboard, backend, all_segments)
        except Exception as e:
            Logger.warn(f"总结词生成异常，使用模板兜底: {e}")
            commentary.summary = _fallback_summary(storyboard)

    # 开场白（残局概况+子力对比+攻守方），插在解说最前。与总结词同机制由 LLM 生成，
    # 失败回退纯模板（_compose_opening 保证非空），因此每个视频都有开场白。
    try:
        commentary.opening = generate_opening(storyboard, backend)
    except Exception as e:
        Logger.warn(f"开场白生成异常，使用模板兜底: {e}")
        commentary.opening = _compose_opening(storyboard)

    status = "正常" if not commentary.fallback_used else f"部分回退({commentary.chunks_succeeded}/{total_chunks})"
    Logger.success(f"解说生成完成: {len(all_segments)} 段, {status}")
    return commentary


# ============================================================
#  Puzzle 战术讲解解说生成（新增，不改原有函数）
# ============================================================

def _get_depth_instruction(rating: int) -> str:
    """三档分层（决策二）：<1500 / 1500-2200 / >2200。
    三档在内容焦点、句式约束、字数预算、可讲深度上做硬区分，避免趋同。"""
    if rating < 1500:
        return (
            "基础讲解，面向完全没有战术经验的初学者，重点讲清「是什么」：\n"
            "- 识别战术模式：这是什么类型的战术？用最通俗的话说\n"
            "- 走法解释：这一步具体做了什么（吃了什么子、保护了什么子）\n"
            "- 结果说明：吃掉了什么子？获得了多少子力优势？\n"
            "- 每步50-100字，用短句，像给朋友现场讲棋\n"
            "- 禁止使用「牵制」「转化」「威胁」「掩护」「切入点」「战术嗅觉」「战术视野」等抽象术语\n"
            "- 只说具体的：吃了什么子、走到哪、逼对方怎样"
        )
    elif rating < 2200:
        return (
            "进阶讲解，面向有一定棋力的棋手，重点讲清「为什么」：\n"
            "- 战术原理：这个战术为什么能成立？对方的失误在哪？\n"
            "- 前提条件：我方子力配置有什么优势？对方哪些子处于不利位置？\n"
            "- 关键手分析：这一步的精妙之处在哪里？为什么非走不可？\n"
            "- 对方困境：对方为什么无法有效应对？有哪些选择，为什么都不行？\n"
            "- 每步60-120字，重点步可多写，语言专业但不晦涩"
        )
    else:
        return (
            "深度讲解，面向有较强计算力的棋手，重点讲清「怎么发现」：\n"
            "- 战术嗅觉：如何在实战中发现这类机会？从哪个信号看出战术存在？\n"
            "- 计算深度：需要看到几步之后？关键变化是什么？\n"
            "- 变着分析：如果这步之后对方不走正解，最强的变着是什么？走错会怎样？\n"
            "- 相关战术：这个战术与其他战术有什么关联？本步的战术本质是什么？\n"
            "- 每步100-180字，可深入拆解计算线路，适合有一定基础的棋手"
        )


def _build_puzzle_json_header(storyboard: dict) -> str:
    """战术分析专家人设 + 四层框架要求 + 标签定义注入 + depth_instruction。"""
    tactic_name = storyboard.get("tactic_name", "战术练习")
    tactic_focus = storyboard.get("tactic_focus", {})
    theme_defs = tactic_focus.get("theme_definitions", "")
    assertions = tactic_focus.get("assertions", [])
    narrative_mode = tactic_focus.get("narrative_mode", "tactical_solution")
    difficulty = storyboard.get("difficulty_level", "intermediate")
    opening_context = storyboard.get("opening_context", "")
    target_length = storyboard.get("target_length", "600-1500字")
    rating = storyboard.get("rating", storyboard.get("difficulty_hint", 1500))
    puzzle_side = storyboard.get("puzzle_side", "白方")
    defending_side = storyboard.get("defending_side", "黑方")

    # Rating → depth_instruction
    depth = ""
    try:
        depth = _get_depth_instruction(int(rating))
    except (ValueError, TypeError):
        depth = _get_depth_instruction(1500)

    node_count = len(storyboard.get("nodes", []))

    lines = [
        "你是一位专业的国际象棋战术分析专家。你的任务是深入拆解棋局中的战术结构，"
        "帮助观众理解每一步背后的逻辑。",
        "",
        "你的讲解信条：",
        "1. 深入浅出：用清晰的语言解释复杂的战术概念",
        "2. 焦点突出：每段优先讲清两件事——这个战术标签的本质，以及它在本步如何具体落实；其余信息点到为止",
        "3. 实战导向：帮助观众培养战术嗅觉，学会在实战中发现类似机会",
        "4. 专业准确：使用正确的棋术术语，避免模糊表述",
        "5. 叙事自然：用连贯的段落串联信息，不要逐条罗列'第一层...第二层...'，禁止出现层号或编号词",
        "",
        f"【战术主题】{tactic_name}",
    ]

    if theme_defs:
        lines.extend([
            "",
            "【标签定义】以下是你应该围绕讲解的战术概念，请融入讲解中：",
            theme_defs,
        ])

    # 主战术深度锚点：单独拎出主标签的机理与关键手，作为「讲透」的抓手。
    # 信息来自知识库，模型据此把抽象概念落地到本局这几步，而非泛泛而谈。
    primary_key = tactic_focus.get("primary_theme", "")
    if primary_key:
        try:
            from src.themes_kb import get_theme as _get_theme
            pt = _get_theme(primary_key)
        except Exception:
            pt = None
        if pt:
            anchor = [
                "",
                f"【主战术深度锚点】本题核心战术是【{pt['cn']}】，请把它讲透，做到以下三层：",
                f"  1. 机理：{pt['cn']}为什么能成立——{pt.get('definition', '')}",
            ]
            if pt.get("key_move_signal"):
                anchor.append(f"  2. 关键手：在本局，{pt['cn']}的关键手表现为——{pt['key_move_signal']}"
                              "。请结合给定走法，明确指出哪一步是这个关键手，它具体做了什么。")
            if pt.get("typical_consequence"):
                anchor.append(f"  3. 结果：{pt['cn']}得手后的典型收益是——{pt['typical_consequence']}"
                              "。请说明本局实际兑现了什么（净赢的子力／被控的线路／对方的困境）。")
            anchor.append("至少要有一处把这个战术概念与本局的具体走法结合起来讲清楚，"
                          "不要只复述定义，也不要只描述走法，要让观众看懂「概念如何在这盘棋里发生」。")
            # 联动叙事：核心战术与次要战术存在辅助关系时，提示组合讲解
            synergy = tactic_focus.get("synergy_themes", [])
            if synergy:
                anchor.append(
                    f"本题还涉及与【{pt['cn']}】相互辅助的战术：{'、'.join(synergy)}。"
                    f"请把它们作为{pt['cn']}的配合手段串起来讲——说明它们如何服务于核心战术，"
                    "而不是各讲各的、平行罗列。")

            # 关键手定位（已用棋盘事实算好）：直接把"哪一步是关键手 + 理由"喂给模型，
            # 避免模型把第一步将军/吃子讲成核心战术。
            key_idx = tactic_focus.get("key_move_idx") or 0
            key_san = tactic_focus.get("key_move_san", "") or ""
            key_reason = tactic_focus.get("key_move_reason", "") or ""
            if key_idx and key_san:
                anchor.append(
                    f"【已算出的关键手】本题核心战术的关键手是第{key_idx}手 {key_san}。"
                    f"理由：{key_reason or '由棋盘事实算出'}。"
                    f"讲解时务必让观众看到「这一步才是核心」，"
                    f"不要把任何其他子（如纯将军、过渡吃子）误讲成核心战术。")
            lines.extend(anchor)

    if assertions:
        for a in assertions:
            if a:
                lines.append(f"【核心约束】{a}")

    if opening_context:
        lines.append(f"【开局背景】{opening_context}")

    lines.extend([
        "",
        f"【叙事视角】从{puzzle_side}（解题方）视角讲解。",
    ])
    if narrative_mode == "defensive_resource":
        lines.append(
            f"本题是防守型战术——重点讲{puzzle_side}在劣势中如何找到唯一防守资源化解危机。"
        )
    else:
        lines.append(
            f"重点讲{puzzle_side}如何主动发现战术机会，通过强制手段获得优势或杀棋。"
        )

    # 分级约束补充：低级更强硬地禁止抽象术语，高级放开变着/计算深度
    rating_int = 0
    try:
        rating_int = int(rating)
    except (ValueError, TypeError):
        pass
    if rating_int < 1500:
        lines.extend([
            "",
            "【难度约束·低级】本题面向初学者，你必须：",
            "- 只用「吃了X」「走到Y」「逼对方Z」这类具体描述，禁止「牵制」「转化」「威胁」「掩护」",
            "- 每步50-100字，用短句，像给朋友现场讲棋",
            "- 第一步就直接说「白方/黑方吃了对方的X」，不要做任何铺垫",
        ])
    elif rating_int >= 2200:
        lines.extend([
            "",
            "【难度约束·高级】本题面向有经验的棋手，你应当：",
            "- 在关键步讲清计算线路：如果对方走了X，己方如何应对Y，最终得到Z",
            "- 可以提变着：「如果对方不走X，而是走Y，则…」",
            "- 每步100-180字，允许深入拆解，适合有一定基础的棋手",
        ])

    lines.extend([
        "",
        f"【讲解深度要求】{depth}",
        "",
        "【讲解要点】每段自然成段、不逐条编号，但要兼顾「概念」与「落地」两层：",
        "- 概念层：这一步用到的战术标签是什么、为什么在这里能成立（依据上面给的定义/识别/前提，对方的弱点或失误在哪）",
        "- 落地层：这个战术在本局如何具体兑现——哪一步是关键手、它具体做了什么、带来什么确定结果（净赢的子力、被控的关键线路、对方的困境）",
        "- 关键步（标注为重点节点的）要把上面两层都讲透，让观众看懂战术机理；过渡步可从简，一两句交代清楚即可",
        "- 强制性来源、对方为何无法应对、实战识别等可在合适处自然带出，不必每步都展开",
        "",
        "【解说规则】",
        f"- 正好{node_count}个segment，不增不减",
        "- 如果需要推理，请只把推理过程写在最前面的 <think>...</think> 中；关闭 think 后只能输出 JSON，不得把推理过程写入 voiceover",
        "- 不要开场白和总结词，直接切入战术分析",
        "- 第一步就要进入战术讲解，不要铺垫局面背景",
        "- 整体围绕战术标签讲，每步尽量结合相关标签解释",
        "- 使用标签中的专业术语，但要解释清楚",
        f"- 全局字数预算控制在{target_length}",
        "- 关键步可写到150-200字把战术讲透，过渡步60-120字从简；宁可在关键步多花笔墨，也不要每步都泛泛而谈",
        "- 禁止使用引擎术语：评估值、分数、厘兵、mate in N",
        "- 禁止虚构或假设走法",
        "- voiceover用纯中文口播，禁止出现棋盘坐标（如h7、g5）",
        "- 禁止输出单独的大写棋子字母（N/B/R/Q/K），请用'马/象/车/后/王'",
        "- 禁止使用括号（包括中英文括号）：如果需要补充说明，请用逗号、破折号或'比如''也就是'等词自然地融入句子，而不是塞进括号",
        "- 指位置时改用方位关系：「底线」「边线」「中心」「王前」「同一条斜线」等",
        "- 不要把【】标签名或标签符号念出来，只讲战术内容本身",
        "- 讲解中要包含具体的走法细节（吃了什么子、走到哪个格子附近），不能全是空泛的形容",
        "- quietMove等安静步骤要完整描述局面变化，句子保持完整，不要断句或留残句",
        "",
        "【JSON格式】",
        '{"segments":[{"id":int,"sub_endgame":"","voiceover":"string","pacing":"slow|normal|fast|pause_before|pause_after"},...]}',
        "segments数量必须等于节点数。sub_endgame字段固定输出空字符串即可。",
    ])
    return "\n".join(lines)


_SAN_PIECE_MAP = {'N': '马', 'B': '象', 'R': '车', 'Q': '后', 'K': '王'}


def _san_piece_to_chinese(moves_str: str) -> str:
    """将 SAN 走法中的棋子字母转为中文。如 'Nf6'→'马'，无棋子字母时（兵走法）返回'兵'。"""
    if moves_str.startswith("O-O-O"):
        return "后翼易位"
    if moves_str.startswith("O-O"):
        return "王翼易位"
    for piece in ('N', 'B', 'R', 'Q', 'K'):
        if moves_str.startswith(piece):
            return _SAN_PIECE_MAP[piece]
    return "兵"


def _build_puzzle_chunk_prompt(header: str, chunk_nodes: list, chunk_idx: int,
                                total_chunks: int) -> str:
    """构建 puzzle 分块 prompt。"""
    is_last = (chunk_idx == total_chunks - 1)
    lines = [header]

    chunk_rule = ""
    if total_chunks > 1:
        if is_last:
            chunk_rule = "本段包含最后几步，允许在最后一步做总结性收束。"
        else:
            chunk_rule = "本段只解说这些步骤，禁止提前总结。"
    lines.append(f"--- 第{chunk_idx + 1}/{total_chunks}段节点 {'(最后)' if is_last else ''} ---")
    if chunk_rule:
        lines.append(chunk_rule)
    lines.append("")

    for node in chunk_nodes:
        nid = node["id"]
        lines.append(f"--- 节点{nid} ---")
        lines.append(f"走法: {_san_piece_to_chinese(node['moves'])}（{node.get('turn', '')}）")
        lines.append(f"状态: {'将军' if node.get('is_check') else '非将军'}"
                     f" | {'吃子' if node.get('is_capture') else '未吃子'}"
                     f" | {'已将杀' if node.get('is_checkmate') else '未将杀'}")

        # 确定性事实：吃掉的具体子力、对方应招数（让解说有硬料可写，挤掉套话）
        captured = node.get("captured_piece_cn", "")
        if captured:
            lines.append(f"[核心] 吃掉的子力: 对方的{captured}")
        reply_count = node.get("legal_reply_count_after")
        if isinstance(reply_count, int) and not node.get("is_checkmate"):
            # 用中文数字表述，避免模型照搬阿拉伯数字被 voiceover 语法卡掉
            cn_num = "零一二三四五六七八九"[reply_count] if 0 <= reply_count < 10 else str(reply_count)
            if reply_count == 0:
                pass  # 0 应招即将杀，上一行已标注
            elif reply_count <= 3:
                lines.append(f"[核心] 走后对方仅剩{cn_num}个合法应招，回旋余地极小")
            elif reply_count <= 8:
                lines.append(f"[核心] 走后对方合法应招收缩到{cn_num}个，明显受限")

        # —— 核心材料（必须讲清）——
        theme_ctx = node.get("theme_context", "")
        if theme_ctx:
            lines.append(f"[核心] 战术关联: {node.get('related_theme', '')} — {theme_ctx}")

        geo = node.get("puzzle_tactical_facts", [])
        if geo:
            lines.append("[核心] 本局确定事实:")
            for gf in geo:
                lines.append(f"  · {gf}")

        must = node.get("must_mention", [])
        if must:
            lines.append(f"[核心] 应提及: {'；'.join(must)}")

        teaching = node.get("teaching_point", "")
        if teaching:
            lines.append(f"[核心] 棋理事实: {teaching}")

        # 关键手定位提示：本节点是不是核心/次要标签的关键手，避免模型把将军/吃子讲成"核心战术"
        is_core_key = node.get("is_core_theme_key_move")
        roles = node.get("theme_key_roles") or []
        key_reason = node.get("theme_key_reason", "")
        if is_core_key and key_reason:
            lines.append(f"[核心] 关键手（核心战术落点）: {key_reason}")
        elif roles and key_reason:
            roles_cn = "、".join(roles)
            lines.append(f"[参考] 本步承担标签角色({roles_cn})：{key_reason}")

        # —— 参考材料（自然时一笔带过，不展开）——
        prereq = node.get("prerequisite_facts", "")
        if prereq:
            lines.append(f"[参考] 战术前提: {prereq}")

        mistakes = node.get("common_mistakes", [])
        if mistakes:
            lines.append(f"[参考] 常见误区: {'；'.join(mistakes[:2])}")

        tactical = node.get("tactical_narratives", [])
        if tactical:
            lines.append("[参考] 棋理分析:")
            for tn in tactical:
                lines.append(f"  · {tn}")

        # pacing 提示
        pacing = node.get("suggested_pacing", "normal")
        if pacing in ("slow", "pause_before", "pause_after"):
            lines.append(f"节奏: {pacing} — 这是关键节点，请重点展开讲解")

        lines.append("")

    return "\n".join(lines)


def _score_puzzle_depth(text: str, kp: dict) -> bool:
    """关键手段落的深度校验：至少覆盖 2/3 类关键词（原因/变化/困境）。

    单一关键词（如仅含"迫使"）不足以证明深度，必须同时包含至少两类。
    """
    cause_words = ("因为", "所以", "正是", "从而", "导致", "意味着", "因此")
    change_words = ("之前", "之后", "一旦", "不同于", "改变")
    constraint_words = ("迫使", "无法", "必须", "不能", "只能", "否则")
    categories = sum([
        any(w in text for w in cause_words),
        any(w in text for w in change_words),
        any(w in text for w in constraint_words),
    ])
    return categories >= 2


def _validate_puzzle_voiceover_surface(text: str) -> tuple:
    """校验谜题口播表层字符与思考泄漏。

    不做删除；发现英文、数字、Markdown 符号或明显思考痕迹时直接判失败，
    交给重试或模板兜底，避免污染字幕/TTS。
    """
    if re.search(r"[A-Za-z0-9*_#`\[\]{}<>|\\/]", text):
        return False, "voiceover含英文/数字/Markdown符号"

    thinking_leaks = (
        "我需要", "让我", "先看", "首先我", "接下来我", "题目要求",
        "提示词", "用户", "输出", "这个节点", "这个segment", "思考过程",
        "推理过程", "我会", "我应该", "需要分析", "需要判断",
    )
    if any(w in text for w in thinking_leaks):
        return False, "voiceover含思考过程泄漏"

    return True, ""


def _validate_puzzle_segment(seg: dict, node: dict) -> tuple:
    """puzzle 专用校验：保留 JSON 结构/数量/长度/pacing 校验，
    移除将杀/将军/吃子的真值禁止校验（puzzle 中这些是正常内容）。
    不强制每段提及 related_theme（实施决策 B：只做软约束）。
    """
    seg_id = seg.get("id")
    if not isinstance(seg_id, int):
        return False, f"id={seg_id}不是有效整数"

    voiceover = seg.get("voiceover")
    if not isinstance(voiceover, str) or not voiceover.strip():
        return False, "voiceover为空"

    # 表层安全校验：非法字符/Markdown/思考泄漏一律失败重试，不能进入 TTS。
    surface_ok, surface_err = _validate_puzzle_voiceover_surface(voiceover.strip())
    if not surface_ok:
        return False, surface_err

    # 最短长度校验
    min_len = 28 if node.get("is_checkmate_after") and len(node.get("moves", "")) <= 4 else 48
    if len(voiceover.strip()) < min_len:
        return False, f"voiceover过短({len(voiceover.strip())}<{min_len})"

    pacing = seg.get("pacing", "normal")
    pacing = str(pacing).strip().lower()
    if pacing not in ALLOWED_PACING:
        return False, f"pacing='{pacing}'不合法"

    # 反套话检测：使用 puzzle 轻量版检测（不做形容词清洗，避免'精准/精确'等战术词被误杀）
    cleaned = _reduce_cliches_puzzle(voiceover.strip())
    if len(cleaned) < len(voiceover.strip()) * 0.3:
        return False, "voiceover套话占比过高"

    # 确保 segment id 与节点 id 对齐（多 chunk 时 LLM 可能从 1 重新编号）
    seg["id"] = node["id"]

    return True, ""


def _auto_fix_puzzle_voiceover(text: str, node: dict) -> str:
    """puzzle 专用自动修复：坐标清洗 + 标签标记删除 + 括号展开 + 轻量反套话 + 标点收敛。"""
    fixed = text

    # 坐标兜底清洗
    fixed = _strip_coordinates(fixed)

    # 删除标签标记泄漏（如【优势】【叉击】等被模型原样输出的内容）
    fixed = re.sub(r"[【][^】]{1,20}[】]", "", fixed)

    # 括号展开：把括号内容融入句子，避免口播出现括号停顿
    fixed = _expand_inline_brackets(fixed)

    # 不完整句子修复：删除「这步X。」后面直接接另一句的残句结构
    fixed = re.sub(r"(这步[^。]{0,6})。(实战|这是|这步|黑方|白方|面对|面对)", r"\1，\2", fixed)

    # 轻量反套话（不做形容词删除，保留'精准/精确'等战术语义词）
    fixed = _reduce_cliches_puzzle(fixed)

    # 标点收敛
    fixed = re.sub(r"[，,]{2,}", "，", fixed)
    fixed = re.sub(r"。{2,}", "。", fixed)
    fixed = re.sub(r"\s{2,}", " ", fixed)
    fixed = re.sub(r"[，、]+。", "。", fixed)
    fixed = re.sub(r"^[，、。]+", "", fixed)
    fixed = fixed.strip()

    return fixed


# ── Puzzle 专用轻量反套话表 ───────────────────────────────────────────
# 与 _CLICHE_PATTERNS 的区别：跳过形容词类清洗（精准/精确/精妙 等在战术讲解中承载实际语义）。
_PUZZLE_CLICHE_PATTERNS = [
    (r"看似(?:平淡无奇|平平无奇|平淡|不起眼|普通|简单)[，,]?(?:实则|却|其实)?", ""),
    (r"别看这一步[^，。]{0,6}[，,]", ""),
    (r"如同?利剑出鞘", ""),
    (r"如洪水般(不可阻挡)?", "持续"),
    (r"天罗地网", "严密的控制"),
    (r"天衣无缝", "完整"),
    (r"密不透风", "严密"),
    (r"无形的牢笼", "包围"),
    (r"胜利的天平(开始|彻底)?(倾斜)?", "优势"),
    (r"(已如)?囊中之物", "胜势已成"),
    (r"不可阻挡", ""),
    (r"暗藏杀机", ""),
    (r"耐心的围猎", "稳步驱赶"),
    (r"致命一击", "最后一击"),
    (r"愈发默契", "更协调"),
    (r"(配合|协调)(愈发|越来越)默契", "配合更协调"),
    (r"默契(的)?配合", "配合"),
    (r"步步为营", "稳步推进"),
    (r"[，,]?\s*为(?:后续|接下来|下一步|最终|后面|最后)(?:的)?[^，。、！]{0,16}"
     r"(?:做准备|做好准备|奠定[了]?(?:坚实)?基础|创造[了]?[^，。]{0,8}条件|铺平[了]?道路|埋下伏笔)", ""),
]


def _reduce_cliches_puzzle(text: str) -> str:
    """Puzzle 轻量反套话：只删废话模板，保留战术形容词。"""
    out = text
    for pat, repl in _PUZZLE_CLICHE_PATTERNS:
        out = re.sub(pat, repl, out)
    return out


def _validate_puzzle_chunk(data: dict, chunk_nodes: list) -> tuple:
    """逐段校验 puzzle chunk。"""
    segments = data.get("segments")
    if not isinstance(segments, list):
        return False, "顶层缺少segments数组"
    if len(segments) != len(chunk_nodes):
        return False, f"segments数量{len(segments)}与节点数{len(chunk_nodes)}不一致"

    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            return False, f"第{i+1}个segment不是对象"
        ok, err = _validate_puzzle_segment(seg, chunk_nodes[i])
        if not ok:
            return False, f"segment[{i}]{err}"

    return True, ""


# ============================================================
#  Puzzle 双关键点强约束（对应实施文档 §3.2）
#  谜题链路必须且只须讲透两个关键点：
#    关键点1（机理）：标签代表的战术策略是什么、为什么成立
#    关键点2（落地）：该战术在本局如何兑现——哪步是关键手、做了什么、什么结果
#  其余效果可让步，但这两点必须覆盖。下方为骨架提取 / 评分 / 模板 / 润色四件套。
# ============================================================

# 落地层「确定结果」判定词：解说命中其一即视为讲到了战术兑现的结果
_PUZZLE_RESULT_WORDS = ("赢", "得子", "得回", "多子", "失", "丢", "被迫", "无法",
                        "困", "杀", "优势", "子力", "胜势", "制胜", "致胜")

_DIGIT_CN = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}


def _expand_inline_brackets(text: str) -> str:
    """把括号内容融入句子，避免口播/字幕出现括号停顿。

    供知识库种子规范化与 LLM 文本清洗共用，确保两条路径行为一致。
    调用方需自行先剔除不该展开的括注（如引擎评估值括注）。
    """
    if not text:
        return text
    out = text
    # 1) （如/比如/例如X、Y、Z）→ ，比如X、Y、Z
    out = re.sub(r"[（(]\s*(?:如|比如|例如)\s*([^（）()]+)[）)]", r"，比如\1", out)
    # 2) （含顿号/逗号的列举）→ ，也就是X
    out = re.sub(r"[（(]([^（）()]*[、，][^（）()]*)[）)]", r"，也就是\1", out)
    # 3) 剩余短括号（1-20字）→ 直接去掉括号
    out = re.sub(r"[（(]([^（）()]{1,20})[）)]", r"，\1", out)
    # 清理可能产生的双逗号
    out = re.sub(r"[，,]{2,}", "，", out)
    return out


def _safe_puzzle_seed_text(text: str) -> str:
    """把知识库/骨架中的棋谱坐标和阿拉伯数字转成适合口播的中文表达。

    这是源数据规范化，不处理模型已生成文本；目的是避免模板兜底或 prompt 锚点
    自身把 f2/f7、5-7 这类不可播字符带入口播。
    """
    if not text:
        return ""
    out = str(text)
    out = re.sub(r"f2\s*/\s*f7|f7\s*/\s*f2", "王前弱格", out, flags=re.I)
    out = re.sub(r"\bf[27]\b", "王前弱格", out, flags=re.I)
    out = re.sub(r"\b[a-h][1-8]\b", "关键格", out, flags=re.I)
    out = out.replace("fried liver攻击", "经典弃子攻击")
    out = re.sub(r"[A-Za-z]", "", out)
    out = "".join(_DIGIT_CN.get(ch, ch) for ch in out)
    out = re.sub(r"[/\\*_#`\[\]{}<>|]", "", out)
    # 剔除引擎术语括注（如"（评估约六零零厘兵以上）"），口播不能出现评估值/厘兵
    out = re.sub(r"[（(][^（）()]*(?:评估|厘兵|分值)[^（）()]*[）)]", "", out)
    out = out.replace("厘兵", "").replace("评估值", "").replace("评估", "")
    # 展开剩余的列举型/说明型括号，避免知识库种子文本把括号带入口播与字幕
    out = _expand_inline_brackets(out)
    out = re.sub(r"[，、]{2,}", "，", out)
    return out.strip()


def _describe_key_move(node: dict) -> tuple:
    """从节点生成无坐标的关键手描述。返回 (描述句, 棋子中文名)。"""
    piece = _san_piece_to_chinese(node.get("moves", ""))
    turn = node.get("turn", "")
    side = "黑方" if "黑" in turn else "白方"

    actions = []
    if node.get("is_capture"):
        actions.append("吃子")
    if node.get("is_checkmate_after"):
        actions.append("形成将杀")
    elif node.get("is_check"):
        actions.append("将军")
    action_text = "、".join(actions) if actions else "走到关键位置"
    return f"{side}用{piece}{action_text}", piece


def _resolve_key_move_idx(nodes: list, primary_key: str):
    """定位关键手节点 id。

    优先级：关联到主标签的节点 → 首个将杀/将军/吃子节点 → 首个节点。
    """
    if not nodes:
        return None
    # 1. 关联到主标签的节点
    if primary_key:
        for node in nodes:
            if node.get("related_theme") == primary_key:
                return node["id"]
    # 2. 首个将杀 / 将军 / 吃子节点
    for node in nodes:
        if (node.get("is_checkmate_after") or node.get("is_check")
                or node.get("is_capture")):
            return node["id"]
    # 3. 兜底首个节点
    return nodes[0]["id"]


def build_puzzle_keypoint_skeleton(storyboard: dict) -> dict:
    """构建谜题双关键点骨架（对应 §6.2.1）。

    数据来源：主标签知识库字段（机理）+ 本局实际走法与净子力事实（落地）。
    返回的骨架供评分器、模板、润色器共用；缺失主标签时返回空 dict。
    """
    nodes = storyboard.get("nodes", [])
    if not nodes:
        return {}

    tactic_focus = storyboard.get("tactic_focus", {})
    primary_key = tactic_focus.get("primary_theme", "")

    theme = None
    if primary_key:
        try:
            from src.themes_kb import get_theme
            theme = get_theme(primary_key)
        except Exception:
            theme = None
    if not theme:
        return {}

    # 优先使用 storyboard 阶段已算好的关键手定位（用棋盘事实评分），
    # 兜底才用旧的 _resolve_key_move_idx（基于将军/吃子等简单信号）。
    key_move_idx = tactic_focus.get("key_move_idx") or 0
    if not key_move_idx:
        key_move_idx = _resolve_key_move_idx(nodes, primary_key)
    key_node = next((n for n in nodes if n["id"] == key_move_idx), nodes[0])
    key_move_desc, key_move_piece = _describe_key_move(key_node)

    # 落地层「实际结果」：优先净子力事实，其次将杀，最后中性兜底
    actual_result = ""
    last_node = nodes[-1]
    for fact in last_node.get("puzzle_tactical_facts", []):
        if any(w in fact for w in ("净赢", "净得", "净多", "多得", "赢得")):
            actual_result = fact
            break
    if not actual_result:
        if any(n.get("is_checkmate_after") for n in nodes):
            actual_result = "完成将杀，直接终结对局"
        else:
            actual_result = "取得明显优势"

    # 机理层匹配词：标签中文名 + 别名，供评分器判断「是否讲到战术是什么」
    aliases = [_safe_puzzle_seed_text(a) for a in theme.get("aliases_cn", [])]
    tactic_cn = _safe_puzzle_seed_text(theme.get("cn", primary_key))
    concept_words = [w for w in [tactic_cn] + aliases if w]

    # 深度层：局面证据 + 变化对比 + 对方困境（关键手段落专用）
    definition = _safe_puzzle_seed_text(theme.get("definition", ""))
    consequence = _safe_puzzle_seed_text(theme.get("typical_consequence", ""))
    key_move_signal = _safe_puzzle_seed_text(theme.get("key_move_signal", ""))
    recognition = _safe_puzzle_seed_text(theme.get("recognition", ""))

    # 句2用识别特征/局面证据，与句4的对方困境(consequence)区分，避免模板复读
    local_weakness = recognition or definition or ""

    before_after = ""
    if key_move_signal:
        before_after = key_move_signal

    defender_problem = ""
    if consequence:
        defender_problem = consequence

    return {
        # 关键点 1：战术策略是什么（机理）
        "tactic_cn": tactic_cn,
        "tactic_concept_words": concept_words,
        "tactic_definition": definition,
        "tactic_recognition": _safe_puzzle_seed_text(theme.get("recognition", "")),
        # 关键点 2：战术如何在本局使用（落地）
        "key_move_idx": key_move_idx,
        "key_move_desc": _safe_puzzle_seed_text(key_move_desc),
        "key_move_piece": key_move_piece,
        "key_move_signal": key_move_signal,
        "consequence": consequence,
        "actual_result": _safe_puzzle_seed_text(actual_result),
        # 深度层：关键手段落专用，增强具体性
        "local_weakness": local_weakness,
        "before_after": before_after,
        "defender_problem": defender_problem,
    }


def _score_puzzle_keypoints(text: str, kp: dict) -> dict:
    """谜题双关键点覆盖评分（一票否决，对应 §7.2.1）。

    两个关键点都覆盖才 pass=True；任一缺失即判不合格。
    """
    if not text or not kp:
        return {"kp1_covered": False, "kp2_covered": False,
                "pass": False, "issues": ["缺少文本或骨架"]}

    issues = []

    # 关键点 1（机理）：命中战术中文名或其别名
    concept_words = kp.get("tactic_concept_words", [])
    kp1_covered = any(w and w in text for w in concept_words)
    if not kp1_covered:
        issues.append(f"未讲清战术策略「{kp.get('tactic_cn', '')}」是什么（关键点1·机理缺失）")

    # 关键点 2（落地）：命中关键手棋子 + 确定结果词
    key_piece = kp.get("key_move_piece", "")
    has_key_move = bool(key_piece) and key_piece in text
    has_result = any(w in text for w in _PUZZLE_RESULT_WORDS)
    kp2_covered = has_key_move and has_result
    if not kp2_covered:
        issues.append("未讲清战术在本局如何兑现（关键点2·落地缺失）")

    return {
        "kp1_covered": kp1_covered,
        "kp2_covered": kp2_covered,
        "pass": kp1_covered and kp2_covered,
        "issues": issues,
    }


def _compose_puzzle_voiceover(node: dict, kp: dict) -> str:
    """谜题关键手节点的模板填空（4 句固定结构，保底，对应 §9.6）。

    4 句结构：机理 → 证据 → 变化 → 困境/结果。
    保证纯模板下也 100% 覆盖双关键点 + 有具体棋理深度。
    """
    tactic_cn = kp.get("tactic_cn", "该战术")
    definition = kp.get("tactic_definition", "")
    key_move_desc = kp.get("key_move_desc", "这一手")
    before_after = kp.get("before_after", "").rstrip("。！？，、；：")
    defender_problem = kp.get("defender_problem", "").rstrip("。！？，、；：")
    actual_result = kp.get("actual_result", "取得优势").rstrip("。！？，、；：")

    # 句 1：指出战术名和核心机理
    # 去除定义末尾的标点，避免与外层句号重复
    def_clean = definition.rstrip("。！？，、；：") if definition else ""
    sent1 = f"这里的核心是{tactic_cn}——{def_clean}" if def_clean else f"这里运用的战术是{tactic_cn}"

    # 句 2：指出本局里的具体证据
    local_weakness = kp.get("local_weakness", "").rstrip("。！？，、；：")
    if local_weakness:
        sent2 = f"本局中，{local_weakness}"
    else:
        sent2 = f"关键手是{key_move_desc}"

    # 句 3：指出关键手改变了什么
    if before_after:
        sent3 = f"{key_move_desc}，{before_after}"
    else:
        sent3 = f"{key_move_desc}，把战术从可能变成现实"

    # 句 4：指出对方为什么难受 + 最终结果
    if defender_problem:
        # 句中已含"对方"则不再加前缀，避免"对方…对方"重复
        if "对方" in defender_problem:
            sent4 = f"{defender_problem}，{actual_result}"
        else:
            sent4 = f"对方{defender_problem}，{actual_result}"
    else:
        sent4 = f"最终{actual_result}"

    return f"{sent1}。{sent2}。{sent3}。{sent4}。"


def _polish_puzzle_voiceover(node: dict, kp: dict, prev_context: str,
                             backend) -> str:
    """谜题关键手节点的 LLM 润色（双关键点强约束，对应 §10.3.1）。"""
    prompt = f"""你在讲解一道国际象棋战术题。请用自然口语化的中文写这一步的解说。

【本题战术】{kp.get('tactic_cn', '')}

【必须讲清的两个关键点（缺一不可）】
1. 这个战术是什么、为什么能成立：{kp.get('tactic_definition', '')}
2. 这个战术在本局如何兑现：关键手是「{kp.get('key_move_desc', '')}」，{kp.get('consequence', '')}，最终{kp.get('actual_result', '')}

【深度素材（必须用上至少两个）】
- 局面证据：{kp.get('local_weakness', '（无）')}
- 变化对比：{kp.get('before_after', '（无）')}
- 对方困境：{kp.get('defender_problem', '（无）')}

上一段结尾：{prev_context or '（无）'}

要求：
- 如果需要推理，请只把推理过程写在最前面的思考标签中；关闭思考后只能输出给观众听的中文解说
- 必须按以下 4 句结构组织：第 1 句指出战术名和核心机理；第 2 句指出本局里的具体证据；第 3 句指出关键手改变了什么；第 4 句指出对方为什么难受以及最终结果
- 必须出现「{kp.get('tactic_cn', '')}」这个词
- 必须提到关键手用的是哪个子（{kp.get('key_move_piece', '')}）以及最终得到的结果
- 120-200字，自然口语，禁止棋子英文、坐标、套话模板词
- 禁止编造走法"""
    return _strip_thinking(
        backend.generate(prompt, grammar=_PUZZLE_PLAIN_CN_GRAMMAR)).strip()


def _compose_puzzle_intro(kp: dict, storyboard: dict) -> str:
    """谜题开场白模板：3 套自然半模板轮换，稳定不依赖 LLM。

    字段来自 keypoint_skeleton（确定性），不出现坐标/英文/Markdown。
    """
    tactic_cn = kp.get("tactic_cn", "战术")
    puzzle_side = storyboard.get("puzzle_side", "")
    key_move_piece = kp.get("key_move_piece", "")
    recognition = kp.get("tactic_recognition", "")
    # 去掉尾部标点，避免与模板自带句号拼成"。。"
    consequence = kp.get("defender_problem", "").rstrip("。！？，、；：")

    # 模板 A：问题导向
    intro_a = f"这道题的重点不是先算很长的变化，而是先发现{tactic_cn}这个战术信号。"
    if recognition:
        intro_a += f"机会来自{recognition}，接下来要看懂{key_move_piece}为什么能成为突破点。"
    else:
        intro_a += f"接下来要看懂{key_move_piece}为什么能成为突破点。"

    # 模板 B：实战导向
    intro_b = f"实战里遇到这种局面，更重要的是看清{tactic_cn}这个主题。"
    if consequence:
        # consequence 已含"对方"则不再加前缀，避免"对方就会对方…"重复
        if "对方" in consequence:
            intro_b += f"一旦关键手出现，{consequence}。"
        else:
            intro_b += f"一旦关键手出现，对方就会{consequence}。"

    # 模板 C：悬念导向
    intro_c = f"这题表面上只是一步普通走法，但真正的看点是{tactic_cn}。"
    intro_c += "关键不在于这步走得漂亮，而在于它让对方马上陷入被动。"

    import hashlib
    idx = int(hashlib.md5(tactic_cn.encode()).hexdigest()[:8], 16) % 3
    return [intro_a, intro_b, intro_c][idx]


def _enforce_puzzle_keypoints(segments: list, nodes: list, kp: dict,
                              backend) -> bool:
    """确保关键手 segment 覆盖双关键点（多重失败安全，对应 §10.2）。

    流程：评分 → 不达标则 LLM 润色 → 仍不达标则模板兜底（模板保证 100% 覆盖）。
    返回该 segment 最终是否覆盖双关键点。
    """
    if not kp or kp.get("key_move_idx") is None:
        return False

    key_id = kp["key_move_idx"]
    seg = next((s for s in segments if s.id == key_id), None)
    node = next((n for n in nodes if n["id"] == key_id), None)
    if seg is None or node is None:
        Logger.warn(f"  双关键点诊断：未定位到关键手节点(key_id={key_id}, "
                    f"seg={'有' if seg else '无'}, node={'有' if node else '无'})")
        return False

    # 诊断：打印关键手节点的入口评分明细，便于确认走了哪条分支
    surface_ok, surface_reason = _validate_puzzle_voiceover_surface(seg.voiceover)
    kp_score = _score_puzzle_keypoints(seg.voiceover, kp)
    depth_ok = _score_puzzle_depth(seg.voiceover, kp)
    Logger.info(
        f"  双关键点诊断 节点{key_id}: 表层={'过' if surface_ok else '挂(' + surface_reason + ')'} "
        f"机理={'过' if kp_score['kp1_covered'] else '挂'} "
        f"落地={'过' if kp_score['kp2_covered'] else '挂'} "
        f"深度={'过' if depth_ok else '挂'}")

    # 已覆盖且表层安全且有深度则不动
    if surface_ok and kp_score["pass"] and depth_ok:
        Logger.info(f"  关键手节点{key_id}已通过所有评分，无需重试")
        return True

    # 找上一段做承接上下文
    prev_context = ""
    prev_seg = next((s for s in segments if s.id == key_id - 1), None)
    if prev_seg is not None:
        prev_context = prev_seg.voiceover[-40:]

    # LLM 润色重试
    try:
        polished = _polish_puzzle_voiceover(node, kp, prev_context, backend)
        surface_ok, _ = _validate_puzzle_voiceover_surface(polished)
        if (polished and surface_ok
                and _score_puzzle_keypoints(polished, kp)["pass"]
                and _score_puzzle_depth(polished, kp)):
            # 过一遍 puzzle auto-fix：展开括号、清坐标、收敛标点（清洗不减关键词，不影响已通过的评分）
            seg.voiceover = _auto_fix_puzzle_voiceover(polished, node)
            Logger.info(f"  关键手节点{key_id}经润色已覆盖双关键点且有深度")
            return True
    except Exception as e:
        Logger.warn(f"  关键手润色失败，转模板兜底: {e}")

    # 模板兜底（保证覆盖）。同样过一遍 auto-fix，避免知识库种子文本里的括号泄漏到口播。
    seg.voiceover = _auto_fix_puzzle_voiceover(_compose_puzzle_voiceover(node, kp), node)
    Logger.info(f"  关键手节点{key_id}降级到模板，已覆盖双关键点")
    return True


def generate_puzzle_structured(board: chess.Board, storyboard: dict) -> GeneratedCommentary:
    """Puzzle 战术讲解主入口。结构与 generate_structured 同构（分块/GBNF/重试/修复），但：
    - 用 _build_puzzle_json_header 取代 _build_json_header
    - 用 _validate_puzzle_chunk 取代 _validate_storyboard_chunk
    - 不生成 opening / summary
    - 复用 _parse_storyboard_json / _auto_fix_voiceover / _strip_coordinates / _reduce_cliches
    """
    nodes = storyboard.get("nodes", [])
    commentary = GeneratedCommentary()

    if not nodes:
        Logger.warn("Puzzle 分镜数据为空，无法生成解说")
        return commentary

    backend = create_backend_from_env()

    node_count = len(nodes)
    total_chunks = max(1, (node_count + CHUNK_SIZE - 1) // CHUNK_SIZE)

    json_header = _build_puzzle_json_header(storyboard)
    all_segments = []
    commentary.chunks_total = total_chunks

    for chunk_idx in range(total_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, node_count)
        chunk_nodes = nodes[start:end]

        json_prompt = _build_puzzle_chunk_prompt(
            json_header, chunk_nodes, chunk_idx, total_chunks)
        # 用 puzzle 收紧语法：voiceover 在采样阶段即禁止 Markdown/英文/数字
        chunk_grammar = _build_puzzle_chunk_grammar(len(chunk_nodes))

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
                continue

            data = _parse_storyboard_json(raw_text)
            if data is _INVALID_JSON_SENTINEL:
                err_msg = "输出不是合法JSON"
                continue

            # 校验前预处理：用 puzzle 专用 auto-fix
            segments = data.get("segments")
            if isinstance(segments, list) and len(segments) == len(chunk_nodes):
                for si, seg in enumerate(segments):
                    seg["voiceover"] = _auto_fix_puzzle_voiceover(
                        seg.get("voiceover", ""), chunk_nodes[si])

            ok, err_msg = _validate_puzzle_chunk(data, chunk_nodes)
            if ok:
                chunk_segments = _finalize_chunk_segments(data, chunk_nodes)
                all_segments.extend(chunk_segments)
                commentary.chunks_succeeded += 1
                success = True
                break

            # 逐段修复（puzzle 专用，不用残局版 _repair_failed_segments）
            if isinstance(segments, list) and len(segments) == len(chunk_nodes):
                for si, seg in enumerate(segments):
                    node = chunk_nodes[si]
                    original_vo = seg.get("voiceover", "")
                    fixed_vo = _auto_fix_puzzle_voiceover(original_vo, node)
                    if fixed_vo != original_vo:
                        seg["voiceover"] = fixed_vo
                data["segments"] = segments
                repaired_ok, _ = _validate_puzzle_chunk(data, chunk_nodes)
                if repaired_ok:
                    chunk_segments = _finalize_chunk_segments(data, chunk_nodes)
                    all_segments.extend(chunk_segments)
                    commentary.chunks_succeeded += 1
                    success = True
                    break

        if not success:
            Logger.warn(f"  Puzzle块{chunk_idx + 1}结构化生成失败，回退文本模式")
            commentary.fallback_used = True
            text_output = _generate_chunk_fallback(json_prompt)
            fallback_parts = _split_fallback_text(text_output, chunk_nodes) if text_output else {}

            chunk_segments = []
            for node in chunk_nodes:
                nid = node["id"]
                if nid in fallback_parts:
                    voice = fallback_parts[nid]
                elif text_output:
                    voice = text_output[:MAX_CHARS] if nid == chunk_nodes[0]["id"] else node.get("san", f"第{nid}步")
                else:
                    voice = "这一步继续推进战术思路，配合前后手形成压力，为关键手兑现战术效果做铺垫。"
                surface_ok, _ = _validate_puzzle_voiceover_surface(voice)
                if not surface_ok:
                    voice = "这一步继续推进战术思路，配合前后手形成压力，为关键手兑现战术效果做铺垫。"
                chunk_segments.append(StoryboardSegment(
                    id=nid,
                    sub_endgame="",
                    voiceover=voice,
                    pacing=normalize_pacing(node.get("suggested_pacing", "normal")),
                ))
            all_segments.extend(chunk_segments)

    commentary.segments = all_segments

    # 跨段去重
    if all_segments and not commentary.fallback_used:
        try:
            _dedupe_across_segments(all_segments)
        except Exception:
            pass

    # 双关键点强约束（§3.2）：确保关键手 segment 同时覆盖「机理」与「落地」。
    # 去重之后执行，避免去重误删刚补上的关键点内容。
    try:
        keypoint_skeleton = build_puzzle_keypoint_skeleton(storyboard)
        if keypoint_skeleton:
            covered = _enforce_puzzle_keypoints(
                all_segments, nodes, keypoint_skeleton, backend)
            if not covered:
                Logger.warn("  未能定位关键手节点，双关键点强约束跳过")
        else:
            Logger.warn("  无主标签骨架，双关键点强约束跳过")
    except Exception as e:
        Logger.warn(f"  双关键点强约束执行异常: {e}")

    commentary.raw_text = "\n".join(
        f"第{seg.id}步：{seg.voiceover}" for seg in all_segments
    )

    # Puzzle 开场白：基于骨架的半模板，稳定不依赖 LLM
    try:
        kp_for_intro = build_puzzle_keypoint_skeleton(storyboard)
        if kp_for_intro:
            commentary.opening = _compose_puzzle_intro(kp_for_intro, storyboard)
    except Exception:
        pass

    # Puzzle 不生成 summary（留在空字符串）
    from src.llm_backend import release_backend
    try:
        release_backend()
    except Exception:
        pass

    status = "正常" if not commentary.fallback_used else f"部分回退({commentary.chunks_succeeded}/{total_chunks})"
    Logger.success(f"Puzzle 解说生成完成: {len(all_segments)} 段, {status}")
    return commentary
