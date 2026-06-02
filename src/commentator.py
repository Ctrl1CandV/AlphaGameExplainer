from src.common import Logger, StoryboardSegment, StoryboardVisuals, StoryboardArrow
from src.common import GeneratedCommentary, ALLOWED_PACING, ALLOWED_ARROW_COLORS
from src.common import is_valid_square_name, normalize_pacing
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

    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            return False, f"第{i+1}个segment不是对象"
        node = chunk_nodes[i]

        seg_id = seg.get("id")
        if not isinstance(seg_id, int):
            return False, f"segment[{i}]的id无效"
        if seg_id != node["id"]:
            seg["id"] = node["id"]  # 自动修正，不阻塞

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
    except Exception as e:
        pass
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
            pass
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
    """LLM 总结失败时的纯中文兜底总结。

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
    commentary.backend = backend.name

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
                    repaired_ok, repaired_err = _validate_storyboard_chunk(repaired, chunk_nodes)
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
                    visuals=_build_visuals_from_node(node),
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

    header = _build_header(storyboard)
    example = _get_example(storyboard.get("endgame_name", "残局"))

    all_parts = []
    for chunk_idx in range(total_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, node_count)
        chunk_nodes = nodes[start:end]

        prompt = _build_chunk_prompt(header, chunk_nodes, chunk_idx, total_chunks, example if chunk_idx == 0 else "")

        result = _generate_chunk_fallback(prompt)
        if not result:
            continue

        if len(result) > MAX_CHARS:
            prompt_short = prompt + f"\n\n每步尽量写成2-4句，但总共不超过{MAX_CHARS//2}字。重新输出。"
            retry = _generate_chunk_fallback(prompt_short)
            if retry:
                result = retry
            else:
                result = result[:MAX_CHARS]

        all_parts.append(result)

    if not all_parts:
        return ""

    final = "\n".join(all_parts).strip()
    return final
