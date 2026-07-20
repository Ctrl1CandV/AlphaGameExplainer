"""
Puzzle战术讲解解说生成

从commentator.py拆分而来。本模块只含Puzzle链路特有的逻辑：
分层讲解指令、JSON header 构建、双关键点骨架/评分/模板/润色、关键手诊断与强约束执行、自动修复与开场白模板
通用工具已迁出到text_filters / grammar / validators / json_utils，本模块通过公共名引用
"""
from src.commentator.text_filters import (
    strip_thinking, reduce_cliches_puzzle, strip_coordinates, digits_to_cn,
    dedupe_across_segments, expand_inline_brackets, safe_puzzle_seed_text,
)
from src.commentator.grammar import build_puzzle_chunk_grammar, PUZZLE_PLAIN_CN_GRAMMAR, build_retry_prompt
from src.commentator.validators import validate_puzzle_chunk, validate_puzzle_voiceover_surface
from src.commentator.examples import commentary_example_mode, get_commentary_example
from src.commentator.json_utils import parse_single_segment, INVALID_JSON_SENTINEL
from src.infra.llm_backend import create_backend_from_env
from src.common import GeneratedCommentary, Logger
from src.analysis.themes_kb import get_theme
from typing import Optional
import hashlib
import re

# Puzzle战术讲解解说生成
def _get_depth_instruction(rating: int) -> str:
    """
    三档分层：<1500 / 1500-2200 / >2200
    三档在内容焦点、句式约束、字数预算、可讲深度上做硬区分，避免趋同
    """
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
            "- 战术嗅觉：如何在实战中发现这类机会？从哪个已提供的信号看出战术存在？\n"
            "- 计算深度：只解释程序已给出的关键变化和确定结果，不自行补算分支\n"
            "- 相关战术：这个战术与已选协同战术有什么关联？本步的战术本质是什么？\n"
            "- 每步100-180字，可深入拆解已验证线路，适合有一定基础的棋手"
        )

def _build_puzzle_json_header(storyboard: dict) -> str:
    """ 战术分析专家人设 + 四层框架要求 + 标签定义注入 + depth_instruction """
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
        "2. 焦点突出：每段优先讲清两件事——核心战术的本质，以及它在本步如何具体落实；其余信息点到为止",
        "3. 事实优先：程序标注的主主题、关键手和确定结果不可改写；信息不足时不自行补算变着",
        "4. 叙事自然：用连贯的段落串联信息，不要逐条罗列层号或编号词",
        "5. 具体表达：首句直接写棋子动作、战术问题或局面变化，每段至少解释一个已提供的具体因果",
        "6. 避免空话：不用「看似平淡实则」「胜利的天平」「致命一击」或泛泛的「为后续做准备」；相邻段不要重复同一开头和比喻",
        "",
        f"【战术主题】{tactic_name}",
    ]

    if theme_defs:
        lines.extend([
            "",
            "【标签定义】以下是你应该围绕讲解的战术概念，请融入讲解中：",
            theme_defs,
        ])

    # 主战术深度锚点：单独拎出主标签的机理与关键手，作为讲透的抓手
    # 信息来自知识库，模型据此把抽象概念落地到本局这几步，而非泛泛而谈
    primary_key = tactic_focus.get("primary_theme", "")
    if primary_key:
        try:
            pt = get_theme(primary_key)
        except Exception:
            pt = None
        if pt:
            anchor = [
                "",
                f"【主战术深度锚点】本题核心战术是【{pt['cn']}】，请把它讲透，做到以下三层：",
                f"  1. 机理：{pt['cn']}为什么能成立——{pt.get('definition', '')}",
            ]
            if pt.get("key_move_signal"):
                anchor.append(f"  2. 关键手：在本局，{pt['cn']}的关键手表现为——{pt['key_move_signal']}。请结合给定走法，明确指出哪一步是这个关键手，它具体做了什么。")
            if pt.get("typical_consequence"):
                anchor.append(f"  3. 结果：{pt['cn']}得手后的典型收益是——{pt['typical_consequence']}。请说明本局实际兑现了什么（净赢的子力／被控的线路／对方的困境）。")
            anchor.append("至少要有一处把这个战术概念与本局的具体走法结合起来讲清楚，不要只复述定义，也不要只描述走法，要让观众看懂「概念如何在这盘棋里发生」。")
            
            # 联动叙事：核心战术与次要战术存在辅助关系时，提示组合讲解
            synergy = tactic_focus.get("synergy_themes", [])
            if synergy:
                anchor.append(
                    f"本题还涉及与【{pt['cn']}】相互辅助的战术：{'、'.join(synergy)}。"
                    f"请把它们作为{pt['cn']}的配合手段串起来讲——说明它们如何服务于核心战术，"
                    "而不是各讲各的、平行罗列。"
                )

            # 关键手定位：直接把哪一步是关键手 + 理由喂给模型，避免模型把第一步将军/吃子讲成核心战术
            key_idx = tactic_focus.get("key_move_idx") or 0
            key_san = tactic_focus.get("key_move_san", "") or ""
            key_reason = tactic_focus.get("key_move_reason", "") or ""
            if key_idx and key_san:
                anchor.append(
                    f"【已算出的关键手】本题核心战术的关键手是第{key_idx}手 {key_san}。"
                    f"理由：{key_reason or '由棋盘事实算出'}。"
                    f"讲解时务必让观众看到「这一步才是核心」，"
                    f"不要把任何其他子（如纯将军、过渡吃子）误讲成核心战术。"
                )
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

    # 前置注入：注入解题开局双方子力盘点，作为不可改写的全局事实
    # 此前storyboard已算出 white_material/black_material，却从未进入正文生成
    white_material = storyboard.get("white_material", "")
    black_material = storyboard.get("black_material", "")
    if white_material and black_material:
        lines.extend([
            "",
            "【解题开局子力（不可改写事实，禁止自行增减棋子种类和数量）】",
            f"- 白方：{white_material}",
            f"- 黑方：{black_material}",
            f"- 解题方：{puzzle_side}；防守方：{defending_side}",
            "只能依据上面给出的子力讲解，禁止凭空增加或减少后、车、象、马、兵；"
            "各步的子力得失请严格依据每个节点给出的[核心]子力结论，不要自行计算净赢多少。",
        ])

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
            "- 在关键步讲清程序已提供的战术依据、强制性和确定结果",
            "- 只有节点明确提供替代变化时才能分析变着；未提供时不得自行编造线路",
            "- 每步100-180字，允许深入拆解已验证信息，适合有一定基础的棋手",
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
    """ 将SAN走法中的棋子字母转为中文，如'Nf6'→'马'，无棋子字母时返回'兵' """
    if moves_str.startswith("O-O-O"):
        return "后翼易位"
    if moves_str.startswith("O-O"):
        return "王翼易位"
    for piece in ('N', 'B', 'R', 'Q', 'K'):
        if moves_str.startswith(piece):
            return _SAN_PIECE_MAP[piece]
    return "兵"

def _puzzle_example_role(chunk_nodes: list) -> str:
    if any(node.get("is_core_theme_key_move") for node in chunk_nodes):
        return "climax"
    if chunk_nodes and chunk_nodes[0].get("id") == 1:
        return "setup"
    return "resolution"

def _build_puzzle_chunk_prompt(
        header: str, chunk_nodes: list, chunk_idx: int,
        total_chunks: int, primary_theme: str = ""
    ) -> str:
    """ 构建puzzle分块prompt """
    is_last = (chunk_idx == total_chunks - 1)
    lines = [header]

    use_example = chunk_idx == 0 or any(node.get("is_core_theme_key_move") for node in chunk_nodes)
    if (use_example and commentary_example_mode() == "matched" and primary_theme):
        role = _puzzle_example_role(chunk_nodes)
        example = get_commentary_example("puzzle", primary_theme, role)
        if example:
            lines.extend([
                "【表达范例】",
                "只模仿信息密度、组织和口语节奏；范例中的棋子、战术和结果不是当前棋局事实。",
                example, "",
            ])

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
        lines.append(
            f"状态: {'将军' if node.get('is_check') else '非将军'}"
            f" | {'吃子' if node.get('is_capture') else '未吃子'}"
            f" | {'已将杀' if node.get('is_checkmate') else '未将杀'}"
        )

        # 确定性事实：吃掉的具体子力和对方应招数，让解说有硬料可写，挤掉套话
        captured = node.get("captured_piece_cn", "")
        if captured:
            lines.append(f"[核心] 吃掉的子力: 对方的{captured}")

        # 前置注入：本步不可改写的子力得失结论，这是根治手段——
        # 把"吃回/兑子/真净赢"的确定判断在生成前喂给模型，而不是等生成后再拦
        material_fact = node.get("material_fact", "")
        if material_fact:
            lines.append(f"[不可改写] 子力结论: {material_fact}")
        reply_count = node.get("legal_reply_count_after")
        if isinstance(reply_count, int) and not node.get("is_checkmate"):
            # 用中文数字表述，避免模型照搬阿拉伯数字被voiceover语法卡掉
            cn_num = "零一二三四五六七八九"[reply_count] if 0 <= reply_count < 10 else str(reply_count)
            if reply_count == 0:
                pass
            elif reply_count <= 3:
                lines.append(f"[核心] 走后对方仅剩{cn_num}个合法应招，回旋余地极小")
            elif reply_count <= 8:
                lines.append(f"[核心] 走后对方合法应招收缩到{cn_num}个，明显受限")

        # 核心材料
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

        # 参考材料
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

        # pacing提示
        pacing = node.get("suggested_pacing", "normal")
        if pacing in ("slow", "pause_before", "pause_after"):
            lines.append(f"节奏: {pacing} — 这是关键节点，请重点展开讲解")

        lines.append("")

    return "\n".join(lines)

def _score_puzzle_depth(text: str, kp: dict) -> bool:
    """ 关键手段落的深度校验：至少覆盖2到3类关键词，单一关键词不足以证明深度，必须同时包含至少两类 """
    cause_words = ("因为", "所以", "正是", "从而", "导致", "意味着", "因此")
    change_words = ("之前", "之后", "一旦", "不同于", "改变")
    constraint_words = ("迫使", "无法", "必须", "不能", "只能", "否则")
    categories = sum([
        any(w in text for w in cause_words),
        any(w in text for w in change_words),
        any(w in text for w in constraint_words),
    ])
    return categories >= 2

def _auto_fix_puzzle_voiceover(text: str, node: dict) -> str:
    """ puzzle专用自动修复：坐标清洗、标签标记删除、括号展开、轻量反套话、标点收敛 """
    fixed = text

    # 坐标兜底清洗
    fixed = strip_coordinates(fixed)
    fixed = digits_to_cn(fixed)
    fixed = fixed.replace("%", "").replace("％", "")

    # 删除标签标记泄漏
    fixed = re.sub(r"[【][^】]{1,20}[】]", "", fixed)

    # 括号展开：把括号内容融入句子，避免口播出现括号停顿
    fixed = expand_inline_brackets(fixed)

    # 不完整句子修复
    fixed = re.sub(r"(这步[^。]{0,6})。(实战|这是|这步|黑方|白方|面对|面对)", r"\1，\2", fixed)

    # 轻量反套话，不做形容词删除，保留'精准/精确'等战术语义词
    fixed = reduce_cliches_puzzle(fixed)

    # 标点收敛
    fixed = re.sub(r"[，,]{2,}", "，", fixed)
    fixed = re.sub(r"。{2,}", "。", fixed)
    fixed = re.sub(r"\s{2,}", " ", fixed)
    fixed = re.sub(r"[，、]+。", "。", fixed)
    fixed = re.sub(r"^[，、。]+", "", fixed)
    fixed = fixed.strip()

    return fixed


"""
Puzzle双关键点强约束，谜题链路必须且只须讲透两个关键点：
    关键点1（机理）：标签代表的战术策略是什么、为什么成立
    关键点2（落地）：该战术在本局如何兑现——哪步是关键手、做了什么、什么结果
其余效果可让步，但这两点必须覆盖，下方为骨架提取/评分/模板/润色四件套
"""

# 落地层「确定结果」判定词：解说命中其一即视为讲到了战术兑现的结果
_PUZZLE_RESULT_WORDS = (
    "赢", "得子", "得回", "多子", "失", "丢", "被迫", "无法",
    "困", "杀", "优势", "子力", "胜势", "制胜", "致胜"
)

_DIGIT_CN = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}

def _describe_key_move(node: dict) -> tuple:
    """ 从节点生成无坐标的关键手描述，返回 (描述句, 棋子中文名) """
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
    """
    定位关键手节点id
    优先级：关联到主标签的节点 → 首个将杀/将军/吃子节点 → 首个节点
    """
    if not nodes:
        return None
    # 1. 关联到主标签的节点
    if primary_key:
        for node in nodes:
            if node.get("related_theme") == primary_key:
                return node["id"]
    # 2. 首个将杀/将军/吃子节点
    for node in nodes:
        if (
            node.get("is_checkmate_after") or node.get("is_check") or node.get("is_capture")
        ):
            return node["id"]
    # 3. 兜底首个节点
    return nodes[0]["id"]

def build_puzzle_keypoint_skeleton(storyboard: dict) -> dict:
    """
    构建谜题双关键点骨架
    数据来源：主标签知识库字段（机理）+ 本局实际走法与净子力事实（落地）
    返回的骨架供评分器、模板、润色器共用；缺失主标签时返回空dict
    """
    nodes = storyboard.get("nodes", [])
    if not nodes:
        return {}

    tactic_focus = storyboard.get("tactic_focus", {})
    primary_key = tactic_focus.get("primary_theme", "")

    theme = None
    if primary_key:
        try:
            theme = get_theme(primary_key)
        except Exception:
            theme = None
    if not theme:
        return {}

    # 优先使用storyboard阶段已算好的关键手定位
    # 兜底才用旧的_resolve_key_move_idx
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
    aliases = [safe_puzzle_seed_text(a) for a in theme.get("aliases_cn", [])]
    tactic_cn = safe_puzzle_seed_text(theme.get("cn", primary_key))
    concept_words = [w for w in [tactic_cn] + aliases if w]

    # 深度层：局面证据 + 变化对比 + 对方困境（关键手段落专用）
    definition = safe_puzzle_seed_text(theme.get("definition", ""))
    consequence = safe_puzzle_seed_text(theme.get("typical_consequence", ""))
    key_move_signal = safe_puzzle_seed_text(theme.get("key_move_signal", ""))
    recognition = safe_puzzle_seed_text(theme.get("recognition", ""))

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
        "tactic_recognition": safe_puzzle_seed_text(theme.get("recognition", "")),
        # 关键点 2：战术如何在本局使用（落地）
        "key_move_idx": key_move_idx,
        "key_move_desc": safe_puzzle_seed_text(key_move_desc),
        "key_move_piece": key_move_piece,
        "key_move_signal": key_move_signal,
        "consequence": consequence,
        "actual_result": safe_puzzle_seed_text(actual_result),
        # 深度层：关键手段落专用，增强具体性
        "local_weakness": local_weakness,
        "before_after": before_after,
        "defender_problem": defender_problem,
    }

def _score_puzzle_keypoints(text: str, kp: dict) -> dict:
    """ 谜题双关键点覆盖评分，两个关键点都覆盖才pass=True，任一缺失即判不合格 """
    if not text or not kp:
        return {
            "kp1_covered": False, "kp2_covered": False,
            "pass": False, "issues": ["缺少文本或骨架"]
        }

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
    """
    谜题关键手节点的模板填空
    4句结构：机理 → 证据 → 变化 → 困境/结果
    保证纯模板下也100%覆盖双关键点 + 有具体棋理深度
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
        # 只有当句子本身没有主语时才补"对方"前缀。此前只判"对方"，导致
        # 以"己方/白方/黑方/双方"开头的 consequence（如 advantage 标签的
        # "己方以优势姿态进入战术阶段…"）被硬加前缀拼出"对方己方…"的病句。
        if defender_problem.lstrip().startswith(("对方", "己方", "白方", "黑方", "双方")):
            sent4 = f"{defender_problem}，{actual_result}"
        else:
            sent4 = f"对方{defender_problem}，{actual_result}"
    else:
        sent4 = f"最终{actual_result}"

    return f"{sent1}。{sent2}。{sent3}。{sent4}。"


def _polish_puzzle_voiceover(
    node: dict, kp: dict, prev_context: str, backend
) -> str:
    """ 谜题关键手节点的LLM润色 """
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
    return strip_thinking(
        backend.generate(prompt, grammar=PUZZLE_PLAIN_CN_GRAMMAR)
    ).strip()

def _compose_puzzle_intro(kp: dict, storyboard: dict) -> str:
    """
    谜题开场白模板：3套自然半模板轮换，稳定不依赖LLM
    字段来自keypoint_skeleton，不出现坐标/英文/Markdown
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
    surface_ok, surface_reason = validate_puzzle_voiceover_surface(seg.voiceover)
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
        surface_ok, _ = validate_puzzle_voiceover_surface(polished)
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


def _puzzle_post_process(commentary: GeneratedCommentary, all_segments: list,
                         nodes: list, storyboard: dict, backend) -> None:
    """Puzzle 后处理：跨段去重 + 双关键点强约束 + 开场白（原地修改 commentary）。

    对应原 generate_puzzle_structured 尾部行 2710-2741 的逻辑。
    """
    # 跨段去重
    if all_segments and not commentary.fallback_used:
        try:
            dedupe_across_segments(all_segments)
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


def _puzzle_fallback_wrapper(chunk_nodes: list, json_prompt: str) -> list:
    """generator 的 build_fallback_voiceover 回调适配器。

    **SPEC §8（2026-07-19）后已废弃**：内容级失败不再模板兜底，generator 改为
    标记 aborted 并中止本片。本函数保留仅为维持 CommentaryConfig 回调签名兼容，
    generator 不再调用它；若意外被调到，记录警告并返回空列表，绝不产出模板句。
    """
    Logger.warn("_puzzle_fallback_wrapper 在 SPEC §8 后不应被调用（已废弃），返回空列表")
    return []


def _puzzle_chunk_prompt_wrapper(storyboard: dict):
    """构造 Puzzle 版 build_chunk_prompt 闭包，适配 generator 的统一签名。"""
    primary_theme = (storyboard.get("tactic_focus", {}) or {}).get("primary_theme", "")

    def wrapper(header, chunk_nodes, chunk_idx, total_chunks, all_nodes):
        return _build_puzzle_chunk_prompt(
            header, chunk_nodes, chunk_idx, total_chunks, primary_theme)
    return wrapper


def generate_puzzle_structured(board, storyboard: dict) -> GeneratedCommentary:
    """Puzzle 战术讲解主入口。通过回调注入 Puzzle 专用函数到通用生成框架。"""
    from src.commentator.generator import generate_commentary, CommentaryConfig
    from src.commentator.grammar import build_puzzle_chunk_grammar
    from src.commentator.validators import validate_puzzle_chunk
    from src.infra.llm_backend import create_backend_from_env

    backend = create_backend_from_env()
    config = CommentaryConfig(
        build_header=_build_puzzle_json_header,
        build_chunk_prompt=_puzzle_chunk_prompt_wrapper(storyboard),
        build_grammar=build_puzzle_chunk_grammar,
        validate_chunk=validate_puzzle_chunk,
        auto_fix_voiceover=_auto_fix_puzzle_voiceover,
        repair_failed_segments=None,
        build_fallback_voiceover=_puzzle_fallback_wrapper,
        post_process=_puzzle_post_process,
    )
    return generate_commentary(storyboard, backend, config)
