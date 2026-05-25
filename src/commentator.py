from src.common import Logger
from dotenv import load_dotenv
import ollama
import chess
import os
import re

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
CHUNK_SIZE = 4
MAX_CHARS = 1800

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


def _build_chunk_prompt(header: str, chunk_nodes: list, chunk_idx: int, total_chunks: int, example: str) -> str:
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
    parts.append("")

    for node in chunk_nodes:
        node_id = node["id"]
        star = "★" if node["is_critical"] else "·"
        phase_label = f" — {node['phase']}" if node.get("phase") else ""
        phase_hint = node.get("phase_hint", "")
        sub_name = node.get("sub_endgame_name", "")
        forbidden = node.get("forbidden_concepts", [])
        allowed = node.get("allowed_concepts", [])

        parts.append(f"{star}第{node_id}步：{node['moves']}（{node['move_count']}着, {node['turn']}）{phase_label}")

        if node.get("endgame_changed"):
            parts.append(f"  ⚠ 残局类型已切换为: {sub_name}，必须使用新类型的概念体系")
        if sub_name:
            parts.append(f"  【当前残局】{sub_name}")
        if allowed:
            parts.append(f"  【允许概念】{'、'.join(allowed)}")
        if forbidden:
            parts.append(f"  【禁止概念】{'、'.join(forbidden)}")

        if node.get("situation_before") and node["is_critical"]:
            parts.append(f"  局面特征：{node['situation_before']}")
        if node.get("transition_summary") and node["is_critical"]:
            parts.append(f"  局面变化：{node['transition_summary']}")
        if node.get("process_summary") and node["is_critical"]:
            parts.append(f"  过程：{node['process_summary']}")
        if node.get("teaching_focus"):
            parts.append(f"  教学重点：{node['teaching_focus']}")
        if node.get("phase_milestone"):
            parts.append("  阶段提示：这一节点承担阶段转换作用")

        if node["is_critical"]:
            eval_delta = node.get("eval_delta")
            if eval_delta is not None:
                if eval_delta < -200:
                    parts.append("  局势判断：重大失误！优势大幅缩水")
                elif eval_delta < -50:
                    parts.append("  局势判断：略有瑕疵，走得不够精确")
                elif eval_delta > 50:
                    parts.append("  局势判断：好棋！出乎意料地扩大了优势")
                else:
                    parts.append("  局势判断：稳健的行棋，维持当前局势")

            if node.get("trap"):
                parts.append(f"  陷阱提示：如果走错会{node['trap']}")
            if node.get("counterfactual_hint"):
                parts.append(f"  反事实提示：{node['counterfactual_hint']}")
            parts.append(f"  战略意图：{phase_hint or '关键转折点，局面发生实质性改变'}")
            parts.append("  输出要求：至少写2-3句，必须讲清楚为何局面被推进或被守住")
        else:
            if node.get("king_change"):
                parts.append(f"  王位变化：{node['king_change']}")
            if node.get("same_position"):
                parts.append("  结果判定：起止局面相同，属于反复试探，没有实质突破")
            parts.append(f"  战略意图：{phase_hint or '等招调整，改善子力位置，等待对方破绽'}")
            if node["move_count"] >= 3:
                parts.append(f"  输出要求：列出这{node['move_count']}着中的1-2个关键转折位置，然后概括目的")
            else:
                parts.append("  输出要求：不能只说来回移动，必须解释调车是在测试什么、控制什么或等待什么")

        parts.append("")

    return "\n".join(parts)


def _strip_thinking(text: str) -> str:
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    return text.strip()


def _generate_chunk(model: str, prompt: str) -> str:
    prompt_no_think = prompt + "\n/no_think"

    for use_think in (True, False):
        try:
            pieces = []
            kwargs = {"model": model, "prompt": prompt_no_think, "stream": True,
                      "options": {"temperature": 0.2}}
            if use_think:
                kwargs["think"] = False
            for chunk in ollama.generate(**kwargs):
                piece = chunk.response if hasattr(chunk, "response") else chunk.get("response", "")
                if piece:
                    pieces.append(piece)
            text = "".join(pieces)
            if text:
                return _strip_thinking(text)
        except Exception as e:
            Logger.warn(f"  ollama{' think=False' if use_think else ''} 异常: {type(e).__name__}: {e}")
    return ""


def generate(board: chess.Board, storyboard: dict) -> str:
    nodes = storyboard.get("nodes", [])
    if not nodes:
        Logger.warn("分镜数据为空，无法生成解说")
        return ""

    node_count = len(nodes)
    total_chunks = max(1, (node_count + CHUNK_SIZE - 1) // CHUNK_SIZE)
    Logger.info(f"生成解说: {node_count} 节点 → {total_chunks} 块 (模型:{OLLAMA_MODEL})")

    header = _build_header(storyboard)
    example = _get_example(storyboard.get("endgame_name", "残局"))

    all_parts = []
    for chunk_idx in range(total_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, node_count)
        chunk_nodes = nodes[start:end]

        prompt = _build_chunk_prompt(header, chunk_nodes, chunk_idx, total_chunks, example if chunk_idx == 0 else "")
        Logger.info(f"  [{chunk_idx + 1}/{total_chunks}] 节点{chunk_nodes[0]['id']}-{chunk_nodes[-1]['id']} (提示词{len(prompt)}字)")

        result = _generate_chunk(OLLAMA_MODEL, prompt)
        if not result:
            Logger.warn(f"  块{chunk_idx + 1}失败，跳过")
            continue

        if len(result) > MAX_CHARS:
            Logger.warn(f"  块{chunk_idx + 1}过长({len(result)}>{MAX_CHARS})，限长重试")
            prompt_short = prompt + f"\n\n每步严格控制在1-3句、总共不超过{MAX_CHARS//2}字。重新输出。"
            retry = _generate_chunk(OLLAMA_MODEL, prompt_short)
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
