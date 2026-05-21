from src.common import Logger, CHINESE_PIECE
from dotenv import load_dotenv
import ollama
import chess
import os
import re

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


_EXAMPLE_BY_ENDGAME = {
    "单车杀王": (
        "第1步：Ra5+。白车移至a5将军，画出了第一条控制线——黑王被限制在第5排以上，活动空间开始缩小。\n"
        "第2步：Kd4→Kc4→Kd3→Ke3（4着）。白王稳步向中心推进，与车形成配合之势，逐步压缩黑王活动范围。\n"
        "第3步：Rh4。关键一手！车横向移至h4，与白王形成横排对王——黑王被锁定在棋盘边缘，再无法回到中心。\n"
        "第4步：Kf2→Kd3→Ke3→Ra4→Rh4→Ke1→Kd2→Kc3（8着）。双方反复争夺对王位置，白方用车等招一步步收紧包围圈。\n"
        "第5步：Ra3#。车在底线将军，黑王无路可逃，完成将杀。\n"
        "总结：单车杀王的核心是盒子法和对王配合——用车画控制线→王车合围压缩→在边线将杀，关键是不给黑王逃脱的机会。"
    ),
    "车兵对车": (
        "第1步：Rc1。白车退至c1建立菲利多防线——白王守住兵前的关键格e3，车在后方控制c线，黑方无法正面突破。\n"
        "第2步：Kf6→Ke6→Kd5（3着）。黑王向白兵逼近，试图将白王挤出防线，双方围绕兵前方的关键格展开争夺。\n"
        "第3步：Ra1+。黑车突然从侧翼将军！白王被迫离开d线，菲利多防线被正面打破，局面出现转折。\n"
        "第4步：Kc3→Kd2→Ke3→Ra3→Rb3（5着）。白王被迫后退，黑车封锁第3排切断白王与兵的联系，白兵彻底暴露。\n"
        "第5步：Rxb3。黑车吃掉白兵，局面转为单车杀王，黑方确立胜势。\n"
        "总结：车兵对车的关键是弱方建立菲利多防线。白王守兵前、车在后方骚扰是标准防守；防线一旦被侧翼突破，兵失去保护则必败。"
    ),
    "单兵残局": (
        "第1步：Kd4。白王抢占关键格d4——正对黑王形成对王，黑王被迫后退，为兵推进扫清障碍。\n"
        "第2步：Kd6→Ke5（2着）。黑王被迫退至d6防守，白王牢牢占据关键格，兵可以安全前进。\n"
        "第3步：e4。白兵在王的保护下开始推进，向升变格e8迈出第一步。\n"
        "第4步：Ke7→Kd6→Ke5→Kf5→Ke6（5着）。双方王围绕兵前空间反复争夺，白方始终保持对王优势，黑王无法阻挡。\n"
        "第5步：e8=Q+。兵到达底线升变为后！局面转为单后杀王，白方必胜。\n"
        "总结：单兵残局的核心是对王与关键格。占据关键格→保护兵推进→升变取胜，关键在于不让对方王形成对王防御。"
    ),
    "单后杀王": (
        "第1步：Qf6。白后从远处控制黑王逃跑路线，将黑王限制在棋盘右下角区域。\n"
        "第2步：Kc6→Kd5→Ke4（3着）。白王向黑王稳步靠近，准备配合后完成合围。\n"
        "第3步：Qg7。关键一手！后将黑王锁定在边线，保持安全距离避免逼和。\n"
        "第4步：Kd3→Ke4→Qf7→Qg6（4着）。白王继续逼近，后调整位置收紧包围圈，黑王被压向角落。\n"
        "第5步：Qb7#。后与白王配合在底线将杀，黑王困在a8无路可逃。\n"
        "总结：单后杀王的关键是保持后的安全距离避免逼和，用己方王配合后逐步将对方王逼至边线或角落将杀。"
    ),
}

_EXAMPLE_FALLBACK = (
    "第1步：Nf3。白马跳至f3控制中心d4和e5格，这是开局阶段的标准发展。\n"
    "第2步：e5→d6→Nc6（3着）。黑方在中心展开反击，用兵和马争夺中心空间。\n"
    "第3步：Bg5。关键一手！象牵制黑方f6马，间接削弱了黑方对d5格的控制。\n"
    "第4步：Qe7→O-O-O→Kb8（3着）。黑方完成后翼易位，王转移到安全位置，准备展开反击。\n"
    "第5步：Qa4#。后将杀！黑王被困在b8无路可逃。\n"
    "总结：先控制中心，再展开子力，最后集中火力发动攻击完成将杀。"
)


def _get_example(endgame_name: str) -> str:
    return _EXAMPLE_BY_ENDGAME.get(endgame_name, _EXAMPLE_FALLBACK)


def generate(board: chess.Board, storyboard: dict) -> str:
    prompt = _build_prompt(board, storyboard)
    Logger.info("提示词如下：" + prompt)
    Logger.info("调用 Ollama 生成解说...")
    try:
        result = _try_generate(prompt, len(storyboard.get("nodes", [])))
        Logger.success(f"解说生成完成 ({len(result)} 字符)")
        return result.strip()
    except Exception as e:
        Logger.error(f"Ollama 调用失败: {e}")
        raise


def _build_prompt(board: chess.Board, storyboard: dict) -> str:
    endgame_name = storyboard.get("endgame_name", "残局")
    motifs = storyboard.get("motifs", [])
    mistakes = storyboard.get("mistakes", [])
    nodes = storyboard.get("nodes", [])
    role_summary = storyboard.get("role_summary", "")
    concept_binding = storyboard.get("concept_binding", [])
    hard_constraints = storyboard.get("hard_constraints", [])
    compact_mode = storyboard.get("compact_mode", False)
    node_count = len(nodes)
    target_length = storyboard.get("target_length", "800-1100字")

    parts = [
        "你是专业的国际象棋残局教练。请输出专业、严谨、以棋理为核心的中文解说，纯文本输出。",
        "要求：只依据给定走法和局面信息解说，禁止虚构剧情，禁止空泛比喻。你的任务是解释局面变化，而不是自由创作。",
        "",
        f"【残局类型】{endgame_name}",
        f"【核心策略】{storyboard.get('context', '残局局面分析')}",
    ]
    if motifs:
        parts.append(f"【关键概念】{'、'.join(motifs)}")
    if mistakes:
        parts.append(f"【易犯错误】{'、'.join(mistakes)}")

    if role_summary:
        parts.append(f"【攻守角色】{role_summary}")
    if concept_binding:
        parts.append(f"【概念绑定】{'；'.join(concept_binding)}")
    if hard_constraints:
        parts.append(f"【事实约束】{'；'.join(hard_constraints)}")

    parts.extend([
        "",
        "【解说规则】",
        f"- 下面提供的是分镜节点，不是原始每一着。你只能输出第1步到第{node_count}步这{node_count}个节点，绝不能额外扩写成更多步",
        "- 必须完整覆盖第1步到最后一步，不得跳号、漏步、合并步号",
        "- 每一步都按这个顺序组织：走法是什么 → 局面变化是什么 → 为什么这样走有用 → 这一段的残局教学点是什么",
        "- 必须优先使用提供的局面变化摘要、教学重点和战略意图，不能自行发明不存在的突破、升变或防线",
        "- 若残局概念与事实约束冲突，一律以事实约束为准",
        "- ★关键步骤：详细分析局面发生了什么实质变化",
        "- ·多着调整步骤：概括整段机动的目的，但仍要点出关键轨迹和为什么有用",
        "- 车只能描述为控制纵线/横线/关键格，绝不能写成对角线、斜线或斜向牵制",
        "- 如果节点起止局面相同，只能解释为反复试探、等招或调车，不得写成突破",
        "- 禁止使用引擎术语：评估值、分数、厘兵、半着、DTM、mate in N",
        f"- 全文以{target_length}为宜；长线残局可以比短残局更充分，但不要空话重复",
        "- 最后用一句话总结该残局类型的核心规律",
        "",
        "【真实对局分镜】（请严格基于以下提供的数据生成解说，绝不能虚构走法！）：",
        "",
    ])

    if compact_mode:
        parts.extend([
            "【长线残局模式】",
            "- 普通节点通常写1-2句；阶段切换节点和关键节点可以写2-3句",
            "- 不要把所有节点都压成一句话，尤其要保留阶段变化、教学重点和关键转折的细节",
            "- 如果某个普通节点承担了明显的阶段转换或教学任务，也要适当展开，而不是只作一句带过",
            "",
        ])

    if nodes:
        for node in nodes:
            node_id = node["id"]
            star = "★" if node["is_critical"] else "·"
            phase_label = f" — {node['phase']}" if node.get("phase") else ""
            phase_hint = node.get("phase_hint", "")

            # 无论是否关键步，都必须把真实走法喂给LLM
            move_text = node.get("moves_display", node["moves"]) if compact_mode else node["moves"]
            parts.append(f"{star}第{node_id}步：{move_text}（{node['move_count']}着, {node['turn']}）{phase_label}")
            if node.get("actor_role"):
                parts.append(f"  当前行动方身份：{node['actor_role']}")
            if (not compact_mode) or node["detail_level"] == "high":
                parts.append(f"  局面特征：{node.get('situation_before', '')}")
            parts.append(f"  局面变化摘要：{node.get('transition_summary', '')}")
            if node.get("teaching_focus"):
                parts.append(f"  教学重点：{node['teaching_focus']}")
            if node.get("phase_milestone"):
                parts.append("  阶段提示：这一节点承担阶段转换作用，解说时要点明目标为什么发生变化")

            if node["is_critical"]:
                eval_delta = node.get("eval_delta")
                if eval_delta is not None:
                    if eval_delta < -200:
                        parts.append(f"  局势判断：重大失误！优势大幅缩水")
                    elif eval_delta < -50:
                        parts.append(f"  局势判断：略有瑕疵，走得不够精确")
                    elif eval_delta > 50:
                        parts.append(f"  局势判断：好棋！出乎意料地扩大了优势")
                    else:
                        parts.append(f"  局势判断：稳健的行棋，维持当前局势")

                if node.get("trap"):
                    parts.append(f"  陷阱提示：如果走错会{node['trap']}")
                if node.get("counterfactual_hint"):
                    parts.append(f"  反事实提示：{node['counterfactual_hint']}")
                parts.append(f"  战略意图：{phase_hint or '关键转折点，局面发生实质性改变'}")
                parts.append("  输出要求：这一节点至少写2-3句，不能只复述走法，必须讲清楚为何局面被推进或被守住")

            else:
                king_change = node.get("king_change", "")
                situation_after = node.get("situation_after", "")
                if king_change:
                    parts.append(f"  王位变化：{king_change}")
                if situation_after and ((not compact_mode) or node["detail_level"] == "high"):
                    parts.append(f"  终了局面：{situation_after}")
                if node.get("same_position"):
                    parts.append("  结果判定：起止局面相同，这一段属于反复试探，没有形成实质突破")
                if node.get("counterfactual_hint") and (node["detail_level"] == "high" or node.get("phase_milestone")):
                    parts.append(f"  反事实提示：{node['counterfactual_hint']}")
                parts.append(f"  战略意图：{phase_hint or '等招调整，改善子力位置，等待对方破绽'}")
                parts.append("  输出要求：这一节点不能只说来回移动，必须解释这些调车是在测试什么、控制什么或等待什么；若是阶段切换节点，允许写2句")

            parts.append("")

    return "\n".join(parts)


def _try_generate(prompt: str, expected_steps: int = 0) -> str:
    for attempt in range(3):
        if attempt > 0:
            Logger.warn(f"解说格式异常，重试 (第{attempt}次)")

        resp = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={
                "temperature": 0.35,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.12
            },
        )
        commentary = resp.get("response", "")
        if not commentary:
            continue

        step_ids = {int(x) for x in re.findall(r"第\s*(\d+)\s*步", commentary)}
        if expected_steps > 0 and step_ids == set(range(1, expected_steps + 1)):
            return commentary
        if expected_steps == 0 and step_ids:
            return commentary
        Logger.warn("步号不完整，重试")

    Logger.warn("多次重试仍格式不完整，使用最后一次结果")
    return commentary
