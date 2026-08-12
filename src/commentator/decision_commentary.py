"""决策管线解说生成（ADR-020 阶段 7，对齐 endgame_commentary.py 结构）。

输入：decision_builder 产出的比较式 storyboard。
输出：GeneratedCommentary（开场 + 分段解说 + 总结）。

叙事弧（PLAN-009 阶段 7）：诊断 → 提问 → 反事实（一句话，轴 3 位置）
→ 计划甲 → 回溯 → 计划乙 → 对比 → 条件性建议（Tier A 追加实战实际
选择）。

设计要点：
- **无坐标无走法**（ADR-020 约束 5）：prompt 注入与解说全程不含具体着法
  与坐标——只讲战略/机理/趋势/代价（形态化中文）；
- **校验三层（P9 修订）**：① 复用 `validate_puzzle_voiceover_surface`
  表层硬闸；② 战略名提及；③ **独有事实校验**——各计划段必须命中自己
  的 `unique_facts`（防同质化真实防线——「提及」≠「有区分度」）；
- **措辞降级（P12）**：不说「大师的选择」，也不说「更多选了」——
  `provenance` 是**这一局**的实战续着（单局个例），说「更多」就把 n=1
  谎报成频率统计。实测：原措辞让 LLM 写出「实战中多数人选择推进悬兵」，
  凭一局断言多数人偏好，与 ADR-020 认识论边界冲突（08.04 修）。
  正确口径是「这一局的实战里走的是某条路」＋「只是一个例子，不是标准答案」。
  真要给频率事实，得接 `tools/pgn_plan_stats.py` 的多局统计（P17 设施），
  那是后续阶段的事；
- **ADR-018 对齐**：注入对比式表达范例锚定形态（不带内容）；
- **PLAN-008 教训**：禁用词不在指令里反复出现（负面提及强化被禁内容）；
- 段级失败不报废（该段不输出）；管线级失败 = 整片不出（P11/SPEC §8）。
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

try:  # 直接运行自检时补充项目根到 sys.path
    from src.commentator.text_filters import (
        clean_cjk_text,
        clean_summary_text,
        reduce_cliches,
        safe_decision_seed_text,
        strip_coordinates,
        strip_thinking,
    )
    from src.commentator.grammar import build_chunk_grammar
    from src.commentator.validators import validate_puzzle_voiceover_surface
    from src.common import GeneratedCommentary
    from src.infra.llm_backend import create_backend_from_env
except ModuleNotFoundError:
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    "..", "..")))
    from src.commentator.text_filters import (
        clean_cjk_text,
        clean_summary_text,
        reduce_cliches,
        safe_decision_seed_text,
        strip_coordinates,
        strip_thinking,
    )
    from src.commentator.grammar import build_chunk_grammar
    from src.commentator.validators import validate_puzzle_voiceover_surface
    from src.common import GeneratedCommentary
    from src.infra.llm_backend import create_backend_from_env

logging.getLogger("chess.engine").setLevel(logging.CRITICAL)

CHUNK_SIZE = 8          # 决策解说节点少（5 个叙事单元）——单 chunk
MAX_RETRIES = 1

# 单段口播字数硬上限（08.04 修）。表层硬闸只查字符与思考泄漏、不查长度，
# 故长度必须在本模块单独拦。取 130 而非 prompt 里写的 110：prompt 是软目标，
# 硬闸留 20 字余量，避免把「稍微超一点但内容合格」的段判废造成无谓重试。
# 依据：ChatTTS 是 GPU 逐字合成，实测约 27s/96 字，130 字约 36s——单段可接受；
# 无上限时总结/对比段常破 150 字，是阶段 8a TTS 卡顿的上游根因。
MAX_VOICEOVER_CHARS = 130

# 节点类型 → 图例中文名（prompt 末尾的 id 图例用）。
#
# 必须按实际节点动态生成图例，不能写死「0=开场，1/2=各计划，3=对比，4=总结」
# （08.04 修的两个内容级 bug 的共同根因）：节点 id 由 `_build_decision_nodes`
# 按实际可行计划数动态编号，单线退化时（可行计划 1 个，P11 允许且算成功产出）
# 节点只有 [opening(0), plan(1), summary(2)]——没有对比节点。此时写死图例会
# 告诉 LLM「2=第二个计划」，于是模型凭空编造一个可行性闸没通过的计划写进
# id 2，而 post_process 按 node 类型把它装进 commentary.summary。
# 实测表现：总结段与计划段逐字重复 + 开场宣称「有两个方向」而实际只有一个
# （讲了不存在的计划 = 教错棋，违反 ADR-015「程序算事实、LLM 只表达」边界）。
_NODE_ROLE_CN = {
    "opening": "开场",
    "plan": "计划",
    "rejected": "为什么不那样走",
    "compare": "对比",
    "summary": "总结",
}

# 轴 4 措辞限权（ADR-021 §措辞限权）：validator 可机判化。
# 正面锚词——对照段必须命中至少一个（缺失即判废）。
AXIS4_ANCHOR_WORDS = ("稍逊", "略差", "不如", "差了", "稍差", "稍弱", "逊色")
# 等强措辞硬错误短表（轴 4 对照段出现即判废——三词上限防枚举漏）。
AXIS4_FORBIDDEN_EQUAL = ("各有取舍", "各有侧重", "看你风格")
# 灾难定性硬错误（窗口内 ≤150 的对照不是败着）。
AXIS4_FORBIDDEN_DISASTER = ("败着", "就输了", "送子", "直接丢", "彻底丢")

# 维度中文名（与 decision_builder._DIM_CN 同源——独有事实校验用关键词）
_DIM_CN = {
    "opp_isolated_qside": "后翼孤立兵",
    "opp_isolated_center": "中心孤立兵",
    "opp_isolated_kside": "王翼孤立兵",
    "opp_backward": "后退兵",
    "passed_diff": "通路兵",
    "mover_pawns_past_mid": "兵过中线",
    "pawn_islands_diff": "兵岛",
    "open_files": "开放线",
    "half_open_own": "半开放线",
    "outposts": "前哨",
    "knight_bishop_diff": "轻子",
    "opp_king_exposure": "王暴露",
}

# 认识论边界与表达形态锚定（ADR-020 约束 4/6 + ADR-018 范例）
_HEADER_CONSTRAINTS = [
    "你是国际象棋战略教练，用比较式叙事讲解一个局面上真实存在的战略选择。",
    "程序提供的计划、趋势与结构事实是唯一事实来源；只讲其中有明确支撑的内容。",
    "不评价着法优劣之外的事；不用「引擎认为」「算法证明」类措辞（认识论边界）。",
    "不给具体走法与坐标——只讲战略意图、结构变化与代价。",
    "两条路线各有取舍——但必须讲出**具体差异**：甲带来的结构改变是什么，乙带来的又是什么，"
    "不要用「两条路各有千秋」「都有可取之处」这类空话收尾。",
    "叙事节奏：诊断局面 → 提出战略问题 → 反事实铺垫（一句话）→ 计划甲 → "
    "回到分岔点 → 计划乙 → 直接对比 → 条件性建议。",
    "对比式表达范例（只学形态，不抄内容）："
    "「后翼这条线的代价是兵链松动，换来的是对方后翼出现一个长期弱点；"
    "中心那条线则相反，兵形稳固但推进空间有限——两种取舍正好对应局面给出的两个答案。」",
]


def _route_plan_name(storyboard: dict, idx: int) -> str:
    routes = storyboard.get("routes", [])
    return routes[idx].get("name", f"计划{idx + 1}") if idx < len(routes) else "?"


def _trend_cn(trend: dict) -> str:
    """趋势中文摘要（形态化注入——无坐标）。"""
    parts = []
    for t in trend.get("trends", []) or []:
        dim = _DIM_CN.get(t.dimension, t.dimension)
        parts.append(f"{'增强' if t.direction == 'increasing' else '削弱'}{dim}")
    shift = trend.get("archetype_shift")
    if shift:
        parts.append(f"结构类型从{shift[0]}转为{shift[1]}")
    return "、".join(parts) if parts else "无明显趋势"


def _tradeoffs_cn(tm: dict) -> str:
    """代价中文摘要（承诺度/难度——无坐标）。"""
    parts = []
    if tm.get("pawn_moves"):
        parts.append(f"兵着承诺 {tm['pawn_moves']} 步")
    if tm.get("captures"):
        parts.append(f"兑子 {tm['captures']} 次")
    if tm.get("open_files_delta"):
        parts.append(f"开放线变化 {tm['open_files_delta']:+d}")
    if tm.get("weak_square_hint"):
        parts.append(f"末端结构：{tm['weak_square_hint']}")
    if tm.get("corridor_roots") is not None:
        parts.append(f"选择余地 {'宽' if tm['corridor_roots'] >= 3 else '窄'}"
                     f"（好着走廊 {tm['corridor_roots']}）")
    return "、".join(parts) if parts else "无明显代价"


def _strip_fact_numbers(fact: str) -> str:
    """剥离独有事实尾部的结构特征数值（「（0.75 vs 0.5）」）。

    unique_facts 形如「XX末端YY显著（0.75 vs 0.5）」——括号里的小数对比值
    是程序判定「显著」用的结构特征值，对观众无任何意义（不是步数、不是子力
    价值），照念会让解说冒出「零点七五对零点五」这类让人一头雾水的数字
    （08.11 用户实测反馈）。LLM 原样照抄 prompt 里给的数值。

    校验侧零风险：`_kw_matches` 按区域词（后翼/中心）+主体词（孤立兵/开放线）
    匹配，从不匹配数值部分——剥掉括号后「XX末端YY显著」照样能命中校验。

    只剥小数对比括号，不动「兵着承诺 4 步」这类有意义整数（观众能理解步数）。
    """
    import re
    # 「（0.75 vs 0.5）」「(0.2 vs 0.0)」——含小数点的对比括号整体剥掉
    return re.sub(r"[（(]\s*\d+\.\d+\s*vs\s*\d+\.\d+\s*[)）]", "", fact).strip()


def _baseline_cn(baseline) -> str:
    """反事实基线数值 → 中文紧迫性程度词（开场段注入用）。

    baseline 是引擎 eval 的 cp 整数（如 96、-20），开场段把「基线评估为
    {baseline}」照直注入，LLM 念成「基线评估为九十六」（08.11 用户实测反馈：
    一头雾水）。对观众有意义的是「这个局面差到什么程度」，不是数值本身。
    转成三档中文程度词：明显不利 / 相当差 / 略处下风（|cp|≥100/50/<50）。
    """
    if baseline is None:
        return "相当被动"
    v = abs(int(baseline))
    if v >= 100:
        return "明显不利"
    if v >= 50:
        return "相当被动"
    return "略处下风"


def _match_provenance_plan(board, storyboard: dict,
                           provenance_san: Optional[str]) -> Optional[str]:
    """实战续着（SAN）映射到计划名（Tier A——P12 措辞降级用）。

    续着首着 ∈ 哪个计划的 direction 候选集 → 该计划名。候选集为空或
    无法匹配 → None（该维度缺席，不阻塞）。
    """
    if not provenance_san:
        return None
    try:
        import chess
        from src.analysis.direction import direction_candidates
        moves = {board.san(m) for m in board.legal_moves}
        if provenance_san not in moves:
            return None
        mv = next(m for m in board.legal_moves if board.san(m) == provenance_san)
        for route in storyboard.get("routes", []):
            direction = route.get("direction", {})
            if not direction:
                continue
            cands = {board.san(m) for m in direction_candidates(board, direction)}
            if board.san(mv) in cands:
                return route.get("name")
    except Exception:
        return None
    return None


def _build_decision_nodes(storyboard: dict) -> List[dict]:
    """storyboard → 叙事弧节点列表（每叙事单元一个 node）。"""
    routes = storyboard.get("routes", [])
    nodes = [
        {"id": 0, "type": "opening",
         "archetype": storyboard.get("decision_point", {}).get("archetype", ""),
         "premise": storyboard.get("decision_point", {}).get("strategic_premise", ""),
         "baseline": storyboard.get("decision_point", {}).get("baseline"),
         "n_plans": len(routes)},
    ]
    for i, route in enumerate(routes):
        nodes.append({"id": i + 1, "type": "plan", "idx": i,
                      "name": route.get("name", "?"),
                      "mechanism": route.get("mechanism", ""),
                      "trend": route.get("trend", {}),
                      "tradeoffs": route.get("tradeoffs", {}),
                      "unique_facts": route.get("unique_facts", [])})
    if len(routes) >= 2:
        nodes.append({"id": len(routes) + 1, "type": "compare",
                      "names": [_route_plan_name(storyboard, i)
                                for i in range(len(routes))],
                      "axes": storyboard.get("comparison_axes", {})})
    nodes.append({"id": len(nodes), "type": "summary",
                  "names": [_route_plan_name(storyboard, i)
                            for i in range(len(routes))],
                  "provenance_plan": storyboard.get("provenance_plan")})
    # 轴 4 对照节点（ADR-021）：rejected_route 非空时在 summary 前插入。
    # axis_type=4 时 routes 只有 1 条（正选），compare 节点不生成（len<2）。
    # 对照段插在 plan 之后、summary 之前——叙事弧：
    #   诊断 → 正选计划 → 「为什么不那样走」→ 条件性建议。
    rejected = storyboard.get("rejected_route")
    if rejected and len(routes) == 1:
        # summary 的 id 让位给对照段——重新分配 id
        summary_node = nodes.pop()  # 取出刚加的 summary
        summary_id = summary_node["id"]
        nodes.append({
            "id": summary_id,
            "type": "rejected",
            "name": rejected.get("name", "次优选择"),
            "gap_level": rejected.get("gap_level", ""),
            "unique_facts": rejected.get("unique_facts", []),
            "primary_name": routes[0].get("name", "正选"),
        })
        summary_node["id"] = summary_id + 1
        nodes.append(summary_node)
    return nodes


def build_header(storyboard: dict) -> str:
    """人设 + 认识论边界 + 战略知识注入（无坐标无走法）。"""
    dp = storyboard.get("decision_point", {})
    parts = [
        "你是一位国际象棋战略教练。程序标注的战略计划、结构趋势与代价数据是唯一事实来源。"
        "只输出合法JSON，不加任何解释或markdown标记。",
        "",
        "你的讲解信条：",
    ] + ["- " + c for c in _HEADER_CONSTRAINTS]
    parts += [
        "",
        f"【局面类型】{dp.get('archetype', '中局')}",
        # KB theory/mechanism 是**带坐标的知识库原文**（如马洛齐的
        # 「c4+e4 双兵锁住 d5」），而本 prompt 同时要求「不给坐标」且成稿要过
        # 表层硬闸（无英文/数字）。原实现把原文照直注入，模型自然照抄坐标，
        # 后处理 strip_coordinates 只吃「字母+数字」，留下 `+` `-` `/` 与
        # 光秃秃的纵线字母（「d 兵」「c 线」），表层校验判不合格 → 重试耗尽 →
        # 整片按 §8 放弃。实测 maroczy 连续两轮均卡在此处（其 theory 坐标
        # 密度最高：c4+e4、d5、d5/b5 三处）。
        # 解法是在**注入前**把坐标转成区域词（与 puzzle 链路对 KB 种子文本
        # 调 safe_puzzle_seed_text 同一策略）：模型看到的就是可播的中文，
        # 不必依赖它「记得别抄坐标」，也不必靠后处理补救。
        f"【战略前提】{safe_decision_seed_text(dp.get('strategic_premise', ''))}",
        f"【反事实基线】若不做任何战略推进（等待），局面会"
        f"{_baseline_cn(dp.get('baseline'))}——这是紧迫性的度量，"
        "只在开场用一句话铺垫（说程度，不报数值）。",
    ]
    return "\n".join(parts)


def _plan_prompt_text(node: dict, is_first: bool) -> str:
    """单个计划节点的形态化事实（无坐标无走法）。"""
    lines = []
    if is_first:
        lines.append("【计划甲】")
    else:
        lines.append("【计划乙】——先回到分岔点，再展开这一条路线")
    lines.append(f"计划名：{node['name']}")
    # 同【战略前提】：KB mechanism 原文带坐标（「用 b4-b5 兵推进冲击对方
    # c 线兵」），注入前先转区域词，避免模型照抄后卡表层硬闸。
    lines.append(f"机理：{safe_decision_seed_text(node['mechanism'])}")
    trend = _trend_cn(node.get("trend", {}))
    if trend:
        lines.append(f"趋势：{trend}")
    tc = _tradeoffs_cn(node.get("tradeoffs", {}))
    if tc:
        lines.append(f"代价：{tc}")
    facts = node.get("unique_facts", [])
    if facts:
        lines.append("独有结构事实（必须讲出至少一条，且只能讲这些程序给出的事实）：")
        # 注入前剥离结构特征数值（「（0.75 vs 0.5）」）——观众听不懂小数
        # 对比，LLM 会照念成「零点七五对零点五」（08.11 实测）。校验
        # `_kw_matches` 只按区域词+主体词匹配，剥数值不影响命中。
        lines.extend(f"  - {_strip_fact_numbers(f)}" for f in facts)
    return "\n".join(lines)


def build_chunk_prompt(header: str, chunk_nodes: list, chunk_idx: int,
                       total_chunks: int, all_nodes: list,
                       generated_segments: list) -> str:
    """每节点构建形态化 prompt（对齐 generator 契约）。"""
    parts = [header, "", "【本段要讲解的叙事单元】"]
    for node in chunk_nodes:
        ntype = node.get("type")
        if ntype == "opening":
            # 计划数必须显式给出（08.04 修）。原文只写「存在战略选择（几个
            # 方向）」，LLM 便按常态假设两条路线，在只有 1 个计划通过可行性
            # 闸时仍讲「有两个方向」——讲了一条被闸门淘汰的路，违反 ADR-015
            # 「程序算事实、LLM 只表达」边界，属教错棋。
            n = int(node.get("n_plans", 0) or 0)
            if n >= 2:
                choice_line = (f"再点出这个局面上有 {n} 个可行方向"
                               "（后面会逐条展开）")
            else:
                choice_line = ("再说明这个局面上**只有一条方向经得起验算**——"
                               "不要提「两个方向」「另一种选择」，"
                               "也不要暗示存在未展开的备选路线")
            parts.append(
                "【开场】局面概览：先诊断局面类型与核心矛盾（战略前提），"
                f"{choice_line}，"
                f"最后用一句话铺垫反事实紧迫性（{ _baseline_cn(node.get('baseline'))}，"
                "只说程度不报数值）。")
        elif ntype == "plan":
            parts.append(_plan_prompt_text(node, is_first=node.get("idx") == 0))
        elif ntype == "rejected":
            # 轴 4 对照段（ADR-021）：「为什么不那样走」。
            # 措辞范式：诱惑力 → 但代价是 → 所以正选更优。定性为「更差但合理」。
            # gap 量级由程序注入（文本零数字）。
            # R2 修复（PLAN-008 教训）：不在指令里枚举禁用词（负面提及强化被禁内容），
            # 改用正面定性指令；validator 侧的硬错误清单兜底。
            # R3 修复：显式给 LLM 对照段的字数约束（全局 60-110 vs 对照段 ≤60）。
            gap_word = node.get("gap_level", "近一个兵")
            facts = node.get("unique_facts", [])
            fact_lines = ("\n".join(f"  - {_strip_fact_numbers(f)}"
                                   for f in facts) if facts else "  - （无独有事实——讲评估差距即可）")
            parts.append(
                f"【为什么不那样走】这一段讨论「{node['name']}」这条路线——"
                f"它**看起来也合理**，但比正选「{node.get('primary_name', '')}」"
                f"差约{gap_word}。"
                "先说出这条路的**诱惑力**（为什么人想走它），"
                "然后说出它的**代价**——必须包含以下独有事实中的至少一条：\n"
                f"{fact_lines}\n"
                f"最后用一句话收束：正因为付出{gap_word}的代价，"
                f"「{node.get('primary_name', '')}」更值得选。"
                "**措辞要求**：把这条路线定性为「稍逊的合理替代」——"
                f"用程度词（稍逊/略差/差了约{gap_word}）描述差距，"
                "定性落点必须是「合理但更差」。"
                "**这一段要简短——控制在 60 字以内（比正选段更短）。**")
        elif ntype == "compare":
            parts.append(
                f"【对比】直接对比两条路线（{node['names'][0]} vs {node['names'][1]}）："
                "各自的结构改变、代价与承诺度差异；然后给条件性建议"
                "（什么局面倾向哪条）。对比必须落到具体结构差异，"
                "不得用「各有千秋」收尾。")
        elif ntype == "summary":
            prov = node.get("provenance_plan")
            # 与开场同源的单线适配（08.04 修）：原文无条件写「收束两路对比」，
            # 单计划时会逼 LLM 编造第二条路线来「对比」。
            names = node.get("names", []) or []
            if len(names) >= 2:
                parts.append(
                    f"【总结】收束两路对比（{'、'.join(names)}），"
                    "重申关键取舍，给一句条件性建议。")
            else:
                only = names[0] if names else "这条路线"
                parts.append(
                    f"【总结】只有「{only}」一条路线经得起验算，"
                    "所以收束时重申它的核心代价与执行要点，"
                    "**不做左右对比、不提不存在的备选**，"
                    "给一句「什么情况下要格外小心」的提醒。")
            if prov:
                # 措辞必须限定为「这一局」（08.04 修）。`provenance` 是**单局**
                # PGN 续走的首着（挖掘器 continuation[0]），不是频率统计。
                # 原文写「更多选了……作为频率事实」，LLM 忠实照做，产出
                # 「实战中多数人选择推进悬兵」——用 n=1 的样本讲群体倾向，
                # 是无根据的推断（实测 hanging demo 成片总结段原句）。
                # 真正的频率口径需要 `pgn_plan_stats` 的多局统计，那是另一条
                # 数据通路（P17 设施），本字段给不出。
                # 说「这一局走的是 X」是可核验的事实；说「多数人选 X」不是。
                # 措辞用正面指令而非「不要说 X」：本模块 docstring 记的
                # PLAN-008 教训是「禁用词在指令里出现会强化被禁内容」，
                # 故只给可照抄的句式，不点名要避开的说法。
                parts.append(
                    f"补充（Tier A）：这一局的实战续走里，执子方走的是"
                    f"「{prov}」这个方向。用一句话点到即止，句式照"
                    f"「这一局里他选的是{prov}」这种**限定到单局**的说法，"
                    "作为一个具体例子，不展开、不引申。")
    parts.append("")
    # 字数上限是硬约束（08.04 修）：TTS 是 GPU 逐字合成，实测 ChatTTS 约
    # 27s/96 字。此前只写「一句到三句」无字数上界，总结/对比段常破 150 字，
    # 单段合成远超合理时长，是阶段 8a TTS 环节卡顿的上游根因。对齐 puzzle
    # 口径给每段 60~110 字硬上限。
    parts.append("按节点 id 输出 segments，每段两到三句、控制在 60 到 110 字之间，"
                 "口语化，不念坐标不走着法。超过 110 字会被判废重来，务必精炼。"
                 "**每条路线用它的计划名指代**（如「推进悬兵」「保持悬兵」），"
                 "不要用「这条路」「方案一」之类含糊指代。")
    parts.append("输出格式（严格，不加 markdown 标记）：")
    parts.append('{"segments": [{"id": 0, "voiceover": "..."}, '
                 '{"id": 1, "voiceover": "..."}]}')
    # id 图例必须由**实际节点**生成，不能硬编码（08.04 修）。
    #
    # 原写法是固定的「0=开场，1/2=各计划，3=对比，4=总结」，但 id 在
    # `_build_decision_nodes` 里是动态的：只有 1 个计划可行时不生成对比节点，
    # summary 的 id 就落到 2 而不是 4。于是 LLM 按错误图例把 id 2 当成
    # 「第二个计划」来写，而 post_process 按 node type 又把它装进了
    # commentary.summary——一处错配同时产生两个症状：
    #   ① 总结段变成第二计划的描述（与计划段内容重复）；
    #   ② 解说讲了一个**未通过可行性闸**的计划（违反 ADR-015「程序算事实、
    #      LLM 只表达」的边界，属于会教错棋的内容级错误）。
    # 实测现场（悬兵局面，可行计划 1 个）：段2 与总结逐字相同，且都在讲
    # 可行性闸淘汰掉的「保持悬兵」。
    legend = "、".join(
        f"{n['id']}={_NODE_ROLE_CN.get(n.get('type'), n.get('type', '?'))}"
        + (f"（{n.get('name')}）" if n.get("type") in ("plan", "rejected") else "")
        for n in chunk_nodes
    )
    parts.append(f"id 用数字，本次只输出这些 id：{legend}。"
                 "不要输出未列出的 id，也不要漏掉任何一个。")
    return "\n".join(parts)


def build_grammar(chunk_size: int) -> str:
    """复用残局链路的 JSON 语法约束。"""
    return build_chunk_grammar(chunk_size)


# 独有事实校验用关键词（区域词 + 主体词拆分——voiceover 口语化时
# 「对方后翼出现孤立兵」不连续含「后翼孤立兵」，须按词匹配）
_REGION_WORDS = ("后翼", "中心", "王翼")
_SUBJECT_WORDS = ("孤立兵", "后退兵", "通路兵", "兵过中线", "兵岛",
                  "开放线", "半开放线", "前哨", "轻子", "王暴露")


# 主体词的口语变形容忍（08.04 修）。
#
# 原实现要求主体词以**连续字符串**出现在 voiceover 里，但口语表达会在词中间
# 插字：事实写「己方兵过中线」，解说说「兵**推**过**了**中线」；事实写
# 「王暴露」，解说说「王**的**暴露面」。连续匹配一律判未命中——这是假阴性，
# 会把讲对了的段判废（独有事实校验是 P9 防同质化的主防线，误杀比漏放更糟：
# 漏放只是少一层保险，误杀会让合格解说反复重试直至整片放弃）。
#
# 解法：对易被拆开的主体词声明「必需词组合」——全部出现即命中，不要求连续、
# 不要求顺序。仍然要求实质内容词，所以「两条路各有千秋」这类套话依然拦得住
# （它既无「兵」也无「中线」）。未声明的词保持原连续匹配（它们在中文里本就
# 是固定搭配，如「孤立兵」「开放线」「前哨」）。
_SUBJECT_ALIASES = {
    "兵过中线": ("兵", "中线"),
    "王暴露": ("王", "暴露"),
}


def _subject_in_text(subject: str, text: str) -> bool:
    """主体词是否出现在文本中（含口语变形容忍）。"""
    parts = _SUBJECT_ALIASES.get(subject)
    if parts:
        return all(p in text for p in parts)
    return subject in text


def _kw_matches(voiceover: str, fact: str) -> bool:
    """一条独有事实是否命中 voiceover（按区域词+主体词匹配）。"""
    for subject in _SUBJECT_WORDS:
        if subject not in fact:
            continue
        if not _subject_in_text(subject, voiceover):
            continue
        # 事实含区域词时 voiceover 也必须含同一区域（区分「后翼」与
        # 「中心」孤立兵）
        for region in _REGION_WORDS:
            if region in fact:
                return region in voiceover
        return True
    return False


def _check_unique_facts(voiceover: str, facts: list) -> bool:
    """独有事实命中：voiceover 命中任一事实（按区域+主体词匹配）。"""
    if not facts:
        return True  # 无独有事实（单计划）——该层跳过
    return any(_kw_matches(voiceover, f) for f in facts)


def _plan_display_name(name: str) -> str:
    """计划名主体（去括号——「保持悬兵（利用动态潜力）」→「保持悬兵」）。

    战略名提及校验用主体名：LLM 自然表达会省略括号里的修饰，完整名
    要求过严会误杀合法解说（实跑验证的首个教训）。
    """
    return name.split("（")[0].split("(")[0].strip()


def _rejected_max_chars(chunk_nodes: list) -> int:
    """轴 4 对照段字数上限 = 同 chunk 内 plan 段实际长度的 60%（ADR-021 d）。

    校验在 validate_chunk 内逐段跑——此时同 chunk 的 plan 段可能尚未生成
    （取决于 LLM 输出顺序），无法精确取其字数。保守策略：用 plan 段的字数
    上限 MAX_VOICEOVER_CHARS(130) 的 60% = 78 作为硬上界——略宽于实际 60%
    （plan 段通常 60-110 字，真实 60% = 36-66），但比 130 紧很多，足以防
    对照段喧宾夺主。
    """
    return int(MAX_VOICEOVER_CHARS * 0.6)


def validate_chunk(data: dict, chunk_nodes: list) -> tuple:
    """四层校验：① 表层硬闸；② 字数上限；③ 战略名提及；④ 独有事实命中（P9）。

    返回 (ok, errors)。段级失败不报废——校验失败的段在 post_process 里
    丢弃（该段不输出）。

    **轴 4 对照段（rejected）的限权失败采用段级语义**（ADR-021 SPEC §8 兼容
    + peer_review R1）：rejected 段的错（锚词/禁词/字数/独有事实）不升级为
    chunk 级失败，而是将该段标记为 ``_reject_drop``——generator 不重试、
    post_process 丢弃该段，整片正常出（对照段缺席=退回单线，符合 ADR-021）。
    """
    errors = []
    segments = data.get("segments", [])
    node_by_id = {int(n["id"]): n for n in chunk_nodes}
    for si, seg in enumerate(segments):
        # id 解析：LLM 可能用节点类型名（'opening'）或数字——失败时
        # 按输出顺序对应节点（LLM 按序输出是常态）
        raw_id = seg.get("id", -1)
        try:
            nid = int(raw_id)
        except (TypeError, ValueError):
            nid = chunk_nodes[si]["id"] if si < len(chunk_nodes) else -1
        node = node_by_id.get(nid)
        if node is None:
            errors.append(f"segment id {raw_id} 无对应节点")
            continue
        text = str(seg.get("voiceover", ""))
        # ① 表层硬闸（坐标/英文数字/Markdown/思考泄漏——不含长度）
        ok_surface, surface_issues = validate_puzzle_voiceover_surface(text)
        if not ok_surface:
            errors.append(f"id {nid} 表层校验失败: {surface_issues[:80]}")
        # ② 字数上限硬闸（08.04 补）。prompt 里的「60~110 字」是软约束，
        # LLM 常突破；而 TTS 是 GPU 逐字合成（实测 ChatTTS 约 27s/96 字），
        # 超长段直接拖垮阶段 8a 的合成环节。表层硬闸只查字符不查长度，
        # 故在此设硬上限——失败进重试链（MAX_RETRIES），不静默放行。
        if len(text) > MAX_VOICEOVER_CHARS:
            errors.append(
                f"id {nid} 超长（{len(text)} 字 > {MAX_VOICEOVER_CHARS}）")
        if node.get("type") == "rejected":
            # 轴 4 对照段限权（ADR-021，PLAN-012 阶段 3）。
            # R1 修复：错不进 errors（不触发 generator 重试/aborted），
            # 而是标 ``_reject_drop`` 让 post_process 丢弃——段级语义，
            # 确保对照段失败时整片仍出（退回单线，不阻塞）。
            rej_errors = []
            plan_max = _rejected_max_chars(chunk_nodes)
            if len(text) > plan_max:
                rej_errors.append(f"超限({len(text)}>{plan_max})")
            # a) 正面锚词：必须命中至少一个「更差」锚词
            if not any(w in text for w in AXIS4_ANCHOR_WORDS):
                rej_errors.append("缺更差锚词")
            # b) 等强措辞硬错误（三词短表）
            for w in AXIS4_FORBIDDEN_EQUAL:
                if w in text:
                    rej_errors.append(f"等强措辞「{w}」")
            # c) 灾难定性硬错误
            for w in AXIS4_FORBIDDEN_DISASTER:
                if w in text:
                    rej_errors.append(f"灾难定性「{w}」")
            # d) 独有事实命中（与 plan 节点同口径——防套话）
            if not _check_unique_facts(text, node.get("unique_facts", [])):
                rej_errors.append("未命中独有事实")
            if rej_errors:
                seg["_reject_drop"] = "; ".join(rej_errors)
        # ② 战略名提及（主体名——plan 段提本计划；compare/summary 提任一）
        if node.get("type") in ("plan", "compare", "summary"):
            if node.get("type") == "plan":
                names = [_plan_display_name(node["name"])]
            else:
                names = [_plan_display_name(n) for n in node.get("names", [])]
            if names and not any(n and n in text for n in names):
                errors.append(f"id {nid} 未提及战略名 {names}")
    return (not errors, errors)


def auto_fix_voiceover(text: str, node: dict) -> str:
    """自动修复（文本清洗——无 LLM 二次调用）。"""
    t = strip_thinking(text)
    t = strip_coordinates(t)
    t = reduce_cliches(t)
    return t


def build_fallback_voiceover(chunk_nodes: list, json_prompt: str) -> list:
    """无 LLM 兜底：按节点类型生成静态解说（仍走三层校验）。"""
    segments = []
    for node in chunk_nodes:
        ntype = node.get("type")
        if ntype == "opening":
            text = (f"这个局面呈现{node.get('n_plans', 1)}条可行的战略路线。"
                    "先看清兵形的前提，再比较各条路线的取舍。")
        elif ntype == "plan":
            # 先规范化再截断（08.04 修）：mechanism 是带坐标的 KB 原文，
            # 若先 [:40] 可能正好切在坐标中间（「b4-b5」→「b4-b」），残段
            # 连 strip_coordinates 都认不出来，会以裸字母进到 TTS。
            mech = safe_decision_seed_text(node.get("mechanism", ""))[:40]
            text = (f"方案是{node['name']}——{mech}"
                    f"；{_trend_cn(node.get('trend', {}))}。"
                    f"{_tradeoffs_cn(node.get('tradeoffs', {}))}。")
        elif ntype == "compare":
            text = ("两条路线的主要差别在于结构走向与代价。"
                    "选择取决于局面更看重哪一端。")
        elif ntype == "rejected":
            # 轴 4 兜底（ADR-021）：锚词 + gap 量级 + 收束
            gap_word = node.get("gap_level", "近一个兵")
            text = (f"为什么不走{node.get('name', '那条')}？"
                    f"它稍逊于正选——差了约{gap_word}。"
                    "结构上付出了代价，所以不是最优选择。")
        else:
            text = ("综合来看，两条路线各有明确的结构取舍："
                    "一条换来兵形稳固，另一条换来对方弱点的形成。")
        segments.append({"id": node["id"], "sub_endgame": "",
                         "voiceover": auto_fix_voiceover(text, node),
                         "pacing": "normal"})
    return segments


def post_process(commentary: GeneratedCommentary, all_segments: list,
                 nodes: list, storyboard: dict, backend) -> None:
    """校验失败的段丢弃（段级失败语义）；组装 opening/summary。

    轴 4 对照段（rejected）被 validate_chunk 标 ``_reject_drop`` 时，
    在此丢弃——对照段缺席 = 退回单线，整片正常出（ADR-021 §SPEC §8 兼容）。
    """
    node_by_id = {int(n["id"]): n for n in nodes}
    kept = []
    for seg in all_segments:
        # 轴 4 段级丢弃（R1 修复：不阻塞整片）
        drop_reason = getattr(seg, "_reject_drop", None) or seg.get("_reject_drop") if isinstance(seg, dict) else getattr(seg, "_reject_drop", None)
        if drop_reason:
            node = node_by_id.get(int(getattr(seg, "id", -1) if not isinstance(seg, dict) else seg.get("id", -1)))
            Logger.info(f"[Decision] 对照段丢弃（{drop_reason}）——退回单线")
            continue
        node = node_by_id.get(int(getattr(seg, "id", -1)))
        if node is None:
            continue
        text = getattr(seg, "voiceover", "")
        ok_surface, _ = validate_puzzle_voiceover_surface(text)
        if node.get("type") in ("plan", "rejected"):
            ok_facts = _check_unique_facts(text, node.get("unique_facts", []))
        else:
            ok_facts = True
        if ok_surface and ok_facts:
            kept.append(seg)
    commentary.segments = kept
    for seg in kept:
        node = node_by_id.get(int(getattr(seg, "id", -1)))
        if node is None:
            continue
        text = clean_cjk_text(getattr(seg, "voiceover", ""))
        if node.get("type") == "opening":
            commentary.opening = clean_summary_text(text)
        elif node.get("type") == "summary":
            commentary.summary = clean_summary_text(text)
    commentary.raw_text = "\n".join(
        getattr(s, "voiceover", "") for s in kept)


def _build_config() -> dict:
    """组装 CommentaryConfig 回调集（对齐 endgame_commentary 契约）。"""
    return {
        "build_header": build_header,
        "build_chunk_prompt": build_chunk_prompt,
        "build_grammar": build_grammar,
        "validate_chunk": validate_chunk,
        "auto_fix_voiceover": auto_fix_voiceover,
        "repair_failed_segments": None,
        "build_fallback_voiceover": build_fallback_voiceover,
        "post_process": post_process,
    }


def generate_decision_commentary(
    decision_input,
    decision_storyboard: dict,
    backend=None,
) -> GeneratedCommentary:
    """决策解说入口（复用 generator.generate_commentary）。

    `decision_input`：decision_builder.DecisionInput（含 fen/provenance）。
    provenance 续着映射到计划名（P12 措辞降级——频率事实）。
    """
    import chess

    from src.commentator.generator import CommentaryConfig, generate_commentary

    board = chess.Board(decision_input.fen)
    prov_plan = _match_provenance_plan(board, decision_storyboard,
                                       decision_input.provenance)
    sb = dict(decision_storyboard)
    sb["provenance_plan"] = prov_plan
    sb["nodes"] = _build_decision_nodes(sb)

    config = CommentaryConfig(
        build_header=build_header,
        build_chunk_prompt=build_chunk_prompt,
        build_grammar=build_grammar,
        validate_chunk=validate_chunk,
        auto_fix_voiceover=auto_fix_voiceover,
        repair_failed_segments=None,
        build_fallback_voiceover=build_fallback_voiceover,
        post_process=post_process,
    )
    backend = backend or create_backend_from_env()
    commentary = generate_commentary(sb, backend, config)
    return commentary


if __name__ == "__main__":
    """阶段 7 单元测试：三层校验逻辑（纯函数——不调 LLM）。"""
    results = []

    # 1. 独有事实命中
    ok1 = _check_unique_facts("这条路会让对方后翼出现孤立兵",
                              ["甲末端后翼孤立兵显著（0.33 vs 0.0）"])
    ok1b = not _check_unique_facts("这条路各有千秋", ["甲末端后翼孤立兵显著"])
    results.append(("独有事实命中/未命中", ok1 and ok1b,
                    f"hit={ok1} miss={ok1b}"))

    # 2. 战略名提及（validate_chunk）
    node = {"id": 1, "type": "plan", "name": "少数派攻击",
            "unique_facts": ["甲末端后翼孤立兵显著"]}
    data_ok = {"segments": [{"id": 1, "voiceover": "少数派攻击的路线会让对方后翼孤立兵"}]}
    data_bad = {"segments": [{"id": 1, "voiceover": "这条路各有千秋"}]}
    ok2, err2 = validate_chunk(data_ok, [node])
    bad2, _ = validate_chunk(data_bad, [node])
    results.append(("validate_chunk 三层校验", ok2 and not bad2,
                    f"ok={ok2} bad={bad2} err={err2[:60]}"))

    # 3. fallback 兜底（无 LLM）
    nodes = [{"id": 0, "type": "opening", "n_plans": 2, "baseline": -20},
             {"id": 1, "type": "plan", "name": "少数派攻击",
              "mechanism": "后翼兵推进", "trend": {}, "tradeoffs": {},
              "unique_facts": []}]
    segs = build_fallback_voiceover(nodes, "")
    ok3 = len(segs) == 2 and all(s["voiceover"] for s in segs)
    results.append(("fallback 兜底", ok3, str([s["voiceover"][:20] for s in segs])))

    # 4. provenance → 计划名映射（P12）
    import chess as _ch
    b = _ch.Board("r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/2NBPN2/PPQ2PPP/"
                  "R4RK1 w - - 8 11")
    sb = {"routes": [{"name": "少数派攻击",
                      "direction": {"pawn_files": ["a", "b"],
                                    "target_zone": "queenside"}}]}
    got = _match_provenance_plan(b, sb, "a3")
    results.append(("provenance→计划名（P12）", got == "少数派攻击",
                    f"a3 -> {got}"))

    ok = True
    for name, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        ok &= passed
    print("阶段 7 单元测试:", "全部通过" if ok else "存在失败")
    raise SystemExit(0 if ok else 1)
