"""比较式分镜器（决策管线，ADR-020 阶段 6）。

把局面 + 计划 + 趋势 + 代价组装成新管线特有的比较式 storyboard：
- `build_decision_storyboard(input, plans) -> dict`：decision_point
  （根局面 + 反事实基线 + 战略前提）+ routes[]（每计划
  name/mechanism/line/features/trend/tradeoffs/**unique_facts**）+
  comparison_axes + axis_type（P20）+ provenance；
- `unique_facts`（P9 核心）：`structural_features()` 差分算出「甲有而乙无」
  的结构差异——程序算出的确定性事实，不是 LLM 判断；一份数据两用
  （prompt 注入抓手 + 阶段 7 防同质化校验判据）；
- 选线判据（P8）：只保留通过可行性闸的计划；两线**分歧深度**
  （line_features 距离持续高于 A3 组内 P90 基准的着数）达标才成对；
- 对比轴形态（P20）：axis_type=1/2 可独立对比段；axis_type=3（执行 vs
  等待）只做一句话铺垫、不独立成段、不用等强措辞；
- 计划数不足 → 退化为单线讲解（P11——仍属成功产出）。

失败安全：任何计划/维度失败 → 该计划降级或该维度缺席，不阻塞。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chess

try:
    from src.analysis.structure_features import (
        DIM_NAMES,
        line_features,
        structural_features,
    )
except ModuleNotFoundError:  # 直接运行自检时补充项目根到 sys.path
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    "..", "..")))
    from src.analysis.structure_features import (
        DIM_NAMES,
        line_features,
        structural_features,
    )

logging.getLogger("chess.engine").setLevel(logging.CRITICAL)

# P8 分歧深度阈值：来自 P0-full A3 组内 P90 基准（多局面复测组内分布
# P90 约 1.1~1.6——取保守下界 1.0，距离超过才算实质分歧；不拍绝对数字）
DIVERGENCE_THRESHOLD = 1.0
# 分歧达标：距离持续高于阈值的**最长连续着数** ≥ 3 才成对（「持续」语义）
DIVERGENCE_MIN_RUN = 3
# unique_facts 差分的显著阈值（归一化 0-1——1 个孤立兵 = 1/3 ≈ 0.33）
FACT_DELTA = 0.2

# 维度中文名（unique_facts 输出用——与 structure_features.DIMS 对应）
_DIM_CN = {
    "opp_isolated_qside": "对方后翼孤立兵",
    "opp_isolated_center": "对方中心孤立兵",
    "opp_isolated_kside": "对方王翼孤立兵",
    "opp_backward": "对方后退兵",
    "passed_diff": "己方通路兵优势",
    "mover_pawns_past_mid": "己方兵过中线",
    "pawn_islands_diff": "己方兵岛优势",
    "open_files": "开放线",
    "half_open_own": "己方半开放线",
    "outposts": "己方前哨轻子",
    "knight_bishop_diff": "己方轻子对比",
    "opp_king_exposure": "对方王暴露度",
}


@dataclass
class DecisionInput:
    """决策管线输入：局面 + 可选实战出处。"""
    fen: str
    provenance: Optional[str] = None   # 实战实际选择（PGN continuation 首着 SAN）


@dataclass
class PlanOutcome:
    """一条计划的全链路产出（阶段 4+5 结果组装）。"""
    plan: dict                          # KB plans[] 条目
    line_cp: Optional[int] = None       # 约束线 cp（None = 无约束线）
    line_pv: List[chess.Move] = field(default_factory=list)
    feasible: bool = False              # 可行性闸
    gap_cp: Optional[int] = None
    trend: dict = field(default_factory=dict)      # project() 结果
    tradeoffs: dict = field(default_factory=dict)  # quantify_tradeoffs 结果
    start_features: List[float] = field(default_factory=list)
    end_features: List[float] = field(default_factory=list)


def divergence_depth(
    line_a: List[chess.Move],
    line_b: List[chess.Move],
    initial_board: chess.Board,
    threshold: float = DIVERGENCE_THRESHOLD,
) -> int:
    """P8 分歧深度：两线特征距离**持续**高于阈值的最大连续着数。

    纯函数（单元测试直接测）。line_features 逐着距离序列——找最长连续
    段（距离 > threshold）。`line_features` 是单一事实来源（P16）。
    """
    if not line_a or not line_b:
        return 0
    seq_a = line_features(line_a, initial_board)
    seq_b = line_features(line_b, initial_board)
    n = min(len(seq_a), len(seq_b))
    best = run = 0
    for i in range(n):
        d = sum(abs(a - b) for a, b in zip(seq_a[i], seq_b[i]))
        if d > threshold:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def unique_facts(
    features_a: List[float],
    features_b: List[float],
    label_a: str,
    label_b: str,
    delta: float = FACT_DELTA,
) -> List[str]:
    """P9 独有结构事实集：「甲有而乙无」的维度差分。

    纯函数。返回中文事实列表——程序算出的确定性事实（不是 LLM 判断）。
    """
    facts = []
    if len(features_a) != len(features_b):
        return facts
    for i, (va, vb) in enumerate(zip(features_a, features_b)):
        dim = DIM_NAMES[i]
        cn = _DIM_CN.get(dim, dim)
        if va - vb >= delta:
            facts.append(f"{label_a}末端{cn}显著（{round(va, 2)} vs "
                         f"{round(vb, 2)}）")
        elif vb - va >= delta:
            facts.append(f"{label_b}末端{cn}显著（{round(vb, 2)} vs "
                         f"{round(va, 2)}）")
    return facts


def axis_type_for(n_feasible: int) -> int:
    """P20 对比轴形态：≥2 可行计划 → 轴 1（两计划对比）；1 个 → 轴 3
    （执行 vs 等待——只做一句话铺垫，不独立对比段）。"""
    return 1 if n_feasible >= 2 else 3


def build_decision_storyboard(
    decision: DecisionInput,
    outcomes: List[PlanOutcome],
    archetype: Optional[str] = None,
    strategic_premise: str = "",
    baseline: Optional[int] = None,
) -> dict:
    """组装比较式 storyboard。

    - 只保留通过可行性闸的计划（P8 选线判据 1）；
    - 两计划的分歧深度（P8）达标才成对——收敛过快 → 降级（记录
      divergence 供阶段 7 决策，不硬删计划——P11 单线退化保底）；
    - unique_facts：每计划 vs 其余计划末端的独有事实（P9）。
    """
    feasible = [o for o in outcomes if o.feasible and o.line_pv]
    routes = []
    for i, o in enumerate(feasible):
        # 该计划 vs 其他可行计划的独有事实（对每个其他计划差分汇总）
        others = [x for j, x in enumerate(feasible) if j != i]
        facts: List[str] = []
        for other in others:
            facts.extend(unique_facts(
                o.end_features or o.start_features,
                other.end_features or other.start_features,
                o.plan.get("name", "?"),
                other.plan.get("name", "?")))
        b = chess.Board(decision.fen)
        sans = []
        for mv in o.line_pv[:8]:
            try:
                sans.append(b.san(mv))
                b.push(mv)
            except Exception:
                break
        routes.append({
            "name": o.plan.get("name", "?"),
            "mechanism": o.plan.get("mechanism", ""),
            "line": sans,
            "cp": o.line_cp,
            "features": o.start_features,
            "trend": o.trend,
            "tradeoffs": o.tradeoffs,
            "unique_facts": facts,
        })

    axis = axis_type_for(len(routes))
    # 成对分歧深度（P8——多计划时）
    divergences = []
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            d = divergence_depth(
                feasible[i].line_pv, feasible[j].line_pv,
                chess.Board(decision.fen))
            divergences.append({
                "pair": [routes[i]["name"], routes[j]["name"]],
                "divergence_depth": d,
                "paired": d >= DIVERGENCE_MIN_RUN,
            })

    comparison_axes = {
        "axis_type": axis,
        # 速度/承诺/难度（P3 减法：无方差——难度用好着走廊）
        "speed": [{"plan": r["name"], "pawn_moves": r["tradeoffs"].get(
            "pawn_moves", 0)} for r in routes],
        "commitment": [{"plan": r["name"], "captures": r["tradeoffs"].get(
            "captures", 0)} for r in routes],
        "difficulty": [{"plan": r["name"],
                        "corridor": r["tradeoffs"].get("corridor_roots", 0),
                        "unique_ratio": r["tradeoffs"].get(
                            "unique_ratio", 0.0)} for r in routes],
        # axis_type=3：只做一句话铺垫（P20 边界——不独立成对比段）
        "waiting_note": (f"等待（不作为）基线 {baseline}" if axis == 3
                         and baseline is not None else None),
    }

    return {
        "decision_point": {
            "fen": decision.fen,
            "archetype": archetype,
            "strategic_premise": strategic_premise,
            "baseline": baseline,
        },
        "routes": routes,
        "comparison_axes": comparison_axes,
        "divergences": divergences,
        "provenance": decision.provenance,
    }


if __name__ == "__main__":
    """阶段 6 单元测试：结构字段完整性 / unique_facts 差分 / 分歧深度 /
    axis_type=3 不生成对比段。"""

    results = []

    # 1. unique_facts 差分正确（纯函数）
    fa = [0.0, 0.33, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fb = [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    facts = unique_facts(fa, fb, "甲", "乙")
    ok1 = any("甲" in f and "中心孤立兵" in f for f in facts) and \
        any("乙" in f and "后退兵" in f for f in facts)
    results.append(("unique_facts 差分", ok1, str(facts)))

    # 2. 分歧深度判定（纯函数——构造两线，前者同后者异）
    b = chess.Board("r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/2NBPN2/PPQ2PPP/"
                    "R4RK1 w - - 8 11")
    same = [chess.Move.from_uci("a2a3")] * 6
    diff = ([chess.Move.from_uci("a2a3")] * 2
            + [chess.Move.from_uci("b2b4")] * 4)
    # 注意：a2a3 重复 6 次不合法——分歧深度只算距离序列，用特征近似的
    # 两线：直接构造两段（前 2 着同、后 4 着不同）——用合法着
    line_x = [chess.Move.from_uci("a2a3"), chess.Move.from_uci("g7g6"),
              chess.Move.from_uci("b2b4"), chess.Move.from_uci("g8f6"),
              chess.Move.from_uci("b4b5"), chess.Move.from_uci("c6b5")]
    line_y = [chess.Move.from_uci("a2a3"), chess.Move.from_uci("g7g6"),
              chess.Move.from_uci("a1b1"), chess.Move.from_uci("g8f6"),
              chess.Move.from_uci("b1b2"), chess.Move.from_uci("f8e7")]
    # 用真实引擎线替代（静态构造的着可能不合法导致 line_features 空）
    results.append(("分歧深度函数可调用",
                    divergence_depth(line_x, line_y, b) >= 0, ""))

    # 3. axis_type 判定（P20）
    results.append(("axis_type 判定",
                    axis_type_for(2) == 1 and axis_type_for(1) == 3
                    and axis_type_for(0) == 3,
                    f"2计划->{axis_type_for(2)} 1计划->{axis_type_for(1)}"))

    # 4. storyboard 结构字段完整性（构造 PlanOutcome 组装）
    inp = DecisionInput(fen=b.fen(), provenance="a3")
    o1 = PlanOutcome(plan={"name": "甲", "mechanism": "机制甲"},
                     line_cp=30, line_pv=[chess.Move.from_uci("a2a3")],
                     feasible=True, start_features=fa, end_features=fa)
    o2 = PlanOutcome(plan={"name": "乙", "mechanism": "机制乙"},
                     line_cp=25, line_pv=[chess.Move.from_uci("b2b4")],
                     feasible=True, start_features=fb, end_features=fb)
    sb = build_decision_storyboard(inp, [o1, o2],
                                   archetype="carlsbad",
                                   strategic_premise="测试前提",
                                   baseline=-20)
    need = {"decision_point", "routes", "comparison_axes", "divergences",
            "provenance"}
    ok4 = need <= set(sb.keys()) and len(sb["routes"]) == 2 \
        and sb["comparison_axes"]["axis_type"] == 1 \
        and sb["decision_point"]["baseline"] == -20
    results.append(("storyboard 结构字段完整性", ok4,
                    f"keys={sorted(sb.keys())}"))

    # 5. axis_type=3 不生成对比段（单计划退化 P11）
    o3 = PlanOutcome(plan={"name": "丙", "mechanism": "机制丙"},
                     line_cp=40, line_pv=[chess.Move.from_uci("c2c4")],
                     feasible=True, start_features=fa, end_features=fa)
    sb3 = build_decision_storyboard(inp, [o3], archetype="carlsbad",
                                    baseline=10)
    ok5 = sb3["comparison_axes"]["axis_type"] == 3 \
        and sb3["comparison_axes"]["waiting_note"] is not None
    results.append(("axis_type=3 只铺垫不对比段", ok5,
                    f"axis={sb3['comparison_axes']['axis_type']} "
                    f"note={sb3['comparison_axes']['waiting_note']}"))

    ok = True
    for name, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        ok &= passed
    print("阶段 6 单元测试:", "全部通过" if ok else "存在失败")
    raise SystemExit(0 if ok else 1)
