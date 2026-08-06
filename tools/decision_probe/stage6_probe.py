"""阶段 6 验证探针：示例局面产出比较式 storyboard（PLAN-009 阶段 6 验证）。

对 6 原型示例局面跑全链路（explore_forward → assess_feasibility →
project → quantify_tradeoffs → build_decision_storyboard），打印
storyboard 供人工核对结构与棋理。

用法：
    python -m tools.decision_probe.stage6_probe
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(_ROOT, ".env"))

import chess  # noqa: E402

from src.analysis.structure_features import structural_features  # noqa: E402
from src.solver.branch_explorer import (  # noqa: E402
    assess_feasibility,
    explore_forward,
    explore_open,
    waiting_baseline,
)
from src.solver.consequence_projector import (  # noqa: E402
    quantify_tradeoffs,
    project,
)
from src.storyboard.decision_builder import (  # noqa: E402
    DecisionInput,
    PlanOutcome,
    build_decision_storyboard,
)

SITUATIONS = [
    ("卡尔斯巴德", "carlsbad", "r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/"
     "2NBPN2/PPQ2PPP/R4RK1 w - - 8 11", "a3"),
    ("悬兵", "hanging", "2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/"
     "1R2R1K1 w - - 1 21", "d5"),
    ("马洛齐", "maroczy", "r2q1rk1/pp2ppbp/3pbnp1/8/2P1P3/2N1B3/"
     "PP1QBPPP/R3K2R w KQ - 5 11", "O-O"),
]


def main() -> None:
    import json

    sf = os.getenv("STOCKFISH_PATH", "")
    if not os.path.isabs(sf):
        sf = os.path.normpath(os.path.join(_ROOT, sf))
    kb = json.load(open(os.path.join(_ROOT, "data", "structure_kb.json"),
                        encoding="utf-8"))

    for sit_name, arch_key, fen, prov in SITUATIONS:
        b = chess.Board(fen)
        print("\n" + "=" * 76)
        print(f"=== {sit_name}（实战续着 {prov}）===")
        opens = explore_open(b, sf, k=4, depth=14)
        baseline = waiting_baseline(b, sf, depth=12)
        outcomes = []
        for plan in kb[arch_key]["plans"]:
            line = explore_forward(b, plan, sf, depth=14)
            if line is None or not line.pv:
                continue
            feas, gap = assess_feasibility(
                line.cp, opens[0].cp if opens else None)
            tr = project(line, b, sf)
            tm = quantify_tradeoffs(line, b, sf, open_lines=opens)
            outcomes.append(PlanOutcome(
                plan=plan, line_cp=line.cp, line_pv=line.pv,
                feasible=feas, gap_cp=gap, trend=tr,
                tradeoffs=tm.__dict__,
                start_features=structural_features(b),
                end_features=tr.get("end_features", [])))

        sb = build_decision_storyboard(
            DecisionInput(fen=fen, provenance=prov), outcomes,
            archetype=arch_key,
            strategic_premise=kb[arch_key]["theory"],
            baseline=baseline)

        print(f"  决策点: {sb['decision_point']['archetype']} | "
              f"前提: {sb['decision_point']['strategic_premise'][:40]}... | "
              f"基线: {sb['decision_point']['baseline']}")
        print(f"  轴类型: {sb['comparison_axes']['axis_type']}"
              f"{'（单线+等待铺垫）' if sb['comparison_axes']['axis_type'] == 3 else ''}")
        for r in sb["routes"]:
            print(f"  [计划] {r['name']} cp={r['cp']} 可行")
            print(f"    线: {' '.join(r['line'])}")
            trends = r["trend"].get("trends", [])
            if trends:
                print(f"    趋势: {[(t.dimension, t.direction) for t in trends]}")
            if r["trend"].get("archetype_shift"):
                print(f"    结构转换: {r['trend']['archetype_shift']}")
            for f in r["unique_facts"][:3]:
                print(f"    独有事实: {f}")
        for d in sb["divergences"]:
            print(f"  分歧深度: {d['pair'][0]} vs {d['pair'][1]} = "
                  f"{d['divergence_depth']} 着 {'✓成对' if d['paired'] else '✗收敛'}")
        if sb["comparison_axes"]["waiting_note"]:
            print(f"  等待铺垫: {sb['comparison_axes']['waiting_note']}")


if __name__ == "__main__":
    main()
