"""阶段 7 验证探针：--text 模式实跑决策解说（PLAN-009 阶段 7 验证）。

用悬兵示例局面（双计划可行 + 分歧成对）跑全链路 →
generate_decision_commentary（真实 LLM）→ 人工核对：
  无坐标泄漏 / 无硬事实错 / 双计划命中各自独有事实 / 认识论措辞合规 /
  轴 3 未被写成等强对比。

用法：
    python -m tools.decision_probe.stage7_probe
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
from src.commentator.decision_commentary import (  # noqa: E402
    generate_decision_commentary,
)
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


def main() -> None:
    import json

    sf = os.getenv("STOCKFISH_PATH", "")
    if not os.path.isabs(sf):
        sf = os.path.normpath(os.path.join(_ROOT, sf))
    kb = json.load(open(os.path.join(_ROOT, "data", "structure_kb.json"),
                        encoding="utf-8"))

    fen = "2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/1R2R1K1 w - - 1 21"
    b = chess.Board(fen)
    opens = explore_open(b, sf, k=4, depth=14)
    baseline = waiting_baseline(b, sf, depth=12)
    outcomes = []
    for plan in kb["hanging"]["plans"]:
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
        DecisionInput(fen=fen, provenance="d5"), outcomes,
        archetype="hanging",
        strategic_premise=kb["hanging"]["theory"],
        baseline=baseline)

    print("=" * 76)
    print("=== 悬兵决策解说（真实 LLM）===")
    print("=" * 76)
    commentary = generate_decision_commentary(
        DecisionInput(fen=fen, provenance="d5"), sb)

    if commentary.opening:
        print(f"\n【开场】{commentary.opening}")
    for seg in commentary.segments:
        ntype = {0: "开场", 1: "计划甲", 2: "计划乙", 3: "对比", 4: "总结"}.get(
            int(getattr(seg, "id", -1)), "?")
        print(f"\n[{ntype}] {getattr(seg, 'voiceover', '')}")
    if commentary.summary:
        print(f"\n【总结】{commentary.summary}")
    print(f"\n--- chunks {commentary.chunks_succeeded}/{commentary.chunks_total}"
          f" retries {commentary.retries_total} ---")

    # 人工核对提示
    print("\n核对清单：")
    print("  1. 无坐标/无走法泄漏（不得出现 a3/e4 类）")
    print("  2. 双计划（推进悬兵/保持悬兵）都出现且各命中独有事实")
    print("  3. 无「引擎认为」「大师的选择」类措辞（认识论/P12）")
    print("  4. 对比落到具体结构差异（无「各有千秋」套话）")


if __name__ == "__main__":
    main()
