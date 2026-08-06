"""阶段 4 验证探针：示例局面全链路对照（PLAN-009 阶段 4 验证方式）。

对阶段 1 的 6 原型示例局面各跑一遍：
  - explore_forward：前向约束线（首着 + 前 6 着 + cp）——人工确认
    「前向线确实体现该战略」；
  - explore_open：自由 MultiPV top3——反向验证对照；
  - assess_feasibility：可行性闸（计划最优 vs 全局最优 gap）；
  - assess_endorsement：背书判定（方向内着占比）；
  - waiting_baseline：等待基线（不作为的代价）。

用法：
    python -m tools.decision_probe.stage4_probe
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

from src.solver.branch_explorer import (  # noqa: E402
    assess_endorsement,
    assess_feasibility,
    explore_forward,
    explore_open,
    waiting_baseline,
)

# 6 原型示例局面（阶段 1/2 实测 FEN）
SITUATIONS = [
    ("卡尔斯巴德", "r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/2NBPN2/PPQ2PPP/"
     "R4RK1 w - - 8 11"),
    ("孤后兵 IQP", "r1bq1rk1/pp2bppp/2n2n2/3p4/N7/5NP1/PP2PPBP/"
     "R1BQ1RK1 w - - 2 11"),
    ("悬兵", "2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/"
     "1R2R1K1 w - - 1 21"),
    ("马洛齐", "r2q1rk1/pp2ppbp/3pbnp1/8/2P1P3/2N1B3/PP1QBPPP/"
     "R3K2R w KQ - 5 11"),
    ("石墙", "rn3rk1/pb2q1pp/1ppbpn2/3pNp2/2PP4/1P4P1/PB1NPPBP/"
     "R2Q1RK1 w - - 2 11"),
    ("多数兵", "r2qr3/3bRpk1/p2p2p1/3P2Qp/1p6/1N3P2/PPP3PP/"
     "1K1R4 w - - 1 21"),
]


def main() -> None:
    import json

    sf = os.getenv("STOCKFISH_PATH", "")
    if not os.path.isabs(sf):
        sf = os.path.normpath(os.path.join(_ROOT, sf))
    kb = json.load(open(os.path.join(_ROOT, "data", "structure_kb.json"),
                        encoding="utf-8"))

    # 局面 → 计划（KB plans 全部计划按局面测；悬兵/石墙等只测该局面的
    # 适用计划——本探针按 KB 顺序全测，标注 mover_side 供人工判断）
    for sit_name, fen in SITUATIONS:
        b = chess.Board(fen)
        print("\n" + "=" * 72)
        print(f"=== {sit_name} ===")
        arch_key = {"卡尔斯巴德": "carlsbad", "孤后兵 IQP": "iqp",
                    "悬兵": "hanging", "马洛齐": "maroczy",
                    "石墙": "stonewall", "多数兵": "majority"}[sit_name]
        plans = kb[arch_key]["plans"]

        opens = explore_open(b, sf, k=4, depth=14)
        print(f"  全局 MultiPV top4: "
              f"{[(b.san(l.move), l.cp) for l in opens]}")
        baseline = waiting_baseline(b, sf, depth=12)
        print(f"  等待基线: {baseline}")

        for plan in plans:
            line = explore_forward(b, plan, sf, depth=14)
            if line is None or not line.pv:
                print(f"  [{plan['name']}] 无约束线")
                continue
            b2 = b.copy()
            sans = []
            for mv in line.pv[:6]:
                try:
                    sans.append(b2.san(mv))
                    b2.push(mv)
                except Exception:
                    break
            feas, gap = assess_feasibility(line.cp,
                                           opens[0].cp if opens else None)
            end = assess_endorsement(b, [plan], opens)
            e = end[plan["name"]]
            print(f"  [{plan['name']}] 首着={b.san(line.move)} cp={line.cp}")
            print(f"      前6着: {' '.join(sans)}")
            print(f"      可行性 gap={gap} {'✓' if feas else '✗不可行'}"
                  f" | 背书 ratio={e['ratio']} ({e['in_direction']}/{e['total']})"
                  f" {'✓' if e['endorsed'] else '✗'}")


if __name__ == "__main__":
    main()
