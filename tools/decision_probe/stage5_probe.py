"""阶段 5 验证探针：示例计划后果投射 + 代价量化（PLAN-009 阶段 5 验证）。

对阶段 1 的 6 原型示例局面每计划跑：
  - project：延伸推演 + 单调趋势（第 8/14/20 着采样）+ 对方应招扰动
    + 结构类型转换（P19）；
  - quantify_tradeoffs：承诺度（兵着/兑子）、开放线差分、弱格提示、
    好着走廊宽度、唯一好着密度。

人工核对：趋势与真实棋理一致；扰动一致率记录在案（若绝大多数趋势被
扰动否掉——投射深度过深，需缩短）。

用法：
    python -m tools.decision_probe.stage5_probe
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

from src.solver.branch_explorer import explore_forward, explore_open  # noqa: E402
from src.solver.consequence_projector import (  # noqa: E402
    quantify_tradeoffs,
    project,
)

SITUATIONS = [
    ("卡尔斯巴德", "carlsbad", "r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/"
     "2NBPN2/PPQ2PPP/R4RK1 w - - 8 11"),
    ("孤后兵 IQP", "iqp", "r1bq1rk1/pp2bppp/2n2n2/3p4/N7/5NP1/PP2PPBP/"
     "R1BQ1RK1 w - - 2 11"),
    ("悬兵", "hanging", "2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/"
     "1R2R1K1 w - - 1 21"),
    ("马洛齐", "maroczy", "r2q1rk1/pp2ppbp/3pbnp1/8/2P1P3/2N1B3/PP1QBPPP/"
     "R3K2R w KQ - 5 11"),
    ("石墙", "stonewall", "rn3rk1/pb2q1pp/1ppbpn2/3pNp2/2PP4/1P4P1/"
     "PB1NPPBP/R2Q1RK1 w - - 2 11"),
    ("多数兵", "majority", "r2qr3/3bRpk1/p2p2p1/3P2Qp/1p6/1N3P2/PPP3PP/"
     "1K1R4 w - - 1 21"),
]


def main() -> None:
    import json

    sf = os.getenv("STOCKFISH_PATH", "")
    if not os.path.isabs(sf):
        sf = os.path.normpath(os.path.join(_ROOT, sf))
    kb = json.load(open(os.path.join(_ROOT, "data", "structure_kb.json"),
                        encoding="utf-8"))

    consistencies = []
    for sit_name, arch_key, fen in SITUATIONS:
        b = chess.Board(fen)
        print("\n" + "=" * 76)
        print(f"=== {sit_name} ===")
        for plan in kb[arch_key]["plans"]:
            line = explore_forward(b, plan, sf, depth=14)
            if line is None or not line.pv:
                print(f"  [{plan['name']}] 无约束线")
                continue
            res = project(line, b, sf)
            tm = quantify_tradeoffs(line, b, sf,
                                    open_lines=explore_open(b, sf, k=4,
                                                            depth=14))
            consistencies.append(res["perturb_consistency"])
            print(f"  [{plan['name']}] 首着={b.san(line.move)} cp={line.cp}")
            if res["trends"]:
                for t in res["trends"]:
                    print(f"    趋势: {t.dimension} {t.direction} "
                          f"{[round(x, 2) for x in t.samples]}")
            else:
                print(f"    趋势: 无（扰动一致率 "
                      f"{res['perturb_consistency']}——全部被否）")
            if res["archetype_shift"]:
                print(f"    结构转换: {res['archetype_shift'][0]} → "
                      f"{res['archetype_shift'][1]}（P19）")
            print(f"    代价: 兵着 {tm.pawn_moves} 兑子 {tm.captures} "
                  f"开放线Δ {tm.open_files_delta:+d} "
                  f"走廊 {tm.corridor_roots} 唯一密度 {tm.unique_ratio}"
                  f"{' 弱格:' + tm.weak_square_hint if tm.weak_square_hint else ''}")

    if consistencies:
        print("\n" + "=" * 76)
        print(f"扰动一致率: {[round(c, 2) for c in consistencies]} "
              f"均值 {round(sum(consistencies) / len(consistencies), 2)}"
              f"（若绝大多数趋势被否——投射深度过深，需缩短）")


if __name__ == "__main__":
    main()
