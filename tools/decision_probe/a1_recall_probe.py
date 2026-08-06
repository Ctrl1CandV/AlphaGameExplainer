"""P0-lite A1：方向谓词召回性验证。

验的是什么
----------
FINDINGS-002 §3.2 A1 修订版：`searchmoves` 的语义是**限制搜索起点集合**，引擎在
集合内自由深搜并自行选最强。所以方向谓词需要保证的是「正确执行着 ∈ 候选集」
（**召回**），不需要保证排序正确（精度）。

这是前向机制成立的**必要条件**——候选集捞不到正确执行着，引擎就根本看不见
该计划的正确下法，产出的「计划执行线」是假的。判据门槛因此定在 ≥90%，
不是及格线。

ground truth 的来源与局限
-------------------------
本探针用**文献标注的计划执行着**做 ground truth（Soltis/Pachman 等对各原型
战略计划的经典描述）。FINDINGS-002 P17 要求最终改用 **PGN 实战频率统计**
（免人工标注），但那依赖 Elite DB 落地；在此之前，文献标注是可用的过渡口径——
因为这些原型的计划执行手在文献中是高度一致、无争议的（少数派攻击就是 b4/Rb1
那一套），主观性远低于「候选计划是否合理」那类判断。

**因此本轮结论的定位是：谓词形态是否可写、召回机制是否成立。**
PGN 频率版复验属 P17 落地后的动作，不能用本轮结果替代。

用法
----
    python -m tools.decision_probe.a1_recall_probe
    python -m tools.decision_probe.a1_recall_probe --top-n 8
"""
from typing import Dict, List
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import chess  # noqa: E402

from tools.decision_probe.engine_probe import (  # noqa: E402
    direction_candidates,
    direction_score,
)

OUT_DIR = os.path.join(_ROOT, "data", "quality_benchmark_decision")

# 测试集：原型 → 局面 + 计划 + 文献标注的执行着
#
# FEN 来自 opening_m5_probe 实跑产出（真实开局主线推演到中局），不是手工构造。
# `plan.direction` 是 KB schema 的候选形态，本探针就是要验证它写得对不对。
# `literature_moves` 是文献描述的该计划执行手（SAN），召回率的分母。
CASES = [
    {
        "archetype": "卡尔斯巴德结构",
        "fen": "r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/2NBPN2/PPQ2PPP/R4RK1 w - - 8 11",
        "plans": [
            {
                "name": "少数派攻击",
                "direction": {
                    # A1 首轮实测补 "a"：a4 是支援 b5 突破的标准手，只写 ["b"]
                    # 会漏掉它（只拿到 zone 分 1.0，排不进 top-N）。
                    # KB 编写规范由此得出一条：pawn_files 要含**支援兵线**，
                    # 不只是主突破兵线。
                    "pawn_files": ["a", "b"],
                    "target_zone": "queenside",
                    "break_squares": ["b5"],
                },
                # Soltis《Pawn Structure Chess》卡尔斯巴德章：少数派攻击的执行
                # 是 b4-b5 推进配合车上 b 线。a4 是常见的支援手。
                "literature_moves": ["b4", "Rab1", "Rfb1", "a4", "Rb1"],
                # A1 污染检查用（P0-full）：文献着集合里**混着典型错误着**，
                # 「文献 vs 非文献」二分不能当「正确 vs 错误」。这里显式划出
                # 该局面下的错误执行着（KB typical_mistakes「过早推进未做好
                # 车的配合」→ 车未到位就 b4；实测 b4=-45，正确执行着 a4=40）。
                "mistake_moves": ["b4"],
                "source": "Soltis / Nimzowitsch",
            },
            {
                "name": "中心突破",
                "direction": {
                    "pawn_files": ["e"],
                    "target_zone": "center",
                    "break_squares": ["e4"],
                },
                # 对立计划：放弃后翼行动，改在中心求 e4 突破。
                "literature_moves": ["e4", "Rae1", "Rfe1", "Re1"],
                # 错误着：e4 突破缺乏子力支援（KB typical_mistakes 原文），
                # 实测 e4=-96，正确执行着 Rae1=48。
                "mistake_moves": ["e4"],
                "source": "Pachman",
            },
        ],
    },
    {
        "archetype": "孤后兵 IQP",
        "fen": "r1bq1rk1/pp2bppp/2n2n2/3p4/N7/5NP1/PP2PPBP/R1BQ1RK1 w - - 2 11",
        "plans": [
            {
                "name": "对孤兵施压",
                "direction": {
                    "pawn_files": ["d"],
                    "target_zone": "center",
                    "break_squares": ["d4"],
                    # A1 首轮实测补：施压类计划必须声明被围攻的格子。
                    # 孤兵在 d5，Nc5/Nc3/Be3 都是「攻 d5」而非「走进 center」。
                    "pressure_squares": ["d5"],
                },
                # IQP 反方标准计划：重子压 d 线、轻子围攻 d5。
                #
                # 【首轮 ground truth 修正】原写 ["Rd1","Rad1","Rfd1","Nc5","Be3","Nc3"]，
                # 经 python-chess 逐着核验，Nc5 与 Be3 **几何上并不攻击 d5**
                # （马在 c5 攻 d7/b7/a6/a4/b3/d3/e6/e4；象在 e3 与 d5 不同斜线），
                # 是手写标注时凭印象填错。真正施压 d5 的着为：
                #   Nc3 直接攻 d5；Nb6 直接攻 d5；e4 兵攻 d5；
                #   Ne5 让开 f3，打通 g2 象到 d5 的斜线（间接施压）。
                # 这一处错误是 P17「ground truth 免人工标注化」的实测证据：
                # 手写文献着会带错，PGN 实战频率统计不会。
                "literature_moves": ["Rd1", "Rad1", "Rfd1", "Nc3", "Nb6", "e4", "Ne5"],
                # 错误着（该局面下评估全部为负，实测：Nb6=-489, Ne5=-444,
                # e4=-96——Nc3=22 才是正确执行着）：
                #   - Nb6/Ne5：方向对（施压 d5）但该局面执行坏——马深入被驱逐；
                #   - e4：过早兑孤兵，围攻与推进脱节（KB typical_mistakes）。
                "mistake_moves": ["Nb6", "Ne5", "e4"],
                "source": "Pachman / Watson",
            },
        ],
    },
    {
        "archetype": "荷兰石墙",
        "fen": "rn3rk1/pb2q1pp/1ppbpn2/3pNp2/2PP4/1P4P1/PB1NPPBP/R2Q1RK1 w - - 2 11",
        "plans": [
            {
                "name": "后翼扩张",
                "direction": {
                    "pawn_files": ["c", "b"],
                    "target_zone": "queenside",
                    "break_squares": ["c5"],
                },
                "literature_moves": ["c5", "b4", "Rc1", "Rab1", "Rfc1", "Rb1", "Rc2"],
                # 错误着：子力未协调就过早兵扩张（实测 c5=-139, b4=-84，
                # 正确执行着 Rc1=31）——先把重子调上后翼线再推兵。
                "mistake_moves": ["c5", "b4"],
                "source": "Soltis",
            },
            {
                # 阶段 1 新增：中心突破（e3-e4 打开石墙中心线）
                "name": "中心突破",
                "direction": {
                    "pawn_files": ["e"],
                    "target_zone": "center",
                    "break_squares": ["e4"],
                },
                # e2-e3 准备 e4（e2-e4 两步跳不合法——首着 e3 是文献执行）。
                "literature_moves": ["e3"],
                "source": "Pachman",
            },
        ],
    },
    {
        "archetype": "马洛齐束缚",
        "fen": "r2q1rk1/pp2ppbp/3pbnp1/8/2P1P3/2N1B3/PP1QBPPP/R3K2R w KQ - 5 11",
        "plans": [
            {
                "name": "王翼进攻",
                "direction": {
                    "pawn_files": ["h", "g"],
                    "target_zone": "kingside",
                    "break_squares": ["h5"],
                },
                # 马洛齐束缚方的经典方案：h4-h5 冲击 + 车上 h 线。
                "literature_moves": ["h4", "g4", "Rh4", "h3", "Rdg1"],
                # 错误着：子力未到位就过早王翼兵冲击（实测 h4=-69, g4=-180，
                # 该局面下唯一合理执行是 h3=-66 的安静准备）。
                "mistake_moves": ["h4", "g4"],
                "source": "Soltis 西西里章",
            },
            {
                # 阶段 1 新增（KB 扩容）：马洛齐后翼扩张——与王翼进攻构成轴 1
                "name": "后翼扩张",
                "direction": {
                    "pawn_files": ["b", "a"],
                    "target_zone": "queenside",
                    "break_squares": ["b5"],
                },
                # b4-b5 冲击黑后翼兵（配合车 b 线），压制黑后翼扩大空间。
                "literature_moves": ["b4", "a4", "Rb1"],
                "source": "Pachman",
            },
        ],
    },
    {
        "archetype": "悬兵结构",
        "fen": "2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/1R2R1K1 w - - 1 21",
        "plans": [
            {
                # 阶段 1 新增：悬兵推进（c5/d5 突破）
                "name": "推进悬兵",
                "direction": {
                    "pawn_files": ["c", "d"],
                    "target_zone": "center",
                    "break_squares": ["c5", "d5"],
                },
                # c4-c5 推进换取空间（d4-d5 被黑 d6 兵挡——只列合法执行着；
                # 文献序列中 c5 突破是悬兵推进的核心手）。
                "literature_moves": ["c5"],
                "source": "Soltis 悬兵章",
            },
            {
                # 阶段 1 新增：保持悬兵（轻子中心化利用双兵动态）
                "name": "保持悬兵（利用动态潜力）",
                "direction": {
                    "target_zone": "center",
                    "outpost_squares": ["e5", "c5"],
                },
                # Nf3-e5 占中心前哨（outpost e5），保持 c4+d4 对中心的控制。
                "literature_moves": ["Ne5"],
                "source": "Soltis 悬兵章",
            },
        ],
    },
    {
        "archetype": "通路兵/多数兵结构",
        # Dragon 变例局面：白后翼 a2b2c2 3v2 多数（黑 a6+b4）+ 白 g2/h2
        # 兵在（王翼行动可测）——Najdorf 局面 g 线无兵（g2 是后）、h 兵
        # 已过线，王翼行动测不了，故统一用 Dragon。
        "fen": "r2qr3/3bRpk1/p2p2p1/3P2Qp/1p6/1N3P2/PPP3PP/1K1R4 w - - 1 21",
        "plans": [
            {
                # 阶段 1 新增：多数翼推进（a4-b4-c4 制造通路兵）。
                # 注意 b2-b4 被黑 b4 兵占（非法）——文献着过合法性过滤后
                # 只列适用者（a4/c4）。
                "name": "多数翼推进",
                "direction": {
                    "pawn_files": ["a", "b", "c"],
                    "target_zone": "queenside",
                    "break_squares": ["c5", "b5"],
                },
                "literature_moves": ["a4", "b4", "c4"],
                "source": "Soltis 多数兵章",
            },
            {
                # 阶段 1 新增：王翼行动（放弃多数推进转攻王翼）
                "name": "王翼行动",
                "direction": {
                    "pawn_files": ["g", "h"],
                    "target_zone": "kingside",
                    "break_squares": ["h5"],
                },
                "literature_moves": ["g4", "h4"],
                "source": "Pachman",
            },
        ],
    },
]


def evaluate_case(board: chess.Board, plan: dict, top_n: int) -> dict:
    """算一个计划的召回率：文献执行着有多少落在 top-N 候选集内。"""
    candidates = direction_candidates(board, plan["direction"], top_n=top_n)
    cand_set = {board.san(mv) for mv in candidates}

    # 文献着法先过合法性过滤——该局面下不合法的着不计入分母
    # （文献描述的是整个计划，不是每一手在当前局面都可走）
    legal_sans = {board.san(mv) for mv in board.legal_moves}
    applicable = [s for s in plan["literature_moves"] if s in legal_sans]
    inapplicable = [s for s in plan["literature_moves"] if s not in legal_sans]

    hit = [s for s in applicable if s in cand_set]
    missed = [s for s in applicable if s not in cand_set]

    recall = (len(hit) / len(applicable) * 100) if applicable else None

    # 附带记录每个文献着的得分，便于诊断谓词维度权重
    scores = {}
    for mv in board.legal_moves:
        san = board.san(mv)
        if san in applicable:
            scores[san] = direction_score(board, mv, plan["direction"])

    return {
        "plan": plan["name"],
        "source": plan["source"],
        "applicable": applicable,
        "inapplicable": inapplicable,
        "hit": hit,
        "missed": missed,
        "recall_pct": recall,
        "candidate_set": sorted(cand_set),
        "literature_move_scores": scores,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=10,
                    help="候选集大小（FINDINGS A1 修订：召回优先，8~10）")
    args = ap.parse_args()

    print(f"A1 召回性验证 | top-N={args.top_n}")
    print("ground truth：文献标注的计划执行着（P17 落地后改 PGN 实战频率复验）")
    print("判据：正确执行着落在候选集内的比例 ≥90%（必要条件，非及格线）")
    print("-" * 76)

    records: List[dict] = []
    for case in CASES:
        board = chess.Board(case["fen"])
        print(f"\n{case['archetype']}")
        for plan in case["plans"]:
            r = evaluate_case(board, plan, args.top_n)
            r["archetype"] = case["archetype"]
            r["fen"] = case["fen"]
            records.append(r)

            rec = r["recall_pct"]
            rec_s = f"{rec:5.1f}%" if rec is not None else "  n/a "
            flag = "✓" if (rec is not None and rec >= 90) else "✗"
            print(f"  {flag} {r['plan']:12} 召回 {rec_s} "
                  f"({len(r['hit'])}/{len(r['applicable'])})")
            if r["missed"]:
                print(f"      漏掉: {', '.join(r['missed'])}")
                for m in r["missed"]:
                    print(f"        {m} 得分 {r['literature_move_scores'].get(m, 0):.1f}")
            if r["inapplicable"]:
                print(f"      本局面不适用（不计分母）: {', '.join(r['inapplicable'])}")

    # 汇总
    valid = [r for r in records if r["recall_pct"] is not None]
    total_applicable = sum(len(r["applicable"]) for r in valid)
    total_hit = sum(len(r["hit"]) for r in valid)
    overall = (total_hit / total_applicable * 100) if total_applicable else 0.0
    per_plan_pass = sum(1 for r in valid if r["recall_pct"] >= 90)

    print("\n" + "=" * 76)
    print("A1 召回汇总")
    print("=" * 76)
    print(f"  计划数              {len(valid)}")
    print(f"  文献执行着（适用）    {total_applicable}")
    print(f"  召回命中            {total_hit}")
    print(f"  整体召回率          {overall:.1f}%")
    print(f"  单计划达标（≥90%）   {per_plan_pass}/{len(valid)}")

    print("\n" + "=" * 76)
    if overall >= 90 and per_plan_pass == len(valid):
        verdict = (f"A1 通过（整体召回 {overall:.1f}%，全部计划达标）"
                   f"→ 方向谓词形态可写，前向机制的必要条件成立")
        print(f"  ✓ {verdict}")
    elif overall >= 90:
        verdict = (f"A1 部分通过（整体 {overall:.1f}% 达标，但 "
                   f"{len(valid) - per_plan_pass} 个计划单独不达标）"
                   f"→ 谓词可写但个别计划需调维度权重")
        print(f"  ~ {verdict}")
    else:
        verdict = (f"A1 不通过（整体召回 {overall:.1f}% < 90%）"
                   f"→ 谓词漏掉正确执行着，前向路线需重新设计打分维度")
        print(f"  ✗ {verdict}")
    print("=" * 76)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "a1_recall_probe_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"top_n": args.top_n},
            "summary": {
                "plan_count": len(valid),
                "total_applicable": total_applicable,
                "total_hit": total_hit,
                "overall_recall_pct": round(overall, 1),
                "per_plan_pass": per_plan_pass,
                "ground_truth": "文献标注（P17 落地后改 PGN 实战频率复验）",
                "verdict": verdict,
            },
            "records": records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
