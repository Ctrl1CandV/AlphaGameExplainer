"""M5 对照探针：从标准开局着法序列生成中局局面，测 M5 存活率。

存在意义
--------
`m5_smoke.py` 证明了 puzzle 库的局面 M5 存活 0%，但那只证明「puzzle 库不行」，
**没有证明「普通中局局面行」**。后者是 ADR-020 换源决策的真正前提，必须独立验证——
否则可能出现最坏情况：换了源，但中局局面同样过不了 M5，整个方案的价值主张落空。

本探针不依赖任何下载数据：用 python-chess 播放公认的开局着法序列到中局，
再跑同一套 M5 判据（复用 `engine_probe`，保证与 m5_smoke 口径完全一致）。

判据口径与 m5_smoke 完全相同（同一个 `assess_m5` 函数），所以两组存活率可直接对比。
这是本探针的设计要点：**对照实验必须用同一把尺子**。

用法
----
    python -m tools.decision_probe.opening_m5_probe [--depth 14] [--standout 150]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import chess  # noqa: E402

from tools.decision_probe.engine_probe import (  # noqa: E402
    DEFAULT_EQUIV_CP,
    DEFAULT_MULTIPV,
    DEFAULT_STANDOUT_CP,
    EngineProbe,
    assess_m5,
    count_material,
    distinct_direction_count,
    equivalent_first_moves,
    resolve_stockfish,
)

# 公认开局主线（SAN）。选取标准：
#   1. 覆盖 structure_kb 首批目标原型（卡尔斯巴德/IQP/马洛齐/石墙/贝诺尼/龙式等）；
#   2. 都是教科书主线，不是冷门变着——保证「典型中局」的代表性；
#   3. 长度 16~24 ply，落在中局而非开局尾。
# 非法着法会被优雅跳过并计入 skipped（防手写笔误污染结论）。
OPENING_LINES: Dict[str, List[str]] = {
    "QGD兑变_卡尔斯巴德": [
        "d4", "d5", "c4", "e6", "Nc3", "Nf6", "cxd5", "exd5", "Bg5", "Be7",
        "e3", "c6", "Bd3", "Nbd7", "Qc2", "O-O", "Nf3", "Re8", "O-O", "Nf8",
    ],
    "QGD塔拉什_IQP": [
        "d4", "d5", "c4", "e6", "Nc3", "c5", "cxd5", "exd5", "Nf3", "Nc6",
        "g3", "Nf6", "Bg2", "Be7", "O-O", "O-O", "dxc5", "Bxc5", "Na4", "Be7",
    ],
    "尼姆佐印度_e3": [
        "d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3", "O-O", "Bd3", "d5",
        "Nf3", "c5", "O-O", "Nc6", "a3", "Bxc3", "bxc3", "dxc4", "Bxc4", "Qc7",
    ],
    "西西里马洛齐束缚": [
        "e4", "c5", "Nf3", "Nc6", "d4", "cxd4", "Nxd4", "g6", "c4", "Nf6",
        "Nc3", "d6", "Be2", "Nxd4", "Qxd4", "Bg7", "Be3", "O-O", "Qd2", "Be6",
    ],
    "荷兰石墙": [
        "d4", "e6", "c4", "f5", "Nf3", "Nf6", "g3", "d5", "Bg2", "c6",
        "O-O", "Bd6", "b3", "Qe7", "Bb2", "O-O", "Nbd2", "b6", "Ne5", "Bb7",
    ],
    "贝诺尼现代": [
        "d4", "Nf6", "c4", "c5", "d5", "e6", "Nc3", "exd5", "cxd5", "d6",
        "e4", "g6", "Nf3", "Bg7", "Be2", "O-O", "O-O", "Re8", "Nd2", "Na6",
    ],
    "西西里龙式": [
        "e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "g6",
        "Be3", "Bg7", "f3", "O-O", "Qd2", "Nc6", "O-O-O", "d5", "exd5", "Nxd5",
    ],
    "西西里谢维宁根": [
        "e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "e6",
        "Be2", "Be7", "O-O", "O-O", "f4", "Nc6", "Be3", "Bd7", "Nb3", "a6",
    ],
    "卡罗康推进": [
        "e4", "c6", "d4", "d5", "e5", "Bf5", "Nf3", "e6", "Be2", "c5",
        "Be3", "Qb6", "Nc3", "Nc6", "O-O", "cxd4", "Nxd4", "Nxd4", "Bxd4", "Bc5",
    ],
    "法兰西温纳维尔": [
        "e4", "e6", "d4", "d5", "Nc3", "Bb4", "e5", "c5", "a3", "Bxc3",
        "bxc3", "Ne7", "Qg4", "O-O", "Nf3", "Nbc6", "Bd3", "f5", "Qg3", "c4",
    ],
    "斯拉夫防御": [
        "d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "dxc4", "a4", "Bf5",
        "e3", "e6", "Bxc4", "Bb4", "O-O", "O-O", "Qe2", "Nbd7", "Rd1", "Qe7",
    ],
    "王印度经典": [
        "d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Nf3", "O-O",
        "Be2", "e5", "O-O", "Nc6", "d5", "Ne7", "b4", "Nh5", "Re1", "f5",
    ],
    "格林菲尔德兑变": [
        "d4", "Nf6", "c4", "g6", "Nc3", "d5", "cxd5", "Nxd5", "e4", "Nxc3",
        "bxc3", "Bg7", "Nf3", "c5", "Be3", "Qa5", "Qd2", "O-O", "Rc1", "cxd4",
    ],
    "西班牙封闭": [
        "e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7",
        "Re1", "b5", "Bb3", "d6", "c3", "O-O", "h3", "Na5", "Bc2", "c5",
    ],
    "意大利慢棋": [
        "e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "d6",
        "O-O", "O-O", "Re1", "a6", "Nbd2", "Ba7", "Nf1", "Ne7", "Ng3", "Ng6",
    ],
    "英国式对称": [
        "c4", "c5", "Nf3", "Nf6", "Nc3", "d5", "cxd5", "Nxd5", "g3", "Nc6",
        "Bg2", "e6", "O-O", "Be7", "d3", "O-O", "Be3", "Nxe3", "fxe3", "Qb6",
    ],
    "后印度": [
        "d4", "Nf6", "c4", "e6", "Nf3", "b6", "g3", "Bb7", "Bg2", "Be7",
        "O-O", "O-O", "Nc3", "d5", "cxd5", "exd5", "Ne5", "Na6", "Bf4", "c5",
    ],
    "加泰罗尼亚": [
        "d4", "Nf6", "c4", "e6", "g3", "d5", "Bg2", "Be7", "Nf3", "O-O",
        "O-O", "dxc4", "Qc2", "a6", "Qxc4", "b5", "Qc2", "Bb7", "Bd2", "Be4",
    ],
    "四骑士": [
        "e4", "e5", "Nf3", "Nc6", "Nc3", "Nf6", "d4", "exd4", "Nxd4", "Bb4",
        "Nxc6", "bxc6", "Bd3", "d5", "exd5", "cxd5", "O-O", "O-O", "Bg5", "c6",
    ],
    "伦敦体系": [
        "d4", "d5", "Nf3", "Nf6", "Bf4", "e6", "e3", "Bd6", "Bxd6", "Qxd6",
        "Nbd2", "O-O", "c4", "c6", "Bd3", "Nbd7", "O-O", "b6", "Qc2", "Bb7",
    ],
}


def build_position(sans: List[str]) -> chess.Board | None:
    """播放 SAN 序列，返回终局面；任一着法非法则返回 None。"""
    board = chess.Board()
    for san in sans:
        try:
            board.push_san(san)
        except ValueError:
            return None
    return board


def main() -> int:
    ap = argparse.ArgumentParser(description="标准开局中局局面的 M5 对照探针")
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--standout", type=int, default=DEFAULT_STANDOUT_CP)
    ap.add_argument("--equiv", type=int, default=DEFAULT_EQUIV_CP)
    ap.add_argument("--multipv", type=int, default=DEFAULT_MULTIPV)
    ap.add_argument("--m6-abs-max", type=int, default=300,
                    help="M6 评估窗口：|cp| 上限（超出=已定胜负，无战略取舍）")
    args = ap.parse_args()

    try:
        sf = resolve_stockfish()
    except FileNotFoundError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    print(f"开局主线 {len(OPENING_LINES)} 条 | depth={args.depth} "
          f"standout={args.standout}cp equiv={args.equiv}cp k={args.multipv}")
    print("对照口径：与 m5_smoke 复用同一 assess_m5 函数，两组存活率可直接对比")
    print("-" * 76)

    records = []
    skipped = []

    with EngineProbe(sf) as probe:
        for i, (name, sans) in enumerate(sorted(OPENING_LINES.items()), 1):
            board = build_position(sans)
            if board is None:
                skipped.append(name)
                print(f"[{i:2}/{len(OPENING_LINES)}] {name:22} 跳过（SAN 序列含非法着）")
                continue

            material = count_material(board)
            m5 = assess_m5(board, probe, k=args.multipv, depth=args.depth,
                           standout_cp=args.standout)

            rec = {
                "name": name,
                "fen": board.fen(),
                "ply": len(sans),
                "material": material,
                "m5_pass": m5.passed,
                "m5_reason": m5.reason,
                "gap_cp": m5.gap_cp,
                "root_cp": m5.top1_cp,
            }

            if m5.passed:
                # M6：评估窗口（排除已定胜负的局面）
                m6_pass = m5.top1_cp is not None and abs(m5.top1_cp) <= args.m6_abs_max
                rec["m6_pass"] = m6_pass
                # M8：近等强首着 ≥2 且方向不同（复用 engine_probe 的单一定义）
                has_choice, equiv_count = equivalent_first_moves(
                    m5.multipv, equiv_cp=args.equiv)
                zone_count = distinct_direction_count(m5.multipv, equiv_cp=args.equiv)
                rec["equiv_count"] = equiv_count
                rec["zone_count"] = zone_count
                rec["m8_pass"] = m6_pass and has_choice and zone_count >= 2
                flag = "PASS" if rec["m8_pass"] else ("M6止" if not m6_pass else "M8止")
                print(f"[{i:2}/{len(OPENING_LINES)}] {name:22} M5✓ "
                      f"eval={m5.top1_cp:+5}cp 等强{equiv_count}着 区域{zone_count} → {flag}")
            else:
                rec["m6_pass"] = False
                rec["m8_pass"] = False
                rec["equiv_count"] = 0
                rec["zones"] = []
                print(f"[{i:2}/{len(OPENING_LINES)}] {name:22} M5✗ {m5.reason}")

            records.append(rec)

    total = len(records)
    if total == 0:
        print("无有效局面", file=sys.stderr)
        return 1

    m5_pass = sum(1 for r in records if r["m5_pass"])
    m6_pass = sum(1 for r in records if r["m6_pass"])
    m8_pass = sum(1 for r in records if r["m8_pass"])

    def pct(n: int) -> float:
        return round(100.0 * n / total, 1)

    print("-" * 76)
    print()
    print("=" * 76)
    print("开局中局局面漏斗（分母 = 成功构造的局面数）")
    print("=" * 76)
    rows = [
        ("有效局面", total),
        (f"M5 无强制战术（standout={args.standout}cp）", m5_pass),
        (f"M6 评估窗口 |cp|≤{args.m6_abs_max}", m6_pass),
        (f"M8 近等强首着 ≥2 且方向 ≥2", m8_pass),
    ]
    for label, n in rows:
        print(f"  {label:44} {n:4}   {pct(n):5.1f}%")

    if skipped:
        print(f"\n  跳过（SAN 笔误）：{len(skipped)} 条 —— {', '.join(skipped)}")

    fails: Dict[str, int] = {}
    for r in records:
        if not r["m5_pass"]:
            fails[r["m5_reason"]] = fails.get(r["m5_reason"], 0) + 1
    if fails:
        print("\nM5 未通过原因：")
        for reason, n in sorted(fails.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:36} {n:4}")

    gaps = sorted(r["gap_cp"] for r in records if r["gap_cp"] is not None)
    if gaps:
        def q(p: float) -> int:
            return gaps[min(len(gaps) - 1, int(len(gaps) * p))]
        print("\n首选-次选 gap 分布（cp）：")
        print(f"  P10 {q(0.10):5}   中位 {q(0.50):5}   P90 {q(0.90):5}   "
              f"最小/最大 {gaps[0]} / {gaps[-1]}")

    # 与 puzzle 源的对照裁决
    print()
    print("=" * 76)
    print("对照裁决（本探针 vs m5_smoke 的 puzzle 源）")
    print("=" * 76)
    print(f"  puzzle 源 M5 存活：  0.0%  （0/79，gap 最小 274cp）")
    print(f"  开局中局 M5 存活： {pct(m5_pass):5.1f}%  ({m5_pass}/{total})")
    print()
    if m5_pass == 0:
        verdict = ("两个源 M5 都归零 → **换源无法解决问题**。ADR-020 的价值主张"
                   "（存在可讲的战略取舍局面）需重新审视，M5 判据本身可能过严。")
        print("  ✗ " + verdict)
    elif pct(m5_pass) < 30:
        verdict = (f"中局源 M5 存活 {pct(m5_pass)}%，显著高于 puzzle 源的 0% 但偏低 → "
                   "换源方向成立，但 M5/M6/M8 阈值需在阶段 3 用真实 PGN 校准。")
        print("  ~ " + verdict)
    else:
        verdict = (f"中局源 M5 存活 {pct(m5_pass)}% vs puzzle 源 0% → "
                   "**换源决策的价值前提得到验证**：普通中局确实存在无强制战术的局面，"
                   "ADR-020 的 PGN 主源路线成立。")
        print("  ✓ " + verdict)
    print("=" * 76)

    out_dir = os.path.join(_ROOT, "data", "quality_benchmark_decision")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "opening_m5_probe_result.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "config": {
                "depth": args.depth,
                "standout_cp": args.standout,
                "equiv_cp": args.equiv,
                "multipv_k": args.multipv,
                "m6_abs_cp_max": args.m6_abs_max,
            },
            "summary": {
                "total": total,
                "m5_pass": m5_pass,
                "m6_pass": m6_pass,
                "m8_pass": m8_pass,
                "m5_survival_pct": pct(m5_pass),
                "m8_survival_pct": pct(m8_pass),
                "skipped": skipped,
                "m5_fail_buckets": fails,
                "verdict": verdict,
            },
            "positions": records,
        }, fh, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
