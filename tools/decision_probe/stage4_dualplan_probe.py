"""阶段 4 双计划筛假设冒烟（PLAN-010，T3 假设验证，半天量级）。

**核心假设**：从 PGN 中局面里筛出「≥2 条计划同时过机制闸」的局面，
存活率是否足以支撑决策管线的产量？这是投入完整挖掘器交付（阶段 5）
之前的最低成本验证——对齐 PLAN-009 当年 M5 冒烟的做法。

漏斗（复用产品链路单一事实来源，约束 8）：
  PGN 中局面（复用 pgn_m5_probe 的采样器）
    → detect_pawn_structure 识别原型
    → in_production 产品池闸（与 decision_pipeline 同口径）
    → applicable_mover_side 角色筛（同口径）
    → 对该原型每条计划跑 explore_forward + assess_feasibility + goal_trajectory
       机制闸（feasible = cp差通过 AND goal_ok，与 decision_pipeline 逐字同口径）
    → 统计「≥2 计划 feasible=True」的局面（双计划候选）
    → 结构可分粗筛（REV-002）：双计划候选再算两计划终局特征**两两距离的最大值**，
       仅保留 max_pair ≥ 0.5 的局面（防「名义双计划、实际两条线趋同」）

**「结构可分粗筛」≠ 阶段 3 的 A3 判定**（后者用两两距离的中位、且 estimator
待 planner 裁决）。本层刻意用 `max(pair_dists) ≥ 0.5` 这一**松代理**：只要任一
计划对分得够开就放行，宁可漏筛不可过严（矿脉理论无限，从容错）。字段/统计
一律叫 `distinct_prefilter`，与「A3」脱钩，避免与阶段 3 的 2/5 横向误读。

**裁决按「已验证原型」口径复算，不看全原型 headline**（阶段 3 peer_review
校准）：阶段 3 P0-full A3 实测通过的只有 carlsbad/maroczy（`A3_VERIFIED`），
iqp 从未测、majority 已失败、hanging 争议。全原型的 gate/prefilter 会被
iqp+majority 主导而虚高，故 Go/No-Go 只对 `A3_VERIFIED` 口径的千局外推
对标 `min_double`，全原型口径仅作对照记录、不作裁决依据。

**桶去重致千局外推是上界**（bucket_cap=2，位置产出对局数次线性），
外推数标注为「乐观上界」。

**KB 快照钉版本（REV-002）**：结果文件记录 data/structure_kb.json 的
git blob hash + 文件内容 sha256。存活率是「KB goal 校准 × PGN 矿脉丰富度」
的联合结果，不钉版本则后续无法区分「矿脉稀薄」与「goal 刚被收紧」。

用法：
    python -m tools.decision_probe.stage4_dualplan_probe \
        --pgn data/pgn/lichess_elite_2025-11.pgn --games 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import chess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..")))

from tools.decision_probe.pgn_m5_probe import (  # noqa: E402
    iter_midgame_positions,
)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KB_PATH = os.path.join(_ROOT, "data", "structure_kb.json")
OUT_DIR = os.path.join(_ROOT, "data", "quality_benchmark_decision")

# 阶段 3 P0-full 实测 A3 通过的原型（实现判据 2/5）——本冒烟据 peer_review
# 校准项处理：carlsbad/maroczy 已验证可分离，正常计入；hanging/majority
# A3 未过（且 estimator 缺陷待 planner 裁决），单列不与已验证原型同等计入
# 存活率分母；stonewall 已 in_production:false，被产品池闸拦下不计。
A3_VERIFIED = {"carlsbad", "maroczy"}


def _kb_snapshot() -> Dict[str, str]:
    """钉住 KB 版本：git blob hash（若已跟踪）+ 文件内容 sha256。"""
    snap: Dict[str, str] = {}
    try:
        h = subprocess.run(
            ["git", "hash-object", KB_PATH],
            cwd=_ROOT, capture_output=True, text=True, timeout=10)
        if h.returncode == 0:
            snap["git_blob_hash"] = h.stdout.strip()
    except Exception:
        pass
    try:
        with open(KB_PATH, "rb") as fh:
            snap["content_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        pass
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_ROOT, capture_output=True, text=True, timeout=10)
        if head.returncode == 0:
            snap["head_commit"] = head.stdout.strip()
    except Exception:
        pass
    return snap


def _plan_passes_gate(board: chess.Board, plan: dict, sf: str,
                      opens_top_cp: Optional[int]) -> Optional[dict]:
    """单条计划是否过机制闸——与 decision_pipeline 逐字同口径（约束 8）。

    feasible = assess_feasibility(cp差) AND goal_trajectory.goal_ok。
    返回 None 表示无约束线（explore_forward 失败）；否则返回诊断 dict。
    """
    from src.solver.branch_explorer import explore_forward, assess_feasibility
    from src.analysis.structure_features import goal_trajectory, structural_features

    line = explore_forward(board, plan, sf, depth=14)
    if line is None or not line.pv:
        return None
    feas, gap = assess_feasibility(line.cp, opens_top_cp)
    goal = plan.get("structural_goal") or {}
    mech_ok = True
    if goal:
        traj = goal_trajectory(board, line.pv, goal, board.turn)
        mech_ok = bool(traj["goal_ok"])
    return {
        "name": plan.get("name"),
        "feasible": bool(feas and mech_ok),
        "feas_cp": bool(feas),
        "mech_ok": mech_ok,
        "gap_cp": gap,
        "line_pv": line.pv,
    }


def _end_features(board: chess.Board, pv: List[chess.Move], n: int = 8):
    """约束线前 n 着的终点特征（锚定决策点走子方，与 A3 同口径）。"""
    from src.analysis.structure_features import structural_features
    mover = board.turn
    b = board.copy()
    fv = structural_features(b, mover)
    for mv in pv[:n]:
        try:
            b.push(mv)
        except Exception:
            break
        fv = structural_features(b, mover)
    return fv


def run(pgn_path: str, max_games: int, out_path: str,
        min_double: int = 30) -> Dict:
    from src.analysis.structure_id import (
        detect_pawn_structure, applicable_mover_side)
    from src.analysis.structure_features import feature_distance
    from src.solver.branch_explorer import explore_open

    sf = os.getenv("STOCKFISH_PATH", "")
    if not os.path.isabs(sf):
        sf = os.path.normpath(os.path.join(_ROOT, sf))
    if not os.path.isfile(sf):
        print(f"找不到 Stockfish: {sf}", file=sys.stderr)
        sys.exit(2)

    kb = json.load(open(KB_PATH, encoding="utf-8"))
    snapshot = _kb_snapshot()

    print(f"PGN: {pgn_path} | 读取上限 {max_games} 局")
    print(f"KB 快照: {snapshot.get('git_blob_hash', '?')[:12]} "
          f"(HEAD {snapshot.get('head_commit', '?')[:8]})")
    print("-" * 72)

    stats = Counter()
    per_arch = defaultdict(lambda: Counter())
    double_plan_positions = []
    seen_buckets: Counter = Counter()
    t0 = time.time()

    for pos in iter_midgame_positions(pgn_path, max_games, seen_buckets):
        stats["sampled"] += 1
        board = chess.Board(pos["fen"])
        arch, conf, _ = detect_pawn_structure(board)
        if arch is None:
            stats["no_archetype"] += 1
            continue
        stats["archetype_hit"] += 1
        per_arch[arch]["hit"] += 1

        # 产品池闸（与 decision_pipeline 同口径）
        if not kb[arch].get("in_production", True):
            stats["not_in_production"] += 1
            per_arch[arch]["not_in_production"] += 1
            continue

        # 角色筛（与 decision_pipeline 同口径）
        side = applicable_mover_side(board, arch)
        plans = kb[arch]["plans"]
        if side is not None:
            applicable = [p for p in plans if p.get("mover_side") == side]
            if not applicable:
                stats["no_applicable_plan"] += 1
                per_arch[arch]["no_applicable_plan"] += 1
                continue
            plans = applicable
        if len(plans) < 2:
            stats["fewer_than_2_plans"] += 1
            per_arch[arch]["fewer_than_2_plans"] += 1
            continue
        per_arch[arch]["reached_gate"] += 1

        # 机制闸：对每条计划跑 explore_forward + feasibility + goal_ok
        opens = explore_open(board, sf, k=4, depth=14)
        opens_top = opens[0].cp if opens else None
        gate_results = []
        for plan in plans:
            r = _plan_passes_gate(board, plan, sf, opens_top)
            if r is not None:
                gate_results.append(r)
        feasible_plans = [r for r in gate_results if r["feasible"]]

        if len(feasible_plans) >= 2:
            stats["double_plan_gate"] += 1
            per_arch[arch]["double_plan_gate"] += 1

            # A3 预筛（REV-002）：两计划终局特征距离 > 粗放下限
            fvs = [_end_features(board, r["line_pv"]) for r in feasible_plans]
            pair_dists = []
            for i in range(len(fvs)):
                for j in range(i + 1, len(fvs)):
                    pair_dists.append(feature_distance(fvs[i], fvs[j]))
            max_pair = max(pair_dists) if pair_dists else 0.0
            # 粗放下限：0.5（同原型内换根着组内距离量级的下沿，宁松不严）
            a3_ok = max_pair >= 0.5
            if a3_ok:
                stats["double_plan_a3ok"] += 1
                per_arch[arch]["double_plan_a3ok"] += 1
            double_plan_positions.append({
                "fen": pos["fen"], "archetype": arch,
                "feasible_plans": [r["name"] for r in feasible_plans],
                "max_pair_distance": round(max_pair, 3),
                "a3_prefilter_ok": a3_ok,
                "url": pos.get("url", ""),
            })

        if stats["sampled"] % 25 == 0:
            print(f"  已抽 {stats['sampled']} | 识别原型 "
                  f"{stats['archetype_hit']} | 双计划过闸 "
                  f"{stats['double_plan_gate']} | 过A3预筛 "
                  f"{stats['double_plan_a3ok']} | {time.time() - t0:.0f}s")

    elapsed = time.time() - t0
    sampled = stats["sampled"] or 1

    # verified-only 口径（peer_review #1 修复）：Go/No-Go 只看 A3 实测通过的
    # 原型（carlsbad/maroczy），不被 iqp（未测）/majority（阶段 3 已失败）
    # 主导虚高。全原型口径仅作对照。
    ver_gate = sum(c["double_plan_gate"] for a, c in per_arch.items()
                   if a in A3_VERIFIED)
    ver_a3ok = sum(c["double_plan_a3ok"] for a, c in per_arch.items()
                   if a in A3_VERIFIED)
    # 千局外推（peer_review 次要项：桶去重使产出次线性，此为**上界**）
    proj_all = round(1000.0 * stats["double_plan_gate"] / sampled, 1)
    proj_ver_gate = round(1000.0 * ver_gate / sampled, 1)
    proj_ver_a3ok = round(1000.0 * ver_a3ok / sampled, 1)
    # 裁决：verified 口径的千局外推双计划过闸数 ≥ min_double 才判成立
    # （对齐 PLAN 判据「千局量级 ≥30 双计划局面」，且用 verified 口径）
    verdict_supported = proj_ver_gate >= min_double

    result = {
        "kb_snapshot": snapshot,
        "pgn": os.path.basename(pgn_path),
        "max_games": max_games,
        "elapsed_s": round(elapsed, 1),
        "stats": dict(stats),
        "per_archetype": {a: dict(c) for a, c in per_arch.items()},
        "a3_verified_archetypes": sorted(A3_VERIFIED),
        "double_plan_positions": double_plan_positions,
        "min_double_threshold": min_double,
        "verdict": {
            "supported": verdict_supported,
            "basis": "verified-only 口径千局外推双计划过闸数 vs min_double",
            "note": "千局外推是上界（桶去重使产出次线性）；A3 预筛=max≥0.5 粗筛"
                    "非阶段 3 A3 判定；全原型口径含 iqp(未测)/majority(阶段3已失败)"
                    "仅作对照不作裁决依据",
        },
        "rates": {
            "archetype_hit_pct": round(100.0 * stats["archetype_hit"] / sampled, 1),
            "double_plan_gate_pct_all": round(100.0 * stats["double_plan_gate"] / sampled, 2),
            "double_plan_a3ok_pct_all": round(100.0 * stats["double_plan_a3ok"] / sampled, 2),
            "verified_gate": ver_gate,
            "verified_a3ok": ver_a3ok,
            "proj_per_1000_all_gate": proj_all,
            "proj_per_1000_verified_gate": proj_ver_gate,
            "proj_per_1000_verified_a3ok": proj_ver_a3ok,
        },
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)

    print("-" * 72)
    print(f"抽样 {stats['sampled']} | 识别原型 {stats['archetype_hit']} "
          f"({result['rates']['archetype_hit_pct']}%)")
    print(f"[对照·全原型] 双计划过闸 {stats['double_plan_gate']} | "
          f"过A3预筛 {stats['double_plan_a3ok']} | 千局外推(上界) ≈{proj_all}")
    print(f"[裁决·verified] carlsbad+maroczy 双计划过闸 {ver_gate} | "
          f"过A3预筛 {ver_a3ok} | 千局外推(上界) ≈{proj_ver_gate}")
    print("逐原型：")
    for a, c in sorted(per_arch.items()):
        verified = "✓验证" if a in A3_VERIFIED else "⚠未验证/单列(不计入裁决)"
        print(f"  {a} [{verified}]: 识别 {c['hit']} | 双计划过闸 "
              f"{c['double_plan_gate']} | 过A3粗筛 {c['double_plan_a3ok']}")
    print(f"耗时 {elapsed:.0f}s，结果写入 {out_path}")
    print("-" * 72)
    print(f"假设裁决（阈值 verified 千局 ≥{min_double}）："
          f"{'成立 ✓' if verdict_supported else '不成立 ✗——按 PLAN 回退分支处理'}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default=os.path.join(
        _ROOT, "data", "pgn", "lichess_elite_2025-11.pgn"))
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(
        OUT_DIR, "stage4_dualplan_result.json"))
    ap.add_argument("--min-double", type=int, default=30)
    args = ap.parse_args()
    run(args.pgn, args.games, args.out, args.min_double)


if __name__ == "__main__":
    main()
