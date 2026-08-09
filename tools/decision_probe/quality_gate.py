"""决策管线质量门槛（PLAN-011 阶段 0）。

把 PLAN-010 交付的探针整合为可重复的质量门槛，记录基线，作为后续每个
KB 改动的验收尺。

两层门（REV-001 分层 + peer_review A1 修正——确定性指标在 storyboard 层量）：
  **快速门**（确定性，每次 KB 改动必跑）：
    - A2/A3 复用 `p0_full_probe`（引擎 Threads=1，跑两遍一致）
    - **storyboard 层真比较式率**：直接调产品同一组函数
      （detect_pawn_structure → explore_open/forward → assess_feasibility →
      goal_trajectory → build_decision_storyboard，全是引擎+python、秒级、
      无 LLM/TTS/渲染），在 storyboard 层量 routes/axis_type/divergences。
      这不是 HANDOFF #8 的「门内重编排」——用的是产品自己的同一组单一
      事实来源函数，判据零分叉（peer_review A1：只读 sidecar 的过度字面
      解读会把确定性指标拖到全链路末尾，被 LLM/TTS 非确定失败挟持）。
  **慢速门**（非确定，每里程碑抽样）：
    - 跑完整 `_run_decision_pipeline` → 读产品 sidecar 统计段级缺失
    - 崩溃（异常）与 SPEC §8 优雅放弃（return ""）**分桶**（peer_review A2）：
      崩溃是回归信号、放弃是预期行为，混桶会失去检测崩溃回归的能力。

样本集：复用 `stage4_dualplan_result.json` 的 40 个 double_plan_positions，
按原型分层抽样（按 fen 排序稳抽样，防 stage4 重排漂移——peer_review B5）。

用法：
    python -m tools.decision_probe.quality_gate           # 两层门全跑
    python -m tools.decision_probe.quality_gate --fast     # 只跑快速门
    python -m tools.decision_probe.quality_gate --baseline # 落基线快照
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from typing import Dict, List, Optional

import chess

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

KB_PATH = os.path.join(_ROOT, "data", "structure_kb.json")
STAGE4_PATH = os.path.join(_ROOT, "data", "quality_benchmark_decision",
                           "stage4_dualplan_result.json")
OUT_DIR = os.path.join(_ROOT, "data", "quality_benchmark_decision")


def _kb_snapshot() -> Dict[str, str]:
    """KB 版本快照（git blob hash + content sha256 + HEAD）。

    content_sha256 与 git_blob_hash 任一取不到 → fail-loud（peer_review B8）：
    阶段0 的全部意义是「可追溯基线」，无版本锚的基线落盘等于自欺。
    """
    snap: Dict[str, str] = {}
    try:
        with open(KB_PATH, "rb") as fh:
            snap["content_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    except Exception as e:
        print(f"!! KB 文件读取失败，无法生成版本锚: {type(e).__name__}: {e}",
              file=sys.stderr)
        sys.exit(1)
    try:
        h = subprocess.run(["git", "hash-object", KB_PATH],
                           cwd=_ROOT, capture_output=True, text=True, timeout=10)
        if h.returncode == 0:
            snap["git_blob_hash"] = h.stdout.strip()
    except Exception:
        pass  # git 可能在非 git 环境，sha256 已够唯一锚定
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=_ROOT, capture_output=True, text=True, timeout=10)
        if head.returncode == 0:
            snap["head_commit"] = head.stdout.strip()
    except Exception:
        pass
    return snap


def _load_samples(per_arch: int = 3) -> List[dict]:
    """从 stage4 双计划局面分层抽样（按 fen 排序稳抽样——peer_review B5）。"""
    r = json.load(open(STAGE4_PATH, encoding="utf-8"))
    pos = r.get("double_plan_positions", [])
    by_arch = defaultdict(list)
    for p in pos:
        by_arch[p["archetype"]].append(p)
    samples = []
    for arch, plist in sorted(by_arch.items()):
        # 按 fen 排序再取前 N，防 stage4 重排导致基线样本漂移
        for p in sorted(plist, key=lambda x: x["fen"])[:per_arch]:
            samples.append({"fen": p["fen"], "archetype": arch,
                            "url": p.get("url", "")})
    return samples


def _storyboard_compare(board: chess.Board, sf: str) -> dict:
    """在 storyboard 层量真比较式（peer_review A1：确定性指标回引擎层）。

    直接调产品同一组单一事实来源函数，不另写判据：
    detect_pawn_structure → applicable_mover_side → explore_open/forward →
    assess_feasibility → goal_trajectory → build_decision_storyboard。
    全是引擎+python、Threads=1 确定、秒级、无 LLM/TTS/渲染。
    """
    from src.analysis.structure_id import detect_pawn_structure, applicable_mover_side
    from src.solver.branch_explorer import explore_open, explore_forward, assess_feasibility
    from src.analysis.structure_features import goal_trajectory
    from src.storyboard.decision_builder import build_decision_storyboard, PlanOutcome, axis_type_for

    kb = json.load(open(KB_PATH, encoding="utf-8"))
    arch, _, _ = detect_pawn_structure(board)
    if arch is None or not kb.get(arch, {}).get("in_production", True):
        return {"archetype": arch, "n_routes": 0, "paired": False,
                "real_compare": False, "axis_type": None,
                "reason": "no_archetype_or_not_in_production"}
    plans = kb[arch]["plans"]
    side = applicable_mover_side(board, arch)
    if side is not None:
        plans = [p for p in plans if p.get("mover_side") == side]

    opens = explore_open(board, sf, k=4, depth=14)
    opens_top = opens[0].cp if opens else None
    outcomes = []
    for plan in plans:
        line = explore_forward(board, plan, sf, depth=14)
        if line is None or not line.pv:
            continue
        feas, gap = assess_feasibility(line.cp, opens_top)
        goal = plan.get("structural_goal") or {}
        mech_ok = True
        if goal:
            tr = goal_trajectory(board, line.pv, goal, board.turn)
            mech_ok = bool(tr["goal_ok"])
        if feas and mech_ok:
            outcomes.append(PlanOutcome(
                plan=plan, line_cp=line.cp, line_pv=line.pv,
                feasible=True, gap_cp=gap, trend={}, tradeoffs={},
                start_features=[], end_features=[]))

    from src.storyboard.decision_builder import DecisionInput
    sb = build_decision_storyboard(
        DecisionInput(fen=board.fen()), outcomes,
        archetype=arch, strategic_premise=kb[arch].get("theory", ""))
    routes = sb.get("routes", [])
    divergences = sb.get("divergences", [])
    n_routes = len(routes)
    paired = any(d.get("paired") for d in divergences) if divergences else False
    # real = routes≥2 & 任一 pair paired（peer_review B1: axis_type==1 与
    # routes≥2 等价，删冗余；peer_review B2: goal_ok 已由选线闸保证，不重复判）
    real = n_routes >= 2 and paired
    return {"archetype": arch, "n_routes": n_routes, "paired": paired,
            "real_compare": real, "axis_type": sb.get("comparison_axes", {}).get("axis_type")}


def run_fast_gate(samples: Optional[List[dict]] = None) -> dict:
    """快速门（确定性）：A2/A3 + storyboard 层真比较式率。"""
    from tools.decision_probe.p0_full_probe import run_a2_goals, run_a3_separability
    sf = os.getenv("STOCKFISH_PATH", "")
    if not sf:
        print("!! STOCKFISH_PATH 未设置", file=sys.stderr)
        sys.exit(2)
    if not os.path.isabs(sf):
        sf = os.path.normpath(os.path.join(_ROOT, sf))

    # KB 原型覆盖核对（peer_review C7）
    kb = json.load(open(KB_PATH, encoding="utf-8"))
    kb_archetypes = set(kb.keys())
    a2_a3_archetypes = {"carlsbad", "hanging", "stonewall", "maroczy",
                        "majority", "iqp"}  # p0_full_probe 硬编码样本表
    uncovered = kb_archetypes - a2_a3_archetypes
    if uncovered:
        print(f"⚠ KB 有原型但 p0_full_probe 样本表未覆盖: {uncovered} "
              f"（A2/A3 会静默跳过，阶段2 须先注入样本表）")

    print("=" * 60)
    print("快速门（A2/A3 + storyboard 真比较式，引擎 Threads=1，确定性）")
    print("=" * 60)
    try:
        a2 = run_a2_goals(sf)
        a3 = run_a3_separability(sf)
    except Exception as e:
        print(f"!! 快速门 A2/A3 崩溃: {type(e).__name__}: {e}", file=sys.stderr)
        raise

    a3_pass = sum(1 for v in a3["per_situation"].values() if v["passed"])
    a3_total = len(a3["per_situation"])

    # storyboard 层真比较式率（确定性，秒级）
    sb_results = []
    if samples:
        print(f"\nstoryboard 层真比较式（{len(samples)} 局面）：")
        for i, s in enumerate(samples):
            board = chess.Board(s["fen"])
            r = _storyboard_compare(board, sf)
            r["fen"] = s["fen"]
            sb_results.append(r)
            print(f"  [{i+1}/{len(samples)}] {r.get('archetype','?'):10s} "
                  f"routes={r['n_routes']} paired={r['paired']} "
                  f"{'✓真比较' if r['real_compare'] else '✗'}")

    # 比率计算（peer_review B3）
    sb_per_arch = defaultdict(lambda: {"sampled": 0, "real_compare": 0})
    for r in sb_results:
        a = r.get("archetype", "?")
        sb_per_arch[a]["sampled"] += 1
        if r["real_compare"]:
            sb_per_arch[a]["real_compare"] += 1
    sb_rates = {}
    for a, c in sb_per_arch.items():
        n = c["sampled"] or 1
        sb_rates[a] = {"real_compare": c["real_compare"], "sampled": c["sampled"],
                        "rate_pct": round(100.0 * c["real_compare"] / n, 1)}

    print(f"\n快速门汇总：A2 {a2['achieved']}/{a2['total']}={a2['rate_pct']}% | "
          f"A3 {a3_pass}/{a3_total} | storyboard 真比较式率 "
          f"{ {a: v['rate_pct'] for a, v in sb_rates.items()} }")
    return {"a2": a2, "a3": a3, "storyboard": sb_results,
            "storyboard_rates": sb_rates}


def _read_sidecar(video_path: str) -> Optional[dict]:
    """从视频路径推断 sidecar 路径并读取。"""
    sidecar_path = os.path.splitext(video_path)[0] + "_review.json"
    if not os.path.isfile(sidecar_path):
        return None
    try:
        return json.load(open(sidecar_path, encoding="utf-8"))
    except Exception:
        return None


def run_slow_gate(samples: List[dict]) -> dict:
    """慢速门：跑完整管线 → 读 sidecar 统计段级缺失（含 LLM/TTS，非确定）。

    崩溃（异常）与 SPEC §8 优雅放弃（return ""）**分桶**（peer_review A2）。
    """
    from src.pipeline.decision_pipeline import _run_decision_pipeline

    print("\n" + "=" * 60)
    print(f"慢速门（完整管线，{len(samples)} 局面，含 LLM/TTS，非确定）")
    print("=" * 60)

    per_arch = defaultdict(lambda: {"sampled": 0, "pipeline_fail": 0,
                                     "crash": 0, "position_with_missing": 0})
    positions = []
    t0 = time.time()

    for i, s in enumerate(samples):
        arch = s["archetype"]
        per_arch[arch]["sampled"] += 1
        out_dir = tempfile.mkdtemp(prefix=f"qgate_{arch}_")
        try:
            video_path = _run_decision_pipeline(s["fen"], output_dir=out_dir)
        except Exception as e:
            # peer_review A2：崩溃单独记档，不进 pipeline_fail
            per_arch[arch]["crash"] += 1
            positions.append({"fen": s["fen"], "archetype": arch,
                              "url": s["url"], "status": "crash",
                              "error": f"{type(e).__name__}: {e}"[:200]})
            print(f"  [{i+1}/{len(samples)}] {arch} 崩溃: {type(e).__name__}: {e}")
            continue

        if not video_path:
            per_arch[arch]["pipeline_fail"] += 1
            positions.append({"fen": s["fen"], "archetype": arch,
                              "url": s["url"], "status": "pipeline_fail"})
            print(f"  [{i+1}/{len(samples)}] {arch} 管线级放弃（SPEC §8）")
            continue

        side = _read_sidecar(video_path)
        if side is None:
            per_arch[arch]["pipeline_fail"] += 1
            positions.append({"fen": s["fen"], "archetype": arch,
                              "url": s["url"], "status": "no_sidecar"})
            print(f"  [{i+1}/{len(samples)}] {arch} 无 sidecar")
            continue

        routes = side.get("routes", [])
        n_routes = len(routes)
        # 段级缺失：计划段 id 是否都在 segments 里（peer_review B4 改名）
        seg_ids = {seg.get("id") for seg in side.get("segments", [])}
        plan_ids = set(range(1, n_routes + 1)) if n_routes >= 2 else set()
        missing_plan_segs = sorted(plan_ids - seg_ids) if plan_ids else []
        has_missing = bool(missing_plan_segs)
        if has_missing:
            per_arch[arch]["position_with_missing"] += 1

        positions.append({
            "fen": s["fen"], "archetype": arch, "url": s["url"],
            "status": "ok", "n_routes": n_routes,
            "goal_ok": [r.get("goal_ok") for r in routes],
            "missing_plan_segs": missing_plan_segs,
        })
        print(f"  [{i+1}/{len(samples)}] {arch} routes={n_routes} "
              f"{'△段缺' if has_missing else '✓'} {time.time()-t0:.0f}s")

    # 比率计算（peer_review B3）
    rates = {}
    for a, c in per_arch.items():
        n = c["sampled"] or 1
        rates[a] = dict(c)
        rates[a]["pipeline_fail_rate"] = round(100.0 * c["pipeline_fail"] / n, 1)
        rates[a]["crash_rate"] = round(100.0 * c["crash"] / n, 1)
        rates[a]["missing_rate"] = round(100.0 * c["position_with_missing"] / n, 1)

    print(f"\n慢速门汇总（{time.time()-t0:.0f}s）：")
    for arch, c in sorted(per_arch.items()):
        print(f"  {arch:10s} 抽样{c['sampled']} 放弃{c['pipeline_fail']} "
              f"崩溃{c['crash']} 段缺{c['position_with_missing']}")
    return {"per_archetype_rates": rates, "positions": positions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="只跑快速门")
    ap.add_argument("--baseline", action="store_true",
                    help="落基线快照（写入 quality_baseline.json）")
    ap.add_argument("--per-arch", type=int, default=3,
                    help="每原型抽样数（默认 3）")
    args = ap.parse_args()

    snapshot = _kb_snapshot()
    print(f"KB 快照: {snapshot.get('git_blob_hash', '?')[:12]} "
          f"(HEAD {snapshot.get('head_commit', '?')[:8]})")

    samples = _load_samples(per_arch=args.per_arch)
    result = {"kb_snapshot": snapshot, "timestamp": time.strftime("%Y-%m-%d %H:%M"),
              "per_arch_sampled": args.per_arch}

    fast = run_fast_gate(samples=samples if not args.fast else samples[:6])
    result["fast_gate"] = fast

    if not args.fast:
        slow = run_slow_gate(samples)
        result["slow_gate"] = slow

    out_name = ("quality_baseline.json" if args.baseline
                else "quality_gate_result.json")
    out_path = os.path.join(OUT_DIR, out_name)
    os.makedirs(OUT_DIR, exist_ok=True)
    # peer_review C6：落盘前自检 gitignore
    ci = subprocess.run(["git", "check-ignore", out_path],
                        cwd=_ROOT, capture_output=True)
    if ci.returncode == 0:
        print(f"!! 输出路径被 .gitignore 排除，基线无法入库: {out_path}",
              file=sys.stderr)
        sys.exit(1)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(f"\n结果写入 {out_path}")
    return result


if __name__ == "__main__":
    main()
