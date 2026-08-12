"""KB-free 等强异 zone 占比验证探针 + 轴 4 素材下界（PLAN-012 阶段 1）。

**阶段 0 裁决优化**：fork stage4 探针加日志，不另起独立采样器。复用
`iter_midgame_positions`（同一采样器 → 同一局面宇宙 → 直接对比 stage4
522 基线），在现有 explore_open 调用后追加完整候选集落盘 + per-plan
机制闸结果落盘。

**一次引擎运行同时产出**：
  (a) KB-free fork-stats：等强异 zone 首着对占比（回答 H1 本体论稀缺）
  (b) 轴 4 池内层可得率：feasible=False 且 mech_ok=True 且 gap∈(80,150]
      的计划占比（阶段 0 确认 stage4 丢了此数据，须带日志重跑）
  (c) 轴 4 K2 下界：opens 中 ∃ gap∈(0,150] 异 zone 次优首着
  (d) stage4 回归校验：同口径双计划过闸率

**指标正名（对抗审查重大 1）**：「等强异 zone 首着对占比」是真战略分岔
的**代理下界**——zone 三桶（落点 file a-c/d-e/f-h）粒度粗，出子次序
（Nc3 vs Nf3 落点不同 zone）会假阳性、同 zone 真分歧（保持 vs 推进
都在 center）会假阴性。报告同时给出 M5-forcing 排除前后的占比，且
**不外推为「战略分岔占比」**。阶段 1.6 的人工分层抽检（30 局面）校准
假阳性率。

**checkpoint**：逐局面 JSONL append（含 bucket 签名），中断重跑自动
跳过已处理 bucket。PGN 从头重读（解析快，引擎是瓶颈）。

**两档**：
  - 代理档 ``--no-kb-analysis``：只跑 explore_open（KB-free），跳过
    KB 约束路径（explore_forward/feasibility/goal_ok），约快 3~5 倍。
    用于定 gap/zone 分布、定正式档 k 与 depth。
  - 正式档（默认）：全量 KB 约束分析 + KB-free，产出 (a)~(d) 全套。

用法::

    # 代理档（定分布，~15 分钟）
    python -m tools.decision_probe.fork_stats_probe --games 250 --k 3 --depth 10 --no-kb-analysis
    # 正式档（定标后）
    python -m tools.decision_probe.fork_stats_probe --games 1000 --k 5 --depth 14
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

import chess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..")))

from tools.decision_probe.pgn_m5_probe import iter_midgame_positions  # noqa: E402
from tools.decision_probe.stage4_dualplan_probe import (  # noqa: E402
    A3_VERIFIED,
    _kb_snapshot,
    _plan_passes_gate,
)

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KB_PATH = os.path.join(_ROOT, "data", "structure_kb.json")
OUT_DIR = os.path.join(_ROOT, "data", "quality_benchmark_decision")

# ---- 阈值（全部引用既有定义，不发明新口径）---------------------------

# M5 standout：首选-次选 ≥ 此值判 forcing（唯一好着，非战略局面）。
# 出处：engine_probe.DEFAULT_STANDOUT_CP = 150，与 M5 冒烟同口径。
STANDOUT_CP = 150

# 等强阈值：gap ≤ 此值判「等强」（战略分岔的两个首着评估接近）。
# 单一事实来源：direction.DEFAULT_EQUIV_CP = 60。
from src.analysis.direction import (  # noqa: E402
    DEFAULT_EQUIV_CP,
    direction_zone,
    equivalence_gap,
)

# 可行性闸阈值：gap > 此值判「不可行」（轴 4 池内层来源的下界）。
# 出处：branch_explorer.DEFAULT_FEASIBLE_CP = 80。
FEASIBLE_CP = 80

# 轴 4 gap 窗口上界（对抗审查重大 2）：超过此值是「犯错」不是「更差但合理」。
# 对齐 STANDOUT_CP——150cp 在引擎语义上已是「唯一好着」级别。
AXIS4_GAP_HI = 150

# M6 评估窗口：|opens[0].cp| ≤ 此值的局面才谈战略（一边倒的局面所有着都行）。
# 出处：pgn_m5_probe.M6_ABS_CP_MAX = 300。
M6_ABS_CP_MAX = 300

# gap 分布直方图分桶（对齐项目既有阈值：30/60/80/150）。
GAP_BINS = [30, 60, 80, 150, 300]


# ---- 逐局面计算（纯函数）---------------------------------------------


def _opens_records(board: chess.Board, opens: list) -> List[dict]:
    """MultiPV 候选集 → 可序列化记录（SAN/UCI/cp/zone）。

    board 不被修改（所有候选都是同一局面的合法着，board.san 只读）。
    """
    out = []
    for ln in opens:
        try:
            san = board.san(ln.move)
        except Exception:
            san = "?"
        out.append({
            "san": san,
            "uci": ln.move.uci(),
            "cp": ln.cp,
            "is_mate": ln.is_mate,
            "zone": direction_zone(ln.move),
        })
    return out


def _m5_forcing(opens_recs: List[dict]) -> bool:
    """M5 standout 判定：首选-次选 ≥ STANDOUT_CP → forcing（非战略局面）。

    候选不足 2 个时，只有 1 个合法着 → trivially forcing。
    用 equivalence_gap（单一事实来源）而非单向差，口径与 _fork_stats 一致。
    """
    if len(opens_recs) < 2:
        return True
    return equivalence_gap(
        opens_recs[0]["cp"], opens_recs[1]["cp"]) >= STANDOUT_CP


def _fork_stats(opens_recs: List[dict]) -> dict:
    """KB-free 等强异 zone 统计（代理下界）。

    遍历 opens 所有两两组合，统计：
    - equiv_pairs：gap ≤ DEFAULT_EQUIV_CP(60) 的对数
    - diff_zone_pairs：zone 不同的对数
    - equiv_diff_zone_pairs：两者皆满足（核心指标——战略分岔代理）
    - min_gap：最小 gap（首选 vs 次选）
    """
    equiv_pairs = 0
    diff_zone_pairs = 0
    equiv_diff_zone = 0
    min_gap = None
    n = len(opens_recs)
    for i in range(n):
        for j in range(i + 1, n):
            g = equivalence_gap(opens_recs[i]["cp"], opens_recs[j]["cp"])
            if min_gap is None or g < min_gap:
                min_gap = g
            equiv = g <= DEFAULT_EQUIV_CP
            diff_z = opens_recs[i]["zone"] != opens_recs[j]["zone"]
            if equiv:
                equiv_pairs += 1
            if diff_z:
                diff_zone_pairs += 1
            if equiv and diff_z:
                equiv_diff_zone += 1
    return {
        "equiv_diff_zone_pairs": equiv_diff_zone,
        "equiv_pairs": equiv_pairs,
        "diff_zone_pairs": diff_zone_pairs,
        "min_gap": min_gap,
        "n_candidates": n,
    }


def _axis4_k2_material(opens_recs: List[dict]) -> dict:
    """轴 4 K2 素材下界：∃ opens[j](j≥1) 使 gap∈(0, AXIS4_GAP_HI] 且异 zone。

    正选参照 = opens[0]（KB-free 口径；产品代码阶段 2 改用 KB target_zone，
    见对抗审查次要 1）。返回首个命中的 j 的诊断；无命中返回 has_k2=False。
    """
    if len(opens_recs) < 2:
        return {"has_k2": False}
    p0 = opens_recs[0]
    for j in range(1, len(opens_recs)):
        g = equivalence_gap(p0["cp"], opens_recs[j]["cp"])
        if 0 < g <= AXIS4_GAP_HI and opens_recs[j]["zone"] != p0["zone"]:
            return {
                "has_k2": True,
                "k2_san": opens_recs[j]["san"],
                "k2_gap": g,
                "k2_zone": opens_recs[j]["zone"],
                "primary_zone": p0["zone"],
            }
    return {"has_k2": False}


def _gap_bin(gap: Optional[int]) -> str:
    """gap → 分布桶标签。"""
    if gap is None:
        return "n/a"
    for hi in GAP_BINS:
        if gap <= hi:
            return f"≤{hi}"
    return f">{GAP_BINS[-1]}"


# ---- checkpoint --------------------------------------------------------


def _load_seen_fens(jsonl_path: str) -> set:
    """读已有 JSONL，重建 seen_fens 以跳过已处理局面。

    **不重建 seen_buckets**（peer_review M1 修复）：iter_midgame_positions
    在 yield 前就给 bucket +1，若主循环用 seen_fens 跳过该 yield，bucket
    计数已涨但局面未处理——同桶后续**不同 FEN** 的新局面会被误判为
    「桶满」而永久跳过。改用空 Counter + seen_fens 精确去重：PGN 从头
    重读（解析快，引擎是瓶颈），已处理 FEN 被精确跳过，不丢新局面。

    代价：resume 需增大 --games 才能产出新局面（同 max_games 内的局面
    已全在 seen_fens 中，全部跳过）。这对代理档（一次性跑完）无影响；
    对正式档（可能中断重跑），增大 --games 即可续跑。
    """
    seen_fens: set = set()
    if not os.path.isfile(jsonl_path):
        return seen_fens
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                f = rec.get("fen")
                if f:
                    seen_fens.add(f)
            except (json.JSONDecodeError, AttributeError):
                continue
    return seen_fens


def _append_record(record: dict, jsonl_path: str) -> None:
    """逐局面 append（checkpoint 粒度）。"""
    with open(jsonl_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---- 主循环 -----------------------------------------------------------


def run(
    pgn_path: str,
    max_games: int,
    out_path: str,
    jsonl_path: str,
    k: int = 5,
    depth: int = 14,
    kb_analysis: bool = True,
) -> dict:
    """主入口：扫描 PGN，逐局面跑 MultiPV + 可选 KB 约束分析，落盘 JSONL。

    返回汇总 dict 并写入 summary JSON。
    """
    from src.solver.branch_explorer import explore_open

    sf = os.getenv("STOCKFISH_PATH", "")
    if not os.path.isabs(sf):
        sf = os.path.normpath(os.path.join(_ROOT, sf))
    if not os.path.isfile(sf):
        print(f"找不到 Stockfish: {sf}", file=sys.stderr)
        sys.exit(2)

    kb = json.load(open(KB_PATH, encoding="utf-8")) if kb_analysis else None
    snapshot = _kb_snapshot()

    # checkpoint resume（peer_review M1 修复：只用 seen_fens，不重建
    # seen_buckets——避免 generator 增量后 position 被 skip 导致同桶新局面丢失）
    seen_fens = _load_seen_fens(jsonl_path)
    resumed = len(seen_fens)

    mode_label = "正式档(全量)" if kb_analysis else "代理档(KB-free only)"
    print(f"PGN: {pgn_path} | 上限 {max_games} 局 | k={k} depth={depth} | {mode_label}")
    if kb_analysis and depth != 14:
        print(f"注：KB 约束路径（_plan_passes_gate）内部 explore_forward 固定 depth=14，"
              f"与 explore_open depth={depth} 不一致——正式档应用 depth=14")
    print(f"KB 快照: {snapshot.get('git_blob_hash', '?')[:12]}")
    if resumed:
        print(f"checkpoint resume：已有 {resumed} 条记录，跳过已处理 FEN")
    print("-" * 72)

    t0 = time.time()
    n_sampled = 0
    n_logged = 0

    # 空 Counter：不预填 seen_buckets（见 _load_seen_fens docstring）。
    # iter_midgame_positions 从头读 PGN，已处理 FEN 被 seen_fens 精确跳过。
    for pos in iter_midgame_positions(pgn_path, max_games, Counter()):
        if pos["fen"] in seen_fens:
            continue
        n_sampled += 1
        board = chess.Board(pos["fen"])

        # === KB-free：每个局面都跑 MultiPV（fork-stats 分母 = 全部采样）===
        opens = explore_open(board, sf, k=k, depth=depth)
        if not opens:
            continue
        opens_recs = _opens_records(board, opens)
        m5_force = _m5_forcing(opens_recs)
        fstats = _fork_stats(opens_recs)
        a4_k2 = _axis4_k2_material(opens_recs)
        top_cp = opens_recs[0]["cp"]

        # === KB 约束路径（可选，仅 archetype hit 才有意义）===
        arch = None
        plans_gate: List[dict] = []
        n_feasible = None
        a4_pool_has = False
        if kb_analysis:
            from src.analysis.structure_id import (
                detect_pawn_structure, applicable_mover_side)
            arch, _, _ = detect_pawn_structure(board)
            if arch is not None and kb[arch].get("in_production", True):
                side = applicable_mover_side(board, arch)
                plans = kb[arch]["plans"]
                if side is not None:
                    applicable = [p for p in plans
                                  if p.get("mover_side") == side]
                    if applicable:
                        plans = applicable
                opens_top = opens[0].cp if opens else None
                n_feasible = 0
                for plan in plans:
                    r = _plan_passes_gate(board, plan, sf, opens_top)
                    if r is None:
                        continue
                    plans_gate.append({
                        "name": r["name"],
                        "feasible": r["feasible"],
                        "feas_cp": r["feas_cp"],
                        "mech_ok": r["mech_ok"],
                        "gap_cp": r["gap_cp"],
                        "line_cp": r["line_cp"],
                    })
                    if r["feasible"]:
                        n_feasible += 1
                    # 轴 4 池内层来源判定（阶段 0 确认 stage4 丢了此数据）：
                    # feasible=False 且 mech_ok=True（仅 gap 超 80 被拒）
                    # 且 gap∈(80,150] → 可作轴 4 对照
                    if (not r["feasible"] and r["mech_ok"]
                            and r["gap_cp"] is not None
                            and FEASIBLE_CP < r["gap_cp"] <= AXIS4_GAP_HI):
                        a4_pool_has = True

        rec = {
            "fen": pos["fen"],
            "bucket": pos["bucket"],
            "url": pos.get("url", ""),
            "ply": pos.get("ply"),
            "top_cp": top_cp,
            "m5_forcing": m5_force,
            "opens": opens_recs,
            "fork_stats": fstats,
            "axis4_k2": a4_k2,
            "archetype": arch,
            "plans_gate": plans_gate,
            "n_feasible": n_feasible,
            "axis4_pool_has": a4_pool_has,
        }
        _append_record(rec, jsonl_path)
        n_logged += 1

        if n_sampled % 25 == 0:
            print(f"  已抽 {n_sampled} | 已记 {n_logged} | "
                  f"{time.time() - t0:.0f}s")

    elapsed = time.time() - t0
    print("-" * 72)
    print(f"采样 {n_sampled} | 记录 {n_logged} | 耗时 {elapsed:.0f}s")

    summary = summarize(jsonl_path)
    summary["run_config"] = {
        "pgn": os.path.basename(pgn_path),
        "max_games": max_games,
        "k": k,
        "depth": depth,
        "kb_analysis": kb_analysis,
        "resumed_records": resumed,
    }
    summary["kb_snapshot"] = snapshot
    summary["elapsed_s"] = round(elapsed, 1)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(f"汇总写入 {out_path}")
    return summary


def summarize(jsonl_path: str) -> dict:
    """从 JSONL 聚合统计（可独立调用，不重跑引擎）。"""
    recs: List[dict] = []
    if os.path.isfile(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    total = len(recs)
    if total == 0:
        return {"total": 0, "note": "无记录"}

    n_forcing = sum(1 for r in recs if r.get("m5_forcing"))
    n_quiet = total - n_forcing
    # M6 过滤：|top_cp| ≤ 300（一边倒的局面不谈战略）
    n_m6 = sum(1 for r in recs
               if abs(r.get("top_cp", 999)) <= M6_ABS_CP_MAX)
    n_quiet_m6 = sum(1 for r in recs
                     if not r.get("m5_forcing")
                     and abs(r.get("top_cp", 999)) <= M6_ABS_CP_MAX)

    # (a) fork-stats：等强异 zone 首着对
    n_fork = sum(1 for r in recs
                 if r.get("fork_stats", {}).get("equiv_diff_zone_pairs", 0) > 0)
    n_fork_quiet = sum(1 for r in recs
                       if not r.get("m5_forcing")
                       and r.get("fork_stats", {}).get(
                           "equiv_diff_zone_pairs", 0) > 0)
    n_fork_quiet_m6 = sum(1 for r in recs
                          if not r.get("m5_forcing")
                          and abs(r.get("top_cp", 999)) <= M6_ABS_CP_MAX
                          and r.get("fork_stats", {}).get(
                              "equiv_diff_zone_pairs", 0) > 0)

    # (c) 轴 4 K2 下界
    n_a4_k2 = sum(1 for r in recs
                  if r.get("axis4_k2", {}).get("has_k2"))
    n_a4_k2_quiet = sum(1 for r in recs
                        if not r.get("m5_forcing")
                        and r.get("axis4_k2", {}).get("has_k2"))

    # gap 分布（首选 vs 次选）
    gap_bins = Counter()
    top1_top2_gaps = []
    for r in recs:
        opens = r.get("opens", [])
        if len(opens) >= 2:
            g = equivalence_gap(opens[0]["cp"], opens[1]["cp"])
            top1_top2_gaps.append(g)
            gap_bins[_gap_bin(g)] += 1

    # zone 组合分布（首选 vs 次选）
    zone_pairs = Counter()
    for r in recs:
        opens = r.get("opens", [])
        if len(opens) >= 2:
            pair = tuple(sorted([opens[0]["zone"], opens[1]["zone"]]))
            zone_pairs[f"{pair[0]}-{pair[1]}"] += 1

    # KB 约束口径（若有 plans_gate 数据）
    kb_stats = {}
    recs_with_gate = [r for r in recs if r.get("plans_gate")]
    if recs_with_gate:
        n_arch_hit = sum(1 for r in recs if r.get("archetype"))
        n_with_gate = len(recs_with_gate)
        n_feasible_dist = Counter(
            r.get("n_feasible") for r in recs_with_gate)
        n_axis4_pool = sum(1 for r in recs_with_gate
                           if r.get("axis4_pool_has"))
        # 轴 4 触发域：n_feasible==1
        n_trigger = n_feasible_dist.get(1, 0)
        # 轴 1 材料：n_feasible>=2
        n_axis1 = sum(v for k, v in n_feasible_dist.items()
                      if isinstance(k, int) and k >= 2)
        kb_stats = {
            "archetype_hit": n_arch_hit,
            "with_gate_data": n_with_gate,
            "n_feasible_dist": dict(n_feasible_dist),
            "axis4_trigger_domain(n==1)": n_trigger,
            "axis1_material(n>=2)": n_axis1,
            "axis4_pool_source_available": n_axis4_pool,
        }

    def pct(num, denom):
        return round(100.0 * num / denom, 1) if denom else 0.0

    return {
        "total": total,
        "m5_forcing": n_forcing,
        "m5_quiet": n_quiet,
        "m6_balanced": n_m6,
        "m5_quiet_and_m6": n_quiet_m6,
        "fork_stats": {
            "equiv_diff_zone_total": n_fork,
            "pct_of_all": pct(n_fork, total),
            "pct_of_quiet": pct(n_fork_quiet, n_quiet),
            "pct_of_quiet_m6": pct(n_fork_quiet_m6, n_quiet_m6),
            "note": ("指标正名：等强异 zone 首着对占比——真战略分岔的代理下界。"
                     "zone 三桶粒度粗，出子次序假阳性 + 同zone真分歧假阴性。"
                     "阶段1.6 人工抽检校准。"),
        },
        "axis4_k2_material": {
            "has_k2_total": n_a4_k2,
            "pct_of_all": pct(n_a4_k2, total),
            "has_k2_quiet": n_a4_k2_quiet,
            "pct_of_quiet": pct(n_a4_k2_quiet, n_quiet),
        },
        "gap_distribution_top1_top2": dict(gap_bins),
        "zone_pair_distribution_top1_top2": dict(zone_pairs),
        "gap_median": (statistics.median(top1_top2_gaps)
                       if top1_top2_gaps else None),
        "kb_constrained": kb_stats,
    }


def _self_test() -> None:
    """纯函数自测——构造 mock opens_recs 验证四个核心函数行为。"""

    results = []

    def mock(san, cp, zone, is_mate=False):
        return {"san": san, "uci": "", "cp": cp, "is_mate": is_mate,
                "zone": zone}

    # 1. _m5_forcing
    forcing = _m5_forcing([mock("e4", 200, "center"), mock("a3", 30, "queenside")])
    quiet = _m5_forcing([mock("e4", 50, "center"), mock("a3", 30, "queenside")])
    results.append(("_m5_forcing (gap200→True, gap20→False)",
                    forcing and not quiet, f"forcing={forcing} quiet={quiet}"))

    # 2. _fork_stats: 等强异 zone
    fs1 = _fork_stats([mock("a4", 50, "queenside"), mock("h4", 45, "kingside")])
    ok2 = fs1["equiv_diff_zone_pairs"] >= 1
    results.append(("_fork_stats (gap5异zone→有分岔)",
                    ok2, str(fs1)))

    # 3. _fork_stats: 等强同 zone → 0 分岔
    fs2 = _fork_stats([mock("c4", 50, "queenside"), mock("a3", 45, "queenside")])
    ok3 = fs2["equiv_diff_zone_pairs"] == 0 and fs2["equiv_pairs"] >= 1
    results.append(("_fork_stats (gap5同zone→无分岔)",
                    ok3, str(fs2)))

    # 4. _fork_stats: 异 zone 不等强 → 0 分岔
    fs3 = _fork_stats([mock("e4", 200, "center"), mock("a3", 30, "queenside")])
    ok4 = fs3["equiv_diff_zone_pairs"] == 0
    results.append(("_fork_stats (gap170异zone但不等强→无分岔)",
                    ok4, str(fs3)))

    # 5. _axis4_k2_material: 有 K2 (gap 80, 异 zone)
    a4_1 = _axis4_k2_material([mock("e4", 50, "center"), mock("a4", -30, "queenside")])
    ok5 = a4_1["has_k2"] is True and a4_1["k2_gap"] == 80
    results.append(("_axis4_k2 (gap80异zone→has_k2)",
                    ok5, str(a4_1)))

    # 6. _axis4_k2_material: 无 K2 (gap 200, 超 150)
    a4_2 = _axis4_k2_material([mock("e4", 200, "center"), mock("a3", 0, "queenside")])
    ok6 = a4_2["has_k2"] is False
    results.append(("_axis4_k2 (gap200超150→no k2)",
                    ok6, str(a4_2)))

    # 7. _axis4_k2_material: 无 K2 (同 zone)
    a4_3 = _axis4_k2_material([mock("c4", 50, "queenside"), mock("a3", -30, "queenside")])
    ok7 = a4_3["has_k2"] is False
    results.append(("_axis4_k2 (异zone false,同zone→no k2)",
                    ok7, str(a4_3)))

    # 8. _gap_bin
    ok8 = (_gap_bin(20) == "≤30" and _gap_bin(50) == "≤60"
           and _gap_bin(75) == "≤80" and _gap_bin(120) == "≤150"
           and _gap_bin(400) == ">300")
    results.append(("_gap_bin 分桶", ok8,
                    f"20→{_gap_bin(20)} 50→{_gap_bin(50)} 75→{_gap_bin(75)} "
                    f"120→{_gap_bin(120)} 400→{_gap_bin(400)}"))

    ok = True
    for name, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        ok &= passed
    print("纯函数自测:", "全部通过" if ok else "存在失败")
    raise SystemExit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="KB-free fork-stats + 轴 4 素材探针（PLAN-012 阶段 1）")
    ap.add_argument("--pgn", default=os.path.join(
        _ROOT, "data", "pgn", "lichess_elite_2025-11.pgn"))
    ap.add_argument("--games", type=int, default=250,
                    help="PGN 读取上限（代理档 250 ≈ 500 去重局面）")
    ap.add_argument("--k", type=int, default=5,
                    help="MultiPV 候选数（代理档 3 / 正式档 5）")
    ap.add_argument("--depth", type=int, default=14,
                    help="搜索深度（代理档 10 / 正式档 14）")
    ap.add_argument("--no-kb-analysis", action="store_true",
                    help="跳过 KB 约束路径（代理档，只跑 KB-free explore_open）")
    ap.add_argument("--out", default=os.path.join(
        OUT_DIR, "fork_stats_probe_result.json"))
    ap.add_argument("--jsonl", default=os.path.join(
        OUT_DIR, "fork_stats_probe_records.jsonl"))
    ap.add_argument("--summarize-only", action="store_true",
                    help="不跑引擎，只从已有 JSONL 聚合统计")
    ap.add_argument("--self-test", action="store_true",
                    help="纯函数自测（不跑引擎，不读 PGN）")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    if args.summarize_only:
        summary = summarize(args.jsonl)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=1)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return

    run(args.pgn, args.games, args.out, args.jsonl,
        k=args.k, depth=args.depth,
        kb_analysis=not args.no_kb_analysis)


if __name__ == "__main__":
    main()
