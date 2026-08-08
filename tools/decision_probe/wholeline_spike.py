"""整线机制保真约束 spike（PLAN-010，立场 B 架构方向验证，时限性探索）。

**背景**：ADR-020 立场 B「只约束首着方向、不约束整条线」（explore_forward
只把 direction_candidates 塞进 root_moves，之后引擎自由深搜）。PLAN-010
阶段 2/3/4 反复出现同一症状——线偏离计划机制：
  - 阶段 2：maroczy 两条约束线互相命中对方 goal 维（都奔 b5）；
  - 阶段 3：maroczy 王翼进攻首着是 Bg5 退象（与 h4 兵冲机制无关）；
  - 阶段 4：40 个名义双计划里 15 个两线趋同、7 个终局结构完全相同（0.0）。

**假设**：根着单点约束不足以让线忠于计划；若沿整条线持续施加方向约束
（每步都在 direction_candidates 内选最强着），线会更贴合机制，从而
  - 压低 A3 组内方差（同计划不同根着的线不再互相漂移），
  - 抬高 A3 跨计划分离（不同计划的线各自沿不同方向走）。

**本 spike 验证什么**：对 A3 未过/临界原型，比较两种取线方式的 A3：
  (a) baseline = explore_forward（根着约束，引擎自由深搜）；
  (b) wholeline = 逐步在 direction_candidates 内选引擎最强着（整线方向约束）。
若 (b) 的 A3 通过数/余量显著优于 (a)，则支持「整线约束」值得立 ADR；
若无改善甚至更差，则立场 B 不是瓶颈，回退「收缩产品野心」方向。

**机制保真度（fidelity）**：推进类计划（有 pawn_files）统计线前 N 着里
「目标兵线上的兵推进」次数——直接量化「线是否真的在推计划的兵」。
这是探索用的独立度量，不复用 direction_score（其 docstring 明言只用于
选根着、不用于验线，混用会违反单一事实来源的语义边界）。

时限性：只跑少量原型/局面，depth 从低（12），单引擎会话复用，产出
诊断性结论而非生产判据。结论交 planner 决定是否立 ADR/后继 PLAN。
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional

import chess

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
KB_PATH = os.path.join(_ROOT, "data", "structure_kb.json")
OUT = os.path.join(_ROOT, "data", "quality_benchmark_decision",
                   "wholeline_spike_result.json")

from src.analysis.direction import direction_candidates  # noqa: E402
from src.analysis.structure_id import applicable_mover_side  # noqa: E402
from src.analysis.structure_features import (  # noqa: E402
    feature_distance, structural_features)
from src.solver.branch_explorer import _open_engine, explore_forward  # noqa: E402

# 聚焦 A3 未过/临界 + 一个对照（carlsbad 稳健过）。FEN 同 p0_full_probe。
# 键 = KB 原型名（iqp_holder 的 KB 键是 "iqp"，用元组标注局面别名）。
SITUATIONS = {
    "carlsbad": "4r3/pp2r1k1/2p2ppn/3pP2p/3P3P/2NRP2K/PP3P2/2R5 w - - 0 31",
    "hanging": "2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/1R2R1K1 w - - 1 21",
    "stonewall": "rn3rk1/pb2q1pp/1ppbpn2/3pNp2/2PP4/1P4P1/PB1NPPBP/R2Q1RK1 w - - 2 11",
    "maroczy": "r2q1rk1/pp2ppbp/3pbnp1/8/2P1P3/2N1B3/PP1QBPPP/R3K2R w KQ - 5 11",
    "majority": "r2qr3/3bRpk1/p2p2p1/3P2Qp/1p6/1N3P2/PPP3PP/1K1R4 w - - 1 21",
    "iqp": "r1bq1rk1/pp2ppbp/5np1/n7/3P4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 2 11",
}

A3_GROUP_ROOTS = 3
LINE_N = 6          # spike 时限性：比 p0_full_probe 的 8 着短，两变体同口径即可
SPIKE_DEPTH = 10    # 时限性：低于生产 depth=18，换速度（诊断性结论非生产判据）


def _pawn_advance_fidelity(board: chess.Board, line: List[chess.Move],
                           direction: dict, n: int = LINE_N) -> int:
    """线前 n 着里「目标兵线上的兵推进」次数（推进类计划的机制保真度）。

    白方推进 = 目标格 rank 高于起始格；黑方相反。只数 direction.pawn_files
    上的兵——计划机制的直接执行手。无 pawn_files 的计划（保持类）返回 -1
    表示不适用。
    """
    files = {chess.FILE_NAMES.index(f) for f in direction.get("pawn_files", [])
             if f in chess.FILE_NAMES}
    if not files:
        return -1
    b = board.copy()
    cnt = 0
    for mv in line[:n]:
        piece = b.piece_at(mv.from_square)
        if piece is not None and piece.piece_type == chess.PAWN:
            if chess.square_file(mv.to_square) in files:
                advancing = (chess.square_rank(mv.to_square) >
                             chess.square_rank(mv.from_square)) if b.turn == chess.WHITE \
                    else (chess.square_rank(mv.to_square) <
                          chess.square_rank(mv.from_square))
                if advancing:
                    cnt += 1
        try:
            b.push(mv)
        except Exception:
            break
    return cnt


def _end_fv(board: chess.Board, line: List[chess.Move], n: int = LINE_N):
    """线前 n 着终点特征（锚定决策点走子方，与 _line_features 同口径）。"""
    mover = board.turn
    b = board.copy()
    fv = structural_features(b, mover)
    for mv in line[:n]:
        try:
            b.push(mv)
        except Exception:
            break
        fv = structural_features(b, mover)
    return fv


def _wholeline_line(board: chess.Board, plan: dict, engine,
                    depth: int, max_moves: int = LINE_N) -> List[chess.Move]:
    """整线方向约束取线：每步在 direction_candidates 内选引擎最强着。

    与 explore_forward（根着单点约束）的区别：这里**每一步**都重新计算
    direction_candidates 并限制引擎只在该集内选——线被持续拉回计划方向。
    候选集为空即停（线自然终止）。
    """
    direction = plan.get("direction") or {}
    b = board.copy()
    line: List[chess.Move] = []
    for _ in range(max_moves):
        cands = direction_candidates(b, direction, top_n=10)
        if not cands:
            break
        try:
            info = engine.analyse(b, chess.engine.Limit(depth=depth),
                                  root_moves=cands)
        except Exception:
            break
        pv = info.get("pv") or []
        if not pv:
            break
        mv = pv[0]
        if mv not in b.legal_moves:
            break
        line.append(mv)
        b.push(mv)
    return line


def _a3_medians(within: List[float], cross: List[float]):
    import statistics
    w = statistics.median(within) if within else None
    c = statistics.median(cross) if cross else None
    passed = (w is not None and c is not None and c > w)
    return w, c, passed


def run() -> Dict:
    sf = os.getenv("STOCKFISH_PATH", "")
    if not os.path.isabs(sf):
        sf = os.path.normpath(os.path.join(_ROOT, sf))
    kb = json.load(open(KB_PATH, encoding="utf-8"))

    print(f"整线约束 spike | depth={SPIKE_DEPTH} | {len(SITUATIONS)} 原型")
    print("=" * 78)

    engine = _open_engine(sf)
    results = {}
    try:
        for sit, fen in SITUATIONS.items():
            board = chess.Board(fen)
            plans = kb[sit]["plans"]
            # 角色闸（同 p0_full_probe，带非空守卫）：iqp 施压/持有计划互斥，
            # stonewall 的 mover_side 标注与角色语义错位时守卫防过滤为空。
            side = applicable_mover_side(board, sit)
            if side is not None:
                filtered = [p for p in plans if p.get("mover_side") == side]
                if filtered:
                    plans = filtered
            t0 = time.time()

            # 每计划两种取线：baseline（explore_forward）/ wholeline（逐步约束）
            base_lines: Dict[str, List[List[chess.Move]]] = {}
            whole_lines: Dict[str, List[List[chess.Move]]] = {}
            fidelity_base: Dict[str, List[int]] = {}
            fidelity_whole: Dict[str, List[int]] = {}

            for plan in plans:
                direction = plan.get("direction") or {}
                cands = direction_candidates(board, direction, top_n=10)
                bl, wl = [], []
                fb, fw = [], []
                for root in cands[:A3_GROUP_ROOTS]:
                    # baseline：root_moves=[root] 深搜、引擎自由续着（现状口径）
                    try:
                        info = engine.analyse(board,
                                              chess.engine.Limit(depth=SPIKE_DEPTH),
                                              root_moves=[root])
                        pv = info.get("pv") or []
                    except Exception:
                        pv = []
                    if pv:
                        bl.append(list(pv))
                        fb.append(_pawn_advance_fidelity(board, pv, direction))
                        # wholeline：同一 root 起步，其后每步方向约束——与
                        # baseline 唯一差异是「首着之后是否持续约束」，隔离变量。
                        b2 = board.copy()
                        b2.push(root)
                        rest = _wholeline_line(b2, plan, engine, SPIKE_DEPTH,
                                               max_moves=LINE_N - 1)
                        wline = [root] + rest
                        wl.append(wline)
                        fw.append(_pawn_advance_fidelity(board, wline, direction))
                if len(bl) >= 2:
                    base_lines[plan["name"]] = bl
                    fidelity_base[plan["name"]] = fb
                if len(wl) >= 2:
                    whole_lines[plan["name"]] = wl
                    fidelity_whole[plan["name"]] = fw

            def distances(lines_by_plan):
                within, cross = [], []
                names = list(lines_by_plan.keys())
                fvs = {nm: [_end_fv(board, ln) for ln in lines_by_plan[nm]]
                       for nm in names}
                for nm in names:
                    L = fvs[nm]
                    for i in range(len(L)):
                        for j in range(i + 1, len(L)):
                            within.append(feature_distance(L[i], L[j]))
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        for la in fvs[names[i]]:
                            for lb in fvs[names[j]]:
                                cross.append(feature_distance(la, lb))
                return within, cross

            bw, bc = distances(base_lines)
            ww, wc = distances(whole_lines)
            b_wmed, b_cmed, b_pass = _a3_medians(bw, bc)
            w_wmed, w_cmed, w_pass = _a3_medians(ww, wc)

            results[sit] = {
                "baseline": {"within_median": b_wmed, "cross_median": b_cmed,
                             "passed": b_pass},
                "wholeline": {"within_median": w_wmed, "cross_median": w_cmed,
                              "passed": w_pass},
                "fidelity_baseline": fidelity_base,
                "fidelity_wholeline": fidelity_whole,
            }
            print(f"\n[{sit}] {time.time() - t0:.0f}s")
            print(f"  baseline : 组内={b_wmed} 跨={b_cmed} "
                  f"{'✓' if b_pass else '✗'}  fidelity={fidelity_base}")
            print(f"  wholeline: 组内={w_wmed} 跨={w_cmed} "
                  f"{'✓' if w_pass else '✗'}  fidelity={fidelity_whole}")
    finally:
        engine.quit()

    # 汇总
    base_pass = sum(1 for r in results.values() if r["baseline"]["passed"])
    whole_pass = sum(1 for r in results.values() if r["wholeline"]["passed"])
    print("\n" + "=" * 78)
    print(f"A3 通过数：baseline {base_pass}/{len(results)} → "
          f"wholeline {whole_pass}/{len(results)}")

    out = {"depth": SPIKE_DEPTH, "results": results,
           "base_pass": base_pass, "whole_pass": whole_pass}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"结果写入 {OUT}")
    return out


if __name__ == "__main__":
    run()
