"""B2 候选集覆盖实战频率 top-2 探针（P0-lite）。

判据（FINDINGS-002 §3.3 B2）：系统 top-k 候选计划覆盖该兵形实战频率 top-2
的计划方向 ≥80%。

- 实战频率：P17 统计（卡尔斯巴德 170 局计划分桶）——top-2 方向 = 少数派攻击 / 中心突破；
- KB 覆盖（程序化判定）：实际计划着 ∈ 任一 KB 计划 `direction_candidates` 并集
  （单一事实来源：direction 语义唯一，覆盖判定与 A1 同口径）；
- IQP 角色拆分：对方持孤兵（施压方走子）→ 用 KB v0 的施压/推进计划测覆盖；
  己方持孤兵（持有方走子）→ KB v0 缺持有方计划（保持/推进），判据暂不可测，
  输出持有方实际着法分布供 KB 扩容决策（P4 修订里 IQP 白方「保持 vs 推 d5」
  是轴 1 对比，本缺口必须在阶段 1 前补上）。

用法：
    "C:\\Users\\LiuYiJie\\.conda\\envs\\commentary\\python.exe" -m tools.decision_probe.b2_coverage_probe \\
        --pgn data/pgn/lichess_elite_2025-11.pgn
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import chess

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from tools.decision_probe.engine_probe import direction_candidates  # noqa: E402
from tools.pgn_plan_stats import open_pgn_stream, _iter_kept_games  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "data", "quality_benchmark_decision")
KB_PATH = os.path.join(_ROOT, "data", "structure_kb.json")

TOP_N = 10
PLY_MIN, PLY_MAX = 16, 32
MIN_PIECE_COUNT = 18
IQP_POOL_OPENINGS = ["Queen's Gambit Accepted", "Semi-Tarrasch", "Ragozin",
                     "Panov", "Alapin", "Queen's Gambit Declined: Tarrasch"]


def _kb_plan_candidates(board: chess.Board, archetype: str, plans: list) -> set:
    """某原型全部计划的 direction_candidates 并集（SAN 集合）。"""
    out = set()
    for p in plans:
        for mv in direction_candidates(board, p["direction"], top_n=TOP_N):
            out.add(board.san(mv))
    return out


def _mover_holds_iqp(board: chess.Board) -> bool:
    """走子方是否持 d4 孤立兵（归一化视角：mover=白）。"""
    files = {chess.square_file(s) for s in board.pieces(chess.PAWN, chess.WHITE)}
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        if chess.square_file(sq) == 3 and chess.square_rank(sq) == 3:
            if 2 not in files and 4 not in files:
                return True
    return False


def _sample_iqp_positions(pgn_path: str, max_samples: int) -> list:
    """采样 IQP 决策点位置（对方持孤兵角色为主），记录后续 8 着。"""
    out = []
    for game in _iter_kept_games(open_pgn_stream(pgn_path)):
        if len(out) >= max_samples:
            break
        opening = game.headers.get("Opening", "")
        if not any(k in opening for k in IQP_POOL_OPENINGS):
            continue
        b = game.board()
        node = game
        ply = 0
        while node.next() is not None and ply < PLY_MAX:
            node = node.next()
            b.push(node.move)
            ply += 1
            if ply < PLY_MIN or ply > PLY_MAX:
                continue
            if len(b.piece_map()) < MIN_PIECE_COUNT or b.is_check():
                continue
            # 对方持孤兵（d5 孤立）且轮到白方——施压方决策点
            #
            # `b.turn == WHITE` 是必须的（08.04 补）：原代码注释写了「且轮到
            # 白方」但没写这个判断，于是 20 个采样里混进 5 个「轮到黑方走」的
            # 局面。run_iqp 按 `i % 2 == 0` 认定偶数位是施压方（白方）的着，
            # 在那 5 个样本上数的其实是黑方的着，必然判未覆盖——B2 卡在
            # 70%（14/20）正是这 5 个（25%）造成的假阴性，不是 KB 缺计划。
            if b.turn != chess.WHITE:
                continue
            bf = {chess.square_file(s) for s in
                  b.pieces(chess.PAWN, chess.BLACK)}
            opp_isolated = any(
                chess.square_file(s) == 3 and chess.square_rank(s) == 4
                for s in b.pieces(chess.PAWN, chess.BLACK)
            ) and 2 not in bf and 4 not in bf
            if not opp_isolated:
                continue
            cont = []
            nxt = node
            while nxt.next() is not None and len(cont) < 8:
                nxt = nxt.next()
                cont.append(nxt.move.uci())
            out.append({"fen": b.fen(), "ply": ply, "opening": opening,
                        "continuation": cont})
            break
    return out


def run_carlsbad() -> dict:
    """卡尔斯巴德：P17 样本 + KB 计划覆盖。"""
    kb = json.load(open(KB_PATH, encoding="utf-8"))
    stats = json.load(open(os.path.join(
        OUT_DIR, "p17_stats_carlsbad.json"), encoding="utf-8"))
    samples = stats["samples"]
    plans = kb["carlsbad"]["plans"]

    # 频率 top-2（P17 计划分桶）
    fam = Counter(s["first_plan_family"] for s in samples
                  if s["first_plan_family"])
    top2 = [k for k, _ in fam.most_common(2) if k != "未定型(窗口内无计划承诺着)"]
    print(f"卡尔斯巴德实战计划频率 top-2: {top2}")
    print(f"  KB 计划: {[p['name'] for p in plans]}")

    covered = with_plan = 0
    misses = []
    for s in samples:
        plan_san = s.get("first_plan_move_san")
        if not plan_san:
            continue  # 未定型：窗口内无计划着（P17 已报 71.8%），不计入
        with_plan += 1
        board = chess.Board(s["fen"])
        cands = _kb_plan_candidates(board, "carlsbad", plans)
        if plan_san in cands:
            covered += 1
        else:
            misses.append({"plan_san": plan_san, "opening": s["opening"]})
    rate = round(100.0 * covered / with_plan, 1) if with_plan else 0.0
    print(f"覆盖判定：{covered}/{with_plan} = {rate}%"
          f"（判据 ≥80%，未定型样本不计入）")
    for m in misses[:5]:
        print(f"    未覆盖: {m}")
    return {"type": "carlsbad", "top2_directions": top2,
            "kb_plans": [p["name"] for p in plans],
            "with_plan": with_plan, "covered": covered, "rate_pct": rate}


def run_iqp(pgn_path: str, max_samples: int) -> dict:
    """IQP：施压方角色覆盖判定 + 持有方角色缺口报告。"""
    kb = json.load(open(KB_PATH, encoding="utf-8"))
    plans = kb["iqp"]["plans"]  # v0: 施压 + 推进消除（对方持孤兵视角）
    positions = _sample_iqp_positions(pgn_path, max_samples)
    print(f"\nIQP 施压方决策点采样: {len(positions)}")

    covered = with_plan = 0
    misses = []
    for pos in positions:
        board = chess.Board(pos["fen"])
        # 施压方前 6 着内任一着命中 KB 候选并集即视为覆盖——
        # 首着可能是发展着（O-O/Nf3 等，非计划承诺），计划着常在第 2~4 着。
        cands = _kb_plan_candidates(board, "iqp", plans)
        hit_san = None
        temp = board.copy()
        # 施压方 = 决策点的走子方。**从棋盘推导而非假定奇偶**（08.04 修）：
        # 原写法 `i % 2 != 0: continue` 隐含「continuation[0] 一定是施压方的
        # 着」，一旦采样混入「轮到对方走」的局面就会数错颜色。采样器已补
        # turn 检查，这里再按 temp.turn 判一次——两处都改，同类 bug 不复发。
        pressuring = board.turn
        for u in pos["continuation"][:6]:
            try:
                mv = chess.Move.from_uci(u)
                san = temp.san(mv)
            except Exception:
                break
            is_pressuring_move = (temp.turn == pressuring)
            if is_pressuring_move and san in cands:
                hit_san = san
                break
            try:
                temp.push(mv)
            except Exception:
                break
        with_plan += 1
        if hit_san is not None:
            covered += 1
        else:
            misses.append({"first_moves": pos["continuation"][:6],
                           "opening": pos["opening"]})
    rate = round(100.0 * covered / with_plan, 1) if with_plan else 0.0
    print(f"施压方覆盖判定（前 6 着窗口）：{covered}/{with_plan} = {rate}%")
    if misses:
        print("  未覆盖样本（前 6 着）:")
        for m in misses[:6]:
            print(f"    {m['first_moves']}  [{m['opening'][:40]}]")
    # KB 缺口提示已过期（08.04 核）：阶段 1 扩容（commit 838a938）已补齐 IQP
    # 持有方两计划（保持孤兵 / 推进兑掉），本探针的 plans 取的就是 kb["iqp"]
    # 全集。此处如实报告 KB 现状，不再输出「缺持有方计划」的陈旧结论。
    holder_plans = [p["name"] for p in plans
                    if "保持" in p["name"] or "推进" in p["name"]]
    print(f"  KB 现有 IQP 计划 {len(plans)} 条（含持有方 {len(holder_plans)} 条："
          f"{'、'.join(holder_plans) if holder_plans else '无'}）。")
    return {"type": "iqp", "positions": len(positions),
            "with_plan": with_plan, "covered": covered, "rate_pct": rate,
            "kb_plans": [p["name"] for p in plans],
            "kb_holder_plans": holder_plans}


def main() -> None:
    ap = argparse.ArgumentParser(description="B2 候选集覆盖实战频率 top-2 探针")
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--max-iqp", type=int, default=20)
    args = ap.parse_args()
    pgn = args.pgn if os.path.isabs(args.pgn) else os.path.join(_ROOT, args.pgn)

    carlsbad = run_carlsbad()
    iqp = run_iqp(pgn, args.max_iqp)

    # 判定：两项均 ≥80% 才算通过（FINDINGS-002 §3.3 B2 判据）。
    #
    # 08.04 修：原逻辑把「无 ❌」的分支写成了「部分通过…缺口待补」，
    # 即两项全达标时反而输出未通过措辞——判定与文案倒置。同时 KB 已在
    # 838a938 补齐 IQP 持有方计划，「KB 缺口待补」的说法本身已过期。
    c_ok = carlsbad["rate_pct"] >= 80
    i_ok = iqp["rate_pct"] >= 80
    verdict_parts = [
        f"卡尔斯巴德覆盖 {carlsbad['rate_pct']}%({'✅' if c_ok else '❌'})",
        f"IQP 施压方覆盖 {iqp['rate_pct']}%({'✅' if i_ok else '❌'})",
    ]
    verdict = (("B2 ✅ 通过（判据 ≥80%）：" if (c_ok and i_ok)
                else "B2 ❌ 未通过（判据 ≥80%）：")
               + "；".join(verdict_parts))
    print("\n" + "=" * 64)
    print(verdict)
    print("=" * 64)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "b2_coverage_probe_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "carlsbad": carlsbad, "iqp": iqp},
                  f, ensure_ascii=False, indent=1)
    print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()
