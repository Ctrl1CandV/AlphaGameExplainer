"""B1 原型识别命中率探针（P0-lite）——structure_id 对真实中局局面的识别验证。

判据（FINDINGS-002 §3.3 B1）：20~30 真实局面，系统主识别命中 ≥70%。

ground truth 双源交叉（P14：ECO 反推为辅助交叉，非依赖）：
- 源 1（开局信号，独立于 structure_id）：按 lichess Opening 头筛对局池
  （QGD Exchange → carlsbad 池；QGA/Semi-Tarrasch/Ragozin/Panov/Alapin/Tarrasch
  → iqp 池）；
- 源 2（内联几何，探针内独立实现）：在决策点窗口（ply 16~32）内用简单的
  兵形骨架检查确认结构确实形成——开局长尾（如 Tarrasch 未形成 IQP）被过滤，
  不作为样本；
- 双源都成立的位置才入样本池，再与 structure_id 的识别比对。

用法：
    "C:\\Users\\LiuYiJie\\.conda\\envs\\commentary\\python.exe" -m tools.decision_probe.b1_identify_probe \\
        --pgn data/pgn/lichess_elite_2025-11.pgn
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import chess

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from tools.pgn_plan_stats import open_pgn_stream, _iter_kept_games  # noqa: E402
from src.analysis.structure_id import detect_pawn_structure  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "data", "quality_benchmark_decision")

# 开局池：Opening 头子串 → 期望原型（源 1）
POOLS = {
    "carlsbad": {
        "expect": "carlsbad",
        "opening_contains": ["Queen's Gambit Declined: Exchange",
                             "QGD: Exchange"],
    },
    "iqp": {
        "expect": "iqp",
        "opening_contains": ["Queen's Gambit Accepted", "Semi-Tarrasch",
                             "Ragozin", "Panov", "Alapin",
                             "Queen's Gambit Declined: Tarrasch"],
    },
}

PLY_MIN, PLY_MAX = 16, 32   # 决策点窗口（第 8~16 回合）——结构新鲜期
MIN_PIECE_COUNT = 18


def _inline_carlsbad(b: chess.Board) -> bool:
    """内联卡尔斯巴德骨架（源 2，独立于 structure_id 的实现）。"""
    wf = {chess.square_file(s) for s in b.pieces(chess.PAWN, chess.WHITE)}
    bf = {chess.square_file(s) for s in b.pieces(chess.PAWN, chess.BLACK)}
    w_d4 = any(chess.square_rank(s) == 3 and chess.square_file(s) == 3
               for s in b.pieces(chess.PAWN, chess.WHITE))
    b_d5 = any(chess.square_rank(s) == 4 and chess.square_file(s) == 3
               for s in b.pieces(chess.PAWN, chess.BLACK))
    if not (w_d4 and b_d5):
        return False
    if 2 in wf or 4 in bf:
        return False
    # 黑方 c 兵未推进到 c5（rank < 4）
    for sq in b.pieces(chess.PAWN, chess.BLACK):
        if chess.square_file(sq) == 2 and chess.square_rank(sq) < 4:
            return False
    return True


def _inline_iqp(b: chess.Board) -> bool:
    """内联孤后兵检查（源 2）：任一方 d4/d5 孤立兵。"""
    for color in (chess.WHITE, chess.BLACK):
        files = {chess.square_file(s) for s in b.pieces(chess.PAWN, color)}
        for sq in b.pieces(chess.PAWN, color):
            if chess.square_file(sq) == 3 and chess.square_rank(sq) in (3, 4):
                if 2 not in files and 4 not in files:
                    return True
    return False


def _sample_pool(pgn_path: str, pool: dict, max_samples: int) -> list:
    """从开局池对局中采样决策点位置（双源交叉后）。"""
    checks = {"carlsbad": _inline_carlsbad, "iqp": _inline_iqp}
    check = checks[pool["expect"]]
    out, skipped = [], 0
    for game in _iter_kept_games(open_pgn_stream(pgn_path)):
        if len(out) >= max_samples:
            break
        opening = game.headers.get("Opening", "")
        if not any(k in opening for k in pool["opening_contains"]):
            continue
        b = game.board()
        node = game
        ply = 0
        found = None
        while node.next() is not None and ply < PLY_MAX:
            node = node.next()
            b.push(node.move)
            ply += 1
            if ply < PLY_MIN or ply > PLY_MAX:
                continue
            if len(b.piece_map()) < MIN_PIECE_COUNT or b.is_check():
                continue
            if check(b):
                found = {"fen": b.fen(), "ply": ply,
                         "opening": opening, "eco": game.headers.get("ECO", "")}
                break
        if found is None:
            skipped += 1
            continue
        out.append(found)
    return out, skipped


def run(pgn_path: str, max_per_pool: int) -> float:
    records, total, hit = [], 0, 0
    per_pool = {}
    for pool_name, pool in POOLS.items():
        samples, skipped = _sample_pool(pgn_path, pool, max_per_pool)
        h = 0
        for s in samples:
            got, conf, _ = detect_pawn_structure(chess.Board(s["fen"]))
            ok = got == pool["expect"]
            h += int(ok)
            records.append({**s, "expect": pool["expect"], "got": got,
                            "hit": ok})
        rate = round(100.0 * h / len(samples), 1) if samples else 0.0
        per_pool[pool_name] = {"samples": len(samples), "hit": h,
                               "rate_pct": rate, "skipped_no_structure": skipped}
        total += len(samples)
        hit += h
        print(f"  {pool_name:10s} 期望 {pool['expect']:8s} 命中 {h}/{len(samples)}"
              f" ({rate}%)  [窗口内未形成结构 {skipped} 局]")
        if len(samples) < 10:
            print(f"    ⚠️ 样本 {len(samples)} < 10，参考性弱")

    rate = round(100.0 * hit / total, 1) if total else 0.0
    passed = rate >= 70.0
    verdict = (f"B1 {'✅ 通过' if passed else '❌ 不通过'}"
               f"（命中率 {rate}%，判据 ≥70%，总样本 {total}）")
    print("=" * 64)
    print(verdict)
    print("=" * 64)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "b1_identify_probe_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {"ply_range": [PLY_MIN, PLY_MAX],
                       "min_piece_count": MIN_PIECE_COUNT,
                       "max_per_pool": max_per_pool},
            "summary": {"total": total, "hit": hit, "rate_pct": rate,
                        "passed": passed, "verdict": verdict,
                        "per_pool": per_pool},
            "records": records,
        }, f, ensure_ascii=False, indent=1)
    print(f"结果已写入 {out_path}")
    return rate


def main() -> None:
    ap = argparse.ArgumentParser(description="B1 原型识别命中率探针（双源交叉）")
    ap.add_argument("--pgn", required=True, help=".pgn 或 .zip")
    ap.add_argument("--max-per-pool", type=int, default=15)
    args = ap.parse_args()
    pgn = args.pgn if os.path.isabs(args.pgn) else os.path.join(_ROOT, args.pgn)
    run(pgn, args.max_per_pool)


if __name__ == "__main__":
    main()
