"""PGN 中局局面 M5 探针——验证「换源后局面能否过 M5」这个核心假设。

M5 冒烟证明了 puzzle 库不行（存活 0/79），但那只是否证旧源。
本脚本验证新源：从强手对局 PGN 抽中局局面，看 M5 存活率。

这是 ADR-020 架构决策 1（PGN 主源）的正面验证。若 PGN 源的 M5 存活率
同样极低，则整个方案的输入层假设崩塌，必须在动工前知道。

同时产出：
  - M6 评估窗口存活率（局面是否均衡~温和优势）
  - M8 近等强首着数分布（是否真有多个方向可选）
  - 每局面的 direction_zone 分布（候选首着是否落在不同区域）
  - 时限筛（P19）与去重（P21）的实际衰减

用法：
    python -m tools.decision_probe.pgn_m5_probe --pgn data/pgn/xxx.pgn --games 300
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from collections import Counter
from typing import Dict, Iterator, List, Optional

import chess
import chess.pgn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tools.decision_probe.engine_probe import (  # noqa: E402
    DEFAULT_EQUIV_CP,
    DEFAULT_MULTIPV,
    DEFAULT_STANDOUT_CP,
    EngineProbe,
    direction_zone,
    resolve_stockfish,
)

# ---------------------------------------------------------------- 抽样参数

MIN_PLY = 40            # 第 20 回合之后（ply = 2 × move）
MAX_PLY = 70            # 第 35 回合之前——再往后多半进残局
MIN_PIECE_COUNT = 18    # M3 子力窗口
M6_ABS_CP_MAX = 300     # M6 评估窗口：|eval| ≤ 3 兵

# P19 时限筛：只要 classical / rapid
MIN_BASE_SECONDS = 180


def parse_time_control(tc: str) -> Optional[int]:
    """从 TimeControl 头解析基础时限秒数。`600+5` -> 600；`-` -> None。"""
    if not tc or tc == "-":
        return None
    base = tc.split("+")[0].strip()
    try:
        return int(base)
    except ValueError:
        return None


def structure_signature(board: chess.Board) -> str:
    """兵形签名（P21 去重用）——只看兵的位置，忽略其他子力。

    同一开局体系下大量对局会到达相同兵形，不去重的话「候选池上千」
    实际只是几十个不同局面被重复计数。
    """
    wp = sorted(chess.square_name(s) for s in board.pieces(chess.PAWN, chess.WHITE))
    bp = sorted(chess.square_name(s) for s in board.pieces(chess.PAWN, chess.BLACK))
    return f"W:{','.join(wp)}|B:{','.join(bp)}"


def material_signature(board: chess.Board) -> str:
    """子力签名——与兵形签名合成分桶键。"""
    counts = []
    for color in (chess.WHITE, chess.BLACK):
        for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            counts.append(len(board.pieces(pt, color)))
    return "-".join(str(c) for c in counts)


# ---------------------------------------------------------------- PGN 读取


def open_pgn_stream(path: str) -> io.TextIOBase:
    """支持 .zip（取其中第一个 .pgn）与裸 .pgn。"""
    if path.lower().endswith(".zip"):
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if n.lower().endswith(".pgn")]
        if not names:
            raise RuntimeError(f"zip 内无 .pgn 文件: {path}")
        raw = zf.open(names[0])
        return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_midgame_positions(
    pgn_path: str,
    max_games: int,
    seen_buckets: Counter,
    bucket_cap: int = 2,
) -> Iterator[Dict]:
    """流式读 PGN，每局最多取 1 个中局局面（P21：不逐着抽）。

    产出 dict：fen / ply / time_control / white_elo / black_elo / url /
              bucket / continuation（实战续走前 6 着 UCI，供 P12 对照）
    """
    handle = open_pgn_stream(pgn_path)
    games_read = 0
    stats = Counter()

    try:
        while games_read < max_games:
            try:
                game = chess.pgn.read_game(handle)
            except Exception:
                stats["parse_error"] += 1
                continue
            if game is None:
                break
            games_read += 1

            headers = game.headers
            tc_base = parse_time_control(headers.get("TimeControl", ""))
            if tc_base is None or tc_base < MIN_BASE_SECONDS:
                stats["timecontrol_rejected"] += 1
                continue

            board = game.board()
            moves = list(game.mainline_moves())
            if len(moves) < MIN_PLY + 6:
                stats["too_short"] += 1
                continue

            # 推进到抽样窗口，取第一个满足子力条件的局面
            picked = None
            for idx, mv in enumerate(moves):
                board.push(mv)
                ply = idx + 1
                if ply < MIN_PLY:
                    continue
                if ply > MAX_PLY:
                    break
                if len(board.piece_map()) < MIN_PIECE_COUNT:
                    continue
                if board.is_check():
                    continue  # 被将军的局面不谈战略选择
                bucket = f"{structure_signature(board)}#{material_signature(board)}"
                if seen_buckets[bucket] >= bucket_cap:
                    stats["bucket_full"] += 1
                    continue
                seen_buckets[bucket] += 1
                picked = {
                    "fen": board.fen(),
                    "ply": ply,
                    "time_control": headers.get("TimeControl", ""),
                    "white_elo": headers.get("WhiteElo", ""),
                    "black_elo": headers.get("BlackElo", ""),
                    "url": headers.get("LichessURL", "") or headers.get("Site", ""),
                    "eco": headers.get("ECO", ""),
                    "opening": headers.get("Opening", ""),
                    "bucket": bucket,
                    "continuation": [m.uci() for m in moves[idx + 1: idx + 7]],
                }
                break

            if picked is None:
                stats["no_position_in_window"] += 1
                continue
            stats["sampled"] += 1
            yield picked
    finally:
        handle.close()
        iter_midgame_positions.last_stats = stats  # type: ignore[attr-defined]
        iter_midgame_positions.games_read = games_read  # type: ignore[attr-defined]


# ---------------------------------------------------------------- 主流程


def run(
    pgn_path: str,
    max_games: int,
    depth: int,
    standout_cp: int,
    equiv_cp: int,
    multipv_k: int,
    out_path: str,
) -> Dict:
    sf = resolve_stockfish()
    if not sf:
        print("找不到 Stockfish", file=sys.stderr)
        sys.exit(2)

    print(f"PGN: {pgn_path}")
    print(f"读取上限 {max_games} 局 | depth={depth} standout={standout_cp}cp "
          f"equiv={equiv_cp}cp k={multipv_k}")
    print(f"抽样窗口 ply {MIN_PLY}~{MAX_PLY} | 子力 ≥{MIN_PIECE_COUNT} | "
          f"时限 ≥{MIN_BASE_SECONDS}s | 每桶上限 2")
    print("-" * 72)

    seen_buckets: Counter = Counter()
    positions = list(iter_midgame_positions(pgn_path, max_games, seen_buckets))
    sample_stats = getattr(iter_midgame_positions, "last_stats", Counter())
    games_read = getattr(iter_midgame_positions, "games_read", 0)

    print(f"读入 {games_read} 局 → 抽样 {len(positions)} 个中局局面")
    for k, v in sample_stats.most_common():
        print(f"    {k:28s} {v}")
    print("-" * 72)

    if not positions:
        print("无可用局面，检查 PGN 与抽样窗口", file=sys.stderr)
        sys.exit(1)

    records: List[Dict] = []
    m5_pass = m6_pass = m8_pass = 0
    gaps: List[int] = []
    equiv_counts: List[int] = []
    zone_spread: Counter = Counter()
    m5_fail: Counter = Counter()

    t0 = time.time()
    with EngineProbe(sf, depth=depth, multipv=multipv_k) as probe:
        for i, pos in enumerate(positions, 1):
            board = chess.Board(pos["fen"])
            cands = probe.analyse(board)
            rec = dict(pos)

            if not cands:
                rec["m5"] = False
                rec["m5_reason"] = "引擎无输出"
                m5_fail["引擎无输出"] += 1
                records.append(rec)
                continue

            best = cands[0]
            rec["eval_cp"] = best.score_cp
            rec["best_move"] = best.move_uci

            # M5：无强制战术
            forced, reason, gap = probe.has_forcing_tactic(
                board, cands, standout_cp=standout_cp)
            rec["m5"] = not forced
            rec["m5_reason"] = reason
            rec["gap_cp"] = gap
            if gap is not None:
                gaps.append(gap)
            if forced:
                m5_fail[reason or "未知"] += 1
                records.append(rec)
                if i % 25 == 0 or i == len(positions):
                    print(f"[{i:4d}/{len(positions)}] M5 存活 {m5_pass} "
                          f"M6 {m6_pass} M8 {m8_pass}")
                continue
            m5_pass += 1

            # M6：评估窗口
            in_window = (best.score_cp is not None
                         and best.mate is None
                         and abs(best.score_cp) <= M6_ABS_CP_MAX)
            rec["m6"] = in_window
            if not in_window:
                records.append(rec)
                continue
            m6_pass += 1

            # M8：近等强首着数 ≥2 且方向不同
            equiv = probe.near_equal_moves(cands, equiv_cp=equiv_cp)
            zones = sorted({direction_zone(chess.Move.from_uci(c.move_uci))
                            for c in equiv})
            rec["equiv_count"] = len(equiv)
            rec["equiv_moves"] = [c.move_uci for c in equiv]
            rec["equiv_zones"] = zones
            equiv_counts.append(len(equiv))
            zone_spread[len(zones)] += 1
            rec["m8"] = len(equiv) >= 2 and len(zones) >= 2
            if rec["m8"]:
                m8_pass += 1
            records.append(rec)

            if i % 25 == 0 or i == len(positions):
                print(f"[{i:4d}/{len(positions)}] M5 存活 {m5_pass} "
                      f"M6 {m6_pass} M8 {m8_pass}")

    elapsed = time.time() - t0
    n = len(positions)
    print("-" * 72)
    print(f"耗时 {elapsed:.1f}s（{elapsed / max(n, 1):.2f}s/局面）")

    def pct(x: int) -> float:
        return round(100.0 * x / n, 1) if n else 0.0

    summary = {
        "pgn": os.path.basename(pgn_path),
        "games_read": games_read,
        "positions_sampled": n,
        "sample_stats": dict(sample_stats),
        "unique_buckets": len(seen_buckets),
        "m5_passed": m5_pass,
        "m6_passed": m6_pass,
        "m8_passed": m8_pass,
        "m5_pct": pct(m5_pass),
        "m6_pct": pct(m6_pass),
        "m8_pct": pct(m8_pass),
        "m5_fail_buckets": dict(m5_fail),
        "zone_spread": dict(zone_spread),
    }

    if gaps:
        gs = sorted(gaps)
        summary["gap_percentiles"] = {
            "min": gs[0],
            "p10": gs[len(gs) // 10],
            "p50": gs[len(gs) // 2],
            "p90": gs[min(len(gs) - 1, len(gs) * 9 // 10)],
            "max": gs[-1],
        }
    if equiv_counts:
        es = sorted(equiv_counts)
        summary["equiv_count_percentiles"] = {
            "min": es[0],
            "p50": es[len(es) // 2],
            "max": es[-1],
        }

    print()
    print("=" * 72)
    print("PGN 中局局面漏斗（分母 = 抽样局面数）")
    print("=" * 72)
    print(f"  {'抽样局面':38s} {n:5d}   100.0%")
    print(f"  {'M5 无强制战术':38s} {m5_pass:5d}   {pct(m5_pass):5.1f}%")
    print(f"  {'M6 评估窗口内':38s} {m6_pass:5d}   {pct(m6_pass):5.1f}%")
    print(f"  {'M8 近等强首着 ≥2 且方向 ≥2':38s} {m8_pass:5d}   {pct(m8_pass):5.1f}%")
    print(f"\n  去重后不同兵形桶: {len(seen_buckets)}")

    if m5_fail:
        print("\nM5 未通过原因：")
        for k, v in m5_fail.most_common():
            print(f"  {k:38s} {v:5d}")

    if "gap_percentiles" in summary:
        g = summary["gap_percentiles"]
        print(f"\n首选-次选 gap 分布（cp）：")
        print(f"  最小 {g['min']}  P10 {g['p10']}  中位 {g['p50']}  "
              f"P90 {g['p90']}  最大 {g['max']}")

    if zone_spread:
        print(f"\n近等强首着覆盖的方向区域数分布：")
        for k in sorted(zone_spread):
            print(f"  {k} 个区域: {zone_spread[k]} 局面")

    # 与 puzzle 源对照裁决
    print()
    print("=" * 72)
    if m5_pass == 0:
        verdict = ("PGN 源同样归零——输入层假设崩塌，方案需重新评估（考虑"
                   "放宽 M5 定义或承认「战略决策局面」在强手对局中也是稀有事件）")
    elif pct(m5_pass) < 20:
        verdict = (f"PGN 源 M5 存活 {pct(m5_pass)}%——显著优于 puzzle 源的 0%，"
                   f"换源方向成立，但产能需按此率重算")
    else:
        verdict = (f"PGN 源 M5 存活 {pct(m5_pass)}%——换源决策获得强力正面验证，"
                   f"输入层不再是瓶颈")
    summary["verdict"] = verdict
    print(verdict)
    if m8_pass:
        print(f"M8 最终存活 {pct(m8_pass)}%（{m8_pass}/{n}）——这是真实可用产能率")
    print("=" * 72)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "depth": depth,
                "standout_cp": standout_cp,
                "equiv_cp": equiv_cp,
                "multipv_k": multipv_k,
                "min_ply": MIN_PLY,
                "max_ply": MAX_PLY,
                "min_piece_count": MIN_PIECE_COUNT,
                "m6_abs_cp_max": M6_ABS_CP_MAX,
                "min_base_seconds": MIN_BASE_SECONDS,
            },
            "summary": summary,
            "positions": records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, help=".pgn 或 .zip")
    ap.add_argument("--games", type=int, default=300, help="读取局数上限")
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--standout", type=int, default=DEFAULT_STANDOUT_CP)
    ap.add_argument("--equiv", type=int, default=DEFAULT_EQUIV_CP)
    ap.add_argument("--multipv", type=int, default=DEFAULT_MULTIPV)
    ap.add_argument("--out", default=os.path.join(
        "data", "quality_benchmark_decision", "pgn_m5_probe_result.json"))
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pgn = args.pgn if os.path.isabs(args.pgn) else os.path.join(root, args.pgn)
    out = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    run(pgn, args.games, args.depth, args.standout, args.equiv, args.multipv, out)


if __name__ == "__main__":
    main()
