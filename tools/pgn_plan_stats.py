"""P17 统计对照设施：PGN 计划频率统计（免标注 ground truth）。

定位（FINDINGS-002 P17）：
- A1（正确执行着）与 B2（候选计划真实性）的 ground truth 采用「PGN 实战频率统计」，
  本设施从 PGN 库统计「到达某兵形后，走子方实际采用了哪些着法、各占多少比例」。
- 打破循环依赖：统计用「开局着法序列锚定」（不依赖 structure_id——B1 待验的模块
  不能给自己造 ground truth）。
- 频率是「棋手实际会考虑什么」的证据，不是「什么最优」的证据。
- 计划层面聚合由人工完成：看频率表 top-10 着法手工归入计划桶（6 原型 × 一次性，
  同时是 KB 编写的输入，一份工作两用）。

自检（P17 自身必须先验证）：先跑已知答案的原型——卡尔斯巴德白方应以
「少数派攻击（b4/a4 系列）」为主、「中心突破（e4）」为次。统计口径正确才可用
它做 A1/B2 的判据。

时限筛（P24 已裁决）：读 Event 头官方分类（Rated Rapid/Classical game 及其
tournament/swiss 变体），**不用 TimeControl 秒数分桶**（180+0 被官方标为 Blitz，
秒数口径与官方口径产能差 15 倍）。

用法：
    "C:\\Users\\LiuYiJie\\.conda\\envs\\commentary\\python.exe" -m tools.pgn_plan_stats \\
        --pgn data/pgn/lichess_elite_2025-11.pgn --anchor carlsbad
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional

import chess
import chess.pgn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.decision_probe.pgn_m5_probe import open_pgn_stream  # noqa: E402


# ---------------------------------------------------------------- 时限筛（P24）

def is_rapid_or_classical(event: str) -> bool:
    """P24 口径：以 lichess 官方 Event 头分类为准，只保留 rapid / classical。"""
    return ("Rapid" in event) or ("Classical" in event)


# ---------------------------------------------------------------- 锚定定义

# 卡尔斯巴德：QGD 兑变 1.d4 d5 2.c4 e6 3.Nc3 Nf6 4.cxd5 exd5（8 半着）
# 锚定后轮到白方（ply 9 = 白方第 5 着）。
CARLSBAD_PREFIX = ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "cxd5", "exd5"]

ANCHORS: Dict[str, dict] = {
    "carlsbad": {
        "cn": "卡尔斯巴德结构",
        "expect": "carlsbad",  # B1：锚定局面应被 structure_id 识别为该原型
        "prefix_san": CARLSBAD_PREFIX,
        # 锚定后观察多少半着内的「计划承诺着」（兵突破/进攻性兵着）。
        # 开发性兵着（e3/g3/h3）不在集合内——它们不是计划承诺。
        # 卡尔斯巴德的 b4 少数派攻击通常在第 11~13 回合（ply 22~26）才启动，
        # 窗口须覆盖到 ply 28（第 14 回合）左右。
        "plan_window_plies": 20,
        "plan_families": {
            "后翼突破(少数派攻击方向)": {"b4", "b5", "a4", "a3"},
            "中心突破": {"e4", "e5"},
        },
    },
    # IQP 两条到达线（B1 ground truth）：锚定后白方 d4 兵孤立（c/e 线无白兵）。
    # 注意 SAN 消歧陷阱（P17 教训的又一例）：a1d1 的 SAN 是 "Rad1" 不是 "Rd1"
    # （f1 车同在 rank 1 可到 d1）。前缀一律用 python-chess 逐步核对后再写入。
    # 帕诺夫（卡罗康）主线：1.e4 c6 2.d4 d5 3.exd5 cxd5 4.c4 Nf6 5.Nc3 e6
    #                   6.Nf3 Be7 7.Bd3 dxc4 8.Bxc4（15 半着，白方 d4 已孤立）
    "iqp_panov": {
        "cn": "孤后兵结构（帕诺夫）",
        "expect": "iqp",
        "prefix_san": ["e4", "c6", "d4", "d5", "exd5", "cxd5", "c4", "Nf6",
                       "Nc3", "e6", "Nf3", "Be7", "Bd3", "dxc4", "Bxc4"],
        "plan_window_plies": 12,
        "plan_families": {
            "推进消除孤兵": {"d5"},
            "保持孤兵": {"e4", "e5"},
        },
    },
    # QGA 主线：1.d4 d5 2.c4 dxc4 3.Nf3 Nf6 4.e3 e6 5.Bxc4 c5 6.O-O a6
    #           7.Qe2 Nc6 8.Rad1 cxd4 9.exd4（17 半着）
    "iqp_qga": {
        "cn": "孤后兵结构（QGA 主线）",
        "expect": "iqp",
        "prefix_san": ["d4", "d5", "c4", "dxc4", "Nf3", "Nf6", "e3", "e6",
                       "Bxc4", "c5", "O-O", "a6", "Qe2", "Nc6", "Rad1",
                       "cxd4", "exd4"],
        "plan_window_plies": 12,
        "plan_families": {
            "推进消除孤兵": {"d5"},
            "保持孤兵": {"e4", "e5"},
        },
    },
    # 尼姆佐维奇 4.e3（白方 IQP），两条到达线：
    #   A: 7...Nc6 8.a3 cxd4 9.exd4 dxc4 10.Bxc4
    #   B: 7...dxc4 8.Bxc4 Nc6 9.a3 cxd4 10.exd4
    "iqp_nimzo": {
        "cn": "孤后兵结构（尼姆佐维奇 4.e3）",
        "expect": "iqp",
        "prefix_san": ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3", "O-O",
                       "Bd3", "d5", "Nf3", "c5", "O-O", "Nc6", "a3",
                       "cxd4", "exd4", "dxc4", "Bxc4"],
        "prefix_variants": [
            ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "e3", "O-O", "Bd3",
             "d5", "Nf3", "c5", "O-O", "dxc4", "Bxc4", "Nc6", "a3",
             "cxd4", "exd4"],
        ],
        "plan_window_plies": 12,
        "plan_families": {
            "推进消除孤兵": {"d5"},
            "保持孤兵": {"e4", "e5"},
        },
    },
    # 半塔拉什（对方黑方持孤兵 d5，白方施压）——实测数据中的主流 IQP 到达线：
    #   1.d4 Nf6 2.Nf3 d5 3.c4 e6 4.Nc3 c5 5.cxd5 cxd4 6.Qa4+ Bd7 7.Qxd4 exd5
    "iqp_semi_tarrasch": {
        "cn": "孤后兵结构（半塔拉什）",
        "expect": "iqp",
        "prefix_san": ["d4", "Nf6", "Nf3", "d5", "c4", "e6", "Nc3", "c5",
                       "cxd5", "cxd4", "Qa4", "Bd7", "Qxd4", "exd5"],
        "plan_window_plies": 12,
        "plan_families": {
            "推进消除孤兵": {"d5"},
            "保持孤兵": {"e4", "e5"},
        },
    },
    # 拉戈津（白方持孤兵 d4）：1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.Nf3 d5
    #   5.cxd5 exd5 6.Bg5 h6 7.Bh4 c5 8.e3 Nc6 9.Bd3 cxd4 10.exd4
    "iqp_ragozin": {
        "cn": "孤后兵结构（拉戈津）",
        "expect": "iqp",
        "prefix_san": ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4", "Nf3", "d5",
                       "cxd5", "exd5", "Bg5", "h6", "Bh4", "c5", "e3",
                       "Nc6", "Bd3", "cxd4", "exd4"],
        "plan_window_plies": 12,
        "plan_families": {
            "推进消除孤兵": {"d5"},
            "保持孤兵": {"e4", "e5"},
        },
    },
}


def _match_prefix(game, prefix_san: List[str]):
    """按 SAN 序列匹配开局前缀，命中返回 (锚定位置棋盘, 最后一个前缀着节点)。

    续走收集必须从锚定节点继续（`game.next()` 恒返回首着，不是锚定之后）。
    将军/将杀后缀（+/#）在比较前剥离——SAN 生成器会附加，前缀书写不写都行。
    """
    board = game.board()
    node = game
    for expected in prefix_san:
        node = node.next()          # 游标推进（game.next() 恒返回首子节点，不可复用）
        if node is None or node.move is None:
            return None
        try:
            san = board.san(node.move).rstrip("+#")
        except Exception:
            return None
        if san != expected:
            return None
        board.push(node.move)
    return board, node


def _plan_family(move_san: Optional[str], anchor: dict) -> str:
    """把窗口内首个计划承诺着（SAN 着名）归入计划分桶；无则「未定型」。"""
    if not move_san:
        return "未定型(窗口内无计划承诺着)"
    for family, moves in anchor["plan_families"].items():
        if move_san in moves:
            return family
    return "其它兵着"


# ---------------------------------------------------------------- 采样


def _iter_kept_games(handle):
    """Event 头预筛 + 保留局 seek 回解析。

    `read_headers` 的语义是「读 header 并**跳过整局 movetext**」——所以保留局
    必须先用 `handle.tell()` 记偏移、读完后 `seek` 回去再 `read_game`（python-chess
    官方模式）。全量 read_game 解析 28 万局 movetext 约 8 分钟，而 93.6% 是 blitz，
    预筛后只解析 17,989 局，约 1 分钟。
    """
    if not handle.seekable():
        # zip 流等不可 seek 场景：退化为全量 read_game（慢路径，仅调试用）
        while True:
            try:
                game = chess.pgn.read_game(handle)
            except Exception:
                continue
            if game is None:
                return
            if is_rapid_or_classical(game.headers.get("Event", "")):
                yield game
        return
    while True:
        offset = handle.tell()
        headers = chess.pgn.read_headers(handle)
        if headers is None:
            return
        if not is_rapid_or_classical(headers.get("Event", "")):
            continue  # movetext 已被 read_headers 跳过
        handle.seek(offset)
        try:
            game = chess.pgn.read_game(handle)
        except Exception:
            continue
        if game is not None:
            yield game


def collect_samples(
    pgn_path: str,
    anchor: dict,
    max_games: int = 0,
    max_samples: int = 2000,
) -> tuple[List[dict], Counter]:
    """流式读 PGN，收集所有到达锚定结构的位置样本。

    返回 (samples, stats)。stats 记录各过滤环节的淘汰数（透明度/校准用）。
    """
    samples: List[dict] = []
    stats: Counter = Counter()
    prefixes = [anchor["prefix_san"]] + anchor.get("prefix_variants", [])
    window = anchor["plan_window_plies"]

    handle = open_pgn_stream(pgn_path)
    games_read = 0
    try:
        for game in _iter_kept_games(handle):
            if max_games and games_read >= max_games:
                break
            games_read += 1

            matched = None
            matched_len = 0
            for prefix in prefixes:
                m = _match_prefix(game, prefix)
                if m is not None:
                    matched, matched_len = m, len(prefix)
                    break
            if matched is None:
                stats["prefix_miss"] += 1
                continue
            board, anchor_node = matched

            # 锚定后的实战续走（用于计划承诺着判定）
            cont: List[str] = []
            temp = board.copy()
            nxt = anchor_node.next()
            first_uci = None
            first_san = None
            plan_san = None
            family_moves = {m for fam in anchor["plan_families"].values()
                            for m in fam}
            while nxt is not None and len(cont) < window:
                mv = nxt.move
                nxt = nxt.next()
                if mv is None:
                    break
                try:
                    san = temp.san(mv)   # SAN 着名（"b4"/"a4"），与分桶集合同语义
                except Exception:
                    san = ""
                if len(cont) == 0:
                    first_uci = mv.uci()
                    first_san = san
                # 计划承诺着只统计锚定走子方的着（偶数索引 = 锚定方）
                if plan_san is None and len(cont) % 2 == 0 and san in family_moves:
                    plan_san = san
                try:
                    temp.push(mv)
                except Exception:
                    break
                cont.append(mv.uci())

            samples.append({
                "fen": board.fen(),
                "ply": matched_len,
                "mover_color": "white" if board.turn == chess.WHITE else "black",
                "mover_move_uci": first_uci,
                "mover_move_san": first_san or "",
                "continuation_uci": cont,
                "first_plan_move_san": plan_san,
                "first_plan_family": _plan_family(plan_san, anchor),
                "event": game.headers.get("Event", ""),
                "eco": game.headers.get("ECO", ""),
                "opening": game.headers.get("Opening", ""),
                "url": game.headers.get("LichessURL", "")
                        or game.headers.get("Site", ""),
            })
            stats["sampled"] += 1
            if max_samples and len(samples) >= max_samples:
                break
    finally:
        handle.close()
    stats["games_read"] = games_read
    return samples, stats


# ---------------------------------------------------------------- 主流程


def run(pgn_path: str, anchor_name: str, max_games: int,
        max_samples: int, out_path: str) -> dict:
    anchor = ANCHORS.get(anchor_name)
    if anchor is None:
        print(f"未知锚定 {anchor_name!r}，可用：{sorted(ANCHORS)}", file=sys.stderr)
        sys.exit(2)

    print(f"PGN: {pgn_path}")
    print(f"锚定: {anchor['cn']}（前缀 {len(anchor['prefix_san'])} 半着，"
          f"计划窗口 {anchor['plan_window_plies']} 半着）")
    print(f"时限筛: Event 头 rapid/classical only（P24 口径）")
    print("-" * 72)

    t0 = time.time()
    samples, stats = collect_samples(pgn_path, anchor, max_games, max_samples)
    elapsed = time.time() - t0

    print(f"读入 {stats['games_read']} 局，{elapsed:.1f}s，"
          f"锚定命中 {len(samples)} 局")
    for k, v in stats.most_common():
        if k != "games_read":
            print(f"    {k:32s} {v}")
    print("-" * 72)

    if len(samples) < 30:
        print(f"⚠️ 样本量 {len(samples)} < 30——按 P17 判据标记「样本不足」，"
              f"该原型的统计判据降级为人工抽查（可加月份数据，勿放宽时限筛）")
        verdict = "样本不足"
    else:
        # 立即选择频率表（锚定位置走子方实际走的着）
        imm: Counter = Counter()
        for s in samples:
            if s["mover_move_san"]:
                imm[s["mover_move_san"]] += 1
        n = len(samples)
        print(f"锚定位置走子方（{'白' if samples[0]['mover_color']=='white' else '黑'}方）"
              f"立即选择 top-10：")
        for san, c in imm.most_common(10):
            print(f"    {san:6s} {c:4d}  {100.0 * c / n:5.1f}%")

        # 计划分桶（自检判据：少数派攻击为主流 + 中心突破为次）
        fam: Counter = Counter(s["first_plan_family"] for s in samples)
        print(f"\n计划承诺着分桶（锚定后 {anchor['plan_window_plies']} 半着内首个兵突破）：")
        for k, c in fam.most_common():
            print(f"    {k:34s} {c:4d}  {100.0 * c / n:5.1f}%")

        minority = fam.get("后翼突破(少数派攻击方向)", 0)
        center = fam.get("中心突破", 0)
        print()
        print("=" * 72)
        if minority > center and minority / n >= 0.25:
            print(f"自检通过：少数派攻击方向 {minority/n:.0%} > 中心突破 "
                  f"{center/n:.0%}——与文献结论（卡尔斯巴德白方以少数派攻击为主流、"
                  f"中心突破为备选）一致，统计口径可信")
            verdict = "自检通过"
        else:
            print(f"⚠️ 自检未复现预期：少数派方向 {minority/n:.0%} vs 中心 "
                  f"{center/n:.0%}——检查锚定前缀或分桶集合（不排除结论偏差）")
            verdict = "自检未复现"
        print("=" * 72)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "anchor": anchor_name,
                "cn": anchor["cn"],
                "prefix_san": anchor["prefix_san"],
                "plan_window_plies": anchor["plan_window_plies"],
                "plan_families": {
                    k: sorted(v) for k, v in anchor["plan_families"].items()
                },
                "time_filter": "Event 头 rapid/classical（P24 口径）",
                "max_games": max_games,
                "max_samples": max_samples,
            },
            "summary": {
                "games_read": stats["games_read"],
                "samples": len(samples),
                "stats": dict(stats),
                "verdict": verdict,
                "elapsed_s": round(elapsed, 1),
            },
            "samples": samples,
        }, f, ensure_ascii=False, indent=1)
    print(f"\n结果已写入 {out_path}")
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser(description="P17 PGN 计划频率统计（免标注 ground truth）")
    ap.add_argument("--pgn", required=True, help=".pgn 或 .zip")
    ap.add_argument("--anchor", required=True, choices=sorted(ANCHORS),
                    help="锚定原型")
    ap.add_argument("--games", type=int, default=0, help="读取局数上限（0=全部）")
    ap.add_argument("--max-samples", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(
        "data", "quality_benchmark_decision", "p17_stats_{anchor}.json"))
    args = ap.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pgn = args.pgn if os.path.isabs(args.pgn) else os.path.join(root, args.pgn)
    out = args.out if os.path.isabs(args.out) else os.path.join(
        root, args.out.format(anchor=args.anchor))
    run(pgn, args.anchor, args.games, args.max_samples, out)


if __name__ == "__main__":
    main()
