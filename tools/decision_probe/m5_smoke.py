"""M5 冒烟：验证或否证「puzzle 库不是战略局面矿」（FINDINGS-002 P15）。

背景：ADR-020 原方案把 Lichess puzzle 库当中局局面矿，依据是「中局 + 非强制取胜
= 17.7%」。但那个 17.7% 测的是 `themes` 标签（含 middlegame、不含 mate/crushing），
**不是 M5 判定**。而 puzzle 库的入选原理是「解题方每步唯一最佳、次佳明显更差」，
与 M5「无强制战术」结构性冲突。

本脚本用现有 79 个 test_puzzles 样本实测 M5 存活率，回答两件事：
  1. M2（themes 口径）到 M5（引擎口径）之间的真实衰减有多陡；
  2. 换 PGN 主源的决策是否成立。

判读（FINDINGS-002 P15 §结果分支）：
  - M5 存活 <5%  → 换源决策**确认成立**，puzzle 库彻底降为辅助；
  - M5 存活 5~15% → 换源仍成立（产能太低），但 puzzle 库可留作 Tier B 补充；
  - M5 存活 >15% → 需重新评估——原方案的输入层可能仍可用，换源理由减弱。

用法（须用项目 conda 环境）：
    python -m tools.decision_probe.m5_smoke
    python -m tools.decision_probe.m5_smoke --depth 14 --standout 120
    python -m tools.decision_probe.m5_smoke --sweep     # 阈值敏感性
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import argparse
import json
import os
import sys
import time

# 允许 `python tools/decision_probe/m5_smoke.py` 直接跑（补 sys.path）
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import chess
from dotenv import load_dotenv

from tools.decision_probe.engine_probe import (
    DEFAULT_EQUIV_CP,
    DEFAULT_STANDOUT_CP,
    EngineProbe,
    assess_m5,
    equivalence_gap,
    solving_position,
)

load_dotenv()

PUZZLE_DIR = os.path.join(_ROOT, "test_puzzles")
OUT_DIR = os.path.join(_ROOT, "data", "quality_benchmark_decision")

# M3 子力窗口（ADR-020 漏斗）：总子力 ≥18 才认为子力丰富、战略成立
MIN_PIECE_COUNT = 18

# M6 评估窗口：均衡~温和优势。超出则「随便走都赢/都输」，战略差异无教学意义
M6_ABS_CP_MAX = 300


@dataclass
class SampleResult:
    puzzle_id: str
    rating: int
    themes: str
    opening_tags: str
    # 零成本闸（纯 python-chess / 字段）
    m1_middlegame: bool = False
    m2_non_forcing_theme: bool = False
    m3_material: bool = False
    piece_count: int = 0
    # 引擎闸
    m5_passed: bool = False
    m5_reason: str = ""
    m6_passed: bool = False
    m8_passed: bool = False
    m8_equiv_count: int = 0
    top1_cp: Optional[int] = None
    top2_cp: Optional[int] = None
    gap_cp: Optional[int] = None
    legal_count: int = 0
    engine_ran: bool = False
    error: str = ""


def load_samples() -> List[dict]:
    """读取 test_puzzles 下的全部样本。"""
    if not os.path.isdir(PUZZLE_DIR):
        raise FileNotFoundError(f"样本目录不存在: {PUZZLE_DIR}")
    out = []
    for name in sorted(os.listdir(PUZZLE_DIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(PUZZLE_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [skip] {name}: {e}")
            continue
        data["_id"] = os.path.splitext(name)[0]
        out.append(data)
    return out


def assess_m8(
    board: chess.Board,
    probe: EngineProbe,
    k: int,
    depth: int,
    equiv_cp: int,
) -> tuple:
    """M8：是否存在多个方向不同的近等强首着。

    返回 (passed, equiv_count)。这里只做「近等强首着数 ≥2」的粗判——
    「方向不同」需要 KB 的 direction 谓词，属 P0-lite 范畴，本冒烟不涉及。
    注意本函数与 assess_m5 共用同一个 `equivalence_gap` 定义（ADR-020 单一事实来源）。
    """
    res = probe.multipv(board, k=k, depth=depth)
    if res.best is None:
        return False, 0
    base = res.best.cp
    equiv = [ln for ln in res.lines if equivalence_gap(base, ln.cp) <= equiv_cp]
    return len(equiv) >= 2, len(equiv)


def run(depth: int, standout_cp: int, equiv_cp: int, k: int, limit: int) -> List[SampleResult]:
    sf_path = os.getenv("STOCKFISH_PATH", "")
    if sf_path and not os.path.isabs(sf_path):
        sf_path = os.path.normpath(os.path.join(_ROOT, sf_path))

    samples = load_samples()
    if limit:
        samples = samples[:limit]
    print(f"样本 {len(samples)} 个 | depth={depth} standout={standout_cp}cp "
          f"equiv={equiv_cp}cp k={k}")
    print(f"Stockfish: {sf_path}")
    print("-" * 72)

    results: List[SampleResult] = []
    t0 = time.time()

    with EngineProbe(sf_path) as probe:
        for i, s in enumerate(samples, 1):
            themes = s.get("themes", "") or ""
            theme_set = set(themes.split())
            r = SampleResult(
                puzzle_id=s.get("_id", "?"),
                rating=s.get("rating", 0),
                themes=themes,
                opening_tags=s.get("openingTags", "") or "",
            )

            # --- 零成本闸：M1 / M2 / M3 ---
            r.m1_middlegame = "middlegame" in theme_set
            r.m2_non_forcing_theme = not (theme_set & {"mate", "crushing"})

            uci_moves = (s.get("moves", "") or "").split()
            board = solving_position(s.get("fen", ""), uci_moves)
            if board is None:
                r.error = "FEN/预备着解析失败"
                results.append(r)
                print(f"[{i:3d}/{len(samples)}] {r.puzzle_id:8s} ERROR {r.error}")
                continue

            r.piece_count = len(board.piece_map())
            r.m3_material = r.piece_count >= MIN_PIECE_COUNT

            # 只对过了零成本闸的样本付引擎成本（ADR-020「便宜的先跑」）
            if not (r.m1_middlegame and r.m2_non_forcing_theme and r.m3_material):
                results.append(r)
                gate = ("M1" if not r.m1_middlegame else
                        "M2" if not r.m2_non_forcing_theme else "M3")
                print(f"[{i:3d}/{len(samples)}] {r.puzzle_id:8s} 止于 {gate}"
                      f"（子力 {r.piece_count}）")
                continue

            # --- 引擎闸：M5 / M6 / M8 ---
            try:
                v = assess_m5(board, probe, k=k, depth=depth, standout_cp=standout_cp)
                r.engine_ran = True
                r.m5_passed = v.passed
                r.m5_reason = v.reason
                r.top1_cp, r.top2_cp, r.gap_cp = v.top1_cp, v.top2_cp, v.gap_cp
                r.legal_count = v.legal_count

                if v.top1_cp is not None:
                    r.m6_passed = abs(v.top1_cp) <= M6_ABS_CP_MAX

                if r.m5_passed:
                    r.m8_passed, r.m8_equiv_count = assess_m8(
                        board, probe, k=k, depth=depth, equiv_cp=equiv_cp)
            except Exception as e:
                r.error = f"{type(e).__name__}: {e}"

            results.append(r)
            mark = "PASS" if r.m5_passed else "fail"
            extra = f" M8={'Y' if r.m8_passed else 'n'}({r.m8_equiv_count})" if r.m5_passed else ""
            print(f"[{i:3d}/{len(samples)}] {r.puzzle_id:8s} M5={mark} "
                  f"{r.m5_reason}{extra}")

    print("-" * 72)
    print(f"耗时 {time.time() - t0:.1f}s")
    return results


def summarize(results: List[SampleResult], standout_cp: int) -> Dict:
    n = len(results)
    m1 = [r for r in results if r.m1_middlegame]
    m2 = [r for r in m1 if r.m2_non_forcing_theme]
    m3 = [r for r in m2 if r.m3_material]
    ran = [r for r in m3 if r.engine_ran]
    m5 = [r for r in ran if r.m5_passed]
    m6 = [r for r in m5 if r.m6_passed]
    m8 = [r for r in m6 if r.m8_passed]

    def pct(x: int) -> str:
        return f"{x / n * 100:.1f}%" if n else "n/a"

    print()
    print("=" * 72)
    print("漏斗逐级存活（分母 = 全部样本）")
    print("=" * 72)
    rows = [
        ("总样本", n),
        ("M1 中局标签", len(m1)),
        ("M2 非强制取胜标签", len(m2)),
        ("M3 子力 ≥18", len(m3)),
        ("  （引擎实跑）", len(ran)),
        (f"M5 无强制战术（standout={standout_cp}cp）", len(m5)),
        ("M6 评估窗口内", len(m6)),
        ("M8 近等强首着 ≥2", len(m8)),
    ]
    for label, cnt in rows:
        print(f"  {label:36s} {cnt:4d}   {pct(cnt)}")

    # M2→M5 衰减是本次冒烟的核心数字：它量化了「themes 口径」与「引擎口径」的差距
    print()
    if m2:
        decay = (1 - len(m5) / len(m2)) * 100
        print(f"  M2 → M5 衰减: {len(m2)} → {len(m5)}（掉 {decay:.1f}%）")
        print(f"  ↑ 这是 FINDINGS-002 P15 的核心量：17.7% 那个数字止于 M2 口径")
    if m5:
        print(f"  M5 幸存者中 M8 命中: {len(m8)}/{len(m5)}")

    # M5 失败原因分布——看衰减主要发生在哪一类
    print()
    print("M5 未通过原因分布：")
    buckets: Dict[str, int] = {}
    for r in ran:
        if r.m5_passed:
            continue
        key = r.m5_reason.split("（")[0].split("（")[0]
        if "唯一好着" in key:
            key = "存在唯一好着"
        elif "将杀线" in key:
            key = "存在将杀线"
        buckets[key] = buckets.get(key, 0) + 1
    for key, cnt in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"  {key:36s} {cnt:4d}")

    # gap 分布：直接决定 standout 阈值该定在哪
    gaps = sorted(r.gap_cp for r in ran if r.gap_cp is not None)
    if gaps:
        print()
        print("首选-次选 gap 分布（cp，用于校准 standout 阈值）：")
        for q, label in ((0.10, "P10"), (0.25, "P25"), (0.50, "中位"),
                         (0.75, "P75"), (0.90, "P90")):
            idx = min(int(len(gaps) * q), len(gaps) - 1)
            print(f"  {label:6s} {gaps[idx]:6d}")
        print(f"  最小/最大 {gaps[0]} / {gaps[-1]}")

    # 走向判读
    survival = len(m5) / n * 100 if n else 0.0
    print()
    print("=" * 72)
    if survival < 5:
        verdict = "换源决策确认成立——puzzle 库降为辅助，PGN 主源为唯一可行路径"
    elif survival < 15:
        verdict = "换源仍成立（产能过低），puzzle 库可留作 Tier B 补充"
    else:
        verdict = "需重新评估——原输入层可能仍可用，换源理由减弱"
    print(f"M5 存活率 {survival:.1f}% → {verdict}")
    print("=" * 72)

    return {
        "sample_count": n,
        "m1_middlegame": len(m1),
        "m2_non_forcing_theme": len(m2),
        "m3_material": len(m3),
        "engine_ran": len(ran),
        "m5_passed": len(m5),
        "m6_passed": len(m6),
        "m8_passed": len(m8),
        "m5_survival_pct": round(survival, 2),
        "m2_to_m5_decay_pct": round((1 - len(m5) / len(m2)) * 100, 2) if m2 else None,
        "m5_fail_buckets": buckets,
        "gap_percentiles": {
            "p10": gaps[min(int(len(gaps) * 0.10), len(gaps) - 1)] if gaps else None,
            "p50": gaps[min(int(len(gaps) * 0.50), len(gaps) - 1)] if gaps else None,
            "p90": gaps[min(int(len(gaps) * 0.90), len(gaps) - 1)] if gaps else None,
        },
        "verdict": verdict,
    }


def sweep(results: List[SampleResult]) -> None:
    """阈值敏感性：standout 取不同值时 M5 存活率如何变化。

    复用已有的 gap 数据离线重算，不重跑引擎——所以这是零成本的。
    """
    ran = [r for r in results if r.engine_ran]
    print()
    print("=" * 72)
    print("standout 阈值敏感性（离线重算，不重跑引擎）")
    print("=" * 72)
    print(f"  {'standout(cp)':>14s} {'M5 通过':>10s} {'占全样本':>10s}")
    n = len(results)
    for th in (80, 100, 120, 150, 200, 250, 300):
        # 复现 assess_m5 的判定链：终局/将军/将杀/单着 一律不通过，与阈值无关
        passed = sum(
            1 for r in ran
            if r.gap_cp is not None and r.gap_cp < th
            and "将杀" not in r.m5_reason and "将军" not in r.m5_reason
        )
        print(f"  {th:14d} {passed:10d} {passed / n * 100:9.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description="M5 冒烟（FINDINGS-002 P15 走向闸门）")
    ap.add_argument("--depth", type=int, default=16, help="MultiPV 搜索深度")
    ap.add_argument("--standout", type=int, default=DEFAULT_STANDOUT_CP,
                    help="唯一好着阈值 cp（首选优于次选此值即判有强制战术）")
    ap.add_argument("--equiv", type=int, default=DEFAULT_EQUIV_CP,
                    help="M8 近等强阈值 cp")
    ap.add_argument("--k", type=int, default=5, help="MultiPV 条数")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个（调试用）")
    ap.add_argument("--sweep", action="store_true", help="额外输出阈值敏感性")
    ap.add_argument("--out", default="", help="结果 JSON 路径（默认写入 benchmark 目录）")
    args = ap.parse_args()

    try:
        results = run(args.depth, args.standout, args.equiv, args.k, args.limit)
    except FileNotFoundError as e:
        print(f"[fatal] {e}", file=sys.stderr)
        return 2

    summary = summarize(results, args.standout)
    if args.sweep:
        sweep(results)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(OUT_DIR, "m5_smoke_result.json")
    payload = {
        "config": {
            "depth": args.depth,
            "standout_cp": args.standout,
            "equiv_cp": args.equiv,
            "multipv_k": args.k,
            "min_piece_count": MIN_PIECE_COUNT,
            "m6_abs_cp_max": M6_ABS_CP_MAX,
        },
        "summary": summary,
        "samples": [asdict(r) for r in results],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
