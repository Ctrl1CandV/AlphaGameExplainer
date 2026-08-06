# -*- coding: utf-8 -*-
"""
generate_commentary.py — 批量生成解说词，输出到 data/ 下指定目录

用法：
  python tools/quality_audit/generate_commentary.py --which both
  python tools/quality_audit/generate_commentary.py --which endgame --no-skip
  python tools/quality_audit/generate_commentary.py --which puzzle --filter 00008

输出格式：每个样本一个 .txt，头部元信息 + 正文解说词。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common import Logger
from src.infra.llm_backend import release_backend

# 默认配置
DEFAULT_CONFIG = {
    "benchmark_dir": "data/quality_benchmark_phase_end",
    "puzzle_dir": "test_puzzles",
    "endgame_dir": "test_endgames",
}


def _compose_output(sample_id: str, sample_type: str, meta: dict,
                    elapsed: float, commentary) -> str:
    """组装输出 txt。"""
    lines = [
        "=" * 50,
        f"类型: {sample_type}",
        f"样例: {sample_id}",
        f"FEN: {meta.get('fen', '')}",
        f"主题: {meta.get('themes', '')}",
        f"难度: {meta.get('rating', '')}",
        f"状态: {'FALLBACK' if commentary.fallback_used else 'SUCCESS'}",
        f"用时: {elapsed:.1f}s",
        f"Chunks: {commentary.chunks_succeeded}/{commentary.chunks_total}"
        f" (重试{commentary.retries_total})",
        "=" * 50,
        "",
    ]
    # 正文
    if commentary.opening:
        lines.append(commentary.opening)
        lines.append("")
    if commentary.raw_text:
        lines.append(commentary.raw_text)
        lines.append("")
    if commentary.summary:
        lines.append(commentary.summary)
    return "\n".join(lines)


def _write_failure(out_path: Path, sample_id: str, sample_type: str,
                   meta: dict, elapsed: float, err: str):
    """写入失败记录。"""
    lines = [
        "=" * 50,
        f"类型: {sample_type}",
        f"样例: {sample_id}",
        f"FEN: {meta.get('fen', '')}",
        f"状态: FAILED",
        f"用时: {elapsed:.1f}s",
        f"错误: {err}",
        "=" * 50,
        "",
        "【生成失败，无解说词】",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ===== Puzzle =====
def generate_one_puzzle(json_path: Path, out_dir: Path, skip_existing: bool) -> bool:
    sample_id = json_path.stem
    out_path = out_dir / f"puzzle_{sample_id}.txt"
    if skip_existing and out_path.exists():
        Logger.info(f"[skip] {out_path.name}")
        return True

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    meta = {
        "fen": raw.get("fen", ""),
        "themes": ",".join(raw.get("themes", [])),
        "rating": str(raw.get("rating", "")),
    }
    input_text = json_path.read_text(encoding="utf-8")
    Logger.info(f"[puzzle] {sample_id} 开始")
    t0 = time.time()
    try:
        from src.pipeline.puzzle_pipeline import _run_puzzle_pipeline
        result = _run_puzzle_pipeline(input_text)
        elapsed = time.time() - t0
        if result is None:
            _write_failure(out_path, sample_id, "puzzle", meta, elapsed,
                           "pipeline returned None")
            return False
        commentary = result[0] if isinstance(result, tuple) else result
        txt = _compose_output(sample_id, "puzzle", meta, elapsed, commentary)
        out_path.write_text(txt, encoding="utf-8")
        Logger.success(f"[puzzle] {sample_id} 完成 ({elapsed:.1f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        Logger.error(f"[puzzle] {sample_id} 异常: {e}")
        traceback.print_exc()
        _write_failure(out_path, sample_id, "puzzle", meta, elapsed, str(e))
        return False


# ===== Endgame =====
def generate_one_endgame(fen_path: Path, out_dir: Path, skip_existing: bool) -> bool:
    sample_id = fen_path.stem
    out_path = out_dir / f"endgame_{sample_id}.txt"
    if skip_existing and out_path.exists():
        Logger.info(f"[skip] {out_path.name}")
        return True

    input_text = fen_path.read_text(encoding="utf-8").strip()
    meta = {"fen": input_text.split("\n")[0], "themes": "", "rating": ""}
    Logger.info(f"[endgame] {sample_id} 开始")
    t0 = time.time()
    try:
        from src.pipeline.endgame_pipeline import _run_pipeline
        result = _run_pipeline(input_text)
        elapsed = time.time() - t0
        if result is None:
            _write_failure(out_path, sample_id, "endgame", meta, elapsed,
                           "pipeline returned None")
            return False
        commentary = result[0]
        txt = _compose_output(sample_id, "endgame", meta, elapsed, commentary)
        out_path.write_text(txt, encoding="utf-8")
        Logger.success(f"[endgame] {sample_id} 完成 ({elapsed:.1f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        Logger.error(f"[endgame] {sample_id} 异常: {e}")
        traceback.print_exc()
        _write_failure(out_path, sample_id, "endgame", meta, elapsed, str(e))
        return False


# ===== 主入口 =====
def main():
    ap = argparse.ArgumentParser(description="批量生成解说词")
    ap.add_argument("--which", default="both",
                    choices=["puzzle", "endgame", "both"],
                    help="生成哪类样本")
    ap.add_argument("--no-skip", action="store_true",
                    help="不跳过已存在的输出文件")
    ap.add_argument("--filter", default="",
                    help="只处理 stem 以此前缀开头的样本")
    ap.add_argument("--outdir", default=DEFAULT_CONFIG["benchmark_dir"],
                    help="输出目录（相对项目根）")
    args = ap.parse_args()

    out_dir = (PROJECT_ROOT / args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = not args.no_skip
    total_ok, total_fail = 0, 0

    try:
        if args.which in ("puzzle", "both"):
            puzzle_dir = (PROJECT_ROOT / DEFAULT_CONFIG["puzzle_dir"]).resolve()
            for p in sorted(puzzle_dir.glob("*.json")):
                if args.filter and not p.stem.startswith(args.filter):
                    continue
                if generate_one_puzzle(p, out_dir, skip):
                    total_ok += 1
                else:
                    total_fail += 1

        if args.which in ("endgame", "both"):
            eg_dir = (PROJECT_ROOT / DEFAULT_CONFIG["endgame_dir"]).resolve()
            for p in sorted(eg_dir.glob("*.fen")):
                if args.filter and not p.stem.startswith(args.filter):
                    continue
                if generate_one_endgame(p, out_dir, skip):
                    total_ok += 1
                else:
                    total_fail += 1
    finally:
        try:
            release_backend()
        except Exception:
            pass

    Logger.info(f"[汇总] 成功 {total_ok}, 失败 {total_fail}, "
                f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
