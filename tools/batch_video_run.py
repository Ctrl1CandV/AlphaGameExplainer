# -*- coding: utf-8 -*-
"""
batch_video_run.py — 批量跑视频样本，保存视频 + 完整终端输出日志

用法：
  python tools/batch_video_run.py                  # 跑全部（endgame 10 + puzzle 10）
  python tools/batch_video_run.py --which endgame  # 只跑 endgame
  python tools/batch_video_run.py --which puzzle   # 只跑 puzzle
  python tools/batch_video_run.py --no-skip        # 强制重跑已存在的样本

输出：
  output/batch_videos/
    ├── endgame_KRvK_1.mp4        # 视频文件
    ├── endgame_KRvK_1.log        # 对应终端完整输出（含解说词预览）
    ├── puzzle_00008.mp4
    ├── puzzle_00008.log
    └── ...

日志文件包含完整终端输出（Logger + 解说词预览），供后期人工筛选解说质量。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Python 解释器路径（conda 环境）
PYTHON_EXE = r"C:\Users\LiuYiJie\.conda\envs\commentary\python.exe"

# 输出目录
OUTPUT_DIR = PROJECT_ROOT / "output" / "batch_videos"

# 视频产出固定路径（video_composer 写死）
VIDEO_OUTPUT = PROJECT_ROOT / "output" / "analysis.mp4"

# ============================================================
# 精选样本列表
# ============================================================

# Endgame 10 个（第二批）：覆盖多种类型 + 不同局面编号
ENDGAME_SAMPLES = [
    "KRvK_2",       # 单车杀王（不同局面）
    "KQvK_3",       # 单后杀王
    "KPvK_3",       # 王兵对王
    "KPPvK_2",      # 双兵对王
    "KBNvK_3",      # 马象杀王
    "KBBvK_2",      # 双象杀王
    "KQRvK_2",      # 后车杀王
    "KQvKR_3",      # 后对车
    "KRPvKR_2",     # 车兵对车
    "KRvKB_2",      # 车对象（新类型）
]

# Puzzle 10 个（第二批）：难度梯度 1211~2632 + 不同战术主题
PUZZLE_SAMPLES = [
    "00Mke",   # 1211, crushing endgame long
    "00ouE",   # 1264, attraction mate sacrifice endgame
    "00f1Y",   # 1310, discoveredAttack endgame
    "00g5H",   # 1367, deflection endgame long
    "00Myw",   # 1422, fork master middlegame
    "0000D",   # 1491, advantage endgame short
    "00HLP",   # 1500, fork middlegame
    "002rd",   # 1795, kingsideAttack pin middlegame
    "00Feu",   # 1989, fork endgame
    "002e5",   # 2632, sacrifice middlegame long
]


def run_one(sample_id: str, sample_type: str, input_path: Path,
            skip_existing: bool) -> dict:
    """运行单个样本，返回结果摘要 dict。"""
    prefix = "endgame" if sample_type == "endgame" else "puzzle"
    video_name = f"{prefix}_{sample_id}.mp4"
    log_name = f"{prefix}_{sample_id}.log"
    video_out = OUTPUT_DIR / video_name
    log_out = OUTPUT_DIR / log_name

    result = {
        "sample_id": sample_id,
        "type": sample_type,
        "video": video_name,
        "log": log_name,
        "status": "PENDING",
        "elapsed": 0.0,
    }

    # 跳过已存在
    if skip_existing and video_out.exists() and log_out.exists():
        result["status"] = "SKIPPED"
        return result

    # 构建命令
    cmd = [PYTHON_EXE, str(PROJECT_ROOT / "main.py")]
    if sample_type == "puzzle":
        cmd.append("--puzzle")
    cmd.append(str(input_path))

    print(f"\n{'='*60}")
    print(f"[{sample_type.upper()}] {sample_id} 开始生成...")
    print(f"  输入: {input_path}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        # 强制子进程以 UTF-8 输出，避免 Windows GBK 控制台编码导致中文丢失
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,  # 单样本最多 10 分钟
            env=child_env,
        )
        elapsed = time.time() - t0
        result["elapsed"] = elapsed

        # 合并 stdout + stderr 写入日志
        log_content = (
            f"{'='*60}\n"
            f"样本: {sample_id}\n"
            f"类型: {sample_type}\n"
            f"输入文件: {input_path}\n"
            f"用时: {elapsed:.1f}s\n"
            f"返回码: {proc.returncode}\n"
            f"{'='*60}\n\n"
            f"--- STDOUT ---\n{proc.stdout}\n\n"
            f"--- STDERR ---\n{proc.stderr}\n"
        )
        log_out.write_text(log_content, encoding="utf-8")

        if proc.returncode != 0:
            result["status"] = "FAILED"
            print(f"  [FAILED] 返回码 {proc.returncode}，详见 {log_name}")
            return result

        # 移动视频到输出目录
        if VIDEO_OUTPUT.exists():
            shutil.move(str(VIDEO_OUTPUT), str(video_out))
            result["status"] = "OK"
            print(f"  [OK] {elapsed:.1f}s → {video_name}")
        else:
            result["status"] = "NO_VIDEO"
            print(f"  [WARN] 进程成功但未找到视频文件 {VIDEO_OUTPUT}")

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        result["elapsed"] = elapsed
        result["status"] = "TIMEOUT"
        log_out.write_text(
            f"样本: {sample_id}\n类型: {sample_type}\n状态: TIMEOUT ({elapsed:.1f}s)\n",
            encoding="utf-8"
        )
        print(f"  [TIMEOUT] 超过 600s 限制")

    except Exception as e:
        elapsed = time.time() - t0
        result["elapsed"] = elapsed
        result["status"] = "ERROR"
        log_out.write_text(
            f"样本: {sample_id}\n类型: {sample_type}\n状态: ERROR\n异常: {e}\n",
            encoding="utf-8"
        )
        print(f"  [ERROR] {e}")

    return result


def main():
    ap = argparse.ArgumentParser(description="批量生成视频样本（含终端日志）")
    ap.add_argument("--which", default="both",
                    choices=["puzzle", "endgame", "both"],
                    help="生成哪类样本（默认 both）")
    ap.add_argument("--no-skip", action="store_true",
                    help="强制重跑已存在的样本")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    skip = not args.no_skip
    results: list[dict] = []

    print(f"批量视频生成")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"Python:   {PYTHON_EXE}")
    print(f"跳过已有: {skip}")

    # ---- Endgame ----
    if args.which in ("endgame", "both"):
        eg_dir = PROJECT_ROOT / "test_endgames"
        for sid in ENDGAME_SAMPLES:
            fen_path = eg_dir / f"{sid}.fen"
            if not fen_path.exists():
                print(f"\n[WARN] 文件不存在，跳过: {fen_path}")
                results.append({
                    "sample_id": sid, "type": "endgame",
                    "video": "", "log": "", "status": "MISSING", "elapsed": 0,
                })
                continue
            r = run_one(sid, "endgame", fen_path, skip)
            results.append(r)

    # ---- Puzzle ----
    if args.which in ("puzzle", "both"):
        pz_dir = PROJECT_ROOT / "test_puzzles"
        for sid in PUZZLE_SAMPLES:
            json_path = pz_dir / f"{sid}.json"
            if not json_path.exists():
                print(f"\n[WARN] 文件不存在，跳过: {json_path}")
                results.append({
                    "sample_id": sid, "type": "puzzle",
                    "video": "", "log": "", "status": "MISSING", "elapsed": 0,
                })
                continue
            r = run_one(sid, "puzzle", json_path, skip)
            results.append(r)

    # ---- 汇总报告 ----
    print(f"\n\n{'='*60}")
    print("批量生成汇总")
    print(f"{'='*60}")
    ok = sum(1 for r in results if r["status"] == "OK")
    failed = sum(1 for r in results if r["status"] in ("FAILED", "ERROR", "TIMEOUT"))
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")
    missing = sum(1 for r in results if r["status"] == "MISSING")
    total_time = sum(r["elapsed"] for r in results)

    for r in results:
        status_icon = {
            "OK": "✓", "SKIPPED": "→", "FAILED": "✗",
            "ERROR": "!", "TIMEOUT": "⏱", "MISSING": "?", "NO_VIDEO": "△",
        }.get(r["status"], "?")
        print(f"  [{status_icon}] {r['type']:8s} {r['sample_id']:12s} "
              f"{r['status']:8s} {r['elapsed']:.1f}s")

    print(f"\n成功 {ok} | 失败 {failed} | 跳过 {skipped} | 缺失 {missing}")
    print(f"总耗时: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"\n提示: 日志文件(.log)含完整解说词预览，可用于二次筛选。")

    # 写一份汇总文件方便后续查阅
    summary_path = OUTPUT_DIR / "_summary.txt"
    lines = [
        f"批量视频生成汇总 - {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"总耗时: {total_time:.0f}s",
        f"成功 {ok} | 失败 {failed} | 跳过 {skipped} | 缺失 {missing}",
        "",
        f"{'状态':8s} {'类型':8s} {'样本ID':12s} {'用时':8s} {'视频文件'}",
        "-" * 60,
    ]
    for r in results:
        lines.append(
            f"{r['status']:8s} {r['type']:8s} {r['sample_id']:12s} "
            f"{r['elapsed']:6.1f}s  {r['video']}"
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"汇总已写入: {summary_path}")


if __name__ == "__main__":
    main()
