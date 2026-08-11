"""批量跑演示片单，产出一组交差用的 decision 视频。

读 demo_playlist.json 片单 → 逐个跑 run_decision_video（完整管线：识别→
引擎→storyboard→LLM 解说→TTS→渲染→合成）→ 把 mp4 + sidecar 复制到
output/demo_<timestamp>/ → 打印汇总报告（成功/放弃/崩溃分桶）。

用法：
    python -m tools.decision_probe.run_demo_batch
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

PLAYLIST_PATH = os.path.join(_ROOT, "tools", "decision_probe", "demo_playlist.json")


def main():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
    from src.pipeline.decision_pipeline import run_decision_video

    playlist = json.load(open(PLAYLIST_PATH, encoding="utf-8"))
    items = playlist["playlist"]
    n = len(items)

    # 输出到带时间戳的演示目录，避免覆盖既有 output 文件
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    demo_dir = os.path.join(_ROOT, "output", f"demo_{ts}")
    os.makedirs(demo_dir, exist_ok=True)
    print(f"演示输出目录: {demo_dir}")
    print(f"片单: {n} 个局面\n")

    results = []
    t0 = time.time()

    for i, item in enumerate(items):
        arch = item["archetype"]
        fen = item["fen"]
        url = item.get("url", "")
        plans = item.get("plans", [])
        print("=" * 60)
        print(f"[{i+1}/{n}] {arch} | {plans}")
        print(f"  fen: {fen}")
        print(f"  url: {url}")
        print("=" * 60)

        ti = time.time()
        try:
            video_path = run_decision_video(fen)
        except Exception as e:
            # 崩溃单独记（不应发生，但兜底）
            results.append({"arch": arch, "fen": fen, "url": url,
                            "status": "crash",
                            "error": f"{type(e).__name__}: {e}"[:200],
                            "elapsed": round(time.time() - ti, 0)})
            print(f"  ❌ 崩溃: {type(e).__name__}: {e}\n")
            continue

        if not video_path:
            # SPEC §8 优雅放弃（无原型/不在产品池/无可行计划/解说 aborted）
            results.append({"arch": arch, "fen": fen, "url": url,
                            "status": "abandoned",
                            "elapsed": round(time.time() - ti, 0)})
            print(f"  ⚠ 管线级放弃（SPEC §8，LLM 措辞层偶发，可重跑）\n")
            continue

        # 成功：把 mp4 + sidecar 复制到演示目录，加原型前缀便于排序
        base = os.path.basename(video_path)
        sidecar = os.path.splitext(video_path)[0] + "_review.json"
        dest_mp4 = os.path.join(demo_dir, f"{arch}_{base}")
        dest_json = os.path.join(demo_dir, f"{arch}_{os.path.basename(sidecar)}")
        shutil.copy2(video_path, dest_mp4)
        if os.path.isfile(sidecar):
            shutil.copy2(sidecar, dest_json)

        # 从 sidecar 取四维关键数据写进报告
        four_dim = {}
        try:
            sc = json.load(open(dest_json, encoding="utf-8"))
            divs = sc.get("divergences", [])
            four_dim = {
                "routes": len(sc.get("routes", [])),
                "paired": divs[0].get("paired") if divs else None,
                "divergence_depth": divs[0].get("divergence_depth") if divs else None,
                "axis_type": sc.get("comparison_axes", {}).get("axis_type"),
                "seg_count": len(sc.get("segments", [])),
            }
        except Exception:
            pass

        results.append({"arch": arch, "fen": fen, "url": url,
                        "status": "ok", "video": dest_mp4,
                        "sidecar": dest_json,
                        "four_dim": four_dim,
                        "elapsed": round(time.time() - ti, 0)})
        print(f"  ✅ 出片: {os.path.basename(dest_mp4)} "
              f"({four_dim}) {results[-1]['elapsed']:.0f}s\n")

    # 汇总报告
    total = time.time() - t0
    ok = [r for r in results if r["status"] == "ok"]
    abandoned = [r for r in results if r["status"] == "abandoned"]
    crash = [r for r in results if r["status"] == "crash"]

    print("\n" + "=" * 60)
    print(f"批量演示完成（{total:.0f}s = {total/60:.1f}min）")
    print(f"  出片 {len(ok)} / 放弃 {len(abandoned)} / 崩溃 {len(crash)}")
    print("=" * 60)
    for r in results:
        mark = {"ok": "✅", "abandoned": "⚠", "crash": "❌"}[r["status"]]
        fd = r.get("four_dim", {})
        extra = f" paired={fd.get('paired')} depth={fd.get('divergence_depth')}" if fd else ""
        print(f"  {mark} {r['arch']:10s} {r['status']:9s}{extra}")
        if r["status"] == "ok":
            print(f"      → {os.path.basename(r['video'])}")
    print(f"\n演示目录: {demo_dir}")

    # 落一份报告 JSON 到演示目录
    report = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
              "demo_dir": demo_dir, "total_sec": round(total, 0),
              "ok": len(ok), "abandoned": len(abandoned), "crash": len(crash),
              "results": results}
    with open(os.path.join(demo_dir, "_batch_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"报告: {os.path.join(demo_dir, '_batch_report.json')}")

    return 0 if not crash else 1


if __name__ == "__main__":
    sys.exit(main())
