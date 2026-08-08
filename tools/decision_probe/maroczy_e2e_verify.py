# -*- coding: utf-8 -*-
"""maroczy KB goal 修复的端到端验证（PLAN-010 验收 #4 的最终确认）。

阶段 2 已在探针层证实两计划走不同进步维；本脚本跑**完整决策管线**
（引擎 + LLM + TTS + 视频），确认机制闸在 maroczy 决策点真实放行两条计划、
产出比较式叙事（routes=2、goal_ok 均 True、unique_facts 非空）——即 KB 修复
让 maroczy 从 PLAN-009 的"单线退化"恢复为"双计划对比"。

用法：conda run -n explainer python tools/decision_probe/maroczy_e2e_verify.py
"""
import json
import os
import sys
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(_ROOT, ".env"))

# 决策点 FEN 同 structure_id / a1_recall 自检（白 c4+e4 马洛齐束缚，白走子）
MAROCZY_FEN = "r2q1rk1/pp2ppbp/3pbnp1/8/2P1P3/2N1B3/PP1QBPPP/R3K2R w KQ - 5 11"


def main():
    from src.pipeline.decision_pipeline import _run_decision_pipeline

    out_dir = tempfile.mkdtemp(prefix="maroczy_e2e_")
    print(f"maroczy 决策点端到端验证 | output_dir={out_dir}")
    print(f"FEN: {MAROCZY_FEN}")
    print("-" * 72)

    output_path = _run_decision_pipeline(MAROCZY_FEN, output_dir=out_dir)

    # sidecar 命名与 decision_pipeline / test_decision_smoke 同口径：
    # <video_stem>_review.json
    sidecar = (os.path.splitext(output_path)[0] + "_review.json"
               if output_path else "")
    if not (sidecar and os.path.isfile(sidecar)):
        cands = [os.path.join(out_dir, f) for f in os.listdir(out_dir)
                 if f.endswith(".json")]
        sidecar = cands[0] if cands else ""

    print(f"视频输出: {output_path or '(无)'}")
    if not os.path.isfile(sidecar):
        print("!! 无 sidecar，无法核对 routes")
        sys.exit(1)

    r = json.load(open(sidecar, encoding="utf-8"))
    print(f"archetype      = {r.get('archetype')}")
    print(f"routes 数      = {len(r.get('routes', []))}")
    print(f"axis_type      = {r.get('comparison_axes', {}).get('axis_type')}")
    ok = True
    if r.get("archetype") != "maroczy":
        print("!! archetype 非 maroczy")
        ok = False
    if len(r.get("routes", [])) != 2:
        print(f"!! routes={len(r.get('routes', []))}，期望 2（KB 修复目标：双计划）")
        ok = False
    for i, rt in enumerate(r.get("routes", [])):
        goal_ok = rt.get("goal_ok")
        uf = len(rt.get("unique_facts", []) or [])
        plan = rt.get("plan", {}).get("name") or rt.get("plan_name")
        print(f"  route[{i}] plan={plan} goal_ok={goal_ok} unique_facts={uf}")
        if goal_ok is not True:
            print(f"    !! route[{i}] goal_ok 非 True")
            ok = False
        if uf == 0:
            print(f"    !! route[{i}] unique_facts 为空（对比不具体）")
            ok = False

    print("-" * 72)
    print("结论：" + ("✓ maroczy 双计划对比成立，KB 修复端到端生效" if ok
                      else "✗ 未达成双计划对比，需排查"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
