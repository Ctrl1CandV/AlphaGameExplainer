"""decision 管线自身回归资产（PLAN-010 阶段0 步骤6）。

老管线夹具（test_endgame_pipeline_smoke / test_puzzle_pipeline_smoke）护不住
decision 管线的行为——机制闸、goal 判定、routes 数全部是 decision 管线专属逻辑，
老管线完全不涉及。这是本 PLAN 阶段2（KB goal 修复）改动前必须先建好的护栏。

断言基准取自 `tests/fixtures/decision_hanging_baseline.json`——由本轮真实
--text 端到端运行产出的 sidecar（悬兵局面，两条计划：推进悬兵/保持悬兵，
均通过机制闸），已固化进 tests/fixtures/（不依赖 .gitignore 排除的 output/
目录是否存在）。

**范围偏差说明**（如实记录，不隐藏）：PLAN 步骤6原文写"不出视频，省时间"，
但 `_run_decision_pipeline` 是单一函数，没有独立的纯文本计算入口——
engine 计算 → 解说生成 → TTS → 视频渲染全部耦合在一次调用里，中途没有
"计算完就返回"的分支。拆出一条独立的文本级路径需要改动 decision_pipeline.py
本身（超出阶段0"引入测试设施"的范围，且会制造一条与生产路径不同的旁路，
正是本 PLAN 反复强调要避免的"单一事实来源"分裂）。故本测试改为调用真实的
`_run_decision_pipeline`（不传 output_dir，跳过 sidecar 落盘与视频文件命名），
仍然走完整链路（含 TTS 与视频合成），标记为 @pytest.mark.slow。
换来的收益：测试永远验证生产代码的真实行为，不会因为测试专用旁路而与
生产链路口径脱节——这与 PLAN 对挖掘器"禁止自写判定逻辑"的单一事实来源
要求是同一原则的延伸。
"""
import json
import os

import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
BASELINE_PATH = os.path.join(FIXTURES_DIR, "decision_hanging_baseline.json")

HANGING_FEN = "2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/1R2R1K1 w - - 1 21"


def _baseline():
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_baseline_fixture_itself_is_well_formed():
    """基准文件本身的结构断言（纯文件校验，无引擎依赖，秒级）。

    这条不标 slow——它保护的是"基准文件没有被意外改坏/截断"，
    与下面调引擎的真实回归测试互补。
    """
    baseline = _baseline()
    assert baseline["archetype"] == "hanging"
    assert len(baseline["routes"]) == 2
    assert baseline["comparison_axes"]["axis_type"] == 1
    for route in baseline["routes"]:
        assert route["goal_ok"] is True
        assert len(route["unique_facts"]) > 0
    assert len(baseline["segments"]) == 5


@pytest.mark.slow
def test_hanging_decision_pipeline_matches_baseline_shape(tmp_path):
    """真实调用 `_run_decision_pipeline`（含引擎+LLM API+TTS+视频合成）。

    不直接调用内部计算函数（如 explore_forward/goal_trajectory）逐一拼接——
    那样测的是"各函数分别正确"，测不出"管线真的把它们正确串起来"，
    而阶段2/5要改的正是这条串接逻辑（机制闸位置、routes 挂载方式）。

    本测试是唯一覆盖"改 KB goal 后，机制闸是否还能在这个局面上正确
    区分两条计划"的自动化护栏——阶段2改 maroczy goal 前，先跑一次本测试
    确认悬兵局面（不涉及本次改动的原型）行为不变，是零回归的直接证据。
    """
    from src.pipeline.decision_pipeline import _run_decision_pipeline

    baseline = _baseline()
    out_dir = str(tmp_path)
    output_path = _run_decision_pipeline(HANGING_FEN, output_dir=out_dir)

    # 管线级失败语义（SPEC §8）：非空路径才算成功产出
    assert output_path, "决策管线在悬兵局面上应产出视频（历史基线曾成功）"
    assert os.path.isfile(output_path)

    # sidecar 与基准做核心字段对比（sidecar 由 output_dir 触发写出）
    sidecar_path = os.path.splitext(output_path)[0] + "_review.json"
    assert os.path.isfile(sidecar_path), "output_dir 非空时应产出评审 sidecar"
    with open(sidecar_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)

    assert sidecar["archetype"] == baseline["archetype"]
    assert len(sidecar["routes"]) == len(baseline["routes"]) == 2
    assert sidecar["comparison_axes"]["axis_type"] == 1
    # 核心机制闸判据：两条计划都应通过（goal_ok=True）——这是"真比较式"
    # 而非"单线退化"的直接证据，与阶段2/6 要验证的目标同一件事
    for route in sidecar["routes"]:
        assert route["goal_ok"] is True, (
            f"计划「{route['name']}」goal_ok 应为 True（悬兵局面两条计划"
            "均应通过机制闸，回归基线已验证——若本次为 False 说明发生了"
            "非预期的回归，需排查是否有无关改动影响了 goal_trajectory/"
            "explore_forward 链路）"
        )
        assert len(route.get("unique_facts", [])) > 0
