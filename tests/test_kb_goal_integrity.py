"""KB `structural_goal` 完整性静态守卫（零引擎、零成本）。

PLAN-010 阶段 2 修复 F5（maroczy 两条同方计划 goal 完全相同 →
机制闸退化为单计划）后，peer_review F-C 指出：单行数据修复没有任何
自动化护栏——被 revert、或未来按 carlsbad「模板」把 maroczy 后翼扩张
也改成 OR 组，都不会被现有测试拦住（`test_decision_smoke.py` 用的是
hanging 局面，不涉 maroczy）。

本文件是那道护栏：纯静态读 `data/structure_kb.json`，不起引擎，
锁住阶段 2 定稿的两条具体结论不回退：
1. **同原型内同方计划的 `structural_goal` 不得同质化**（F5 病灶本身）；
2. **maroczy 后翼扩张必须用 `opp_isolated_qside` 单维谓词、禁用 OR 组**
   （阶段 2 实测：maroczy 起点 `opp_backward=1`，套 carlsbad 式
   `{any:[opp_isolated_qside, opp_backward]}` 会经后退兵支路在起点即
   平凡满足，静默回退 3a 检查——peer_review F-E）。

注：这是「防具体已知回退」的护栏，不是「goal 全局质量」的判据；
maroczy 两计划的 A3 可分离性（交叉满足）仍是阶段 3 职责，不在此断言。
"""
import json
import os

import pytest

_KB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "structure_kb.json")


@pytest.fixture(scope="module")
def kb():
    with open(_KB_PATH, encoding="utf-8") as f:
        return json.load(f)


def _goal_signature(goal: dict) -> str:
    """把 structural_goal 归一为可比较的规范签名。

    OR 组（`any`）与普通 dict 都覆盖：普通 dict 用「排序后的
    dim>=pred」拼接；OR 组用「any(」包裹各子 goal 签名（排序）。
    两条计划签名相同即视为同质化。
    """
    if "any" in goal:
        subs = sorted(_goal_signature(g) for g in goal["any"])
        return "any(" + "|".join(subs) + ")"
    return "&".join(f"{k}{v}" for k, v in sorted(goal.items()))


def test_maroczy_two_plans_have_distinct_goals(kb):
    """F5 直接护栏：maroczy 王翼进攻 / 后翼扩张 goal 不得同质化。"""
    plans = kb["maroczy"]["plans"]
    goals = [_goal_signature(p["structural_goal"]) for p in plans]
    assert len(goals) == len(set(goals)), (
        f"maroczy 两计划 goal 同质化（F5 回退）：{goals}")


def test_maroczy_qside_plan_uses_single_dim_not_or_group(kb):
    """F-E 护栏：maroczy 后翼扩张须用 opp_isolated_qside 单维、禁用 OR 组。

    起点 opp_backward=1，OR 组会平凡满足；阶段 2 刻意选单维。
    """
    qside = next(p for p in kb["maroczy"]["plans"] if p["name"] == "后翼扩张")
    goal = qside["structural_goal"]
    assert "any" not in goal, (
        "maroczy 后翼扩张禁用 OR 组（起点 opp_backward=1 会平凡满足）")
    assert goal == {"opp_isolated_qside": ">=1"}, (
        f"maroczy 后翼扩张 goal 偏离阶段 2 定稿：{goal}")


def test_all_same_side_plan_pairs_within_archetype_not_homogeneous(kb):
    """全 KB 泛化护栏：任一原型内、同一 mover_side 的两计划 goal 不得完全同质。

    这是 F5 的一般化——阶段 2 只逐个修了 maroczy，但同质化是可复发的
    KB 编辑错误，这条守卫对全部 6 原型生效，未来新增计划也受约束。
    exclude：不同 mover_side（如 iqp 的 mover vs opponent 计划语义互斥，
    共享谓词不算同质化，见 F5 事实表原始判断）。
    """
    offenders = []
    for arch, entry in kb.items():
        by_side = {}
        for p in entry.get("plans", []):
            sig = _goal_signature(p["structural_goal"])
            key = (p.get("mover_side", "mover"), sig)
            by_side.setdefault(key, []).append(p["name"])
        for (side, sig), names in by_side.items():
            if len(names) > 1:
                offenders.append(f"{arch}/{side}: {names} 共享 goal {sig}")
    assert not offenders, "同方计划 goal 同质化：\n" + "\n".join(offenders)
