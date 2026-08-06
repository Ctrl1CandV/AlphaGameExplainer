"""decision_pipeline._jsonable 单测（PLAN-010 阶段0 步骤3，真纯函数）。

断言口径迁自 PLAN-009 实施记录附录B：StructureTrend/TradeoffMetrics 可
序列化、自环 dict 与自环 dataclass 均被深度护栏拦住、字段集与 asdict 一致。

注意：直接 import src.pipeline.decision_pipeline 会拉起 media 栈依赖
（tts_engine/board_renderer/video_composer），PLAN-010 元信息已记录这一 import
成本。经实测（阶段0冒烟）该模块可在当前环境正常 import，故直接测试真实函数，
不做额外隔离——若未来环境变化导致 import 变重，可重新评估抽出纯 helper。
"""
import dataclasses
import json

import pytest

from src.pipeline.decision_pipeline import _jsonable
from src.solver.consequence_projector import StructureTrend, TradeoffMetrics


def test_structure_trend_dataclass_serializable():
    """StructureTrend dataclass 可转为可 json.dump 的 dict。"""
    trend = StructureTrend(dimension="mover_pawns_past_mid",
                            direction="increasing",
                            samples=[0.2, 0.4, 0.6])
    result = _jsonable(trend)
    assert isinstance(result, dict)
    # json.dump 不应抛异常
    json.dumps(result, ensure_ascii=False)
    assert result["dimension"] == "mover_pawns_past_mid"
    assert result["direction"] == "increasing"
    assert result["samples"] == [0.2, 0.4, 0.6]
    assert result["robust"] is True
    assert result["perturb_results"] == []


def test_tradeoff_metrics_dataclass_serializable():
    """TradeoffMetrics dataclass 可转为可 json.dump 的 dict。"""
    tm = TradeoffMetrics(pawn_moves=3, captures=2, open_files_delta=1,
                          weak_square_hint="d5弱格", corridor_roots=4,
                          unique_ratio=0.5)
    result = _jsonable(tm)
    json.dumps(result, ensure_ascii=False)
    assert result == {
        "pawn_moves": 3, "captures": 2, "open_files_delta": 1,
        "weak_square_hint": "d5弱格", "corridor_roots": 4,
        "unique_ratio": 0.5,
    }


def test_field_set_matches_asdict():
    """_jsonable 的字段集与 dataclasses.asdict 一致（只做类型转换不裁字段）。"""
    trend = StructureTrend(dimension="passed_diff", direction="decreasing",
                            samples=[0.1, 0.05, 0.0], robust=False,
                            perturb_results=[True, False])
    via_jsonable = set(_jsonable(trend).keys())
    via_asdict = set(dataclasses.asdict(trend).keys())
    assert via_jsonable == via_asdict


def test_archetype_shift_tuple_serializable():
    """archetype_shift 是元组（如 ("hanging", "iqp")），必须能被序列化为 list。"""
    shift = ("hanging", "iqp")
    result = _jsonable(shift)
    assert result == ["hanging", "iqp"]
    json.dumps(result, ensure_ascii=False)


def test_nested_dict_with_dataclass_values():
    """真实 sidecar 场景：dict 套 dataclass 列表（trend 结果的实际形状）。"""
    payload = {
        "trends": [
            StructureTrend(dimension="opp_backward", direction="decreasing",
                            samples=[0.5, 0.3, 0.1]),
        ],
        "archetype_shift": ("hanging", "iqp"),
        "end_features": [0.1, 0.2, 0.3],
    }
    result = _jsonable(payload)
    json.dumps(result, ensure_ascii=False)  # 不应抛 TypeError
    assert isinstance(result["trends"][0], dict)
    assert result["archetype_shift"] == ["hanging", "iqp"]
    assert result["end_features"] == [0.1, 0.2, 0.3]


def test_self_referential_dict_guarded_by_depth():
    """自环 dict：a["self"] = a。深度护栏必须拦住，不抛 RecursionError。"""
    a = {"name": "loop"}
    a["self"] = a
    # 不应抛出 RecursionError；深度超限后退化为字符串
    result = _jsonable(a)
    assert isinstance(result, dict)
    assert result["name"] == "loop"
    # 递归到深处后必然被 str() 兜底，不会无限递归
    json.dumps(result, ensure_ascii=False)


@dataclasses.dataclass
class _SelfRefDataclass:
    """仅用于本测试的自环 dataclass（模拟"将来有人往 storyboard 挂自引用对象"）。"""
    name: str
    child: object = None


def test_self_referential_dataclass_guarded_by_depth():
    """自环 dataclass：obj.child = obj。深度护栏必须拦住（docstring 明确声称的场景）。

    docstring 原文："若有人往 storyboard 挂上互相引用的对象，无限递归会以
    RecursionError 表现"——这正是该护栏要防的场景，必须直接测。
    """
    node = _SelfRefDataclass(name="root")
    node.child = node
    result = _jsonable(node)
    assert isinstance(result, dict)
    assert result["name"] == "root"
    # 不应抛 RecursionError；最终应能被 json 序列化（深处退化为字符串）
    json.dumps(result, ensure_ascii=False)


def test_unknown_type_falls_back_to_str():
    """陌生类型（非 dataclass/dict/list/tuple/基本类型）退化为字符串，
    不抛异常，不让整份写失败。

    注：chess.Move 在本项目锁定的 python-chess 版本里实际是 dataclass，
    会走 dataclass 分支被递归拆解为字段字典，不落入本兜底分支——
    用 set（真正的陌生类型，不属于任何一条分支）测试兜底行为。
    """
    obj = {1, 2, 3}
    result = _jsonable(obj)
    assert isinstance(result, str)
    json.dumps(result, ensure_ascii=False)


def test_none_and_primitives_passthrough():
    """None/str/int/float/bool 原样返回。"""
    assert _jsonable(None) is None
    assert _jsonable("text") == "text"
    assert _jsonable(42) == 42
    assert _jsonable(3.14) == 3.14
    assert _jsonable(True) is True
