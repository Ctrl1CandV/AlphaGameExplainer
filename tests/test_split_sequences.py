"""decision_pipeline._split_sequences 单测（PLAN-010 阶段0 步骤3，真纯函数）。

断言口径：1/2/4 计划及无计划段四种情形，每序列 <=1 计划段、段总数守恒；
2 计划路径与改动前逐段一致（零回归——08.04 修复记录明确要求）。

`_split_sequences` 只依赖 Segment.moves 是否非空来判定"计划段"，不涉及
引擎/LLM/IO，是真纯函数。
"""
from src.common import Segment
from src.pipeline.decision_pipeline import _split_sequences


def _seg(idx, moves=None, phase=""):
    """构造最小 Segment，moves 用非空列表模拟"带走法的计划段"。"""
    return Segment(move_idx=idx, text=f"seg{idx}", moves=moves or [], phase=phase)


def _has_moves(seg):
    return bool(seg.moves)


def test_single_plan_one_sequence():
    """1 个计划 -> 1 个序列（不满足 len(plan_idx) >= 2 的切分条件）。"""
    segs = [
        _seg(0, phase="decision"),           # 开场
        _seg(1, moves=["e2e4"], phase="p1"),  # 计划段
        _seg(2, phase="summary"),             # 总结
    ]
    seqs = _split_sequences(segs)
    assert len(seqs) == 1
    assert seqs[0] == segs
    total = sum(len(s) for s in seqs)
    assert total == len(segs)


def test_two_plans_matches_pre_change_behavior():
    """2 计划：切点为 [:plan_idx[1]] / [plan_idx[1]:]，与 08.04 修复前逐段一致
    （REV 记录的零回归要求——2 计划路径行为不变）。"""
    segs = [
        _seg(0, phase="decision"),            # 开场，idx0，无 moves
        _seg(1, moves=["a2a3"], phase="p1"),   # 计划甲，idx1，plan_idx[0]=1
        _seg(2, moves=["b2b4"], phase="p2"),   # 计划乙，idx2，plan_idx[1]=2
        _seg(3, phase="compare"),              # 对比，idx3
        _seg(4, phase="summary"),              # 总结，idx4
    ]
    seqs = _split_sequences(segs)
    assert len(seqs) == 2
    # 序列1 = 开场 + 计划甲（下标 0:2，即 [:plan_idx[1]]，plan_idx[1]=2）
    assert seqs[0] == segs[0:2]
    # 序列2 = 计划乙 + 对比 + 总结（下标 2:）
    assert seqs[1] == segs[2:]
    # 每序列内带 moves 的段数 <= 1
    for seq in seqs:
        assert sum(1 for s in seq if _has_moves(s)) <= 1
    # 段总数守恒
    assert sum(len(s) for s in seqs) == len(segs)


def test_four_plans_no_two_plan_segments_share_a_sequence():
    """4 计划（对齐 iqp 实测：4 条计划全部可行）：每序列 <=1 计划段，
    不能出现 08.04 修复前那种"计划乙/丙/丁塞进同一序列"的 bug。"""
    segs = [
        _seg(0, phase="decision"),
        _seg(1, moves=["m1"], phase="p1"),
        _seg(2, moves=["m2"], phase="p2"),
        _seg(3, moves=["m3"], phase="p3"),
        _seg(4, moves=["m4"], phase="p4"),
        _seg(5, phase="compare"),
        _seg(6, phase="summary"),
    ]
    seqs = _split_sequences(segs)
    # 4 个计划段 -> 4 个序列
    assert len(seqs) == 4
    for seq in seqs:
        moves_count = sum(1 for s in seq if _has_moves(s))
        assert moves_count <= 1, f"序列内出现 {moves_count} 个计划段，违反≤1约束"
    # 段总数守恒（无重复、无丢失）
    assert sum(len(s) for s in seqs) == len(segs)
    # 末序列须含对比+总结（不带 moves 的收尾段）
    assert seqs[-1][-2:] == segs[-2:]


def test_no_plan_segments_returns_single_sequence():
    """无计划段（全部 moves 为空）：< 2 个 plan_idx，退化为单序列。"""
    segs = [_seg(0), _seg(1), _seg(2)]
    seqs = _split_sequences(segs)
    assert len(seqs) == 1
    assert seqs[0] == segs


def test_empty_segments_list():
    """空输入：不抛异常，返回单个空列表的序列（< 2 个 plan_idx 分支）。"""
    seqs = _split_sequences([])
    assert seqs == [[]]
