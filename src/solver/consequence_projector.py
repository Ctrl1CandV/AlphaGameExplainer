"""后果投射器（决策管线，ADR-020 阶段 5）。

对每条成立计划产出：
- `project`：长期结构趋势（P2 定稿——单调性 + 对方应招扰动双闸）
  + 结构类型转换（P19）；
- `quantify_tradeoffs`：代价量化（对手获得/承诺度/执行难度——P3 减法
  裁决：不用评估波动方差与下限，风险用「好着走廊宽度」表达）。

趋势定义（P2）：
- **单调性**：第 8/14/20 着三点 `structural_features()` 采样，某维单调
  变化且总变化量 ≥ 阈值才叫「持续扩大/持续收缩」；振荡 → 该维度无趋势
  （原规格「多深度一致」自相矛盾——趋势本就要求变化）；
- **稳健性 = 对方应招扰动**：把计划线第 2/4/6 着的对方走法换成该点
  MultiPV 内第二选择，各推一次，趋势方向保持才报（最大不确定来源是
  对手选择；扰动结果本身是教学素材——「不管对方怎么应，后翼推进都会
  留下孤兵」）。成本 1→4 次投射。

失败安全：任何维度计算失败 → 该维度缺席，不阻塞。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import chess
import chess.engine

try:
    from src.analysis.structure_features import DIM_NAMES, structural_features
    from src.analysis.structure_id import detect_pawn_structure
    from src.solver.branch_explorer import _open_engine
except ModuleNotFoundError:  # 直接运行自检时补充项目根到 sys.path
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    "..", "..")))
    from src.analysis.structure_features import DIM_NAMES, structural_features
    from src.analysis.structure_id import detect_pawn_structure
    from src.solver.branch_explorer import _open_engine

logging.getLogger("chess.engine").setLevel(logging.CRITICAL)

# 采样着点（相对当前局面）：第 8/14/20 着（P2 定稿）
SAMPLE_PLIES = (8, 14, 20)
# 扰动点：第 2/4/6 着的对方走法（成本 1→4 次投射；过慢可减到第 2 着）
PERTURB_PLIES = (2, 4, 6)
# 单调趋势的最小总变化量（归一化 0-1 特征——1 个孤立兵 = 1/3 ≈ 0.33，
# 0.15 约为半个兵形事件的保守下界）
MIN_TREND_DELTA = 0.15
# 延伸推演深度
EXTEND_DEPTH = 14


@dataclass
class StructureTrend:
    """一条单调结构趋势。`robust=False` 表示被扰动否掉（不输出）。"""
    dimension: str                       # DIM_NAMES 维度名
    direction: str                       # "increasing" / "decreasing"
    samples: List[float]                 # 3 点采样值（归一化）
    robust: bool = True                  # 扰动后仍成立
    perturb_results: List[bool] = field(default_factory=list)  # 各扰动点


@dataclass
class TradeoffMetrics:
    """代价量化（P3 减法裁决：无评估波动方差/下限）。"""
    pawn_moves: int = 0                  # 承诺度：线内兵着数
    captures: int = 0                    # 承诺度：线内兑子数
    open_files_delta: int = 0            # 末端 vs 起点开放线变化（对手获得代理）
    weak_square_hint: str = ""           # 末端结构提示（弱格/孤兵等，中文）
    corridor_roots: int = 0              # 好着走廊宽度：根着近等强数
    unique_ratio: float = 0.0            # 唯一好着密度：沿线节点唯一着占比


def _monotonic(samples: List[float], min_delta: float = MIN_TREND_DELTA
               ) -> Optional[str]:
    """三点采样是否单调（纯函数，单元测试直接测）。

    返回 "increasing" / "decreasing" / None（振荡或变化不足）。
    严格单调方向：每相邻点同向且总变化量 ≥ min_delta。
    """
    if len(samples) < 3:
        return None
    deltas = [b - a for a, b in zip(samples, samples[1:])]
    if all(d >= 0 for d in deltas) and (samples[-1] - samples[0]) >= min_delta:
        return "increasing"
    if all(d <= 0 for d in deltas) and (samples[0] - samples[-1]) >= min_delta:
        return "decreasing"
    return None


def _extend(board: chess.Board, pv: List[chess.Move], target_ply: int,
            sf_path: str, depth: int = EXTEND_DEPTH) -> Optional[chess.Board]:
    """从 board 推演 pv 到 target_ply（相对当前局面的着数）。

    pv 不够则引擎继续双方对下补齐（延伸推演）。失败返回 None（失败安全）。
    """
    b = board.copy()
    engine = None
    try:
        engine = _open_engine(sf_path)
        ply = 0
        for mv in pv:
            b.push(mv)
            ply += 1
            if ply >= target_ply:
                return b
        while ply < target_ply:
            info = engine.analyse(b, chess.engine.Limit(depth=depth))
            pv2 = info.get("pv")
            if not pv2:
                return b
            b.push(pv2[0])
            ply += 1
        return b
    except Exception:
        return None
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass


def _trends_from_boards(start: chess.Board, boards: List[chess.Board],
                        ) -> List[StructureTrend]:
    """从起点与各采样点棋盘提取单调趋势（纯特征层，无引擎）。"""
    trends = []
    fv_start = structural_features(start)
    fv_pts = [structural_features(b) for b in boards]
    for i, dim in enumerate(DIM_NAMES):
        samples = [fv_start[i]] + [fv[i] for fv in fv_pts]
        direction = _monotonic(samples)
        if direction is None:
            continue
        trends.append(StructureTrend(
            dimension=dim, direction=direction, samples=samples))
    return trends


def project(
    plan_line,
    start: chess.Board,
    sf_path: str,
    sample_plies: Tuple[int, ...] = SAMPLE_PLIES,
    perturb_plies: Tuple[int, ...] = PERTURB_PLIES,
) -> dict:
    """投射计划线：延伸推演 + 单调趋势 + 对方应招扰动 + 结构转换（P19）。

    `plan_line` 为 branch_explorer.BranchLine（含 pv）。返回：
    {
      "trends": [StructureTrend, ...],      # 扰动后仍成立（robust=True）
      "rejected_trends": [StructureTrend],  # 被扰动否掉的趋势（记录用）
      "archetype_shift": Optional[(from, to)],  # 结构类型转换（P19）
      "end_features": List[float],          # 末端特征（tradeoffs 用）
      "perturb_consistency": float,         # 扰动一致率（验证记录）
    }
    采样点超出推演能力（线太短/引擎失败）时对应点缺席。
    """
    if plan_line is None or not plan_line.pv:
        return {"trends": [], "rejected_trends": [], "archetype_shift": None,
                "end_features": [], "perturb_consistency": 1.0}

    # 主推演：pv 推演 + 引擎补齐到最远采样点
    max_ply = max(sample_plies)
    boards: List[Optional[chess.Board]] = []
    end_board = _extend(start, plan_line.pv, max_ply, sf_path)
    if end_board is None:
        return {"trends": [], "rejected_trends": [], "archetype_shift": None,
                "end_features": [], "perturb_consistency": 1.0}
    # 采样点棋盘：从 start 逐步推演 pv（+引擎补齐）——复用 _extend 到各点
    for ply in sample_plies:
        boards.append(_extend(start, plan_line.pv, ply, sf_path))

    base_trends = _trends_from_boards(start, [b for b in boards if b])
    if not base_trends:
        return {"trends": [], "rejected_trends": [], "archetype_shift": None,
                "end_features": structural_features(end_board),
                "perturb_consistency": 1.0}

    # 扰动：第 2/4/6 着的对方走法换成该点 MultiPV 第二选择
    # （perturb_ply 是 1-based 的被替换着序号——2/4/6 为偶数 = 对方着）
    # 逐条趋势独立判定（P2「趋势结论保持才报」的保守语义）：每条基趋势
    # 在**全部**扰动点都保持（方向不变且仍单调）才报；扰动后消失/反向的
    # 维度不报。首版用「集合子集」判定过严——延伸推演的引擎非确定性使
    # 扰动线自然增删维度，集合比较会把保留的趋势也全盘否掉。
    engine = None
    perturb_pool: List[set] = []
    try:
        engine = _open_engine(sf_path)
        for p_ply in perturb_plies:
            if p_ply > len(plan_line.pv):
                perturb_pool.append({(t.dimension, t.direction)
                                     for t in base_trends})
                continue
            # 推演到被替换着之前（pv 前 p_ply-1 着）——轮到 pv[p_ply-1]
            # 的走子方（对方）——multipv 第二选择即替换着
            b = start.copy()
            for mv in plan_line.pv[:p_ply - 1]:
                b.push(mv)
            infos = engine.analyse(b, chess.engine.Limit(depth=12), multipv=3)
            pv_list = [list(info.get("pv", [])) for info in infos
                       if info.get("pv")]
            if len(pv_list) < 2:
                perturb_pool.append({(t.dimension, t.direction)
                                     for t in base_trends})
                continue
            alt = pv_list[1]
            p_boards = []
            for ply in sample_plies:
                if ply <= p_ply:
                    p_boards.append(_extend(start, plan_line.pv, ply, sf_path))
                else:
                    # 扰动线：start→pv 到 p_ply-1→替换着→延伸
                    p_boards.append(_extend_perturbed(
                        start, plan_line.pv, alt, p_ply, ply, sf_path))
            p_trends = _trends_from_boards(start, [x for x in p_boards if x])
            perturb_pool.append({(t.dimension, t.direction)
                                 for t in p_trends})
    except Exception:
        perturb_pool = [{(t.dimension, t.direction)
                         for t in base_trends}] * len(perturb_plies)
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass

    kept, rejected = [], []
    for t in base_trends:
        key = (t.dimension, t.direction)
        t.perturb_results = [key in pool for pool in perturb_pool]
        t.robust = all(t.perturb_results)
        (kept if t.robust else rejected).append(t)

    # 扰动一致率：基趋势在扰动下的平均保持率（P2「记录在案」）
    if base_trends and perturb_pool:
        per_trend = [sum(t.perturb_results) / len(t.perturb_results)
                     for t in base_trends]
        consistency = sum(per_trend) / len(per_trend)
    else:
        consistency = 1.0

    # 结构类型转换（P19）：起点 vs 末端
    shift = None
    arch0, _, _ = detect_pawn_structure(start)
    arch1, _, _ = detect_pawn_structure(end_board)
    if arch0 and arch1 and arch0 != arch1:
        shift = (arch0, arch1)

    return {
        "trends": kept,
        "rejected_trends": rejected,
        "archetype_shift": shift,
        "end_features": structural_features(end_board),
        "perturb_consistency": round(consistency, 3),
    }


def _extend_perturbed(start: chess.Board, pv: List[chess.Move],
                      alt: List[chess.Move], perturb_ply: int,
                      target_ply: int, sf_path: str,
                      depth: int = EXTEND_DEPTH) -> Optional[chess.Board]:
    """扰动线延伸：start → pv 前 perturb_ply-1 着 → 换 alt[0]（替换
    第 perturb_ply 着）→ 引擎补齐到 target_ply。

    与 `_extend` 的差异只在扰动点替换对方走法；引擎补齐逻辑相同。
    perturb_ply 为 1-based 被替换着序号（2/4/6——对方着）。
    """
    b = start.copy()
    engine = None
    try:
        engine = _open_engine(sf_path)
        ply = 0
        for mv in pv[:perturb_ply - 1]:
            b.push(mv)
            ply += 1
        if alt:
            b.push(alt[0])
            ply += 1
        while ply < target_ply:
            info = engine.analyse(b, chess.engine.Limit(depth=depth))
            pv2 = info.get("pv")
            if not pv2:
                return b
            b.push(pv2[0])
            ply += 1
        return b
    except Exception:
        return None
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass


def quantify_tradeoffs(
    plan_line,
    start: chess.Board,
    sf_path: str,
    open_lines=None,
) -> TradeoffMetrics:
    """代价量化（P3 减法裁决）。

    - 承诺度：线内兵着数 + 兑子数（pv 静态统计）；
    - 对手获得代理：末端 vs 起点开放线变化（开放线是双方共享但常被
      对方利用的资源）——12 维 `open_files` 差分；
    - 弱格提示：末端结构识别（structure_id 命中/孤立兵等，中文）——
      简化版：末端特征里的孤立/后退兵提示；
    - 好着走廊宽度：根着 MultiPV 近等强着数（open_lines 传入时）；
    - 唯一好着密度：沿线 4 节点各 multipv=3，「唯一好着」节点占比。
    """
    metrics = TradeoffMetrics()
    if plan_line is None or not plan_line.pv:
        return metrics

    # 承诺度（pv 静态统计——from_square 的棋子必须在 push 前检查，
    # push 后原格已空）
    b = start.copy()
    for mv in plan_line.pv:
        captured = b.piece_at(mv.to_square)
        is_pawn = (b.piece_type_at(mv.from_square) == chess.PAWN)
        b.push(mv)
        if is_pawn:
            metrics.pawn_moves += 1
        if captured is not None:
            metrics.captures += 1

    # 末端开放线差分（对手获得代理）
    end = _extend(start, plan_line.pv, SAMPLE_PLIES[-1], sf_path)
    if end is not None:
        f0 = structural_features(start)
        f1 = structural_features(end)
        metrics.open_files_delta = round(
            (f1[DIM_NAMES.index("open_files")]
             - f0[DIM_NAMES.index("open_files")]) * 8)
        # 弱格提示（末端结构，中文）
        from src.analysis.structure_features import _raw_features
        raw = _raw_features(end)
        hints = []
        if raw["opp_isolated_qside"]:
            hints.append("对方后翼孤兵")
        if raw["opp_isolated_center"]:
            hints.append("对方中心孤兵")
        if raw["opp_backward"]:
            hints.append("对方后退兵")
        if raw["mover_pawns_past_mid"]:
            hints.append("己方兵过线")
        metrics.weak_square_hint = "/".join(hints)

    # 好着走廊宽度（根着 MultiPV 近等强数——open_lines 传入）
    if open_lines and len(open_lines) >= 2:
        from src.solver.branch_explorer import equivalence_gap
        base = open_lines[0].cp
        metrics.corridor_roots = sum(
            1 for ln in open_lines if equivalence_gap(base, ln.cp) <= 80)

    # 唯一好着密度：沿线 4 节点（0/4/8/12 着）multipv=3
    engine = None
    try:
        engine = _open_engine(sf_path)
        unique_nodes = 0
        nodes = 0
        b = start.copy()
        for i, mv in enumerate(plan_line.pv):
            if i in (0, 4, 8, 12):
                infos = engine.analyse(b, chess.engine.Limit(depth=12),
                                       multipv=3)
                cps = []
                for info in infos:
                    sc = info.get("score")
                    if sc is None:
                        continue
                    rel = sc.relative
                    m = rel.mate()
                    cps.append(10000 if m and m > 0 else
                               (-10000 if m and m < 0 else (rel.score() or 0)))
                if len(cps) >= 2:
                    nodes += 1
                    if abs(cps[0] - cps[1]) > 80:
                        unique_nodes += 1
            b.push(mv)
        metrics.unique_ratio = round(unique_nodes / nodes, 3) if nodes else 0.0
    except Exception:
        pass
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass
    return metrics


if __name__ == "__main__":
    """阶段 5 单元测试：单调性判定（纯函数）+ 已知趋势局面投射。

    1. 振荡序列不输出（_monotonic 直接测）；
    2. 单调增/减序列正确判定；
    3. 已知趋势局面：多数翼推进（a4-b4-c4 制造通路兵）——passed_diff
       或 mover_pawns_past_mid 应出现单调趋势（真实引擎投射）。
    """
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    "..", "..")))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))

    results = []

    # 1. 振荡不输出
    osc = _monotonic([0.5, 0.2, 0.8])
    results.append(("振荡序列不输出", osc is None, f"got={osc}"))

    # 2. 单调判定
    inc = _monotonic([0.1, 0.3, 0.6])
    dec = _monotonic([0.8, 0.5, 0.2])
    flat = _monotonic([0.1, 0.12, 0.13])
    results.append(("单调增/减/不足判定",
                    inc == "increasing" and dec == "decreasing"
                    and flat is None,
                    f"inc={inc} dec={dec} flat={flat}"))

    # 3. 已知趋势局面投射（悬兵推进 d5——通路兵/过线趋势）
    sf = os.getenv("STOCKFISH_PATH", "")
    if not os.path.isabs(sf):
        sf = os.path.normpath(os.path.join(os.getcwd(), sf))
    from src.solver.branch_explorer import BranchLine

    b = chess.Board("2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/"
                    "1R2R1K1 w - - 1 21")
    # 用悬兵推进的实际约束线（explore_forward 一步到位）
    from src.solver.branch_explorer import explore_forward
    plan = {"name": "推进悬兵", "direction": {"pawn_files": ["c", "d"],
            "target_zone": "center", "break_squares": ["c5", "d5"]}}
    line = explore_forward(b, plan, sf, depth=14)
    if line is not None and line.pv:
        res = project(line, b, sf)
        dims = [t.dimension for t in res["trends"]]
        print(f"  推进悬兵趋势: {[(t.dimension, t.direction) for t in res['trends']]}")
        print(f"  扰动一致率: {res['perturb_consistency']} "
              f"| 被否: {[t.dimension for t in res['rejected_trends']]}")
        print(f"  结构转换: {res['archetype_shift']}")
        results.append(("已知趋势局面产出趋势", len(res["trends"]) > 0,
                        f"趋势数={len(res['trends'])} dims={dims}"))
    else:
        results.append(("已知趋势局面产出趋势", False, "无约束线"))

    ok = True
    for name, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        ok &= passed
    print("阶段 5 单元测试:", "全部通过" if ok else "存在失败")
    sys.exit(0 if ok else 1)
