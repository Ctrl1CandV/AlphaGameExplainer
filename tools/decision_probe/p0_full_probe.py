"""P0-full 四项验证探针（PLAN-009 阶段 0.6）——P0-lite 通过后的走向闸门。

四项（FINDINGS-002 §3.2/§3.3 修订版判据）：
- **A1 污染检查**：候选集内典型错误着的引擎评估**不压过**正确执行着
  （只有压过才需干预——引擎会因评估更低而不选错误着，压过才是真污染）；
- **A2 结构目标达成**：约束线跑完 `structural_goal` 达成或朝达成显著移动
  **≥70%**（不要求原型保持——推进型计划本就改变兵形，P19）；
- **A3 可分离性（自校准）**：跨计划距离显著 > 同计划组内距离
  （组内 = 同一计划的不同执行线；不拍绝对阈值，P8 阈值一并产出）；
- **B3 口径一致性**：M8 判「方向不同」的定义与 KB `direction` 是同一套语义
  （`direction_zone` 单一事实来源，检查 import 同一对象）。

走向分支（FINDINGS §3.5）：A 与 B 都过 → 推进阶段 1；A 过 B 不过 →
识别/候选重设计；**A 不过 → 前向路线不成立，转反向为主或放弃**。

用法：
    "C:\\Users\\LiuYiJie\\.conda\\envs\\commentary\\python.exe" -m tools.decision_probe.p0_full_probe
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

import chess

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv  # noqa: E402
load_dotenv(os.path.join(_ROOT, ".env"))

from src.analysis.direction import direction_zone  # noqa: E402
from src.analysis.structure_features import (  # noqa: E402
    feature_distance,
    structural_features,
)
from src.solver.branch_explorer import explore_forward  # noqa: E402
from tools.decision_probe.a1_recall_probe import CASES  # noqa: E402
from tools.decision_probe.engine_probe import (  # noqa: E402
    resolve_stockfish,
)

OUT_DIR = os.path.join(_ROOT, "data", "quality_benchmark_decision")
KB_PATH = os.path.join(_ROOT, "data", "structure_kb.json")

# 污染检查：错误着比正确着高出多少才算「压过」。
# 80cp 对齐可行性闸宽松端（PLAN-009 DEFAULT_FEASIBLE_CP），理由：引擎
# 评估噪声（±30cp 常见）+ 与运行时可行性闸同口径。**这是 FINDINGS 判据
# 字面（'>' 无阈值）的登记化放宽**（peer_review Major 指出隐性放宽——
# 已在此登记；正式修订登记待 planner 确认写入 FINDINGS §3.2）。
POLLUTION_CP = 80
LINE_LEN = 16          # A2/A3 取约束线前 16 着（8 个回合——首着准备+推进+
                       # 突破+后果，短于 10 着只能看到准备阶段）
A3_GROUP_ROOTS = 3     # 组内取前 3 个候选根着各出一条执行线

# A1 污染检查排除的样本：**计划在该局面不可行**——污染检查的语义是
# 「典型错误着不压过正确执行着」，前提是计划本身可行（错误着 = 执行时机
# 错误）。下列局面全局最优（MultiPV 实测）不支持对应计划，其「错误着」
# 实为核心执行着（过早形态），标为错误是「计划不可行」的同义反复，
# 0 污染结论平凡（引擎不会执行不可行计划）——样本层排除（peer_review
# Major 4/7）：
#   - 卡尔斯巴德/中心突破：carlsbad 原 FEN 下 e4 被 Nf6 盯防（KB
#     typical_mistakes「e4 突破缺乏子力支援」），全局最优 h3/a3/Rab1；
#   - 马洛齐束缚/王翼进攻：白未易位（全局最优 O-O/Rc1），h4-g4 冲击
#     时机错误——该局面应先完成王翼安全，王翼进攻是次级计划。
EXCLUDE_A1_PLANS = {
    ("卡尔斯巴德结构", "中心突破"),
    ("马洛齐束缚", "王翼进攻"),
}


def _resolve_sf() -> str:
    sf = os.getenv("STOCKFISH_PATH", "")
    if not os.path.isabs(sf):
        sf = os.path.normpath(os.path.join(_ROOT, sf))
    return sf


# ---------------------------------------------------------------- A1 污染检查


def run_a1_pollution(sf: str, depth: int = 12) -> dict:
    """候选集内典型错误着评估是否压过正确执行着。

    判据（FINDINGS-002 §3.2 修订版语义）：
      「典型错误着不能压过正确执行着」——只有错误着评估高到引擎会选它，
      前向机制才会把「计划的错误执行」当计划产出，那才是真污染。

    实现要点：
    - 正确/错误二分**不按**「文献 vs 非文献」——文献着集合里混着典型错误着
      （卡尔斯巴德过早 b4=-45、e4=-96、IQP 的 Nb6=-489/Ne5=-444 都是文献
      着），非文献着里也有同计划的合理准备手（a3=41 与文献 a4=40 几乎等强）。
      正确/错误来自 CASES 的显式 `mistake_moves` 字段（KB typical_mistakes
      推导 + 单根着实测交叉验证）。
    - 压过基准取**正确执行着的最佳**（best_correct）而非最差：引擎在候选集内
      选评估最高者，只有错误着超过所有正确执行着才会被执行；用最差正确着
      做基准会被文献里「方向对但该局面执行差」的着（如 Ne5）拖垮。
    - 两组都只取**落在候选集内**的着——候选集外的着引擎根本看不见，无污染
      可言（那是 A1 召回失败，不是污染）。
    """
    from src.analysis.direction import direction_candidates

    records = []
    polluted = 0
    total = 0
    skipped = 0
    for case in CASES:
        board = chess.Board(case["fen"])
        legal_sans = {board.san(mv) for mv in board.legal_moves}
        for plan in case["plans"]:
            if (case["archetype"], plan["name"]) in EXCLUDE_A1_PLANS:
                records.append({
                    "archetype": case["archetype"], "plan": plan["name"],
                    "skipped": True,
                    "reason": "计划在该局面不可行（全局最优不支持）——污染检查无意义",
                })
                skipped += 1
                continue
            cands = {board.san(mv) for mv in
                     direction_candidates(board, plan["direction"], top_n=10)}
            mistakes = [s for s in plan.get("mistake_moves", [])
                        if s in legal_sans and s in cands]
            correct = [s for s in plan.get("literature_moves", [])
                       if s in legal_sans and s in cands
                       and s not in mistakes]
            if not mistakes or not correct:
                # 候选集内无错误着或正确着：单独统计，不计入通过率分母
                # （peer_review Minor：原实现计入 total 但不 increment
                # polluted，会虚高通过率）
                skipped += 1
                records.append({
                    "archetype": case["archetype"], "plan": plan["name"],
                    "skipped": True, "reason": "候选集内无错误着或正确着",
                })
                continue
            total += 1

            evals = {}
            for san in mistakes + correct:
                mv = next(m for m in board.legal_moves if board.san(m) == san)
                evals[san] = _single_root_eval(board, mv, sf, depth)
            mistake_evals = [evals[s] for s in mistakes if evals[s] is not None]
            correct_evals = [evals[s] for s in correct if evals[s] is not None]
            if not mistake_evals or not correct_evals:
                continue
            best_mistake = max(mistake_evals)
            best_correct = max(correct_evals)
            is_polluted = best_mistake > best_correct + POLLUTION_CP
            polluted += int(is_polluted)
            records.append({
                "archetype": case["archetype"],
                "plan": plan["name"],
                "best_mistake": best_mistake,
                "best_correct": best_correct,
                "gap": best_mistake - best_correct,
                "mistakes_in_cands": mistakes,
                "correct_in_cands": correct,
                "polluted": is_polluted,
            })
            print(f"  A1 {case['archetype']}/{plan['name']}: "
                  f"错误max={best_mistake} 正确max={best_correct} "
                  f"gap={best_mistake - best_correct} "
                  f"{'⚠️污染' if is_polluted else '✓'}")
    rate = round(100.0 * (total - polluted) / total, 1) if total else 0.0
    passed = polluted == 0
    print(f"A1 污染检查: {polluted}/{total} 污染，通过率 {rate}%"
          f"（判据：0 污染）")
    return {"total": total, "polluted": polluted, "rate_pct": rate,
            "passed": passed, "records": records}


def _single_root_eval(board, move, sf, depth):
    """单根着深搜评估（root_moves=[move]）。失败返回 None。"""
    from src.solver.branch_explorer import _open_engine
    engine = None
    try:
        engine = _open_engine(sf)
        info = engine.analyse(board, chess.engine.Limit(depth=depth),
                              root_moves=[move])
        score = info.get("score")
        if score is None:
            return None
        rel = score.relative
        m = rel.mate()
        if m is not None:
            return 10000 if m > 0 else -10000
        return rel.score() or 0
    except Exception:
        return None
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass


# ---------------------------------------------------------------- A2 结构目标达成


def _goal_progress(board, line, goal, max_len=LINE_LEN):
    """沿线朝 goal 的最大进步。**逻辑已收口到 `structure_features`**。

    返回 (start_satisfied, satisfied_after_start, progress_dict)，语义与
    A2 判据一致（FINDINGS §3.2「约束线跑完达成或朝达成方向显著移动」）：
    - start_satisfied：起点是否已满足 goal——诊断用（区分推进型/保持型样本）；
    - satisfied_after_start：**push 至少一步后**的任一点满足（起点不算，
      peer_review Critical 2 建议 b：否则保持型样本无论线怎么走都通过）；
    - progress_dict：各目标维朝目标方向的最大进步（原始计数单位）。

    本函数现在只是 `goal_trajectory` 的薄封装（08.04 修）。此前它自带一份
    独立实现，与产品链路的判据各算一套，且**漏传 `mover_color`**——
    `_raw_features(b)` 缺省按 `board.turn` 归一化，而线内每 push 一着 turn
    就翻转，「我方/对方」语义逐着颠倒。实测黑方决策点（中心突破，goal
    `mover_pawns_past_mid >= 1`）：锚定视角时 d5d4 一着即达成、progress=1，
    不锚定则整条线恒为 0——同一条线两套结论，A2 通过率被系统性低估。
    KB 6 原型 14 条计划里 7 条命中此偏差。

    顺带删掉了 `"<="` 分支：`goal_satisfied` 只支持 `>=` / `==`（未知谓词
    保守判不满足），KB 全部 structural_goal 实测也只用这两种，那个分支是
    永不执行的死代码，留着只会让人以为 `<=` 可用。
    """
    from src.analysis.structure_features import goal_satisfied, goal_trajectory

    # 视角锚点 = **决策点**走子方，整条线共用（见 goal_trajectory docstring）
    mover = board.turn
    tr = goal_trajectory(board, line, goal, mover, max_len)
    return (goal_satisfied(board, goal, mover),
            bool(tr["goal_reached"]), tr["goal_progress"])


def _mirror_direction(direction: dict) -> dict:
    """把原色视角的 direction 变换为 mirror 归一化（mover=白）视角。

    KB direction 的 break/pressure/outpost 格是**原色几何**（如消除计划
    break d4 = 黑 d5 推 d4 吃白 d4）。运行时局面统一 mirror 归一化
    （走子方=白），黑方计划（mover_side: "opponent"）的格子必须 rank
    翻转（d4→d5）后才匹配。file 不变、rank 镜像——与 board.mirror()
    同一变换（P22）。
    这是阶段 1 KB mirror 编写规范落地前的探针适配；规范落地后 KB 数据
    本身按 mover=白视角写，本函数退役。
    """
    def flip(sq: str) -> str:
        sqi = chess.parse_square(sq)
        return chess.square_name(
            chess.square(chess.square_file(sqi), 7 - chess.square_rank(sqi)))
    out = dict(direction)
    for k in ("break_squares", "pressure_squares", "outpost_squares"):
        if k in out:
            out[k] = [flip(s) for s in out[k]]
    return out


def run_a2_goals(sf: str, depth: int = 14) -> dict:
    """约束线是否达成 structural_goal 或朝达成显著移动。

    判据（peer_review 修正后）：
    - **推进型**（起点未满足）：约束线**制造**目标——终点达成或任一目标
      维度显著移动（原始计数进步 ≥1）；
    - **保持型**（起点已满足，如 IQP 施压/保持）：约束线**不破坏**目标
      ——终点仍满足。原实现起点满足即 ok=True（无论线怎么走都通过），
      3 个样本是无效验证——已修。
    - 多数推进型计划 16 着内**不能**达成目标（少数派攻击的 b4-b5 兑兵
      制造孤兵需要整条执行序列），达成本身受线长限制——「显著移动」
      判据兜底（约束线是否**朝目标走**才是 A2 要验的机制成立性）。

    样本按「计划角色 ↔ 局面角色」匹配（mover_side 语义）：
    - 施压方计划（针对对方孤兵）在黑 d5 孤兵局面测；
    - 持有方计划（保持/兑掉己方孤兵）在白 d4 孤兵局面测（mirror 版，
      原局面白 d4 不孤立——e2 兵存在，测持有方计划是角色错配）；
    - **推进消除孤兵已移除**：该计划是黑方执行（黑 d5 推 d4 兑白 d4
      孤兵），正确测法需「黑持 d5 兵 + 白持 d4 孤兵」的专属局面——
      现有样本集（黑 d5 孤兵 FEN 白 c/e 有兵）与 mirror 版（黑 d5 不
      孤立）都使 goal 起点平凡满足，是无效样本（peer_review Critical）；
      阶段 1 扩充样本时补专属局面；
    - 中心突破从 elite PGN 实扫（QID Spassky ply=62 真实局面）——
      原 carlsbad FEN 下 e4 被 Nf6 盯防不可行（KB typical_mistakes
      「e4 突破缺乏子力支援」教科书场景，M8 闸门运行时滤掉），换可行
      局面重验（用户裁决：样本正确化，判据不动）。
    """
    kb = json.load(open(KB_PATH, encoding="utf-8"))
    samples = [
        ("carlsbad", kb["carlsbad"],
         "r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/2NBPN2/PPQ2PPP/R4RK1 w - - 8 11",
         {"少数派攻击"}, False),
        ("carlsbad", kb["carlsbad"],
         "1k2r3/1bp1q3/1p1br3/p2p1ppp/3P1P1P/1P2P1P1/P1R2K2/2BQRB2 w - - 0 32",
         {"中心突破"}, False),
        ("iqp", kb["iqp"],
         "r1bq1rk1/pp2bppp/2n2n2/3p4/N7/5NP1/PP2PPBP/R1BQ1RK1 w - - 2 11",
         {"对孤兵施压"}, False),
        ("iqp", kb["iqp"],
         "r1bq1rk1/pp2ppbp/5np1/n7/3P4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 2 11",
         {"保持孤兵（利用动态潜力）"}, False),
        ("iqp", kb["iqp"],
         "r1bq1rk1/pp2ppbp/5np1/n7/3P4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 2 11",
         {"推进兑掉孤兵"}, False),
        # 阶段 3 新增（PLAN-010）：maroczy 两计划——阶段 2 goal 去同质化后
        # 才有可测内容（改前两条 goal 完全相同 mover_pawns_past_mid>=1）。
        # 决策点 FEN 同 structure_id/a1_recall 自检（白 c4+e4 束缚，白走子）。
        # 王翼进攻 goal=mover_pawns_past_mid>=1，后翼扩张 goal=opp_isolated_qside>=1。
        # 两者起点均未满足（实测 mover_pawns_past_mid=0 / opp_isolated_qside=0），
        # 属推进型，A2 验「约束线是否朝各自 goal 移动」。
        ("maroczy", kb["maroczy"],
         "r2q1rk1/pp2ppbp/3pbnp1/8/2P1P3/2N1B3/PP1QBPPP/R3K2R w KQ - 5 11",
         {"王翼进攻", "后翼扩张"}, False),
        # 阶段 3 新增（PLAN-010）：majority 两计划。决策点 FEN 同自检（Dragon
        # 白后翼 3v2 多数）。**已知样本局限（阶段 2 复核 + 阶段 3 peer_review）**：
        # 该局面下「王翼行动」goal=mover_pawns_past_mid>=1 起点即满足（d5 兵已过
        # 中线，属后翼/中心兵，与王翼机制无关），属保持型平凡满足——A2 判「不
        # 破坏」通过是空泛 pass；「多数翼推进」goal=passed_diff>=1 实测线内达成
        # （passed_diff=1、reached=True），但 passed_diff 是「己-彼通路兵差」，
        # +1 可能来自对方丢通路兵而非己方造出——goal 谓词区分不了「己方推进见
        # 效」与「对方白丢」，这条 pass 语义存疑。此局面对 majority 的 goal 验证
        # 力弱，A2 通过高估了真实验证强度，真验证移交阶段 6 换局面（见 PLAN 阶段 2/6）。
        ("majority", kb["majority"],
         "r2qr3/3bRpk1/p2p2p1/3P2Qp/1p6/1N3P2/PPP3PP/1K1R4 w - - 1 21",
         {"多数翼推进", "王翼行动"}, False),
        # PLAN-011 阶段 2：Benoni 两计划 A2 样本。
        ("benoni", kb["benoni"],
         "r2q1rk1/pp1n1ppp/3p4/2pPp3/2P1P3/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 13",
         {"中心突破", "后翼扩张"}, False),
    ]
    records = []
    achieved = total = 0
    for archetype, entry, fen, plan_names, mirror in samples:
        board = chess.Board(fen)
        for plan in entry["plans"]:
            if plan_names is not None and plan["name"] not in plan_names:
                continue
            goal = plan.get("structural_goal")
            if not goal:
                continue
            total += 1
            line = explore_forward(board, plan, sf, depth=depth)
            if line is None or not line.pv:
                records.append({"archetype": archetype, "plan": plan["name"],
                                "achieved": False, "reason": "无约束线"})
                print(f"  A2 {archetype}/{plan['name']}: 无约束线 ⚠️")
                continue
            start_sat, sat_after, progress = _goal_progress(board, line.pv, goal)
            significant = any(p >= 1 for p in progress.values())
            # 判据（peer_review Critical 2 建议 b）：satisfied 必须发生在
            # push 至少一步之后（起点满足不算）；或显著移动
            ok = sat_after or significant
            achieved += int(ok)
            records.append({"archetype": archetype, "plan": plan["name"],
                            "achieved": ok, "start_satisfied": start_sat,
                            "satisfied_after_start": sat_after,
                            "progress": progress,
                            "significant": significant,
                            "first_move": board.san(line.move),
                            "reason": "" if ok else "线内未达成且无显著移动"})
            print(f"  A2 {archetype}/{plan['name']}: "
                  f"{'✓' if ok else '✗'} 首着={board.san(line.move)} "
                  f"起点满足={start_sat} 线内满足={sat_after} 进步={progress}")
    rate = round(100.0 * achieved / total, 1) if total else 0.0
    passed = rate >= 70.0
    print(f"A2 结构目标达成/显著移动: {achieved}/{total} = {rate}%（判据 ≥70%）")
    return {"total": total, "achieved": achieved, "rate_pct": rate,
            "passed": passed, "records": records}


# ---------------------------------------------------------------- A3 可分离性


def _line_features(board, line, n=8):
    """取约束线前 n 着的特征序列（含起点），返回终点特征。"""
    b = board.copy()
    fv = structural_features(b)
    for i, mv in enumerate(line[:n]):
        try:
            b.push(mv)
        except Exception:
            break
        fv = structural_features(b)
    return fv


def _single_root_line(board, move, sf, depth):
    """单根着深搜返回 pv 线（root_moves=[move]）。失败返回 None。

    A3 需要**每个候选根着各出一条执行线**——explore_open(k=1) 只返回全局
    top1 的线，筛根着匹配基本全军覆没（首着以外的根着永远拿不到线，这是
    首版 A3 无数据的直接原因）。root_moves=[move] 是正解：引擎在该根着
    约束下深搜，返回该着的真实执行线。
    """
    from src.solver.branch_explorer import _open_engine
    engine = None
    try:
        engine = _open_engine(sf)
        info = engine.analyse(board, chess.engine.Limit(depth=depth),
                              root_moves=[move])
        pv = info.get("pv")
        if not pv:
            return None
        return list(pv)
    except Exception:
        return None
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass


def run_a3_separability(sf: str, depth: int = 14) -> dict:
    """跨计划距离 vs 同计划组内距离（自校准，不拍阈值）。

    样本（用户裁决限定真实取舍计划对 + 多局面复测——单局面临界不稳定
    已被 3 次运行证实）：
    - **carlsbad**：Grünfeld 三马体系 ply=60 真实局面（elite PGN 实扫），
      两计划约束线实测均含执行手（少数派 b4-b5、中心 e4）；
    - **悬兵**：Alapin ply=40（推进悬兵 c5 / 保持悬兵前哨）；
    - **石墙**：荷兰石墙（后翼扩张 / 中心突破 e3）。
    每局面独立判据「跨计划中位 > 组内中位」（FINDINGS「跨计划距离显著
    大于组内距离」的稳健量化——中位数是「噪声本底」的中心估计；P90 与
    A1 并列包含截断内在冲突不可用，见 2026-08-03 实施记录）；**全部局面
    通过才算 A3 成立**。距离 = P16 特征向量加权曼哈顿（feature_distance，
    P8 阈值一并产出）。

    组内：searchmoves 集内换根着（候选集前 A3_GROUP_ROOTS 个——FINDINGS
    §3.2「同 plan 的 searchmoves 集内换根着即得」）；
    跨计划：两计划线全部组合对（n×m）。
    """
    kb = json.load(open(KB_PATH, encoding="utf-8"))
    situations = {
        "carlsbad": (kb["carlsbad"]["plans"],
                     "4r3/pp2r1k1/2p2ppn/3pP2p/3P3P/2NRP2K/PP3P2/2R5 w - - 0 31"),
        "hanging": (kb["hanging"]["plans"],
                    "2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/"
                    "1R2R1K1 w - - 1 21"),
        "stonewall": (kb["stonewall"]["plans"],
                      "rn3rk1/pb2q1pp/1ppbpn2/3pNp2/2PP4/1P4P1/PB1NPPBP/"
                      "R2Q1RK1 w - - 2 11"),
        # 阶段 3 新增（PLAN-010）：maroczy 两计划（王翼进攻/后翼扩张）——
        # 阶段 2 交叉矩阵已实测两条约束线在 16 着内会互相命中对方 goal 维
        # （引擎在两种方向约束下都倾向 b5 后翼破坏），预判 A3 大概率不过；
        # 本样本正是为把这个疑点从「交叉矩阵旁证」变成「A3 判据正式取证」。
        # 决策点 FEN 同 a1_recall/structure_id 自检。
        "maroczy": (kb["maroczy"]["plans"],
                    "r2q1rk1/pp2ppbp/3pbnp1/8/2P1P3/2N1B3/PP1QBPPP/"
                    "R3K2R w KQ - 5 11"),
        # PLAN-011 阶段 1-2：majority A3 样本从 Dragon 自检 FEN 换成 stage4
        # 实战局面（PLAN-010 只在 Dragon FEN 测，它恰好不通过 margin -0.0417；
        # 多局面重测 4/6 通过——Dragon 是样本缺陷不是原型本质不可分）。
        # 换成 stage4 局面 [0]（margin +0.17）与 [2]（margin +0.38）较稳健。
        "majority_a": (kb["majority"]["plans"],
                       "1r3r1k/p1nqp1bp/3p1p2/2p2P2/2P2N2/"
                       "4B1P1/PP4QP/4RRK1 w - - 3 21"),
        "majority_b": (kb["majority"]["plans"],
                       "2r1nr1k/p3qp1P/1n2p1pP/1p1pP3/3N1B2/"
                       "2P1QP2/P1P5/2K2R1R w - - 0 21"),
        # 补测新增（2026-08-06，untangler 复盘 + 用户裁决）：iqp 从未进过 A3
        # 探针，却是阶段 4 双计划筛最大供给源（15 过闸/9 过粗筛）——「未测」
        # 不是「失败」，是唯一能实质翻转 verified 口径结论的变量。
        # iqp 4 条计划分属两种 mover_side 角色，A3 只应在「同一决策点走子方
        # 真能执行」的计划间测可分离性（与 decision_pipeline 角色闸同口径），
        # 故拆施压方 / 持有方两个局面、由 applicable_mover_side 筛计划：
        #   施压方（白走子、黑持 d5 孤兵）→ 对孤兵施压 / 推进消除孤兵；
        #   持有方（白走子、白持 d4 孤兵）→ 保持孤兵 / 推进兑掉孤兵。
        #
        # **iqp_pressure 已移除（08.06，independent_analysis 分歧 2）**：该 FEN
        # 白方无 d 线兵，「推进消除孤兵」（direction pawn_files=["d"]）永远无法
        # 触发方向约束，退化为松散中心手——A2 docstring 早已明说此计划需
        # 「黑持 d5 + 白持 d4 孤兵」的专属局面、现有样本是无效样本。拿无效样本
        # 测 A3 会把「样本退化」误读成「结构趋同」。待构造可执行局面后补回。
        "iqp_holder": (kb["iqp"]["plans"],
                       "r1bq1rk1/pp2ppbp/5np1/n7/3P4/2N2N2/"
                       "PP2BPPP/R1BQ1RK1 w - - 2 11"),
        # PLAN-011 阶段 1-3：iqp 多局面重测 1/6 通过（margin +0.08）——
        # iqp_holder 自检 FEN 不过（margin -0.2375）不代表全灭，换 stage4
        # 实战局面补充。该局面 stage4 实扫、过了双计划机制闸。
        "iqp_stage4": (kb["iqp"]["plans"],
                       "2r2rk1/1b2qppp/pQ6/1p1pP3/4nPp1/"
                       "3B4/PPP1N3/1K1R3R w - - 0 21"),
        # PLAN-011 阶段 2：Benoni A3 样本。
        "benoni": (kb["benoni"]["plans"],
                   "r2q1rk1/pp1n1ppp/3p4/2pPp3/2P1P3/2N2N2/"
                   "PP3PPP/R1BQ1RK1 w - - 0 13"),
    }

    from src.analysis.direction import direction_candidates
    from src.analysis.structure_id import applicable_mover_side

    per_situation = {}
    all_within, all_cross = [], []
    for sit_name, (plans, fen) in situations.items():
        board = chess.Board(fen)
        # 角色闸（与 decision_pipeline 同口径）：只测决策点走子方真能执行的
        # 计划。iqp 的施压方/持有方计划互斥，混测会把「对手的计划」算进可分
        # 离性。**仅当过滤后计划集非空才应用**（08.06 修，independent_analysis
        # 分歧 3）：首版无守卫，stonewall 在该 FEN 返回 "opponent" 而 KB 两条
        # 计划都是 mover_side="mover"，被过滤为空、A3 数据静默丢失，连带让
        # estimator 修复承诺的 stonewall 重审流产。守卫后 stonewall 保留全部
        # 计划（其 mover_side 标注与 applicable_mover_side 角色语义的错位是
        # KB 层问题，另行回流，不在本探针内静默吞掉）。
        arch = sit_name.split("_")[0]
        side = applicable_mover_side(board, arch)
        if side is not None:
            filtered = [p for p in plans if p.get("mover_side") == side]
            if filtered:
                plans = filtered
        within = []
        lines_by_plan = {}   # plan_name -> [pv, ...]
        for plan in plans:
            cands = direction_candidates(board, plan["direction"], top_n=10)
            line_list = []
            for root in cands[:A3_GROUP_ROOTS]:
                pv = _single_root_line(board, root, sf, depth)
                if pv is None or not pv:
                    continue
                line_list.append(pv)
            if len(line_list) >= 2:
                lines_by_plan[plan["name"]] = line_list
                for i in range(len(line_list)):
                    for j in range(i + 1, len(line_list)):
                        d = feature_distance(
                            _line_features(board, line_list[i]),
                            _line_features(board, line_list[j]))
                        within.append(d)
        # 跨计划：两计划文献执行线全部组合对
        cross = []
        names = list(lines_by_plan.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                for la in lines_by_plan[names[i]]:
                    for lb in lines_by_plan[names[j]]:
                        d = feature_distance(
                            _line_features(board, la),
                            _line_features(board, lb))
                        cross.append(d)
        ws, cs = sorted(within), sorted(cross)
        # 中位数用 statistics.median（真中位）——08.06 修（用户裁决）。
        # 原实现 `ws[len(ws)//2]` 对偶数长度列表取的是上中位（组内均 6 个
        # 距离 → 下标 3 ≈ 58 分位），跨计划 9 个（奇数）才是真中位，判据
        # 实际算「跨计划真中位 > 组内 58 分位」，比 docstring 承诺的
        # 「中位 > 中位」更严、偏向 FAIL。阶段 3 peer_review 发现、PLAN-010
        # 复盘数值复核证实，用户裁决修复并重审 stonewall/hanging。
        w_med = statistics.median(ws) if ws else None
        c_med = statistics.median(cs) if cs else None
        passed = (w_med is not None and c_med is not None
                  and c_med > w_med)
        per_situation[sit_name] = {
            "within": within, "within_median": w_med,
            "cross": cross, "cross_median": c_med, "passed": passed,
        }
        all_within += within
        all_cross += cross
        print(f"  A3 [{sit_name}] 组内中位={w_med} 跨计划中位={c_med} "
              f"{'✓' if passed else '✗'}")
        print(f"      组内={[round(d, 2) for d in within]}")
        print(f"      跨={[round(d, 2) for d in cross]}")

    passed_all = all(v["passed"] for v in per_situation.values())
    passed = passed_all
    print(f"A3 可分离性: {sum(1 for v in per_situation.values() if v['passed'])}/"
          f"{len(per_situation)} 局面通过（判据：全部通过）")
    return {"per_situation": per_situation, "passed": passed,
            "all_within": all_within, "all_cross": all_cross}


# ---------------------------------------------------------------- B3 口径一致性


def run_b3_caliber() -> dict:
    """M8 方向定义与 KB direction 是否同一套语义。

    两层检查（peer_review Minor：原实现只查引用相等，近乎无操作）：
    1. 引用层：engine_probe.direction_zone is src.direction_zone
       （单一事实来源——M8 与 KB 调同一函数对象）；
    2. 语义层：KB 所有计划的 target_zone 取值必须 ⊆ direction_zone 的
       返回值集合（{queenside, center, kingside}）——若 KB 用了
       direction_zone 表达不了的区域名（如 "qside"），即使引用同一
       函数，M8 与 KB 的「方向」语义也会静默错配。
    """
    import tools.decision_probe.engine_probe as ep
    same = ep.direction_zone is direction_zone

    kb = json.load(open(KB_PATH, encoding="utf-8"))
    kb_zones = set()
    for archetype in kb.values():
        for plan in archetype.get("plans", []):
            z = plan.get("direction", {}).get("target_zone")
            if z:
                kb_zones.add(z)
    valid_zones = {"queenside", "center", "kingside"}
    zones_ok = kb_zones <= valid_zones
    passed = same and zones_ok
    print(f"B3 口径一致性: 引用同一对象={same} | "
          f"KB target_zone 取值 {sorted(kb_zones)} ⊆ {sorted(valid_zones)}"
          f"={zones_ok}")
    return {"same_object": same, "kb_zones": sorted(kb_zones),
            "zones_ok": zones_ok, "passed": passed}


def main() -> None:
    sf = _resolve_sf()
    print(f"Stockfish: {sf} exists={os.path.exists(sf)}")
    t0 = time.time()
    results = {}
    results["a1"] = run_a1_pollution(sf)
    results["a2"] = run_a2_goals(sf)
    results["a3"] = run_a3_separability(sf)
    results["b3"] = run_b3_caliber()

    passed_all = all(r.get("passed") for r in results.values())
    verdict = ("P0-full ✅ 全部通过 → 可进入阶段 1 KB 扩容"
               if passed_all else
               "P0-full ❌ 存在未通过项 → 按 FINDINGS §3.5 走向表处置")
    print("=" * 64)
    print(verdict)
    print("=" * 64)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "p0_full_probe_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "elapsed_s": round(time.time() - t0, 1),
                   **results}, f, ensure_ascii=False, indent=1)
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
