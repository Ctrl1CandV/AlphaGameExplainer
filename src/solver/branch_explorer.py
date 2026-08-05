"""分支探索器（决策管线，ADR-020 阶段 4）。

职责（P0-full 与运行时共用）：
- `explore_forward`：用 `direction_candidates` 限根着集（searchmoves 语义，
  走 python-chess 的 `root_moves` 参数），引擎在集合内自由深搜 →
  「若采纳战略 P 的最强执行线」（立场 B 正向）；
- `explore_open`：自由 MultiPV（k=4~6）→ 反向验证 + 等强性对比 +
  执行难度（好着走廊）数据（立场 B 反向）；
- `assess_feasibility`：可行性闸——计划最优 vs 全局最优的差距，**必须调
  `equivalence_gap`**（单一事实来源，与挖掘器 M8 同口径，FINDINGS-002 P7）；
- `waiting_baseline`：反事实基线（P6 定稿）——FEN 翻转走子权的 null move
  语义，评估「不作为」的代价（轴 3 紧迫性的数据来源）。

进程管理对齐 `stockfish_analyzer.py` 既有模式：popen_uci → configure →
finally quit。不复用该模块函数：它是单线求解语义，这里要 MultiPV 分布。

颜色归一化：调用方保证 board 已归一化（走子方=白）；评估分统一换算为
**走子方视角**（正数=走子方占优），与 engine_probe 的 PvLine 约定一致。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import chess
import chess.engine

try:
    from src.analysis.direction import (
        DEFAULT_EQUIV_CP,
        direction_candidates,
        equivalence_gap,
    )
except ModuleNotFoundError:  # 直接运行自检时补充项目根到 sys.path
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    "..", "..")))
    from src.analysis.direction import (
        DEFAULT_EQUIV_CP,
        direction_candidates,
        equivalence_gap,
    )

logging.getLogger("chess.engine").setLevel(logging.CRITICAL)

# 将杀分的 cp 折算上限（对齐 tools/decision_probe/engine_probe.py 的约定）：
# python-chess 要求给 mate_score 才能把 Mate(n) 转成 int；取 10000 远大于
# 任何实战 cp 差，保证「有杀」在数值比较中永远压过「无杀」。
MATE_CP = 10000

DEFAULT_DEPTH = 18        # 前向约束线的搜索深度
DEFAULT_OPEN_K = 5        # 自由 MultiPV 候选数（ADR-020 阶段 4 的 k=4~6 取中）
DEFAULT_FEASIBLE_CP = 80  # 可行性闸初值（PLAN-009 阶段 4：50~80，取宽松端）
# 实战续走首着的最大容许损失（阶段 9 双重校验②）。PLAN-009 写「≥-30cp」，
# 即相对该局面最优着最多差 30cp——超过就认为实战这一手本身有瑕疵，不作为
# 「这个水平的棋手更多选了它」的证据注入解说。
DEFAULT_ACTUAL_LOSS_CP = 30


@dataclass
class BranchLine:
    """MultiPV 的一条候选线。"""
    move: chess.Move
    cp: int                        # 走子方视角的 cp（将杀已折算为 ±MATE_CP 量级）
    is_mate: bool = False
    mate_in: Optional[int] = None  # 正数=走子方将杀对手，负数=被将杀
    pv: List[chess.Move] = field(default_factory=list)


def _open_engine(sf_path: str) -> chess.engine.SimpleEngine:
    """启动引擎并做统一配置（线程/哈希）。

    **`Threads: 1` 是可复现性要求，不是性能取舍**（08.04 修，原值 2）。
    多线程 Stockfish 的搜索结果本质不可复现：各线程共享置换表，写入顺序
    随 OS 调度而变，同一局面同一深度每次跑出的 PV 都可能不同。实测本机
    （maroczy 决策点 depth=14）：`Threads=2` 连跑 3 次得到 3 条不同的线
    （长度 9 / 15 / 20）；`Threads=1` 连跑 3 次逐着完全一致。

    为什么这条链路必须确定性：决策管线的产出**判据建立在线的内容之上**
    ——A2 轨迹一致性（`goal_trajectory` 沿线取样）、P8 分歧深度、代价量化
    全都读 `line_pv`。线一变，判据结论跟着变：实测同一计划连跑 3 次
    `goal_ok` 翻转 True/True/False，于是「这条计划的机制是否成立」这种
    应当客观的事实，变成了一次掷硬币。阶段 9 评审因此无法复现。
    单线程慢一些（本机实测每次 analyse 增加约 1~2s，一条片子多十几秒），
    换来的是判据可复现——对「程序算事实、LLM 只表达」的边界，这是必要条件。
    """
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"Hash": 128, "Threads": 1})
    return engine


def _line_from_info(board: chess.Board, info) -> BranchLine:
    """把一次 analyse 的 info 转成 BranchLine（走子方视角 cp + mate 折算）。

    `score.relative` 已是轮到走棋一方的视角（analyse 的走棋方 = board.turn），
    无需再按颜色换算。
    """
    pv = list(info.get("pv", []))
    move = pv[0] if pv else chess.Move.null()
    score = info.get("score")
    cp, is_mate, mate_in = 0, False, None
    if score is not None:
        try:
            rel = score.relative
            m = rel.mate()
            if m is not None:
                is_mate = True
                mate_in = m
                cp = MATE_CP if m > 0 else -MATE_CP
            else:
                cp = rel.score() or 0
        except Exception:
            cp = 0
    return BranchLine(move=move, cp=cp, is_mate=is_mate, mate_in=mate_in, pv=pv)


def explore_forward(
    board: chess.Board,
    plan: dict,
    sf_path: str,
    depth: int = DEFAULT_DEPTH,
) -> Optional[BranchLine]:
    """正向：方向约束下的最强执行线（立场 B 正向）。

    `plan` 为 KB 的 plans[] 条目（含 direction）；根着集 =
    `direction_candidates(board, plan["direction"])`，引擎在集内自由深搜。
    候选集为空或引擎失败返回 None（失败安全，调用方降级）。
    """
    cands = direction_candidates(board, plan["direction"], top_n=10)
    if not cands:
        return None
    engine = None
    try:
        engine = _open_engine(sf_path)
        info = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            root_moves=cands,
        )
        return _line_from_info(board, info)
    except Exception:
        return None
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass


def explore_open(
    board: chess.Board,
    sf_path: str,
    k: int = DEFAULT_OPEN_K,
    depth: int = DEFAULT_DEPTH,
) -> List[BranchLine]:
    """反向：自由 MultiPV（立场 B 反向），供等强性/背书/好着走廊使用。"""
    engine = None
    try:
        engine = _open_engine(sf_path)
        infos = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=k,
        )
        return [_line_from_info(board, info) for info in infos]
    except Exception:
        return []
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass


def assess_feasibility(
    plan_eval: Optional[int],
    open_eval: Optional[int],
    threshold_cp: int = DEFAULT_FEASIBLE_CP,
) -> tuple:
    """可行性闸：计划最优 vs 全局最优的差距是否可接受。

    返回 (feasible, gap_cp)。**必须经 `equivalence_gap` 换算**（单一事实来源，
    与挖掘器 M8 同一口径——否则出现 FINDINGS-002 P7 的挖矿/运行时脱节）。
    任一评估缺失返回 (False, None)（失败安全）。
    """
    if plan_eval is None or open_eval is None:
        return False, None
    gap = equivalence_gap(plan_eval, open_eval)
    return gap <= threshold_cp, gap


def assess_endorsement(
    board: chess.Board,
    plans: list,
    open_lines: List[BranchLine],
    threshold: float = 0.4,
) -> dict:
    """引擎背书判定：方向内的着在 MultiPV top-k 中占比 ≥threshold 才算背书。

    背书 = 「方向客观成立」的加分证据（消化 P5）：引擎自由 MultiPV 里
    落在计划方向候选集内的首着占比——说明该方向是引擎认可的整体着法
    家族，不是 top-1 单点巧合。**背书不参与叙事主次**（主次由 KB 计划
    顺序 + 形态差异度决定），只作为方向客观性的佐证。

    返回 {plan_name: {"endorsed": bool, "in_direction": n, "total": k,
                      "ratio": float}}。open_lines 为空或 plan 无 direction
    时该计划记 endorsed=False（失败安全）。
    """
    out = {}
    if not open_lines:
        for plan in plans:
            out[plan.get("name", "?")] = {
                "endorsed": False, "in_direction": 0, "total": 0, "ratio": 0.0,
            }
        return out
    cand_sets = {}
    for plan in plans:
        name = plan.get("name", "?")
        direction = plan.get("direction")
        if not direction:
            out[name] = {"endorsed": False, "in_direction": 0,
                         "total": len(open_lines), "ratio": 0.0}
            continue
        cand_sets[name] = {
            board.san(m) for m in direction_candidates(board, direction, top_n=10)
        }
    for plan in plans:
        name = plan.get("name", "?")
        if name not in cand_sets:
            continue
        total = len(open_lines)
        in_dir = sum(1 for ln in open_lines
                     if board.san(ln.move) in cand_sets[name])
        ratio = in_dir / total if total else 0.0
        out[name] = {
            "endorsed": ratio >= threshold,
            "in_direction": in_dir,
            "total": total,
            "ratio": round(ratio, 3),
        }
    return out


def waiting_baseline(
    board: chess.Board,
    sf_path: str,
    depth: int = 12,
) -> Optional[int]:
    """反事实基线（P6 定稿）：FEN 翻转走子权的 null move 语义。

    实现：取当前局面，翻转走子权（`board.turn = not board.turn`）、清空 ep
    格（否则过路兵字段与新走子方矛盾，FEN 非法），送引擎评估——「对方连走
    两步」的评估 = 不作为的代价基线（轴 3「执行 vs 等待」的数据来源）。

    正确性边界：己方正被将军时不可用（翻转后局面非法）→ 显式检查
    `is_check()`，为真返回 None（失败安全，该维度缺席不阻塞）。
    """
    if board.is_check():
        return None
    b = board.copy()
    b.turn = not b.turn
    b.ep_square = None
    engine = None
    try:
        engine = _open_engine(sf_path)
        info = engine.analyse(b, chess.engine.Limit(depth=depth))
        score = info.get("score")
        if score is None:
            return None
        rel = score.relative
        m = rel.mate()
        if m is not None:
            return MATE_CP if m > 0 else -MATE_CP
        return rel.score() or 0
    except Exception:
        return None
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass


def assess_actual_move(
    board: chess.Board,
    move: chess.Move,
    sf_path: str,
    depth: int = 12,
    max_loss_cp: int = DEFAULT_ACTUAL_LOSS_CP,
) -> tuple:
    """实战续走首着的评估筛（阶段 9 双重校验②，P12/P19/P24）。

    返回 `(passed, loss_cp)`：`loss_cp` = 这一手相对该局面最优着的净损失
    （走子方视角，非负；0 表示它本身就是最优着）。`loss_cp <= max_loss_cp`
    时 `passed=True`。任一评估拿不到 → `(False, None)`（失败安全：拿不到
    证据就不注入，不猜）。

    **为什么必须有这道筛**：实战对照段会说「这个水平的棋手在实战中更多选了
    某条路」。若那一手本身是失误，这句话就把一个错误决策讲成了参考答案——
    比不讲更糟。PLAN-009 阶段 9 明确要求「两条都过才注入」，这是第二条。
    （第一条时限筛在阶段 0 挖掘时已完成：`mine_decision_positions` 的 G1
    读 `Event` 头官方分类，实测 20000 局里 18582 局因 blitz/bullet 被排除，
    故产品链路无需重复筛——见 PLAN-009 阶段 9「本阶段无需重复筛」。）

    判据用**相对损失**而非绝对 cp：局面本身可能已经劣势（如 -200cp），此时
    绝对阈值会把所有着法一律判失误；「未失误」的正确含义是「没有比该局面
    最优着差太多」。两次 analyse 同深度同引擎配置（`Threads=1` 确定性），
    结果可复现。
    """
    if move not in board.legal_moves:
        return False, None
    engine = None
    try:
        engine = _open_engine(sf_path)
        limit = chess.engine.Limit(depth=depth)

        def _cp(info) -> Optional[int]:
            score = info.get("score")
            if score is None:
                return None
            rel = score.relative
            m = rel.mate()
            if m is not None:
                return MATE_CP if m > 0 else -MATE_CP
            return rel.score() or 0

        best_cp = _cp(engine.analyse(board, limit))
        move_cp = _cp(engine.analyse(board, limit, root_moves=[move]))
        if best_cp is None or move_cp is None:
            return False, None
        loss = max(0, best_cp - move_cp)
        return loss <= max_loss_cp, loss
    except Exception:  # noqa: BLE001
        return False, None
    finally:
        if engine is not None:
            try:
                engine.quit()
            except Exception:
                pass


if __name__ == "__main__":
    """阶段 4 单元测试（PLAN-009：mock 或固定局面）。

    四项：
    1. 方向约束只从给定根着集起步——explore_forward 的 pv 首着 ∈
       direction_candidates（固定局面 + 真实引擎）；
    2. 可行性闸在「方向次优」局面正确标记——中心突破在 carlsbad 原
       FEN 不可行（e4 被 Nf6 盯防），assess_feasibility 应返回
       feasible=False 且 gap 大；
    3. 背书判据在「top-1 巧合落点」不误判——构造 MultiPV 中仅 1 个着
       落在方向内的样本，占比 <40% → 不背书；
    4. waiting_baseline 在被将军局面正确返回 None。
    """
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    "..", "..")))
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))

    sf = os.getenv("STOCKFISH_PATH", "")
    if not os.path.isabs(sf):
        sf = os.path.normpath(os.path.join(os.getcwd(), sf))

    results = []

    # --- 1. 方向约束起步集 ---
    b = chess.Board("r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/2NBPN2/PPQ2PPP/"
                    "R4RK1 w - - 8 11")
    plan = {"name": "少数派攻击", "direction": {"pawn_files": ["a", "b"],
            "target_zone": "queenside", "break_squares": ["b5"]}}
    cands = {b.san(m) for m in direction_candidates(b, plan["direction"])}
    line = explore_forward(b, plan, sf, depth=14)
    first_in = line is not None and line.pv and b.san(line.pv[0]) in cands
    results.append(("方向约束起步集", first_in,
                    f"首着={b.san(line.pv[0]) if line and line.pv else '?'}"))

    # --- 2. 可行性闸正确标记（纯函数测试——不依赖引擎，稳定）---
    # 次优：计划 -50 vs 全局 50 → gap 100 > 80 → 不可行
    feas_bad, gap_bad = assess_feasibility(-50, 50)
    # 近等强：计划 40 vs 全局 50 → gap 10 ≤ 80 → 可行
    feas_ok, gap_ok = assess_feasibility(40, 50)
    passed2 = (not feas_bad and gap_bad == 100
               and feas_ok and gap_ok == 10)
    results.append(("可行性闸次优/近等正确标记", passed2,
                    f"次优(-50,50)->({feas_bad},{gap_bad}) "
                    f"近等(40,50)->({feas_ok},{gap_ok})"))

    # --- 3. 背书判据「top-1 巧合落点」不误判 ---
    # 构造：MultiPV 5 条线里只有 1 条首着落在方向内（占比 20% < 40%）
    fake_lines = [BranchLine(move=chess.Move.from_uci("g1f3"), cp=30 - i,
                             pv=[chess.Move.from_uci("g1f3")])
                  for i in range(5)]
    fake_lines[4] = BranchLine(move=chess.Move.from_uci("a2a4"), cp=25,
                               pv=[chess.Move.from_uci("a2a4")])
    end = assess_endorsement(b, [plan], fake_lines)
    endorsed = end["少数派攻击"]["endorsed"]
    results.append(("背书 top-1 巧合不误判", not endorsed,
                    f"ratio={end['少数派攻击']['ratio']} "
                    f"in={end['少数派攻击']['in_direction']}/5"))

    # --- 4. waiting_baseline 被将军返回 None ---
    # 构造被将军局面：黑象 b4 沿 c3-d2-e1 斜线将军白王 e1
    # （d2 必须无兵——首版 FEN d2 有白兵挡住象线，is_check 恒 False）
    check_fen = "rnbqk1nr/pppp1ppp/8/4p3/1b2P3/5N2/PPP2PPP/RNBQK2R w KQkq - 0 4"
    b_check = chess.Board(check_fen)
    if b_check.is_check():
        base = waiting_baseline(b_check, sf, depth=10)
        results.append(("waiting_baseline 被将军 None", base is None,
                        f"got={base}"))
    else:
        results.append(("waiting_baseline 被将军 None", False, "局面非将军"))

    ok = True
    for name, passed, detail in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        ok &= passed
    print("阶段 4 单元测试:", "全部通过" if ok else "存在失败")
    sys.exit(0 if ok else 1)
