"""方向与等强的单一事实来源（决策管线，ADR-020 §决策 7）。

「方向」「等强」两个概念在全链路只有一处定义，本模块是它们的宿主：

- `direction_zone` / `direction_score` / `direction_candidates`：方向（选根着用）
- `equivalence_gap` / `is_equivalent`：等强

挖掘器、可行性闸、引擎背书、可分离性判定、口径一致性全部调本模块，
禁止各模块自行实现同类判定——否则必然出现 FINDINGS-002 P7 那类
「挖矿口径 Go 通过、运行时口径 80% 降级」的脱节。

**正式化说明（2026-08-03）**：本模块由 `tools/decision_probe/engine_probe.py`
的对应实现迁移而来（探针已改 import 本模块，删除本地副本——单一事实来源
不允许双实现）。同时按 B2 实测补了 `outpost_squares` 打分维度（见 direction_score
第 7 条注释）。
"""
from __future__ import annotations

from typing import List

import chess

# M8/可行性闸的「近等强」阈值（ADR-020 与 PLAN-009 阶段 4 的初值，待数据校准）。
DEFAULT_EQUIV_CP = 60

ZONE_QUEENSIDE = "queenside"
ZONE_CENTER = "center"
ZONE_KINGSIDE = "kingside"


def equivalence_gap(cp_a: int, cp_b: int) -> int:
    """两个评估之间的差距（cp，非负）。

    这是「等强」概念在全链路的**唯一**定义。M5（是否有唯一好着）、
    M8（是否多个近等强首着）、可行性闸（计划最优 vs 全局最优）、
    A3（线间比较）全部通过本函数换算，禁止各处自行相减或自定阈值语义。
    """
    return abs(cp_a - cp_b)


def is_equivalent(cp_a: int, cp_b: int, threshold_cp: int = DEFAULT_EQUIV_CP) -> bool:
    """在给定阈值下两个评估是否算「近等强」。"""
    return equivalence_gap(cp_a, cp_b) <= threshold_cp


def direction_zone(move: chess.Move) -> str:
    """一个着法指向哪个战略区域，以**目标格**为准（落点比出发点更能代表意图）。

    这是「方向」概念在全链路的唯一定义。分区按目标格 file：
    a~c = 后翼，d~e = 中心，f~h = 王翼。

    颜色无关：`board.mirror()` 是上下翻转 + 颜色互换，不改变 file
    （已实测，见 FINDINGS-002 P22），故归一化前后本函数结果一致。
    """
    f = chess.square_file(move.to_square)
    if f <= 2:
        return ZONE_QUEENSIDE
    if f <= 4:
        return ZONE_CENTER
    return ZONE_KINGSIDE


def _pressure_delta(board: chess.Board, move: chess.Move, targets: List[int]) -> int:
    """走这一着后，走子方对目标格集合的攻击数净增量。

    「围攻某个格子」是文献计划的第三种常见表达（前两种是兵推进方向、目标区域）。
    典型例：IQP 施压方要围攻 d5 孤兵，关键着是把子力送到控制 d5 的格子
    （Nc3/Nc5/Bf4/Qd3…）——这些格子散布在后翼/中心/王翼三个区，
    用单一 target_zone 无法表达。A1 召回验证正是靠这条测出 schema 缺维度。
    """
    mover = board.turn
    before = sum(len(board.attackers(mover, sq)) for sq in targets)
    tmp = board.copy(stack=False)
    tmp.push(move)
    after = sum(len(tmp.attackers(mover, sq)) for sq in targets)
    return after - before


def direction_score(board: chess.Board, move: chess.Move, direction: dict) -> float:
    """一个着法与某战略方向的对齐度（ADR-020 §决策 7 的单一事实来源函数）。

    `direction` 即 KB 的 `plans[].direction` 字段，形如：
        {"pawn_files": ["b"], "target_zone": "queenside",
         "break_squares": ["b5"], "pressure_squares": [], "outpost_squares": []}

    **语义是「选根着」不是「验线」**（FINDINGS-002 P1 / A2 的 schema 拆分理由）：
    本函数只用于挑出进 `searchmoves` 的候选根着集，验证「整条线是否实现了计划」
    是 `structural_goal` 的职责，两者形式完全不同，不可混用。

    打分维度（加权求和，非归一化——只用于排序取 top-N）：
      0. 兵推进落点黑名单      直接 0 分  排除「推进」语义（保持类计划）
      1. 目标兵线上的兵推进      +3.0   计划的直接执行手
      2. 落点在目标区域          +1.0   方向一致
      3. 重子调到目标兵线        +2.0   **计划的准备手**（关键，见下）
      4. 命中 break_squares      +2.0   文献点明的突破格
      5. 轻子走向目标区域        +0.5   弱支持
      6. 走后攻击到施压目标格    +2.5   围攻类计划（pressure_squares）
      7. 轻子落点命中据点格      +1.0   中心据点占领（outpost_squares）

    第 0 条是 08.04 实测补的排除维度：保持类计划（保持悬兵/保持孤兵——
    no pawn_files + 有 outpost）若只配 `target_zone`，其「落点在目标区域
    +1.0」会把**推进兵的走法**（如悬兵的 d4-d5）也捞进候选集——导致
    「保持悬兵」首着变成 d5 推进、与「推进悬兵」画面雷同（视频只演示
    一个方案）。配 `exclude_pawn_targets`（如 ["d5", "c5"]）后，兵推进
    落入这些格直接 0 分——保持类候选集只剩中心组织手与前哨占位，
    与推进类计划的首着自然区分。默认缺省为空集，既有计划打分行为不变。

    第 3 条是本函数设计上最要紧的一维。少数派攻击的文献执行序列是
    「车调 b/c 线 → 推 b4 → b5 兑掉 → 施压孤兵」——**准备手占了前半段**。
    若只给兵推进打分，A1 召回必然失败（正确执行着的前半段全被漏掉），
    而 A1 召回是前向机制成立的必要条件。

    第 7 条是 B2 实测（2026-08-03）补的维度：IQP 施压方的 Ne5 在非 fianchetto
    局面（白象在 c4/d3，无 g2 象斜线可打通）对 d5 无攻击增量，第 6 条抓不到，
    被漏出候选集——但 Ne5 是中心据点占领，是施压计划的常见执行手。
    `outpost_squares`（轻子占领的中心控制格）与 `pressure_squares`（被围攻的
    目标格）互补：前者抓「据点」，后者抓「围攻」。
    """
    score = 0.0
    to_sq = move.to_square
    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0.0

    # 0. 兵推进落点黑名单（保持类计划：排除「推进」语义——默认缺省为空集）
    exclude_pawn_targets = {
        chess.parse_square(s) for s in direction.get("exclude_pawn_targets", [])
        if len(s) == 2 and s[0] in chess.FILE_NAMES and s[1].isdigit()
    }
    if piece.piece_type == chess.PAWN and to_sq in exclude_pawn_targets:
        return 0.0

    target_files = {
        chess.FILE_NAMES.index(f) for f in direction.get("pawn_files", [])
        if f in chess.FILE_NAMES
    }
    target_zone = direction.get("target_zone", "")
    break_squares = {
        chess.parse_square(s) for s in direction.get("break_squares", [])
        if len(s) == 2 and s[0] in chess.FILE_NAMES and s[1].isdigit()
    }
    outpost_squares = {
        chess.parse_square(s) for s in direction.get("outpost_squares", [])
        if len(s) == 2 and s[0] in chess.FILE_NAMES and s[1].isdigit()
    }

    to_file = chess.square_file(to_sq)

    # 1. 目标兵线上的兵推进
    if piece.piece_type == chess.PAWN and to_file in target_files:
        score += 3.0

    # 2. 落点在目标区域
    if target_zone and direction_zone(move) == target_zone:
        score += 1.0

    # 3. 重子调到目标兵线（计划的准备手）
    if piece.piece_type in (chess.ROOK, chess.QUEEN) and to_file in target_files:
        score += 2.0

    # 4. 文献点明的突破格
    if to_sq in break_squares:
        score += 2.0

    # 5. 轻子走向目标区域
    if piece.piece_type in (chess.KNIGHT, chess.BISHOP) and target_zone:
        if direction_zone(move) == target_zone:
            score += 0.5

    # 6. 走后攻击到「施压目标格」（+2.5）
    #
    # 这一维是 A1 首轮实测（召回 82.4% < 90%）暴露出来的必需维度。
    # IQP「对孤兵施压」的文献执行着 Nc5 / Nc3 得分 0.0 被漏掉——因为它们落在
    # file c（queenside），而计划的 target_zone 是 center，前五维全不命中。
    # 根因：「围攻某个格子」这类计划的执行着**天然散布在多个区域**（施压 d5 的
    # 手段有 Nc3/Nc5/Ne5/Bf4/Rd1，横跨三个 zone），单一 target_zone 表达不了。
    # 用走后局面判 attackers：马跳 c5 是否真的攻到 d5 取决于落点，只有 push
    # 之后才能确定。
    pressure_squares = [
        chess.parse_square(s) for s in direction.get("pressure_squares", [])
        if len(s) == 2 and s[0] in chess.FILE_NAMES and s[1].isdigit()
    ]
    if pressure_squares:
        # 用「攻击数净增量」而非「是否被攻击」——首版二值守卫因 d5 早已被
        # d1 后攻击而恒 False，整维静默失效。净增量对「已有 1 个攻击者、
        # 本着再加 1 个」这种真实加压手同样敏感。
        delta = _pressure_delta(board, move, pressure_squares)
        if delta > 0:
            score += 2.5 * min(delta, 2)  # 净增 ≥2 个攻击者封顶，避免单着刷分

    # 7. 轻子落点命中「据点格」（+1.0）——见函数 docstring 第 7 条
    if piece.piece_type in (chess.KNIGHT, chess.BISHOP) and to_sq in outpost_squares:
        score += 1.0

    return score


def direction_candidates(
    board: chess.Board, direction: dict, top_n: int = 10
) -> List[chess.Move]:
    """按 `direction_score` 取 top-N 合法着，作为 `searchmoves` 候选集。

    这是前向约束的入口：ADR-020 立场 B 的「只约束首着方向、不约束整条线」
    就落在这里——限定根着集合后，引擎在集合内自由深搜。

    **并列包含式截断（A1 实测得出的必需规则）**：不在同分组中间切断。
    实测现场：卡尔斯巴德少数派攻击的得分分布是 4.0×4 / 3.5×2 / 3.0×5，
    `top_n=10` 恰好切进 3.0 那一组，把文献执行着 `Rab1` 排除在外——而它与
    组内另外 4 个着**同分**，谁进谁出纯属排序偶然。这类失败不是谓词不准，
    是截断规则的人为假象，会让 A1 召回从 100% 假摔到 94.4%。

    解法：取到第 N 名的分数后，把所有 ≥ 该分数的着全部纳入。候选集可能略大
    于 N，这在 `searchmoves` 语义下无害（引擎自己选最强），而漏掉正确执行着
    是致命的——召回优先，精度由引擎兜底。
    """
    scored = []
    for mv in board.legal_moves:
        s = direction_score(board, mv, direction)
        if s > 0:
            scored.append((s, mv))
    if not scored:
        return []
    scored.sort(key=lambda x: -x[0])
    threshold = scored[min(top_n - 1, len(scored) - 1)][0]
    return [mv for s, mv in scored if s >= threshold]
