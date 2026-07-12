"""Puzzle 战术关键手定位器。

目标：基于棋盘事实，为核心标签（机理类）确定"真正体现该战术的那一步"，
解决 "核心标签选对了但关键手讲错" 的问题（典型场景：sacrifice 被误选成
前面那步将军）。

设计原则：
 1) 完全在 Python 层用棋盘事实做判定，不让 LLM 猜；
 2) 每类标签一个独立评分函数，返回 (score, reason)；多候选取最高分；
 3) 评分失败 / 分数全为 0 时回退到通用信号（将军/将杀/吃高价值子）；
 4) 不影响残局链路。

从 storyboard.py 提取。第二阶段重点修复对象（根因D，puzzle key_move error 82-87%）。
"""
from src.chess_utils.tactic import is_fork, is_pin, is_skewer, is_discovered
from src.chess_utils.material import PIECE_VALUES
from src.common import Logger
from typing import List
import chess

# 评估类标签：只描述结果，不参与"关键手"定位（避免 crushing 抢机理叙事的关键手）
_OUTCOME_THEMES = {
    "crushing", "advantage", "equality", "winning",
    "long", "short", "oneMove", "veryLong",
}


def collect_puzzle_move_facts(board: chess.Board, moves: List[chess.Move]):
    """复盘整条解法，抽取每步的棋盘事实，供关键手评分使用。

    返回 list[dict]，每个元素是一步的事实。包含但不限于：
      - idx: 1-based 步号
      - san / uci
      - board_before / board_after（独立 chess.Board 实例，避免外部修改污染）
      - mover_color / piece_type（主动行棋方与移动子种类）
      - is_check / is_capture / is_checkmate
      - captured_piece_type（被吃子种类，无则为 None）
      - gives_mover_attacked（移动子落点后是否被对方攻击）
      - material_delta_white（白方视角的子力净值变化，单位为 PIECE_VALUES）
      - legal_reply_count_after（走后对方合法应招数）

    不抛异常：所有字段用 try/except 兜底，缺失即为 None/0。
    """
    facts = []
    temp = board.copy()
    for i, move in enumerate(moves):
        board_before = temp.copy()
        mover_color = temp.turn
        is_check = bool(temp.gives_check(move))
        is_capture = bool(temp.is_capture(move))
        is_checkmate = False
        piece_type = board_before.piece_at(move.from_square)
        mover_pt = piece_type.piece_type if piece_type else None

        captured_piece_type = None
        if is_capture:
            if temp.is_en_passant(move):
                captured_piece_type = chess.PAWN
            else:
                cap = temp.piece_at(move.to_square)
                if cap:
                    captured_piece_type = cap.piece_type

        temp.push(move)
        board_after = temp.copy()
        is_checkmate = board_after.is_checkmate()

        # 子力净值变化（白方视角）
        mat_before_white = 0
        mat_after_white = 0
        try:
            for sq, p in board_before.piece_map().items():
                mat_before_white += PIECE_VALUES.get(p.piece_type, 0) * (1 if p.color == chess.WHITE else -1)
            for sq, p in board_after.piece_map().items():
                mat_after_white += PIECE_VALUES.get(p.piece_type, 0) * (1 if p.color == chess.WHITE else -1)
            material_delta_white = mat_after_white - mat_before_white
        except Exception:
            material_delta_white = 0

        gives_mover_attacked = False
        try:
            if mover_pt is not None and mover_pt != chess.KING:
                enemy = not mover_color
                if board_after.is_attacked_by(enemy, move.to_square):
                    gives_mover_attacked = True
        except Exception:
            gives_mover_attacked = False

        legal_reply_count_after = 0
        try:
            legal_reply_count_after = sum(1 for _ in board_after.legal_moves)
        except Exception:
            pass

        san = ""
        try:
            san = board_before.san(move)
        except Exception:
            san = move.uci() if move else ""

        facts.append({
            "idx": i + 1,
            "move": move,
            "san": san,
            "board_before": board_before,
            "board_after": board_after,
            "mover_color": mover_color,
            "piece_type": mover_pt,
            "is_check": is_check,
            "is_capture": is_capture,
            "is_checkmate": is_checkmate,
            "captured_piece_type": captured_piece_type,
            "gives_mover_attacked": gives_mover_attacked,
            "material_delta_white": material_delta_white,
            "legal_reply_count_after": legal_reply_count_after,
        })
    return facts


def _score_mate_key_move(theme_key, fact, facts, mover_color):
    """将杀/将型：直接取最后一步（将杀那步）。"""
    if not fact.get("is_checkmate"):
        return 0.0, ""
    return 1.0, "本手完成将杀，是战术的终局兑现。"


def _score_fork_key_move(theme_key, fact, facts, mover_color):
    """叉击：走子后该子同时攻击两个高价值目标。"""
    try:
        if is_fork(fact["board_before"], fact["move"], fact["board_after"]):
            return 0.9, "本手用一子同时叉住两个目标，是本战术的叉击手。"
    except Exception:
        pass
    return 0.0, ""


def _score_pin_key_move(theme_key, fact, facts, mover_color):
    """牵制：走子后建立/强化牵制关系。"""
    try:
        if is_pin(fact["board_before"], fact["move"], fact["board_after"]):
            return 0.9, "本手建立了牵制（被牵子无法擅动），是本战术的关键。"
    except Exception:
        pass
    return 0.0, ""


def _score_skewer_key_move(theme_key, fact, facts, mover_color):
    """串击：走子后形成串击。"""
    try:
        if is_skewer(fact["board_before"], fact["move"], fact["board_after"]):
            return 0.9, "本手形成串击，逼对方高价值子先动、暴露低价值子。"
    except Exception:
        pass
    return 0.0, ""


def _score_discovered_key_move(theme_key, fact, facts, mover_color):
    """闪击：移开遮挡子，露出远射攻击线。"""
    try:
        if is_discovered(fact["board_before"], fact["move"], fact["board_after"]):
            return 0.9, "本手移开遮挡子，露出身后远射子的攻击线，是闪击手。"
    except Exception:
        pass
    return 0.0, ""


def _score_double_check_key_move(theme_key, fact, facts, mover_color):
    """双将：走子后同时存在两路将军。"""
    if not fact.get("is_check"):
        return 0.0, ""
    board_after = fact["board_after"]
    enemy_king = board_after.king(not mover_color)
    if enemy_king is None:
        return 0.0, ""
    attackers = [p for p in board_after.attackers(mover_color, enemy_king)]
    if len(attackers) >= 2:
        return 0.95, "本手产生双将，对方王只能移动，无法用吃子或垫子解将。"
    return 0.0, ""


def _score_promotion_key_move(theme_key, fact, facts, mover_color):
    """升变：走到底线的那一步（或同回合吃子+升变）。"""
    move = fact.get("move")
    if move is None:
        return 0.0, ""
    if move.promotion:
        return 1.0, "本手完成升变，把兵变成更有力的子（如后），是战术的兑现点。"
    return 0.0, ""


def _score_advanced_pawn_key_move(theme_key, fact, facts, mover_color):
    """通路兵：推动己方兵逼近升变 / 清除升变路径障碍。"""
    if fact.get("piece_type") != chess.PAWN:
        return 0.0, ""
    move = fact.get("move")
    if move is None:
        return 0.0, ""
    to_rank = chess.square_rank(move.to_square)
    if move.promotion:
        return 0.0, ""
    moved_toward_promotion = (
        (mover_color == chess.WHITE and to_rank >= 5) or
        (mover_color == chess.BLACK and to_rank <= 2)
    )
    if moved_toward_promotion and not fact.get("is_capture"):
        return 0.8, "本手推进己方兵逼近升变格，是通路兵战术的关键推进。"
    if fact.get("is_capture") and fact.get("captured_piece_type") == chess.PAWN:
        return 0.7, "本手吃掉前方障碍兵，己方通路兵推进道路被打通。"
    return 0.0, ""


def _score_sacrifice_key_move(theme_key, fact, facts, mover_color):
    """弃子：主动让高价值子被对方吃/进入对方攻击范围，后续获得决定性收益。"""
    if not fact.get("move"):
        return 0.0, ""
    if fact.get("is_checkmate"):
        return 0.0, ""
    if fact.get("piece_type") in (None, chess.KING):
        return 0.0, ""

    score = 0.0
    reasons = []
    mover_pt = fact.get("piece_type")
    mover_value = PIECE_VALUES.get(mover_pt, 0)
    if mover_value < 2:
        return 0.0, ""

    if fact.get("is_capture") or fact.get("gives_mover_attacked"):
        score += 0.4
        reasons.append("主动让高价值子进入对方攻击范围")

    delta = fact.get("material_delta_white", 0)
    if mover_color == chess.WHITE and delta < 0:
        score += 0.2
        reasons.append("本手暂时净亏子力")
    elif mover_color == chess.BLACK and delta > 0:
        score += 0.2
        reasons.append("本手暂时净亏子力")

    for j in range(fact["idx"], min(fact["idx"] + 2, len(facts))):
        nf = facts[j]
        if nf.get("is_capture") and nf.get("captured_piece_type") == mover_pt:
            score += 0.3
            reasons.append("对方随后回吃了弃子")
            break

    for j in range(fact["idx"], min(fact["idx"] + 4, len(facts))):
        nf = facts[j]
        if nf.get("is_checkmate"):
            score += 0.3
            reasons.append("通过弃子最终完成将杀")
            break
        if nf.get("legal_reply_count_after", 99) <= 2:
            score += 0.2
            reasons.append("弃子后对方合法应招明显收缩")
            break

    if score <= 0:
        return 0.0, ""

    reason = "本手是弃子： " + "，".join(reasons) + "，是本题的弃子关键手。"
    return min(score, 1.0), reason


def _score_capturing_defender_key_move(theme_key, fact, facts, mover_color):
    """消除防守子：本手是吃子 + 后续 1-2 ply 内吃到原保护子。"""
    if not fact.get("is_capture"):
        return 0.0, ""
    idx = fact["idx"]
    nxt_idx = idx
    if 0 <= nxt_idx < len(facts):
        nxt = facts[nxt_idx]
        if nxt.get("is_capture") and nxt.get("idx", 0) == idx + 1:
            return 0.7, "本手先吃掉防守子，下一手即可白吃目标，是消除防守子的关键。"
    if not fact.get("gives_mover_attacked"):
        return 0.5, "本手主动吃掉对方防守子，使目标失去保护。"
    return 0.0, ""


def _score_deflection_key_move(theme_key, fact, facts, mover_color):
    """驱离/引离：迫使对方子离开关键职责。"""
    if fact.get("is_checkmate"):
        return 0.0, ""
    s, r = _score_sacrifice_key_move(theme_key, fact, facts, mover_color)
    if s > 0.4 and r:
        return 0.6, "本手通过弃子把对方防守子引离关键岗位。" + r
    return 0.0, ""


def _score_clearance_key_move(theme_key, fact, facts, mover_color):
    """腾线/清障：移开己方子打开远射线路。"""
    if fact.get("is_capture") and fact.get("captured_piece_type") == chess.PAWN:
        return 0.5, "本手清除线路上的障碍，为后续远射打开通路。"
    if not fact.get("is_capture"):
        try:
            before_att = 0
            after_att = 0
            for sq, p in fact["board_before"].piece_map().items():
                if p.color == mover_color and p.piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP):
                    before_att += len(fact["board_before"].attacks(sq))
            for sq, p in fact["board_after"].piece_map().items():
                if p.color == mover_color and p.piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP):
                    after_att += len(fact["board_after"].attacks(sq))
            if after_att - before_att >= 6:
                return 0.6, "本手移开己方子，打开己方远射子攻击线。"
        except Exception:
            pass
    return 0.0, ""


def _score_intermezzo_key_move(theme_key, fact, facts, mover_color):
    """中间手：解对方威胁前先插入更强的一手。"""
    if fact.get("is_checkmate"):
        return 0.0, ""
    if fact.get("is_check") and fact.get("is_capture"):
        return 0.6, "本手在对方威胁前插入带将军的吃子，是典型的中间手。"
    return 0.0, ""


def _score_exposed_king_key_move(theme_key, fact, facts, mover_color):
    """暴露王：利用对方王暴露的弱点发动攻击（将军/逼迫）。"""
    if fact.get("is_check") and not fact.get("is_capture"):
        return 0.6, "本手对暴露的王发动将军，迫其在开阔地带逃生。"
    if fact.get("is_check") and fact.get("is_capture"):
        return 0.7, "本手对暴露的王发动弃子将军，强制改变局面走向。"
    return 0.0, ""


def _score_kingside_attack_key_move(theme_key, fact, facts, mover_color):
    """王翼进攻：突破王前防线。"""
    if not fact.get("is_capture"):
        return 0.0, ""
    move = fact.get("move")
    if move is None:
        return 0.0, ""
    to_file = chess.square_file(move.to_square)
    if 3 <= to_file <= 5 and fact.get("gives_mover_attacked"):
        return 0.7, "本手在王翼区域弃子，撕开对方王前防线。"
    return 0.0, ""


def _score_attacking_f2_f7_key_move(theme_key, fact, facts, mover_color):
    """攻击 f2/f7：直接落到 f2/f7 攻击、或者弃子突破该格。"""
    move = fact.get("move")
    if move is None:
        return 0.0, ""
    to_sq = move.to_square
    if to_sq in (chess.F2, chess.F7):
        return 0.9, "本手直接落到 f2/f7，突破对方开局薄弱防线。"
    if fact.get("is_capture") and fact.get("gives_mover_attacked"):
        return 0.6, "本手通过弃子制造对 f2/f7 区域的攻击压力。"
    return 0.0, ""


def _score_hanging_piece_key_move(theme_key, fact, facts, mover_color):
    """悬子：直接吃掉对方无保护子。"""
    if not fact.get("is_capture"):
        return 0.0, ""
    if not fact.get("gives_mover_attacked"):
        return 0.8, "本手吃掉对方无保护的子（悬子），净赚一子。"
    return 0.0, ""


def _score_overloading_key_move(theme_key, fact, facts, mover_color):
    """过载：对方一子承担多个防守任务，本手攻击使其无法兼顾。"""
    s, r = _score_sacrifice_key_move(theme_key, fact, facts, mover_color)
    if s >= 0.5:
        return 0.5, "本手利用过载：对方防守子无法兼顾两个任务，被迫放弃其一。"
    return 0.0, ""


# 标签 → 评分函数的注册表。未列出的标签走通用回退。
_THEME_SCORERS = {
    "fork": _score_fork_key_move,
    "pin": _score_pin_key_move,
    "skewer": _score_skewer_key_move,
    "discoveredAttack": _score_discovered_key_move,
    "doubleCheck": _score_double_check_key_move,
    "discoveredCheck": _score_discovered_key_move,
    "mate": _score_mate_key_move,
    "mateIn1": _score_mate_key_move,
    "mateIn2": _score_mate_key_move,
    "mateIn3": _score_mate_key_move,
    "backRankMate": _score_mate_key_move,
    "smotheredMate": _score_mate_key_move,
    "promotion": _score_promotion_key_move,
    "sacrifice": _score_sacrifice_key_move,
    "capturingDefender": _score_capturing_defender_key_move,
    "deflection": _score_deflection_key_move,
    "attraction": _score_deflection_key_move,
    "clearance": _score_clearance_key_move,
    "intermezzo": _score_intermezzo_key_move,
    "overloading": _score_overloading_key_move,
    "exposedKing": _score_exposed_king_key_move,
    "kingsideAttack": _score_kingside_attack_key_move,
    "attackingF2F7": _score_attacking_f2_f7_key_move,
    "hangingPiece": _score_hanging_piece_key_move,
    "advancedPawn": _score_advanced_pawn_key_move,
}


def _fallback_key_move_idx(facts):
    """通用回退：取将军/将杀/吃高价值子中分数最高的一步。"""
    best_idx = 1 if facts else 0
    best_score = -1.0
    for fact in facts:
        s = 0.0
        if fact.get("is_checkmate"):
            s = 1.0
        elif fact.get("is_check"):
            s = 0.6
        elif fact.get("is_capture"):
            cap_v = PIECE_VALUES.get(fact.get("captured_piece_type"), 0)
            s = 0.3 + 0.05 * cap_v
        if s > best_score:
            best_score = s
            best_idx = fact["idx"]
    return best_idx


def locate_theme_key_moves(facts, effective_themes, main_theme_key):
    """对 effective_themes 中的每个标签定位其关键手。

    返回 dict：
      {
        "<theme_key>": {"idx": int, "score": float, "reason": str},
        ...
        "core": {...},
        "key_move_idx": int,
        "key_move_san": str,
        "key_move_reason": str,
      }
    """
    result = {}
    if not facts or not effective_themes:
        idx0 = _fallback_key_move_idx(facts) if facts else 0
        san0 = facts[idx0 - 1]["san"] if facts and 0 < idx0 <= len(facts) else ""
        return {
            "core": {"idx": idx0, "score": 0.0, "reason": "通用回退：选最具决定性的一步。"},
            "key_move_idx": idx0,
            "key_move_san": san0,
            "key_move_reason": "通用回退：选最具决定性的一步。",
        }

    mover_color = facts[0]["mover_color"] if facts else chess.WHITE

    for theme_key in effective_themes:
        if theme_key in _OUTCOME_THEMES:
            continue
        scorer = _THEME_SCORERS.get(theme_key)
        if scorer is None:
            continue
        best = None
        for fact in facts:
            try:
                score, reason = scorer(theme_key, fact, facts, mover_color)
            except Exception as e:
                Logger.warn(f"标签 {theme_key} 评分异常: {e}")
                continue
            if score <= 0:
                continue
            if best is None or score > best["score"]:
                best = {"idx": fact["idx"], "score": score, "reason": reason}
        if best is not None:
            result[theme_key] = best

    core_loc = result.get(main_theme_key)
    if core_loc is None:
        idx0 = _fallback_key_move_idx(facts)
        san0 = facts[idx0 - 1]["san"] if 0 < idx0 <= len(facts) else ""
        core_loc = {
            "idx": idx0,
            "score": 0.0,
            "reason": "通用回退：选最具决定性的一步。",
        }
    result["core"] = core_loc
    result["key_move_idx"] = core_loc["idx"]
    san_key = ""
    if 0 < core_loc["idx"] <= len(facts):
        san_key = facts[core_loc["idx"] - 1]["san"]
    result["key_move_san"] = san_key
    result["key_move_reason"] = core_loc["reason"]
    return result
