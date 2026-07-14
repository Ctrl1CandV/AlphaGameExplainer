from src.analysis.endgame_kb import describe_endgame, get_forbidden_concepts, match as match_endgame
from src.common import CompressedStep, Logger, AnalyzedMove, PIECE_VALUES, piece_cn
from src.chess_utils.position import piece_square as _piece_square, piece_squares as _piece_squares
from src.chess_utils.material import material_score as _material_score, color_name as _color_name, side_material_desc as _side_material_desc
from src.analysis.insight_extractor import extract_for_compressed
from src.storyboard.compressor import _role_meta
from typing import List, Optional, Tuple
import chess

LONG_MOVE_THRESHOLD = 18
COMPACT_NODE_THRESHOLD = 7


def _piece_label(piece_type: chess.PieceType) -> str:
    return piece_cn(piece_type)

def _transition_summary(fen_before: str, fen_after: str) -> str:
    b1 = chess.Board(fen_before)
    b2 = chess.Board(fen_after)
    parts = []
    tracked = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    for color, name in ((chess.WHITE, "白"), (chess.BLACK, "黑")):
        for piece_type in tracked:
            before_sqs = _piece_squares(b1, color, piece_type)
            after_sqs = _piece_squares(b2, color, piece_type)
            if before_sqs == after_sqs:
                continue
            label = _piece_label(piece_type)
            if before_sqs and after_sqs:
                before_text = "、".join(chess.square_name(sq) for sq in before_sqs)
                after_text = "、".join(chess.square_name(sq) for sq in after_sqs)
                parts.append(f"{name}{label}{before_text}→{after_text}")
            elif before_sqs and not after_sqs:
                parts.append(f"{name}{label}消失")
            elif not before_sqs and after_sqs:
                after_text = "、".join(chess.square_name(sq) for sq in after_sqs)
                parts.append(f"{name}{label}出现在{after_text}")
    return "；".join(parts) if parts else "起止局面结构没有实质变化，主要是反复试探与等招"

def _krpkr_phase_hint(board_before: chess.Board, board_after: chess.Board, role_meta: dict, same_position: bool) -> Tuple[str, str]:
    if same_position:
        return "反复试探", "这段变化首尾回到同一局面，本质是调车试探与等待，没有形成真正突破"
    strong = role_meta.get("strong_color")
    weak = role_meta.get("weak_color")
    if strong is None or weak is None:
        return "", ""
    strong_pawn_before = _piece_square(board_before, strong, chess.PAWN)
    strong_pawn_after = _piece_square(board_after, strong, chess.PAWN)
    weak_king_before = board_before.king(weak)
    strong_king_before = board_before.king(strong)
    if strong_pawn_before is None:
        return "残局转换", "有兵方的兵已经不在棋盘上，讲解重点转为转换后的子力残局"
    step = 1 if strong == chess.WHITE else -1
    front_rank = chess.square_rank(strong_pawn_before) + step
    front_sq = chess.square(chess.square_file(strong_pawn_before), front_rank) if 0 <= front_rank <= 7 else None
    if strong_pawn_after is not None and strong_pawn_after != strong_pawn_before:
        return "推进兵势", "有兵方开始推进兵，说明局面从比拼站位转入计算升变节奏"
    if weak_king_before is not None and front_sq is not None and weak_king_before == front_sq:
        return "防线成型", "无兵方的王仍站在兵前关键格附近，当前重点是守住兵前并等待有兵方露出破绽"
    if strong_king_before is not None and chess.square_distance(strong_king_before, strong_pawn_before) <= 1:
        return "争取突破", "有兵方的王与兵保持紧密联系，下一目标通常是切断防守方王车联系或准备搭桥"
    return "争夺关键格", "双方仍在围绕兵前关键格、侧翼骚扰位和切断线路来回调整，谁先站稳关键格谁就更接近目标"

def _hard_constraints(board: chess.Board, endgame_name: str, role_meta: dict) -> List[str]:
    rules = []
    wp = _piece_square(board, chess.WHITE, chess.PAWN)
    bp = _piece_square(board, chess.BLACK, chess.PAWN)
    if wp is not None and bp is None:
        rules.append("只有白方有兵，只有白方可能升变，黑方无兵，绝不能写黑方升变")
    elif bp is not None and wp is None:
        rules.append("只有黑方有兵，只有黑方可能升变，白方无兵，绝不能写白方升变")
    if endgame_name == "车兵对车" and role_meta:
        strong = _color_name(role_meta["strong_color"])
        weak = _color_name(role_meta["weak_color"])
        rules.append(f"菲利多防线只能绑定到{weak}的防守任务，卢塞纳桥位只能绑定到{strong}的进攻任务")
    return rules

def _suggest_pacing(node: dict, cs, compressed: list) -> str:
    is_last = cs.idx == len(compressed)
    tags = node.get("tags", [])

    if node.get("endgame_changed"):
        return "slow"
    if "将军" in tags and not is_last:
        return "slow"
    if node.get("same_position") and len(cs.sans) >= 3:
        return "fast"
    if node.get("is_critical") and node.get("phase_milestone"):
        return "pause_before"
    if node.get("is_critical") and node.get("eval_delta") is not None and abs(node["eval_delta"]) > 200:
        return "slow"
    if is_last and node.get("is_critical"):
        return "pause_after"
    if node.get("is_critical"):
        return "slow"
    return "normal"


def _winner_name(outcome: Optional[chess.Outcome]) -> str:
    if outcome is None or outcome.winner is None:
        return ""
    return "白方" if outcome.winner == chess.WHITE else "黑方"


def _collect_node_move_info(board_before: chess.Board, cs) -> dict:
    """从一个压缩节点的所有走法中提取动作事实"""
    temp = board_before.copy()
    checking_types = set()
    captured_types = set()
    moved_piece_types = set()
    king_moved = False

    for san in cs.sans:
        try:
            move = temp.parse_san(san)
        except ValueError:
            continue
        piece = temp.piece_at(move.from_square)
        if piece:
            moved_piece_types.add(piece.piece_type)
            if piece.piece_type == chess.KING:
                king_moved = True
        if temp.is_capture(move):
            captured_piece = temp.piece_at(move.to_square)
            if captured_piece:
                captured_types.add(captured_piece.piece_type)
        temp.push(move)
        if temp.is_check():
            checkers = temp.checkers()
            for sq in checkers:
                p = temp.piece_at(sq)
                if p:
                    checking_types.add(p.piece_type)

    return {
        "king_moved": king_moved,
        "moved_piece_types": sorted(moved_piece_types),
        "checking_piece_types": sorted(checking_types),
        "captured_piece_types": sorted(captured_types),
    }


def _detect_repetition_maneuver(compressed: list, idx: int, kb_name: str) -> tuple:
    """检测节点是否属于反复试探机动，以及重复次数和模式"""
    cs = compressed[idx]
    if len(cs.sans) < 2:
        return False, 0, ""

    dest_squares = []
    temp_board = chess.Board(cs.fen_before)
    for san in cs.sans:
        try:
            move = temp_board.parse_san(san)
        except ValueError:
            continue
        dest_squares.append(move.to_square)
        temp_board.push(move)

    if len(dest_squares) < 3:
        return False, 0, ""

    unique = list(dict.fromkeys(dest_squares))
    if len(unique) <= 2 and len(dest_squares) >= 3:
        repeat_count = 1
        for j in range(idx + 1, len(compressed)):
            next_cs = compressed[j]
            next_temp = chess.Board(next_cs.fen_before)
            next_dests = []
            for san in next_cs.sans:
                try:
                    m = next_temp.parse_san(san)
                except ValueError:
                    continue
                next_dests.append(m.to_square)
                next_temp.push(m)
            next_unique = list(dict.fromkeys(next_dests))
            if len(next_unique) <= 2 and set(next_unique) == set(unique):
                repeat_count += 1
                continue
            break

        if repeat_count >= 2:
            pattern_squares = [chess.square_name(sq) for sq in unique]
            return True, repeat_count + 1, f"{pattern_squares[0]}-{pattern_squares[1]}" if len(pattern_squares) == 2 else "-".join(pattern_squares)

    return False, 0, ""


def _classify_goal(board_before: chess.Board, board_after: chess.Board, cs, role_meta: dict) -> str:
    weak_color = role_meta.get("weak_color")
    if weak_color is None:
        return "improve_piece_coordination"

    if board_after.is_checkmate() or board_after.is_game_over():
        return "convert_to_mate"

    wk_before = board_before.king(weak_color)
    wk_after = board_after.king(weak_color)
    if wk_before is None or wk_after is None:
        return "improve_piece_coordination"

    before_escapes = sum(1 for _ in board_before.legal_moves)
    after_escapes = sum(1 for _ in board_after.legal_moves)

    weak_rank = chess.square_rank(wk_after)
    weak_file = chess.square_file(wk_after)
    on_edge = weak_rank in (0, 7) or weak_file in (0, 7)
    in_corner = (weak_rank in (0, 7) and weak_file in (0, 7))

    if in_corner:
        return "drive_to_corner"
    if on_edge and after_escapes < before_escapes:
        return "drive_to_edge"
    if after_escapes < before_escapes:
        return "shrink_space"
    if cs is not None and getattr(cs, "fen_before", "") == getattr(cs, "fen_after", ""):
        return "hold_net"

    return "improve_piece_coordination"


def _assign_claim_level(node: dict, goal: str, is_last: bool) -> str:
    if node.get("is_checkmate_after"):
        return "terminal"
    if node.get("is_game_over_after"):
        return "terminal"
    if is_last and goal == "convert_to_mate":
        return "forcing"
    if node.get("is_check_after"):
        return "forcing"
    if node.get("legal_reply_count_after", 10) <= 2:
        return "forcing"
    if goal in ("drive_to_corner", "drive_to_edge", "shrink_space"):
        return "constraining"
    return "positioning"


def _assign_video_density(node: dict, contains_rep: bool, repeat_count: int) -> dict:
    if contains_rep and repeat_count >= 3:
        return {"density": "low", "summary_only": True}
    if node.get("is_critical") or node.get("is_checkmate_after") or node.get("endgame_changed"):
        return {"density": "high", "summary_only": False}
    return {"density": "medium", "summary_only": False}


# ============================================================
#  Narrative Planner (ADR-012)
#  程序化计算叙事弧线：tension_score → narrative_role → tone_hint/word_budget
#  不引入新 LLM 调用，纯棋盘事实+位置计算
# ============================================================

_ROLE_TONE = {
    "setup": "平稳叙述",
    "build_up": "节奏加快",
    "climax": "强调关键",
    "falling_action": "舒缓解释",
    "resolution": "总结收束",
}

_ROLE_WORD_BUDGET = {
    "setup": "40-70字",
    "build_up": "70-110字",
    "climax": "110-160字",
    "falling_action": "40-70字",
    "resolution": "60-100字",
}


def _assign_narrative_role(idx: int, total: int, tension: float, node: dict) -> str:
    """根据节点位置+张力分数+特殊事件判定叙事角色。

    规则（ADR-012）：
    - 将杀/终局 → resolution（无论位置）
    - 第一个节点 → setup（开局铺垫）
    - 最后一个节点（非将杀）→ falling_action（回落收束）
    - tension >= 0.6 → climax（高潮/关键转折）
    - 前半段非climax → build_up；后半段非climax → falling_action
    """
    if node.get("is_checkmate_after") or node.get("is_game_over_after"):
        return "resolution"
    if idx == 0:
        return "setup"
    if tension >= 0.6:
        return "climax"
    if idx == total - 1:
        return "falling_action"
    if idx < total * 0.5:
        return "build_up"
    return "falling_action"


def build(board: chess.Board, compressed: List[CompressedStep], winner_color=None,
          enable_insight: bool = True) -> dict:
    """基于压缩节点构建叙事分镜，注入局面特征与分阶段解说提示。

    winner_color: 实际终局赢家颜色(chess.WHITE/BLACK)，由 pipeline 复盘得出。
    用于让攻守立场从真实结果反推，避免解说与画面相反。

    enable_insight: 是否启用棋理洞察层（src/insight_extractor）。默认开启；
    任何异常都会被吞掉退回"无洞察"，保证不破坏原有链路。
    """
    kb = match_endgame(board)
    if kb is not None:
        endgame_name = kb["name"]
    else:
        endgame_name = describe_endgame(board)["name"]
    phases = kb["phases"] if kb else []
    role_meta = _role_meta(board, endgame_name, winner_color=winner_color)
    n = len(compressed)

    # 棋理洞察：失败安全地提取，下面按节点注入。提取失败/禁用时为空 dict 列表。
    insights = []
    if enable_insight:
        try:
            insights = extract_for_compressed(
                compressed, board, role_meta if role_meta else None,
                endgame_name)
        except Exception as e:
            Logger.warn(f"棋理洞察提取失败，退回无洞察模式: {e}")
            insights = []

    for i, cs in enumerate(compressed):
        if phases and n > 0:
            ratio = i / max(n - 1, 1)
            pi = min(int(ratio * len(phases)), len(phases) - 1)
            cs.phase = phases[pi][0]
            cs.phase_hint = phases[pi][1]

    nodes_out = []
    start_board_for_material = board.copy()
    prev_phase = ""
    prev_endgame_name = ""
    prev_endgame_type = ""
    n_compressed = len(compressed)

    for idx_cs, cs in enumerate(compressed):
        board_before = chess.Board(cs.fen_before)
        board_after = chess.Board(cs.fen_after)

        move_info = _collect_node_move_info(board_before, cs)
        contains_rep, rep_count, rep_pattern = _detect_repetition_maneuver(compressed, idx_cs, endgame_name)
        goal = _classify_goal(board_before, board_after, cs, role_meta if role_meta else {})
        is_last_node = idx_cs == n_compressed - 1

        sub_endgame = describe_endgame(board_before)
        sub_name = sub_endgame["name"]
        sub_type = sub_endgame.get("type", "unknown")
        endgame_changed = (sub_type != prev_endgame_type) and prev_endgame_type != "" and sub_type != "unknown"
        if sub_name != prev_endgame_name:
            prev_endgame_name = sub_name
            prev_endgame_type = sub_type

        allowed = sub_endgame.get("motifs", [])
        forbidden = get_forbidden_concepts(board_before, sub_endgame)

        if len(cs.sans) > 1:
            turn = "双方交替"
        else:
            turn = "白方走" if board_before.turn == chess.WHITE else "黑方走"

        phase_hint = getattr(cs, "phase_hint", "")
        same_position = cs.fen_before == cs.fen_after
        if kb and kb.get("name") == "车兵对车":
            cs.phase, phase_hint = _krpkr_phase_hint(board_before, board_after, role_meta, same_position)
        elif same_position and len(cs.sans) >= 2:
            cs.phase = "反复试探"
            phase_hint = "这段变化的起止局面相同，属于反复调车试探与等招，并未形成实质突破"

        actor_role = ""
        if role_meta:
            actor_role = "强方" if board_before.turn == role_meta["strong_color"] else "弱方"

        phase_milestone = bool(cs.phase and cs.phase != prev_phase)
        detail_level = "high" if cs.is_critical or phase_milestone or len(cs.sans) >= 6 else "medium"
        outcome_after = board_after.outcome() if board_after.is_game_over() else None
        legal_reply_count_after = sum(1 for _ in board_after.legal_moves)
        is_capture_node = "吃子" in cs.tags
        has_check_in_node = "将军" in cs.tags

        node = {
            "id": cs.idx,
            "sans": list(cs.sans),
            "turn": turn,
            "moves": " → ".join(cs.sans),
            "move_count": len(cs.sans),
            "is_critical": cs.is_critical,
            "phase": cs.phase,
            "phase_hint": phase_hint,
            "tags": cs.tags,
            "fen_before": cs.fen_before,
            "transition_summary": _transition_summary(cs.fen_before, cs.fen_after),
            "eval_delta": getattr(cs, "eval_delta", None),
            "same_position": same_position,
            "actor_role": actor_role,
            "phase_milestone": phase_milestone,
            "detail_level": detail_level,
            "sub_endgame_name": sub_name,
            "endgame_changed": endgame_changed,
            "allowed_concepts": allowed,
            "forbidden_concepts": forbidden,
            "is_capture_node": is_capture_node,
            "has_check_in_node": has_check_in_node,
            # 根因B/E修复（前置注入）：把本节点起始局面双方真实子力直接给模型，
            # 杜绝"兵残局里凭空造后"这类捏造（KPvK 系列高频）。是不可改写事实。
            "white_material": _side_material_desc(board_before, chess.WHITE),
            "black_material": _side_material_desc(board_before, chess.BLACK),
            "is_check_after": board_after.is_check(),
            "is_checkmate_after": board_after.is_checkmate(),
            "is_stalemate_after": board_after.is_stalemate(),
            "is_game_over_after": board_after.is_game_over(),
            "legal_reply_count_after": legal_reply_count_after,
            "winner_after": _winner_name(outcome_after),
            "king_moved": move_info["king_moved"],
            "moved_piece_types": move_info["moved_piece_types"],
            "checking_piece_types": move_info["checking_piece_types"],
            "captured_piece_types": move_info["captured_piece_types"],
            "contains_repetition_maneuver": contains_rep,
            "repeat_count": rep_count,
            "maneuver_pattern": rep_pattern,
            "position_goal": goal,
            "is_last_node": is_last_node,
        }
        claim_level = _assign_claim_level(node, goal, is_last_node)
        node["claim_level"] = claim_level
        video_info = _assign_video_density(node, contains_rep, rep_count)
        node["video_density"] = video_info["density"]
        node["summary_only"] = video_info["summary_only"]

        node["suggested_phase_label"] = cs.phase if cs.phase else ""
        node["suggested_pacing"] = _suggest_pacing(node, cs, compressed)

        # 注入棋理洞察（失败安全：insights 为空时全部跳过，node 不含这些字段，
        # 下游 commentator 读不到即按旧行为处理）。
        if idx_cs < len(insights):
            insight = insights[idx_cs]
            tp = insight.get("teaching_point", "")
            if tp:
                node["teaching_point"] = tp
            mm = insight.get("must_mention", [])
            if mm:
                node["must_mention"] = mm
            sc = insight.get("spatial_change", {})
            if sc:
                node["spatial_change"] = sc

            # 战术叙述（新）：纯棋理中文，不给结论只给前提
            tn = insight.get("tactical_narratives", [])
            if tn:
                node["tactical_narratives"] = tn

            # 关键手判定回灌：insight_extractor._compute_importance 已用棋盘事实
            # （活动空间锐减/逼到边角/双重攻击/对王等）算出本节点的重要级别与理由。
            # 本地小模型缺乏独立棋理推理力，拿不出「哪步关键、为什么关键」的综合判断，
            # 只能堆套话填空。这里把已算好的结论+理由回灌给 prompt，让模型从「自己判断」
            # 降级为「把给定结论讲透」——这是小模型能胜任的任务。
            # 注意：仅作为「供讲解的判定依据」注入，不再据此上调 is_critical（避免污染压缩）。
            importance = insight.get("importance", "")
            reasons = insight.get("importance_reasons", []) or []
            if importance:
                node["move_importance"] = importance
            if reasons:
                node["importance_reasons"] = reasons

            # Narrative Planner (ADR-012)：注入张力/叙事角色/字数预算/语气提示
            tension = insight.get("tension_score")
            if tension is not None:
                node["tension_score"] = tension
                role = _assign_narrative_role(idx_cs, n_compressed, tension, node)
                node["narrative_role"] = role
                node["tone_hint"] = _ROLE_TONE.get(role, "")
                node["word_budget"] = _ROLE_WORD_BUDGET.get(role, "60-90字")

        # 引擎信号（中性观察，不给结论）：
        # 利用节点已有的 eval_delta / is_only_move 生成量化参考句。
        eval_signals = []
        ed = getattr(cs, "eval_delta", None)
        # 排除终局哨兵 9999 和极端值，只对合理的评估变化生成信号。
        # 不断言"扩大/缩小"方向：eval_delta 是逐着按走子方视角算、再跨多着
        # 求和得到的，多着合并节点里符号可能反号，断言方向会把增大优势的一步
        # 说成"优势缩小"。这里只陈述"显著变化"的量级，方向交给画面与其他信号。
        if ed is not None and 200 < abs(ed) < 9000:
            eval_signals.append(
                f"局面评估值在这一步后发生了显著变化（约{abs(int(ed))}厘兵）。")

        # 唯一好着（用真实信号，不用 candidates 代理量）
        if cs.is_only_move:
            eval_signals.append(
                "除正解外，其他候选走法都会让胜势大幅缩水——"
                "这是当前局面下唯一能保住胜利果实的选择。")

        if eval_signals:
            node["eval_signals"] = eval_signals

        nodes_out.append(node)
        prev_phase = cs.phase

    total_halfmoves = sum(len(cs.sans) for cs in compressed)

    strong_color = role_meta.get("strong_color")
    weak_color = role_meta.get("weak_color")
    winning_side = _color_name(strong_color) if strong_color is not None else ""
    losing_side = _color_name(weak_color) if weak_color is not None else ""
    narrative_mode = "winning_conversion" if role_meta else "balanced"

    # 开场白素材（永远可得，不依赖 KB）：双方子力中文描述。
    # 强弱方未知（材料均势且无终局信息）时按颜色给出，仍可用于子力对比介绍。
    if strong_color is not None:
        strong_material = _side_material_desc(board, strong_color)
        weak_material = _side_material_desc(board, weak_color)
    else:
        strong_material = ""
        weak_material = ""

    return {
        "endgame_name": endgame_name,
        "endgame_matched": kb is not None,
        "white_material": _side_material_desc(board, chess.WHITE),
        "black_material": _side_material_desc(board, chess.BLACK),
        "strong_material": strong_material,
        "weak_material": weak_material,
        "context": kb["theory"] if kb else "残局局面分析",
        "phases": phases,
        "motifs": kb.get("motifs", []) if kb else [],
        "mistakes": kb.get("mistakes", []) if kb else [],
        "opening": kb.get("opening", {}) if kb else {},
        "role_summary": role_meta.get("role_summary", ""),
        "concept_binding": role_meta.get("concept_binding", []),
        "hard_constraints": _hard_constraints(board, endgame_name, role_meta),
        "winning_side": winning_side,
        "losing_side": losing_side,
        "narrative_mode": narrative_mode,
        "compact_mode": total_halfmoves >= LONG_MOVE_THRESHOLD or n >= COMPACT_NODE_THRESHOLD,
        "target_length": _target_length(n),
        "has_sub_endgame_switch": any(
            node.get("endgame_changed") for node in nodes_out
        ),
        "nodes": nodes_out,
    }


def _target_length(node_count: int) -> str:
    """全局字数预算随节点数连续计算（取代旧的 ≥7 二档硬阶梯）。

    每节点约 90 字、上界约 120 字，整体夹在 [700, 2000] 区间。节点越多
    预算越大，但因节点数本身已被自适应预算压成次线性，长解法的总字数不会
    失控膨胀，与「越长压得越狠」一致。
    """
    lo = max(700, node_count * 90)
    hi = max(1000, node_count * 120)
    return f"{lo}-{hi}字"
