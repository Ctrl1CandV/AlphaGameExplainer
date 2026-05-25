from src.common import CompressedStep, Logger, AnalyzedMove
from src.endgame_knowledge import match as match_endgame
from src.endgame_knowledge import describe_endgame, get_forbidden_concepts
from typing import List, Optional, Tuple
import chess

MAX_NODE_SPAN = 4
LONG_NODE_SPAN = 6
LONG_MOVE_THRESHOLD = 18
COMPACT_NODE_THRESHOLD = 7

def _tag_position(board: chess.Board, move: chess.Move) -> List[str]:
    tags = []
    if board.gives_check(move):
        tags.append("将军")
    if board.is_capture(move):
        tags.append("吃子")
    t = _detect_opposition(board)
    if t:
        tags.append(t)
    t = _detect_zugzwang_hint(board)
    if t:
        tags.append(t)
    return tags

def _detect_opposition(board: chess.Board) -> str:
    kings = [sq for sq, p in board.piece_map().items() if p.piece_type == chess.KING]
    if len(kings) != 2:
        return ""
    wk = next(sq for sq, p in board.piece_map().items()
              if p.piece_type == chess.KING and p.color == chess.WHITE)
    bk = next(sq for sq, p in board.piece_map().items()
              if p.piece_type == chess.KING and p.color == chess.BLACK)
    df = abs(chess.square_file(wk) - chess.square_file(bk))
    dr = abs(chess.square_rank(wk) - chess.square_rank(bk))
    if df == 0 and dr == 2:
        return "对王(竖排)"
    if df == 2 and dr == 0:
        return "对王(横排)"
    if df == 2 and dr == 2:
        return "斜线对王"
    if df == 0 and dr == 0:
        return ""
    if df <= 1 and dr <= 1:
        return "近距离对峙"
    return ""

def _detect_zugzwang_hint(board: chess.Board) -> str:
    if board.is_check() or board.is_game_over():
        return ""
    opponent = chess.WHITE if board.turn == chess.BLACK else chess.BLACK
    threat_sqs = list(board.attackers(opponent, board.king(board.turn)))
    if len(threat_sqs) == 1:
        return "仅一安全格"
    if len(threat_sqs) >= 3:
        return "王被困"
    return ""

def _is_semantic_boundary(entry: dict, prev_entry: Optional[dict], board_before: chess.Board) -> bool:
    """ 确定是否应该在此步骤前断节点 """
    if prev_entry is None:
        return True
    if "将军" in entry["tags"] or "吃子" in entry["tags"]:
        return True
    if entry.get("only") and not entry.get("is_last"):
        return True
    if entry["fen_before"] == entry.get("fen_after", ""):
        return True
    dist = 0
    before_sqs = set(board_before.piece_map().keys())
    after_board = chess.Board(entry["fen_after"])
    after_sqs = set(after_board.piece_map().keys())
    moved = before_sqs.symmetric_difference(after_sqs)
    kings_moved = any(board_before.piece_at(sq) and board_before.piece_at(sq).piece_type == chess.KING for sq in moved)
    if kings_moved and prev_entry is not None:
        prev_after = chess.Board(prev_entry["fen_after"])
        prev_kings = {sq for sq, p in prev_after.piece_map().items() if p.piece_type == chess.KING}
        cur_kings = {sq for sq, p in board_before.piece_map().items() if p.piece_type == chess.KING}
        if prev_kings != cur_kings:
            dist = min((chess.square_distance(a, b) for a in prev_kings for b in cur_kings), default=0)
            if dist >= 2:
                return True
    return False

def _process_summary(entries: List[dict]) -> str:
    """ 生成节点的过程摘要，描述中间发生了什么 """
    if len(entries) <= 1:
        return entries[0]["san"] if entries else ""

    parts = []
    checked = any("将军" in e.get("tags", []) for e in entries)
    captured = any("吃子" in e.get("tags", []) for e in entries)
    has_promotion = any("=" in e["san"] for e in entries)
    kings_moved = any(
        chess.Board(e["fen_before"]).king(chess.WHITE) != chess.Board(e["fen_after"]).king(chess.WHITE) or
        chess.Board(e["fen_before"]).king(chess.BLACK) != chess.Board(e["fen_after"]).king(chess.BLACK)
        for e in entries
    )
    first = entries[0]
    last = entries[-1]
    first_after = chess.Board(first["fen_after"])
    last_before = chess.Board(last["fen_before"])

    if has_promotion:
        promo_moves = [e["san"] for e in entries if "=" in e["san"]]
        parts.append(f"兵连续推进并升变：{'、'.join(promo_moves)}")
    if checked:
        parts.append("含将军走法，压缩对方王活动空间")
    if captured:
        parts.append("含吃子，改变子力对比")
    if kings_moved:
        parts.append("双方王位置发生关键变化")
    if not parts:
        parts.append("调整子力位置，改善站位")

    return "；".join(parts)

def _is_swing_move(item: dict, prev_item: Optional[dict]) -> bool:
    if prev_item is None or item.get("eval_delta") is None or prev_item.get("eval_delta") is None:
        return False
    return abs(item["eval_delta"]) < 30 and abs(prev_item.get("eval_delta", 999)) < 30

def _kbnk_corner_state(board: chess.Board, role_meta: dict) -> str:
    strong = role_meta.get("strong_color")
    weak = role_meta.get("weak_color")
    if strong is None or weak is None:
        return ""
    wk = board.king(weak)
    bishop_sq = _piece_square(board, strong, chess.BISHOP)
    if wk is None or bishop_sq is None:
        return ""
    if chess.square_name(wk) in {"a1", "a8", "h1", "h8"}:
        king_color = (chess.square_file(wk) + chess.square_rank(wk)) % 2
        bishop_color = (chess.square_file(bishop_sq) + chess.square_rank(bishop_sq)) % 2
        return "正确角" if king_color == bishop_color else "错误角"
    if chess.square_file(wk) in (0, 7) or chess.square_rank(wk) in (0, 7):
        return "边线"
    return "中心"

def compress(board: chess.Board, analyzed_moves: List[AnalyzedMove]) -> List[CompressedStep]:
    """ 语义压缩：按将军/吃子/升变/转折等边 界切分节点，单节点≤4步 """
    temp = board.copy()
    kb = match_endgame(board)
    endgame_name = kb["name"] if kb else "残局"
    role_meta = _role_meta(board, endgame_name)
    kbnk_mode = endgame_name == "象马杀王" and len(analyzed_moves) >= LONG_MOVE_THRESHOLD
    long_line_mode = len(analyzed_moves) >= LONG_MOVE_THRESHOLD
    max_span = LONG_NODE_SPAN if long_line_mode else MAX_NODE_SPAN

    per_move = []
    prev_score = None
    for i, am in enumerate(analyzed_moves):
        if temp.is_game_over():
            break
        turn = "白方" if temp.turn == chess.WHITE else "黑方"
        san = temp.san(am.move)
        tags = _tag_position(temp.copy(), am.move)
        score = am.score
        only = am.is_only_move
        fen_before = temp.fen()
        temp.push(am.move)
        fen_after = temp.fen()

        if score is not None and prev_score is not None and len(per_move) > 0:
            per_move[-1]["eval_delta"] = -(score + prev_score)

        entry = {
            "idx": i + 1, "san": san, "tags": tags,
            "only": only, "eval": score, "eval_delta": None,
            "turn": turn, "fen_before": fen_before, "fen_after": fen_after,
            "trap": am.trap_san,
            "candidates": am.candidates,
        }
        per_move.append(entry)
        prev_score = score

    if prev_score is not None and len(per_move) > 0:
        if temp.is_game_over():
            outcome = temp.outcome()
            if outcome and outcome.winner is not None:
                per_move[-1]["eval_delta"] = 9999
        else:
            per_move[-1]["eval_delta"] = None

    for idx, item in enumerate(per_move):
        is_first = idx == 0
        is_last = idx == len(per_move) - 1
        big_delta = item.get("eval_delta") is not None and abs(item["eval_delta"]) > 200
        state_changed = False
        if kbnk_mode:
            before_state = _kbnk_corner_state(chess.Board(item["fen_before"]), role_meta)
            after_state = _kbnk_corner_state(chess.Board(item["fen_after"]), role_meta)
            state_changed = before_state != after_state
        item["is_first"] = is_first
        item["is_last"] = is_last
        item["big_delta"] = big_delta
        item["state_changed"] = state_changed

    # 语义边界分组
    groups = []
    cur_group = []
    for idx, item in enumerate(per_move):
        board_before = chess.Board(item["fen_before"])
        prev = cur_group[-1] if cur_group else None
        boundary = _is_semantic_boundary(item, prev, board_before)
        span_full = len(cur_group) >= max_span

        if boundary and cur_group:
            groups.append(cur_group)
            cur_group = []
        elif span_full:
            groups.append(cur_group)
            cur_group = []

        cur_group.append(item)

    if cur_group:
        groups.append(cur_group)

    compressed = []
    for grp in groups:
        if not grp:
            continue
        first = grp[0]
        last = grp[-1]
        is_critical = any(
            g["is_first"] or g["is_last"] or
            "将军" in g["tags"] or "吃子" in g["tags"] or
            g.get("big_delta") or g.get("state_changed") or
            (g["only"] and not long_line_mode)
            for g in grp
        )
        all_tags = list(set(t for g in grp for t in g["tags"]))
        swing = False
        if len(grp) >= 3:
            swing = all(
                _is_swing_move(grp[j], grp[j - 1] if j > 0 else grp[0])
                for j in range(1, len(grp))
            )
        if swing:
            all_tags.append("对王调整")

        total_delta = sum(g.get("eval_delta", 0) for g in grp if g.get("eval_delta") is not None)
        trap = next((g["trap"] for g in grp if g.get("trap")), None) or ""
        process_summ = _process_summary(grp)

        compressed.append(CompressedStep(
            idx=len(compressed) + 1,
            sans=[g["san"] for g in grp],
            fen_before=first["fen_before"],
            fen_after=last["fen_after"],
            is_critical=is_critical,
            trap=trap,
            tags=all_tags,
            eval_delta=total_delta,
            candidates=list(set(c for g in grp for c in g.get("candidates", []))),
        ))
        compressed[-1].process_summary = process_summ

    compressed = _merge_repetitive(compressed)

    Logger.info(f"压缩: {len(per_move)} 步 → {len(compressed)} 节点")
    return compressed

def _same_pieces(a_fen: str, b_fen: str) -> bool:
    try:
        a_counts = {}
        for p in chess.Board(a_fen).piece_map().values():
            key = (p.piece_type, p.color)
            a_counts[key] = a_counts.get(key, 0) + 1
        b_counts = {}
        for p in chess.Board(b_fen).piece_map().values():
            key = (p.piece_type, p.color)
            b_counts[key] = b_counts.get(key, 0) + 1
        return a_counts == b_counts
    except Exception:
        return False

def _merge_repetitive(steps: List[CompressedStep]) -> List[CompressedStep]:
    if len(steps) < 3:
        return steps
    merged = [steps[0]]
    for cur in steps[1:]:
        prev = merged[-1]
        if (not prev.is_critical and not cur.is_critical
                and _same_pieces(prev.fen_before, cur.fen_before)
                and len(prev.sans) < 10):
            prev.sans.extend(cur.sans)
            prev.fen_after = cur.fen_after
            prev.tags = list(set(prev.tags + cur.tags))
            if len(prev.sans) >= 6:
                prev.tags.append("对王调整")
        else:
            merged.append(cur)
    for i, s in enumerate(merged):
        s.idx = i + 1
    return merged

def _describe_situation(board: chess.Board) -> str:
    """生成局面的中文描述：王位与对王关系、兵排位、车控制线"""
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return ""
    wk_name = chess.square_name(wk)
    bk_name = chess.square_name(bk)
    df = abs(chess.square_file(wk) - chess.square_file(bk))
    dr = abs(chess.square_rank(wk) - chess.square_rank(bk))

    lines = [f"白王{wk_name}，黑王{bk_name}"]
    if df == 0 and dr == 2:
        who = "白方" if board.turn == chess.BLACK else "黑方"
        lines.append(f"（竖排对王，{who}占据主动）")
    elif df == 2 and dr == 0:
        lines.append("（横排对王）")
    elif df == 2 and dr == 2:
        lines.append("（斜线对王）")
    elif df <= 1 and dr <= 1:
        lines.append("（近距离对峙）")

    for sq, p in board.piece_map().items():
        if p.piece_type == chess.PAWN:
            color = "白" if p.color == chess.WHITE else "黑"
            rank = chess.square_rank(sq)
            lines.append(f"{color}兵{chess.square_name(sq)}（第{rank + 1}排{'，已过中线' if (p.color == chess.WHITE and rank >= 4) or (p.color == chess.BLACK and rank <= 3) else ''}）")
        elif p.piece_type == chess.ROOK:
            color = "白" if p.color == chess.WHITE else "黑"
            file_char = chr(97 + chess.square_file(sq))
            rank_num = chess.square_rank(sq) + 1
            lines.append(f"{color}车{chess.square_name(sq)}（控制{file_char}列和第{rank_num}排）")

    return "；".join(lines)

def _compare_kings(fen_before: str, fen_after: str) -> str:
    """比较两个局面的王位变化"""
    try:
        b1 = chess.Board(fen_before)
        b2 = chess.Board(fen_after)
        wk1 = b1.king(chess.WHITE)
        wk2 = b2.king(chess.WHITE)
        bk1 = b1.king(chess.BLACK)
        bk2 = b2.king(chess.BLACK)
        parts = []
        if wk1 is not None and wk2 is not None and wk1 != wk2:
            parts.append(f"白王{chess.square_name(wk1)}→{chess.square_name(wk2)}")
        if bk1 is not None and bk2 is not None and bk1 != bk2:
            parts.append(f"黑王{chess.square_name(bk1)}→{chess.square_name(bk2)}")
        if not parts:
            return "王位未变，车反复等招"
        return "，".join(parts)
    except Exception:
        return ""

def _material_score(board: chess.Board, color: chess.Color) -> int:
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    total = 0
    for piece in board.piece_map().values():
        if piece.color == color and piece.piece_type != chess.KING:
            total += values.get(piece.piece_type, 0)
    return total

def _color_name(color: chess.Color) -> str:
    return "白方" if color == chess.WHITE else "黑方"

def _piece_square(board: chess.Board, color: chess.Color, piece_type: chess.PieceType):
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type == piece_type:
            return sq
    return None

def _piece_squares(board: chess.Board, color: chess.Color, piece_type: chess.PieceType) -> List[int]:
    squares = []
    for sq, piece in board.piece_map().items():
        if piece.color == color and piece.piece_type == piece_type:
            squares.append(sq)
    return sorted(squares)

def _piece_label(piece_type: chess.PieceType) -> str:
    mapping = {
        chess.KING: "王",
        chess.QUEEN: "后",
        chess.ROOK: "车",
        chess.BISHOP: "象",
        chess.KNIGHT: "马",
        chess.PAWN: "兵",
    }
    return mapping.get(piece_type, "子")

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

def _compact_moves_display(sans: List[str]) -> str:
    if len(sans) <= 4:
        return " → ".join(sans)
    if len(sans) <= 8:
        return f"{sans[0]} → {sans[1]} → ... → {sans[-2]} → {sans[-1]}"
    return f"{sans[0]} → ... → {sans[-1]}"

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

def _krpkr_teaching_focus(board_before: chess.Board, board_after: chess.Board, role_meta: dict, same_position: bool) -> str:
    strong = role_meta.get("strong_color")
    weak = role_meta.get("weak_color")
    if strong is None or weak is None:
        return ""
    strong_pawn_before = _piece_square(board_before, strong, chess.PAWN)
    strong_pawn_after = _piece_square(board_after, strong, chess.PAWN)
    strong_rook_before = _piece_square(board_before, strong, chess.ROOK)
    strong_rook_after = _piece_square(board_after, strong, chess.ROOK)
    weak_rook_before = _piece_square(board_before, weak, chess.ROOK)
    weak_rook_after = _piece_square(board_after, weak, chess.ROOK)
    if same_position:
        return "这一段的教学重点是等招试探：双方都在确认关键格和切断线路是否会松动，而不是立即突破。"
    if strong_pawn_before is not None and strong_pawn_after is not None and strong_pawn_before != strong_pawn_after:
        return "这一段的教学重点是兵的推进时机：兵每前进一步，升变距离都会缩短，但前提是王车配合不能散。"
    if weak_rook_before is not None and weak_rook_after is not None and chess.square_rank(weak_rook_before) != chess.square_rank(weak_rook_after):
        return "这一段的教学重点是防守车换排骚扰：无兵方通过横向调车寻找更好的将军和切断位置。"
    if strong_rook_before is not None and strong_rook_after is not None and chess.square_file(strong_rook_before) != chess.square_file(strong_rook_after):
        return "这一段的教学重点是有兵方调整车位，为兵让路，同时准备从侧面或后方掩护推进。"
    return "这一段的教学重点是围绕兵前关键格和王车联系做准备，暂时还没有进入最后的技术兑现阶段。"

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

def _counterfactual_hint(node: dict) -> str:
    if node.get("trap"):
        return f"如果这一步处理失当，对手常见的反击是{node['trap']}"
    if node.get("same_position"):
        return "如果贸然打破当前站位，往往会先暴露关键格或让对方王重新获得活动空间"
    if node.get("phase") == "推进兵势":
        return "如果此时推进准备不足，兵可能失去王车保护，反而难以继续前进"
    if node.get("phase") == "防线成型":
        return "如果这一阶段守不住兵前关键格，原本可和的局面就可能迅速恶化"
    if node.get("phase") == "争取突破":
        return "如果没有先改善王车站位，后续就算强行推进，也很难把优势真正兑现"
    return ""

def _generic_teaching_focus(phase: str, phase_hint: str, same_position: bool) -> str:
    if same_position:
        return "这一段的教学重点是等招与站位保持：虽然没有直接突破，但关键控制线和关键格都不能轻易放松。"
    if phase == "建立控制线":
        return "这一段的教学重点是先把对方王的活动空间框住，后续推进才会有明确方向。"
    if phase == "王车合围":
        return "这一段的教学重点是让王和主力子形成呼应，避免单独将军把对方王放跑。"
    if phase == "驱赶到边":
        return "这一段的教学重点是先把对方王从中心赶到边线，缩小其可用逃跑空间。"
    if phase == "引导至正确角落":
        return "这一段的教学重点是把对方王送往唯一可被彻底封死的角落，而不是只求把王赶到任意边角。"
    if phase == "完成将杀":
        return "这一段的教学重点是把各子控制网拼完整，让对方王每一个逃格都被封死。"
    if phase_hint:
        return phase_hint
    return "这一段的教学重点是改善站位并为下一阶段目标做准备。"

def _role_meta(board: chess.Board, endgame_name: str) -> dict:
    white_score = _material_score(board, chess.WHITE)
    black_score = _material_score(board, chess.BLACK)
    if white_score == black_score:
        return {}
    strong = chess.WHITE if white_score > black_score else chess.BLACK
    weak = chess.BLACK if strong == chess.WHITE else chess.WHITE
    meta = {
        "strong_color": strong,
        "weak_color": weak,
        "role_summary": f"{_color_name(strong)}是强方，目标是扩大优势并转化为胜势；{_color_name(weak)}是弱方，目标是组织防守争取和棋。",
        "concept_binding": [],
    }
    if endgame_name == "车兵对车":
        meta["role_summary"] = f"{_color_name(strong)}是强方（有兵方），目标是推进兵升变或吃兵转胜；{_color_name(weak)}是弱方（无兵方），目标是建立防线守和。"
        meta["concept_binding"] = [
            f"菲利多防线属于{_color_name(weak)}的防守策略",
            f"卢塞纳桥位属于{_color_name(strong)}的进攻策略",
        ]
    return meta

def build(board: chess.Board, compressed: List[CompressedStep]) -> dict:
    """基于压缩节点构建叙事分镜，注入局面特征与分阶段解说提示"""
    kb = match_endgame(board)
    phases = kb["phases"] if kb else []
    role_meta = _role_meta(board, kb["name"] if kb else "残局")
    n = len(compressed)

    for i, cs in enumerate(compressed):
        if phases and n > 0:
            ratio = i / max(n - 1, 1)
            pi = min(int(ratio * len(phases)), len(phases) - 1)
            cs.phase = phases[pi][0]
            cs.phase_hint = phases[pi][1]

    nodes_out = []
    prev_phase = ""
    prev_endgame_name = ""
    prev_endgame_type = ""
    for cs in compressed:
        board_before = chess.Board(cs.fen_before)
        board_after = chess.Board(cs.fen_after)

        sub_endgame = describe_endgame(board_before)
        sub_name = sub_endgame["name"]
        sub_type = sub_endgame.get("type", "unknown")
        endgame_changed = (sub_type != prev_endgame_type) and prev_endgame_type != "" and sub_type != "unknown"
        if sub_name != prev_endgame_name:
            if prev_endgame_name:
                Logger.info(f"  子残局切换: {prev_endgame_name} → {sub_name}")
            prev_endgame_name = sub_name
            prev_endgame_type = sub_type

        allowed = sub_endgame.get("motifs", [])
        forbidden = get_forbidden_concepts(board_before, sub_endgame)

        if len(cs.sans) > 1:
            turn = "双方交替"
        else:
            turn = "白方走" if board_before.turn == chess.WHITE else "黑方走"

        situation_before = _describe_situation(board_before)
        situation_after = _describe_situation(board_after)
        phase_hint = getattr(cs, "phase_hint", "")
        same_position = cs.fen_before == cs.fen_after
        if kb and kb.get("name") == "车兵对车":
            cs.phase, phase_hint = _krpkr_phase_hint(board_before, board_after, role_meta, same_position)
        elif same_position and len(cs.sans) >= 2:
            Logger.warn(f"节点{cs.idx}起止局面相同，按反复试探处理")
            cs.phase = "反复试探"
            phase_hint = "这段变化的起止局面相同，属于反复调车试探与等招，并未形成实质突破"

        actor_role = ""
        if role_meta:
            actor_role = "强方" if board_before.turn == role_meta["strong_color"] else "弱方"

        phase_milestone = bool(cs.phase and cs.phase != prev_phase)
        detail_level = "high" if cs.is_critical or phase_milestone or len(cs.sans) >= 6 else "medium"

        node = {
            "id": cs.idx,
            "turn": turn,
            "moves": " → ".join(cs.sans),
            "moves_display": _compact_moves_display(cs.sans),
            "move_count": len(cs.sans),
            "is_critical": cs.is_critical,
            "phase": cs.phase,
            "phase_hint": phase_hint,
            "tags": cs.tags,
            "trap": cs.trap,
            "fen_before": cs.fen_before,
            "situation_before": situation_before,
            "transition_summary": _transition_summary(cs.fen_before, cs.fen_after),
            "process_summary": getattr(cs, "process_summary", ""),
            "teaching_focus": _krpkr_teaching_focus(board_before, board_after, role_meta, same_position) if kb and kb.get("name") == "车兵对车" else _generic_teaching_focus(cs.phase, phase_hint, same_position),
            "eval_delta": getattr(cs, "eval_delta", None),
            "same_position": same_position,
            "actor_role": actor_role,
            "phase_milestone": phase_milestone,
            "detail_level": detail_level,
            "sub_endgame_name": sub_name,
            "endgame_changed": endgame_changed,
            "allowed_concepts": allowed,
            "forbidden_concepts": forbidden,
        }
        node["counterfactual_hint"] = _counterfactual_hint(node)

        if not cs.is_critical and len(cs.sans) >= 2:
            node["king_change"] = _compare_kings(cs.fen_before, cs.fen_after)
            node["situation_after"] = situation_after

        nodes_out.append(node)
        prev_phase = cs.phase

    total_halfmoves = sum(len(cs.sans) for cs in compressed)

    return {
        "endgame_name": kb["name"] if kb else "残局",
        "context": kb["theory"] if kb else "残局局面分析",
        "phases": phases,
        "motifs": kb.get("motifs", []) if kb else [],
        "mistakes": kb.get("mistakes", []) if kb else [],
        "role_summary": role_meta.get("role_summary", ""),
        "concept_binding": role_meta.get("concept_binding", []),
        "hard_constraints": _hard_constraints(board, kb["name"] if kb else "残局", role_meta),
        "compact_mode": total_halfmoves >= LONG_MOVE_THRESHOLD or n >= COMPACT_NODE_THRESHOLD,
        "target_length": "1200-1600字" if n >= COMPACT_NODE_THRESHOLD else "800-1100字",
        "has_sub_endgame_switch": any(
            node.get("endgame_changed") for node in nodes_out
        ),
        "nodes": nodes_out,
    }