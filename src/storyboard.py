from src.common import CompressedStep, Logger, AnalyzedMove
from src.endgame_knowledge import match as match_endgame
from src.endgame_knowledge import describe_endgame, get_forbidden_concepts
from typing import List, Optional, Tuple
import chess

MAX_NODE_SPAN = 4
LONG_NODE_SPAN = 6
LONG_MOVE_THRESHOLD = 18
COMPACT_NODE_THRESHOLD = 7
MAX_SPAN_CAP = 10        # 压缩跨度上限（再长的解法每节点也不超过10着）

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
    endgame_name = kb["name"] if kb else describe_endgame(board)["name"]
    role_meta = _role_meta(board, endgame_name)
    kbnk_mode = endgame_name == "象马杀王" and len(analyzed_moves) >= LONG_MOVE_THRESHOLD
    long_line_mode = len(analyzed_moves) >= LONG_MOVE_THRESHOLD
    # 压缩跨度随总步数平滑增长，取代旧的 4/6 硬开关：
    #   每节点最多合并的着数 = 4 + (总着数-6)//6，封顶 MAX_SPAN_CAP。
    #   设计为在 18 着阈值处恰好≈6（对齐旧 LONG_NODE_SPAN），短解法仍是4、
    #   长解法继续爬升到10——60着的解法不再用固定6硬切导致节点过多、解说啰嗦。
    total_moves = len(analyzed_moves)
    max_span = min(MAX_NODE_SPAN + max(0, total_moves - 6) // 6, MAX_SPAN_CAP)

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

        compressed.append(CompressedStep(
            idx=len(compressed) + 1,
            sans=[g["san"] for g in grp],
            fen_before=first["fen_before"],
            fen_after=last["fen_after"],
            is_critical=is_critical,
            is_only_move=any(g.get("only") for g in grp),
            trap=trap,
            tags=all_tags,
            eval_delta=total_delta,
            candidates=list(set(c for g in grp for c in g.get("candidates", []))),
        ))

    compressed = _merge_check_sequences(compressed)
    compressed = _merge_repetitive(compressed)

    # 自适应节点预算：节点数随解法长度次线性增长（越长压得越狠），只减不增。
    # 取代旧的「固定阈值 / max_span 上限被语义边界压制」导致的压缩比反相关问题。
    try:
        target_nodes = _adaptive_node_budget(len(per_move))
        compressed = _merge_to_budget(compressed, target_nodes)
    except Exception as e:
        Logger.warn(f"自适应节点预算合并跳过: {e}")

    Logger.info(f"压缩: {len(per_move)} 步 → {len(compressed)} 节点")
    return compressed


def _adaptive_node_budget(total_moves: int) -> int:
    """目标压缩节点数：随解法着数次线性增长，封顶 16、保底 6。

    设计意图：解法越长，单位内容越接近「重复的逼王过程」，应当压得越狠
    （压缩比随长度单调增大），而不是节点数线性膨胀让解说啰嗦、给 AI 留出
    编故事的空节点。系数 0.2 让 15 着≈7 节点、57 着≈15 节点。
    """
    return max(6, min(16, round(4 + 0.2 * total_moves)))


def _node_is_hard_keep(s) -> bool:
    """硬保护节点：含吃子或将军的关键事件，永不作为被吸收的 victim。

    将杀节点是最后一步，由首尾保护覆盖；首尾在 _merge_to_budget 中单独排除。
    """
    return ("吃子" in s.tags) or ("将军" in s.tags)


def _merge_to_budget(steps: List[CompressedStep], target: int) -> List[CompressedStep]:
    """把压缩节点二次合并到接近 target 个（只减不增）。

    规则：
      - 首、尾节点永远保留（叙事开局/收官锚点）；
      - 含吃子/将军的关键事件节点永不被消除（可作为吸收者接纳邻居）；
      - 其余节点按「非关键优先、着数少优先」被选作 victim，并入相邻节点；
      - soft_cap 限制单节点合并后的着数，避免一个画面播太久。
    所有着法仅重新分组、总数不变（不变量，由测试保证）。
    """
    if len(steps) <= target:
        return steps

    work = list(steps)
    total_sans = sum(len(s.sans) for s in work)
    # soft_cap 限制合并后单节点的着数，避免一个画面播太久。
    # 旧实现是"软上限"：都超时仍会 fallback 合并，导致节点可以膨胀到 14+ 步。
    # 新实现收紧为真上限：只有在总量可控时才允许合并，否则跳过该 victim。
    soft_cap = max(6, total_sans // max(target, 1) + 2)
    guard = 0
    skipped_victims = set()  # 本轮因超 soft_cap 而跳过的 victim 索引，不污染 is_critical

    while len(work) > target and guard < 2000:
        guard += 1
        # 选 victim：排除首尾、硬保护节点、以及已被标记跳过的节点
        victim_idx = None
        best_key = None
        for i in range(1, len(work) - 1):
            s = work[i]
            if _node_is_hard_keep(s) or i in skipped_victims:
                continue
            key = (0 if not s.is_critical else 1, len(s.sans))
            if best_key is None or key < best_key:
                best_key = key
                victim_idx = i
        if victim_idx is None:
            break

        i = victim_idx
        v = work[i]
        left = work[i - 1] if i - 1 >= 0 else None
        right = work[i + 1] if i + 1 < len(work) else None

        def fits(t):
            return t is not None and len(t.sans) + len(v.sans) <= soft_cap

        # 只在不超 soft_cap 时合并；都超则标记为不可合并，继续尝试下一个 victim
        if fits(left) and fits(right):
            into_left = len(left.sans) <= len(right.sans)
        elif fits(left):
            into_left = True
        elif fits(right):
            into_left = False
        else:
            # 都超 soft_cap，标记跳过（不污染 is_critical）
            skipped_victims.add(i)
            continue

        if into_left and left is not None:
            left.sans = left.sans + v.sans
            left.fen_after = v.fen_after
            left.tags = list(set(left.tags + v.tags))
            left.is_critical = left.is_critical or v.is_critical
            left.is_only_move = left.is_only_move or v.is_only_move
            left.eval_delta = (left.eval_delta or 0) + (v.eval_delta or 0)
        elif right is not None:
            right.sans = v.sans + right.sans
            right.fen_before = v.fen_before
            right.tags = list(set(right.tags + v.tags))
            right.is_critical = right.is_critical or v.is_critical
            right.is_only_move = right.is_only_move or v.is_only_move
            right.eval_delta = (right.eval_delta or 0) + (v.eval_delta or 0)
        else:
            break
        del work[i]

    for i, s in enumerate(work):
        s.idx = i + 1
    return work

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

def _merge_check_sequences(steps: List[CompressedStep]) -> List[CompressedStep]:
    """将交替将军→非将军的驱赶序列分段合并为叙事节点，每段最多合并8个原始节点"""
    if len(steps) < 4:
        return steps

    n = len(steps)
    skip = [False] * n

    for i in range(n):
        if skip[i]:
            continue

        if "将军" not in steps[i].tags:
            continue

        j = i + 1
        check_count = 1
        quiet_count = 0
        total_sans = len(steps[i].sans)
        while j < n and (j - i) < 8:
            cs_j = steps[j]
            if "吃子" in cs_j.tags:
                break
            if total_sans + len(cs_j.sans) > 12:
                break
            has_check = "将军" in cs_j.tags
            had_check = "将军" in steps[j - 1].tags
            if has_check:
                check_count += 1
            else:
                quiet_count += 1
            if has_check == had_check:
                if (j - i) >= 4:
                    j += 0
                break
            total_sans += len(cs_j.sans)
            j += 1

        run_len = j - i
        if check_count < 2 or run_len < 3:
            continue

        first = steps[i]
        last = steps[j - 1]

        all_sans = []
        for k in range(i, j):
            all_sans.extend(steps[k].sans)

        first_board = chess.Board(first.fen_before)
        last_board = chess.Board(last.fen_after)
        bk_before = first_board.king(chess.BLACK)
        bk_after = last_board.king(chess.BLACK)
        wk_before = first_board.king(chess.WHITE)
        wk_after = last_board.king(chess.WHITE)

        king_parts = []
        if bk_before is not None and bk_after is not None and bk_before != bk_after:
            king_parts.append(f"黑王{chess.square_name(bk_before)}→{chess.square_name(bk_after)}")
        if wk_before is not None and wk_after is not None and wk_before != wk_after:
            king_parts.append(f"白王{chess.square_name(wk_before)}→{chess.square_name(wk_after)}")

        maneuver_pattern = "将军驱赶"
        if check_count >= 4:
            maneuver_pattern = "连续将军驱赶"

        is_repeating = False
        for seg_len in (2, 3, 4):
            if len(all_sans) >= seg_len * 2 and all_sans[:seg_len] == all_sans[seg_len:seg_len * 2]:
                is_repeating = True
                break
        if is_repeating or (king_parts and any(
                any(pat in p for pat in ("h7→h8", "h8→h7", "f7→f8", "f8→f7"))
                for p in king_parts)):
            maneuver_pattern = "反复试探等待"

        merged_cs = CompressedStep(
            idx=0,
            sans=all_sans,
            fen_before=first.fen_before,
            fen_after=last.fen_after,
            is_critical=True,
            is_only_move=any(steps[k].is_only_move for k in range(i, j)),
            trap=first.trap or "",
            tags=[maneuver_pattern],
            eval_delta=sum((steps[k].eval_delta or 0) for k in range(i, j)),
            candidates=[],
        )
        steps[i] = merged_cs
        for k in range(i + 1, j):
            skip[k] = True

    result = [s for idx, s in enumerate(steps) if not skip[idx]]
    for idx, s in enumerate(result):
        s.idx = idx + 1
    return result

def _merge_repetitive(steps: List[CompressedStep]) -> List[CompressedStep]:
    if len(steps) < 3:
        return steps
    # 合并后单节点子步数硬上限：防止 KQvKR 等全程子力不变的残局被压成巨块。
    # 旧逻辑用 len(prev.sans) < 10 做吸收前检查，吸收后可超过 10（如 14 步），
    # 导致 LLM 拿到一个「14 步驱赶」的节点写不出贴合画面的解说。
    # 新逻辑检查吸收后总量，保证每节点最多 8 步。
    _MERGE_REPETITIVE_CAP = 8
    merged = [steps[0]]
    for cur in steps[1:]:
        prev = merged[-1]
        if (not prev.is_critical and not cur.is_critical
                and _same_pieces(prev.fen_before, cur.fen_before)
                and len(prev.sans) + len(cur.sans) <= _MERGE_REPETITIVE_CAP):
            prev.sans.extend(cur.sans)
            prev.fen_after = cur.fen_after
            prev.tags = list(set(prev.tags + cur.tags))
            prev.is_only_move = prev.is_only_move or cur.is_only_move
            if len(prev.sans) >= 6:
                prev.tags.append("对王调整")
        else:
            merged.append(cur)
    for i, s in enumerate(merged):
        s.idx = i + 1
    return merged

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

def _role_meta(board: chess.Board, endgame_name: str, winner_color=None) -> dict:
    white_score = _material_score(board, chess.WHITE)
    black_score = _material_score(board, chess.BLACK)
    # 优先按「实际终局赢家」定强弱（winner_color 由 pipeline 复盘终局得出）：
    # 终局里赢的一方就是强方，这样解说立场永远与画面一致。
    # 无终局信息(线被截断)或和棋时，回退按材料判断。
    if winner_color is not None:
        strong = winner_color
        weak = chess.BLACK if strong == chess.WHITE else chess.WHITE
        if (white_score > black_score and strong != chess.WHITE) or \
           (black_score > white_score and strong != chess.BLACK):
            Logger.warn(f"立场修正: 材料强方≠实际赢家，按终局结果以{_color_name(strong)}为取胜方解说")
    else:
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
            from src.insight_extractor import extract_for_compressed
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
            pass
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
            "moves_display": _compact_moves_display(cs.sans),
            "move_count": len(cs.sans),
            "is_critical": cs.is_critical,
            "phase": cs.phase,
            "phase_hint": phase_hint,
            "tags": cs.tags,
            "trap": cs.trap,
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

            # importance 不再注入为"结论标签"（旧行为：注入 importance=high/low
            # 并据此上调 is_critical）。改为仅用于内部参考，不写入 node，
            # 让 LLM 自己从 tactical_narratives / teaching_point / spatial_change
            # 中判断哪一步是关键手。
            # 旧逻辑保留但不再向 prompt 注入 importance 标签。

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

    return {
        "endgame_name": endgame_name,
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
