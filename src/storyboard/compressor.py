from src.common import CompressedStep, Logger, AnalyzedMove
from src.analysis.endgame_kb import match as match_endgame, describe_endgame
from src.chess_utils.position import tag_position as _tag_position
from src.chess_utils.position import piece_square as _piece_square
from src.chess_utils.material import material_score as _material_score, color_name as _color_name
import chess
from typing import List, Optional

MAX_NODE_SPAN = 4
LONG_MOVE_THRESHOLD = 18
COMPACT_NODE_THRESHOLD = 7
# 压缩跨度上限
MAX_SPAN_CAP = 10

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

        compressed.append(CompressedStep(
            idx=len(compressed) + 1,
            sans=[g["san"] for g in grp],
            fen_before=first["fen_before"],
            fen_after=last["fen_after"],
            is_critical=is_critical,
            is_only_move=any(g.get("only") for g in grp),
            tags=all_tags,
            eval_delta=total_delta,
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
            tags=[maneuver_pattern],
            eval_delta=sum((steps[k].eval_delta or 0) for k in range(i, j)),
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
