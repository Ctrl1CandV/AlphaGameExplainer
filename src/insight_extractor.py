from src.common import PIECE_CN as _PIECE_CN, piece_cn as _piece_cn, PIECE_VALUES as _PIECE_VALUES
from typing import List, Optional
import chess

"""
棋理洞察提取层
职责：用纯 chess.Board API（零引擎依赖）从每个压缩节点的起止局面中算出
**结构化、可表达的棋理事实**，喂给 LM，解决"信息贫瘠→只能堆比喻"的根因
设计约束（与项目其余部分约定一致）：
  1. 纯 board API，不持有 Stockfish/表库实例——所有指标 from chess.Board
     需要"是否唯一好棋"这类引擎信息时，由调用方从 AnalyzedMove 透传，不在此重算
  2. 产出文本一律**关系化、不含坐标**：画面里棋盘/箭头已标出精确格子
     语音的职责是用方位关系说清"这一步改变了什么"。所以 teaching_point
     must_mention 用"边线""角落""逃格""对王"这类词，绝不出现 a1-h8
  3. 失败安全：单节点提取抛异常只返回空洞察，不影响其余节点，更不影响主链路
对外主入口：extract_for_compressed(compressed, root_board, role_meta, endgame_name)
"""

_REGION_CN = {
    "corner": "角落", "edge": "边线", "center": "中心", "near_center": "中心一带",
}

def _king_safe_squares(board: chess.Board, color: chess.Color) -> set:
    """
    返回color方王在当前局面下"能安全去"的相邻格集合（近似王活动度）
    判定：相邻格中，非己方占用、不与对方王相邻、不被对方攻击
    这是衡量"王还剩多少活动空间"的稳健指标，且不依赖轮到谁走
    （legal_moves 只算轮走方，残局里对方王常常不是轮走方）
    注：滑子穿过王当前格的 x 光攻击会被王自身遮挡而少算，属残局叙事可接受的近似
    """
    ksq = board.king(color)
    if ksq is None:
        return set()
    enemy = not color
    enemy_king = board.king(enemy)
    enemy_king_zone = set(chess.SquareSet(chess.BB_KING_ATTACKS[enemy_king])) if enemy_king is not None else set()
    out = set()
    for sq in chess.SquareSet(chess.BB_KING_ATTACKS[ksq]):
        piece = board.piece_at(sq)
        if piece is not None and piece.color == color:
            continue
        if sq in enemy_king_zone:
            continue
        if board.is_attacked_by(enemy, sq):
            continue
        out.add(sq)
    return out

def _square_region(sq: int) -> str:
    """把一个格子归到棋盘区域：corner / edge / center / near_center。"""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    if f in (0, 7) and r in (0, 7):
        return "corner"
    if f in (0, 7) or r in (0, 7):
        return "edge"
    if f in (3, 4) and r in (3, 4):
        return "center"
    return "near_center"

def _detect_opposition(board: chess.Board) -> str:
    """ 两王相对态势（本地最小实现，避免与 toryboard形成循环依赖） """
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return ""
    df = abs(chess.square_file(wk) - chess.square_file(bk))
    dr = abs(chess.square_rank(wk) - chess.square_rank(bk))
    if df == 0 and dr == 2:
        return "正对王（竖向，逼对方让路）"
    if df == 2 and dr == 0:
        return "正对王（横向，逼对方让路）"
    if df == 2 and dr == 2:
        return "斜向对王"
    return ""

def _replay_node(board_before: chess.Board, sans: List[str],
                 strong_color: Optional[chess.Color]):
    """
    回放节点内所有着，提取动作事实。
    返回 dict：
      strong_actions: [(piece_type, gives_check, is_capture, is_promo), ...] 强方的着
      weak_king_fled: 弱方王是否在本节点内移动过
      last_check: 本节点最后一着是否将军
      board_after: 走完后的局面
    """
    temp = board_before.copy()
    strong_actions = []
    weak_king_fled = False
    last_check = False
    for san in sans:
        try:
            mv = temp.parse_san(san)
        except ValueError:
            continue
        mover = temp.turn
        pc = temp.piece_at(mv.from_square)
        chk = temp.gives_check(mv)
        cap = temp.is_capture(mv)
        promo = mv.promotion is not None
        temp.push(mv)
        last_check = chk
        if strong_color is not None and mover == strong_color:
            strong_actions.append((pc.piece_type if pc else None, chk, cap, promo))
        elif pc is not None and pc.piece_type == chess.KING:
            weak_king_fled = True
    return {
        "strong_actions": strong_actions,
        "weak_king_fled": weak_king_fled,
        "last_check": last_check,
        "board_after": temp,
    }


def _action_phrase(action, endgame_name: str) -> str:
    """ 把单个强方动作转成关系化短语 """
    pt, chk, cap, promo = action
    name = _piece_cn(pt)
    if promo:
        return "兵推到底线升变"
    if pt == chess.PAWN:
        return "推进兵、向升变格逼近"
    if chk and cap:
        return f"{name}吃子并将军"
    if cap:
        return f"{name}吃掉对方子力"
    if chk:
        return f"{name}将军、逼对方王让步"
    if pt == chess.KING:
        return "王跟上来、缩短与对方王的距离"
    if pt == chess.KNIGHT:
        return "马跳到卡住对方王逃格的位置"
    if pt == chess.BISHOP:
        return "象沿斜线封住一条逃逸路线"
    if pt == chess.ROOK:
        return "车横切一条线、把对方王压在一侧"
    if pt == chess.QUEEN:
        return "后从远处罩住对方王的活动区"
    return f"{name}调整站位"

def _compose_teaching(facts: dict, endgame_name: str) -> str:
    """ 把结构化事实拼成 1-3 句关系化教学点，事实不足时返回空串 """
    clauses = []

    # 1) 多着合并的机动块：整体描述，不逐着数
    if facts.get("maneuver_label"):
        clauses.append(facts["maneuver_label"])
    else:
        # 2) 单/少着：强方最后一个有意义动作
        last_action = facts.get("last_strong_action_phrase")
        if last_action:
            clauses.append(last_action)

    # 3) 对王态势（如果成立）
    if facts.get("opposition"):
        clauses.append(facts["opposition"])

    # 4) 空间变化（具体数字，最有信息量）
    wb = facts.get("weak_before")
    wa = facts.get("weak_after")
    if wb is not None and wa is not None and facts.get("role_known"):
        if facts.get("is_checkmate_after"):
            clauses.append("对方王已无处可逃，被将死")
        elif wa <= 1 and wa < wb:
            clauses.append("对方王几乎被锁死，只剩一个格子可动")
        elif wa < wb:
            clauses.append(f"对方王能走的格子从{wb}个减到{wa}个")
        elif wa == wb and facts.get("same_position"):
            clauses.append("局面回到原样，本质是调子试探、等一步")

    # 5) 阶段里程碑（首次到边线/角落）
    milestone = facts.get("milestone")
    if milestone:
        clauses.append(milestone)

    if not clauses:
        return ""
    text = "，".join(clauses) + "。"
    return text

def _compose_must_mention(facts: dict) -> List[str]:
    """1-3 条最该讲到的硬事实（关系化），供 prompt 软提示。"""
    bullets = []
    if facts.get("last_strong_action_phrase") and not facts.get("maneuver_label"):
        bullets.append(facts["last_strong_action_phrase"])
    if facts.get("maneuver_label"):
        bullets.append(facts["maneuver_label"])
    wb, wa = facts.get("weak_before"), facts.get("weak_after")
    if facts.get("role_known") and wb is not None and wa is not None and wa < wb:
        bullets.append(f"对方王活动格收窄（{wb}→{wa}）")
    if facts.get("milestone"):
        bullets.append(facts["milestone"])
    # 去重、限长
    seen = set()
    out = []
    for b in bullets:
        if b and b not in seen:
            seen.add(b)
            out.append(b)
    return out[:3]

def _material_signature(board: chess.Board, color: chess.Color) -> tuple:
    """ 返回 color 方除王外的子力组成 """
    counts = {}
    for piece in board.piece_map().values():
        if piece.color == color and piece.piece_type != chess.KING:
            counts[piece.piece_type] = counts.get(piece.piece_type, 0) + 1
    return tuple(sorted(counts.items()))

def _sig_name(sig: tuple) -> str:
    """ 子力签名 → 中文简称 """
    if not sig:
        return "单王"
    parts = []
    for pt, cnt in sig:
        name = _PIECE_CN.get(pt, "?")
        parts.append(f"{cnt}{name}" if cnt > 1 else name)
    return "".join(parts)


def _extract_tactical_narrative(cs, board_before, board_after, role_meta) -> List[str]:
    """
    检测本节点中需要棋理推理才能理解的战术关系。
    只产出 1-3 句纯棋理中文叙述（无坐标、无评判词）。
    失败安全：任何异常返回空列表。
    """
    try:
        narratives = []
        weak_color = role_meta.get("weak_color") if role_meta else None
        strong_color = role_meta.get("strong_color") if role_meta else None

        if not strong_color or not weak_color:
            return []

        # ---- 1. 逐着检测双重攻击 + 被迫丢子 ----
        temp = board_before.copy()
        for san in cs.sans:
            try:
                move = temp.parse_san(san)
            except ValueError:
                continue

            mover_color = temp.turn
            gives_check = temp.gives_check(move)

            # 只有强方的着才需要分析战术结构
            if gives_check and mover_color == strong_color:
                # 推演后局面
                temp2 = temp.copy()
                temp2.push(move)

                # 走子的位置是否同时攻击对方无保护的大子？
                moved_sq = move.to_square
                for attacked_sq in temp2.attacks(moved_sq):
                    target = temp2.piece_at(attacked_sq)
                    if (target is None or target.color != weak_color
                            or target.piece_type == chess.KING
                            or target.piece_type == chess.PAWN):
                        continue

                    # 检查这个大子是否被保护
                    defenders = temp2.attackers(target.color, attacked_sq)
                    if defenders:
                        continue  # 有保护，不算双重攻击

                    # 双重攻击成立。现在检查弱方的应将是否能保住它。
                    # "保住"定义：应将后强方是否仍能无代价吃掉该子
                    # （子仍在盘上 ∧ 仍被强方攻击 ∧ 无足够保护）。
                    # 旧逻辑只查"子是否还在原格"，不识别子已逃走/王吃掉将军子。
                    weak_name = "白" if weak_color == chess.WHITE else "黑"
                    target_name = _PIECE_CN.get(target.piece_type, "子")
                    can_save = False
                    for reply in temp2.legal_moves:
                        temp3 = temp2.copy()
                        temp3.push(reply)
                        target_piece = temp3.piece_at(attacked_sq)
                        if target_piece is None:
                            # 子已不在原格（逃走或被吃）→ 检查它是否在新格安全
                            continue
                        # 子还在原格：强方是否仍能攻击它？
                        strong_attackers = temp3.attackers(strong_color, attacked_sq)
                        weak_defenders = temp3.attackers(weak_color, attacked_sq)
                        if not strong_attackers:
                            # 强方已无法攻击该子 → 保住
                            can_save = True
                            break
                        # 强方仍能攻击：检查弱方保护是否足够（子交换不亏）
                        if len(weak_defenders) >= len(strong_attackers):
                            can_save = True
                            break

                    if not can_save:
                        narratives.append(
                            f"这一着同时做了两件事：给{weak_name}王将军，同时直接攻击{weak_name}{target_name}。"
                            f"{weak_name}方必须应将，但在所有合法的应将走法中，"
                            f"没有一步能同时保住{weak_name}{target_name}——"
                            f"这意味着{weak_name}{target_name}必定在下一步被吃掉。"
                        )
                    else:
                        narratives.append(
                            f"这一着同时将军并攻击{weak_name}{target_name}——一子两用，"
                            f"对方必须应将的同时还要处理{target_name}的威胁。"
                        )
                    break  # 一个节点只报告一次双重攻击

            temp.push(move)

        # ---- 2. 检测残局类型质变 ----
        if cs.fen_before and cs.fen_after:
            try:
                bf = chess.Board(cs.fen_before)
                af = chess.Board(cs.fen_after)
                strong_before = _material_signature(bf, strong_color)
                strong_after = _material_signature(af, strong_color)
                weak_before = _material_signature(bf, weak_color)
                weak_after = _material_signature(af, weak_color)

                # 子力组成变了 → 残局类型变了
                if strong_before != strong_after or weak_before != weak_after:
                    before_full = f"{_sig_name(strong_before)}对{_sig_name(weak_before)}"
                    after_full = f"{_sig_name(strong_after)}对{_sig_name(weak_after)}"
                    if before_full != after_full:
                        # 判断是否"简化到已知必胜残局"
                        # 只有当强方仍保留至少一车或一后时才断言必胜；
                        # 单马/单象/双马对单王是理论和棋（不能逼杀），不能断言必胜。
                        strong_has_heavy = any(
                            pt in (chess.QUEEN, chess.ROOK)
                            for pt, _ in strong_after
                        )
                        if not weak_after and strong_has_heavy:
                            narratives.append(
                                f"这一步之后，局面从「{before_full}」变为「{after_full}」——"
                                f"残局类型发生了质变。{_sig_name(strong_after)}对单王是已知的必胜残局，"
                                f"后续推进只是时间问题。"
                            )
                        elif not weak_after:
                            narratives.append(
                                f"这一步之后，局面从「{before_full}」变为「{after_full}」——"
                                f"残局类型发生了质变。"
                            )
                        else:
                            narratives.append(
                                f"这一步之后，局面从「{before_full}」变为「{after_full}」——"
                                f"残局类型发生了改变。"
                            )
            except Exception:
                pass

        # 去重限长
        seen = set()
        out = []
        for n in narratives:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out[:3]

    except Exception:
        return []


def _compute_importance(facts: dict, cs_is_critical: bool) -> tuple:
    """ 语义重要性评分 → (level, reasons)。比旧的纯标签判定更贴近棋理 """
    score = 0
    reasons = []

    if facts.get("is_checkmate_after"):
        return "high", ["形成将杀，收官节点"]

    if facts.get("maneuver_label"):
        score += 15
        reasons.append("成段的驱赶/机动，演示核心技法")

    wb, wa = facts.get("weak_before"), facts.get("weak_after")
    if facts.get("role_known") and wb is not None and wa is not None and wb > 0:
        red = round((1 - wa / wb) * 100)
        if red >= 50:
            score += 25
            reasons.append(f"对方王活动空间锐减{red}%")
        elif red >= 25:
            score += 12
            reasons.append(f"对方王活动空间收窄{red}%")
        if wa <= 1:
            score += 20
            reasons.append("对方王几乎被锁死")

    if facts.get("milestone"):
        score += 20
        reasons.append(facts["milestone"])

    if facts.get("last_check"):
        score += 8

    if facts.get("opposition"):
        score += 8
        reasons.append("形成对王，掌握主动权")

    # 兼容旧判定：原本就被标为关键的，至少给中等
    if cs_is_critical:
        score += 8

    if score >= 28:
        return "high", reasons
    if score >= 13:
        return "medium", reasons
    return "low", reasons

def _puzzle_tactical_facts(board_before: chess.Board, cs, board_after: chess.Board,
                           role_meta: Optional[dict]) -> List[str]:
    """E 类：puzzle 战术几何事实（Phase 1：走子后攻击目标 + 悬子检测）。

    返回关系化中文叙述列表，不含坐标。
    """
    facts = []
    try:
        weak_color = role_meta.get("weak_color") if role_meta else None
        strong_color = role_meta.get("strong_color") if role_meta else None
        if not strong_color or not weak_color:
            return facts

        temp = board_before.copy()
        for san in cs.sans:
            try:
                move = temp.parse_san(san)
            except ValueError:
                continue

            mover = temp.turn
            temp.push(move)

            if mover != strong_color:
                continue

            # 1) 走子后该子攻击了哪些高价值目标
            moved_sq = move.to_square
            high_value_targets = []
            for atk_sq in temp.attacks(moved_sq):
                target = temp.piece_at(atk_sq)
                if target is None or target.color != weak_color:
                    continue
                if target.piece_type in (chess.KING, chess.PAWN):
                    continue
                high_value_targets.append(_piece_cn(target.piece_type))
            if high_value_targets:
                targets_str = "、".join(high_value_targets)
                facts.append(f"走子后，该子同时瞄住了对方的{targets_str}")

            # 2) 检测悬子（hanging piece）：无保护或保护不足的弱方子
            for sq, piece in temp.piece_map().items():
                if piece.color != weak_color or piece.piece_type == chess.KING:
                    continue
                if piece.piece_type == chess.PAWN:
                    continue
                attackers = temp.attackers(strong_color, sq)
                defenders = temp.attackers(weak_color, sq)
                if attackers and len(defenders) < len(attackers):
                    cn = _piece_cn(piece.piece_type)
                    if len(defenders) == 0:
                        facts.append(f"对方的{cn}处于悬空状态，没有任何保护")
                    else:
                        facts.append(f"对方的{cn}保护不足，攻方子力多于守方")

        # 去重
        seen = set()
        out = []
        for f in facts:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out[:4]
    except Exception:
        return []




def _net_material_fact(
        start_board: chess.Board, moves: list, strong_color: chess.Color
    ) -> str:
    """计算整串走法走完后，强方相对开局的子力净值，返回一句中文叙述。
    无净值优势时返回空串，由调用方处理杀棋判定。"""
    try:
        strong_captured = {}   # 强方吃掉的对方子（仅真实吃子）
        weak_captured = {}     # 弱方吃掉的强方子（仅真实吃子）
        has_promotion = False  # 强方是否包含升变（用于纯升变文本）
        temp = start_board.copy()

        for move in moves:
            mover = temp.turn

            # 只追踪真实吃子，不混入升变
            captured_type = None
            if temp.is_en_passant(move):
                captured_type = chess.PAWN
            else:
                victim = temp.piece_at(move.to_square)
                if victim is not None:
                    captured_type = victim.piece_type
            if captured_type is not None:
                bucket = strong_captured if mover == strong_color else weak_captured
                bucket[captured_type] = bucket.get(captured_type, 0) + 1

            # 强方升变仅标记，不混入吃子桶（升变≠吃子，混入会污染对消逻辑的文本输出）
            if move.promotion and mover == strong_color:
                has_promotion = True

            temp.push(move)

        def _value(bucket):
            return sum(_PIECE_VALUES.get(pt, 0) * n for pt, n in bucket.items())

        net_points = _value(strong_captured) - _value(weak_captured)

        # 纯升变（无吃子）：净点值为 0，但升变本身就是实质性收益
        if net_points == 0 and has_promotion:
            return "经过升变，强方获得了更强的子力"

        if net_points <= 0:
            return ""

        # 对消双方共有的棋子类型，尽量映射到单一棋子名
        remain = dict(strong_captured)
        for pt, n in weak_captured.items():
            remain[pt] = remain.get(pt, 0) - n
        remain = {pt: n for pt, n in remain.items() if n > 0}

        if len(remain) == 1:
            pt, n = next(iter(remain.items()))
            unit = _piece_cn(pt)
            count_cn = "一" if n == 1 else ("两" if n == 2 else str(n))
            return f"经过这一连串交换，强方净赢{count_cn}个{unit}"

        # 无法映射到单一棋子：吃子+升变混杂时用点值
        return f"经过这一连串交换，强方净多得约{net_points}个兵的子力价值"
    except Exception:
        return ""

def extract_for_node(
    cs, root_winner_strong: Optional[chess.Color],
    role_meta: Optional[dict], endgame_name: str, prev_state: dict,
    mode: str = "endgame"
    ) -> dict:
    """
    对单个压缩节点提取洞察。prev_state 跨节点累计（已到边线/角落标记）。
    失败安全：任何异常都返回空洞察 dict，不抛出。

    mode="endgame"（默认）：走现有逻辑，零影响。
    mode="puzzle"：关闭残局专用事实（B/C 类），保留战术叙述（D 类），
                  新增战术几何事实（E 类）。
    """
    try:
        board_before = chess.Board(cs.fen_before)
        strong_color = None
        weak_color = None
        if role_meta:
            strong_color = role_meta.get("strong_color")
            weak_color = role_meta.get("weak_color")

        replay = _replay_node(board_before, list(cs.sans), strong_color)
        board_after = replay["board_after"]

        facts = {
            "role_known": weak_color is not None,
            "same_position": cs.fen_before == cs.fen_after,
            "is_checkmate_after": board_after.is_checkmate(),
            "last_check": replay["last_check"],
        }

        is_puzzle = (mode == "puzzle")

        # 强方最后一个动作短语（两种模式通用）
        if replay["strong_actions"]:
            facts["last_strong_action_phrase"] = _action_phrase(
                replay["strong_actions"][-1], endgame_name)

        # 多着机动块（B 类：puzzle 模式下跳过）
        if not is_puzzle:
            maneuver_tags = {"将军驱赶", "连续将军驱赶", "反复试探等待", "对王调整"}
            hit = [t for t in (cs.tags or []) if t in maneuver_tags]
            if hit and len(cs.sans) >= 3:
                label_map = {
                    "连续将军驱赶": "这是一连串将军驱赶，把对方王一路逼着退",
                    "将军驱赶": "用将军一步步把对方王往边角赶",
                    "反复试探等待": "反复调子试探、等一步，逼对方先动",
                    "对王调整": "围绕对王来回调整，争夺关键格",
                }
                facts["maneuver_label"] = label_map.get(hit[0], "")

        # 对王（B 类：puzzle 模式下跳过）
        if not is_puzzle:
            opp = _detect_opposition(board_after)
            if opp:
                facts["opposition"] = opp

        # 空间变化（B 类：puzzle 模式下跳过弱方王活动度分析）
        if weak_color is not None and not is_puzzle:
            safe_before = _king_safe_squares(board_before, weak_color)
            safe_after = _king_safe_squares(board_after, weak_color)
            facts["weak_before"] = len(safe_before)
            facts["weak_after"] = len(safe_after)
            facts["escapes_cut"] = len(safe_before - safe_after)

            wk_after = board_after.king(weak_color)
            region = _square_region(wk_after) if wk_after is not None else "center"
            facts["king_region"] = region
            on_edge = region in ("edge", "corner")
            in_corner = region == "corner"

            # 阶段里程碑（B 类：puzzle 模式下跳过）
            if in_corner and not prev_state.get("in_corner"):
                facts["milestone"] = "对方王首次被逼进角落，进入收网阶段"
                prev_state["in_corner"] = True
                prev_state["on_edge"] = True
            elif on_edge and not prev_state.get("on_edge"):
                facts["milestone"] = "对方王首次被压到边线"
                prev_state["on_edge"] = True

            facts["on_edge"] = on_edge
            facts["in_corner"] = in_corner
        elif weak_color is not None and is_puzzle:
            # puzzle 模式：仍记录 is_checkmate_after 供后续使用
            facts["weak_before"] = None
            facts["weak_after"] = None

        teaching = _compose_teaching(facts, endgame_name)
        must = _compose_must_mention(facts)
        importance, reasons = _compute_importance(facts, getattr(cs, "is_critical", False))

        # 战术叙述（D 类：puzzle 模式下保留但关闭 C 类"必胜残局"断言）
        if is_puzzle:
            # 只提取双重攻击部分，跳过残局类型质变→必胜残局断言
            tactical_narratives = _extract_tactical_narrative_puzzle(
                cs, board_before, board_after, role_meta)
        else:
            tactical_narratives = _extract_tactical_narrative(
                cs, board_before, board_after, role_meta)

        spatial = {}
        if facts.get("role_known") and facts.get("weak_before") is not None:
            wb, wa = facts["weak_before"], facts["weak_after"]
            spatial = {
                "weak_before": wb,
                "weak_after": wa,
                "reduction_pct": round((1 - wa / wb) * 100) if wb > 0 else 0,
                "king_region": _REGION_CN.get(facts.get("king_region", ""), ""),
                "escapes_cut": facts.get("escapes_cut", 0),
            }

        result = {
            "teaching_point": teaching,
            "must_mention": must,
            "importance": importance,
            "importance_reasons": reasons,
            "spatial_change": spatial,
            "tactical_narratives": tactical_narratives,
        }

        # E 类：puzzle 战术几何事实（Phase 1）
        if is_puzzle:
            geo_facts = _puzzle_tactical_facts(board_before, cs, board_after, role_meta)
            if geo_facts:
                result["puzzle_tactical_facts"] = geo_facts

        return result
    except Exception:
        return {
            "teaching_point": "",
            "must_mention": [],
            "importance": "medium",
            "importance_reasons": [],
            "spatial_change": {},
            "tactical_narratives": [],
        }


def _extract_tactical_narrative_puzzle(cs, board_before, board_after, role_meta) -> List[str]:
    """puzzle 模式专用战术叙述：只提取双重攻击部分，跳过残局类型质变断言（C 类关闭）。"""
    try:
        narratives = []
        weak_color = role_meta.get("weak_color") if role_meta else None
        strong_color = role_meta.get("strong_color") if role_meta else None

        if not strong_color or not weak_color:
            return []

        # 逐着检测双重攻击 + 被迫丢子（与 endgame 版相同逻辑）
        temp = board_before.copy()
        for san in cs.sans:
            try:
                move = temp.parse_san(san)
            except ValueError:
                continue

            mover_color = temp.turn
            gives_check = temp.gives_check(move)

            if gives_check and mover_color == strong_color:
                temp2 = temp.copy()
                temp2.push(move)

                moved_sq = move.to_square
                for attacked_sq in temp2.attacks(moved_sq):
                    target = temp2.piece_at(attacked_sq)
                    if (target is None or target.color != weak_color
                            or target.piece_type == chess.KING
                            or target.piece_type == chess.PAWN):
                        continue

                    defenders = temp2.attackers(target.color, attacked_sq)
                    if defenders:
                        continue

                    weak_name = "白" if weak_color == chess.WHITE else "黑"
                    target_name = _PIECE_CN.get(target.piece_type, "子")
                    can_save = False
                    for reply in temp2.legal_moves:
                        temp3 = temp2.copy()
                        temp3.push(reply)
                        target_piece = temp3.piece_at(attacked_sq)
                        if target_piece is None:
                            continue
                        strong_attackers = temp3.attackers(strong_color, attacked_sq)
                        weak_defenders = temp3.attackers(weak_color, attacked_sq)
                        if not strong_attackers:
                            can_save = True
                            break
                        if len(weak_defenders) >= len(strong_attackers):
                            can_save = True
                            break

                    if not can_save:
                        narratives.append(
                            f"这一着同时做了两件事：给{weak_name}王将军，同时直接攻击{weak_name}{target_name}。"
                            f"{weak_name}方必须应将，但在所有合法的应将走法中，"
                            f"没有一步能同时保住{weak_name}{target_name}——"
                            f"这意味着{weak_name}{target_name}必定在下一步被吃掉。"
                        )
                    else:
                        narratives.append(
                            f"这一着同时将军并攻击{weak_name}{target_name}——一子两用，"
                            f"对方必须应将的同时还要处理{target_name}的威胁。"
                        )
                    break

            temp.push(move)

        # 去重限长
        seen = set()
        out = []
        for n in narratives:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out[:3]

    except Exception:
        return []

def extract_for_compressed(
    compressed: List, root_board: chess.Board,
    role_meta: Optional[dict] = None, endgame_name: str = "",
    mode: str = "endgame"
    ) -> List[dict]:
    """对整个压缩序列提取洞察，返回与 compressed 等长的 list。

    role_meta: storyboard._role_meta 的产物（含 strong_color/weak_color）。
               为空时仍可提取动作/对王等与立场无关的事实。
    mode: "endgame"（默认，零影响）或 "puzzle"（关闭残局专用事实、启用战术几何）。
    """
    insights = []
    prev_state = {"on_edge": False, "in_corner": False}
    root_winner_strong = role_meta.get("strong_color") if role_meta else None
    for cs in compressed:
        insights.append(extract_for_node(
            cs, root_winner_strong, role_meta, endgame_name, prev_state, mode=mode))
    return insights