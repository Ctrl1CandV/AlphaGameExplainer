"""棋理洞察提取层（src/insight_extractor.py）。

职责：用纯 chess.Board API（零引擎依赖）从每个压缩节点的起止局面中算出
**结构化、可表达的棋理事实**，喂给 LLM，解决"信息贫瘠→只能堆比喻"的根因。

设计约束（与项目其余部分约定一致）：
  1. 纯 board API，不持有 Stockfish/表库实例——所有指标 from chess.Board。
     需要"是否唯一好棋"这类引擎信息时，由调用方从 AnalyzedMove 透传，不在此重算。
  2. 产出文本一律**关系化、不含坐标**：画面里棋盘/箭头已标出精确格子，
     语音的职责是用方位关系说清"这一步改变了什么"。所以 teaching_point /
     must_mention 用"边线""角落""逃格""对王"这类词，绝不出现 a1-h8。
  3. 失败安全：单节点提取抛异常只返回空洞察，不影响其余节点，更不影响主链路。

对外主入口：extract_for_compressed(compressed, root_board, role_meta, endgame_name)
"""
from typing import List, Optional, Dict
import chess


_PIECE_CN = {
    chess.KING: "王", chess.QUEEN: "后", chess.ROOK: "车",
    chess.BISHOP: "象", chess.KNIGHT: "马", chess.PAWN: "兵",
}


def _piece_cn(pt: Optional[int]) -> str:
    return _PIECE_CN.get(pt, "子")


def _king_safe_squares(board: chess.Board, color: chess.Color) -> set:
    """返回 color 方王在当前局面下"能安全去"的相邻格集合（近似王活动度）。

    判定：相邻格中，非己方占用、不与对方王相邻、不被对方攻击。
    这是衡量"王还剩多少活动空间"的稳健指标，且不依赖轮到谁走
    （legal_moves 只算轮走方，残局里对方王常常不是轮走方）。
    注：滑子穿过王当前格的 x 光攻击会被王自身遮挡而少算，
    属残局叙事可接受的近似。
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
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    if f in (0, 7) and r in (0, 7):
        return "corner"
    if f in (0, 7) or r in (0, 7):
        return "edge"
    if f in (3, 4) and r in (3, 4):
        return "center"
    return "near_center"


_REGION_CN = {
    "corner": "角落", "edge": "边线", "center": "中心", "near_center": "中心一带",
}


def _detect_opposition(board: chess.Board) -> str:
    """两王相对态势（本地最小实现，避免与 storyboard 形成循环依赖）。"""
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
    """回放节点内所有着，提取动作事实。

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
    """把单个强方动作转成关系化短语（无坐标）。"""
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
    """把结构化事实拼成 1-3 句关系化教学点（无坐标）。事实不足时返回空串。"""
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


def _compute_importance(facts: dict, cs_is_critical: bool) -> tuple:
    """语义重要性评分 → (level, reasons)。比旧的纯标签判定更贴近棋理。"""
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


def extract_for_node(cs, root_winner_strong: Optional[chess.Color],
                     role_meta: Optional[dict], endgame_name: str,
                     prev_state: dict) -> dict:
    """对单个压缩节点提取洞察。prev_state 跨节点累计（已到边线/角落标记）。

    失败安全：任何异常都返回空洞察 dict，不抛出。
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

        # 强方最后一个动作短语
        if replay["strong_actions"]:
            facts["last_strong_action_phrase"] = _action_phrase(
                replay["strong_actions"][-1], endgame_name)

        # 多着机动块：用 tags 里的标签作整体描述
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

        # 对王
        opp = _detect_opposition(board_after)
        if opp:
            facts["opposition"] = opp

        # 空间变化（弱方王活动度）
        if weak_color is not None:
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

            # 阶段里程碑：首次到边线 / 首次到角落
            if in_corner and not prev_state.get("in_corner"):
                facts["milestone"] = "对方王首次被逼进角落，进入收网阶段"
                prev_state["in_corner"] = True
                prev_state["on_edge"] = True
            elif on_edge and not prev_state.get("on_edge"):
                facts["milestone"] = "对方王首次被压到边线"
                prev_state["on_edge"] = True

            facts["on_edge"] = on_edge
            facts["in_corner"] = in_corner

        teaching = _compose_teaching(facts, endgame_name)
        must = _compose_must_mention(facts)
        importance, reasons = _compute_importance(facts, getattr(cs, "is_critical", False))

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

        return {
            "teaching_point": teaching,
            "must_mention": must,
            "importance": importance,
            "importance_reasons": reasons,
            "spatial_change": spatial,
        }
    except Exception:
        # 失败安全：返回空洞察，调用方按"无洞察"处理，行为退回旧链路
        return {
            "teaching_point": "",
            "must_mention": [],
            "importance": "medium",
            "importance_reasons": [],
            "spatial_change": {},
        }


def extract_for_compressed(compressed: List, root_board: chess.Board,
                           role_meta: Optional[dict] = None,
                           endgame_name: str = "") -> List[dict]:
    """对整个压缩序列提取洞察，返回与 compressed 等长的 list。

    role_meta: storyboard._role_meta 的产物（含 strong_color/weak_color）。
               为空时仍可提取动作/对王等与立场无关的事实。
    """
    insights = []
    prev_state = {"on_edge": False, "in_corner": False}
    root_winner_strong = role_meta.get("strong_color") if role_meta else None
    for cs in compressed:
        insights.append(extract_for_node(
            cs, root_winner_strong, role_meta, endgame_name, prev_state))
    return insights
