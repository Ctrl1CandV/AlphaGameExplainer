from src.common import CompressedStep, Logger, AnalyzedMove, PIECE_VALUES, piece_cn
from src.chess_utils.material import material_score as _material_score, material_balance as _material_balance, side_material_desc as _side_material_desc, absent_major_pieces as _absent_major_pieces
from src.chess_utils.tactic import is_fork as _is_fork, is_pin as _is_pin, is_skewer as _is_skewer, is_discovered as _is_discovered
from src.analysis.themes_kb import get_theme, select_core_theme, select_narrative_stance, related_intersection
from src.analysis.insight_extractor import extract_for_compressed, per_step_material_fact
from src.storyboard.key_move_locator import collect_puzzle_move_facts as _collect_puzzle_move_facts, locate_theme_key_moves as _locate_theme_key_moves
from typing import List
import chess


def associate_move_with_theme(board_before: chess.Board, move: chess.Move,
                               board_after: chess.Board,
                               effective_themes: List[str],
                               main_theme_key: str = "") -> str:
    """确定走法与哪个标签最相关（仅在候选标签集合内择一，决策三）。

    优先级：终局杀型 → 叉击 → 牵制 → 串击 → 闪击 → 程序选定的主标签兜底。
    """
    # 1. 终局：从 effective_themes 反查杀型标签
    if board_after.is_checkmate():
        mate_in_themes = [t for t in effective_themes if t.endswith("Mate")]
        if mate_in_themes:
            return mate_in_themes[0]
        if "mate" in effective_themes:
            return "mate"

    # 2. 几何可判的战术：仅在 effective_themes 内择一
    if "fork" in effective_themes and _is_fork(board_before, move, board_after):
        return "fork"
    if "pin" in effective_themes and _is_pin(board_before, move, board_after):
        return "pin"
    if "skewer" in effective_themes and _is_skewer(board_before, move, board_after):
        return "skewer"
    if "discoveredAttack" in effective_themes and _is_discovered(board_before, move, board_after):
        return "discoveredAttack"

    # 3. 兜底：使用程序选出的主标签，不依赖原始标签顺序
    if main_theme_key in effective_themes:
        return main_theme_key
    return effective_themes[0] if effective_themes else ""


def _puzzle_target_length(node_count: int, rating: int) -> str:
    """动态字数预算（实施决策 D）：按节点数 × 每节点字数预算，结合 Rating 分层。"""
    if rating < 1500:
        per_node = (90, 140)
    elif rating < 2200:
        per_node = (120, 190)
    else:
        per_node = (150, 240)

    min_len = max(180, node_count * per_node[0])
    max_len = max(320, node_count * per_node[1])
    return f"{min(min_len, 2200)}-{min(max_len, 3200)}字"


def build_for_puzzle( board: chess.Board, moves: List[chess.Move], puzzle) -> dict:
    """
    每个move直接成一个节点，逐步推演棋盘，提取每步事实
    关联标签，注入标签定义，返回与build()输出格式兼容的storyboard dict
    puzzle: PuzzleData，含 effective_themes、rating、opening_tags等
    """

    effective = puzzle.effective_themes
    # 核心标签：优先选战术机理（fork/sacrifice…），而非评估分类（crushing/advantage）
    # 根因3修复：机理标签选择与叙事基调选择正交处理，见 select_narrative_stance。
    main_theme_key = select_core_theme(effective)
    main_theme = get_theme(main_theme_key) or {}
    # 次要标签 + 与核心存在联动关系的标签（供组合叙事）
    secondary_keys = [k for k in effective if k != main_theme_key]
    synergy_keys = related_intersection(main_theme_key, secondary_keys)[:2]

    # 关键手定位：复盘整条解法 → 对每个机理类标签用棋盘事实评分 → 选最高分。
    # 解决"核心标签选对了但关键手讲错"（如 sacrifice 被误选成前面那步将军）。
    move_facts = _collect_puzzle_move_facts(board, moves)
    key_move_map = _locate_theme_key_moves(move_facts, effective, main_theme_key)

    # 单步 CompressedStep 包装，供 insight_extractor 复用
    temp = board.copy()
    single_steps = []
    for i, move in enumerate(moves):
        fen_before = temp.fen()
        san = temp.san(move)
        is_check = temp.gives_check(move)
        is_capture = temp.is_capture(move)
        tags = []
        if is_check:
            tags.append("将军")
        if is_capture:
            tags.append("吃子")
        temp.push(move)
        fen_after = temp.fen()
        single_steps.append(CompressedStep(
            idx=i + 1,
            sans=[san],
            fen_before=fen_before,
            fen_after=fen_after,
            is_critical=True,  # puzzle 每步都是关键手
            tags=tags,
        ))

    # 棋理洞察（puzzle 模式）
    # 构造 puzzle 专用 role_meta（实施决策 C：以 board.turn 走子方为解题方/强方）
    puzzle_role_meta = {
        "strong_color": board.turn,
        "weak_color": not board.turn,
    }
    insights = []
    try:
        insights = extract_for_compressed(
            single_steps, board, role_meta=puzzle_role_meta, endgame_name="",
            mode="puzzle")
    except Exception as e:
        Logger.warn(f"Puzzle 棋理洞察提取失败: {e}")
        insights = []

    # 攻守视角（实施决策 C）
    puzzle_side_color = board.turn
    puzzle_side = "白方" if puzzle_side_color == chess.WHITE else "黑方"
    defending_side = "黑方" if puzzle_side_color == chess.WHITE else "白方"

    # 根因3修复：叙事基调不能再由 select_core_theme 的 tier 系统单点决定。
    # defensiveMove 语义上和 crushing/advantage/equality 一样是"评估类"标签
    # （描述谁占优、叙事基调），却被误放进"机理"桶，导致 crushing+defensiveMove
    # 共存时（Lichess 常见的多标签噪声）机理桶的 tier 0 会压过评估桶的 tier 2，
    # 把明明是碾压进攻的题目错误地定为"防守化解危机"。stance_key 从
    # select_narrative_stance 单独选出，与 main_theme_key（机理教学核心）解耦，
    # 两者互不干扰。
    stance_key = select_narrative_stance(effective)
    narrative_mode = "defensive_resource" if stance_key == "defensiveMove" else "tactical_solution"

    # 兜底核验：即使标签系统选出 defensiveMove，也要用棋盘事实核实解题方
    # 起始局面是否真的处于劣势。若解题方在解题局面开局时子力已占优或均势
    # （不是绝对劣势），则不允许叙事基调被判定为"防守化解危机"——这是用
    # 客观子力事实否决标签噪声的最后一道保险（对应 puzzle_0039T/puzzle_002Uy
    # 两个已确认的攻守颠倒案例：两题都是 crushing+defensiveMove 共存，且解题
    # 方在起始局面并非劣势）。
    if narrative_mode == "defensive_resource":
        solver_material = _material_score(board, puzzle_side_color)
        opponent_material = _material_score(board, not puzzle_side_color)
        if solver_material >= opponent_material:
            narrative_mode = "tactical_solution"

    # 逐节点构建
    temp = board.copy()
    nodes_out = []
    start_board_for_material = board.copy()
    # 根因C修复（前置注入）：解题开局时解题方相对对方的子力差，作为每步
    # per-step material fact 的基线，用于区分"吃回/兑子/真净赢"（见 puzzle_001aK）。
    solver_start_balance = _material_balance(board, puzzle_side_color)
    # PLAN-003 B+：累计「截至当前节点、前序所有节点被吃过的棋子类型集合」。
    # 供 validator 的 validate_material_existence 作为第三个放行来源——解决
    # 「前序节点吃了大子、后续节点回顾该子战术成果却被判捏造」的假阳性
    # （实测样本 puzzle_002Hv 节点1吃马、后续提马被判失败）。注意只记类型
    # 不记数量/时间线，这是 validator 固有局限（见 validators.py 注释）。
    captured_types_sofar = set()
    _CN_TO_PIECE_TYPE = {"后": chess.QUEEN, "车": chess.ROOK,
                         "象": chess.BISHOP, "马": chess.KNIGHT}
    for i, move in enumerate(moves):
        board_before = temp.copy()
        is_check = temp.gives_check(move)
        is_capture = temp.is_capture(move)
        is_checkmate_after = False
        fen_before = temp.fen()
        turn = "白方走" if temp.turn == chess.WHITE else "黑方走"
        san = temp.san(move)
        # 本步确定性子力得失结论（不可改写事实，前置注入 prompt）
        # P0-2：提前计算是否形成将杀，传入 per_step_material_fact 使其 short-circuit
        # 避免 LLM 拿不到"将杀"信号后自行发明得子措辞（002rd Qh6# / 001wR Qd8#）。
        _is_cm_board = board_before.copy()
        _is_cm_board.push(move)
        is_checkmate_after = _is_cm_board.is_checkmate()
        material_fact = per_step_material_fact(
            board_before, san, puzzle_side_color, solver_start_balance,
            is_checkmate_after=is_checkmate_after)
        # 抽取被吃子的具体类型（确定性事实，供解说"吃掉了什么子"而非泛泛"吃子"）
        captured_piece_cn = ""
        if is_capture:
            if temp.is_en_passant(move):
                captured_piece_cn = "兵"
            else:
                captured = temp.piece_at(move.to_square)
                if captured:
                    captured_piece_cn = piece_cn(captured.piece_type)
        temp.push(move)
        fen_after = temp.fen()
        board_after = temp.copy()
        # is_checkmate_after 已在本循环顶部通过 _is_cm_board 计算，此处无需重复。

        # 标签关联
        related_theme = associate_move_with_theme(
            board_before, move, board_after, effective, main_theme_key)
        theme_entry = get_theme(related_theme) or {}

        # 标签上下文稍后按节点在解法中的角色选择，避免每步重复整份知识库。
        theme_context = ""

        # 关键手定位结果回灌：本步属于哪些标签的关键手、是不是核心关键手
        step_idx = i + 1
        roles_here = [k for k, loc in key_move_map.items()
                      if isinstance(loc, dict) and loc.get("idx") == step_idx
                      and k not in ("core",)]
        is_core_key = (key_move_map.get("core", {}) or {}).get("idx") == step_idx
        key_move_reason = ""
        if is_core_key:
            key_move_reason = key_move_map.get("key_move_reason", "") or ""
        elif roles_here:
            # 取第一个次要标签的理由
            key_move_reason = key_move_map[roles_here[0]].get("reason", "")

        # 节点知识按叙事作用选择：关键手讲机理，前置步讲识别/前提，
        # 后续步仍只给概念定义；本局结果由 material_fact、关键手理由等确定性字段提供。
        if theme_entry:
            theme_cn = theme_entry.get("cn", related_theme)
            if is_core_key:
                selected_theme_text = theme_entry.get("definition", "")
            elif step_idx < (key_move_map.get("key_move_idx", 0) or 1):
                selected_theme_text = (theme_entry.get("recognition", "")
                                       or theme_entry.get("prerequisite", "")
                                       or theme_entry.get("definition", ""))
            else:
                selected_theme_text = theme_entry.get("definition", "")
            if selected_theme_text:
                theme_context = f"本步关联【{theme_cn}】：{selected_theme_text}"

        # 注入洞察
        insight = insights[i] if i < len(insights) else {}
        teaching_point = insight.get("teaching_point", "")
        must_mention = insight.get("must_mention", [])
        spatial_change = insight.get("spatial_change", {})
        tactical_narratives = insight.get("tactical_narratives", [])
        puzzle_tactical_facts = insight.get("puzzle_tactical_facts", [])

        node = {
            "id": i + 1,
            "san": san,
            "move_count": 1,
            "turn": turn,
            "moves": san,
            "fen_before": fen_before,
            "fen_after": fen_after,
            "is_check": is_check,
            "is_capture": is_capture,
            "is_checkmate": is_checkmate_after,
            "is_checkmate_after": is_checkmate_after,
            "is_check_after": board_after.is_check(),
            "is_game_over_after": board_after.is_game_over(),
            "is_capture_node": is_capture,
            "has_check_in_node": is_check,
            "captured_piece_cn": captured_piece_cn,
            # PLAN-004 阶段 B：本节点起始局面上不存在的大子种类（中文棋子名列表）。
            # 供 prompt 负面事实注入，消除"提后但局面无后"这类真幻觉。puzzle 用单字段 san，
            # 升变走法（san 含 =Q/=R/=B/=N）对应新棋子不计 absent。兵/王不标。
            # previously_captured_types 传入对齐 validator B+（peer_review O1 修复）：
            # 前序被吃的大子允许回顾，不计 absent，避免 prompt 禁令压制合法历史叙述。
            "absent_pieces": _absent_major_pieces(
                board_before, [san] if san else None, captured_types_sofar),
            # PLAN-003 B+：截至本节点的前序累计被吃棋子类型（snapshot，供 validator 放行合理回顾）。
            # 注意是「注入前」的快照——本节点自己吃的子不计入本节点的放行集（本节点吃子后该子
            # 理应在 fen_before 里或被 material_fact 覆盖，不需走此通道）。
            # 用 sorted(list) 而非 set，与 captured_piece_types 等字段惯例一致（JSON 可序列化）。
            "previously_captured_piece_types": sorted(captured_types_sofar),
            "material_fact": material_fact,
            "legal_reply_count_after": sum(1 for _ in board_after.legal_moves),
            # 关键手定位（新增）
            "theme_key_roles": roles_here,
            "is_core_theme_key_move": is_core_key,
            "theme_key_reason": key_move_reason,
            # Puzzle 专用字段
            "related_theme": related_theme,
            "theme_context": theme_context,
            "prerequisite_facts": theme_entry.get("prerequisite", ""),
            "common_mistakes": theme_entry.get("common_mistakes", []),
            "typical_consequence": theme_entry.get("typical_consequence", ""),
            "defense_reference": theme_entry.get("defense_reference", ""),
            # 棋盘事实
            "teaching_point": teaching_point,
            "must_mention": must_mention,
            "spatial_change": spatial_change,
            "tactical_narratives": tactical_narratives,
            "puzzle_tactical_facts": puzzle_tactical_facts,
            # 兼容字段
            "tags": single_steps[i].tags if i < len(single_steps) else [],
            "suggested_pacing": "slow" if is_check or is_capture or is_checkmate_after else "normal",
            "phase": "",
            "phase_hint": "",
            "claim_level": "terminal" if is_checkmate_after else "forcing" if is_check else "positioning",
        }
        # PLAN-006 阶段 D：Puzzle 教学角色→emphasis_level 映射
        if is_core_key:
            node["emphasis_level"] = "pivotal"
        elif i == 0:
            node["emphasis_level"] = "important"  # 识别阶段/setup
        else:
            node["emphasis_level"] = "routine"    # follow_up
        # PLAN-003 B+：本节点吃掉的棋子类型累加进全局集，供后续节点的放行快照使用。
        # 兵不校验（PIECE_CN_TO_TYPE 不含兵），跳过；中文反查枚举。
        if captured_piece_cn in _CN_TO_PIECE_TYPE:
            captured_types_sofar.add(_CN_TO_PIECE_TYPE[captured_piece_cn])
        nodes_out.append(node)

    # P0-3 清理：移除旧的 _net_material_fact 整串净值追加块。
    # 该块输出"强方净多得约X个兵的子力价值"格式，是 Phase 0 量化痼疾的
    # 污染源（LLM 把它当模板金句拷贝进每步）。per_step_material_fact 已逐节点
    # 提供精确的定性结论，两者冲突——保留新代码、删除旧代码。

    # 组装 storyboard
    # Header 只注入主主题及最多两个真正联动的协同主题，而且只保留定义；
    # 关键手信号和结果由后续主主题锚点按需注入，不在这里重复整份知识库。
    prompt_theme_keys = [main_theme_key] if main_theme_key else []
    prompt_theme_keys.extend(k for k in synergy_keys if k not in prompt_theme_keys)
    theme_def_lines = []
    for key in prompt_theme_keys:
        theme = get_theme(key)
        if theme:
            theme_def_lines.append(f"【{theme['cn']}】{theme.get('definition', '')}")
    theme_defs_text = "\n".join(theme_def_lines)
    tactic_name = main_theme.get("cn", "战术练习")
    if synergy_keys:
        other_names = []
        for k in synergy_keys:
            t = get_theme(k)
            if t:
                other_names.append(t["cn"])
        if other_names:
            tactic_name += " + " + " + ".join(other_names[:2])

    # 关键手定位诊断日志（失败安全：异常吞掉，不影响主流程）
    try:
        core_loc = key_move_map.get("core", {}) or {}
        Logger.info(
            f"  关键手定位 核心标签={main_theme_key} 关键手步号={core_loc.get('idx')} "
            f"SAN={key_move_map.get('key_move_san', '')} 评分={core_loc.get('score', 0):.2f}")
        for tk, loc in key_move_map.items():
            if tk in ("core",):
                continue
            if isinstance(loc, dict):
                Logger.info(
                    f"    标签 {tk}: 关键手步号={loc.get('idx')} 评分={loc.get('score', 0):.2f}")
    except Exception as e:
        Logger.warn(f"关键手定位日志失败: {e}")

    # 联动标签中文名：供 prompt/锚点组合叙事（如"通过弃子打出碾压"）
    synergy_names = []
    for k in synergy_keys:
        t = get_theme(k)
        if t:
            synergy_names.append(t["cn"])

    return {
        "endgame_name": tactic_name,
        "endgame_matched": False,
        "tactic_name": tactic_name,
        "tactic_focus": {
            "primary_theme": main_theme_key,
            "theme_definitions": theme_defs_text,
            "assertions": [main_theme.get("assertion", "")] if main_theme else [],
            "narrative_mode": narrative_mode,
            "synergy_themes": synergy_names,
            # 关键手定位（新增）：全局统一的关键手步号 + 理由
            "key_move_idx": key_move_map.get("key_move_idx", 0),
            "key_move_san": key_move_map.get("key_move_san", ""),
            "key_move_reason": key_move_map.get("key_move_reason", ""),
        },
        "difficulty_hint": puzzle.rating,
        "difficulty_level": main_theme.get("difficulty_level", "intermediate"),
        "rating": puzzle.rating,
        "opening_context": puzzle.opening_tags,
        "attacking_side": puzzle_side,
        "defending_side": defending_side,
        "puzzle_side": puzzle_side,
        "narrative_mode": narrative_mode,
        "target_length": _puzzle_target_length(len(moves), puzzle.rating),
        "context": f"战术讲解：{tactic_name}",
        "phases": [],
        "motifs": [],
        "mistakes": [],
        "opening": {},
        "role_summary": "",
        "concept_binding": [],
        "hard_constraints": [],
        "winning_side": puzzle_side,
        "losing_side": defending_side,
        "white_material": _side_material_desc(board, chess.WHITE),
        "black_material": _side_material_desc(board, chess.BLACK),
        "strong_material": "",
        "weak_material": "",
        "compact_mode": False,
        "has_sub_endgame_switch": False,
        "nodes": nodes_out,
    }
