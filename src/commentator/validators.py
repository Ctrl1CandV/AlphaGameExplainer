import re

from src.common import ALLOWED_PACING
from src.commentator.text_filters import reduce_cliches, reduce_cliches_puzzle
import chess

# 最短解说词长度：概括模式节点 28 字，其余取该常量。
# 原定义于 commentator.py 模块级（行 13），仅 _validate_single_segment 使用，
# 随校验器一起迁入本模块。
MIN_VOICEOVER_LEN = 48

# 中文棋子名 → python-chess piece_type，用于子力存在性校验（根因2修复）。
# "王"不校验（双方永远有王，且"王"字在"对王""王翼"等短语中噪声太大）。
PIECE_CN_TO_TYPE = {"后": chess.QUEEN, "车": chess.ROOK, "象": chess.BISHOP, "马": chess.KNIGHT}

# 单字棋子名会误命中的常见复合词（这些词里的字不指棋子）。校验前先把这些词
# 从文本里抹掉，再判断裸棋子字是否仍然出现，避免"之后/随后/马上/象征"等误判。
# 例："白后跃至底线" 抹词后仍含"后"→ 真提及；"黑王随后退守" 抹掉"随后"后不含
# 裸"后"→ 非提及。
PIECE_CN_DECOYS = {
    "后": ("之后", "随后", "然后", "过后", "而后", "其后", "此后", "事后",
           "先后", "前后", "日后", "往后", "退后", "稍后", "落后", "背后",
           "身后", "最后", "后方", "后面", "后续", "后排", "后翼", "后来",
           "后期", "后半", "后手", "王后方"),
    "象": ("象征", "现象", "形象", "印象", "对象", "想象", "抽象", "气象",
           "景象", "迹象", "万象", "象棋", "好像", "图象"),
    "马": ("马上", "马虎", "立马", "马不停蹄", "马前卒", "兵马", "人马"),
    "车": ("塞车", "火车", "汽车", "列车", "车轮", "车厢", "堵车", "马车"),
}


def mentions_piece(text: str, cn_name: str) -> bool:
    """判断文本是否真正提及某种棋子（而非命中同字复合词）。

    先把该棋子字对应的已知干扰复合词从文本里剔除，再看裸棋子字是否残留。
    这样"随后/之后/马上/象征"等不会误判为提及后/马/象。
    """
    if cn_name not in text:
        return False
    stripped = text
    for decoy in PIECE_CN_DECOYS.get(cn_name, ()):  # 抹掉干扰词
        stripped = stripped.replace(decoy, "")
    return cn_name in stripped


def validate_material_existence(text: str, node: dict) -> tuple:
    """校验 voiceover 提到的棋子种类，在节点起始局面上真实存在。

    根因2修复：模型会在描述升变/机动意图时提前把"兵"讲成"后"等不存在的
    棋子（如 KPvK 系列反复出现"白后跃至底线"，但局面里只有兵，尚未升变），
    或在 puzzle 里凭空多造车/象（子力盘点 6/6 全错）。这里用 fen_before 解析
    出该节点开始时盘面真实拥有的棋子种类，与文本中提到的棋子名做交叉核对；
    若本节点内确实发生了升变（sans 中含 promotion 标记），则放行对应新棋子
    种类的提及。

    残局与 puzzle 链路共用（此前只在残局 validate_single_segment 调用，
    puzzle 子力捏造是审计最高频问题，必须同样拦截）。

    只校验"新增"的棋子种类（如声称有后但没后），不校验"不再提及"的棋子，
    避免误伤合理省略。失败安全：任何解析异常直接放行，不阻塞主流程。

    PLAN-003 B+（第三个放行来源）：node 携带 `previously_captured_piece_types`
    （截至本节点、前序所有节点累计被吃过的棋子类型集合，由 builder 预填）。
    当前节点回顾「前序被吃的大子」是合理的战术叙述（如吃马后讲马的战术作用），
    不应判为捏造。三类放行来源取并集：present_types | promoted_types |
    previously_captured_types，任一命中即放行。
    固有局限（不处理）：
    1. 本函数只校验棋子「类型」不校验数量/时间线/颜色，无法区分「合理回顾」
       与「把已吃子当成还在棋盘上的活子」——后者由上游 material_fact 注入
       与 prompt 约束兜底，非 validator 职责。
    2. 放行集是 per-type、color-agnostic、永久累积的：一旦某类大子在前序任一
       节点被吃，后续所有节点对该类子的提及永久放行。在长残局/多吃子 puzzle
       中，累计集合可能覆盖全部四种被校验类型 {后,车,象,马}，使本校验对后续
       节点接近失效。这是为修复「跨节点历史叙述假阳性」刻意接受的放宽边界——
       实测真幻觉（如 0039T/0048h/004Lu 造后）都是造「从未在棋盘出现过的子」，
       不在前序被吃集合内，仍被正确拦截。若未来出现「造一个前序已被吃过的
       同型子」的幻觉新模式，需收紧为按颜色累计或加时间衰减。
    """
    fen_before = node.get("fen_before", "")
    if not fen_before:
        return True, ""
    try:
        board = chess.Board(fen_before)
    except Exception:
        return True, ""

    # 本节点内是否发生升变：sans 里任意一着以 =Q/=R/=B/=N 结尾即视为对应
    # 棋子类型允许被提及，即使升变前的局面上没有这枚棋子。
    # puzzle 节点用单字段 "san"；残局节点用 "sans" 列表——两者都兼容。
    promoted_types = set()
    sans = node.get("sans") or ([node.get("san")] if node.get("san") else [])
    for san in sans:
        if not isinstance(san, str) or "=" not in san:
            continue
        promo_letter = san.rsplit("=", 1)[-1][:1]
        promo_map = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}
        if promo_letter in promo_map:
            promoted_types.add(promo_map[promo_letter])

    present_types = {p.piece_type for p in board.piece_map().values()}

    # PLAN-003 B+：前序累计被吃棋子类型（builder 预填，缺失时退化为空集，保持向后兼容）
    previously_captured_types = set(node.get("previously_captured_piece_types") or [])

    for cn_name, piece_type in PIECE_CN_TO_TYPE.items():
        if not mentions_piece(text, cn_name):
            continue
        if piece_type in present_types or piece_type in promoted_types:
            continue
        if piece_type in previously_captured_types:
            continue
        return False, f"提到了'{cn_name}'，但本节点起始局面上不存在该棋子（且本节点未发生对应升变，前序节点也未吃过该类子）"

    return True, ""


def validate_single_segment(seg: dict, node: dict) -> tuple:
    seg_id = seg.get("id")
    if not isinstance(seg_id, int):
        return False, f"id={seg_id}不是有效整数"
    if seg_id != node["id"]:
        seg["id"] = node["id"]  # 自动修正

    voiceover = seg.get("voiceover")
    if not isinstance(voiceover, str) or not voiceover.strip():
        return False, f"voiceover为空"
    min_len = 28 if node.get("summary_only") else MIN_VOICEOVER_LEN
    if len(voiceover.strip()) < min_len:
        return False, f"voiceover过短({len(voiceover.strip())}<{min_len})"

    pacing = seg.get("pacing", "normal")
    pacing = str(pacing).strip().lower()
    if pacing not in ALLOWED_PACING:
        return False, f"pacing='{pacing}'不合法"

    # PLAN-003 B1：去除 sub_endgame 非空门。该字段在媒体/TTS/字幕层无消费者
    # （仅 generator 写入），puzzle 链路本就允许为空且无害，endgame 链路保留
    # 此门只会把模型偶尔输出空串的样本误杀整片（实测 KBBvK_3 类）。
    # 字段仍允许存在，只是不再因空值判失败；如需回填，由 generator/auto-fix
    # 用 node.sub_endgame_name 补，而非在 validator 拦截。

    text = voiceover.strip()
    _CHECKMATE_BANNED = ("将杀", "绝杀", "杀王", "无路可走", "无路可逃",
                         "死局", "终局已定", "锁定胜局")
    if node.get("is_checkmate_after") is False and any(word in text for word in _CHECKMATE_BANNED):
        return False, "错误宣称将杀"

    # 根因5修复：将杀断言位置校验。多着节点的 is_checkmate_after 只反映
    # "整个节点走完之后"的终态，不代表节点内每一着都已将杀。审计报告里
    # 反复出现"第6步（仅王逼近）声称形成将杀"——模型把节点终态提前贴到了
    # 描述中间过程的句子上。这里用一个位置启发式：多着节点若确实以将杀
    # 收尾，将杀类断言只能出现在文本后半段（对应"最后一着"），不能出现在
    # 描述前面机动过程的前半段。
    if node.get("is_checkmate_after") and node.get("move_count", 1) > 1:
        hit_positions = [text.find(w) for w in _CHECKMATE_BANNED if w in text]
        hit_positions = [p for p in hit_positions if p >= 0]
        if hit_positions and min(hit_positions) < len(text) * 0.5:
            return False, "多着节点内提前宣称将杀——将杀只发生在最后一着，前面的机动过程不能提前断言"

    allows_check_word = node.get("is_check_after") or node.get("has_check_in_node")
    if not allows_check_word and "将军" in text:
        return False, "错误宣称将军"

    king_moved = node.get("king_moved", False)
    checking_types = node.get("checking_piece_types", [])
    king_claims_check = king_moved and chess.KING not in checking_types
    if king_claims_check and any(word in text for word in ("王将军", "王形成杀", "王绝杀", "王直接", "致命将军")):
        return False, "错误宣称王将军——国际象棋中王不能直接将军"

    if not node.get("is_capture_node") and any(word in text for word in ("吃掉", "兑掉", "吞掉")):
        return False, "错误宣称吃子"

    # 根因2修复：子力存在性校验。审计报告里高频出现"局面根本没有后，
    # 却讲白后/黑后如何行动"的捏造（如 KPvK 系列把兵推进想象成后已存在）。
    # 用 fen_before 程序化核实——本节点若不含升变走法且局面确实无后，
    # voiceover 提到"后"就是硬性捏造，可零成本、无 LLM 依赖地拦截。
    ok, err = validate_material_existence(text, node)
    if not ok:
        return False, err

    if node.get("is_game_over_after") and node.get("legal_reply_count_after", 1) == 0:
        if any(word in text for word in ("黑方应", "白方应", "下一步", "随后再")):
            return False, "在终局后继续虚构后续走法"

    _NEUTRALITY_BANNED = ("双方等待", "局势平衡", "互相试探", "积蓄力量", "均势", "双方都在")
    if any(word in text for word in _NEUTRALITY_BANNED):
        return False, "含有均势叙事词——这是必胜残局变现，必须从强方主导推进角度写"

    if node.get("summary_only") and len(text) > 120:
        return False, f"概括模式节点过长({len(text)}>120)"

    return True, ""


def validate_storyboard_chunk(data: dict, chunk_nodes: list) -> tuple:
    segments = data.get("segments")
    if not isinstance(segments, list):
        return False, "顶层缺少segments数组"
    if len(segments) != len(chunk_nodes):
        return False, f"segments数量{len(segments)}与节点数{len(chunk_nodes)}不一致"

    # 逐段校验复用 validate_single_segment（单一事实来源），整块通过才算通过。
    # 错误信息加 segment[i] 前缀，保留 _build_retry_prompt 依赖的关键词（宣称/过短等）。
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            return False, f"第{i+1}个segment不是对象"
        ok, err = validate_single_segment(seg, chunk_nodes[i])
        if not ok:
            return False, f"segment[{i}]{err}"

    return True, ""


def validate_puzzle_voiceover_surface(text: str) -> tuple:
    """校验谜题口播表层字符与思考泄漏。

    不做删除；发现英文、数字、Markdown 符号或明显思考痕迹时直接判失败，
    交给重试或模板兜底，避免污染字幕/TTS。
    """
    if re.search(r"[A-Za-z0-9*_#`\[\]{}<>|\\/]", text):
        return False, "voiceover含英文/数字/Markdown符号"

    thinking_leaks = (
        "我需要", "让我", "先看", "首先我", "接下来我", "题目要求",
        "提示词", "用户", "输出", "这个节点", "这个segment", "思考过程",
        "推理过程", "我会", "我应该", "需要分析", "需要判断",
    )
    if any(w in text for w in thinking_leaks):
        return False, "voiceover含思考过程泄漏"

    return True, ""


def validate_puzzle_segment(seg: dict, node: dict) -> tuple:
    """puzzle 专用校验：保留 JSON 结构/数量/长度/pacing 校验，
    移除将杀/将军/吃子的真值禁止校验（puzzle 中这些是正常内容）。
    不强制每段提及 related_theme（实施决策 B：只做软约束）。
    """
    seg_id = seg.get("id")
    if not isinstance(seg_id, int):
        return False, f"id={seg_id}不是有效整数"

    voiceover = seg.get("voiceover")
    if not isinstance(voiceover, str) or not voiceover.strip():
        return False, "voiceover为空"

    # 表层安全校验：非法字符/Markdown/思考泄漏一律失败重试，不能进入 TTS。
    surface_ok, surface_err = validate_puzzle_voiceover_surface(voiceover.strip())
    if not surface_ok:
        return False, surface_err

    # 根因2修复（补 puzzle 链路）：子力存在性校验。puzzle 子力捏造是审计中
    # 最高频的问题（6/6 全中，如凭空多造车/象/后），此前该校验只在残局
    # validate_single_segment 调用，puzzle 链路完全漏掉。这里补上——puzzle
    # 节点带 fen_before/san 字段，与残局共用同一 validate_material_existence。
    mat_ok, mat_err = validate_material_existence(voiceover.strip(), node)
    if not mat_ok:
        return False, mat_err

    # 最短长度校验
    min_len = 28 if node.get("is_checkmate_after") and len(node.get("moves", "")) <= 4 else 48
    if len(voiceover.strip()) < min_len:
        return False, f"voiceover过短({len(voiceover.strip())}<{min_len})"

    pacing = seg.get("pacing", "normal")
    pacing = str(pacing).strip().lower()
    if pacing not in ALLOWED_PACING:
        return False, f"pacing='{pacing}'不合法"

    # 反套话检测：使用 puzzle 轻量版检测（不做形容词清洗，避免'精准/精确'等战术词被误杀）
    cleaned = reduce_cliches_puzzle(voiceover.strip())
    if len(cleaned) < len(voiceover.strip()) * 0.3:
        return False, "voiceover套话占比过高"

    # 本轮实测新增：非吃子步禁止断言"本步吃掉了具体某子"。
    # 案例 puzzle_001aK 第5步是 Kf2（白王避将后撤，非吃子），解说却说
    # "直接吃掉这枚无主的马"——把一步非吃子讲成吃子得子。这里只拦"指向本步
    # 的具体吃子断言"（如"这一吃""直接吃掉这枚/这个/那枚"），不拦战术定义里
    # 泛指的"可直接吃掉而不受惩罚"这类概念叙述，避免误伤机理讲解。
    if not node.get("is_capture"):
        _CONCRETE_CAPTURE_CLAIMS = ("这一吃", "直接吃掉这", "直接吃掉那",
                                    "果断吃掉这", "果断吃掉那", "吃掉这枚",
                                    "吃掉那枚", "吃掉这个", "吃掉那个")
        if any(w in voiceover for w in _CONCRETE_CAPTURE_CLAIMS):
            return False, "非吃子步却断言本步吃掉了具体子力"

    # 确保 segment id 与节点 id 对齐（多 chunk 时 LLM 可能从 1 重新编号）
    seg["id"] = node["id"]

    return True, ""


def validate_puzzle_chunk(data: dict, chunk_nodes: list) -> tuple:
    """逐段校验 puzzle chunk。"""
    segments = data.get("segments")
    if not isinstance(segments, list):
        return False, "顶层缺少segments数组"
    if len(segments) != len(chunk_nodes):
        return False, f"segments数量{len(segments)}与节点数{len(chunk_nodes)}不一致"

    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            return False, f"第{i+1}个segment不是对象"
        ok, err = validate_puzzle_segment(seg, chunk_nodes[i])
        if not ok:
            return False, f"segment[{i}]{err}"

    return True, ""
