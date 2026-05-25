from typing import Optional
import chess

_PIECE_MAP = {
    "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP,
    "N": chess.KNIGHT, "P": chess.PAWN,
}

ENDGAME_KB = [
    {
        "type": "KRvK",
        "name": "单车杀王",
        "theory": (
            "单车杀王是基础残局之一。核心思路是「盒子法」：用车画一条线，"
            "将对方王关进一个逐渐缩小的矩形，白王配合逼近，最终将对方王逼至棋盘边缘完成将杀。"
        ),
        "phases": [
            ("建立控制线", "用车切断对方王的逃路，将其限制在棋盘一侧"),
            ("王车合围", "白王向中心推进，与车形成配合。利用「对王」之势配合车的将军，将对方王逐排逼退"),
            ("收网将杀", "对方王被逼至底线或边线，车在安全距离将军即可将杀"),
        ],
        "motifs": [
            "盒子法：用车画线限制对方王的活动范围",
            "对王：两王相对时车将军可逼退一整排",
            "等招：车在不失去控制的前提下平移，迫使对方王走到不利位置",
        ],
        "mistakes": [
            "过早将军，对方王反而逃脱控制线",
            "车失去对关键列/排的控制",
            "形成无子可动局面（逼和）",
        ],
    },
    {
        "type": "KQvK",
        "name": "单后杀王",
        "theory": (
            "单后杀王比单车更简单。后同时具备车和象的移动能力，"
            "配合己方王逐步将对方王逼至棋盘边缘即可完成将杀。"
            "核心是避免形成无子可动局面（逼和）。"
        ),
        "phases": [
            ("逼近驱赶", "后从远处控制对方王的活动空间，己方王向对方王靠近"),
            ("边缘收网", "将对方王逼至底线或边线。注意保持后的安全距离，避免逼和"),
            ("完成将杀", "己方王保护后，后在底线将军将杀"),
        ],
        "motifs": ["后用骑士距离控制对方王的逃跑路线", "己方王始终向对方王靠拢"],
        "mistakes": ["后贴脸将军被吃掉", "逼和：对方无子可动但未被将军"],
    },
    {
        "type": "KBBvK",
        "name": "双象杀王",
        "theory": (
            "双象杀王需要在相邻两条斜线上建立控制网。"
            "两象并排推进，王在中间配合，将对方王逐步逼向棋盘一角完成将杀。"
        ),
        "phases": [
            ("象网初建", "将双象放置于相邻的斜线上，形成不可穿越的屏障"),
            ("推进压缩", "双象交替推进，己方王填补空隙，压缩对方王的活动空间"),
            ("角落将杀", "对方王被逼至角落，一象将军、王封口即可将杀"),
        ],
        "motifs": ["双象形成V字封锁线", "王填补象之间的空隙"],
        "mistakes": ["双象分开太远形成漏洞", "忽略了逼和的可能性"],
    },
    {
        "type": "KBNvK",
        "name": "象马杀王",
        "theory": (
            "象马杀王是公认最难的常见残局之一，需要将对方王逼到与象同色的角落。"
            "核心是 W 形驱赶模式：王、象、马三者协作形成推进队形。"
        ),
        "phases": [
            ("驱赶到边", "利用王、象、马的协同，将对方王从中心驱赶至棋盘边缘"),
            ("引导至正确角落", "将对方王引导至与己方象同色的角落（唯一能完成将杀的角落）"),
            ("完成将杀", "王压制对方王，马和象配合完成将杀"),
        ],
        "motifs": ["W形驱赶：马控制对方王的关键逃跑格", "象控制与自身同色的斜线", "王始终靠近对方王施加压力"],
        "mistakes": ["将对方王逼到错误颜色的角落（无法将杀）", "让马失去对关键格的控制"],
    },
    {
        "type": "KPvK",
        "name": "单兵残局",
        "theory": (
            "单兵对单王的关键是判断兵是否能升变。若己方王能保护兵安全到底线，则必胜；"
            "若对方王能阻止升变或形成逼和，则为和棋。"
            "核心概念是「对王」和「关键格」。"
        ),
        "phases": [
            ("争夺关键格", "双方王争夺兵前方和两侧的关键格。若白王占据关键格，兵可安全推进"),
            ("兵推进升变", "在王的保护下步步推进，必要时利用对王将对方王挤开"),
            ("升后将杀", "兵到达底线升变为后，随后以后杀王的方式结束"),
        ],
        "motifs": ["对王：主动方利用对王将对方王挤离关键路线", "关键格：兵前方两排的三格为关键格", "三角迂回：王绕到兵的另一侧来保持对王"],
        "mistakes": ["兵推进过快失去王保护", "不掌握对王概念被对方王逼和"],
    },
    {
        "type": "KRPvKR",
        "name": "车兵对车",
        "theory": (
            "车兵对车是最常见的实战残局之一。弱方（无兵方）在标准防守位置（菲利多防线）可以守住；"
            "强方若兵已过中线，通常可以取胜。关键在于弱方王的位置。"
        ),
        "phases": [
            ("判断防线", "判断弱方是否能建立菲利多防线或侧边防线的防守阵型"),
            ("推进防守", "强方用车切断联系推进兵；弱方坚守防线"),
            ("决定性突破", "或强方突破防线升变取胜，或弱方成功固守成和"),
        ],
        "motifs": ["菲利多防线：弱方王在兵前方，车在第六排骚扰", "卢塞纳桥位：强方王在兵前，用车搭桥掩护升变"],
        "mistakes": ["弱方王被压在兵前方无法动弹", "强方过早用车保护兵放弃灵活性"],
    },
]

def _piece_signature(board: chess.Board, color: chess.Color) -> tuple:
    """ 获取某一方具体的棋子情况 """
    counts = {}
    for piece in board.piece_map().values():
        if piece.color == color and piece.piece_type != chess.KING:
            counts[piece.piece_type] = counts.get(piece.piece_type, 0) + 1
    return tuple(sorted(counts.items()))

def _parse_type(type_key: str):
    parts = type_key.split("v")
    if len(parts) != 2:
        return None, None
    w_chars, b_chars = parts
    w_dict = {_PIECE_MAP[c]: w_chars.count(c) for c in _PIECE_MAP if w_chars.count(c) > 0}
    b_dict = {_PIECE_MAP[c]: b_chars.count(c) for c in _PIECE_MAP if b_chars.count(c) > 0}
    return tuple(sorted(w_dict.items())), tuple(sorted(b_dict.items()))

_PATTERN_CACHE = {}
for entry in ENDGAME_KB:
    _PATTERN_CACHE[entry["type"]] = _parse_type(entry["type"])

def match(board: chess.Board) -> Optional[dict]:
    if board.is_game_over():
        return None
    w_sig = _piece_signature(board, chess.WHITE)
    b_sig = _piece_signature(board, chess.BLACK)
    for entry in ENDGAME_KB:
        w_pat, b_pat = _PATTERN_CACHE[entry["type"]]
        if w_sig == w_pat and b_sig == b_pat:
            return entry
        if w_sig == b_pat and b_sig == w_pat:
            return entry
    return None


_CHINESE_TYPE = {
    chess.QUEEN: "后", chess.ROOK: "车", chess.BISHOP: "象",
    chess.KNIGHT: "马", chess.PAWN: "兵",
}


def _sig_name(sig: tuple) -> str:
    parts = []
    sig_dict = dict(sig)
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
        count = sig_dict.get(pt, 0)
        if count > 0:
            name = _CHINESE_TYPE.get(pt, "?")
            if count == 1:
                parts.append(name)
            else:
                parts.append(f"{count}{name}")
    return "".join(parts) if parts else "单王"


def describe_endgame(board: chess.Board) -> dict:
    kb = match(board)
    if kb:
        return {
            "type": kb["type"],
            "name": kb["name"],
            "theory": kb["theory"],
            "phases": kb["phases"],
            "motifs": kb["motifs"],
            "mistakes": kb["mistakes"],
            "matched": True,
        }
    w_sig = _piece_signature(board, chess.WHITE)
    b_sig = _piece_signature(board, chess.BLACK)
    w_name = _sig_name(w_sig)
    b_name = _sig_name(b_sig)
    if w_name == "单王" and b_name == "单王":
        name = "单王残局"
    else:
        name = f"{w_name}对{b_name}"
    return {
        "type": "unknown",
        "name": name,
        "theory": "",
        "phases": [],
        "motifs": [],
        "mistakes": [],
        "matched": False,
    }


def get_forbidden_concepts(board: chess.Board, endgame_info: dict) -> list:
    rules = []
    piece_map = board.piece_map()
    has_pawn = any(p.piece_type == chess.PAWN for p in piece_map.values())
    has_rook = any(p.piece_type == chess.ROOK for p in piece_map.values())
    has_bishop = any(p.piece_type == chess.BISHOP for p in piece_map.values())
    has_knight = any(p.piece_type == chess.KNIGHT for p in piece_map.values())
    has_queen = any(p.piece_type == chess.QUEEN for p in piece_map.values())

    if not has_pawn:
        rules.append("禁止提及升变、兵推进、关键格")
        rules.append("禁止提及「菲利多防线」「卢塞纳桥位」")
    if not has_rook:
        rules.append("禁止提及车控制线、横线/纵线割裂、盒子法")
    if not has_bishop:
        rules.append("禁止提及斜线控制、象网、同色角落")
    if not has_knight:
        rules.append("禁止提及马控制、W形驱赶")
    if not has_queen:
        rules.append("禁止提及后控制线、骑士距离")
    if wdl := _try_wdl(board):
        if wdl <= 0:
            rules.append("当前局面黑方必败，禁止写成黑方优势或反击机会")
        if wdl >= 0:
            rules.append("当前局面白方必败，禁止写成白方优势或反击机会")
    return rules


def _try_wdl(board: chess.Board) -> Optional[int]:
    try:
        import chess.syzygy
        syzygy_dir = None
        import os
        from dotenv import load_dotenv
        load_dotenv()
        syzygy_dir = os.getenv("SYZYGY_PATH", "")
        if not syzygy_dir:
            return None
        with chess.syzygy.open_tablebase(syzygy_dir) as tb:
            return tb.probe_wdl(board)
    except Exception:
        return None