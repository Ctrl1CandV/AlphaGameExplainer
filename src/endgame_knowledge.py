from src.common import piece_cn
from typing import Optional
import chess
import json
import os

_PIECE_MAP = {
    "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP,
    "N": chess.KNIGHT, "P": chess.PAWN,
}

# 残局知识库数据外置为 JSON，便于后续补充新残局而无需改动代码逻辑。
_KB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "endgame_kb.json",
)

with open(_KB_PATH, "r", encoding="utf-8") as _f:
    ENDGAME_KB = json.load(_f)

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

def _sig_name(sig: tuple) -> str:
    parts = []
    sig_dict = dict(sig)
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
        count = sig_dict.get(pt, 0)
        if count > 0:
            name = piece_cn(pt)
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
            "opening": kb.get("opening", {}),
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
        "opening": {},
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
    return rules
