from src.common import GameData, PuzzleData, PIECE_VALUES
from src.themes_kb import filter_themes
import chess.pgn
import chess
import io

def _count_material(board: chess.Board):
    """ 当前双方的子力统计 """
    white, black = 0, 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == chess.WHITE:
            white += value
        else:
            black += value
    return white, black

def is_endgame(board: chess.Board) -> bool:
    """ 判断当前局面是否为残局：材料分低或子数 ≤7（可被表库覆盖） """
    white, black = _count_material(board)
    if white <= 14 or black <= 14 or (white + black) <= 20:
        return True
    if len(board.piece_map()) <= 6:
        return True
    return False

"""
国际象棋的FEN表示共有六个部分:
1. 棋子位置: 用字母表示白方和黑方，数字表示连续空格，/分隔行
2. 轮到谁走棋: w表白方b表黑方
3. 王车易位权利: KQkq分别表示白方王翼、后翼和黑方王翼、后翼；-表示无易位权
4. 过路兵目标格: 上一步如果走了过路兵，这里标记可被吃的位置；-表示无过路兵机会
5. 半个回合计数: 用于50步规则，从最后一次吃子或兵移动后的半回合数
6. 完整回合数: 从对局开始当前回合数
"""
def is_fen(text: str) -> bool:
    parts = text.strip().split()
    return len(parts) == 6 and "/" in parts[0]

def parse_fen(fen_str: str) -> GameData:
    """ 解析FEN字符串 """
    chess.Board(fen_str)
    return GameData(initial_fen=fen_str)

def parse_pgn(pgn_str: str) -> GameData:
    """ 解析PGN文本，返回GameData """
    game = chess.pgn.read_game(io.StringIO(pgn_str.strip()))
    if not game:
        raise ValueError("无法解析PGN")

    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return GameData(initial_fen=board.fen())

def parse(input_text: str) -> GameData:
    """ 统一入口：自动判断PGN或FEN """
    text = input_text.strip()
    game_data = parse_fen(text) if is_fen(text) else parse_pgn(text)
    board = chess.Board(game_data.initial_fen)
    if not is_endgame(board):
        raise ValueError("仅支持残局局面，当前输入为中局或开局")
    return game_data

# Puzzle战术讲解输入解析

def _build_puzzle_data(
        fen: str, moves_str: str, themes_str: str,
        rating: int = 0, opening: str = ""
    ) -> PuzzleData:
    """
    公共构造：moves_str.split() 逐个 chess.Move.from_uci()；
    themes_str.split() → themes_kb.filter_themes() 拆 A/B 类。
    """
    raw_themes = [t.strip() for t in themes_str.split() if t.strip()]
    effective, _auxiliary = filter_themes(raw_themes)

    moves = []
    for tok in moves_str.split():
        tok = tok.strip()
        if not tok:
            continue
        try:
            moves.append(chess.Move.from_uci(tok))
        except ValueError as e:
            raise ValueError(f"UCI 走法格式错误: '{tok}'") from e

    # Lichess 约定：Moves[0] 为对方预备步，Moves[1:] 为解答步
    # 仅 1 步时（FEN 已为解题位置）不拆分
    prelude_move = None
    if len(moves) >= 2:
        prelude_move = moves[0]
        moves = moves[1:]

    return PuzzleData(
        fen=fen,
        prelude_move=prelude_move,
        moves=moves,
        effective_themes=effective,
        raw_themes=raw_themes,
        rating=rating,
        opening_tags=opening,
    )

def _parse_puzzle_json(text: str) -> PuzzleData:
    """ 解析单题JSON格式 """
    import json as _json
    data = _json.loads(text)
    return _build_puzzle_data(
        fen=data["fen"],
        moves_str=data.get("moves", ""),
        themes_str=data.get("themes", ""),
        rating=data.get("rating", 0),
        opening=data.get("openingTags", ""),
    )

def parse_puzzle_input(input_text: str) -> PuzzleData:
    """ 统一入口 """
    text = input_text.strip()
    if text.startswith("{"):
        return _parse_puzzle_json(text)
    else:
        return None