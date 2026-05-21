from src.common import GameData, PIECE_VALUES
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
    """ 判断当前局面是否为残局 """
    white, black = _count_material(board)
    return white <= 14 or black <= 14 or (white + black) <= 20

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
