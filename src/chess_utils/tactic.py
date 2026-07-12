"""战术几何检测——fork/pin/skewer/discovered。

从 storyboard.py 提取的四个纯棋盘几何判定函数。
不依赖 storyboard 的任何上下文（不读 CompressedStep、不读 role_meta）。
被 storyboard.associate_move_with_theme 和 storyboard.key_move_locator 共用。
"""
import chess
from src.chess_utils.material import PIECE_VALUES


def is_fork(board_before: chess.Board, move: chess.Move,
            board_after: chess.Board) -> bool:
    """走子后该子是否同时攻击两个或以上对方高价值目标（非兵）。"""
    moved_sq = move.to_square
    mover_color = board_before.turn
    enemy_color = not mover_color

    targets = []
    for atk_sq in board_after.attacks(moved_sq):
        target = board_after.piece_at(atk_sq)
        if target is None or target.color != enemy_color:
            continue
        if target.piece_type == chess.PAWN:
            continue
        targets.append(target)
    return len(targets) >= 2


def is_pin(board_before: chess.Board, move: chess.Move,
           board_after: chess.Board) -> bool:
    """走子后是否建立了牵制：己方远射子→对方被牵子→对方更高价值子共线。"""
    mover_color = board_before.turn
    enemy_color = not mover_color

    for sq, piece in board_after.piece_map().items():
        if piece.color != mover_color:
            continue
        if piece.piece_type not in (chess.QUEEN, chess.ROOK, chess.BISHOP):
            continue
        for atk_sq in board_after.attacks(sq):
            target = board_after.piece_at(atk_sq)
            if target is None or target.color != enemy_color:
                continue
            ray_dir = (chess.square_file(atk_sq) - chess.square_file(sq),
                       chess.square_rank(atk_sq) - chess.square_rank(sq))
            if ray_dir == (0, 0):
                continue
            df = 1 if ray_dir[0] > 0 else (-1 if ray_dir[0] < 0 else 0)
            dr = 1 if ray_dir[1] > 0 else (-1 if ray_dir[1] < 0 else 0)
            next_sq = chess.square(chess.square_file(atk_sq) + df,
                                   chess.square_rank(atk_sq) + dr)
            if next_sq is None:
                continue
            behind = board_after.piece_at(next_sq)
            if behind is None or behind.color != enemy_color:
                continue
            if PIECE_VALUES.get(target.piece_type, 0) <= PIECE_VALUES.get(behind.piece_type, 0):
                return True
    return False


def is_skewer(board_before: chess.Board, move: chess.Move,
              board_after: chess.Board) -> bool:
    """走子后是否建立了串击：己方远射子→对方高价值子→对方低价值子共线。"""
    mover_color = board_before.turn
    enemy_color = not mover_color

    for sq, piece in board_after.piece_map().items():
        if piece.color != mover_color:
            continue
        if piece.piece_type not in (chess.QUEEN, chess.ROOK, chess.BISHOP):
            continue
        for atk_sq in board_after.attacks(sq):
            target = board_after.piece_at(atk_sq)
            if target is None or target.color != enemy_color:
                continue
            df = (1 if chess.square_file(atk_sq) > chess.square_file(sq)
                  else (-1 if chess.square_file(atk_sq) < chess.square_file(sq) else 0))
            dr = (1 if chess.square_rank(atk_sq) > chess.square_rank(sq)
                  else (-1 if chess.square_rank(atk_sq) < chess.square_rank(sq) else 0))
            if df == 0 and dr == 0:
                continue
            next_sq = chess.square(chess.square_file(atk_sq) + df,
                                   chess.square_rank(atk_sq) + dr)
            if next_sq is None:
                continue
            behind = board_after.piece_at(next_sq)
            if behind is None or behind.color != enemy_color:
                continue
            if PIECE_VALUES.get(target.piece_type, 0) > PIECE_VALUES.get(behind.piece_type, 0):
                return True
    return False


def is_discovered(board_before: chess.Board, move: chess.Move,
                  board_after: chess.Board) -> bool:
    """走子是否产生了闪击：移动的子让开了身后远射子的攻击线。"""
    mover_color = board_before.turn
    enemy_color = not mover_color

    from_sq = move.from_square

    for sq, piece in board_before.piece_map().items():
        if piece.color != mover_color:
            continue
        if piece.piece_type not in (chess.QUEEN, chess.ROOK, chess.BISHOP):
            continue
        for atk_sq in board_before.attacks(sq):
            target = board_before.piece_at(atk_sq)
            if target is None or target.color != enemy_color:
                continue
            between = chess.SquareSet(chess.between(sq, atk_sq))
            if from_sq in between:
                return True
    return False
