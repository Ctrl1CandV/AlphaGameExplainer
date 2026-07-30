"""Puzzle 预备步旁白生成（纯模板，不依赖 LLM）。

从 storyboard.py 提取。交代对方走法 + 子力对比 + 优势 + 轮到谁。
每个分支提供多条随机变体，避免同类型预备着逐字雷同。
"""
from src.chess_utils.material import PIECE_VALUES, side_material_desc
import chess
import random


def _is_castle_san(san: str) -> bool:
    """是否王车易位 SAN。标准写法 O-O / O-O-O，老式 PGN 用数字零 0-0 / 0-0-0，都覆盖。"""
    return san.startswith("O") or san.startswith("0")


def _prelude_san_piece_cn(san: str) -> str:
    """从SAN走法提取棋子中文名。"""
    # 王车易位 SAN（O-O / O-O-O / 0-0 / 0-0-0）字母 O/数字 0 不匹配 N/B/R/Q/K，
    # 旧逻辑会误返回"兵"，产出"推进一兵"这类事实错误。易位以王为主导，返回"王"。
    if _is_castle_san(san):
        return "王"
    _map = {"N": "马", "B": "象", "R": "车", "Q": "后", "K": "王"}
    for letter, cn in _map.items():
        if san.startswith(letter):
            return cn
    return "兵"


def _advantage_desc(board: chess.Board) -> str:
    """根据子力价值判断当前局面优势方。"""
    white_val = sum(
        PIECE_VALUES[p.piece_type] for p in board.piece_map().values()
        if p.color == chess.WHITE and p.piece_type != chess.KING
    )
    black_val = sum(
        PIECE_VALUES[p.piece_type] for p in board.piece_map().values()
        if p.color == chess.BLACK and p.piece_type != chess.KING
    )
    diff = white_val - black_val
    if diff >= 5:
        return "白方子力大幅领先"
    if diff >= 2:
        return "白方子力占优"
    if diff <= -5:
        return "黑方子力大幅领先"
    if diff <= -2:
        return "黑方子力占优"
    return "双方子力均势"


def build_prelude_narration(prelude_san: str, board_after: chess.Board, puzzle_side: str) -> str:
    """纯模板生成预备着开场旁白（一到两句），交代对方走法 + 子力对比 + 优势 + 轮到谁。

    每个分支内置多条变体随机抽取，消除同类型预备着逐字相同的机械感。
    """
    opponent = "白方" if puzzle_side == "黑方" else "黑方"
    piece_cn_name = _prelude_san_piece_cn(prelude_san)
    is_capture = "x" in prelude_san
    is_check = board_after.is_check()

    advantage = _advantage_desc(board_after)

    # 王车易位（O-O / O-O-O / 0-0 / 0-0-0）是王与车联动的双子走法，单独分支给专用措辞，
    # 必须前置到 is_capture/is_check 判定之前：易位可能同时将军（O-O+ / O-O#），
    # 否则会被 is_check 分支截获，回退到通用"王走子并将军"，丢掉易位语义。
    if _is_castle_san(prelude_san):
        if is_check:
            move_part = random.choice([
                f"{opponent}完成王车易位，王躲进安全格、车快速出击并顺势将军",
                f"{opponent}选择易位，王转移到安全位置的同时车冲出来直接将军",
                f"{opponent}果断易位，王和车一步到位，车还顺便叫了将",
            ])
        else:
            move_part = random.choice([
                f"{opponent}完成王车易位，王躲进安全格同时车快速出击",
                f"{opponent}选择易位，把王安置到安全区域，车也顺势投入战斗",
                f"{opponent}走了一步易位，王转移到侧翼，车也找到了活跃的位置",
            ])
    elif is_capture and is_check:
        move_part = random.choice([
            f"{opponent}{piece_cn_name}直接吃子并将军，撕开对方防线",
            f"{opponent}{piece_cn_name}果断吃掉一子还带着将军，攻势很猛",
            f"{opponent}用{piece_cn_name}吃子的同时还叫了将，一举两得",
        ])
    elif is_capture:
        move_part = random.choice([
            f"{opponent}{piece_cn_name}果断吃子，直接赚到一个子",
            f"{opponent}{piece_cn_name}吃掉一个子，先捞实惠",
            f"{opponent}用{piece_cn_name}白吃一子，先赚一手",
        ])
    elif is_check:
        move_part = random.choice([
            f"{opponent}{piece_cn_name}走子并将军，施加压力",
            f"{opponent}{piece_cn_name}直接叫将，抢先发难",
            f"{opponent}用{piece_cn_name}将军，主动挑起战斗",
        ])
    else:
        # 普通调整步：按棋子类型给更具体的措辞，避免千篇一律"为后续战术做铺垫"。
        # 马走"日"字跳跃不存在"线路"概念，单独分支；象/车/后沿直线或斜线可用"线路"。
        if piece_cn_name == "兵":
            move_part = random.choice([
                f"{opponent}推进一兵，试探性地改变局面结构",
                f"{opponent}挺了一步兵，悄悄调整阵型",
                f"{opponent}动了一步兵，看似平淡实则在酝酿想法",
            ])
        elif piece_cn_name == "王":
            move_part = random.choice([
                f"{opponent}王挪动位置，重新组织防守站位",
                f"{opponent}调整了一下王的位置，加强防护",
                f"{opponent}把王转移了一下，给后续行动腾出空间",
            ])
        elif piece_cn_name == "马":
            move_part = random.choice([
                f"{opponent}马跳到一个新的位置，寻找不同的进攻方向",
                f"{opponent}把马调了个方向，瞄准新的目标",
                f"{opponent}马重新选了一个落点，酝酿新的进攻计划",
            ])
        else:
            move_part = random.choice([
                f"{opponent}{piece_cn_name}换到另一条线路，寻找新的进攻角度",
                f"{opponent}调整了{piece_cn_name}的位置，瞄准不同的方向",
                f"{opponent}{piece_cn_name}换了个位置，意图重新组织进攻",
            ])

    # 子力描述：均势时轻量化，有差距时才展开详细对比
    white_mat = side_material_desc(board_after, chess.WHITE)
    black_mat = side_material_desc(board_after, chess.BLACK)

    if advantage == "双方子力均势":
        material_part = random.choice([
            f"目前双方子力相当，局面还很开放。",
            f"盘面上双方实力接近，谁先犯错谁吃亏。",
            f"子力上两边差不多，关键看接下来的操作。",
        ])
    else:
        material_part = random.choice([
            f"目前白方有{white_mat}，黑方有{black_mat}，{advantage}。",
            f"子力方面，白方{white_mat}对黑方{black_mat}，{advantage}。",
        ])

    closing = random.choice([
        f"现在轮到{puzzle_side}，需要找出最强的战术手段。",
        f"该{puzzle_side}走了，看看能不能抓住机会。",
        f"轮到{puzzle_side}行动。",
    ])

    return f"{move_part}。{material_part}{closing}"
