"""structure_id.applicable_mover_side 单测（PLAN-010 阶段0 步骤3，真纯函数）。

断言口径：iqp/hanging/stonewall 三个有角色歧义的原型各自筛对两种角色；
carlsbad/maroczy/majority 三个无歧义原型恒返回 None；未知原型/无角色指标局面
安全返回 None。

全部 FEN 在编写测试前已用 detect_pawn_structure + applicable_mover_side 实测
核实角色归属，不凭 FEN 外观猜测——IQP/stonewall 的角色方向容易凭直觉搞反
（walker 的 mover_side 语义是"谁持有该结构"而非"谁在攻击"）。
"""
import chess
import pytest

from src.analysis.structure_id import applicable_mover_side, detect_pawn_structure


# --- IQP：对方（黑）持 d5 孤兵，走子方（白）是施压方 -> role="opponent" ---
# （取自 structure_id.py 自检用例"IQP(对方持孤兵)"）
IQP_OPPONENT_FEN = "r1bq1rk1/pp2bppp/2n2n2/3p4/N7/5NP1/PP2PPBP/R1BQ1RK1 w - - 2 11"
# --- IQP：走子方（白）自己持有 d4 孤兵 -> role="mover"（最小化构造，避免依赖
#     复杂局面偶然触发别的原型判据） ---
IQP_MOVER_FEN = "4k3/8/8/8/3P4/8/8/4K3 w - - 0 1"

# --- stonewall：对方（黑）持有 d5+f5+e6 石墙，走子方是对抗方 -> "opponent" ---
# （取自 structure_id.py 自检用例"荷兰石墙"）
STONEWALL_OPPONENT_FEN = ("rn3rk1/pb2q1pp/1ppbpn2/3pNp2/2PP4/1P4P1/"
                          "PB1NPPBP/R2Q1RK1 w - - 2 11")
# --- stonewall：走子方（白）自己持有 d4+f4+e3 石墙 -> "mover"（最小化构造） ---
STONEWALL_MOVER_FEN = "4k3/8/8/8/3P1P2/4P3/8/4K3 w - - 0 1"

# --- hanging：对方（黑）持有 c5+d5 双悬兵，走子方是对抗方 -> "opponent" ---
HANGING_OPPONENT_FEN = "4k3/8/8/2pp4/8/8/8/4K3 w - - 0 1"
# --- hanging：走子方局面（取自 structure_id.py 自检"悬兵 Alapin"） -> "mover" ---
HANGING_MOVER_FEN = "2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/1R2R1K1 w - - 1 21"

# --- 无角色歧义的三个原型：判据本身锚定走子方，恒返回 None ---
CARLSBAD_FEN = ("r1bqrnk1/pp2bppp/2p2n2/3p2B1/3P4/2NBPN2/"
               "PPQ2PPP/R4RK1 w - - 8 11")
MAROCZY_FEN = ("r2q1rk1/pp2ppbp/3pbnp1/8/2P1P3/2N1B3/"
              "PP1QBPPP/R3K2R w KQ - 5 11")
MAJORITY_FEN = ("r2qr3/3bRpk1/p2p2p1/3P2Qp/1p6/1N3P2/"
                "PPP3PP/1K1R4 w - - 1 21")


def test_iqp_opponent_side_when_opponent_holds_pawn():
    board = chess.Board(IQP_OPPONENT_FEN)
    arch, _, _ = detect_pawn_structure(board)
    assert arch == "iqp"
    assert applicable_mover_side(board, "iqp") == "opponent"


def test_iqp_mover_side_when_mover_holds_pawn():
    board = chess.Board(IQP_MOVER_FEN)
    arch, _, _ = detect_pawn_structure(board)
    assert arch == "iqp"
    assert applicable_mover_side(board, "iqp") == "mover"


def test_iqp_mirror_preserves_role():
    """颜色归一化（P22）：镜像局面（走子方颜色互换）不应改变判定角色——
    这才是"颜色无关"的真正含义（镜像前后结论一致，不是制造对偶关系）。"""
    board = chess.Board(IQP_OPPONENT_FEN)
    mirrored = board.mirror()
    assert board.turn != mirrored.turn  # 确认真的互换了颜色/走子方
    assert applicable_mover_side(board, "iqp") == "opponent"
    assert applicable_mover_side(mirrored, "iqp") == "opponent"


def test_stonewall_opponent_side_when_opponent_holds_wall():
    board = chess.Board(STONEWALL_OPPONENT_FEN)
    arch, _, _ = detect_pawn_structure(board)
    assert arch == "stonewall"
    assert applicable_mover_side(board, "stonewall") == "opponent"


def test_stonewall_mover_side_when_mover_holds_wall():
    board = chess.Board(STONEWALL_MOVER_FEN)
    arch, _, _ = detect_pawn_structure(board)
    assert arch == "stonewall"
    assert applicable_mover_side(board, "stonewall") == "mover"


def test_hanging_opponent_side_when_opponent_holds_pawns():
    board = chess.Board(HANGING_OPPONENT_FEN)
    arch, _, _ = detect_pawn_structure(board)
    assert arch == "hanging"
    assert applicable_mover_side(board, "hanging") == "opponent"


def test_hanging_mover_side_when_mover_holds_pawns():
    board = chess.Board(HANGING_MOVER_FEN)
    arch, _, _ = detect_pawn_structure(board)
    assert arch == "hanging"
    assert applicable_mover_side(board, "hanging") == "mover"


@pytest.mark.parametrize("fen,archetype", [
    (CARLSBAD_FEN, "carlsbad"),
    (MAROCZY_FEN, "maroczy"),
    (MAJORITY_FEN, "majority"),
])
def test_non_ambiguous_archetypes_return_none(fen, archetype):
    """carlsbad/maroczy/majority：判据本身锚定走子方，无角色歧义，恒 None。"""
    board = chess.Board(fen)
    arch, _, _ = detect_pawn_structure(board)
    assert arch == archetype
    assert applicable_mover_side(board, archetype) is None


def test_unknown_archetype_returns_none():
    """未定义的原型名：安全返回 None，不抛异常。"""
    board = chess.Board(IQP_MOVER_FEN)
    assert applicable_mover_side(board, "nonexistent_archetype") is None


def test_no_role_indicators_returns_none():
    """初始局面对 iqp/hanging/stonewall 均无角色指标，安全返回 None。"""
    board = chess.Board()
    for archetype in ("iqp", "hanging", "stonewall"):
        assert applicable_mover_side(board, archetype) is None
