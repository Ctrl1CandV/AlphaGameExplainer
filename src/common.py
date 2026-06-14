from dataclasses import dataclass, field
from colorama import init, Fore, Style
from typing import List, Optional
from datetime import datetime
import chess
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 国际象棋材料价值评估
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

# 棋子类型 → 中文名
PIECE_CN = {
    chess.KING: "王", chess.QUEEN: "后", chess.ROOK: "车",
    chess.BISHOP: "象", chess.KNIGHT: "马", chess.PAWN: "兵",
}

ALLOWED_PACING = {"fast", "normal", "slow", "pause_before", "pause_after"}

init()
class Logger:
    @staticmethod
    def _ts():
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def info(msg):
        print(f"{Fore.BLUE}[{Logger._ts()}] {msg}{Style.RESET_ALL}")

    @staticmethod
    def success(msg):
        print(f"{Fore.GREEN}[{Logger._ts()}] {msg}{Style.RESET_ALL}")

    @staticmethod
    def warn(msg):
        print(f"{Fore.YELLOW}[{Logger._ts()}] {msg}{Style.RESET_ALL}")

    @staticmethod
    def error(msg):
        print(f"{Fore.RED}[{Logger._ts()}] {msg}{Style.RESET_ALL}")

@dataclass
class GameData:
    initial_fen: str

@dataclass
class AnalyzedMove:
    """ 带元数据的已分析走法，由SF逐步搜索或多源表库查询产生 """
    move: chess.Move
    score: Optional[int] = None                             # 走棋前局面的centipawn评估值，走棋方视角
    is_only_move: bool = False                              # 是否为唯一好着
    source: str = "sf"                                      # 来源，sf或syzygy

@dataclass
class Segment:
    move_idx: int               # 对应第几个讲解节点
    text: str                   # 该节点的解说文本，用于合成语音和字幕生成
    pacing: str = "normal"      # 解说节奏，slow、normal、fast、pause_before、pause_after
    audio_path: str = ""        # TTS生成的音频文件路径
    duration_s: float = 0.0     # 音频时长
    start_time: float = 0.0     # 在最终视频中的不含片头静音起始时间
    moves: List[chess.Move] = field(default_factory=list)  # 本节点包含的子步走法，按顺序在节点时长内依次播放
    phase: str = ""             # 当前节点所属残局阶段名，渲染器用于阶段过渡提示

@dataclass
class CompressedStep:
    """ 压缩后的讲解节点，对应1到N步的实际走法 """
    idx: int
    sans: List[str] = field(default_factory=list)
    fen_before: str = ""
    fen_after: str = ""
    is_critical: bool = False                           # 是否包含关键事件
    is_only_move: bool = False                          # 是否包含唯一好着
    phase: str = ""
    tags: List[str] = field(default_factory=list)
    eval_delta: Optional[float] = None                  # 该节点的总评估变化

@dataclass
class StoryboardSegment:
    """ LLM生成解说的结构化输出格式，一个节点对应一个segment """
    id: int
    sub_endgame: str            # 当前子残局类型名
    voiceover: str              # LLM生成的口播解说文本，是最终进入TTS的文字
    pacing: str = "normal"

@dataclass
class GeneratedCommentary:
    """ 解说生成的完整结果，汇总本轮LLM生成的所有产出 """
    segments: List[StoryboardSegment] = field(default_factory=list)
    raw_text: str = ""                  # 纯文本版的解说词
    opening: str = ""                   # 开场白文本，包含残局类型、子力对比和取胜思路，插在解说最前
    summary: str = ""                   # 结尾总结词，独立于分步解说
    chunks_total: int = 0
    chunks_succeeded: int = 0
    retries_total: int = 0
    fallback_used: bool = False         # 是否回退到了纯文本模式

@dataclass
class PuzzleData:
    """
    Puzzle战术讲解输入数据，来自Lichess puzzle数据库或手动输入
    Lichess约定：Moves中第一步为对方预备步
    FEN为预备步走子前的局面，展示给解题方的局面是推完第一步之后的局面，第二步才是解答的起始步。
    无预备步时，prelude_move为None，moves即为全部走法。
    """
    fen: str                                                    # 初始局面FEN
    prelude_move: Optional[chess.Move] = None                   # 对方预备步
    moves: List[chess.Move] = field(default_factory=list)       # 正解走法序列
    effective_themes: List[str] = field(default_factory=list)   # A类需要深度讲解的战术标签
    raw_themes: List[str] = field(default_factory=list)         # 原始标签串
    rating: int = 0                                             # 难度评分，以此选择讲解深度挡位(0-1500-2200-∞)
    opening_tags: str = ""                                      # 开局分类标签

def normalize_pacing(value: str) -> str:
    v = (value or "").strip().lower()
    return v if v in ALLOWED_PACING else "normal"

def resolve_path(path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))

def piece_cn(piece_type) -> str:
    return PIECE_CN.get(piece_type, "子")

def extract_moves(board: chess.Board, analyzed: List[AnalyzedMove]) -> List[chess.Move]:
    """从 AnalyzedMove 列表提取合法走法序列"""
    moves, temp = [], board.copy()
    for am in analyzed:
        if temp.is_game_over():
            break
        if am.move in temp.legal_moves:
            moves.append(am.move)
            temp.push(am.move)
    return moves

def determine_winner(board: chess.Board, analyzed_moves: List[AnalyzedMove]):
    """
    复盘解法到终局，返回实际获胜方颜色 (chess.WHITE / chess.BLACK)，无法判定返回 None。
    用于让解说立场从实际终局结果反推，而不是按初始子力猜强弱——
    否则求解器若走出反常线路，解说会与画面完全相反。
    """
    temp = board.copy()
    for am in analyzed_moves:
        if temp.is_game_over():
            break
        if am.move not in temp.legal_moves:
            break
        temp.push(am.move)
    if not temp.is_game_over():
        return None
    outcome = temp.outcome()
    if outcome is None:
        return None
    return outcome.winner  # chess.WHITE / chess.BLACK / None（和棋）

def check_draw(board: chess.Board, analyzed_moves: List[AnalyzedMove], tablebase_solver=None) -> str:
    """
    检查局面是否为和棋。优先使用表库判定，其次复盘走法序列查看终局结果。
    返回非空字符串表示和棋错误信息，空字符串表示非和棋可继续。
    """
    if tablebase_solver is not None:
        try:
            tablebase_solver.open()
        except Exception:
            pass
        is_draw = tablebase_solver.is_draw(board)
        if is_draw is True:
            return "此局面为理论上的和棋（或超出50步规则无法兑现的必胜），无法生成必胜解说。"
        if is_draw is False:
            return ""

    temp = board.copy()
    for am in analyzed_moves:
        if temp.is_game_over():
            break
        temp.push(am.move)

    if not temp.is_game_over():
        return ""

    outcome = temp.outcome()
    if outcome is None:
        return ""
    if outcome.winner is None:
        return "该残局最终局面为逼和/和棋，无法生成必胜解说。"

    return ""