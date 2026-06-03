from dataclasses import dataclass, field
from colorama import init, Fore, Style
from typing import List, Optional
from datetime import datetime
import chess
import os

init()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载 .env 并配置 ffmpeg（在 pydub import 前加入 PATH，避免时序警告）
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except Exception:
    pass

_FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "")
if _FFMPEG_PATH:
    _FFMPEG_DIR = os.path.dirname(_FFMPEG_PATH)
    if os.path.isdir(_FFMPEG_DIR) and _FFMPEG_DIR not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")
    os.environ["IMAGEIO_FFMPEG_EXE"] = _FFMPEG_PATH  # moviepy 2.x

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
ALLOWED_ARROW_COLORS = {"red", "green", "blue", "yellow"}

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
    initial_fen: str                                        # 局面的FEN表示
    moves: List[chess.Move] = field(default_factory=list)   # 走法列表，python-chess的Move对象
    moves_san: List[str] = field(default_factory=list)      # 走法的SAN记谱

@dataclass
class AnalyzedMove:
    """ 带元数据的已分析走法，由SF逐步搜索或多源表库查询产生 """
    move: chess.Move
    score: Optional[int] = None                             # 走棋前局面的centipawn评估值，走棋方视角
    candidates: List[str] = field(default_factory=list)     # MultiPV候选走法的SAN列表
    is_only_move: bool = False                              # 是否为唯一好着
    trap_san: Optional[str] = None                          # 看似合理但其实导致大劣的陷阱走法SAN
    source: str = "sf"                                      # 来源:"chessdb" / "sf" / "syzygy" / "gaviota"
    dtm: Optional[int] = None                               # 距杀步数(Gaviota DTM), 正=走棋方胜, 负=走棋方负

@dataclass
class Segment:
    move_idx: int               # 对应第几个讲解节点（节点级分段；旧语义为第几步棋）
    text: str                   # 该节点的解说文本
    pacing: str = "normal"      # 解说节奏: slow/normal/fast/pause_before/pause_after
    audio_path: str = ""        # TTS生成的音频文件路径
    duration_s: float = 0.0     # 音频时长
    start_time: float = 0.0     # 在最终视频中的起始时间（不含片头静音，由渲染器按实际帧时长回填）
    moves: List[chess.Move] = field(default_factory=list)  # 本节点包含的子步走法，按顺序在节点时长内依次播放
    phase: str = ""             # 当前节点所属残局阶段名（如"建立控制线"），渲染器用于阶段过渡提示

@dataclass
class CompressedStep:
    """ 压缩后的讲解节点，对应1到N步的实际走法 """
    idx: int
    sans: List[str] = field(default_factory=list)
    fen_before: str = ""
    fen_after: str = ""
    is_critical: bool = False
    is_only_move: bool = False
    phase: str = ""
    candidates: List[str] = field(default_factory=list)
    trap: str = ""
    tags: List[str] = field(default_factory=list)
    eval_delta: Optional[float] = None

@dataclass
class StoryboardArrow:
    from_sq: str
    to_sq: str
    color: str
    label: str = ""

@dataclass
class StoryboardVisuals:
    extra_highlights: List[str] = field(default_factory=list)
    arrows: List[StoryboardArrow] = field(default_factory=list)
    phase_label: str = ""

@dataclass
class StoryboardSegment:
    id: int
    sub_endgame: str
    voiceover: str
    pacing: str = "normal"
    visuals: StoryboardVisuals = field(default_factory=StoryboardVisuals)

@dataclass
class GeneratedCommentary:
    segments: List[StoryboardSegment] = field(default_factory=list)
    raw_text: str = ""
    opening: str = ""                                       # 开场白（残局类型+子力对比+取胜思路），插在解说最前
    summary: str = ""                                       # 结尾总结词（技法/经验），独立于分步解说
    backend: str = ""
    chunks_total: int = 0
    chunks_succeeded: int = 0
    retries_total: int = 0
    fallback_used: bool = False

def is_valid_square_name(square: str) -> bool:
    if not isinstance(square, str) or len(square) != 2:
        return False
    return square[0] in "abcdefgh" and square[1] in "12345678"

def normalize_pacing(value: str) -> str:
    v = (value or "").strip().lower()
    return v if v in ALLOWED_PACING else "normal"

def resolve_path(path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))

def piece_cn(piece_type) -> str:
    return PIECE_CN.get(piece_type, "子")


@dataclass
class PuzzleData:
    """Puzzle 战术讲解输入数据，来自 Lichess puzzle 数据库或手动输入。"""
    fen: str                                    # 初始局面 FEN
    moves: List[chess.Move] = field(default_factory=list)   # 正解走法序列（不校验合法性）
    effective_themes: List[str] = field(default_factory=list)  # A 类有效标签（已过滤、保序）
    auxiliary_themes: List[str] = field(default_factory=list)  # B 类辅助标签
    raw_themes: List[str] = field(default_factory=list)        # 原始全量标签
    rating: int = 0
    popularity: int = 0
    opening_tags: str = ""
    puzzle_id: str = ""