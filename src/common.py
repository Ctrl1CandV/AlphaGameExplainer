from dataclasses import dataclass, field
from colorama import init, Fore, Style
from typing import List, Optional
from datetime import datetime
import chess
import os

init()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 国际象棋材料价值评估
PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

CHINESE_PIECE = {
    chess.KING: "王", chess.QUEEN: "后", chess.ROOK: "车",
    chess.BISHOP: "象", chess.KNIGHT: "马", chess.PAWN: "兵",
}

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
class Segment:
    move_idx: int               # 对应第几步棋
    text: str                   # 该步的解说文本
    audio_path: str = ""        # TTS生成的音频文件路径
    duration_s: float = 0.0     # 音频时长
    start_time: float = 0.0     # 在最终视频中的起始时间

@dataclass
class CompressedStep:
    """ 压缩后的讲解节点，对应1到N步的实际走法 """
    idx: int
    sans: List[str] = field(default_factory=list)
    fen_before: str = ""
    fen_after: str = ""
    is_critical: bool = False
    phase: str = ""
    candidates: List[str] = field(default_factory=list)
    trap: str = ""
    tags: List[str] = field(default_factory=list)
    eval_delta: Optional[float] = None

def resolve_path(path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))