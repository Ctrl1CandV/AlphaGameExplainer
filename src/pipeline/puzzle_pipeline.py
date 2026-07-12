"""Puzzle 战术讲解管线。

从原 pipeline.py 提取。执行 4 步文本管线或含视频的完整管线。
包含 Segment 重复构造 bug 修复（问题 F）。
"""
from src.common import Logger, Segment
from src.media.video_composer import compose, cleanup_artifacts, INTRO_SEC
from src.media.tts_engine import synthesize as tts_synthesize, build_puzzle_segments
from src.storyboard.puzzle_builder import build_for_puzzle
from src.storyboard.prelude import build_prelude_narration
from src.media.subtitle_gen import generate as gen_subtitles, build_cues
from src.media.board_renderer import render_animated_frames
from src.commentator.puzzle_commentary import generate_puzzle_structured
from src.parser import parse_puzzle_input

import chess
import sys


def _run_puzzle_pipeline(input_text: str):
    """
    执行Puzzle战术讲解管线，返回如下信息：
    - commentary: GeneratedCommentary类，完整的LLM解说词文本
    - board: Board类，预备着后的初始局面
    - puzzle: PuzzleData类，战术讲解输入信息
    - storyboard: LLM生成解说词时的剧本
    - prelude_san: 预备着SAN表示
    - pre_fen: 初始局面
    - prelude_narration: 开场白文本
    """
    Logger.info("=" * 20 + "Puzzle战术讲解开始运行" + "=" * 20)

    Logger.info("[1/4]解析Puzzle输入...")
    puzzle = parse_puzzle_input(input_text)
    if puzzle is None:
        Logger.error("Puzzle文件输出格式错误")
        sys.exit(1)
    board = chess.Board(puzzle.fen)

    if not board.is_valid():
        status = board.status()
        if status != chess.STATUS_OPPOSITE_CHECK:
            Logger.error(f"非法初始局面: FEN不合法 (status={status})")
            return None

    # 记录预备步信息，供视频或文本输出使用
    prelude_san, pre_fen = "", puzzle.fen
    if puzzle.prelude_move is not None:
        prelude_san = board.san(puzzle.prelude_move)
        board.push(puzzle.prelude_move)
    Logger.info(f"  标签:{puzzle.effective_themes}, 步数:{len(puzzle.moves)}, Rating:{puzzle.rating}")

    Logger.info("[2/4]构建战术分镜...")
    storyboard = build_for_puzzle(board, puzzle.moves, puzzle)

    Logger.info("[3/4]生成战术解说...")
    try:
        commentary = generate_puzzle_structured(board, storyboard)
    except Exception as e:
        Logger.error(f"Puzzle解说生成失败: {e}")
        return None

    # 纯模板生成预备着旁白
    prelude_narration = ""
    if prelude_san:
        puzzle_side = "白方" if board.turn == chess.WHITE else "黑方"
        prelude_narration = build_prelude_narration(prelude_san, board, puzzle_side)

    Logger.info("[4/4]战术解说完成")
    return commentary, board, puzzle, storyboard, prelude_san, pre_fen, prelude_narration


def run_puzzle(input_text: str) -> str:
    """输出纯解说文本，对应 --text"""
    result = _run_puzzle_pipeline(input_text)
    if result is None:
        return
    commentary, prelude_narration = result[0], result[-1]
    if commentary.opening:
        print(commentary.opening + "\n")
    if prelude_narration:
        print(prelude_narration + "\n")
    print(commentary.raw_text)


def run_puzzle_video(input_text: str, voice_prompt: str = "") -> str:
    """输出视频 puzzle 模式的对应视频，沿用现有片头片尾保证链路"""
    result = _run_puzzle_pipeline(input_text)
    if result is None:
        return

    commentary, board, puzzle, storyboard, prelude_san, pre_fen, prelude_narration = result
    moves, nodes = puzzle.moves, storyboard.get("nodes", [])
    if not moves:
        Logger.error("无有效走法序列")
        return
    tactic_name = storyboard.get("tactic_name", "战术练习")

    # TTS语音合成
    Logger.info("TTS语音合成...")
    segments = build_puzzle_segments(commentary, moves, nodes)

    # 预备着段：棋盘从预备步前的局面开始，先动画演示对方的铺垫手
    if prelude_san and puzzle.prelude_move is not None:
        prelude_text = prelude_narration or f"对方走了{prelude_san}，局面来到当前状态。"
        segments.insert(0, Segment(
            move_idx=0,
            text=prelude_text,
            pacing="normal",
            moves=[puzzle.prelude_move],
            phase="",
        ))

    # 开场白段：基于骨架的半模板，插在预备着之后、正式解说之前
    # [问题F修复] 原代码创建了 intro_seg 后未使用，实际插入的是重复构造的 Segment。
    # 修复后直接使用同一个对象，消除死变量。
    if commentary.opening:
        intro_segment = Segment(
            move_idx=-1,
            text=commentary.opening,
            pacing="normal",
            moves=[], phase=""
        )
        insert_pos = 0 if not (prelude_san and puzzle.prelude_move is not None) else 1
        segments.insert(insert_pos, intro_segment)

    segments = tts_synthesize(segments, voice_prompt=voice_prompt)

    # 生成视频
    Logger.info("生成视频...")
    panel_info = {"endgame_name": tactic_name}

    # 从预备步前的局面开始渲染，预备着段将动画演示这一步
    initial_fen = pre_fen if prelude_san else board.fen()
    frame_paths, frame_durations = render_animated_frames(segments, initial_fen, panel_info=panel_info)

    # puzzle链路跳过片头片尾，字幕偏移仅含初始局面展示时长
    srt_path = gen_subtitles(segments, offset_s=INTRO_SEC)
    cues = build_cues(segments, offset_s=INTRO_SEC)

    try:
        output_path = compose(
            frame_paths=frame_paths,
            frame_durations=frame_durations,
            segments=segments,
            srt_path=srt_path,
            endgame_name=tactic_name,
            cues=cues,
            initial_fen=board.fen(),
            skip_title=True,
            skip_outro=True,
        )
        Logger.success(f"Puzzle视频已生成:{output_path}")
    finally:
        cleanup_artifacts(frame_paths, srt_path, segments)
