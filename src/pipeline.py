from src.common import (
    extract_moves, determine_winner, check_draw,
    Logger, resolve_path, Segment
)
from src.video_composer import compose, cleanup_artifacts, LEAD_SILENCE, INTRO_SEC
from src.tts_engine import synthesize as tts_synthesize, build_node_segments
from src.storyboard import compress, build, build_prelude_narration
from src.subtitle_gen import generate as gen_subtitles, build_cues
from src.board_renderer import render_animated_frames
from src.commentator import generate_structured
from src.stockfish_analyzer import get_solution
from src.llm_backend import release_backend
from src.tablebase import TablebaseSolver
from src.parser import parse

from dotenv import load_dotenv
import chess
import sys
import os
load_dotenv()

def _run_pipeline(input_text: str):
    """ 执行5步文本管线，返回(commentary, board, game_data)"""
    Logger.info("=" * 20 + "AlphaGameExplainer 开始运行" + "=" * 20)
    stockfish_path = resolve_path(os.getenv("STOCKFISH_PATH", ""))
    syzygy_path = os.getenv("SYZYGY_PATH", "")

    tablebase_solver = None
    if syzygy_path:
        tablebase_solver = TablebaseSolver(syzygy_dir=syzygy_path)

    Logger.info("[1/5]解析对局...")
    game_data = parse(input_text)
    board = chess.Board(game_data.initial_fen)

    if not board.is_valid():
        status = board.status()
        # 正在被将军的状态是合法的
        if status != chess.STATUS_OPPOSITE_CHECK:
            Logger.error(f"非法初始局面: FEN不合法(status={status})，无法生成解说")
            return None

    Logger.info("[2/5]查询最优解法...")
    analyzed_moves = get_solution(board, stockfish_path, tablebase_solver, syzygy_path)
    if not analyzed_moves:
        Logger.warn("未能找到解法")
        return None

    draw_error = check_draw(board, analyzed_moves, tablebase_solver)
    if draw_error:
        Logger.error(draw_error)
        return None

    Logger.info("[3/5]节点压缩...")
    compressed = compress(board, analyzed_moves)

    Logger.info("[4/5]构建叙事分镜...")
    winner_color = determine_winner(board, analyzed_moves)
    storyboard = build(board, compressed, winner_color=winner_color)

    Logger.info("[5/5]生成中文解说...")
    try:
        commentary = generate_structured(board, storyboard)
    except Exception:
        Logger.warn(f"结构化生成失败")
        sys.exit(1)

    try:
        release_backend()
        if tablebase_solver:
            tablebase_solver.close()
    except Exception:
        pass

    return commentary, board, game_data, analyzed_moves, storyboard, compressed, winner_color

def run(input_text: str) -> str:
    """ 运行现有5步管线，只返回解说文本"""
    result = _run_pipeline(input_text)
    if result is None:
        return
    commentary = result[0]
    if commentary.opening:
        print(commentary.opening + "\n")
    print(commentary.raw_text)
    if commentary.summary:
        print("\n" + commentary.summary)

def run_video(input_text: str, voice_prompt: str = "", endgame_name: str = "") -> str:
    """ 运行完整7步管线，生成mp4解说视频，返回输出视频路径 """
    result = _run_pipeline(input_text)
    if result is None:
        return

    commentary, board, _, analyzed_moves, storyboard, compressed, winner_color = result
    moves = extract_moves(board, analyzed_moves)
    if not moves:
        Logger.error("无法提取有效走法序列")
        return

    # [6/7] TTS 语音合成
    Logger.info("[6/7]TTS 语音合成...")
    segments = build_node_segments(commentary, moves, compressed)
    # 开场白段，插在最前，挂初始局面静态展示
    if commentary.opening:
        segments.insert(0, Segment(
            move_idx=0,
            text=commentary.opening,
            pacing="slow",
            moves=[],
        ))
    # 追加结尾总结，挂到最终局面上播放
    if commentary.summary:
        segments.append(Segment(
            move_idx=(len(compressed) if compressed else len(moves)) + 1,
            text=commentary.summary,
            pacing="slow",
            moves=[],
        ))
    segments = tts_synthesize(segments, voice_prompt=voice_prompt)

    # [7/7] 生成视频
    Logger.info("[7/7] 生成视频...")
    scores = [move.score for move in analyzed_moves]
    endgame = endgame_name or storyboard.get("endgame_name", "残局")
    panel_info = {"endgame_name": endgame} if scores else None
    if panel_info:
        panel_info["scores"], panel_info["winner_color"] = scores, winner_color

    """
    构建子步索引映射 {move_idx: (sub_idx, total)} 用于颜色轮换
    节点级渲染：每个segment在其音频时长内顺序播放本节点的全部子步
    并把实际渲染时长回填到seg.duration_s / seg.start_time
    供下方字幕与composer严格对齐，以彻底消除 MIN_HOLD 地板导致的累积漂移
    """
    frame_paths, frame_durations = render_animated_frames(
        segments, board.fen(), panel_info=panel_info
    )
    srt_path = gen_subtitles(segments, offset_s=LEAD_SILENCE)
    cues = build_cues(segments, offset_s=LEAD_SILENCE)

    try:
        output_path = compose(
            frame_paths=frame_paths,
            frame_durations=frame_durations,
            segments=segments,
            srt_path=srt_path,
            endgame_name=endgame,
            cues=cues,
            initial_fen=board.fen(),
        )
        Logger.success(f"视频已生成:{output_path}")
    finally:
        cleanup_artifacts(frame_paths, srt_path, segments)


#  Puzzle 战术讲解管线
from src.commentator import generate_puzzle_structured
from src.tts_engine import build_puzzle_segments
from src.storyboard import build_for_puzzle
from src.parser import parse_puzzle_input

def _run_puzzle_pipeline(input_text: str):
    """
    执行Puzzle战术讲解管线， 返回如下信息
    - commentary GeneratedCommentary类，完整的LLM解说词文本
    - board Board类，预备着后的初始局面
    - puzzle PuzzleData类，战术讲解输入信息
    - storyboard LLM生成解说词时的剧本，保存整个管线的核心数据结构
    - prelude_san 预备着SAN表示
    - pre_fen 初始局面
    - prelude_narration 开场白文本
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
    """ 输出纯解说文本，对应--text """
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
    """ 输出视频puzzle模式的对应视频，沿用现有片头片尾保证链路 """
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
    if commentary.opening:
        intro_seg = Segment(
            move_idx=-1,
            text=commentary.opening,
            pacing="normal",
            moves=[], phase=""
        )
        segments.insert(
            0 if not (prelude_san and puzzle.prelude_move is not None) else 1,
            Segment(
                move_idx=-1,
                text=commentary.opening,
                pacing="normal",
                moves=[], phase=""
        ))
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