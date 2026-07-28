"""残局讲解管线。

从原 pipeline.py 提取。执行 5 步文本管线或 7 步视频管线。
"""
from src.common import (
    extract_moves, determine_winner, check_draw,
    Logger, resolve_path, Segment
)
from src.media.video_composer import compose, cleanup_artifacts, LEAD_SILENCE, INTRO_SEC
from src.media.tts_engine import synthesize as tts_synthesize, build_node_segments
from src.storyboard.compressor import compress
from src.storyboard.endgame_builder import build
from src.media.subtitle_gen import generate as gen_subtitles, build_cues
from src.media.board_renderer import render_animated_frames
from src.commentator.endgame_commentary import generate_structured
from src.solver.stockfish_analyzer import get_solution
from src.infra.llm_backend import release_backend
from src.solver.tablebase import TablebaseSolver
from src.parser import parse

from dotenv import load_dotenv
import chess
import sys
import os
load_dotenv()


def _run_pipeline(input_text: str):
    """执行5步文本管线，返回(commentary, board, game_data, analyzed_moves, storyboard, compressed, winner_color)"""
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

    # SPEC §8：内容级失败即舍弃——generator 标记 aborted 时放弃本片，走 return None 通道
    # （与上游 if not analyzed_moves: return None 一致），不产出废片。
    if getattr(commentary, "aborted", False):
        Logger.warn(
            f"内容级失败，放弃本片生成（chunk {commentary.aborted_chunk}：{commentary.aborted_reason}）"
        )
        try:
            release_backend()
            if tablebase_solver:
                tablebase_solver.close()
        except Exception:
            pass
        return None

    try:
        release_backend()
        if tablebase_solver:
            tablebase_solver.close()
    except Exception:
        pass

    return commentary, board, game_data, analyzed_moves, storyboard, compressed, winner_color


def run(input_text: str) -> str:
    """运行现有5步管线，只返回解说文本"""
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
    """运行完整7步管线，生成mp4解说视频，返回输出视频路径"""
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
    # PLAN-006 阶段 C：从 storyboard 节点注入 emphasis_level（TTS 二维查表用）
    _node_emph = {n["id"]: n.get("emphasis_level", "important") for n in storyboard.get("nodes", [])}
    # PLAN-006 阶段 D：slide_sec 随 emphasis 微调（pivotal 略慢给观众消化，routine 略快保持节奏）
    # PLAN-006 阶段 D（REVIEW-002 V2）：±0.1s 差异低于人眼运动感知阈值（~0.3s），
    # 放大到 planner 裁决的 0.60/0.30，配合 V1 辉光增强让画面节奏真正拉开。
    _SLIDE_BY_EMPHASIS = {"pivotal": 0.60, "important": 0.45, "routine": 0.30}
    for seg in segments:
        seg.emphasis_level = _node_emph.get(seg.move_idx, "important")
        seg.slide_sec = _SLIDE_BY_EMPHASIS.get(seg.emphasis_level, 0.45)
    if commentary.opening:
        segments.insert(0, Segment(
            move_idx=0,
            text=commentary.opening,
            pacing="slow",
            moves=[],
        ))
    if commentary.summary:
        segments.append(Segment(
            move_idx=(len(compressed) if compressed else len(moves)) + 1,
            text=commentary.summary,
            pacing="slow",
            moves=[],
        ))

    # PLAN-005 调试：视频生成前把完整解说词打印到终端，便于人工审阅解说质量
    Logger.info("===== 解说词预览（视频生成前）=====")
    if commentary.opening:
        Logger.info(f"[开场白] {commentary.opening}")
    for ss in commentary.segments:
        Logger.info(f"[节点{ss.id}] {ss.voiceover}")
    if commentary.summary:
        Logger.info(f"[总结] {commentary.summary}")
    Logger.info("===== 解说词预览结束 =====")

    segments = tts_synthesize(segments, voice_prompt=voice_prompt)

    # [7/7] 生成视频
    Logger.info("[7/7] 生成视频...")
    scores = [move.score for move in analyzed_moves]
    endgame = endgame_name or storyboard.get("endgame_name", "残局")
    panel_info = {"endgame_name": endgame} if scores else None
    if panel_info:
        panel_info["scores"], panel_info["winner_color"] = scores, winner_color

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
