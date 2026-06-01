from src.stockfish_analyzer import get_solution
from src.common import Logger, resolve_path, GeneratedCommentary, Segment, AnalyzedMove, CompressedStep
from src.storyboard import compress, build
from src.commentator import generate_structured, generate
from src.llm_backend import release_backend
from src.tablebase import TablebaseSolver
from dotenv import load_dotenv
from src.parser import parse
from typing import List
import chess
import os

load_dotenv()


def run(input_text: str) -> str:
    """运行现有 5 步管线，返回解说文本"""
    result = _run_pipeline(input_text)
    if result is None:
        return ""
    commentary, _board, _game_data, _analyzed_moves, _storyboard, _compressed, _winner = result
    print(commentary.raw_text)
    if commentary.summary:
        print("\n" + commentary.summary)
    return commentary.raw_text


def _extract_moves(board: chess.Board, analyzed: List[AnalyzedMove]) -> List[chess.Move]:
    """从AnalyzedMove列表提取合法走法"""
    moves = []
    temp = board.copy()
    for am in analyzed:
        if temp.is_game_over():
            break
        if am.move in temp.legal_moves:
            moves.append(am.move)
            temp.push(am.move)
    return moves


def run_video(input_text: str, voice_prompt: str = "",
              endgame_name: str = "") -> str:
    """
    运行完整 7 步管线，生成 .mp4 解说视频。
    返回输出视频路径。
    """
    result = _run_pipeline(input_text)
    if result is None:
        return ""

    commentary, board, game_data, analyzed_moves, storyboard, compressed, winner_color = result
    moves = _extract_moves(board, analyzed_moves)
    if not moves:
        Logger.error("无法提取有效走法序列")
        return ""

    endgame = endgame_name or storyboard.get("endgame_name", "残局")

    # [6/7] TTS 语音合成
    Logger.info("[6/7] TTS 语音合成...")
    from src.tts_engine import synthesize as tts_synthesize

    segments = _build_move_segments(commentary, moves, board, compressed)
    # 追加结尾总结段：挂到最终局面上播放（技法/经验总结）
    if commentary.summary:
        segments.append(Segment(
            move_idx=len(moves) + 1,
            text=commentary.summary,
            pacing="slow",
        ))
    segments = tts_synthesize(segments, voice_prompt=voice_prompt)

    # [7/7] 生成视频
    Logger.info("[7/7] 生成视频...")
    from src.board_renderer import render_animated_frames
    from src.subtitle_gen import generate as gen_subtitles
    from src.video_composer import compose, LEAD_SILENCE

    scores = [am.score for am in analyzed_moves]
    panel_info = {"endgame_name": endgame} if scores else None
    if panel_info:
        panel_info["scores"] = scores
        panel_info["winner_color"] = winner_color

    # 构建子步索引映射 {move_idx: (sub_idx, total)} 用于颜色轮换
    sub_step_indices = {}
    if compressed:
        move_idx = 0
        for cs in compressed:
            for sub_idx in range(len(cs.sans)):
                sub_step_indices[move_idx + sub_idx] = (sub_idx, len(cs.sans))
            move_idx += len(cs.sans)

    frame_paths, frame_durations = render_animated_frames(
        moves, board.fen(), segments, panel_info=panel_info,
        sub_step_indices=sub_step_indices)
    # 字幕起始偏移 = 视频开头静音(标题卡+初始局面)，与音频严格对齐
    from src.subtitle_gen import build_cues
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
        Logger.success(f"视频已生成: {output_path}")
        return output_path
    finally:
        _cleanup_temp_files(frame_paths, srt_path, segments)


def _cleanup_temp_files(frame_paths: list, srt_path: str, segments: list):
    """清理临时文件（帧图片、音频段、字幕、批次文件等）"""
    import shutil
    import os as _os

    # 帧图片
    for p in frame_paths:
        try:
            _os.remove(p)
        except Exception:
            pass
    # 音频段
    for seg in segments:
        if seg.audio_path and _os.path.exists(seg.audio_path):
            try:
                _os.remove(seg.audio_path)
            except Exception:
                pass
    # 字幕
    if _os.path.exists(srt_path):
        try:
            _os.remove(srt_path)
        except Exception:
            pass
    # 批次文件 + 静音 + 标题卡
    audio_dir = _os.path.join("output", "audio")
    frames_dir = _os.path.join("output", "frames")
    for d in (audio_dir, frames_dir):
        if _os.path.isdir(d):
            try:
                shutil.rmtree(d)
            except Exception:
                pass


def _build_move_segments(commentary: GeneratedCommentary, moves: List[chess.Move],
                          board: chess.Board,
                          compressed: List[CompressedStep] = None) -> List[Segment]:
    """将解说词按 compressed step 映射到每步走法，继承 pacing"""
    # 构建 compressed step → voiceover + pacing 查找表
    voice_map: dict = {}
    if commentary.segments:
        for seg in commentary.segments:
            voice_map[seg.id] = (seg.voiceover, seg.pacing)

    # 构建 compressed step → [move indices] 映射
    step_moves: dict = {}
    if compressed:
        move_idx = 0
        for cs in compressed:
            n = len(cs.sans)
            step_moves[cs.idx] = list(range(move_idx, move_idx + n))
            move_idx += n

    result = []
    temp = board.copy()
    for i, move in enumerate(moves):
        san = temp.san(move)

        # 查找该 move 属于哪个 compressed step
        text = f"第{i + 1}步: {san}"
        pacing = "normal"
        if step_moves:
            for step_id, move_indices in step_moves.items():
                if i in move_indices:
                    vo, pac = voice_map.get(step_id, (None, "normal"))
                    if vo:
                        # 若一步含多着，只取第一着展示完整解说；
                        # 其余跟随步置空文本——静默快速走子，避免 TTS 念
                        # "SAN（续前）"造成中英混读、碎读、含糊。
                        if i == move_indices[0]:
                            text = vo
                        else:
                            text = ""
                    pacing = pac
                    break

        result.append(Segment(move_idx=i + 1, text=text, pacing=pacing))
        temp.push(move)
    return result


def _run_pipeline(input_text: str):
    """执行 5 步文本管线，返回 (commentary, board, game_data)"""
    Logger.info("=" * 20 + "AlphaGameExplainer 开始运行" + "=" * 20)

    stockfish_path = resolve_path(os.getenv("STOCKFISH_PATH", "stockfish-windows-x86-64-avx2.exe"))
    syzygy_path = os.getenv("SYZYGY_PATH", "")
    gaviota_path = os.getenv("GAVIOTA_PATH", "")

    tablebase_solver = None
    if syzygy_path or gaviota_path:
        tablebase_solver = TablebaseSolver(
            syzygy_dir=syzygy_path,
            gaviota_dir=gaviota_path,
        )
        Logger.info(f"表库配置: Syzygy={syzygy_path or '未设置'}, Gaviota={gaviota_path or '未设置'}")

    Logger.info("[1/5] 解析对局...")
    game_data = parse(input_text)
    board = chess.Board(game_data.initial_fen)

    if not board.is_valid():
        Logger.error(f"非法初始局面: FEN不合法 (status={board.status()})，无法生成解说")
        return None

    Logger.info("[2/5] 查询最优解法...")
    analyzed_moves = get_solution(board, stockfish_path, tablebase_solver, syzygy_path)
    if not analyzed_moves:
        Logger.warn("未能找到解法")
        return None

    draw_error = _check_draw(board, analyzed_moves, tablebase_solver)
    if draw_error:
        Logger.error(draw_error)
        Logger.error("当前版本仅支持必胜残局解说，和棋局面暂不处理。")
        return None

    Logger.info("[3/5] 节点压缩...")
    compressed = compress(board, analyzed_moves)

    Logger.info("[4/5] 构建叙事分镜...")
    winner_color = _determine_winner(board, analyzed_moves)
    storyboard = build(board, compressed, winner_color=winner_color)

    Logger.info("[5/5] 生成中文解说...")
    try:
        commentary = generate_structured(board, storyboard)
    except Exception as e:
        Logger.warn(f"结构化生成失败，回退纯文本: {e}")
        text = generate(board, storyboard)
        commentary = GeneratedCommentary(raw_text=text, fallback_used=True)

    if not commentary.summary:
        from src.commentator import _fallback_summary
        commentary.summary = _fallback_summary(storyboard)

    if commentary.segments:
        Logger.info(f"  {len(commentary.segments)} 段 - pacing分布: " +
                    ", ".join(f"{p}={sum(1 for s in commentary.segments if s.pacing == p)}"
                              for p in ["slow", "normal", "fast", "pause_before", "pause_after"]
                              if any(s.pacing == p for s in commentary.segments)))

    try:
        release_backend()
    except Exception:
        pass
    if tablebase_solver:
        try:
            tablebase_solver.close()
        except Exception:
            pass

    return commentary, board, game_data, analyzed_moves, storyboard, compressed, winner_color


def _determine_winner(board, analyzed_moves):
    """复盘解法到终局，返回实际获胜方颜色(chess.WHITE/BLACK)，无法判定返回 None。

    用于让解说立场从「实际终局结果」反推，而不是按初始子力猜强弱。
    否则求解器若走出反常线路（如强方弃子被杀），解说会与画面完全相反。
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
    return outcome.winner  # chess.WHITE / chess.BLACK / None(和棋)


def _check_draw(board, analyzed_moves, tablebase_solver) -> str:
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
