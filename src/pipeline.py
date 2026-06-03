from src.common import Logger, resolve_path, GeneratedCommentary, Segment, AnalyzedMove, CompressedStep
from src.commentator import generate_structured, generate
from src.stockfish_analyzer import get_solution
from src.llm_backend import release_backend
from src.storyboard import compress, build
from src.tablebase import TablebaseSolver
from dotenv import load_dotenv
from src.parser import parse
from typing import List
import chess
import os

load_dotenv()

def run(input_text: str) -> str:
    """ 运行现有5步管线，只返回解说文本"""
    result = _run_pipeline(input_text)
    if result is None:
        return ""
    commentary, _board, _game_data, _analyzed_moves, _storyboard, _compressed, _winner = result
    if commentary.opening:
        print(commentary.opening + "\n")
    print(commentary.raw_text)
    if commentary.summary:
        print("\n" + commentary.summary)
    return commentary.raw_text

def _extract_moves(board: chess.Board, analyzed: List[AnalyzedMove]) -> List[chess.Move]:
    """ 从AnalyzedMove列表提取合法走法 """
    moves, temp = [], board.copy()
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

    segments = _build_node_segments(commentary, moves, compressed)
    # 开场白段：插在最前，挂初始局面静态展示（moves 为空，渲染器按静态定格处理），
    # 与视频开头的棋盘展示同步——声音介绍局面，画面停在初始局面。
    if commentary.opening:
        segments.insert(0, Segment(
            move_idx=0,
            text=commentary.opening,
            pacing="slow",
            moves=[],
        ))
    # 追加结尾总结段：挂到最终局面上播放（技法/经验总结）。moves 为空，渲染器按静态定格处理。
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
    from src.board_renderer import render_animated_frames
    from src.subtitle_gen import generate as gen_subtitles
    from src.video_composer import compose, LEAD_SILENCE

    scores = [am.score for am in analyzed_moves]
    panel_info = {"endgame_name": endgame} if scores else None
    if panel_info:
        panel_info["scores"] = scores
        panel_info["winner_color"] = winner_color

    # 构建子步索引映射 {move_idx: (sub_idx, total)} 用于颜色轮换
    # 节点级渲染：每个 segment 在其音频时长内顺序播放本节点的全部子步，
    # 并把「实际渲染时长」回填到 seg.duration_s / seg.start_time，
    # 供下方字幕与 composer 严格对齐（彻底消除 MIN_HOLD 地板导致的累积漂移）。
    frame_paths, frame_durations = render_animated_frames(
        segments, board.fen(), panel_info=panel_info)
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


def _build_node_segments(commentary: GeneratedCommentary, moves: List[chess.Move],
                         compressed: List[CompressedStep] = None) -> List[Segment]:
    """按压缩节点分段：一个节点 = 一段解说 + 该节点的全部子步走法。

    这是「解决音画粒度错位」的核心改动。旧实现把节点整段解说塞给第一个子步、
    其余子步置空文本（导致首步静止十几秒念完、后续子步无声飞闪、解说视角与
    画面错位）。现在改为节点级分段：

      - 每段携带本节点的全部 moves，由 board_renderer 在该段音频时长内
        顺序播放这些子步并均摊定格，解说推进时棋子也在持续走；
      - 一段一段音频，不再有空文本段，从根上消除空段累积漂移。

    无压缩信息时退化为逐步分段（每段一个走法），保持鲁棒。
    """
    # 节点 id → (voiceover, pacing) 查找表
    voice_map: dict = {}
    if commentary.segments:
        for seg in commentary.segments:
            voice_map[seg.id] = (seg.voiceover, seg.pacing)

    result: List[Segment] = []

    if compressed:
        move_cursor = 0
        for cs in compressed:
            n = len(cs.sans)
            node_moves = moves[move_cursor:move_cursor + n]
            move_cursor += n
            if not node_moves:
                continue
            vo, pac = voice_map.get(cs.idx, (None, "normal"))
            text = vo if vo else ""
            result.append(Segment(
                move_idx=cs.idx,
                text=text,
                pacing=pac or "normal",
                moves=list(node_moves),
                phase=getattr(cs, "phase", ""),
            ))
        # 解法被截断、moves 比 compressed 覆盖的还多时，剩余走法兜底成一段静默节点
        if move_cursor < len(moves):
            result.append(Segment(
                move_idx=(compressed[-1].idx if compressed else 0) + 1,
                text="",
                pacing="normal",
                moves=list(moves[move_cursor:]),
            ))
        return result

    # 无压缩信息：逐步分段
    for i, move in enumerate(moves):
        result.append(Segment(move_idx=i + 1, text="", pacing="normal", moves=[move]))
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

    Logger.info("[1/5] 解析对局...")
    game_data = parse(input_text)
    board = chess.Board(game_data.initial_fen)

    if not board.is_valid():
        # STATUS_OPPOSITE_CHECK (1024) 表示"轮到走棋的一方正在将军对方"，
        # 这在国际象棋中完全合法（白方刚走了一步将军，轮到白方继续走）。
        # python-chess 的 is_valid() 对此返回 False 是出于 FEN 一致性检查，
        # 但在残局分析场景中这种局面是合理的，不应拒绝。
        status = board.status()
        if status != chess.STATUS_OPPOSITE_CHECK:
            Logger.error(f"非法初始局面: FEN不合法 (status={status})，无法生成解说")
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

    # 开场白兜底：结构化生成异常回退纯文本时 commentary 不带开场白，
    # 这里用纯模板补上，保证每个视频都有开场白。
    if not commentary.opening:
        from src.commentator import _compose_opening
        commentary.opening = _compose_opening(storyboard)

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


# ============================================================
#  Puzzle 战术讲解管线（新增，不复用 _run_pipeline）
# ============================================================

def _run_puzzle_pipeline(input_text: str):
    """执行 Puzzle 战术讲解管线，返回 (commentary, board, puzzle, storyboard)。"""
    from src.parser import parse_puzzle_input
    from src.storyboard import build_for_puzzle
    from src.commentator import generate_puzzle_structured

    Logger.info("=" * 20 + "Puzzle 战术讲解开始运行" + "=" * 20)

    Logger.info("[1/4] 解析 Puzzle 输入...")
    puzzle = parse_puzzle_input(input_text)
    board = chess.Board(puzzle.fen)

    if not board.is_valid():
        status = board.status()
        if status != chess.STATUS_OPPOSITE_CHECK:
            Logger.error(f"非法初始局面: FEN不合法 (status={status})")
            return None

    # Lichess 预备步：推进对方铺垫手，使棋盘到达解题起始位置
    if puzzle.prelude_move is not None:
        board.push(puzzle.prelude_move)

    Logger.info(f"  标签: {puzzle.effective_themes}, 步数: {len(puzzle.moves)}, Rating: {puzzle.rating}")

    Logger.info("[2/4] 构建战术分镜...")
    storyboard = build_for_puzzle(board, puzzle.moves, puzzle)

    Logger.info("[3/4] 生成战术解说...")
    try:
        commentary = generate_puzzle_structured(board, storyboard)
    except Exception as e:
        Logger.error(f"Puzzle 解说生成失败: {e}")
        return None

    Logger.info("[4/4] 解说生成完成")
    return commentary, board, puzzle, storyboard


def run_puzzle(input_text: str) -> str:
    """输出纯解说文本（对应 --puzzle --text）。"""
    result = _run_puzzle_pipeline(input_text)
    if result is None:
        return ""
    commentary, _board, _puzzle, _storyboard = result
    print(commentary.raw_text)
    return commentary.raw_text


def _build_puzzle_segments(commentary, moves: list, nodes: list) -> list:
    """按节点构造 Segment 列表（puzzle 版：无开场白/总结段，phase 为空）。"""
    voice_map = {}
    if commentary.segments:
        for seg in commentary.segments:
            voice_map[seg.id] = (seg.voiceover, seg.pacing)

    result = []
    for node in nodes:
        nid = node["id"]
        vo, pac = voice_map.get(nid, (None, "normal"))
        text = vo if vo else ""
        # 每节点一个 move（puzzle 不压缩）
        node_moves = []
        if nid <= len(moves):
            node_moves = [moves[nid - 1]]
        result.append(Segment(
            move_idx=nid,
            text=text,
            pacing=pac or "normal",
            moves=node_moves,
            phase="",
        ))
    return result


def run_puzzle_video(input_text: str, voice_prompt: str = "") -> str:
    """输出视频（对应 --puzzle）。Phase 1 先沿用现有片头片尾保证链路跑通。"""
    result = _run_puzzle_pipeline(input_text)
    if result is None:
        return ""

    commentary, board, puzzle, storyboard = result
    moves = puzzle.moves
    nodes = storyboard.get("nodes", [])
    if not moves:
        Logger.error("无有效走法序列")
        return ""

    tactic_name = storyboard.get("tactic_name", "战术练习")

    # TTS 语音合成
    Logger.info("TTS 语音合成...")
    from src.tts_engine import synthesize as tts_synthesize

    segments = _build_puzzle_segments(commentary, moves, nodes)
    segments = tts_synthesize(segments, voice_prompt=voice_prompt)

    # 生成视频
    Logger.info("生成视频...")
    from src.board_renderer import render_animated_frames
    from src.subtitle_gen import generate as gen_subtitles
    from src.video_composer import compose, LEAD_SILENCE, INTRO_SEC

    panel_info = {"endgame_name": tactic_name}

    frame_paths, frame_durations = render_animated_frames(
        segments, board.fen(), panel_info=panel_info)

    # puzzle 链路跳过片头片尾，字幕偏移仅含初始局面展示时长
    subtitle_offset = INTRO_SEC

    from src.subtitle_gen import build_cues
    srt_path = gen_subtitles(segments, offset_s=subtitle_offset)
    cues = build_cues(segments, offset_s=subtitle_offset)

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
        Logger.success(f"Puzzle 视频已生成: {output_path}")
        return output_path
    finally:
        _cleanup_temp_files(frame_paths, srt_path, segments)


def run_puzzle_csv(filepath: str, limit: int = 0, text_only: bool = False) -> str:
    """批量处理 CSV 文件（Phase 3）。

    产物写入 output/puzzle_batch/{时间戳}/，单题命名 {PuzzleId}.mp4，
    批次根目录生成 index.json。
    """
    from src.parser import parse_puzzle_csv
    from datetime import datetime
    import json as _json

    Logger.info(f"批量处理 CSV: {filepath}")
    puzzles = parse_puzzle_csv(filepath)
    if not puzzles:
        Logger.error("CSV 中未解析到有效 Puzzle")
        return ""

    if limit > 0:
        puzzles = puzzles[:limit]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join("output", "puzzle_batch", timestamp)
    os.makedirs(batch_dir, exist_ok=True)

    index = []
    total = len(puzzles)
    for i, puzzle in enumerate(puzzles):
        pid = puzzle.puzzle_id or f"row_{i + 1:05d}"
        Logger.info(f"[{i + 1}/{total}] 处理 {pid}...")

        status = "success"
        error = ""
        output_path = ""
        try:
            # 重新序列化为 text 格式以复用 _run_puzzle_pipeline
            input_text = (
                f'{{"fen":"{puzzle.fen}","moves":"{" ".join(m.uci() for m in puzzle.moves)}",'
                f'"themes":"{" ".join(puzzle.raw_themes)}","rating":{puzzle.rating},'
                f'"popularity":{puzzle.popularity},"openingTags":"{puzzle.opening_tags}",'
                f'"puzzle_id":"{puzzle.puzzle_id}"}}'
            )
            if text_only:
                result = _run_puzzle_pipeline(input_text)
                if result is None:
                    raise RuntimeError("管线返回空结果")
                commentary, _board, _pz, _sb = result
                output_path = os.path.join(batch_dir, f"{pid}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(commentary.raw_text)
            else:
                output_path = run_puzzle_video(input_text)
                if not output_path:
                    raise RuntimeError("视频生成失败")
                # 移动到批次目录
                dest = os.path.join(batch_dir, f"{pid}.mp4")
                import shutil
                shutil.move(output_path, dest)
                output_path = dest
        except Exception as e:
            status = "failed"
            error = str(e)
            Logger.error(f"  {pid} 失败: {e}")

        index.append({
            "puzzle_id": pid,
            "rating": puzzle.rating,
            "themes": puzzle.effective_themes,
            "output_path": output_path,
            "status": status,
            "error": error,
        })

    # 写 index.json
    index_path = os.path.join(batch_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        _json.dump(index, f, ensure_ascii=False, indent=2)

    succeeded = sum(1 for e in index if e["status"] == "success")
    Logger.success(f"批量处理完成: {succeeded}/{total}, 索引: {index_path}")
    return batch_dir
