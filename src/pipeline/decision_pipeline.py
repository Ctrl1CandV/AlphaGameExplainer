"""决策管线（ADR-020 第三条决策管线，阶段 8a 线性视频版）。

从决策输入到视频：
  1. 全链路计算：识别原型 → KB 计划 → explore_forward / assess_feasibility
     → project（趋势）→ quantify_tradeoffs → build_decision_storyboard
     → generate_decision_commentary；
  2. 视频组装（阶段 8a 线性版）：叙事单元 → Segment——着法（moves）驱动
     画面动画，解说文本（text）进 TTS/字幕（**着法与口播分离**——画面
     演示计划线，口播无坐标无走法，ADR-020 约束 5）；
  3. 两条计划用两个独立渲染序列先后播放（演示计划甲 → 演示计划乙——
     各自从决策点局面渲染，不回溯不预览未来——阶段 8a 定义）；
  4. TTS / 字幕 / 合成复用 puzzle 管线同一套设施（compose 等）。

8b（回溯 + 未来局面预览）为 board_renderer 加法扩展，v2 再做。
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import List, Optional, Tuple

try:  # 直接运行自检时补充项目根到 sys.path
    from src.common import Logger, Segment
    from src.media.board_renderer import FRAMES_DIR, render_animated_frames
    from src.media.subtitle_gen import build_cues, generate as gen_subtitles
    from src.media.tts_engine import synthesize as tts_synthesize
    from src.media.video_composer import INTRO_SEC, cleanup_artifacts, compose
except ModuleNotFoundError:
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         "..", ".."))
    sys.path.insert(0, _ROOT)
    from src.common import Logger, Segment
    from src.media.board_renderer import FRAMES_DIR, render_animated_frames
    from src.media.subtitle_gen import build_cues, generate as gen_subtitles
    from src.media.tts_engine import synthesize as tts_synthesize
    from src.media.video_composer import INTRO_SEC, cleanup_artifacts, compose

import chess

# 计划线演示着数（画面动画长度——口播无坐标，着法仅驱动画面）
#
# 取值依据（08.04 修，原值 10 过快）：renderer 把**一段的音频时长**摊给该段
# 内所有子步，所以每步停留 ≈ 段时长 / 着数。决策管线一条计划只有一段解说，
# 段时长由该段字数决定：
#   字数上限 130 字 → ChatTTS 约 20~25s → 10 着时每步仅约 2s（含滑动开销
#   后定格不足 1.5s），观众看不清任何一步；6 着时每步约 3.5~4s，与老管线
#   节奏相当（endgame/puzzle 的 node 普遍 1~3 着配一段解说）。
# 6 半回合 = 3 个完整回合，足够展示「推进 → 对方应对 → 跟进」这一最小
# 计划轮廓，再长则单段内信息过载且与解说文本脱节（解说只讲结构趋势，
# 不逐着解释）。若将来要演示更长的线，正确做法是拆成多段各配解说
# （阶段 8b 回溯版一并处理），不是继续加大本值。
LINE_DISPLAY_PLY = 6

# 渲染子目录（两次独立渲染避免帧文件编号冲突）
_SEQ_A_DIR = os.path.join(os.path.dirname(FRAMES_DIR), "frames_seq_a")
_SEQ_B_DIR = os.path.join(os.path.dirname(FRAMES_DIR), "frames_seq_b")


def _render_sequence(segments: List[Segment], initial_fen: str,
                     panel_info: Optional[dict], subdir: str
                     ) -> Tuple[List[str], List[float]]:
    """渲染单序列到独立目录（08.04 改：直接用 renderer 的 frames_dir 参数）。

    前版做法是「渲染到公共 FRAMES_DIR → shutil.move 搬到子目录」，绕过
    renderer 而非扩展它，带来两个问题：多一轮全帧磁盘搬运；且无法阻止
    renderer 回填 `start_time`（第二序列会把 seq_b 覆盖成 B 内相对时间，
    字幕 cue 错乱），只能在外面再重算一次时间轴打补丁。

    现按 PLAN-009 阶段 8b 的既定方式对 board_renderer 做加法式扩展：
      - `frames_dir`：直接写目标目录，无需搬运；
      - `write_start_time=False`：分序列渲染时不回填相对时间轴，由调用方
        在两序列都渲染完后统一按段序累加（见 `_rebuild_global_timeline`）。
    两参数都有默认值，既有 endgame/puzzle 调用不传即保持原行为（零回归）。
    """
    if os.path.isdir(subdir):          # 复跑残留帧会混入本次输出
        shutil.rmtree(subdir, ignore_errors=True)
    return render_animated_frames(segments, initial_fen,
                                  panel_info=panel_info,
                                  frames_dir=subdir,
                                  write_start_time=False)


def _rebuild_global_timeline(segments: List[Segment]) -> None:
    """按段序重算全局 start_time（分序列渲染的必要收尾）。

    `_render_sequence` 传 `write_start_time=False`，故此处是 start_time 的
    唯一写入点——字幕 cue 与画面时间轴据此对齐。`duration_s` 由 renderer
    按各段实际帧时长写好，这里只做前缀累加，不改时长。
    """
    cursor = 0.0
    for seg in segments:
        seg.start_time = cursor
        cursor += seg.duration_s


def build_video_segments(
    decision_storyboard: dict,
    commentary,
) -> List[Segment]:
    """决策解说 → 视频 segments（叙事单元 → Segment）。

    开场/对比/总结段 moves 空（静态画面）；计划段 moves = 计划线前
    LINE_DISPLAY_PLY 着（从决策点出发——画面演示，口播无坐标）。
    """
    routes = decision_storyboard.get("routes", [])
    segs: List[Segment] = []

    # 开场（决策点静态画面）
    segs.append(Segment(
        move_idx=0, text=getattr(commentary, "opening", "") or
        "这个局面存在多条可行的战略路线。",
        moves=[], phase="decision"))

    # 计划段（各自从决策点渲染）
    plan_sans = []
    for i, route in enumerate(routes):
        line = route.get("_line_pv", []) or []
        moves = line[:LINE_DISPLAY_PLY]
        text = ""
        for seg in getattr(commentary, "segments", []):
            if int(getattr(seg, "id", -1)) == i + 1:
                text = getattr(seg, "voiceover", "")
                break
        segs.append(Segment(
            move_idx=i + 1, text=text or f"方案：{route.get('name', '?')}",
            moves=moves, phase=route.get("name", "plan")))

    # 对比段（停在计划末局面）
    cmp_text = ""
    for seg in getattr(commentary, "segments", []):
        if int(getattr(seg, "id", -1)) == len(routes) + 1:
            cmp_text = getattr(seg, "voiceover", "")
            break
    segs.append(Segment(
        move_idx=len(routes) + 1, text=cmp_text, moves=[],
        phase="compare"))

    # 总结段
    summary = getattr(commentary, "summary", "") or ""
    segs.append(Segment(move_idx=len(routes) + 2, text=summary, moves=[],
                        phase="summary"))
    return segs


def _split_sequences(segments: List[Segment]) -> Tuple[List[Segment], List[Segment]]:
    """8a 线性版双序列：序列 A = 开场+计划甲；序列 B = 计划乙+对比+总结。

    两条计划各自从决策点局面渲染（不回溯——阶段 8a 定义）。
    """
    # 计划段是带 moves 的段（第 1、2 个带 moves 的）；序列 A 到第一个
    # 计划段结束，序列 B 从第二个计划段开始。
    plan_idx = [i for i, s in enumerate(segments) if s.moves]
    if len(plan_idx) < 2:
        return segments, []
    cut = plan_idx[1]
    return segments[:cut], segments[cut:]


def _synthesize_voice(segments: List[Segment],
                      voice_prompt: Optional[str]) -> Optional[List[Segment]]:
    """整批合成语音，对齐 endgame/puzzle 两条管线的既有调用方式。

    一次 `tts_synthesize(全部段)`：ChatTTS 模型只加载一次、说话人向量复用，
    音色跨段一致；`synthesize` 内部已实现「ChatTTS 优先 → 逐段失败才走
    pyttsx3 兜底 → 统一重算时间轴」的完整降级链，无需在管线侧再包一层。

    失败返回 None，调用方按 SPEC §8 / FINDINGS P11 放弃整片——决策管线
    自身就是产品，没有「去掉该功能仍可交付的主体」可回退，出无声片或
    音画错位片比不出片更糟。

    历史教训（08.04 修）：前版在管线侧套了「逐段独立线程 + 45s join 超时」，
    三个缺陷叠加导致阶段 8a 卡死——
      1. `join` 超时只是不再等待，**并不终止线程**。ChatTTS 是 GPU 推理，
         超时段仍在后台占显存续跑，后续段再起新线程并发争抢，越跑越慢，
         最终全段超时全段降级；
      2. 超时分支引用了内层函数的形参 `target`，外层作用域并无此名 →
         必然 `NameError`，管线在第一次超时处直接崩溃（现场表现：只产出
         seg_000.wav，帧目录为空，成片是上一轮旧产物）；
      3. 阈值本身不成立。实测 ChatTTS 96 字需约 27s（GPU），而解说段当时
         无字数上限，长段必然突破 45s。
    根因不在 TTS 设施（实测模型加载 1s 正常），在管线侧自造的超时机制 +
    解说无字数预算。前者删除，后者由 decision_commentary 的字数约束解决。
    """
    try:
        return tts_synthesize(segments, voice_prompt=voice_prompt)
    except Exception as e:  # noqa: BLE001
        Logger.error(f"TTS 合成失败: {e}")
        return None


def _run_decision_pipeline(input_fen: str, provenance: Optional[str] = None,
                           output_dir: Optional[str] = None,
                           voice_prompt: Optional[str] = None) -> str:
    """决策管线主入口（8a 线性视频版）。返回输出视频路径。"""
    import json

    from src.analysis.structure_features import structural_features
    from src.commentator.decision_commentary import generate_decision_commentary
    from src.solver.branch_explorer import (
        assess_feasibility,
        explore_forward,
        explore_open,
        waiting_baseline,
    )
    from src.solver.consequence_projector import quantify_tradeoffs, project
    from src.storyboard.decision_builder import (
        DecisionInput,
        PlanOutcome,
        build_decision_storyboard,
    )

    Logger.info("=" * 20 + "Decision 决策管线开始运行" + "=" * 20)

    # 1. 全链路计算
    sf = os.getenv("STOCKFISH_PATH", "")
    if not os.path.isabs(sf):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            "..", ".."))
        sf = os.path.normpath(os.path.join(root, sf))
    kb = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
            __file__)))), "data", "structure_kb.json"), encoding="utf-8"))

    from src.analysis.structure_id import detect_pawn_structure
    board = chess.Board(input_fen)
    arch, _, _ = detect_pawn_structure(board)
    if arch is None:
        Logger.warn("无法识别兵形原型——按 SPEC §8 放弃（决策管线无回退主体）")
        return ""

    # 产品池闸门（08.04）：原型须通过 P0-full A3 可分离性才可出片。
    # 闸门由 KB 的 `in_production` 字段驱动（缺省 True——已验证原型无需标注），
    # 判据与理由写在 KB 条目里，启用/停用只改数据不改代码。
    # 当前停用：stonewall（封闭兵链，两计划推进后结构差异过小，A3 未过）。
    # 识别得出的原型不在池内时按 SPEC §8 放弃——宁可不出片，也不出
    # 「两条路其实分不开却讲成对比」的误导内容。
    if not kb[arch].get("in_production", True):
        Logger.warn(
            f"原型 {arch}（{kb[arch].get('cn', '')}）不在产品池——"
            f"{kb[arch].get('in_production_note', '未通过可分离性验证')}"
            "按 SPEC §8 放弃本片生成")
        return ""
    plans = kb[arch]["plans"]

    opens = explore_open(board, sf, k=4, depth=14)
    baseline = waiting_baseline(board, sf, depth=12)
    outcomes = []
    for plan in plans:
        line = explore_forward(board, plan, sf, depth=14)
        if line is None or not line.pv:
            continue
        feas, gap = assess_feasibility(
            line.cp, opens[0].cp if opens else None)
        tr = project(line, board, sf)
        tm = quantify_tradeoffs(line, board, sf, open_lines=opens)
        outcomes.append(PlanOutcome(
            plan=plan, line_cp=line.cp, line_pv=line.pv,
            feasible=feas, gap_cp=gap, trend=tr,
            tradeoffs=tm.__dict__,
            start_features=structural_features(board),
            end_features=tr.get("end_features", [])))

    sb = build_decision_storyboard(
        DecisionInput(fen=input_fen, provenance=provenance), outcomes,
        archetype=arch, strategic_premise=kb[arch]["theory"],
        baseline=baseline)
    # 计划线挂回 storyboard（视频渲染用——口播无坐标，画面需着法）
    for route, outcome in zip(sb["routes"], outcomes):
        route["_line_pv"] = outcome.line_pv

    commentary = generate_decision_commentary(
        DecisionInput(fen=input_fen, provenance=provenance), sb)
    # 解说级失败（SPEC §8 语义：管线级失败 = 整片不出）——本地降级 LLM
    # 输出不合格时会 aborted——直接放弃，不生成空解说视频
    if getattr(commentary, "aborted", False):
        Logger.error(f"解说生成失败（{getattr(commentary, 'aborted_reason', '?')}）"
                     "——按 SPEC §8 放弃本片生成")
        return ""
    # 解说预览（诊断——对照视频字幕核对解说段完整性）
    Logger.info("===== 解说词预览 =====")
    if commentary.opening:
        Logger.info(f"[开场] {commentary.opening[:60]}...")
    for seg in getattr(commentary, "segments", []):
        Logger.info(f"[段{getattr(seg, 'id', '?')}] "
                    f"{getattr(seg, 'voiceover', '')[:60]}...")
    if commentary.summary:
        Logger.info(f"[总结] {commentary.summary[:60]}...")
    Logger.info(f"===== 解说段数 {len(commentary.segments)} =====")

    # 2. 视频组装
    segments = build_video_segments(sb, commentary)
    # TTS 整批合成（对齐两条老管线）。失败即整片不出——决策管线自身就是
    # 产品，无可回退主体（SPEC §8 / FINDINGS P11）；估算时长产出的是音画
    # 脱节的无声片，比不出片更糟。
    tts_segments = _synthesize_voice(segments, voice_prompt)
    if tts_segments is None:
        Logger.error("TTS 合成失败——按 SPEC §8 放弃本片生成")
        return ""
    segments = tts_segments

    # 诊断：方案数（P8 选线只保留可行计划——若仅 1 计划可行会单线退化）
    Logger.info(f"[Decision] 可行计划 {len(sb['routes'])} 个 -> "
                f"视频序列 {len(segments)} 段（带走法段 "
                f"{sum(1 for s in segments if s.moves)}）")

    # 3. 双序列渲染拼接（8a 线性——各自从决策点）
    seq_a, seq_b = _split_sequences(segments)
    panel_info = {"endgame_name": kb[arch]["cn"]}
    paths_a, durs_a = _render_sequence(seq_a, input_fen, panel_info, _SEQ_A_DIR)
    if seq_b:
        paths_b, durs_b = _render_sequence(seq_b, input_fen, panel_info,
                                           _SEQ_B_DIR)
        frame_paths = paths_a + paths_b
        frame_durations = durs_a + durs_b
    else:
        frame_paths, frame_durations = paths_a, durs_a

    # 全局时间轴由本处统一赋值。两次渲染都传了 write_start_time=False，
    # 渲染器不再回填「本次调用内相对时间」，这里按 A→B 段序累加即是全局值。
    # duration_s 已由渲染器写入（画面占用时长，与调用次序无关），可直接累加。
    cursor = 0.0
    for seg in segments:
        seg.start_time = cursor
        cursor += seg.duration_s

    # 4. 字幕 + 合成（跳过片头片尾，puzzle 同款）
    srt_path = gen_subtitles(segments, offset_s=INTRO_SEC)
    cues = build_cues(segments, offset_s=INTRO_SEC)
    try:
        output_path = compose(
            frame_paths=frame_paths,
            frame_durations=frame_durations,
            segments=segments,
            srt_path=srt_path,
            endgame_name=kb[arch]["cn"],
            cues=cues,
            initial_fen=input_fen,
            skip_title=True,
            skip_outro=True,
        )
        Logger.success(f"Decision 视频已生成: {output_path}")
        return output_path
    finally:
        cleanup_artifacts(frame_paths, srt_path, segments)
        for d in (_SEQ_A_DIR, _SEQ_B_DIR):
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    """阶段 8a 验证：悬兵示例局面产出线性视频（肉眼验收）。"""
    import os
    import sys

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, root)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(root, ".env"))

    fen = "2r1r1k1/pp2bppp/1nnp4/5q2/2PP4/1Q3NBP/P2N1PP1/1R2R1K1 w - - 1 21"
    out = _run_decision_pipeline(fen, provenance="d5",
                                 output_dir=os.path.join(root, "output"))
    print(f"输出: {out}")
