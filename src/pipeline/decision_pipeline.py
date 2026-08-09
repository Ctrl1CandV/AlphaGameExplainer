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
    from src.infra.llm_backend import release_backend
    from src.media.board_renderer import FRAMES_DIR, render_animated_frames
    from src.media.subtitle_gen import build_cues, generate as gen_subtitles
    from src.media.tts_engine import synthesize as tts_synthesize
    from src.media.video_composer import INTRO_SEC, cleanup_artifacts, compose
except ModuleNotFoundError:
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         "..", ".."))
    sys.path.insert(0, _ROOT)
    from src.common import Logger, Segment
    from src.infra.llm_backend import release_backend
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

# 渲染子目录：每条计划一个独立目录，避免帧文件编号互相覆盖。
#
# 08.04 由固定的 A/B 两个常量改为按序号生成。原设计写死两个目录，隐含
# 「可行计划恰好 2 条」的假设——而 IQP 的 KB 有 4 条计划、实测 4 条全部
# 通过可行性闸，第 3、4 条无处可去（详见 `_split_sequences`）。
def _seq_dir(idx: int) -> str:
    """第 idx 条渲染序列的帧目录（idx 从 0 起）。"""
    return os.path.join(os.path.dirname(FRAMES_DIR), f"frames_seq_{idx}")


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

    # 对比段（停在计划末局面）。**只有两条以上计划才有对比段**——
    # 必须与 `decision_commentary._build_decision_nodes` 的 `if len(routes) >= 2`
    # 同条件（08.04 修）。此前本处无条件取 id = len(routes)+1，而单线退化时
    # （可行计划 1 个，P11 允许且算成功产出）解说节点只有
    # [opening(0), plan(1), summary(2)]，len(routes)+1 恰好等于 2 = **总结节点**
    # 的 id。于是总结文本被当成对比段取走，紧接着又被总结段取一次，成片里
    # 同一段话连播两遍（实测 majority 局面：段2 与总结相似度 0.991）。
    # 节点结构是解说侧定义的，视频侧不能自己推算 id——条件必须两处一致。
    if len(routes) >= 2:
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
    segs.append(Segment(move_idx=len(segs), text=summary, moves=[],
                        phase="summary"))
    return segs


def _split_sequences(segments: List[Segment]) -> List[List[Segment]]:
    """按计划切分渲染序列：**每条计划一个序列**，各自从决策点局面渲染。

    返回序列列表（1 个计划 → 1 个序列，N 个计划 → N 个序列）：
      序列 1 = 开场 + 计划甲
      序列 i = 计划 i（2 ≤ i < N）
      序列 N = 计划 N + 对比 + 总结
    切点取每个「带 moves 的段」的下标——不含 moves 的段（开场/对比/总结）
    归入相邻序列，静态定格不需要独立起始局面。

    **必须按计划数动态切，不能只切两段**（08.04 修）。前版硬编码
    `cut = plan_idx[1]` 只在第二条计划处切一刀，隐含假设「恰好 2 条计划」。
    但可行计划数由可行性闸决定，KB 的 iqp 有 4 条计划、实测 4 条全部可行：
    计划乙/丙/丁被塞进同一个序列，renderer 于是把它们的着法**连续 push 到
    同一块棋盘上**——计划丙的首着在计划乙走完后的局面里根本不合法，
    `gives_check()` 内部 push 直接 AssertionError 崩掉整条管线
    （实测 iqp：`push() expects move to be pseudo-legal, but got a8d8`）。
    这不是渲染细节，是画面语义错误：阶段 8a 的定义是「每条计划各自从决策点
    出发演示」，把两条计划接在一条时间线上演示，等于告诉观众「先走甲再走乙」，
    而它们本是同一个决策点上互斥的两个选择。

    2 条计划时切点与前版完全一致（`[:plan_idx[1]], [plan_idx[1]:]`），
    既有 carlsbad/hanging 等双计划局面零行为变化。
    """
    plan_idx = [i for i, s in enumerate(segments) if s.moves]
    if len(plan_idx) < 2:
        return [segments]
    # 每个计划段起点即一刀；首刀之前（开场）并入第一序列
    cuts = plan_idx[1:]
    seqs, start = [], 0
    for cut in cuts:
        seqs.append(segments[start:cut])
        start = cut
    seqs.append(segments[start:])          # 末序列含对比+总结
    return [s for s in seqs if s]


def _release_llm() -> None:
    """释放 LLM 后端（显存 + 缓存），失败只告警不打断出片。

    吞异常与两条老管线一致：后端已经用完，释放失败最坏是这一轮 TTS 掉到
    CPU（慢但仍出片），不该因清理动作把已经算完的解说丢掉。
    """
    try:
        release_backend()
    except Exception as e:  # noqa: BLE001
        Logger.warn(f"释放 LLM 后端失败（不影响本片生成）: {e}")


def _jsonable(obj, _depth: int = 0):
    """把 storyboard 里的 dataclass / 元组递归转成 JSON 可写的原生类型。

    sidecar 的 `trend` 直接来自 `project()`，其 `trends` / `rejected_trends`
    装的是 `StructureTrend` dataclass，`archetype_shift` 是元组——`json.dump`
    对二者都不认，整份 sidecar 写到一半抛 `TypeError`。08.04 实测：文件已
    open("w") 截断、错误发生在 dump 中途，于是磁盘上留下一个**被截断的半份
    JSON**；阶段 9 评审脚本 `json.load` 读它必崩（JSONDecodeError）。

    只做类型转换、不裁字段：评审要判「产出这一片时」的完整结构数据，
    裁字段等于把判据的输入悄悄改小。

    `_depth` 是自环护栏：当前 `project()` 的产物是纯树（无环），但本函数是
    通用转换器，将来若有人往 storyboard 挂上互相引用的对象，无限递归会以
    `RecursionError` 表现成「sidecar 神秘丢失」。超深即退化为字符串——
    与陌生类型的兜底同一策略：宁可这一处不精确，也不让整份 sidecar 写不出。
    """
    import dataclasses

    if _depth > 12:
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # 用 fields+getattr 而非 dataclasses.asdict：asdict 会**自己**深递归
        # 整棵嵌套结构，递归发生在本函数之外，上面的 `_depth` 护栏管不到它
        # （实测自引用 dataclass 仍抛 RecursionError）。逐字段浅取、由本函数
        # 统一递归，护栏才对所有路径生效。
        return {f.name: _jsonable(getattr(obj, f.name, None), _depth + 1)
                for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, _depth + 1) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)          # 兜底：陌生类型退化为字符串，绝不让整份写失败


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


def _screen_actual_move(board: chess.Board, actual: str,
                        sf_path: str) -> Optional[str]:
    """实战续走首着过双重校验则返回其 SAN，否则 None（该段缺席）。

    `actual` 接受 SAN（`Nb4`）或 UCI（`c6b4`）——挖掘器产出的 continuation
    是 UCI，人工填写时更习惯 SAN，两种都认比要求调用方统一格式更稳妥
    （格式转换是纯函数，判错的代价却是整段内容缺席或讲错）。

    返回 SAN 而非 Move：下游 `_match_provenance_plan` 与 prompt 注入都按
    SAN 处理，在此统一转换一次，避免每个消费点各转一遍。
    """
    from src.solver.branch_explorer import assess_actual_move

    mv = None
    try:                                   # 先按 SAN 解析（人工填写的常见形态）
        mv = board.parse_san(actual)
    except Exception:                      # noqa: BLE001
        try:                               # 再按 UCI（挖掘器 continuation 格式）
            cand = chess.Move.from_uci(actual)
            if cand in board.legal_moves:
                mv = cand
        except Exception:                  # noqa: BLE001
            mv = None
    if mv is None:
        Logger.warn(f"实战续走首着 {actual!r} 不是本局面的合法着——"
                    "跳过实战对照段（不注入未校验内容）")
        return None

    passed, loss = assess_actual_move(board, mv, sf_path)
    san = board.san(mv)
    if not passed:
        Logger.info(f"[Decision] 实战首着 {san} 未过评估筛"
                    f"（净损失 {loss}cp）——不注入实战对照段")
        return None
    Logger.info(f"[Decision] 实战首着 {san} 过评估筛（净损失 {loss}cp）")
    return san


def _decision_core(input_fen: str, provenance: Optional[str] = None) -> Optional[dict]:
    """决策管线**前半段**：识别 → 引擎探索 → storyboard → LLM 解说 → 释放 LLM。

    返回 bundle dict（含 `arch`/`kb`/`board`/`sb`/`by_name`/`commentary`），
    或 `None`（任一 SPEC §8 放弃点——无原型/不在产品池/无可行计划/解说 aborted）。

    **文本路径（`run_decision --text`）与视频路径（`_run_decision_pipeline`）
    共用此函数**（PLAN-011 阶段 3）——此前 `_run_decision_pipeline` 是单一耦合
    函数、无中间出口，文本模式无法复用。拆分点在原「2. 视频组装」注释处（LLM
    解说完成、`_release_llm` 之后）：前半段是「算内容」（识别/引擎/storyboard/
    解说，含 SPEC §8 全部放弃闸），后半段是「出视频」（TTS/渲染/合成/sidecar）。
    拆分是纯函数提取，前半段逐行行为不变。

    `provenance`：实战续走**首着**（SAN 或 UCI），经双重校验后才进解说。
    **不是开局名**——传开局名会被当作非法着拒掉，实战对照段静默缺席
    （08.04 实测：stage9 runner 误传 `pick["opening"]`，5 个 demo 的
    Tier A 段全程没出现过）。
    """
    import json

    from src.analysis.structure_features import (
        goal_trajectory,
        structural_features,
    )
    from src.commentator.decision_commentary import generate_decision_commentary
    from src.solver.branch_explorer import (
        assess_actual_move,
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

    from src.analysis.structure_id import (
        applicable_mover_side,
        detect_pawn_structure,
    )
    board = chess.Board(input_fen)
    arch, _, _ = detect_pawn_structure(board)
    if arch is None:
        Logger.warn("无法识别兵形原型——按 SPEC §8 放弃（决策管线无回退主体）")
        return None

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
        return None
    # 只保留**决策点走子方真能执行**的计划（08.04 补，Critical）。
    #
    # KB 的 iqp 条目里 4 条计划分属两种角色：2 条是孤兵持有方的选择
    # （保持/推进兑掉），2 条是面对孤兵一方的选择（施压/推进消除）。而
    # `_iqp_check` 是「任一方持孤兵即命中」，管线此前完全不读 `mover_side`
    # 字段——于是把双方的计划一起当成「走子方的四条路」讲。
    # 实测 iqp demo（黑方走子、白持 d4 孤兵）四条全讲，其中两条是白方的
    # 选择，黑方根本走不了：观众照着讲解走，走的是对手的计划。这是硬事实
    # 错（SPEC §8 零容忍），不是表达瑕疵。
    # 角色判不出（None）时不过滤——carlsbad/maroczy/majority 的判据本身
    # 锚定走子方，KB 里这些原型全部计划都是 mover_side="mover"，无歧义。
    side = applicable_mover_side(board, arch)
    plans = kb[arch]["plans"]
    if side is not None:
        applicable = [p for p in plans if p.get("mover_side") == side]
        if applicable:
            dropped = [p["name"] for p in plans if p not in applicable]
            if dropped:
                Logger.info(f"[Decision] 走子方角色={side}，滤除对手方计划 {dropped}")
            plans = applicable
        else:
            # KB 该原型没有本角色的计划——无可讲内容，按 SPEC §8 放弃，
            # 不退化成「讲对手的计划」。
            Logger.warn(f"原型 {arch} 无 mover_side={side} 的计划——"
                        "按 SPEC §8 放弃本片生成")
            return None

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

        # 机制成立闸（08.04 补，用户裁决"闸门化"）：可行性不只看评估差，
        # 还要看这条计划的**机制在这个局面是否真的跑得起来**。
        #
        # A2 轨迹一致性（`goal_ok` = 线内达成 structural_goal 或朝达成
        # 显著移动）此前只是阶段 9 的**事后评审维度**，不参与选线——于是
        # 机制不成立的计划照样进片。实测 maroczy 决策点的「王翼进攻」：
        # goal 是「己方兵过中线」，而方向约束下的最强线首着是 Bf8（退象），
        # 全线 19 个半回合黑方一个兵都没过中线（逐着值恒 0）。解说照着
        # KB 的 mechanism 讲「用王翼兵冲击对方兵墙」，画面却在演退象——
        # 讲的是一个在此局面并不成立的计划。
        #
        # 判据复用 `goal_trajectory`（与评审脚本 d2 同一函数，单一事实
        # 来源），视角锚定决策点走子方。无 structural_goal 的计划不拦
        # （KB 目前全都有，留作 schema 演进的失败安全）。
        goal = plan.get("structural_goal") or {}
        mech_ok = True
        if goal:
            traj = goal_trajectory(board, line.pv, goal, board.turn)
            mech_ok = bool(traj["goal_ok"])
            if not mech_ok:
                Logger.info(
                    f"[Decision] 计划「{plan.get('name')}」机制在本局面不成立"
                    f"（结构目标既未达成也未朝达成移动），不参与选线")
        outcomes.append(PlanOutcome(
            plan=plan, line_cp=line.cp, line_pv=line.pv,
            feasible=feas and mech_ok, gap_cp=gap, trend=tr,
            tradeoffs=tm.__dict__,
            start_features=structural_features(board),
            end_features=tr.get("end_features", [])))

    # 实战对照段的双重校验（08.04 补，阶段 9 P12/P19/P24）。
    #
    # 校验放在**管线内部**而非调用方：调用方只需给出「实战续走首着」这个
    # 事实，是否够格进解说由管线判。放在外面则每个调用方都得自己记得筛，
    # 漏一个就等于把未校验的实战着直接讲成参考答案。
    #
    # 两道筛（PLAN-009 阶段 9「两条都过才注入」）：
    #   ① 时限筛——已在阶段 0 挖掘完成（G1 读 Event 头官方分类，实测
    #      20000 局里 18582 局因 blitz/bullet 排除），本阶段无需重复；
    #   ② 评估筛——`assess_actual_move` 判该着相对最优着的净损失 ≤30cp。
    # 未过（或根本不是合法着）即置 None：宁可缺这一段，也不把一手失误
    # 讲成「这个水平的棋手更多选了它」。
    #
    # 历史教训：阶段 9 首轮 5 个 demo 里这一段**一句都没出现**——runner 把
    # `pick["opening"]`（开局名，如 "Sicilian Defense: O'Kelly Variation"）
    # 当 provenance 传了进来，而 `_match_provenance_plan` 第一步就是
    # `if provenance_san not in moves: return None`，开局名当然不是合法着，
    # 于是 5 个局面全部静默返回 None，4 份 sidecar 的 Tier A 痕迹都是 None。
    # 参数类型错配 + 静默失败 = 功能整体缺席却没有任何报错。现在校验挪进
    # 管线，非法着会明确 warn，不再静默。
    if provenance:
        provenance = _screen_actual_move(board, provenance, sf)

    sb = build_decision_storyboard(
        DecisionInput(fen=input_fen, provenance=provenance), outcomes,
        archetype=arch, strategic_premise=kb[arch]["theory"],
        baseline=baseline)
    # 零可行计划闸（08.04 补，与上面的机制闸配套）。
    #
    # 机制闸把「结构目标不成立」的计划也判为不可行后，`sb["routes"]` 可能为空
    # ——此前只有 cp 差一条判据，至少总有一条计划留下，这个分支到不了。
    # 必须在**解说生成之前**拦：否则白烧一次 LLM 调用，再产出一部
    # 「开场说有可行方向、后面一条都没讲」的片子。决策管线自身就是产品，
    # 无可回退主体（SPEC §8 / P11），无内容即整片不出。
    if not sb.get("routes"):
        Logger.warn(f"原型 {arch}（{kb[arch].get('cn', '')}）在本局面无可行计划"
                    "——按 SPEC §8 放弃本片生成")
        return None
    # 计划线挂回 storyboard（视频渲染用——口播无坐标，画面需着法）。
    #
    # 按**计划名**匹配，不按位置（08.04 修）。`sb["routes"]` 是
    # decision_builder 过滤后的可行子集（`[o for o in outcomes if o.feasible
    # and o.line_pv]`），而 `outcomes` 含全部算过的计划——两者长度不等时
    # `zip` 会错位配对：若计划甲不可行被滤掉，routes[0]（实为计划乙）会被
    # 挂上计划甲的着法线，于是画面演示甲的走法、解说讲乙的战略，是「硬事实
    # 错」级别的缺陷（SPEC §8 零容忍）。两个计划都可行时位置恰好对齐，所以
    # 此前的 5 个 demo 掩盖了它——一旦某原型有计划被可行性闸滤掉就会显形。
    by_name = {o.plan.get("name"): o for o in outcomes}
    for route in sb["routes"]:
        outcome = by_name.get(route.get("name"))
        if outcome is None:      # 理应不发生（routes 是 outcomes 子集）
            Logger.warn(f"计划「{route.get('name')}」找不到对应线，画面将无着法")
            continue
        route["_line_pv"] = outcome.line_pv

    # LLM 用完立即释放（08.04 补——本管线原先漏了，两条老管线各有 3 处）。
    # 必须在 TTS 之前：本机 RTX 4060 Laptop 共 8GB 显存，本地兜底模型
    # Qwen3.5-9B-Q6_K 权重 7.0GB（`LLAMA_CPP_N_GPU_LAYERS=-1` 全量上卡）几乎占满，
    # ChatTTS 按「加载时空闲显存」选设备（阈值约 2GB，见 tts_engine
    # `_free_gpu_before_tts`），后端不释放时它静默回退 CPU——合成慢一个数量级，
    # 正是阶段 8a「TTS 卡住」的直接成因之一。
    # 同时清掉 LLM_BACKEND_CACHE：本管线常在同进程内连跑多个 FEN，缓存里
    # 若留着上一轮已降级到本地的后端实例，后续 FEN 会一路用本地模型，
    # 再也不尝试 API——批量出片时质量与显存双输。
    # 用 finally 而非顺序调用：解说生成抛异常时也必须释放，否则批量出片
    # （stage9_demo_run 同进程连跑多个 FEN）会一路带着 14.5GB 残留跑下去。
    try:
        commentary = generate_decision_commentary(
            DecisionInput(fen=input_fen, provenance=provenance), sb)
    finally:
        _release_llm()
    # 解说级失败（SPEC §8 语义：管线级失败 = 整片不出）——本地降级 LLM
    # 输出不合格时会 aborted——直接放弃，不生成空解说视频
    if getattr(commentary, "aborted", False):
        Logger.error(f"解说生成失败（{getattr(commentary, 'aborted_reason', '?')}）"
                     "——按 SPEC §8 放弃本片生成")
        return None
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

    # 拆分点（PLAN-011 阶段 3）：前半段「算内容」到此结束，打包返回。
    # `by_name` 供视频路径挂线 + sidecar A2 轨迹；`board` 供 sidecar 视角锚定。
    return {"arch": arch, "kb": kb, "board": board, "sb": sb,
            "by_name": by_name, "commentary": commentary}


def _run_decision_pipeline(input_fen: str, provenance: Optional[str] = None,
                           output_dir: Optional[str] = None,
                           voice_prompt: Optional[str] = None) -> str:
    """决策管线视频路径（8a 线性视频版）。返回输出视频路径，放弃时返回 ""。

    前半段（识别→引擎→storyboard→解说）委托 `_decision_core`（与文本路径共用）；
    本函数是**后半段**：视频组装 → TTS → 多序列渲染 → 字幕合成 → 评审 sidecar。
    `_decision_core` 返回 None（任一 SPEC §8 放弃点）时本函数返回 ""，行为与
    拆分前一致。
    """
    import json
    from src.analysis.structure_features import goal_trajectory

    core = _decision_core(input_fen, provenance)
    if core is None:
        return ""      # SPEC §8 放弃点已在 _decision_core 内 warn/error
    arch = core["arch"]
    kb = core["kb"]
    board = core["board"]
    sb = core["sb"]
    by_name = core["by_name"]
    commentary = core["commentary"]

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

    # 3. 多序列渲染拼接（8a 线性——每条计划各自从决策点出发）
    seqs = _split_sequences(segments)
    panel_info = {"endgame_name": kb[arch]["cn"]}
    frame_paths: List[str] = []
    frame_durations: List[float] = []
    seq_dirs = []
    for i, seq in enumerate(seqs):
        d = _seq_dir(i)
        seq_dirs.append(d)
        paths_i, durs_i = _render_sequence(seq, input_fen, panel_info, d)
        frame_paths += paths_i
        frame_durations += durs_i

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
    # 批量产出时必须给每片独立文件名（08.04 修）。`output_dir` 此前是死参数——
    # 签名有、__main__ 传了、函数体从未使用，而 compose 内部把输出硬编码为
    # output/analysis.mp4。单片生成看不出问题，阶段 9 要一次跑多个 demo 时
    # 后一片会静默覆盖前一片。文件名用「原型 + FEN 短哈希」：原型便于人工
    # 辨认，哈希保证同原型多局面不撞名且可追溯到具体输入。
    out_file = ""
    if output_dir:
        import hashlib
        tag = hashlib.md5(input_fen.encode("utf-8")).hexdigest()[:6]
        out_file = os.path.join(output_dir, f"decision_{arch}_{tag}.mp4")
        # 评审 sidecar（阶段 9）：把**产出这一片时**的 storyboard 与解说文本
        # 落盘，供评审脚本判程序侧四维（事实正确/轨迹一致/对比区分度/措辞合规）。
        # 不落盘就只能重跑纯函数复算，而复算既费引擎时间、又可能因引擎非确定性
        # 与成片不一致——评审必须针对成片本身，否则判的不是同一个东西。
        try:
            # A2 轨迹一致性（d2 判据）在此处算并落盘。`project()` 只产结构趋势，
            # 不含「目标是否达成」——评审若自己重算就得重跑引擎，且可能与成片
            # 不一致。用决策点走子方锚定视角（`board.turn`）：线末的 turn 往往
            # 是对手，不锚定会把「我方/对方」判反。
            # 同上：按计划名取线与目标，不按位置（routes 是 outcomes 的
            # 可行子集，位置不可靠）。
            mover = board.turn
            traj = {}
            for name, outcome in by_name.items():
                traj[name] = goal_trajectory(
                    board, outcome.line_pv,
                    outcome.plan.get("structural_goal", {}) or {}, mover)

            side = {
                "fen": input_fen,
                "archetype": arch,
                "archetype_cn": kb[arch]["cn"],
                "video": out_file,
                "routes": [
                    {"name": r.get("name"),
                     "mechanism": r.get("mechanism"),
                     "unique_facts": r.get("unique_facts", []),
                     "trend": _jsonable(r.get("trend", {})),
                     "tradeoffs": _jsonable(r.get("tradeoffs", {})),
                     **traj.get(r.get("name"), {})}
                    for r in sb.get("routes", [])
                ],
                "comparison_axes": sb.get("comparison_axes", {}),
                # divergences（P8 分歧深度）：storyboard 顶层字段，纯 JSON-able
                # （pair/depth/paired）。PLAN-011 阶段 0 加——质量门槛脚本需从
                # sidecar 统计「真比较式率」（routes≥2 & axis_type=1 & 任一 pair
                # paired=True），此前 divergences 只在 storyboard 内存、没复制
                # 到 sidecar，门槛拿不到。产品行为零变化（字段已计算好，只是复制）。
                "divergences": sb.get("divergences", []),
                "opening_text": getattr(commentary, "opening", ""),
                "summary_text": getattr(commentary, "summary", ""),
                "segments": [
                    {"id": getattr(s, "id", None),
                     "voiceover": getattr(s, "voiceover", "")}
                    for s in getattr(commentary, "segments", [])
                ],
                "seg_durations": [
                    {"idx": i, "text_len": len(sg.text or ""),
                     "duration_s": round(getattr(sg, "duration_s", 0.0), 2),
                     "speech_s": round(getattr(sg, "speech_duration_s", 0.0), 2),
                     "n_moves": len(getattr(sg, "moves", []) or [])}
                    for i, sg in enumerate(segments)
                ],
            }
            # 先完整序列化成字符串、再落盘（08.04 修）。此前直接
            # `json.dump(side, f)` 是流式写入：中途遇到不可序列化对象会抛异常，
            # 而已写出的半截 JSON 留在磁盘上——外层 except 只 warn「不影响成片」，
            # 于是成片正常、sidecar 静默残缺，评审脚本再 `json.load` 就炸在
            # `JSONDecodeError`（实测 618 字节截断文件）。序列化成功才开文件，
            # 失败时磁盘上不留任何东西，评审脚本按「sidecar 缺失」正常处理。
            blob = json.dumps(side, ensure_ascii=False, indent=1)
            with open(os.path.splitext(out_file)[0] + "_review.json",
                      "w", encoding="utf-8") as f:
                f.write(blob)
        except Exception as e:  # noqa: BLE001
            # 带上异常类型名（08.04 修）。此前只打 `{e}`，而本处真实发生过的是
            # `goal_trajectory` 漏 import 导致的 NameError——消息体只有一个函数名，
            # 读日志时极易当成「数据里缺字段」这类无害情况忽略过去，实际是编码错误。
            # NameError/AttributeError/TypeError 一律属于「代码写错了」，必须能从
            # 日志一眼区分于「这一局面的数据凑不出 sidecar」。
            Logger.warn(f"评审 sidecar 落盘失败（不影响成片）: "
                        f"{type(e).__name__}: {e}")
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
            output_path=out_file,
        )
        Logger.success(f"Decision 视频已生成: {output_path}")
        return output_path
    finally:
        cleanup_artifacts(frame_paths, srt_path, segments)
        # 清全部实际用到的序列目录（08.04 改：原先写死 A/B 两个，
        # 计划数 >2 时 seq_c 及之后的帧会留在磁盘上，下一次跑同原型时
        # 被 `_render_sequence` 的 rmtree 兜住不至于混帧，但残留仍占空间）
        for d in seq_dirs:
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)


def run_decision(input_fen: str, provenance: Optional[str] = None) -> None:
    """输出纯解说文本，对应 `--decision --text`（对齐 puzzle 的 run_puzzle）。

    与视频路径共用 `_decision_core`（PLAN-011 阶段 3）——文本模式只跑到
    「算完内容」就打印解说，不进 TTS/渲染/合成，省去视频链路的耗时；且因
    共用 core，SPEC §8 全部放弃闸（无原型/不在产品池/无可行计划/解说 aborted）
    与视频路径完全一致，不存在「文本能出、视频放弃」这类口径分裂。
    core 返回 None（任一放弃点）时静默返回，放弃原因已由 core 内 warn/error。
    """
    core = _decision_core(input_fen, provenance)
    if core is None:
        return
    commentary = core["commentary"]
    if commentary.opening:
        print(commentary.opening + "\n")
    for seg in getattr(commentary, "segments", []):
        print(f"[{getattr(seg, 'id', '?')}] "
              f"{getattr(seg, 'voiceover', '')}\n")
    if commentary.summary:
        print(commentary.summary)


def run_decision_video(input_fen: str, voice_prompt: str = "",
                       provenance: Optional[str] = None) -> str:
    """生成决策讲解视频，对应 `--decision`（默认）。返回视频路径，放弃返回 ""。

    薄封装 `_run_decision_pipeline`（视频后半段）——统一默认输出到 output/，
    与两条老管线的产出目录对齐；空 `voice_prompt` 归一为 None 交由底层取默认。
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = os.path.join(root, "output")
    os.makedirs(output_dir, exist_ok=True)
    return _run_decision_pipeline(input_fen, provenance=provenance,
                                  output_dir=output_dir,
                                  voice_prompt=voice_prompt or None)


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
