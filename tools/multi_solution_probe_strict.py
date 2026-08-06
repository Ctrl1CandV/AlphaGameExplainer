"""严口径复验：攻击 PLAN-010 的 70% 激活率。

宽口径（原 probe）用 3x3 区域作签名键，"左上"vs"中上"可能只差一格（c8 vs d8），
观众感知不到差异却被计为 2 簇。本脚本用两档更严的口径复验：

严口径 A（象限）：杀王格按 4 象限（左上/右上/左下/右下）归并，去掉"中"带。
严口径 B（象限 + 感知阈值）：在 A 之上，要求两解至少一个轴有可感知差异——
    步数差 >= 4（约一个回合半以上）或 平均容错宽度差 >= 3。
"""
import os, glob, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SYZYGY_PATH", "syzygy")
import chess
from src.solver.tablebase import TablebaseSolver

solver = TablebaseSolver(syzygy_dir="syzygy")
solver.open()


def winning_moves(b):
    out = []
    for mv in b.legal_moves:
        t = b.copy(stack=False)
        t.push(mv)
        if t.is_checkmate():
            out.append((mv, 0))
            continue
        w = solver.probe_wdl(t)
        if w is None or w >= 0:
            continue
        d = solver._syzygy.probe_dtz(t) if solver._syzygy else None
        out.append((mv, abs(d) if d else 999))
    return out


def solve_line(b0, first, cap=80):
    b = b0.copy(stack=False)
    b.push(first)
    n, widths = 1, []
    while not b.is_game_over() and n < cap:
        widths.append(len(winning_moves(b)))
        pair = solver._best_move_syzygy(b)
        if not pair:
            return None
        b.push(pair[0])
        n += 1
        if b.is_game_over():
            break
        widths.append(len(winning_moves(b)))
        pair = solver._best_move_syzygy(b)
        if not pair:
            return None
        b.push(pair[0])
        n += 1
    if not b.is_checkmate():
        return None
    return n, b.king(b.turn), widths


def quadrant(sq):
    """4 象限，无中间带——观众能明确区分的粒度"""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return ("左" if f <= 3 else "右") + ("下" if r <= 3 else "上")


STEP_GAP, WIDTH_GAP = 4, 3.0
act_wide = act_quad = act_percept = 0
total = 0
detail_rows = []

for path in sorted(glob.glob("test_endgames/*.fen")):
    name = os.path.basename(path)
    fen = open(path, encoding="utf-8").read().strip().split("\n")[0]
    try:
        board = chess.Board(fen)
    except Exception:
        continue
    if not solver.is_hit(board):
        continue
    total += 1
    wm = winning_moves(board)
    if not wm:
        continue
    wm.sort(key=lambda x: x[1])
    picks = wm[:4] + wm[len(wm) // 2: len(wm) // 2 + 3] + wm[-3:]
    seen, lines = set(), []
    for mv, _d in picks:
        if mv in seen:
            continue
        seen.add(mv)
        r = solve_line(board, mv)
        if r:
            lines.append(r)
    if not lines:
        continue

    # 宽口径（3x3，复现原 probe）
    def region3(sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        return ("左" if f <= 2 else "右" if f >= 5 else "中",
                "下" if r <= 2 else "上" if r >= 5 else "中")
    if len({region3(k) for _n, k, _w in lines}) >= 2:
        act_wide += 1

    # 严口径 A：象限
    quads = {}
    for n, ksq, widths in lines:
        q = quadrant(ksq)
        avgw = sum(widths) / max(1, len(widths))
        cur = quads.get(q)
        if cur is None or n < cur[0]:
            quads[q] = (n, avgw)
    quad_ok = len(quads) >= 2
    if quad_ok:
        act_quad += 1

    # 严口径 B：象限 + 感知阈值
    percept_ok = False
    note = ""
    if quad_ok:
        vals = list(quads.values())
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                ds = abs(vals[i][0] - vals[j][0])
                dw = abs(vals[i][1] - vals[j][1])
                if ds >= STEP_GAP or dw >= WIDTH_GAP:
                    percept_ok = True
                    note = f"步差{ds} 宽差{dw:.1f}"
                    break
            if percept_ok:
                break
        if not percept_ok:
            vals_s = sorted(quads.items())
            note = "同象限组差异均低于阈值: " + ", ".join(
                f"{k}{v[0]}步宽{v[1]:.1f}" for k, v in vals_s)
    if percept_ok:
        act_percept += 1

    detail_rows.append((name, len(quads), "PASS" if percept_ok else "FAIL", note))

print(f"{'样本':<16}{'象限簇':>6}{'感知闸':>8}  说明")
print("-" * 96)
for r in detail_rows:
    print(f"{r[0]:<16}{r[1]:>6}{r[2]:>8}  {r[3]}")
print("-" * 96)
print(f"表库可解样本 {total}")
print(f"宽口径(3x3区域)      激活 {act_wide:>2}  -> {act_wide/total*100:.0f}%")
print(f"严口径A(4象限)       激活 {act_quad:>2}  -> {act_quad/total*100:.0f}%")
print(f"严口径B(象限+感知阈) 激活 {act_percept:>2}  -> {act_percept/total*100:.0f}%")
solver.close()
