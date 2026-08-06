"""一次性探针：全量残局多解激活率 + 容错宽度可测性。用完即删。"""
import os, sys, glob, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess
from dotenv import load_dotenv
load_dotenv()
from src.solver.tablebase import TablebaseSolver

SYZ = os.getenv("SYZYGY_PATH", "syzygy")
solver = TablebaseSolver(syzygy_dir=SYZ)
solver.open()


def winning_moves(board):
    """保持必胜的全部合法走法 -> [(move, child_dtz)]"""
    out = []
    for m in board.legal_moves:
        t = board.copy(stack=False)
        t.push(m)
        if t.is_checkmate():
            out.append((m, 0))
            continue
        w = solver.probe_wdl(t)
        if w is None or w >= 0:      # 对手不再必败 -> 丢胜
            continue
        try:
            d = solver._syzygy.probe_dtz(t)
        except Exception:
            d = None
        out.append((m, abs(d) if d is not None else 999))
    return out


def solve_line(board, first, cap=60):
    """从 first 起用表库最优策略走到终局，返回 (sans, mate_square, widths)"""
    b = board.copy(stack=False)
    strong = b.turn
    sans, widths = [], []
    if b.turn == strong:
        widths.append(len(winning_moves(b)))
    sans.append(b.san(first)); b.push(first)
    while not b.is_game_over() and len(sans) < cap:
        if b.turn == strong:
            widths.append(len(winning_moves(b)))
        pair = solver._best_move_syzygy(b)
        if not pair:
            return None
        mv = pair[0]
        sans.append(b.san(mv)); b.push(mv)
    if not b.is_checkmate():
        return None
    ksq = b.king(b.turn)
    return sans, ksq, widths


def region(sq):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return ("左" if f <= 2 else "右" if f >= 5 else "中",
            "下" if r <= 2 else "上" if r >= 5 else "中")


rows = []
for path in sorted(glob.glob("test_endgames/*.fen")):
    name = os.path.basename(path)
    fen = open(path, encoding="utf-8").read().strip().split("\n")[0]
    try:
        board = chess.Board(fen)
    except Exception:
        continue
    if not solver.is_hit(board):
        rows.append((name, "无表库", 0, 0, "", ""))
        continue
    wm = winning_moves(board)
    if not wm:
        rows.append((name, "无必胜", 0, 0, "", ""))
        continue
    # 按 dtz 分桶采样，最多 10 条根着，覆盖快/中/慢
    wm.sort(key=lambda x: x[1])
    picks = wm[:4] + wm[len(wm)//2: len(wm)//2+3] + wm[-3:]
    seen, lines = set(), []
    for mv, _d in picks:
        if mv in seen:
            continue
        seen.add(mv)
        r = solve_line(board, mv)
        if r:
            lines.append(r)
    if not lines:
        rows.append((name, "解线失败", len(wm), 0, "", ""))
        continue
    # 解签名 = (杀王区域, 步数奇偶档)
    sigs = {}
    for sans, ksq, widths in lines:
        key = region(ksq)
        avgw = sum(widths) / max(1, len(widths))
        cur = sigs.get(key)
        if cur is None or len(sans) < cur[0]:
            sigs[key] = (len(sans), round(avgw, 1), round(min(widths), 1) if widths else 0)
    detail = " | ".join(
        f"{k[0]}{k[1]}角 {v[0]}步 宽{v[1]}(最窄{v[2]})" for k, v in sorted(sigs.items())
    )
    lens = [len(s) for s, _, _ in lines]
    ws = [round(sum(w) / max(1, len(w)), 1) for _, _, w in lines]
    rows.append((name, "OK", len(wm), len(sigs), detail,
                 f"步{min(lens)}-{max(lens)} 宽{min(ws)}-{max(ws)}"))

solver.close()

print(f"{'样本':<16}{'状态':<8}{'必胜着':>5}{'签名簇':>6}  差异明细")
print("-" * 130)
act = 0
for name, st, nw, ns, detail, span in rows:
    if st == "OK" and ns >= 2:
        act += 1
    print(f"{name:<16}{st:<8}{nw:>5}{ns:>6}  {detail}")
tot = len(rows)
ok = sum(1 for r in rows if r[1] == "OK")
print("-" * 130)
print(f"总样本 {tot} / 表库可解 {ok} / 签名簇>=2（可激活）{act}  ->  激活率 {act/max(1,tot)*100:.0f}%")
print("\n=== 容错宽度跨度（同一样本内不同解的宽窄差）===")
for name, st, nw, ns, detail, span in rows:
    if st == "OK":
        print(f"  {name:<16}{span}")
