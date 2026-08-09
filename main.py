from src.pipeline import (
    run, run_video, run_puzzle, run_puzzle_video,
    run_decision, run_decision_video,
)
from src.common import Logger
import sys
import os

def main():
    """
    默认生成视频，--text仅输出解说文本
    --puzzle切换到Puzzle战术讲解链路
    --decision切换到多战略意图讲解链路（输入单个 FEN）
    残局模式支持传入文件路径（.fen/.pgn/.txt），或不传参进入交互式输入
    """
    text_mode = "--text" in sys.argv
    puzzle_mode = "--puzzle" in sys.argv
    decision_mode = "--decision" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    # Puzzle单题模式
    if puzzle_mode:
        if not args:
            Logger.error("请指定Puzzle输入文件路径")
            sys.exit(1)
        path = args[0]
        with open(path, "r", encoding="utf-8") as f:
            input_text = f.read()

        try:
            if text_mode:
                run_puzzle(input_text)
            else:
                run_puzzle_video(input_text)
        except Exception as e:
            Logger.error(str(e))
            sys.exit(1)
        return

    # 残局讲解：支持文件路径或交互式输入
    if args:
        # 文件模式：从文件读取 FEN/PGN
        path = args[0]
        if not os.path.isfile(path):
            Logger.error(f"文件不存在: {path}")
            sys.exit(1)
        try:
            with open(path, "r", encoding="utf-8") as f:
                input_text = f.read().strip()
        except UnicodeDecodeError:
            Logger.error(f"文件编码错误，请确保是 UTF-8 文本文件: {path}")
            sys.exit(1)
        except OSError as e:
            Logger.error(f"读取文件失败: {e}")
            sys.exit(1)

        if not input_text:
            Logger.error(f"文件内容为空: {path}")
            sys.exit(1)

        Logger.info(f"已从文件读取: {path}")
    else:
        # 交互式输入
        Logger.info("请输入PGN或FEN内容(输入END结束，或直接传入.fen文件路径):")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        input_text = "\n".join(lines)

    try:
        if decision_mode:
            # 多战略意图讲解：输入是单个 FEN（决策点局面）。文本模式共用
            # `_decision_core` 只算内容不出视频；两路径的 SPEC §8 放弃闸一致。
            if text_mode:
                run_decision(input_text)
            else:
                run_decision_video(input_text)
        elif text_mode:
            run(input_text)
        else:
            run_video(input_text)
    except Exception as e:
        Logger.error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()