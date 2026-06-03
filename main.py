from src.common import Logger
from src.pipeline import run, run_video, run_puzzle, run_puzzle_video, run_puzzle_csv
import sys


def main():
    """默认生成视频，--text 仅输出解说文本。
    --puzzle 切换到 Puzzle 战术讲解链路。
    --csv 配合 --puzzle 批量处理 CSV 文件。
    """
    text_mode = "--text" in sys.argv
    puzzle_mode = "--puzzle" in sys.argv
    csv_mode = "--csv" in sys.argv

    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    # 提取 --limit N
    limit = 0
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--limit" and i + 1 < len(sys.argv):
            try:
                limit = int(sys.argv[i + 1])
            except ValueError:
                pass

    # CSV 批量模式
    if puzzle_mode and csv_mode:
        if args:
            path = args[0]
        else:
            Logger.error("请指定 CSV 文件路径")
            sys.exit(1)
        run_puzzle_csv(path, limit=limit, text_only=text_mode)
        return

    # Puzzle 单题模式
    if puzzle_mode:
        if args:
            path = args[0]
            with open(path, "r", encoding="utf-8") as f:
                input_text = f.read()
        else:
            Logger.info("请输入Puzzle数据(JSON/TXT/CSV行, 输入END结束):")
            lines = []
            while True:
                line = input()
                if line.strip().upper() == "END":
                    break
                lines.append(line)
            input_text = "\n".join(lines)

        try:
            if text_mode:
                run_puzzle(input_text)
            else:
                output = run_puzzle_video(input_text)
                print(f"\n视频已生成: {output}")
        except Exception as e:
            Logger.error(str(e))
            sys.exit(1)
        return

    # 原链路（残局讲解）
    if args:
        path = args[0]
        with open(path, "r", encoding="utf-8") as f:
            input_text = f.read()
    else:
        Logger.info("请输入PGN或FEN内容(输入END结束):")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        input_text = "\n".join(lines)

    try:
        if text_mode:
            run(input_text)
        else:
            output = run_video(input_text)
            print(f"\n视频已生成: {output}")
    except Exception as e:
        Logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
