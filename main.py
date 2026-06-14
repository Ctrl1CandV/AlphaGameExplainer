from src.pipeline import run, run_video, run_puzzle, run_puzzle_video
from src.common import Logger
import sys

def main():
    """
    默认生成视频，--text仅输出解说文本
    --puzzle切换到Puzzle战术讲解链路
    """
    text_mode = "--text" in sys.argv
    puzzle_mode = "--puzzle" in sys.argv
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

    # 残局讲解
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
            run_video(input_text)
    except Exception as e:
        Logger.error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()