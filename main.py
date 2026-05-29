from src.common import Logger
from src.pipeline import run, run_video
import sys


def main():
    """默认生成视频，--text 仅输出解说文本"""
    text_mode = "--text" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

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
