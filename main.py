from src.common import Logger
from src.pipeline import run
import sys

def main():
    # 当输入为pgn文件时
    if len(sys.argv) > 1:
        path = sys.argv[1]
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
        run(input_text)
    except Exception as e:
        Logger.error(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()