"""带颜色的终端日志工具。从 common.py 拆出。"""
from colorama import init, Fore, Style
from datetime import datetime

init()


class Logger:
    @staticmethod
    def _ts():
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def info(msg):
        print(f"{Fore.BLUE}[{Logger._ts()}] {msg}{Style.RESET_ALL}")

    @staticmethod
    def success(msg):
        print(f"{Fore.GREEN}[{Logger._ts()}] {msg}{Style.RESET_ALL}")

    @staticmethod
    def warn(msg):
        print(f"{Fore.YELLOW}[{Logger._ts()}] {msg}{Style.RESET_ALL}")

    @staticmethod
    def error(msg):
        print(f"{Fore.RED}[{Logger._ts()}] {msg}{Style.RESET_ALL}")
