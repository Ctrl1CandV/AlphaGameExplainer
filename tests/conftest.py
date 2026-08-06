"""pytest 全局 fixture/路径设置。

项目根目录已经是 pytest.ini 所在目录（rootdir），src/ 包本身可直接 import；
这里只做防御性补充，保证从任意工作目录跑 `pytest` 都能找到项目根。
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
