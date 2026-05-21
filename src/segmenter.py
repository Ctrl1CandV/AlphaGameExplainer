import re
from typing import List
from src.common import Segment


def segment(commentary: str, move_count: int) -> List[Segment]:
    """
    将解说文本按 [第N步] 标签拆分为段落。
    若 LLM 未按格式输出，则按棋步数均分。
    """
    pattern = r"第\s*(\d+)\s*步[：:\s]*"
    parts = re.split(pattern, commentary)

    if len(parts) > 1:
        result = []
        for i in range(1, len(parts), 2):
            idx = int(parts[i])
            text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            result.append(Segment(move_idx=idx, text=text))
        result.sort(key=lambda s: s.move_idx)
        return result

    total = len(commentary)
    chunk = max(60, total // max(move_count, 1))
    result = []
    for i in range(move_count):
        start = i * chunk
        end = start + chunk if i < move_count - 1 else total
        result.append(Segment(move_idx=i + 1, text=commentary[start:end].strip()))
    return result
