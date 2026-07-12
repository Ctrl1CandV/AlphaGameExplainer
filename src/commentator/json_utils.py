"""JSON 解析/修复工具。

从 commentator.py 提取。处理 LLM 输出的 JSON 解析、常见格式修复。
零外部依赖（仅 json + re 标准库）。
"""
import json
import re
from typing import Optional


def extract_json_text(text: str) -> str:
    t = text.strip()
    if not t:
        return ""
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    brace_start = t.find("{")
    brace_end = t.rfind("}")
    if brace_start == -1 or brace_end <= brace_start:
        return ""
    return t[brace_start:brace_end + 1]


def repair_common_json_issues(text: str) -> str:
    fixed = text.strip()
    fixed = re.sub(r"^```(?:json)?\s*", "", fixed)
    fixed = re.sub(r"\s*```$", "", fixed)
    fixed = fixed.replace("\u201c", "\"").replace("\u201d", "\"")
    fixed = fixed.replace("\u2018", "'").replace("\u2019", "'")
    fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)
    return fixed


# 哨兵对象，表示 JSON 解析彻底失败
INVALID_JSON_SENTINEL = object()


def parse_storyboard_json(text: str) -> dict:
    candidates = []
    extracted = extract_json_text(text)
    if extracted:
        candidates.append(extracted)
        candidates.append(repair_common_json_issues(extracted))
    repaired_full = repair_common_json_issues(text)
    if repaired_full and repaired_full not in candidates:
        candidates.append(repaired_full)

    for json_text in candidates:
        if not json_text:
            continue
        try:
            data = json.loads(json_text)
            if isinstance(data, dict) and "segments" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            continue
    return INVALID_JSON_SENTINEL


def parse_single_segment(raw_text: str) -> Optional[dict]:
    t = raw_text.strip()
    brace_start = t.find("{")
    brace_end = t.rfind("}")
    if brace_start == -1 or brace_end <= brace_start:
        return None
    try:
        obj = json.loads(t[brace_start:brace_end + 1])
        if isinstance(obj, dict) and "id" in obj:
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return None
