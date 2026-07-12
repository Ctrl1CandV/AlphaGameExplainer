"""GBNF 语法定义（llama.cpp token 级语法约束）。

残局 JSON 语法、Puzzle 收紧中文语法、总结词/开场白语法、单段修复语法。
从 commentator.py 提取，纯字符串模板，零外部依赖。
"""


def build_chunk_grammar(n_segments: int) -> str:
    """残局 chunk JSON 语法。"""
    if n_segments <= 0:
        return ""
    seg_repeat = "segment" + "".join(' ws "," ws segment' for _ in range(n_segments - 1))
    return (
        'root ::= "{" ws "\\"segments\\"" ws ":" ws "[" ws ' + seg_repeat + ' ws "]" ws "}"\n'
        'segment ::= "{" ws "\\"id\\"" ws ":" ws integer ws "," ws '
        '"\\"sub_endgame\\"" ws ":" ws string ws "," ws '
        '"\\"voiceover\\"" ws ":" ws string ws "," ws '
        '"\\"pacing\\"" ws ":" ws pacing ws "}"\n'
        'pacing ::= "\\"slow\\"" | "\\"normal\\"" | "\\"fast\\"" | "\\"pause_before\\"" | "\\"pause_after\\""\n'
        'integer ::= [0-9]+\n'
        'string ::= "\\"" [^"\\\\x00-\\x1F]* "\\""\n'
        'ws ::= [ \\t\\n]*'
    )


def build_puzzle_chunk_grammar(n_segments: int) -> str:
    """Puzzle 专用收紧语法：voiceover 在采样阶段就只允许中文字符与中文标点。

    从根本上杜绝 Markdown 符号、英文字母、阿拉伯数字混入口播稿。
    与 build_chunk_grammar 的唯一区别：voiceover 用 cnstring 取代通用 string；
    sub_endgame 固定为空串。
    """
    if n_segments <= 0:
        return ""
    seg_repeat = "segment" + "".join(' ws "," ws segment' for _ in range(n_segments - 1))
    cn_punct = "，。、；：？！…—·「」『』（）《》〈〉“”‘’　"
    return (
        'root ::= ws think? "{" ws "\\"segments\\"" ws ":" ws "[" ws ' + seg_repeat + ' ws "]" ws "}"\n'
        'think ::= "<think>" thinkchar* "</think>" ws\n'
        'thinkchar ::= [^<]\n'
        'segment ::= "{" ws "\\"id\\"" ws ":" ws integer ws "," ws '
        '"\\"sub_endgame\\"" ws ":" ws "\\"\\"" ws "," ws '
        '"\\"voiceover\\"" ws ":" ws cnstring ws "," ws '
        '"\\"pacing\\"" ws ":" ws pacing ws "}"\n'
        'pacing ::= "\\"slow\\"" | "\\"normal\\"" | "\\"fast\\"" | "\\"pause_before\\"" | "\\"pause_after\\""\n'
        'integer ::= [0-9]+\n'
        'cnstring ::= "\\"" cnchar* "\\""\n'
        'cnchar ::= [\\u4e00-\\u9fff' + cn_punct + ']\n'
        'ws ::= [ \\t\\n]*'
    )


# 纯中文自由文本语法（无 JSON 包裹）：供润色器等单串生成使用。
PUZZLE_PLAIN_CN_GRAMMAR = (
    'root ::= think? cnchar+\n'
    'think ::= "<think>" thinkchar* "</think>" ws\n'
    'thinkchar ::= [^<]\n'
    'cnchar ::= [\\u4e00-\\u9fff，。、；：？！…—·「」『』（）《》〈〉“”‘’]\n'
    'ws ::= [ \\t\\n]*'
)

# 单段修复语法
SEGMENT_GRAMMAR = (
    'root ::= "{" ws "\\"id\\"" ws ":" ws integer ws "," ws '
    '"\\"sub_endgame\\"" ws ":" ws string ws "," ws '
    '"\\"voiceover\\"" ws ":" ws string ws "," ws '
    '"\\"pacing\\"" ws ":" ws pacing ws "}"\n'
    'pacing ::= "\\"slow\\"" | "\\"normal\\"" | "\\"fast\\"" | "\\"pause_before\\"" | "\\"pause_after\\""\n'
    'integer ::= [0-9]+\n'
    'string ::= "\\"" [^"\\\\x00-\\x1F]* "\\""\n'
    'ws ::= [ \\t\\n]*'
)

# 总结词专用语法：token 级只允许中文 + 中文标点。2-3 句。
SUMMARY_GRAMMAR = (
    'root ::= sentence sentence sentence?\n'
    'sentence ::= cjk (sep cjk)* end\n'
    'cjk ::= han+\n'
    'han ::= [\\u4e00-\\u9fff]\n'
    'sep ::= "，" | "、"\n'
    'end ::= "。" | "！" | "？"'
)


def build_retry_prompt(prompt: str, error_msg: str, attempt: int = 1) -> str:
    """构建重试 prompt（残局和 Puzzle 共用）。"""
    if "JSON" in error_msg or "不是合法" in error_msg:
        hint = (
            "请只输出一个合法JSON对象，不要加 ```json 代码块、"
            "markdown标记、或任何解释性文字。输出体必须以 { 开头、以 } 结尾。"
        )
    elif "宣称" in error_msg:
        if "将杀" in error_msg:
            hint = (
                f"上一轮输出包含不准确的终结性描述。{error_msg}。"
                "请检查每个节点的走后真值：只有明确写「已形成将杀」的节点才能写将杀/绝杀。"
                "其他节点请改用「压缩空间」「封住逃格」「确立胜势」等描述。只输出合法JSON。"
            )
        else:
            hint = (
                f"上一轮输出包含不准确的战术描述。{error_msg}。"
                "请根据节点信息中的实际走法和状态来调整用词。只输出合法JSON。"
            )
    elif "过短" in error_msg:
        hint = "请补足解说信息量：普通节点至少55字，summary_only节点也要用一句完整地交代机动目的。"
    else:
        hint = "请修改输出以通过校验，只输出合法JSON对象；不要输出visuals字段。"

    return prompt + f"\n\n上一轮输出校验失败: {error_msg}。{hint}"
