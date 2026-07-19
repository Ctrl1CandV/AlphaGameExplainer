"""GBNF 语法定义（llama.cpp token 级语法约束）。

残局 JSON 语法、Puzzle 收紧中文语法、总结词/开场白语法、单段修复语法。
从 commentator.py 提取，纯字符串模板，零外部依赖。

输出模式标签（API 后端用）：
- build_chunk_grammar / build_puzzle_chunk_grammar / SEGMENT_GRAMMAR 标 JSON
  （JSON 外壳，API 走 response_format=json_object）
- PUZZLE_PLAIN_CN_GRAMMAR 标 PURE_CN（无 JSON 包裹，纯中文自由文本）
- 调用方传 None 表示普通文本
GrammarConstraint 是 str 子类，对 llama.cpp 与既有 truthy 判断完全兼容；
未知非空 grammar 在 API 模式 fail closed（见 PLAN-002 行为规约 §11）。
"""
from enum import Enum


class OutputMode(Enum):
    """API 后端的输出模式分类，由 grammar 标签决定。"""
    JSON = "json"          # JSON 结构，走 response_format=json_object
    PURE_CN = "pure_cn"    # 纯中文自由文本，追加中文格式约束


class GrammarConstraint(str):
    """带输出模式标签的 grammar 字符串。

    str 子类：与 llama_cpp.LlamaGrammar.from_string、if grammar 判断、
    字符串拼接完全兼容；额外携带 .output_mode 供 API 后端决定 response_format。
    """

    def __new__(cls, value: str, output_mode: "OutputMode | str | None" = None):
        # 允许传 str 字面量（默认 JSON 兼容旧行为）或 OutputMode
        if isinstance(output_mode, str) and not isinstance(output_mode, OutputMode):
            try:
                output_mode = OutputMode(output_mode)
            except ValueError:
                output_mode = None
        obj = super().__new__(cls, value)
        obj.output_mode = output_mode
        return obj


def _tag(value: str, mode: OutputMode) -> GrammarConstraint:
    """把纯字符串 grammar 标记为指定输出模式。空串保持空串（falsy 不变）。"""
    if not value:
        return GrammarConstraint(value, mode)
    return GrammarConstraint(value, mode)


def build_chunk_grammar(n_segments: int) -> GrammarConstraint:
    """残局 chunk JSON 语法。"""
    if n_segments <= 0:
        return _tag("", OutputMode.JSON)
    seg_repeat = "segment" + "".join(' ws "," ws segment' for _ in range(n_segments - 1))
    return _tag(
        'root ::= "{" ws "\\"segments\\"" ws ":" ws "[" ws ' + seg_repeat + ' ws "]" ws "}"\n'
        'segment ::= "{" ws "\\"id\\"" ws ":" ws integer ws "," ws '
        '"\\"sub_endgame\\"" ws ":" ws string ws "," ws '
        '"\\"voiceover\\"" ws ":" ws string ws "," ws '
        '"\\"pacing\\"" ws ":" ws pacing ws "}"\n'
        'pacing ::= "\\"slow\\"" | "\\"normal\\"" | "\\"fast\\"" | "\\"pause_before\\"" | "\\"pause_after\\""\n'
        'integer ::= [0-9]+\n'
        'string ::= "\\"" [^"\\\\x00-\\x1F]* "\\""\n'
        'ws ::= [ \\t\\n]*',
        OutputMode.JSON,
    )


def build_puzzle_chunk_grammar(n_segments: int) -> GrammarConstraint:
    """Puzzle 专用收紧语法：voiceover 在采样阶段就只允许中文字符与中文标点。

    从根本上杜绝 Markdown 符号、英文字母、阿拉伯数字混入口播稿。
    与 build_chunk_grammar 的唯一区别：voiceover 用 cnstring 取代通用 string；
    sub_endgame 固定为空串。

    标记为 JSON（外层是 segments JSON），API 模式走 json_object；
    cnstring 的 token 级中文锁在 API 模式下丢失，由 validator+retry+后处理兜底。
    """
    if n_segments <= 0:
        return _tag("", OutputMode.JSON)
    seg_repeat = "segment" + "".join(' ws "," ws segment' for _ in range(n_segments - 1))
    cn_punct = "，。、；：？！…—·「」『』（）《》〈〉“”‘’　"
    return _tag(
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
        'ws ::= [ \\t\\n]*',
        OutputMode.JSON,
    )


# 纯中文自由文本语法（无 JSON 包裹）：供润色器等单串生成使用。
PUZZLE_PLAIN_CN_GRAMMAR = _tag(
    'root ::= think? cnchar+\n'
    'think ::= "<think>" thinkchar* "</think>" ws\n'
    'thinkchar ::= [^<]\n'
    'cnchar ::= [\\u4e00-\\u9fff，。、；：？！…—·「」『』（）《》〈〉“”‘’]\n'
    'ws ::= [ \\t\\n]*',
    OutputMode.PURE_CN,
)

# 单段修复语法
SEGMENT_GRAMMAR = _tag(
    'root ::= "{" ws "\\"id\\"" ws ":" ws integer ws "," ws '
    '"\\"sub_endgame\\"" ws ":" ws string ws "," ws '
    '"\\"voiceover\\"" ws ":" ws string ws "," ws '
    '"\\"pacing\\"" ws ":" ws pacing ws "}"\n'
    'pacing ::= "\\"slow\\"" | "\\"normal\\"" | "\\"fast\\"" | "\\"pause_before\\"" | "\\"pause_after\\""\n'
    'integer ::= [0-9]+\n'
    'string ::= "\\"" [^"\\\\x00-\\x1F]* "\\""\n'
    'ws ::= [ \\t\\n]*',
    OutputMode.JSON,
)

# 总结词专用语法：token 级只允许中文 + 中文标点。2-3 句。
# 注意：生产代码中 endgame summary/opening 实际不传 grammar（见 endgame_commentary.py
# 注释，_SUMMARY_GRAMMAR 会阻止 <think>），这里保留定义但已无活跃调用方。
SUMMARY_GRAMMAR = _tag(
    'root ::= sentence sentence sentence?\n'
    'sentence ::= cjk (sep cjk)* end\n'
    'cjk ::= han+\n'
    'han ::= [\\u4e00-\\u9fff]\n'
    'sep ::= "，" | "、"\n'
    'end ::= "。" | "！" | "？"',
    OutputMode.PURE_CN,
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
