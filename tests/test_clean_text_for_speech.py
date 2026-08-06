"""tts_engine._clean_text_for_speech 单测（PLAN-010 阶段0 步骤3，adversary REV-001 补）。

零引擎依赖的纯文本替换：升变/坐标/纵线转写、全角分号冒号归一、破折号/问号/
全角括号处理、重复"该格该格"收敛、无匹配原样返回。全部断言值经函数实测确认
（不凭正则规则推断输出——见 PLAN-010 阶段0 对 safe_decision_seed_text 的教训）。
"""
from src.media.tts_engine import _clean_text_for_speech


def test_promotion_notation_to_chinese():
    """升变记号（字母数字=棋子字母[+#]?）→「升变」。"""
    assert _clean_text_for_speech("白后走到 g1=Q+ 完成升变") == "白后走到 升变 完成升变"


def test_bare_coordinate_to_placeholder():
    """孤立坐标 a1-h8 → 「该格」。"""
    assert _clean_text_for_speech("象走到 h7 格") == "象走到 该格 格"


def test_file_line_reference_to_zhe_yi_xian():
    """纵线表述「a线」→「这一线」。"""
    assert _clean_text_for_speech("车沿 a线 推进") == "车沿 这一线 推进"


def test_dash_variants_to_comma():
    """破折号（—/–/―，含全角）→ 逗号，语义等价停顿。"""
    assert _clean_text_for_speech("这条路稳——但空间受限") == "这条路稳，但空间受限"


def test_question_mark_to_period():
    """问号（中英文）→ 句号，ChatTTS 靠语调读疑问语气而非字符。"""
    assert _clean_text_for_speech("该先动哪一翼？") == "该先动哪一翼。"


def test_semicolon_and_colon_normalize_to_comma():
    """全角/半角分号、冒号 → 逗号（ChatTTS 词表不含，实测确认）。"""
    assert _clean_text_for_speech("甲…；乙…") == "甲…，乙…"
    assert _clean_text_for_speech("代价是：兵形松动") == "代价是，兵形松动"


def test_parentheses_become_commas_around_content():
    """全角/半角括号本体去掉、内容保留并前后补逗号（括注要念，不能删）。"""
    result = _clean_text_for_speech("走 h7 与 g5 （保护后翼）之后")
    assert result == "走 该格 与 该格 ，保护后翼，之后"
    assert "（" not in result and "）" not in result


def test_duplicate_placeholder_collapses_to_zhe_xie_ge():
    """连续多个坐标替换后产生的「该格该格」收敛为「这些格」。"""
    assert _clean_text_for_speech("兵从h7h7") == "兵从这些格"
    assert _clean_text_for_speech("马从h7h7h7") == "马从这些格"


def test_two_placeholder_with_and_collapses_to_liang_ge():
    """「该格和该格」收敛为「两个关键格」。"""
    assert _clean_text_for_speech("象控制h7和h7") == "象控制两个关键格"


def test_no_special_chars_passthrough():
    """无匹配内容原样返回（不误伤纯中文文本，对齐老管线零回归要求）。"""
    text = "这条路线更稳健，但空间受限，需要长期忍耐。"
    assert _clean_text_for_speech(text) == text


def test_empty_string_returns_empty():
    assert _clean_text_for_speech("") == ""


def test_old_pipeline_typical_texts_unaffected():
    """老管线（endgame/puzzle）典型文本零残留、语义不变——回归口径对齐 PLAN-009
    实施记录附录B「老管线 5 条典型文本零残留」。这些文本本身不含坐标/特殊
    标点，函数应原样返回。"""
    typical_texts = [
        "白方用车将黑王逼到边线，形成必胜的将杀网。",
        "黑方唯一的应将方式是把王移向角落。",
        "这一步是全局最关键的一手，直接决定了残局的走向。",
    ]
    for t in typical_texts:
        assert _clean_text_for_speech(t) == t
