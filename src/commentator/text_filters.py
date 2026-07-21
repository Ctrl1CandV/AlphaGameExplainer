"""文本清洗与过滤工具。

从 commentator.py 提取。汇总残局链路与 puzzle 链路共用的所有文本处理：
反套话、去坐标、跨段去重、CJK 白名单清洗、括号展开、知识库种子规范化。
零外部依赖（仅 re 标准库）。
"""
import re


def strip_thinking(text: str) -> str:
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    return text.strip()


# 反套话替换表：从实际 KBNvK 样本统计出的高频空洞修辞。
# 原则：只清纯修辞，不动有信息量的词。多数直接删除（删掉不影响句意，
# 因为它们本就不承载事实），少数替换成中性词以保句子通顺。
# 删除类用空串，会在 _auto_fix_voiceover 末尾的标点收敛里清掉留下的多余标点。
_CLICHE_PATTERNS = [
    # (正则, 替换) —— 顺序敏感：先长后短
    # 「看似平淡，实则…」家族：AI 最爱的伪悬念开头，不承载任何棋理事实。
    # 删掉前半截后，后半截的实质内容仍能独立成句。
    (r"看似(?:平淡无奇|平平无奇|平淡|不起眼|普通|简单)[，,]?(?:实则|却|其实)?", ""),
    (r"别看这一步[^，。]{0,6}[，,]", ""),
    (r"如同?利剑出鞘", ""),
    (r"如洪水般(不可阻挡)?", "持续"),
    (r"天罗地网", "严密的控制"),
    (r"天衣无缝", "完整"),
    (r"密不透风", "严密"),
    (r"无形的牢笼", "包围"),
    (r"胜利的天平(开始|彻底)?(倾斜)?", "优势"),
    (r"(已如)?囊中之物", "胜势已成"),
    (r"不可阻挡", ""),
    (r"暗藏杀机", ""),
    (r"耐心的围猎", "稳步驱赶"),
    (r"致命一击", "最后一击"),
    # 形容词类修饰：删去后句子仍通顺
    (r"精妙地?", ""),
    (r"精湛地?", ""),
    (r"精准地?", ""),
    (r"精确地?", ""),
    (r"精心(计算|策划)?地?", ""),
    (r"深思熟虑", ""),
    (r"完美(的|地)?", ""),
    (r"愈发默契", "更协调"),
    (r"(配合|协调)(愈发|越来越)默契", "配合更协调"),
    (r"默契(的)?配合", "配合"),
    (r"步步为营", "稳步推进"),
    # 纯过渡凑字尾巴：「为后续/最终…做准备/奠定基础/创造条件/铺平道路」。
    # 这类句尾不承载任何棋理事实，是 AI 最爱的空洞承诺，每次出现都删。
    (r"[，,]?\s*为(?:后续|接下来|下一步|最终|后面|最后)(?:的)?[^，。、！]{0,16}"
     r"(?:做准备|做好准备|奠定[了]?(?:坚实)?基础|创造[了]?[^，。]{0,8}条件|铺平[了]?道路|埋下伏笔)", ""),
    # PLAN-004 阶段 A 实测追加（phase4.2 正文高频空泛结论句）：
    # 「稳获子力或位置优势」「转化为胜势」「力量对比」「大局已定」——模型爱用的无落点结论。
    # 用删除类（空串）而非替换：避免否定/条件语境（"不能把优势转化为胜势"）下替换串改变原意；
    # 删除后留下的多余标点由 _auto_fix_voiceover 末尾的标点收敛清理。
    (r"稳获(?:子力或位置优势|子力优势|子力)", ""),
    (r"(?:把|将|使)?(?:优势|胜势)(?:稳稳|稳步|逐步)?(?:转化|兑现)为?(?:了)?(?:胜势|胜利|胜局)", ""),
    (r"力量对比(?:悬殊|发生[了]?(?:变化|逆转|倾斜))", ""),
    (r"大局已定", ""),
    (r"完全掌握[了]?(?:主动权|局面)", ""),
    # 信条7提到的空泛结论词，与 prompt 对齐做后处理兜底
    (r"锁死", "封锁"),
    (r"彻底封锁", "封锁"),
]


def reduce_cliches(text: str) -> str:
    """删减空洞套话/重复比喻。不改变事实性内容，只去修辞。"""
    out = text
    for pat, repl in _CLICHE_PATTERNS:
        out = re.sub(pat, repl, out)
    return out


# ── Puzzle 专用轻量反套话表 ───────────────────────────────────────────
# 与 _CLICHE_PATTERNS 的区别：跳过形容词类清洗（精准/精确/精妙 等在战术讲解中承载实际语义）。
_PUZZLE_CLICHE_PATTERNS = [
    (r"看似(?:平淡无奇|平平无奇|平淡|不起眼|普通|简单)[，,]?(?:实则|却|其实)?", ""),
    (r"别看这一步[^，。]{0,6}[，,]", ""),
    (r"如同?利剑出鞘", ""),
    (r"如洪水般(不可阻挡)?", "持续"),
    (r"天罗地网", "严密的控制"),
    (r"天衣无缝", "完整"),
    (r"密不透风", "严密"),
    (r"无形的牢笼", "包围"),
    (r"胜利的天平(开始|彻底)?(倾斜)?", "优势"),
    (r"(已如)?囊中之物", "胜势已成"),
    (r"不可阻挡", ""),
    (r"暗藏杀机", ""),
    (r"耐心的围猎", "稳步驱赶"),
    (r"致命一击", "最后一击"),
    (r"愈发默契", "更协调"),
    (r"(配合|协调)(愈发|越来越)默契", "配合更协调"),
    (r"默契(的)?配合", "配合"),
    (r"步步为营", "稳步推进"),
    (r"[，,]?\s*为(?:后续|接下来|下一步|最终|后面|最后)(?:的)?[^，。、！]{0,16}"
     r"(?:做准备|做好准备|奠定[了]?(?:坚实)?基础|创造[了]?[^，。]{0,8}条件|铺平[了]?道路|埋下伏笔)", ""),
    # PLAN-004 阶段 A 实测追加（phase4.2 puzzle 正文高频空泛结论句），与残局表对齐。
    (r"稳获(?:子力或位置优势|子力优势|子力)", ""),
    (r"(?:把|将|使)?(?:优势|胜势)(?:稳稳|稳步|逐步)?(?:转化|兑现)为?(?:了)?(?:胜势|胜利|胜局)", ""),
    (r"力量对比(?:悬殊|发生[了]?(?:变化|逆转|倾斜))", ""),
    (r"大局已定", ""),
    (r"完全掌握[了]?(?:主动权|局面)", ""),
]


def reduce_cliches_puzzle(text: str) -> str:
    """Puzzle 轻量反套话：只删废话模板，保留战术形容词。"""
    out = text
    for pat, repl in _PUZZLE_CLICHE_PATTERNS:
        out = re.sub(pat, repl, out)
    return out


# 坐标兜底清洗：prompt 已要求"禁坐标"，但 LLM 偶尔仍会吐出 e8/f8 这类格子名。
# 坐标一旦混进 TTS 会被逐字母念（"e-eight"），非常刺耳，所以定稿前用正则强制清除。
# 三层策略，从精到糙：
#   1) 白名单移动动词+坐标 → 整体收成方位动词（最自然，直接丢掉坐标）；
#   2) 通用「介词+坐标」→「介词+那一格」（动词不在白名单时，保住前面的动词不被截断）；
#   3) catch-all 清掉任何残留的孤立坐标。
_COORD = r"[a-h][1-8]"
_MOVE_TO_COORD = [
    (re.compile(rf"(?:被迫)?(?:退守|退回到|退回|退到|退至|后撤到|撤回到|撤到)\s*{_COORD}\s*格?"), "后退"),
    (re.compile(rf"(?:移到|移至|走到|走向|来到|落到|落在|停在|占据)\s*{_COORD}\s*格?"), "就位"),
    (re.compile(rf"(?:跳到|跳向|跳至|跳上|跃到|跃向)\s*{_COORD}\s*格?"), "跳出"),
    (re.compile(rf"(?:切入到?|进到|进至|挺进到|推进到|杀到)\s*{_COORD}\s*格?"), "切入"),
]
# 通用介词：动词未被白名单覆盖时（如"逼到f8""压向a7"），把坐标换成"那一格"，
# 让前置动词与句子结构完整保留，避免 catch-all 把动词截成残句。
_COORD_PREP = re.compile(rf"(?<=[一-鿿])(到|至|向|在|于)\s*{_COORD}\s*格?")
# catch-all：清除剩余的孤立坐标，连同可能的前导介词与"格"后缀一起吃掉。
_COORD_CATCHALL = re.compile(rf"(?:从|由|到|至|向|于|在|经)?\s*{_COORD}\s*格?")


def strip_coordinates(text: str) -> str:
    """清除 voiceover 中泄漏的棋盘坐标，防止进入 TTS 被逐字母念读。"""
    out = text
    for pat, repl in _MOVE_TO_COORD:
        out = pat.sub(repl, out)
    out = _COORD_PREP.sub(r"\1那一格", out)
    out = _COORD_CATCHALL.sub("", out)
    return out


# 跨段去重：chunk 之间各自独立生成、LLM 看不到全局，导致同一句套话
# （"逐步收紧包围圈""围绕对王争夺关键格"…）在多段里反复出现。这里在所有
# segment 汇总后做一次全局扫描：每个短语家族首次出现保留原文，第二次及以后
# 轮换成同义变体，保住语义、消除字面复读感。变体本身也要无坐标、不空洞。
_REPEAT_FAMILIES = [
    (re.compile(r"逐步收紧包围圈|不断收紧包围圈|收紧包围圈"),
     ["把包围圈又收小一圈", "进一步缩小对方王的活动范围", "继续收网", "再压掉一块活动空间"]),
    (re.compile(r"围绕对王(?:来回|反复)?调整[，、]?\s*争夺关键格|围绕对王(?:来回|反复)调整"),
     ["贴着对方王不断换位、卡住要害格", "在关键格上与对方王反复周旋", "一格一格地抢占对方王身边的要点"]),
    (re.compile(r"等待最佳时机完成最后一击|等待最佳时机|等待[^，。]{0,6}最后一击"),
     ["伺机收官", "只待最后一着到位", "等收官的时机成熟"]),
    (re.compile(r"为(?:下一步|后续)的?致命打击做准备|为致命一击蓄势"),
     ["为收官铺路", "把收杀的条件一点点凑齐"]),
    (re.compile(r"只能被动应对|只能被动防守|被动应对"),
     ["几乎没有还手余地", "走一步看一步，毫无主动权", "只能跟着白方的节奏走"]),
    (re.compile(r"逐步压缩(?:其|对方王的?)?(?:活动)?空间|不断压缩(?:其|对方)?(?:活动)?空间"),
     ["把对方王能落脚的格子越夺越少", "活动范围被一截截切掉", "腾挪余地越来越小"]),
]


def dedupe_across_segments(segments) -> None:
    """对已汇总的 StoryboardSegment 列表原地去重高频套话短语。

    首次命中保留原文；之后每次命中按家族轮换替换为同义变体。失败安全：
    任何异常都跳过该家族，不影响解说主体。
    """
    if not segments:
        return
    for pat, variants in _REPEAT_FAMILIES:
        if not variants:
            continue
        hit = 0
        for seg in segments:
            vo = getattr(seg, "voiceover", "") or ""
            if not vo:
                continue

            # 逐次替换：每个命中都计数，首个全局命中保留原文，其余按家族轮换变体
            def _sub_one(text):
                nonlocal hit
                out_parts = []
                last = 0
                for m in pat.finditer(text):
                    out_parts.append(text[last:m.start()])
                    if hit == 0:
                        out_parts.append(m.group(0))  # 首次保留
                    else:
                        out_parts.append(variants[(hit - 1) % len(variants)])
                    hit += 1
                    last = m.end()
                out_parts.append(text[last:])
                return "".join(out_parts)

            try:
                seg.voiceover = _sub_one(vo)
            except Exception:
                continue


def clean_cjk_text(text: str) -> str:
    """白名单清洗：只保留中文字符 + 常用中文标点，其余一律剔除。

    总结词、开场白等所有喂 TTS 的文本共用此清洗，避免白名单逻辑重复。
    """
    t = text.strip()
    t = re.sub(r"[^一-鿿，。、！？]", "", t)
    t = re.sub(r"[，、]{2,}", "，", t)
    t = re.sub(r"。{2,}", "。", t)
    t = re.sub(r"[，、！？]+。", "。", t)
    t = re.sub(r"^[，、！？。]+", "", t)
    return t.strip()


def clean_summary_text(text: str) -> str:
    """清洗总结词：先去引号与「总结」前缀，再走公共 CJK 白名单清洗。

    前缀统一由 generate_summary 末尾补回，保证不重复也不残缺。
    """
    t = text.strip().strip("「」\"'`").strip()
    t = re.sub(r"^总结(一下)?[，,：:]?", "", t).strip()
    return clean_cjk_text(t)


def has_forbidden_chars(text: str) -> bool:
    """是否仍含字母/数字（清洗失败的标志）。"""
    return bool(re.search(r"[A-Za-z0-9]", text))


def clean_opening_text(text: str) -> str:
    """清洗开场白：先去引号与「开场」前缀，再走公共 CJK 白名单清洗。"""
    t = text.strip().strip("「」\"'`").strip()
    t = re.sub(r"^开场[白]?[，,：:]?", "", t).strip()
    return clean_cjk_text(t)


# 阿拉伯数字 → 中文数字（单字映射），供 safe_puzzle_seed_text 把棋谱里的数字转成口播友好形式。
_DIGIT_CN = {
    "0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
    "5": "五", "6": "六", "7": "七", "8": "八", "9": "九",
}

# 连续阿拉伯数字匹配，供 digits_to_cn 整段读法转换
_DIGITS_TO_CN = re.compile(r"[0-9]+")


def digits_to_cn(text: str) -> str:
    """把口播文本中的阿拉伯数字转中文，连续数字按数位整体读（33→三十三，5→五）。

    用于残局/Puzzle voiceover 后处理：GBNF 的 cnstring 只在 Puzzle chunk 采样期锁中文，
    残局 chunk 与 fallback 文本路径不锁，API 模式下模型偶尔输出"5个减到4个""33%"等
    阿拉伯数字，TTS/字幕不能出现数字字符。

    读法规则（覆盖 voiceover 实际场景：活动格数、百分比、倍数、年份/编号）：
    - 1 位：直接读（5→五、9→九）
    - 2 位：按中文数位（12→十二、33→三十三、10→十、20→二十）
    - 3 位及以上：逐字读（2026→二零二六，符合年份/编号口播习惯；避免引入万/亿复杂度）

    不处理小数点/百分号等附带符号，由调用方其他规则清理。
    """
    if not text:
        return text
    return _DIGITS_TO_CN.sub(_replace_digits, text)


def _replace_digits(match) -> str:
    """正则回调：把一段连续阿拉伯数字按上述读法转中文。"""
    digits = match.group(0)
    n = len(digits)
    if n == 1:
        return _DIGIT_CN[digits]
    if n == 2:
        return _two_digits_to_cn(digits)
    # 3 位及以上逐字读（年份、编号、大数避免错误读法）
    return "".join(_DIGIT_CN[ch] for ch in digits)


def _two_digits_to_cn(two: str) -> str:
    """两位数按中文数位读：10→十、20→二十、33→三十三、12→十二、05→零五。"""
    tens, ones = two[0], two[1]
    if tens == "0":
        return ("零" if ones != "0" else "零零") + (_DIGIT_CN[ones] if ones != "0" else "")
    if ones == "0":
        # 整十：10→十（不读「一十」）、20→二十、90→九十
        return ("十" if tens == "1" else _DIGIT_CN[tens] + "十")
    if tens == "1":
        return "十" + _DIGIT_CN[ones]
    return _DIGIT_CN[tens] + "十" + _DIGIT_CN[ones]


def expand_inline_brackets(text: str) -> str:
    """把括号内容融入句子，避免口播/字幕出现括号停顿。

    供知识库种子规范化与 LLM 文本清洗共用，确保两条路径行为一致。
    调用方需自行先剔除不该展开的括注（如引擎评估值括注）。
    """
    if not text:
        return text
    out = text
    # 1) （如/比如/例如X、Y、Z）→ ，比如X、Y、Z
    out = re.sub(r"[（(]\s*(?:如|比如|例如)\s*([^（）()]+)[）)]", r"，比如\1", out)
    # 2) （含顿号/逗号的列举）→ ，也就是X
    out = re.sub(r"[（(]([^（）()]*[、，][^（）()]*)[）)]", r"，也就是\1", out)
    # 3) 剩余短括号（1-20字）→ 直接去掉括号
    out = re.sub(r"[（(]([^（）()]{1,20})[）)]", r"，\1", out)
    # 清理可能产生的双逗号
    out = re.sub(r"[，,]{2,}", "，", out)
    return out


def safe_puzzle_seed_text(text: str) -> str:
    """把知识库/骨架中的棋谱坐标和阿拉伯数字转成适合口播的中文表达。

    这是源数据规范化，不处理模型已生成文本；目的是避免模板兜底或 prompt 锚点
    自身把 f2/f7、5-7 这类不可播字符带入口播。
    """
    if not text:
        return ""
    out = str(text)
    out = re.sub(r"f2\s*/\s*f7|f7\s*/\s*f2", "王前弱格", out, flags=re.I)
    out = re.sub(r"\bf[27]\b", "王前弱格", out, flags=re.I)
    out = re.sub(r"\b[a-h][1-8]\b", "关键格", out, flags=re.I)
    out = out.replace("fried liver攻击", "经典弃子攻击")
    out = re.sub(r"[A-Za-z]", "", out)
    out = "".join(_DIGIT_CN.get(ch, ch) for ch in out)
    out = re.sub(r"[/\\*_#`\[\]{}<>|]", "", out)
    # 剔除引擎术语括注（如"（评估约六零零厘兵以上）"），口播不能出现评估值/厘兵
    out = re.sub(r"[（(][^（）()]*(?:评估|厘兵|分值)[^（）()]*[）)]", "", out)
    out = out.replace("厘兵", "").replace("评估值", "").replace("评估", "")
    # 展开剩余的列举型/说明型括号，避免知识库种子文本把括号带入口播与字幕
    out = expand_inline_brackets(out)
    out = re.sub(r"[，、]{2,}", "，", out)
    return out.strip()
