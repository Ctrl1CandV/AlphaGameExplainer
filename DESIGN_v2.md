# AlphaGameExplainer 解说深度优化设计文档 v3

> **状态**: 设计阶段 | **日期**: 2026-06-03
>
> 本文档聚焦一个核心问题：**如何让 LLM "自己想出"哪一步是关键手，而不是靠我们告诉它。**

---

## 目录

- [一、问题诊断](#一问题诊断)
  - [1.1 具体案例](#11-具体案例)
  - [1.2 根因：喂结论 vs 喂前提](#12-根因喂结论-vs-喂前提)
  - [1.3 为什么当前架构做不到](#13-为什么当前架构做不到)
- [二、方案设计](#二方案设计)
  - [2.1 核心原则](#21-核心原则)
  - [2.2 方案 A：战术叙述提取器](#22-方案-a战术叙述提取器)
  - [2.3 方案 B：prompt 重构——从命令式到邀请式](#23-方案-bprompt-重构从命令式到邀请式)
  - [2.4 方案 C：引擎信号作为中性观察](#24-方案-c引擎信号作为中性观察)
- [三、关于"引擎作为外部审核员"的深度分析](#三关于引擎作为外部审核员的深度分析)
- [四、实施计划](#四实施计划)
- [五、预期效果](#五预期效果)

---

## 一、问题诊断

### 1.1 具体案例

局面 `8/8/8/3k4/3r4/8/8/1K1Q4 w - - 0 1`（白后白王 vs 黑车黑王）：

解法中关键的几步：白后走到一个格 → **同时将军黑王 + 攻击黑车** → 黑方必须应将 → 所有合法应将中**没有一个能同时保住车** → 黑车被吃 → 局面从"后对车"塌缩为"后对单王"（已知的简单必胜残局）。

这是**全局的胜负手**。但当前系统里，LLM 对这一段的解说跟对其他步骤的解说**没有本质区别**——它不知道这步改变了局面的结构。

### 1.2 根因：喂结论 vs 喂前提

当前 `insight_extractor._compute_importance` 的做法：

```python
# 喂的是 "结论"
importance = "high"
importance_reasons = ["对方王活动空间锐减50%", "形成对王，掌握主动权"]
```

然后在 `commentator.py` 的 prompt 中：

```
详略: 重点节点 — 这是关键转折，请写得更有张力，点出它为什么重要
importance: high
importance_reasons: 对方王活动空间锐减50%；形成对王，掌握主动权
```

LLM 从头到尾没有做**任何推理**。它只是把我们给的结论换了一种说法——就像学生抄答案，而不是自己解题。更要命的是，**模型根本"看不到"这一步的战术结构**。它不知道：

- 白后的落点同时做了两件事（将军 + 攻击车）
- 黑方应将的走法没有一个能保住车
- 吃完车后残局类型发生了质变

所以它**不可能**讲出"后一子两用、王顾此失彼"这种有洞察力的解说——因为它没有能推理的原料。

```
根因链条：
  模型拿到的不是 "这一着同时做了X和Y，对方无法两全"
  而是               "importance=high, 请重点强调"
    → 模型没有棋理事实可以思考
      → 只能把"重要"换成"关键"换成"转折"
        → 套话
```

### 1.3 为什么当前架构做不到

`insight_extractor._compute_importance` 的评分逻辑：

```python
score = 0
if maneuver_label:    score += 15
if reduction >= 50%:  score += 25   # 活动空间减半 → "重要"
if reduction >= 25%:  score += 12
if wa <= 1:           score += 20
if milestone:         score += 20
if cs_is_critical:    score += 8
# ... 阈值判断 → importance = "high"/"medium"/"low"
```

这是**启发式打分**，不是**棋理分析**。它能量化"王的活动空间变小了"，但无法识别"这一着是双重攻击+被迫丢子"。后者需要的是**战术模式匹配**，而不是**特征加权求和**。

---

## 二、方案设计

### 2.1 核心原则

```
旧范式：提取特征 → 打分 → 输出结论标签 → LLM 复述标签
新范式：分析棋局 → 生成棋理叙述 → LLM 理解叙述 → LLM 自己形成判断
         ↑                    ↑                  ↑
     (纯 board API)    (中文自然语言)    (模型的内在推理能力)
```

三个操作原则：

1. **喂前提，不喂结论**。给的是"这一着同时做了X和Y，对方无法两全"，不是"这一步很重要请重点讲"。
2. **用棋理语言，不用结构化标签**。`tactic_type: double_attack` 需要模型"翻译"，不如直接给"这一着同时将军并攻击黑车"——模型读到就像棋手看到棋盘。
3. **让模型自己判断**。不指定哪步是关键手。用人物设定（persona）邀请模型自己判断，而不是用指令命令它。

### 2.2 方案 A：战术叙述提取器

**目标**：用纯 `chess.Board` API 检测每一步中是否存在**需要推理才能理解的战术关系**，并产出 1-3 句棋理叙述（中文、无坐标、无评判词）。

这是最核心的改动。新增 `insight_extractor.py` 中的一个函数。

#### 2.2.1 检测的战术模式

| 模式 | 定义 | 为什么需要推理 |
|------|------|---------------|
| **双重攻击** | 一步同时将军 + 攻击对方无保护的大子 | 涉及"必须应将"规则的推理 |
| **被迫丢子** | 对方所有应将都无法保住被攻击的子 | 需要穷举对方合法应将 |
| **残局类型质变** | 吃完子后残局类型改变（如 KRvKQ→KQvK） | 需要理解"简化"的战略含义 |
| **唯一好着** | 其他走法会导致评估暴跌 | 需要 MultiPV 或 SF 佐证 |
| **零化着** | 吃子/升变，重置 50 步计数 | 残局中的关键节奏点 |

#### 2.2.2 核心函数

```python
def _extract_tactical_narrative(cs, board_before, board_after, role_meta, endgame_name):
    """
    对单个压缩节点提取战术叙述（棋理中文，无坐标、无评判词）。
    返回 List[str]，每个是一句棋理观察。
    
    核心逻辑：不判断"重要/不重要"，只描述"发生了什么"和"为什么对方无法两全"。
    """
    narratives = []
    weak_color = role_meta.get("weak_color") if role_meta else None
    strong_color = role_meta.get("strong_color") if role_meta else None

    # ---- 检测1：双重攻击 ----
    for san in cs.sans:
        try:
            move = board_before.parse_san(san)
        except ValueError:
            continue
        
        gives_check = board_before.gives_check(move)
        temp = board_before.copy()
        temp.push(move)
        
        if gives_check:
            # 走完后，走子是否同时攻击对方无保护的大子？
            moved_piece = temp.piece_at(move.to_square)
            if moved_piece:
                for attacked_sq in temp.attacks(move.to_square):
                    target = temp.piece_at(attacked_sq)
                    if (target and target.color != moved_piece.color
                            and target.piece_type != chess.KING
                            and target.piece_type in (chess.QUEEN, chess.ROOK)):
                        # 这个大子有没有被保护？
                        defenders = temp.attackers(target.color, attacked_sq)
                        if not defenders:
                            # 双重攻击成立。进一步检查：对方的应将能否保住它？
                            can_save = False
                            for legal_reply in temp.legal_moves:
                                temp2 = temp.copy()
                                temp2.push(legal_reply)
                                if temp2.piece_at(attacked_sq) is not None:
                                    if temp2.attackers(target.color, attacked_sq):
                                        can_save = True
                                        break
                            
                            piece_name = {chess.QUEEN: "后", chess.ROOK: "车",
                                          chess.BISHOP: "象", chess.KNIGHT: "马"}.get(target.piece_type, "子")
                            
                            if not can_save:
                                narratives.append(
                                    f"这一着同时做了两件事：给黑王将军，同时直接攻击黑{piece_name}。"
                                    f"黑方必须应将，但在所有合法的应将走法中，"
                                    f"没有一步能同时保住黑{piece_name}——这意味着黑{piece_name}必定在下一步被吃掉。"
                                )
                            else:
                                narratives.append(
                                    f"这一着同时将军并攻击黑{piece_name}——一子两用。"
                                )
        
        board_before.push(move)

    # ---- 检测2：残局类型质变 ----
    # 比较节点前后的子力组成
    pieces_before = _material_signature(chess.Board(cs.fen_before))
    pieces_after = _material_signature(chess.Board(cs.fen_after))
    if pieces_before != pieces_after:
        before_desc = _describe_material(pieces_before)
        after_desc = _describe_material(pieces_after)
        if before_desc != after_desc:
            narratives.append(
                f"这一步之后，局面从「{before_desc}」变为「{after_desc}」——"
                f"残局类型发生了质变。"
            )

    return narratives
```

#### 2.2.3 输出示例（对比）

对于那个"白后吃车"的节点：

```python
# 旧 insight_extractor 输出：
{
    "teaching_point": "后将军、逼对方王让步，对方王活动格收窄（4→2）",
    "importance": "high",
    "importance_reasons": ["对方王活动空间收窄50%"],
}

# 新 insight_extractor 额外输出：
{
    "tactical_narratives": [
        "这一着同时做了两件事：给黑王将军，同时直接攻击黑车。"
        "黑方必须应将，但在所有合法的应将走法中，"
        "没有一步能同时保住黑车——这意味着黑车必定在下一步被吃掉。",

        "这一步之后，局面从「后对车」变为「后对单王」——"
        "残局类型发生了质变。「后对单王」是已知的简单必胜残局。",
    ],
}
```

旧输出给 LLM 的是"王空间收窄 50% + importance=high"——逻辑断裂。
新输出给 LLM 的是**完整的因果链**："一子两用 → 对方应将却无法两全 → 必丢车 → 残局质变"。

LLM 拿到后者，不需要被"告诉"这是关键手——**因果链本身就在说"这是关键手"**。

### 2.3 方案 B：prompt 重构——从命令式到邀请式

**核心改动**：在 prompt 中去掉"结论注入"，升级人物设定。

#### 2.3.1 删除：结论注入

当前 `commentator.py:_build_chunk_prompt` 中有这些结论注入：

```
详略: 重点节点 — 这是关键转折，请写得更有张力，点出它为什么重要
importance: high
importance_reasons: 对方王活动空间锐减50%；...
```

**全部删除**。不给模型下"重要/不重要"的判决。

保留的是**中性棋理事实**（来自 insight_extractor）：
- `teaching_point` — 保留（已是棋理事实）
- `spatial_change` — 保留（纯数据）
- `must_mention` — 保留但改名为 `chess_facts`，语气从"必须讲到"改为"棋理观察"

#### 2.3.2 新增：战术叙述注入

在 `_build_chunk_prompt` 的节点信息中新增一段：

```python
# 在节点 prompt 块中追加（如果该节点有 tactical_narratives）
tactical_narratives = node.get("tactical_narratives", [])
if tactical_narratives:
    parts.append("棋理分析（由棋盘直接算出的战术事实）:")
    for tn in tactical_narratives:
        parts.append(f"  · {tn}")
```

注意格式——用 `·` 开头，中文字，完全是叙述式的。不写"请重点讲这个"，不写"这是关键"。

#### 2.3.3 升级：系统 message + json_header 的人物设定

当前 `llm_backend.py` 的 system message 是：

```
你是国际象棋赛事解说员。解说基于节点信息的「状态」字段中的真值。
只有「已将杀」的节点才能说将杀/绝杀...
```

改为：

```python
SYSTEM_MESSAGE = (
    "你是一位会自己看棋的国际象棋教练。你的讲解信条：\n"
    "- 你不需要被告知哪一步重要——你会从每个节点的棋理事实中自己判断；\n"
    "- 当你在节点信息中看到一着同时做了多件事、让对方无法两全、或改变了残局结构时，"
    "你会自然地讲出它为什么是关键手；\n"
    "- 你的判断来自对棋局的理解，而不是对指令的服从。\n"
    "- 底线：只有「已将杀」的节点才能说将杀/绝杀。其他节点用推进性描述。"
    "多个segment之间要有承接关系。不要复述提示词。"
)
```

对应修改 `commentator.py:_build_json_header` 的前几行——把"你的任务是...你必须..."改成"你是一位...你自然会在看到..."。

**关键区别**：

```
旧：你是国际象棋赛事解说员。要求：只依据给定走法和局面信息解说。
    → "你是一个工具，请遵守以下规则"

新：你是一位会自己看棋的教练。你不需要被告知哪一步重要。
    → "你是一个有判断力的人，以下是你可以用来判断的事实"
```

### 2.4 方案 C：引擎信号作为中性观察

这里引出用户的核心问题：**是否可以用 Stockfish 作为"外部审核员"协作 LLM？**

在深入分析之前，先明确 SF 能提供什么、不能提供什么：

| SF 能提供 | SF 不能提供 |
|-----------|------------|
| 评估值变化（eval delta） | 解释"为什么评估变了" |
| MultiPV 多候选着评估差 | 识别战术模式（叉子、牵制等） |
| 将杀距离变化 | 判断"这是关键手"（那是人的判断） |
| 唯一好着检测 | 将评估值翻译成教练语言 |

**结论：SF 不是审核员，而是信号源。** 把它的输出作为"中性观察"喂给 LLM，能增加量化维度的事实密度。

#### 2.4.1 具体做法

在 `storyboard.py:build()` 中，对**已经做了 heavy search 的关键节点**（目前有 `_sf_step_heavy` 在 `key_indices`），多加两个字段：

```python
# 1. 评估值跳跃
if eval_before is not None and eval_after is not None:
    delta = abs(eval_after - eval_before)
    if delta > 200:
        node["eval_jump"] = (
            f"这一步之后，局面的评估值发生了显著变化（优势扩大了约{delta}厘兵），"
            f"说明这一步极大地推进了胜势。"
        )

# 2. 将杀距离缩短
if mate_before is not None and mate_after is not None:
    if mate_after < mate_before:
        node["mate_progress"] = (
            f"距将杀的步数从约{mate_before}步缩短到约{mate_after}步。"
        )

# 3. MultiPV 唯一好着（已在 is_only_move 中部分覆盖，增强描述）
if is_only_move and len(candidates) >= 1:
    node["only_move_narrative"] = (
        "注意：除这一步外，其他候选走法都会让胜势大幅缩水——"
        "这是当前局面下唯一能保住胜利果实的选择。"
    )
```

这些**全部是中性观察**。不写"这是关键手"，只写"评估值跳了多少"、"将杀近了多远"、"有没有替代选择"。LLM 自己决定要不要把这些量化信息转化为"关键手"的判断。

#### 2.4.2 注入 prompt 的方式

与战术叙述完全一样——追加到节点信息块中：

```python
# 在 _build_chunk_prompt 中：
eval_signals = []
if node.get("eval_jump"):      eval_signals.append(node["eval_jump"])
if node.get("mate_progress"):  eval_signals.append(node["mate_progress"])
if node.get("only_move_narrative"): eval_signals.append(node["only_move_narrative"])
if eval_signals:
    parts.append("引擎数据（量化参考，不是判决）:")
    for es in eval_signals:
        parts.append(f"  · {es}")
```

注意 `"引擎数据（量化参考，不是判决）"` 这个前缀——明确告诉 LLM 这些数字只是参考，判不判关键手是 LLM 自己的事。

---

## 三、关于"引擎作为外部审核员"的深度分析

你提出的"集成专业引擎作为外部审核员"是一个有洞察力的问题。让我分析三种可能的协作模式：

### 模式 A：后置审核（引擎审 LLM 输出）

```
LLM 生成解说 → 提取解说中的棋理声明 → 引擎验证声明 → 驳回/修正
```

**不可行**。原因：
1. **NL 理解鸿沟**：LLM 的解说用的是自然语言（"黑王被逼到了边线"），没有工具能从自然语言解说中精确提取可验证的棋理命题。这是 NLP 难题。
2. **延迟翻倍**：每次生成后再走一遍验证，时间成本不可接受。
3. **修正困难**：审出错误后怎么修？重新生成？局部替换？目前都没有可靠机制。

### 模式 B：前置信号（引擎给 LLM 提供输入）

```
引擎分析节点 → 输出量化信号 → 注入 prompt → LLM 参考信号进行解说
```

**推荐**。这就是方案 C。引擎的 evaluative delta、mate distance、MultiPV 差距作为中性观察注入 prompt，LLM 自己决定怎么用。

优势：
- 不需要 NL 理解
- 引擎计算已有（`_sf_step_heavy` 在关键节点做了）
- 信号是纯数字 → 文本转换规则完全可控

### 模式 C：协作判断（引擎 + 规则共同判定关键手）

```
引擎评估变化 + 战术提取器检测模式 → 综合判断 → 输出"中性事实包" → LLM 解说
```

**最佳方案**。把引擎的量化信号和 `insight_extractor` 的战术检测结果**组合**成一个"事实包"，但不做"重要性"的综合打分。LLM 拿到的是：

```
棋理分析:
  · 这一着同时将军并攻击黑车，黑方应将却无法保住车
引擎数据:
  · 这一步之后，评估值从+200厘兵跳升到+500厘兵
  · 距将杀从12步缩短到7步
  · 其他走法都会让优势回落到均势
```

读到这些事实，LLM 不需要任何额外的"重要性"标签——它自然就会把这一步讲成胜负手。因为它拿到了三个维度的证据：**战术结构的几何必然性** + **评估值的量化跳跃** + **候选着之间的对比差异**。

### 最终建议：选模式 C，但引擎部分用"轻量化信号"

不需要"完整的 SF 审核员"。只需要：

1. **战术提取器**（方案 A）做几何分析——这是 80% 的价值
2. **引擎信号**（方案 C 轻量版）做量化佐证——在已有 heavy search 的节点多生成 1-2 行文本描述
3. **Persona 升级**（方案 B）邀请 LLM 自己判断——这是"最后一公里"

三者协作：几何分析给的是**必要性**（为什么对方无法两全），引擎信号给的是**量级感**（这一步到底推进了多少），persona 给的是**自主权**（你来判断，你来表达）。

---

## 四、实施计划

### 改动文件（4 个，按顺序实施）

| 步骤 | 文件 | 改动内容 | 工作量 |
|------|------|---------|--------|
| 1 | `src/insight_extractor.py` | 新增 `_extract_tactical_narrative()`，在 `extract_for_node()` 中调用 | ~120 行 |
| 2 | `src/storyboard.py` | `build()` 中注入 `tactical_narratives` 到节点 dict；增强 SF 信号字段 | ~40 行 |
| 3 | `src/commentator.py` | `_build_json_header()` 升级人物设定；`_build_chunk_prompt()` 注入战术叙述和引擎信号、删除 `importance` 结论注入 | ~30 行修改 |
| 4 | `src/llm_backend.py` | 更新 system message | ~5 行修改 |

**不改动的文件**：`common.py`, `pipeline.py`, `board_renderer.py`, `video_composer.py`, `tts_engine.py`, `tablebase.py`, `stockfish_analyzer.py`, `parser.py`

### 实施步骤

**步骤 1（核心）**：`insight_extractor.py` 新增战术叙述提取器

- 实现 `_detect_double_threat()` + `_detect_material_shift()` + `_compose_tactical_narrative()`
- 全部用 `chess.Board` API，零引擎依赖
- 失败安全：任何异常返回空列表，不影响下游
- 输出只含中文棋理叙述，不含坐标、标签、评判词

**步骤 2（透传）**：`storyboard.py` 透传新字段 + 增强引擎信号

- 在 `build()` 中调用 `extract_for_compressed` 后，把 `tactical_narratives` 注入 node dict
- 对已有 heavy search 的节点，生成 1-2 句引擎信号的中文描述

**步骤 3（prompt 重构）**：`commentator.py` 升级

- `_build_json_header`：开头 5 行升级为"会自己看棋的教练" persona
- `_build_chunk_prompt`：删除 `importance` 和 `importance_reasons` 的注入；新增 `tactical_narratives` 和 `eval_signals` 的注入
- `_build_chunk_prompt`：将 `must_mention` 改名为 `chess_facts`，语气从"必须讲到"改为"棋理观察"

**步骤 4（收尾）**：`llm_backend.py` system message 升级

### 验证方法

用 `8/8/8/3k4/3r4/8/8/1K1Q4 w - - 0 1` 这个局面跑一遍，人工检查：

1. `insight_extractor` 是否正确检测出双重攻击和残局质变
2. 生成的解说是否包含"一子两用""无法两全""残局质变"等棋理分析
3. 是否不再出现"请重点强调""这是关键转折"等指令泄漏

---

## 五、预期效果

以白后吃黑车这一步为例：

**当前输出**（旧版）：
> 白后走到关键位置，给黑王施加了巨大压力。这一步之后黑王的活动空间被大幅压缩，白方优势继续扩大。

**预期输出**（新版）：
> 这是全局的胜负手。白后这一着一子两用——既叫了将，又直接盯住了黑车。黑方现在必须应将，但仔细看：所有能应将的走法里，没有一步能同时保住车。这意味着黑车必丢，而一旦黑车被吃，局面就从难缠的「后对车」变成了「后对单王」——那是已知的必胜残局，后面的推进只是时间问题。

区别在于：
- 旧版讲的是**现象**（空间压缩了、优势扩大了）
- 新版讲的是**因果**（为什么空间会被压缩？因为必须应将。为什么优势会扩大？因为对方无法两全，必丢大子。丢了大子为什么决定性的？因为残局类型发生了质变。）

**LLM 能讲出后者，不是因为它变聪明了，而是因为我们给了它能推理的原料。**

---

> **文档版本**: v3.0 | **日期**: 2026-06-03
