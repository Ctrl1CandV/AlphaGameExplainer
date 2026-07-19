# AlphaGameExplainer · 项目记忆

> 本文件是项目主记忆体，描述项目**当前状态**。新概念/术语 → 追加到「领域语言」；架构决策 → 追加到「架构决策」并写 ADR；平台限制/陷阱 → 追加到「硬约束」。每条不超过两行，只追加不重排。

## 领域语言

- **残局讲解链路**：PGN/FEN → 表库求解或 Stockfish 单线求解 → 节点压缩 → 分镜 → LLM 中文解说 → TTS → 视频。
- **Puzzle 链路**：战术讲解链路，含双关键点强约束（机理/落地）。独立于残局讲解链路。
- **单线求解**：当前 `stockfish_analyzer.get_solution` 只返回最优单条走法序列，不含多分支对比。
- **node / chunk / segment**：node 是 storyboard 压缩后的讲解节点（1-N 着）；chunk 是 LLM 生成时的分块（4 node/chunk）；segment 是 LLM 输出的单个解说段。
- **GBNF 语法**：llama.cpp 的 token 级语法约束，用于残局 JSON 和 Puzzle cnstring 锁死中文。
- **讲解词中文纯净化**：voiceover 禁止英文/数字/坐标/Markdown。已有 `_strip_coordinates` / `_clean_cjk_text` 兜底。
- **Narrative Planner / narrative_role**（**已实施，ADR-012**）：程序化叙事规划层，为每个节点计算tension_score(0-1)和叙事角色（setup/build_up/climax/falling_action/resolution），动态分配字数预算和tone_hint。属于 storyboard 阶段程序化计算，不引入新LLM调用。代码落地于 insight_extractor.py(_compute_tension) + endgame_builder.py(_assign_narrative_role) + endgame_commentary.py(prompt注入)。
- **单Pass flat generation**：当前架构——算法/事实→单次prompt→LLM一次性输出全部解说词。模型同时承担规划+叙事+格式，是质量瓶颈之一，但当前首要瓶颈是上游算法层真值错误。
- **DS-MHP-lite 动态证据路径**（**已废弃，ADR-011**）：曾计划的证据路径RAG。废弃——上游已为真+Planner已给叙事骨架后边际收益低，且属内容型注入有GIGO风险。由 ADR-017 提示词工程深化取代其价值定位。
- **Plan-Then-Generate 两阶段生成**（**暂缓**）：Pass1 Content Planner输出结构化要点JSON，Pass2 Draft Generator基于要点生成口播成稿。暂缓——上游事实不稳时两阶段会放大错误。
- **Best-of-N (BoN)**（**已废弃，ADR-013**）：曾计划关键段生成N个候选选优。废弃——多输出选优不能有效提质，反而拉长生成时间并引入生成/评分的额外问题。（真值合法性核查维度早已剥离落地于 validators.py，属 ADR-015 范畴，不受本废弃影响）
- **Playbook 进化 / QLoRA 阶段性固化**（**已废弃，ADR-014**）：曾计划人工门控解说模式库+QLoRA固化。废弃——重且有把错误模式固化进context/权重的风险，性价比低。
- **提示词工程深化 / KB知识立体化利用**（**首要主线，ADR-017**）：在上游信息已充分为真的前提下，深化利用KB（json）知识构造提示词——负面约束注入（把套话词表写进prompt禁用清单）+ KB知识按局面立体化组织 + 可选全局润色Pass。无大结构改动，是当前最高ROI优化。
- **知识库扩容 + 优质范例库**（**必须做，ADR-018**）：扩容 endgame_kb/puzzle_themes，并新建解说范例库锚定「怎么说」的风格（非内容，区别于已废弃ADR-011）。范例经爬取 + 强模型API（Claude Opus 4.8/GPT-5.6）离线生成，非人工手写。
- **云端 API 并行生成后端**（**首要实施，ADR-019**）：DeepSeek API 为主 + 本地 llama.cpp 兜底（单次调用失败即切本地，带连续失败熔断）。客户端用 OpenAI SDK 指向 OpenAI 兼容端点 `https://api.deepseek.com`（非 /anthropic、非原生 requests），模型id/base_url 经 env 可配置，默认 `deepseek-v4-flash`。**关键约束：HTTP API 无法用 GBNF，Puzzle 采样期 `cnstring` 中文锁定丢失，纯中文/JSON 结构完全落到既有 validator+retry+后处理**，须先冒烟验证通过率再全量接入。方案见 docs/plans/PLAN-002。
- **多战略计划讲解**（**长期方向，非当前主线**）：给定中局局面，引擎产出多条导向不同终局的走法分支，事后命名对比讲解。不追求算法找到「真正的人类战略意图」，只追求「不同分支→不同终局」可量化验证。
- **终局差异化gate / 事后命名 / 融入式解说**（**长期方向**）：多战略计划链路的子组件，随 ADR-006~009 一起暂缓。
- **项目结构重构**（**已完成，ADR-016**）：按管线分层将 commentator.py(2752行)/storyboard.py(2078行) 拆分为 8 个子包（infra/chess_utils/analysis/solver/storyboard/commentator/pipeline/media），消除12+处函数级重复，单向依赖。重构不改运行时行为，用回归样本验证逐段语义等价。

## 架构决策

- **[ADR-006]【暂缓】多战略计划采用伪逻辑路线** —— 多分支→终局差异化gate→事后命名，不强求算法理解人类战略，只确保分支导向不同终局可量化验证。非当前主线，排期靠后。
- **[ADR-007]【暂缓】分支探索引擎：Lc0 BT3-768x15 为主，Stockfish MultiPV 5 降级备选** —— Lc0 policy取根候选+searchmoves提取多分支；NPS不足或Lc0不可用时回退SF MultiPV。随多战略计划方向暂缓。
- **[ADR-008]【暂缓】多计划对比采用融入式解说结构** —— 在分步解说关键分叉点插入分支描述，不独立成前置段落。随多战略计划方向暂缓。
- **[ADR-009]【暂缓】事后命名采用API预标注+Faiss检索库** —— 离线强模型API预标注局面特征→入库Faiss；在线近邻检索，低置信度降级为特征描述型名字。随多战略计划方向暂缓。
- **[ADR-010]【已废弃】解说质量提升采用五层分层架构** —— 原L1→L5（Narrative Planner→DS-MHP-lite→两阶段生成→评分器+BoN→Playbook进化）整体架构已被"上游算法层优先"路线取代；验证优先原则保留，但L2-L5在truth错误率达标前不启动代码开发。
- **[ADR-011]【已废弃】DS-MHP-lite动态证据路径替代静态RAG** —— 被 ADR-017 取代。内容型证据路径在 Planner 已给叙事骨架后边际收益低、复杂度高，且属内容注入有 GIGO 风险；解说提质改走 ADR-017 提示词工程路线。从未落地代码。
- **[ADR-012]【已采纳·已实施】Narrative Planner程序化叙事规划** —— storyboard阶段计算tension_score(0-1)/narrative_role(setup/build_up/climax/falling_action/resolution)/动态字数预算/tone_hint。纯Python计算无LLM调用，代码已落地于 src/analysis/insight_extractor.py(_compute_tension) + src/storyboard/endgame_builder.py(_assign_narrative_role) + src/commentator/endgame_commentary.py(prompt注入)。tension归一化分母40->80（climax档溢出修复）。
- **[ADR-013]【已废弃】程序化多维评分器+Best-of-N选优** —— 多输出选优对解说提质增量小，反而拉长生成时间并引入生成/评分的额外问题，弃用。**注**：真值合法性核查维度早已剥离落地于 validators.py/审计工具（属 ADR-015 范畴），不随本决策废弃而移除。
- **[ADR-014]【已废弃】人工门控Playbook进化，QLoRA降级为长期固化** —— 机制重、有把错误模式固化进context/权重的风险，且解说提质ROI低于提示词工程/范例库路线，弃用。
- **[ADR-015]【已采纳】上游真值修复采用"前置事实注入+后验硬事实核查"双保险** —— storyboard/insight每节点算确定性事实（子力/吃子/升变/将杀/per-step material delta）注入prompt白名单；commentator校验链路后验核查棋子存在性/将杀位置。代码已落地，上游全量验证通过（30残局+39 Puzzle零问题）。
- **[ADR-016]【已采纳并已实施】全项目结构重构——按管线分层拆分巨型文件** —— 将 commentator.py(2752行)和 storyboard.py(2078行)按管线拆分为子包；提取 chess_utils/ 消除12+处函数级重复；单向依赖禁止循环。重构不改运行时行为，用回归样本验证逐段语义等价。**遗留**：ADR-016 中 P7「docs/CLAUDE.md/benchmark纳入版本控制」未执行，.gitignore 仍排除这些路径，待处理。

## 硬约束

- **硬件**：4070 Ti Super 16G 显存。Qwen3.6-27B 4-bit（~14.5GB）几乎占满显存，**无法与 Lc0 BT3（~2.6GB）或 ChatTTS（~2GB）同时驻留**，必须串行加载/释放（管线已用 `release_backend()` 在 TTS 前释放 LLM）。多战略计划链路（ADR-007 需 Lc0）与 LLM 生成需分阶段错峰用显存。
- **n_ctx=4096**：本地 llama.cpp 当前默认生成上下文窗口，受 Qwen3.6-27B 4-bit 占满显存后 KV cache 余量所限，可经 `LLAMA_CPP_N_CTX` 覆盖。非硬产品上限；prompt 仍需预算控制。切到 API 后端后此约束不适用于 API 路径（DeepSeek 128k 窗口），仅本地兜底路径受限。
- **讲解词中文纯净化**：voiceover 禁止英文/数字/坐标/Markdown。已有 `_strip_coordinates` / `_clean_cjk_text` 兜底。
- **Syzygy 表库覆盖 3-6 子**：超出范围的残局走 Stockfish 求解。

## 当前状态

**第一阶段已结束**（2026-07-09 上游算法层优化合入+回归验证）：前置事实注入、per-step material delta、叙事基调解耦、空间轨迹、后验硬事实核查、P0遗留bug修复。

**结构重构已完成**（2026-07-13，ADR-016）：commentator.py/storyboard.py 拆分为 8 个子包，消除函数级重复，单向依赖。

**Phase 2 已完成**（2026-07-14）：
- 根因D 关键手定位器回归--fallback 率 51%->15%，评估类标签走解题方第一步策略，补充22个缺失评分器（25->47），新增 _solver_fallback_key_move_idx/_solver_fallback_reason（src/storyboard/key_move_locator.py）
- ADR-012 Narrative Planner--tension_score/narrative_role/word_budget/tone_hint 注入 prompt，tension 归一化分母 40->80（climax档溢出修复），locate_theme_key_moves 入口加字符串自动 split 守卫
- 同事审查4个问题已全部修复

**验证状态**：
- 上游全量验证通过：残局30样本（29/30零问题，1短弧线无climax属合理边界）+ Puzzle39样本（零问题）
- 效果验证通过：KRvK残局（Narrative Planner弧线生效）+ 001aK Puzzle（关键手定位正确，material fact注入生效）

**第三阶段已启动**（2026-07-16，解说生成层优化）：上游真值已稳、Planner已减负，瓶颈转移到"把正确事实转化为优质解说词"。本阶段决策：
- 废弃 ADR-011（证据路径）/013（BoN选优）/014（Playbook进化）——边际收益低或有固化错误风险
- ADR-017 提示词工程与KB利用深化=首要主线（E1 主动约束 + E2 选择性知识注入，已落地）
- ADR-018 范例库=最小人工审核范例集试验，默认关闭，待消融验证

**Phase 3 首轮基线 + reviewer 复盘（2026-07-17）**：`data/quality_benchmark_phase3/` 69 样本（none/4096），58 SUCCESS / 11 FALLBACK。reviewer 路径 B 判定不通过——3 个 Endgame 原始 JSON 泄漏进配音、8 个 Puzzle 通用 fallback 无区分度、审计指标失真（forbidden 69/69 误报、Puzzle 套话漏检）、逐调用可观测性缺失。二次方案 `docs/plans/PLAN-001-Phase3解说质量修复与验证闭环.md`（草案）。

**当前首要实施=API 后端（ADR-019 提前动工）**：上司已批准 API 方案与风险。`docs/plans/PLAN-002-DeepSeek-API为主本地兜底后端.md`（草案）——DeepSeek API 为主 + 本地 llama.cpp 单次调用兜底（2 次重试 + 连续失败熔断），模型 id/base_url env 可配置。**排序：PLAN-002 先行，PLAN-001 在新后端下再推进**。PLAN-002 阶段 0 为强制冒烟闸门：实测去 GBNF 后中文纯净率/JSON 通过率，不达标不进入全量接入。

**工具链**：`tools/quality_audit/` 质量审计工具（默认流程已取消 AI 标注，仅生成 + 人工查看）。

## 项目结构

按管线单向分层（ADR-016），详细目录读代码为准，不在此复制。核心层：
- `src/infra/`：`llm_backend.py`(后端抽象+单例+release)、`logger.py`
- `src/chess_utils/`：material/position/tactic（消除函数级重复）
- `src/analysis/`：`insight_extractor.py`(棋理事实+_compute_tension)、`endgame_kb.py`、`themes_kb.py`
- `src/solver/`：`stockfish_analyzer.py`(单线三阶段)、`tablebase.py`(Syzygy)
- `src/storyboard/`：compressor/`key_move_locator.py`/`endgame_builder.py`(_assign_narrative_role)/puzzle_builder/prelude
- `src/commentator/`：endgame/puzzle 解说、`generator.py`(主流程)、`validators.py`、`text_filters.py`、`grammar.py`(GBNF)、`json_utils.py`
- `src/pipeline/`：endgame(5步)/puzzle(4步) 管线；`src/media/`：渲染/TTS/字幕/合成
- `data/`(endgame_kb/puzzle_themes/commentary_examples/quality_benchmark_phaseN)、`syzygy/`、`tools/quality_audit/`、`tools/lc0/`、`main.py`、`docs/`(SPEC/plans/adr)

## 文档与记忆系统

- 四层文档职责：CLAUDE.md=长期领域语言/架构决策索引/硬约束/当前状态；docs/adr/=决策论证；docs/SPEC.md=行为契约+高层状态（不存详细施工步骤）；docs/plans/PLAN-XXX=详细实施路线与全过程证据。
- **CLAUDE.md**（本文件）：项目当前状态。每次会话开始先读。
- **docs/SPEC.md**：行为契约+高层状态；当前指针指向 PLAN-002（API 后端）。
- **docs/plans/**：PLAN-001（Phase 3 质量修复，reviewer 二次方案，待裁决）、PLAN-002（DeepSeek API 为主本地兜底后端，草案，首要实施）。
- **docs/REFACTORING_PLAN.md**：全项目结构重构详细设计方案（ADR-016，已实施）。
- **docs/Phase2 Review Findings.md**：Phase 2 代码审查发现（4个问题已全部修复）。
- **docs/adr/ADR-XXX.md**：架构决策完整论证（注意状态字段：已采纳/暂缓/已废弃）。
