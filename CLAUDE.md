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
- **Decision 决策管线 / 多战略意图讲解**（**第三条管线，ADR-020，生产入口已落地 PLAN-011**）：给定中局局面，识别兵形原型→取文献战略计划→`searchmoves` 方向约束产出各计划执行线→反向 MultiPV 验证方向价值→后果投射+代价量化→比较式叙事。与 Endgame/Puzzle 平行，独立失败域。`python main.py --decision <fen文件>` 出视频，`--decision --text <fen文件>` 只出解说文本（共用 `_decision_core`，SPEC §8 放弃闸两路径一致）。产品池 6 原型（carlsbad/iqp/hanging/maroczy/majority/benoni）+ 2 停用（stonewall/dutch_kid，`in_production:false`）。载体从 v1「Puzzle 末端展望」推翻而来。
- **战略 = 对未来结构类型的偏好**（ADR-020 立场 A）：绕开「战略意图无法从着法唯一恢复」的认识论死结——不恢复心理意图，只判定结构偏好方向，未来结构全部 python-chess 可算。
- **direction / structural_goal 双字段**（ADR-020 R1）：`direction` 是着法打分谓词（选根着进 searchmoves），`structural_goal` 是结构状态谓词（验「这条线真实现了计划」）。必须拆——真实计划执行序列不是每着同向，用着法方向一致率验线任何真实计划都过不了。
- **单一事实来源四函数**（ADR-020 R1）：`direction_score` / `structural_features` / `line_features` / `equivalence_gap`。「方向」「等强」「结构特征」三概念全链路唯一定义，禁止各模块各写一套（否则挖矿口径与运行时口径脱节）。
- **四类对比轴**（ADR-020 R1 + ADR-021 扩展）：轴 1 两个计划（等强，主力）/ 轴 2 同计划两种执行时机（等强，补充，**代码未启用**）/ 轴 3 执行 vs 等待（**等待必然更差，只做一句话铺垫、不独立成对比段、不用等强措辞**）/ 轴 4 正选 vs 更差的合理替代（ADR-021，**对照必然更差**：仅 n_feasible==1 降级触发，gap∈(0,150]cp，机制闸淘汰者永不进对照，空差分即放弃退回单线；措辞限权全机判——量级词程序注入、锚词正面校验、等强/灾难措辞硬错误）。
- **终局差异化gate / 事后命名 / 融入式解说**（**已废弃**）：v1 多战略链路子组件，随 ADR-006~009 被 ADR-020 收束/废弃。
- **项目结构重构**（**已完成，ADR-016**）：按管线分层将 commentator.py(2752行)/storyboard.py(2078行) 拆分为 8 个子包（infra/chess_utils/analysis/solver/storyboard/commentator/pipeline/media），消除12+处函数级重复，单向依赖。重构不改运行时行为，用回归样本验证逐段语义等价。

## 架构决策

- **[ADR-006]【被 ADR-020 收束】多战略计划采用伪逻辑路线** —— 保留「不强求算法理解真正人类战略」的哲学内核；载体从「多分支终局对比」改为「中局战略取舍教学」。
- **[ADR-007]【被 ADR-020 废弃】分支探索引擎 Lc0 为主** —— ADR-020 用 SF `searchmoves` 方向约束（前向）+ MultiPV（反向），**Lc0 彻底移出方案**（不提供不可替代价值，且背着「policy 多样性=战略多样性」未验证假设）。
- **[ADR-008]【被 ADR-020 废弃】多计划对比采用融入式解说结构** —— ADR-020 用独立管线的比较式叙事（诊断→提问→计划甲→回溯→计划乙→对比），既非融入式也非末端展望式。
- **[ADR-009]【被 ADR-020 废弃】事后命名采用API预标注+Faiss检索库** —— ADR-020 用 structure_kb 封闭词表（棋类文献权威来源），战略名先于搜索已知，无事后检索命名。
- **[ADR-010]【已废弃】解说质量提升采用五层分层架构** —— 原L1→L5（Narrative Planner→DS-MHP-lite→两阶段生成→评分器+BoN→Playbook进化）整体架构已被"上游算法层优先"路线取代；验证优先原则保留，但L2-L5在truth错误率达标前不启动代码开发。
- **[ADR-011]【已废弃】DS-MHP-lite动态证据路径替代静态RAG** —— 被 ADR-017 取代。内容型证据路径在 Planner 已给叙事骨架后边际收益低、复杂度高，且属内容注入有 GIGO 风险；解说提质改走 ADR-017 提示词工程路线。从未落地代码。
- **[ADR-012]【已采纳·已实施】Narrative Planner程序化叙事规划** —— storyboard阶段计算tension_score(0-1)/narrative_role(setup/build_up/climax/falling_action/resolution)/动态字数预算/tone_hint。纯Python计算无LLM调用，代码已落地于 src/analysis/insight_extractor.py(_compute_tension) + src/storyboard/endgame_builder.py(_assign_narrative_role) + src/commentator/endgame_commentary.py(prompt注入)。tension归一化分母40->80（climax档溢出修复）。
- **[ADR-013]【已废弃】程序化多维评分器+Best-of-N选优** —— 多输出选优对解说提质增量小，反而拉长生成时间并引入生成/评分的额外问题，弃用。**注**：真值合法性核查维度早已剥离落地于 validators.py/审计工具（属 ADR-015 范畴），不随本决策废弃而移除。
- **[ADR-014]【已废弃】人工门控Playbook进化，QLoRA降级为长期固化** —— 机制重、有把错误模式固化进context/权重的风险，且解说提质ROI低于提示词工程/范例库路线，弃用。
- **[ADR-015]【已采纳】上游真值修复采用"前置事实注入+后验硬事实核查"双保险** —— storyboard/insight每节点算确定性事实（子力/吃子/升变/将杀/per-step material delta）注入prompt白名单；commentator校验链路后验核查棋子存在性/将杀位置。代码已落地，上游全量验证通过（30残局+39 Puzzle零问题）。
- **[ADR-016]【已采纳并已实施】全项目结构重构——按管线分层拆分巨型文件** —— 将 commentator.py(2752行)和 storyboard.py(2078行)按管线拆分为子包；提取 chess_utils/ 消除12+处函数级重复；单向依赖禁止循环。重构不改运行时行为，用回归样本验证逐段语义等价。**遗留**：ADR-016 中 P7「docs/CLAUDE.md/benchmark纳入版本控制」未执行，.gitignore 仍排除这些路径，待处理。
- **[ADR-020]【已采纳·R1 已修订，待动工】第三条决策管线——多战略意图讲解** —— 新建 Decision 管线（与 Endgame/Puzzle 平行），中局输入 + structure_kb 知识层 + 正向 searchmoves 方向约束 + 反向 MultiPV 交叉验证 + 后果投射 + 比较式叙事。**同时收束 ADR-006（内核保留）、废弃 ADR-007/008/009**。R1（2026-08-02）三项修订：输入主源改 Lichess Elite DB（PGN）、KB schema 拆 `direction`/`structural_goal`、对比轴扩展为三类（轴 3 限权）；新增单一事实来源与颜色归一化两条横切约束；知识层两级降一级（失衡轴不做）。方案见 PLAN-009，评审见 FINDINGS-002。动工前有三道闸门（M5 冒烟 → P0-lite/full → 挖掘 Go/No-Go）。
- **[ADR-021]【已采纳·已实施，PLAN-012 完成】轴 4「为什么不那样走」非对称对比降级形态** —— 扩展 ADR-020 架构决策 9：新增第四类对比轴（正选 vs 更差的合理替代），轴 1/2/3 不变。触发严格降级（仅 n_feasible==1）；对照两层来源：KB 池内仅 gap 被拒计划（mech_ok=True 且 gap∈(80,150]）→ MultiPV 次优 K2（异 zone、gap∈(0,150]、后续 4 着无 ≥200cp 骤降）；空差分即放弃退回单线。背景：PLAN-010 实测双计划过闸率 7.66%，轴 1 等强对比只覆盖少数局面。措辞限权：gap 量级中文分档程序注入（文本零数字）、「更差」锚词正面校验、等强三词（各有取舍/各有侧重/看你风格）与灾难定性为硬错误、对照段 ≤ 正选段 60% 字数。质量门：上线当轮重钉快速门基线，慢速门从单计划局面池定向抽样。
- **[ADR-022]【已采纳·已实施阶段0-4.1，PLAN-013】P16 翼向扩维与子空间 A3 测量仪器修复** —— PLAN-012 证伪 H1 后确立「天花板在测量」：①P16 维度集 append-only 扩 3 翼向维（`mover_qside/center/kside_pawns_past_mid`，12→15，旧聚合维保留、旧 goal 零改动；**已实施** main `0e979cd`）；②子空间 A3 成 gate 度量唯一口径（两计划 goal 声明维并集求距离，denoise 引擎续走趋同；全空间距离旁路落盘不进 gate；生产侧 P8/unique_facts 保持全空间；**已实施** a3_subspace_metrics.py + run_a3_separability 改造）；③生产侧漂移是预期变更（PLAN-012「轴 1 零回归」口径解除，改「内容质量不降级」抽验）；④维度集边界 ≤20 + 追加三件套。**结果**：benoni 解禁（A2+A3 过闸，5→6 在产）；stonewall A3 margin 0.708 显著通过但 A2 中心突破未达+待用户确认维持停用；dutch_kid 新原型识别器定稿（33 局面 9.4%）但 A3 子空间 5/5 证伪（翼向推进型中局续走趋同）→ in_production:false。effdims `>=` 系判据修正（progress≥1→max>=target）+ τ=0.2 定档（REV-002 planner 裁决已落地：连带加严地板为「每计划≥1」、stonewall 维持停用）。引擎复跑 5/9 通过=hanging+majority_a/b+iqp_holder+benoni，134 测试零回归。

## 硬约束

- **硬件**：4070 Ti Super 16G 显存。Qwen3.6-27B 4-bit（~14.5GB）几乎占满显存，**无法与 ChatTTS（~2GB）同时驻留**，必须串行加载/释放（管线已用 `release_backend()` 在 TTS 前释放 LLM）。Decision 管线（ADR-020）只用 Stockfish（CPU，不占显存），**Lc0 已移出方案**，故无显存竞争；但引擎调用须在 LLM 加载前完成（现有 puzzle/endgame 管线的既有次序天然满足）。
- **n_ctx=4096**：本地 llama.cpp 当前默认生成上下文窗口，受 Qwen3.6-27B 4-bit 占满显存后 KV cache 余量所限，可经 `LLAMA_CPP_N_CTX` 覆盖。非硬产品上限；prompt 仍需预算控制。切到 API 后端后此约束不适用于 API 路径（DeepSeek 128k 窗口），仅本地兜底路径受限。
- **讲解词中文纯净化**：voiceover 禁止英文/数字/坐标/Markdown。已有 `_strip_coordinates` / `_clean_cjk_text` 兜底。
- **Syzygy 表库覆盖 3-6 子**：超出范围的残局走 Stockfish 求解。
- **Python 运行环境**：项目用 conda 环境 `explainer`（`C:\Users\11487\.conda\envs\explainer`，2026-08 迁移到本机后适配，PLAN-010 F7/阶段 7 实测确认）。**历史**：迁移前旧机器用环境名 `commentary`（`C:\Users\LiuYiJie\.conda\envs\commentary`），文档旧版曾记该名，已不符。系统默认 `python`（WindowsApps stub）无依赖、无输出，**不能直接 `python` 跑项目代码**。命令行必须用 `"C:\Users\11487\.conda\envs\explainer\python.exe"`、`conda run -n explainer python ...` 或先 `conda activate explainer`。`chess` / `pydub` / `chatTTS` / `pytest` 等依赖仅装在该环境。**torch/torchaudio/numpy 版本配对（PLAN-011 阶段4 实测）**：用户级 site-packages（`AppData\Roaming\Python\Python312`）若装了与 conda 环境不同源的 torch/numpy，会**遮蔽** conda 配套版本导致 ABI 不匹配——ChatTTS 经 numba→numpy、dvae→torchaudio 硬依赖，任一错配 TTS 初始化失败、视频链路卡死。正确配对：`torch 2.6.0+cu124` + `torchaudio 2.6.0+cu124` + `numpy 2.4.6`（满足 numba ≤2.4 要求）。2026-08-09 已卸载用户级遮蔽的 torch 2.13.0+cpu / torchvision 0.28.0 / numpy 2.5.1 恢复配对。装包前确认 `import torch; print(torch.__version__, torch.__file__)` 指向 conda 环境（非 AppData\Roaming）。

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

**Phase 4 结构可靠性修复已完成**（2026-07-21，PLAN-003 收口）：API 接入后解说词质量大幅提升（0 泄漏、0 硬事实错、套话密度降 54%），但首轮全量报废率 27.5%（19/69）。经 B+（validator 跨节点历史宽容）+ B1（去 sub_endgame 门）+ C1a（segment 数量对齐 id-only）+ B2（过短阈值 48→44 宽松化）四阶段修复，**报废率降至 13.0%（9/69）**，残留失败全可解释（真幻觉 6 + 过短 2 + 坏JSON 1）、零 validator 假阳性、零结构类。原「≤5%」阈值经 untangler 总裁决不再作硬收口门槛。**PLAN-003 收口，PLAN-001 解锁待启动**——残留真幻觉修复（提"后"但局面无后、提前宣称将杀、时间线错位）+ 真内容不足（19 字类）属模型/prompt 范畴，转入 PLAN-001。

**PLAN-001 作废 + PLAN-004 重写并冻结**（2026-07-21）：PLAN-001 因基线过期、阶段 C 与 §8 冲突，经 adversary 重写为 PLAN-004。REVIEW-002 修正重心（质量抬升为主线）+ 删除幻影靶。REV-003 按用户反馈收缩——phase4.2 已是有效基线（B2 仅改阈值 48→44），已有 E1/E2 基础不需要大设计，删除遥测/审计v2/统一校验/fallback哲学/模板修复等过度工程化阶段。**最终 4 阶段：A（提示词优化：正向引导+衔接+针对性修正）→ B（幻觉减少：absent_pieces 负面事实 + prompt 约束）→ C（回归验证）→ D（范例消融，可选）**。**阶段 A、B 已实施完成**（2026-07-21，developer；阶段 A 收尾追加 prelude O-O 王车易位事实错误 bug 修复）。

**PLAN-004 收口**（2026-07-22，常规优化最后一个总结性计划）：A/B/C/C1/C2 全部完成。阶段 C phase5 全量回归 + subagent 独立抽验暴露两个必修缺口并修复：**C1**（main `9a8e43e`）validator 后字假阳性修复（"左后/最后/侧后方/吃兵后"等 8 个合法叙述不再被误杀）+ 套话删除残句清理；**C2**（main `73fb968`）endgame 链路补表层泄漏硬闸（phase5 终稿实测 KBNvK_5"B限制了"/KRvK_2"a线"真实泄漏溜进成片，补齐后 SPEC §8 零泄漏从"尽力删除"变"硬保证"，对齐 puzzle 口径）+ leaks 门 bug（forbidden 步号伪影）。**phase_final 全量实测：63 成功/6 失败，aborted 率 8.7%**（优于 phase4.2 基线 13.0%），零泄漏（3 个真实坐标泄漏被 C2 表层闸正确拦截，宁丢片不泄漏）、零硬事实错误进片。**残留 6 失败全可解释**：3 真实泄漏（C2 拦截，正确）+ 升变预期叙述（004LZ，合理可能性，API 时代应放宽）+ 提后无后（003aS，非确定性真幻觉）+ 边缘过短（KBBvK_1 差 4 字）——均属模型能力边界或下一代方案范畴。

**PLAN-006/007 独立审查与修复**（2026-07-28，REVIEW-002）：对两阶段方案+实施全面审查，9 个问题已修复或记录。**PLAN-006 音画缺陷**：V1 pivotal 辉光 ×1.4 增强被 `_draw_glow` 的 `min(1.0)` 截断——三轮回退后仅剩的两个视觉效果之一实际失效（新增 `GLOW_MAX_INTENSITY=1.4` + alpha 封顶 255）；V2 slide_sec ±0.1s 低于运动感知阈值，planner 对 M4 的裁决（0.60/0.30）从未落地；A1 `speech_duration_s` 扣了 `pre_s` 但字幕起点未偏移，pivotal 段字幕早出 0.2~0.4s（`pre_silence_s` 死字段已写回并被字幕消费）；A3 important/routine 的 speed 都是 5、pre_s 都是 0，三档实际塌成两档。**PLAN-007 安全边界**：P1 润色绕过 validator——它跑在链路最后、作用于成片文本，却是唯一无 §8 舍弃通道的一环，写坏直接进片，且 REVIEW-001 标称「已修 validator 重验」而代码实无（现复用 `config.validate_chunk` 单段重验，不通过保留原文）；P2 `_should_skip` 对 Fallback/API 后端恒 False（现显式检查熔断/永久失败）；P3 puzzle 的 prev_voiceover 注入计划有而代码无（已补齐）。**E2 遗留**：emphasis 三档分布的 ≤65% routine 闸门属阶段 A 要求的验证步骤，从未执行，仍待补。

**解说词质量遗留问题**（2026-07-28）：`docs/FINDINGS-001-解说词质量遗留问题.md`——detect 模式实测（双象杀王）暴露 3 个润色按设计管不了的问题：总结词双开场语气词+口语风格断层（post_process 范畴）、节点把 `spatial_change` 逐着真值念成数字流水账且为全片最长段、`strip_coordinates` 清洗后残句「白方通过的走位」。三者属 prompt/post_process 层，已全部纳入 PLAN-008（分别为 F1/F2/F3）。**更正**：原记录称流水账段为「routine 节点、印证 emphasis 详略未生效」，经 PLAN-008 REVIEW-002 实测证伪——现场段实为 `important`/`pivotal` 档长多着节点（mc=8，traj_len=9），且 trajectory 注入条件（`len>=3`）根本不按 emphasis 分级，与 emphasis 详略无关。

**PLAN-008 立项与实施完成**（2026-07-28）：修 PLAN-006/007 遗留的表达层缺陷（F1 总结词双开场 / F2 数字流水账 / F3 坐标残句 / F4 puzzle 开场白僵硬）。阶段 A·A1（`_SPOKEN_OPENERS` 剥离）+ A2（`_THROUGH_PREP_DE` 悬空介词清理）已落地；**阶段 B 经 REVIEW-002 实测重写并冻结后已实施**——原方案「routine 不注入 trajectory」的核心前提被实测证伪（F2 源头是 important/pivotal 长多着节点），改为**空间过程形态化注入**：`summarize_trajectory`（insight_extractor.py）把逐着数字串 `2→3→3→…→1`+"必须按这个顺序如实叙述"（实测覆盖 204/221=92.3% 节点）换成「起止真值+形态词（单向/有回升/波动/持平）+区域锚点」，全 emphasis 档统一。原型+真实接入全量实测 **221/221 PASS**，peer_review 修复 1 major（无轨迹谎报波动）+6 minor。B2 header 信条禁逐着罗列格数（含中文数词），B5 `scan_process_enumeration` 审计标记（只标记不判废，基线复现 15.3%）。**阶段 C 已实施**：`generate_puzzle_intro`（puzzle_commentary.py）LLM 生成+模板兜底，同战术标签两次开场白不再雷同（修复 F4）；实施中发现 `_puzzle_intro_is_bad` 不能复用 segment 的 thinking_leaks 校验（"接下来我"误杀合法开场过渡语），改为开场白只查硬特征。--text 实跑 3 残局+2 puzzle 并做新旧代码对照，确认 F1/F2/F4 生效。**同时清掉 PLAN-006 E2 验证债**：emphasis 三档实测 pivotal 29.0%/important 36.2%/routine 34.8%，≤65% 闸门 PASS。**新发现遗留 F5**（解说词缺宾语残句"登上，"+"那一格"突兀，新旧代码都有非本次回归）+ **B7**（pivotal ≤3 上限在 2/30 样本失效，属 PLAN-006 范畴）已记录待 planner 评估。code-developer 自审闭环完成，待用户全量验证+reviewer 复盘。

**reviewer REVIEW-003 + 修缮收尾**（2026-07-28）：reviewer 判路径 B（F2 未解决，起草 PLAN-009 全域数字关闭+validator 硬拦截）。code-developer 基于用户「放权、改善而非修复、容忍瑕疵」哲学重新校准：实测证实 reviewer 的 28.6% 用了过宽口径（把单次起止对比「从三个减到两个」也算罪证），而这类是**含棋理、可容忍**的；F2 真病灶（多步过程枚举）**已被 B1 切断**。倾向**不采纳 PLAN-009 收紧方向**（会损失 teaching_point/must_mention 棋理真值，解说退回空话）。修缮：C1 echo 检测补齐（reviewer R-5）；prompt 强化"禁那一格"实测反效果（负面提及强化反模式）已回退；`_MOVE_TO_COORD` 脱节仅 2% 容忍不改。三个阶段均判定主要目标达成。遗留 F5（坐标清洗残骸可容忍）+ B7（pivotal 上限）+ PLAN-009 草案待 planner 裁决。**教训**：负面 prompt 提及会强化被禁内容——禁用某词时不要在指令里反复出现该词。

**视频视听优化立项 PLAN-005**（2026-07-25，主指针）：解说质量常规优化收口后转向「解说如何被更好地听见、看见」。三条必要改动，均限定媒体层（`src/common.py` + `src/media/*`）加法式扩展，不重构、不换框架、不动解说生成。**Core 1**（字幕真实语音同步）：TTS 记录真实语音截止时长 `speech_duration_s`（不含尾静音），字幕预算从被渲染器覆盖的 `duration_s` 改用 `speech_duration_s`，修末条字幕拖入尾静音的真实 bug；不追句级精度（TTS/字幕分句逻辑不一致，直通会错位）。**Core 2**（画面整体质量）：箭头 2x overlay 抗锯齿 + glow 改高斯柔光 + 格子高亮改柔和圆角环 + 棋子/棋盘投影缓存，按元素分治不做统一超采样。**Core 3**（落子结果澄清）：将军画攻击线（红虚线，区别实线走子箭头）+ 将杀标王无逃生格（低干扰半标记），均 python-chess 确定性计算。砍掉音效（空资产+盖人声）与战术射线（易错）。经 external validate_approach（longcat）判「有条件推荐」，两前提已纳入设计。

**第三条决策管线立项 + 两轮评审修订**（2026-08-01 立项，2026-08-02 修订，ADR-020 + PLAN-009 + FINDINGS-002）：多战略意图讲解从「Puzzle 末端展望」（v1，废弃）演进为「独立第三条 Decision 管线」（v2）。理论地基：战略 = 对未来结构类型的偏好（可计算/可对比/可前向），正向 searchmoves 方向约束 + 反向 MultiPV 交叉验证。**FINDINGS-002 两轮评审后用户裁决三项重量级修订**：① 输入主源从 Lichess puzzle 库改 **Lichess Elite DB（PGN）**——puzzle 库的入选原理（解题方每步唯一最佳）与 M5「无强制战术」结构性冲突，原 17.7% 测的是 themes 标签而非 M5；② KB schema 拆 `direction`（选根着）/`structural_goal`（验线）——一个字段无法同时承担着法打分与结构状态判定；③ 对比来源扩展为三类轴（两个计划 / 同计划两种执行 / 执行 vs 等待，轴 3 限权只做一句话）。新增两条横切约束：**单一事实来源**（direction_score/structural_features/line_features/equivalence_gap 四函数全链路唯一）+ **颜色归一化**（`board.mirror()`，KB 只写走子方视角）。知识层由两级降一级（失衡轴不做，PGN 源矿脉无限、识别不到直接丢弃）。**动工前三道闸门**：M5 冒烟（半天，验证换源判断）→ P0-lite（纯静态零引擎，谓词召回 ≥90%/识别 ≥70%/候选覆盖实战 top-2 ≥80%）→ P0-full（带引擎，污染检查/结构目标达成/可分离性自校准）。P0 判据已全部**免人工标注化**（用 PGN 实战频率统计替代评审标注）。含最小可行路径（2 原型 + 只做轴 1 + 单月数据 + 只出文本，两周 demo）对冲工程量膨胀。当前状态：**待启动，主线仍是 ADR-017**。

**工具链**：`tools/quality_audit/` 质量审计工具（默认流程已取消 AI 标注，仅生成 + 人工查看）。`tools/decision_probe/`（**仅 test 分支**）：决策管线探针——`p0_full_probe`（A2/A3）、`stage4_dualplan_probe`（双计划筛）、`maroczy_e2e_verify`（端到端）、`quality_gate`（质量门槛两层门，PLAN-011 阶段0）。

**Decision 管线实施进展**（PLAN-009/010/011，2026-08）：
- **PLAN-009**（阶段 1-9，已完成）：ADR-020 决策管线从立项到可用——M5 冒烟/P0-lite/P0-full 三闸门通过、KB 6 原型落地、引擎确定性根治、机制闸、实战对照段、mover_side 角色闸。
- **PLAN-010**（已完成）：KB 质量提升——P16 十二维特征向量定稿、direction/structural_goal 双字段、可分离性多局面实测。关键结论：**carlsbad/hanging/maroczy 的 paired/A3 临界是引擎续走趋同的系统性问题**（wholeline spike 实证约束越强越收敛），是 KB 计划/goal 设计属性非执行保真度属性——记 known limitation 不强修。Tier A 措辞（"这一局里他选的是 X"）已在 PLAN-009 阶段9 落地。
- **PLAN-011**（阶段 0-3 完成 `24efb6d`，阶段 4 收尾中）：交付质量冲刺——质量门槛两层门（快速门确定性/storyboard 真比较式率、慢速门段级缺失率）+ 存量 5 原型质量结论（majority 改善、iqp partial、carlsbad/hanging known limit）+ 新增 Benoni 原型但 A3 全不过判 `in_production:false`（P16 十二维翼盲，诚实证伪产 12 维可表达性缺口报告）+ `--decision` 生产入口（文本/视频双路径，拆分 `_decision_core`）。**阶段 4**：2026-08-09 实测 `python main.py --decision <fen>` 产出完整正片（`output/decision_hanging_75ac1d.mp4` + sidecar，四维达标）；修 TTS 环境冲突（用户级 torch 2.13.0+cpu / numpy 2.5.1 遮蔽 conda 配对版本，见硬约束）；慢速门基线补跑中。
- **PLAN-012**（**已完成** 2026-08-12，main `5941486`，ADR-021）：KB-free 验证证伪 H1「只有少数结构有讲解价值」——等强异 zone 首着对保守口径 51.7%（首选 vs 次选 gap≤60）、宽口径 92.7%（top-5 任意对），战略分岔素材不稀缺，**天花板在测量（P16 翼盲）与方法（KB 九闸），不在棋本身**。轴 4 非对称对比全链路落地：n_feasible==1 时 K2（MultiPV 次优异 zone，主供源 87.5%）或池内层（16.1%）造「更差但合理」对照，12.5% 无源退单线；解说段级语义（对照段失败静默退回单线不阻塞出片）；4 样本 text 验证合规、轴 1 零回归。验收债（快速门重钉/慢速门轴 4 定向）由 PLAN-013 吸收。
- **PLAN-013**（**阶段 0-4.1+REV-002 完成** 2026-08-14，ADR-022；main `0e979cd`+`8afdac1`+`4751ac3`，test `c8be7ae`+`10c5fb6`+`9745eee`+`30b8783`）：测量仪器修复与 KB 战略储备扩充——P16 v2 翼向扩维（12→15 append-only）+ DIM_CN 单一来源 + 子空间 A3 v2（gate 度量 denoise，a3_subspace_metrics.py）→ benoni 解禁（A2+A3 过闸 5→6 在产）/ stonewall 复测（维持停用）/ dutch_kid 新原型（33 局面 9.4% 命中但 A3 5/5 证伪 in_production:false）→ K2 gap 下界裁决（保留 0）+ R3 自然吸收。**REV-002 planner 裁决三点已落地**：①effdims `>=` 系判据修正接受（progress≥1→max>=target，连带加严地板为「每计划≥1」hanging 翻转通过/stonewall 拦截单边分离）；②τ=0.2 定档（P25 依据 + round 浮点保护）；③stonewall 维持停用（margin 0.708 但单边分离 + 程序性条件未满足）。引擎复跑 5/9=hanging+majority_a/b+iqp_holder+benoni。134 测试零回归。**待后续**：4.2 存量抽验（需 LLM）/ 4.3 质量门重钉+对照表 / 4.5 端到端（用户后续安排）。

## 项目结构

按管线单向分层（ADR-016），详细目录读代码为准，不在此复制。核心层：
- `src/infra/`：`llm_backend.py`(后端抽象+单例+release)、`logger.py`
- `src/chess_utils/`：material/position/tactic（消除函数级重复）
- `src/analysis/`：`insight_extractor.py`(棋理事实+_compute_tension)、`endgame_kb.py`、`themes_kb.py`、`structure_id.py`(兵形原型识别)、`structure_features.py`(P16 十二维)、`direction.py`(方向打分)
- `src/solver/`：`stockfish_analyzer.py`(单线三阶段)、`tablebase.py`(Syzygy)
- `src/storyboard/`：compressor/`key_move_locator.py`/`endgame_builder.py`(_assign_narrative_role)/puzzle_builder/prelude/`decision_builder.py`(决策比较式叙事)
- `src/commentator/`：endgame/puzzle/`decision_commentary.py`(决策解说) 、`generator.py`(主流程)、`validators.py`、`text_filters.py`、`grammar.py`(GBNF)、`json_utils.py`
- `src/pipeline/`：endgame(5步)/puzzle(4步)/decision(识别→引擎→storyboard→解说→TTS→渲染) 三管线；`src/media/`：渲染/TTS/字幕/合成
- `data/`(`structure_kb.json` 决策知识库 / endgame_kb/puzzle_themes/commentary_examples/`quality_benchmark_decision`/quality_benchmark_phaseN)、`syzygy/`、`tools/`、`main.py`、`docs/`(SPEC/plans/adr)

## 文档与记忆系统

- 四层文档职责：CLAUDE.md=长期领域语言/架构决策索引/硬约束/当前状态；docs/adr/=决策论证；docs/SPEC.md=行为契约+高层状态（不存详细施工步骤）；docs/plans/PLAN-XXX=详细实施路线与全过程证据。
- **CLAUDE.md**（本文件）：项目当前状态。每次会话开始先读。
- **docs/SPEC.md**：行为契约+高层状态。
- **docs/plans/**：PLAN-001（Phase 3 质量修复，**已作废** 2026-07-21，后继 PLAN-004）、PLAN-002（DeepSeek API 为主本地兜底后端，已完成）、PLAN-003（Phase 4 结构可靠性修复，已完成）、PLAN-004（API 时代解说质量修复与验证闭环，**已完成** 2026-07-22 收口）、PLAN-005（视频生成视听优化，**已完成** 2026-07-27 收口）、PLAN-006（解说节奏与情感表达优化，执行中，待端到端观感验证）、PLAN-007（解说词二次润色，执行中，待切 `ENABLE_POLISH=true`）、PLAN-008（表达层精修与约束松绑，**已完成** 2026-07-28）、PLAN-009（多战略意图讲解第三条决策管线，**已完成** 阶段1-9）、PLAN-010（决策管线阶段9遗留问题修复与KB质量提升，**已完成**）、PLAN-011（决策管线交付质量冲刺与战略储备扩充，**已完成** 2026-08-09 收口）、PLAN-012（Decision 扩张验证与轴 4 非对称对比，**已完成** 2026-08-12，H1 证伪 + 轴 4 全链路）、PLAN-013（Decision 测量仪器修复与 KB 战略储备扩充，**阶段 0-4.1 完成** 2026-08-13，benoni 解禁+dutch_kid 证伪+子空间 A3 v2）。
- **docs/FINDINGS-002**：多战略意图讲解方案评审与遗留问题（P0 双闸门判据 + P1~P23 问题清单），PLAN-009 各阶段动工前逐条回查。
- **docs/HANDOFF-001/002**：决策管线遗留问题与交付验收交接台账（不进 git，会话间移交用）。
- **docs/REFACTORING_PLAN.md**：全项目结构重构详细设计方案（ADR-016，已实施）。
- **docs/Phase2 Review Findings.md**：Phase 2 代码审查发现（4个问题已全部修复）。
- **docs/adr/ADR-XXX.md**：架构决策完整论证（注意状态字段：已采纳/暂缓/已废弃）。
