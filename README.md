# AlphaGameExplainer

国际象棋自动解说视频生成系统。给定一个棋局，自动完成「棋理分析 → 中文解说 → 语音合成 → 竖版视频渲染」，产出可直接发布的讲解视频（720×960 竖版，带 TTS 配音与同步字幕）。

设计哲学：**程序计算事实，LLM 负责表达**。所有棋盘事实由 python-chess 与 Stockfish 确定性计算，LLM 只在既定事实内组织语言，经多道校验闸拦截幻觉、坐标泄漏与事实错误——宁可不出片，也不出误导内容。

## 三条讲解管线

系统包含三条相互独立的讲解管线，各自独立的失败域：

| 管线 | 输入 | 讲什么 |
|---|---|---|
| **Endgame 残局** | 残局 FEN / PGN | 最优杀法、关键手、残局技巧 |
| **Puzzle 战术** | 战术题 | 战术机理（叉/牵制/闪击）与落地手段 |
| **Decision 决策** | 中局局面 | 多个战略计划的取舍对比 |

## 快速开始

### 环境

- Python 3.12（项目使用 conda 环境，依赖见 `requirements.txt`）
- Stockfish 引擎（仓库内含 Windows x64 avx2 版本）
- Syzygy 3-6 子表库（`syzygy/` 目录）
- DeepSeek API Key（配置在 `.env` 的 `DEEPSEEK_API_KEY`；无 key 时回退本地 llama.cpp）

```bash
pip install -r requirements.txt
```

### 运行

```bash
# 残局讲解（默认出视频；--text 只出解说文本）
python main.py path/to/endgame.fen
python main.py --text path/to/endgame.fen

# 战术题讲解
python main.py --puzzle path/to/puzzle.json

# 中局决策讲解（输入单个 FEN 文件）
python main.py --decision path/to/position.fen
python main.py --decision --text path/to/position.fen
```

不带文件参数运行残局模式会进入交互式输入（粘贴 PGN/FEN，输入 `END` 结束）。

输出视频位于 `output/` 目录。

## 目录结构

```
main.py                  # 单一入口，命令行开关切换三条管线
src/
  infra/                 # LLM 后端抽象（API + 本地兜底）、日志
  chess_utils/           # 子力/局面/战术的公共计算
  analysis/              # 棋理事实抽取、兵形识别、特征向量、知识库
  solver/                # Stockfish 求解、Syzygy 表库、分支探索、后果投射
  storyboard/            # 节点压缩、关键手定位、分镜构建
  commentator/           # 三条管线的解说生成 + 校验 + 文本清洗
  pipeline/              # 三条管线的端到端编排
  media/                 # 棋盘渲染、TTS、字幕、视频合成
data/                    # 知识库（残局/战术/兵形）、解说范例、质量基线
syzygy/                  # Syzygy 残局表库
docs/                    # 架构决策(ADR) / 实施计划(PLAN) / 行为契约(SPEC) / 工作小结
assets/                  # 棋子图片、TTS 参考音色
```

## 文档导航

- **工作小结**：`docs/工作小结-AlphaGameExplainer.md` — 项目全貌与三阶段回顾
- **项目记忆**：`CLAUDE.md` — 领域语言、架构决策索引、硬约束、当前状态
- **行为契约**：`docs/SPEC.md` — 高层状态与计划索引
- **架构决策**：`docs/adr/` — 各项技术决策的完整论证
- **实施计划**：`docs/plans/` — 各阶段详细实施路线与证据

## 测试

回归测试网与研发探针在 `test` 分支（`tests/` + `tools/`）：

```bash
pytest tests/ -m "not slow"    # 快速回归
pytest tests/                  # 全量（含引擎/TTS 慢速用例）
```
