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

## 部署（迁移到新电脑）

按顺序执行。前置依赖（Python 环境、外部程序）必须先装好，否则运行会报错。

### 1. 系统前置依赖

| 依赖 | 说明 | 验证命令 |
|---|---|---|
| **Python 3.12** | 建议用 conda 建独立环境，避免污染系统 Python | `python --version` |
| **FFmpeg** | moviepy 合成视频必需。安装后加入系统 PATH，或在 `.env` 填绝对路径 | `ffmpeg -version` |
| **Stockfish** | 仓库已含 Windows x64 avx2 版（`stockfish-windows-x86-64-avx2.exe`）。非 Windows 或非 avx2 CPU 需自行下载对应版本 | — |

> **CPU 注意**：仓库自带的是 avx2 版本。若目标机 CPU 不支持 avx2（较老机型），Stockfish 会启动失败，需从官网下载对应指令集版本替换，并更新 `.env` 的 `STOCKFISH_PATH`。

### 2. 建环境、装依赖

```bash
# 建 conda 环境（推荐）
conda create -n explainer python=3.12
conda activate explainer

# 装 Python 依赖
pip install -r requirements.txt
```

> **TTS 依赖版本敏感**：ChatTTS 经 numba/torchaudio 硬依赖 torch 与 numpy，版本错配会导致 TTS 初始化失败、视频链路卡死。若视频链路报 TTS 相关错误，确认 `torch`/`torchaudio`/`numpy` 三者版本配套（详见 `CLAUDE.md` 硬约束段）。只出文本（`--text`）不加载 TTS，可先用文本模式验证主链路。

### 3. 配置 `.env`

项目根目录的 `.env` 存放密钥与本机路径，**不进版本库**。从模板复制一份再填：

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

然后编辑 `.env`，**至少**填这几项：

| 变量 | 必填 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 是 | DeepSeek API 密钥；解说生成的主后端 |
| `STOCKFISH_PATH` | 是 | 引擎路径。用仓库自带版本时填相对路径 `stockfish-windows-x86-64-avx2.exe` 即可 |
| `SYZYGY_PATH` | 否 | 残局表库目录，填 `syzygy`。缺失时残局走 Stockfish 求解，不影响运行 |
| `FFMPEG_PATH` | 否 | FFmpeg 已加入系统 PATH 时留空即可；否则填 ffmpeg 可执行文件的绝对路径 |

> 路径变量支持相对路径（自动相对项目根解析）或绝对路径。跨机部署优先用相对路径。

### 4. 验证部署

```bash
# 最轻量：只出解说文本，不碰 TTS/视频，先确认引擎 + API 通
python main.py --text path/to/endgame.fen

# 完整链路：出视频（需 FFmpeg + TTS 就绪）
python main.py path/to/endgame.fen
```

文本模式跑通说明「引擎求解 + API 解说」链路正常；视频模式跑通说明「TTS + FFmpeg 渲染」链路也正常。

## 运行

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

不带文件参数运行残局模式会进入交互式输入（粘贴 PGN/FEN，输入 `END` 结束）。输出视频位于 `output/` 目录。

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
