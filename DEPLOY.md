# AlphaGameExplainer 完整部署文档

> 最后更新：2026-05-30  
> 适用系统：Windows 10/11 x64  
> 最低硬件：32GB 内存 + NVIDIA GPU（16GB 显存推荐）

---

## 目录

1. [基础环境](#1-基础环境)
2. [Python Conda 环境](#2-python-conda-环境)
3. [PyTorch（CUDA 版）](#3-pytorchcuda-版)
4. [llama-cpp-python（CUDA 版）](#4-llama-cpp-pythoncuda-版)
5. [CUDA DLL 依赖修复](#5-cuda-dll-依赖修复)
6. [其余 Python 依赖](#6-其余-python-依赖)
7. [GGUF 模型文件](#7-gguf-模型文件)
8. [Stockfish 引擎](#8-stockfish-引擎)
9. [FFmpeg](#9-ffmpeg)
10. [Syzygy 表库](#10-syzygy-表库)
11. [环境变量配置（.env）](#11-环境变量配置env)
12. [ChatTTS 模型（首次运行自动下载）](#12-chattts-模型首次运行自动下载)
13. [IndexTTS2（可选，视频模式需要）](#13-indextts2可选视频模式需要)
14. [启动运行](#14-启动运行)
15. [依赖版本速查表](#15-依赖版本速查表)

---

## 1. 基础环境

### 1.1 Conda（Miniconda）

如果尚未安装：

```powershell
# 下载 Miniconda（推荐）
# 安装版：https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
# 安装时勾选"Add to PATH"
```

验证：

```powershell
conda --version
# 应输出：conda 25.1.1 或更高
```

### 1.2 NVIDIA 显卡驱动

驱动版本需 ≥ 545，可从 [NVIDIA 官网](https://www.nvidia.com/download/index.aspx) 下载。

```powershell
nvidia-smi
# 确认能看到 GPU 名称、驱动版本、CUDA 版本
# 本项目验证环境：RTX 4070 16GB, 驱动 560.94, CUDA 12.6
```

> **注意**：本文档假设你有 NVIDIA 独立显卡。若用 CPU 推理（不推荐），需更换 llama-cpp-python 为 CPU 版（`--index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`），推理速度约 3-5 倍慢于 GPU。

### 1.3 CUDA Toolkit

**不需要安装完整的 CUDA Toolkit。** 本项目通过 pip 包提供 CUDA 12 运行时 DLL，见第 3、4、5 节。

如果你机器上已经装了 CUDA 13.x Toolkit（路径 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`），可保留，不冲突。

---

## 2. Python Conda 环境

创建独立 conda 环境，Python 3.12：

```powershell
conda create -n explainer python=3.12 -y
```

> **说明**：环境名为 `explainer`，Python 版本 3.12.13。如果用其他名字，后续命令中替换 `explainer` 即可。

激活环境：

```powershell
conda activate explainer
```

> **注意**：若 `conda activate` 在当前终端失效（PowerShell 执行策略问题），所有 `pip`/`python` 命令改用绝对路径：
> ```powershell
> C:\Users\<用户名>\.conda\envs\explainer\python.exe -m pip ...
> ```

---

## 3. PyTorch（CUDA 版）

> ⚠️ **安装顺序很重要**：必须先装 PyTorch CUDA 版，再装 ChatTTS。
> 如果先装 ChatTTS，它会自动拉取 `torch` CPU 版，之后装 CUDA 版可能不覆盖。

ChatTTS 依赖 torch。**必须装 CUDA 版**，理由同上。

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

- 版本：`torch` 2.x.x + cu124
- 大小：约 2.5 GB
- 下载源：PyTorch 官方

验证 CUDA 可用：
```powershell
python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
```

---

## 4. llama-cpp-python（CUDA 版）

**核心依赖**，负责加载 GGUF 模型做 GPU 推理。

```powershell
# ★ 推荐：直接用 GitHub Release 的预编译 CUDA wheel（最可靠）
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.23-cu124/llama_cpp_python-0.3.23-py3-none-win_amd64.whl
```

- 版本：`llama-cpp-python 0.3.23` + cu124
- 大小：约 463 MB（wheel，含编译好的 llama.cpp + CUDA DLL）
- 下载源：GitHub Release（国内可能较慢，但不会被 PyPI 源码包干扰）

> **为什么不用 `--extra-index-url`？**
> PyPI 上同名包 `llama-cpp-python 0.3.23` 只有源码 tar.gz（68MB），pip 在 extra-index 和 PyPI 之间可能优先选 PyPI 的源码包，导致触发本地 C++ 编译（需 cmake + VS Build Tools），且编译出的是 CPU 版。直接用 GitHub Release URL 可 100% 确定拿到 CUDA 预编译 wheel。

> **国内加速**：如果 GitHub Release 下载慢（463MB），开代理后重试：
> ```powershell
> $env:HTTPS_PROXY="http://127.0.0.1:7890"
> $env:HTTP_PROXY="http://127.0.0.1:7890"
> pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.23-cu124/llama_cpp_python-0.3.23-py3-none-win_amd64.whl
> ```

> **如果 pip install 直接 URL 报错**，备选方案（需科学上网稳定）：
> ```powershell
> pip install llama-cpp-python --only-binary :all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
> ```

---

## 5. CUDA DLL 依赖修复

> **这是关键步骤，缺少则 llama-cpp-python 无法加载 GPU 模型。**

`llama-cpp-python` 的 `ggml-cuda.dll` 硬链接了 `cudart64_12.dll` 和 `cublas64_12.dll`（文件名带版本号 12）。Windows 加载 DLL 只看文件名，所以必须提供 CUDA 12 版本的 DLL。

### 5.1 安装 CUDA 12 运行时

```powershell
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12
```

- `nvidia-cuda-runtime-cu12`：约 3.6 MB
- `nvidia-cublas-cu12`：约 553 MB（含 cublas + cublasLt）
- `nvidia-cuda-nvrtc-cu12`：约 30 MB

### 5.2 复制 DLL 到 llama_cpp 目录

> 以下命令在 **PowerShell** 中执行（不是 cmd）。

```powershell
$src1 = "$env:CONDA_PREFIX\Lib\site-packages\nvidia\cuda_runtime\bin"
$src2 = "$env:CONDA_PREFIX\Lib\site-packages\nvidia\cublas\bin"
$dst  = "$env:CONDA_PREFIX\Lib\site-packages\llama_cpp\lib"

Copy-Item "$src1\*.dll" -Destination $dst -Force
Copy-Item "$src2\*.dll" -Destination $dst -Force
```

> 如果用 cmd，等价命令：
> ```cmd
> copy "%CONDA_PREFIX%\Lib\site-packages\nvidia\cuda_runtime\bin\*.dll" "%CONDA_PREFIX%\Lib\site-packages\llama_cpp\lib\"
> copy "%CONDA_PREFIX%\Lib\site-packages\nvidia\cublas\bin\*.dll" "%CONDA_PREFIX%\Lib\site-packages\llama_cpp\lib\"
> ```

复制完成后，`llama_cpp\lib` 下应有以下 CUDA DLL：

| 文件 | 大小 |
|---|---|
| `cudart64_12.dll` | ~0.6 MB |
| `cublas64_12.dll` | ~98 MB |
| `cublasLt64_12.dll` | ~638 MB |

### 5.3 验证

```powershell
python -c "from llama_cpp import Llama; print('llama_cpp + CUDA OK')"
```

若无报错即修复成功。

---

## 6. 其余 Python 依赖

```powershell
pip install python-chess stockfish pyttsx3 pydub soundfile ChatTTS Pillow moviepy numpy pysrt python-dotenv colorama
```

> 建议用清华镜像加速（不须代理）：
> ```powershell
> pip install python-chess stockfish pyttsx3 pydub soundfile ChatTTS Pillow moviepy numpy pysrt python-dotenv colorama -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

各包用途：

| 包 | 用途 |
|---|---|
| `python-chess` | 国际象棋规则引擎、Syzygy/Gaviota 表库接口 |
| `stockfish` | Stockfish UCI 引擎 Python 封装 |
| `pyttsx3` | 离线 TTS 回退方案 |
| `pydub` | 音频处理 |
| `soundfile` | 音频文件读写 |
| `ChatTTS` | ChatTTS 语音合成（主 TTS 方案） |
| `Pillow` | 棋盘渲染 |
| `moviepy` | 视频合成 |
| `numpy` | 数学计算 |
| `pysrt` | 字幕处理 |
| `python-dotenv` | .env 环境变量加载 |
| `colorama` | 终端彩色日志 |

---

## 7. GGUF 模型文件

### 7.1 推荐模型（按优先级）

| 优先级 | 模型 | 文件 | 大小 | 适用场景 |
|--------|------|------|------|----------|
| 🥇 | Qwen3.6-27B Q3_K_M | `Qwen3.6-27B-Q3_K_M.gguf` | 13.6 GB | **16GB 显存最佳选择**，余量充裕 |
| 🥈 | Qwen3.6-27B IQ4_XS | `Qwen3.6-27B-IQ4_XS.gguf` | 15.4 GB | 量化质量更高，但 16GB 较紧 |
| 🥉 | Qwen3-14B Q5_K_M | `Qwen_Qwen3-14B-Q5_K_M.gguf` | ~10 GB | 速度快，适合测试/中轻度残局 |
| 备 | Qwen3.5-27B Q4_K_S | `Qwen3.5-27B-Q4_K_S.gguf` | ~15 GB | 标准 Transformer（无 SSM），老版 llama.cpp 也行 |

> **来源**：unsloth 仓库的 GGUF 不含 MTP 层，是唯一确认兼容的量化源。
> - 仓库地址：https://huggingface.co/unsloth/Qwen3.6-27B-GGUF
> - 国内镜像：`export HF_ENDPOINT=https://hf-mirror.com`（见第 12 节）

### 7.2 下载指令

```powershell
# Q3_K_M（推荐，13.6 GB）
huggingface-cli download unsloth/Qwen3.6-27B-GGUF \
    --include "Qwen3.6-27B-Q3_K_M.gguf" \
    --local-dir "D:/Program Files/HuggingFace"

# IQ4_XS（高画质，15.4 GB）
huggingface-cli download unsloth/Qwen3.6-27B-GGUF \
    --include "Qwen3.6-27B-IQ4_XS.gguf" \
    --local-dir "D:/Program Files/HuggingFace"
```

### 7.3 模型兼容性说明

> ⚠️ **这是最常见的部署故障点，请仔细阅读。**

Qwen3.6 使用 **qwen35** 架构，包含可选的 **MTP（Multi-Token Prediction）层**。MTP 层含 SSM（State Space Model）张量，需要较新版本的 llama.cpp 支持。

- **unsloth 仓库的 GGUF 不含 MTP 层** → 兼容 llama-cpp-python 0.3.23 ✅
- **bartowski / Qwen 官方仓库的 GGUF 含 MTP 层** → 需要编译最新版 llama.cpp（源码编译），0.3.23 的预编译 wheel 会报错 ❌

**如果你看到以下错误，说明模型含 MTP 层：**

```
llama_model_load: error loading model: missing tensor 'blk.64.ssm_conv1d.weight'
```

解决方案：换用 unsloth 仓库的不含 MTP 版本，或从源码编译最新 llama-cpp-python（需 VS Build Tools）。

### 7.4 显存计算

```
IQ4_XS (15.4GB) + KV缓存(4096 tokens) + 计算缓冲 ≈ 18-19 GB → 16GB 不够，会溢出
Q3_K_M (13.6GB) + KV缓存(4096 tokens) + 计算缓冲 ≈ 16-17 GB → 刚好，可能少量溢出
Q3_K_M (13.6GB) + KV缓存(2048 tokens) + 计算缓冲 ≈ 15-16 GB → 安全
```

> 如果选 IQ4_XS，建议 `.env` 中设置 `LLAMA_CPP_N_CTX=2048` 或 `LLAMA_CPP_N_GPU_LAYERS=50` 控制显存。

### 7.5 其他可选模型

| 模型 | 量化 | 大小 | 来源 |
|---|---|---|---|
| Qwen3.6-27B Q4_K_S | Q4_K_S | 15.9 GB | unsloth/Qwen3.6-27B-GGUF |
| Qwen3.6-27B Q3_K_L | Q3_K_L | 14.4 GB | unsloth/Qwen3.6-27B-GGUF |
| Qwen3-14B Q5_K_M | Q5_K_M | ~10 GB | bartowski/Qwen_Qwen3-14B-GGUF |
| Qwen3.5-9B Q4_K_M | Q4_K_M | ~6 GB | unsloth/Qwen3.5-9B-GGUF |

---

## 8. Stockfish 引擎

### 8.1 下载

从 [Stockfish 官方下载页](https://stockfishchess.org/download/) 获取 Windows x64 AVX2 版本。

- 文件名：`stockfish-windows-x86-64-avx2.exe`
- 版本：17 或更高
- 放置位置：**项目根目录**（与 `main.py` 同级），或任意路径后在 `.env` 中配置

### 8.2 验证

```powershell
.\stockfish-windows-x86-64-avx2.exe bench 16 1 1
# 应有正常输出，最后显示 "Nodes/second: xxxxxx" 即正常
```

---

## 9. FFmpeg

### 9.1 下载

从 [FFmpeg 官网](https://ffmpeg.org/download.html) 下载 Windows 版：

- 版本：8.1.1 Essentials Build（或更新的 essentials build）
- 下载页面：https://www.gyan.dev/ffmpeg/builds/
- 选择：`ffmpeg-release-essentials.zip`

### 9.2 安装

解压到固定路径，例如 `D:/Program Files/ffmpeg-8.1.1-essentials_build/`。

确保 `bin/ffmpeg.exe` 存在：

```powershell
Test-Path "D:/Program Files/ffmpeg-8.1.1-essentials_build/bin/ffmpeg.exe"
```

---

## 10. Syzygy 表库

项目已包含残局表库（`syzygy/` 目录，约 500+ 对 `.rtbw`/`.rtbz` 文件）。

- **格式**：文件为 **Gaviota** 格式（双文件 `.rtbw` + `.rtbz`），不是 Syzygy 格式。目录名叫 `syzygy` 是历史遗留
- 路径：项目根目录下的 `syzygy/`（`.env` 中 `SYZYGY_PATH=syzygy` 指向这里）
- 覆盖范围：K v K, KQ v K, KR v K, KP v K, KPP v K, KBP v K 等 ≤5 子残局
- **无需额外下载**
- **注意**：默认 `SYZYGY_PATH=syzygy` 会尝试以 Syzygy 格式读取，失败则静默跳过，求解自动回退到 Stockfish。如需启用 Gaviota 表库（≤5 子 DTM 精确求解），在 `.env` 中设 `GAVIOTA_PATH=syzygy`

---

## 11. 环境变量配置（.env）

在项目根目录创建/编辑 `.env` 文件：

```env
# ===== Stockfish 引擎 =====
STOCKFISH_PATH=stockfish-windows-x86-64-avx2.exe

# ===== LLM 配置 =====
LLM_BACKEND=llama_cpp
LLM_TEMPERATURE=0.2

# ===== llama-cpp-python (本地 GGUF 模型) =====
LLAMA_CPP_MODEL_PATH=D:/Program Files/HuggingFace/Qwen3.6-27B-IQ4_XS.gguf
LLAMA_CPP_N_GPU_LAYERS=-1
LLAMA_CPP_N_CTX=4096
LLAMA_CPP_N_BATCH=512
LLAMA_CPP_VERBOSE=false

# ===== FFmpeg =====
FFMPEG_PATH=D:/Program Files/ffmpeg-8.1.1-essentials_build/bin/ffmpeg.exe

# ===== 输出 & 表库 =====
OUTPUT_DIR=./output
SYZYGY_PATH=syzygy

# ===== TTS =====
TTS_RATE=180
```

### 关键参数说明

> ⚠️ **常见错误**：变量名是 `LLAMA_CPP_*`（驼峰命名，有 A），不是 `LLM_CPP_*`（少了 A）。代码 `llm_backend.py` 读的是 `LLAMA_CPP_MODEL_PATH` 等，写错变量名会导致所有 llama.cpp 参数被静默忽略、使用默认值，启动时报"模型路径未设置"。

| 变量 | 说明 |
|---|---|
| `LLAMA_CPP_MODEL_PATH` | GGUF 模型文件绝对路径 |
| `LLAMA_CPP_N_GPU_LAYERS` | `-1` = 全部层 offload 到 GPU（显存不足时自动溢到内存）。写入 `0` = 纯 CPU |
| `LLAMA_CPP_N_CTX` | 上下文窗口大小（tokens）。27B 模型建议 2048-4096，根据显存调整 |
| `LLAMA_CPP_N_BATCH` | 批处理大小，影响 prompt 处理速度 |
| `LLM_TEMPERATURE` | 生成温度，0.0-1.0，越低越确定 |
| `STOCKFISH_PATH` | Stockfish exe 文件名（项目目录下）或绝对路径 |
| `FFMPEG_PATH` | ffmpeg.exe 绝对路径 |
| `SYZYGY_PATH` | 表库目录，相对路径 `syzygy` 即可。目录内是 Gaviota 格式（`.rtbw`/`.rtbz`），如需启用 Gaviota 表库额外设 `GAVIOTA_PATH=syzygy` |

---

## 12. ChatTTS 模型（首次运行自动下载）

### 12.0 配置 HuggingFace 镜像（国内用户必须）

ChatTTS 首次调用会从 HuggingFace 下载模型权重（约 1-2 GB）。国内直连 HuggingFace 会被阻断（`ConnectionResetError`），必须配置镜像：

```env
# 在 .env 中或系统环境变量中设置：
HF_ENDPOINT=https://hf-mirror.com
```

> 本项目 `src/__init__.py` 已内置镜像回退逻辑，但建议显式设置。

### 12.1 手动预下载（可选）

```powershell
# 可选：手动预先下载，避免首次运行等待
python -c "from ChatTTS import Chat; chat = Chat(); chat.load(compile=False, source='huggingface')"
```

- 需要网络连接（或 HuggingFace 镜像）
- 下载一次后缓存，后续无需再下载

---

## 13. IndexTTS2（可选，视频模式需要）

IndexTTS2 用于生成高质量 TTS 语音（视频模式），需要独立的 Python 3.10 环境：

```powershell
# 进入 index-tts 目录
cd ../index-tts

# 安装 uv（Python 包管理器）
pip install uv

# 同步依赖
uv sync

# 下载模型权重
uv run modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

> 视频模式默认优先尝试 IndexTTS2，不可用时回退到 ChatTTS，再不行回退到 pyttsx3。

---

## 14. 启动运行

> 项目有两种模式：
> - `--text`：纯文本解说输出，**不需要 FFmpeg / TTS**，适合快速测试 LLM 是否正常
> - 不带 `--text`：完整视频模式，生成 `.mp4` 视频，需要 FFmpeg + TTS 全部就绪

### 14.1 纯文本模式（推荐首次测试）

```powershell
python main.py --text
```

交互式输入 FEN 或 PGN，输入 `END` 结束：

```
请输入PGN或FEN内容(输入END结束):
1K6/1P1k4/8/8/8/8/4R3/8 w - - 0 1
END
```

### 14.2 视频模式（生成 mp4 讲解视频）

```powershell
python main.py
```

需要 **FFmpeg + MoviePy + TTS（ChatTTS / pyttsx3）全部就绪**。会生成 `output/analysis.mp4`。

### 14.3 从文件批量运行

```powershell
# 纯文本模式
python main.py --text test.fen

# 视频模式
python main.py test.fen
```

### 14.4 测试用 FEN

```
# 单车杀王 — 最简单，首测用
8/8/8/3k4/8/8/8/3RK3 w - - 0 1

# 车兵对车 — 实战常见
k7/7R/8/8/8/8/5r2/K7 w - - 0 1

# 象马杀王 — 最复杂
k7/2K5/8/8/8/8/8/N2B4 w - - 0 1
```

---

## 15. 依赖版本速查表

| 组件 | 版本 | 安装方式 |
|---|---|---|
| Python | 3.12.13 | conda create |
| PyTorch | 2.x (cu124) | pip (--index-url cu124) |
| llama-cpp-python | 0.3.23 (cu124) | pip (--extra-index-url abetlen cu124) |
| CUDA Runtime | 12.9.79 | pip (nvidia-cuda-runtime-cu12) |
| cuBLAS | 12.9.2.10 | pip (nvidia-cublas-cu12) |
| Stockfish | 17 | 官网下载 exe |
| FFmpeg | 8.1.1 Essentials | gyan.dev 下载 |
| Syzygy 表库 | 6-men 子集 | 项目内置 |
| ChatTTS | 0.2.x | pip |
| python-chess | ≥1.999, <2.0 | pip |
| pydub | ≥0.25.1, <1.0 | pip |
| moviepy | ≥2.0.0, <3.0 | pip |
| numpy | ≥1.26.0, <2.0 | pip |

---

## 常见问题

### Q1: `conda activate explainer` 无效果

PowerShell 执行策略限制。临时解决：

```powershell
C:\Users\<用户名>\.conda\envs\explainer\python.exe -m pip install <包名>
```

或在 PowerShell 中以管理员运行 `Set-ExecutionPolicy RemoteSigned`。

### Q2: 安装 llama-cpp-python 时下载超时

直连 GitHub 不稳定，开代理后重试（见第 4 节代理提示）。

### Q3: `Failed to load shared library llama.dll (or one of its dependencies)`

🎯 **CUDA DLL 未正确部署**。请回第 5 节，确认 `cudart64_12.dll` / `cublas64_12.dll` / `cublasLt64_12.dll` 已复制到 `llama_cpp\lib` 目录。

### Q4: `Failed to load model from file: missing tensor 'blk.64.ssm_conv1d.weight'`

🎯 **GGUF 模型包含 MTP/SSM 层**。你下载的是 Qwen 官方或 bartowski 仓库的版本（含 MTP），需要换用 **unsloth 仓库的不含 MTP 版本**。详见第 7.3 节。

也可见于启动时日志中显示 `general.architecture = qwen35` 但加载到第 64 层时报错。

如果确实要用 MTP 版模型，需从源码编译最新 llama-cpp-python（见第 4 节备选方案末尾的说明）。

### Q5: `OSError: [WinError -1073741795] Windows Error 0xc000001d`

🎯 **illegal instruction**。CPU 版 llama-cpp-python 的 qwen35 DeltaNet kernel 在部分 CPU 上触发此错误。请装 CUDA 版 wheel（GPU 推理绕过此问题），见第 4 节。

### Q6: GPU 显存不足推理极慢

模型 15.4GB + 缓冲 ≈17GB，对于 16GB 显存的 4070，模型层部分溢出到内存是正常的。可尝试：

1. 换更小的量化：**Q3_K_M（13.6GB）是 16GB 显存的最佳选择**
2. 降低上下文窗口：`.env` 中 `LLAMA_CPP_N_CTX=2048`
3. 减少 offload 层数：`LLAMA_CPP_N_GPU_LAYERS=50`（手动控制哪些层上 GPU）

### Q7: 启动报"模型路径未设置"但 .env 明明写了路径

🎯 **变量名写错了**。代码读的是 `LLAMA_CPP_MODEL_PATH`（有 A），你的 `.env` 写的是 `LLM_CPP_MODEL_PATH`（少了 A）。详见第 11 节的红色警告。

检查方法：
```powershell
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('LLAMA_CPP_MODEL_PATH'))"
```
若输出 `None` 就是变量名不对。
