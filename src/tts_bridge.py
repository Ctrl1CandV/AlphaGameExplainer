"""TTS 桥接模块 - 通过 subprocess + JSON 文件调用 IndexTTS2 (独立 Python 3.10 环境)"""
import subprocess
import json
import os
import sys
from src.common import Logger

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INDEX_TTS_DIR = os.path.abspath(os.path.join(_PROJECT_ROOT, "..", "..", "index-tts"))
_TTS_SERVER = os.path.join(_INDEX_TTS_DIR, "tts_server.py")

EMOTION_VECTORS = {
    "default":  None,
    "calm":     [0, 0, 0, 0, 0, 0, 0, 0.9],
    "excited":  [0, 0, 0, 0, 0, 0, 0.8, 0.2],
    "tense":    [0, 0, 0, 0.6, 0, 0, 0, 0.4],
    "solemn":   [0, 0, 0, 0, 0, 0.2, 0, 0.8],
}


def _tts_available() -> bool:
    """检查 IndexTTS2 环境和模型权重是否就绪"""
    if not os.path.exists(_TTS_SERVER):
        return False
    ckpt = os.path.join(_INDEX_TTS_DIR, "checkpoints")
    if not os.path.isdir(ckpt):
        return False
    # 检查关键模型权重文件 (IndexTTS2 使用 .pth / .safetensors)
    for f in os.listdir(ckpt):
        if f in ("gpt.pth", "s2mel.pth") or f.endswith(".safetensors"):
            return True
    return False


def synthesize_batch(items: list, voice_prompt: str, output_dir: str) -> list:
    """
    批量调用 IndexTTS2 合成语音。
    items: [{"text": str, "output_path": str, "emotion": str, "speed": float}, ...]
    返回: [{"output_path": str, "duration": float}, ...]  失败返回空列表
    """
    if not _tts_available():
        Logger.warn("IndexTTS2 模型权重未下载，将使用 pyttsx3 回退")
        Logger.warn("下载命令: cd ../index-tts && uv run modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints")
        return []

    # 构建批量输入
    batch = []
    for item in items:
        entry = {
            "text": item["text"],
            "output_path": item["output_path"],
        }
        emo = item.get("emotion", "default")
        if emo != "default" and emo in EMOTION_VECTORS:
            entry["emotion_vector"] = EMOTION_VECTORS[emo]
        if item.get("speed", 1.0) != 1.0:
            entry["speed"] = item["speed"]
        batch.append(entry)

    input_file = os.path.abspath(os.path.join(output_dir, "_tts_batch.json"))
    result_file = os.path.abspath(os.path.join(output_dir, "_tts_result.json"))

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump({"voice_prompt": voice_prompt, "items": batch}, f, ensure_ascii=False)

    cfg_path = os.path.join(_INDEX_TTS_DIR, "checkpoints", "config.yaml")
    model_dir = os.path.join(_INDEX_TTS_DIR, "checkpoints")

    # 命令诊断（可手动测试）
    venv_python = os.path.join(_INDEX_TTS_DIR, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        cmd = [venv_python, _TTS_SERVER,
               "--batch", input_file, "--cfg", cfg_path, "--model_dir", model_dir]
    else:
        cmd = ["uv", "run", "python", _TTS_SERVER,
               "--batch", input_file, "--cfg", cfg_path, "--model_dir", model_dir]
    # 动态超时：120s 模型加载 + 每段 120s（CPU 推理保守估计）
    timeout = max(300, 120 + len(batch) * 120)
    est_min = timeout / 60
    Logger.info(f"IndexTTS2: {' '.join(cmd)}")
    Logger.info(f"批量合成 {len(batch)} 段 (模型首次加载约1分钟，超时设 {timeout}s ≈ {est_min:.0f}min)...")
    try:
        # stdout/stderr 直通终端，结果从 JSON 文件读取
        proc = subprocess.run(
            cmd,
            cwd=_INDEX_TTS_DIR,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            Logger.warn(f"IndexTTS2 退出码 {proc.returncode}")
            return []

        if not os.path.exists(result_file):
            Logger.warn(f"IndexTTS2 未生成结果文件: {result_file}")
            Logger.warn(f"  检查 stderr 输出排查模型加载/合成错误")
            return []

        with open(result_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        ok = sum(1 for r in results if r.get("status") == "ok")
        Logger.success(f"IndexTTS2 合成: {ok}/{len(results)} 段成功")
        return results

    except subprocess.TimeoutExpired:
        Logger.error(f"IndexTTS2 合成超时 ({timeout}s ≈ {timeout/60:.0f}min)，回退 pyttsx3")
        Logger.error("  可手动测试: cd ../index-tts && " + " ".join(cmd))
        return []
    except Exception as e:
        Logger.error(f"IndexTTS2 调用异常: {e}")
        return []
