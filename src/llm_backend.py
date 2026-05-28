from abc import ABC, abstractmethod
from dotenv import load_dotenv
from src.common import Logger
import os

load_dotenv()

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, grammar: str = None) -> str:
        ...

    def close(self):
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

class OllamaBackend(LLMBackend):
    def __init__(self, model: str = None, temperature: float = 0.2):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.temperature = temperature

    def generate(self, prompt: str, grammar: str = None) -> str:
        import ollama

        prompt_no_think = prompt + "\n/no_think"
        for use_think in (True, False):
            try:
                pieces = []
                kwargs = {
                    "model": self.model,
                    "prompt": prompt_no_think,
                    "stream": True,
                    "options": {"temperature": self.temperature},
                }
                if use_think:
                    kwargs["think"] = False
                for chunk in ollama.generate(**kwargs):
                    piece = chunk.response if hasattr(chunk, "response") else chunk.get("response", "")
                    if piece:
                        pieces.append(piece)
                text = "".join(pieces)
                if text:
                    return text
            except Exception as e:
                Logger.warn(f"  ollama generate 异常: {type(e).__name__}: {e}")
        return ""

class LlamaCppBackend(LLMBackend):
    def __init__(
        self,
        model_path: str = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
        n_batch: int = 512,
        verbose: bool = False,
        temperature: float = 0.2,
    ):
        self.model_path = model_path or os.getenv("LLAMA_CPP_MODEL_PATH", "")
        self.n_gpu_layers = int(os.getenv("LLAMA_CPP_N_GPU_LAYERS", str(n_gpu_layers)))
        self.n_ctx = int(os.getenv("LLAMA_CPP_N_CTX", str(n_ctx)))
        self.n_batch = int(os.getenv("LLAMA_CPP_N_BATCH", str(n_batch)))
        self.verbose = os.getenv("LLAMA_CPP_VERBOSE", str(verbose)).lower() in ("true", "1", "yes")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", str(temperature)))
        self._llm = None
        self._grammar_cache = {}

        if not self.model_path:
            Logger.warn("LLAMA_CPP_MODEL_PATH 未设置，LlamaCppBackend 将不可用")

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        if not self.model_path:
            raise RuntimeError("LLAMA_CPP_MODEL_PATH 未设置，无法加载模型")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        Logger.info(f"加载 LlamaCpp 模型: {os.path.basename(self.model_path)}")
        Logger.info(f"  n_gpu_layers={self.n_gpu_layers}, n_ctx={self.n_ctx}, n_batch={self.n_batch}")

        from llama_cpp import Llama

        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            verbose=self.verbose,
        )
        Logger.success(f"LlamaCpp 模型加载完成")

    def generate(self, prompt: str, grammar: str = None) -> str:
        try:
            self._ensure_loaded()
        except Exception as e:
            Logger.error(f"LlamaCpp 加载失败: {e}")
            return ""

        try:
            grammar_obj = None
            if grammar:
                if grammar not in self._grammar_cache:
                    from llama_cpp import LlamaGrammar
                    self._grammar_cache[grammar] = LlamaGrammar.from_string(grammar, verbose=self.verbose)
                grammar_obj = self._grammar_cache[grammar]

            result = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": (
                        "你是国际象棋赛事解说员。解说基于节点信息的「状态」字段中的真值。"
                        "只有「已将杀」的节点才能说将杀/绝杀。其他节点用推进性描述（压缩空间、控制关键格、驱赶对方王）。"
                        "如果要求JSON，只输出指定字段，不自行增加字段。"
                        "多个segment之间要有承接关系。不要复述提示词。"
                    )},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1400,
                temperature=self.temperature,
                stop=[],
                grammar=grammar_obj,
            )
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return text.strip()
        except Exception as e:
            Logger.warn(f"  llama.cpp generate 异常: {type(e).__name__}: {e}")
            return ""

    def close(self):
        if self._llm is not None:
            try:
                self._llm.close()
            except Exception:
                pass
            self._llm = None
            self._grammar_cache.clear()
            Logger.info("LlamaCpp 模型已释放")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

LLM_BACKEND_CACHE = {}


def create_backend_from_env() -> LLMBackend:
    global LLM_BACKEND_CACHE

    backend_type = os.getenv("LLM_BACKEND", "ollama").strip().lower()
    cache_key = backend_type

    if cache_key in LLM_BACKEND_CACHE:
        return LLM_BACKEND_CACHE[cache_key]

    if backend_type == "llama_cpp":
        backend = LlamaCppBackend()
    else:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        backend = OllamaBackend(temperature=temperature)

    LLM_BACKEND_CACHE[cache_key] = backend
    Logger.info(f"LLM 后端: {backend.name}")
    return backend

def release_backend():
    global LLM_BACKEND_CACHE
    for backend in LLM_BACKEND_CACHE.values():
        try:
            backend.close()
        except Exception:
            pass
    LLM_BACKEND_CACHE.clear()
