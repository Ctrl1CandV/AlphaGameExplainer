from src.infra.logger import Logger
from dotenv import load_dotenv
import os

load_dotenv()

# 共享system message：两后端共用同一事实边界
# 提取自原LlamaCppBackend.generate内联字符串，逐字节等价，保证本地行为不变
SYSTEM_MESSAGE = (
    "你是一位国际象棋教练。程序标注的关键手、状态和确定性棋理事实是当前棋局的唯一真值；"
    "引擎数据等量化信息只作参考，不得据此改写状态。未标注重要程度时，也只能依据已提供事实组织详略。"
    "知识片段和范例只供表达参考，不得覆盖当前事实；信息不足时保守描述，不自行补算走法或变化。"
    "只有标注「已将杀」的节点才能说将杀或绝杀，其他节点使用推进性描述。"
    "如果要求JSON，只输出指定字段，不自行增加字段。多个segment之间要有承接关系。"
    "不要复述提示词。"
)

class LlamaCppBackend:
    """ llama.cpp后端，单例缓存在模块级LLM_BACKEND_CACHE中 """

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

        Logger.info(f"加载 LLM 模型: {os.path.basename(self.model_path)}")

        from llama_cpp import Llama
        self._llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            verbose=self.verbose,
        )
        Logger.success(f"LLM 模型就绪")

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

            system_content = SYSTEM_MESSAGE

            result = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1400,
                temperature=self.temperature,
                stop=[],
                grammar=grammar_obj,
            )
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return text.strip()
        except Exception:
            return ""

    def close(self):
        if self._llm is not None:
            try:
                self._llm.close()
            except Exception:
                pass
            self._llm = None
            self._grammar_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # generator读取此属性决定主chunk JSON的内容重试预算，本地模式沿用既有MAX_RETRIES
    content_retry_limit = 1


class DeepSeekBackend:
    """
    DeepSeek API后端（OpenAI SDK兼容），实现generate(prompt, grammar)->str契约

    职责：
    - 只负责一次HTTP 传输 + 响应提取 + 输出模式适配，不判内容/schema/事实合法性
    - SDK max_retries=0，一次generate()最多一个API HTTP请求
    - 只把异常/不可用状态/None/空白/null/undefined判为""（传输级失败）；
      任何其他非空内容原样返回，由generator的parser/validator判定
    - grammar GrammarConstraint.output_mode标签映射response_format：
      JSON → json_object；PURE_CN → 不设 response_format，追加纯中文格式约束；
      None → 普通文本；未知非空 grammar → fail closed

    content_retry_limit=2：API模式内容重试两次，由generator读取
    """

    # 推理模型：reasoning_tokens在content之前消耗，本地1400全被推理吃光
    DEFAULT_MAX_TOKENS = 32768

    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.timeout = float(os.getenv("LLM_API_TIMEOUT", "120"))
        self.max_tokens = int(os.getenv("DEEPSEEK_MAX_TOKENS", str(self.DEFAULT_MAX_TOKENS)))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self._client = None
        # 永久失败标记：缺ke或鉴权401/403时置True，FallbackBackend据此立即开路
        self.is_permanently_broken = not bool(self.api_key)
        # 可观测统计：供冒烟/回归记录token开销与finish_reason
        self.http_call_count = 0
        self.last_finish_reason = None
        self.last_reasoning_tokens = None
        self.last_completion_tokens = None
        if self.is_permanently_broken:
            Logger.warn("DEEPSEEK_API_KEY 未配置，DeepSeekBackend 标记为永久失败")

    def _ensure_client(self):
        if self._client is not None:
            return
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置")
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0
        )

    def generate(self, prompt: str, grammar: str = None) -> str:
        # 构造期判定的永久失败（缺 key）：直接返回，不重复请求
        if self.is_permanently_broken:
            return ""

        try:
            self._ensure_client()
        except Exception as e:
            Logger.error(f"DeepSeek client 初始化失败: {self._sanitize_err(e)}")
            return ""

        mode = self._resolve_output_mode(grammar)
        if mode == "unknown":
            return ""  # fail closed：未知 grammar 不猜测模式，触发 FallbackBackend 兜底本地

        messages = self._build_messages(prompt, mode)
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if mode == "json":
            kwargs["response_format"] = {"type": "json_object"}

        self.http_call_count += 1
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # 401/403 鉴权失败视为永久错误：标记后 FallbackBackend 立即开路，不重复请求
            if self._is_auth_error(e):
                self.is_permanently_broken = True
                Logger.warn("DeepSeek 鉴权失败（401/403），标记永久失败，后续调用直接走本地")
            else:
                Logger.warn(f"DeepSeek API 调用失败: {self._sanitize_err(e)}")
            return ""

        content = self._extract_content(resp)
        # 记录本次调用的 finish_reason 与 token 开销（PLAN-002 §13/REV-003 实测支撑）
        self._record_usage(resp)
        if content is None:
            return ""
        text = content.strip()
        # 传输级无结果：null/undefined/空白一律视为失败，交 generator/fallback 链处理
        if text in ("", "null", "undefined", "None"):
            return ""
        return text

    def _record_usage(self, resp) -> None:
        """ 从响应提取finish_reason与 usage，供观测max_tokens是否截断，失败安全 """
        try:
            self.last_finish_reason = getattr(resp.choices[0], "finish_reason", None)
        except Exception:
            self.last_finish_reason = None
        try:
            usage = resp.usage
            self.last_completion_tokens = getattr(usage, "completion_tokens", None)
            details = getattr(usage, "completion_tokens_details", None)
            self.last_reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
        except Exception:
            pass

    @staticmethod
    def _is_auth_error(e) -> bool:
        """ 识别401/403鉴权失败 """
        status = getattr(e, "status_code", None)
        return status in (401, 403)

    def _resolve_output_mode(self, grammar) -> str:
        """ 从grammar解析输出模式，返回'json'/'pure_cn'/'text'/'unknown' """
        if not grammar:
            return "text"
        # GrammarConstraint带output_mode标签
        output_mode = getattr(grammar, "output_mode", None)
        if output_mode is not None:
            from src.commentator.grammar import OutputMode
            if output_mode == OutputMode.JSON:
                return "json"
            if output_mode == OutputMode.PURE_CN:
                return "pure_cn"
            return "unknown"
        return "unknown"

    def _build_messages(self, prompt: str, mode: str) -> list:
        """ 构造messages，PURE_CN追加格式约束 """
        system = SYSTEM_MESSAGE
        if mode == "pure_cn":
            system = (
                system + "口播文本必须是纯中文加中文标点，"
                "不得出现英文字母、阿拉伯数字、坐标、Markdown 符号或思考过程。"
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

    @staticmethod
    def _extract_content(resp) -> str:
        """ 从SDK响应提取content。任何结构异常返回None """
        try:
            choice = resp.choices[0]
            return choice.message.content
        except Exception:
            return None

    @staticmethod
    def _sanitize_err(e) -> str:
        """
        脱敏异常文本：剔除key与Authorization头，避免凭据/隐私入日志
        先脱敏再截断，避免把脱敏后的占位符切掉
        """
        import re as _re
        msg = str(e)
        msg = _re.sub(r"sk-[A-Za-z0-9]{4,}", "sk-***", msg)
        msg = _re.sub(r"Bearer\s+[A-Za-z0-9_\-]+", "Bearer ***", msg)
        return msg[:200]

    def close(self):
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # API模式内容重试预算：含首试共3次生成
    content_retry_limit = 2

class FallbackBackend:
    """
    主后端失败时按调用即时降级到次后端，带运行级传输熔断
    职责：
    - 主=API、次=本地懒加载；API传输级失败（返回空串）→ 本次调用立即走本地
    - 内容/schema/事实不合格不切本地：API返回非空内容（即便校验失败）也算成功，
      重置失败计数，由generator的retry/repair/fallback链处理
    - 连续传输失败达阈值后熔断开路，本次运行余下调用直接走本地
    - 缺key/鉴权失败视为永久错误，立即开路不重复请求
    - 双后端都失败返回""，交既有generator/fallback链处理，不承诺必然非空

    次后端通过local_factory延迟创建，仅在首次降级时加载本地模型
    """

    def __init__(self, primary, local_factory, failure_threshold: int = None):
        self._primary = primary
        self._local_factory = local_factory
        self._local = None
        self._failure_count = 0
        self._circuit_open = False
        self._permanent_failure = False
        self.failure_threshold = int(
            failure_threshold if failure_threshold is not None
            else os.getenv("LLM_FALLBACK_THRESHOLD", "3")
        )

    def _get_local(self):
        """ 懒加载本地后端。仅在首次降级时构造，避免happy path占显存 """
        if self._local is None:
            self._local = self._local_factory()
        return self._local

    def generate(self, prompt: str, grammar: str = None) -> str:
        # 同步主后端的永久失败标记，一旦主后端标记，立即开路，本次运行余下调用直接走本地
        if getattr(self._primary, "is_permanently_broken", False):
            self._permanent_failure = True
            self._circuit_open = True

        # 永久错误或熔断已开：直接走本地，不再请求 API
        if self._permanent_failure or self._circuit_open:
            return self._get_local().generate(prompt, grammar=grammar)

        try:
            text = self._primary.generate(prompt, grammar=grammar)
        except Exception as e:
            # 主后端异常视为传输级失败（DeepSeekBackend 内部已捕获并返回 ""，
            # 这里兜底防止未预期异常穿透）
            Logger.warn(f"主后端异常，降级本地: {e}")
            text = ""

        if text:
            # 非空响应视为成功：重置失败计数（即便上层 validator 会拒绝此内容，
            # 那是 generator 的职责，不计入传输熔断）
            self._failure_count = 0
            return text

        # 传输级失败：计数 + 必要时开路，本次调用立即走本地
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._circuit_open = True
            Logger.warn(
                f"API 连续传输失败 {self._failure_count} 次（阈值 {self.failure_threshold}），"
                f"熔断开路，本次运行余下调用直接走本地"
            )
        return self._get_local().generate(prompt, grammar=grammar)

    def close(self):
        """关闭主后端与已加载的本地后端。幂等：重复调用安全。"""
        for backend in (self._primary, self._local):
            if backend is None:
                continue
            try:
                backend.close()
            except Exception:
                pass
        self._primary = None
        self._local = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # 与主后端一致的内容重试预算；generator 读取此属性
    content_retry_limit = 2


# 模块级单例缓存。
# 必须单例：4070 Ti Super 16G 显存，Qwen3.6-27B 4-bit ~14.5GB 几乎占满，
# 重复加载会导致 OOM。release_backend() 在 TTS 前释放显存。
LLM_BACKEND_CACHE = {}


def create_backend_from_env():
    """按 LLM_BACKEND 环境变量分派后端。

    - 缺省或 `llama_cpp`：返回 LlamaCppBackend 单例（旧行为，未配置时零变化）
    - `deepseek`：返回 FallbackBackend（主=DeepSeek API，次=本地懒加载），
      详见 PLAN-002 阶段 2/3。该分支在阶段 1 只占位，阶段 2 接入。
    - 其它值：warn 并回退本地，避免未知配置导致崩溃。
    """
    global LLM_BACKEND_CACHE

    backend_kind = os.getenv("LLM_BACKEND", "llama_cpp").strip().lower() or "llama_cpp"
    cache_key = backend_kind

    if cache_key in LLM_BACKEND_CACHE:
        return LLM_BACKEND_CACHE[cache_key]

    if backend_kind == "llama_cpp":
        backend = LlamaCppBackend()
    elif backend_kind == "deepseek":
        # 阶段 2 接入；阶段 1 先占位，避免提前引入 openai 依赖。
        backend = _create_deepseek_fallback_backend()
    else:
        Logger.warn(f"未知 LLM_BACKEND={backend_kind!r}，回退 llama_cpp")
        backend = LlamaCppBackend()
        cache_key = "llama_cpp"

    LLM_BACKEND_CACHE[cache_key] = backend
    return backend


def _create_deepseek_fallback_backend():
    """构造 deepseek 模式的 FallbackBackend（主=DeepSeek API，次=本地懒加载）。

    PLAN-002 阶段 3：API 为主，本地通过 LlamaCppBackend 工厂延迟创建，
    仅在首次传输降级时加载模型（显存约束）。传输熔断阈值由 LLM_FALLBACK_THRESHOLD 控制。
    """
    return FallbackBackend(
        primary=DeepSeekBackend(),
        local_factory=lambda: LlamaCppBackend(),
    )


def release_backend():
    global LLM_BACKEND_CACHE
    for backend in LLM_BACKEND_CACHE.values():
        try:
            backend.close()
        except Exception:
            pass
    LLM_BACKEND_CACHE.clear()
