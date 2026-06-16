"""
全局配置 - LLM API 连接参数
所有 demo 文件统一从此处导入

使用方式（在任意子目录的 Python 文件中）：
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import client, MODEL_NAME

环境变量配置：
    复制 .env.example 为 .env 并填入真实值，或在 shell 中 export

可配置项：
    LLM_API_KEY          必填 —— 网关 API Key
    LLM_BASE_URL         可选 —— 默认指向公司 LLM 网关
    LLM_MODEL_NAME       可选 —— 通用对话模型（默认 sonnet）
    LLM_EMBEDDING_MODEL  可选 —— 向量化模型
    LLM_JUDGE_MODEL      可选 —— LLM-as-Judge 评估用模型（通常需更强）

设计要点：
    client / API_KEY 采用惰性加载 —— 仅在首次使用 client 时校验 Key，
    因此纯本地脚本（如 tokenization_demo.py）即使没配置 key 也能跑。
"""

import os

from openai import OpenAI

# 自动加载 .env（若安装了 python-dotenv）
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 模型配置 —— 直接读取，没有副作用
# ---------------------------------------------------------------------------
BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1",
)
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "aws-claude-sonnet-4-6")
EMBEDDING_MODEL = os.getenv("LLM_EMBEDDING_MODEL", "text-embedding-3-small")
JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "aws-claude-opus-4-7")


# ---------------------------------------------------------------------------
# 惰性 client —— 只有真正用 client 时才校验 API Key
# ---------------------------------------------------------------------------
def _require_api_key() -> str:
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise RuntimeError(
            "未配置 LLM_API_KEY 环境变量。\n"
            "请复制 .env.example 为 .env 并填入真实 Key，"
            "或执行：export LLM_API_KEY=your_key_here"
        )
    return key


class _LazyClient:
    """OpenAI 客户端的惰性代理 —— 首次属性访问时才真正构造。"""

    def __init__(self) -> None:
        self._real: OpenAI | None = None

    def _get(self) -> OpenAI:
        if self._real is None:
            self._real = OpenAI(api_key=_require_api_key(), base_url=BASE_URL)
        return self._real

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


client: OpenAI = _LazyClient()  # type: ignore[assignment]


def __getattr__(name: str):
    """模块级惰性属性 —— `from config import API_KEY` 时才校验。"""
    if name == "API_KEY":
        return _require_api_key()
    raise AttributeError(f"module 'config' has no attribute {name!r}")
