"""
全局配置文件 - LLM API 连接参数
所有 demo 文件统一从此处导入配置

使用方式（在任意子目录的 Python 文件中）：
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import client, MODEL_NAME

环境变量配置：
    复制 .env.example 为 .env 并填入真实值，或在 shell 中 export：
        export LLM_API_KEY=...
        export LLM_BASE_URL=...
        export LLM_MODEL_NAME=...
"""

import os

from openai import OpenAI

# 自动加载 .env（如果安装了 python-dotenv 且存在 .env 文件）
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "aws-claude-sonnet-4-6")

if not API_KEY:
    raise RuntimeError(
        "未配置 LLM_API_KEY 环境变量。\n"
        "请复制 .env.example 为 .env 并填入真实 key，"
        "或在 shell 中执行：export LLM_API_KEY=your_key_here"
    )

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
