"""
全局配置文件 - LLM API 连接参数
所有 demo 文件统一从此处导入配置

使用方式（在任意子目录的 Python 文件中）：
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import client, MODEL_NAME
"""

from openai import OpenAI

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJmTnRMMnNDWkRIMVY3QTMzbzFSelZuOHphM3F3UDlmNiJ9.6VRPdWAyerGbSlqmsnegT9CGgJceX_leAgEX86nfKOI"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "aws-claude-sonnet-4-6"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
