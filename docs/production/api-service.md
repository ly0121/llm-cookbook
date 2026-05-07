---
title: API 服务化
---

# API 服务化（FastAPI + LLM）

将 LLM 应用从脚本模式升级为持续运行的 API 服务，支持多客户端并发访问。

## 1. 脚本 vs 服务

| 维度 | 脚本模式 | 服务模式 |
|------|---------|---------|
| 使用方式 | 手动运行 | 任何客户端随时调用 |
| 并发 | 单次执行 | 多请求并行处理 |
| 管理 | 无 | 认证、限流、监控 |

## 2. FastAPI 核心优势

| 特性 | 对 LLM 的意义 |
|------|-------------|
| 原生异步 | IO 密集型 LLM 调用不阻塞 |
| Pydantic 类型 | 自动校验请求参数 |
| 自动文档 | /docs 页面自助对接 |
| StreamingResponse | 原生支持 SSE 流式 |

## 3. 请求/响应模型

```python
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str = Field(description="用户消息")
    session_id: str = Field(default="default")
    stream: bool = Field(default=False)
    temperature: float = Field(default=0.7, ge=0, le=2)

class ChatResponse(BaseModel):
    reply: str
    usage: dict
    latency_ms: float
```

## 4. 流式 API

```python
from fastapi.responses import StreamingResponse

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    async def generate():
        async for chunk in chain.astream(request.message):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

## 5. 认证与鉴权

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header()):
    if x_api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
```

## 6. 速率限制

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")
async def chat(request: ChatRequest):
    ...
```

## 7. 健康检查

```python
@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}

@app.get("/ready")
async def ready():
    # 检查 LLM 连接、向量库连接等
    return {"status": "ready"}
```

## 8. 部署方案

| 方案 | 适用 | 命令 |
|------|------|------|
| Uvicorn | 开发 | `uvicorn app:app --reload` |
| Gunicorn+Uvicorn | 生产 | `gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker` |
| Docker | 容器化 | Dockerfile + docker-compose |
| K8s | 大规模 | Deployment + HPA |

## 9. 性能优化

- 连接池复用 HTTP 客户端
- 响应缓存（Redis）
- 异步并发处理
- 负载均衡（多实例）

::: warning 需要本地运行
完整实现见 `api_service/fastapi_server.py` 和 `api_service/test_client.py`。
:::

---

::: tip 下一步
- [安全护栏](/production/guardrails) — API 的安全防护层
- [评估体系](/production/evaluation) — 监控服务质量
:::
