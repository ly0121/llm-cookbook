---
title: 流式输出
---

# 流式输出（Streaming）

流式输出让用户在 LLM 生成的同时看到逐字输出，将"8秒空白"变为"即时响应"的体验。

## 1. 为什么需要流式

```
阻塞式: 点击 → [8秒空白] → 突然出现一大段
流式:   点击 → 0.3s后开始 → 一个字一个字蹦出来
```

同样等 8 秒，流式让用户感觉"快了 10 倍"。关键指标：**TTFT**（Time To First Token）。

## 2. 协议对比

| 协议 | 方向 | 连接 | 适用场景 |
|------|------|------|---------|
| SSE | 服务端→客户端 | HTTP 长连接 | LLM 流式（首选） |
| WebSocket | 双向 | 升级协议 | 实时聊天 |
| Long Polling | 客户端轮询 | 多次 HTTP | 兼容旧系统 |

**SSE 格式：**
```
data: {"token": "你"}
data: {"token": "好"}
data: [DONE]
```

## 3. LangChain 流式接口

```python
# 同步流式
for chunk in chain.stream({"question": "你好"}):
    print(chunk, end="", flush=True)

# 异步流式
async for chunk in chain.astream({"question": "你好"}):
    print(chunk, end="", flush=True)

# 事件流（获取中间步骤）
async for event in chain.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

## 4. Agent 流式

Agent 执行过程中逐步骤/逐 token 输出：

```python
# 逐步骤
for step in agent_executor.stream({"input": "查天气"}):
    if "output" in step:
        print(f"Final: {step['output']}")
    else:
        print(f"Step: {step}")
```

## 5. FastAPI SSE 实现

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    async def generate():
        async for chunk in chain.astream(request.dict()):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

## 6. 前端消费 SSE

```javascript
const eventSource = new EventSource('/chat/stream');
eventSource.onmessage = (event) => {
    if (event.data === '[DONE]') {
        eventSource.close();
        return;
    }
    const { token } = JSON.parse(event.data);
    appendToUI(token);
};
```

## 7. 生产注意事项

| 问题 | 解决方案 |
|------|---------|
| 背压（慢客户端） | 缓冲区限制 + 超时断连 |
| 连接中断 | 客户端自动重连 + 续传 |
| Nginx 缓冲 | `X-Accel-Buffering: no` |
| 错误处理 | 在流中发送错误事件 |

::: warning 需要本地运行
完整实现见 `streaming/stream_demo.py`。
:::

---

::: tip 下一步
- [异步并发](/engineering/async) — 异步编程与并发 LLM 调用
- [API 服务](/production/api-service) — 完整的 FastAPI 服务化方案
:::
