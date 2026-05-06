# 流式输出（Streaming）完全知识手册

> 本文档系统性讲解 LLM 应用中流式输出的原理、协议、实现方案。
> 配合 `stream_demo.py` 代码阅读效果更佳。

---

## 目录

1. [流式输出的本质与原理](#1-流式输出的本质与原理)
2. [SSE 协议详解](#2-sse-协议详解)
3. [WebSocket vs SSE vs Long Polling 对比](#3-websocket-vs-sse-vs-long-polling-对比)
4. [LangChain 流式接口](#4-langchain-流式接口)
5. [Agent 流式输出](#5-agent-流式输出)
6. [LangGraph 流式模式](#6-langgraph-流式模式)
7. [异步流式编程](#7-异步流式编程)
8. [流式输出在前端的实现](#8-流式输出在前端的实现)
9. [背压处理与慢客户端](#9-背压处理与慢客户端)
10. [流式错误处理](#10-流式错误处理)
11. [生产环境流式架构](#11-生产环境流式架构)

---

## 1. 流式输出的本质与原理

### 1.1 为什么需要流式

```
对应 stream_demo.py 开头的"前置科普一"：

阻塞式（.invoke）的用户体验：
  ┌─────────────────────────────────────────────────────────┐
  │  点击发送 → [          8秒空白         ] → 突然出现一大段 │
  │  用户感受："卡了？挂了？要刷新吗？"                      │
  └─────────────────────────────────────────────────────────┘

流式（.stream）的用户体验：
  ┌─────────────────────────────────────────────────────────┐
  │  点击发送 → 0.3s后开始 → 一│个│字│一│个│字│蹦│出│来│    │
  │  用户感受："在思考了！好快！"                             │
  └─────────────────────────────────────────────────────────┘

同样等 8 秒，流式让用户感觉"快了 10 倍"。
这是所有主流 AI 产品（ChatGPT、Claude）采用流式输出的根本原因。
```

### 1.2 LLM 天然适合流式

```
LLM 的生成机制（参见 llm/KNOWLEDGE.md 第7节）：

  模型逐 token 生成，每个 token 一经计算出来就可以发送：

  时间线：
  t=0    t=0.1   t=0.2   t=0.3   t=0.4   ...   t=3.0
   │       │       │       │       │              │
   "你"    "好"    "！"    "我"    "是"    ...   "<EOS>"
   ↓       ↓       ↓       ↓       ↓              ↓
  发送    发送    发送    发送    发送    ...    发送完毕

  阻塞式：等 t=3.0 全部生成完才一次性返回
  流式：  t=0.1 就开始发送第一个 token

  关键指标：
    TTFT (Time To First Token) = 首个 token 到达时间
    流式模式的 TTFT ≈ 0.1~0.5s（取决于 Prompt 长度和模型）
    阻塞模式的 TTFT = 总生成时间（可能 5~30s）
```

### 1.3 流式传输的底层实现

```
HTTP 层面的实现原理：

  普通请求（阻塞）：
    客户端 → 请求 → 服务器处理 → 一次性响应 → 连接关闭

  流式请求（SSE/Chunked Transfer）：
    客户端 → 请求 → 服务器开始响应
                   → 发送 chunk_1
                   → 发送 chunk_2
                   → 发送 chunk_3
                   → ...
                   → 发送结束标记 → 连接关闭

  HTTP 响应头：
    Transfer-Encoding: chunked          ← 分块传输
    Content-Type: text/event-stream     ← SSE 格式
    Cache-Control: no-cache             ← 禁用缓存
    Connection: keep-alive              ← 保持连接
```

---

## 2. SSE 协议详解

### 2.1 SSE 是什么

```
SSE = Server-Sent Events（服务器发送事件）

一种基于 HTTP 的单向推送协议：
  服务器 → 客户端（单向）
  客户端只能接收，不能通过同一连接发送数据

对比 stream_demo.py 第5章的 SSE 格式化输出：
  data: {"token": "你"}
  data: {"token": "好"}
  data: [DONE]
```

### 2.2 SSE 报文格式

```
SSE 有严格的文本格式规范：

┌─────────────────────────────────────────────────────────┐
│ 字段         │ 含义                                     │
├──────────────┼──────────────────────────────────────────┤
│ data:        │ 事件数据（必需）                         │
│ event:       │ 事件类型（可选，默认 "message"）          │
│ id:          │ 事件 ID（可选，用于断线重连）             │
│ retry:       │ 重连间隔毫秒（可选）                     │
│ (空行)       │ 事件之间的分隔符                         │
└──────────────┴──────────────────────────────────────────┘

完整的 SSE 报文示例：

  event: token
  id: 1
  data: {"content": "你", "index": 0}

  event: token
  id: 2
  data: {"content": "好", "index": 1}

  event: tool_call
  id: 3
  data: {"name": "search", "args": {"query": "天气"}}

  event: done
  id: 4
  data: {"finish_reason": "stop", "usage": {"total_tokens": 50}}

关键规则：
  ① 每个字段占一行，格式为 "field: value\n"
  ② 事件之间用空行("\n\n")分隔
  ③ 以 ":" 开头的行是注释（可用作心跳保活）
  ④ data 可以多行（同一事件内多个 data: 会被拼接）
```

### 2.3 OpenAI 风格的 SSE

```
OpenAI API 的流式响应格式（已成为行业标准）：

  data: {"id":"chatcmpl-abc","choices":[{"delta":{"role":"assistant"}}]}

  data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"你"}}]}

  data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"好"}}]}

  data: {"id":"chatcmpl-abc","choices":[{"delta":{}},"finish_reason":"stop"}]}

  data: [DONE]

注意 delta vs message：
  非流式：response.choices[0].message.content     ← 完整消息
  流式：  response.choices[0].delta.content        ← 增量片段

对应 llm/KNOWLEDGE.md 第12.3节的详细说明。
```

---

## 3. WebSocket vs SSE vs Long Polling 对比

### 3.1 三种实时通信方案

```
┌──────────────────────────────────────────────────────────────────────┐
│           │ Long Polling    │ SSE              │ WebSocket           │
├───────────┼─────────────────┼──────────────────┼─────────────────────┤
│ 方向      │ 单向(伪双向)    │ 单向(服务器→客户端)│ 双向                │
│ 协议      │ HTTP            │ HTTP             │ WS (独立协议)       │
│ 连接      │ 反复建立/关闭   │ 持久连接          │ 持久连接            │
│ 实现复杂度│ 低              │ 低               │ 中~高               │
│ 浏览器支持│ 所有            │ 所有现代浏览器    │ 所有现代浏览器      │
│ 自动重连  │ 需自行实现      │ 浏览器原生支持    │ 需自行实现          │
│ 代理兼容性│ 好              │ 好               │ 部分代理不支持      │
│ 适用场景  │ 旧系统兼容      │ LLM 流式输出     │ 聊天室/游戏/协作    │
└───────────┴─────────────────┴──────────────────┴─────────────────────┘
```

### 3.2 为什么 LLM 应用首选 SSE

```
LLM 流式输出的特点决定了 SSE 是最佳选择：

  ① 单向性：LLM 生成是服务器向客户端推送，不需要双向通信
  ② 简单性：基于 HTTP，不需要额外协议升级
  ③ 原生重连：浏览器 EventSource API 自动处理断线重连
  ④ 兼容性：所有 CDN、反向代理、负载均衡器都能正确转发
  ⑤ 可调试性：用 curl 就能测试，数据是可读文本

什么时候用 WebSocket？
  - 用户可以"中断"AI 生成（需要客户端→服务器发消息）
  - 多人协作实时编辑
  - 需要频繁双向通信的场景

实际方案：SSE 用于输出流，普通 POST 用于发送输入
  POST /api/chat         → 发送用户消息
  GET  /api/chat/stream  → SSE 接收 AI 回复
```

---

## 4. LangChain 流式接口

### 4.1 四种流式 API 概览

```
对应 stream_demo.py "前置科普二"的表格：

┌───────────────────┬────────────────────────────────────────┐
│  API              │  适用场景                               │
├───────────────────┼────────────────────────────────────────┤
│  .stream()        │  最简单，逐 chunk 输出，同步            │
│  .astream()       │  .stream() 的异步版本                  │
│  .astream_events()│  全链路事件流，能看到每个组件（最强大） │
│  .astream_log()   │  JSON Patch 日志流                     │
└───────────────────┴────────────────────────────────────────┘
```

### 4.2 .stream() 详解

```
最简单的流式调用方式（对应 stream_demo.py 第1章）：

  chain = prompt | llm | StrOutputParser()

  for chunk in chain.stream({"question": "什么是黑洞？"}):
      print(chunk, end="", flush=True)

每个 chunk 的内容取决于链的最后一个组件：
  prompt | llm | StrOutputParser  → chunk 是 str
  prompt | llm                    → chunk 是 AIMessageChunk

stream_demo.py 的演示展示了两种情况的对比。

关键点：
  ① flush=True 确保立即输出（不被缓冲区延迟）
  ② end="" 避免每个 chunk 后自动换行
  ③ chunk 大小不固定（可能是一个字，也可能是几个字）
```

### 4.3 .astream() 详解

```
.stream() 的异步版本，用于 async 环境：

  async for chunk in chain.astream({"question": "什么是黑洞？"}):
      print(chunk, end="", flush=True)

适用场景：
  - FastAPI 路由处理函数（async def endpoint）
  - asyncio 事件循环中
  - 需要同时处理多个流式请求时

在脚本中使用：
  asyncio.run(my_async_function())
```

### 4.4 .astream_events() 详解

```
最强大的流式 API（对应 stream_demo.py 第3章）：

能看到链中每一层的事件，事件类型：
  on_chain_start       → 链开始执行
  on_chain_end         → 链执行完毕
  on_chat_model_start  → LLM 开始推理
  on_chat_model_stream → LLM 产出一个 token（核心！）
  on_chat_model_end    → LLM 推理完毕
  on_tool_start        → 工具开始执行
  on_tool_end          → 工具执行完毕
  on_parser_start      → 解析器开始
  on_parser_end        → 解析器结束

使用方式：
  async for event in chain.astream_events(input, version="v2"):
      if event["event"] == "on_chat_model_stream":
          token = event["data"]["chunk"].content
          print(token, end="")

⚠️ 注意事项：
  ① 只有异步版本（必须用 async for）
  ② 必须指定 version="v2"（v1 已废弃）
  ③ 事件量大，需要过滤关注的事件类型
```

---

## 5. Agent 流式输出

### 5.1 Agent 流式的特殊性

```
对应 stream_demo.py 第2章的说明：

普通 Chain 流式：一维流，全是文本 token
  token → token → token → ... → 结束

Agent 流式：多维流，混合步骤和 token
  ┌────────────────────────────────────────────────────┐
  │ 步骤1: LLM 推理（决定调用工具）    ← 不一定需要展示 │
  │ 步骤2: 工具执行（等待结果）        ← 展示 loading  │
  │ 步骤3: LLM 推理（决定再调工具）    ← 不一定需要展示 │
  │ 步骤4: 工具执行                    ← 展示 loading  │
  │ 步骤5: LLM 生成最终回答            ← 需要逐字展示  │
  └────────────────────────────────────────────────────┘
```

### 5.2 步骤级流式（AgentExecutor.stream）

```
对应 stream_demo.py 第2章的演示：

  for event in agent_executor.stream({"input": "..."}):
      if "actions" in event:      # Agent 决定调用工具
          print(f"调用: {event['actions'][0].tool}")
      elif "steps" in event:      # 工具返回结果
          print(f"结果: {event['steps'][0].observation}")
      elif "output" in event:     # 最终回答（完整文本）
          print(f"回答: {event['output']}")

特点：
  - 每完成一个步骤产出一个事件
  - 最终回答是完整文本（不是逐 token）
  - 适合展示"Agent 在做什么"的进度条
```

### 5.3 Token 级流式（astream_events）

```
对应 stream_demo.py 第3章 demo_agent_astream_events：

  async for event in agent_executor.astream_events(input, version="v2"):
      if event["event"] == "on_tool_start":
          print(f"工具调用: {event['name']}")
      elif event["event"] == "on_tool_end":
          print(f"工具返回: {event['data']['output']}")
      elif event["event"] == "on_chat_model_stream":
          content = event["data"]["chunk"].content
          if content and not event["data"]["chunk"].tool_calls:
              print(content, end="")  # 最终回答逐 token

关键技巧：过滤 tool_calls
  LLM 生成工具调用时也会产出 on_chat_model_stream 事件，
  但此时 chunk.tool_calls 不为空——这些不是给用户看的文本。
  只有 tool_calls 为空时的 content 才是最终回答。
```

---

## 6. LangGraph 流式模式

### 6.1 两种 stream_mode

```
对应 stream_demo.py 第4章：

stream_mode="values"（默认）：
  每个节点执行后，产出当前完整 State 快照。
  类比：每次拍一张全景照片

  for state in graph.stream(input, stream_mode="values"):
      print(state)  # {"topic": "AI", "summary": "...", "expanded": "..."}

stream_mode="updates"：
  每个节点执行后，只产出该节点修改的字段。
  类比：只拍变化的部分

  for update in graph.stream(input, stream_mode="updates"):
      # update = {"node_name": {"changed_field": "new_value"}}
      for node, changes in update.items():
          print(f"{node} 更新了: {changes}")
```

### 6.2 选择建议

```
┌─────────────────────┬──────────────────────────────────────────┐
│ stream_mode         │ 适用场景                                  │
├─────────────────────┼──────────────────────────────────────────┤
│ "values"            │ 需要完整状态（如前端重建整个页面）        │
│ "updates"           │ 只关心增量变化（前端局部更新，带宽更小）  │
└─────────────────────┴──────────────────────────────────────────┘

生产环境推荐 "updates"：
  - 传输数据量更小
  - 前端可以做精准的局部渲染
  - 更容易实现进度条（知道哪个节点在执行）
```

### 6.3 LangGraph + astream_events

```
LangGraph 也支持 astream_events，可以获取节点内部 LLM 的逐 token 输出：

  async for event in app.astream_events(input, version="v2"):
      if event["event"] == "on_chat_model_stream":
          # 某个节点内的 LLM 正在生成
          node_name = event["tags"][0] if event["tags"] else "unknown"
          print(f"[{node_name}] {event['data']['chunk'].content}", end="")

这是实现"多 Agent 系统中每个 Agent 都流式输出"的关键技术。
```

---

## 7. 异步流式编程

### 7.1 为什么流式需要异步

```
对应 stream_demo.py "前置科普三"：

同步流式的问题：
  for chunk in chain.stream(input):  # 当前线程被阻塞
      send_to_client(chunk)          # 如果客户端慢，线程等待

  → 一个线程只能服务一个客户端
  → 1000 个并发 = 1000 个线程 = 资源耗尽

异步流式：
  async for chunk in chain.astream(input):
      await send_to_client(chunk)  # 等待时让出 CPU

  → 一个线程可以服务数千个客户端
  → 1000 个并发 = 1 个线程 + 事件循环 = 资源高效
```

### 7.2 asyncio 基础模式

```python
import asyncio

async def stream_to_client(chain, input_data):
    """异步流式发送给客户端"""
    async for chunk in chain.astream(input_data):
        yield chunk  # 作为异步生成器

# 在脚本中运行（对应 stream_demo.py 的方式）：
asyncio.run(demo_function())

# 在 FastAPI 中（天然异步）：
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for chunk in chain.astream({"question": request.question}):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 7.3 并发流式请求

```python
async def handle_multiple_streams():
    """同时处理多个流式请求"""
    tasks = [
        process_stream(chain, {"question": "问题1"}),
        process_stream(chain, {"question": "问题2"}),
        process_stream(chain, {"question": "问题3"}),
    ]
    # 三个流同时进行，互不阻塞
    await asyncio.gather(*tasks)
```

---

## 8. 流式输出在前端的实现

### 8.1 浏览器 EventSource API

```javascript
// 对应 stream_demo.py 第5章展示的前端代码

// 方式一：原生 EventSource（仅支持 GET）
const es = new EventSource('/api/chat/stream?q=你好');
es.onmessage = (event) => {
    if (event.data === '[DONE]') {
        es.close();
        return;
    }
    const { token } = JSON.parse(event.data);
    document.getElementById('output').textContent += token;
};
es.onerror = (err) => {
    console.error('SSE 错误:', err);
    es.close();
};
```

### 8.2 fetch + ReadableStream（支持 POST）

```javascript
// 方式二：fetch API（支持 POST 请求，更灵活）
async function streamChat(question) {
    const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        // 解析 SSE 格式
        const lines = text.split('\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') return;
                const { token } = JSON.parse(data);
                appendToOutput(token);
            }
        }
    }
}
```

### 8.3 React 组件示例

```javascript
// React Hook for SSE streaming
function useStreamingChat() {
    const [content, setContent] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);

    const sendMessage = async (question) => {
        setContent('');
        setIsStreaming(true);

        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value);
            // 逐步拼接，触发 React 重渲染
            setContent(prev => prev + parseSSEToken(chunk));
        }
        setIsStreaming(false);
    };

    return { content, isStreaming, sendMessage };
}
```

### 8.4 Vue 3 组合式 API 示例

```javascript
// Vue 3 Composable
export function useStreaming() {
    const content = ref('');
    const loading = ref(false);

    async function send(question) {
        content.value = '';
        loading.value = true;

        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            body: JSON.stringify({ question }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            content.value += parseToken(decoder.decode(value));
        }
        loading.value = false;
    }

    return { content, loading, send };
}
```

---

## 9. 背压处理与慢客户端

### 9.1 什么是背压问题

```
背压（Backpressure）：生产者速度 > 消费者速度 时的数据积压

  LLM 生成速度:    50 tokens/s
  网络发送速度:    10 tokens/s（慢网络客户端）
       ↓
  缓冲区不断增长 → 内存溢出 → 服务崩溃

  ┌────────┐     ┌────────────────┐     ┌──────────┐
  │  LLM   │ →→→ │  缓冲区(爆满)  │ →→  │ 慢客户端 │
  │ 50t/s  │     │  内存持续增长   │     │  10t/s   │
  └────────┘     └────────────────┘     └──────────┘
```

### 9.2 解决方案

```
① 缓冲区上限 + 丢弃策略：
  MAX_BUFFER = 1000  # 最多缓存 1000 个 token
  if buffer.full():
      buffer.pop_oldest()  # 丢弃最旧的（或断开连接）

② 异步队列 + 超时：
  import asyncio

  queue = asyncio.Queue(maxsize=100)

  async def producer():
      async for chunk in chain.astream(input):
          try:
              await asyncio.wait_for(queue.put(chunk), timeout=5.0)
          except asyncio.TimeoutError:
              # 客户端太慢，断开连接
              break

  async def consumer(websocket):
      while True:
          chunk = await queue.get()
          await websocket.send(chunk)

③ HTTP/2 流控：
  HTTP/2 协议原生支持流级别的流控（WINDOW_UPDATE 帧）
  服务端会自动感知客户端的接收能力

④ 生产环境建议：
  - 设置连接超时（30~60s 无活动断开）
  - 设置单次响应最大长度（max_tokens）
  - 监控每个连接的缓冲区使用量
  - 超过阈值主动断开并返回错误
```

---

## 10. 流式错误处理

### 10.1 错误类型

```
流式过程中可能遇到的错误：

  ① 连接建立阶段：
     - 网络不可达（ConnectionError）
     - 认证失败（401）
     - 限流（429 Rate Limit）

  ② 传输中阶段：
     - 网络中断（客户端断网）
     - 服务端内部错误（500）
     - 模型输出被内容过滤器截断
     - 超时（响应时间过长）

  ③ 解析阶段：
     - SSE 格式解析错误
     - JSON 解析错误（chunk 被截断）
```

### 10.2 服务端错误处理

```python
# FastAPI 流式错误处理模式
async def generate_stream(question: str):
    try:
        async for chunk in chain.astream({"question": question}):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        # 在 SSE 流中发送错误事件
        error_data = json.dumps({"error": str(e), "type": type(e).__name__})
        yield f"event: error\ndata: {error_data}\n\n"

@app.post("/chat/stream")
async def stream_endpoint(request: ChatRequest):
    return StreamingResponse(
        generate_stream(request.question),
        media_type="text/event-stream",
        headers={
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )
```

### 10.3 客户端错误处理

```javascript
// 前端健壮的流式处理
async function robustStream(question) {
    const MAX_RETRIES = 3;
    let retries = 0;

    while (retries < MAX_RETRIES) {
        try {
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                body: JSON.stringify({ question }),
                signal: AbortSignal.timeout(30000),  // 30s 超时
            });

            if (!response.ok) {
                if (response.status === 429) {
                    // 限流：等待后重试
                    await sleep(2 ** retries * 1000);
                    retries++;
                    continue;
                }
                throw new Error(`HTTP ${response.status}`);
            }

            // 正常处理流...
            return await processStream(response);

        } catch (error) {
            if (error.name === 'TimeoutError') {
                retries++;
                continue;
            }
            throw error;
        }
    }
    throw new Error('Max retries exceeded');
}
```

---

## 11. 生产环境流式架构

### 11.1 典型架构

```
┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────┐
│  前端    │ SSE │  Nginx   │     │  FastAPI     │     │  LLM     │
│ (React)  │←────│ (反代)   │←────│  (Python)    │←────│  (GPU)   │
│          │     │          │     │              │     │          │
│ EventSrc │     │ 禁缓冲   │     │ astream()   │     │ 逐token  │
└──────────┘     └──────────┘     └──────────────┘     └──────────┘
                       │
                       ↓
                 ┌──────────┐
                 │  Redis   │  ← 可选：用于多实例广播
                 │ (Pub/Sub)│
                 └──────────┘
```

### 11.2 Nginx 配置

```nginx
# 流式输出的 Nginx 配置关键点
location /api/chat/stream {
    proxy_pass http://backend;

    # 禁用所有缓冲！否则 Nginx 会攒够一批数据再转发
    proxy_buffering off;
    proxy_cache off;

    # SSE 专用头
    proxy_set_header Connection '';
    proxy_http_version 1.1;
    chunked_transfer_encoding off;

    # 超时设置（流式可能持续很久）
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
}
```

### 11.3 FastAPI 完整实现

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

app = FastAPI()
chain = prompt | llm | StrOutputParser()

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        try:
            async for chunk in chain.astream({"question": request.question}):
                if chunk:
                    data = json.dumps({"token": chunk}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

### 11.4 多实例部署的流式广播

```
问题：多个 FastAPI 实例（水平扩展）时，用户连接的实例可能不是
     处理请求的实例。

解决方案：Redis Pub/Sub 广播

  ┌──────────┐   publish    ┌─────────┐   subscribe   ┌──────────┐
  │ Instance1│ ──────────→  │  Redis  │ ←──────────── │ Instance2│
  │ (处理LLM)│              │ Pub/Sub │               │ (SSE连接) │
  └──────────┘              └─────────┘               └──────────┘
                                 │
                            subscribe
                                 ↓
                            ┌──────────┐
                            │ Instance3│
                            │ (SSE连接) │
                            └──────────┘

  Instance1 处理 LLM 请求，将 token 发布到 Redis channel
  持有 SSE 连接的 Instance2/3 订阅该 channel，转发给客户端
```

### 11.5 性能优化清单

```
┌────────────────────────────────────────────────────────────────┐
│ 优化项                 │ 措施                                   │
├────────────────────────┼────────────────────────────────────────┤
│ 首字延迟(TTFT)         │ 精简 Prompt、预热连接池、就近部署      │
│ 吞吐量                 │ 异步处理、连接复用、LLM 批处理         │
│ 内存控制               │ 缓冲区上限、超时断开、流控              │
│ 可靠性                 │ 心跳保活、自动重连、断点续传            │
│ 可观测性               │ 每条流的 TTFT/TPS 监控、错误率告警     │
│ 成本控制               │ max_tokens 限制、中途停止能力          │
└────────────────────────┴────────────────────────────────────────┘

心跳保活示例：
  async def event_generator():
      last_data_time = time.time()
      async for chunk in chain.astream(input):
          yield f"data: {chunk}\n\n"
          last_data_time = time.time()
      # 如果超过 15s 没有数据，发送心跳注释
      # (实际需要配合 asyncio.wait 实现)
      # yield ": heartbeat\n\n"
```

---

## 附录 A：本文件夹代码与知识点对应

| 代码文件 | 覆盖的知识点 | 对应本文档章节 |
|---------|-------------|---------------|
| `stream_demo.py` 第1章 | Chain .stream() 同步流式 | 第4.2节 |
| `stream_demo.py` 第2章 | Agent 步骤级流式 | 第5.2节 |
| `stream_demo.py` 第3章 | astream_events 全链路事件 | 第4.4节、第5.3节 |
| `stream_demo.py` 第4章 | LangGraph stream_mode | 第6节 |
| `stream_demo.py` 第5章 | SSE 格式化输出 | 第2节、第11.3节 |

---

## 附录 B：流式 API 选择速查

```
对应 stream_demo.py 结尾的总结表格：

  简单 Chain 打字机效果       → .stream()
  FastAPI 异步服务            → .astream()
  Agent 全链路监控/逐 token   → .astream_events()
  LangGraph 节点级追踪        → .stream(stream_mode="updates")
  前端 SSE 推送              → .stream() + SSE 格式化

关键记忆点：
  ① .stream() 同步，用 for 循环
  ② .astream_events() 异步，用 async for
  ③ Agent .stream() 是步骤级（非逐 token）
  ④ Agent 逐 token 必须用 astream_events
  ⑤ LangGraph 用 stream_mode 控制输出粒度
  ⑥ 生产环境 SSE 推送 + Nginx 禁缓冲
```

---

> **下一步学习**：阅读 `langchain/KNOWLEDGE.md` 了解 LangChain 框架的完整知识体系，然后前往 `agent/` 学习 Agent 架构设计。
