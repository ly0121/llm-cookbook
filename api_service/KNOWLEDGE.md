# API 服务化（FastAPI + LLM）完全知识手册

> 本文档系统讲解将 LLM 应用服务化的所有核心知识点，从 FastAPI 基础到生产部署。
> 配合 `fastapi_server.py` 和 `test_client.py` 代码阅读效果更佳。

---

## 目录

1. [LLM 服务化的必要性](#1-llm-服务化的必要性)
2. [FastAPI 框架基础](#2-fastapi-框架基础)
3. [API 设计模式](#3-api-设计模式)
4. [请求/响应模型设计](#4-请求响应模型设计)
5. [流式 API 实现](#5-流式-api-实现)
6. [认证与鉴权](#6-认证与鉴权)
7. [速率限制（Rate Limiting）](#7-速率限制rate-limiting)
8. [负载均衡策略](#8-负载均衡策略)
9. [连接池与资源管理](#9-连接池与资源管理)
10. [错误处理与标准化响应](#10-错误处理与标准化响应)
11. [健康检查与优雅关闭](#11-健康检查与优雅关闭)
12. [部署方案](#12-部署方案)
13. [性能优化](#13-性能优化)
14. [监控与告警](#14-监控与告警)
15. [API 版本管理](#15-api-版本管理)

---

## 1. LLM 服务化的必要性

### 1.1 脚本模式 vs 服务模式

```
对应 fastapi_server.py 顶部的前置科普：

  脚本模式（当前项目大多数代码）：
    python xxx.py → 执行 → 输出 → 退出
    只有你能用，每次手动运行

  服务模式（生产必须）：
    启动 → 持续监听 → 任何客户端随时调用 → 返回结果
    ┌─────────────────────────────────────────┐
    │  前端 App ────→                          │
    │  后端微服务 ──→  LLM API 服务  →  响应   │
    │  定时任务 ────→                          │
    │  第三方系统 ──→                          │
    └─────────────────────────────────────────┘
```

### 1.2 服务化带来的价值

```
┌────────────────┬──────────────────────────────────────────┐
│  维度           │  价值                                     │
├────────────────┼──────────────────────────────────────────┤
│  多客户端复用   │  一个服务供多个前端/系统调用              │
│  弹性伸缩      │  根据流量自动增减实例                     │
│  安全隔离      │  API Key 控制访问权限                     │
│  监控可观测    │  统一采集日志、指标、链路追踪             │
│  版本管理      │  灰度发布、AB 测试                       │
│  流量控制      │  限流、降级、熔断                         │
└────────────────┴──────────────────────────────────────────┘
```

---

## 2. FastAPI 框架基础

### 2.1 核心特性

```
对应 fastapi_server.py 第 1 章：

FastAPI 之所以成为 LLM 服务的首选框架：

  ┌───────────────────┬────────────────────────────────────────┐
  │  特性             │  对 LLM 应用的意义                      │
  ├───────────────────┼────────────────────────────────────────┤
  │  原生异步         │  LLM调用是IO密集型，异步不阻塞          │
  │  类型安全         │  Pydantic 校验请求参数，防止无效输入     │
  │  自动文档         │  /docs 页面自动生成，前端自助对接        │
  │  高性能           │  基于 Starlette，性能接近 Go/Node       │
  │  流式支持         │  StreamingResponse 天然支持 SSE         │
  └───────────────────┴────────────────────────────────────────┘
```

### 2.2 ASGI 架构

```
ASGI（Async Server Gateway Interface）调用链：

  客户端 HTTP 请求
       ↓
  ┌──────────────┐
  │   uvicorn    │  ← ASGI 服务器（接收连接、管理事件循环）
  └──────┬───────┘
         ↓
  ┌──────────────┐
  │   Starlette  │  ← ASGI 框架（路由、中间件）
  └──────┬───────┘
         ↓
  ┌──────────────┐
  │   FastAPI    │  ← 应用层（业务逻辑、数据校验）
  └──────────────┘

对应 fastapi_server.py 第 5 章：
  uvicorn.run(app, host="0.0.0.0", port=8000)
  uvicorn = 服务员，FastAPI = 厨师
```

### 2.3 异步处理模型

```python
# async def 声明异步端点 → 不会阻塞事件循环

@app.post("/api/qa")
async def qa_endpoint(request: QuestionRequest):
    # await 异步调用 LLM（等待期间可处理其他请求）
    answer = await qa_chain.ainvoke({"question": request.question})
    return QuestionResponse(answer=answer, ...)

# 对比同步写法（会阻塞！）：
@app.post("/api/qa_sync")
def qa_sync(request: QuestionRequest):
    # 这里会阻塞整个事件循环！其他请求都得等！
    answer = qa_chain.invoke({"question": request.question})
    return QuestionResponse(answer=answer, ...)
```

---

## 3. API 设计模式

### 3.1 RESTful 设计

```
对应 fastapi_server.py 的端点设计：

  POST /api/qa          → 问答（创建一次问答交互）
  POST /api/translate   → 翻译（创建一次翻译任务）
  POST /api/qa/stream   → 流式问答
  GET  /health          → 健康检查（读取服务状态）

LLM API 设计的特殊性：
  ① 几乎全是 POST（因为输入在 Body 中，可能很长）
  ② 不适合纯 CRUD（不是数据库操作，是"计算"）
  ③ 流式端点是刚需（SSE 或 WebSocket）
  ④ 超时时间长（普通API < 1s，LLM API 可能 5-30s）
```

### 3.2 LangServe 模式

```
对应 fastapi_server.py 第 3 章：

LangServe 自动为一个 Chain 生成完整 API：

  add_routes(app, qa_chain, path="/langserve/qa")

  自动生成：
    POST /langserve/qa/invoke        → 同步调用
    POST /langserve/qa/stream        → 流式调用
    POST /langserve/qa/batch         → 批量调用
    GET  /langserve/qa/input_schema  → 输入格式
    GET  /langserve/qa/playground    → 调试 UI

  优势：零代码生成标准化 API
  劣势：自定义灵活性受限

对应 test_client.py 中的测试：
  test_langserve_invoke() → 测试 invoke 端点
  test_langserve_batch()  → 测试 batch 端点
```

---

## 4. 请求/响应模型设计

### 4.1 Pydantic Schema 设计

```python
# 对应 fastapi_server.py 第 2 章的数据模型定义

class QuestionRequest(BaseModel):
    """请求体——定义客户端必须发送什么"""
    question: str = Field(
        description="用户的问题",
        examples=["什么是量子计算？"],
        min_length=1,        # 不允许空字符串
        max_length=2000,     # 限制最大长度
    )

class QuestionResponse(BaseModel):
    """响应体——定义服务端返回什么"""
    question: str = Field(description="原始问题")
    answer: str = Field(description="AI 的回答")
    model: str = Field(description="使用的模型名称")

# Pydantic 自动实现：
#   ① 类型校验（question 必须是 str）
#   ② 文档生成（Swagger UI 展示 description）
#   ③ 序列化（Python 对象 ↔ JSON 自动转换）
```

### 4.2 设计原则

```
好的 API Schema 设计原则：

  ① 字段命名清晰（question 而非 q）
  ② 提供 examples（前端参考）
  ③ 设置合理约束（min_length, max_length）
  ④ 响应包含请求回显（方便调试）
  ⑤ 错误响应标准化

  请求格式（客户端发送）：
  {
    "question": "什么是量子计算？"
  }

  成功响应格式：
  {
    "question": "什么是量子计算？",
    "answer": "量子计算是利用量子力学原理...",
    "model": "gpt-4"
  }

  错误响应格式：
  {
    "detail": "LLM 调用失败: rate limit exceeded"
  }
```

---

## 5. 流式 API 实现

### 5.1 SSE（Server-Sent Events）

```
对应 fastapi_server.py 的流式端点实现：

SSE 协议格式：
  Content-Type: text/event-stream

  data: {"token": "量"}

  data: {"token": "子"}

  data: {"token": "计算"}

  data: [DONE]

  每条消息以 "data: " 开头，以两个换行 "\n\n" 结束。

为什么 LLM 需要流式？
  普通接口：用户等 5 秒才看到完整回答
  流式接口：0.3 秒后逐字显示（ChatGPT 的打字效果）
```

### 5.2 FastAPI 流式实现

```python
# 对应 fastapi_server.py 第 2 章 qa_stream_endpoint

@app.post("/api/qa/stream")
async def qa_stream_endpoint(request: QuestionRequest):
    async def generate_sse():
        """异步生成器：逐 token 产出 SSE 事件"""
        async for chunk in qa_chain.astream({"question": request.question}):
            if chunk:
                data = json.dumps({"token": chunk}, ensure_ascii=False)
                yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

# 对应 test_client.py 的 test_stream() 客户端消费：
for line in resp.iter_lines(decode_unicode=True):
    if line.startswith("data: "):
        data_str = line[6:]
        if data_str == "[DONE]": break
        token = json.loads(data_str).get("token", "")
```

---

## 6. 认证与鉴权

### 6.1 API Key 认证

```python
# 最简单的认证方式——适合内部服务/小规模使用

from fastapi import Header, HTTPException

VALID_API_KEYS = {"sk-abc123", "sk-def456"}

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="无效的 API Key")

@app.post("/api/qa", dependencies=[Depends(verify_api_key)])
async def qa_endpoint(request: QuestionRequest):
    ...

# 客户端调用：
# curl -H "X-API-Key: sk-abc123" -X POST /api/qa
```

### 6.2 JWT 认证

```
JWT（JSON Web Token）认证流程：

  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │  客户端   │     │  认证服务 │     │  API服务  │
  └────┬─────┘     └────┬─────┘     └────┬─────┘
       │ 1.登录请求     │                 │
       │───────────────→│                 │
       │ 2.返回JWT Token│                 │
       │←───────────────│                 │
       │                                  │
       │ 3.带JWT请求API                    │
       │─────────────────────────────────→│
       │               4.验证JWT签名       │
       │ 5.返回数据                        │
       │←─────────────────────────────────│

JWT 结构：
  Header.Payload.Signature
  {"alg":"HS256"}.{"user_id":"u123","exp":1234567890}.签名

优势：无状态（服务端不需存储 session）
适用：多服务架构、前端直接调用
```

### 6.3 OAuth2

```
适用场景：第三方应用接入你的 LLM API

  ① 用户授权第三方 App 使用你的 AI 服务
  ② 第三方 App 获得 access_token
  ③ 用 access_token 调用你的 API
  ④ Token 过期后用 refresh_token 续期

FastAPI 内置 OAuth2 支持：
  from fastapi.security import OAuth2PasswordBearer
  oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
```

---

## 7. 速率限制（Rate Limiting）

### 7.1 为什么需要限流

```
LLM API 调用成本高（每次调用都消耗 GPU 算力和费用）：

  不限流的后果：
    ① 单用户刷爆 API → 其他用户无法使用
    ② 恶意攻击 → 账单爆炸
    ③ 下游 LLM 限流 → 你的服务全部 429 错误

  限流策略：
    RPM（Requests Per Minute）= 每分钟请求数
    TPM（Tokens Per Minute）= 每分钟 Token 消耗
```

### 7.2 实现方式

```python
# 方式一：滑动窗口（内存实现）
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)  # user_id -> [timestamps]

    def is_allowed(self, user_id: str) -> bool:
        now = time.time()
        # 清理过期记录
        self.requests[user_id] = [
            t for t in self.requests[user_id]
            if now - t < self.window
        ]
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        self.requests[user_id].append(now)
        return True

# 方式二：令牌桶（Redis 实现，适合分布式）
# 每秒往桶里放 N 个令牌，请求来了取一个，桶空则拒绝

# 方式三：使用 slowapi 库
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/qa")
@limiter.limit("10/minute")
async def qa_endpoint(request: Request, body: QuestionRequest):
    ...
```

---

## 8. 负载均衡策略

### 8.1 部署架构

```
生产环境多实例部署：

  客户端请求
       ↓
  ┌──────────────┐
  │  Nginx/ALB   │  ← 负载均衡器
  └──┬───┬───┬───┘
     ↓   ↓   ↓
  ┌────┐┌────┐┌────┐
  │实例1││实例2││实例3│  ← FastAPI 服务实例
  └────┘└────┘└────┘
     ↓   ↓   ↓
  ┌──────────────┐
  │  LLM API    │  ← 下游 LLM 服务
  └──────────────┘
```

### 8.2 均衡算法

```
┌─────────────────┬──────────────────────────────────────────┐
│  算法            │  适用场景                                 │
├─────────────────┼──────────────────────────────────────────┤
│  轮询           │  实例配置相同                              │
│  加权轮询       │  实例配置不同（按算力分配）                │
│  最少连接       │  LLM 请求耗时差异大时最佳                  │
│  一致性哈希     │  需要会话亲和性（缓存命中）                │
└─────────────────┴──────────────────────────────────────────┘

LLM 服务推荐：最少连接（Least Connections）
  因为 LLM 请求耗时差异很大（1s ~ 30s），
  最少连接能避免慢请求堆积在某个实例上。
```

---

## 9. 连接池与资源管理

### 9.1 HTTP 连接池

```python
# LLM API 客户端应复用连接（避免每次请求建立 TCP 连接）

import httpx

# 全局连接池（应用启动时创建，关闭时销毁）
http_client = httpx.AsyncClient(
    timeout=60.0,
    limits=httpx.Limits(
        max_connections=100,       # 最大连接数
        max_keepalive_connections=20,  # 保活连接数
    ),
)

# FastAPI 生命周期管理
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    yield
    # 关闭时
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)
```

### 9.2 LLM 客户端复用

```
对应 fastapi_server.py 的全局 LLM 实例：

  llm = ChatOpenAI(...)  ← 全局单例，所有请求共享

  原因：
    ① ChatOpenAI 内部维护了 HTTP 连接池
    ② 每次请求创建新实例 = 每次新建 TCP 连接（慢！）
    ③ 连接池自动管理连接的创建、复用、回收

  注意事项：
    ① ChatOpenAI 是线程安全和协程安全的
    ② 不需要加锁就能并发使用
    ③ 但要注意下游的并发限制（RPM/TPM）
```

---

## 10. 错误处理与标准化响应

### 10.1 错误分类

```
对应 fastapi_server.py 中的 HTTPException 使用：

  ┌─────────────────────────────────────────────────────────────┐
  │  HTTP 状态码   │  含义             │  LLM 场景             │
  ├─────────────────────────────────────────────────────────────┤
  │  400          │  请求参数错误      │  问题为空/格式不对    │
  │  401          │  未认证           │  缺少 API Key        │
  │  403          │  无权限           │  API Key 无此接口权限 │
  │  422          │  数据验证失败      │  Pydantic 校验不通过 │
  │  429          │  请求过多         │  触发限流            │
  │  500          │  服务内部错误      │  LLM 调用失败       │
  │  502          │  上游服务错误      │  LLM API 返回错误   │
  │  503          │  服务不可用       │  模型过载           │
  │  504          │  网关超时         │  LLM 响应超时       │
  └─────────────────────────────────────────────────────────────┘
```

### 10.2 标准化错误响应

```python
# 统一错误格式

from fastapi import Request
from fastapi.responses import JSONResponse

class APIError(BaseModel):
    code: str           # 机器可读错误码
    message: str        # 人类可读描述
    details: dict = {}  # 额外信息

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": "服务内部错误，请稍后重试",
            "details": {"error_type": type(exc).__name__},
        },
    )

# 对应 fastapi_server.py 中：
#   raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")
```

---

## 11. 健康检查与优雅关闭

### 11.1 健康检查

```
对应 fastapi_server.py 第 4 章：

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_NAME, ...}

K8s 探针配置对应：
  livenessProbe:   服务是否存活（死了就重启）
    → GET /health 返回 200

  readinessProbe:  服务是否就绪（没准备好就不分流量）
    → GET /health 返回 200 且 LLM 连接正常

  startupProbe:    服务是否启动完成（模型加载需要时间）
    → GET /health 首次返回 200

深度健康检查（检查下游依赖）：
  async def deep_health():
      try:
          await llm.ainvoke({"question": "ping"})  # 实际调一下LLM
          return {"status": "healthy", "llm": "connected"}
      except:
          return {"status": "degraded", "llm": "disconnected"}
```

### 11.2 优雅关闭（Graceful Shutdown）

```
服务关闭时的正确做法：

  ① 停止接收新请求
  ② 等待正在处理的请求完成（设超时）
  ③ 关闭连接池
  ④ 退出进程

  uvicorn 默认支持 SIGTERM 优雅关闭：
    收到 SIGTERM → 停止接收 → 等待进行中请求 → 退出

  配置等待时间：
    uvicorn app:app --timeout-graceful-shutdown 30
```

---

## 12. 部署方案

### 12.1 Docker 部署

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 生产用 gunicorn + uvicorn worker
CMD ["gunicorn", "fastapi_server:app",
     "-k", "uvicorn.workers.UvicornWorker",
     "-w", "4",
     "--bind", "0.0.0.0:8000",
     "--timeout", "120"]
```

### 12.2 Kubernetes 部署

```yaml
# deployment.yaml 核心配置
spec:
  replicas: 3                    # 3 个实例
  containers:
    - name: llm-api
      image: llm-api:latest
      resources:
        requests: {cpu: "500m", memory: "512Mi"}
        limits: {cpu: "2000m", memory: "2Gi"}
      livenessProbe:
        httpGet: {path: /health, port: 8000}
        periodSeconds: 10
      readinessProbe:
        httpGet: {path: /health, port: 8000}
        initialDelaySeconds: 5
  # HPA 自动伸缩
  autoscaling:
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilization: 70%
```

### 12.3 Serverless 部署

```
适用场景：流量不稳定、不想管服务器

  AWS Lambda + API Gateway
  阿里云函数计算
  Vercel Serverless Functions

  优势：按调用付费、自动伸缩、零运维
  劣势：冷启动延迟（首次请求慢 2-5s）
        不适合流式响应（部分平台不支持）
        执行时间限制（通常 < 300s）

  LLM 场景评估：
    低频调用（<100/天）→ Serverless 省钱
    高频调用（>1000/天）→ 常驻服务更划算
```

---

## 13. 性能优化

### 13.1 缓存策略

```
相同问题直接返回缓存结果（避免重复调用 LLM）：

  语义缓存（推荐）：
    不是严格匹配问题文本，而是匹配语义相似的问题
    "量子计算是什么" ≈ "什么是量子计算" → 命中缓存

  实现：
    from langchain.cache import InMemoryCache
    import langchain
    langchain.llm_cache = InMemoryCache()

    # 或用 Redis 缓存（生产推荐）
    from langchain_community.cache import RedisSemanticCache

  缓存命中率预期：
    通用问答：20-40%
    客服FAQ：60-80%
    翻译：30-50%
```

### 13.2 批处理与异步队列

```
高并发场景的处理模式：

  模式一：同步串行（最慢）
    请求1 → LLM → 响应1 → 请求2 → LLM → 响应2
    总时间 = N × 单次时间

  模式二：异步并发（快，对应 async_concurrent/ 项目）
    请求1,2,3... 同时 → LLM → 同时响应
    总时间 ≈ 单次时间

  模式三：异步队列（最稳健）
    请求 → 消息队列(Redis/RabbitMQ) → Worker消费 → 回调通知
    适用：可以接受延迟的批量任务（报告生成、数据分析）

  对应 fastapi_server.py 中：
    单次请求用 await qa_chain.ainvoke()
    LangServe 的 /batch 端点实现了批量调用
```

---

## 14. 监控与告警

### 14.1 关键指标（Metrics）

```
LLM API 服务必须监控的指标：

  ┌─────────────────────────────────────────────────────────────┐
  │  延迟指标：                                                  │
  │    P50/P95/P99 响应时间                                     │
  │    首 Token 延迟（TTFT, Time To First Token）               │
  │    完整响应延迟（TTLR, Time To Last Response）              │
  │                                                             │
  │  吞吐指标：                                                  │
  │    QPS（每秒请求数）                                         │
  │    并发连接数                                                │
  │    Token 消耗速率                                            │
  │                                                             │
  │  错误指标：                                                  │
  │    错误率（4xx/5xx 占比）                                    │
  │    LLM 调用失败率                                           │
  │    超时率                                                    │
  │                                                             │
  │  资源指标：                                                  │
  │    CPU/内存使用率                                            │
  │    连接池利用率                                              │
  │    队列堆积长度                                              │
  └─────────────────────────────────────────────────────────────┘
```

### 14.2 告警规则

```
建议的告警阈值：

  P99 延迟 > 30s           → 告警（LLM 响应异常慢）
  错误率 > 5%              → 告警（可能服务异常）
  QPS 超过容量 80%         → 告警（需要扩容）
  LLM 调用失败率 > 10%    → 告警（下游可能故障）
  连接池使用率 > 90%       → 告警（需要增加连接）

实现工具：
  Prometheus + Grafana（指标采集 + 可视化）
  OpenTelemetry（分布式链路追踪）
  ELK Stack（日志聚合分析）
```

---

## 15. API 版本管理

### 15.1 版本策略

```
为什么需要版本管理？

  ① 接口升级不能破坏老客户端
  ② 新旧版本需要共存一段时间
  ③ 灰度发布需要按版本分流

版本号方案：
  URL 路径版本：  /v1/api/qa, /v2/api/qa（推荐）
  请求头版本：    X-API-Version: 2
  查询参数版本：  /api/qa?version=2

推荐做法（URL 路径版本）：
  from fastapi import APIRouter

  v1_router = APIRouter(prefix="/v1")
  v2_router = APIRouter(prefix="/v2")

  @v1_router.post("/qa")
  async def qa_v1(...): ...

  @v2_router.post("/qa")
  async def qa_v2(...): ...  # 新增 streaming 参数

  app.include_router(v1_router)
  app.include_router(v2_router)
```

### 15.2 版本生命周期

```
版本管理最佳实践：

  v1 发布 → v2 发布 → v1 标记废弃 → v1 下线
  │         │         │              │
  ├─────────┼─────────┼──────────────┤
  │  v1 活跃 │ v1+v2  │ v1 废弃期   │ v1 下线
              共存     （返回 Warning）

  废弃期至少 3-6 个月
  下线前通知所有调用方迁移
```

---

## 附录 A：代码与知识点对应

| 代码位置 | 覆盖知识点 | 对应本文章节 |
|---------|------------|-------------|
| `fastapi_server.py` FastAPI() | 应用创建、ASGI | 第2节 |
| `fastapi_server.py` Pydantic模型 | 请求响应设计 | 第4节 |
| `fastapi_server.py` StreamingResponse | 流式SSE实现 | 第5节 |
| `fastapi_server.py` add_routes | LangServe模式 | 第3节 |
| `fastapi_server.py` /health | 健康检查 | 第11节 |
| `fastapi_server.py` uvicorn.run | 服务启动 | 第2、12节 |
| `test_client.py` 全部函数 | 客户端调用验证 | 第3-5节 |

---

## 附录 B：推荐学习路径

```
入门阶段（3天）：
  运行 fastapi_server.py + test_client.py
  打开 http://localhost:8000/docs 体验 Swagger UI
  第1-5节 → 理解基础架构和流式实现

进阶阶段（1周）：
  第6-10节 → 实现认证、限流、错误处理
  给现有代码添加 API Key 认证

生产阶段（2周）：
  第11-15节 → 部署、监控、版本管理
  用 Docker 打包部署
  接入 Prometheus 监控
```

---

> **下一步学习**：运行 `fastapi_server.py` 启动服务，然后用 `test_client.py` 验证所有端点。之后前往 `async_concurrent/KNOWLEDGE.md` 深入理解异步并发原理。
