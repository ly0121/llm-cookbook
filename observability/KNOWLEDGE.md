# LLM 可观测性（Observability）完全知识手册

> 本文档系统性地覆盖 LLM 应用中可观测性的所有核心知识点。
> 配合 `tracing_demo.py` 代码阅读效果更佳。

---

## 目录

1. [可观测性三支柱](#1-可观测性三支柱logging-tracing-metrics)
2. [为什么 LLM 应用特别需要可观测性](#2-为什么-llm-应用特别需要可观测性)
3. [分布式追踪原理](#3-分布式追踪原理)
4. [LangSmith 详解](#4-langsmith-详解)
5. [LangFuse 详解](#5-langfuse-详解)
6. [OpenTelemetry 集成](#6-opentelemetry-集成)
7. [LangChain Callbacks 系统详解](#7-langchain-callbacks-系统详解)
8. [自定义 Callback Handler](#8-自定义-callback-handler)
9. [关键指标监控](#9-关键指标监控)
10. [日志最佳实践](#10-日志最佳实践)
11. [告警策略设计](#11-告警策略设计)
12. [成本监控与预算控制](#12-成本监控与预算控制)
13. [性能瓶颈分析](#13-性能瓶颈分析)
14. [生产环境可观测性架构](#14-生产环境可观测性架构)

---

## 1. 可观测性三支柱（Logging, Tracing, Metrics）

### 1.1 三支柱概览

```
对应 tracing_demo.py 前置科普二

  ┌──────────────┬────────────────────────────────────────────┐
  │  支柱         │  说明                                       │
  ├──────────────┼────────────────────────────────────────────┤
  │  Logging     │  离散事件记录：什么时候发生了什么事          │
  │  (日志)      │  "2024-01-15 10:30:22 LLM调用完成 耗时2.1s" │
  │              │                                            │
  │  Tracing     │  因果链路追踪：一次请求经过了哪些步骤       │
  │  (追踪)      │  请求→Prompt→LLM→解析→返回 各步耗时        │
  │              │                                            │
  │  Metrics     │  聚合统计指标：系统整体健康状况             │
  │  (指标)      │  P99延迟=3.2s, 错误率=0.5%, 日消耗=10万t   │
  └──────────────┴────────────────────────────────────────────┘
```

### 1.2 三者关系

```
  Logging → 回答"发生了什么？"（单个事件）
  Tracing → 回答"怎么发生的？"（因果链条）
  Metrics → 回答"整体怎么样？"（宏观趋势）

  问题排查流程:
    Metrics 发现异常 (错误率飙升)
       ↓
    Tracing 定位链路 (哪个环节出问题)
       ↓
    Logging 查看细节 (具体什么错误)
```

---

## 2. 为什么 LLM 应用特别需要可观测性

### 2.1 LLM 应用的特殊挑战

```
对应 tracing_demo.py 前置科普一

  ┌─────────────────────────────────────────────────────────────┐
  │  挑战一：不确定性                                            │
  │    传统软件: 同一输入 → 同一输出（确定性）                    │
  │    LLM 应用: 同一输入 → 不同输出（随机性）                    │
  │    → 必须记录每次输入和输出，才能排查问题                     │
  │                                                             │
  │  挑战二：多步骤链路                                          │
  │    RAG: 用户问题 → 嵌入 → 检索 → 重排 → Prompt → LLM → 解析 │
  │    哪一步出错了？哪一步最慢？没有追踪就是黑盒！              │
  │                                                             │
  │  挑战三：成本不可预测                                        │
  │    每次调用消耗不同数量的 Token                               │
  │    不监控 → 月底账单吓死人                                   │
  │                                                             │
  │  挑战四：质量难以量化                                        │
  │    "回答正确吗？" "幻觉了吗？" "有害内容吗？"               │
  │    需要持续监控和评估                                        │
  └─────────────────────────────────────────────────────────────┘
```

### 2.2 可观测性能回答的关键问题

```
  运营问题:
    - 今天花了多少钱？(成本监控)
    - 哪个功能最费钱？(成本归因)
    - 用户最常问什么？(需求分析)

  质量问题:
    - 回答准确率是多少？(质量指标)
    - 哪类问题容易出错？(弱点分析)
    - 模型升级后效果变好了吗？(A/B 测试)

  性能问题:
    - 为什么这个请求这么慢？(瓶颈定位)
    - 平均延迟是多少？P99 呢？(SLA 监控)
    - 缓存命中率如何？(优化方向)
```

---

## 3. 分布式追踪原理

### 3.1 核心概念

```
  Trace（追踪）: 一次完整请求的全生命周期

  Span（跨度）: Trace 中的一个操作步骤

  Context Propagation（上下文传播）: 跨服务传递追踪信息

  示例：一次 RAG 问答的 Trace 结构:

  Trace: "用户问'什么是AI'"
  │
  ├── Span: chain_start (总计 3200ms)
  │   ├── Span: embedding (120ms)
  │   │   └── 输入: "什么是AI" → 输出: [0.12, 0.34, ...]
  │   │
  │   ├── Span: vector_search (80ms)
  │   │   └── 检索到 3 个文档
  │   │
  │   ├── Span: prompt_format (2ms)
  │   │   └── 拼接 prompt + 检索结果
  │   │
  │   ├── Span: llm_call (2800ms)  ← 最慢!
  │   │   ├── model: gpt-4
  │   │   ├── tokens: 入500 出200
  │   │   └── 输出: "AI是..."
  │   │
  │   └── Span: output_parse (5ms)
  │       └── 格式化输出
  │
  总延迟: 3200ms, 其中 LLM 占 87%
```

### 3.2 Span 数据结构

```
  {
    "trace_id": "abc123",          # 唯一追踪 ID
    "span_id": "span_456",         # 当前 Span ID
    "parent_span_id": "span_123",  # 父 Span ID (形成树结构)
    "operation": "llm_call",       # 操作名称
    "start_time": "2024-01-15T10:30:22.000Z",
    "end_time": "2024-01-15T10:30:24.800Z",
    "duration_ms": 2800,
    "status": "ok",
    "attributes": {
      "model": "gpt-4",
      "prompt_tokens": 500,
      "completion_tokens": 200,
      "temperature": 0.7
    }
  }
```

---

## 4. LangSmith 详解

### 4.1 概述

```
LangSmith = LangChain 官方的可观测性 + 评估平台

  核心功能:
  ┌────────────────────────────────────────────────────────┐
  │  1. 追踪 (Tracing)                                     │
  │     自动记录 LangChain 链路的每一步                      │
  │     输入/输出/延迟/Token/错误 全部可视化                 │
  │                                                        │
  │  2. 评估 (Evaluation)                                   │
  │     定义评估指标 → 跑数据集 → 自动打分                  │
  │     比较不同 Prompt/模型的效果                           │
  │                                                        │
  │  3. 数据集管理 (Datasets)                               │
  │     收集生产中的 (输入,输出) 对                          │
  │     标注 → 构建评估集 → 持续迭代                        │
  │                                                        │
  │  4. 监控 (Monitoring)                                   │
  │     实时仪表盘: 延迟、成本、错误率                       │
  │     告警: 指标异常时通知                                 │
  └────────────────────────────────────────────────────────┘
```

### 4.2 接入方式

```python
# 只需设置环境变量，LangChain 自动上报追踪数据
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_xxx..."
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# 之后所有 LangChain 调用自动被追踪，无需改代码!
chain = prompt | llm | parser
result = chain.invoke({"question": "什么是AI?"})
# → LangSmith 仪表盘自动显示这次调用的完整追踪
```

### 4.3 优缺点

```
优点:
  - 与 LangChain 深度集成（零代码改动）
  - UI 强大，可视化效果好
  - 内置评估框架
  - 支持在线标注和协作

缺点:
  - 商业产品（免费额度有限）
  - 数据存在外部（合规敏感场景不适用）
  - 绑定 LangChain 生态
```

---

## 5. LangFuse 详解

### 5.1 概述

```
LangFuse = 开源的 LLM 可观测性平台

  与 LangSmith 定位相同，但:
  - 开源 (MIT License)
  - 可自部署 (Docker / Kubernetes)
  - 数据留在自己的基础设施中
  - 不绑定特定框架 (LangChain/LlamaIndex/自定义都支持)
```

### 5.2 接入方式

```python
# 方式一: LangChain Callback 集成
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    public_key="pk-xxx",
    secret_key="sk-xxx",
    host="https://your-langfuse-instance.com"
)

chain.invoke(
    {"question": "什么是AI?"},
    config={"callbacks": [langfuse_handler]}
)

# 方式二: Python SDK 手动追踪
from langfuse import Langfuse

langfuse = Langfuse()
trace = langfuse.trace(name="rag-query")
span = trace.span(name="llm-call", input={"prompt": "..."})
# ... 调用 LLM ...
span.end(output={"response": "..."}, metadata={"tokens": 500})
```

### 5.3 自部署架构

```
  ┌────────────────────────────────────────────┐
  │  你的应用服务器                             │
  │  ├── LangChain App                        │
  │  └── LangFuse SDK → 发送追踪数据          │
  └──────────────────────┬─────────────────────┘
                         ↓
  ┌────────────────────────────────────────────┐
  │  LangFuse Server (Docker)                  │
  │  ├── API Server                            │
  │  ├── Web UI (仪表盘)                       │
  │  └── PostgreSQL (数据存储)                  │
  └────────────────────────────────────────────┘
```

---

## 6. OpenTelemetry 集成

### 6.1 什么是 OpenTelemetry

```
OpenTelemetry (OTel) = 可观测性的"通用标准"

  传统:
    应用 → 自定义格式 → Datadog
    应用 → 自定义格式 → Jaeger
    应用 → 自定义格式 → 每换一个后端就要改代码

  OpenTelemetry:
    应用 → OTel 标准格式 → Exporter → Datadog / Jaeger / Grafana ...
                                       换后端不用改应用代码!
```

### 6.2 LLM 应用中的 OTel 集成

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# 初始化
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-app")

# 手动创建 Span
with tracer.start_as_current_span("llm_call") as span:
    span.set_attribute("llm.model", "gpt-4")
    span.set_attribute("llm.prompt_tokens", 500)
    result = llm.invoke(prompt)
    span.set_attribute("llm.completion_tokens", 200)
```

### 6.3 OTel 语义约定（LLM 扩展）

```
OpenTelemetry 正在制定 LLM 专用语义约定:

  gen_ai.system = "openai"           # AI 系统
  gen_ai.request.model = "gpt-4"    # 请求模型
  gen_ai.response.model = "gpt-4"   # 实际使用模型
  gen_ai.usage.prompt_tokens = 500  # 输入 token
  gen_ai.usage.completion_tokens = 200  # 输出 token
  gen_ai.request.temperature = 0.7  # 温度参数
```

---

## 7. LangChain Callbacks 系统详解

### 7.1 事件生命周期

```
对应 tracing_demo.py 第1章

LangChain 在执行链路时，会在关键时刻触发回调事件:

  Chain 执行:
    on_chain_start  → 链开始
    on_chain_end    → 链结束
    on_chain_error  → 链出错

  LLM 调用:
    on_llm_start           → LLM 开始
    on_llm_new_token       → 流式输出每个 token
    on_llm_end             → LLM 结束
    on_llm_error           → LLM 出错

  Tool 调用:
    on_tool_start   → 工具开始
    on_tool_end     → 工具结束
    on_tool_error   → 工具出错

  Retriever:
    on_retriever_start → 检索开始
    on_retriever_end   → 检索结束
```

### 7.2 Callback 传递方式

```python
# 方式一: 全局设置 (影响所有调用)
from langchain_core.globals import set_verbose, set_debug
set_verbose(True)   # 打印链路信息
set_debug(True)     # 打印详细调试信息

# 方式二: 构造时传入 (影响特定实例)
llm = ChatOpenAI(callbacks=[handler1, handler2])

# 方式三: 调用时传入 (最灵活, 推荐)
result = chain.invoke(
    {"question": "..."},
    config={"callbacks": [handler]}  # ← 推荐方式
)

# 对应 tracing_demo.py:
result = chain.invoke(
    {"question": q},
    config={"callbacks": [handler]},
)
```

---

## 8. 自定义 Callback Handler

### 8.1 基本结构

```python
# 对应 tracing_demo.py 第1章 ObservabilityHandler

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from uuid import UUID

class MyHandler(BaseCallbackHandler):
    """自定义回调处理器"""

    def on_llm_start(self, serialized, prompts, *, run_id: UUID, **kwargs):
        """LLM 开始时: 记录开始时间和输入"""
        self.start_time = time.time()
        print(f"LLM 开始: {prompts[0][:50]}...")

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs):
        """LLM 结束时: 计算延迟，提取 token 用量"""
        elapsed = time.time() - self.start_time
        token_usage = response.llm_output.get("token_usage", {})
        print(f"LLM 结束: {elapsed*1000:.0f}ms, {token_usage}")

    def on_llm_error(self, error, *, run_id: UUID, **kwargs):
        """LLM 出错时: 记录错误"""
        print(f"LLM 错误: {error}")
```

### 8.2 生产级 Handler 设计

```python
# 对应 tracing_demo.py 第3章 JSONLogHandler

class ProductionHandler(BaseCallbackHandler):
    """生产级 Handler: 结构化日志 + 指标收集"""

    def __init__(self):
        self._start_times = {}
        self.metrics = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_errors": 0,
            "latencies": [],
        }

    def on_llm_end(self, response, *, run_id, **kwargs):
        elapsed = time.time() - self._start_times[run_id]
        usage = response.llm_output.get("token_usage", {})

        # 结构化日志 (可被 ELK/Datadog 采集)
        log_entry = {
            "level": "INFO",
            "event": "llm_call_complete",
            "run_id": str(run_id)[:8],
            "latency_ms": round(elapsed * 1000),
            "tokens": usage.get("total_tokens", 0),
            "model": "gpt-4",
            "timestamp": datetime.now().isoformat(),
        }
        logger.info(json.dumps(log_entry))

        # 指标聚合
        self.metrics["total_calls"] += 1
        self.metrics["total_tokens"] += usage.get("total_tokens", 0)
        self.metrics["latencies"].append(elapsed)
```

---

## 9. 关键指标监控

### 9.1 核心指标体系

```
  ┌─────────────────────────────────────────────────────────────┐
  │  类别        │ 指标              │ 说明                      │
  ├─────────────────────────────────────────────────────────────┤
  │  延迟        │ avg_latency       │ 平均响应时间              │
  │              │ p50_latency       │ 50% 请求低于此值          │
  │              │ p95_latency       │ 95% 请求低于此值          │
  │              │ p99_latency       │ 99% 请求低于此值          │
  │              │ ttft              │ 首 Token 时间(流式)       │
  ├─────────────────────────────────────────────────────────────┤
  │  Token       │ prompt_tokens     │ 输入 Token 总量           │
  │              │ completion_tokens │ 输出 Token 总量           │
  │              │ avg_tokens/call   │ 每次调用平均 Token        │
  ├─────────────────────────────────────────────────────────────┤
  │  成本        │ daily_cost        │ 日成本                    │
  │              │ cost_per_request  │ 每次请求成本              │
  │              │ cost_by_chain     │ 按链路归因成本            │
  ├─────────────────────────────────────────────────────────────┤
  │  可靠性      │ error_rate        │ 错误率                    │
  │              │ retry_rate        │ 重试率                    │
  │              │ timeout_rate      │ 超时率                    │
  │              │ fallback_rate     │ 降级率                    │
  ├─────────────────────────────────────────────────────────────┤
  │  业务        │ cache_hit_rate    │ 缓存命中率                │
  │              │ satisfaction      │ 用户满意度                │
  │              │ hallucination_rate│ 幻觉率                    │
  └─────────────────────────────────────────────────────────────┘
```

### 9.2 指标计算示例

```python
# 对应 tracing_demo.py 第2章 get_summary()

def compute_metrics(traces):
    """从追踪数据计算指标"""
    llm_calls = [t for t in traces if t["event"] == "llm_end"]

    latencies = [t["latency_ms"] for t in llm_calls]
    latencies.sort()

    return {
        "total_calls": len(llm_calls),
        "avg_latency_ms": sum(latencies) / len(latencies),
        "p50_latency_ms": latencies[len(latencies) // 2],
        "p95_latency_ms": latencies[int(len(latencies) * 0.95)],
        "p99_latency_ms": latencies[int(len(latencies) * 0.99)],
        "total_tokens": sum(t["token_usage"].get("total_tokens", 0) for t in llm_calls),
        "error_count": len([t for t in traces if t["event"] == "llm_error"]),
    }
```

---

## 10. 日志最佳实践

### 10.1 结构化日志 vs 非结构化日志

```
对应 tracing_demo.py 第3章

非结构化 (难以程序化分析):
  "2024-01-15 10:30:22 INFO LLM调用成功，耗时1234ms，消耗token 500"

结构化 JSON (可被日志系统自动解析):
  {
    "timestamp": "2024-01-15T10:30:22.000Z",
    "level": "INFO",
    "event": "llm_call_complete",
    "latency_ms": 1234,
    "tokens": 500,
    "model": "gpt-4",
    "trace_id": "abc123"
  }

结构化日志的优势:
  - ELK Stack 可自动索引每个字段
  - Grafana 可基于字段做聚合查询
  - 告警系统可基于字段值触发规则
```

### 10.2 日志级别规范

```
  DEBUG:  开发调试信息 (完整 prompt 内容、中间变量)
          生产环境通常关闭

  INFO:   正常业务事件 (调用成功、缓存命中)
          每次 LLM 调用记录一条

  WARNING: 需要关注但不影响功能 (重试成功、降级触发)
           运维需要定期检查

  ERROR:  影响功能的错误 (LLM 调用失败、解析错误)
          需要告警和处理

  CRITICAL: 系统级严重故障 (断路器打开、所有模型不可用)
            需要立即人工介入
```

### 10.3 敏感信息处理

```
LLM 日志中可能包含敏感信息:
  - 用户的隐私数据 (姓名、身份证、医疗记录)
  - API 密钥
  - 内部 System Prompt

处理策略:
  1. Prompt 内容只记录前 N 个字符 (preview)
  2. 用户输入做脱敏处理 (PII masking)
  3. API Key 永远不记录
  4. 分级存储 (含敏感信息的日志加密)
```

---

## 11. 告警策略设计

### 11.1 告警规则设计

```
  ┌──────────────────────────────────────────────────────────┐
  │  级别  │ 条件                       │ 通知方式           │
  ├──────────────────────────────────────────────────────────┤
  │  P1    │ 错误率 > 50% 持续 2min     │ 电话 + 短信        │
  │  P1    │ 断路器打开                  │ 电话 + 短信        │
  │  P1    │ 所有降级失败                │ 电话 + 短信        │
  ├──────────────────────────────────────────────────────────┤
  │  P2    │ 错误率 > 5% 持续 5min      │ 即时消息(钉钉)     │
  │  P2    │ P99延迟 > 10s 持续 5min    │ 即时消息           │
  │  P2    │ 429错误 > 50/min           │ 即时消息           │
  ├──────────────────────────────────────────────────────────┤
  │  P3    │ 日成本超预算 80%           │ 邮件               │
  │  P3    │ 缓存命中率 < 30%           │ 邮件               │
  │  P3    │ 错误率 > 1% 持续 30min     │ 邮件               │
  └──────────────────────────────────────────────────────────┘
```

### 11.2 避免告警疲劳

```
原则:
  1. 只告警可行动的问题 (有人需要做什么)
  2. 合并同类告警 (不要每个错误都发一条)
  3. 设置告警冷却期 (同一问题 30min 内只通知一次)
  4. 区分告警和通知 (P3 是通知，P1 才是告警)
  5. 定期 Review 告警规则 (删除无人响应的告警)
```

---

## 12. 成本监控与预算控制

### 12.1 成本追踪

```python
# 对应 tracing_demo.py 第4章 CostTracker

class CostTracker(BaseCallbackHandler):
    """Token 成本追踪器"""

    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},     # $/1M tokens
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    }

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        model = "gpt-4o"  # 从 serialized 中获取
        price = self.PRICING[model]
        cost = (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000

        self.total_cost += cost
```

### 12.2 预算控制策略

```
  ┌─────────────────────────────────────────────────────────┐
  │  预算控制层次                                            │
  │                                                         │
  │  Level 1: 软限制 (告警)                                 │
  │    日消耗 > 预算 80% → 发告警                           │
  │    仍然允许继续调用                                      │
  │                                                         │
  │  Level 2: 降级限制                                      │
  │    日消耗 > 预算 100% → 自动降级到便宜模型              │
  │    GPT-4 → GPT-3.5-turbo (成本降 20x)                  │
  │                                                         │
  │  Level 3: 硬限制 (熔断)                                 │
  │    日消耗 > 预算 150% → 停止 LLM 调用                   │
  │    返回兜底响应 + 通知管理员                             │
  └─────────────────────────────────────────────────────────┘
```

### 12.3 成本归因

```
按维度归因成本:
  - 按用户: user_123 本月消耗 ¥50
  - 按功能: RAG问答 ¥200/天, 摘要 ¥50/天, 翻译 ¥30/天
  - 按链路: embedding ¥20, LLM ¥180, rerank ¥10
  - 按模型: GPT-4 ¥150, GPT-3.5 ¥30, embedding ¥20

归因方式: 在每次调用时打标签 (tag)
  config={"tags": ["feature:rag", "user:123", "env:prod"]}
```

---

## 13. 性能瓶颈分析

### 13.1 常见瓶颈

```
  典型 RAG 链路耗时分布:

  ┌──────────────────────────────────────────────────────────┐
  │  步骤           │ 典型耗时    │ 占比    │ 优化方向        │
  ├──────────────────────────────────────────────────────────┤
  │  Embedding      │ 50-200ms   │ 3-5%   │ 批量/缓存       │
  │  向量搜索       │ 10-100ms   │ 1-3%   │ 索引优化        │
  │  Rerank         │ 200-500ms  │ 5-15%  │ 减少候选数      │
  │  LLM 生成      │ 1-5s       │ 75-90% │ 缓存/小模型     │
  │  输出解析       │ 1-10ms     │ <1%    │ 通常不是瓶颈    │
  └──────────────────────────────────────────────────────────┘

  结论: LLM 生成是绝对瓶颈 (占 75-90% 时间)
```

### 13.2 优化策略

```
  针对 LLM 生成慢:
    1. 缓存 (→ caching/KNOWLEDGE.md)
    2. 流式输出 (提升感知速度)
    3. 缩短 Prompt (减少 token → 生成更快)
    4. 降级到更快的模型 (GPT-3.5 比 GPT-4 快 3x)
    5. 并行调用 (多个独立 LLM 调用并发执行)

  针对 Embedding 慢:
    1. 批量 embed (一次性 embed 多个文本)
    2. 缓存 embedding 结果
    3. 使用本地小模型 (避免网络延迟)

  针对向量搜索慢:
    1. HNSW 索引 (近似最近邻，牺牲精度换速度)
    2. 缩小搜索范围 (pre-filter)
    3. GPU 加速搜索
```

---

## 14. 生产环境可观测性架构

### 14.1 完整架构图

```
  ┌────────────────────────────────────────────────────────────┐
  │                    LLM Application                         │
  │                                                            │
  │  ┌──────────────────────────────────────────────────────┐ │
  │  │  LangChain Chain                                     │ │
  │  │  ├── Callback Handler 1: StructuredLogger            │ │
  │  │  ├── Callback Handler 2: MetricsCollector            │ │
  │  │  └── Callback Handler 3: CostTracker                 │ │
  │  └──────────────────────────────────────────────────────┘ │
  └──────────┬─────────────────────┬──────────────┬───────────┘
             ↓                     ↓              ↓
  ┌──────────────┐    ┌──────────────────┐   ┌──────────────┐
  │  日志系统     │    │  指标系统         │   │  追踪系统     │
  │  ELK Stack   │    │  Prometheus      │   │  Jaeger /    │
  │  / Datadog   │    │  + Grafana       │   │  LangSmith   │
  └──────────────┘    └──────────────────┘   └──────────────┘
             ↓                     ↓              ↓
  ┌────────────────────────────────────────────────────────────┐
  │                   告警系统 (PagerDuty / 钉钉)               │
  │                                                            │
  │  规则引擎: 指标异常 → 触发告警 → 通知对应负责人            │
  └────────────────────────────────────────────────────────────┘
```

### 14.2 实施清单

```
  Phase 1: 基础可观测性 (1-2天)
    □ 实现自定义 Callback Handler (参考 tracing_demo.py)
    □ 记录每次 LLM 调用的延迟和 Token 消耗
    □ 输出结构化 JSON 日志

  Phase 2: 指标与告警 (3-5天)
    □ 接入 Prometheus (采集指标)
    □ 搭建 Grafana 仪表盘
    □ 配置基础告警规则 (错误率/延迟)

  Phase 3: 追踪与分析 (1-2周)
    □ 接入 LangSmith 或 LangFuse
    □ 实现成本归因 (按功能/用户)
    □ 建立性能基线 (Baseline)

  Phase 4: 持续优化
    □ 定期审查监控仪表盘
    □ A/B 测试新 Prompt/模型时用追踪对比
    □ 预算告警和自动降级
    □ 质量评估流水线 (定期跑评估集)
```

---

## 附录 A：本文件夹代码与知识点对应

| 代码位置 | 覆盖知识点 | 对应本文档章节 |
|---------|-----------|---------------|
| `tracing_demo.py` 前置科普 | 可观测性必要性、三支柱 | 第1-2节 |
| `tracing_demo.py` 第1章 | Callback Handler、事件生命周期 | 第7-8节 |
| `tracing_demo.py` 第2章 | 可观测性报告、指标聚合 | 第9节 |
| `tracing_demo.py` 第3章 | 结构化 JSON 日志 | 第10节 |
| `tracing_demo.py` 第4章 | Token 成本追踪 | 第12节 |

---

## 附录 B：推荐学习路径

```
入门（1天）：
  第1-2节 → 理解为什么需要可观测性
  第7-8节 → 运行 tracing_demo.py，理解 Callback 机制
  第10节 → 学习结构化日志

进阶（3-5天）：
  第3节 → 理解分布式追踪概念
  第4-5节 → 选择并接入 LangSmith 或 LangFuse
  第9节 → 定义关键监控指标

生产（1-2周）：
  第6节 → 接入 OpenTelemetry
  第11-12节 → 设计告警和预算控制
  第14节 → 搭建完整可观测性架构
```

---

> **下一步学习**：前往 `error_handling/KNOWLEDGE.md` 了解如何处理监控到的错误，或前往 `caching/KNOWLEDGE.md` 学习通过缓存优化成本和性能指标。
