---
title: 可观测性
---

# LLM 可观测性（Observability）

LLM 应用的不确定性和多步骤链路使得可观测性成为生产必需品。

## 1. 三支柱

| 支柱 | 回答的问题 | 示例 |
|------|-----------|------|
| Logging | 发生了什么？ | "LLM调用完成 耗时2.1s" |
| Tracing | 怎么发生的？ | 请求→Prompt→LLM→解析 各步耗时 |
| Metrics | 整体怎么样？ | P99=3.2s, 错误率=0.5% |

**排查流程：** Metrics 发现异常 → Tracing 定位环节 → Logging 查看细节

## 2. LLM 应用的特殊挑战

| 挑战 | 传统软件 | LLM 应用 |
|------|---------|---------|
| 确定性 | 同输入同输出 | 同输入不同输出 |
| 链路 | 请求→处理→响应 | 嵌入→检索→重排→Prompt→LLM→解析 |
| 成本 | 固定 | 按 token 计费，需实时监控 |

## 3. LangSmith

LangChain 官方可观测性平台：

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="ls-xxx"
# 代码不用改，自动发送 trace 数据
```

功能：链路追踪、评估、监控、Playground 调试、数据集管理。

## 4. LangFuse（开源替代）

```python
from langfuse.callback import CallbackHandler

handler = CallbackHandler(
    public_key="pk-xxx",
    secret_key="sk-xxx",
)
chain.invoke(input, config={"callbacks": [handler]})
```

## 5. Callbacks 系统

```python
from langchain_core.callbacks import BaseCallbackHandler

class CostTracker(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        self.total_tokens += usage.get("total_tokens", 0)
        print(f"累计 tokens: {self.total_tokens}")
```

## 6. 关键监控指标

| 指标 | 说明 | 告警阈值 |
|------|------|---------|
| TTFT | 首 token 延迟 | >3s |
| E2E Latency | 端到端延迟 | >10s |
| Error Rate | 错误率 | >5% |
| Token/Request | 平均 token 消耗 | 突增 50% |
| Cost/Day | 日成本 | 超预算 |
| Cache Hit Rate | 缓存命中率 | <50% |

## 7. 日志最佳实践

```python
import structlog

logger = structlog.get_logger()

logger.info("llm_call_complete",
    model="gpt-4o",
    latency_ms=2100,
    input_tokens=150,
    output_tokens=300,
    cache_hit=False,
    session_id="user_001",
)
```

原则：结构化 JSON 格式、包含 trace_id、不记录敏感内容。

## 8. 生产架构

```
应用 → [Callbacks] → LangSmith/LangFuse（追踪）
                   → Prometheus（指标）
                   → ELK/Loki（日志）
                   → Grafana（告警面板）
```

::: warning 需要本地运行
完整实现见 `observability/tracing_demo.py`。
:::

---

::: tip 下一步
- [API 服务](/production/api-service) — 在服务中集成监控
- [评估体系](/production/evaluation) — 用可观测数据驱动评估
:::
