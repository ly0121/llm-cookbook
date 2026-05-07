---
title: 错误处理
---

# 错误处理与重试

LLM API 调用面临网络波动、速率限制等问题，完善的容错机制是生产系统的必备能力。

## 1. 错误分类

| 层级 | 错误类型 | 频率 |
|------|---------|------|
| 网络层 | ConnectionError, TimeoutError | 5-10% |
| API 层 | 429 Rate Limit | 40-50%（最常见） |
| API 层 | 500/502/503 服务端错误 | 15-20% |
| 业务层 | OutputParserError | 3-5% |

## 2. 可重试 vs 不可重试

| 可重试（暂时性错误） | 不可重试（永久性错误） |
|---------------------|---------------------|
| 429 Rate Limit | 401 Unauthorized |
| 500/502/503 | 400 Bad Request |
| Timeout | 上下文超长 |
| 网络中断 | 内容安全过滤 |

## 3. 重试策略

| 策略 | 做法 | 适用 |
|------|------|------|
| 固定间隔 | 每次等 2s | 简单场景 |
| 指数退避 | 2s→4s→8s→16s | 推荐 |
| 指数退避+抖动 | 加随机偏移 | 高并发（避免惊群） |

## 4. Tenacity 库

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((TimeoutError, ConnectionError))
)
async def call_llm(prompt):
    return await llm.ainvoke(prompt)
```

## 5. 断路器模式（Circuit Breaker）

连续失败时"熔断"，避免雪崩：

```
关闭状态 → 正常调用
         → 连续失败 N 次
打开状态 → 直接返回降级结果（不调用 API）
         → 等待冷却时间
半开状态 → 尝试一次调用
         → 成功: 恢复关闭
         → 失败: 回到打开
```

## 6. 降级方案（Fallback）

```python
async def call_with_fallback(prompt):
    try:
        return await call_gpt4(prompt)       # 主模型
    except Exception:
        try:
            return await call_gpt35(prompt)  # 降级模型
        except Exception:
            return "抱歉，服务暂时不可用"     # 兜底
```

## 7. 速率限制处理

```python
import asyncio

async def handle_rate_limit(func, *args):
    for attempt in range(5):
        try:
            return await func(*args)
        except RateLimitError as e:
            wait_time = e.retry_after or (2 ** attempt)
            await asyncio.sleep(wait_time)
    raise Exception("Rate limit exceeded after retries")
```

## 8. 输出解析错误修复

```python
from langchain.output_parsers import RetryOutputParser

retry_parser = RetryOutputParser.from_llm(
    parser=pydantic_parser,
    llm=llm,
    max_retries=2,
)
# 解析失败时自动让 LLM 修正输出格式
```

## 9. 全局异常处理架构

```
请求 → [输入校验] → [速率限制] → [LLM 调用(重试+超时)]
                                        ↓ 失败
                                   [降级方案]
                                        ↓ 失败
                                   [兜底响应]
     → [输出校验] → [响应]
```

::: warning 需要本地运行
完整实现见 `error_handling/retry_demo.py`，包含 Tenacity 重试、断路器、降级链等完整代码。
:::

---

::: tip 下一步
- [可观测性](/engineering/observability) — 错误监控与告警
- [API 服务](/production/api-service) — 服务端全局异常处理
:::
