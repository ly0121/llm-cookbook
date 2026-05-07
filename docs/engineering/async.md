---
title: 异步与并发
---

# 异步与并发编程

LLM 调用是 IO 密集型操作（99% 时间在等网络），异步编程可将 3 个 10 秒的请求从串行 30 秒压缩到并行 10 秒。

## 1. 同步 vs 异步

```
同步（排队模式）：
  请求1 ████████  请求2 ████████  请求3 ████████
  总耗时: 30s

异步（并发模式）：
  请求1 ████████
  请求2 ████████
  请求3 ████████
  总耗时: 10s    加速比 = 3x
```

## 2. Python asyncio 核心概念

| 概念 | 说明 |
|------|------|
| Event Loop | 事件循环，调度协程执行 |
| coroutine | `async def` 定义的协程函数 |
| await | 挂起当前协程，让出控制权 |
| asyncio.gather | 并发执行多个协程 |
| Semaphore | 控制最大并发数 |

## 3. 并发 LLM 调用

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

async def concurrent_calls(prompts: list[str]):
    tasks = [llm.ainvoke(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results

# 同时发送 10 个请求
results = asyncio.run(concurrent_calls(prompts))
```

## 4. 信号量控制并发

避免触发 API Rate Limit：

```python
semaphore = asyncio.Semaphore(5)  # 最多同时 5 个请求

async def limited_call(prompt):
    async with semaphore:
        return await llm.ainvoke(prompt)

tasks = [limited_call(p) for p in prompts]
results = await asyncio.gather(*tasks)
```

## 5. 批量处理策略

| 策略 | 实现 | 适用场景 |
|------|------|---------|
| gather | `asyncio.gather(*tasks)` | 少量并发 |
| 分批 | 每批 N 个，batch 间 sleep | 大量请求 |
| 队列 | Producer-Consumer 模式 | 持续流入 |

## 6. 速率限制

```python
import time

class RateLimiter:
    def __init__(self, max_per_minute: int):
        self.interval = 60.0 / max_per_minute
        self.last_call = 0

    async def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.interval:
            await asyncio.sleep(self.interval - elapsed)
        self.last_call = time.time()
```

## 7. 多线程 vs 多进程 vs 异步

| 方式 | 适用场景 | LLM 调用推荐 |
|------|---------|-------------|
| 异步 IO | IO 密集（网络请求） | 首选 |
| 多线程 | IO 密集（兼容同步库） | 备选 |
| 多进程 | CPU 密集（计算） | 不适合 |

## 8. 错误处理

```python
async def safe_call(prompt):
    try:
        return await asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        return "超时"
    except Exception as e:
        return f"错误: {e}"
```

::: warning 需要本地运行
完整实现见 `async_concurrent/async_demo.py`，包含性能对比基准测试。
:::

---

::: tip 下一步
- [缓存](/engineering/caching) — 避免重复调用，降低成本
- [错误处理](/engineering/error-handling) — 重试与容错机制
:::
