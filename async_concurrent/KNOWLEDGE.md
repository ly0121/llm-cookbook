# 异步与并发编程（Async/Concurrent）完全知识手册

> 本文档系统讲解 Python 异步编程和并发 LLM 调用的所有核心知识点。
> 配合 `async_demo.py` 代码阅读效果更佳。

---

## 目录

1. [同步 vs 异步编程模型](#1-同步-vs-异步编程模型)
2. [Python asyncio 基础](#2-python-asyncio-基础)
3. [async/await 语法详解](#3-asyncawait-语法详解)
4. [并发 LLM 调用模式](#4-并发-llm-调用模式)
5. [信号量控制并发数](#5-信号量控制并发数)
6. [批量处理策略](#6-批量处理策略)
7. [速率限制实现](#7-速率限制实现)
8. [异步与同步代码的互操作](#8-异步与同步代码的互操作)
9. [多线程 vs 多进程 vs 异步 IO](#9-多线程-vs-多进程-vs-异步-io)
10. [生产者-消费者模式](#10-生产者-消费者模式)
11. [异步错误处理与超时控制](#11-异步错误处理与超时控制)
12. [aiohttp vs httpx 异步 HTTP 客户端](#12-aiohttp-vs-httpx-异步-http-客户端)
13. [性能对比与基准测试](#13-性能对比与基准测试)
14. [实际场景：并发调用多模型/多提示](#14-实际场景并发调用多模型多提示)

---

## 1. 同步 vs 异步编程模型

### 1.1 直觉理解

```
对应 async_demo.py 顶部的"奶茶店"比喻：

  同步（Synchronous）= 排队模式：
  ┌─────────────────────────────────────────────────────────────┐
  │  请求1 ████████████                                         │
  │  请求2             ████████████                              │
  │  请求3                         ████████████                  │
  │  总耗时：|──────────── 30s ────────────────|                │
  │  （3个请求 × 10s = 30s）                                    │
  └─────────────────────────────────────────────────────────────┘

  异步（Asynchronous）= 并发模式：
  ┌─────────────────────────────────────────────────────────────┐
  │  请求1 ████████████                                         │
  │  请求2 ████████████                                         │
  │  请求3 ████████████                                         │
  │  总耗时：|── 10s ──|                                        │
  │  （3个请求同时进行，取最慢的那个）                            │
  └─────────────────────────────────────────────────────────────┘

  加速比 = 同步耗时 / 异步耗时 = 30s / 10s = 3x
```

### 1.2 为什么 LLM 调用特别适合异步

```
LLM 调用的耗时分解：

  ┌──────┐    ┌────────────────────────┐    ┌──────┐
  │网络IO│    │    GPU 推理（等待中）    │    │网络IO│
  │ 50ms │    │      2000-5000ms       │    │ 50ms │
  └──────┘    └────────────────────────┘    └──────┘
               ↑ CPU 在这段时间完全空闲！

  同步模式：CPU 傻等 5 秒（浪费！）
  异步模式：CPU 去处理其他请求，LLM 回来了再继续

  关键洞察：
    LLM 调用是 IO 密集型（不是 CPU 密集型）
    99% 的时间在"等网络"，CPU 根本没干活
    异步就是让 CPU 在等待期间去干别的事
```

---

## 2. Python asyncio 基础

### 2.1 事件循环（Event Loop）

```
事件循环 = 异步编程的"调度中心"

  ┌─────────────────────────────────────────────────────────────┐
  │                     事件循环                                  │
  │                                                             │
  │    ┌─────┐    ┌─────┐    ┌─────┐                          │
  │    │任务A│    │任务B│    │任务C│   ← 就绪队列              │
  │    └──┬──┘    └──┬──┘    └──┬──┘                          │
  │       ↓          ↓          ↓                              │
  │    执行到 await → 暂停 → 执行下一个任务                     │
  │       ↓                                                    │
  │    IO 完成 → 恢复执行                                       │
  │                                                             │
  │    循环往复，直到所有任务完成                                 │
  └─────────────────────────────────────────────────────────────┘

  类比：餐厅只有 1 个服务员（单线程事件循环）
    但他不会站在一桌旁边等客人吃完
    而是：点单A → 点单B → 点单C → A菜好了上菜 → ...
```

### 2.2 协程（Coroutine）

```python
# 协程 = 可以暂停和恢复的函数

import asyncio

# 用 async def 声明协程函数
async def fetch_answer(question: str) -> str:
    # await 暂停当前协程，让出控制权给事件循环
    result = await chain.ainvoke({"question": question})
    return result

# 协程函数调用后得到的是"协程对象"（不会立即执行！）
coro = fetch_answer("什么是黑洞？")  # 此时没有执行任何代码

# 必须通过事件循环来运行协程
result = asyncio.run(coro)  # 这时才真正执行
```

### 2.3 任务（Task）

```python
# Task = 被调度执行的协程（注册到事件循环中）

async def main():
    # 创建 Task：协程被注册到事件循环，准备执行
    task1 = asyncio.create_task(fetch_answer("什么是黑洞？"))
    task2 = asyncio.create_task(fetch_answer("什么是量子？"))

    # 两个 task 并发执行
    result1 = await task1
    result2 = await task2

# 关系链：
#   async def → 协程函数
#   协程函数() → 协程对象
#   asyncio.create_task(协程对象) → Task
#   事件循环调度 Task 执行
```

---

## 3. async/await 语法详解

### 3.1 async def

```python
# 声明协程函数（可暂停的函数）

# 普通函数
def sync_func():
    return "hello"

# 协程函数
async def async_func():
    return "hello"

# 区别：
sync_func()   # 返回 "hello"（立即执行）
async_func()  # 返回 <coroutine object>（不执行！需要 await）
```

### 3.2 await

```python
# await 做了什么？
#   ① 暂停当前协程
#   ② 把控制权还给事件循环
#   ③ 等异步操作完成后恢复执行

async def example():
    # ① 发送请求给 LLM（非阻塞）
    # ② 暂停，事件循环去处理其他任务
    # ③ LLM 返回结果，恢复执行，result 拿到值
    result = await chain.ainvoke({"question": "test"})

    # 只能 await "可等待对象"：
    #   - 协程（async def 的返回值）
    #   - Task（asyncio.create_task 的返回值）
    #   - Future（底层 IO 操作的包装）
```

### 3.3 async for / async with

```python
# 异步迭代器（对应 async_demo.py 第 3 章的 .astream()）
async for chunk in chain.astream({"question": "什么是AI？"}):
    print(chunk, end="")

# 异步上下文管理器（对应第 5 章的 Semaphore）
async with semaphore:
    result = await chain.ainvoke(input)

# 等价展开：
#   async with semaphore:
#     → await semaphore.acquire()  # 获取令牌
#     → ... 执行代码 ...
#     → semaphore.release()        # 归还令牌
```

---

## 4. 并发 LLM 调用模式

### 4.1 asyncio.gather

```python
# 对应 async_demo.py 第 1 章：方式二

# 同时执行多个协程，等待全部完成
tasks = [
    chain.ainvoke({"question": q})
    for q in QUESTIONS
]
results = await asyncio.gather(*tasks)

# gather 的特点：
#   ① 所有任务同时开始
#   ② 返回顺序和输入顺序一致（不管谁先完成）
#   ③ 默认一个失败则全部取消
#   ④ return_exceptions=True 可容忍部分失败

# 对应代码中的性能对比：
#   同步：5个请求串行 → 总时间 = 5 × 单次时间
#   gather：5个请求并发 → 总时间 ≈ 单次时间（快 5 倍！）
```

### 4.2 asyncio.as_completed

```python
# gather 等全部完成才返回，as_completed 谁先完成先处理谁

async def process_questions(questions):
    tasks = {
        asyncio.create_task(chain.ainvoke({"question": q})): q
        for q in questions
    }

    for completed in asyncio.as_completed(tasks.keys()):
        result = await completed
        # 这里按完成顺序处理，而非输入顺序
        print(f"完成: {result[:30]}...")

# 适用场景：
#   gather：需要所有结果才能继续（如汇总报告）
#   as_completed：可以逐个处理结果（如实时展示进度）
```

### 4.3 .abatch() — LangChain 内置批量并发

```python
# 对应 async_demo.py 第 2 章

inputs = [{"question": q} for q in QUESTIONS]

# 不限并发
results = await chain.abatch(inputs)

# 限制并发数（防止打爆 API）
results = await chain.abatch(
    inputs,
    config={"max_concurrency": 2},  # 最多同时 2 个
)

# abatch vs 手动 gather：
#   abatch 更简洁，自带 max_concurrency 支持
#   手动 gather 更灵活，可以自定义超时/重试逻辑
```

---

## 5. 信号量控制并发数

### 5.1 Semaphore 原理

```
对应 async_demo.py 第 5 章：

Semaphore（信号量）= 令牌桶并发控制

  ┌─────────────────────────────────────────────────────────────┐
  │  桶里有 N 个令牌（如 N=2）                                   │
  │                                                             │
  │  任务1 想执行 → 取走令牌 ✅ → 开始执行                       │
  │  任务2 想执行 → 取走令牌 ✅ → 开始执行                       │
  │  任务3 想执行 → 桶空了 ❌ → 等待...                          │
  │  任务4 想执行 → 桶空了 ❌ → 等待...                          │
  │                                                             │
  │  任务1 完成 → 归还令牌 → 任务3 拿到令牌 → 开始执行           │
  │  任务2 完成 → 归还令牌 → 任务4 拿到令牌 → 开始执行           │
  └─────────────────────────────────────────────────────────────┘

  时间线（Semaphore(2)，4个任务）：
    t=0:  任务1,2 开始执行
    t=3:  任务1完成，任务3 开始
    t=4:  任务2完成，任务4 开始
    t=6:  任务3完成
    t=7:  任务4完成
    总时间 ≈ 7s（而非串行的 12s，也非无限并发的 3s）
```

### 5.2 代码实现

```python
# 对应 async_demo.py 第 5 章

semaphore = asyncio.Semaphore(2)  # 最多同时 2 个请求

async def rate_limited_invoke(question: str, index: int):
    """受信号量限制的调用"""
    async with semaphore:  # 获取令牌（桶空则等待）
        print(f"开始：{question}")
        result = await chain.ainvoke({"question": question})
        print(f"完成：{question}")
        return result
    # async with 退出时自动归还令牌

# 启动所有任务（但最多同时执行 2 个）
results = await asyncio.gather(*[
    rate_limited_invoke(q, i)
    for i, q in enumerate(questions, 1)
])
```

### 5.3 Semaphore vs max_concurrency

```
两种并发控制的对比：

  ┌─────────────────┬───────────────────┬───────────────────────┐
  │  方式            │  Semaphore        │  max_concurrency      │
  ├─────────────────┼───────────────────┼───────────────────────┤
  │  使用场景       │  任何异步代码      │  仅 .abatch() 中      │
  │  灵活性         │  高（可组合逻辑）  │  低（仅限流）         │
  │  作用范围       │  全局/自定义       │  单次 abatch 调用     │
  │  额外逻辑       │  可加超时/重试     │  不可自定义          │
  │  代码量         │  较多              │  一行参数            │
  └─────────────────┴───────────────────┴───────────────────────┘

  选择建议：
    简单批量调用 → max_concurrency
    需要自定义逻辑（超时、重试、日志） → Semaphore
```

---

## 6. 批量处理策略

### 6.1 固定大小批处理

```python
# 将大量任务分成固定大小的批次执行

async def batch_process(questions: list, batch_size: int = 5):
    all_results = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        # 每批并发执行
        results = await asyncio.gather(*[
            chain.ainvoke({"question": q}) for q in batch
        ])
        all_results.extend(results)
        # 批次间可加延迟（防限流）
        await asyncio.sleep(1.0)
    return all_results

# 100 个问题，每批 5 个：
# 批1(5并发) → 等1s → 批2(5并发) → 等1s → ...
# 总时间 ≈ 20批 × (单次耗时 + 1s间隔)
```

### 6.2 动态批处理

```python
# 根据 API 响应动态调整批大小

async def adaptive_batch(questions: list):
    batch_size = 10  # 初始批大小
    results = []

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        try:
            batch_results = await asyncio.gather(*[
                chain.ainvoke({"question": q}) for q in batch
            ])
            results.extend(batch_results)
            # 成功则尝试增大批次
            batch_size = min(batch_size + 2, 20)
        except Exception as e:
            if "rate_limit" in str(e):
                # 被限流则缩小批次并等待
                batch_size = max(batch_size // 2, 1)
                await asyncio.sleep(5.0)
                # 重试当前批次
                i -= batch_size

    return results
```

---

## 7. 速率限制实现

### 7.1 令牌桶算法（Token Bucket）

```
令牌桶 = 速率限制的经典算法

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  每秒往桶里放 r 个令牌（rate）                               │
  │  桶最多装 b 个令牌（burst）                                  │
  │  每个请求取走 1 个令牌                                       │
  │  桶空则等待（或拒绝）                                        │
  │                                                             │
  │     ┌───┐ ← 令牌以固定速率填充                              │
  │     │ r │                                                   │
  │     └─┬─┘                                                   │
  │       ↓                                                     │
  │   ┌───────┐                                                 │
  │   │●●●●●○○│ ← 桶（最大容量 b=7，当前 5 个令牌）             │
  │   └───┬───┘                                                 │
  │       ↓                                                     │
  │     请求取走令牌                                              │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

```python
import time

class TokenBucket:
    def __init__(self, rate: float, burst: int):
        self.rate = rate      # 每秒补充的令牌数
        self.burst = burst    # 桶的最大容量
        self.tokens = burst   # 当前令牌数
        self.last_time = time.monotonic()

    async def acquire(self):
        while True:
            now = time.monotonic()
            # 补充令牌
            elapsed = now - self.last_time
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_time = now

            if self.tokens >= 1:
                self.tokens -= 1
                return  # 获得令牌，继续执行
            else:
                # 等待直到有令牌
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)

# 使用：每秒最多 5 个请求，允许突发 10 个
bucket = TokenBucket(rate=5, burst=10)

async def rate_limited_call(question):
    await bucket.acquire()  # 等待令牌
    return await chain.ainvoke({"question": question})
```

### 7.2 滑动窗口算法（Sliding Window）

```python
from collections import deque

class SlidingWindowLimiter:
    """滑动窗口速率限制器"""
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        self.timestamps = deque()

    async def acquire(self):
        while True:
            now = time.monotonic()
            # 移除窗口外的旧记录
            while self.timestamps and now - self.timestamps[0] > self.window:
                self.timestamps.popleft()

            if len(self.timestamps) < self.max_requests:
                self.timestamps.append(now)
                return
            else:
                # 等待最早的请求过期
                wait = self.window - (now - self.timestamps[0])
                await asyncio.sleep(wait)

# 使用：60 秒内最多 20 个请求
limiter = SlidingWindowLimiter(max_requests=20, window_seconds=60)
```

---

## 8. 异步与同步代码的互操作

### 8.1 在同步代码中调用异步函数

```python
# 场景：你的主程序是同步的，但想用异步 LLM 调用

# 方式一：asyncio.run()（最简单，对应 async_demo.py 底部）
async def main():
    result = await chain.ainvoke({"question": "test"})
    return result

result = asyncio.run(main())  # 阻塞直到完成

# 方式二：在已有事件循环中运行（Jupyter Notebook 场景）
import nest_asyncio
nest_asyncio.apply()  # 允许嵌套事件循环
result = asyncio.run(main())

# 方式三：在新线程中运行事件循环
import concurrent.futures

def run_async_in_thread(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
```

### 8.2 在异步代码中调用同步函数

```python
# 场景：异步端点中需要调用阻塞的同步库

import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def async_endpoint():
    # 将同步阻塞调用放到线程池中执行（不阻塞事件循环）
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        sync_blocking_function,  # 同步函数
        arg1, arg2,             # 参数
    )
    return result

# 实际场景：
#   ① 同步数据库驱动（psycopg2）
#   ② CPU 密集计算（文本处理）
#   ③ 不支持异步的第三方 SDK
```

---

## 9. 多线程 vs 多进程 vs 异步 IO

### 9.1 三种并发模型对比

```
┌─────────────────────────────────────────────────────────────────┐
│  维度        │  多线程           │  多进程          │  异步IO     │
├─────────────────────────────────────────────────────────────────┤
│  并行方式    │  OS 线程切换      │  多CPU核心       │  事件循环    │
│  GIL 影响   │  受限(CPU任务)    │  不受限          │  不受限      │
│  内存开销    │  中(共享内存)     │  高(独立内存)    │  低(单线程)  │
│  适合任务    │  IO密集+简单      │  CPU密集         │  IO密集      │
│  切换成本    │  中(OS调度)       │  高(进程切换)    │  低(用户态)  │
│  编程复杂度  │  中(需加锁)       │  中(需IPC)       │  中(需async) │
│  LLM调用    │  可用但非最优     │  不推荐          │  最佳选择    │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 为什么 LLM 调用推荐异步 IO

```
LLM 调用 = 99% 时间在等 IO（网络）

  多线程方案：
    线程1: 发请求 → 等5秒 → 收响应
    线程2: 发请求 → 等5秒 → 收响应
    ...
    问题：100个并发 = 100个线程 = 大量内存 + 上下文切换开销

  异步IO方案：
    1个线程：发100个请求 → 等最慢那个5秒 → 全部收到
    问题：无！完美契合 IO 密集场景

  结论：
    LLM API 调用 → asyncio（最佳）
    CPU密集处理（如大批量文本预处理）→ multiprocessing
    混合场景 → asyncio + run_in_executor
```

---

## 10. 生产者-消费者模式

### 10.1 异步队列

```python
# 生产者-消费者模式：解耦请求接收和处理

import asyncio

async def producer(queue: asyncio.Queue, questions: list):
    """生产者：将问题放入队列"""
    for q in questions:
        await queue.put(q)
    # 放入结束信号
    await queue.put(None)

async def consumer(queue: asyncio.Queue, consumer_id: int):
    """消费者：从队列取问题并处理"""
    while True:
        question = await queue.get()
        if question is None:
            await queue.put(None)  # 传递结束信号给下一个消费者
            break
        result = await chain.ainvoke({"question": question})
        print(f"Consumer-{consumer_id}: {result[:30]}...")
        queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=10)  # 有界队列（背压控制）

    questions = ["Q1", "Q2", ..., "Q20"]

    # 1个生产者 + 3个消费者
    producer_task = asyncio.create_task(producer(queue, questions))
    consumer_tasks = [
        asyncio.create_task(consumer(queue, i))
        for i in range(3)
    ]

    await producer_task
    await asyncio.gather(*consumer_tasks)
```

### 10.2 应用场景

```
适用场景：
  ① 大批量文档处理（1000+ 文档需要 LLM 摘要）
  ② 实时流量处理（请求速率 > 处理速率）
  ③ 优先级队列（重要请求优先处理）

  有界队列的作用（背压 Backpressure）：
    queue = asyncio.Queue(maxsize=10)
    当队列满了，生产者自动等待 → 不会无限堆积内存
```

---

## 11. 异步错误处理与超时控制

### 11.1 超时控制

```python
# 对应 async_demo.py 第 4 章

# asyncio.wait_for — 给任何协程加超时限制
try:
    result = await asyncio.wait_for(
        chain.ainvoke({"question": "复杂问题..."}),
        timeout=30.0,  # 30 秒超时
    )
except asyncio.TimeoutError:
    print("请求超时！")
    # 超时后协程会被取消（CancelledError）

# 为什么必须设超时？
#   ① LLM 服务可能卡死（不返回）
#   ② 网络抖动导致连接挂起
#   ③ 一个卡住的请求可能拖垮整个服务
#   生产建议：所有 LLM 调用都设 30-60s 超时
```

### 11.2 部分失败处理

```python
# 对应 async_demo.py 第 4 章的 safe_invoke

async def safe_invoke(question: str, timeout_sec: float = 30.0):
    """带超时和异常捕获的安全包装"""
    try:
        result = await asyncio.wait_for(
            chain.ainvoke({"question": question}),
            timeout=timeout_sec,
        )
        return {"status": "success", "answer": result}
    except asyncio.TimeoutError:
        return {"status": "timeout", "answer": None}
    except Exception as e:
        return {"status": "error", "answer": str(e)}

# gather + return_exceptions=True：
results = await asyncio.gather(
    safe_invoke("Q1"),
    safe_invoke("Q2"),
    safe_invoke("Q3"),
    return_exceptions=True,  # 不让一个失败取消其他任务
)

# 设计原则：
#   ① 每个请求独立处理错误（互不影响）
#   ② 返回统一的状态结构（status + answer）
#   ③ 调用方可以根据 status 决定是否重试
```

### 11.3 重试策略

```python
# 指数退避重试（Exponential Backoff）

async def invoke_with_retry(question: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(
                chain.ainvoke({"question": question}),
                timeout=30.0,
            )
        except (asyncio.TimeoutError, Exception) as e:
            if attempt == max_retries - 1:
                raise  # 最后一次直接抛出
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            print(f"重试 {attempt+1}/{max_retries}，等待 {wait_time}s")
            await asyncio.sleep(wait_time)

# 重试策略选择：
#   超时错误 → 重试（可能是暂时的网络问题）
#   429 限流 → 重试（加长等待时间）
#   400 参数错误 → 不重试（重试也不会成功）
#   500 服务器错误 → 有限重试
```

---

## 12. aiohttp vs httpx 异步 HTTP 客户端

### 12.1 对比

```
┌─────────────────┬──────────────────────┬──────────────────────┐
│  维度            │  aiohttp             │  httpx               │
├─────────────────┼──────────────────────┼──────────────────────┤
│  API 风格       │  异步原生            │  同步+异步双模式     │
│  与 requests 兼容│ 完全不同            │  几乎相同            │
│  性能           │  略高                │  略低                │
│  HTTP/2 支持   │  不原生             │  原生支持            │
│  流式响应       │  支持                │  支持                │
│  LangChain 使用│  不直接使用          │  底层使用 httpx      │
│  生态成熟度     │  老牌，社区大        │  新兴，设计更现代    │
└─────────────────┴──────────────────────┴──────────────────────┘

LangChain 底层使用 httpx（所以 .ainvoke() 天然支持异步）。
如果你需要自己调 LLM API，推荐 httpx。
```

### 12.2 httpx 异步使用

```python
import httpx

async def call_llm_directly(question: str):
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": "Bearer sk-xxx"},
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": question}],
            },
        )
        return response.json()["choices"][0]["message"]["content"]
```

---

## 13. 性能对比与基准测试

### 13.1 实测数据

```
对应 async_demo.py 第 1 章的性能对比：

  测试条件：5 个 LLM 问题，单次平均耗时 3s

  同步串行 .invoke()：
    总耗时 ≈ 15s（5 × 3s）

  异步并发 asyncio.gather：
    总耗时 ≈ 3s（取最慢那个）
    加速比 ≈ 5x

  异步并发 + Semaphore(2)：
    总耗时 ≈ 9s（2个一批 × 3批 × 3s）
    加速比 ≈ 1.7x（安全但更慢）

  .abatch(max_concurrency=2)：
    总耗时 ≈ 9s（同 Semaphore(2)）

性能公式（理论）：
  T_sync = N × t_single
  T_async = t_single（无限并发）
  T_limited = ceil(N/C) × t_single（并发数=C）
```

### 13.2 基准测试方法

```python
import time
import statistics

async def benchmark(func, inputs, concurrency=None, runs=3):
    """基准测试工具"""
    times = []
    for _ in range(runs):
        start = time.time()
        if concurrency:
            sem = asyncio.Semaphore(concurrency)
            async def limited(inp):
                async with sem:
                    return await func(inp)
            await asyncio.gather(*[limited(i) for i in inputs])
        else:
            await asyncio.gather(*[func(i) for i in inputs])
        times.append(time.time() - start)

    return {
        "mean": statistics.mean(times),
        "p50": statistics.median(times),
        "min": min(times),
        "max": max(times),
    }
```

---

## 14. 实际场景：并发调用多模型/多提示

### 14.1 多模型并发（择优选择）

```python
# 同时调用多个 LLM，选最快/最好的结果

async def multi_model_race(question: str):
    """竞速模式：谁先返回用谁的"""
    models = [
        ChatOpenAI(model="gpt-4o-mini", ...),
        ChatOpenAI(model="qwen-turbo", ...),
        ChatOpenAI(model="deepseek-chat", ...),
    ]

    tasks = [m.ainvoke(question) for m in models]

    # as_completed 取第一个完成的
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            # 取消其他还在运行的任务
            for t in tasks:
                t.cancel()
            return result
        except Exception:
            continue  # 这个失败了，等下一个

    raise Exception("所有模型都失败了")
```

### 14.2 多提示并发（集成决策）

```python
# 同一问题用不同提示获取多角度回答

async def multi_perspective(question: str):
    """多角度分析：用不同 prompt 获取不同视角"""
    perspectives = [
        ("技术视角", "从技术实现角度分析："),
        ("商业视角", "从商业价值角度分析："),
        ("风险视角", "从潜在风险角度分析："),
    ]

    async def get_perspective(name, prefix):
        result = await chain.ainvoke({
            "question": f"{prefix}{question}"
        })
        return (name, result)

    results = await asyncio.gather(*[
        get_perspective(name, prefix)
        for name, prefix in perspectives
    ])

    # 汇总多角度分析
    summary = "\n".join(f"【{name}】{answer}" for name, answer in results)
    return summary
```

### 14.3 流水线并发

```python
# 多步骤流水线：每步依赖上一步结果，但不同文档可以流水线并发

async def pipeline(document: str):
    """处理流水线：摘要 → 提取关键词 → 生成标签"""
    summary = await summarize_chain.ainvoke({"text": document})
    keywords = await extract_chain.ainvoke({"text": summary})
    tags = await tag_chain.ainvoke({"keywords": keywords})
    return {"summary": summary, "keywords": keywords, "tags": tags}

# 10 个文档并发走流水线
documents = [doc1, doc2, ..., doc10]
results = await asyncio.gather(*[
    pipeline(doc) for doc in documents
])
# 每个文档内部是串行的（有依赖），但文档之间是并发的
```

---

## 附录 A：代码与知识点对应

| 代码位置 | 覆盖知识点 | 对应本文章节 |
|---------|------------|-------------|
| `async_demo.py` 第1章 | 同步vs异步性能对比 | 第1、13节 |
| `async_demo.py` 第2章 | .abatch() 批量并发 | 第4、6节 |
| `async_demo.py` 第3章 | .astream() 异步流式 | 第3节 |
| `async_demo.py` 第4章 | 超时控制+部分失败 | 第11节 |
| `async_demo.py` 第5章 | Semaphore 信号量 | 第5节 |

---

## 附录 B：速查表

```
┌────────────────────────────┬───────────────────────────────────┐
│  需求                       │  推荐方案                          │
├────────────────────────────┼───────────────────────────────────┤
│  单个异步调用               │  await chain.ainvoke(input)        │
│  多个并发调用               │  asyncio.gather(*tasks)            │
│  谁先完成先处理             │  asyncio.as_completed(tasks)       │
│  批量并发（简洁版）         │  await chain.abatch(inputs)        │
│  限制并发数（简洁版）       │  abatch(config={max_concurrency})  │
│  限制并发数（灵活版）       │  asyncio.Semaphore(n)              │
│  超时保护                   │  asyncio.wait_for(coro, timeout)   │
│  异步流式                   │  async for chunk in .astream()     │
│  部分失败容忍               │  gather(return_exceptions=True)    │
│  速率限制                   │  TokenBucket / SlidingWindow       │
│  同步调异步                 │  asyncio.run(coro)                 │
│  异步调同步                 │  loop.run_in_executor(...)         │
│  生产者-消费者              │  asyncio.Queue                     │
└────────────────────────────┴───────────────────────────────────┘
```

---

## 附录 C：推荐学习路径

```
入门阶段（3天）：
  运行 async_demo.py，观察同步 vs 异步的性能差异
  第1-3节 → 理解 asyncio 基本概念

进阶阶段（1周）：
  第4-7节 → 掌握并发控制和速率限制
  尝试修改 Semaphore 值，观察行为变化

高级阶段（2周）：
  第8-12节 → 理解互操作、生产者-消费者、HTTP客户端
  第13-14节 → 实战场景练习

生产应用：
  结合 api_service/ 项目，在 FastAPI 中使用异步调用
  为所有 LLM 调用加上超时 + 重试 + Semaphore
```

---

> **下一步学习**：运行 `async_demo.py` 观察实际性能对比，然后前往 `api_service/KNOWLEDGE.md` 了解如何在 FastAPI 中应用异步并发技术。
