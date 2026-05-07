"""
╔══════════════════════════════════════════════════════════════════╗
║         项目十一：异步与并发 — async/await 让 LLM 调用飞起来        ║
║         从"排队买奶茶"到"同时点 5 杯"的性能飞跃                    ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普一：同步 vs 异步——用奶茶店理解】
═══════════════════════════════════════════════════════════════════

  同步（Sync）= 排队买奶茶：
  ┌─────────────────────────────────────────────────────────────┐
  │  你：点第 1 杯 → 等 3 分钟 → 拿到                          │
  │  你：点第 2 杯 → 等 3 分钟 → 拿到                          │
  │  你：点第 3 杯 → 等 3 分钟 → 拿到                          │
  │  总耗时：9 分钟（3+3+3）                                    │
  └─────────────────────────────────────────────────────────────┘

  异步（Async）= 同时点 3 杯，等最慢那杯好就行：
  ┌─────────────────────────────────────────────────────────────┐
  │  你：同时点 3 杯 → 服务员并行制作 → 3 分钟后全部拿到        │
  │  总耗时：3 分钟（取最慢的那杯）                              │
  │  快了 3 倍！                                                 │
  └─────────────────────────────────────────────────────────────┘

  对应到 LLM 调用：
    同步 .invoke()：一个请求处理完才处理下一个
    异步 .ainvoke()：多个请求"同时"发出去，谁先回来就先处理谁

═══════════════════════════════════════════════════════════════════
【前置科普二：为什么 LLM 调用特别适合异步？】
═══════════════════════════════════════════════════════════════════

LLM 调用的耗时构成：
  ① 网络延迟（~50ms）
  ② LLM 推理生成（~2000-5000ms）← 绝大部分时间在这里！
  ③ 响应传输（~50ms）

在步骤②期间，你的 CPU 在干嘛？
  同步模式：CPU 在"干等"（sleeping），啥也没干！
  异步模式：CPU 去处理其他请求了，等 LLM 回来再继续！

  ┌─────────────────────────────────────────────────────────────┐
  │  比喻：同步 = 你站在微波炉前盯着食物转                       │
  │        异步 = 你把食物放进去，去干别的事，叮！再回来取        │
  └─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
【前置科普三：Python asyncio 核心概念速览】
═══════════════════════════════════════════════════════════════════

  ┌────────────────┬──────────────────────────────────────────┐
  │  概念           │  说明                                     │
  ├────────────────┼──────────────────────────────────────────┤
  │  async def     │  声明一个"协程函数"（可暂停的函数）       │
  │  await         │  暂停当前协程，等待异步操作完成            │
  │  asyncio.gather│  同时运行多个协程，等全部完成              │
  │  事件循环      │  调度器，决定哪个协程该运行了              │
  └────────────────┴──────────────────────────────────────────┘

  LangChain 的异步 API 命名规律：
    .invoke()           → .ainvoke()
    .stream()           → .astream()
    .batch()            → .abatch()
    .stream_events()    → .astream_events()
    （前面加 a = async）
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import asyncio
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "kivy-kimi-k2_5"

llm = ChatOpenAI(model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, temperature=0.7)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位简洁的科普作家，回答控制在50字以内。"),
        ("human", "{question}"),
    ]
)
parser = StrOutputParser()
chain = prompt | llm | parser

print("=" * 60)
print("项目十一：异步与并发 — async/await")
print("=" * 60)
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：同步 vs 异步——性能对比
# 目标：用实际计时证明异步的性能优势
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTIONS = [
    "什么是黑洞？",
    "什么是量子纠缠？",
    "什么是暗物质？",
    "什么是引力波？",
    "什么是中子星？",
]

print("=" * 60)
print("第 1 章：同步 vs 异步 性能对比")
print("=" * 60)
print()

# ── 方式一：同步串行（一个一个来）────────────────────────

print("【方式一：同步串行 .invoke()】")
print("  每个问题等前一个完成后才开始...")
print()

start = time.time()
sync_results = []
for i, q in enumerate(QUESTIONS, 1):
    t0 = time.time()
    result = chain.invoke({"question": q})
    elapsed = time.time() - t0
    sync_results.append(result)
    print(f"  [{i}] {q} → {elapsed:.1f}s → {result[:30]}...")

sync_total = time.time() - start
print()
print(f"  ⏱️  同步总耗时：{sync_total:.1f}s")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚠️ 重要：Python 3.9 的 asyncio.run() 在脚本中多次调用会
#    导致事件循环关闭后 httpx 连接池报错。
#    解决方案：把所有异步演示放在一个 async def main() 中，
#    只调用一次 asyncio.run(main())。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def main():
    """所有异步演示的统一入口"""

    # ── 方式二：异步并发（同时发出去）────────────────────────
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("【方式二：异步并发 asyncio.gather + .ainvoke()】")
    print("  所有问题同时发出，等最慢的那个完成...")
    print()

    start = time.time()

    # asyncio.gather 接收多个协程，同时运行它们
    # 类比：同时下 5 个外卖单，等最慢那个到就全齐了
    tasks = [chain.ainvoke({"question": q}) for q in QUESTIONS]
    async_results = await asyncio.gather(*tasks)

    async_total = time.time() - start

    for i, (q, result) in enumerate(zip(QUESTIONS, async_results), 1):
        print(f"  [{i}] {q} → {result[:30]}...")

    print()
    print(f"  ⏱️  异步总耗时：{async_total:.1f}s")
    print()

    speedup = sync_total / async_total if async_total > 0 else 0
    print(f"  📊 性能对比：")
    print(f"     同步：{sync_total:.1f}s")
    print(f"     异步：{async_total:.1f}s")
    print(f"     加速比：{speedup:.1f}x 🚀")
    print()
    print("  💡 为什么快了这么多？")
    print("     同步：请求1完成 → 请求2开始 → 请求2完成 → 请求3开始...")
    print("     异步：请求1/2/3/4/5 同时发出 → 等最慢的那个 → 全部完成")
    print()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 第 2 章：.abatch() — LangChain 内置的批量并发
    # 目标：不需要手动 gather，用 .abatch() 一行搞定
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("=" * 60)
    print("第 2 章：.abatch() — 内置批量并发")
    print("=" * 60)
    print()

    # ── .batch() vs .abatch() ─────────────────────────────────
    #
    # .batch(inputs)   → 同步批量（内部可能并行，取决于实现）
    # .abatch(inputs)  → 异步批量（明确并发执行）
    #
    # .abatch 的优势：
    #   ① 代码更简洁（不需要手动 asyncio.gather）
    #   ② 可以设置 max_concurrency 限制并发数（防止打爆 API）
    #   ③ 返回顺序和输入顺序一致

    print("【.abatch() 一行搞定并发】")
    print()

    inputs = [{"question": q} for q in QUESTIONS]

    # ── 不限制并发 ─────────────────────────────────────────
    start = time.time()
    results = await chain.abatch(inputs)
    elapsed = time.time() - start

    for i, (q, r) in enumerate(zip(QUESTIONS, results), 1):
        print(f"  [{i}] {q} → {r[:30]}...")
    print()
    print(f"  ⏱️  abatch（无并发限制）：{elapsed:.1f}s")
    print()

    # ── 限制并发数为 2 ─────────────────────────────────────
    #
    # 为什么要限制并发？
    #   ① API 有速率限制（RPM/TPM），并发太高会被 429 限流
    #   ② 避免同时发太多请求导致 LLM 服务过载
    #   ③ 在速度和稳定性之间取平衡
    #
    # max_concurrency=2 表示：最多同时执行 2 个请求

    start = time.time()
    results_limited = await chain.abatch(
        inputs,
        config={"max_concurrency": 2},
    )
    elapsed_limited = time.time() - start

    print(f"  ⏱️  abatch（max_concurrency=2）：{elapsed_limited:.1f}s")
    print()
    print("  💡 限制并发后更慢，但更安全（不会触发 API 限流）。")
    print("     生产中建议根据 API 的 RPM 限制设置合理的并发数。")
    print()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 第 3 章：异步流式 .astream() — 异步 + 打字机效果
    # 目标：在异步环境中实现逐 token 输出
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("=" * 60)
    print("第 3 章：异步流式 .astream()")
    print("=" * 60)
    print()

    # ── .stream() vs .astream() ───────────────────────────────
    #
    # .stream()  → 同步迭代器（for chunk in ...）
    # .astream() → 异步迭代器（async for chunk in ...）
    #
    # 在 FastAPI 等异步框架中，必须用 .astream()！
    # 因为同步 .stream() 会阻塞事件循环，导致其他请求无法处理。

    print("【.astream() — 异步流式逐 token】")
    print()
    print("  ❓ 问题：什么是量子计算机？")
    print("  🤖 回答：", end="", flush=True)

    token_count = 0
    async for chunk in chain.astream({"question": "什么是量子计算机？"}):
        print(chunk, end="", flush=True)
        token_count += 1

    print()
    print(f"  📊 共 {token_count} 个 chunk")
    print()
    print("  💡 .astream() 用法和 .stream() 完全对称：")
    print("     同步：for chunk in chain.stream(input)")
    print("     异步：async for chunk in chain.astream(input)")
    print()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 第 4 章：并发 + 超时控制 — 生产环境必备
    # 目标：设置请求超时、处理并发中的部分失败
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("=" * 60)
    print("第 4 章：超时控制 + 部分失败处理")
    print("=" * 60)
    print()

    # ── 为什么需要超时？─────────────────────────────────────
    #
    # LLM API 可能因为：
    #   ① 网络抖动 → 请求卡住不返回
    #   ② 服务过载 → 响应极慢（30秒+）
    #   ③ 生成死循环 → 一直在输出
    #
    # 没有超时控制 = 一个请求卡住就把整个服务拖垮！

    print("【asyncio.wait_for — 给异步调用加超时】")
    print()

    # ── 正常调用（应该成功）────────────────────────────────
    print("  ✅ 正常调用（超时30秒）：")
    try:
        result = await asyncio.wait_for(
            chain.ainvoke({"question": "1+1等于几？"}),
            timeout=30.0,
        )
        print(f"     结果：{result}")
    except asyncio.TimeoutError:
        print("     ❌ 超时！")
    print()

    # ── 演示超时处理（设置极短超时）─────────────────────────
    print("  ⏰ 极短超时（0.1秒，故意触发超时）：")
    try:
        result = await asyncio.wait_for(
            chain.ainvoke({"question": "请详细解释相对论的所有推导过程"}),
            timeout=0.1,
        )
        print(f"     结果：{result}")
    except asyncio.TimeoutError:
        print("     ❌ 超时！请求被取消（asyncio.TimeoutError）")
        print("     💡 生产中应设置合理超时（如 30-60 秒）并返回友好错误")
    print()

    # ── gather 中的部分失败处理 ────────────────────────────────

    print("【asyncio.gather(return_exceptions=True) — 容忍部分失败】")
    print()
    print("  场景：3 个并发请求，用 safe_invoke 包装保证互不干扰")
    print()

    async def safe_invoke(question: str, timeout_sec: float = 30.0):
        """带超时的安全调用包装"""
        try:
            result = await asyncio.wait_for(
                chain.ainvoke({"question": question}),
                timeout=timeout_sec,
            )
            return {"status": "success", "question": question, "answer": result}
        except asyncio.TimeoutError:
            return {"status": "timeout", "question": question, "answer": None}
        except Exception as e:
            return {"status": "error", "question": question, "answer": str(e)}

    questions = [
        "什么是DNA？",
        "什么是RNA？",
        "什么是蛋白质？",
    ]

    start = time.time()
    results = await asyncio.gather(
        *[safe_invoke(q, timeout_sec=30.0) for q in questions]
    )
    elapsed = time.time() - start

    success_count = 0
    for r in results:
        status_icon = "✅" if r["status"] == "success" else "❌"
        answer_display = (
            r["answer"][:30] + "..."
            if r["answer"] and len(r["answer"]) > 30
            else r["answer"]
        )
        print(f"  {status_icon} [{r['status']:7s}] {r['question']} → {answer_display}")
        if r["status"] == "success":
            success_count += 1

    print()
    print(f"  ⏱️  耗时：{elapsed:.1f}s | 成功：{success_count}/{len(results)}")
    print()
    print("  💡 safe_invoke 包装保证：即使某个请求失败，其他请求不受影响。")
    print()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 第 5 章：Semaphore — 精细并发控制
    # 目标：用信号量限制全局并发数，保护 API 限流
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("=" * 60)
    print("第 5 章：Semaphore — 信号量精细并发控制")
    print("=" * 60)
    print()

    # ── Semaphore 是什么？──────────────────────────────────────
    #
    # Semaphore（信号量）= "令牌桶"：
    #   桶里有 N 个令牌。
    #   要执行任务，先从桶里取一个令牌。
    #   任务完成后，把令牌还回去。
    #   如果桶空了，新任务就等着（直到有人还令牌）。
    #
    # 比 max_concurrency 更灵活：
    #   max_concurrency 只能在 .abatch() 中用。
    #   Semaphore 可以在任何异步代码中用，包括自定义逻辑。
    #
    # 场景：API 限制 5 QPS，你有 20 个请求 → Semaphore(5)

    print("【asyncio.Semaphore — 令牌桶并发控制】")
    print()

    # 创建信号量：最多同时 2 个请求
    semaphore = asyncio.Semaphore(2)

    async def rate_limited_invoke(question: str, index: int):
        """受信号量限制的调用"""
        async with semaphore:
            t0 = time.time()
            print(f"  🟢 [{index}] 开始：{question}")
            result = await chain.ainvoke({"question": question})
            elapsed = time.time() - t0
            print(f"  ✅ [{index}] 完成：{elapsed:.1f}s")
            return result

    sem_questions = [
        "太阳有多热？",
        "月球有多远？",
        "火星有水吗？",
        "木星有多大？",
    ]

    print(f"  📋 4 个任务，Semaphore(2)=最多同时执行 2 个")
    print()

    start = time.time()
    await asyncio.gather(
        *[rate_limited_invoke(q, i) for i, q in enumerate(sem_questions, 1)]
    )
    total = time.time() - start

    print()
    print(f"  ⏱️  总耗时：{total:.1f}s")
    print("  💡 观察：[1][2] 同时开始，[3][4] 等前面完成后才开始。")
    print("     这就是 Semaphore 的'令牌桶'效果。")
    print()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 总结
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    print("=" * 60)
    print("🎉 项目十一学习完毕！")
    print("=" * 60)
    print()
    print("💡 异步并发速查表：")
    print()
    print("  ┌────────────────────────┬───────────────────────────────────┐")
    print("  │  需求                   │  推荐方案                          │")
    print("  ├────────────────────────┼───────────────────────────────────┤")
    print("  │  单个异步调用           │  await chain.ainvoke(input)        │")
    print("  │  多个并发调用           │  asyncio.gather(*tasks)            │")
    print("  │  批量并发（简洁版）     │  await chain.abatch(inputs)        │")
    print("  │  限制并发数             │  abatch(config={max_concurrency})  │")
    print("  │  精细并发控制           │  asyncio.Semaphore(n)              │")
    print("  │  超时保护               │  asyncio.wait_for(coro, timeout)   │")
    print("  │  异步流式               │  async for chunk in .astream()     │")
    print("  │  部分失败容忍           │  gather(return_exceptions=True)    │")
    print("  └────────────────────────┴───────────────────────────────────┘")
    print()
    print("💡 生产要点：")
    print("   ① LLM 调用是 I/O 密集型，异步能显著提升吞吐量")
    print("   ② 一定要设置超时，否则一个卡住的请求会拖垮服务")
    print("   ③ 用 Semaphore 或 max_concurrency 保护 API 限流")
    print("   ④ 用 return_exceptions=True 保证并发任务互不干扰")
    print("   ⑤ FastAPI 中天然是异步环境，直接用 ainvoke/astream")
    print("=" * 60)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 入口：单次 asyncio.run() 调用所有异步演示
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

asyncio.run(main())
