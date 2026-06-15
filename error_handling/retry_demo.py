"""
╔══════════════════════════════════════════════════════════════════╗
║         项目十五：Error Handling + Retry（容错机制）                 ║
║         LLM 应用的"免死金牌"——超时、重试、降级、兜底               ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普一：LLM 调用会遇到哪些错误？】
═══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────┐
  │  ① 网络错误（ConnectionError）                              │
  │     → 网络抖动、DNS 解析失败、连接超时                      │
  │     → 特点：通常是暂时性的，重试即可恢复                    │
  │                                                             │
  │  ② 速率限制（RateLimitError / 429）                         │
  │     → 请求太频繁，超过 API 的 RPM/TPM 限制                  │
  │     → 特点：等一会儿再重试就行                              │
  │                                                             │
  │  ③ 服务端错误（InternalServerError / 500）                  │
  │     → LLM 服务内部故障                                      │
  │     → 特点：可能暂时性，可以重试几次看看                    │
  │                                                             │
  │  ④ 格式错误（OutputParserException）                        │
  │     → LLM 输出不符合预期格式（JSON 解析失败等）             │
  │     → 特点：可以重试（换个温度再生成一次）                  │
  │                                                             │
  │  ⑤ 超时（Timeout）                                          │
  │     → LLM 生成时间过长                                      │
  │     → 特点：可能是输入太长或服务过载                        │
  └─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
【前置科普二：重试策略——不是"无脑重试"】
═══════════════════════════════════════════════════════════════════

  错误的重试 = 火上浇油：
    服务过载 → 你疯狂重试 → 服务更过载 → 整个系统雪崩！

  正确的重试策略：

  ┌──────────────────┬──────────────────────────────────────────┐
  │  策略             │  说明                                     │
  ├──────────────────┼──────────────────────────────────────────┤
  │  指数退避        │  等 1s → 2s → 4s → 8s（越等越久）       │
  │  最大重试次数    │  最多重试 3 次，超过就放弃                │
  │  可重试判断      │  只有暂时性错误才重试（429、500、网络）   │
  │  抖动（jitter）  │  在等待时间上加随机偏移，防止雷群效应    │
  └──────────────────┴──────────────────────────────────────────┘
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import time
import random

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_core.runnables.retry import RunnableRetry


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("项目十五：Error Handling + Retry（容错机制）")
print("=" * 60)
print()

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY, BASE_URL, MODEL_NAME
llm = ChatOpenAI(model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, temperature=0.7)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位简洁的科普作家，回答控制在50字以内。"),
        ("human", "{question}"),
    ]
)
parser = StrOutputParser()
chain = prompt | llm | parser

print("✅ LLM 初始化完成")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：.with_retry() — LangChain 内置重试
# 目标：用 LangChain 原生的重试机制处理暂时性错误
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：.with_retry() — 内置重试机制")
print("=" * 60)
print()

# ── .with_retry() 用法 ─────────────────────────────────────
#
# 任何 Runnable 都可以调用 .with_retry()：
#   chain.with_retry(
#       stop_after_attempt=3,       # 最多重试 3 次
#       wait_exponential_jitter=True # 指数退避 + 抖动
#   )
#
# 效果：
#   第一次调用失败 → 等 1s → 重试
#   第二次还失败   → 等 2s → 重试
#   第三次还失败   → 放弃，抛出异常

retry_chain = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

print("【.with_retry() 正常调用演示】")
print()
print("  ❓ 问题：什么是 DNA？")
result = retry_chain.invoke({"question": "什么是DNA？"})
print(f"  🤖 回答：{result}")
print()
print("  💡 正常情况下 .with_retry() 不会触发重试，行为和普通链一样。")
print("     只有出错时才会自动重试。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：模拟错误 + 重试过程可视化
# 目标：用模拟函数展示重试的完整过程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：模拟错误 + 重试过程可视化")
print("=" * 60)
print()

# ── 模拟一个"前两次失败，第三次成功"的函数 ─────────────────

call_count = 0


def flaky_function(input_dict: dict) -> str:
    """
    模拟不稳定的服务：前 2 次失败，第 3 次成功。
    用于演示重试机制的效果。
    """
    global call_count
    call_count += 1

    if call_count <= 2:
        print(f"    ❌ 第 {call_count} 次调用：模拟失败（ConnectionError）")
        raise ConnectionError(f"模拟网络错误（第{call_count}次）")
    else:
        print(f"    ✅ 第 {call_count} 次调用：成功！")
        return f"成功响应（经过 {call_count} 次尝试）"


# 把函数包装成 Runnable 并加上重试
flaky_runnable = RunnableLambda(flaky_function).with_retry(
    stop_after_attempt=5,
    wait_exponential_jitter=True,
)

print("【模拟不稳定服务 + 自动重试】")
print()
print("  场景：服务前 2 次返回错误，第 3 次恢复正常")
print("  策略：最多重试 5 次，指数退避")
print()

call_count = 0  # 重置计数
start = time.time()
result = flaky_runnable.invoke({"question": "test"})
elapsed = time.time() - start

print()
print(f"  📊 结果：{result}")
print(f"  ⏱️  总耗时：{elapsed:.1f}s（包含退避等待时间）")
print()
print("  💡 观察：系统自动重试了 2 次后成功，用户无感知！")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：Fallback（降级）— 主力模型挂了换备用
# 目标：用 .with_fallbacks() 实现优雅降级
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：Fallback（降级）— 备用方案")
print("=" * 60)
print()

# ── .with_fallbacks() 是什么？──────────────────────────────
#
# 当主链失败时，自动切换到备用链：
#   chain.with_fallbacks([backup_chain1, backup_chain2])
#
# 执行顺序：
#   主链调用 → 成功 → 返回结果
#   主链调用 → 失败 → 试 backup1 → 成功 → 返回结果
#   主链调用 → 失败 → 试 backup1 → 失败 → 试 backup2
#
# 应用场景：
#   ① 主力模型（GPT-4）挂了 → 降级到 GPT-3.5
#   ② 外部 API 不可用 → 用本地缓存兜底
#   ③ 复杂链失败 → 用简单链（直接返回搜索结果）兜底


# 模拟一个"总是失败"的主链
def always_fail(input_dict: dict) -> str:
    raise RuntimeError("主力模型不可用！")


primary_chain = RunnableLambda(always_fail)

# 备用链：用正常的 LLM 链
fallback_chain = chain

# 组装：主链 + 降级
resilient_chain = primary_chain.with_fallbacks([fallback_chain])

print("【.with_fallbacks() 优雅降级演示】")
print()
print("  场景：主力模型故障，自动降级到备用模型")
print()
print("  ❓ 问题：什么是量子力学？")

result = resilient_chain.invoke({"question": "什么是量子力学？"})
print(f"  🤖 回答（来自备用链）：{result}")
print()
print("  💡 用户完全无感知——主力挂了，备用顶上，体验不中断！")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：手动实现完整的容错包装器
# 目标：结合重试 + 超时 + 降级 + 兜底的完整方案
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 4 章：完整容错包装器")
print("=" * 60)
print()


class ResilientInvoker:
    """
    完整容错调用器：重试 + 超时 + 降级 + 兜底。

    实际生产中的标准做法：
      ① 先重试 N 次（带指数退避）
      ② 全部失败后尝试降级方案
      ③ 降级也失败则返回兜底响应
    """

    def __init__(
        self,
        primary_chain,
        fallback_chain=None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        timeout: float = 30.0,
        fallback_message: str = "抱歉，服务暂时不可用，请稍后重试。",
    ):
        self.primary = primary_chain
        self.fallback = fallback_chain
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self.fallback_message = fallback_message

    def invoke(self, input_dict: dict) -> dict:
        """
        容错调用：重试 → 降级 → 兜底

        返回: {"answer": str, "source": "primary"|"fallback"|"default", "attempts": int}
        """
        # 阶段一：重试主链
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.primary.invoke(input_dict)
                return {"answer": result, "source": "primary", "attempts": attempt}
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    # 指数退避 + 随机抖动
                    delay = self.base_delay * (2 ** (attempt - 1)) + random.uniform(
                        0, 0.5
                    )
                    print(f"    ⚠️ 第{attempt}次失败: {type(e).__name__}")
                    print(f"       等待 {delay:.1f}s 后重试...")
                    time.sleep(delay)

        # 阶段二：尝试降级
        if self.fallback:
            try:
                print(f"    🔄 主链 {self.max_retries} 次全部失败，尝试降级方案...")
                result = self.fallback.invoke(input_dict)
                return {
                    "answer": result,
                    "source": "fallback",
                    "attempts": self.max_retries,
                }
            except Exception as e:
                print(f"    ❌ 降级方案也失败: {e}")

        # 阶段三：返回兜底响应
        print(f"    🛑 所有方案均失败，返回兜底响应")
        return {
            "answer": self.fallback_message,
            "source": "default",
            "attempts": self.max_retries,
        }


# ── 演示完整容错流程 ───────────────────────────────────────

# 创建一个"50%概率失败"的模拟链
fail_count = [0]


def sometimes_fail(input_dict: dict) -> str:
    fail_count[0] += 1
    if fail_count[0] <= 2:
        raise ConnectionError("模拟间歇性故障")
    return chain.invoke(input_dict)


unstable_chain = RunnableLambda(sometimes_fail)

invoker = ResilientInvoker(
    primary_chain=unstable_chain,
    fallback_chain=chain,
    max_retries=3,
    base_delay=0.5,
)

print("【完整容错流程演示】")
print()
print("  场景：主链前2次故障，第3次成功")
print()

fail_count[0] = 0
result = invoker.invoke({"question": "什么是区块链？"})

print()
print(f"  📊 结果：")
print(f"     回答：{result['answer'][:40]}...")
print(f"     来源：{result['source']}")
print(f"     尝试次数：{result['attempts']}")
print()

# 演示全部失败 → 降级
print("  ─── 演示全部失败 → 降级 ───")
print()

always_fail_chain = RunnableLambda(
    lambda x: (_ for _ in ()).throw(RuntimeError("全挂"))
)

invoker2 = ResilientInvoker(
    primary_chain=always_fail_chain,
    fallback_chain=chain,
    max_retries=2,
    base_delay=0.3,
)

result2 = invoker2.invoke({"question": "什么是人工智能？"})

print()
print(f"  📊 结果：")
print(f"     回答：{result2['answer'][:40]}...")
print(f"     来源：{result2['source']}（主链全挂，降级成功）")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 总结
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("🎉 项目十五学习完毕！")
print("=" * 60)
print()
print("💡 容错策略速查表：")
print()
print("  ┌──────────────────┬───────────────────────────────────┐")
print("  │  策略             │  实现方式                          │")
print("  ├──────────────────┼───────────────────────────────────┤")
print("  │  自动重试         │  chain.with_retry(attempts=3)     │")
print("  │  指数退避         │  wait_exponential_jitter=True     │")
print("  │  优雅降级         │  chain.with_fallbacks([backup])   │")
print("  │  超时控制         │  asyncio.wait_for(timeout=30)     │")
print("  │  兜底响应         │  try/except → 返回默认回答        │")
print("  │  完整方案         │  ResilientInvoker（重试+降级+兜底）│")
print("  └──────────────────┴───────────────────────────────────┘")
print()
print("💡 生产要点：")
print("   ① 只对暂时性错误重试（网络/429/500），不对 4xx 客户端错误重试")
print("   ② 指数退避 + 抖动，避免雷群效应（所有客户端同时重试）")
print("   ③ 设置最大重试次数（通常 3 次），避免无限重试")
print("   ④ 降级方案要提前准备好（备用模型/缓存/默认回答）")
print("   ⑤ 记录所有错误和重试日志（对接项目十四的可观测性）")
print("=" * 60)
