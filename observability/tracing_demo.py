"""
╔══════════════════════════════════════════════════════════════════╗
║         项目十四：Observability（可观测性）                          ║
║         从 print 调试到全链路追踪——看清 AI 应用的每一步             ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普一：为什么 print() 调试在生产中不够？】
═══════════════════════════════════════════════════════════════════

  开发阶段：print("这里到了") print(f"结果是{x}")
  → 能看到输出，够用了。

  生产阶段：
  ┌─────────────────────────────────────────────────────────────┐
  │  问题一：用户反馈"回答不对"，但你不知道哪一步出了问题       │
  │    • 是 Prompt 拼错了？                                      │
  │    • 是检索到了错误文档？                                    │
  │    • 是 LLM 幻觉了？                                        │
  │    → print 已经淹没在日志海洋中，找不到了！                  │
  │                                                             │
  │  问题二：服务突然变慢，不知道瓶颈在哪                        │
  │    • 是 Embedding 计算慢？                                   │
  │    • 是向量搜索慢？                                          │
  │    • 是 LLM 生成慢？                                         │
  │    → 没有计时数据，无法定位！                                │
  │                                                             │
  │  问题三：Token 费用超预算，不知道哪个链消耗最多              │
  │    → 没有 Token 统计，无法优化！                             │
  └─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
【前置科普二：可观测性三大支柱】
═══════════════════════════════════════════════════════════════════

  ┌──────────────┬────────────────────────────────────────────┐
  │  支柱         │  说明                                       │
  ├──────────────┼────────────────────────────────────────────┤
  │  Logging     │  结构化日志：记录每步的输入输出和时间       │
  │  Tracing     │  链路追踪：看到完整调用链的层级关系         │
  │  Metrics     │  指标监控：Token消耗、延迟P99、成功率       │
  └──────────────┴────────────────────────────────────────────┘

  工具生态：
    • LangSmith：LangChain 官方，最深度集成
    • LangFuse：开源替代，可私有化部署
    • 本项目：用 LangChain Callbacks 手动实现（理解底层原理）
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import time
import json
from datetime import datetime
from typing import Any
from uuid import UUID

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("项目十四：Observability（可观测性）")
print("=" * 60)
print()

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "kivy-kimi-k2_5"

llm = ChatOpenAI(model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, temperature=0.7)
print("✅ LLM 初始化完成")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：自定义 Callback Handler（可观测性的基石）
# 目标：用 Callback 捕获 LLM 调用的全生命周期事件
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：自定义 Callback Handler")
print("=" * 60)
print()

# ── Callback 机制是什么？──────────────────────────────────
#
# LangChain 在执行链/LLM/工具时，会在关键时刻"发出事件"：
#   on_llm_start     → LLM 开始生成
#   on_llm_end       → LLM 生成完毕
#   on_llm_error     → LLM 报错
#   on_chain_start   → 链开始执行
#   on_chain_end     → 链执行完毕
#   on_tool_start    → 工具开始调用
#   on_tool_end      → 工具调用完毕
#
# 你实现一个 CallbackHandler，就能"监听"这些事件！
# 类比：Callback = AI应用的"监控摄像头"


class ObservabilityHandler(BaseCallbackHandler):
    """
    自定义可观测性 Handler。

    记录每次 LLM 调用的：
      - 开始/结束时间（计算延迟）
      - 输入 Prompt
      - 输出内容
      - Token 消耗
      - 错误信息（如果有）
    """

    def __init__(self):
        self.traces = []  # 存储所有追踪记录
        self._start_times = {}  # 临时存储开始时间

    def on_llm_start(self, serialized: dict, prompts: list, *, run_id: UUID, **kwargs):
        """LLM 开始生成时触发"""
        self._start_times[run_id] = time.time()
        trace = {
            "event": "llm_start",
            "run_id": str(run_id),
            "timestamp": datetime.now().isoformat(),
            "model": serialized.get("kwargs", {}).get("model_name", "unknown"),
            "prompt_preview": prompts[0][:100] if prompts else "",
        }
        self.traces.append(trace)
        print(f"    📡 [on_llm_start] LLM 开始生成 (run_id: {str(run_id)[:8]}...)")

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs):
        """LLM 生成完毕时触发"""
        elapsed = time.time() - self._start_times.get(run_id, time.time())

        # 提取 Token 使用量
        token_usage = {}
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]

        # 提取输出文本
        output_text = ""
        if response.generations and response.generations[0]:
            output_text = response.generations[0][0].text

        trace = {
            "event": "llm_end",
            "run_id": str(run_id),
            "timestamp": datetime.now().isoformat(),
            "latency_ms": round(elapsed * 1000),
            "token_usage": token_usage,
            "output_preview": output_text[:80],
        }
        self.traces.append(trace)

        prompt_tokens = token_usage.get("prompt_tokens", "?")
        completion_tokens = token_usage.get("completion_tokens", "?")
        total_tokens = token_usage.get("total_tokens", "?")

        print(f"    📡 [on_llm_end] 生成完毕")
        print(f"       延迟: {elapsed*1000:.0f}ms | Token: {prompt_tokens}+{completion_tokens}={total_tokens}")

    def on_llm_error(self, error: Exception, *, run_id: UUID, **kwargs):
        """LLM 出错时触发"""
        trace = {
            "event": "llm_error",
            "run_id": str(run_id),
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
        }
        self.traces.append(trace)
        print(f"    🚨 [on_llm_error] 错误: {str(error)[:50]}")

    def on_chain_start(self, serialized: dict, inputs: dict, *, run_id: UUID, **kwargs):
        """链开始执行时触发"""
        self._start_times[run_id] = time.time()
        chain_name = (serialized or {}).get("name", (serialized or {}).get("id", ["unknown"])[-1])
        print(f"    📡 [on_chain_start] 链开始: {chain_name}")

    def on_chain_end(self, outputs: dict, *, run_id: UUID, **kwargs):
        """链执行完毕时触发"""
        elapsed = time.time() - self._start_times.get(run_id, time.time())
        print(f"    📡 [on_chain_end] 链完毕: {elapsed*1000:.0f}ms")

    def get_summary(self) -> dict:
        """生成可观测性摘要"""
        llm_calls = [t for t in self.traces if t["event"] == "llm_end"]
        total_latency = sum(t.get("latency_ms", 0) for t in llm_calls)
        total_tokens = sum(
            t.get("token_usage", {}).get("total_tokens", 0) for t in llm_calls
        )
        return {
            "total_llm_calls": len(llm_calls),
            "total_latency_ms": total_latency,
            "avg_latency_ms": total_latency // len(llm_calls) if llm_calls else 0,
            "total_tokens": total_tokens,
            "errors": len([t for t in self.traces if t["event"] == "llm_error"]),
        }


# ── 使用 Callback 追踪 LLM 调用 ──────────────────────────

handler = ObservabilityHandler()

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位简洁的科普作家，回答控制在50字以内。"),
    ("human", "{question}"),
])
chain = prompt | llm | StrOutputParser()

print("【带 Callback 追踪的 LLM 调用】")
print()

# config={"callbacks": [...]} 传入自定义 handler
questions = ["什么是黑洞？", "什么是量子计算？", "什么是深度学习？"]

for q in questions:
    print(f"  ❓ {q}")
    result = chain.invoke(
        {"question": q},
        config={"callbacks": [handler]},
    )
    print(f"  🤖 {result[:40]}...")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：可观测性报告——从追踪数据中提取洞察
# 目标：汇总统计信息，形成"可观测性报告"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：可观测性报告")
print("=" * 60)
print()

summary = handler.get_summary()

print("╔══════════════════════════════════════════════════════════╗")
print("║                  可观测性报告                            ║")
print("╠══════════════════════════════════════════════════════════╣")
print(f"║  LLM 调用次数    ：{summary['total_llm_calls']}")
print(f"║  总延迟          ：{summary['total_latency_ms']}ms")
print(f"║  平均延迟        ：{summary['avg_latency_ms']}ms")
print(f"║  总 Token 消耗   ：{summary['total_tokens']}")
print(f"║  错误次数        ：{summary['errors']}")
print("╠══════════════════════════════════════════════════════════╣")
print("║  详细追踪记录（最近 3 条 llm_end）                       ║")
print("╠══════════════════════════════════════════════════════════╣")

llm_end_traces = [t for t in handler.traces if t["event"] == "llm_end"]
for t in llm_end_traces[-3:]:
    latency = t.get("latency_ms", 0)
    tokens = t.get("token_usage", {}).get("total_tokens", "?")
    output = t.get("output_preview", "")[:35]
    print(f"║  {latency:>5}ms | {tokens:>4} tokens | {output}...")

print("╚══════════════════════════════════════════════════════════╝")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：结构化日志（JSON 格式，方便 ELK/Datadog 采集）
# 目标：输出标准 JSON 日志，可被日志系统采集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：结构化 JSON 日志")
print("=" * 60)
print()

# ── 为什么要结构化日志？──────────────────────────────────
#
# 非结构化日志：
#   "2024-01-15 LLM调用成功，耗时1234ms，消耗token 500"
#   → 难以被程序解析！只能人眼看。
#
# 结构化日志（JSON）：
#   {"time": "2024-01-15", "event": "llm_end", "latency_ms": 1234, "tokens": 500}
#   → ELK/Datadog/Grafana 可以自动解析、聚合、可视化！


class JSONLogHandler(BaseCallbackHandler):
    """输出标准 JSON 格式日志的 Handler（可对接 ELK/Datadog）"""

    def __init__(self):
        self._start_times = {}

    def on_llm_start(self, serialized: dict, prompts: list, *, run_id: UUID, **kwargs):
        self._start_times[run_id] = time.time()
        log = {
            "level": "INFO",
            "event": "llm_start",
            "run_id": str(run_id)[:8],
            "timestamp": datetime.now().isoformat(),
        }
        print(f"  LOG: {json.dumps(log, ensure_ascii=False)}")

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs):
        elapsed = time.time() - self._start_times.get(run_id, time.time())
        token_usage = {}
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]

        log = {
            "level": "INFO",
            "event": "llm_end",
            "run_id": str(run_id)[:8],
            "timestamp": datetime.now().isoformat(),
            "latency_ms": round(elapsed * 1000),
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
        }
        print(f"  LOG: {json.dumps(log, ensure_ascii=False)}")

    def on_llm_error(self, error: Exception, *, run_id: UUID, **kwargs):
        log = {
            "level": "ERROR",
            "event": "llm_error",
            "run_id": str(run_id)[:8],
            "timestamp": datetime.now().isoformat(),
            "error": str(error)[:100],
        }
        print(f"  LOG: {json.dumps(log, ensure_ascii=False)}")


json_handler = JSONLogHandler()

print("【JSON 格式日志输出（可对接 ELK/Datadog/Grafana）】")
print()

result = chain.invoke(
    {"question": "什么是机器学习？"},
    config={"callbacks": [json_handler]},
)
print()
print(f"  回答：{result[:40]}...")
print()
print("  💡 这些 JSON 日志可以被：")
print("     • ELK Stack 采集 → 全文搜索 + 聚合分析")
print("     • Datadog / Grafana → 实时监控仪表盘")
print("     • 自定义告警系统 → 延迟过高/错误率飙升时发告警")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：Token 成本监控
# 目标：追踪 Token 消耗，计算费用
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 4 章：Token 成本监控")
print("=" * 60)
print()


class CostTracker(BaseCallbackHandler):
    """Token 成本追踪器"""

    # 假设的价格（每千 token）
    PRICE_PER_1K_INPUT = 0.01  # 输入 ¥0.01/千token
    PRICE_PER_1K_OUTPUT = 0.03  # 输出 ¥0.03/千token

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def on_llm_end(self, response: LLMResult, **kwargs):
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.total_input_tokens += usage.get("prompt_tokens", 0)
            self.total_output_tokens += usage.get("completion_tokens", 0)
            self.call_count += 1

    @property
    def total_cost(self) -> float:
        input_cost = self.total_input_tokens / 1000 * self.PRICE_PER_1K_INPUT
        output_cost = self.total_output_tokens / 1000 * self.PRICE_PER_1K_OUTPUT
        return input_cost + output_cost

    def report(self):
        print("  ┌──────────────────────────────────────────┐")
        print(f"  │  调用次数    ：{self.call_count}")
        print(f"  │  输入 Token  ：{self.total_input_tokens:,}")
        print(f"  │  输出 Token  ：{self.total_output_tokens:,}")
        print(f"  │  总 Token    ：{self.total_input_tokens + self.total_output_tokens:,}")
        print(f"  │  预估费用    ：¥{self.total_cost:.4f}")
        print("  └──────────────────────────────────────────┘")


cost_tracker = CostTracker()

# 执行多个调用来积累数据
print("  ⏳ 执行 3 次 LLM 调用，追踪 Token 消耗...")
print()

for q in ["DNA的结构", "RNA的功能", "蛋白质折叠"]:
    chain.invoke({"question": q}, config={"callbacks": [cost_tracker]})

print("【Token 成本报告】")
cost_tracker.report()
print()
print("  💡 生产中接入此 Handler，可以：")
print("     ① 按用户/按链路 统计 Token 消耗")
print("     ② 设置 Token 预算上限，超限告警")
print("     ③ 月度账单可视化")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 总结
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("🎉 项目十四学习完毕！")
print("=" * 60)
print()
print("💡 可观测性实现方案：")
print()
print("  ┌──────────────────┬───────────────────────────────────┐")
print("  │  方案             │  适用场景                          │")
print("  ├──────────────────┼───────────────────────────────────┤")
print("  │  自定义 Callback │  理解底层原理 + 轻量监控           │")
print("  │  LangSmith       │  LangChain 官方，深度集成          │")
print("  │  LangFuse        │  开源，可私有部署，成本低          │")
print("  │  OpenTelemetry   │  通用标准，对接 Datadog/Jaeger    │")
print("  └──────────────────┴───────────────────────────────────┘")
print()
print("💡 关键 Callback 事件：")
print("   on_llm_start / on_llm_end     → 追踪 LLM 调用")
print("   on_chain_start / on_chain_end  → 追踪链路执行")
print("   on_tool_start / on_tool_end    → 追踪工具调用")
print("   on_llm_error                   → 追踪错误")
print("=" * 60)
