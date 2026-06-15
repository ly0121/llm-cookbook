"""
╔══════════════════════════════════════════════════════════════════╗
║         项目：LLMOps 可观测性（Observability）全面实验            ║
║         探索调用追踪、性能监控、成本计算、语义缓存等核心能力       ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题：为什么 LLM 应用需要可观测性？】
═══════════════════════════════════════════════════════════════════

当 LLM 从实验走向生产环境，你会面临一系列"看不见"的问题：

  用户反馈"好慢" → 到底慢在哪里？网络？模型推理？后处理？
  月底账单超预算 → 哪些调用花了最多钱？能不能缓存？
  输出质量下降   → 是 prompt 变了？还是模型版本变了？

  ┌─────────────────────────────────────────────────────────────┐
  │  可观测性三大支柱（The Three Pillars）                        │
  │                                                             │
  │  1. Tracing（追踪）                                          │
  │     记录每次调用的完整链路：输入 → 处理 → 输出               │
  │     回答：一次请求经历了哪些步骤？每步花了多久？             │
  │                                                             │
  │  2. Metrics（指标）                                          │
  │     量化系统表现：延迟、吞吐量、token 用量、错误率            │
  │     回答：系统整体表现如何？有没有退化趋势？                 │
  │                                                             │
  │  3. Cost（成本）                                             │
  │     追踪每次调用的花费，按模型/功能/用户维度汇总             │
  │     回答：钱花在了哪里？怎么优化？                           │
  │                                                             │
  │  + 语义缓存（Semantic Cache）                                │
  │     用相似度匹配避免重复调用，降低延迟和成本                  │
  └─────────────────────────────────────────────────────────────┘

本文件通过真实 API 调用，完整演示 LLMOps 可观测性的核心实践。
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：LLMOps 可观测性总览
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import time

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import client, MODEL_NAME

print("=" * 60)
print("第 0 章：LLMOps 可观测性总览")
print("=" * 60)
print()
print("""
┌──────────────────────────────────────────────────────────────┐
│           LLM 应用可观测性架构（Observability Stack）          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  用户请求                                                     │
│    ↓                                                         │
│  [Tracer 开始记录] ─────────────────────────────────────┐    │
│    ↓                                                    │    │
│  Prompt 组装（记录输入 + 时间戳）                         │    │
│    ↓                                                    │    │
│  语义缓存检查 ─── 命中 → 直接返回（记录缓存命中）        │    │
│    ↓ 未命中                                             │    │
│  API 调用（记录 TTFT、总延迟、token 数）                  │    │
│    ↓                                                    │    │
│  后处理（记录处理时间）                                   │    │
│    ↓                                                    │    │
│  [Tracer 结束] ← 计算成本、汇总指标 ───────────────────┘    │
│    ↓                                                         │
│  返回响应 + 写入监控系统                                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘

为什么可观测性如此重要？
  - 没有度量就没有优化：你无法改善你无法衡量的东西
  - 生产环境问题定位：从"用户说慢"到"定位到第3步耗时异常"
  - 成本控制：发现不必要的重复调用，引入缓存节省开支
  - 质量监控：追踪输出质量随时间的变化趋势
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：调用链路追踪（Tracing）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 链路追踪的核心思想：
#   把一次完整的 LLM 调用拆解为多个"步骤"（Span），
#   记录每个步骤的开始时间、结束时间、输入输出。
#
#   ┌────────────────────────────────────────────────────────┐
#   │  Trace（一次完整调用）                                   │
#   │  ├─ Span 1: prompt_assembly (组装 prompt)              │
#   │  │   └─ 输入: 用户消息  输出: 完整 messages             │
#   │  ├─ Span 2: llm_call (调用 LLM API)                   │
#   │  │   └─ 输入: messages  输出: response                 │
#   │  └─ Span 3: post_process (后处理)                      │
#   │       └─ 输入: response  输出: 最终结果                 │
#   └────────────────────────────────────────────────────────┘
#
#   好处：
#   1. 出问题时快速定位是哪个环节慢/出错
#   2. 记录输入输出便于复现和调试
#   3. 统计各环节的平均耗时，找到瓶颈

print("=" * 60)
print("第 1 章：调用链路追踪（Tracing）")
print("=" * 60)
print()


class Span:
    """表示追踪链路中的一个步骤"""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration_ms = None
        self.input_data = None
        self.output_data = None
        self.metadata = {}

    def start(self, input_data=None):
        """开始记录这个步骤"""
        self.start_time = time.time()
        self.input_data = input_data

    def end(self, output_data=None):
        """结束记录这个步骤"""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.output_data = output_data

    def __repr__(self):
        return f"Span({self.name}, {self.duration_ms:.1f}ms)"


class Tracer:
    """
    调用链路追踪器。
    记录一次完整 LLM 调用中每个步骤的时间和数据。
    """

    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.spans = []
        self.start_time = time.time()

    def create_span(self, name: str) -> Span:
        """创建一个新的步骤"""
        span = Span(name)
        self.spans.append(span)
        return span

    def get_total_duration_ms(self) -> float:
        """获取整个追踪的总耗时"""
        return (time.time() - self.start_time) * 1000

    def print_report(self):
        """打印追踪报告"""
        print(f"  ┌─ Trace ID: {self.trace_id}")
        print(f"  │  总耗时: {self.get_total_duration_ms():.1f}ms")
        print(f"  │")
        for i, span in enumerate(self.spans):
            connector = "└" if i == len(self.spans) - 1 else "├"
            print(f"  │  {connector}─ [{span.name}] {span.duration_ms:.1f}ms")
            if span.input_data:
                # 截取前50字符显示
                input_str = str(span.input_data)[:50]
                print(f"  │     输入: {input_str}...")
            if span.output_data:
                output_str = str(span.output_data)[:50]
                print(f"  │     输出: {output_str}...")
        print(f"  └─ 追踪完毕")


# ── 演示：追踪一次完整的 LLM 调用 ──────────────────────────
print("── 演示：追踪一次完整的 LLM 调用过程 ──────────────────")
print()

tracer = Tracer(trace_id="trace_001")

# 步骤1：组装 prompt
span1 = tracer.create_span("prompt_assembly")
span1.start(input_data="用户问题: 什么是机器学习？")

system_msg = "你是一位AI教授，回答简洁明了，控制在50字以内。"
user_msg = "什么是机器学习？"
messages = [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": user_msg},
]
span1.end(output_data=f"组装完成, {len(messages)} 条消息")

# 步骤2：调用 LLM API
span2 = tracer.create_span("llm_api_call")
span2.start(input_data=f"模型={MODEL_NAME}, 消息数={len(messages)}")

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    temperature=0.7,
    max_tokens=100,
)
result_text = response.choices[0].message.content.strip()
span2.end(output_data=result_text)
span2.metadata["token_usage"] = {
    "prompt_tokens": response.usage.prompt_tokens,
    "completion_tokens": response.usage.completion_tokens,
    "total_tokens": response.usage.total_tokens,
}

# 步骤3：后处理
span3 = tracer.create_span("post_process")
span3.start(input_data=result_text)

# 模拟后处理：去除多余空白、计算字数
final_result = result_text.strip()
char_count = len(final_result)
span3.end(output_data=f"处理完成, {char_count} 字")

# 打印追踪报告
tracer.print_report()
print()
print(f"  最终回答: {final_result}")
print()
print("  观察要点：")
print("   - llm_api_call 通常是最耗时的步骤（网络 + 模型推理）")
print("   - prompt_assembly 和 post_process 通常很快（毫秒级）")
print("   - 通过追踪可以精确定位性能瓶颈")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：性能指标监控（Metrics）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# LLM 性能的核心指标：
#
#   ┌────────────────────────────────────────────────────────┐
#   │  指标名称           │ 含义                              │
#   ├────────────────────────────────────────────────────────┤
#   │  TTFT               │ Time To First Token               │
#   │                     │ 首 token 延迟，用户感知的响应速度  │
#   │  总延迟             │ 从发送请求到收到完整响应的时间     │
#   │  Tokens/s           │ 每秒生成的 token 数（吞吐量）     │
#   │  Token 使用量       │ prompt + completion 的 token 总数  │
#   └────────────────────────────────────────────────────────┘
#
#   TTFT 为什么重要？
#     用户体验 = 感知延迟，而非总延迟
#     TTFT 低 → 用户很快看到第一个字 → 感觉"很快"
#     即使总生成时间长，低 TTFT 也能给用户好的体验
#
#   测量 TTFT 的方法：使用 streaming 模式
#     stream=True → 模型每生成一个 token 就立刻返回
#     第一个 chunk 到达的时间 = TTFT

print("=" * 60)
print("第 2 章：性能指标监控（Metrics）")
print("=" * 60)
print()

# ── 2.1 使用 Streaming 测量 TTFT ──────────────────────────
print("── 2.1 使用 Streaming 测量 TTFT（首 Token 延迟）────────")
print()


def measure_streaming_metrics(prompt: str, system_prompt: str = "你是一位助手。") -> dict:
    """
    使用 streaming 模式测量 LLM 调用的性能指标。

    返回：
        dict 包含 ttft_ms, total_latency_ms, completion_tokens, tokens_per_second
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    start_time = time.time()
    first_token_time = None
    token_count = 0
    full_response = ""

    # 使用 streaming 模式调用
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=200,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            token_count += 1
            full_response += chunk.choices[0].delta.content

    end_time = time.time()

    # 计算指标
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    total_latency_ms = (end_time - start_time) * 1000
    generation_time_s = (end_time - first_token_time) if first_token_time else 0
    tokens_per_second = token_count / generation_time_s if generation_time_s > 0 else 0

    return {
        "ttft_ms": ttft_ms,
        "total_latency_ms": total_latency_ms,
        "completion_tokens": token_count,
        "tokens_per_second": tokens_per_second,
        "response": full_response,
    }


# 执行多次调用，收集指标
test_prompts = [
    "用一句话解释什么是深度学习。",
    "Python 和 Java 的主要区别是什么？简要回答。",
    "为什么天空是蓝色的？用一句话解释。",
]

all_metrics = []
for i, prompt in enumerate(test_prompts, 1):
    print(f"  调用 {i}: {prompt}")
    metrics = measure_streaming_metrics(prompt)
    all_metrics.append(metrics)
    print(f"    TTFT（首 token 延迟）: {metrics['ttft_ms']:.0f}ms")
    print(f"    总延迟: {metrics['total_latency_ms']:.0f}ms")
    print(f"    生成 token 数: {metrics['completion_tokens']}")
    print(f"    生成速度: {metrics['tokens_per_second']:.1f} tokens/s")
    print(f"    回答: {metrics['response'][:60]}...")
    print()

# ── 2.2 汇总统计 ──────────────────────────────────────────
print("── 2.2 性能指标汇总统计 ────────────────────────────────")
print()

avg_ttft = sum(m["ttft_ms"] for m in all_metrics) / len(all_metrics)
avg_latency = sum(m["total_latency_ms"] for m in all_metrics) / len(all_metrics)
avg_tokens_per_s = sum(m["tokens_per_second"] for m in all_metrics) / len(all_metrics)
total_tokens = sum(m["completion_tokens"] for m in all_metrics)

print(f"  ┌─────────────────────────────────────────────────┐")
print(f"  │  性能指标汇总（{len(all_metrics)} 次调用）               │")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  平均 TTFT:          {avg_ttft:>8.0f} ms              │")
print(f"  │  平均总延迟:         {avg_latency:>8.0f} ms              │")
print(f"  │  平均生成速度:       {avg_tokens_per_s:>8.1f} tokens/s       │")
print(f"  │  总 token 消耗:      {total_tokens:>8d} tokens          │")
print(f"  └─────────────────────────────────────────────────┘")
print()
print("  观察要点：")
print("   - TTFT 反映用户感知的响应速度，越低越好")
print("   - tokens/s 反映模型的生成吞吐量")
print("   - 生产环境中应设置 TTFT 告警阈值（如 >2000ms 告警）")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：成本计算器（Cost Tracker）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# LLM API 的计费模式：
#   按 token 数量计费，输入和输出分开计价
#
#   ┌────────────────────────────────────────────────────────┐
#   │  成本公式：                                             │
#   │                                                        │
#   │  单次调用成本 = prompt_tokens × 输入单价                │
#   │              + completion_tokens × 输出单价             │
#   │                                                        │
#   │  通常输出单价 > 输入单价（生成比理解更贵）              │
#   │                                                        │
#   │  优化策略：                                             │
#   │  1. 精简 prompt（减少输入 token）                       │
#   │  2. 限制 max_tokens（控制输出长度）                     │
#   │  3. 语义缓存（避免重复调用）                            │
#   │  4. 选择合适模型（简单任务用便宜模型）                  │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 3 章：成本计算器（Cost Tracker）")
print("=" * 60)
print()


class CostTracker:
    """
    LLM 调用成本追踪器。
    按模型价格计算每次调用的成本，支持按日/月汇总。
    """

    # 模型价格表（每 1000 tokens 的价格，单位：元）
    PRICING = {
        "kivy-kimi-k2_5": {
            "input_per_1k": 0.002,    # 输入：0.002 元 / 1K tokens
            "output_per_1k": 0.006,   # 输出：0.006 元 / 1K tokens
        },
        "gpt-4o": {
            "input_per_1k": 0.01,
            "output_per_1k": 0.03,
        },
        "gpt-4o-mini": {
            "input_per_1k": 0.00015,
            "output_per_1k": 0.0006,
        },
    }

    def __init__(self):
        self.records = []

    def record_call(self, model: str, prompt_tokens: int, completion_tokens: int,
                    timestamp: float = None):
        """记录一次 API 调用的 token 使用"""
        if timestamp is None:
            timestamp = time.time()

        pricing = self.PRICING.get(model, self.PRICING["kivy-kimi-k2_5"])
        input_cost = (prompt_tokens / 1000) * pricing["input_per_1k"]
        output_cost = (completion_tokens / 1000) * pricing["output_per_1k"]
        total_cost = input_cost + output_cost

        record = {
            "timestamp": timestamp,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }
        self.records.append(record)
        return record

    def get_total_cost(self) -> float:
        """获取所有记录的总成本"""
        return sum(r["total_cost"] for r in self.records)

    def get_total_tokens(self) -> dict:
        """获取总 token 使用量"""
        return {
            "prompt_tokens": sum(r["prompt_tokens"] for r in self.records),
            "completion_tokens": sum(r["completion_tokens"] for r in self.records),
            "total_tokens": sum(r["total_tokens"] for r in self.records),
        }

    def get_daily_summary(self) -> dict:
        """按日汇总成本"""
        from datetime import datetime
        daily = {}
        for r in self.records:
            day = datetime.fromtimestamp(r["timestamp"]).strftime("%Y-%m-%d")
            if day not in daily:
                daily[day] = {"cost": 0, "calls": 0, "tokens": 0}
            daily[day]["cost"] += r["total_cost"]
            daily[day]["calls"] += 1
            daily[day]["tokens"] += r["total_tokens"]
        return daily

    def get_monthly_projection(self) -> float:
        """根据当前使用量预估月度成本"""
        if not self.records:
            return 0
        # 计算每次调用的平均成本，假设每天调用量相同
        avg_cost_per_call = self.get_total_cost() / len(self.records)
        # 假设每天调用次数与当前记录相同，乘以30天
        calls_per_day = len(self.records)  # 简化假设
        return avg_cost_per_call * calls_per_day * 30

    def print_report(self):
        """打印成本报告"""
        tokens = self.get_total_tokens()
        print(f"  ┌─────────────────────────────────────────────────┐")
        print(f"  │  成本报告（共 {len(self.records)} 次调用）                    │")
        print(f"  ├─────────────────────────────────────────────────┤")
        print(f"  │  输入 tokens:     {tokens['prompt_tokens']:>8d}                  │")
        print(f"  │  输出 tokens:     {tokens['completion_tokens']:>8d}                  │")
        print(f"  │  总 tokens:       {tokens['total_tokens']:>8d}                  │")
        print(f"  │  ─────────────────────────────────────────────  │")
        print(f"  │  总成本:          {self.get_total_cost():>10.6f} 元           │")
        print(f"  │  月度预估:        {self.get_monthly_projection():>10.4f} 元           │")
        print(f"  └─────────────────────────────────────────────────┘")


# ── 演示：追踪多次调用的成本 ──────────────────────────────
print("── 演示：追踪多次调用的成本 ────────────────────────────")
print()

cost_tracker = CostTracker()

# 模拟几次真实调用并追踪成本
cost_test_prompts = [
    ("你好，介绍一下你自己。", "你是一位友好的助手。"),
    ("用Python写一个快速排序。", "你是一位编程专家，只输出代码。"),
    ("总结一下量子计算的核心原理。", "你是一位物理学教授，回答简洁。"),
    ("翻译成英文：今天天气很好。", "你是一位翻译。"),
    ("给我讲一个关于猫的笑话。", "你是一位幽默大师。"),
]

for prompt, sys_prompt in cost_test_prompts:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=150,
    )

    # 记录成本
    record = cost_tracker.record_call(
        model=MODEL_NAME,
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
    )

    print(f"  调用: {prompt[:20]}...")
    print(f"    tokens: {response.usage.prompt_tokens}(输入) + {response.usage.completion_tokens}(输出)")
    print(f"    成本: {record['total_cost']:.6f} 元")
    print()

# 打印成本汇总报告
cost_tracker.print_report()
print()

# ── 对比不同模型的成本 ────────────────────────────────────
print("── 对比不同模型的成本（模拟计算）──────────────────────")
print()

# 假设相同的 token 用量，对比不同模型的成本差异
sample_prompt_tokens = 500
sample_completion_tokens = 200

print(f"  假设场景：输入 {sample_prompt_tokens} tokens，输出 {sample_completion_tokens} tokens")
print()
for model_name, pricing in CostTracker.PRICING.items():
    input_cost = (sample_prompt_tokens / 1000) * pricing["input_per_1k"]
    output_cost = (sample_completion_tokens / 1000) * pricing["output_per_1k"]
    total = input_cost + output_cost
    monthly = total * 1000  # 假设每月1000次调用
    print(f"  {model_name:20s}: 单次 {total:.6f} 元 | 月估(1000次) {monthly:.4f} 元")
print()
print("  观察要点：")
print("   - 不同模型价格差异可达 100 倍")
print("   - 简单任务用便宜模型，复杂任务用强模型，可大幅降低成本")
print("   - 输出 token 通常比输入 token 贵 2-3 倍")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：语义缓存（Semantic Cache）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 语义缓存的核心思想：
#   如果用户问了一个与之前"意思相近"的问题，
#   直接返回缓存的答案，无需再次调用 LLM。
#
#   ┌────────────────────────────────────────────────────────┐
#   │  传统缓存 vs 语义缓存：                                 │
#   │                                                        │
#   │  传统缓存：                                             │
#   │    "什么是AI？"  ≠  "AI是什么？"  ← 字符串不同，未命中  │
#   │                                                        │
#   │  语义缓存：                                             │
#   │    "什么是AI？"  ≈  "AI是什么？"  ← 语义相似，命中！    │
#   │                                                        │
#   │  实现方式：                                             │
#   │    1. 将问题转为 embedding 向量                         │
#   │    2. 计算新问题与缓存中所有问题的余弦相似度            │
#   │    3. 如果相似度超过阈值 → 缓存命中，直接返回           │
#   │    4. 如果未命中 → 调用 LLM，并将结果存入缓存           │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 4 章：语义缓存（Semantic Cache）")
print("=" * 60)
print()


class SemanticCache:
    """
    基于 embedding 相似度的语义缓存。
    使用余弦相似度判断新问题是否与缓存中的问题语义相近。
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        初始化语义缓存。

        参数：
            similarity_threshold: 相似度阈值，超过此值视为缓存命中
        """
        self.cache = []  # 存储 {"embedding": [...], "question": str, "answer": str}
        self.similarity_threshold = similarity_threshold
        self.hit_count = 0
        self.miss_count = 0

    def _get_embedding(self, text: str) -> list:
        """获取文本的 embedding 向量"""
        response = client.embeddings.create(
            model="kivy-text-embedding-3-large",
            input=text,
        )
        return response.data[0].embedding

    def _cosine_similarity(self, vec_a: list, vec_b: list) -> float:
        """计算两个向量的余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = sum(a * a for a in vec_a) ** 0.5
        magnitude_b = sum(b * b for b in vec_b) ** 0.5
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        return dot_product / (magnitude_a * magnitude_b)

    def lookup(self, question: str) -> tuple:
        """
        查询缓存。

        返回：
            (命中, 答案, 相似度) - 命中时返回缓存答案，未命中返回 None
        """
        if not self.cache:
            return False, None, 0.0

        query_embedding = self._get_embedding(question)

        best_similarity = 0.0
        best_answer = None

        for entry in self.cache:
            similarity = self._cosine_similarity(query_embedding, entry["embedding"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = entry["answer"]

        if best_similarity >= self.similarity_threshold:
            self.hit_count += 1
            return True, best_answer, best_similarity
        else:
            self.miss_count += 1
            return False, None, best_similarity

    def store(self, question: str, answer: str):
        """将问答对存入缓存"""
        embedding = self._get_embedding(question)
        self.cache.append({
            "embedding": embedding,
            "question": question,
            "answer": answer,
        })

    def get_stats(self) -> dict:
        """获取缓存命中统计"""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache),
        }


# ── 演示：语义缓存的效果 ──────────────────────────────────
print("── 演示：语义缓存的效果（命中 vs 未命中延迟对比）────────")
print()

cache = SemanticCache(similarity_threshold=0.85)

# 定义测试问题：第一组是原始问题，第二组是语义相似的问题
original_questions = [
    "什么是人工智能？",
    "Python 的优点有哪些？",
    "怎么学习编程？",
]

similar_questions = [
    "人工智能是什么意思？",     # 与第1题语义相似
    "Python 有什么好处？",      # 与第2题语义相似
    "编程应该怎么入门学习？",   # 与第3题语义相似
]

# 第一轮：原始问题（全部未命中，需要调用 LLM）
print("  【第一轮】原始问题（缓存为空，全部未命中）")
print()

for q in original_questions:
    start = time.time()

    # 查询缓存
    hit, cached_answer, similarity = cache.lookup(q)

    if hit:
        answer = cached_answer
        source = "缓存命中"
    else:
        # 未命中，调用 LLM
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一位助手，回答简洁，控制在30字以内。"},
                {"role": "user", "content": q},
            ],
            temperature=0.7,
            max_tokens=60,
        )
        answer = response.choices[0].message.content.strip()
        # 存入缓存
        cache.store(q, answer)
        source = "LLM 调用"

    elapsed = (time.time() - start) * 1000
    print(f"  问: {q}")
    print(f"    答: {answer}")
    print(f"    来源: {source} | 耗时: {elapsed:.0f}ms")
    print()

# 第二轮：语义相似的问题（应命中缓存）
print("  【第二轮】语义相似问题（应命中缓存）")
print()

for q in similar_questions:
    start = time.time()

    # 查询缓存
    hit, cached_answer, similarity = cache.lookup(q)

    if hit:
        answer = cached_answer
        source = f"缓存命中 (相似度={similarity:.3f})"
    else:
        # 未命中，调用 LLM
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一位助手，回答简洁，控制在30字以内。"},
                {"role": "user", "content": q},
            ],
            temperature=0.7,
            max_tokens=60,
        )
        answer = response.choices[0].message.content.strip()
        cache.store(q, answer)
        source = f"LLM 调用 (最高相似度={similarity:.3f})"

    elapsed = (time.time() - start) * 1000
    print(f"  问: {q}")
    print(f"    答: {answer}")
    print(f"    来源: {source} | 耗时: {elapsed:.0f}ms")
    print()

# 打印缓存统计
stats = cache.get_stats()
print(f"  ┌─────────────────────────────────────────────────┐")
print(f"  │  语义缓存统计                                    │")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  缓存大小:       {stats['cache_size']:>4d} 条                       │")
print(f"  │  命中次数:       {stats['hit_count']:>4d} 次                       │")
print(f"  │  未命中次数:     {stats['miss_count']:>4d} 次                       │")
print(f"  │  命中率:         {stats['hit_rate']*100:>6.1f}%                     │")
print(f"  └─────────────────────────────────────────────────┘")
print()
print("  观察要点：")
print("   - 缓存命中时，省去了 LLM API 调用，延迟大幅降低")
print("   - 语义缓存比精确匹配更智能，能识别意思相近的问题")
print("   - 阈值设置很关键：太高→命中率低，太低→可能返回不相关答案")
print("   - 生产环境中缓存可节省 30-70% 的 API 调用成本")
print()


# ── 总结 ──────────────────────────────────────────────────
print("=" * 60)
print("全部实验完毕！核心总结：")
print("=" * 60)
print("""
  ┌────────────────────────────────────────────────────────────┐
  │  能力             │ 解决的问题          │ 关键实现           │
  ├────────────────────────────────────────────────────────────┤
  │  调用链路追踪     │ 定位性能瓶颈        │ Span + Tracer     │
  │  性能指标监控     │ 量化系统表现        │ TTFT + Streaming  │
  │  成本计算器       │ 控制预算开支        │ Token 计费模型    │
  │  语义缓存         │ 减少重复调用        │ Embedding 相似度  │
  └────────────────────────────────────────────────────────────┘

  最佳实践：
  1. 从第一天就加入可观测性，不要等出问题再补
  2. 设置关键指标告警：TTFT > 阈值、成本超预算、错误率上升
  3. 语义缓存是最有效的成本优化手段之一
  4. 追踪数据可用于持续优化 prompt 和模型选择
""")
