---
title: 可观测性与LLMOps
---

<script setup>
const code1 = `# LLM 调用链路追踪器（模拟 RAG Pipeline）
import time
import random

class Span:
    """追踪单元：记录一个步骤的执行信息"""
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.start_time = None
        self.end_time = None
        self.metadata = {}
        self.status = "pending"

    def start(self):
        self.start_time = time.time()
        self.status = "running"
        return self

    def end(self, status="success"):
        self.end_time = time.time()
        self.status = status
        return self

    @property
    def duration_ms(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

class Tracer:
    """简易链路追踪器"""
    def __init__(self, trace_name):
        self.trace_name = trace_name
        self.trace_id = f"trace_{random.randint(10000, 99999)}"
        self.spans = []

    def create_span(self, name, parent=None):
        span = Span(name, parent)
        self.spans.append(span)
        return span

    def summary(self):
        print(f"{'='*60}")
        print(f"Trace: {self.trace_name} (ID: {self.trace_id})")
        print(f"{'='*60}")
        total = sum(s.duration_ms for s in self.spans)
        print(f"{'步骤':<20} {'耗时(ms)':<12} {'状态':<10} {'占比':<8}")
        print(f"{'-'*60}")
        for span in self.spans:
            pct = (span.duration_ms / total * 100) if total > 0 else 0
            status_icon = "OK" if span.status == "success" else "FAIL"
            print(f"{span.name:<20} {span.duration_ms:<12.1f} {status_icon:<10} {pct:.1f}%")
            if span.metadata:
                for k, v in span.metadata.items():
                    print(f"  -> {k}: {v}")
        print(f"{'-'*60}")
        print(f"{'总计':<20} {total:<12.1f}")
        print()

        # 可视化时间线
        print("时间线可视化:")
        max_bar = 40
        for span in self.spans:
            pct = (span.duration_ms / total) if total > 0 else 0
            bar_len = int(pct * max_bar)
            bar = "█" * bar_len + "░" * (max_bar - bar_len)
            print(f"  {span.name:<16} |{bar}| {span.duration_ms:.0f}ms")


# ========== 模拟 RAG Pipeline ==========
tracer = Tracer("RAG-Query-Pipeline")

# Step 1: 查询解析
span1 = tracer.create_span("query_parsing").start()
time.sleep(random.uniform(0.01, 0.03))  # 模拟处理
span1.metadata = {"query": "什么是Transformer?", "intent": "knowledge_qa"}
span1.end()

# Step 2: 向量检索
span2 = tracer.create_span("embedding_encode").start()
time.sleep(random.uniform(0.02, 0.05))
span2.metadata = {"model": "text-embedding-3-small", "dimensions": 1536}
span2.end()

# Step 3: 向量数据库查询
span3 = tracer.create_span("vector_search").start()
time.sleep(random.uniform(0.05, 0.12))
span3.metadata = {"top_k": 5, "score_threshold": 0.75, "results_found": 5}
span3.end()

# Step 4: Rerank
span4 = tracer.create_span("rerank").start()
time.sleep(random.uniform(0.03, 0.08))
span4.metadata = {"model": "bge-reranker", "kept": 3}
span4.end()

# Step 5: Prompt 组装
span5 = tracer.create_span("prompt_assembly").start()
time.sleep(random.uniform(0.005, 0.01))
span5.metadata = {"template": "rag_qa_v2", "context_tokens": 1200}
span5.end()

# Step 6: LLM 推理
span6 = tracer.create_span("llm_inference").start()
time.sleep(random.uniform(0.15, 0.35))
span6.metadata = {"model": "gpt-4o", "input_tokens": 1450, "output_tokens": 280, "temperature": 0.7}
span6.end()

# Step 7: 后处理
span7 = tracer.create_span("post_processing").start()
time.sleep(random.uniform(0.01, 0.02))
span7.metadata = {"citations_added": 3, "format": "markdown"}
span7.end()

# 输出追踪结果
tracer.summary()

# 性能分析建议
print()
print("=== 性能优化建议 ===")
llm_pct = span6.duration_ms / sum(s.duration_ms for s in tracer.spans) * 100
if llm_pct > 50:
    print(f"⚠ LLM 推理占总耗时 {llm_pct:.0f}%，建议:")
    print("  - 考虑使用更快的模型 (如 gpt-4o-mini)")
    print("  - 启用流式输出降低 TTFT")
    print("  - 实施语义缓存减少重复调用")
if span3.duration_ms > 100:
    print(f"⚠ 向量检索耗时 {span3.duration_ms:.0f}ms，建议优化索引或使用 ANN 加速")
`

const code2 = `# Token 成本计算器与优化策略分析

class ModelPricing:
    """模型定价信息（美元 / 1M tokens）"""
    MODELS = {
        "gpt-4o": {"input": 2.50, "output": 10.00, "speed": 100},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "speed": 150},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00, "speed": 60},
        "claude-3.5-sonnet": {"input": 3.00, "output": 15.00, "speed": 90},
        "claude-3-haiku": {"input": 0.25, "output": 1.25, "speed": 200},
        "deepseek-v3": {"input": 0.27, "output": 1.10, "speed": 120},
        "qwen-plus": {"input": 0.80, "output": 2.00, "speed": 110},
    }

class CostCalculator:
    """Token 成本计算器"""

    def __init__(self):
        self.pricing = ModelPricing.MODELS

    def calculate_cost(self, model, input_tokens, output_tokens):
        """计算单次调用成本"""
        p = self.pricing[model]
        input_cost = (input_tokens / 1_000_000) * p["input"]
        output_cost = (output_tokens / 1_000_000) * p["output"]
        return input_cost + output_cost

    def daily_cost(self, model, calls_per_day, avg_input, avg_output):
        """计算日成本"""
        single = self.calculate_cost(model, avg_input, avg_output)
        return single * calls_per_day

    def monthly_cost(self, model, calls_per_day, avg_input, avg_output):
        """计算月成本"""
        return self.daily_cost(model, calls_per_day, avg_input, avg_output) * 30


# ========== 场景分析 ==========
calc = CostCalculator()

print("=" * 65)
print("场景: 企业客服系统 - 日均 10,000 次对话")
print("=" * 65)
print(f"参数: 平均输入 800 tokens, 平均输出 400 tokens")
print()

# 对比各模型成本
D = chr(36)  # dollar sign
calls_per_day = 10000
avg_input = 800
avg_output = 400

print(f"{'模型':<22} {'单次成本':<14} {'日成本':<12} {'月成本':<12} {'速度'}")
print("-" * 65)

results = []
for model, info in ModelPricing.MODELS.items():
    single = calc.calculate_cost(model, avg_input, avg_output)
    daily = calc.daily_cost(model, calls_per_day, avg_input, avg_output)
    monthly = calc.monthly_cost(model, calls_per_day, avg_input, avg_output)
    results.append((model, single, daily, monthly, info["speed"]))
    print(f"{model:<22} {D}{single:<13.5f} {D}{daily:<11.2f} {D}{monthly:<11.2f} {info['speed']} t/s")

print()
print("=" * 65)
print("优化策略模拟")
print("=" * 65)

# 策略 1: 语义缓存
cache_hit_rate = 0.35  # 35% 缓存命中率
base_monthly = calc.monthly_cost("gpt-4o", calls_per_day, avg_input, avg_output)
cached_monthly = base_monthly * (1 - cache_hit_rate)
savings_cache = base_monthly - cached_monthly

print(f"\\n策略 1: 语义缓存 (命中率 {cache_hit_rate*100:.0f}%)")
print(f"  原始月成本 (gpt-4o): {D}{base_monthly:.2f}")
print(f"  缓存后月成本:        {D}{cached_monthly:.2f}")
print(f"  节省:                {D}{savings_cache:.2f} ({cache_hit_rate*100:.0f}%)")

# 策略 2: 模型路由（简单问题用小模型，复杂问题用大模型）
simple_ratio = 0.7  # 70% 简单问题
complex_ratio = 0.3
simple_monthly = calc.monthly_cost("gpt-4o-mini", int(calls_per_day * simple_ratio), avg_input, avg_output)
complex_monthly = calc.monthly_cost("gpt-4o", int(calls_per_day * complex_ratio), avg_input, avg_output)
routed_monthly = simple_monthly + complex_monthly
savings_route = base_monthly - routed_monthly

print(f"\\n策略 2: 智能路由 (简单 {simple_ratio*100:.0f}% -> mini, 复杂 {complex_ratio*100:.0f}% -> 4o)")
print(f"  路由后月成本: {D}{routed_monthly:.2f}")
print(f"  节省:         {D}{savings_route:.2f} ({savings_route/base_monthly*100:.0f}%)")

# 策略 3: Prompt 压缩（减少 input tokens）
compression_rate = 0.4  # 压缩 40% 的 input
compressed_input = int(avg_input * (1 - compression_rate))
compressed_monthly = calc.monthly_cost("gpt-4o", calls_per_day, compressed_input, avg_output)
savings_compress = base_monthly - compressed_monthly

print(f"\\n策略 3: Prompt 压缩 (压缩率 {compression_rate*100:.0f}%)")
print(f"  压缩后月成本: {D}{compressed_monthly:.2f}")
print(f"  节省:         {D}{savings_compress:.2f} ({savings_compress/base_monthly*100:.0f}%)")

# 综合策略
combined_monthly = calc.monthly_cost("gpt-4o-mini", int(calls_per_day * simple_ratio * (1-cache_hit_rate)), compressed_input, avg_output) + \\
                   calc.monthly_cost("gpt-4o", int(calls_per_day * complex_ratio * (1-cache_hit_rate)), compressed_input, avg_output)
savings_combined = base_monthly - combined_monthly

print(f"\\n策略 4: 综合优化 (缓存 + 路由 + 压缩)")
print(f"  优化后月成本: {D}{combined_monthly:.2f}")
print(f"  总节省:       {D}{savings_combined:.2f} ({savings_combined/base_monthly*100:.0f}%)")

print()
print("=" * 65)
print("结论: 通过组合优化策略，可将成本降低至原始的", end=" ")
print(f"{combined_monthly/base_monthly*100:.0f}%")
`
</script>

# 可观测性与 LLMOps

LLM 应用不同于传统软件，其输出具有非确定性，且成本与质量高度依赖运行时状态。可观测性是保障 LLM 应用稳定运行的基石。

## 1. LLMOps 概述

### 与传统 MLOps 的区别

| 维度 | 传统 MLOps | LLMOps |
|------|-----------|--------|
| 模型训练 | 自行训练，数据驱动 | 多数使用 API，Prompt 驱动 |
| 评估方式 | 固定测试集 + 指标 | 人工评估 + LLM-as-Judge |
| 版本管理 | 模型权重 + 代码 | Prompt + 编排逻辑 + 配置 |
| 成本结构 | GPU 训练成本为主 | Token 消耗（按调用计费） |
| 延迟特征 | 推理延迟稳定 | TTFT + 流式生成，变化大 |
| 可观测重点 | 精度/召回率/AUC | 幻觉率/延迟/Token 用量/成本 |
| 迭代速度 | 周/月级 | 小时/天级（改 Prompt 即可） |

::: info LLMOps 核心理念
LLMOps = **Prompt 管理** + **链路追踪** + **质量评估** + **成本控制** + **持续迭代**

它不是 MLOps 的替代，而是针对 LLM 应用特性的扩展与适配。
:::

### LLMOps 架构全景

```
┌─────────────────────────────────────────────────────────────┐
│                      LLMOps Platform                        │
├─────────────┬──────────────┬──────────────┬────────────────┤
│  Prompt     │   Tracing    │  Evaluation  │    Cost &      │
│  Registry   │   & Logging  │  & Testing   │    Billing     │
├─────────────┼──────────────┼──────────────┼────────────────┤
│ - 版本管理   │ - 调用链路    │ - 自动评估    │ - Token 计量   │
│ - A/B 测试  │ - 延迟分析    │ - 人工标注    │ - 成本归因     │
│ - 模板库    │ - 错误追踪    │ - 回归测试    │ - 预算告警     │
└─────────────┴──────────────┴──────────────┴────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM Application Layer                           │
│   RAG Pipeline / Agent / Chatbot / Code Gen / ...           │
└─────────────────────────────────────────────────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Providers                                 │
│   OpenAI / Anthropic / 本地部署 / 多模型路由                   │
└─────────────────────────────────────────────────────────────┘
```

## 2. 链路追踪

链路追踪（Tracing）是可观测性的核心，它让你看到每一次 LLM 调用的完整执行路径。

### 主流追踪工具对比

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| **LangSmith** | LangChain 官方，集成度高 | LangChain 生态用户 |
| **Phoenix (Arize)** | 开源，支持多框架 | 需要本地部署的团队 |
| **OpenTelemetry** | 通用标准，生态广泛 | 已有 OTel 基础设施 |
| **Langfuse** | 开源，UI 友好 | 中小团队快速上手 |
| **Helicone** | 代理模式，零侵入 | 不想改代码的场景 |

### 追踪数据模型

```
Trace (一次完整请求)
├── Span: query_parsing        [12ms]
├── Span: embedding            [35ms]
├── Span: vector_search        [89ms]
├── Span: rerank               [56ms]
├── Span: prompt_assembly      [8ms]
├── Span: llm_inference        [245ms]  ← 通常最耗时
│   ├── metadata: model=gpt-4o
│   ├── metadata: input_tokens=1450
│   ├── metadata: output_tokens=280
│   └── metadata: temperature=0.7
└── Span: post_processing      [15ms]
```

### 实践：构建简易追踪器

下面实现一个 LLM 调用追踪器，模拟追踪一个完整的 RAG Pipeline：

<PythonRunner :browser-runnable="true" :code="code1" />

::: tip 生产环境建议
- 使用 OpenTelemetry SDK 作为追踪基础设施
- 将 Trace 数据发送到 Jaeger/Tempo 等后端
- 结合 LangSmith/Langfuse 获得 LLM 特化的分析视图
:::

## 3. 关键监控指标

### 性能指标

| 指标 | 含义 | 目标值 | 告警阈值 |
|------|------|--------|----------|
| **TTFT** (Time to First Token) | 首 Token 延迟 | < 500ms | > 2s |
| **Tokens/s** | 生成速度 | > 50 t/s | < 20 t/s |
| **E2E Latency** | 端到端延迟 | < 3s | > 10s |
| **并发数** | 同时处理请求数 | 按容量规划 | > 80% 容量 |

### 质量指标

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **幻觉率** | 输出包含虚假信息的比例 | LLM-as-Judge / 人工抽检 |
| **回答相关性** | 输出与问题的匹配度 | Embedding 相似度 / 评分模型 |
| **引用准确率** | RAG 中引用来源的正确率 | 自动验证引用链接 |
| **拒答率** | 模型拒绝回答的比例 | 关键词 + 分类器检测 |

### 成本指标

| 指标 | 含义 | 关注点 |
|------|------|--------|
| **Token 消耗** | 输入/输出 Token 总量 | 按用户/功能/模型拆分 |
| **单次调用成本** | 每次请求的费用 | 异常高消耗检测 |
| **成本/收入比** | LLM 成本占收入比例 | 商业可持续性 |
| **错误率** | API 调用失败比例 | 429/500/超时分别统计 |

### 指标采集架构

```
┌──────────┐    ┌──────────────┐    ┌─────────────┐
│ LLM App  │───>│  Collector   │───>│  Time-Series│
│          │    │  (OTel/自研)  │    │  DB (Prom)  │
└──────────┘    └──────────────┘    └─────────────┘
                                          │
                                          ▼
                                    ┌─────────────┐
                                    │  Grafana    │
                                    │  Dashboard  │
                                    └─────────────┘
                                          │
                                          ▼
                                    ┌─────────────┐
                                    │  AlertMgr   │
                                    │  (告警通知)   │
                                    └─────────────┘
```

## 4. 成本控制与优化

### Token 成本计算

LLM API 的定价通常按 Token 数量计费，输入和输出分别定价。下面的计算器帮助你对比不同模型和优化策略的成本差异：

<PythonRunner :browser-runnable="true" :code="code2" />

### 核心优化策略

::: tip 四大成本优化方向
1. **语义缓存** — 相似问题直接返回缓存结果
2. **模型路由** — 按复杂度分流到不同模型
3. **Prompt 压缩** — 减少不必要的 Context
4. **批处理** — 合并请求降低固定开销
:::

#### 语义缓存策略

```
用户查询 ──> Embedding ──> 相似度检索
                              │
                    ┌─────────┴─────────┐
                    │                   │
              相似度 > 0.95        相似度 < 0.95
                    │                   │
                    ▼                   ▼
              返回缓存结果          调用 LLM
                                       │
                                       ▼
                                  存入缓存
```

**缓存设计要点：**
- 使用 Embedding 相似度而非精确匹配
- 设置合理的 TTL（避免过时信息）
- 按业务场景区分缓存策略（实时性要求不同）
- 监控缓存命中率，持续优化

#### 模型路由策略

| 请求类型 | 路由模型 | 判断依据 |
|----------|----------|----------|
| 简单问答 | gpt-4o-mini / Haiku | 短输入、无复杂推理 |
| 复杂推理 | gpt-4o / Sonnet | 多步逻辑、专业问题 |
| 代码生成 | 专用代码模型 | 代码相关 Intent |
| 多语言翻译 | 擅长多语言的模型 | 翻译类 Intent |

## 5. CI/CD 与模型迭代

### Prompt 版本管理

```
prompts/
├── rag_qa/
│   ├── v1.0.yaml      # 初始版本
│   ├── v1.1.yaml      # 优化格式要求
│   ├── v2.0.yaml      # 重构上下文组织方式
│   └── config.yaml    # 当前生产版本指向
├── summarize/
│   ├── v1.0.yaml
│   └── config.yaml
└── intent_classify/
    ├── v1.0.yaml
    └── config.yaml
```

### 灰度发布流程

```
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  开发    │────>│  评估    │────>│  灰度    │────>│  全量    │
│  & 测试  │     │  Gate    │     │  5-20%   │     │  发布    │
└─────────┘     └──────────┘     └──────────┘     └──────────┘
                     │                 │
                     ▼                 ▼
               自动评估通过?       指标无退化?
               - 准确率 > 85%     - 延迟无增加
               - 幻觉率 < 5%     - 错误率无上升
               - 延迟 < 3s       - 用户满意度持平
```

### 回滚机制

| 触发条件 | 回滚操作 | 恢复时间目标 |
|----------|----------|--------------|
| 错误率突增 > 5% | 自动回滚到上一版本 | < 1 分钟 |
| 幻觉率 > 10% | 人工确认后回滚 | < 5 分钟 |
| 成本突增 > 200% | 自动切换低成本模型 | < 1 分钟 |
| 延迟 P99 > 10s | 自动降级（缩短 Prompt） | < 30 秒 |

## 6. 告警策略

### 分级告警设计

| 级别 | 触发条件 | 通知方式 | 响应要求 |
|------|----------|----------|----------|
| **P0 (严重)** | 服务完全不可用 / 数据泄露 | 电话 + 短信 | 15 分钟内响应 |
| **P1 (高)** | 错误率 > 10% / 延迟 > 10s | 即时通讯 + 短信 | 30 分钟内响应 |
| **P2 (中)** | 成本异常 / 幻觉率升高 | 即时通讯 | 2 小时内响应 |
| **P3 (低)** | 缓存命中率下降 / 小幅延迟增加 | 邮件 / 工单 | 下个工作日处理 |

### 告警规则示例

```yaml
# Prometheus AlertManager 规则示例
groups:
  - name: llm_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(llm_requests_errors_total[5m]) / rate(llm_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: P1
        annotations:
          summary: "LLM 错误率超过 5%"

      - alert: HighLatency
        expr: histogram_quantile(0.99, llm_request_duration_seconds) > 10
        for: 5m
        labels:
          severity: P1
        annotations:
          summary: "LLM P99 延迟超过 10 秒"

      - alert: CostSpike
        expr: increase(llm_token_cost_dollars[1h]) > 2 * avg_over_time(llm_token_cost_dollars[7d])
        labels:
          severity: P2
        annotations:
          summary: "LLM 成本异常突增"

      - alert: HighHallucinationRate
        expr: rate(llm_hallucination_detected[1h]) / rate(llm_responses_total[1h]) > 0.1
        for: 30m
        labels:
          severity: P2
        annotations:
          summary: "幻觉率超过 10%"
```

::: warning 告警疲劳
告警规则需要持续调优。常见的反模式：
- 阈值设置过低导致频繁告警
- 缺少分级机制，所有告警同等对待
- 没有自动恢复检测，告警持续不消除

建议定期 Review 告警历史，调整阈值和抑制规则。
:::

## 7. 最佳实践总结

| 阶段 | 关键动作 | 工具推荐 |
|------|----------|----------|
| 开发期 | 本地追踪 + 单元评估 | Langfuse / Phoenix |
| 测试期 | 自动化评估 + 回归测试 | promptfoo / DeepEval |
| 灰度期 | 实时监控 + 对比分析 | Grafana + LangSmith |
| 生产期 | 全链路追踪 + 告警 + 成本控制 | OTel + Prometheus + 自研 |

---

::: info 延伸阅读
- [OpenTelemetry 官方文档](https://opentelemetry.io/docs/)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Phoenix by Arize](https://docs.arize.com/phoenix)
- [Langfuse 开源项目](https://langfuse.com/)
:::
