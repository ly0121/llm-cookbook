---
title: LLM评测与测试
---

<script setup>
const code1 = `# RAGAS 风格 Faithfulness（忠实度）评分
# 检查答案是否基于给定上下文（grounded in context）

def extract_statements(text):
    """将文本拆分为独立陈述句"""
    # 简化版：按句号分割
    statements = [s.strip() for s in text.replace('！', '。').replace('？', '。').split('。') if s.strip()]
    return statements

def check_statement_in_context(statement, context):
    """检查单条陈述是否能从上下文推断（简化版：关键词重叠度）"""
    statement_words = set(statement)
    context_words = set(context)
    overlap = statement_words & context_words
    # 计算字符级重叠率作为简化判据
    if len(statement_words) == 0:
        return 0.0
    return len(overlap) / len(statement_words)

def faithfulness_score(answer, context, threshold=0.5):
    """
    计算 Faithfulness 分数
    = 能从 context 推断的陈述数 / 总陈述数
    """
    statements = extract_statements(answer)
    if not statements:
        return 1.0  # 空答案视为完全忠实

    supported = 0
    details = []
    for stmt in statements:
        score = check_statement_in_context(stmt, context)
        is_supported = score >= threshold
        if is_supported:
            supported += 1
        details.append({
            'statement': stmt,
            'overlap_score': round(score, 3),
            'supported': is_supported
        })

    final_score = supported / len(statements)
    return final_score, details

# === 测试用例 ===
context = """
Transformer架构由Google团队在2017年发表的论文"Attention Is All You Need"中提出。
该架构完全基于自注意力机制，摒弃了传统的循环神经网络结构。
BERT是基于Transformer的编码器部分构建的预训练模型，于2018年发布。
"""

# 忠实的答案（基于上下文）
answer_faithful = "Transformer由Google在2017年提出，基于自注意力机制。BERT基于Transformer编码器构建。"

# 不忠实的答案（包含上下文中没有的信息）
answer_unfaithful = "Transformer由OpenAI在2015年提出，使用CNN结构。目前已被Mamba架构取代。"

print("=" * 50)
print("RAGAS Faithfulness 评分演示")
print("=" * 50)

print("\\n【上下文】")
print(context.strip())

print("\\n" + "-" * 50)
print("【测试1: 忠实的答案】")
print(f"答案: {answer_faithful}")
score1, details1 = faithfulness_score(answer_faithful, context)
print(f"\\nFaithfulness 分数: {score1:.2f}")
print("各陈述详情:")
for d in details1:
    status = '✓ 支持' if d['supported'] else '✗ 不支持'
    print(f"  [{status}] (重叠率:{d['overlap_score']}) {d['statement']}")

print("\\n" + "-" * 50)
print("【测试2: 不忠实的答案】")
print(f"答案: {answer_unfaithful}")
score2, details2 = faithfulness_score(answer_unfaithful, context)
print(f"\\nFaithfulness 分数: {score2:.2f}")
print("各陈述详情:")
for d in details2:
    status = '✓ 支持' if d['supported'] else '✗ 不支持'
    print(f"  [{status}] (重叠率:{d['overlap_score']}) {d['statement']}")

print("\\n" + "=" * 50)
print("结论: 忠实度分数越高，答案越贴合上下文")
print(f"忠实答案: {score1:.2f} vs 不忠实答案: {score2:.2f}")
`

const code2 = `# LLM-as-a-Judge: 基于评分准则的模拟评估
# 模拟用强模型对弱模型输出进行打分

class JudgeRubric:
    """评分准则定义"""
    def __init__(self, name, criteria, scale=(1, 5)):
        self.name = name
        self.criteria = criteria  # dict: {分数: 描述}
        self.scale = scale

class LLMJudge:
    """模拟 LLM-as-a-Judge 评估器"""

    def __init__(self, rubrics):
        self.rubrics = rubrics

    def score_response(self, question, response, reference=None):
        """对模型回答进行多维度评分（规则模拟版）"""
        scores = {}
        for rubric in self.rubrics:
            score = self._evaluate_dimension(rubric, question, response, reference)
            scores[rubric.name] = score
        return scores

    def _evaluate_dimension(self, rubric, question, response, reference):
        """基于简单规则模拟评分（实际应调用GPT-4等强模型）"""
        if rubric.name == '相关性':
            # 检查回答与问题的关键词重叠
            q_chars = set(question)
            r_chars = set(response)
            overlap = len(q_chars & r_chars) / max(len(q_chars), 1)
            return min(5, max(1, int(overlap * 6)))

        elif rubric.name == '完整性':
            # 基于回答长度评估（简化）
            length = len(response)
            if length > 200: return 5
            elif length > 100: return 4
            elif length > 50: return 3
            elif length > 20: return 2
            else: return 1

        elif rubric.name == '准确性':
            # 与参考答案对比（如有）
            if reference:
                ref_chars = set(reference)
                resp_chars = set(response)
                overlap = len(ref_chars & resp_chars) / max(len(ref_chars), 1)
                return min(5, max(1, int(overlap * 6)))
            return 3  # 无参考时给中间分

        elif rubric.name == '流畅性':
            # 检查是否有重复、乱码等
            if len(set(response)) / max(len(response), 1) > 0.3:
                return 4
            return 2

        return 3

    def generate_report(self, question, response, scores):
        """生成评估报告"""
        total = sum(scores.values())
        max_total = len(scores) * 5
        overall = total / max_total * 100

        report = []
        report.append(f"问题: {question}")
        report.append(f"回答: {response[:80]}{'...' if len(response) > 80 else ''}")
        report.append("")
        report.append("评分详情:")
        report.append("-" * 40)
        for dim, score in scores.items():
            bar = '█' * score + '░' * (5 - score)
            report.append(f"  {dim}: [{bar}] {score}/5")
        report.append("-" * 40)
        report.append(f"  综合得分: {total}/{max_total} ({overall:.1f}%)")

        quality = '优秀' if overall >= 80 else '良好' if overall >= 60 else '一般' if overall >= 40 else '较差'
        report.append(f"  质量等级: {quality}")
        return "\\n".join(report)


# === 定义评分准则 ===
rubrics = [
    JudgeRubric('相关性', {
        5: '完全切题，直接回答问题',
        3: '部分相关，有些偏题',
        1: '完全不相关'
    }),
    JudgeRubric('完整性', {
        5: '全面覆盖所有要点',
        3: '覆盖主要内容但有遗漏',
        1: '严重不完整'
    }),
    JudgeRubric('准确性', {
        5: '所有信息准确无误',
        3: '大部分准确，少量错误',
        1: '存在严重事实错误'
    }),
    JudgeRubric('流畅性', {
        5: '表达清晰流畅',
        3: '基本通顺',
        1: '语句混乱'
    }),
]

judge = LLMJudge(rubrics)

# === 评估多个模型的回答 ===
question = "什么是Transformer架构的核心创新？"
reference = "Transformer的核心创新是自注意力机制（Self-Attention），它能并行处理序列中所有位置的关系，取代了RNN的循环结构。"

responses = {
    '模型A (高质量)': 'Transformer架构的核心创新是自注意力机制（Self-Attention）。它允许模型在处理序列时，能够同时关注输入序列中所有位置之间的关系，而不需要像RNN那样逐步处理。这使得模型可以并行计算，大大提升了训练效率。',
    '模型B (中等质量)': 'Transformer主要用了attention机制，比RNN快。',
    '模型C (低质量)': '深度学习是人工智能的分支，包括CNN、RNN等多种网络结构。',
}

print("=" * 50)
print("LLM-as-a-Judge 评估报告")
print("=" * 50)
print(f"\\n参考答案: {reference}")
print()

for model_name, response in responses.items():
    print(f"\\n{'=' * 50}")
    print(f"【{model_name}】")
    print("=" * 50)
    scores = judge.score_response(question, response, reference)
    report = judge.generate_report(question, response, scores)
    print(report)

print("\\n" + "=" * 50)
print("总结: LLM-as-a-Judge 通过多维度评分准则")
print("可以系统化地评估模型输出质量，适用于大规模评测")
`
</script>

# LLM 评测与测试

评测是衡量大语言模型能力、指导模型迭代优化的核心环节。一个完善的评测体系需要覆盖模型的多维度能力，并结合离线和在线方法。

## 1. 评测体系概述

::: info 核心原则
评测的目标不是给出绝对分数，而是帮助我们做出决策：选哪个模型、哪个 Prompt 更好、哪次迭代有效。
:::

### 离线评测 vs 在线评测

| 维度 | 离线评测（Offline） | 在线评测（Online） |
|------|---------------------|---------------------|
| **时机** | 模型上线前 | 模型上线后 |
| **数据** | 固定测试集 | 真实用户流量 |
| **指标** | 准确率、BLEU、ROUGE 等 | 用户满意度、点击率、留存 |
| **优点** | 可复现、成本低 | 反映真实效果 |
| **缺点** | 可能脱离真实场景 | 受流量波动影响 |
| **代表方法** | Benchmark、人工标注 | A/B 测试、用户反馈 |

### 评测流程

```
需求定义 → 指标选择 → 数据准备 → 评测执行 → 结果分析 → 迭代优化
    ↑                                                         |
    └─────────────────────────────────────────────────────────┘
```

## 2. 基准测试集（Benchmarks）

主流基准测试覆盖不同能力维度：

| 基准测试 | 评测能力 | 任务类型 | 数据规模 | 语言 |
|----------|----------|----------|----------|------|
| **MMLU** | 综合知识 | 多选题（57个学科） | ~16,000 题 | 英文 |
| **CMMLU** | 中文综合知识 | 多选题（67个学科） | ~12,000 题 | 中文 |
| **HumanEval** | 代码生成 | 函数补全 | 164 题 | Python |
| **GSM8K** | 数学推理 | 小学数学应用题 | 8,500 题 | 英文 |
| **HELM** | 多维综合 | 多场景多指标 | 大规模 | 英文 |
| **C-Eval** | 中文学科 | 多选题（52个学科） | ~13,000 题 | 中文 |

::: tip 如何选择基准测试
- **通用能力对比**：MMLU / CMMLU
- **代码能力**：HumanEval / MBPP
- **推理能力**：GSM8K / MATH / ARC
- **中文场景**：CMMLU / C-Eval / SuperCLUE
:::

### 指标解读注意事项

```
常见陷阱：
├── 数据污染：训练集与测试集重叠导致虚高
├── 刷榜优化：针对特定测试集优化而非通用能力
├── 评测方式差异：0-shot vs 5-shot 结果不可直接对比
└── 版本差异：同一 benchmark 不同版本分数不同
```

## 3. RAG 评测框架（RAGAS）

RAG（检索增强生成）系统需要专门的评测方法，RAGAS 是目前最流行的 RAG 评测框架。

### 核心指标

| 指标 | 评估对象 | 计算方式 | 含义 |
|------|----------|----------|------|
| **Faithfulness** | 生成质量 | 答案中可从上下文推导的陈述比例 | 答案是否忠于检索内容 |
| **Answer Relevancy** | 生成质量 | 答案与问题的语义相关度 | 答案是否切题 |
| **Context Precision** | 检索质量 | 相关上下文排在前面的比例 | 检索排序是否合理 |
| **Context Recall** | 检索质量 | 参考答案中能从上下文推导的比例 | 检索是否覆盖完整 |

### Faithfulness 计算流程

```
1. 将答案拆分为独立陈述（statements）
2. 逐条判断每个陈述是否能从 context 中推断
3. Faithfulness = 可推断的陈述数 / 总陈述数
```

::: info RAGAS 评测不需要人工标注
RAGAS 的一大优势是利用 LLM 本身进行评估，无需人工标注 ground truth（Context Recall 除外），大幅降低评测成本。
:::

### 交互示例：Faithfulness 评分

<PythonRunner :code="code1" />

## 4. LLM-as-a-Judge

利用强大的 LLM（如 GPT-4）作为评判者，对其他模型的输出进行评估。

### 方法分类

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| **单点评分** | 对单个回答按准则打 1-5 分 | 质量评估 |
| **成对比较** | 比较两个回答哪个更好 | 模型对比 |
| **参考对照** | 与标准答案对比评分 | 有 ground truth 时 |
| **多维评分** | 从多个维度分别打分 | 细粒度分析 |

### 评分准则设计

```
评分准则 (Rubric) 示例：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
维度：相关性 (Relevance)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5分：完全切题，直接回答问题核心
4分：大部分相关，有少量额外信息
3分：部分相关，有些偏题
2分：勉强相关，主要内容偏离
1分：完全不相关
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

::: warning LLM-as-a-Judge 的局限性
- **位置偏差**：倾向于给排在前面的答案更高分
- **冗长偏差**：倾向于给更长的回答更高分
- **自我偏好**：可能偏好自己生成的内容
- **缓解方法**：交换位置重复评测、控制长度、多模型交叉评估
:::

### 交互示例：LLM-as-Judge 评分

<PythonRunner :code="code2" />

## 5. A/B 测试方法

A/B 测试是在线评测的金标准方法，通过随机分流真实用户来比较不同方案的效果。

### 实施流程

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  用户请求    │────→│  流量分配器   │────→│  方案A (50%) │
│             │     │  (随机分流)   │────→│  方案B (50%) │
└─────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │  指标收集     │
                                          │  统计检验     │
                                          │  结论决策     │
                                          └──────────────┘
```

### 关键要素

| 要素 | 说明 |
|------|------|
| **分流策略** | 用户级（同一用户始终看同一方案）vs 请求级 |
| **样本量** | 需提前做 Power Analysis，确保统计显著性 |
| **观测周期** | 通常 1-2 周，避免新奇效应 |
| **核心指标** | 1-2 个主指标 + 多个护栏指标 |
| **统计检验** | t-test / Mann-Whitney / Bootstrap |

### LLM 场景常用指标

```
用户体验指标：
├── 回答采纳率（用户是否接受了回答）
├── 追问率（是否需要多轮才能解决）
├── 会话时长（越短通常越好）
├── 用户显式反馈（点赞/点踩）
└── 留存率（用户是否持续使用）
```

## 6. 评测工具

### 工具对比

| 工具 | 定位 | 特点 | 适用场景 |
|------|------|------|----------|
| **Promptfoo** | Prompt 评测 | YAML 配置、支持多 Provider、CI 集成 | Prompt 迭代优化 |
| **OpenCompass** | 综合评测平台 | 中文友好、支持 50+ 数据集、分布式 | 模型能力全面评测 |
| **lm-evaluation-harness** | 学术基准评测 | EleutherAI 出品、标准化、可复现 | 学术论文标准评测 |
| **RAGAS** | RAG 评测 | 专注 RAG、无需标注、自动化 | RAG 系统优化 |
| **DeepEval** | 单元测试风格 | Pytest 集成、CI/CD 友好 | 工程化测试 |

### Promptfoo 配置示例

```yaml
# promptfooconfig.yaml
providers:
  - openai:gpt-4
  - openai:gpt-3.5-turbo

prompts:
  - "请回答：{{question}}"
  - "你是专家，请详细回答：{{question}}"

tests:
  - vars:
      question: "什么是机器学习？"
    assert:
      - type: contains
        value: "数据"
      - type: llm-rubric
        value: "回答应准确、完整地解释机器学习的概念"
```

### OpenCompass 使用流程

```bash
# 安装
pip install opencompass

# 运行评测
python run.py --models hf_llama_7b --datasets mmlu_ppl ceval_ppl

# 查看结果
python summarize.py
```

::: tip 评测最佳实践
1. **明确目标**：先想清楚评测要回答什么问题
2. **组合方法**：离线 Benchmark + 在线 A/B 测试结合
3. **持续迭代**：评测集要定期更新，避免过拟合
4. **多维度覆盖**：不要只看单一指标
5. **版本管理**：记录每次评测的模型版本、Prompt 版本、数据版本
:::

## 总结

| 方法 | 最适合 | 成本 | 可信度 |
|------|--------|------|--------|
| Benchmark | 快速筛选、能力定位 | 低 | 中 |
| RAGAS | RAG 系统评测 | 中 | 中高 |
| LLM-as-Judge | 开放式评测、大规模 | 中 | 中 |
| 人工评测 | 最终质量把关 | 高 | 高 |
| A/B 测试 | 上线决策 | 高 | 最高 |

评测不是一次性工作，而是贯穿 LLM 应用全生命周期的持续过程。建议建立自动化评测 Pipeline，在每次模型或 Prompt 变更时自动触发评测，确保质量不退化。
