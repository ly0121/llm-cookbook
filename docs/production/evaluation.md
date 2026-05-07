---
title: 评估体系
---

# LLM 评估体系

从"感觉还行"到"数据说话"，评估是科学开发 LLM 应用的基石。

## 1. 为什么需要评估

```
无评估: 改了参数 → "试了几个问题，感觉还行" → 上线 → 用户投诉
有评估: 改了参数 → 跑评估集 → 忠实度 82%→85% ✅ → 有信心上线
```

三大价值：质量保障、回归检测、优化指导。

## 2. LLM 评估的挑战

| 挑战 | 说明 |
|------|------|
| 非确定性 | 同输入不同输出，需多次评估取均值 |
| 主观性 | "好"的定义因场景而异 |
| 多维度 | 准确性、流畅性、安全性需分别评估 |
| 规模化 | 人工评估成本高，自动评估有偏差 |

## 3. 自动评估指标

| 指标 | 适用 | 说明 |
|------|------|------|
| BLEU | 翻译 | n-gram 重合度 |
| ROUGE | 摘要 | 召回导向 |
| BERTScore | 通用 | 语义相似度 |
| Exact Match | 事实问答 | 精确匹配 |

## 4. RAG 专用指标

| 维度 | 评估什么 | 计算方式 |
|------|---------|---------|
| Context Relevance | 检索内容相关性 | relevant_chunks / total |
| Faithfulness | 回答忠于上下文 | supported_claims / total |
| Answer Correctness | 最终答案正确性 | 与标准答案比对 |

## 5. LLM-as-Judge

用另一个 LLM 评估目标 LLM 的输出：

```python
judge_prompt = """评估以下回答的质量(1-5分):

问题: {question}
参考答案: {reference}
被评估回答: {answer}

评分标准:
5 = 完全正确且全面
4 = 基本正确，有小瑕疵
3 = 部分正确
2 = 大部分错误
1 = 完全错误

评分(只输出数字):"""
```

优势：可规模化、成本低。风险：评估偏差需校准。

## 6. 评估数据集构建

| 方法 | 说明 |
|------|------|
| 人工标注 | 金标准，成本高 |
| LLM 生成 | 快速，需人工验证 |
| 生产日志 | 真实场景，需脱敏 |
| 对抗样本 | 测试边界情况 |

建议：至少 50-100 条高质量评估样本。

## 7. A/B 测试

```python
# 随机分流
import random

def select_model(session_id: str):
    if hash(session_id) % 100 < 50:
        return "model_a"  # 对照组
    return "model_b"      # 实验组
```

## 8. 持续评估（CI/CD 集成）

```yaml
# 每次 Prompt 改动触发评估
on_pr:
  - run_eval_suite
  - compare_with_baseline
  - if regression > 5%: block_merge
```

## 9. 评估工具

| 工具 | 特点 |
|------|------|
| RAGAS | RAG 自动化评估框架 |
| LangSmith | LangChain 官方 |
| DeepEval | 开源 LLM 评估 |
| 人工评估 | 最终验收金标准 |

::: warning 需要本地运行
完整实现见 `evaluation/rag_eval.py`，包含 LLM-as-Judge 和 RAG 评估流程。
:::

---

::: tip 下一步
- [可观测性](/engineering/observability) — 线上持续监控
- [安全护栏](/production/guardrails) — 评估安全防护
:::
