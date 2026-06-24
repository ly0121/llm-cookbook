---
title: 评估方法学
---

::: info 本章范围
本章评估「模型训练好坏」（PPL / lm-eval-harness benchmark）。Phase 4 的 `evaluation/KNOWLEDGE.md` 将评估「RAG 答案质量」（faithfulness / answer relevance）。两者互不重叠：训练阶段看本章，应用阶段看那章。
:::

# 第 6 章 评估方法学

> 训练有明确的 loss 曲线——评估却没有"正确答案"；评估方法的选择直接决定你对模型好坏的认知

## 为什么评估比训练更难

模型训练可以用 val loss 单一数字追踪进度。但"这个回答是否比那个更好"本质上是主观判断。不同的评估方法衡量的是不同维度的能力，没有一个指标可以代表一切。

评估方法选得不好，会产生误导：某模型在 benchmark 上刷出高分，但在实际对话中一塌糊涂——这种情况在 LLM 领域屡见不鲜。

## Perplexity 的局限

📌 **Perplexity（困惑度）** 是最古老的语言模型评估指标：

$$
\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{t=1}^N \log p(y_t \mid y_{<t})\right)
$$

PPL 越低，模型对测试文本的预测越准确。但 PPL 有两个致命局限：

1. **与下游任务质量相关性弱**：PPL 低的模型不一定在对话、翻译、推理任务上表现好
2. **被 tokenizer 影响**：不同 tokenizer 的 PPL 不可直接比较（词表大小影响每步预测难度）

PPL 适合：比较同一模型在量化前后的精度损失、监控微调过程中是否出现灾难性遗忘。不适合：跨模型比较能力强弱。

## LLM-as-Judge

用更强的 LLM（如 GPT-4）来给被测模型的输出打分。优点：接近人类判断，可自动化；缺点：昂贵，存在偏见（更喜欢自己风格的输出，偏向更长的回答）。

常见范式：
- **Pairwise comparison**：给 GPT-4 两个回答，让它选更好的
- **Absolute scoring**：让 GPT-4 给回答打 1-10 分
- **Criteria-based**：按 helpful / harmless / honest 三维度分别打分

## 标准 Benchmark 家族

| Benchmark | 内容 | 评估维度 | 格式 |
|-----------|------|---------|------|
| **MMLU** | 57 学科多选题（大学水平） | 知识广度 | 4 选 1 |
| **ARC** | 小学理科多选题 | 常识推理 | 4 选 1 |
| **HellaSwag** | 句子续写（选最合理的结尾） | 常识理解 | 4 选 1 |
| **GSM8K** | 小学数学应用题（需推理步骤） | 数学推理 | 开放生成 |
| **HumanEval** | Python 编程填空 | 代码生成 | 开放生成 |
| **TruthfulQA** | 常见错误信念测试 | 诚实性 | 多选 |

::: info Benchmark 污染问题
如果模型的预训练数据中包含了 benchmark 的题目和答案，评分就会虚高。这是领域内的严重问题。缓解方法：使用时间戳更新的 benchmark、私有测试集、或检查训练数据是否包含 benchmark 内容的哈希值。
:::

## lm-evaluation-harness 工程实践

EleutherAI 的 `lm-evaluation-harness` 是运行标准 benchmark 的事实标准工具：

```bash
lm_eval --model hf \
        --model_args pretrained=meta-llama/Llama-3-8B-Instruct \
        --tasks mmlu,arc_challenge,hellaswag,gsm8k \
        --device cuda:0 \
        --batch_size 8
```

`lm_eval` 自动处理 few-shot prompting、loglikelihood 评估、结果汇总，输出标准化分数。支持本地模型（HuggingFace 格式）、API 模型（OpenAI / Anthropic）、量化模型（bitsandbytes / GPTQ）。

## Chatbot Arena 与 MT-Bench

📌 **Chatbot Arena**：LMSYS 的人工盲测平台。用户与两个匿名模型对话，选出更好的那个，用 Elo 评分系统排名。这是目前最接近"用户真实体验"的评估方法，但成本高、速度慢。

**MT-Bench**：80 道多轮对话问题，用 GPT-4 打分（1-10 分）。分为数学、代码、写作、推理等 8 个类别，可以快速定位模型的弱项。对于团队内部评估，MT-Bench 比 Chatbot Arena 更经济实用。

## 评估选型建议

| 目标 | 推荐方法 | 成本 |
|------|---------|------|
| 快速检查量化/微调是否掉点 | PPL 计算 | 极低 |
| 跑标准 benchmark 比较能力 | lm-evaluation-harness | 中 |
| 评估对话质量 | MT-Bench + LLM-as-Judge | 中高 |
| 真实用户体验 | Chatbot Arena | 高 |

## 与生产对应

`lm-evaluation-harness`（`pip install lm-eval`）；PPL 计算用 `transformers` 的 `evaluate` 库；生产监控推荐组合：lm-eval（离线 benchmark）+ 人工抽样（在线质量）。

::: info 关联 demo
- [`11_eval_perplexity.py`](../../ml_foundations/post_training/11_eval_perplexity.py)：Perplexity 计算与局限演示
- [`12_eval_lm_harness.py`](../../ml_foundations/post_training/12_eval_lm_harness.py)：lm-evaluation-harness 标准 benchmark
:::

---

::: tip 下一节
→ [选型决策手册](./selection)
:::

::: info 上一节
← [PTQ：训练后量化](./quantization)
:::
