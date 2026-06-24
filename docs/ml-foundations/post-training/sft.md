---
title: SFT：让模型学会听指令
---

# 第 1 章 SFT：让模型学会"听指令"

> 用极少量高质量数据，改变模型的输出行为——而不改变它已学到的世界知识

SFT（Supervised Fine-Tuning，监督微调）是把 base model 变成对话助手的第一步。模型已有的知识不变，改变的只是**输出行为的条件分布**——从"续写文本"切换到"回应指令"。

## 数据形态：三种主流格式

📌 **chat template** 是不同数据格式的统一抽象，不同模型的 special token 格式不同（Mistral 用 `[INST]`，Qwen 用 `<|im_start|>`），但核心结构一致。

**Alpaca 格式**（单轮指令）：
```json
{
  "instruction": "把下面句子翻译成英文",
  "input": "今天天气很好",
  "output": "The weather is great today."
}
```

**ShareGPT 格式**（多轮对话）：
```json
{
  "conversations": [
    {"from": "human", "value": "什么是 LoRA？"},
    {"from": "gpt", "value": "LoRA 是一种参数高效的微调方法…"},
    {"from": "human", "value": "它比全量微调省多少显存？"},
    {"from": "gpt", "value": "通常节省 80-90% 显存…"}
  ]
}
```

**Chat template 渲染后**（以 LLaMA-3 为例）：
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
什么是 LoRA？<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
LoRA 是一种…<|eot_id|>
```

`tokenizer.apply_chat_template()` 自动处理这个渲染过程。

## 损失函数与 Label Mask

SFT 的训练目标是让模型学会"如何回应"，而不是"如何提问"。对整个序列（包括 user prompt）都算 loss 会出现两个问题：
1. 模型浪费参数容量"记住"用户输入的写法
2. prompt 部分的 loss 稀释有用的梯度信号

解决方案：**对 prompt 部分做 label mask**，只在 assistant 回复的 token 上计算 loss：

```
token 序列: [BOS] [INST] 什么是 LoRA ？ [/INST] LoRA 是 一种 … [EOS]
label mask:  -100  -100  -100 -100 -100  -100  有效  有效 有效  有效
```

PyTorch CrossEntropyLoss 对 label=-100 的位置自动忽略。损失公式：

$$
\mathcal{L}_\text{SFT} = -\frac{1}{|T_\text{response}|} \sum_{t \in T_\text{response}} \log p_\theta(y_t \mid y_{<t}, x)
$$

其中 $T_\text{response}$ 是 assistant 回复的 token 集合，$x$ 是完整对话上下文（含 prompt）。

## 多 Turn 对话的 Mask 策略对比

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **只掩盖最后一轮 prompt** | 只对最后一个 user turn mask | 简单易实现 | 浪费前几轮 assistant 的监督信号 |
| **掩盖所有 user turns** | 所有 human 发言全部 mask=-100 | 充分利用所有 assistant 回复 | 实现稍复杂 |
| **不掩盖任何内容** | 整个序列都算 loss | 最简单 | 模型会"学习提问风格"，效果偏差 |

推荐方案：**掩盖所有 user turns**。`trl.SFTTrainer` 通过 `DataCollatorForCompletionOnlyLM` 配置。

## TRL SFTTrainer 工程实践

`SFTTrainer` 封装了三件事：
1. 调用 `tokenizer.apply_chat_template()` 把对话格式转成 token ids
2. 用 `DataCollatorForCompletionOnlyLM` 自动生成 label mask
3. 标准 `Trainer` 循环（forward → loss → backward → optimizer step）

关键参数：
```python
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    args=SFTConfig(
        max_length=2048,             # TRL 1.6.0 重命名：max_seq_length → max_length
        output_dir="runs/sft",
        num_train_epochs=3,
        per_device_train_batch_size=2,
    ),
)
```

## SFT 的失败模式

**灾难性遗忘（Catastrophic Forgetting）**：过度 SFT 会使模型"忘记"预训练阶段的知识。典型症状：SFT 后模型在通用 benchmark（MMLU / HellaSwag）上得分下降。缓解方法：降低学习率（1e-5 以下）、减少 epoch 数（1-3 轮）、加 KL 惩罚项。

**模板过拟合（Template Overfitting）**：模型学会"说话的格式"但内容空洞。症状：所有回答开头都是"当然！我很乐意帮你…"。根本原因是 SFT 数据里充斥着模板化回答。解决方法：清洗数据，过滤低信息密度的回答。

::: warning 数据质量比数量更重要
1k 条高质量人工标注的数据，通常优于 100k 条 GPT-4 生成的模板化数据。SFT 数据的核心价值在于"展示期望行为的多样性"，而不是数量。
:::

## 与生产对应

`trl.SFTTrainer`；数据格式参考 `datasets.load_dataset("timdettmers/openassistant-guanaco")`；Label mask 实现在 `trl/trainer/sft_trainer.py` 的 `DataCollatorForCompletionOnlyLM` 类。

::: info 关联 demo
- [`01_data_construction.py`](../../ml_foundations/post_training/01_data_construction.py)：动手构造 SFT 数据集；理解 chat template 格式
- [`02_multi_turn_chat.py`](../../ml_foundations/post_training/02_multi_turn_chat.py)：多轮对话的 mask 策略；看清哪些 token 参与 loss
- [`03_sft_full.py`](../../ml_foundations/post_training/03_sft_full.py)：Full-parameter SFT；baseline 对比
:::

---

::: tip 下一节
→ [LoRA：参数高效微调数学](./lora)
:::

::: info 上一节
← [全景：从 base model 到 ChatGPT](./overview)
:::
