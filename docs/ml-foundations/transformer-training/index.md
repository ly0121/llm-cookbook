# 零.5、Transformer 训练实战

> 从 BPE 到 KV Cache，亲手训出第一个能写"伪莎士比亚"的小 LLM

---

## 为什么需要这一章？

完成 `docs/ml-foundations/` 的经典 ML / DL / NLP 三章后，你已经知道"什么是训练"、"什么是反向传播"——但 Transformer 是一头更大的野兽。直接跳入 LLM 应用层，attention、KV cache、temperature 这些词会一直是黑盒。

**本章的目标**：把 LLM 的每一块零件拆开、看透、再装回去。完成后你能：

1. **自己写 self-attention**：从 $\text{softmax}(QK^\top/\sqrt{d_k})V$ 到 NumPy 实现，一行都不跳过
2. **看懂训练超参的含义**：知道 GPT-2 / LLaMA 的 AdamW + cosine warmup + grad_clip 为什么这样设
3. **理解 OpenAI API 的旋钮**：temperature / top_p / top_k 在做什么，什么场景用哪种
4. **自己实现 KV cache**：知道 ChatGPT "第二个 token 比第一个快"背后的数学

::: tip 阅读建议
本章是 hands-on 章节，建议跟着代码跑一遍。`gpt_train.py` 在 Mac MPS 上约 30 秒可见 loss 从 4.22 降到 1.83，CPU 也只需 5-6 分钟。其它 6 个 demo 全部在 1 分钟以内。
:::

::: info 前置知识
- **PyTorch 基础**：张量操作、`nn.Module`、`optimizer.step()`（见 `docs/ml-foundations/deep-learning`）
- **softmax / 交叉熵**：能说清楚 softmax 的输出是概率分布（见 `docs/ml-foundations/classical-ml`）
- **不需要**：CUDA、分布式训练、预训练模型权重
:::

---

## 学习路径总览

7 个 demo 之间的依赖关系如下——前三个互相独立，都汇入主训练脚本，主训练脚本生成的 checkpoint 供后三个使用：

```
   BPE 分词            自注意力           位置编码
bpe_tokenizer.py   attention_from_  positional_
                   scratch.py       encoding.py
       │                 │               │
       └─────────────────┼───────────────┘
                         ↓
                   gpt_train.py
                 （主训练 + checkpoint）
                         │
           ┌─────────────┼─────────────┐
           ↓             ↓             ↓
   采样策略          注意力可视化      KV cache
sampling_         attention_       kv_cache.py
strategies.py     visualization.py
```

::: info 运行前置步骤
```bash
# 下载训练语料（~1MB Tiny Shakespeare）
curl -L -o ml_foundations/transformer_training/data/tiny_shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
后三个 demo（采样、可视化、KV cache）需要先运行 `gpt_train.py` 生成 checkpoint。
:::

---

## 本章包含

### 1. [BPE Tokenization](./tokenization)

> 为什么 LLM 看到的不是字也不是词，而是"子词"

**核心问题**：GPT 系列为什么选择 BPE？词表大小（456 vs 50K vs 128K）对模型有何影响？如何从零实现一个 BPE 训练循环和编解码器？

---

### 2. [自注意力机制](./attention)

> $\text{softmax}(QK^\top/\sqrt{d_k})V$ —— 把这个公式吃透

**核心问题**：Q / K / V 各自代表什么语义？为什么要除以 $\sqrt{d_k}$（不除会怎样）？因果 Mask 如何让模型"只看过去"？多头注意力的分头计算和 concat 为什么有效？

---

### 3. [位置编码](./positional-encoding)

> Attention 本身无序——三种方案各有权衡

**核心问题**：绝对正余弦编码、学习式编码、RoPE 三种方案分别适合什么场景？为什么现代 LLM（LLaMA / Qwen / DeepSeek）都迁移到 RoPE？RoPE 如何天然编码相对位置？

---

### 4. [训练循环](./training)

> AdamW + cosine warmup + grad_clip —— LLM 训练的标准配方

**核心问题**：Transformer Block 的 pre-LN 残差结构是如何组装的？为什么 AdamW 的 $\beta_2=0.95$ 而不是默认的 $0.999$？cosine warmup 的曲线形状为什么这样设计？2000 步后 val loss 从 4.22 降到 1.83 意味着什么？

---

### 5. [生成与采样策略](./generation)

> 同一个模型，4 种采样策略，截然不同的输出风格

**核心问题**：greedy、temperature、top-k、top-p（nucleus sampling）各自的数学定义是什么？ChatGPT 的 temperature / top_p 旋钮在底层做了什么？什么场景下选哪种策略？

---

### 6. [注意力可视化](./attention-visualization)

> 打开黑盒——看训练好的注意力头在关注什么

**核心问题**：不同注意力头学到了哪些不同的语言模式？如何用 ASCII 热图直观展示注意力权重？可视化结果能否验证"语法头 vs 语义头"的直觉？

---

### 7. [KV Cache 推理加速](./inference)

> 为什么 ChatGPT "第二个 token 比第一个快"

**核心问题**：无 cache 的自回归生成为什么是 $O(T^2 \cdot d)$？KV cache 如何把每步复杂度降到 $O(d)$？内存代价有多大（LLaMA-70B 单 batch 需要约 21.5GB）？FlashAttention / PagedAttention / GQA 各解决什么问题？

---

## 模型架构一览

本章训练的 GPT 与 GPT-2 / LLaMA 架构骨架完全一致，只是规模不同：

```
输入 token ids
      │
      ▼
┌─────────────────────────────────────┐
│  Token Embedding  (vocab × d_model) │
│  + Position Embedding (L × d_model) │   ← 学习式 PE，等同 GPT-2
└──────────────────┬──────────────────┘
                   │
          ┌────────▼────────┐
          │   Transformer   │  × N 层（本 demo N=6）
          │      Block      │
          │  ┌───────────┐  │
          │  │  Pre-LN   │  │
          │  │ + Causal  │  │  ← 因果自注意力（上三角 mask）
          │  │ Self-Attn │  │
          │  └─────┬─────┘  │
          │        + (残差)  │
          │  ┌─────▼─────┐  │
          │  │  Pre-LN   │  │
          │  │  + MLP    │  │  ← FFN：d → 4d → d，GELU 激活
          │  └─────┬─────┘  │
          │        + (残差)  │
          └────────┬─────────┘
                   │
          ┌────────▼────────┐
          │  Final LayerNorm │
          └────────┬─────────┘
                   │
          ┌────────▼────────┐
          │   LM Head        │  ← 权重与 Token Embedding 共享（Weight Tying）
          │  (d_model→vocab) │
          └─────────────────┘
                   │
                logits → CrossEntropyLoss（训练）
                       → 采样策略（推理）
```

**缩放规律**：把本 demo 的超参换一下，就是真实的大模型：

| 模型 | 层数 | 头数 | d_model | 参数量 |
|------|------|------|---------|--------|
| **本 demo** | 6 | 6 | 192 | ~3M |
| GPT-2 small | 12 | 12 | 768 | 124M |
| GPT-2 XL | 48 | 25 | 1600 | 1.5B |
| LLaMA-7B | 32 | 32 | 4096 | 7B |

参数量粗估公式：$N \approx 12 \times L \times d^2$（每层 4 个 attention 矩阵 + FFN 扩展比 4）。

---

## 配套代码

所有 demo 在 `ml_foundations/transformer_training/` 目录下：

| 文档 | 配套 demo | 预计运行时长 |
|------|-----------|------------|
| BPE Tokenization | `bpe_tokenizer.py` | < 30s（demo），完整训练 1m+ |
| 自注意力机制 | `attention_from_scratch.py` | < 5s |
| 位置编码 | `positional_encoding.py` | < 5s |
| 训练循环 | `gpt_train.py` | CPU ~5-6 min / MPS ~30s |
| 生成与采样 | `sampling_strategies.py` | < 30s（需要 checkpoint） |
| 注意力可视化 | `attention_visualization.py` | < 10s（需要 checkpoint） |
| KV cache | `kv_cache.py` | < 30s（需要 checkpoint） |

::: warning 硬件说明
所有 demo 在 Mac CPU 上可跑通，无需 NVIDIA GPU、无需下载预训练模型、无需 API key。Apple Silicon 自动启用 MPS 加速主训练（`gpt_train.py`），速度约为 CPU 的 10×。
:::

---

## 推荐学习顺序

::: tip 完整路径（约 1-2 天）
1. **BPE → Attention → PE**：先建立"输入 → 特征"各组件的直觉，三个 demo 各不到 5 分钟（半天）
2. **跑主训练**：`python ml_foundations/transformer_training/gpt_train.py`，等 loss 从 4.22 降到 1.83（CPU 约 5-6 分钟，MPS 约 30 秒），同时阅读训练循环文档
3. **采样实验**：同一 checkpoint 尝试 greedy / temperature / top-k / top-p 四种策略，观察文本风格差异
4. **注意力可视化**：在 checkpoint 上画热度图，直观感受不同注意力头学到的模式
5. **KV cache**：理解推理加速原理，从小 demo 推演到 LLaMA-70B 的 21.5GB cache 问题
:::

::: tip 速览路径（约 2-3 小时）
仅阅读 7 篇文档，不跑代码。优先阅读**自注意力机制**（理解 $\sqrt{d_k}$ 的作用）和 **KV Cache**（理解推理瓶颈），这两篇是理解现代 LLM 工程的基础。
:::

---

## 实测训练效果

本章主训练脚本 `gpt_train.py` 使用以下配置：

| 配置项 | 值 |
|--------|----|
| 模型参数量 | ~3M（6 层 × 6 头 × d=192） |
| 训练步数 | 2000 步 |
| 批大小 | 32 × 128 token |
| 训练数据 | Tiny Shakespeare（~1MB） |
| Tokenizer | 字符级（vocab=65） |

| 设备 | 耗时 | val loss（初始 → 最终） |
|------|------|------------------------|
| CPU（M 系芯片） | ~5-6 分钟 | 4.22 → 1.83 |
| MPS（Apple Silicon GPU） | ~30 秒 | 4.22 → 1.83 |

初始 loss $\approx \ln(65) \approx 4.17$（随机猜测基准）；训练后降至 1.83，说明模型已学到莎士比亚英文的字符统计规律，生成样本已有明显的"伪莎士比亚"风格。

---

## 完成后你能...

- ✅ 解释 self-attention 的每一步矩阵运算，以及为什么除以 $\sqrt{d_k}$
- ✅ 知道 AdamW + cosine warmup + grad_clip 这套训练配方中每个超参的含义
- ✅ 自己实现 BPE 训练循环和编解码器
- ✅ 看懂 nanoGPT / GPT-2 / LLaMA 的源码（架构骨架完全一致，只是规模不同）
- ✅ 理解 vLLM / PagedAttention 在解决什么问题（KV cache 内存碎片）
- ✅ 在技术对话中清晰回答"sinusoidal PE、learned PE、RoPE 有什么区别"

---

## 与项目主线的衔接

完成本章后：

- 进入 [一、LLM 基础](/llm/) 时，embedding、attention、KV cache、temperature 这些概念都不再是黑盒——你亲手实现过它们
- 阅读 LoRA / SFT / DPO 相关章节时，你清楚"LoRA 矩阵加在哪一层的线性投影上"（就是 $W^Q, W^K, W^V, W^O$）
- 遇到 `FlashAttention`、`GQA`、`MLA` 等工程优化时，你知道它们是在 KV cache 基础上的进一步演进

**本章 → 经典 ML / DL / NLP 的对应**：

| 零章知识 | 本章对应 |
|---------|---------|
| MLP（多层感知机） | Transformer Block 中的 FFN 子层 |
| Adam 优化器 | AdamW（解耦 weight decay 的改进版） |
| 交叉熵 Loss | token 级 CrossEntropyLoss，每步对 vocab 大小做 softmax |
| Dropout + 正则 | `weight_decay=0.1` + `dropout=0.1` 防止过拟合 |
| Word2Vec embedding | Token Embedding 矩阵，用 weight tying 与 lm_head 共享 |
| RNN 的序列建模 | Self-Attention 替代 RNN，并行计算且无梯度消失 |

---

## 快速上手

```bash
# 1. 安装依赖
pip install -e ".[ml]"

# 2. 下载训练语料（~1MB）
curl -L -o ml_foundations/transformer_training/data/tiny_shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 3. 运行前三个独立 demo（各 < 5 秒）
python ml_foundations/transformer_training/bpe_tokenizer.py
python ml_foundations/transformer_training/attention_from_scratch.py
python ml_foundations/transformer_training/positional_encoding.py

# 4. 主训练（生成 checkpoint，CPU ~5-6 分钟，MPS ~30 秒）
python ml_foundations/transformer_training/gpt_train.py

# 5. 需要 checkpoint 的后三个 demo
python ml_foundations/transformer_training/sampling_strategies.py
python ml_foundations/transformer_training/attention_visualization.py
python ml_foundations/transformer_training/kv_cache.py
```

**下一步**：进入 [一、LLM 基础](/llm/) 章节，或先深入阅读本章配套的 [KNOWLEDGE.md](https://github.com/datawhalechina/llm-cookbook/blob/master/ml_foundations/transformer_training/KNOWLEDGE.md)（549 行，涵盖所有公式推导和超参解析）。
