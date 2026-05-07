---
title: 什么是大语言模型
---

<script setup>
const code1 = `import numpy as np

# 模拟自注意力计算
np.random.seed(42)

# 假设 3 个 token，每个 4 维
d_k = 4
Q = np.random.randn(3, d_k)  # Query
K = np.random.randn(3, d_k)  # Key
V = np.random.randn(3, d_k)  # Value

# 步骤1：计算注意力分数
scores = Q @ K.T  # (3, 3)
print('原始分数 QK^T:')
print(np.round(scores, 3))

# 步骤2：缩放
scaled = scores / np.sqrt(d_k)
print(f'\\n缩放后 (÷√{d_k}):')
print(np.round(scaled, 3))

# 步骤3：Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

weights = softmax(scaled)
print('\\n注意力权重 (softmax):')
print(np.round(weights, 3))
print('每行和为1:', np.round(weights.sum(axis=1), 3))

# 步骤4：加权求和
output = weights @ V
print(f'\\n输出形状: {output.shape}')
print('输出值:')
print(np.round(output, 3))
`

const code2 = `import numpy as np

# 模拟 Temperature 对生成概率的影响
vocab = ['好', '不错', '热', '冷', '晴']
logits = np.array([2.0, 1.5, 0.8, 0.3, 0.1])  # 模型原始输出

def softmax_with_temperature(logits, T):
    scaled = logits / T
    exp_x = np.exp(scaled - np.max(scaled))
    return exp_x / exp_x.sum()

print('不同 Temperature 下的概率分布：')
print(f'{"词":>4s}  T=0.3(保守)  T=1.0(正常)  T=2.0(创意)')
print('-' * 50)
for temps in [0.3, 1.0, 2.0]:
    probs = softmax_with_temperature(logits, temps)
    if temps == 0.3:
        p03 = probs
    elif temps == 1.0:
        p10 = probs
    else:
        p20 = probs

for i, w in enumerate(vocab):
    print(f'{w:>4s}  {p03[i]:.3f}        {p10[i]:.3f}        {p20[i]:.3f}')

print()
print('T=0.3: 几乎只选"好" → 确定性强')
print('T=2.0: 概率更均匀 → 各词都有机会')
`
</script>

# 什么是大语言模型（LLM）

## 1. 定义

**大语言模型（Large Language Model, LLM）** 是一种基于深度学习的自然语言处理模型，它通过在海量文本数据上训练，学会了"预测下一个词"的能力。

```
核心本质：LLM 是一个极其复杂的"下一个词预测器"

输入：  "今天天气真"
输出概率：好(0.35) 不错(0.25) 热(0.15) 冷(0.10) ...

一个字一个字地预测，串起来就成了连贯的文章。
```

## 2. "大"在哪里

| 维度 | 小模型 | 大语言模型 |
|------|--------|-----------|
| 参数量 | 数百万 ~ 数亿 | 数十亿 ~ 数万亿 |
| 训练数据 | GB 级 | TB 级（万亿 token） |
| 训练算力 | 单卡几天 | 数千 GPU 训练数月 |
| 涌现能力 | 无 | 推理、编程、翻译… |

## 3. 涌现能力

当模型规模达到一定阈值后，会突然"涌现"出小模型不具备的能力：

```
  能力
  ↑
  │                    ╭─── 涌现！
  │                   ╱
  │      ────────────╱
  │     /
  │────/
  │
  └──────────────────────→ 模型参数量
       1B   10B   100B
```

涌现能力包括：逻辑推理、代码生成、多语言翻译、指令理解、少样本学习。

## 4. Transformer 架构

所有现代 LLM 都基于 **Transformer** 架构（2017 年 Google 提出）：

```
输入文本: "大语言模型很强大"

     ┌─────────────────────────────┐
     │      Tokenization（分词）    │
     └──────────────┬──────────────┘
                    ↓
     ┌─────────────────────────────┐
     │    Token Embedding（词嵌入） │
     └──────────────┬──────────────┘
                    ↓
     ┌─────────────────────────────┐
     │  Positional Encoding（位置） │
     └──────────────┬──────────────┘
                    ↓
     ┌─────────────────────────────┐
     │  Transformer Block × N 层   │
     │  ┌───────────────────────┐  │
     │  │ Multi-Head Attention  │  │
     │  │ + Feed-Forward Network│  │
     │  │ + Layer Norm          │  │
     │  │ + Residual Connection │  │
     │  └───────────────────────┘  │
     └──────────────┬──────────────┘
                    ↓
     ┌─────────────────────────────┐
     │     Output: 下一个词概率     │
     └─────────────────────────────┘
```

## 5. 自注意力机制

Transformer 的核心创新——让每个词"看"到所有其他词：

```
句子："小猫追着球跑，它很开心"

"它" 关注的注意力分数：
  小猫: 0.45  ← 最相关！
  追着: 0.05
  球:   0.25
  跑:   0.05
  很:   0.05
  开心: 0.15
```

数学公式：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

<PythonRunner :browser-runnable="true" :code="code1" />

## 6. 文本生成机制

LLM 通过自回归方式逐词生成文本。每一步的"选词"有多种策略：

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| Greedy | 选概率最高的词 | 代码生成 |
| Top-K | 从 top-K 中采样 | 通用对话 |
| Top-P | 从累积概率达 P 的词集采样 | 创意写作 |
| Temperature | 控制概率分布的尖锐程度 | 调节创造力 |

<PythonRunner :browser-runnable="true" :code="code2" />

## 7. 主流模型

| 模型 | 开发者 | 特点 |
|------|--------|------|
| GPT-4o | OpenAI | 综合能力最强 |
| Claude 4 | Anthropic | 安全对齐、长上下文 |
| Llama 3 | Meta | 开源标杆 |
| Qwen 2.5 | 阿里 | 中文能力突出 |
| DeepSeek V3 | DeepSeek | 开源 MoE |

---

::: tip 下一步
- [提示工程](/llm/prompt-engineering) — 掌握与 LLM 高效对话的技巧
- [Tokenization](/llm/tokenization) — 理解文本如何变成数字
- [文本生成机制](/llm/generation) — 深入各种解码策略
:::
