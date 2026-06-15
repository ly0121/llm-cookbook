---
title: Transformer 架构详解
---

<script setup>
const code1 = `import numpy as np

np.random.seed(42)

# ========================================
# 自注意力机制完整计算过程
# ========================================
print("=== 自注意力机制 (Self-Attention) 完整计算 ===\\n")

# 模拟 4 个 token，每个 8 维嵌入
seq_len = 4
d_model = 8
tokens = ['我', '爱', '自然', '语言']

# 随机初始化 token embedding
X = np.random.randn(seq_len, d_model)
print(f"输入矩阵 X: shape = ({seq_len}, {d_model})")
print(f"  含义: {seq_len} 个 token, 每个 {d_model} 维向量\\n")

# 初始化 Q, K, V 投影矩阵
d_k = 4  # Q, K 的维度
d_v = 4  # V 的维度
W_Q = np.random.randn(d_model, d_k) * 0.5
W_K = np.random.randn(d_model, d_k) * 0.5
W_V = np.random.randn(d_model, d_v) * 0.5

# 步骤 1: 线性投影得到 Q, K, V
Q = X @ W_Q  # (4, 4)
K = X @ W_K  # (4, 4)
V = X @ W_V  # (4, 4)
print("步骤 1: 线性投影")
print(f"  Q = X @ W_Q, shape = {Q.shape}")
print(f"  K = X @ W_K, shape = {K.shape}")
print(f"  V = X @ W_V, shape = {V.shape}\\n")

# 步骤 2: 计算注意力分数
scores = Q @ K.T  # (4, 4)
print("步骤 2: 计算注意力分数 QK^T")
print(f"  scores shape = {scores.shape} (每个 token 对其他所有 token 的分数)")
print(f"  scores =")
for i, t in enumerate(tokens):
    row = '  '.join([f'{s:+.2f}' for s in scores[i]])
    print(f"    {t:4s}: [{row}]")

# 步骤 3: 缩放 (除以 sqrt(d_k))
scaled_scores = scores / np.sqrt(d_k)
print(f"\\n步骤 3: 缩放 (除以 sqrt({d_k}) = {np.sqrt(d_k):.2f})")
print(f"  目的: 防止内积过大导致 softmax 梯度消失")

# 步骤 4: 因果掩码 (Causal Mask) — GPT 风格
print(f"\\n步骤 4: 应用因果掩码 (Causal Mask)")
mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
masked_scores = scaled_scores + mask
print("  掩码矩阵 (1=可见, 0=遮挡):")
visibility = 1 - np.triu(np.ones((seq_len, seq_len)), k=1)
for i, t in enumerate(tokens):
    row = '  '.join([f'{int(v)}' for v in visibility[i]])
    print(f"    {t:4s}: [{row}]")
print(f"  含义: 每个 token 只能看到自己和之前的 token")

# 步骤 5: Softmax 归一化
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(masked_scores)
print(f"\\n步骤 5: Softmax 归一化 (每行和为 1)")
print("  注意力权重:")
for i, t in enumerate(tokens):
    row = '  '.join([f'{w:.3f}' for w in attention_weights[i]])
    print(f"    {t:4s}: [{row}]")

# 步骤 6: 加权求和
output = attention_weights @ V  # (4, 4)
print(f"\\n步骤 6: 加权求和 output = weights @ V")
print(f"  output shape = {output.shape}")
print(f"  每个 token 的输出是 V 的加权组合\\n")

# 完整公式总结
print("=" * 50)
print("完整公式: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k) + mask) @ V")
print("=" * 50)
`

const code2 = `import numpy as np

np.random.seed(42)

# ========================================
# 多头注意力 + RoPE 位置编码演示
# ========================================
print("=== 多头注意力 (Multi-Head Attention) ===\\n")

seq_len = 4
d_model = 16
n_heads = 4
d_k = d_model // n_heads  # 每个头的维度 = 4

tokens = ['Transformer', '是', '强大的', '架构']
X = np.random.randn(seq_len, d_model)

print(f"配置: d_model={d_model}, n_heads={n_heads}, d_k={d_k}")
print(f"含义: 把 {d_model} 维拆成 {n_heads} 个头, 每头 {d_k} 维\\n")

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def single_head_attention(X, W_Q, W_K, W_V, d_k):
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    scores = Q @ K.T / np.sqrt(d_k)
    # 因果掩码
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
    weights = softmax(scores + mask)
    return weights @ V, weights

# 模拟多头: 每个头有独立的 W_Q, W_K, W_V
head_outputs = []
print("各头关注的模式 (第一个 token 对各 token 的注意力):")
print(f"{'Token':<12}", end='')
for t in tokens:
    print(f"{t:<10}", end='')
print()
print("-" * 52)

for h in range(n_heads):
    W_Q = np.random.randn(d_model, d_k) * 0.3
    W_K = np.random.randn(d_model, d_k) * 0.3
    W_V = np.random.randn(d_model, d_k) * 0.3
    out, weights = single_head_attention(X, W_Q, W_K, W_V, d_k)
    head_outputs.append(out)

    # 展示最后一个 token 的注意力分布
    last_weights = weights[-1]
    print(f"Head {h+1}:     ", end='')
    for w in last_weights:
        bar = '█' * int(w * 20)
        print(f"{w:.2f}{bar:<6}", end='')
    print()

# 拼接所有头
concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)
W_O = np.random.randn(d_model, d_model) * 0.3
multi_head_output = concat @ W_O

print(f"\\n拼接后: shape = {concat.shape}")
print(f"输出投影后: shape = {multi_head_output.shape}")
print(f"  公式: MultiHead(Q,K,V) = Concat(head1,...,headN) @ W_O")

# ========================================
# RoPE 旋转位置编码
# ========================================
print("\\n\\n=== RoPE 旋转位置编码 ===\\n")

def rope_rotation(x, pos, d):
    """对向量 x 在位置 pos 应用 RoPE"""
    result = np.zeros_like(x)
    for i in range(0, d, 2):
        theta = 1.0 / (10000.0 ** (i / d))
        cos_val = np.cos(pos * theta)
        sin_val = np.sin(pos * theta)
        result[i] = x[i] * cos_val - x[i+1] * sin_val
        result[i+1] = x[i] * sin_val + x[i+1] * cos_val
    return result

# 演示: 同一个向量在不同位置的旋转效果
d = 8
x = np.ones(d)  # 简单向量
print(f"原始向量: {np.round(x, 3)}")
print(f"\\n不同位置的 RoPE 旋转效果:")
for pos in [0, 1, 5, 10, 100]:
    rotated = rope_rotation(x, pos, d)
    print(f"  pos={pos:>3d}: {np.round(rotated, 3)}")

# 验证 RoPE 的关键性质: 内积只取决于相对位置
print("\\nRoPE 核心性质验证: 内积只取决于相对位置差")
q = np.random.randn(d)
k = np.random.randn(d)

# 位置 (3, 5) 相对距离 = 2
q_pos3 = rope_rotation(q, 3, d)
k_pos5 = rope_rotation(k, 5, d)
dot_3_5 = np.dot(q_pos3, k_pos5)

# 位置 (10, 12) 相对距离也 = 2
q_pos10 = rope_rotation(q, 10, d)
k_pos12 = rope_rotation(k, 12, d)
dot_10_12 = np.dot(q_pos10, k_pos12)

print(f"  pos(3,5)  内积 = {dot_3_5:.4f}  (相对距离=2)")
print(f"  pos(10,12) 内积 = {dot_10_12:.4f}  (相对距离=2)")
print(f"  差异: {abs(dot_3_5 - dot_10_12):.6f} (几乎相同!)")
print(f"\\n  结论: RoPE 让注意力天然感知相对位置,而非绝对位置")
`

const code3 = `import numpy as np

np.random.seed(42)

# ========================================
# Transformer Block 完整前向传播
# ========================================
print("=== Transformer Block 完整前向传播 ===\\n")

seq_len = 3
d_model = 8
n_heads = 2
d_ff = 32  # FFN 中间层维度 (通常 4 * d_model)

tokens = ['大', '模型', '强']
X = np.random.randn(seq_len, d_model)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

# ─── RMSNorm (现代 LLM 用 RMSNorm 替代 LayerNorm) ───
def rms_norm(x, weight, eps=1e-6):
    """RMSNorm: 只做缩放，不减均值"""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight

# ─── Multi-Head Attention ───
def multi_head_attention(x, W_QKV, W_O, n_heads):
    seq_len, d_model = x.shape
    d_k = d_model // n_heads

    # 一次性投影得到 Q, K, V
    qkv = x @ W_QKV  # (seq_len, 3*d_model)
    Q, K, V = np.split(qkv, 3, axis=-1)

    # 拆分为多头
    Q = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)  # (n_heads, seq, d_k)
    K = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    # 注意力计算
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
    weights = softmax(scores + mask)
    attn_out = weights @ V  # (n_heads, seq, d_k)

    # 合并多头
    attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, d_model)
    return attn_out @ W_O

# ─── SwiGLU FFN (现代 LLM 的标配) ───
def swiglu_ffn(x, W_gate, W_up, W_down):
    """SwiGLU: gate * up, 然后 down projection"""
    gate = x @ W_gate  # 门控
    up = x @ W_up      # 上投影
    # SiLU 激活: x * sigmoid(x)
    silu_gate = gate * (1 / (1 + np.exp(-gate)))
    hidden = silu_gate * up  # 逐元素相乘
    return hidden @ W_down

# ─── 初始化参数 ───
norm1_weight = np.ones(d_model)
norm2_weight = np.ones(d_model)
W_QKV = np.random.randn(d_model, 3 * d_model) * 0.1
W_O = np.random.randn(d_model, d_model) * 0.1
W_gate = np.random.randn(d_model, d_ff) * 0.1
W_up = np.random.randn(d_model, d_ff) * 0.1
W_down = np.random.randn(d_ff, d_model) * 0.1

# ─── 前向传播 ───
print("输入 X: shape =", X.shape)
print(f"  ({seq_len} tokens, {d_model} dims)\\n")

# 第一个子层: RMSNorm + Multi-Head Attention + 残差
print("── 子层 1: 注意力 ──")
x_norm1 = rms_norm(X, norm1_weight)
print(f"  1. RMSNorm(X)         shape = {x_norm1.shape}")
attn_out = multi_head_attention(x_norm1, W_QKV, W_O, n_heads)
print(f"  2. MultiHeadAttn(...)  shape = {attn_out.shape}")
X_after_attn = X + attn_out  # 残差连接
print(f"  3. 残差连接: X + Attn  shape = {X_after_attn.shape}")
print(f"     残差连接的作用: 梯度可以直接回传,避免梯度消失\\n")

# 第二个子层: RMSNorm + SwiGLU FFN + 残差
print("── 子层 2: 前馈网络 ──")
x_norm2 = rms_norm(X_after_attn, norm2_weight)
print(f"  4. RMSNorm(X)         shape = {x_norm2.shape}")
ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)
print(f"  5. SwiGLU_FFN(...)     shape = {ffn_out.shape}")
print(f"     FFN 中间维度: {d_model} -> {d_ff} -> {d_model}")
X_final = X_after_attn + ffn_out  # 残差连接
print(f"  6. 残差连接: X + FFN   shape = {X_final.shape}\\n")

# 信息流总结
print("=" * 55)
print("Transformer Block 完整信息流:")
print("=" * 55)
print(f"""
  X (输入)
  │
  ├──────────────────────┐
  ↓                      │
  RMSNorm                │ (Pre-Norm)
  ↓                      │
  Multi-Head Attention   │ (信息交换: token间互相看)
  ↓                      │
  + ←────────────────────┘ (残差连接)
  │
  ├──────────────────────┐
  ↓                      │
  RMSNorm                │ (Pre-Norm)
  ↓                      │
  SwiGLU FFN             │ (信息处理: 非线性变换)
  ↓                      │
  + ←────────────────────┘ (残差连接)
  │
  ↓
  输出 (传入下一层)
""")

# 参数量估算
print("── 参数量估算 (以 LLaMA-7B 为例) ──")
D = 4096  # d_model
H = 32    # n_heads
FF = 11008  # d_ff (SwiGLU 用 8/3 * d_model 取整)
L = 32    # 层数
V = 32000  # 词表大小

attn_params = 4 * D * D  # W_Q, W_K, W_V, W_O
ffn_params = 3 * D * FF   # W_gate, W_up, W_down
norm_params = 2 * D        # 两个 RMSNorm
block_params = attn_params + ffn_params + norm_params
total_params = L * block_params + V * D + D  # +embedding +final norm

print(f"  d_model = {D}, n_heads = {H}, d_ff = {FF}, layers = {L}")
print(f"  注意力层参数:  4 * {D}^2 = {attn_params:,}")
print(f"  FFN 层参数:    3 * {D} * {FF} = {ffn_params:,}")
print(f"  每层总参数:    {block_params:,}")
print(f"  {L} 层总参数:  {L * block_params:,}")
print(f"  词嵌入:        {V} * {D} = {V * D:,}")
print(f"  ─────────────────────────────")
print(f"  模型总参数:    {total_params:,} ≈ {total_params/1e9:.1f}B")
print(f"\\n  FP16 显存:     {total_params * 2 / 1e9:.1f} GB")
print(f"  INT4 显存:     {total_params * 0.5 / 1e9:.1f} GB")
`
</script>

# Transformer 架构详解

Transformer 是所有现代大语言模型的基础架构。理解 Transformer 不仅能帮助你更好地使用 LLM，也是理解推理优化、微调、模型选型的前提。

## 从 RNN 到 Transformer

在 Transformer 之前，处理序列的主流是 RNN/LSTM：

```
RNN 的困境：
  h₀ → [我] → h₁ → [爱] → h₂ → [自然] → h₃ → [语言] → h₄ → [处理] → h₅
        ↓          ↓           ↓            ↓            ↓

问题：
  1. 串行计算：必须一个字一个字处理 → 慢！无法利用 GPU 并行
  2. 长距离遗忘：h₅ 几乎"忘了" h₁ 的信息
  3. 梯度消失：反向传播经过多步后梯度趋近于零

Transformer 的革命（2017 年 Google "Attention Is All You Need"）：
  "为什么要一个字一个字看？直接让每个字同时看所有其他字！"

         我  爱  自然  语言  处理
  我  [  ✓   ✓    ✓     ✓     ✓  ]
  爱  [  ✓   ✓    ✓     ✓     ✓  ]  ← 每个词同时关注所有词
  自然 [  ✓   ✓    ✓     ✓     ✓  ]    完全并行，GPU 友好！
  语言 [  ✓   ✓    ✓     ✓     ✓  ]
  处理 [  ✓   ✓    ✓     ✓     ✓  ]
```

## Transformer 的三种变体

```
原始 Transformer = Encoder + Decoder

  ┌──────────┐              ┌──────────────┐    ┌──────────────┐
  │ 仅编码器  │              │  仅解码器     │    │ 编码器-解码器 │
  │  BERT    │              │  GPT 系列    │    │    T5        │
  │  理解任务 │              │  生成任务     │    │  翻译/摘要   │
  └──────────┘              └──────────────┘    └──────────────┘
  分类、NER、                文本生成、对话        Seq2Seq 任务
  句子相似度                 (当今主流 LLM)

当今主流 LLM（GPT、Claude、Llama、Qwen）都是"仅解码器"架构
```

## 自注意力机制（Self-Attention）

### 直觉理解

```
句子："小猫追着球跑，它很开心"

问题："它"指的是谁？

自注意力让"它"去"看"句子中的每个词，计算相关程度：

  "它" 的注意力分数：
    小猫: 0.45  ← 最相关！模型理解"它"指代"小猫"
    追着: 0.05
    球:   0.25  ← 也有一定相关性
    跑:   0.05
    很:   0.05
    开心: 0.15

这就是注意力的本质：让每个词"决定"该关注哪些其他词。
```

### 数学公式

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V

其中：
  Q (Query)  = X · W_Q    "我在找什么信息？"
  K (Key)    = X · W_K    "我有什么信息可以提供？"
  V (Value)  = X · W_V    "我实际携带的信息内容"
  d_k        = Key 的维度  用于缩放，防止内积过大

计算过程分解：

  步骤1: scores = Q · K^T           → 形状 (seq_len × seq_len)
  步骤2: scaled = scores / √d_k     → 防止 softmax 饱和
  步骤3: weights = softmax(scaled)   → 每行和为 1
  步骤4: output = weights · V        → 加权组合
```

**为什么要除以 √d_k？**

```
Q 和 K 元素为标准正态分布时：
  Q·K^T 每个元素的方差 = d_k

  d_k=128 时，内积值很大 → softmax 变成 one-hot → 梯度为 0
  除以 √d_k 后，方差回到 1，softmax 行为正常
```

### 交互示例：自注意力计算

<PythonRunner :browser-runnable="true" :code="code1" />

## 多头注意力（Multi-Head Attention）

一个注意力头只能学一种"关注模式"，但语言理解需要多种角度：

```
头1: 关注语法关系（主语-谓语）
头2: 关注指代关系（代词-实体）
头3: 关注修饰关系（形容词-名词）
头4: 关注位置邻近的词
...

实现：把 d_model 拆分成 h 个头，每头独立计算注意力

  MultiHead(Q, K, V) = Concat(head₁, head₂, ..., head_h) · W_O

  其中 head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

┌───────────────────────────────────────────────────┐
│  输入 X (seq_len × d_model)                        │
│        ↓                                          │
│  ┌────────┬────────┬────────┬────────┐            │
│  │ 头 1   │ 头 2   │ 头 3   │...头 h │            │
│  │(d_model/│(d_model/│(d_model/│(d_model/│           │
│  │   h)   │   h)   │   h)   │   h)   │            │
│  └───┬────┴───┬────┴───┬────┴───┬────┘            │
│      └────────┴────────┴────────┘                  │
│                   ↓ Concat + W_O                   │
│         (seq_len × d_model)                        │
└───────────────────────────────────────────────────┘

典型配置：
  LLaMA-7B:   32 头, d_model=4096,  d_k=128
  LLaMA-70B:  64 头, d_model=8192,  d_k=128
  GPT-3:      96 头, d_model=12288, d_k=128
```

## 因果掩码（Causal Mask）

GPT 等自回归模型中，第 i 个 token 只能看到前面的 token：

```
掩码矩阵（1=可见，0=遮挡）：

         我  爱  自然  语言  处理
  我  [  1   0    0     0     0  ]  ← 只能看到自己
  爱  [  1   1    0     0     0  ]  ← 看到"我"和自己
  自然 [  1   1    1     0     0  ]
  语言 [  1   1    1     1     0  ]
  处理 [  1   1    1     1     1  ]  ← 看到所有前面的词

实现：在 softmax 之前，将未来位置的分数设为 -∞
  masked_scores = scores + mask   (mask 中未来位置为 -∞)
  softmax(-∞) = 0  → 完全不关注未来的词
```

<PythonRunner :browser-runnable="true" :code="code2" />

## 位置编码（Positional Encoding）

Attention 本身不知道词的顺序（"我爱你"和"你爱我"的注意力一样），需要位置编码告诉模型"这是第几个词"。

### 正弦位置编码（原始 Transformer）

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

特点：
  - 任意两个位置的编码都不同
  - 可以（理论上）外推到训练时未见过的更长序列
  - 缺点：无法很好地表达相对位置
```

### RoPE 旋转位置编码（现代 LLM 主流）

```
核心思想：用旋转矩阵编码相对位置

  把向量的每两个维度看成一个复数对，按位置旋转不同角度：
  f(x, pos) = x · e^(i·pos·θ)

关键性质：
  两个位置编码后的内积，只取决于相对距离！
  即 <RoPE(q, pos_i), RoPE(k, pos_j)> 只与 (pos_i - pos_j) 有关

优势：
  - 天然编码相对位置
  - 远距离位置信息自然衰减
  - 通过 NTK-aware 缩放可扩展上下文长度

使用 RoPE 的模型：Llama、Qwen、Mistral、DeepSeek
```

## Feed-Forward Network（前馈网络）

每个 Transformer Block 中，注意力层后面跟着一个 FFN：

```
注意力层的作用: "信息交换"（token 间互相看）
FFN 的作用:     "信息处理"（每个 token 独立做非线性变换）

标准 FFN：
  FFN(x) = W₂ · GELU(W₁ · x)
  维度变化: d_model → 4*d_model → d_model （先扩展再压缩）

SwiGLU FFN（现代 LLM 标配，Llama/Qwen/Mistral 都用）：
  FFN(x) = W_down · (SiLU(W_gate · x) ⊙ W_up · x)

  SiLU(x) = x · sigmoid(x)
  ⊙ 是逐元素乘法（门控机制）

  SwiGLU 比标准 FFN 效果更好（参数多 50% 但精度提升明显）

重要发现：LLM 的"知识"主要存储在 FFN 的参数中
```

## Layer Normalization 与残差连接

### RMSNorm（现代做法）

```
RMSNorm(x) = x / RMS(x) · γ

  RMS(x) = √(mean(x²))

比 LayerNorm 更简单：去掉了减均值的步骤
计算更快，效果相当
Llama、Qwen、Mistral 等都用 RMSNorm
```

### Pre-Norm vs Post-Norm

```
Post-Norm（原始论文）:  LayerNorm(x + Attention(x))
Pre-Norm（现代做法）:   x + Attention(LayerNorm(x))

Pre-Norm 训练更稳定，几乎所有现代 LLM 都用 Pre-Norm
```

### 残差连接的重要性

```
没有残差的深层网络：
  x → F₁ → F₂ → ... → F₉₆ → output
  梯度要经过 96 次链式法则，可能变成 0

有残差的深层网络：
  x ──→ (+) ──→ (+) ──→ ... ──→ output
     ↘  ↗    ↘  ↗
     F₁      F₂

  梯度可以直接通过"高速公路"回传！
  每一层不是"改变"信息，而是"添加"新信息：
  output = 原始 + 第1层学到的 + 第2层学到的 + ...
```

## Transformer Block 完整结构

<PythonRunner :browser-runnable="true" :code="code3" />

## KV Cache（推理加速核心）

```
自回归生成过程中，KV Cache 避免重复计算：

朴素方法（每步重新计算所有 attention）：
  步1: 计算 "今" 的 Q,K,V → 生成 "天"
  步2: 计算 "今天" 的 Q,K,V → 生成 "天"     ← "今"重复计算了！
  步3: 计算 "今天天" 的 Q,K,V → 生成 "气"   ← "今天"重复计算了！

KV Cache（缓存已计算的 K,V）：
  步1: 计算 "今" 的 Q,K,V → 缓存 K₁,V₁ → 生成 "天"
  步2: 只计算 "天" 的 Q  → 拼接 [K₁,K₂], [V₁,V₂] → 生成 "天"
  步3: 只计算 "天" 的 Q  → 拼接 [K₁₂₃], [V₁₂₃] → 生成 "气"

  计算量从 O(n²) 降为 O(n) per step！

KV Cache 内存占用（LLaMA-7B, FP16）：
  单 token: 2 × 32层 × 32头 × 128维 × 2字节 = 0.5 MB
  2048 序列: 0.5MB × 2048 = 1 GB / 请求
  batch=32: 32 GB（比模型本身还大！）

  → 这就是为什么长上下文推理需要 PagedAttention 等技术
```

## MoE（Mixture of Experts）架构

```
传统 Transformer：每个 token 经过完整的 FFN（所有参数）
MoE Transformer：每个 token 只激活部分"专家"

┌────────────────────────────────────────────────┐
│  传统 FFN (Dense):                              │
│    token → [完整 FFN, 所有参数参与] → output     │
│    计算量: 100%                                  │
│                                                │
│  MoE FFN (Sparse):                              │
│    token → Router → 选择 Top-2 专家              │
│         ┌─────────────────────────────┐        │
│         │ Expert 1  Expert 2  ...  Expert 8│     │
│         │   (选中)    (选中)          │     │
│         └─────────────────────────────┘        │
│    → 加权组合选中专家的输出                      │
│    计算量: ~25% (只用 2/8 个专家)                │
└────────────────────────────────────────────────┘

代表模型：
  Mixtral 8×7B:  8 个 7B 专家，每次激活 2 个
                 总参数 47B，激活参数 13B
  DeepSeek V3:   总参数 671B，激活参数 37B
                 → 用超大模型的知识量，小模型的推理成本！

Router (路由器) 的训练挑战：
  - 负载均衡: 避免所有 token 都选同一个专家
  - 辅助损失: 加入 load balancing loss 鼓励均匀使用
```

## 参数量与显存估算

| 模型 | 层数 | d_model | 头数 | 总参数 | FP16 显存 | INT4 显存 |
|------|------|---------|------|--------|----------|----------|
| LLaMA-7B | 32 | 4096 | 32 | 6.7B | 13.4 GB | 3.4 GB |
| LLaMA-13B | 40 | 5120 | 40 | 13B | 26 GB | 6.5 GB |
| LLaMA-70B | 80 | 8192 | 64 | 70B | 140 GB | 35 GB |
| GPT-3 | 96 | 12288 | 96 | 175B | 350 GB | 88 GB |

::: info 记忆公式
每层参数 ≈ 12 × d_model²（注意力 4d² + FFN 约 8d²）

模型总参数 ≈ 12 × d_model² × n_layers

FP16 显存 ≈ 参数量 × 2 bytes

INT4 显存 ≈ 参数量 × 0.5 bytes
:::

## 总结：现代 LLM 的 Transformer 配方

```
现代 LLM (Llama 3 / Qwen 2.5 / DeepSeek V3) 的标准配方：

  ┌─────────────────────────────────────────────┐
  │  位置编码:    RoPE (旋转位置编码)              │
  │  归一化:      RMSNorm (Pre-Norm)             │
  │  注意力:      Grouped-Query Attention (GQA)  │
  │  FFN:         SwiGLU                         │
  │  激活函数:    SiLU (x·sigmoid(x))            │
  │  词表:        BPE, 100K~150K                 │
  │  上下文:      128K tokens                    │
  │  训练数据:    10T~15T tokens                 │
  └─────────────────────────────────────────────┘

  GQA (Grouped-Query Attention):
    - 标准 MHA: Q/K/V 各有 h 组
    - GQA: Q 有 h 组，K/V 只有 h/g 组（多个 Q 头共享 K/V）
    - 减少 KV Cache 大小，加速推理
    - Llama 2 70B: 64 Q头, 8 KV头 (每8个Q共享1组KV)
```

---

::: tip 下一步
- [Embedding 词向量](/llm/embedding) — 理解文本如何变成向量
- [推理部署与加速](/llm/inference) — KV Cache、PagedAttention 等优化技术
- [进阶方向与前沿技术](/llm/advanced-topics) — LoRA、MoE、多模态融合
:::
