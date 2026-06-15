"""
╔══════════════════════════════════════════════════════════════════╗
║         项目：Transformer 架构原理完整实现与演示                  ║
║         从自注意力到完整 Transformer Block 的逐步推导             ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题：Transformer 是如何工作的？】
═══════════════════════════════════════════════════════════════════

Transformer 是所有现代大语言模型（GPT、Claude、Llama、Qwen）的基础架构。
2017 年 Google 的论文 "Attention Is All You Need" 提出了这一革命性架构，
彻底取代了 RNN/LSTM，成为 NLP 的基石。

  核心创新：
    1. 自注意力机制 — 让每个 token 直接"看到"所有其他 token
    2. 完全并行计算 — 不像 RNN 必须串行处理序列
    3. 位置编码 — 用数学方法告诉模型 token 的位置信息

  ┌─────────────────────────────────────────────────────────────┐
  │  Transformer Block 信息流：                                   │
  │                                                             │
  │  输入 X                                                     │
  │    ↓                                                        │
  │  RMSNorm → Multi-Head Attention → 残差连接                  │
  │    ↓                                                        │
  │  RMSNorm → SwiGLU FFN → 残差连接                            │
  │    ↓                                                        │
  │  输出（传入下一层）                                           │
  │                                                             │
  │  现代 LLM 就是堆叠 32~96 层这样的 Block！                    │
  └─────────────────────────────────────────────────────────────┘

本文件通过纯 NumPy 实现 Transformer 的每个核心组件，
让你亲手体验数据在 Transformer 中的流动过程。
"""

import numpy as np

np.random.seed(42)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：自注意力机制（Self-Attention）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 自注意力的核心思想：
#   让每个 token "决定"该关注序列中的哪些其他 token。
#
#   例如："小猫追着球跑，它很开心"
#   "它" 需要知道自己指代的是 "小猫" 而非 "球"
#   自注意力通过 Q·K^T 计算每对 token 间的相关度
#
# 公式：
#   Attention(Q, K, V) = softmax(QK^T / √d_k) · V
#
#   Q = X · W_Q  (Query: "我在找什么信息？")
#   K = X · W_K  (Key:   "我有什么信息可以提供？")
#   V = X · W_V  (Value: "我实际携带的信息内容")

print("=" * 60)
print("第 1 章：自注意力机制 (Self-Attention) 完整计算")
print("=" * 60)
print()


def softmax(x):
    """数值稳定的 softmax 实现"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


# ── 1.1 基本设置 ─────────────────────────────────────────
print("── 1.1 输入准备 ──────────────────────────────────────")
print()

seq_len = 4
d_model = 8
tokens = ['我', '爱', '自然', '语言']

# 随机初始化 token embedding（实际中由词表查找得到）
X = np.random.randn(seq_len, d_model)
print(f"  输入序列: {tokens}")
print(f"  输入矩阵 X: shape = ({seq_len}, {d_model})")
print(f"  含义: {seq_len} 个 token, 每个用 {d_model} 维向量表示")
print()

# ── 1.2 线性投影 ─────────────────────────────────────────
print("── 1.2 线性投影得到 Q, K, V ──────────────────────────")
print()

d_k = 4  # Q, K 的维度
d_v = 4  # V 的维度

# 投影矩阵（实际中这些是通过训练学到的参数）
W_Q = np.random.randn(d_model, d_k) * 0.5
W_K = np.random.randn(d_model, d_k) * 0.5
W_V = np.random.randn(d_model, d_v) * 0.5

# 线性投影
Q = X @ W_Q  # (4, 4)
K = X @ W_K  # (4, 4)
V = X @ W_V  # (4, 4)

print(f"  W_Q: ({d_model}, {d_k})  — 将输入投影到 Query 空间")
print(f"  W_K: ({d_model}, {d_k})  — 将输入投影到 Key 空间")
print(f"  W_V: ({d_model}, {d_v})  — 将输入投影到 Value 空间")
print()
print(f"  Q = X @ W_Q, shape = {Q.shape}")
print(f"  K = X @ W_K, shape = {K.shape}")
print(f"  V = X @ W_V, shape = {V.shape}")
print()
print("  直觉理解：")
print("    Q (Query)  = '我在找什么信息？'")
print("    K (Key)    = '我有什么信息可以提供？'")
print("    V (Value)  = '我实际携带的信息内容'")
print()

# ── 1.3 注意力分数计算 ───────────────────────────────────
print("── 1.3 计算注意力分数 QK^T ───────────────────────────")
print()

scores = Q @ K.T  # (4, 4)
print(f"  scores = Q @ K^T, shape = {scores.shape}")
print(f"  scores[i][j] = token_i 的 Query 与 token_j 的 Key 的点积")
print()
print("  注意力分数矩阵：")
print(f"  {'':8s}", end='')
for t in tokens:
    print(f"{t:>8s}", end='')
print()
for i, t in enumerate(tokens):
    row = ''.join([f'{s:+8.2f}' for s in scores[i]])
    print(f"  {t:4s}  {row}")
print()

# ── 1.4 缩放 ────────────────────────────────────────────
print("── 1.4 缩放（除以 √d_k）─────────────────────────────")
print()

scaled_scores = scores / np.sqrt(d_k)
print(f"  scaled_scores = scores / √{d_k} = scores / {np.sqrt(d_k):.2f}")
print()
print("  为什么要缩放？")
print(f"    当 d_k={d_k} 时，Q·K^T 的方差约为 d_k={d_k}")
print(f"    如果 d_k=128（实际模型中），内积值会非常大")
print(f"    大的值进 softmax → 梯度接近 0 → 训练困难")
print(f"    除以 √d_k 让方差回到 1，softmax 行为正常")
print()

# ── 1.5 因果掩码 ─────────────────────────────────────────
print("── 1.5 因果掩码（Causal Mask）—— GPT 风格 ────────────")
print()

mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
masked_scores = scaled_scores + mask

print("  GPT 等自回归模型：第 i 个 token 只能看到前面的 token")
print()
print("  掩码矩阵 (1=可见, 0=遮挡)：")
visibility = 1 - np.triu(np.ones((seq_len, seq_len)), k=1)
print(f"  {'':8s}", end='')
for t in tokens:
    print(f"{t:>6s}", end='')
print()
for i, t in enumerate(tokens):
    row = ''.join([f'{int(v):>6d}' for v in visibility[i]])
    print(f"  {t:4s}  {row}")
print()
print("  实现方式：将未来位置的分数设为 -∞")
print("  softmax(-∞) = 0 → 完全不关注未来的词")
print()

# ── 1.6 Softmax 归一化 ──────────────────────────────────
print("── 1.6 Softmax 归一化 ────────────────────────────────")
print()

attention_weights = softmax(masked_scores)
print("  attention_weights = softmax(masked_scores)")
print("  每行和为 1，表示概率分布")
print()
print("  注意力权重矩阵：")
print(f"  {'':8s}", end='')
for t in tokens:
    print(f"{t:>8s}", end='')
print()
for i, t in enumerate(tokens):
    row = ''.join([f'{w:8.3f}' for w in attention_weights[i]])
    print(f"  {t:4s}  {row}")
print()
print("  观察：")
print(f"    '{tokens[0]}' 只能看到自己，所以权重为 [1.000, 0, 0, 0]")
print(f"    '{tokens[-1]}' 可以看到所有前面的词，权重分散")
print()

# ── 1.7 加权求和 ─────────────────────────────────────────
print("── 1.7 加权求和得到输出 ──────────────────────────────")
print()

output = attention_weights @ V  # (4, 4)
print(f"  output = attention_weights @ V")
print(f"  output shape = {output.shape}")
print()
print("  含义：每个 token 的输出 = 它能看到的所有 token 的 V 的加权组合")
print("  权重由 Q·K^T 决定（即由语义相关性决定）")
print()
print("  " + "=" * 55)
print("  完整公式: Attention(Q,K,V) = softmax(QK^T/√d_k + mask) @ V")
print("  " + "=" * 55)
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：多头注意力（Multi-Head Attention）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 一个注意力头只能学一种"关注模式"。
# 但语言理解需要同时关注多种关系：
#   头1: 语法关系（主语-谓语）
#   头2: 指代关系（代词-实体）
#   头3: 修饰关系（形容词-名词）
#   头4: 位置邻近
#
# 多头注意力：把 d_model 拆成 h 个头，每头独立计算注意力
#   MultiHead(Q,K,V) = Concat(head1, ..., headh) · W_O

print()
print("=" * 60)
print("第 2 章：多头注意力 (Multi-Head Attention)")
print("=" * 60)
print()

seq_len = 4
d_model = 16
n_heads = 4
d_k = d_model // n_heads  # 每个头的维度 = 4
tokens = ['Transformer', '是', '强大的', '架构']

X = np.random.randn(seq_len, d_model)

print(f"  配置: d_model={d_model}, n_heads={n_heads}, d_k={d_k}")
print(f"  含义: 把 {d_model} 维拆成 {n_heads} 个头, 每头 {d_k} 维")
print()


def single_head_attention(X, W_Q, W_K, W_V, d_k, seq_len):
    """单头注意力计算"""
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    scores = Q @ K.T / np.sqrt(d_k)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
    weights = softmax(scores + mask)
    return weights @ V, weights


# ── 2.1 各头独立计算 ─────────────────────────────────────
print("── 2.1 各头独立计算注意力 ────────────────────────────")
print()

head_outputs = []
print(f"  最后一个 token ('{tokens[-1]}') 对各 token 的注意力分布：")
print()
print(f"  {'Head':<10}", end='')
for t in tokens:
    print(f"{t:<12}", end='')
print()
print("  " + "-" * 58)

for h in range(n_heads):
    W_Q = np.random.randn(d_model, d_k) * 0.3
    W_K = np.random.randn(d_model, d_k) * 0.3
    W_V = np.random.randn(d_model, d_k) * 0.3
    out, weights = single_head_attention(X, W_Q, W_K, W_V, d_k, seq_len)
    head_outputs.append(out)

    # 展示最后一个 token 的注意力分布
    last_weights = weights[-1]
    print(f"  Head {h+1}:   ", end='')
    for w in last_weights:
        bar = '#' * int(w * 15)
        print(f"{w:.3f}{bar:<8}", end='')
    print()

print()
print("  观察：不同头学到不同的注意力模式")
print("  有的头更关注近邻词，有的头更关注语义相关词")
print()

# ── 2.2 拼接与输出投影 ───────────────────────────────────
print("── 2.2 拼接所有头 + 输出投影 ─────────────────────────")
print()

concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, d_model)
W_O = np.random.randn(d_model, d_model) * 0.3
multi_head_output = concat @ W_O

print(f"  各头输出: {n_heads} 个 shape=({seq_len}, {d_k}) 的矩阵")
print(f"  拼接后:   shape = {concat.shape}")
print(f"  W_O 投影: ({d_model}, {d_model})")
print(f"  最终输出: shape = {multi_head_output.shape}")
print()
print("  公式: MultiHead(Q,K,V) = Concat(head1,...,headN) @ W_O")
print()
print("  典型配置:")
print("    LLaMA-7B:   32 头, d_model=4096,  d_k=128")
print("    LLaMA-70B:  64 头, d_model=8192,  d_k=128")
print("    GPT-3:      96 头, d_model=12288, d_k=128")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：RoPE 旋转位置编码
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Attention 本身不知道词的顺序（"我爱你"和"你爱我"的注意力一样）
# 需要位置编码告诉模型"这是第几个词"
#
# RoPE 的核心思想：
#   把向量的每两个维度看成一个复数对，按位置旋转不同角度
#   关键性质：两个编码后向量的内积只取决于相对距离！
#
# 使用 RoPE 的模型：Llama、Qwen、Mistral、DeepSeek

print()
print("=" * 60)
print("第 3 章：RoPE 旋转位置编码")
print("=" * 60)
print()


def rope_rotation(x, pos, d):
    """对向量 x 在位置 pos 应用 RoPE 旋转"""
    result = np.zeros_like(x)
    for i in range(0, d, 2):
        # 每对维度使用不同频率的旋转
        theta = 1.0 / (10000.0 ** (i / d))
        cos_val = np.cos(pos * theta)
        sin_val = np.sin(pos * theta)
        # 旋转矩阵作用于 (x[i], x[i+1]) 对
        result[i] = x[i] * cos_val - x[i + 1] * sin_val
        result[i + 1] = x[i] * sin_val + x[i + 1] * cos_val
    return result


# ── 3.1 不同位置的旋转效果 ───────────────────────────────
print("── 3.1 同一向量在不同位置的旋转效果 ──────────────────")
print()

d = 8
x = np.ones(d)
print(f"  原始向量: {np.round(x, 3)}")
print()
print("  不同位置的 RoPE 旋转结果:")
for pos in [0, 1, 5, 10, 50, 100]:
    rotated = rope_rotation(x, pos, d)
    print(f"    pos={pos:>3d}: [{', '.join([f'{v:+.3f}' for v in rotated])}]")

print()
print("  观察：")
print("    pos=0 时向量不变（cos(0)=1, sin(0)=0）")
print("    位置越大，旋转角度越大，向量变化越明显")
print("    低频维度变化慢，高频维度变化快")
print()

# ── 3.2 验证 RoPE 核心性质 ───────────────────────────────
print("── 3.2 验证核心性质：内积只取决于相对位置 ─────────────")
print()

q = np.random.randn(d)
k = np.random.randn(d)

print("  设 q 和 k 为两个随机向量")
print()

# 测试多组绝对位置，但相对距离相同
print("  相对距离 = 2 的不同绝对位置对：")
pairs = [(1, 3), (5, 7), (10, 12), (50, 52), (100, 102)]
dots = []
for pos_q, pos_k in pairs:
    q_rotated = rope_rotation(q, pos_q, d)
    k_rotated = rope_rotation(k, pos_k, d)
    dot = np.dot(q_rotated, k_rotated)
    dots.append(dot)
    print(f"    pos({pos_q:>3d}, {pos_k:>3d}): 内积 = {dot:.6f}")

print()
print(f"  最大差异: {max(dots) - min(dots):.10f}")
print("  结论: 内积几乎完全相同！RoPE 让注意力天然感知相对位置")
print()

# 对比不同相对距离
print("  不同相对距离的内积（相对距离越大，相关性通常衰减）:")
for dist in [0, 1, 2, 5, 10, 50]:
    q_r = rope_rotation(q, 0, d)
    k_r = rope_rotation(k, dist, d)
    dot = np.dot(q_r, k_r)
    print(f"    相对距离={dist:>3d}: 内积 = {dot:+.6f}")
print()
print("  RoPE 的优势：")
print("    - 天然编码相对位置（不需要额外的位置嵌入矩阵）")
print("    - 远距离信息自然衰减")
print("    - 通过 NTK-aware 缩放可扩展上下文长度")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：Feed-Forward Network（前馈网络）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 注意力层的作用：信息交换（token 间互相看）
# FFN 的作用：    信息处理（每个 token 独立做非线性变换）
#
# 标准 FFN：  FFN(x) = W2 · GELU(W1 · x)
# SwiGLU FFN：FFN(x) = W_down · (SiLU(W_gate · x) ⊙ W_up · x)
#
# SwiGLU 是现代 LLM 的标配（Llama/Qwen/Mistral/DeepSeek 都用）

print()
print("=" * 60)
print("第 4 章：Feed-Forward Network (SwiGLU)")
print("=" * 60)
print()

d_model_ffn = 8
d_ff = 32  # 中间层维度，通常为 4*d_model 或 8/3*d_model

# ── 4.1 标准 FFN ─────────────────────────────────────────
print("── 4.1 标准 FFN（原始 Transformer）────────────────────")
print()


def gelu(x):
    """GELU 激活函数: x * Φ(x)"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def standard_ffn(x, W1, W2):
    """标准 FFN: W2 · GELU(W1 · x)"""
    hidden = gelu(x @ W1)  # (seq, d_ff)
    return hidden @ W2  # (seq, d_model)


W1 = np.random.randn(d_model_ffn, d_ff) * 0.1
W2 = np.random.randn(d_ff, d_model_ffn) * 0.1
x_sample = np.random.randn(1, d_model_ffn)

ffn_out = standard_ffn(x_sample, W1, W2)
print(f"  输入: shape = {x_sample.shape}")
print(f"  W1:   ({d_model_ffn}, {d_ff}) — 扩展维度")
print(f"  W2:   ({d_ff}, {d_model_ffn}) — 压缩回来")
print(f"  输出: shape = {ffn_out.shape}")
print(f"  维度变化: {d_model_ffn} → {d_ff} → {d_model_ffn} (先扩展再压缩)")
print(f"  参数量: {d_model_ffn * d_ff + d_ff * d_model_ffn:,} = 2 * d_model * d_ff")
print()

# ── 4.2 SwiGLU FFN ──────────────────────────────────────
print("── 4.2 SwiGLU FFN（现代 LLM 标配）───────────────────")
print()


def silu(x):
    """SiLU (Swish) 激活函数: x * sigmoid(x)"""
    return x * (1 / (1 + np.exp(-x)))


def swiglu_ffn(x, W_gate, W_up, W_down):
    """
    SwiGLU FFN:
      gate = SiLU(x @ W_gate)
      up = x @ W_up
      hidden = gate ⊙ up  (逐元素相乘 = 门控机制)
      output = hidden @ W_down
    """
    gate = silu(x @ W_gate)  # 门控信号
    up = x @ W_up  # 上投影
    hidden = gate * up  # 门控：决定哪些信息通过
    return hidden @ W_down  # 下投影回原始维度


W_gate = np.random.randn(d_model_ffn, d_ff) * 0.1
W_up = np.random.randn(d_model_ffn, d_ff) * 0.1
W_down = np.random.randn(d_ff, d_model_ffn) * 0.1

swiglu_out = swiglu_ffn(x_sample, W_gate, W_up, W_down)
print(f"  SwiGLU 公式: FFN(x) = (SiLU(x@W_gate) ⊙ x@W_up) @ W_down")
print()
print(f"  W_gate: ({d_model_ffn}, {d_ff}) — 门控投影")
print(f"  W_up:   ({d_model_ffn}, {d_ff}) — 上投影")
print(f"  W_down: ({d_ff}, {d_model_ffn}) — 下投影")
print(f"  输出: shape = {swiglu_out.shape}")
print(f"  参数量: {d_model_ffn * d_ff * 3:,} = 3 * d_model * d_ff (比标准 FFN 多 50%)")
print()
print("  SwiGLU vs 标准 FFN：")
print("    - 多了一个门控矩阵 W_gate，参数量多 50%")
print("    - 但精度提升明显，性价比高")
print("    - 门控机制让网络能够选择性地传递信息")
print()

# 展示 SiLU 激活函数
print("  SiLU 激活函数特性 (x * sigmoid(x)):")
test_x = np.array([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])
test_silu = silu(test_x)
for xi, yi in zip(test_x, test_silu):
    bar = '#' * max(0, int((yi + 0.5) * 8))
    print(f"    x={xi:+.1f} → SiLU(x)={yi:+.3f}  {bar}")
print()
print("  特点：允许小的负值通过（不像 ReLU 完全截断负值）")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 章：Layer Normalization 与残差连接
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# RMSNorm（现代 LLM 主流）：
#   RMSNorm(x) = x / RMS(x) * γ
#   比 LayerNorm 更简单：去掉了减均值的步骤
#
# Pre-Norm（现代做法）：
#   x + Attention(RMSNorm(x))  而非  LayerNorm(x + Attention(x))
#
# 残差连接：
#   让梯度可以直接通过"高速公路"回传，避免梯度消失

print()
print("=" * 60)
print("第 5 章：RMSNorm 与残差连接")
print("=" * 60)
print()


# ── 5.1 RMSNorm vs LayerNorm ────────────────────────────
print("── 5.1 RMSNorm vs LayerNorm 对比 ────────────────────")
print()


def layer_norm(x, gamma, beta, eps=1e-6):
    """标准 LayerNorm: 减均值 + 除标准差 + 缩放偏移"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    return gamma * normalized + beta


def rms_norm(x, gamma, eps=1e-6):
    """RMSNorm: 只除 RMS，不减均值，没有偏移"""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


d = 8
x_test = np.array([[3.0, -1.5, 2.0, 0.5, -2.0, 1.0, -0.5, 4.0]])
gamma = np.ones(d)
beta = np.zeros(d)

ln_out = layer_norm(x_test, gamma, beta)
rms_out = rms_norm(x_test, gamma)

print(f"  输入向量: {x_test[0]}")
print()
print(f"  LayerNorm 输出: [{', '.join([f'{v:.3f}' for v in ln_out[0]])}]")
print(f"    步骤: (x - mean) / std * γ + β")
print(f"    mean={np.mean(x_test):.3f}, std={np.std(x_test):.3f}")
print()
print(f"  RMSNorm 输出:   [{', '.join([f'{v:.3f}' for v in rms_out[0]])}]")
print(f"    步骤: x / RMS(x) * γ")
print(f"    RMS={np.sqrt(np.mean(x_test**2)):.3f}")
print()
print("  RMSNorm 优势：")
print("    - 少一次求均值操作，计算更快")
print("    - 没有 beta 参数，参数量更少")
print("    - 实验表明效果与 LayerNorm 相当")
print("    - Llama、Qwen、Mistral 等都使用 RMSNorm")
print()

# ── 5.2 残差连接 ─────────────────────────────────────────
print("── 5.2 残差连接的重要性 ──────────────────────────────")
print()

print("  没有残差的深层网络：")
print("    x → F1 → F2 → ... → F96 → output")
print("    梯度要经过 96 次链式法则，可能变成 0！")
print()
print("  有残差的深层网络：")
print("    x ──→ (+) ──→ (+) ──→ ... ──→ output")
print("       \\  /    \\  /")
print("        F1      F2")
print()
print("    梯度可以直接通过'高速公路'回传！")
print("    output = x + F1(x) + F2(x+F1(x)) + ...")
print()

# 演示：信号在深层传播中的衰减
print("  信号衰减实验（模拟 32 层传播）：")
signal = np.ones(8) * 1.0
signal_no_res = signal.copy()
signal_with_res = signal.copy()

for layer in range(32):
    # 模拟一个收缩变换
    W = np.random.randn(8, 8) * 0.3
    transform = signal_no_res @ W
    signal_no_res = transform  # 无残差
    signal_with_res = signal_with_res + signal_with_res @ W * 0.1  # 有残差

print(f"    初始信号范数:   {np.linalg.norm(signal):.4f}")
print(f"    32层后(无残差): {np.linalg.norm(signal_no_res):.4f}")
print(f"    32层后(有残差): {np.linalg.norm(signal_with_res):.4f}")
print()
print("  结论: 没有残差连接，信号经过多层后会严重衰减或爆炸")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 6 章：完整 Transformer Block 前向传播
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 一个完整的 Transformer Block:
#   1. RMSNorm → Multi-Head Attention → 残差连接
#   2. RMSNorm → SwiGLU FFN → 残差连接
#
# 现代 LLM 堆叠 32~96 个这样的 Block

print()
print("=" * 60)
print("第 6 章：完整 Transformer Block 前向传播")
print("=" * 60)
print()

seq_len = 3
d_model = 8
n_heads = 2
d_ff = 32
d_k = d_model // n_heads
tokens_block = ['大', '模型', '强']

X_block = np.random.randn(seq_len, d_model)

print(f"  配置: seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
print(f"  输入 X: shape = {X_block.shape}")
print()


def multi_head_attention_block(x, W_QKV, W_O, n_heads):
    """完整的多头注意力（合并 QKV 投影，高效实现）"""
    seq_len, d_model = x.shape
    d_k = d_model // n_heads

    # 一次性投影得到 Q, K, V（实际实现中的优化）
    qkv = x @ W_QKV  # (seq_len, 3*d_model)
    Q, K, V = np.split(qkv, 3, axis=-1)

    # 拆分为多头
    Q = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    # 注意力计算
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
    weights = softmax(scores + mask)
    attn_out = weights @ V

    # 合并多头
    attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, d_model)
    return attn_out @ W_O


# 初始化参数
norm1_weight = np.ones(d_model)
norm2_weight = np.ones(d_model)
W_QKV = np.random.randn(d_model, 3 * d_model) * 0.1
W_O = np.random.randn(d_model, d_model) * 0.1
W_gate_block = np.random.randn(d_model, d_ff) * 0.1
W_up_block = np.random.randn(d_model, d_ff) * 0.1
W_down_block = np.random.randn(d_ff, d_model) * 0.1

# ── 前向传播 ─────────────────────────────────────────────
print("  ┌───────────────────────────────────────────────────┐")
print("  │         Transformer Block 前向传播过程            │")
print("  └───────────────────────────────────────────────────┘")
print()

# 子层1: RMSNorm + Attention + 残差
print("  ── 子层 1: 注意力 ──")
x_norm1 = rms_norm(X_block, norm1_weight)
print(f"    1. RMSNorm(X)           shape = {x_norm1.shape}")

attn_out = multi_head_attention_block(x_norm1, W_QKV, W_O, n_heads)
print(f"    2. MultiHeadAttn(norm)  shape = {attn_out.shape}")

X_after_attn = X_block + attn_out  # 残差连接
print(f"    3. X + Attn (残差连接)  shape = {X_after_attn.shape}")
print(f"       残差: 梯度可以直接回传，避免梯度消失")
print()

# 子层2: RMSNorm + SwiGLU FFN + 残差
print("  ── 子层 2: 前馈网络 ──")
x_norm2 = rms_norm(X_after_attn, norm2_weight)
print(f"    4. RMSNorm(X)           shape = {x_norm2.shape}")

ffn_out_block = swiglu_ffn(x_norm2, W_gate_block, W_up_block, W_down_block)
print(f"    5. SwiGLU_FFN(norm)     shape = {ffn_out_block.shape}")
print(f"       维度变化: {d_model} → {d_ff} → {d_model}")

X_final = X_after_attn + ffn_out_block  # 残差连接
print(f"    6. X + FFN (残差连接)   shape = {X_final.shape}")
print()

# 信息流图
print("  完整信息流：")
print("    X (输入)")
print("    │")
print("    ├──────────────────┐")
print("    ↓                  │")
print("    RMSNorm            │  (Pre-Norm)")
print("    ↓                  │")
print("    Multi-Head Attn    │  (token 间信息交换)")
print("    ↓                  │")
print("    + ←────────────────┘  (残差连接)")
print("    │")
print("    ├──────────────────┐")
print("    ↓                  │")
print("    RMSNorm            │  (Pre-Norm)")
print("    ↓                  │")
print("    SwiGLU FFN         │  (每个 token 独立非线性变换)")
print("    ↓                  │")
print("    + ←────────────────┘  (残差连接)")
print("    │")
print("    ↓")
print("    输出 (传入下一层)")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 7 章：KV Cache（推理加速核心）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 自回归生成时，每生成一个 token 都要重新计算 attention
# KV Cache 缓存已计算的 K 和 V，避免重复计算
# 将 per-step 计算量从 O(n^2) 降为 O(n)

print()
print("=" * 60)
print("第 7 章：KV Cache 推理加速")
print("=" * 60)
print()

print("── 7.1 朴素方法 vs KV Cache 对比 ────────────────────")
print()

# 模拟自回归生成过程
d_model_kv = 8
d_k_kv = 4
gen_len = 6

W_Q_kv = np.random.randn(d_model_kv, d_k_kv) * 0.5
W_K_kv = np.random.randn(d_model_kv, d_k_kv) * 0.5
W_V_kv = np.random.randn(d_model_kv, d_k_kv) * 0.5

# 模拟已有的 token embeddings
all_tokens = np.random.randn(gen_len, d_model_kv)
gen_tokens = ['今', '天', '天', '气', '真', '好']

print("  朴素方法（每步重新计算所有 K,V）：")
naive_ops = 0
for step in range(1, gen_len + 1):
    # 每步都对所有已有 token 重新计算
    ops = step * d_model_kv * d_k_kv * 2  # Q,K 的投影
    naive_ops += ops
    prefix = ''.join(gen_tokens[:step])
    print(f"    步骤 {step}: 计算 '{prefix}' 全部 {step} 个token的K,V → 计算量 ∝ {step}")

print(f"  总操作量: 1+2+3+...+{gen_len} = {gen_len*(gen_len+1)//2} (O(n^2))")
print()

print("  KV Cache 方法（缓存已计算的 K,V）：")
kv_cache_K = []
kv_cache_V = []
cache_ops = 0

for step in range(gen_len):
    # 只计算新 token 的 K,V
    new_k = all_tokens[step:step + 1] @ W_K_kv
    new_v = all_tokens[step:step + 1] @ W_V_kv
    kv_cache_K.append(new_k)
    kv_cache_V.append(new_v)
    cache_ops += 1

    # 新 token 的 Q 与缓存中所有 K 做注意力
    if step < 4:  # 只显示前几步
        print(f"    步骤 {step+1}: 只计算 '{gen_tokens[step]}' 的K,V → 拼接到缓存")
        print(f"           缓存大小: K={len(kv_cache_K)}个, V={len(kv_cache_V)}个")

print(f"    ...")
print(f"  总新增计算量: {gen_len} (O(n))，每步只算1个token的K,V!")
print()

print("  加速比: O(n^2) / O(n) = O(n)")
print(f"  序列长度=2048 时，理论加速 ~1000x (实际因注意力计算仍需全序列)")
print()

# ── 7.2 KV Cache 内存估算 ────────────────────────────────
print("── 7.2 KV Cache 内存占用估算 ─────────────────────────")
print()

print("  以 LLaMA-7B 为例 (FP16)：")
layers = 32
heads = 32
d_head = 128  # d_k per head
bytes_per_param = 2  # FP16

single_token_kv = 2 * layers * heads * d_head * bytes_per_param
print(f"    参数: {layers} 层, {heads} 头, d_k={d_head}, FP16")
print(f"    单 token KV 缓存: 2 * {layers} * {heads} * {d_head} * 2B = {single_token_kv:,} B = {single_token_kv/1024/1024:.2f} MB")
print()

for seq_len_est in [512, 2048, 8192, 128000]:
    total_kv = single_token_kv * seq_len_est
    print(f"    序列长度 {seq_len_est:>6d}: KV Cache = {total_kv/1024/1024/1024:.2f} GB")

print()
print("  batch_size=32, seq=2048:")
kv_batch = single_token_kv * 2048 * 32
print(f"    KV Cache 总计: {kv_batch/1024/1024/1024:.1f} GB (比模型权重 13.4GB 还大！)")
print()
print("  这就是为什么需要 PagedAttention、GQA 等技术来优化 KV Cache")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 8 章：GQA (Grouped-Query Attention)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 标准 MHA: Q/K/V 各有 h 组
# GQA: Q 有 h 组，K/V 只有 h/g 组
# 多个 Q 头共享 K/V，减少 KV Cache 大小

print()
print("=" * 60)
print("第 8 章：GQA (Grouped-Query Attention)")
print("=" * 60)
print()

print("  三种注意力变体：")
print()
print("    MHA (Multi-Head Attention):")
print("      Q: 32 组, K: 32 组, V: 32 组")
print("      每个 Q 头有自己独立的 K/V")
print()
print("    GQA (Grouped-Query Attention):")
print("      Q: 32 组, K: 8 组, V: 8 组")
print("      每 4 个 Q 头共享 1 组 K/V")
print()
print("    MQA (Multi-Query Attention):")
print("      Q: 32 组, K: 1 组, V: 1 组")
print("      所有 Q 头共享 1 组 K/V (极端情况)")
print()

# KV Cache 对比
print("  KV Cache 大小对比 (LLaMA-70B, seq=2048, FP16):")
print()
configs = [
    ("MHA (64Q, 64KV)", 64, 64),
    ("GQA (64Q, 8KV)", 64, 8),
    ("MQA (64Q, 1KV)", 64, 1),
]

d_head_gqa = 128
for name, q_heads, kv_heads in configs:
    kv_size = 2 * 80 * kv_heads * d_head_gqa * 2 * 2048  # 2(K+V) * layers * heads * dim * bytes * seq
    print(f"    {name:20s}: KV Cache = {kv_size/1024/1024/1024:.2f} GB")

print()
print("  GQA 的优势：")
print("    - KV Cache 减少到 MHA 的 1/8（64→8 组 KV）")
print("    - 推理速度大幅提升（内存带宽是瓶颈）")
print("    - 精度损失很小（大量实验验证）")
print()
print("  使用 GQA 的模型：")
print("    - Llama 2 70B: 64 Q头, 8 KV头")
print("    - Llama 3: 全系列使用 GQA")
print("    - Mistral 7B: 32 Q头, 8 KV头")
print("    - Qwen 2.5: 全系列使用 GQA")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 9 章：参数量与显存估算
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print()
print("=" * 60)
print("第 9 章：参数量与显存估算")
print("=" * 60)
print()

print("── 9.1 LLaMA-7B 参数量详细拆解 ──────────────────────")
print()

D = 4096  # d_model
H = 32  # n_heads
FF = 11008  # d_ff (SwiGLU 用 8/3 * d_model 取整)
L = 32  # 层数
V = 32000  # 词表大小

# 每层参数量
attn_params = 4 * D * D  # W_Q, W_K, W_V, W_O
ffn_params = 3 * D * FF  # W_gate, W_up, W_down (SwiGLU)
norm_params = 2 * D  # 两个 RMSNorm 的 gamma
block_params = attn_params + ffn_params + norm_params

# 总参数量
embedding_params = V * D
final_norm_params = D
lm_head_params = D * V  # 通常与 embedding 共享，不额外计数
total_params = L * block_params + embedding_params + final_norm_params

print(f"  模型配置:")
print(f"    d_model = {D}")
print(f"    n_heads = {H}, d_k = {D // H}")
print(f"    d_ff = {FF}")
print(f"    n_layers = {L}")
print(f"    vocab_size = {V}")
print()
print(f"  每层参数量拆解:")
print(f"    注意力: W_Q + W_K + W_V + W_O = 4 * {D}^2 = {attn_params:>12,}")
print(f"    FFN:    W_gate + W_up + W_down = 3 * {D} * {FF} = {ffn_params:>12,}")
print(f"    Norm:   2 * {D} = {norm_params:>12,}")
print(f"    每层合计: {block_params:>12,}")
print()
print(f"  总参数量:")
print(f"    {L} 层 Transformer Block: {L * block_params:>14,}")
print(f"    词嵌入 ({V} * {D}):        {embedding_params:>14,}")
print(f"    最终 RMSNorm:              {final_norm_params:>14,}")
print(f"    ─────────────────────────────────────────")
print(f"    总计:                      {total_params:>14,}")
print(f"                               ≈ {total_params / 1e9:.2f}B")
print()

# ── 9.2 显存估算 ─────────────────────────────────────────
print("── 9.2 不同精度的显存需求 ────────────────────────────")
print()

precisions = [
    ("FP32", 4),
    ("FP16/BF16", 2),
    ("INT8", 1),
    ("INT4", 0.5),
]

print(f"  {'精度':<12} {'每参数字节':<12} {'模型显存':<12} {'说明'}")
print("  " + "-" * 56)
for name, bytes_per in precisions:
    mem_gb = total_params * bytes_per / 1e9
    note = ""
    if name == "FP16/BF16":
        note = "标准推理精度"
    elif name == "INT4":
        note = "消费级 GPU 可用"
    elif name == "FP32":
        note = "训练精度（少用于推理）"
    elif name == "INT8":
        note = "轻量量化"
    print(f"  {name:<12} {bytes_per:<12} {mem_gb:<12.1f} {note}")
print()

# ── 9.3 常见模型参数量对比 ───────────────────────────────
print("── 9.3 常见模型参数量对比 ────────────────────────────")
print()

models = [
    ("LLaMA-7B", 32, 4096, 32, 6.7),
    ("LLaMA-13B", 40, 5120, 40, 13.0),
    ("Qwen2.5-72B", 80, 8192, 64, 72.7),
    ("LLaMA-70B", 80, 8192, 64, 70.0),
    ("GPT-3", 96, 12288, 96, 175.0),
]

print(f"  {'模型':<15} {'层数':<6} {'d_model':<8} {'头数':<6} {'参数量':<10} {'FP16显存':<10} {'INT4显存'}")
print("  " + "-" * 72)
for name, layers, dm, heads, params_b in models:
    fp16 = params_b * 2
    int4 = params_b * 0.5
    print(f"  {name:<15} {layers:<6} {dm:<8} {heads:<6} {params_b:<10.1f}B {fp16:<10.1f}GB {int4:.1f}GB")
print()

# 记忆公式
print("  ┌────────────────────────────────────────────────────────┐")
print("  │  快速估算公式：                                          │")
print("  │                                                        │")
print("  │  每层参数 ≈ 12 * d_model^2                              │")
print("  │  (注意力 4d^2 + FFN 约 8d^2)                            │")
print("  │                                                        │")
print("  │  模型总参数 ≈ 12 * d_model^2 * n_layers                 │")
print("  │                                                        │")
print("  │  FP16 显存 ≈ 参数量(B) * 2 GB                           │")
print("  │  INT4 显存 ≈ 参数量(B) * 0.5 GB                         │")
print("  │                                                        │")
print("  │  训练显存 ≈ 模型显存 * 4~6 (优化器状态+梯度+激活)       │")
print("  └────────────────────────────────────────────────────────┘")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 10 章：MoE (Mixture of Experts) 架构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# MoE 的核心思想：
#   每个 token 只激活一部分专家（FFN），而非全部参数
#   → 用超大模型的知识量，小模型的推理成本

print()
print("=" * 60)
print("第 10 章：MoE (Mixture of Experts) 架构")
print("=" * 60)
print()


def moe_layer(x, experts, router_weights, top_k=2):
    """
    简化的 MoE 层实现

    参数:
        x: 输入 (batch, d_model)
        experts: 专家列表，每个是 (W_gate, W_up, W_down)
        router_weights: 路由器权重 (d_model, n_experts)
        top_k: 每个 token 激活的专家数
    """
    n_experts = len(experts)
    batch_size = x.shape[0]

    # 路由器: 决定每个 token 送给哪些专家
    router_logits = x @ router_weights  # (batch, n_experts)
    router_probs = softmax(router_logits)

    # 选择 Top-K 专家
    top_k_indices = np.argsort(router_probs, axis=-1)[:, -top_k:]
    top_k_probs = np.take_along_axis(router_probs, top_k_indices, axis=-1)

    # 归一化 Top-K 概率
    top_k_probs = top_k_probs / top_k_probs.sum(axis=-1, keepdims=True)

    # 计算各专家输出并加权
    output = np.zeros_like(x)
    for i in range(batch_size):
        for j in range(top_k):
            expert_idx = top_k_indices[i, j]
            W_g, W_u, W_d = experts[expert_idx]
            expert_out = swiglu_ffn(x[i:i + 1], W_g, W_u, W_d)
            output[i] += top_k_probs[i, j] * expert_out[0]

    return output, router_probs


# 设置 MoE 层
d_model_moe = 8
d_ff_moe = 16
n_experts = 8
top_k_experts = 2

# 创建 8 个专家（每个是一个 SwiGLU FFN）
experts = []
for _ in range(n_experts):
    W_g = np.random.randn(d_model_moe, d_ff_moe) * 0.1
    W_u = np.random.randn(d_model_moe, d_ff_moe) * 0.1
    W_d = np.random.randn(d_ff_moe, d_model_moe) * 0.1
    experts.append((W_g, W_u, W_d))

# 路由器权重
router_W = np.random.randn(d_model_moe, n_experts) * 0.3

# 模拟输入
x_moe = np.random.randn(4, d_model_moe)
token_names = ['编程', '烹饪', '数学', '诗歌']

print(f"  配置: {n_experts} 个专家, 每 token 激活 Top-{top_k_experts} 个")
print()

# 前向传播
moe_output, router_probs_all = moe_layer(x_moe, experts, router_W, top_k=top_k_experts)

print("  路由器分配结果：")
print(f"  {'Token':<8}", end='')
for i in range(n_experts):
    print(f"{'E'+str(i+1):<8}", end='')
print("  选中专家")
print("  " + "-" * 76)

for i, name in enumerate(token_names):
    probs = router_probs_all[i]
    top_indices = np.argsort(probs)[-top_k_experts:][::-1]
    print(f"  {name:<8}", end='')
    for j in range(n_experts):
        if j in top_indices:
            print(f"{probs[j]:.3f}*  ", end='')
        else:
            print(f"{probs[j]:.3f}   ", end='')
    chosen = [f"E{idx+1}" for idx in top_indices]
    print(f"  {'+'.join(chosen)}")

print()
print("  MoE 的核心优势：")
print(f"    总参数量: {n_experts} 个专家 × 每专家参数 = 大模型的知识量")
print(f"    激活参数: 只有 {top_k_experts}/{n_experts} = {top_k_experts/n_experts*100:.0f}% 参数参与计算")
print(f"    推理成本: 接近小模型")
print()
print("  代表模型：")
print("    Mixtral 8x7B:  8 专家, Top-2, 总参 47B, 激活 13B")
print("    DeepSeek V3:   256 专家, Top-8, 总参 671B, 激活 37B")
print("    Qwen2.5-MoE:   64 专家, Top-8, 总参 14.3B, 激活 2.7B")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 总结
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print("=" * 60)
print("总结：现代 LLM 的 Transformer 配方")
print("=" * 60)
print("""
  ┌─────────────────────────────────────────────────────────┐
  │  组件             │ 现代选择                             │
  ├─────────────────────────────────────────────────────────┤
  │  位置编码         │ RoPE (旋转位置编码)                  │
  │  归一化           │ RMSNorm (Pre-Norm)                  │
  │  注意力           │ Grouped-Query Attention (GQA)       │
  │  FFN              │ SwiGLU                              │
  │  激活函数         │ SiLU (x * sigmoid(x))              │
  │  推理优化         │ KV Cache + PagedAttention           │
  │  稀疏化           │ MoE (按需激活专家)                  │
  └─────────────────────────────────────────────────────────┘

  完整的 LLM 架构：
    输入 tokens
      ↓
    Token Embedding + RoPE
      ↓
    ┌─────────────────────────┐
    │  Transformer Block × N  │ (N = 32~96)
    │    RMSNorm → GQA Attn   │
    │    RMSNorm → SwiGLU FFN │
    └─────────────────────────┘
      ↓
    RMSNorm
      ↓
    LM Head (线性层 → logits → softmax → 下一个 token 概率)
""")
