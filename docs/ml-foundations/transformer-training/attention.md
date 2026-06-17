# 自注意力机制

> `Attention(Q,K,V) = softmax(QK^T / √d) V` —— 这一行公式重塑了 AI 的走向

---

## 一、为什么 attention 是 Transformer 的核心

### 1.1 RNN 的两大历史遗留病

在 Transformer 出现之前，序列建模的主力是 RNN / LSTM。[深度学习基础](../deep-learning) 一章已经建立了这个概念。RNN 的核心矛盾是：**表达能力与可训练性之间的张力**。

具体表现为两大病症：

**病症一：梯度消失（长程依赖失效）**

反向传播沿时间展开，梯度要连乘 $T$ 次权重矩阵：

$$
\frac{\partial L}{\partial h_1} \propto \prod_{t=2}^{T} W_h \cdot \tanh'(\cdot)
$$

当 $|W_h \cdot \tanh'| < 1$ 时，梯度指数级衰减 —— 序列开头的 token 对训练几乎没有贡献。LSTM 的门控机制缓解了这个问题，但根本上没有解决。

**病症二：串行计算（GPU 利用率低下）**

RNN 的计算图是链状的：必须等 $h_{t-1}$ 算完才能算 $h_t$。

```
x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → ...
                   ↑ 必须等待        ↑ 必须等待
```

这意味着即使你有 1000 个 GPU 核心，RNN 每次只能用其中一个。相比之下，矩阵乘法可以在 GPU 的数千个核心上并行执行。

### 1.2 自注意力：一次性看全局

自注意力的核心思想是：**让每个 token 主动"查询"序列中所有其他 token，按相关度加权聚合信息**。

```
                    self-attention（一次矩阵运算，全并行）
                    ┌─────────────────────────────────────┐
x₁ ─────────────────┤                                     ├──→ y₁
x₂ ─────────────────┤  每个 token 同时看到所有其他 token    ├──→ y₂
x₃ ─────────────────┤                                     ├──→ y₃
...                 └─────────────────────────────────────┘   ...
```

关键优势：

| 维度 | RNN/LSTM | Self-Attention |
|------|---------|----------------|
| 并行性 | 串行，必须等待 $h_{t-1}$ | 全序列一次并行 |
| 任意两 token 距离 | $O(T)$ 步 | $O(1)$ 直接交互 |
| 长程依赖 | 容易衰减 | 位置间直接加权 |
| GPU 利用率 | 极低 | 接近满载 |

::: tip LLM 视角
GPT-3 训练时使用了约 10,000 个 A100 GPU，如果用 RNN，**串行依赖会让 99%+ 的算力空转**。自注意力的全并行特性是 LLM 规模化的前提条件，不是锦上添花，而是必要条件。
:::

### 1.3 一行公式，一个世界

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

这行公式里的每个字符都有深刻含义：

| 符号 | 含义 | 为什么这样设计 |
|------|------|--------------|
| $Q$ | Query 矩阵 | 每个 token 的"搜索请求" |
| $K$ | Key 矩阵 | 每个 token 的"特征标签" |
| $V$ | Value 矩阵 | 每个 token 的"实际内容" |
| $QK^\top$ | 相似度矩阵 | 衡量每对 token 之间的关联 |
| $\sqrt{d_k}$ | 缩放因子 | 防止 softmax 进入饱和区 |
| softmax | 归一化 | 把相似度转为概率分布 |
| $\times V$ | 加权聚合 | 按相关度混合各 token 的内容 |

接下来每一节都会把这行公式的某个部分彻底讲透。

---

## 二、Q/K/V：搜索引擎的隐喻

### 2.1 搜索引擎类比

自注意力和搜索引擎在结构上高度同构：

```
搜索引擎：
  你输入 "如何学 Python"（Query）
        ↓
  与数百万网页标题匹配（Keys）
        ↓
  返回相关性加权后的内容（Values）

自注意力：
  token "猫" 发出查询（Query）
        ↓
  与序列中所有 token 的特征比较（Keys）
        ↓
  加权聚合所有 token 的语义信息（Values）
```

| 搜索引擎 | 自注意力 |
|---------|---------|
| 搜索词（query）| $Q$：当前 token 在"找什么" |
| 网页标题（key）| $K$：其他 token 的"我有什么标签" |
| 网页内容（value）| $V$：其他 token 的"我能提供什么信息" |
| 相关性得分 | $QK^\top / \sqrt{d_k}$ |
| 点击率归一化 | softmax |
| 综合阅读多页内容 | $\text{weights} \times V$ |

### 2.2 为什么用三个独立的投影？

输入 $X \in \mathbb{R}^{T \times d_\text{model}}$ 经过三个独立的线性层变换为 $Q$、$K$、$V$：

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

其中 $W_Q, W_K \in \mathbb{R}^{d_\text{model} \times d_k}$，$W_V \in \mathbb{R}^{d_\text{model} \times d_v}$。

**为什么不直接用 $X$ 本身？**

如果 $Q = K = V = X$，那么"搜索视角"、"被搜索视角"、"内容视角"是同一个向量。三个独立的投影矩阵让模型可以学习三种**完全不同的表示视角**：

- $W_Q$：学"什么特征对于提问有用"
- $W_K$：学"什么特征对于被检索有用"
- $W_V$：学"什么特征对于信息传递有用"

这三种视角往往截然不同。例如动词"吃"在语法依存关系中更容易被名词主语检索（K 视角），但在语义信息传递上提供了事件类型（V 视角），而在寻找宾语时又会发出特定查询（Q 视角）。

::: tip LLM 视角
在 `gpt_train.py` 的 `CausalSelfAttention` 中，Q/K/V 三个投影被**合并为一个矩阵**：

```python
self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
q, k, v = self.qkv(x).split(C, dim=2)
```

这样可以用一次大矩阵乘法代替三次小矩阵乘法，在 GPU 上更高效（矩阵越大，FLOP 利用率越高）。数学上完全等价。
:::

---

## 三、缩放点积注意力（Scaled Dot-Product Attention）

### 3.1 完整数学推导

设输入序列 $X \in \mathbb{R}^{B \times T \times d_\text{model}}$，我们逐步推导每一个维度的变化。

**步骤 1：线性投影**

$$
Q = X W_Q \in \mathbb{R}^{B \times T \times d_k}
$$
$$
K = X W_K \in \mathbb{R}^{B \times T \times d_k}
$$
$$
V = X W_V \in \mathbb{R}^{B \times T \times d_v}
$$

每个 token 的 $d_\text{model}$ 维表示被投影到 $d_k$ 维的查询/键空间，$d_v$ 维的值空间。

**步骤 2：计算相似度矩阵**

$$
\text{scores} = Q K^\top \in \mathbb{R}^{B \times T \times T}
$$

第 $i$ 行第 $j$ 列的元素是 token $i$ 的 query 向量与 token $j$ 的 key 向量的点积：

$$
\text{scores}[i, j] = q_i \cdot k_j = \sum_{l=1}^{d_k} q_{il} \cdot k_{jl}
$$

这是一个衡量"token $i$ 应该关注 token $j$ 多少"的原始得分。

**步骤 3：缩放**

$$
\text{scaled} = \frac{\text{scores}}{\sqrt{d_k}} \in \mathbb{R}^{B \times T \times T}
$$

除以 $\sqrt{d_k}$ 是为了控制方差，第四节会详细推导原因。

**步骤 4：归一化为注意力权重**

$$
\text{weights} = \text{softmax}(\text{scaled},\ \text{dim}=-1) \in \mathbb{R}^{B \times T \times T}
$$

softmax 沿最后一维（即"被关注的 token"维）归一化，使得每一行的权重之和为 1，形成概率分布。

**步骤 5：加权聚合**

$$
\text{output} = \text{weights} \cdot V \in \mathbb{R}^{B \times T \times d_v}
$$

每个输出位置是所有 token 的 Value 向量的加权平均。

### 3.2 维度全程追踪

```
输入 X:     (B, T, d_model)
              ↓ W_Q / W_K / W_V 投影
Q, K:       (B, T, d_k)
V:          (B, T, d_v)
              ↓ Q @ K^T
scores:     (B, T, T)         ← T×T 的相似度矩阵
              ↓ / √d_k
scaled:     (B, T, T)
              ↓ softmax(dim=-1)
weights:    (B, T, T)         ← 每行是概率分布（和为 1）
              ↓ @ V
output:     (B, T, d_v)       ← 与输入序列长度相同，维度变为 d_v
```

::: info 本 demo 配置
`attention_from_scratch.py` 的 `MultiHeadAttention` 默认配置：
- `d_model = 64`，`n_heads = 4`，`d_k = 64/4 = 16`

`gpt_train.py` 的 GPTConfig 默认配置（微型 GPT）：
- `n_embd = 192`，`n_head = 6`，`head_dim = 192/6 = 32`

LLaMA-7B 生产配置：
- `d_model = 4096`，`n_heads = 32`，`head_dim = 128`
:::

### 3.3 NumPy 实现（逐行对照）

`attention_from_scratch.py` 中的 `numpy_attention` 函数完整实现了上述五个步骤：

```python
def numpy_attention(X, Wq, Wk, Wv, mask=None):
    """X: (T, d) → output: (T, d_v); 全用 NumPy 算清楚每一步。"""
    Q = X @ Wq          # (T, d_k)  — 步骤 1
    K = X @ Wk          # (T, d_k)  — 步骤 1
    V = X @ Wv          # (T, d_v)  — 步骤 1
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)             # (T, T) — 步骤 2+3
    if mask is not None:
        scores = np.where(mask, scores, -1e9)    # 因果 mask
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)  # 步骤 4（数值稳定的 softmax）
    out = weights @ V                            # (T, d_v) — 步骤 5
    return out, weights
```

注意 softmax 的实现用了 **数值稳定** 技巧：先减去每行最大值再做 exp，避免大数值导致的 `inf`：

$$
\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
$$

数学上等价于原始 softmax，但数值上更稳定。

---

## 四、为什么除以 √d？

### 4.1 数学证明

假设查询向量和键向量的每个分量独立同分布于标准正态：

$$
q_i, k_i \sim \mathcal{N}(0, 1), \quad i = 1, \ldots, d_k
$$

那么点积 $s = q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的统计量：

**期望：**

$$
\mathbb{E}[s] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \cdot \mathbb{E}[k_i] = 0
$$

（利用 $q_i$ 和 $k_i$ 独立，以及各自均值为 0）

**方差：**

$$
\text{Var}[s] = \sum_{i=1}^{d_k} \text{Var}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2 k_i^2] = \sum_{i=1}^{d_k} 1 \cdot 1 = d_k
$$

**标准差：**

$$
\text{std}[s] = \sqrt{d_k}
$$

**结论**：$s = q \cdot k$ 的标准差随 $d_k$ 增大而增大。除以 $\sqrt{d_k}$ 后，方差归一为 1：

$$
\text{Var}\!\left[\frac{s}{\sqrt{d_k}}\right] = \frac{d_k}{d_k} = 1
$$

### 4.2 不缩放会发生什么？

当 $d_k$ 很大（如 128）时，未缩放的点积值域约在 $[-3\sqrt{128}, +3\sqrt{128}] \approx [-34, +34]$。

softmax 的饱和效应：

```
scores（未缩放）: [  0.5,  34.0, -10.0,   2.0 ]
softmax 输出:    [ ~0.0, ~1.0,   ~0.0,  ~0.0 ]  ← 趋近 one-hot
                          ↑
                     几乎所有权重集中在这里
```

```
scores（已缩放）: [  0.04,  3.0,  -0.88,  0.18 ]
softmax 输出:    [  0.10,  0.64,   0.04,  0.12 ]  ← 分布平滑
```

softmax 进入饱和区的后果：

1. **梯度消失**：饱和区的梯度 $\approx 0$，参数几乎不更新
2. **注意力退化**：几乎所有权重压缩到一个 token，失去"关注多个位置"的能力
3. **训练不稳定**：早期随机初始化时点积值大，模型难以从均匀注意力起步

::: warning 经典错误
去掉 $\sqrt{d_k}$ 是初学者实现 attention 时最常见的 bug 之一。代码可以跑，但训练 loss 下降缓慢，调了很久 lr 都没用，根本原因是 softmax 饱和。

验证方法：在 `attention_from_scratch.py` 中把 `/ np.sqrt(d_k)` 改为 `/ 1`，观察 attention weights 是否变成接近 one-hot 的稀疏分布。
:::

### 4.3 SDPA 验证结果

`attention_from_scratch.py` 中将手写实现与 PyTorch 内置的 `F.scaled_dot_product_attention` 对比：

```
手写 vs F.scaled_dot_product_attention 最大误差: 1.19e-07
→ ✓ 一致
```

误差 $1.19 \times 10^{-7}$ 在 float32 精度（约 $10^{-7}$）范围内，完全由浮点运算顺序导致，数学上等价。

---

## 五、因果 Mask（Causal Mask）

### 5.1 为什么需要因果 Mask？

GPT 是**自回归**（autoregressive）语言模型：训练目标是预测下一个 token。如果第 $t$ 个位置在计算注意力时能"看到" $t+1, t+2, \ldots$ 的 token，那么模型在训练时直接"抄答案"，测试时生成时却没有未来信息 —— 造成**训练/推理不一致**，模型彻底失效。

因果约束：**位置 $t$ 只能关注位置 $0, 1, \ldots, t$，不能关注 $t+1, t+2, \ldots$**

### 5.2 实现原理

通过在 softmax 之前，把"未来"位置的 score 填为 $-\infty$：

$$
\text{masked\_scores}[i, j] = \begin{cases} \text{scores}[i, j] & \text{若 } j \leq i \\ -\infty & \text{若 } j > i \end{cases}
$$

softmax 中 $e^{-\infty} = 0$，因此这些位置的注意力权重恰好为零。

### 5.3 可视化

```
原始 scores:        应用因果 mask:         softmax 后的权重:
┌───────────────┐   ┌────────────────────┐   ┌──────────────────────┐
│ a  b  c  d  e │   │ a  -∞  -∞  -∞  -∞ │   │ 1.00  0     0     0  │
│ f  g  h  i  j │   │ f   g  -∞  -∞  -∞ │   │ 0.43  0.57  0     0  │
│ k  l  m  n  o │ → │ k   l   m  -∞  -∞ │ → │ 0.25  0.31  0.44  0  │
│ p  q  r  s  t │   │ p   q   r   s  -∞ │   │ 0.18  0.22  0.28  0.32│
│ u  v  w  x  y │   │ u   v   w   x   y │   │ 0.14  0.17  0.21  0.25 0.23│
└───────────────┘   └────────────────────┘   └──────────────────────┘
   T=5 序列             上三角填 -∞                下三角为权重（行和=1）
```

- 第 0 行：只能看自己（$w_{00} = 1.0$）
- 第 1 行：能看位置 0 和 1（两个正权重）
- 第 4 行：能看所有 5 个位置

### 5.4 生产代码实现

`gpt_train.py` 的 `CausalSelfAttention` 用 `register_buffer` 预计算一个固定的下三角矩阵：

```python
self.register_buffer(
    "mask",
    torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)),
    persistent=False,
)
```

在 forward 中使用 `masked_fill`：

```python
att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
```

`register_buffer` 的好处：
1. mask 随模型一起移动到 GPU（`.to(device)` 时自动跟随）
2. `persistent=False` 表示不保存到 checkpoint（可以重新计算）
3. 比每次前向传播重新创建 mask 更高效

::: tip LLM 视角
**BERT 不用因果 mask，GPT 系列必须用**。原因是架构目标不同：

- BERT（encoder）：用完整上下文预测 [MASK] token，双向注意力
- GPT（decoder）：自回归生成，只能用历史信息，必须单向注意力

这个设计决定决定了这两类模型的适用场景：BERT 擅长理解（分类、NER），GPT 擅长生成（对话、续写）。现代 LLM 几乎全是 GPT 风格的 decoder-only 架构（Llama、Qwen、Mistral、DeepSeek）。
:::

---

## 六、多头注意力（Multi-Head Attention）

### 6.1 为什么需要多头？

单头注意力只能学习一种"语义关系"：模型必须在一个 $d_k$ 维子空间内同时表达所有类型的关联。

但语言中存在多种同时存在的关系：

```
"The cat that the dog chased ate the mouse."

句法关系：  cat ─── ate（主谓）
共指关系：  that ─── cat（关系从句）
语义关系：  dog ─── chased（施事-动作）
近邻关系：  cat ─── that（局部上下文）
```

单头注意力每次只能"聚焦"一种关系模式，多头让模型**并行学习多种关系**。

::: tip LLM 视角
研究人员（Vig & Belinkov, Clark et al.）在 BERT 中发现：不同的注意力头确实分工明确：
- 某些头专注于语法依存（主谓关系）
- 某些头专注于位置近邻（相邻 token）
- 某些头专注于共指消解（"它"→"猫"）

这不是人工设计的，而是训练自动涌现的结果。
:::

### 6.2 维度变换全流程

多头注意力的核心操作是把 $d_\text{model}$ 维空间**均匀分割**为 $h$ 个子空间，每个子空间独立运行 attention：

```
输入 x:  (B, T, d_model)
           ↓ W_Q / W_K / W_V 投影
Q/K/V:  (B, T, d_model)
           ↓ .view(B, T, n_head, head_dim).transpose(1, 2)
Q/K/V:  (B, n_head, T, head_dim)    ← 每头看到完整序列，但维度更小
           ↓ 每头独立 scaled dot-product attention
out:    (B, n_head, T, head_dim)
           ↓ .transpose(1, 2).contiguous().view(B, T, d_model)
concat: (B, T, d_model)             ← 拼回原维度
           ↓ W_O 输出投影
output: (B, T, d_model)             ← 形状与输入完全相同
```

其中 `head_dim = d_model / n_head`。

**数学形式：**

$$
\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(Q W^Q_i,\ K W^K_i,\ V W^V_i)
$$

### 6.3 本 demo 的具体数字

`gpt_train.py` 的默认配置：`n_embd=192, n_head=6`

```
d_model = 192
n_head  = 6
head_dim = 192 / 6 = 32

每头的 QK^T 矩阵: (B, T, 32) @ (B, 32, T) → (B, T, T)
6 个头并行运算后拼接: 6 × 32 = 192 ← 恢复原维度
```

### 6.4 参数量分析

四个投影矩阵：$W_Q, W_K, W_V \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$，$W_O \in \mathbb{R}^{d_\text{model} \times d_\text{model}}$

$$
\text{MHA 参数量} = 4 \times d_\text{model}^2
$$

**关键：参数量与头数 $h$ 无关！**

增加头数只是重新分割已有的维度，不增加新参数。`attention_from_scratch.py` 验证了这一点：

```python
mha = MultiHeadAttention(d_model=64, n_heads=4)
print(f"参数量: {sum(p.numel() for p in mha.parameters())}")
# 输出: 16384 = 4 × 64² ✓
```

::: info 为什么输出要经过 W_O 再投影？
每个头的输出是 $d_\text{model}/h$ 维的，concat 后是 $d_\text{model}$ 维。$W_O$ 投影让模型可以**混合来自不同头的信息**，而不只是简单拼接。没有 $W_O$，多头在输出端是完全独立的，无法跨头融合信息。
:::

### 6.5 PyTorch 实现（`attention_from_scratch.py`）

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, causal=True):
        B, T, _ = x.shape
        # 投影 + reshape 成多头形状
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # 缩放点积注意力（每头独立）
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)   # (B, h, T, T)
        if causal:
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ V                                       # (B, h, T, d_k)
        # 合并多头 + 输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out), weights
```

---

## 七、复杂度分析

### 7.1 时间复杂度

自注意力的计算瓶颈在两处：

**① $QK^\top$ 矩阵乘法**：$(T \times d_k) \cdot (d_k \times T) \to T \times T$，需要 $O(T^2 \cdot d_k)$ 次乘加运算

**② $\text{weights} \cdot V$ 矩阵乘法**：$(T \times T) \cdot (T \times d_v) \to T \times d_v$，需要 $O(T^2 \cdot d_v)$ 次乘加运算

综合（取 $d_k = d_v = d/h$，$h$ 个头）：

$$
\text{Time} = O(T^2 \cdot d) \quad \text{per layer}
$$

与序列长度的平方成正比！

### 7.2 空间复杂度

注意力矩阵 $\text{weights} \in \mathbb{R}^{B \times h \times T \times T}$，需要存储在显存中：

$$
\text{Space(attention matrix)} = O(B \cdot h \cdot T^2)
$$

即使只看序列长度维度：$O(T^2)$ per layer。

### 7.3 长上下文是大问题

| 序列长度 T | 注意力矩阵大小（单头，float32）|
|-----------|--------------------------|
| 512       | $512^2 \times 4 = 1$ MB |
| 2048      | $2048^2 \times 4 = 16$ MB |
| 8192      | $8192^2 \times 4 = 256$ MB |
| 32768     | $32768^2 \times 4 = 4$ GB |
| 128000    | $128000^2 \times 4 = 61$ GB |

对于 LLaMA-3 这样有 32 个注意力头、32 层的模型，128K 上下文的注意力矩阵峰值显存需求约 $61 \times 32 \times 32 \approx 62$ TB —— **直接用标准 attention 根本不可能**。这就是 FlashAttention 存在的根本动机。

```
显存瓶颈示意:
T=512  ░░░░░                   < 32MB 每层，可接受
T=2K   ░░░░░░░░░░░░░░░░        ~500MB 每层，勉强
T=8K   ░░░░░░░░░░░░░░░░░░░░░░░ 8GB+ 每层，需要优化
T=128K ████████████████████████ 爆显存
```

::: tip LLM 视角
**FlashAttention**（Dao et al., 2022）通过重写 attention 的 CUDA kernel，把中间结果保留在 SRAM（片上缓存）而非 HBM（显存主存），避免了 $T^2$ 显存的实际分配。数学结果完全相同，但内存占用从 $O(T^2)$ 降到 $O(T)$，同时计算速度提升 2-4 倍。

这是目前所有生产级 LLM 推理框架（vLLM、TensorRT-LLM、llama.cpp）的标配。
:::

### 7.4 O(T²) 是根本性限制吗？

从信息论角度看，当 token 总数为 $T$ 时，**描述每对 token 之间的关系就需要 $O(T^2)$ 信息量**。所以 $O(T^2)$ 不仅是 attention 的复杂度，也是"全局两两交互"的信息论下界。

想要突破 $O(T^2)$，就必须放弃"完全两两交互"，退而求其次：
- **Sliding Window**：每个 token 只看局部窗口 → $O(T \cdot w)$
- **稀疏注意力**：只关注重要的 token 对 → 近似 $O(T \log T)$
- **状态空间模型（Mamba）**：用线性递推代替注意力 → $O(T)$，但全局依赖能力不同

---

## 八、现代变体（仅介绍）

自注意力的 $O(T^2)$ 空间复杂度 + KV Cache 内存占用，推动了各种变体的发展。本节只做概念性介绍。

### 8.1 变体一览

**MHA（Multi-Head Attention）**：标准多头注意力，GPT-2 的方案，每个头有独立的 $W_K, W_V$。

**MQA（Multi-Query Attention）**：所有 query 头共享**同一个** K 矩阵和 V 矩阵。KV Cache 大小减少为 $1/h$（$h$ 为头数）。Shazeer 2019 提出，PaLM 采用。代价是模型容量略有下降。

**GQA（Grouped-Query Attention）**：把 $h$ 个 query 头分成 $g$ 组，每组共享一对 K/V。是 MHA 和 MQA 的折中：
- $g = h$：退化为 MHA
- $g = 1$：退化为 MQA
- $g = h/4$（32 头 → 8 组）：KV Cache 减少 4 倍，性能接近 MHA

LLaMA-2 70B、LLaMA-3 系列、Mistral 7B 均采用 GQA。

**MLA（Multi-Head Latent Attention）**：DeepSeek-V2 引入。不直接缓存 K/V，而是把 K/V 压缩成一个低秩 latent 向量再缓存，推理时从 latent 恢复完整 K/V。KV Cache 降至标准 MHA 的约 1/10，同时理论表达能力不低于 MHA。

**Sliding Window Attention**：Mistral 7B 使用。每个 token 只能关注前 $w$ 个 token（如 $w = 4096$），超出窗口外的信息通过多层堆叠间接传递。时间/空间复杂度降到 $O(T \cdot w)$。

### 8.2 对比表格

| 变体 | KV Cache 占用 | 序列中 K/V 数 | 代表模型 |
|------|-------------|-------------|---------|
| **MHA** | 1× | $h$ 对独立 K/V | GPT-2, GPT-3 |
| **MQA** | $1/h$ | 1 对共享 K/V | PaLM |
| **GQA-8** | $1/4$（32→8组）| 8 对共享 K/V | LLaMA-2 70B, LLaMA-3 |
| **MLA** | ~$1/10$ | 低秩 latent | DeepSeek-V2/V3 |
| **Sliding Window** | $1\times$（窗口内）| $h$ 对，但截断 | Mistral 7B |

::: tip LLM 视角
这些变体都在回答同一个问题：**如何在模型能力不明显下降的前提下，减小 KV Cache 的内存占用？**

KV Cache 是 LLM 推理的主要显存消耗：以 LLaMA-70B（$L=80$，$h=64$，$d_\text{head}=128$，fp16）为例，$T=8192$ 的 KV Cache 约 21 GB。GQA 使这一数字降到约 5 GB，使得在单卡 A100（80GB）上服务更多并发请求成为可能。
:::

---

## 九、配套代码

| 文件 | 主题 | 关键函数/类 |
|------|------|-----------|
| [`attention_from_scratch.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/transformer_training/attention_from_scratch.py) | NumPy + PyTorch 双实现，SDPA 验证 | `numpy_attention()`, `MultiHeadAttention`, `causal_mask()` |
| [`gpt_train.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/transformer_training/gpt_train.py) | 生产版 CausalSelfAttention，含 KV cache 接口 | `CausalSelfAttention.forward(x, kv_cache)` |

::: details 自己跑一遍

```bash
cd ml_foundations/transformer_training
python attention_from_scratch.py
```

程序依次执行四个演示：

1. **NumPy 无 mask 注意力**：打印 5×5 注意力权重矩阵（ASCII 热图），每行都有正权重
2. **NumPy 因果 mask 注意力**：上三角被遮蔽，只有下三角有权重
3. **PyTorch 多头注意力**：`(1, 8, 64) → (1, 8, 64)`，权重形状 `(1, 4, 8, 8)`
4. **SDPA 验证**：手写 vs `F.scaled_dot_product_attention` 最大误差 `1.19e-07`

预期输出最后几行：

```
手写 vs F.scaled_dot_product_attention 最大误差: 1.19e-07
→ ✓ 一致

关键收获:
✓ Q·K^T 衡量相似度，除以 √d 防止 softmax 饱和
✓ 因果 mask 让 GPT 只看历史（自回归）
✓ 多头让模型在不同子空间学不同关系
```

:::

::: tip LLM 视角
**为什么 `attention_from_scratch.py` 要同时做 NumPy 和 PyTorch 两个版本？**

NumPy 版本：用于**理解机制**。没有自动微分，没有任何黑盒，每行都是可读的矩阵运算。

PyTorch 版本：用于**验证实现**。通过与 `F.scaled_dot_product_attention`（PyTorch 官方实现，内部用 FlashAttention kernel 优化）对比，确认手写实现的正确性。

这种"先 NumPy 写清楚，再 PyTorch 验证"的方法在研究工作中非常常见，Karpathy 的 nanoGPT 也是这个思路。
:::

---

## 十、延伸阅读

**论文（必读顺序）**

1. Vaswani et al. **"Attention Is All You Need"** (2017) — 原始 Transformer 论文，所有公式的源头，第三节是本页内容的直接来源
2. Shazeer **"Fast Transformer Decoding: One Write-Head is All You Need"** (2019) — MQA 提出，解释了为什么共享 K/V 对模型质量影响有限
3. Ainslie et al. **"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"** (2023) — GQA 方法，LLaMA-2/3 的直接参考
4. Dao et al. **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (2022) — 所有现代推理框架的底层

**视频讲解**

- Andrej Karpathy **"Let's build GPT: from scratch, in code"** (YouTube) — 手写 GPT 包括 attention，是本 demo 最直接的参考视频，强烈推荐

**博客**

- Lilian Weng **"Attention? Attention!"** — 涵盖从 soft attention 到多头注意力的完整历史演化，有大量可视化
- Lilian Weng **"The Transformer Family Version 2.0"** — MQA/GQA/MLA 等现代变体的最全综述
- Jay Alammar **"The Illustrated Transformer"** — 最适合初学者的可视化介绍，图表直观

---

> **下一站**：[位置编码](./positional-encoding) —— Attention 本身是置换不变的，位置编码告诉模型"谁在哪里"

