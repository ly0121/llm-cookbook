# 位置编码

> Sinusoidal → Learned → RoPE —— 三代方案，一条主线：让 Transformer 感知"顺序"

---

## 一、为什么需要位置编码？

Self-attention 有一个常被忽视的特性：**它是置换等变（permutation-equivariant）的**。

也就是说，如果把输入序列中任意两个 token 对调，attention 的输出只是对应行被对调，模型本身感知不到顺序的变化。

用一个极端的例子来说明：

```
序列 A："我  打  你"
序列 B："你  打  我"

在没有位置编码的 Transformer 里，
这两个序列会产生完全相同的内部表示！
```

这在语义上是灾难性的——"我打你"和"你打我"的意思南辕北辙，但模型无法区分。

::: tip LLM 视角
**为什么 RNN/LSTM 没有这个问题？**

RNN 是按时间步逐一计算的，$h_t = f(h_{t-1}, x_t)$，顺序信息"天然地"编码在递推关系里。

Transformer 放弃了时序递推，换来了全序列并行计算的效率优势——但代价是必须**显式地**把位置信息注入进来。这就是位置编码存在的根本原因。
:::

**解决方案**：在把 token 送入 Transformer 之前，给每个位置构造一个"位置向量"，加到（或作用于）token 表示上，让模型能区分位置 0、1、2、3……

三代主流方案的技术路线：

```
绝对正余弦 (Vaswani 2017)  →  学习式绝对 PE (BERT/GPT-2)  →  RoPE (LLaMA/Qwen/DeepSeek)
固定公式注入到 X           →  可学参数注入到 X             →  旋转作用于 Q/K
```

---

## 二、绝对正余弦编码（Sinusoidal，Vaswani 2017）

### 2.1 公式

Transformer 原论文提出的方案，完全由公式定义，没有可学习参数：

$$
PE_{(\text{pos},\ 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(\text{pos},\ 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)
$$

其中：
- $\text{pos} \in \{0, 1, \ldots, T{-}1\}$：token 在序列中的位置
- $i \in \{0, 1, \ldots, d_{\text{model}}/2{-}1\}$：维度索引（每对偶/奇维度共享一个频率）
- $d_{\text{model}}$：嵌入维度（如 512、768）

### 2.2 频率设计的直觉

不同维度对应不同的**频率**（或波长）：

| 维度 $2i$ | 频率 $\omega_i$ | 波长 $\lambda_i$ | 类比 |
|-----------|---------------|----------------|------|
| 0, 1（最低维） | $1 / 1 = 1$ | $2\pi \approx 6.3$ | 秒针（快速变化，区分近邻） |
| 4, 5 | $1 / 10000^{4/d}$ | 更长 | 分针 |
| $d{-}2, d{-}1$（最高维） | $1 / 10000$ | $20000\pi \approx 62832$ | 时针（缓慢变化，感知远程位置） |

```
低维度（高频，短波长）：  ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  精细区分相邻位置
高维度（低频，长波长）：  ∿         ∿         ∿  感知全局位置范围
```

这就像"时钟"编码：用不同进制的"位"共同表示一个数，理论上能无歧义地表示任意整数位置。

### 2.3 代码实现

`sinusoidal_pe(seq_len, d_model)` 的实现非常紧凑（见 `positional_encoding.py`）：

```python
def sinusoidal_pe(seq_len, d_model):
    """绝对正余弦。"""
    pos = np.arange(seq_len)[:, None]           # (T, 1)
    i   = np.arange(d_model)[None, :]           # (1, d)
    angle_rates = 1 / (10000 ** (2 * (i // 2) / d_model))
    angles = pos * angle_rates                   # (T, d)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])       # 偶数维
    pe[:, 1::2] = np.cos(angles[:, 1::2])       # 奇数维
    return pe
```

一个 4 × 8 的小矩阵（4 个位置，8 维）示意输出模式：

```
pos\dim  d0      d1      d2      d3      d4      d5      d6      d7
  0   [  0.000   1.000   0.000   1.000   0.000   1.000   0.000   1.000 ]
  1   [  0.841   0.540   0.100   0.995   0.010   1.000   0.001   1.000 ]
  2   [  0.909  -0.416   0.200   0.980   0.020   1.000   0.002   1.000 ]
  3   [  0.141  -0.990   0.296   0.955   0.030   1.000   0.003   1.000 ]
      └──────── 高频 ────────┘  └──── 低频 ──────┘  └──── 极低频 ────┘
```

低维列变化剧烈，高维列几乎不动——正好对应短波长和长波长。

### 2.4 优点与局限

**优点**：
- 无参数，公式定义，不占模型容量
- 理论上能外推到训练时未见过的序列长度

**局限**：
- 位置向量直接**加**到 token 表示上（$x \leftarrow x + PE$），在深层网络里位置信息容易被稀释
- 虽然理论上可以外推，但实际上超出训练长度后效果明显下降
- 没有"显式"的相对位置感知（后来的 RoPE 解决了这个问题）

::: tip LLM 视角
原始 Transformer 的实验发现：正余弦编码和学习式编码效果几乎相同。但 Vaswani 等人最终选择正余弦，因为它允许模型"外推到比训练时更长的序列"——尽管这个优点在实践中并不总是成立。
:::

---

## 三、学习式位置编码（Learned Absolute PE，BERT/GPT-2）

### 3.1 思路

把位置编码当成一个普通的 Embedding 层：分配一张 $\text{max\_len} \times d_{\text{model}}$ 的查找表，每一行对应一个位置，随机初始化，然后**和模型参数一起端到端训练**：

$$
PE = \text{Embedding}(\text{position\_id})
$$

```python
class LearnedPE(torch.nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = torch.nn.Embedding(max_len, d_model)

    def forward(self, seq_len):
        return self.pe(torch.arange(seq_len))
```

使用时与 token 嵌入直接相加：

$$
x_{\text{input}} = \text{TokEmb}(\text{token\_id}) + \text{PosEmb}(\text{position\_id})
$$

### 3.2 我们的 demo 就用这个

`gpt_train.py` 中的主模型 `GPT` 用的就是学习式 PE：

```python
# gpt_train.py, GPT.__init__
self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # ← 学习式 PE

# forward
x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
```

这里 `cfg.block_size = 128`，意味着最多支持 128 个 token 的上下文。

### 3.3 优缺点

**优点**：
- 实现极其简单
- 在训练长度范围内，表现通常和正余弦相当甚至更好（因为可以学到数据集特有的位置规律）

**致命缺陷——完全无法外推**：

```
训练时：位置 0, 1, 2, ..., 511  →  这 512 行 Embedding 有意义
推理时：输入第 512 个 token     →  位置 512 超出范围！直接报错或行为未定义
```

对于教学小模型，这不是问题（序列从不超过 block_size）。但在生产 LLM 里，上下文需要从 4K 扩展到 128K，这个缺陷是致命的。GPT-2 和早期 BERT 都使用学习式 PE，这也是它们上下文窗口有限（GPT-2 最大 1024）的原因之一。

::: tip LLM 视角
**为什么现代 LLM 不用学习式 PE？**

核心矛盾：训练一个拥有 128K 上下文的模型，学习式 PE 需要预先分配 128K × d 的参数表，而且每个位置都需要被充分训练到。更麻烦的是，如果之后想扩展到 256K 上下文，整个位置表需要推翻重来。RoPE 的出现彻底解决了这个问题。
:::

---

## 四、RoPE：旋转位置编码（Rotary Position Embedding）

RoPE 是 Su et al. 2021（RoFormer 论文）提出的，被 LLaMA、Qwen、DeepSeek 等几乎所有现代主流 LLM 采用。

### 4.1 核心思想的转变

前两种方案都是把位置信息**加到 token 表示 X 上**，然后 X 进入 attention。

RoPE 的思路完全不同：**不动 X，直接在 attention 计算中作用于 Q 和 K**。

$$
\underbrace{\langle \text{RoPE}(q, m),\ \text{RoPE}(k, n) \rangle}_{\text{attention score}} = \text{函数}(q, k, m{-}n)
$$

位置信息通过让 Q 和 K 按各自的位置"旋转"来注入，而两者的内积（即 attention score）只依赖**相对位置** $m - n$。

### 4.2 二维情形：复数平面上的旋转

先用二维的情形建立直觉。把向量 $(q_0, q_1)$ 看作复数 $q_0 + i \cdot q_1$，对位于位置 $m$ 的 token，乘以一个旋转因子 $e^{im\theta}$：

$$
(q_0 + i \cdot q_1) \cdot e^{im\theta} = (q_0 \cos m\theta - q_1 \sin m\theta) + i(q_0 \sin m\theta + q_1 \cos m\theta)
$$

用矩阵形式写出来：

$$
\text{RoPE}_{\text{2D}}(q, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix}
$$

这就是一个标准的二维旋转矩阵——位置 $m$ 对应旋转角 $m\theta$。

```
复数平面示意：

      q1
      │      ·  ←  原始向量 q
      │    /
      │  /  角度 mθ
      │/──────→  q0

      旋转后：向量方向改变了 mθ 角，长度不变
      不同位置 m 旋转不同角度
```

**关键性质**：两个旋转向量的内积：

$$
\langle R(m) q,\ R(n) k \rangle = \langle q,\ R(n-m) k \rangle = \text{函数}(q, k, n-m)
$$

只依赖相对角度差 $n - m$，绝对位置 $m$、$n$ 各自"抵消"了！

### 4.3 高维做法：两两配对分组旋转

$d$ 维向量分成 $d/2$ 个二维组，每组独立旋转：

$$
\text{RoPE}(q, m)^{(i)} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}, \quad i = 0, 1, \ldots, \frac{d}{2}{-}1
$$

每组用不同的频率 $\theta_i$，类似正余弦编码的多频率设计：

$$
\theta_i = \text{base}^{-2i/d}, \quad \text{通常 base} = 10000
$$

```
维度分组旋转示意：

(q0, q1) ──旋转θ0(快)──→ (out0, out1)
(q2, q3) ──旋转θ1──────→ (out2, out3)
(q4, q5) ──旋转θ2──────→ (out4, out5)
  ...
(q_{d-2}, q_{d-1}) ──旋转θ_{d/2-1}(慢)──→ (out_{d-2}, out_{d-1})
```

### 4.4 代码实现

`rope_apply(x, base=10000)` 的实现（见 `positional_encoding.py`）：

```python
def rope_apply(x, base=10000):
    """对 (T, d) 张量应用 RoPE。x 的最后一维必须是偶数。"""
    T, d = x.shape
    assert d % 2 == 0
    pos   = torch.arange(T).float()[:, None]                      # (T, 1)
    freqs = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))   # (d/2,)
    theta = pos * freqs[None, :]                                   # (T, d/2)
    cos, sin = theta.cos(), theta.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]     # 拆偶/奇维度
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin    # 旋转后偶维
    out[..., 1::2] = x1 * sin + x2 * cos    # 旋转后奇维
    return out
```

注意：RoPE **不加**到输入 X 上，而是在 attention 计算前分别作用于 Q 和 K：

```python
# 生产代码中的典型用法（伪代码）
Q = rope_apply(Q)   # Q 按位置旋转
K = rope_apply(K)   # K 按位置旋转
scores = Q @ K.T / sqrt(d_k)   # 内积只依赖相对位置
```

::: tip LLM 视角
**RoPE 为什么作用于 Q/K 而不是 X？**

加到 X 上之后，PE 信息需要经过 $W_Q$、$W_K$ 投影才能影响 attention score，中间可能被稀释或扭曲。直接在 Q/K 上旋转，保证位置信息"原汁原味"地体现在 attention score 里，且数学性质（相对位置依赖）完美成立。
:::

---

## 五、RoPE 的相对位置性质

### 5.1 关键定理

设 $q_m = \text{RoPE}(q, m)$，$k_n = \text{RoPE}(k, n)$，则：

$$
\langle q_m, k_n \rangle = \sum_{i=0}^{d/2-1} \left( q_{2i} k_{2i} + q_{2i+1} k_{2i+1} \right) \cos\!\big((m-n)\theta_i\big) + \left( q_{2i} k_{2i+1} - q_{2i+1} k_{2i} \right) \sin\!\big((m-n)\theta_i\big)
$$

整个表达式只含 $(m - n)$，而不含 $m$ 或 $n$ 各自的绝对值——这就是 **RoPE 天然编码相对位置**的数学保证。

### 5.2 实验验证

`positional_encoding.py` 的验证段落用 4 个不同的 $(m, n)$ 对，但都满足 $m - n = -5$，检验内积是否相同：

```python
for (m, n) in [(0, 5), (3, 8), (10, 15), (2, 7)]:
    q_m = rope_apply(q[None, :].repeat(m+1, 1))[m]
    k_n = rope_apply(k[None, :].repeat(n+1, 1))[n]
    print(f"m={m:3d}, n={n:3d}, m-n={m-n:+4d}  →  q_m·k_n = {(q_m @ k_n).item():+.4f}")
```

**实测输出**：

```
m=  0, n=  5, m-n=  -5  →  q_m·k_n = +9.2770
m=  3, n=  8, m-n=  -5  →  q_m·k_n = +9.2770
m= 10, n= 15, m-n=  -5  →  q_m·k_n = +9.2770
m=  2, n=  7, m-n=  -5  →  q_m·k_n = +9.2770
```

四对不同的 $(m, n)$，只要 $m - n = -5$，内积完全相同（$\approx 9.277$）。

::: tip LLM 视角
这个性质与人类对语言的直觉高度一致：我们感知的是"主语距谓语有多远"，而不是"主语在第几个位置、谓语在第几个位置"。RoPE 让 attention score 直接反映这种相对关系，而绝对正余弦和学习式 PE 都没有这个性质。
:::

### 5.3 可视化理解

```
假设 m-n=-5（q 在 k 前面 5 个位置）：

位置:  0  1  2  3  4  5  6  7  8 ...
       q                 k           → 内积 = 9.277
          q                 k        → 内积 = 9.277
                    q           k    → 内积 = 9.277

无论这对 (q,k) 出现在序列的哪个位置，
只要它们的间距是 5，attention 就以相同的方式响应。
```

---

## 六、长度外推与 base 调整

### 6.1 标准 RoPE 的外推问题

RoPE 虽然在理论上能处理任意长度，但实践中训练时使用的序列长度决定了模型"见过"哪些旋转角度。当推理长度超过训练长度时：

- 高频维度（$\theta_0$）的旋转角度会周期性重复（频率太高），这部分没问题
- **低频维度**（$\theta_{d/2-1}$）的旋转角度可能进入训练时从未出现过的区域 → 外推失效

```
低频维度旋转角度示意（base=10000，d=128）：

训练长度 4096：  旋转角约 4096 × 10000^(-127/128) ≈ 4.10 rad
超出后 8192：  旋转角约 8192 × 10000^(-127/128) ≈ 8.19 rad  ← 未见过！
```

### 6.2 解决方案：调大 base

最直接的修复：把 `base` 从 10000 调大（如 500000），让低频维度的波长更长，覆盖更大的位置范围：

$$
\theta_i = \text{base}^{-2i/d}
$$

`base` 增大 → $\theta_i$ 变小 → 旋转更慢 → 波长更长 → 能支持更长上下文

**现代 LLM 的 base 选择**：

| 模型 | RoPE base | 最大 context |
|------|-----------|-------------|
| LLaMA-1 / 2 | 10,000 | 4K / 4K–32K |
| LLaMA-3 | 500,000 | 8K → 128K（经微调扩展） |
| Qwen-2 | 1,000,000 | 32K |
| DeepSeek-V3 | 10,000（配合 YaRN） | 64K |
| Mistral-7B | 10,000 | 32K（滑动窗口） |

LLaMA-3 把 base 从 10,000 提升到 500,000，是它能支持 128K 上下文的基础技术之一。

### 6.3 YaRN 与 NTK-aware scaling

调大 base 需要重新训练或微调。如果已经有一个用小 base 训练好的模型，可以用**后训练插值**方法：

**NTK-aware scaling**（bloc97 提出）：
- 核心思想：把高频维度做线性频率插值，低频维度不动
- 对 base 进行等效缩放：$\text{base}' = \text{base} \times s^{d/(d-2)}$，其中 $s$ 是扩展倍数

**YaRN**（Yet another RoPE extensioN）：
- 更精细的分段处理：低频维度线性插值，中频维度 NTK 插值，高频维度不动
- DeepSeek-V3 使用 YaRN 在 base=10000 的基础上支持 64K 上下文

::: tip LLM 视角
**实践中的取舍**：

- **从零训练**：直接用大 base（500K–1M），简单可靠
- **已有模型想扩展**：用 YaRN 做短时微调（通常只需几千步），可以 4–8 倍扩展上下文
- **无微调直接外推**：YaRN 可以在损失约 10–15% 性能的前提下直接 2× 外推，适合快速验证
:::

---

## 七、ALiBi（简要提及）

**Attention with Linear Biases**（Press et al. 2022）是另一种思路，代表作是 BLOOM（176B 参数的开源 LLM）。

核心思想不是修改 Q/K，而是在 attention scores 矩阵上直接加一个**线性距离惩罚**：

$$
\text{score}(i, j) = \frac{q_i \cdot k_j}{\sqrt{d_k}} - m_h \cdot |i - j|
$$

其中 $m_h$ 是每个注意力头的斜率（不同头有不同斜率，从小到大），$|i - j|$ 是两个 token 的距离。

```
ALiBi 偏置矩阵示意（距离越远，惩罚越大）：

位置:  0   1   2   3   4
  0  [ 0  -1  -2  -3  -4 ]
  1  [-1   0  -1  -2  -3 ]   × m_h
  2  [-2  -1   0  -1  -2 ]
  3  [-3  -2  -1   0  -1 ]
  4  [-4  -3  -2  -1   0 ]
```

**优点**：训练时用短序列，推理时直接外推到更长序列，性能衰减平稳（因为线性衰减是单调的，不会产生"从未见过的角度"问题）。

**现状**：ALiBi 目前主要见于 BLOOM 等早期开源模型。主流方向已转向 RoPE 系列，因为 RoPE 在多数基准上效果更好，且有成熟的扩展（YaRN、LongRoPE 等）。

---

## 八、方案对比汇总

| 方案 | 训练形式 | 注入位置 | 外推性 | 相对位置感知 | 现代地位 |
|------|---------|---------|--------|------------|---------|
| 绝对正余弦 | 固定公式，无参数 | $x + PE$ | 中（理论可外推，实际有限） | 隐式（无法保证） | 教科书，原始 Transformer |
| 学习式绝对 | 可训练 Embedding | $x + PE$ | 极差（完全无法外推） | 无 | GPT-2 / BERT 历史遗留 |
| RoPE | 固定公式，无参数 | 旋转 Q, K | 好（base 可调；YaRN 可扩展） | 天然显式 | LLaMA / Qwen / DeepSeek 等主流 |
| ALiBi | 固定公式，无参数 | attention score 减偏置 | 极好（直接外推） | 显式（线性惩罚） | BLOOM，目前少数使用 |

---

## 九、我们 demo 的取舍

### 9.1 为什么主训练 demo 用学习式 PE？

`gpt_train.py` 是这套 demo 的核心——它要在 5 分钟内完成训练，让读者亲眼看到 loss 从 4.22 下降到 1.83。

学习式 PE 的优点在教学场景下完全压制其缺陷：
- 实现只需一行：`self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)`
- 没有任何额外的数学概念需要解释
- 对 block_size=128 的小模型，外推不是问题

### 9.2 positional_encoding.py 的价值

`positional_encoding.py` 则专注于"对比三种方案"：

1. **正余弦**：输出 ASCII 热图，直观看到频率随维度的变化规律
2. **学习式**：随机初始化状态（未训练），对比正余弦的规律性
3. **RoPE**：核心实验——验证内积只依赖相对距离 $m-n$

这个文件的存在让读者可以在不修改主训练代码的情况下，独立地理解位置编码的数学原理。

### 9.3 运行

```bash
python ml_foundations/transformer_training/positional_encoding.py
```

期望看到三段输出：
1. 正余弦编码的 ASCII 热图（可以看到明显的条纹模式）
2. 学习式 PE 的热图（初始随机，无规律）
3. RoPE 验证段——4 对 $(m, n)$ 的内积完全相同（$\approx 9.277$）

---

## 十、配套代码

| 文件 | 内容 |
|------|------|
| [`positional_encoding.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/transformer_training/positional_encoding.py) | `sinusoidal_pe()`、`LearnedPE`、`rope_apply()` 三种实现 + RoPE 相对位置验证 |
| [`gpt_train.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/transformer_training/gpt_train.py) | 主训练 demo，使用学习式 PE（`self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)`） |

---

## 十一、延伸阅读

### 核心论文

- Vaswani et al. **"Attention Is All You Need"** (2017) —— 正余弦编码的出处，Transformer 原始论文
- Su et al. **"RoFormer: Enhanced Transformer with Rotary Position Embedding"** (2021) —— RoPE 原始论文，含完整数学推导
- Press et al. **"Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)"** (2022) —— ALiBi 论文

### 技术博客

- bloc97 GitHub Gist，**"NTK-Aware Scaled RoPE"** —— NTK-aware scaling 的提出帖，引发了 RoPE 外推研究热潮
- Peng et al. **"YaRN: Efficient Context Window Extension of Large Language Models"** (2023) —— YaRN 论文，DeepSeek-V3 使用的方法
- EleutherAI blog，**"Rotary Embeddings: A Relative Revolution"** —— RoPE 的直觉解释，配图清晰

### 代码参考

- Andrej Karpathy [**nanoGPT**](https://github.com/karpathy/nanoGPT) —— 学习式 PE 的简洁实现，本 demo 的直接参考
- Meta [**LLaMA 3 代码**](https://github.com/meta-llama/llama3) —— 生产级 RoPE 实现，base=500000

---

> **下一站**：[GPT 训练](./gpt-train) —— 把 tokenizer、attention、位置编码组装成完整模型，跑一次真正的训练。
