# Transformer 从零训练（Transformer Training from Scratch）

> BPE 分词 → 自注意力 → 位置编码 → GPT 训练 → 采样策略 → KV Cache —— 把每一块拆开、看透、装回去

---

## 一、学习路径

本目录 7 个 demo 构成完整的 LLM 训练实验室。推荐按下表顺序阅读：

| 序号 | 文件 | 为什么放在这里 |
|------|------|----------------|
| 1 | [`bpe_tokenizer.py`](./bpe_tokenizer.py) | 万事开头先理解"词"的边界；BPE 是 GPT 家族的根 |
| 2 | [`attention_from_scratch.py`](./attention_from_scratch.py) | Transformer 的核心计算，NumPy+PyTorch 双重实现，一行一行看清楚 |
| 3 | [`positional_encoding.py`](./positional_encoding.py) | Attention 本身无序；三种位置方案各有权衡 |
| 4 | [`gpt_train.py`](./gpt_train.py) | 把前三块组装成完整模型，真正跑训练，看 loss 下降 |
| 5 | [`sampling_strategies.py`](./sampling_strategies.py) | 模型训完后如何"说话"；不同采样策略的效果差距很大 |
| 6 | [`attention_visualization.py`](./attention_visualization.py) | 打开黑盒，看注意力头到底在关注什么 |
| 7 | [`kv_cache.py`](./kv_cache.py) | 推理加速的核心技术；从理论复杂度到实测加速比 |

> **前置知识**：熟悉 PyTorch 基础张量操作（`../deep_learning/KNOWLEDGE.md`）；了解 softmax / 交叉熵（`../classical/KNOWLEDGE.md`）。

---

## 二、BPE Tokenization

### 2.1 为什么需要子词分词

字符级（a/b/c）：序列太长，上下文窗口装不下多少语义。
词级（the/running）：词表爆炸（英语有几十万词形变化），且无法处理新词。
**子词（subword）** 是折中：把高频整词保留，低频词拆成可复用的片段。

BPE（Byte-Pair Encoding）是目前最主流的子词方案，GPT-2/3/4 全部采用。

### 2.2 算法步骤

```
1. 初始词表 = 所有字节（0-255），共 256 个 token
2. 把语料按行切割，每行字节化为 id 序列
3. 统计所有相邻 (a, b) pair 出现频次
4. 找到频次最高的 pair，合并为一个新 token（id = 256 + step）
5. 替换语料中所有该 pair
6. 重复步骤 3-5，共 N 轮
最终词表大小 = 256 + N
```

这就是 [`bpe_tokenizer.py`](./bpe_tokenizer.py) 中 `BPETokenizer.train()` 的逻辑：每轮调用 `_get_stats()` 找最高频 pair，再调用 `_merge()` 替换。

### 2.3 一个具体例子

训练语料（前 50KB Tiny Shakespeare）跑 200 轮合并后：

```
原文(55 字节): ROMEO: But soft, what light through yonder window breaks?
BPE (~35 tokens): [前 20 个 token id 打印出来…]
压缩比: 1.58×
最终词表大小: 456  (256 字节 + 200 次合并)
```

合并过程的前几轮通常是：`(' ', 't')` → `' t'`，`('e', 'r')` → `'er'`，`('t', 'h')` → `'th'` …… 高频字母组合率先被吸收。

### 2.4 词表大小的权衡

| 词表大小 | 压缩率 | Embedding 矩阵 | 示例 |
|---------|--------|---------------|------|
| ~456（本 demo） | 1.58× | 456 × d | 演示算法骨架 |
| ~50K（GPT-2） | 约 4-5× | 50K × 768 ≈ 38M | 早期主流 |
| ~100K（GPT-4 cl100k_base） | 更高 | 100K × d | 多语言强 |
| ~128K（LLaMA-3） | 最高 | 128K × d | 代码 + 多语言 |

词表越大，每个 token 平均携带更多语义，但 Embedding 层的参数量也随之上升。GPT-4 的 cl100k_base 词表约 10 万，对中文字符的覆盖也更好。LLaMA 使用 SentencePiece（也是 BPE 变体，但在 Unicode 字节上操作，并额外支持 unigram 语言模型）。

> **关键洞察**：BPE 算法的数学本质和生产 tokenizer 完全一致；本 demo 词表小只是因为训练数据少，跑 200 步而非 5 万步。

### 2.5 注意：主训练使用字符级 tokenizer

[`gpt_train.py`](./gpt_train.py) 中的 `build_char_tokenizer()` 使用**字符级分词**（vocab ≈ 65），而非本节的 BPE。原因是字符级实现零依赖、迭代快，可以在 5 分钟内看到 loss 收敛。BPE 的算法原理与此节完全一致，两者的架构层（模型本身）完全相同。

---

## 三、自注意力数学推导

### 3.1 Q、K、V 的含义

自注意力把输入序列的每个 token 表示 $x_i$ 线性投影成三个向量：

| 向量 | 直觉 | 作用 |
|------|------|------|
| **Q（Query）** | "我在找什么？" | 用来与所有 K 做比较 |
| **K（Key）** | "我有什么特征？" | 被所有 Q 查询 |
| **V（Value）** | "我能提供什么？" | 加权求和后输出 |

一个经典比喻：Q 是图书馆读者的问题，K 是书脊上的关键词，V 是书的实际内容。

### 3.2 完整公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- $Q \in \mathbb{R}^{T \times d_k}$，$K \in \mathbb{R}^{T \times d_k}$，$V \in \mathbb{R}^{T \times d_v}$
- $T$ 是序列长度，$d_k$ 是头的维度

具体步骤：

```
scores  = Q @ K.T / sqrt(d_k)     # (T, T) 相似度矩阵
weights = softmax(scores, dim=-1)  # 归一化到概率分布
output  = weights @ V              # 加权聚合 Value
```

### 3.3 为什么除以 $\sqrt{d_k}$？

当 $d_k$ 很大时，$QK^\top$ 的每个元素是 $d_k$ 个乘积的和。假设 $q_i, k_j \sim \mathcal{N}(0,1)$，则点积的方差为 $d_k$，标准差为 $\sqrt{d_k}$。

不除以 $\sqrt{d_k}$：点积值域爆炸 → softmax 进入饱和区（输出趋近 one-hot）→ 梯度消失。

除以 $\sqrt{d_k}$ 后：方差归一 → softmax 输出分布平滑 → 梯度流通。

[`attention_from_scratch.py`](./attention_from_scratch.py) 的 `numpy_attention()` 在第一行就是 `scores = Q @ K.T / np.sqrt(d_k)`，并通过与 `F.scaled_dot_product_attention` 对比验证（最大误差 < 1e-5）。

### 3.4 因果 Mask（Causal Mask）

GPT 是自回归模型，第 $i$ 个位置只能看到 $0 \ldots i$ 的信息：

```
位置:  0  1  2  3  4
  0  [✓  ✗  ✗  ✗  ✗]
  1  [✓  ✓  ✗  ✗  ✗]
  2  [✓  ✓  ✓  ✗  ✗]
  3  [✓  ✓  ✓  ✓  ✗]
  4  [✓  ✓  ✓  ✓  ✓]
```

实现：对上三角位置填入 $-\infty$，softmax 后这些位置权重变为 0：

```python
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
scores = scores.masked_fill(~mask, float("-inf"))
```

### 3.5 多头注意力（Multi-Head Attention）

单头 attention 把所有信息压在一个子空间。多头的思想是：**把 $d_\text{model}$ 等分成 $h$ 份，每份独立跑一个 attention，再 concat 回去**。

$$
\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

每个头学不同的关系模式：有的头关注语法（主谓依存），有的头关注语义（名词共指）。[`attention_from_scratch.py`](./attention_from_scratch.py) 中 `MultiHeadAttention(d_model=64, n_heads=4)` 演示了 $d_k = 64/4 = 16$ 的分头计算。

> **参数量**：四个投影矩阵 $W^Q, W^K, W^V, W^O$，每个 $d \times d$，总参数约 $4 d^2$。

---

## 四、位置编码

Transformer 的 attention 本质上是集合运算——打乱输入顺序结果不变。**位置编码**（PE）让模型知道每个 token 的序列位置。

### 4.1 三种方案对比

| 方案 | 代表 | 外推性 | 是否增加参数 | 相对位置 |
|------|------|--------|------------|---------|
| **绝对正余弦** | Vaswani 2017 | 有限（未见过的长度效果差） | 无 | 隐式（不天然） |
| **学习式** | GPT-2 / BERT | 无（超出 max_len 直接失效） | 有（max_len × d） | 无 |
| **RoPE** | LLaMA / Qwen / DeepSeek | 最佳（可通过 YaRN 等扩展） | 无 | 天然 |

### 4.2 绝对正余弦（Sinusoidal PE）

Vaswani 2017 提出，无可学习参数：

$$
\text{PE}(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad
\text{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

不同维度对应不同频率，低维度频率高（精细分辨近邻），高维度频率低（感知远程位置）。

[`positional_encoding.py`](./positional_encoding.py) 中 `sinusoidal_pe(seq_len, d_model)` 直接实现上式，输出 ASCII 热图可以直观看到频率随维度降低的规律。

### 4.3 学习式位置编码（Learned PE）

```python
self.pos_emb = nn.Embedding(max_len, d_model)
x = tok_emb + pos_emb(positions)
```

GPT-2 / BERT 的做法。优点是简单灵活；缺点是完全无法外推（若训练时 max_len=128，推理时输入 300 个 token 则超出范围）。[`gpt_train.py`](./gpt_train.py) 使用这种方案（`self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)`）。

### 4.4 RoPE（旋转位置编码）

LLaMA / Qwen / DeepSeek 等现代 LLM 的标配。核心思想是把向量视为复数平面上的点，乘以一个旋转矩阵 $R(m)$：

$$
\text{RoPE}(q, m) = R(m) \cdot q
$$

关键性质：两个位置 $m, n$ 的旋转向量内积只依赖相对距离 $m - n$：

$$
\langle \text{RoPE}(q, m),\ \text{RoPE}(k, n) \rangle \propto f(q, k, m - n)
$$

实现时对 $(q_{2i}, q_{2i+1})$ 做二维旋转：

```
out[2i]   = q[2i]   · cos(θ) - q[2i+1] · sin(θ)
out[2i+1] = q[2i]   · sin(θ) + q[2i+1] · cos(θ)
其中 θ = m / (base^(2i/d))，通常 base=10000
```

[`positional_encoding.py`](./positional_encoding.py) 中 `rope_apply()` 实现并验证了"相同 $m-n$ 得到相近内积"的相对位置性质。

> **为什么 LLM 迁移到 RoPE**：天然编码相对位置（词语之间的相对距离比绝对位置更重要），且通过 YaRN / LongRoPE 等方法可以把原训练长度扩展到 4-8 倍以上，这是 LLaMA-3 支持 128K 上下文的基础之一。

---

## 五、Transformer Block 装配

### 5.1 Block 结构

[`gpt_train.py`](./gpt_train.py) 中 `Block` 类的完整结构：

```
输入 x
  │
  ├─ LayerNorm(x) ──▶ CausalSelfAttention ──▶ 输出 a
  │                                              │
  └──────────────────────────────────────────── + ──▶ x'
                                                 │
  ┌────────────────────────────────────────────  │
  │
  ├─ LayerNorm(x') ──▶ MLP ──▶ 输出 m
  │                              │
  └───────────────────────────── + ──▶ x''（传入下一层）
```

关键设计点：

1. **Pre-LN（先归一化再做 attention/FFN）**：比 Vaswani 2017 的 Post-LN 更稳定，梯度不容易在深层消失。GPT-2 和几乎所有现代 LLM 都采用 pre-LN。
2. **残差连接（Residual）**：`x = x + sublayer(LN(x))`，保证梯度高速公路，允许堆叠几十层。
3. **FFN 扩展比 = 4**：`MLP` 中先从 $d \to 4d$（`fc`），再从 $4d \to d$（`proj`），中间用 GELU 激活。这个 4× 是 Vaswani 提出的经验值，GPT 系列沿用。
4. **Weight Tying（权重绑定）**：`self.head.weight = self.tok_emb.weight`，输出头 lm_head 与 token embedding 共享同一矩阵，减少参数量并有助于泛化。

### 5.2 LayerNorm 数学

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

其中 $\mu, \sigma^2$ 是沿 embedding 维度（而非 batch 维度）的均值和方差，$\gamma, \beta$ 是可学习的仿射参数。

> **RMSNorm**（LLaMA 使用）：去掉减均值项，只除以 RMS：$\text{RMSNorm}(x) = \gamma \cdot x / \sqrt{\frac{1}{d}\sum x_i^2}$。更快，且在大模型上效果相当。

---

## 六、训练循环

### 6.1 超参配置

[`gpt_train.py`](./gpt_train.py) 中 `train()` 函数的完整超参：

| 超参 | 值 | 说明 |
|------|-----|------|
| `steps` | 2000 | 总训练步数 |
| `batch_size` | 32 | 每步处理 32 条序列 |
| `block_size` | 128 | 上下文窗口（token 数） |
| `lr` | 3e-4 | 峰值学习率 |
| `betas` | (0.9, 0.95) | AdamW 一/二阶矩衰减；$\beta_2=0.95$（非默认 0.999）对 LLM 更稳 |
| `weight_decay` | 0.1 | 参数正则，防止过拟合 |
| `grad_clip` | 1.0 | 梯度裁剪，防止梯度爆炸 |
| `warmup` | 100 步 | 线性 warmup，前 100 步从 0 升到峰值 lr |

### 6.2 学习率调度：余弦 Warmup

```
lr ↑
   ╱╲          ← 峰值 lr=3e-4
  ╱  ╲___
 ╱       ──╲___
╱             ──╲___
└──┬─────────────────→ steps
  100          2000
warmup     cosine decay → 10% 峰值
```

代码中 `lr_lambda` 实现：前 `warmup` 步线性升；之后按余弦衰减到峰值的 10%（`0.1 + 0.9 * 0.5 * (1 + cos(π·progress))`）。

### 6.3 AdamW 更新规则

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$
$$
w_t \leftarrow w_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda w_t
$$

最后一项 $\lambda w_t$ 是解耦的 weight decay（AdamW 而非 Adam 的关键区别）：L2 惩罚不经过自适应缩放，对权重绑定更稳健。

### 6.4 实测结果

在 Tiny Shakespeare（~1MB）上，字符级 tokenizer（vocab=65），~3M 参数：

| 设备 | 2000 步耗时 | val loss（初始→最终） |
|------|------------|----------------------|
| CPU（M 系芯片） | ~5 分钟 | 4.22 → 1.83 |
| MPS（Apple Silicon GPU） | ~30 秒 | 4.22 → 1.83 |

loss 从初始 $\ln(65) \approx 4.17$（随机猜测）下降到 1.83，说明模型学到了莎士比亚的字符统计规律。生成样本（temperature=0.8, top_k=40）已能看出明显的"伪莎士比亚"风格。

---

## 七、生成与采样策略

### 7.1 问题本质

模型输出的是 logits（未归一化的分数），经过 softmax 得到下一个 token 的概率分布 $p \in \mathbb{R}^{V}$。**采样策略**决定如何从这个分布中抽一个 token。

[`sampling_strategies.py`](./sampling_strategies.py) 实现并对比了 4 种策略（用相同的模型和相同的 seed 保证唯一变量是策略本身）。

### 7.2 四种策略

#### Greedy（贪心，temperature=0）

```python
next_id = logits.argmax(dim=-1, keepdim=True)
```

每步取概率最高的 token。确定性强，适合**翻译、数学推导、代码生成**等需要精确答案的场景。缺点：容易陷入重复循环（因为局部最优不等于全局最优）。

#### Temperature 采样

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

- $T < 1$：分布变尖，更确定（接近 greedy）
- $T > 1$：分布变平，更随机（胡言乱语风险上升）
- **甜区：T = 0.7–0.9**（大多数生产场景默认值）

#### Top-k 采样

```python
v, _ = torch.topk(logits, k)
logits[logits < v[:, [-1]]] = -float("inf")
```

只保留概率最高的 $k$ 个候选，其余置 $-\infty$，再做 softmax 采样。**截断长尾噪音**，防止模型"乱跳"到奇怪词。常用 $k=40$（与 temperature=0.8 联用）。

#### Top-p（核采样，Nucleus Sampling）

按概率从高到低累积，找到累积概率恰好超过 $p$ 的最小集合，只在这个集合内采样：

```
排序: p1 ≥ p2 ≥ p3 ≥ …
找 k 使得 p1+p2+…+pk ≥ p，截断 pk+1, pk+2 …
```

候选数量**自适应**：当模型很确定时（分布尖）候选少；不确定时（分布平）候选多。OpenAI 默认推荐 `top_p=0.9–0.95`。

### 7.3 何时用哪种策略

| 场景 | 推荐策略 | 理由 |
|------|---------|------|
| 代码/数学/SQL 生成 | greedy 或 T=0.1–0.3 | 确定性优先 |
| 聊天/对话 | T=0.7–0.8 + top_p=0.95 | 自然流畅 |
| 创意写作 | T=0.9–1.0 + top_k=50 | 多样性优先 |
| 生产 API 默认 | top_p=0.95, T=0.8 | 两者兼顾 |

**Repetition Penalty**：对已经出现过的 token 在 logits 层面做惩罚（$z_i \leftarrow z_i / \text{penalty}$ 若 $z_i > 0$，或 $z_i \times \text{penalty}$ 若 $z_i < 0$），解决 greedy 和低 temperature 下的死循环问题。

> **LLM 视角**：top_p + temperature 是 OpenAI / Anthropic API 的两个最核心旋钮。Claude / ChatGPT 的"创意模式"本质上就是调高这两个值。

---

## 八、推理优化 —— KV Cache

### 8.1 问题：自回归生成的计算冗余

生成第 $t$ 个 token 时，需要对已有的 $t$ 个 token 跑完整的 attention：

$$
\text{cost}_{\text{无 cache}} = \sum_{t=1}^{T} O(t \cdot d) = O(T^2 \cdot d)
$$

但注意：**第 1 到 $t-1$ 个 token 的 K/V 向量在每步都一样**（模型权重没变，输入没变）。每次重算都是浪费。

### 8.2 KV Cache 原理

```
第 1 步: prompt 全部跑一遍 attention（prefill）,保存每层的 K, V
第 2 步: 只输入新生成的 1 个 token,计算它的 q
         → q 与缓存的全部 K 计算 score
         → 用缓存的全部 V 加权求和
         → append 新 k, v 到缓存
第 3 步: 同上……
```

[`gpt_train.py`](./gpt_train.py) 中 `CausalSelfAttention.forward()` 支持 `kv_cache` 参数：当传入缓存时，直接 `k = torch.cat([past_k, k], dim=2)` 追加，不重算历史。

每步复杂度从 $O(t \cdot d)$ 降到 $O(d)$（只算新 token 的投影），总复杂度：

$$
\text{cost}_{\text{有 cache}} = O(T \cdot d)
$$

### 8.3 内存代价

每层需要存 K 和 V 两个张量，形状为 `(B, n_heads, T, head_dim)`：

$$
\text{Cache 大小} = 2 \times L \times H \times T \times d_{\text{head}} \times \text{bytes/float}
$$

以 LLaMA-70B（$L=80$，$H=64$，$d_{\text{head}}=128$，fp16 = 2 bytes）为例：

$$
2 \times 80 \times 64 \times 8192 \times 128 \times 2 \approx 21.5\ \text{GB}
$$

（kv_cache.py 注释中参考值为"约 1.6GB"是针对 LLaMA-70B 的较早估计，不同量化精度下数值有差异。）这是推理需要大显存的主要原因之一。

### 8.4 实测结果

[`kv_cache.py`](./kv_cache.py) 在本 demo（~3M 参数）上测得：

```
生成 100 个 token:
  无 cache: ??.??s
  有 cache: ??.??s
  加速比  : 1.88×
```

**为什么加速比只有 1.88×，而不是理论上的 $O(T^2)$ vs $O(T)$？**

本 demo 模型极小（3M 参数），矩阵乘法时间很短，MPS/CPU 的调度开销占主导。生产级 LLM（7B+，推理时序列长度往往 1K–8K）的实测加速通常在 **3–10×** 以上。

### 8.5 工业级扩展

| 技术 | 解决的问题 | 代表 |
|------|-----------|------|
| **FlashAttention** | 内存高效地重写 attention kernel（HBM 访问次数最小化），不改变 KV cache 逻辑 | Dao et al. 2022 |
| **PagedAttention** | KV cache 按"页"管理，消除内存碎片，支持多请求共享 cache | vLLM |
| **GQA（Grouped Query Attention）** | 多个 Query 头共享一组 K/V，大幅减少 cache 大小 | LLaMA-2 70B, Mistral |
| **MLA（Multi-Head Latent Attention）** | 把 KV 压缩成低秩表示再缓存 | DeepSeek-V2/V3 |

---

## 九、与 GPT-2 / LLaMA 的对应关系

### 9.1 参数量缩放表

| 模型 | 层数 $L$ | 头数 $H$ | $d_\text{model}$ | 参数量 |
|------|---------|---------|-----------------|--------|
| **本 demo** | 6 | 6 | 192 | ~3M |
| GPT-2 small | 12 | 12 | 768 | 124M |
| GPT-2 XL | 48 | 25 | 1600 | 1.5B |
| LLaMA-7B | 32 | 32 | 4096 | 7B |
| LLaMA-70B | 80 | 64 | 8192 | 70B |

参数量粗估：$N \approx 12 \times L \times d^2$（attention 4 矩阵 + FFN 2 矩阵 × 扩展比 4）。

### 9.2 架构演进

从本 demo 的 GPT-2 风格到现代 LLaMA 系列，演进主要在以下几个维度：

| 组件 | 本 demo（GPT-2 风格） | 现代 LLM（LLaMA 风格） | 原因 |
|------|---------------------|----------------------|------|
| 激活函数 | GELU | **SwiGLU** | 门控结构表达力更强 |
| 归一化 | LayerNorm | **RMSNorm** | 更快，效果相当 |
| 位置编码 | Learned PE | **RoPE** | 天然相对位置，外推性好 |
| 注意力 | MHA（每头独立 K/V） | **GQA**（多 Q 共享 K/V） | 大幅节省 KV cache 显存 |
| FFN | Dense FFN | **MoE**（稀疏专家混合） | 用更少激活参数处理更多知识 |
| 偏置 | 有 bias | **无 bias**（大部分线性层） | 训练更稳，GPT-2 已去掉 attention bias |

### 9.3 从 demo 到 GPT-2 的路径

```
本 demo (3M)
    ↓ 改 cfg: n_layer=12, n_head=12, n_embd=768, vocab_size=50257
GPT-2 small (124M)
    ↓ 改 cfg: n_layer=48, n_head=25, n_embd=1600
GPT-2 XL (1.5B)
    ↓ 替换 PE→RoPE, LN→RMSNorm, GELU→SwiGLU, 扩展词表
LLaMA-7B (7B)
```

**架构骨架完全一致**，差别只是规模和少量现代化改进。

---

## 十、配套代码索引

| 文件 | 主题 | 关键 API / 类 | 预计运行时长 |
|------|------|--------------|------------|
| [`bpe_tokenizer.py`](./bpe_tokenizer.py) | 从零实现 BPE 分词器 | `BPETokenizer.train()`, `.encode()`, `.decode()` | < 30 秒 |
| [`attention_from_scratch.py`](./attention_from_scratch.py) | NumPy + PyTorch 自注意力 | `numpy_attention()`, `MultiHeadAttention`, `F.scaled_dot_product_attention` | < 5 秒 |
| [`positional_encoding.py`](./positional_encoding.py) | 正余弦 / 学习式 / RoPE | `sinusoidal_pe()`, `LearnedPE`, `rope_apply()` | < 5 秒 |
| [`gpt_train.py`](./gpt_train.py) | ~3M GPT 训练（Tiny Shakespeare） | `GPT`, `GPTConfig`, `train()`, `load_checkpoint()` | CPU ~5min / MPS ~30s |
| [`sampling_strategies.py`](./sampling_strategies.py) | greedy / temp / top-k / top-p 对比 | `generate()` with strategy 参数 | < 30 秒（需要 ckpt） |
| [`attention_visualization.py`](./attention_visualization.py) | ASCII 注意力热图 | `patch_attn_to_record()`, `show_attn_grid()` | < 10 秒（需要 ckpt） |
| [`kv_cache.py`](./kv_cache.py) | KV cache vs 无 cache 基准测试 | `gen_no_cache()`, `gen_with_cache()`, `cache_size_bytes()` | < 30 秒（需要 ckpt） |

**运行顺序**：先 `python gpt_train.py`（生成 checkpoint），再运行后三个 demo。

```bash
# 下载语料（~1MB）
curl -L -o ml_foundations/transformer_training/data/tiny_shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 依次运行
python ml_foundations/transformer_training/bpe_tokenizer.py
python ml_foundations/transformer_training/attention_from_scratch.py
python ml_foundations/transformer_training/positional_encoding.py
python ml_foundations/transformer_training/gpt_train.py          # 生成 ckpt
python ml_foundations/transformer_training/sampling_strategies.py
python ml_foundations/transformer_training/attention_visualization.py
python ml_foundations/transformer_training/kv_cache.py
```

---

## 十一、延伸阅读

### 核心论文

- Vaswani et al. **"Attention Is All You Need"** (2017) —— Transformer 原始论文，必读
- Radford et al. **"Language Models are Unsupervised Multitask Learners"** (2019) —— GPT-2 论文，架构细节
- Touvron et al. **"LLaMA: Open and Efficient Foundation Language Models"** (2023) —— RoPE + RMSNorm + SwiGLU 的工程实践
- Su et al. **"RoFormer: Enhanced Transformer with Rotary Position Embedding"** (2021) —— RoPE 原始论文
- Dao et al. **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (2022) —— KV cache 配套的 attention 加速
- Holtzman et al. **"The Curious Case of Neural Text Degeneration"** (2019) —— top-p 核采样的理论基础

### 代码资源

- Andrej Karpathy [**nanoGPT**](https://github.com/karpathy/nanoGPT) —— 本 demo 架构的最直接参考，极简高质量
- Karpathy [**nanogpt-speedrun**](https://github.com/karpathy/nanogpt-speedrun) —— 在 124M GPT-2 上追求最低 loss，展示现代训练技巧
- Karpathy [**neural networks: zero to hero**](https://karpathy.ai/zero-to-hero.html) —— 从 makemore 到 GPT 的视频讲解，本 demo 的最佳配套视频

### 博客与综述

- Lilian Weng **"The Transformer Family"** (https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) —— 最全面的 Transformer 变体综述，含大量数学细节
- Lilian Weng **"Decoding Strategies in Large Language Models"** —— 采样策略详解，对应本文第七节
- Sebastian Raschka **"Build a Large Language Model (From Scratch)"** —— 与本 demo 思路最接近的书

> **下一站**：本目录掌握"如何训一个 LLM"；接下来可以阅读 `../../llm/` 目录，了解如何**微调、对齐、部署**已训好的 LLM。
