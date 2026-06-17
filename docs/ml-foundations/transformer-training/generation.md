# 文本生成与采样策略

> 模型训完了，如何让它开口说话？greedy / temperature / top-k / top-p 全解析

---

## 一、为什么需要采样？

### 1.1 训练目标回顾

GPT 系列模型的训练目标是最大化下一个 token 的对数概率：

$$
\mathcal{L} = -\sum_{t} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1})
$$

推理时，模型在每一步输出 logits 向量 $z \in \mathbb{R}^V$（$V$ 为词表大小），经过 softmax 得到概率分布：

$$
P(x_{t+1} = i \mid x_{1..t}) = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

**问题**：从这个分布中取哪一个 token？这就是采样策略要解决的核心问题。

### 1.2 为什么不能总是选最大值？

最直觉的做法是每步选概率最高的 token（argmax），这叫 **greedy 解码**。但它有严重缺陷：

```
prompt: "The sky is very"
step 1: "blue"   (p=0.42)  ← 选了
step 2: "and"    (p=0.38)  ← 选了
step 3: "the"    (p=0.51)  ← 选了
step 4: "sky"    (p=0.49)  ← 选了
step 5: "is"     (p=0.47)  ← 选了
step 6: "very"   (p=0.44)  ← 循环回来了！
step 7: "blue"   (p=0.42)  ← 陷入死循环
```

Greedy 解码很容易陷入这种 **"the the the"** 式的重复循环，因为局部最优并不等于全局最优。

### 1.3 采样的本质

从概率分布中**随机抽样**而非取最大值，引入适度随机性，让模型能"走出"局部最优。

各种采样策略的核心差异在于：**如何对 logits 做预处理再采样？**

```
原始 logits z ∈ R^V
        ↓
  【采样策略处理】
  · 缩放（temperature）
  · 截断（top-k / top-p）
        ↓
   softmax → 概率 p
        ↓
   torch.multinomial(p, 1)  ← 从分布中随机抽一个
```

::: tip LLM 视角
OpenAI / Anthropic 的 API 本质上都是在 `temperature`、`top_k`、`top_p` 这几个旋钮上做文章。每次你在 Playground 里调"创意度"，本质上就是在调这些参数。
:::

---

## 二、Greedy 解码（temperature → 0）

### 2.1 算法

每步选概率最高的 token，完全确定性：

```python
next_id = logits.argmax(dim=-1, keepdim=True)
```

对应我们 demo 的实现：

```python
# sampling_strategies.py - greedy 分支
if strategy == "greedy":
    next_id = logits.argmax(dim=-1, keepdim=True)
```

等价于将 temperature 设为无穷小（接近 0），使分布无限尖锐，argmax token 的概率趋向 1.0。

### 2.2 特点分析

| 属性 | 值 |
|------|-----|
| 确定性 | 完全可复现（同 ckpt + 同 prompt → 永远相同输出） |
| 多样性 | 无（每次输出完全一样） |
| 速度 | 最快（无随机采样开销） |
| 重复风险 | 高 |

### 2.3 适用场景

- **机器翻译**：要最可能正确的翻译
- **数学/代码补全**：需要正确答案，不能"发挥创意"
- **SQL 生成**：语法有对错之分
- **数学推理（CoT）**：配合 self-consistency，对同一问题多次 greedy 采样取多数答案

::: tip LLM 视角
ChatGPT 的 "数学助手" 模式通常把 `temperature=0`（等效 greedy）。OpenAI Codex 系列默认 `temperature=0`，因为代码要能跑通。
:::

---

## 三、Temperature 采样

### 3.1 公式

Temperature $T$ 对 logits 做缩放后再 softmax：

$$
P_T(i) = \frac{\exp\!\left(\dfrac{z_i}{T}\right)}{\displaystyle\sum_j \exp\!\left(\dfrac{z_j}{T}\right)}
$$

```python
# sampling_strategies.py - temperature 分支
T = kwargs.get("temperature", 1.0)
probs = F.softmax(logits / T, dim=-1)
next_id = torch.multinomial(probs, 1)
```

### 3.2 T 对分布的影响

Temperature 对分布有"拉平"或"锐化"的效果：

```
原始 logits: [3.0, 1.5, 0.5, -0.5]
─────────────────────────────────────────────────────
T = 0.1  →  [30, 15, 5, -5]  → softmax 极度锐化
            [0.999, 0.000, 0.000, 0.000]  ≈ greedy

T = 1.0  →  [3.0, 1.5, 0.5, -0.5]  → 原始分布
            [0.64, 0.24, 0.09, 0.03]

T = 2.0  →  [1.5, 0.75, 0.25, -0.25]  → 分布变平
            [0.44, 0.32, 0.22, 0.12]

T → ∞    →  [0, 0, 0, 0]  → 均匀分布
            [0.25, 0.25, 0.25, 0.25]
```

### 3.3 三个极限行为

| Temperature | 分布形态 | 行为 |
|-------------|---------|------|
| $T \to 0$ | 退化为 one-hot（argmax） | 等价于 greedy |
| $T = 1$ | 原始概率分布 | 按模型学到的分布采样 |
| $T \to \infty$ | 均匀分布 | 完全随机，词语乱跳 |

### 3.4 实用值域

- **T = 0.1–0.3**：接近 greedy，适合代码/数学
- **T = 0.6–0.8**：多数生产场景的甜区，兼顾质量和自然度
- **T = 0.9–1.0**：创意写作，输出更有惊喜
- **T > 1.2**：通常开始产生语法错误或词语混乱

::: tip LLM 视角
ChatGPT 的 `temperature` 默认值约在 0.7–1.0 之间（OpenAI 未完全公开）。Claude 的 API 文档建议创意任务用 1.0，精确任务用 0.0。

我们的 demo 用 **T=0.1**（近似确定）和 **T=0.8**（常用创意值）做对比，你可以在 `sampling_strategies.py` 中直接改参数感受差异。
:::

---

## 四、Top-k 采样

### 4.1 算法

只在概率最高的 $k$ 个 token 中采样，其余 token 的 logit 设为 $-\infty$：

```python
# sampling_strategies.py - top_k 分支
k = kwargs.get("k", 40)
T = kwargs.get("temperature", 1.0)
v, _ = torch.topk(logits, k)
logits[logits < v[:, [-1]]] = -float("inf")   # 截断后 k 名
probs = F.softmax(logits / T, dim=-1)
next_id = torch.multinomial(probs, 1)
```

数学上等价于：

$$
P_k(i) = \begin{cases}
\dfrac{\exp(z_i / T)}{\displaystyle\sum_{j \in \text{Top}_k} \exp(z_j / T)} & \text{若 } i \in \text{Top}_k \\
0 & \text{其他}
\end{cases}
$$

### 4.2 特殊情形

| k 值 | 等价于 |
|------|--------|
| k = 1 | Greedy 解码 |
| k = 词表大小 V | 无截断（等价于纯 temperature 采样） |
| k = 40 | 我们 demo 的默认值，也是 GPT-2 论文的默认值 |

### 4.3 优缺点分析

**优点：**
- 过滤"长尾噪音" —— 概率极小的奇怪词被砍掉
- 防止模型"随机跳"到完全不相关的词

**缺点：**
- k 是固定值，**对不同锐度的分布不自适应**：
  - 当模型很确定（分布尖），k=40 可能包含很多无关低概率词
  - 当模型很不确定（分布平），k=40 可能截断太多合法候选

```
尖锐分布 (模型确定时):           平坦分布 (模型不确定时):
token:   A    B    C    D ...    token:   A    B    C    D ...
prob:   0.85 0.10 0.03 0.01 ... prob:   0.05 0.05 0.05 0.05 ...
                ↑                                        ↑
          k=40 保留了很多垃圾           k=40 可能截断了合法候选
```

### 4.4 主流取值

- k = 40：GPT-2 论文默认，也是我们 demo 的参数
- k = 50：HuggingFace `generate()` 的常见推荐
- k = 100：更开放，多样性更高

::: tip LLM 视角
Anthropic Claude API 支持 `top_k` 参数。OpenAI API **不直接暴露** top-k（只有 top_p）。

Top-k 通常和 temperature 联用：`top_k=40, temperature=0.8`。这也是 `gpt_train.py` 中 `generate()` 方法的参数组合：`temperature=1.0, top_k=None`（默认关闭），自行传入覆盖。
:::

---

## 五、Top-p 采样（Nucleus Sampling）

### 5.1 核心思想

Holtzman et al. 2019 提出"核采样"（The Curious Case of Neural Text Degeneration）：

> 不固定候选数量 k，而是固定**累积概率阈值 p**，取最小的候选集合使得累积概率 ≥ p。

### 5.2 算法步骤

1. 按概率从高到低对词表排序
2. 累加概率，找到使累积概率刚好超过 $p$ 的截断点
3. 只在截断点以内的 token 中采样

```python
# sampling_strategies.py - top_p 分支
p = kwargs.get("p", 0.95)
T = kwargs.get("temperature", 1.0)
sorted_logits, sorted_idx = torch.sort(logits, descending=True)
cum_probs = torch.cumsum(F.softmax(sorted_logits / T, dim=-1), dim=-1)

# shift right: 第 0 个 token 永远保留（不截断）
mask = cum_probs > p
mask[..., 1:] = mask[..., :-1].clone()
mask[..., 0] = False

sorted_logits[mask] = -float("inf")
logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
probs = F.softmax(logits, dim=-1)
next_id = torch.multinomial(probs, 1)
```

### 5.3 动态截断的优势

```
分布尖锐时 (模型确定):         分布平坦时 (模型不确定):
─────────────────────────     ─────────────────────────
p1=0.72 ─┐                   p1=0.12
p2=0.15  ├ 累积已超 0.9       p2=0.11
p3=0.06  ┘                   p3=0.10
p4=0.04  ← 截断点在这里        p4=0.09
p5=0.03                      p5=0.08
  ...                         ...  ← 要到更多才超 0.9
                              p9=0.07 ─┐ 累积超 0.9
                                       └ 截断点在这里

只取 3 个候选                  取 9 个候选
```

Top-p 的候选数量**自适应分布锐度**，这是它比 top-k 优越的根本原因。

### 5.4 实用值域

| p 值 | 行为 | 适用 |
|------|------|------|
| p = 1.0 | 无截断（全词表采样） | 关闭 top-p |
| p = 0.95 | 我们 demo 的默认值 | 大多数场景 |
| p = 0.9 | 稍微保守 | 平衡质量与多样性 |
| p = 0.7 | 偏保守 | 接近专业写作 |

::: tip LLM 视角
OpenAI API 的 `top_p` 参数默认值是 **1.0**（不截断），通常与低 temperature 联用。

Anthropic Claude API `top_p` 默认值也是 1.0。官方文档建议：**不要同时调 temperature 和 top_p**，选一个即可。

我们的 demo 中 **top_p=0.95** 与 temperature=0.8 联用，这在开源模型的社区实践中很常见（如 llama.cpp / Ollama 的默认配置）。
:::

---

## 六、Repetition Penalty（重复惩罚）

### 6.1 问题

即使用了采样，低 temperature 或特定数据集训练的模型仍可能产生重复：

```
生成: "ROMEO: I love thee, I love thee, I love thee, I love thee..."
```

### 6.2 实现原理

对已经出现过的 token，对其 logit 施加惩罚系数 $\alpha > 1$：

$$
z'_i = \begin{cases}
z_i / \alpha & \text{若 } z_i > 0 \text{ 且 token } i \text{ 已出现} \\
z_i \times \alpha & \text{若 } z_i < 0 \text{ 且 token } i \text{ 已出现} \\
z_i & \text{其他}
\end{cases}
$$

这个处理使得：无论 logit 正负，已出现 token 的 logit 绝对值都**减小**，降低被再次选中的概率。

HuggingFace `transformers` 的实现（`RepetitionPenaltyLogitsProcessor`）：

```python
# HuggingFace 风格（我们的 demo 未实现此特性）
score = torch.gather(input_ids_scores, 1, input_ids)
score = torch.where(score < 0, score * penalty, score / penalty)
next_token_logits.scatter_(1, input_ids, score)
```

### 6.3 常用值与注意事项

| 惩罚系数 | 效果 |
|---------|------|
| 1.0 | 无惩罚（关闭） |
| 1.1–1.2 | 轻度抑制重复，对流畅度影响小 |
| 1.3–1.5 | 明显减少重复，可能影响正常重复词（如 "the"） |
| > 2.0 | 过强，输出可能变得语义不连贯 |

::: warning 注意
Repetition penalty 会惩罚**所有**出现过的 token，包括英语中合法高频的 "the"、"is"、"a" 等。惩罚系数过大会使输出变得刻意回避常见词，反而不自然。

对话场景（说话人名称会自然重复）也需要注意这一点。
:::

---

## 七、Beam Search 与 LLM 时代为何弃用

### 7.1 什么是 Beam Search？

传统 NLP（机器翻译时代）的标准解码算法：每步保留概率最高的 $B$ 条候选路径，最终取总概率最高的完整序列。

```
B=3 时的搜索树（示意）:

              "The"(0.4)──"cat"(0.6)──"sat"(0.7)  → 路径 A: 0.168
prompt ─┤    "A"  (0.3)──"dog"(0.5)──"ran"(0.8)  → 路径 B: 0.120
              "I"  (0.2)──"saw"(0.7)──"the"(0.6)  → 路径 C: 0.084
```

传统机翻用 B=4~12，每步保留 top-B 路径，分支数受控。

### 7.2 为什么 LLM 不用 Beam Search？

| 维度 | Beam Search | 概率采样（top-k/top-p） |
|------|------------|----------------------|
| 输出多样性 | 低（B 条路径高度相似） | 高（每次随机不同） |
| 输出质量（有标准答案） | 高 | 中等 |
| 输出质量（开放生成） | 机械、公式化 | 自然、有变化 |
| 计算开销 | $B$ 倍内存和计算 | 与 greedy 相当 |
| 长文本 | 路径同质化更严重 | 可控 |

**核心矛盾**：Beam search 最大化序列的整体对数概率，但**最高概率的句子≠最自然的句子**（Holtzman 2019 的核心论点）。

::: tip LLM 视角
GPT-3 的论文（Brown et al. 2020）完全使用 temperature + top-p 采样，不涉及 beam search。

**Beam search 的遗留地盘**：
- 机器翻译（seq2seq，有明确参考译文）
- 语音识别（输出空间有限）
- 代码合成中的候选生成（beam 生成多条 + 单元测试筛选）
:::

---

## 八、各 API 参数对应表

| API | Greedy | Temperature | Top-k | Top-p | 重复惩罚 |
|-----|--------|------------|-------|-------|---------|
| **OpenAI Chat** | `temperature=0` | `temperature` | 不支持 | `top_p` | `frequency_penalty` / `presence_penalty` |
| **Anthropic Claude** | `temperature=0` | `temperature` | `top_k` | `top_p` | 不支持（内置） |
| **HuggingFace `generate()`** | `do_sample=False` | `temperature` | `top_k` | `top_p` | `repetition_penalty` |
| **我们的 demo** | `strategy="greedy"` | `strategy="temperature", temperature=T` | `strategy="top_k", k=40` | `strategy="top_p", p=0.95` | 未实现 |
| **gpt_train.py generate()** | `temperature→0` | `temperature` | `top_k` | 未实现 | 未实现 |

**gpt_train.py 中的 `generate()` 签名**（lines 150–166）：

```python
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=False):
    ...
    logits = logits[:, -1, :] / max(temperature, 1e-5)   # temperature 缩放
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")       # top-k 截断
    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
```

注意 `use_cache=True` 时会使用 KV cache 加速推理（每步只传入最新的 1 个 token）。

---

## 九、推测解码（Speculative Decoding）简介

### 9.1 核心思想

传统自回归生成每步只能出 1 个 token，GPU 利用率很低（生成 1 个 token 只做了很少矩阵乘法）。

推测解码（Leviathan et al. 2022）用一个**小模型（draft model）先快速生成多个候选 token**，再用大模型**一次性验证**：

```
Draft model（小，快）:     token₁ token₂ token₃ token₄  ← 一次生成 4 个候选
                               ↓       ↓      ↓      ↓
Target model（大，慢）:    [验证 1]  [验证 2] [验证 3] [拒绝]  ← 一次前向处理 4 个

接受前 3 个 token，在拒绝点重新采样
```

### 9.2 正确性保证

通过 rejection sampling，接受/拒绝机制保证最终输出的分布与**仅用大模型采样完全一致**——这是推测解码最重要的性质。

### 9.3 实际效果

| 指标 | 值 |
|------|-----|
| 加速比 | 2–3× |
| 输出分布 | 与原模型完全一致（非近似） |
| 前提条件 | draft model 与 target model 词表相同 |

::: tip LLM 视角
- **vLLM** 从 0.3.0 版本支持 speculative decoding（`--speculative-model`）
- **TGI**（Text Generation Inference）也支持
- **Together AI** 的推理服务内置推测解码
- **与 KV cache 完全兼容**，两者可以叠加使用

当前主流搭配：Llama-3-70B 作为 target，Llama-3-8B 作为 draft，加速比约 2-2.5×。
:::

---

## 十、我们 demo 的 5 个对比案例

### 10.1 实验设置

```
模型     : gpt_train.py 训练的 ~3M 字符级 GPT（Tiny Shakespeare）
Prompt   : "ROMEO:"
生成长度 : 200 tokens
随机种子 : torch.manual_seed(42)（所有策略相同）
```

### 10.2 五种策略配置

| 编号 | 策略标签 | 参数 |
|------|---------|------|
| 1 | greedy | `strategy="greedy"` |
| 2 | T=0.1（接近确定） | `strategy="temperature", temperature=0.1` |
| 3 | T=0.8（常用） | `strategy="temperature", temperature=0.8` |
| 4 | top-k=40 | `strategy="top_k", k=40, temperature=0.8` |
| 5 | top-p=0.95 | `strategy="top_p", p=0.95, temperature=0.8` |

### 10.3 典型输出对比

以下是在一次实际运行中三种典型策略的输出示例（节选前 ~120 字符）：

**Greedy（确定性，易重复）：**
```
ROMEO: I will not be a man to the world,
And the world is the world and the world is...
```
> 注意 "the world" 反复出现——greedy 陷入了概率最高的短语循环。

**T=0.8（流畅，自然）：**
```
ROMEO: What, shall I speak so well?
That I have been the father of my love,
And with a most virtuous heart...
```
> 语言风格接近莎士比亚，流畅且有变化。

**top-p=0.95（多样，有惊喜）：**
```
ROMEO: My lord, what means this bloody brawl?
The prince hath sent for you to the court,
Where you shall find him...
```
> 候选集合自适应，生成句式更丰富，偶有意外但有趣的转折。

::: tip LLM 视角
在字符级小模型（3M 参数）上，top-k 和 top-p 的差异相对有限，因为词表只有 65 个字符（top-k=40 几乎等于无截断）。

在生产级 LLM（词表 50K–100K）上，top-p 和 top-k 的差异才会显著体现——分布尖锐时 top-p 只保留很少候选，top-k 却仍保留固定的 40 个，低质量候选混入的概率更高。
:::

---

## 十一、配套代码

| 文件 | 内容 | 关键接口 |
|------|------|---------|
| [`sampling_strategies.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/transformer_training/sampling_strategies.py) | 5 种采样策略并排对比 | `generate(model, encode_ids, max_new_tokens, strategy, **kwargs)` |
| [`gpt_train.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/transformer_training/gpt_train.py) | 内置 temperature + top_k + use_cache | `GPT.generate(idx, max_new_tokens, temperature, top_k, use_cache)` |

::: tip 跑一遍

确保已先训练好 checkpoint：

```bash
cd /path/to/llm-cookbook

# Step 1: 训练模型（如果还没跑过）
python ml_foundations/transformer_training/gpt_train.py

# Step 2: 运行 5 种采样策略对比
python ml_foundations/transformer_training/sampling_strategies.py
```

输出会展示同一 prompt "ROMEO:" 在 5 种策略下的 200-token 生成结果，差异一目了然。
:::

---

## 十二、生产实战经验总结

### 12.1 按任务选策略

| 任务类型 | 推荐配置 | 理由 |
|---------|---------|------|
| **创意写作** | `temperature=0.9, top_p=0.95` | 多样性优先，允许惊喜 |
| **代码生成** | `temperature=0.2` 或 greedy | 正确性优先，确定性强 |
| **机器翻译** | greedy 或 beam=4 | 有标准参考答案 |
| **对话助手** | `temperature=0.7–1.0, repetition_penalty=1.1` | 自然流畅，避免死循环 |
| **数学推理（CoT）** | `temperature=0.0`，多次采样取多数 | Self-consistency 技术 |
| **摘要生成** | `temperature=0.3–0.5, top_p=0.9` | 忠实内容，适度多样 |

### 12.2 常见陷阱

::: warning 常见误区 1：temperature 和 top_p 同时调
Anthropic 官方文档明确指出：**temperature 和 top_p 不要同时远离默认值**。两个参数都会改变分布，叠加效果难以预测。

实践建议：固定其中一个，只调另一个。
:::

::: warning 常见误区 2：top_k=1 不等于 greedy
`top_k=1` 会先截断到 1 个候选，再做 softmax（概率=1.0），再用 `multinomial` 采样。虽然结果相同，但实现路径和 `argmax` 不同，若代码有 bug 可能行为不一致。生产代码中应明确区分。
:::

::: tip LLM 视角
**OpenAI 的最佳实践**（来自官方文档）：
- 精确任务：`temperature=0`
- 平衡任务：`temperature=0.7`（默认值）
- 多样性任务：`temperature=1.0`，保持 `top_p=1.0`

**Anthropic 的最佳实践**：
- 分析/多选：`temperature=0`
- 数据提取：`temperature=0`
- 对话/翻译：`temperature=1`（默认）
- 创意写作：`temperature=1`
:::

### 12.3 调参顺序建议

```
1. 先确定任务类型（确定性 vs 创意）
2. 根据任务选 temperature 范围
3. 若出现重复 → 加 repetition_penalty=1.1
4. 若输出含低质量词 → 加 top_p=0.9 或 top_k=40
5. 组合调参时，每次只改一个参数
```

---

## 十三、延伸阅读

- **Holtzman et al. 2019** — "The Curious Case of Neural Text Degeneration"：top-p（核采样）的原始论文，证明最高概率序列不等于最自然序列，必读。[arXiv:1904.09751](https://arxiv.org/abs/1904.09751)

- **Leviathan et al. 2022** — "Fast Inference from Transformers via Speculative Decoding"：推测解码原始论文。[arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

- **Basu et al. 2020** — "Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity"：自适应温度的另一个方向，根据目标困惑度动态调整截断。[arXiv:2007.14966](https://arxiv.org/abs/2007.14966)

- **vLLM 文档 — Speculative Decoding**：[docs.vllm.ai](https://docs.vllm.ai/en/latest/features/spec_decode.html)，生产级推测解码配置指南。

- **HuggingFace Generation Guide**：[huggingface.co/blog/how-to-generate](https://huggingface.co/blog/how-to-generate)，覆盖所有解码策略的代码示例，与本文形成很好的对照。

- **Lilian Weng — "Decoding Strategies in Large Language Models"**：[lilianweng.github.io](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)，更多采样策略的数学推导和可视化。

---

> **上一节**：[GPT 训练全流程](./training) —— 损失下降、cosine warmup、AdamW
>
> **下一节**：[KV Cache 推理加速](./kv-cache) —— 从 $O(T^2)$ 到 $O(T)$，推理性能的关键
