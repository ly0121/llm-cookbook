# 完整训练流程

> AdamW + cosine warmup + grad clip —— ~3M 参数，2000 步，val loss 4.22 → 1.83，能写"伪莎士比亚"

---

## 一、为什么训练流程如此重要

同样的架构，**训练配方**可以决定模型能不能收敛、最终能力差多少。

```
同一个 Transformer 架构：

  糟糕的配方                  好的配方
  ─────────────              ─────────────
  固定 lr=1e-3               cosine warmup
  普通 Adam (β2=0.999)        AdamW (β2=0.95)
  无 grad clip               grad_clip=1.0
        ↓                          ↓
  loss 震荡 / 发散            稳定下降，val loss 1.83
```

### 历史教训：Pre-LN vs Post-LN

GPT-3 时代（2020 年）的一个关键决策：LayerNorm 放在哪里？

- **Post-LN**（Vaswani 2017 原版）：$x' = \text{LN}(x + \text{sublayer}(x))$
  在 1-3 层的小模型上没问题，但扩展到 96 层时，靠近输入的层梯度极小，需要非常精心的 warmup 策略才能训稳。

- **Pre-LN**（现代标准）：$x' = x + \text{sublayer}(\text{LN}(x))$
  梯度能畅通地通过残差"高速公路"，几十层叠加依然稳定。GPT-2、LLaMA、Qwen 全部采用 Pre-LN。

::: tip LLM 视角
**训练配方 = 架构 + 优化器 + 调度器 + 正则化的组合**

GPT-4 和你用相同代码写的玩具模型，架构本质相同。差距更多在于：
- 数据质量与数量
- 学习率调度的精细程度
- 分布式训练的稳定性技巧

掌握这一页的"小模型训练配方"，你就掌握了 LLM 训练的基本骨架。
:::

### 我们的 demo 成绩单

| 指标 | 数值 |
|------|------|
| 参数量 | ~3M |
| 训练步数 | 2000 steps |
| 初始 val loss | 4.22 (≈ ln 65，随机猜测) |
| 最终 val loss | **1.83** |
| MPS（Apple GPU） | ~30 秒 |
| CPU（M 系芯片） | ~5-6 分钟 |
| 生成质量 | 能写出有格律感的"伪莎士比亚" |

---

## 二、Transformer Block 装配

### Block 结构：Pre-LN 版本

一个 Block = LayerNorm + Attention + 残差 + LayerNorm + FFN + 残差

```
输入 x (B, T, d)
  │
  ├─ LN(x) ──▶ CausalSelfAttention ──▶ a
  │                                     │
  └───────────────────────────── + ──▶ x₁    ← 第一条残差
                                   │
  ┌──────────────────────────────  │
  │
  ├─ LN(x₁) ──▶ MLP ──▶ m
  │                       │
  └───────────────── + ──▶ x₂    ← 第二条残差（传入下一 Block）
```

### Pre-LN vs Post-LN 对比

| 顺序 | 公式 | 稳定性 | 适用范围 |
|------|------|--------|---------|
| Post-LN（原论文） | $x' = \text{LN}(x + \text{Attn}(x))$ | 大模型不稳，需大量 warmup | ≤12 层小模型 |
| Pre-LN（现代） | $x' = x + \text{Attn}(\text{LN}(x))$ | 深层稳定，默认配方 | **所有现代 LLM** |

::: tip LLM 视角
Pre-LN 的核心优势：**残差路径上没有 LayerNorm**。

梯度反向传播时，残差的直通路径（Identity Branch）保证了从输出层到输入层有一条"无障碍高速公路"，即使网络深达 96 层（GPT-3）或 128 层，梯度也不会消失到机器精度以下。
:::

### 残差连接的梯度意义

```
前向：  x ──▶ LN ──▶ Attn ──▶ +──▶ x'
        │                      ↑
        └──────────────────────┘  (residual)

反向梯度：
  ∂L/∂x = ∂L/∂x' × (1 + ∂Attn(LN(x))/∂x)
                    ^^^
                    残差保证 +1 项始终存在
```

无论 Attention 层的梯度有多小，+1 项确保梯度不会变成零。

### FFN 的"扩展-收缩"模式

本 demo 使用 GELU 激活：

$$
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x), \quad W_1 \in \mathbb{R}^{4d \times d},\ W_2 \in \mathbb{R}^{d \times 4d}
$$

扩展比 4× 是 Vaswani 2017 的经验值，GPT 系列沿用至今。

::: tip LLM 视角
**现代 LLM 用 SwiGLU 替换 GELU**（LLaMA / Qwen / Mistral）：

$$
\text{SwiGLU}(x) = (xW_1) \odot \text{SiLU}(xW_2) \cdot W_3
$$

其中 $\text{SiLU}(x) = x \cdot \sigma(x)$（平滑 ReLU 的变体），$\odot$ 是逐元素乘法（门控机制）。

SwiGLU 需要三个权重矩阵（比 GELU 多一个），所以 LLaMA 的 FFN 扩展比通常取 $\frac{8}{3}d \approx 2.67d$，保持总参数量与 GELU + 4× 相当。
:::

### 对应代码（gpt_train.py）

```python
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)      # Pre-LN 1
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)      # Pre-LN 2
        self.mlp = MLP(cfg)

    def forward(self, x, kv_cache=None):
        a, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + a                                # 残差 1
        x = x + self.mlp(self.ln2(x))           # 残差 2
        return x, new_cache
```

---

## 三、Tokenizer + 数据准备

### 字符级 Tokenizer

本 demo 使用最简单的字符级分词：每个字符是一个 token，词表大小约 65（莎士比亚语料的所有不重复字符）。

```
字符集（vocab≈65）：
  a b c d e f g h i j k l m n o p q r s t u v w x y z
  A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
  0-9  . , ! ? ; : ' " - ( ) \n 空格 ...
```

**为什么用字符级？** 教学目的：零依赖、词表固定、5 分钟内可见收敛效果。生产环境用 BPE（GPT-4 的 cl100k_base 词表约 10 万）。

### 数据切分

```python
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]   # 9:1 切分
```

Tiny Shakespeare 语料约 100 万字符，切分后：

- **训练集**：~90 万 token
- **验证集**：~10 万 token（完全不参与梯度更新，用于评估泛化能力）

### get_batch()：随机切片

```python
def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)
```

每次调用随机抽取 `batch_size` 个起始位置，切出长度为 `block_size` 的片段。注意 `y` 是 `x` 向右平移一位，因为训练目标是**预测下一个 token**。

```
x: [T  h  e  r  e  _  i  s  _  a]   (block_size=10 示意)
y: [ h  e  r  e  _  i  s  _  a  _]
    ↑ 每个位置预测下一个字符
```

**每步处理的 token 数**：`batch_size × block_size = 32 × 128 = 4096 token`

---

## 四、参数量估算

### 近似公式

对 decoder-only Transformer，忽略 bias 和 LayerNorm 参数，每层的主要参数来自：

- **Attention**：$W_Q, W_K, W_V, W_O$ 各 $d \times d$，共 $4d^2$
- **FFN**：$W_1 \in \mathbb{R}^{4d \times d}$，$W_2 \in \mathbb{R}^{d \times 4d}$，共 $8d^2$

合计每层 $12d^2$，总参数：

$$
\text{params} \approx 12 \cdot L \cdot d^2 \quad (\text{忽略 embedding})
$$

### 我们的 demo

$$
12 \times 6 \times 192^2 = 12 \times 6 \times 36864 \approx 2{,}654{,}208 \approx 2.65\text{M}
$$

加上 token embedding + position embedding（均为 $V \times d$ 或 $T \times d$，但 weight tying 使输出头与 tok_emb 共享）：

$$
\underbrace{65 \times 192}_{\text{tok\_emb（与输出头 weight tied）}} + \underbrace{128 \times 192}_{\text{pos\_emb}} \approx 12480 + 24576 \approx 37\text{K}
$$

**总计 ≈ 2.69M**，加上 LayerNorm 和 bias 后实际打印约 **~3M**。

### 与大模型对比

| 模型 | L | d | 公式估算 | 实际 |
|------|---|---|---------|------|
| 本 demo | 6 | 192 | 2.65M | ~3M |
| GPT-2 small | 12 | 768 | 85M | 124M |
| LLaMA-7B | 32 | 4096 | 6.4B | 7B（FFN 用 SwiGLU 另算） |
| LLaMA-70B | 80 | 8192 | 64B | 70B |

::: tip LLM 视角
**为什么 LLaMA-7B 的 FFN 不是 $8d^2$？**

LLaMA 用 SwiGLU + 扩展比 $\frac{8}{3}$，每层 FFN 参数约 $2 \times \frac{8}{3}d^2 + \frac{8}{3}d^2 \approx 8d^2$（三个矩阵×更小扩展比，最终接近标准 FFN），加上 Attention 的 $4d^2$，每层约 $12d^2$，与近似公式一致。所以"12Ld²"对 LLaMA 系列同样适用。
:::

---

## 五、AdamW 优化器配方（LLM 标配）

### 为什么不用普通 Adam？

Adam 把 L2 正则（weight decay）放进梯度里：

$$
g_t^{\text{Adam+L2}} = g_t + \lambda \theta_{t-1}
$$

问题：Adam 的自适应步长 $\frac{1}{\sqrt{\hat{v}_t}}$ 会**缩放**这个正则项，使得更新频繁的参数受到更弱的正则化，逻辑上不一致。

**AdamW** 把 weight decay 解耦，直接作用在参数上：

$$
\theta_t \leftarrow \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_{t-1}
$$

第二项的 $\lambda \theta_{t-1}$ 不经过自适应缩放，行为更"正直"。

### 完整更新公式

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad (\text{偏差修正})
$$

$$
\theta_t \leftarrow \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_{t-1}
$$

### LLM 标准超参

本 demo 的配置（对应 `gpt_train.py` 第 200 行）：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,            # 峰值学习率，适合 ~100M 以内的模型
    betas=(0.9, 0.95),  # 注意：β₂=0.95 而非默认 0.999
    weight_decay=0.1,   # 较强正则，LLM 训练标准值
)
```

| 超参 | 本 demo | GPT-3 / LLaMA | 说明 |
|------|---------|--------------|------|
| `lr` | 3e-4 | 1e-4 ~ 3e-4 | 小模型用大 lr |
| `β₁` | 0.9 | 0.9 | 一阶矩（动量） |
| `β₂` | 0.95 | 0.95 | 二阶矩；比默认 0.999 小 |
| `wd` | 0.1 | 0.1 | 解耦 weight decay |
| `ε` | 1e-8 | 1e-8 | 数值稳定项 |

::: tip LLM 视角
**为什么 LLM 用 β₂=0.95 而不是 Adam 默认的 0.999？**

$\beta_2$ 控制二阶矩 $v_t$ 的"记忆长度"：

- $\beta_2 = 0.999$：$v_t$ 约等于最近 1000 步梯度平方的平均，更新步长变化极慢
- $\beta_2 = 0.95$：约等于最近 20 步的平均，**对梯度量级的变化响应更快**

LLM 训练中，loss 可能在某些步骤出现"spike"（尖峰），$\beta_2=0.95$ 能更快适应这种变化，不至于因步长失调导致梯度爆炸。
:::

---

## 六、学习率调度：cosine warmup

### 三阶段设计

```
lr ↑
3e-4 ─            ╱‾‾╲
                 ╱    ╲___
                ╱         ╲___
               ╱               ╲___
3e-5 ─        ╱                    ╲_________
0    ─ ───────
     └──┬──────────────────────────────┬──▶ step
       100                           2000
     warmup         cosine decay         min_lr(10%)
```

**代码实现**：

```python
warmup = 100
def lr_lambda(step):
    if step < warmup:
        return step / warmup                               # 线性 warmup
    progress = (step - warmup) / (steps - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))  # cosine → 10%
```

peak lr = 3e-4，最终 min lr = 3e-5（峰值的 10%）。

### 为什么需要 Warmup？

训练开始时，Adam 的二阶矩 $v_t$ 是从零初始化的。前几步 $\hat{v}_t$ 的估计严重不准（偏低），会导致步长虚高：

$$
\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \approx \frac{g_t}{\sqrt{\epsilon}} \quad (v_t \approx 0)
$$

分母接近 $\sqrt{\epsilon} = 10^{-4}$，等效学习率被放大约 $10^4$ 倍——**直接爆炸**。

Warmup 通过前 100 步用极小的 lr 慢慢"热身"，让 $v_t$ 积累到合理估计值后，再全速训练。

::: tip LLM 视角
GPT-3 用了 **2000 步** warmup（总训练约 30 万步，warmup 占 0.7%）。LLaMA-3 则把 warmup 设为 2000 步但峰值 lr 更低。

一个粗糙的经验法则：**warmup steps ≈ 0.1% × total steps**，但对小模型（< 100M）即便 warmup 100 步（5%）也能工作。
:::

### 为什么 Cosine 比 Linear 更好？

```
Linear decay:         Cosine decay:
lr │╲                 lr │  ╲___
   │ ╲                   │      ╲___
   │  ╲                  │          ╲___
   │   ╲                 │              ╲___
   └────→ step           └──────────────────→ step
   等速下降，后期 lr      后期 lr 下降慢，
   可能太小，学不动       给模型更多"细调"机会
```

Cosine 在训练后期 lr 仍相对较大，模型有更多时间在最优解附近精细搜索。实验表明 cosine 比 linear 和 step decay 通常带来 0.1-0.3 个 val loss 的改善。

---

## 七、梯度裁剪 grad_clip

### 问题：梯度爆炸

即使有 warmup，某些训练步骤仍可能出现异常大的梯度（loss spike）。这会导致参数被更新到一个非常差的位置，后续很难恢复。

### 解决方案

对所有参数的梯度向量计算全局 L2 范数，超阈值时等比例缩放：

$$
\text{if } \|\nabla\|_2 > 1.0: \quad \nabla \leftarrow \nabla \times \frac{1.0}{\|\nabla\|_2}
$$

PyTorch 一行实现：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

放在 `loss.backward()` 之后、`optimizer.step()` 之前。

::: tip LLM 视角
**grad_clip=1.0 是 LLM 训练的行业标准**：

- GPT-3：max_grad_norm = 1.0
- LLaMA 1/2/3：max_grad_norm = 1.0
- Qwen 系列：max_grad_norm = 1.0
- Mistral：max_grad_norm = 1.0

几乎所有公开的 LLM 训练 config 都是 1.0。偶尔见到 0.5（更保守）或不设置（某些微调场景），但 1.0 是安全默认值。
:::

::: warning 不要忽略 grad clip
不加 grad clip 训练小模型通常侥幸没问题，但训练大模型（参数 > 1B）或序列较长时，偶发的梯度爆炸几乎是必然的。

典型症状：训练到 30-40% 时 loss 突然从正常跳到 10+，然后永远回不来——这就是没加 grad clip 的代价。
:::

---

## 八、训练循环（核心模板）

### 每步六个动作

```
step t:
  1. sample_batch   → x (B,T), y (B,T)
  2. model.forward  → loss = cross_entropy(logits, y)
  3. zero_grad      → 清除上一步残留梯度
  4. loss.backward  → 计算所有参数的梯度
  5. clip_grad_norm → 防止梯度爆炸
  6. optimizer.step → 更新参数
     sched.step     → 更新学习率
```

### 完整训练函数（gpt_train.py train() 核心）

```python
for step in range(steps):
    model.train()
    x, y = get_batch(train_data, cfg.block_size, batch_size, DEVICE)

    _, loss, _ = model(x, targets=y)          # forward
    optimizer.zero_grad(set_to_none=True)     # 清零梯度
    loss.backward()                           # backward
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # grad clip
    optimizer.step(); sched.step()            # 更新参数和 lr

    if step % eval_interval == 0 or step == steps - 1:
        model.eval()
        with torch.no_grad():
            x_val, y_val = get_batch(val_data, cfg.block_size, batch_size, DEVICE)
            _, val_loss, _ = model(x_val, targets=y_val)
        print(f"  step {step:5d}/{steps}  lr {sched.get_last_lr()[0]:.5f}"
              f"  train {loss.item():.4f}  val {val_loss.item():.4f}")
```

**注意**：`set_to_none=True` 比 `zero_grad()` 更高效——直接把 `.grad` 设为 None 而不是全零张量，节省内存写入。

### 实际训练日志

运行 `python gpt_train.py` 后输出类似：

```
  设备: mps
  语料: 1115394 字符  |  词表: 65
  模型: GPTConfig(block_size=128, vocab_size=65, n_layer=6, ...)
  参数量: 2.93M

  step     0/2000  lr 0.00000  train 4.2200  val 4.2200  (0.1s)
  step   200/2000  lr 0.00030  train 2.5900  val 2.6200  (3.2s)
  step   400/2000  lr 0.00029  train 2.3100  val 2.3800  (6.1s)
  step   600/2000  lr 0.00027  train 2.1500  val 2.2200  (9.0s)
  step   800/2000  lr 0.00024  train 2.0200  val 2.1000  (12.0s)
  step  1000/2000  lr 0.00020  train 1.9700  val 2.0100  (15.1s)
  step  1200/2000  lr 0.00016  train 1.9100  val 1.9500  (18.0s)
  step  1400/2000  lr 0.00012  train 1.8800  val 1.9200  (21.0s)
  step  1600/2000  lr 0.00009  train 1.8600  val 1.9000  (24.0s)
  step  1800/2000  lr 0.00005  train 1.8300  val 1.8600  (27.1s)
  step  1999/2000  lr 0.00003  train 1.8100  val 1.8300  (30.0s)
```

::: tip 理解这张日志
- **step 0 的 train loss ≈ 4.22**：理论随机猜测值是 $\ln(65) \approx 4.17$，非常接近，说明初始化正确。
- **step 200 的 lr ≈ 0.00030**：恰好是 warmup 结束、峰值学习率。
- **val loss 始终比 train loss 略高**：属于正常现象，差值在 0.02-0.05 之间说明没有严重过拟合。
- **后期 lr 降低时 loss 下降放缓**：这是 cosine decay 的预期行为，模型在精细优化。
:::

---

## 九、Loss 曲线解读

### ASCII Loss 曲线

gpt_train.py 训练结束后会打印如下曲线（T=train，V=val）：

```
  loss
  4.22 │ T
       │  V
  3.50 │    T
       │     V
  2.75 │       T
       │        V
  2.25 │          T
       │           V
  2.00 │             T
       │              V
  1.90 │               T
       │                V
  1.83 │                 T
       │                  V
       └──────────────────────▶ step
         0  200 400 ...      1999
         T=train  V=val
```

### 三种典型形态

**正常收敛**（本 demo 的情况）：

```
loss
  │ ╲
  │  ╲___
  │      ╲___
  │          ╲___________
  └─────────────────────▶ step
  train ≈ val，差值稳定 < 0.1
```

**欠拟合**（模型不够大，或步数太少）：

```
loss
  │ ╲
  │  ╲___________  ← val loss 停在高处不动
  │               ─────────────────
  └─────────────────────────────▶ step
  解决：增加 params（n_layer/n_embd），或增加训练步数
```

**过拟合**（数据太少，或模型太大）：

```
loss
  │ ╲          train ─────────────▶
  │  ╲___
  │      ╲___  val 先降后升 ↗
  │              ╲__________╱‾‾‾‾‾
  └──────────────────────────────▶ step
  解决：增加 dropout，减少训练步数，增加 weight_decay
```

### 我们的 demo 为何轻微过拟合？

Tiny Shakespeare 仅约 100 万字符，~3M 参数模型很容易"记住"部分数据。

$$
\text{token-per-param} = \frac{1{,}000{,}000 \text{ tokens}}{3{,}000{,}000 \text{ params}} \approx 0.33 \text{ tokens/param}
$$

Chinchilla 推荐 20 tokens/param，我们只有约 1/60，必然处于数据欠缺区。但这完全符合教学目的——用小数据快速演示训练动态。

---

## 十、缩放定律 Chinchilla（preview）

### Kaplan 缩放定律（2020）

OpenAI 的 Kaplan 等人发现，模型 loss 遵循幂律：

$$
L(N, D) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty
$$

其中 $N$ 是参数量，$D$ 是训练 token 数，$\alpha_N \approx \alpha_D \approx 0.076$，$L_\infty$ 是理论下界。

这意味着：给定固定算力，**增大模型比增加数据"回报率"更高**——GPT-3 / Gopher 的设计逻辑。

### Chinchilla 修正（2022）

DeepMind 的 Hoffmann 等人更精细地实验后发现，Kaplan 的结论低估了数据的重要性：

$$
\text{最优配置：tokens} \approx 20 \times \text{params}
$$

给定计算预算 $C$（FLOPs），应该平均分配给参数量和训练数据：

| 模型 | 参数 | 训练 token | 实际 tokens/param | Chinchilla 建议 |
|------|------|-----------|-----------------|----------------|
| Gopher-280B | 280B | 300B | 1.07 | 5600B（欠训！）|
| GPT-3 | 175B | 300B | 1.71 | 3500B（欠训！）|
| Chinchilla | 70B | 1.4T | **20** | 1400B（最优）|
| LLaMA-3-8B | 8B | 15T | **1875** | 有意"过训" |

::: tip LLM 视角
**"过训练"（over-trained）是现代小模型的策略**

LLaMA-3-8B 用 15 万亿 token 训练（每参数 1875 token，是 Chinchilla 最优的 94 倍！）。这看起来"浪费"，但逻辑很清晰：

- **推理成本**远大于**训练成本**（一个模型要被调用数十亿次）
- 用更多数据训更小的模型 → 推理时每次调用更便宜
- 虽然单次训练效率低于 Chinchilla 最优，但**总部署成本**更低

这就是为什么 Meta 用 180 亿倍算力比 Chinchilla 最优"多烧"在 LLaMA-3-8B 上——他们在为未来的推理成本买单。
:::

### 本 demo 的定位

```
tokens/param ≈ 0.33（~60× 低于 Chinchilla 最优）

              Chinchilla 最优
                     ↓
  0.33   1    5    20   50   100
  ├──────┼────┼────┼────┼────┼────▶ tokens/param
  ↑
 本 demo
（严重欠数据，但能演示训练动态）
```

本 demo 的目的不是追求最优 loss，而是**在 5 分钟内演示完整的训练流程**。

---

## 十一、生成验证

### 训练后立即生成

训练结束后，`train()` 函数自动用 prompt `"ROMEO:"` 生成 200 个字符：

```python
prompt = torch.tensor([encode("ROMEO:")], dtype=torch.long, device=DEVICE)
out = model.generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40)
print(decode(out[0].tolist()))
```

### 对比：训练前 vs 训练后

**训练前（随机初始化）**：

```
ROMEO:zQxW!pKmN?vR;eLjT.fGhYdSoUcBaIwXqV
mKnZeRjLpQ!tWfYsSvGhUiOdCaXb?wNkPmTqV...
```

完全随机字符串，无任何统计规律。

**训练后（2000 步）**：

```
ROMEO:
I will not be so strangely,
And the world shall be more
Than the king of men be so.
JULIET:
What is the world, my lord?
ROMEO:
The world is the world, my lord,
And the world is the world...
```

虽然语义有重复（"world"出现频繁），但：
- 单词拼写基本正确
- 标点符号合理
- 对话格式（"名字:" + 换行）已学会
- 诗歌格律隐约可见

这是 val loss 1.83 对应的字符级生成质量。

### 训练成功的评判标准

| 指标 | 未达标 | 达标 |
|------|--------|------|
| val loss | > 2.5（欠拟合）或比 train loss 高 > 0.5（过拟合） | **1.7 - 2.0** |
| 生成文本 | 随机字符串 | 英文单词 + 合理标点 |
| 对话格式 | 不出现 | 偶尔出现 |
| 训练时间 | 超过 10 分钟（CPU） | < 6 分钟（CPU） |

---

## 十二、配套代码

| 文件 | 内容 |
|------|------|
| [`gpt_train.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/transformer_training/gpt_train.py) | 完整训练循环，导出 `GPT`, `GPTConfig`, `load_checkpoint`, `encode`, `decode` |

::: tip 跑一遍

```bash
# 先下载 Tiny Shakespeare（约 1MB）
curl -L -o ml_foundations/transformer_training/data/tiny_shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 训练（MPS ~30s，CPU ~5-6min）
cd ml_foundations/transformer_training
python gpt_train.py

# 验证产物
ls data/
# 应该看到：ckpt.pt  loss_log.json  tiny_shakespeare.txt
```

训练完成后可以直接运行 `sampling_strategies.py`、`attention_visualization.py`、`kv_cache.py` 这三个 demo，它们都依赖 `ckpt.pt`。
:::

::: tip LLM 视角
**weight tying（权重绑定）是为什么？**

```python
self.head.weight = self.tok_emb.weight  # gpt_train.py 第 118 行
```

输出头（把 $d$ 维向量映射到 vocab 大小的 logits）和输入 embedding（把 token id 映射到 $d$ 维向量）共享同一矩阵。

两个理由：
1. **节省参数**：vocab × d 的矩阵只存一份（本 demo 节省约 12K 参数，大模型能省数亿）
2. **语义一致性**：同一个 token 在 "输入语义" 和 "输出偏好" 上应该是对应的，共享权重施加了这个归纳偏置

GPT-2、LLaMA 都使用 weight tying。
:::

---

## 十三、延伸阅读

- Vaswani et al. **"Attention Is All You Need"** (2017)，训练附录部分：warmup 策略的最早描述
- Andrej Karpathy [**nanoGPT**](https://github.com/karpathy/nanoGPT)：本 demo 的直接参考，极简高质量实现
- Kaplan et al. **"Scaling Laws for Neural Language Models"** (2020)：第一个系统描述 LLM 幂律缩放的工作
- Hoffmann et al. **"Training Compute-Optimal Large Language Models"** (Chinchilla, 2022)：纠正 Kaplan 结论，数据和参数同等重要
- Black et al. **"GPT-NeoX-20B"** (2022)：最佳工程实践集合，含完整的训练超参配方

---

> **下一站**：[采样策略](./sampling) —— 模型训完了，如何让它"聪明地说话"？greedy / top-k / top-p 的选择艺术。
