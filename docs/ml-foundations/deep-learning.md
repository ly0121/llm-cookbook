# 深度学习基础

> 张量、自动微分、神经网络 —— LLM 的"骨架"

---

## 一、为什么需要深度学习？

经典 ML 在结构化数据上表现优异，但遇到 **图像、语音、文本** 这类高维非结构化数据，瓶颈出现在 **特征工程**：

```
经典 ML 处理图像：              深度学习处理图像：
─────────────────              ─────────────────
人工设计 SIFT/HOG 特征          原始像素 → 多层网络自动学特征
↓                              ↓
喂给 SVM/RF                    端到端训练
↓                              ↓
精度有限（专家瓶颈）              精度突破（特征学习）
```

**深度学习的核心赌注**：用足够多的层 + 足够多的数据 + 足够强的算力，让模型**自己**学到比人类设计更好的特征。

---

## 二、张量（Tensor）：深度学习的基本数据结构

### 2.1 什么是张量？

张量就是 **多维数组**：

| 阶数 | 形状 | 例子 |
|------|------|------|
| 0 阶（标量） | `()` | 损失值 `0.27` |
| 1 阶（向量） | `(d,)` | 词嵌入 `(768,)` |
| 2 阶（矩阵） | `(N, d)` | 一个 batch 的嵌入 `(32, 768)` |
| 3 阶 | `(B, T, d)` | LLM 的隐状态 `(batch, seq_len, hidden)` |
| 4 阶 | `(B, C, H, W)` | CNN 的图像 batch |

::: tip LLM 视角
LLM 内部所有数据都是张量：
- 输入 token ID：`(B, T)`
- 嵌入后：`(B, T, d_model)`
- 注意力得分：`(B, n_heads, T, T)`
- 输出 logits：`(B, T, V)`
:::

### 2.2 PyTorch 基本操作

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape)         # torch.Size([2, 2])
print(x.dtype)         # torch.float32
print(x @ x)           # 矩阵乘法
print(x.sum(dim=0))    # 沿第 0 维求和 → tensor([4., 6.])
```

### 2.3 广播（Broadcasting）

不同形状的张量做运算时，自动按规则对齐：

```
A.shape = (3, 1)        B.shape = (1, 4)
        ↓                       ↓
        ╱──────────────────────╲
                  ↓
        广播为 (3, 4) 后逐元素运算
```

LLM 中的位置编码加法、注意力 mask 都依赖广播。

---

## 三、自动微分（Autograd）：神经网络的"训练引擎"

### 3.1 反向传播是什么？

反向传播 = **链式法则的高效实现**。

考虑一个最简单的复合函数：

$$
y = f(g(h(x)))
$$

求 $\frac{\partial y}{\partial x}$ 时：

$$
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}
$$

**autograd** 的工作就是：
1. **前向**：记录所有运算（构建计算图）
2. **反向**：从 loss 出发，沿图反向应用链式法则
3. 把梯度存到每个 `requires_grad=True` 的张量的 `.grad` 上

### 3.2 PyTorch 的 autograd

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()      # 触发反向传播
print(x.grad)     # dy/dx = 2x + 3 = 7.0
```

::: tip LLM 视角
训练 70 亿参数的 Llama-7B 时，autograd 自动构建 **数千万个节点的计算图**，逐层反向传播 → 给每个参数算出梯度 → 优化器更新。

如果没有 autograd，每加一种新算子都要手推梯度公式，是不可想象的。
:::

### 3.3 计算图与梯度

```
前向：    x ──[× 2]──> a ──[+ 3]──> y
                        ↓
                       loss

反向：    grad_x ←─[× 2]── grad_a ←─[× 1]── grad_y=1
```

每个算子注册 **forward** 和 **backward** 两套实现，autograd 自动串联。

---

## 四、多层感知机（MLP）：最基础的神经网络

### 4.1 单层结构

$$
h = \sigma(Wx + b)
$$

- $W \in \mathbb{R}^{d_{out} \times d_{in}}$：权重矩阵
- $b \in \mathbb{R}^{d_{out}}$：偏置
- $\sigma$：激活函数

### 4.2 为什么需要激活函数？

::: warning 关键洞察
**没有非线性激活，多层网络等价于单层！**

$W_2(W_1 x) = (W_2 W_1) x = W' x$ —— 还是线性。
:::

### 4.3 常见激活函数

| 函数 | 公式 | 特点 | 何时用 |
|------|------|------|--------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | 输出 [0,1]；梯度消失 | 二分类输出 |
| **Tanh** | $\tanh(x)$ | 输出 [-1,1]；同样会消失 | RNN 隐藏层 |
| **ReLU** | $\max(0, x)$ | 计算快；可能"死亡" | CNN/MLP 主流 |
| **GELU** | $x \cdot \Phi(x)$ | 平滑 ReLU | Transformer 主流 |
| **SwiGLU** | $\text{Swish}(xW) \odot (xV)$ | 门控变体 | LLaMA / GPT 系列 |

::: tip LLM 视角
现代 LLM 普遍用 **GELU / SwiGLU**：
- BERT、GPT-2 用 GELU
- LLaMA、Mistral、Qwen 用 SwiGLU（精度更高，参数量略增）
:::

### 4.4 MLP in PyTorch

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)
```

::: tip LLM 关联
**Transformer 的 FFN 子层就是一个两层 MLP**：

$$
\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2
$$

通常 $d_{hidden} = 4 \cdot d_{model}$（如 LLaMA-7B 中 $d_{model}=4096$，FFN 内部维度 $11008$，约 2.7×）。

**LLM 一半以上的参数在 FFN 中**。
:::

---

## 五、训练循环：神经网络是怎么"学"的

### 5.1 五步标准流程

```python
for epoch in range(num_epochs):
    for x, y in dataloader:
        # 1. 前向：算出预测
        y_hat = model(x)

        # 2. 算损失
        loss = criterion(y_hat, y)

        # 3. 清零旧梯度（PyTorch 默认累加）
        optimizer.zero_grad()

        # 4. 反向：算梯度
        loss.backward()

        # 5. 更新参数
        optimizer.step()
```

### 5.2 损失函数

| 任务 | 损失 | PyTorch |
|------|------|---------|
| 回归 | MSE | `nn.MSELoss()` |
| 二分类 | BCE | `nn.BCEWithLogitsLoss()` |
| 多分类 | 交叉熵 | `nn.CrossEntropyLoss()` |
| LLM next-token | 交叉熵 | `nn.CrossEntropyLoss()` |

### 5.3 优化器

| 优化器 | 思想 | 特点 |
|--------|------|------|
| **SGD** | $\theta \leftarrow \theta - \eta \nabla L$ | 简单；需调 lr |
| **SGD + Momentum** | 累积历史梯度方向 | 收敛更快 |
| **Adam** | 自适应学习率 + 动量 | 几乎无需调参，默认首选 |
| **AdamW** | Adam + 解耦的权重衰减 | **LLM 训练标配** |

::: tip LLM 视角
训练 LLM 几乎清一色用 **AdamW**：
- $\beta_1 = 0.9, \beta_2 = 0.95$（注意：通常不是 0.999）
- 学习率：cosine 衰减 + warmup（前 2000 步）
- weight_decay = 0.1

AdamW 把 L2 正则从梯度里拆出来单独处理，避免和 Adam 的自适应步长冲突。
:::

### 5.4 学习率：最重要的超参数

```
loss
  │ ╲           lr 太大 → 震荡发散
  │  ╲
  │   ╲___      lr 合适 → 稳定下降
  │       ───
  │           ─────  lr 太小 → 学不动
  └────────────────→ step
```

**LLM 训练用 cosine + warmup**：

```
lr ↑
   │     ╱──╲___
   │    ╱       ╲___
   │   ╱             ╲___
   │  ╱                   ╲___
   │ ╱                         ╲
   └─────────────────────────────→ step
   warmup    主训练（cosine 退火）
```

---

## 六、卷积神经网络（CNN）

### 6.1 卷积操作

```
输入特征图              卷积核(3x3)         输出特征图
┌─────────────┐         ┌───────┐          ┌───────────┐
│  pixels...  │   ⊗    │ filter│   →      │ activations│
│             │         │       │          │            │
└─────────────┘         └───────┘          └───────────┘
   局部连接 + 权重共享 → 捕获局部模式（边缘、纹理）
```

### 6.2 三大优势

1. **局部连接**：每个神经元只看局部区域 → 参数大幅减少
2. **权重共享**：同一个 filter 扫过整张图 → 平移不变性
3. **层次特征**：浅层学边缘 → 中层学纹理 → 深层学语义

### 6.3 现代 CNN 演进

```
LeNet (1998) → AlexNet (2012) → VGG (2014) → ResNet (2015) → EfficientNet (2019)
   2 层卷积      8 层 + ReLU     19 层      152 层(残差)        架构搜索
```

**ResNet 的"残差连接"**：$h = f(x) + x$ → 让深层网络可训练。

::: tip LLM 关联
**Transformer 也用残差连接**：

```
x → LayerNorm → Attention → + → LayerNorm → FFN → +
↓                            ↑                    ↑
└────────── residual ────────┘                    │
                                                  │
↓─────────────── residual ────────────────────────┘
```

没有残差，深层 Transformer（如 GPT-3 的 96 层）根本训练不了。
:::

---

## 七、循环神经网络（RNN）与 LSTM

### 7.1 RNN：处理序列

每一步把 **前一时刻的隐状态** 喂给当前时刻：

```
   x₁    x₂    x₃    ...   xₜ
   ↓     ↓     ↓           ↓
┌───┐ ┌───┐ ┌───┐       ┌───┐
│RNN│→│RNN│→│RNN│→ ... →│RNN│
└───┘ └───┘ └───┘       └───┘
   ↓     ↓     ↓           ↓
   h₁    h₂    h₃          hₜ
```

数学：

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b)
$$

### 7.2 RNN 的根本问题：梯度消失

反向传播沿时间展开，梯度要乘 $T$ 次 $W_h$：

$$
\frac{\partial L}{\partial h_1} \propto \prod_{t=2}^{T} W_h \cdot \tanh'(\cdot)
$$

- $|W_h \cdot \tanh'| < 1$ → 梯度指数级**消失** → 学不到长期依赖
- $|W_h \cdot \tanh'| > 1$ → 梯度指数级**爆炸** → 训练不稳定

### 7.3 LSTM：用门控对抗梯度消失

LSTM 引入 **细胞状态 $C_t$** + 三道门：

```
       ┌────── 遗忘门 ftt ──────┐
xₜ ───┤                        │
       │  C_{t-1} ─ ⊗ ─── + ───→ Cₜ
       │           ↑     ↑      │
       └─ 输入门 iₜ┘     │      │
                         │      │
                    候选 c̃ₜ     │
                                │
       ┌─ 输出门 oₜ ──── ⊗ ─────→ hₜ
       └────────────────────────┘
```

门控让网络**有选择地**记住或遗忘信息 → 能捕获更长依赖。

### 7.4 RNN 时代到 Transformer 时代

| | RNN/LSTM | Transformer |
|--|---------|-------------|
| 并行性 | ❌ 串行（必须等 $h_{t-1}$） | ✅ 全序列并行 |
| 长程依赖 | 弱（即便 LSTM） | 强（自注意力直接连接） |
| 算力利用 | GPU 利用率低 | 几乎打满 |
| 训练规模 | 难超 1B 参数 | 轻松 100B+ |

::: tip LLM 关联
**为什么 Transformer 取代了 RNN？**
1. **并行**：RNN 必须按时间步走；Transformer 一次算完整个序列 → 训练快几十倍
2. **全局视野**：自注意力让任意两个 token 直接交互（O(1) 距离）
3. **可扩展**：堆到 100+ 层依然能训

但 RNN 的精神在 **状态空间模型（Mamba、RWKV）** 中复活，正在挑战 Transformer 的 O(n²) 注意力。
:::

---

## 八、正则化：防止过拟合

### 8.1 Dropout

训练时随机"丢弃"一部分神经元：

```
全连接               Dropout (p=0.5)
○────○              ○────●
○────○              ●────○      ●：被丢弃
○────○              ○────○
```

强制网络不依赖任何单个神经元 → 等价于训练**指数级的**子网络集成。

```python
self.dropout = nn.Dropout(p=0.1)  # LLM 微调常用 0.1
```

### 8.2 LayerNorm vs BatchNorm

| | BatchNorm | LayerNorm |
|--|----------|-----------|
| 归一化维度 | 沿 batch | 沿特征 |
| 公式 | $(x - \mu_{\text{batch}}) / \sigma_{\text{batch}}$ | $(x - \mu_{\text{feat}}) / \sigma_{\text{feat}}$ |
| 依赖 batch 大小 | 是 | 否 |
| 主流场景 | CNN（图像） | **Transformer**（NLP） |

::: tip LLM 视角
**所有 LLM 都用 LayerNorm（或其变体 RMSNorm）**：
- LLaMA 用 **RMSNorm**：$\frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}$（去掉减均值，更快）
- Pre-LN（在 sublayer 之前归一化）让深层 Transformer 训练更稳

为什么不用 BatchNorm？
- LLM 序列长度可变 → batch 内统计量不稳
- LayerNorm 只看自己这条样本，与 batch 无关
:::

### 8.3 早停（Early Stopping）

```
loss
  │  train ────────────────
  │       ╲
  │  val   ╲___╱──── 过拟合
  │            ↑
  │         早停点
  └────────────────→ epoch
```

监控验证集 loss，连续 N 个 epoch 不下降则停止。

---

## 九、深度学习与 LLM 的对应关系总结

| 经典深度学习 | LLM 中的对应 |
|-----------|-------------|
| 张量 (B, C, H, W) | 张量 (B, T, d) |
| MLP（多层感知机） | Transformer 的 FFN 子层 |
| 残差连接（ResNet） | 每个 sublayer 后的 `x + sublayer(x)` |
| BatchNorm | LayerNorm / RMSNorm |
| Dropout | 微调时常用，预训练大模型几乎不用 |
| ReLU | GELU / SwiGLU |
| Adam | AdamW（解耦权重衰减） |
| LSTM 隐状态 | KV cache（推理时复用历史） |
| 卷积的"权重共享" | 注意力的"参数与位置无关" |

---

## 十、配套代码

| 文件 | 演示主题 |
|------|---------|
| [`pytorch_basics.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/deep_learning/pytorch_basics.py) | 张量 / autograd / 手推梯度下降 |
| [`mlp_from_scratch.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/deep_learning/mlp_from_scratch.py) | NumPy 手写反向传播（XOR 问题） |
| [`mlp_pytorch.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/deep_learning/mlp_pytorch.py) | PyTorch MLP 训练 MNIST 子集 |
| [`cnn_mnist.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/deep_learning/cnn_mnist.py) | 简化 LeNet + 卷积核可视化 |
| [`rnn_lstm.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/deep_learning/rnn_lstm.py) | 字符级 LSTM 语言模型 + 文本生成 |

---

## 十一、延伸阅读

- Goodfellow et al. *Deep Learning* —— 神书（"花书"）
- 李沐《动手学深度学习》—— 中文最佳实战
- Andrej Karpathy [Zero to Hero](https://karpathy.ai/zero-to-hero.html) —— 从 micrograd 到 GPT
- Chris Olah [colah's blog](https://colah.github.io/) —— LSTM/CNN 可视化神作

> **下一站**：[NLP 经典基础](./nlp-foundations) —— 在 Transformer 之前，文本是怎么处理的？
