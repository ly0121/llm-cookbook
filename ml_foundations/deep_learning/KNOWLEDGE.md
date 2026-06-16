# 深度学习基础（Deep Learning）

> 张量、反向传播、优化器、CNN、RNN —— 通向 Transformer 与 LLM 的必经之路

---

## 一、为什么需要深度学习？

经典 ML（线性回归、SVM）在**特征已经足够好**时表现优秀。但真实世界的数据：

- 一张图像有 224×224×3 = 150528 个像素
- 一段文本是任意长的离散符号序列
- 一段音频是 16kHz × 数秒 = 数万采样点

**手工设计特征几乎不可能**。深度学习的核心思想：**让模型自动从原始数据中学出层次化的特征**。

```
浅层特征  →  中层特征  →  高层特征
边缘/角点    眼睛/嘴巴    人脸
字母组合    词汇语义     句子含义
```

> **LLM 视角**：Transformer 的 N 层 Attention + FFN 就是这种"特征逐层抽象"的极致版。每一层都在更高的语义层次上重新组织 token 的表示。

---

## 二、张量（Tensor）：所有计算的载体

### 2.1 维度直觉

| 维度 | 名称 | 例子 |
|------|------|------|
| 0D | 标量 | `loss = 0.42` |
| 1D | 向量 | `embedding = [0.1, -0.3, ...]`（768 维） |
| 2D | 矩阵 | 一个 batch 的 embedding `(B, D)` |
| 3D | 立方体 | 一段文本 `(B, T, D)` — batch × seq_len × dim |
| 4D | 图像 batch | `(B, C, H, W)` — batch × 通道 × 高 × 宽 |

LLM 中最常见的张量形状：`(batch_size, seq_len, hidden_dim)`，例如 `(8, 2048, 4096)`。

### 2.2 关键操作

```python
import torch

x = torch.randn(2, 3)         # 创建 2×3 张量
y = x.T                       # 转置 → 3×2
z = x @ y                     # 矩阵乘 → 2×2
z = x.reshape(6)              # 展平 → (6,)
z = x.unsqueeze(0)            # 加一维 → (1, 2, 3)
z = x.sum(dim=1)              # 沿第二维求和 → (2,)
```

**广播（broadcasting）**：形状不同的张量按"对齐尾部维度"自动扩展：

```
(3, 1) + (1, 4)  →  (3, 4)
(B, 1, D) + (B, T, D)  →  (B, T, D)   ← 经常出现在 attention 实现中
```

---

## 三、自动微分（Autograd）：神经网络能训练的关键

### 3.1 计算图（Computational Graph）

任何复杂运算都可以拆成基本运算（+、*、relu、softmax 等）的组合，构成有向无环图：

```
   x ──┐
        ├──> mul ──> a ──┐
   w ──┘                  ├──> add ──> y ──> loss
   b ──────────────────────┘
```

### 3.2 链式法则（反向传播的数学本质）

$$
\frac{\partial \text{loss}}{\partial w} = \frac{\partial \text{loss}}{\partial y} \cdot \frac{\partial y}{\partial a} \cdot \frac{\partial a}{\partial w}
$$

**关键洞察**：每个节点只需要知道
- 自己的"局部梯度"（如 `mul` 节点：`d(a)/d(x) = w`）
- 上游传下来的全局梯度

就能算出对自己输入的梯度。这种"反向传播"算法让 1 亿参数模型也能高效训练。

### 3.3 PyTorch 实现

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3            # y = 8
y.backward()          # 自动求导
print(x.grad)         # 12 (= 3*x^2 at x=2)
```

只要每个操作都支持反向传播，**任意复杂的网络都能自动算梯度** —— 这就是 PyTorch / TensorFlow / JAX 的基石。

---

## 四、神经网络的基本块：MLP

### 4.1 一层全连接（Linear / Dense）

$$
y = \sigma(Wx + b)
$$

- $W \in \mathbb{R}^{d_{out} \times d_{in}}$：可学习的权重矩阵
- $b \in \mathbb{R}^{d_{out}}$：偏置
- $\sigma$：激活函数（ReLU / GELU / Sigmoid）

### 4.2 为什么需要非线性？

如果所有层都是线性的，多层堆叠等价于一层：
$W_2(W_1 x + b_1) + b_2 = W_2 W_1 x + (W_2 b_1 + b_2) = W' x + b'$

**激活函数引入非线性**，让深度网络能拟合任意函数（万能逼近定理）。

| 激活 | 公式 | 特点 |
|------|------|------|
| Sigmoid | $1/(1+e^{-x})$ | 老古董，梯度消失严重 |
| Tanh | $(e^x - e^{-x})/(e^x + e^{-x})$ | 0 均值，比 sigmoid 好 |
| **ReLU** | $\max(0, x)$ | 现代默认，简单快 |
| **GELU** | $x \cdot \Phi(x)$ | LLM 用，光滑版 ReLU |
| **SwiGLU** | $\text{Swish}(xW_1) \odot (xW_2)$ | LLaMA 用，门控更强 |

### 4.3 训练循环（核心模板）

```python
for epoch in range(num_epochs):
    for x, y in dataloader:
        y_pred = model(x)              # 前向
        loss = criterion(y_pred, y)    # 计算损失
        optimizer.zero_grad()          # 清梯度（防累加）
        loss.backward()                # 反向传播
        optimizer.step()               # 更新参数
```

> **LLM 训练用的是同样的循环**，只是 model 是几百亿参数的 Transformer，dataloader 处理的是几万亿 token。

---

## 五、优化器：SGD → Adam

### 5.1 SGD（Stochastic Gradient Descent）

$$
w \leftarrow w - \eta \cdot \nabla_w L
$$

简单但收敛慢、对学习率敏感、容易卡在鞍点。

### 5.2 加 Momentum

$$
v \leftarrow \beta v + \nabla_w L,\quad w \leftarrow w - \eta v
$$

像球滚下山有惯性，能冲出小坑、加速平坦区。

### 5.3 Adam（自适应学习率）

对每个参数维护两个统计量：
- 一阶矩 $m$（梯度平均）
- 二阶矩 $v$（梯度平方平均）

$$
w \leftarrow w - \eta \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

效果：梯度大的方向自动减小步长，梯度小的方向自动加大步长。**LLM 训练几乎都用 AdamW**（Adam + 解耦的 weight decay）。

### 5.4 学习率调度（Scheduling）

LLM 训练标准做法：
- **Warmup**：前 1-5% 步骤线性升到峰值学习率
- **Cosine Decay**：之后按余弦曲线降到 0

```
lr ↑
   ╱──╲___
  ╱      ──╲___
 ╱            ──╲___
└────────────────────→ steps
warmup    cosine decay
```

---

## 六、CNN：卷积神经网络

### 6.1 卷积操作直觉

把一个小的"滤波器"（如 3×3）在图像上滑动，每个位置做点积：

```
图像:                      滤波器:        输出特征图:
[1 2 3 4]                  [1 0]
[5 6 7 8]    *             [0 1]    =     [7  9 11]
[9 0 1 2]                                  [5  7  9]
[3 4 5 6]
```

### 6.2 核心思想

| 性质 | 含义 |
|------|------|
| **局部性** | 一个像素只与周围像素有关，不需要全连接 |
| **平移不变性** | 同一个滤波器在所有位置共享参数 |
| **层次抽象** | 浅层学边缘，深层学物体 |

### 6.3 与 Transformer 对比

CNN 和 Transformer 都是"特征提取器"，但归纳偏置不同：

|  | CNN | Transformer |
|---|---|---|
| 归纳偏置 | 局部性 + 平移不变性 | 全局交互（attention） |
| 上下文窗口 | 受限（感受野） | 任意（自注意力） |
| 数据效率 | 小数据更好 | 需要大数据 |
| 长序列 | 弱 | 强（直到 2024） |

---

## 七、RNN / LSTM：处理序列数据

### 7.1 RNN 基本结构

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b)
$$

每一步都把"上一步的隐状态"和"当前输入"融合，得到当前隐状态。

```
x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃
              ↑              ↑              ↑
              x₂             x₃             ...
```

### 7.2 RNN 的两大问题

1. **梯度消失/爆炸**：长序列下梯度连乘，要么趋 0 要么炸
2. **无法并行**：必须按时间步串行计算

### 7.3 LSTM：用门控解决梯度问题

```
遗忘门 fₜ = σ(W_f [h_{t-1}, x_t])    决定丢掉多少旧记忆
输入门 iₜ = σ(W_i [h_{t-1}, x_t])    决定写入多少新信息
输出门 oₜ = σ(W_o [h_{t-1}, x_t])    决定输出多少
```

### 7.4 RNN/LSTM 为什么被 Transformer 取代？

- **并行性**：Transformer 一次处理整个序列，训练速度快几十倍
- **长程依赖**：注意力直接连接任意两个位置
- **可扩展**：堆 96 层 RNN 几乎不可能，Transformer 行

但 RNN 的精神仍然存在：
- **状态空间模型 / Mamba** 是改进的 RNN，2024 重新流行
- **KV-cache** 在推理时让 Transformer "退化"成类似 RNN 的逐步生成

---

## 八、正则化：防止过拟合

### 8.1 Dropout

训练时随机丢弃一定比例的神经元（如 10%-50%），强迫网络不依赖某个具体单元：

```
训练:  [a][b][c][d][e]
        ↓ dropout(0.4)
       [a][_][c][_][e]   ← 随机置零
推理:  [a*0.6][b*0.6]... ← 缩放补偿
```

### 8.2 Weight Decay（L2 正则）

每步更新时把权重略微往 0 拉：

$$
w \leftarrow w - \eta(\nabla L + \lambda w)
$$

LLM 训练标准超参：`weight_decay=0.1`。

### 8.3 LayerNorm / RMSNorm

不是严格意义的正则化，但稳定训练必不可少：

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
$$

LLM 几乎都用 RMSNorm（去掉减均值，只除 RMS），更高效：

$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2}}
$$

---

## 九、本目录 demo 速查

| 文件 | 主题 | 关键 API |
|------|------|---------|
| `pytorch_basics.py` | 张量 + autograd | `torch.tensor`, `requires_grad`, `.backward()` |
| `mlp_from_scratch.py` | 纯 NumPy MLP | 手写前向 + 反传，验证 XOR |
| `mlp_pytorch.py` | PyTorch MLP | `nn.Module`, MNIST 子集训练 |
| `cnn_mnist.py` | LeNet 风格 CNN | `nn.Conv2d`, `nn.MaxPool2d` |
| `rnn_lstm.py` | 字符级语言模型 | `nn.LSTM`, 文本生成 |

---

## 十、与 LLM 的衔接

完成本目录学习后，你就掌握了理解 Transformer 所需的所有基本块：

- **张量与自动微分** → Transformer 的所有计算都是张量运算
- **MLP** → Transformer 的 FFN 子层就是 2 层 MLP（hidden_dim 变 4×）
- **激活函数（GELU / SwiGLU）** → LLM 的 FFN 用这些
- **优化器（AdamW + 学习率调度）** → LLM 训练标配
- **正则化（Dropout / RMSNorm）** → 每个 Transformer Block 都有
- **CNN（局部性思想）** → 帮助理解为什么 Transformer 不需要它
- **RNN（序列建模）** → 理解 attention 如何"取代"它

> **下一站**：把这些块按"Multi-Head Attention + FFN + 残差 + Norm"的方式堆叠，就是 Transformer。可以阅读项目根目录的 `llm/transformer_demo.py`（纯 NumPy 实现）继续深入。

---

## 十一、延伸阅读

- Goodfellow et al. *Deep Learning*（"花书"）—— 深度学习圣经
- Andrej Karpathy 的 [neural networks: zero to hero](https://karpathy.ai/zero-to-hero.html) —— 视频系列
- 3Blue1Brown 的神经网络可视化 —— 直觉建立首选
- PyTorch 官方教程
- "Attention Is All You Need" (2017) —— 直接进入 Transformer
