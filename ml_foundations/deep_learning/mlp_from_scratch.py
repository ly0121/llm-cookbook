"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:纯 NumPy 手写 MLP — 反向传播完全展开                 ║
║         理解神经网络训练的"黑盒"内部到底发生了什么                 ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题:不靠 PyTorch,从零实现一个能学 XOR 的 2 层 MLP】
═══════════════════════════════════════════════════════════════════

  为什么 XOR 重要?
    XOR 是经典的"非线性可分"问题 — 单层感知机 1969 年被证明
    无法解决它,直接导致了 AI 第一次寒冬。1986 年反向传播算法
    让 2 层 MLP 轻松解决,神经网络才得以复兴。

  ┌─────────────────────────────────────────────────────────────┐
  │   网络结构(2-3-1 MLP):                                       │
  │                                                             │
  │   输入 x∈ℝ²  →  隐藏 h∈ℝ³  →  输出 y∈ℝ¹                    │
  │                                                             │
  │   前向:                                                       │
  │     z₁ = W₁·x + b₁          (3×2 · 2×1 + 3×1 = 3×1)        │
  │     a₁ = ReLU(z₁)            ← 非线性!                       │
  │     z₂ = W₂·a₁ + b₂         (1×3 · 3×1 + 1×1 = 1×1)        │
  │     ŷ  = sigmoid(z₂)        ← 输出概率                       │
  │     L  = -[y log ŷ + (1-y) log(1-ŷ)]   ← 二元交叉熵         │
  │                                                             │
  │   反向(链式法则展开):                                          │
  │     dL/dz₂ = ŷ - y              (sigmoid+BCE 的优雅性质)    │
  │     dL/dW₂ = (dL/dz₂) · a₁^T                                │
  │     dL/db₂ = dL/dz₂                                          │
  │     dL/da₁ = W₂^T · dL/dz₂                                  │
  │     dL/dz₁ = dL/da₁ * (z₁>0)    (ReLU 梯度只在正值传)        │
  │     dL/dW₁ = (dL/dz₁) · x^T                                 │
  │     dL/db₁ = dL/dz₁                                          │
  └─────────────────────────────────────────────────────────────┘

  与 LLM 的关联:
    Transformer 的每个 FFN 子层就是 2 层 MLP(hidden=4×D),
    全世界数千亿美金算力做的事情,本质上和本文件做的一样 —
    只是规模大了 100 亿倍。
"""

import numpy as np

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(np.float64)


class MLP:
    """2 层 MLP:input_dim → hidden_dim → 1 (二分类)。"""

    def __init__(self, input_dim, hidden_dim, lr=0.5):
        # He 初始化(适合 ReLU)
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(1, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, 1))
        self.lr = lr

    def forward(self, X):
        """X.shape = (input_dim, batch);返回 ŷ (1, batch)。"""
        self.X = X
        self.z1 = self.W1 @ X + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.W2 @ self.a1 + self.b2
        self.y_hat = sigmoid(self.z2)
        return self.y_hat

    def loss(self, y):
        """二元交叉熵。y.shape=(1, batch)。"""
        eps = 1e-9
        return -np.mean(y * np.log(self.y_hat + eps) + (1 - y) * np.log(1 - self.y_hat + eps))

    def backward(self, y):
        """反向传播,计算所有梯度。"""
        m = y.shape[1]
        # 输出层:sigmoid + BCE 的简洁组合
        dz2 = (self.y_hat - y) / m  # (1, m)
        dW2 = dz2 @ self.a1.T  # (1, hidden)
        db2 = dz2.sum(axis=1, keepdims=True)  # (1, 1)
        # 反向到隐藏层
        da1 = self.W2.T @ dz2  # (hidden, m)
        dz1 = da1 * relu_grad(self.z1)
        dW1 = dz1 @ self.X.T
        db1 = dz1.sum(axis=1, keepdims=True)
        return dW1, db1, dW2, db2

    def step(self, dW1, db1, dW2, db2):
        """梯度下降一步。"""
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


def train_xor():
    """训练 MLP 解决 XOR 问题。"""
    # XOR 数据:4 个样本
    X = np.array([
        [0, 0, 1, 1],
        [0, 1, 0, 1],
    ], dtype=np.float64)  # (2, 4)
    y = np.array([[0, 1, 1, 0]], dtype=np.float64)  # (1, 4)

    print("─" * 60)
    print("  数据集:XOR(异或)")
    print("─" * 60)
    print("    输入 (x₁, x₂)    标签 y")
    for i in range(4):
        print(f"    ({int(X[0,i])}, {int(X[1,i])})         {int(y[0,i])}")
    print("\n    线性不可分!需要至少 1 个隐藏层 + 非线性激活")

    # 训练
    model = MLP(input_dim=2, hidden_dim=4, lr=0.5)
    n_epochs = 5000

    print("\n" + "═" * 60)
    print(f"  训练 {n_epochs} epochs,lr=0.5,hidden_dim=4")
    print("═" * 60)
    print(f"  {'epoch':>6s}  {'loss':>10s}  {'acc':>6s}  示意")

    for epoch in range(n_epochs):
        model.forward(X)
        loss = model.loss(y)
        grads = model.backward(y)
        model.step(*grads)

        if epoch in [0, 100, 500, 1000, 2500, n_epochs - 1]:
            preds = model.predict(X)
            acc = (preds == y).mean()
            bar_len = max(0, int(40 - loss * 40))
            bar = "█" * bar_len
            print(f"  {epoch:>6d}  {loss:>10.6f}  {acc*100:>5.1f}%  {bar}")

    # 验证
    print("\n" + "═" * 60)
    print("  验证:")
    print("═" * 60)
    preds = model.predict(X)
    probs = model.forward(X)
    print(f"  {'输入':>10s}  {'真实':>4s}  {'预测概率':>10s}  {'预测':>6s}")
    for i in range(4):
        x_str = f"({int(X[0,i])},{int(X[1,i])})"
        print(f"  {x_str:>10s}  {int(y[0,i]):>4d}  {probs[0,i]:>10.4f}  {int(preds[0,i]):>6d}")

    print("\n  ✓ XOR 学习成功!这证明 2 层 MLP + 非线性能解决线性不可分问题")
    return model


def gradient_check():
    """数值梯度 vs 解析梯度,验证反向传播实现正确。"""
    print("\n" + "═" * 60)
    print("  梯度检查(numerical gradient vs analytical)")
    print("═" * 60)
    print("  方法:用 (f(w+h) - f(w-h)) / (2h) 近似 df/dw")
    print("       与反向传播算的梯度对比,误差应 < 1e-5")

    np.random.seed(0)
    X = np.random.randn(2, 5)
    y = (X[0] + X[1] > 0).astype(np.float64).reshape(1, -1)

    model = MLP(2, 3, lr=0.1)
    model.forward(X)
    dW1, _, _, _ = model.backward(y)

    # 对 W1 的某个元素做数值检查
    h = 1e-5
    i, j = 0, 1
    orig = model.W1[i, j]

    model.W1[i, j] = orig + h
    model.forward(X)
    loss_plus = model.loss(y)

    model.W1[i, j] = orig - h
    model.forward(X)
    loss_minus = model.loss(y)

    model.W1[i, j] = orig
    numerical = (loss_plus - loss_minus) / (2 * h)
    analytical = dW1[i, j]

    print(f"\n  W1[{i},{j}]:")
    print(f"    数值梯度 = {numerical:.8f}")
    print(f"    解析梯度 = {analytical:.8f}")
    print(f"    相对误差 = {abs(numerical - analytical) / max(abs(numerical), 1e-9):.2e}")
    if abs(numerical - analytical) < 1e-5:
        print("  ✓ 反向传播实现正确!")
    else:
        print("  ✗ 梯度不匹配,实现有 bug")


def main():
    print("\n" + "█" * 60)
    print("█" + " " * 12 + "纯 NumPy 手写 MLP — XOR 问题" + " " * 14 + "█")
    print("█" * 60)

    train_xor()
    gradient_check()

    print("\n" + "═" * 60)
    print("  关键收获:")
    print("═" * 60)
    print("  ✓ 反向传播 = 链式法则在计算图上的应用")
    print("  ✓ 每层只需算\"局部梯度\",PyTorch 自动做的就是这个")
    print("  ✓ ReLU 的梯度只在 z>0 时为 1,这是它简单高效的原因")
    print("  ✓ 同样的算法,放大 10 亿倍,就训练出了 GPT-4\n")


if __name__ == "__main__":
    main()
