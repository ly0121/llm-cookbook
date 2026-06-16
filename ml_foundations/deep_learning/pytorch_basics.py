"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:PyTorch 张量与自动微分基础                           ║
║         理解神经网络能"自动学习"的数学引擎                        ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题:PyTorch 是如何自动算梯度的?】
═══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────┐
  │   关键概念:                                                    │
  │                                                             │
  │   1. 张量(Tensor) — 多维数组,所有计算的载体                   │
  │   2. requires_grad — 标记"需要追踪梯度"的张量                 │
  │   3. 计算图(autograd) — 自动构建运算的有向无环图               │
  │   4. backward() — 自动反向传播,沿计算图算梯度                 │
  │                                                             │
  │   计算图示例 (z = x*y + b):                                  │
  │       x ──┐                                                 │
  │           ├──► mul ──► t ──┐                               │
  │       y ──┘                ├──► add ──► z                  │
  │                            │                                │
  │                       b ───┘                                │
  │                                                             │
  │   z.backward() 会沿这个图反向走,                              │
  │   把 dz/dx, dz/dy, dz/db 自动算出来。                        │
  └─────────────────────────────────────────────────────────────┘

  与 LLM 的关联:
    Transformer 一次前向涉及上亿次矩阵乘,但 .backward() 一行
    就能自动算所有几亿参数的梯度。这就是深度学习能"训练大模型"
    的工程基础。
"""

import torch
import torch.nn.functional as F


def section_1_create_tensors():
    """张量的创建与基本属性。"""
    print("\n" + "═" * 60)
    print("  § 1. 张量创建与基本属性")
    print("═" * 60)

    # 从 list 创建
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"\n  从 list 创建:")
    print(f"    a = {a.tolist()}")
    print(f"    shape = {tuple(a.shape)}, dtype = {a.dtype}, device = {a.device}")

    # 常用初始化
    print("\n  常用初始化:")
    print(f"    torch.zeros(2,3)  → shape={tuple(torch.zeros(2,3).shape)}")
    print(f"    torch.ones(2,3)   → shape={tuple(torch.ones(2,3).shape)}")
    print(f"    torch.randn(2,3)  → 标准正态")
    print(f"    torch.arange(6)   → {torch.arange(6).tolist()}")

    # 形状操作
    x = torch.arange(12).reshape(3, 4)
    print("\n  形状操作:")
    print(f"    x = arange(12).reshape(3,4)  shape={tuple(x.shape)}")
    print(f"    x.T              → shape={tuple(x.T.shape)}")
    print(f"    x.reshape(2,6)   → shape={tuple(x.reshape(2,6).shape)}")
    print(f"    x.unsqueeze(0)   → shape={tuple(x.unsqueeze(0).shape)}")
    print(f"    x.flatten()      → shape={tuple(x.flatten().shape)}")


def section_2_broadcasting():
    """广播:形状不同的张量自动对齐。"""
    print("\n" + "═" * 60)
    print("  § 2. 广播(Broadcasting)")
    print("═" * 60)

    print("\n  规则:从尾部开始对齐,缺失维度补 1,大小为 1 的维度可以扩展")

    a = torch.arange(6).reshape(3, 2)  # (3, 2)
    b = torch.tensor([10.0, 20.0])  # (2,) → 广播为 (1, 2) → (3, 2)
    print(f"\n  a.shape = {tuple(a.shape)}")
    print(f"  b.shape = {tuple(b.shape)}")
    print(f"  a + b   = {(a + b).tolist()}  ← b 沿第 0 维广播")

    # LLM 中典型场景:attention mask
    scores = torch.randn(2, 4, 4)  # (batch, seq, seq)
    mask = torch.triu(torch.ones(4, 4), diagonal=1).bool()  # (4, 4)
    print(f"\n  典型 LLM 场景:causal mask")
    print(f"    scores.shape = {tuple(scores.shape)}  (batch, seq, seq)")
    print(f"    mask.shape   = {tuple(mask.shape)}    (seq, seq)")
    print("    masked_scores = scores.masked_fill(mask, -inf)  ← 自动广播")


def section_3_autograd_basics():
    """自动微分:y = x³ 的梯度。"""
    print("\n" + "═" * 60)
    print("  § 3. autograd 自动求导(单变量)")
    print("═" * 60)

    print("\n  计算 y = x³ 在 x=2 处的导数")
    print("  解析解: dy/dx = 3x² → 在 x=2 时 = 12")

    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 3

    print(f"\n  PyTorch 计算:")
    print(f"    x = 2.0  (requires_grad=True)")
    print(f"    y = x³ = {y.item()}")

    y.backward()
    print(f"    y.backward() 之后 x.grad = {x.grad.item()}  ← 与解析解一致")


def section_4_autograd_multivar():
    """多变量自动微分:z = x²y + sin(y)"""
    print("\n" + "═" * 60)
    print("  § 4. autograd 多变量求导")
    print("═" * 60)

    print("\n  z = x²y + sin(y),求 dz/dx 和 dz/dy")
    print("  解析解:")
    print("    dz/dx = 2xy")
    print("    dz/dy = x² + cos(y)")

    x = torch.tensor(3.0, requires_grad=True)
    y = torch.tensor(1.5, requires_grad=True)
    z = x ** 2 * y + torch.sin(y)
    z.backward()

    print(f"\n  在 x=3, y=1.5 处:")
    print(f"    z          = {z.item():.4f}")
    print(f"    x.grad     = {x.grad.item():.4f}  (解析: 2*3*1.5={2*3*1.5})")
    print(f"    y.grad     = {y.grad.item():.4f}  (解析: 9+cos(1.5)={9 + torch.cos(torch.tensor(1.5)).item():.4f})")


def section_5_gradient_descent():
    """用 autograd 手动做 5 步梯度下降。"""
    print("\n" + "═" * 60)
    print("  § 5. 用 autograd 做梯度下降")
    print("═" * 60)

    print("\n  目标:最小化 f(x) = (x-3)²,真实最优 x*=3")

    x = torch.tensor(0.0, requires_grad=True)
    lr = 0.3
    print(f"\n  初始 x = 0.0,学习率 = {lr}")
    print(f"  {'step':>5s}  {'x':>8s}  {'f(x)':>8s}  {'grad':>8s}")
    for step in range(10):
        f = (x - 3) ** 2
        if x.grad is not None:
            x.grad.zero_()
        f.backward()
        print(f"  {step:>5d}  {x.item():>8.4f}  {f.item():>8.4f}  {x.grad.item():>8.4f}")
        with torch.no_grad():
            x -= lr * x.grad

    print(f"\n  最终 x ≈ {x.item():.4f}  (真实 x*=3)")


def section_6_neural_net_one_step():
    """模拟一个最小的"训练步":一层网络,一次更新。"""
    print("\n" + "═" * 60)
    print("  § 6. 最小神经网络训练步(线性回归 1 步)")
    print("═" * 60)

    print("\n  模型: y_pred = W @ x + b")
    print("  目标: 拟合 y = 2x + 1,初始 W=0, b=0")

    # 数据
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])  # y = 2x + 1

    W = torch.zeros(1, 1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.05

    print(f"\n  {'step':>5s}  {'W':>8s}  {'b':>8s}  {'loss':>8s}")
    for step in range(20):
        y_pred = X @ W + b
        loss = ((y_pred - y) ** 2).mean()

        if W.grad is not None:
            W.grad.zero_()
        if b.grad is not None:
            b.grad.zero_()
        loss.backward()

        if step % 4 == 0 or step == 19:
            print(f"  {step:>5d}  {W.item():>8.4f}  {b.item():>8.4f}  {loss.item():>8.4f}")

        with torch.no_grad():
            W -= lr * W.grad
            b -= lr * b.grad

    print(f"\n  最终 W ≈ {W.item():.3f} (目标 2.0),b ≈ {b.item():.3f} (目标 1.0)")
    print("  ✓ 这就是任何深度学习模型训练的核心循环!")


def section_7_no_grad_context():
    """torch.no_grad():推理时省内存。"""
    print("\n" + "═" * 60)
    print("  § 7. torch.no_grad() — 推理 / 评估时关闭追踪")
    print("═" * 60)

    x = torch.tensor(2.0, requires_grad=True)

    print("\n  在 with torch.no_grad() 内的运算不构建计算图,省内存")
    with torch.no_grad():
        y = x * 3
        print(f"    y = x*3 = {y.item()},  y.requires_grad = {y.requires_grad}")

    y2 = x * 3
    print(f"\n  在 with 外:y2.requires_grad = {y2.requires_grad}")
    print("\n  💡 LLM 推理(model.eval() + no_grad)能节省一半显存")


def section_8_common_ops_cheatsheet():
    """LLM 实现里最常用的 10 个张量操作速查。"""
    print("\n" + "═" * 60)
    print("  § 8. LLM 实现里高频操作速查")
    print("═" * 60)

    cheats = [
        ("矩阵乘",        "x @ y         ", "Q·K^T, attention 核心"),
        ("softmax",       "F.softmax(x,-1)", "attention 权重归一化"),
        ("masked_fill",   "x.masked_fill(m,-inf)", "causal mask"),
        ("拼接",          "torch.cat([a,b],dim=-1)", "KV-cache 拼接"),
        ("分块",          "x.chunk(3,dim=-1)", "QKV 拆分"),
        ("归一化",        "F.layer_norm(x,...)", "LayerNorm/RMSNorm"),
        ("reshape",       "x.reshape(B,T,H,D)", "多头 attention 拆头"),
        ("transpose",     "x.transpose(1,2)", "(B,T,H,D) → (B,H,T,D)"),
        ("交叉熵",        "F.cross_entropy", "next-token loss"),
        ("gather",        "x.gather(dim,idx)", "按 token id 取 embedding"),
    ]
    print()
    for name, code, where in cheats:
        print(f"  {name:10s}  {code:25s}  → {where}")


def main():
    print("\n" + "█" * 60)
    print("█" + " " * 13 + "PyTorch 张量与自动微分基础" + " " * 12 + "█")
    print("█" * 60)

    section_1_create_tensors()
    section_2_broadcasting()
    section_3_autograd_basics()
    section_4_autograd_multivar()
    section_5_gradient_descent()
    section_6_neural_net_one_step()
    section_7_no_grad_context()
    section_8_common_ops_cheatsheet()

    print("\n" + "═" * 60)
    print("  本节关键收获:")
    print("═" * 60)
    print("  ✓ 张量是多维数组,reshape/广播/索引是基本功")
    print("  ✓ requires_grad=True 让 PyTorch 自动记录计算图")
    print("  ✓ .backward() 自动反向传播算所有参数梯度")
    print("  ✓ 训练循环 = 前向 → loss → zero_grad → backward → step")
    print("  ✓ 推理时用 no_grad() 能省一半显存\n")


if __name__ == "__main__":
    main()
