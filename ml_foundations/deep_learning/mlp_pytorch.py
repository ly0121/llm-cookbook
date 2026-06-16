"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:PyTorch MLP — MNIST 子集分类                        ║
║         展示工业级训练循环模板                                    ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题:用 PyTorch 把"手写 MLP"重写,体验框架的便利】
═══════════════════════════════════════════════════════════════════

  对比 mlp_from_scratch.py:
    | 步骤      | NumPy 版                     | PyTorch 版                 |
    |----------|----------------------------|---------------------------|
    | 定义模型  | 手动写 W1,b1,W2,b2          | nn.Sequential(Linear, ReLU) |
    | 前向     | 手动写矩阵乘                  | model(x)                   |
    | 反向     | 手动推导链式法则(50+ 行)       | loss.backward()            |
    | 优化     | 手动写 W -= lr*grad          | optimizer.step()           |
    | 总代码   | ~100 行                     | ~30 行                     |

  ┌─────────────────────────────────────────────────────────────┐
  │   网络结构:                                                    │
  │   输入 28×28=784 → Linear(128) → ReLU → Linear(64) → ReLU   │
  │     → Linear(10) → softmax → 类别 0-9                        │
  │                                                             │
  │   训练循环(经典 5 步):                                        │
  │     1. y_pred = model(x)            前向                     │
  │     2. loss = criterion(y_pred, y)  计算损失                 │
  │     3. optimizer.zero_grad()        清梯度                   │
  │     4. loss.backward()              反向传播                 │
  │     5. optimizer.step()             更新参数                 │
  └─────────────────────────────────────────────────────────────┘

  数据:为了 CPU 快速训练,只用 MNIST 训练集前 5000 张 + 测试集前 1000 张
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)


# ─────────────────────────────────────────────────────────────
# 1. 模型定义
# ─────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """3 层全连接网络。"""

    def __init__(self, in_dim=784, hidden=(128, 64), out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], out_dim)

    def forward(self, x):
        # x: (B, 1, 28, 28) → 展平 (B, 784)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 输出 logits,交叉熵会自动 softmax
        return x


# ─────────────────────────────────────────────────────────────
# 2. 数据加载(只取 MNIST 子集让 CPU 快速训练)
# ─────────────────────────────────────────────────────────────
def load_mnist_subset(n_train=5000, n_test=1000, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    print("─" * 60)
    print(f"  加载 MNIST 数据集(首次运行会下载到 {DATA_DIR})")
    print("─" * 60)
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_full = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_ds = Subset(train_full, list(range(n_train)))
    test_ds = Subset(test_full, list(range(n_test)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"  训练集: {len(train_ds)} 张 (从 60000 中取前 {n_train})")
    print(f"  测试集: {len(test_ds)} 张 (从 10000 中取前 {n_test})")
    print(f"  batch_size: {batch_size}")

    # 看一下一个样本
    img, lbl = train_ds[0]
    print(f"  样本形状: {tuple(img.shape)}, 标签: {lbl}")

    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────
# 3. 训练 / 评估
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, n_correct, n_total = 0.0, 0, 0
    for x, y in loader:
        # 经典训练 5 步
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n_correct += (y_pred.argmax(1) == y).sum().item()
        n_total += x.size(0)
    return total_loss / n_total, n_correct / n_total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item() * x.size(0)
            n_correct += (y_pred.argmax(1) == y).sum().item()
            n_total += x.size(0)
    return total_loss / n_total, n_correct / n_total


# ─────────────────────────────────────────────────────────────
# 4. 简单可视化:文本字符画展示一个样本
# ─────────────────────────────────────────────────────────────
def show_sample_ascii(img_tensor, label, pred=None):
    img = img_tensor.squeeze().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    chars = " .:-=+*#%@"
    print(f"\n  样本 真实=[{label}]" + (f" 预测=[{pred}]" if pred is not None else ""))
    for row in img:
        line = "".join(chars[min(int(v * (len(chars) - 1)), len(chars) - 1)] for v in row)
        print(f"    {line}")


# ─────────────────────────────────────────────────────────────
# 5. main
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "█" * 60)
    print("█" + " " * 16 + "PyTorch MLP — MNIST 子集" + " " * 17 + "█")
    print("█" * 60)

    # 数据
    train_loader, test_loader = load_mnist_subset(n_train=5000, n_test=1000, batch_size=64)

    # 模型
    model = MLP(in_dim=784, hidden=(128, 64), out_dim=10)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  模型: {model}")
    print(f"  总参数: {n_params:,}")

    # 优化器 + 损失
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练
    print("\n" + "═" * 60)
    print("  训练循环")
    print("═" * 60)
    n_epochs = 5
    print(f"  {'epoch':>5s}  {'tr_loss':>9s}  {'tr_acc':>7s}  {'te_loss':>9s}  {'te_acc':>7s}")
    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        print(f"  {epoch:>5d}  {tr_loss:>9.4f}  {tr_acc*100:>6.2f}%  {te_loss:>9.4f}  {te_acc*100:>6.2f}%")

    # 看几个测试样本的预测结果
    print("\n" + "═" * 60)
    print("  测试样本可视化(前 3 张)")
    print("═" * 60)
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            preds = model(x).argmax(1)
            for j in range(min(3, x.size(0))):
                show_sample_ascii(x[j], y[j].item(), preds[j].item())
            break

    print("\n" + "═" * 60)
    print("  关键收获:")
    print("═" * 60)
    print("  ✓ nn.Module + forward() = 自动构建计算图")
    print("  ✓ 训练 5 步:forward → loss → zero_grad → backward → step")
    print("  ✓ DataLoader 自动 batch + shuffle + 多进程")
    print("  ✓ Adam 比 SGD 收敛快得多,默认 lr=1e-3 通常够用")
    print("  ✓ MLP 在 MNIST 上 5 epoch 就能达到 ~95% acc(全量数据可达 98%+)")
    print("\n  下一站: cnn_mnist.py — 用 CNN 把同样的任务再上一个台阶\n")


if __name__ == "__main__":
    main()
