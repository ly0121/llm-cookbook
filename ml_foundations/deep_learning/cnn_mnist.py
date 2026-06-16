"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:CNN(LeNet 风格) — MNIST 子集                       ║
║         理解卷积神经网络的局部性 + 平移不变性                     ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题:MLP 已经能做 MNIST,为什么还需要 CNN?】
═══════════════════════════════════════════════════════════════════

  MLP 的两个问题:
    1. 参数爆炸 — 28×28=784 输入 → 128 隐藏 = 100K 参数
                 224×224 真实图像 → 50K 隐藏 = 250 亿参数(不可行)
    2. 不感知空间 — 把图像展平后,左上角和右下角的像素被同等对待
                  违背了"图像中相邻像素关系密切"的事实

  CNN 的两大设计:
  ┌─────────────────────────────────────────────────────────────┐
  │   ① 局部连接(local connectivity)                              │
  │      每个神经元只看输入的一小块区域(3×3 / 5×5)                 │
  │                                                             │
  │   ② 权重共享(weight sharing)                                  │
  │      同一个滤波器在整张图上滑动 — 大幅减少参数                  │
  │      (10万参数 → 几千参数)                                    │
  │                                                             │
  │   组合产生了平移不变性:                                         │
  │      "猫"在图像左上角和右下角都能被检测到                        │
  └─────────────────────────────────────────────────────────────┘

  本 demo 的网络结构(简化 LeNet):
    输入 (1,28,28)
      → Conv2d(1→16, kernel=3, pad=1) → ReLU → MaxPool2d(2)  → (16,14,14)
      → Conv2d(16→32, kernel=3, pad=1) → ReLU → MaxPool2d(2) → (32,7,7)
      → Flatten → Linear(32*7*7 → 64) → ReLU → Linear(64 → 10)

  与 LLM 的关联:
    Vision Transformer(ViT)直接把图像切成 patches 用 attention,
    在大数据集上超过 CNN。但在小数据 / 资源受限场景,CNN 仍是首选。
    "局部性 + 共享" vs "全局 + 数据驱动" — 两种归纳偏置的取舍。
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
# CNN 模型
# ─────────────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    """LeNet 风格,适合 MNIST 28×28 灰度图。"""

    def __init__(self, num_classes=10):
        super().__init__()
        # Conv 块 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # Conv 块 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # MaxPool 用同一个,无参数
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接分类头
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, 7, 7)
        x = x.view(x.size(0), -1)  # 展平 (B, 32*7*7)
        x = F.relu(self.fc1(x))  # (B, 64)
        x = self.fc2(x)  # (B, 10) logits
        return x


# ─────────────────────────────────────────────────────────────
# 数据
# ─────────────────────────────────────────────────────────────
def load_mnist(n_train=5000, n_test=1000, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    print("─" * 60)
    print(f"  加载 MNIST(数据缓存于 {DATA_DIR})")
    print("─" * 60)
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_full = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(Subset(train_full, range(n_train)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(test_full, range(n_test)), batch_size=batch_size, shuffle=False)
    print(f"  训练: {n_train} 张  测试: {n_test} 张  batch={batch_size}")
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────
# 训练 / 评估
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optim, criterion):
    model.train()
    tot_loss, tot_correct, tot_n = 0.0, 0, 0
    for x, y in loader:
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        tot_loss += loss.item() * x.size(0)
        tot_correct += (y_pred.argmax(1) == y).sum().item()
        tot_n += x.size(0)
    return tot_loss / tot_n, tot_correct / tot_n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tot_loss, tot_correct, tot_n = 0.0, 0, 0
    for x, y in loader:
        y_pred = model(x)
        tot_loss += criterion(y_pred, y).item() * x.size(0)
        tot_correct += (y_pred.argmax(1) == y).sum().item()
        tot_n += x.size(0)
    return tot_loss / tot_n, tot_correct / tot_n


# ─────────────────────────────────────────────────────────────
# 可视化:卷积核 + 特征图
# ─────────────────────────────────────────────────────────────
def visualize_conv1_filters(model):
    """打印 conv1 的 16 个 3×3 滤波器(简单字符画)。"""
    print("\n  ──── 第 1 层卷积学到的 16 个 3×3 滤波器 ────")
    weights = model.conv1.weight.detach().squeeze(1)  # (16, 3, 3)
    chars = " .:-=+*#%@"
    for f_idx in range(16):
        w = weights[f_idx]
        w = (w - w.min()) / (w.max() - w.min() + 1e-9)
        print(f"  filter #{f_idx}:")
        for row in w:
            line = "".join(chars[min(int(v * (len(chars) - 1)), len(chars) - 1)] for v in row)
            print(f"    {line}")


@torch.no_grad()
def visualize_feature_maps(model, sample):
    """前向到 conv1 输出,显示前 4 个通道的特征图。"""
    model.eval()
    x = sample.unsqueeze(0)  # (1, 1, 28, 28)
    feat = F.relu(model.conv1(x))  # (1, 16, 28, 28)
    print("\n  ──── 第 1 层卷积特征图(前 4 通道) ────")
    chars = " .:-=+*#%@"
    for ch in range(4):
        fm = feat[0, ch].numpy()
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-9)
        print(f"  channel #{ch}:")
        for row in fm[::2, ::2]:  # 14×14 字符画太宽,降采样到 14×14
            line = "".join(chars[min(int(v * (len(chars) - 1)), len(chars) - 1)] for v in row)
            print(f"    {line}")


def main():
    print("\n" + "█" * 60)
    print("█" + " " * 16 + "CNN — LeNet 风格 / MNIST" + " " * 17 + "█")
    print("█" * 60)

    train_loader, test_loader = load_mnist()

    model = SimpleCNN(num_classes=10)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  模型: {model.__class__.__name__}  参数总数 {n_params:,}")
    print("  对比:同规模 MLP(784→256→10) 参数约 200K — CNN 更轻")

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("\n" + "═" * 60)
    print("  训练循环 (5 epochs)")
    print("═" * 60)
    print(f"  {'epoch':>5s}  {'tr_loss':>9s}  {'tr_acc':>7s}  {'te_loss':>9s}  {'te_acc':>7s}")
    for ep in range(1, 6):
        tr_l, tr_a = train_epoch(model, train_loader, optim, criterion)
        te_l, te_a = evaluate(model, test_loader, criterion)
        print(f"  {ep:>5d}  {tr_l:>9.4f}  {tr_a*100:>6.2f}%  {te_l:>9.4f}  {te_a*100:>6.2f}%")

    print("\n" + "═" * 60)
    print("  可视化")
    print("═" * 60)
    visualize_conv1_filters(model)

    # 取一个测试样本看特征图
    sample, label = test_loader.dataset[0]
    print(f"\n  样本真实标签: {label}")
    visualize_feature_maps(model, sample)

    print("\n" + "═" * 60)
    print("  关键收获:")
    print("═" * 60)
    print("  ✓ Conv2d = 局部连接 + 权重共享,大幅减少参数")
    print("  ✓ MaxPool2d 下采样 + 引入轻微平移不变性")
    print("  ✓ 浅层卷积学边缘 / 角点,深层学物体部件")
    print("  ✓ CNN 在 MNIST 5 epoch 通常达 98%+,优于同规模 MLP")
    print("  ✓ ViT 时代 CNN 仍在小数据 / 边缘设备上有优势\n")


if __name__ == "__main__":
    main()
