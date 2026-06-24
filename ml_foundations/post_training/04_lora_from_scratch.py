"""
╔══════════════════════════════════════════════════════════════════╗
║  04_lora_from_scratch.py — 手写 LoRA 数学                          ║
║                                                                  ║
║  核心问题：W = W₀ + (α/r) BA 为什么 work？r 取多少够用？             ║
║  与生产对应：peft.LoraConfig 后端到底在做什么                       ║
╚══════════════════════════════════════════════════════════════════╝
"""
import sys

# 保护性导入：在没有 torch 的环境中给出中文提示
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("错误：未找到 PyTorch。请先执行 `pip install torch` 后再运行本脚本。")
    sys.exit(1)

torch.manual_seed(42)

# ── 超参数 ──────────────────────────────────────────────────────────
D_IN, D_OUT = 256, 256       # 玩具线性层维度
N_TRAIN = 1024               # 训练样本数
LORA_R = 8                   # 默认 LoRA 秩
LORA_ALPHA = 16              # 默认 LoRA scaling 因子
R_LIST = [4, 8, 16]         # 对比实验用的秩列表
HEATMAP_DIM = 16             # ASCII 热力图维度（16×16 子块）
STEPS = 300                  # 训练步数


# ────────────────────────────────────────────────────────────────────
#  核心模块
# ────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """W·x + (α/r) B·A·x，其中 W 冻结，只训 A、B。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.base = base
        # 冻结原始权重，不参与梯度更新
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r          # 等价于隐式 LR 调节
        # A 随机初始化（小值），B 初始化为 0 → 训练起点等价于纯 base
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LoRA 等式：W_eff = W₀ + (α/r) B A
        return self.base(x) + self.scaling * (x @ self.A.T @ self.B.T)

    def effective_delta(self) -> torch.Tensor:
        """返回低秩增量矩阵 (α/r) B A，形状 [D_OUT, D_IN]。"""
        with torch.no_grad():
            return self.scaling * (self.B @ self.A)


# ────────────────────────────────────────────────────────────────────
#  数据与训练工具
# ────────────────────────────────────────────────────────────────────

def make_toy_data() -> tuple:
    """目标：从「base 任务（已学好的线性映射）」迁移到「base + delta」。"""
    torch.manual_seed(42)
    W_true_base = torch.randn(D_OUT, D_IN) * 0.1
    # 低秩扰动：模拟「新任务 delta 本身就是低秩的」
    delta = torch.randn(D_OUT, LORA_R) @ torch.randn(LORA_R, D_IN) * 0.05
    W_true_new = W_true_base + delta
    X = torch.randn(N_TRAIN, D_IN)
    Y = X @ W_true_new.T
    return X, Y, W_true_base


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def train_loop(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    steps: int = STEPS,
    lr: float = 1e-3,
) -> list:
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr)
    losses = []
    loader = DataLoader(TensorDataset(X, Y), batch_size=64, shuffle=True)
    it = iter(loader)
    for _ in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# ────────────────────────────────────────────────────────────────────
#  可视化工具
# ────────────────────────────────────────────────────────────────────

def ascii_curve(losses: list, label: str, width: int = 60) -> None:
    """在终端打印 loss 下降折线（sparkline 风格）。"""
    print(f"\n── {label} (最终 loss = {losses[-1]:.4f}) ──")
    sampled = [losses[int(i * (len(losses) - 1) / (width - 1))] for i in range(width)]
    lo, hi = min(sampled), max(sampled)
    rng = hi - lo if hi > lo else 1.0
    bars = []
    for v in sampled:
        h = int(7 * (1 - (v - lo) / rng))   # 映射到 0-7
        bars.append("▁▂▃▄▅▆▇█"[h])
    print("loss: " + "".join(bars))


def ascii_heatmap(mat: torch.Tensor, title: str, dim: int = HEATMAP_DIM) -> None:
    """
    将矩阵左上角 dim×dim 子块渲染为 ASCII 热力图。
    用字符集 " .:-=+*#@" 表示从低到高的数值区间。
    """
    chars = " .:-=+*#@"
    # 取左上角子块
    sub = mat[:dim, :dim].detach().float()
    lo, hi = sub.min().item(), sub.max().item()
    rng = hi - lo if abs(hi - lo) > 1e-9 else 1.0
    print(f"\n  [{title}]  (显示 {dim}×{dim} 左上角子块，值域 [{lo:.3f}, {hi:.3f}])")
    print("  ┌" + "─" * dim + "┐")
    for row in sub:
        line = ""
        for v in row:
            idx = int((v.item() - lo) / rng * (len(chars) - 1))
            idx = max(0, min(len(chars) - 1, idx))
            line += chars[idx]
        print("  │" + line + "│")
    print("  └" + "─" * dim + "┘")


def print_param_table(full_params: int) -> None:
    """打印全参 vs LoRA(r=4/8/16) 参数量对比表。"""
    print("\n┌──────────────────┬──────────────┬──────────────┐")
    print("│  配置             │  可训练参数    │  占全参比例   │")
    print("├──────────────────┼──────────────┼──────────────┤")
    print(f"│  全参微调          │ {full_params:>12,} │ {'100.00%':>12} │")
    for r in R_LIST:
        lora_params = r * D_IN + r * D_OUT   # A: r×D_IN, B: D_OUT×r
        pct = 100.0 * lora_params / full_params
        row_label = f"  LoRA r={r:<2}"
        print(f"│{row_label:<18}│ {lora_params:>12,} │ {pct:>11.2f}% │")
    print("└──────────────────┴──────────────┴──────────────┘")


# ────────────────────────────────────────────────────────────────────
#  主函数
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(42)
    # 04 不依赖 transformers RNG (纯手写 LoRA)，省略 set_seed
    print("=" * 66)
    print("  LoRA 核心等式：W = W₀ + (α/r) BA")
    print("  W₀ 冻结；A、B 低秩矩阵，是唯一可训参数")
    print("=" * 66)

    X, Y, W_base = make_toy_data()
    print(f"\n任务：从 base 线性映射 → base + 低秩扰动")
    print(f"      D_IN={D_IN}, D_OUT={D_OUT}, N_TRAIN={N_TRAIN}, 训练步数={STEPS}\n")

    # ── 1) 参数量对比表 ──────────────────────────────────────────────
    full_params = D_IN * D_OUT   # bias=False
    print_param_table(full_params)

    # ── 2) 全参 baseline 训练 ─────────────────────────────────────
    torch.manual_seed(42)
    base_full = nn.Linear(D_IN, D_OUT, bias=False)
    with torch.no_grad():
        base_full.weight.copy_(W_base)
    for p in base_full.parameters():
        p.requires_grad = True
    print(f"\n[训练] 全参微调：可训练参数 = {count_trainable(base_full):,}")
    losses_full = train_loop(base_full, X, Y)

    # ── 3) 多秩 LoRA 对比 ─────────────────────────────────────────
    lora_results = {}
    for r in R_LIST:
        torch.manual_seed(42)
        base_lora = nn.Linear(D_IN, D_OUT, bias=False)
        with torch.no_grad():
            base_lora.weight.copy_(W_base)
        model = LoRALinear(base_lora, r=r, alpha=LORA_ALPHA)
        n_params = count_trainable(model)
        pct = 100.0 * n_params / full_params
        print(f"[训练] LoRA r={r:>2}，α={LORA_ALPHA}：可训练参数 = {n_params:,}  ({pct:.2f}%)")
        losses = train_loop(model, X, Y)
        lora_results[r] = (model, losses)

    # ── 4) Loss 收敛折线 ──────────────────────────────────────────
    print("\n\n══ Loss 收敛曲线（每格 ≈ 1 步） ══")
    ascii_curve(losses_full, "全参微调  (full FT)")
    for r in R_LIST:
        _, losses = lora_results[r]
        ascii_curve(losses, f"LoRA r={r}")

    # ── 5) ASCII 热力图：原矩阵 / 低秩补丁 / 合并结果 ─────────────
    print("\n\n══ ASCII 热力图：原矩阵 + 低秩补丁 ══")
    # 使用默认 r=8 的 LoRA 模型
    best_model, _ = lora_results[LORA_R]
    W0 = best_model.base.weight           # [D_OUT, D_IN]
    delta_mat = best_model.effective_delta()   # [D_OUT, D_IN]
    W_eff = W0 + delta_mat

    ascii_heatmap(W0, "W₀  原始权重矩阵 (冻结)")
    ascii_heatmap(delta_mat, f"(α/r)BA  低秩补丁  r={LORA_R}")
    ascii_heatmap(W_eff, "W_eff = W₀ + 低秩补丁")

    # ── 6) 最终 loss 汇总 ─────────────────────────────────────────
    print("\n\n══ 最终 Loss 汇总 ══")
    print(f"  全参微调         : {losses_full[-1]:.6f}")
    for r in R_LIST:
        _, losses = lora_results[r]
        pct = 100.0 * (r * D_IN + r * D_OUT) / full_params
        print(f"  LoRA r={r:<2} ({pct:.2f}%): {losses[-1]:.6f}")

    # ── 7) 关键收获 ───────────────────────────────────────────────
    print("\n\n=== 关键收获 ===")
    r8_params = R_LIST[1] * D_IN + R_LIST[1] * D_OUT
    pct8 = 100.0 * r8_params / full_params
    print(f"1. r=8 时只训原参数的 {pct8:.2f}%，最终 loss 与全参微调接近 → 低秩假设成立")
    print("2. B 初始化为 0 → 训练起点 ΔW=0，不破坏 base 预训练知识，「贴补丁」语义")
    print("3. scaling = α/r 把超参 α 从秩 r 解耦，换 r 时无需重调 learning rate")
    print("4. 真实 Transformer 里 LoRA 接在 q_proj/v_proj，数学形式与本脚本完全相同")
    print("5. 热力图直观显示 ΔW 低秩结构：补丁稀疏，大部分元素接近 0")


if __name__ == "__main__":
    main()
