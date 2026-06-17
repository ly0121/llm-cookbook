"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:位置编码三流派                                        ║
║         绝对正余弦 vs 学习式 vs RoPE(LLaMA 同款)                  ║
╚══════════════════════════════════════════════════════════════════╝

【核心问题:Transformer 没有循环/卷积,如何感知位置?】

  绝对正余弦 (Vaswani 2017):
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    优点: 无参,可外推; 缺点: 不灵活

  学习式 (BERT/GPT-2):
    Embedding(max_pos, d), 训练时学
    优点: 简单灵活; 缺点: 完全无法外推到训练长度之外

  RoPE (LLaMA/Qwen):
    把 Q/K 视为复数 (q_0+iq_1, q_2+iq_3, ...),按位置旋转
    R(m) Q · R(n)^* K = Q · K · R(m-n)  ← 内积只依赖相对位置
    优点: 天然相对; 长度外推性最佳
"""
import numpy as np
import torch

torch.manual_seed(42)


def sinusoidal_pe(seq_len, d_model):
    """绝对正余弦。"""
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / (10000 ** (2 * (i // 2) / d_model))
    angles = pos * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe

class LearnedPE(torch.nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = torch.nn.Embedding(max_len, d_model)
    def forward(self, seq_len):
        return self.pe(torch.arange(seq_len))

def rope_apply(x, base=10000):
    """对 (T, d) 张量应用 RoPE。x 的最后一维必须是偶数。"""
    T, d = x.shape
    assert d % 2 == 0
    pos = torch.arange(T).float()[:, None]
    freqs = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))[None, :]
    theta = pos * freqs               # (T, d/2)
    cos, sin = theta.cos(), theta.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]   # 拆偶/奇
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def show_heatmap(M, title, max_rows=20, max_cols=40):
    """ASCII 灰度图。"""
    M = np.asarray(M)
    M = M[:max_rows, :max_cols]
    print(f"\n  ── {title}  ({M.shape[0]}×{M.shape[1]}) ──")
    vmin, vmax = M.min(), M.max()
    chars = " ·-+=*#@"
    for row in M:
        line = "".join(
            chars[min(int((v - vmin) / max(vmax - vmin, 1e-9) * (len(chars) - 1)), len(chars) - 1)]
            for v in row
        )
        print(f"  [{line}]")

def main():
    print("\n" + "█" * 60)
    print("█" + " " * 21 + "位置编码三流派" + " " * 24 + "█")
    print("█" * 60)

    seq_len, d_model = 32, 64

    print("\n" + "═" * 60); print("  ※ 1. 绝对正余弦 ※"); print("═" * 60)
    pe_sin = sinusoidal_pe(seq_len, d_model)
    show_heatmap(pe_sin, "Sinusoidal PE", max_rows=seq_len)

    print("\n" + "═" * 60); print("  ※ 2. 学习式 ※"); print("═" * 60)
    learned = LearnedPE(max_len=seq_len, d_model=d_model)
    pe_learn = learned(seq_len).detach().numpy()
    show_heatmap(pe_learn, "Learned PE (随机初始化,未训练)", max_rows=seq_len)

    print("\n" + "═" * 60); print("  ※ 3. RoPE(旋转位置编码) ※"); print("═" * 60)
    Q = torch.randn(seq_len, d_model)
    Q_rot = rope_apply(Q).numpy()
    show_heatmap(Q_rot - Q.numpy(), "RoPE 应用前后差值", max_rows=seq_len)

    print("\n" + "═" * 60); print("  ※ RoPE 关键性质验证:相对位置 ※"); print("═" * 60)
    q = torch.randn(d_model); k = torch.randn(d_model)
    print("  比较 RoPE(q,m) · RoPE(k,n) 与 m-n:")
    for (m, n) in [(0, 5), (3, 8), (10, 15), (2, 7)]:
        q_m = rope_apply(q[None, :].repeat(m+1, 1))[m]
        k_n = rope_apply(k[None, :].repeat(n+1, 1))[n]
        print(f"    m={m:3d}, n={n:3d}, m-n={m-n:+4d}  →  q_m·k_n = {(q_m @ k_n).item():+.4f}")
    print("  (相同 m-n 会得到相近的内积,这就是 RoPE 的相对位置性质)")

    print("\n  关键收获:")
    print("  ✓ 绝对正余弦无参可外推,但表示能力有限")
    print("  ✓ 学习式灵活但无法外推")
    print("  ✓ RoPE 把'位置'编码进 Q/K 的旋转里,内积只依赖相对距离")
    print("  ✓ LLaMA/Qwen/DeepSeek 都用 RoPE,这是现代 LLM 长上下文的关键")

if __name__ == "__main__":
    main()
