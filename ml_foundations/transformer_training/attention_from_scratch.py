"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:自注意力机制从零实现                                  ║
║         softmax(QK^T/√d) V — Transformer 的心脏                   ║
╚══════════════════════════════════════════════════════════════════╝

【核心问题:Q、K、V 三个矩阵到底在做什么?】

  自注意力 = 每个 token 主动"看"序列里所有 token,加权聚合
    Q (Query):  "我想找什么?"
    K (Key):    "我有什么特征?"
    V (Value):  "我的内容是什么?"

  scores = Q @ K^T / √d_k     ← 相似度,除以 √d 防止 softmax 饱和
  weights = softmax(scores)    ← 概率分布
  output = weights @ V         ← 加权聚合

  多头 = 把 Q/K/V 拆成 h 份,每份独立算,最后 concat
       → 不同子空间学不同的"关系类型"
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42); np.random.seed(42)


def numpy_attention(X, Wq, Wk, Wv, mask=None):
    """X: (T, d) → output: (T, d_v); 全用 NumPy 算清楚每一步。"""
    Q = X @ Wq          # (T, d_k)
    K = X @ Wk          # (T, d_k)
    V = X @ Wv          # (T, d_v)
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)             # (T, T)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights = weights / weights.sum(axis=-1, keepdims=True)
    out = weights @ V                            # (T, d_v)
    return out, weights

def causal_mask(T):
    """下三角 mask:位置 i 只能看 0..i。"""
    return np.tril(np.ones((T, T), dtype=bool))

def show_attention(weights, title):
    """ASCII 灰度热度图。"""
    print(f"\n  ── {title} ──")
    chars = " ·∙○●"
    for row in weights:
        line = "".join(chars[min(int(v * len(chars)), len(chars)-1)] for v in row)
        print(f"  [{line}]  " + "  ".join(f"{v:.2f}" for v in row))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, causal=True):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)   # (B, h, T, T)
        if causal:
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ V                                       # (B, h, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(out), weights


def main():
    print("\n" + "█" * 60)
    print("█" + " " * 16 + "自注意力机制从零实现" + " " * 21 + "█")
    print("█" * 60)

    print("\n" + "═" * 60)
    print("  ※ NumPy 单头实现 ※")
    print("═" * 60)
    T, d = 5, 8
    X = np.random.randn(T, d) * 0.5
    Wq = np.random.randn(d, d) * 0.3
    Wk = np.random.randn(d, d) * 0.3
    Wv = np.random.randn(d, d) * 0.3

    out, w = numpy_attention(X, Wq, Wk, Wv, mask=None)
    show_attention(w, f"无 mask 注意力权重 ({T}×{T})")

    out_c, w_c = numpy_attention(X, Wq, Wk, Wv, mask=causal_mask(T))
    show_attention(w_c, f"因果 mask 注意力权重 (上三角填 -inf)")

    print("\n" + "═" * 60)
    print("  ※ PyTorch 多头实现 ※")
    print("═" * 60)
    mha = MultiHeadAttention(d_model=64, n_heads=4)
    x = torch.randn(1, 8, 64)
    y, weights = mha(x, causal=True)
    print(f"  输入: {tuple(x.shape)}  →  输出: {tuple(y.shape)}")
    print(f"  注意力权重: {tuple(weights.shape)}  (B, h, T, T)")
    print(f"  参数量: {sum(p.numel() for p in mha.parameters())} (≈ 4 × d_model²)")

    print("\n  ──── 对比 PyTorch 内置 SDPA 验证手写实现 ────")
    Q = mha.W_q(x).view(1, 8, 4, 16).transpose(1, 2)
    K = mha.W_k(x).view(1, 8, 4, 16).transpose(1, 2)
    V = mha.W_v(x).view(1, 8, 4, 16).transpose(1, 2)
    sdpa = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    manual = (F.softmax(Q @ K.transpose(-2,-1) / 4 + torch.tril(torch.ones(8,8)).log().masked_fill(torch.tril(torch.ones(8,8))==0, float("-inf")), dim=-1)) @ V
    diff = (sdpa - manual).abs().max().item()
    print(f"  手写 vs F.scaled_dot_product_attention 最大误差: {diff:.2e}")
    print(f"  → {'✓ 一致' if diff < 1e-5 else '✗ 不一致'}")

    print("\n  关键收获:")
    print("  ✓ Q·K^T 衡量相似度,除以 √d 防止 softmax 饱和")
    print("  ✓ 因果 mask 让 GPT 只看历史(自回归)")
    print("  ✓ 多头让模型在不同子空间学不同关系")

if __name__ == "__main__":
    main()
