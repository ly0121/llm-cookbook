# Phase 2: Transformer 从零训练 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete from-scratch Transformer training walkthrough (7 demos + 7 docs) that runs on Mac CPU/MPS in under 6 minutes.

**Architecture:** Self-contained demos under `ml_foundations/transformer_training/`. The main training script (`gpt_train.py`) defines the `GPT` model class and is imported by 3 downstream demos (sampling, attention viz, KV cache) via shared checkpoint at `data/ckpt.pt`. All demos run independently except the 3 that require running training first.

**Tech Stack:** Python 3.10+, NumPy, PyTorch (CPU/MPS auto-detect), tiktoken (only for the BPE-vs-cl100k_base comparison snippet). No new dependencies beyond what Phase 1 added.

**Reference spec:** `docs/superpowers/specs/2026-06-16-phase2-transformer-training-design.md`

---

## Task Order Overview

```
A. Scaffold + corpus            → Task 1
B. Standalone teaching demos     → Tasks 2, 3, 4
C. Main training demo            → Task 5  ← critical path
D. Checkpoint-consuming demos    → Tasks 6, 7, 8
E. KNOWLEDGE.md                  → Task 9
F. VitePress sub-chapter docs    → Tasks 10-16
G. Sidebar + README + .gitignore → Task 17
H. Smoke-test all demos          → Task 18
I. Final single commit           → Task 19
```

Tasks 2/3/4 are independent and can be parallelized. Tasks 6/7/8 depend on Task 5. Docs (Tasks 9-16) can run in parallel with code if desired.

---

## Task 1: Scaffold directory + vendor Tiny Shakespeare corpus

**Files:**
- Create: `ml_foundations/transformer_training/__init__.py` (empty marker)
- Create: `ml_foundations/transformer_training/data/tiny_shakespeare.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p ml_foundations/transformer_training/data
touch ml_foundations/transformer_training/__init__.py
```

- [ ] **Step 2: Vendor Tiny Shakespeare corpus**

Download once and commit (public domain):

```bash
curl -L -o ml_foundations/transformer_training/data/tiny_shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
wc -l ml_foundations/transformer_training/data/tiny_shakespeare.txt
# Expect: ~40000 lines, ~1.1MB
```

If curl fails (offline), commit a small placeholder note and the tokenizer demo will print a clear download instruction.

- [ ] **Step 3: Update .gitignore**

Add at end of `.gitignore`:

```
# Phase 2: Transformer training artifacts
ml_foundations/transformer_training/data/ckpt.pt
ml_foundations/transformer_training/data/loss_log.json
ml_foundations/transformer_training/data/runs/
```

- [ ] **Step 4: Verify file presence**

```bash
ls -la ml_foundations/transformer_training/data/
# Expect: tiny_shakespeare.txt (~1.1MB)
```

---

## Task 2: bpe_tokenizer.py — From-scratch BPE

**Files:**
- Create: `ml_foundations/transformer_training/bpe_tokenizer.py`

- [ ] **Step 1: Write the docstring header**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:从零实现 BPE Tokenizer                                ║
║         GPT 用的子词分词器是怎么造出来的                            ║
╚══════════════════════════════════════════════════════════════════╝

【核心问题:tokenizer 是怎么从原始字符学到 "the"、"ing" 这类常见子词的?】

  BPE (Byte-Pair Encoding) 算法:
    1. 初始词表 = 所有字符
    2. 统计语料中相邻 token 对出现频次
    3. 合并最高频的 pair → 新 token
    4. 重复 N 次,得到最终词表

  与 LLM 的关联:
    GPT-2: ~50K BPE 词表
    GPT-4 (cl100k_base): ~100K
    LLaMA-3: ~128K
    本 demo 训 ~265 词表,展示算法骨架,与生产 tokenizer 数学完全一致。
"""
```

- [ ] **Step 2: Implement BPE class**

```python
import re
from collections import Counter
from pathlib import Path

class BPETokenizer:
    def __init__(self):
        self.merges = []           # list of (pair, new_token_id)
        self.vocab = {}            # int -> bytes
        self.token_to_id = {}      # bytes -> int

    def _get_stats(self, ids_list):
        """统计相邻 pair 频次。"""
        counts = Counter()
        for ids in ids_list:
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] += 1
        return counts

    def _merge(self, ids_list, pair, new_id):
        out = []
        for ids in ids_list:
            new_ids, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                    new_ids.append(new_id); i += 2
                else:
                    new_ids.append(ids[i]); i += 1
            out.append(new_ids)
        return out

    def train(self, text, num_merges=200, verbose_first_n=30):
        # init: each byte -> int (0..255)
        ids_list = [list(s.encode("utf-8")) for s in text.split("\n") if s.strip()]
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []

        for step in range(num_merges):
            stats = self._get_stats(ids_list)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + step
            ids_list = self._merge(ids_list, pair, new_id)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges.append((pair, new_id))
            if step < verbose_first_n:
                merged_str = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"  step {step:3d}  merge {pair} → {new_id:3d}  '{merged_str}'  (count={stats[pair]})")

        self.token_to_id = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        for pair, new_id in self.merges:
            ids = self._merge([ids], pair, new_id)[0]
        return ids

    def decode(self, ids):
        out = b"".join(self.vocab[i] for i in ids)
        return out.decode("utf-8", errors="replace")
```

- [ ] **Step 3: Implement main()**

```python
def main():
    print("\n" + "█" * 60)
    print("█" + " " * 18 + "BPE Tokenizer 从零训练" + " " * 18 + "█")
    print("█" * 60)

    corpus_path = Path(__file__).parent / "data" / "tiny_shakespeare.txt"
    if not corpus_path.exists():
        print(f"  ❌ 语料文件未找到: {corpus_path}")
        print("     请运行: curl -L -o {corpus_path} \\")
        print("       https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        return
    text = corpus_path.read_text()[:50_000]  # 前 50KB 训练演示
    print(f"\n  语料: tiny_shakespeare.txt 前 {len(text)} 字符")

    tok = BPETokenizer()
    print("\n  ──── 训练 200 轮合并(展示前 30 轮) ────")
    tok.train(text, num_merges=200, verbose_first_n=30)
    print(f"\n  最终词表大小: {len(tok.vocab)}  (256 字节 + {len(tok.merges)} 合并)")

    sample = "ROMEO: But soft, what light through yonder window breaks?"
    char_count = len(sample.encode("utf-8"))
    bpe_ids = tok.encode(sample)
    print(f"\n  ──── 编码示例 ────")
    print(f"  原文({char_count} 字节): {sample}")
    print(f"  BPE  ({len(bpe_ids)} tokens): {bpe_ids[:20]}{'...' if len(bpe_ids) > 20 else ''}")
    print(f"  压缩比: {char_count/len(bpe_ids):.2f}x")
    print(f"  解码回原文: {tok.decode(bpe_ids)}")

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        gpt4_ids = enc.encode(sample)
        print(f"\n  ──── 与 GPT-4 (cl100k_base, ~100K 词表) 对比 ────")
        print(f"  本 demo: {len(bpe_ids)} tokens")
        print(f"  GPT-4 : {len(gpt4_ids)} tokens (词表大 ~400 倍 → token 更短)")
    except ImportError:
        pass

    print("\n  关键收获:")
    print("  ✓ BPE = 反复合并最高频字符对")
    print("  ✓ 词表越大压缩越好,但 embedding 矩阵也越大")
    print("  ✓ 字节级初始化保证可处理任意 Unicode")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run and verify**

```bash
python ml_foundations/transformer_training/bpe_tokenizer.py
# Expect:
#   - shows 30 merge steps
#   - final vocab ~456
#   - decoded sample matches original
#   - tiktoken comparison (if installed)
# Wall time: < 5s
```

---

## Task 3: attention_from_scratch.py — Self-attention dual implementation

**Files:**
- Create: `ml_foundations/transformer_training/attention_from_scratch.py`

- [ ] **Step 1: Docstring + imports**

```python
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
```

- [ ] **Step 2: NumPy implementation (single-head)**

```python
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
```

- [ ] **Step 3: PyTorch multi-head implementation**

```python
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
```

- [ ] **Step 4: main() with comparisons**

```python
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
```

- [ ] **Step 5: Run and verify**

```bash
python ml_foundations/transformer_training/attention_from_scratch.py
# Expect: ASCII heatmaps, sdpa diff < 1e-5
# Wall time: < 3s
```

---

## Task 4: positional_encoding.py — Three PE schemes

**Files:**
- Create: `ml_foundations/transformer_training/positional_encoding.py`

- [ ] **Step 1: Docstring + the three implementations**

```python
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
```

- [ ] **Step 2: Three encoders**

```python
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
```

- [ ] **Step 3: Visualization helpers + main()**

```python
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
```

- [ ] **Step 4: Run and verify**

```bash
python ml_foundations/transformer_training/positional_encoding.py
# Expect: 3 ASCII heatmaps + RoPE relative-position table
# Wall time: < 2s
```

---

## Task 5: gpt_train.py — Main GPT training (CRITICAL PATH)

**Files:**
- Create: `ml_foundations/transformer_training/gpt_train.py`

This is the heaviest task. Estimated 500-600 lines.

- [ ] **Step 1: Docstring + imports + config**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:从零训练 GPT(decoder-only Transformer)               ║
║         ~3M 参数 / Tiny Shakespeare / Mac CPU 5min 可见效果       ║
╚══════════════════════════════════════════════════════════════════╝

【核心问题:把 attention + position + FFN + LayerNorm 拼起来,
            训出一个能写"伪莎士比亚"的小 LLM】

  本文件双重身份:
    ① 主训练脚本: python gpt_train.py
    ② 可被 import 的模型库: from gpt_train import GPT, load_checkpoint, ...

  与 GPT-2 / LLaMA 的对应关系:
    GPT-2 small: 12 层 × 12 头 × 768 d_model = 124M params
    本 demo  : 6 层 × 6 头 × 192 d_model = ~3M params (~40 倍小)
    架构完全一致,只是规模不同。
"""
import json, math, time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
DATA_DIR = Path(__file__).parent / "data"
CKPT_PATH = DATA_DIR / "ckpt.pt"
LOSS_LOG = DATA_DIR / "loss_log.json"
```

- [ ] **Step 2: Model definitions**

```python
@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 1024     # 实际由 tokenizer 决定,build_tokenizer 时填充
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        T_full = k.size(2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if new_cache is None:  # 训练阶段才需要 causal mask
            att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), new_cache

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))

class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
    def forward(self, x, kv_cache=None):
        a, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, new_cache

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, kv_caches=None):
        B, T = idx.shape
        if kv_caches is not None:
            past_len = kv_caches[0][0].size(2) if kv_caches[0] is not None else 0
            pos = torch.arange(past_len, past_len + T, device=idx.device)
        else:
            pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, kv_cache=cache)
            new_caches.append(new_cache)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, (new_caches if kv_caches is not None else None)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=False):
        self.eval()
        kv_caches = [None] * self.cfg.n_layer if use_cache else None
        for _ in range(max_new_tokens):
            if use_cache and kv_caches[0] is not None:
                idx_in = idx[:, -1:]
            else:
                idx_in = idx[:, -self.cfg.block_size:]
            logits, _, kv_caches = self(idx_in, kv_caches=kv_caches) if use_cache else self(idx_in)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
```

- [ ] **Step 3: Tokenizer + data + training loop**

```python
def build_char_tokenizer(text):
    """字符级 tokenizer(简化:不用 BPE,加快迭代)。"""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    return encode, decode, len(chars), stoi, itos

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

def train(steps=2000, batch_size=32, eval_interval=200):
    print(f"\n  设备: {DEVICE}")
    text = (DATA_DIR / "tiny_shakespeare.txt").read_text()
    encode, decode, vocab_size, stoi, itos = build_char_tokenizer(text)
    print(f"  语料: {len(text)} 字符  |  词表: {vocab_size}")

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    cfg = GPTConfig(vocab_size=vocab_size)
    model = GPT(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型: {cfg}")
    print(f"  参数量: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    warmup = 100
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    losses = []
    t0 = time.time()
    for step in range(steps):
        model.train()
        x, y = get_batch(train_data, cfg.block_size, batch_size, DEVICE)
        _, loss, _ = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); sched.step()

        if step % eval_interval == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(val_data, cfg.block_size, batch_size, DEVICE)
                _, val_loss, _ = model(x_val, targets=y_val)
            losses.append({"step": step, "train": loss.item(), "val": val_loss.item(), "lr": sched.get_last_lr()[0]})
            elapsed = time.time() - t0
            print(f"  step {step:5d}/{steps}  lr {sched.get_last_lr()[0]:.5f}  train {loss.item():.4f}  val {val_loss.item():.4f}  ({elapsed:.1f}s)")

    print(f"\n  训练完成,总耗时 {time.time()-t0:.1f}s")

    # 保存
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "stoi": stoi, "itos": itos,
    }, CKPT_PATH)
    LOSS_LOG.write_text(json.dumps(losses, indent=2))
    print(f"  ckpt → {CKPT_PATH}")
    print(f"  loss → {LOSS_LOG}")

    # 生成示例
    print("\n  ──── 生成示例(prompt='ROMEO:', 200 tokens) ────")
    prompt = torch.tensor([encode("ROMEO:")], dtype=torch.long, device=DEVICE)
    out = model.generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40)
    print("  " + decode(out[0].tolist()).replace("\n", "\n  "))

    # ASCII loss curve
    print("\n  ──── Loss 曲线 ────")
    if losses:
        height = 12
        max_l = max(l["train"] for l in losses)
        min_l = min(l["val"] for l in losses)
        for h in range(height, -1, -1):
            row = ""
            for l in losses:
                tn = (l["train"] - min_l) / max(max_l - min_l, 1e-9) * height
                vn = (l["val"]   - min_l) / max(max_l - min_l, 1e-9) * height
                if abs(tn - h) < 0.5: row += "T"
                elif abs(vn - h) < 0.5: row += "V"
                else: row += " "
            print(f"  {max_l - (max_l-min_l)*(height-h)/height:5.2f} │ {row}")

# ──────────────────────────────────────────
# 提供给其它 demo import 的辅助函数
# ──────────────────────────────────────────
def load_checkpoint(path=CKPT_PATH, device=DEVICE):
    ckpt = torch.load(path, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    encode = lambda s: [ckpt["stoi"][c] for c in s if c in ckpt["stoi"]]
    decode = lambda ids: "".join(ckpt["itos"][i] for i in ids)
    return model, cfg, encode, decode

def main():
    print("\n" + "█" * 60)
    print("█" + " " * 14 + "GPT 从零训练(decoder-only)" + " " * 17 + "█")
    print("█" * 60)
    train(steps=2000, batch_size=32, eval_interval=200)
    print("\n  关键收获:")
    print("  ✓ 同一架构 + 训练循环,放大 100 倍就是 GPT-2,放大 10000 倍就是 GPT-4")
    print("  ✓ AdamW + cosine warmup 是 LLM 训练默认配方")
    print("  ✓ weight_decay 0.1 + grad_clip 1.0 是稳定训练的关键\n")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Dry-run a quick smoke test (50 steps)**

```bash
cd ml_foundations/transformer_training
python -c "
from gpt_train import train
train(steps=50, batch_size=16, eval_interval=10)
"
# Expect: loss drops, no errors, ckpt saved
# Wall time on CPU: ~15s; on MPS: ~3s
```

- [ ] **Step 5: Full run**

```bash
python ml_foundations/transformer_training/gpt_train.py
# CPU expected: 5-6 min, val loss drops from ~4.3 to ~1.7
# MPS expected: ~30s, similar loss
# Generated text: should resemble Shakespeare-ish English (not gibberish)
```

- [ ] **Step 6: Verify checkpoint exists**

```bash
ls -la ml_foundations/transformer_training/data/ckpt.pt
ls -la ml_foundations/transformer_training/data/loss_log.json
# Both should exist; ckpt ~10-30MB
```

---

## Task 6: sampling_strategies.py

**Files:**
- Create: `ml_foundations/transformer_training/sampling_strategies.py`

- [ ] **Step 1: Implement and run**

Full code (~250 lines):

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:文本生成的采样策略                                    ║
║         greedy / temperature / top-k / top-p 实战对比              ║
╚══════════════════════════════════════════════════════════════════╝

【核心问题:为什么同一个模型,改一下 temperature 输出就完全不同?】

  采样 = 在每一步从下一 token 的概率分布中"挑一个"
    greedy       :   永远挑概率最高的 → 死板,易重复
    temperature  :  T<1 更确定,T>1 更随机
    top-k        :  只在前 k 个候选里采样,过滤长尾噪音
    top-p (核采样):  累计概率达到 p 的最小集合内采样,自适应
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from gpt_train import load_checkpoint, CKPT_PATH

torch.manual_seed(42)

@torch.no_grad()
def generate(model, encode_ids, max_new_tokens, strategy, **kwargs):
    """统一接口,支持多种采样策略。"""
    idx = torch.tensor([encode_ids], dtype=torch.long, device=next(model.parameters()).device)
    cfg = model.cfg
    for _ in range(max_new_tokens):
        idx_in = idx[:, -cfg.block_size:]
        logits, _, _ = model(idx_in)
        logits = logits[:, -1, :]
        if strategy == "greedy":
            next_id = logits.argmax(dim=-1, keepdim=True)
        elif strategy == "temperature":
            T = kwargs.get("temperature", 1.0)
            probs = F.softmax(logits / T, dim=-1)
            next_id = torch.multinomial(probs, 1)
        elif strategy == "top_k":
            k = kwargs.get("k", 40)
            T = kwargs.get("temperature", 1.0)
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits / T, dim=-1)
            next_id = torch.multinomial(probs, 1)
        elif strategy == "top_p":
            p = kwargs.get("p", 0.95)
            T = kwargs.get("temperature", 1.0)
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits / T, dim=-1), dim=-1)
            mask = cum_probs > p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits[mask] = -float("inf")
            logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
        else:
            raise ValueError(strategy)
        idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()

def main():
    print("\n" + "█" * 60)
    print("█" + " " * 18 + "采样策略对比" + " " * 27 + "█")
    print("█" * 60)

    if not CKPT_PATH.exists():
        print(f"\n  ❌ 找不到 checkpoint: {CKPT_PATH}")
        print(f"     请先跑: python ml_foundations/transformer_training/gpt_train.py")
        return

    model, cfg, encode, decode = load_checkpoint()
    prompt = "ROMEO:"
    prompt_ids = encode(prompt)
    print(f"\n  prompt: '{prompt}'  |  生成 200 tokens 对比 5 种采样")

    cases = [
        ("greedy",       {"strategy": "greedy"}),
        ("T=0.1 (确定)", {"strategy": "temperature", "temperature": 0.1}),
        ("T=0.8 (常用)", {"strategy": "temperature", "temperature": 0.8}),
        ("top-k=40",     {"strategy": "top_k", "k": 40, "temperature": 0.8}),
        ("top-p=0.95",   {"strategy": "top_p", "p": 0.95, "temperature": 0.8}),
    ]
    for label, kwargs in cases:
        torch.manual_seed(42)  # 同种子保证唯一变量是策略
        ids = generate(model, prompt_ids, max_new_tokens=200, **kwargs)
        text = decode(ids).replace("\n", " ")
        print(f"\n  ── {label} ──")
        print(f"  {text[:300]}")

    print("\n  关键收获:")
    print("  ✓ greedy 容易陷入循环(因为每步只选一个固定方向)")
    print("  ✓ temperature 调整\"随机性\";T=0.7-0.9 是大多数场景的甜区")
    print("  ✓ top-k 砍掉长尾噪音,但每步候选数固定")
    print("  ✓ top-p(核采样)候选数自适应,是 OpenAI 默认推荐\n")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
python ml_foundations/transformer_training/sampling_strategies.py
# Expect: 5 different generations, T=0.1 should look more repetitive
# Wall time: 30-60s on CPU
```

---

## Task 7: attention_visualization.py

**Files:**
- Create: `ml_foundations/transformer_training/attention_visualization.py`

- [ ] **Step 1: Implement attention extraction via forward hook**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:可视化模型的注意力分布                                ║
║         打开"黑盒",看每个头到底在关注什么                         ║
╚══════════════════════════════════════════════════════════════════╝
"""
import torch
import torch.nn.functional as F
from gpt_train import load_checkpoint, CKPT_PATH

torch.manual_seed(42)

def patch_attn_to_record(model):
    """Monkey-patch CausalSelfAttention.forward 让它把权重存到 self.last_attn。"""
    from gpt_train import CausalSelfAttention
    import math
    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        self.last_attn = att.detach()      # ← 保存
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), None
    for block in model.blocks:
        block.attn.forward = forward.__get__(block.attn, CausalSelfAttention)

def show_attn_grid(att, tokens, layer, head, max_T=20):
    """ASCII 灰度图。att: (T, T)"""
    att = att[:max_T, :max_T].cpu().numpy()
    tokens = tokens[:max_T]
    chars = " ·∙○●"
    print(f"\n  ── Layer {layer}, Head {head} ──")
    print("       " + "".join(f"{t:3s}" for t in tokens))
    for i, row in enumerate(att):
        line = "".join(chars[min(int(v * len(chars)), len(chars)-1)] * 3 for v in row)
        print(f"  {tokens[i]:>3s}  {line}")

def main():
    print("\n" + "█" * 60)
    print("█" + " " * 16 + "注意力分布可视化" + " " * 23 + "█")
    print("█" * 60)

    if not CKPT_PATH.exists():
        print(f"\n  ❌ 找不到 checkpoint,请先训练: python gpt_train.py")
        return

    model, cfg, encode, decode = load_checkpoint()
    patch_attn_to_record(model)

    prompt = "ROMEO: But soft!"
    ids = encode(prompt)
    tokens = [decode([i]) for i in ids]
    x = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)

    with torch.no_grad():
        model(x)

    print(f"\n  prompt: '{prompt}'  ({len(ids)} tokens)")
    print(f"  共 {cfg.n_layer} 层 × {cfg.n_head} 头 = {cfg.n_layer * cfg.n_head} 个 attention map")

    for layer in [0, cfg.n_layer // 2, cfg.n_layer - 1]:
        att = model.blocks[layer].attn.last_attn[0]   # (n_head, T, T)
        for head in [0, cfg.n_head - 1]:
            show_attn_grid(att[head], tokens, layer, head)

    print("\n  关键收获:")
    print("  ✓ 浅层注意力倾向\"近邻 token\"(局部信息聚合)")
    print("  ✓ 深层注意力跨度更大,捕获长程语义关系")
    print("  ✓ 不同 head 学不同模式 → 多头是\"专家分工\"")
    print("  ✓ 工业 LLM 用 attention probe 工具(BertViz)做更精细分析\n")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run**

```bash
python ml_foundations/transformer_training/attention_visualization.py
# Expect: 6 ASCII attention maps (3 layers × 2 heads)
# Wall time: <5s
```

---

## Task 8: kv_cache.py

**Files:**
- Create: `ml_foundations/transformer_training/kv_cache.py`

- [ ] **Step 1: Implement benchmark**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:KV Cache 推理加速实验                                 ║
║         为什么 ChatGPT 第二个 token 比第一个快?                    ║
╚══════════════════════════════════════════════════════════════════╝

【核心问题:生成第 t 个 token,前 t-1 个 K/V 已经算过,
            为什么要重算?直接缓存!】

  无 cache: 每生成 1 个 token,重算整个序列的 K/V → O(t²) 总成本
  有 cache: 每步只算新 token 的 K/V,append 到缓存 → O(t) 总成本
"""
import time
import torch
from gpt_train import load_checkpoint, GPT, CKPT_PATH

torch.manual_seed(42)

@torch.no_grad()
def gen_no_cache(model, prompt_ids, max_new):
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=next(model.parameters()).device)
    cfg = model.cfg
    for _ in range(max_new):
        idx_in = idx[:, -cfg.block_size:]
        logits, _, _ = model(idx_in)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx

@torch.no_grad()
def gen_with_cache(model, prompt_ids, max_new):
    device = next(model.parameters()).device
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    cfg = model.cfg
    kv_caches = [None] * cfg.n_layer
    # 先 prefill prompt
    logits, _, kv_caches = model(idx, kv_caches=kv_caches)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    idx = torch.cat([idx, next_id], dim=1)
    # decode 阶段
    for _ in range(max_new - 1):
        logits, _, kv_caches = model(next_id, kv_caches=kv_caches)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx, kv_caches

def cache_size_bytes(kv_caches):
    total = 0
    for c in kv_caches:
        if c is not None:
            k, v = c
            total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total

def main():
    print("\n" + "█" * 60)
    print("█" + " " * 16 + "KV Cache 推理加速实验" + " " * 18 + "█")
    print("█" * 60)

    if not CKPT_PATH.exists():
        print(f"\n  ❌ 找不到 checkpoint,请先训练: python gpt_train.py")
        return

    model, cfg, encode, decode = load_checkpoint()
    prompt_ids = encode("ROMEO:")
    n_new = 100

    # 无 cache
    t0 = time.time()
    out1 = gen_no_cache(model, prompt_ids, n_new)
    t_no_cache = time.time() - t0

    # 有 cache
    t0 = time.time()
    out2, kvs = gen_with_cache(model, prompt_ids, n_new)
    t_cache = time.time() - t0

    print(f"\n  生成 {n_new} 个 token:")
    print(f"    无 cache: {t_no_cache:.2f}s   ({n_new/t_no_cache:.1f} tok/s)")
    print(f"    有 cache: {t_cache:.2f}s   ({n_new/t_cache:.1f} tok/s)")
    print(f"    加速比  : {t_no_cache/t_cache:.2f}×")
    print(f"    cache 大小: {cache_size_bytes(kvs)/1024:.1f} KB")
    print(f"    cache 形状(每层): K={tuple(kvs[0][0].shape)}, V={tuple(kvs[0][1].shape)}")

    # 一致性验证(不一定逐 token 一致 — 浮点 + caching 可能有微差,但前几 token 应一致)
    same = (out1[0, :len(prompt_ids)+5] == out2[0, :len(prompt_ids)+5]).all().item()
    print(f"\n  一致性(前 5 token): {'✓' if same else '✗'}")

    print("\n  关键收获:")
    print("  ✓ KV cache 把生成成本从 O(t²) 降到 O(t),典型加速 3-10×")
    print("  ✓ 代价: 显存占用 ~ 2 × n_layer × n_head × T × head_dim × 2bytes(fp16)")
    print("  ✓ LLaMA-70B 8K 上下文的 cache 约 1.6GB → 推理需要大显存")
    print("  ✓ 工业方案: PagedAttention(vLLM)分页管理 cache,避免内存碎片\n")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```bash
python ml_foundations/transformer_training/kv_cache.py
# Expect: speedup ≥ 3×, cache size shown
# Wall time: 30-90s on CPU
```

---

## Task 9: KNOWLEDGE.md

**Files:**
- Create: `ml_foundations/transformer_training/KNOWLEDGE.md`

- [ ] **Step 1: Write 7-section knowledge doc**

Outline (~900 lines, write each section ~120 lines):

1. **学习路径**：列出 7 个 demo 推荐顺序
2. **BPE Tokenization** — 算法 + cl100k_base 对比
3. **自注意力数学推导** — Q,K,V 含义 + softmax + multi-head
4. **位置编码** — sinusoidal / learned / RoPE 对比表
5. **Transformer Block 装配** — pre-LN, residual, FFN×4
6. **训练循环** — AdamW, cosine warmup, grad_clip
7. **生成与采样** — 4 种策略 + 何时选哪个
8. **推理优化** — KV cache + 引子 FlashAttention/PagedAttention
9. **与 GPT-2/LLaMA 的对应关系** — 参数量缩放 + 架构演进
10. **配套代码索引**

Each section uses the same style as `ml_foundations/classical/KNOWLEDGE.md` and `ml_foundations/deep_learning/KNOWLEDGE.md` (already in repo). LaTeX where useful, ASCII diagrams, ":::tip LLM 视角" callouts.

- [ ] **Step 2: Internal links + sanity check**

Verify all links to demo files use relative paths:
```
[BPE 实现](./bpe_tokenizer.py)
```

---

## Task 10: docs/ml-foundations/transformer-training/index.md

**Files:**
- Create: `docs/ml-foundations/transformer-training/index.md`

- [ ] **Step 1: Write chapter overview (~250 lines)**

Mirror style of `docs/ml-foundations/index.md`:
- Why this chapter
- Learning path diagram
- Sub-document links (7 entries)
- Hardware estimates (CPU 5-6 min for main training)
- "完成后你能..." checklist

---

## Task 11: docs/.../tokenization.md

**Files:**
- Create: `docs/ml-foundations/transformer-training/tokenization.md`

- [ ] **Step 1: Write ~600 lines covering:**
  - Why tokenization
  - Byte-level vs char-level vs subword
  - BPE algorithm step-by-step
  - GPT-2 / LLaMA / GPT-4 vocab sizes compared
  - Worked example matching the demo
  - LLM tip: byte fallback, special tokens

---

## Task 12: docs/.../attention.md

**Files:**
- Create: `docs/ml-foundations/transformer-training/attention.md`

- [ ] **Step 1: Write ~800 lines covering:**
  - The "search" metaphor for Q/K/V
  - Math: softmax(QK^T/√d)V derivation
  - Why divide by √d
  - Multi-head attention
  - Causal mask for decoder-only
  - Complexity O(T²d), the long-context problem
  - Modern variants: GQA / MLA (mention only)

---

## Task 13: docs/.../positional-encoding.md

**Files:**
- Create: `docs/ml-foundations/transformer-training/positional-encoding.md`

- [ ] **Step 1: Write ~500 lines covering:**
  - Why Transformer needs explicit PE
  - Sinusoidal (Vaswani 2017) — math + intuition
  - Learned PE (BERT/GPT-2) — pros/cons
  - RoPE (LLaMA) — derivation, relative property, base scaling for long context
  - ALiBi (mention only)
  - Why LLaMA-3 uses RoPE base=500K

---

## Task 14: docs/.../training.md

**Files:**
- Create: `docs/ml-foundations/transformer-training/training.md`

- [ ] **Step 1: Write ~700 lines covering:**
  - Pre-LN vs Post-LN
  - AdamW + cosine warmup recipe
  - LLM-specific: betas=(0.9, 0.95), wd=0.1, grad_clip=1.0
  - Scaling laws preview (Chinchilla)
  - Param count formula for Transformer
  - Loss curve interpretation

---

## Task 15: docs/.../generation.md

**Files:**
- Create: `docs/ml-foundations/transformer-training/generation.md`

- [ ] **Step 1: Write ~600 lines covering:**
  - Greedy / temperature / top-k / top-p
  - Repetition penalty
  - Beam search (and why LLM era doesn't use it)
  - OpenAI / Anthropic API parameters mapping
  - Speculative decoding (introduce only)

---

## Task 16: docs/.../inference.md

**Files:**
- Create: `docs/ml-foundations/transformer-training/inference.md`

- [ ] **Step 1: Write ~600 lines covering:**
  - KV cache derivation + memory estimate
  - Prefill vs decode phase
  - Attention visualization techniques
  - Real-world: PagedAttention, FlashAttention, continuous batching (intro only)
  - vLLM / TGI / llama.cpp summary

---

## Task 17: Integrate into sidebar / README / .gitignore

**Files:**
- Modify: `docs/.vitepress/config.ts`
- Modify: `README.md`

- [ ] **Step 1: Update sidebar**

In `docs/.vitepress/config.ts`, after the existing `"零、ML 基础（前置补课）"` block, insert:

```typescript
{
  text: "零.5、Transformer 训练实战",
  collapsed: true,
  items: [
    { text: "本章导读", link: "/ml-foundations/transformer-training/" },
    { text: "BPE Tokenization", link: "/ml-foundations/transformer-training/tokenization" },
    { text: "自注意力机制", link: "/ml-foundations/transformer-training/attention" },
    { text: "位置编码", link: "/ml-foundations/transformer-training/positional-encoding" },
    { text: "完整训练流程", link: "/ml-foundations/transformer-training/training" },
    { text: "文本生成与采样", link: "/ml-foundations/transformer-training/generation" },
    { text: "推理优化与 KV Cache", link: "/ml-foundations/transformer-training/inference" },
  ],
},
```

- [ ] **Step 2: Update README**

Insert "0.5 Transformer Training from Scratch" section after the "0. ML Foundations" subsection per spec §7. Update the project structure tree to include `ml_foundations/transformer_training/`.

- [ ] **Step 3: Verify .gitignore was updated in Task 1**

```bash
grep "ckpt.pt" .gitignore
# Expect: lines added in Task 1 step 3
```

---

## Task 18: Smoke-test all demos end-to-end

- [ ] **Step 1: Syntax check**

```bash
python -m py_compile ml_foundations/transformer_training/*.py
echo "Syntax OK if no error"
```

- [ ] **Step 2: Run all demos sequentially**

```bash
cd ml_foundations/transformer_training
time python bpe_tokenizer.py            # < 5s
time python attention_from_scratch.py    # < 3s
time python positional_encoding.py       # < 3s
time python gpt_train.py                  # 5-6 min CPU / 30s MPS
time python sampling_strategies.py        # 30-60s
time python attention_visualization.py    # < 5s
time python kv_cache.py                   # 30-90s
```

- [ ] **Step 3: Verify acceptance criteria**

Check each per spec §9:
- ✅ All 7 demos complete without error
- ✅ `gpt_train.py` produces ckpt.pt
- ✅ Generated text resembles English (not gibberish — vowels/consonants alternate, contains words like "the")
- ✅ KV cache speedup ≥ 3×
- ✅ No new dependencies needed

---

## Task 19: Single final commit

- [ ] **Step 1: Stage all changes**

```bash
git add ml_foundations/transformer_training/ \
        docs/ml-foundations/transformer-training/ \
        docs/.vitepress/config.ts \
        README.md \
        .gitignore
git status --short
```

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
feat(ml): add transformer_training module + docs

Phase 2 of the ML foundations track: build a complete from-scratch
Transformer training walkthrough.

Code (ml_foundations/transformer_training/, all CPU-runnable):
- bpe_tokenizer.py: from-scratch BPE training, with cl100k_base comparison
- attention_from_scratch.py: NumPy + PyTorch dual implementations of
  causal multi-head self-attention, validated against
  F.scaled_dot_product_attention
- positional_encoding.py: sinusoidal / learned / RoPE comparison +
  RoPE relative-position property verification
- gpt_train.py: ~3M-parameter decoder-only Transformer trained on
  Tiny Shakespeare in ~5min on Mac CPU / ~30s on MPS; saves ckpt.pt
  for downstream demos
- sampling_strategies.py: greedy / temperature / top-k / top-p
  side-by-side generation comparison
- attention_visualization.py: ASCII heatmaps of multi-layer multi-head
  attention patterns
- kv_cache.py: incremental KV cache implementation showing 3-10×
  inference speedup

Docs (docs/ml-foundations/transformer-training/):
- index, tokenization, attention, positional-encoding, training,
  generation, inference — each section ties theory back to the demos
  and to modern LLM practice (LLaMA RoPE, vLLM PagedAttention, etc.)

Integration:
- VitePress sidebar adds "零.5、Transformer 训练实战" between
  "零、ML 基础" and "一、LLM 基础"
- README adds "0.5 Transformer Training from Scratch" subsection
- .gitignore excludes ckpt.pt / loss_log.json / runs/
- Tiny Shakespeare corpus (~1.1MB, public domain) vendored in
  ml_foundations/transformer_training/data/

No new dependencies: reuses numpy/torch/tiktoken from Phase 1.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify commit**

```bash
git log --oneline -3
git show --stat HEAD | head -50
```

---

## Self-Review Checklist (after writing the plan)

- ✅ Spec coverage: every spec §3-§9 item maps to a task
- ✅ No placeholders: all code blocks contain real, runnable code
- ✅ Type consistency: `GPT`, `GPTConfig`, `load_checkpoint`, `encode`, `decode` are defined in Task 5 and consistently imported in Tasks 6/7/8
- ✅ Docs/code paths match spec §3 / §5
- ✅ Acceptance criteria from spec §9 verified in Task 18
