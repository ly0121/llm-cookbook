"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:RNN / LSTM 字符级语言模型                            ║
║         理解为什么 Transformer 必须取代 RNN                       ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题:RNN 如何处理序列?LSTM 解决了什么?】
═══════════════════════════════════════════════════════════════════

  任务:训练一个字符级语言模型 — 给定前面若干字符,预测下一个字符。
       这正是 LLM 的迷你版(只是 LLM 用 token 而不是字符)。

  ┌─────────────────────────────────────────────────────────────┐
  │   RNN 的核心递归:                                              │
  │                                                             │
  │     hₜ = tanh(W_h·hₜ₋₁ + W_x·xₜ + b)                       │
  │     yₜ = softmax(W_y·hₜ + b_y)                             │
  │                                                             │
  │   每一步都依赖上一步的隐状态 → 无法并行!                        │
  │   长序列下梯度连乘 → 消失或爆炸                                │
  │                                                             │
  │   LSTM 用三个门控解决:                                        │
  │     fₜ = σ(W_f[hₜ₋₁,xₜ])    遗忘门 — 丢多少旧记忆            │
  │     iₜ = σ(W_i[hₜ₋₁,xₜ])    输入门 — 写多少新信息             │
  │     oₜ = σ(W_o[hₜ₋₁,xₜ])    输出门 — 输出多少                │
  │     cₜ = fₜ⊙cₜ₋₁ + iₜ⊙ĉₜ   细胞状态 — 长期记忆通道           │
  └─────────────────────────────────────────────────────────────┘

  与 Transformer 的对比:
    | 维度       | RNN/LSTM           | Transformer            |
    |-----------|-------------------|------------------------|
    | 并行性     | 必须按时间串行      | 整个序列并行            |
    | 长程依赖   | 衰减(LSTM 改善)   | 任意位置直连(attention) |
    | 训练效率   | 慢                 | 快几十倍(单 GPU)       |
    | 可扩展     | 难超 4 层          | 96+ 层                 |

  为什么仍然学 RNN?
    1. 理解"序列建模"的基础范式
    2. 状态空间模型(Mamba)是 RNN 的复兴
    3. KV-cache 推理本质是 RNN 思想(逐步消费)
"""

import string
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# ─────────────────────────────────────────────────────────────
# 训练语料(中英混合,无需下载)
# ─────────────────────────────────────────────────────────────
CORPUS = """
机器学习是人工智能的核心分支之一,通过算法让计算机从数据中学习规律。
深度学习是机器学习的一个子领域,使用多层神经网络处理复杂模式。
神经网络由层组成,每层包含若干神经元,通过权重连接相邻层。
反向传播算法通过链式法则计算梯度,使深度网络可以训练。
卷积神经网络擅长处理图像,循环神经网络擅长处理序列数据。
注意力机制让模型动态关注输入的不同部分,是 Transformer 的核心。
Transformer 架构通过自注意力实现高效并行计算,推动了大语言模型的兴起。
预训练加微调的范式让模型可以适应各种下游任务。
机器学习的成功依赖于充足的数据、合适的算法和强大的计算资源。
""".strip()


# ─────────────────────────────────────────────────────────────
# 词表与编码
# ─────────────────────────────────────────────────────────────
def build_vocab(text):
    """字符级词表(包括所有出现过的字符)。"""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return chars, stoi, itos


def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)


# ─────────────────────────────────────────────────────────────
# 数据切分:产生(输入序列,下一字符)对
# ─────────────────────────────────────────────────────────────
def make_batches(data, seq_len, batch_size):
    """随机抽 batch_size 个长 seq_len 的序列,target 是错位 1 位。"""
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[s : s + seq_len] for s in starts])
    y = torch.stack([data[s + 1 : s + 1 + seq_len] for s in starts])
    return x, y


# ─────────────────────────────────────────────────────────────
# LSTM 语言模型
# ─────────────────────────────────────────────────────────────
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (B, T) → emb: (B, T, embed_dim)
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)  # out: (B, T, hidden)
        logits = self.fc(out)  # (B, T, vocab_size)
        return logits, hidden


# ─────────────────────────────────────────────────────────────
# 训练
# ─────────────────────────────────────────────────────────────
def train(model, data, vocab_size, n_steps=400, seq_len=40, batch_size=16, lr=3e-3):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("\n" + "═" * 60)
    print(f"  训练 {n_steps} steps  seq_len={seq_len}  batch={batch_size}  lr={lr}")
    print("═" * 60)
    print(f"  {'step':>5s}  {'loss':>8s}  {'perplexity':>11s}  示意")

    for step in range(1, n_steps + 1):
        x, y = make_batches(data, seq_len, batch_size)
        logits, _ = model(x)
        # logits: (B, T, V) → (B*T, V),  y: (B, T) → (B*T,)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

        optim.zero_grad()
        loss.backward()
        # LSTM 容易梯度爆炸,做梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        if step in [1, 50, 100, 200, 300, n_steps]:
            ppl = torch.exp(loss).item()
            bar = "█" * max(1, int(40 - loss.item() * 8))
            print(f"  {step:>5d}  {loss.item():>8.4f}  {ppl:>11.2f}  {bar}")


# ─────────────────────────────────────────────────────────────
# 文本生成
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, prompt, stoi, itos, max_new=80, temperature=0.8):
    model.eval()
    ids = encode(prompt, stoi).unsqueeze(0)  # (1, T)
    hidden = None

    # 先把 prompt 喂进去,拿到 hidden
    logits, hidden = model(ids, hidden)
    last_logit = logits[0, -1]

    out = list(prompt)
    for _ in range(max_new):
        # 温度采样:除以 T 再 softmax
        probs = F.softmax(last_logit / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        out.append(itos[next_id])
        # 把新字符喂进去,继续往前
        new_input = torch.tensor([[next_id]])
        logits, hidden = model(new_input, hidden)
        last_logit = logits[0, -1]
    return "".join(out)


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "█" * 60)
    print("█" + " " * 14 + "RNN/LSTM 字符级语言模型" + " " * 21 + "█")
    print("█" * 60)

    chars, stoi, itos = build_vocab(CORPUS)
    print(f"\n  语料长度: {len(CORPUS)} 字符")
    print(f"  词表大小: {len(chars)}")
    print(f"  字符样本: {chars[:30]}{'...' if len(chars) > 30 else ''}")

    data = encode(CORPUS, stoi)

    # 字符频率
    counter = Counter(CORPUS)
    print(f"\n  Top 10 高频字符:")
    for c, cnt in counter.most_common(10):
        marker = c if c.strip() else "<space>" if c == " " else "<\\n>"
        print(f"    '{marker}'  count={cnt}")

    # 模型
    model = CharLSTM(vocab_size=len(chars), embed_dim=64, hidden_dim=128, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  模型: CharLSTM  参数 {n_params:,}")

    # 训练前的"胡言乱语"
    print("\n  ──── 训练前生成(随机权重,完全无意义) ────")
    sample = generate(model, "机器学习", stoi, itos, max_new=40)
    print(f"  {sample}")

    # 训练
    train(model, data, vocab_size=len(chars), n_steps=400, seq_len=40, batch_size=16, lr=3e-3)

    # 训练后再生成
    print("\n" + "═" * 60)
    print("  训练后生成(温度采样):")
    print("═" * 60)
    for prompt in ["机器学习", "深度", "Transformer", "神经网络"]:
        sample = generate(model, prompt, stoi, itos, max_new=60, temperature=0.8)
        print(f"\n  prompt = '{prompt}'")
        print(f"  生成   = '{sample}'")

    print("\n" + "═" * 60)
    print("  关键收获:")
    print("═" * 60)
    print("  ✓ 字符级语言建模 = 给 (xₜ₋ₙ,...,xₜ₋₁) 预测 xₜ")
    print("  ✓ LSTM 用门控 + 细胞状态缓解了梯度消失")
    print("  ✓ 温度采样:T 大→更随机, T 小→更确定 (LLM 同样原理)")
    print("  ✓ 由于无法并行,RNN 在大模型时代被 Transformer 取代")
    print("  ✓ 但'状态空间模型'(Mamba)是改进版 RNN 的复兴\n")


if __name__ == "__main__":
    main()
