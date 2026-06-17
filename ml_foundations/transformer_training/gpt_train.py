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
