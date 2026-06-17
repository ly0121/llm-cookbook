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
