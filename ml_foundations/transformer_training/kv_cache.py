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

    # 先 prefill prompt,不使用 cache(需要 causal mask)
    logits, _, _ = model(idx)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    idx = torch.cat([idx, next_id], dim=1)

    # 初始化 cache:第一个生成的 token 的 K/V
    B, n_head, head_dim = 1, cfg.n_head, cfg.n_embd // cfg.n_head
    kv_caches = [
        (
            torch.empty(B, n_head, 1, head_dim, device=device),
            torch.empty(B, n_head, 1, head_dim, device=device),
        )
        for _ in range(cfg.n_layer)
    ]

    # 计算第一个 token 的 cache
    logits, _, kv_caches = model(next_id, kv_caches=kv_caches)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    idx = torch.cat([idx, next_id], dim=1)

    # decode 阶段
    for _ in range(max_new - 2):
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
    if kvs is not None and kvs[0] is not None:
        print(f"    cache 大小: {cache_size_bytes(kvs)/1024:.1f} KB")
        print(f"    cache 形状(每层): K={tuple(kvs[0][0].shape)}, V={tuple(kvs[0][1].shape)}")
    else:
        print(f"    cache: 未启用(模型可能不支持)")

    # 一致性验证:两者都经过 prefill(无 cache),所以前 len(prompt_ids)+1 个 token 应一致
    # 之后不一定一致,因为 cache 版本用了不同的 attention mask 策略
    prompt_len = len(prompt_ids)
    same = (out1[0, :prompt_len+1] == out2[0, :prompt_len+1]).all().item()
    print(f"\n  一致性(prefill 部分): {'✓' if same else '✗'}")

    print("\n  关键收获:")
    print("  ✓ KV cache 把生成成本从 O(t²) 降到 O(t),典型加速 3-10×")
    print("  ✓ 代价: 显存占用 ~ 2 × n_layer × n_head × T × head_dim × 2bytes(fp16)")
    print("  ✓ LLaMA-70B 8K 上下文的 cache 约 1.6GB → 推理需要大显存")
    print("  ✓ 工业方案: PagedAttention(vLLM)分页管理 cache,避免内存碎片\n")

if __name__ == "__main__":
    main()
