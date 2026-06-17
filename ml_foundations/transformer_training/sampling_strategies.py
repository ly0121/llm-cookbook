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
