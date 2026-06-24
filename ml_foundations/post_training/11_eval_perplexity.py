"""
╔══════════════════════════════════════════════════════════════════╗
║  11_eval_perplexity.py — 困惑度评估 base / SFT / DPO                ║
║                                                                  ║
║  核心问题：PPL 真的能衡量「模型变好了」吗？                          ║
║  与生产对应：研究里最常用但最容易误导的单指标                        ║
╚══════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    print("缺少依赖 torch，请先安装：pip install torch")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
except ImportError:
    print("缺少依赖 transformers，请先安装：pip install transformers")
    sys.exit(1)

try:
    from datasets import load_dataset
    from peft import PeftModel
except ImportError as e:
    print(f"缺包：{e}")
    sys.exit(1)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_DIR = Path(__file__).parent / "runs" / "03_sft_full"
DPO_DIR = Path(__file__).parent / "runs" / "08_dpo"

EVAL_PROMPTS = [
    "用一句话解释什么是 RAG。",
    "推荐一种适合长跑的呼吸节奏。",
]


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_ppl(model, tokenizer, device, texts: list[str]) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for t in texts:
        ids = tokenizer(t, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        with torch.no_grad():
            out = model(ids, labels=ids)
        n_tok = ids.shape[1]
        total_loss += out.loss.item() * n_tok
        total_tokens += n_tok
    return float(torch.tensor(total_loss / total_tokens).exp())


def load_eval_texts() -> list[str]:
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:50]")
        return [r["text"] for r in ds if len(r["text"]) > 100][:30]
    except Exception as e:
        print(f"下载 WikiText 失败：{e}，使用本地小样本（可设置 HF_ENDPOINT=https://hf-mirror.com 重试）")
        return [
            "Attention is all you need. The Transformer architecture relies entirely on self-attention.",
            "Reinforcement learning from human feedback aligns models with human preferences.",
        ] * 15


def generate(model, tokenizer, device, prompts: list[str]) -> list[str]:
    outs = []
    model.eval()
    for p in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], return_tensors="pt", add_generation_prompt=True
        )
        # transformers 5.x returns BatchEncoding instead of a bare Tensor
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        ids = ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        outs.append(text)
    return outs


def load_variant(name: str, adapter_dir: Optional[Path], device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter_dir is None:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
    elif not adapter_dir.exists():
        print(f"  {name} adapter 不存在（{adapter_dir}），用 base 代替；建议先跑对应 demo")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
    else:
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)
        # 03_sft_full 保存的是全参，不是 PEFT；尝试两种加载方式
        adapter_cfg = adapter_dir / "adapter_config.json"
        if adapter_cfg.exists():
            model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        else:
            del base  # SFT full-weight: free transient base before second load
            model = AutoModelForCausalLM.from_pretrained(str(adapter_dir),
                                                          dtype=torch.float32).to(device)
    return tokenizer, model


def main() -> None:
    set_seed(42)
    torch.manual_seed(42)
    device = pick_device()
    print(f"设备：{device}")

    eval_texts = load_eval_texts()
    print(f"评估文本：{len(eval_texts)} 段（WikiText-2 验证集子集）")

    results = {}
    gens = {}
    for name, adapter in [
        ("base", None),
        ("SFT", SFT_DIR),
        ("DPO", DPO_DIR),
    ]:
        print(f"\n=== 加载 {name} ===")
        tokenizer, model = load_variant(name, adapter, device)
        ppl = compute_ppl(model, tokenizer, device, eval_texts)
        gens[name] = generate(model, tokenizer, device, EVAL_PROMPTS)
        results[name] = ppl
        print(f"  PPL = {ppl:.3f}")
        del model

    print("\n=== PPL 对比表 ===")
    print(f"  {'变体':<10} | {'PPL':>10}")
    print("  " + "─" * 25)
    for name, ppl in results.items():
        print(f"  {name:<10} | {ppl:>10.3f}")

    print("\n=== 生成对比 ===")
    for i, p in enumerate(EVAL_PROMPTS):
        print(f"\nQ: {p}")
        for name in ["base", "SFT", "DPO"]:
            print(f"  [{name}] {gens[name][i]}")

    print("\n=== 关键收获 ===")
    print("1. PPL 仅衡量「在该分布上模型有多惊讶」，与指令跟随质量弱相关")
    print("2. SFT 后 PPL 在 WikiText 上可能反而升高（因为模型偏向 instruct 分布）")
    print("3. 真正评估生成质量需要 MT-Bench / Arena / lm-eval-harness（见 12）")
    print("4. 人工 side-by-side 仍是 gold standard，PPL 是廉价代理指标")


if __name__ == "__main__":
    main()
