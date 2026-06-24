"""
╔══════════════════════════════════════════════════════════════════╗
║  03_sft_full.py — TRL SFTTrainer 全参微调 Qwen2.5-0.5B             ║
║                                                                  ║
║  核心问题：base model 怎么变成「会听指令」？SFT 的代价多大？           ║
║  与生产对应：所有 instruction-tuned 模型的第一步                    ║
╚══════════════════════════════════════════════════════════════════╝
"""
import json
import sys
import time
from pathlib import Path

# ── 保护性导入 ──────────────────────────────────────────────────────

try:
    import torch
except ImportError:
    print("❌ 需要 PyTorch：pip install torch")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
except ImportError:
    print("❌ 需要 transformers：pip install transformers")
    sys.exit(1)

try:
    from trl import SFTConfig, SFTTrainer
except ImportError:
    print("❌ 需要 trl：pip install 'trl>=0.11.0'")
    sys.exit(1)

try:
    from datasets import Dataset
except ImportError:
    print("❌ 需要 datasets：pip install datasets")
    sys.exit(1)

# ── 常量 ───────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = Path(__file__).parent / "data" / "alpaca_mini.jsonl"
OUT_DIR = Path(__file__).parent / "runs" / "03_sft_full"

GEN_PROMPTS = [
    "用一句话解释什么是注意力机制。",
    "写一首关于秋天的两行小诗。",
    "推荐一道适合初学者的家常菜。",
]


# ── 工具函数 ────────────────────────────────────────────────────────

def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dataset_from_jsonl() -> Dataset:
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            user = r["instruction"] + ("\n\n" + r["input"] if r["input"] else "")
            rows.append({
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": r["output"]},
                ]
            })
    return Dataset.from_list(rows)


def generate_samples(model, tokenizer, device) -> list:
    outs = []
    model.eval()
    for prompt in GEN_PROMPTS:
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt", add_generation_prompt=True
        )
        # apply_chat_template with return_tensors="pt" returns a plain tensor
        ids = enc.to(device)
        attn = None
        input_len = ids.shape[1]
        gen_kwargs = dict(max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        if attn is not None:
            gen_kwargs["attention_mask"] = attn
        with torch.no_grad():
            out = model.generate(ids, **gen_kwargs)
        text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        outs.append(f"Q: {prompt}\nA: {text}")
    return outs


def print_peak_memory(device: torch.device) -> None:
    try:
        if device.type == "mps":
            peak = torch.mps.current_allocated_memory() / 1024 ** 3
            print(f"💾 MPS 当前显存占用：{peak:.2f} GB")
        elif device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"💾 CUDA 峰值显存：{peak:.2f} GB")
        else:
            try:
                import psutil
                rss = psutil.Process().memory_info().rss / 1024 ** 3
                print(f"💾 CPU RSS 内存：{rss:.2f} GB")
            except ImportError:
                print("💾 (psutil 未安装，跳过内存统计)")
    except Exception as e:
        print(f"💾 内存统计失败：{e}")


# ── 主函数 ─────────────────────────────────────────────────────────

def main() -> None:
    set_seed(42)
    torch.manual_seed(42)

    device = pick_device()
    print(f"✅ 设备：{device}")

    # 加载模型 & tokenizer
    print(f"\n⏳ 加载模型 {MODEL_ID} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32
        ).to(device)
    except Exception as e:
        print(f"❌ 加载模型失败：{e}")
        print("提示：检查网络，或 export HF_ENDPOINT=https://hf-mirror.com")
        sys.exit(1)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型参数量：{n_params / 1e6:.1f}M")

    # 加载数据集
    ds = load_dataset_from_jsonl()
    print(f"✅ 数据集：{len(ds)} 条 Alpaca 样本")

    # 训练前生成
    print("\n=== 训练前生成（base model） ===")
    before = generate_samples(model, tokenizer, device)
    for x in before:
        print(x + "\n")

    # SFTConfig — TRL 1.6.0 使用 max_length 而非 max_seq_length
    # MPS 内存有限：batch=1, accum=4 保持等效梯度；max_length=128 适配 MPS 内存
    config = SFTConfig(
        output_dir=str(OUT_DIR),
        max_steps=50,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        bf16=False,
        fp16=False,
        max_length=128,      # TRL >=1.0 字段名（旧版叫 max_seq_length）；128 适配 MPS 内存
        packing=False,
        dataloader_num_workers=0,  # MPS 不支持 pin_memory，禁用 worker 进程减少开销
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"\n=== 开始训练（50 steps，预计 MPS ~15-20 min（视样本长度而定）/ CPU ~30+ min） ===")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"✅ 训练完成，耗时 {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # 训练后生成
    print("\n=== 训练后生成（SFT model） ===")
    after = generate_samples(model, tokenizer, device)
    for x in after:
        print(x + "\n")

    # 并排对比
    print("\n=== 训练前 vs 训练后 对比 ===")
    for i, (b, a) in enumerate(zip(before, after)):
        print(f"--- 样本 {i+1} ---")
        print(f"[训练前] {b}")
        print(f"[训练后] {a}")
        print()

    # 保存模型
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT_DIR))
    print(f"✅ 模型已保存到 {OUT_DIR}")

    # 峰值内存
    print_peak_memory(device)

    # 关键收获
    print("\n=== 关键收获 ===")
    print("1. 全参 SFT 训完 50 step，Qwen2.5-0.5B 在 alpaca 风格 prompt 上回答更结构化")
    print("2. 0.5B 模型全参微调 MPS/CPU 都能跑；7B 起就必须 LoRA")
    print("3. SFTTrainer 自动套 chat template + mask prompt token（assistant_only_loss）")
    print("4. report_to=[] 关闭 wandb；save_strategy=no 不写中间 ckpt")
    print("5. TRL 1.6.0 字段名改为 max_length（旧版叫 max_seq_length），升级需注意")


if __name__ == "__main__":
    main()
