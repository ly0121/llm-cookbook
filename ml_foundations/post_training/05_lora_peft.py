"""
╔══════════════════════════════════════════════════════════════════╗
║  05_lora_peft.py — PEFT 库的工程化 LoRA                            ║
║                                                                  ║
║  核心问题：target_modules 怎么选？adapter 文件多大？merge 怎么做？   ║
║  与生产对应：HuggingFace 微调 90% 用这套                            ║
╚══════════════════════════════════════════════════════════════════╝

与 Task 5（03_sft_full.py）对比：
  - 同样的 base model、同样的数据、同样的 SFTTrainer
  - 不同：只训练 q_proj + v_proj 的 LoRA 权重（~0.5M 可训参数）
  - 好处：adapter 文件 ~6MB，base model ~1GB；推理时可动态插拔
"""
import json
import sys
import time
from pathlib import Path

# ── 保护性导入（各自独立 try/except，方便定位缺包） ──────────────────────

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
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError:
    print("❌ 需要 peft：pip install 'peft>=0.13.0'")
    sys.exit(1)

try:
    from datasets import Dataset
except ImportError:
    print("❌ 需要 datasets：pip install datasets")
    sys.exit(1)

# ── 常量 ───────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = Path(__file__).parent / "data" / "alpaca_mini.jsonl"
OUT_DIR = Path(__file__).parent / "runs" / "05_lora_peft"
MERGED_DIR = Path(__file__).parent / "runs" / "05_lora_peft_merged"

GEN_PROMPT = "用一句话解释什么是注意力机制。"


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


def generate_one(model, tokenizer, device, prompt: str) -> str:
    model.eval()
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt", add_generation_prompt=True
    )
    # apply_chat_template with return_tensors="pt" returns a plain tensor
    if hasattr(enc, "input_ids"):
        ids = enc.input_ids.to(device)
    else:
        ids = enc.to(device)
    input_len = ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=80, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()


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

    # ── 加载 base model ────────────────────────────────────────────
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
    print(f"✅ 模型总参数量：{n_params / 1e6:.1f}M")

    # ── LoRA 配置 ──────────────────────────────────────────────────
    # task_type 可用字符串 "CAUSAL_LM" 或 TaskType.CAUSAL_LM（peft>=0.13 均支持）
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # 只适配注意力的 Q/V 投影
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    print("\n=== LoRA 参数统计 ===")
    model.print_trainable_parameters()  # 预期：~0.5M trainable, ~0.1% of total

    # ── 数据集 ────────────────────────────────────────────────────
    ds = load_dataset_from_jsonl()
    print(f"✅ 数据集：{len(ds)} 条 Alpaca 样本")

    # ── SFTConfig ─────────────────────────────────────────────────
    # MPS 内存有限：batch=1, accum=4 保持等效批量梯度；max_length=128 减少序列长度
    # lr=2e-4 比全参 SFT 的 2e-5 高 10×：LoRA 参数量小，需要更大步长驱动收敛
    config = SFTConfig(
        output_dir=str(OUT_DIR),
        max_steps=50,
        per_device_train_batch_size=1,     # MPS 内存限制，等效 batch=4（accum=4）
        gradient_accumulation_steps=4,     # 与 Task 5 相同设置，保证公平对比
        learning_rate=2e-4,                # LoRA 标准经验值：比全参高 10×
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=False,
        fp16=False,
        max_length=128,                    # TRL >=1.0 字段名（旧版叫 max_seq_length）
        packing=False,
        dataloader_num_workers=0,          # MPS 不支持 pin_memory，禁用 worker
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    # ── 训练 ──────────────────────────────────────────────────────
    print(f"\n=== 开始 LoRA 训练（50 steps，预计 MPS ~10-15 min / CPU ~20 min） ===")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"✅ 训练完成，耗时 {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── 保存 adapter ───────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))          # 只写 adapter 权重，不含 base
    tokenizer.save_pretrained(str(OUT_DIR))

    adapter_size_bytes = sum(
        f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()
    )
    adapter_size_mb = adapter_size_bytes / 1024 ** 2
    print(f"\n✅ adapter 已保存到 {OUT_DIR}（共 {adapter_size_mb:.2f} MB）")
    print(f"   对比：base model Qwen2.5-0.5B ~1 GB，adapter 仅 {adapter_size_mb/1000:.2f}x 大小")

    # ── 重新加载 adapter 并 merge_and_unload ──────────────────────
    print("\n=== 重新加载 adapter + merge_and_unload 演示 ===")
    base2 = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32
    ).to(device)
    peft_model = PeftModel.from_pretrained(base2, str(OUT_DIR))

    # 生成：adapter 挂载版
    resp_peft = generate_one(peft_model, tokenizer, device, GEN_PROMPT)
    print(f"[adapter 挂载] Q: {GEN_PROMPT}")
    print(f"              A: {resp_peft}")

    # merge：把 B·A 加回 W₀，变成普通 nn.Linear
    merged = peft_model.merge_and_unload()

    # 生成：merge 版
    resp_merged = generate_one(merged, tokenizer, device, GEN_PROMPT)
    print(f"\n[merge 后]    Q: {GEN_PROMPT}")
    print(f"              A: {resp_merged}")

    match = resp_peft == resp_merged
    print(f"\n✅ 两者输出{'一致' if match else '不一致（浮点误差可接受）'}")

    # 保存 merged model（可选，~1GB）
    # MERGED_DIR.mkdir(parents=True, exist_ok=True)
    # merged.save_pretrained(str(MERGED_DIR))
    # print(f"✅ merged model 保存到 {MERGED_DIR}（与 base 同尺寸）")

    # ── 峰值内存 ───────────────────────────────────────────────────
    print_peak_memory(device)

    # ── 关键收获 ───────────────────────────────────────────────────
    print("\n=== 关键收获 ===")
    print(f"1. r=8 + target=[q_proj, v_proj]：adapter 仅 ~{adapter_size_mb:.1f} MB，base model ~1 GB，节省 99%+ 存储")
    print("2. lr=2e-4 比全参 SFT 的 2e-5 高 10×，是 LoRA 的常见经验值（参数少，需要更大步长）")
    print("3. merge_and_unload 把 B·A 矩阵加回 W₀，输出是普通 nn.Module，可被任意框架加载")
    print("4. target_modules 加入 mlp（gate/up/down_proj）能进一步提效但会翻倍 adapter 参数量")
    print("5. PeftModel.from_pretrained 支持热插拔：同一 base model 可在推理时动态切换多个 adapter")


if __name__ == "__main__":
    main()
