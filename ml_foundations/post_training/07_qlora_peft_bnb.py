"""
╔══════════════════════════════════════════════════════════════════╗
║  07_qlora_peft_bnb.py — bitsandbytes 4-bit QLoRA（CUDA 路线）        ║
║                                                                  ║
║  核心问题：为什么 4-bit base + LoRA adapter 在 24GB GPU 上能微调 7B？  ║
║  与生产对应：HF PEFT + bitsandbytes NF4，业界 SOTA 单卡 QLoRA 配方     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import json
import sys
from pathlib import Path

import torch

# ── Early device check (before optional heavy imports) ─────────────────────
# bitsandbytes 4-bit only works on CUDA; bail early on Mac MPS / CPU.
if not torch.cuda.is_available():
    print(
        "bitsandbytes 4-bit 不支持 Apple MPS / CPU。"
        "请使用 06_qlora_mlx.py（Apple Silicon）"
        "或在 Google Colab/CUDA 环境运行本脚本。"
    )
    sys.exit(0)

from datasets import Dataset
from transformers import AutoTokenizer, set_seed

# ── 路径常量 ────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_HERE = Path(__file__).parent
DATA_PATH = _HERE / "data" / "alpaca_mini.jsonl"
OUT_DIR = _HERE / "runs" / "07_qlora_peft_bnb"

SEED = 42


# ── 步骤 0：环境检测 ─────────────────────────────────────────────────────────

def check_env() -> None:
    """检查依赖包是否齐全（CUDA 已由模块级早检测确认）。"""
    # 检测：CUDA 可用但缺包 → 提示安装后退出
    missing = []
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        missing.append("bitsandbytes")

    try:
        import peft  # noqa: F401
    except ImportError:
        missing.append("peft")

    try:
        import trl  # noqa: F401
    except ImportError:
        missing.append("trl")

    if missing:
        print(f"缺少依赖包：{', '.join(missing)}")
        print("请先安装：pip install bitsandbytes peft trl")
        sys.exit(0)


# ── 步骤 1：加载数据集 ────────────────────────────────────────────────────────

def load_dataset() -> Dataset:
    """从 data/alpaca_mini.jsonl 加载对话数据，转为 messages 格式。"""
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
    print(f"  加载样本数：{len(rows)}")
    return Dataset.from_list(rows)


# ── 步骤 2：构建 BnB 配置 + 加载模型 ─────────────────────────────────────────

def build_bnb_config():
    """构建 bitsandbytes NF4 量化配置。"""
    from transformers import BitsAndBytesConfig  # guarded: only on CUDA path

    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("=== BitsAndBytesConfig ===")
    print("  量化类型       : NF4（4-bit Normal Float）")
    print("  compute dtype  : bfloat16（前向/反向用 bf16，权重存 4-bit）")
    print("  double quant   : True（量化常数本身再量化，省 ~0.4 bit/param）")
    return cfg


# ── 步骤 3：PEFT LoRA 配置 ───────────────────────────────────────────────────

def build_lora_config():
    """构建与 05_lora_peft.py / 06_qlora_mlx.py 对齐的 LoRA 配置。"""
    from peft import LoraConfig  # guarded

    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


# ── 步骤 4：训练 ─────────────────────────────────────────────────────────────

def run_training() -> None:
    """在 CUDA 上执行一轮 QLoRA 训练（seed=42 可复现）。"""
    from peft import get_peft_model, prepare_model_for_kbit_training  # guarded
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # guarded
    from trl import SFTConfig, SFTTrainer  # guarded
    import transformers  # for set_seed

    print(f"\n{'='*60}")
    print(f"  SEED = {SEED}")
    print(f"{'='*60}")

    # Explicit seed initialization
    transformers.set_seed(SEED)
    torch.manual_seed(SEED)

    bnb_config = build_bnb_config()

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 以 4-bit 加载 base model
    print(f"\n正在以 NF4 4-bit 加载 {MODEL_ID} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # 准备 kbit 训练（冻结 4-bit 权重，LayerNorm 转 fp32）
    model = prepare_model_for_kbit_training(model)
    print("  prepare_model_for_kbit_training 完成")

    # 注入 LoRA
    lora_cfg = build_lora_config()
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 加载数据
    ds = load_dataset()

    # TRL SFTTrainer
    sft_config = SFTConfig(
        output_dir=str(OUT_DIR),
        max_steps=50,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=True,
        max_length=256,
        packing=False,
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"\n=== 开始 QLoRA 训练（50 steps，Colab T4 ~2 min） ===")
    trainer.train()

    # 保存 LoRA adapter
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"\n4-bit base + LoRA adapter 已保存到 {OUT_DIR}")


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # 环境检测（顺序：无 CUDA → 退出；缺包 → 退出；否则继续）
    check_env()

    print("环境检测通过：CUDA 可用，依赖包齐全\n")
    print(f"Base model : {MODEL_ID}")
    print(f"Data       : {DATA_PATH}")
    print(f"Output     : {OUT_DIR}")
    print(f"SEED       : {SEED}")

    # 单次训练，SEED=42
    run_training()

    print("\n" + "="*60)
    print("=== 关键收获 ===")
    print("1. base model 以 NF4 4-bit 加载，显存比 fp16 省 ~4x")
    print("   (Qwen2.5-0.5B 从 ~1GB 降到 ~280MB)")
    print("2. compute dtype=bfloat16：4-bit 只存权重，矩阵乘还是高精度，不掉点")
    print("3. prepare_model_for_kbit_training：冻结 4-bit 权重 + LayerNorm 转 fp32")
    print("4. double quant=True：对量化常数本身二次量化，再省 ~0.4 bit/param")
    print("5. LoRA 只更新 q_proj/v_proj（r=8）：可训参数 <1%，显存极低")
    print("="*60)


if __name__ == "__main__":
    main()
