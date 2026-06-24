"""
╔══════════════════════════════════════════════════════════════════╗
║  08_dpo_alignment.py — TRL DPOTrainer 偏好对齐                   ║
║                                                                  ║
║  核心问题：为什么 DPO 一招打 PPO 三步？                            ║
║  与生产对应：Llama-3 / Qwen-2 等模型的对齐阶段都跑过类似流程       ║
╚══════════════════════════════════════════════════════════════════╝

DPO (Direct Preference Optimization) 直接把「chosen vs rejected」
奖励信号蒸馏进策略梯度，无需单独训练 reward model 或跑 PPO rollout。

损失函数：
  L = -log σ( β * [ log π_θ(c|x)/π_ref(c|x)
                    - log π_θ(r|x)/π_ref(r|x) ] )

本脚本：
  1. 优先从 runs/03_sft_full/ 加载 SFT checkpoint（更贴近生产流程）
  2. 若 SFT checkpoint 不存在，回退到 Qwen2.5-0.5B-Instruct base
  3. 套 LoRA adapter，ref_model=None（PEFT 自动 disable adapter 当 ref）
  4. DPOTrainer 50 step，打印 loss / reward margin 轨迹
  5. SFT 前 vs DPO 后生成对比
  6. 保存 adapter 到 runs/08_dpo/
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
    from datasets import Dataset
except ImportError:
    print("❌ 需要 datasets：pip install datasets")
    sys.exit(1)

try:
    from peft import LoraConfig, get_peft_model
    from trl import DPOConfig, DPOTrainer
except ImportError as e:
    print(f"❌ 缺包：{e}")
    print("提示：pip install 'trl>=0.11.0' 'peft>=0.13.0'")
    sys.exit(1)

# ── 路径常量 ────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
DATA_PATH = _HERE / "data" / "dpo_pairs_mini.jsonl"
SFT_CKPT = _HERE / "runs" / "03_sft_full"
OUT_DIR = _HERE / "runs" / "08_dpo"
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

EVAL_PROMPTS = [
    "我朋友刚被裁员心情很差，我该怎么安慰？",
    "解释一下为什么天空是蓝色的，给一个 10 岁的孩子听。",
]


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(device: torch.device):
    """
    优先加载 SFT checkpoint；不存在则回退 base。
    Task 5 注意事项：config.json torch_dtype=null，需显式 dtype=torch.float32。
    """
    sft_model_file = SFT_CKPT / "model.safetensors"
    if sft_model_file.exists():
        model_path = str(SFT_CKPT)
        source_label = f"SFT checkpoint ({SFT_CKPT})"
    else:
        model_path = BASE_MODEL_ID
        source_label = f"base model ({BASE_MODEL_ID})"
        print("⚠️  未找到 runs/03_sft_full/model.safetensors，"
              "回退到 base model（演示独立性）")

    print(f"📦 加载模型来源：{source_label}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32
    ).to(device)
    return model, tokenizer, source_label


def load_pairs() -> Dataset:
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "prompt": r["prompt"],
                "chosen": r["chosen"],
                "rejected": r["rejected"],
            })
    print(f"📄 加载 DPO 数据：{len(rows)} 条偏好对")
    return Dataset.from_list(rows)


def generate(model, tokenizer, device, prompts: list) -> list:
    outs = []
    model.eval()
    for p in prompts:
        result = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            return_tensors="pt",
            add_generation_prompt=True,
        )
        # newer transformers may return BatchEncoding; extract tensor
        if hasattr(result, "input_ids"):
            ids = result.input_ids.to(device)
        else:
            ids = result.to(device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(
            out[0][ids.shape[1]:], skip_special_tokens=True
        ).strip()
        outs.append(f"Q: {p}\nA: {text}")
    return outs


def main() -> None:
    set_seed(42)
    torch.manual_seed(42)

    t0 = time.time()
    device = pick_device()
    print(f"✅ 设备：{device}")

    # 1. 加载模型
    model, tokenizer, source_label = load_model_and_tokenizer(device)

    # 2. DPO 前生成（作为 before 对比）
    print("\n=== DPO 训练前生成 ===")
    before_gens = generate(model, tokenizer, device, EVAL_PROMPTS)
    for x in before_gens:
        print(x + "\n")

    # 3. 套 LoRA（省 reference model 显存）
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 4. 加载数据
    ds = load_pairs()

    # 5. DPO 配置（TRL 1.6.0：max_prompt_length removed in TRL 1.6.0; max_length=128 covers both chosen 和 rejected）
    config = DPOConfig(
        output_dir=str(OUT_DIR),
        max_steps=50,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,          # DPO 典型范围 1e-6 ~ 1e-5；plan 草拟过 5e-5，
                                       # 但 100 行 preference dataset 下更小 lr 更稳，且
                                       # 11_eval_perplexity.py 实测 DPO PPL 64.460 ≈ base
        beta=0.1,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        max_length=128,              # 同时覆盖 chosen 和 rejected 序列
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # PEFT 模型下：disable adapter → 充当 ref（省 1× 显存）
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print(f"\n=== 开始 DPO 训练（50 steps，MPS ~5 min / CPU ~15 min） ===")
    print("   DPO loss 理论起点 ≈ ln(2) ≈ 0.693")
    trainer.train()

    # 6. 打印 loss / reward margin 轨迹
    log_history = trainer.state.log_history
    print("\n=== 训练轨迹 ===")
    print(f"{'step':>6}  {'loss':>8}  {'reward_margin':>14}  {'chosen_rwd':>12}  {'rejected_rwd':>13}")
    for entry in log_history:
        if "loss" not in entry:
            continue
        step = entry.get("step", "?")
        loss = entry.get("loss", float("nan"))
        chosen = entry.get("rewards/chosen", float("nan"))
        rejected = entry.get("rewards/rejected", float("nan"))
        margin = chosen - rejected if (chosen == chosen and rejected == rejected) else float("nan")
        print(f"{step:>6}  {loss:>8.4f}  {margin:>14.4f}  {chosen:>12.4f}  {rejected:>13.4f}")

    # 7. DPO 后生成对比
    print("\n=== DPO 训练后生成 ===")
    after_gens = generate(model, tokenizer, device, EVAL_PROMPTS)
    for x in after_gens:
        print(x + "\n")

    # 8. 并排对比
    print("=== SFT / base 前 vs DPO 后 并排对比 ===")
    for i, (b, a) in enumerate(zip(before_gens, after_gens), 1):
        print(f"── 示例 {i} ──")
        print(f"[训练前] {b}")
        print(f"[DPO 后] {a}")
        print()

    # 9. 保存
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"✅ DPO adapter 已保存到 {OUT_DIR}")

    # 10. 性能统计
    elapsed = time.time() - t0
    if device.type == "mps":
        # MPS 没有 memory_reserved；用 allocated 作近似
        peak_mb = torch.mps.current_allocated_memory() / 1024 ** 2
        mem_note = f"MPS allocated {peak_mb:.0f} MB"
    elif device.type == "cuda":
        peak_mb = torch.cuda.max_memory_reserved() / 1024 ** 2
        mem_note = f"CUDA peak {peak_mb:.0f} MB"
    else:
        mem_note = "CPU（无显存统计）"
    print(f"\n⏱  总用时：{elapsed:.1f}s | 内存：{mem_note}")

    # 11. 关键收获
    print("\n=== 关键收获 ===")
    print("1. DPO 损失 = -log σ(β·[log π_θ(c)/π_ref(c) - log π_θ(r)/π_ref(r)])")
    print("   β=0.1 让策略不至于过度偏离 reference，防止 reward hacking")
    print("2. ref_model=None + PEFT：TRL 自动 disable LoRA adapter 充当 reference，"
          "节省约 1× 参数量的显存")
    print("3. DPO 比 PPO 节省 reward model + rollout 两个环节，端到端更稳定")
    print("4. reward margin = chosen_reward - rejected_reward；上升说明模型正确区分偏好")
    print("5. 50 step × 100 对数据仅能看到风格微倾斜；工业级对齐需要 50k+ 高质量偏好对")


if __name__ == "__main__":
    main()
