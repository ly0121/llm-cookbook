"""
╔══════════════════════════════════════════════════════════════════╗
║  01_data_construction.py — SFT 数据从原始 JSON 到三张量            ║
║                                                                  ║
║  核心问题：为什么 SFT 损失只对 response token 算，不对 prompt？      ║
║  与生产对应：transformers DataCollatorForLanguageModeling 在做啥   ║
╚══════════════════════════════════════════════════════════════════╝
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, set_seed

DATA_PATH = Path(__file__).parent / "data" / "alpaca_mini.jsonl"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def load_alpaca_mini(n: int = 3) -> list[dict]:
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def apply_chat_template(tokenizer, row: dict) -> str:
    user_msg = row["instruction"]
    if row["input"]:
        user_msg += "\n\n" + row["input"]
    messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": row["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def build_input_and_labels(tokenizer, row: dict) -> dict:
    """对 prompt 段 mask（labels = -100），只对 response 算 loss。"""
    user_msg = row["instruction"]
    if row["input"]:
        user_msg += "\n\n" + row["input"]
    prompt_msgs = [{"role": "user", "content": user_msg}]
    prompt_str = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_str = apply_chat_template(tokenizer, row)

    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_str, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def print_three_tensors(tokenizer, sample: dict, max_show: int = 40) -> None:
    print("─" * 70)
    print(f"  token | id    | label  | text")
    print("─" * 70)
    for i, (tid, lab) in enumerate(zip(sample["input_ids"], sample["labels"])):
        if i >= max_show:
            print(f"  ... (still {len(sample['input_ids']) - max_show} tokens)")
            break
        tok = tokenizer.decode([tid]).replace("\n", "\\n")
        lab_show = "MASKED" if lab == -100 else f"{lab}"
        print(f"  {i:5d} | {tid:5d} | {lab_show:>6} | {tok!r}")


def main() -> None:
    set_seed(42)
    torch.manual_seed(42)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"❌ 下载 tokenizer 失败：{e}")
        print("提示：检查网络 / 设置 HF_ENDPOINT=https://hf-mirror.com")
        sys.exit(1)

    rows = load_alpaca_mini(n=3)
    print(f"✅ 加载 {len(rows)} 条 Alpaca 样本，使用 tokenizer: {MODEL_ID}\n")

    for idx, row in enumerate(rows):
        print(f"=== 样本 {idx + 1} ===")
        print(f"instruction: {row['instruction'][:80]}…")
        sample = build_input_and_labels(tokenizer, row)
        print(f"序列长度：{len(sample['input_ids'])} tokens；被 mask: "
              f"{sum(1 for l in sample['labels'] if l == -100)}")
        print_three_tensors(tokenizer, sample, max_show=30)
        print()

    print("=== 关键收获 ===")
    print("1. labels = input_ids，但 prompt 段被替换为 -100（忽略 loss）")
    print("2. 只有 assistant 回复段参与梯度计算，模型学的是「怎么回答」")
    print("3. 如果不 mask prompt，模型会同时学「怎么提问」，浪费容量且伤害指令跟随")
    print("4. chat template 自动注入 <|im_start|>/<|im_end|> 等 Qwen 特殊 token")


if __name__ == "__main__":
    main()
