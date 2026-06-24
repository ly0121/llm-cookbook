"""一次性生成器：从 UltraFeedback-binarized 抽样 100 对到 dpo_pairs_mini.jsonl"""
import json
import random
from datasets import load_dataset

random.seed(42)
ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
indices = random.sample(range(len(ds)), 100)
out_path = "ml_foundations/post_training/data/dpo_pairs_mini.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for i in indices:
        row = ds[i]
        f.write(json.dumps({
            "prompt": row["prompt"],
            "chosen": row["chosen"][-1]["content"],
            "rejected": row["rejected"][-1]["content"],
        }, ensure_ascii=False) + "\n")
print(f"Wrote 100 rows to {out_path}")
