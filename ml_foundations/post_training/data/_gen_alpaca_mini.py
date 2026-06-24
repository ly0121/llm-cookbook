"""一次性生成器：从 stanford-alpaca 抽样 200 条到 alpaca_mini.jsonl"""
import json
import random
from datasets import load_dataset

random.seed(42)
ds = load_dataset("tatsu-lab/alpaca", split="train")
indices = random.sample(range(len(ds)), 200)
out_path = "ml_foundations/post_training/data/alpaca_mini.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for i in indices:
        row = ds[i]
        f.write(json.dumps({
            "instruction": row["instruction"],
            "input": row["input"],
            "output": row["output"],
        }, ensure_ascii=False) + "\n")
print(f"Wrote 200 rows to {out_path}")
