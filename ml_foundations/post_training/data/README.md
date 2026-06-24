# Phase 3 数据集说明

| 文件 | 行数 | 来源 | 用途 |
|------|------|------|------|
| `alpaca_mini.jsonl` | 200 | `tatsu-lab/alpaca` 随机抽样（seed=42） | demo 01/03/05/06/07 SFT/LoRA 训练 |
| `dpo_pairs_mini.jsonl` | 100 | `HuggingFaceH4/ultrafeedback_binarized` 随机抽样（seed=42） | demo 08 DPO 偏好对齐 |

## 重新生成

```bash
python ml_foundations/post_training/data/_gen_alpaca_mini.py
python ml_foundations/post_training/data/_gen_dpo_pairs_mini.py
```

数据是公有领域 / Apache 2.0 许可，可直接 commit。
