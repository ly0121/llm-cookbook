# Phase 3: 训练后期与对齐 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete post-training & alignment walkthrough (12 demos + 1 KNOWLEDGE.md + 9 docs) on Qwen2.5-0.5B that runs on Mac CPU/MPS, covering SFT / LoRA / QLoRA / DPO / quantization / evaluation.

**Architecture:** Self-contained numbered demos under `ml_foundations/post_training/`. Each demo loads its own HuggingFace model independently (no cross-script imports). Two QLoRA implementations (Apple MLX native + PEFT/bnb) to cover both Mac and Linux/CUDA paths. PPO is documented and demonstrated structurally but not actually trained.

**Tech Stack:** Python 3.10+, PyTorch (CPU/MPS auto-detect), HuggingFace transformers/peft/trl/datasets/accelerate, Apple MLX (`mlx-lm`) for Mac-native QLoRA, optional `bitsandbytes` + `llama-cpp-python` + `lm-eval` for cloud / quantization / benchmark demos.

**Reference spec:** `docs/superpowers/specs/2026-06-18-phase3-post-training-design.md`

## Global Constraints

- Base model: `Qwen/Qwen2.5-0.5B-Instruct` (default), `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (fallback when user prefers smaller chat-tuned baseline). Both have full chat templates.
- All demos except `07_qlora_peft_bnb.py` MUST run on Mac CPU/MPS within budget (see per-task budgets).
- Phase 1/2 convention: NO unit tests, NO CI; verification is the demo's own print output + a smoke-test pass.
- All training artifacts (adapters, merged models, GGUF files, eval caches) go to `ml_foundations/post_training/runs/` and are `.gitignore`-d. Only the two tiny `data/*.jsonl` files are committed.
- Single final commit at the very end (Task 26). Smaller commits during development MUST be amended or squashed before the final push.
- Every demo file: ASCII box-art docstring header, Chinese comments and `print` output, `if __name__ == "__main__": main()` entry, `torch.manual_seed(42)` / `transformers.set_seed(42)`, friendly Chinese error messages on missing deps / OOM / network failures.
- Devices auto-detected via `torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")`; printed at start of every training demo.
- New dependencies pinned in `requirements.txt`: `transformers>=4.45.0`, `peft>=0.13.0`, `trl>=0.11.0`, `datasets>=3.0.0`, `accelerate>=1.0.0`, plus Mac/CUDA-optional packages annotated.

---

## Task Order Overview

```
A. Scaffold + datasets               → Task 1
B. No-HF-model demos (parallel)      → Tasks 2, 3, 4
C. SFT critical path                 → Task 5  ← critical path
D. PEFT LoRA                         → Task 6
E. QLoRA dual paths (parallel)       → Tasks 7, 8
F. DPO + PPO intro (parallel)        → Tasks 9, 10
G. Quantization inference            → Task 11
H. Evaluation closure (parallel)     → Tasks 12, 13
I. KNOWLEDGE.md                      → Task 14
J. VitePress 9 docs (parallel)       → Tasks 15-23
K. Integration: sidebar/README/etc.  → Task 24
L. Smoke-test all demos              → Task 25
M. Final single commit               → Task 26
```

Parallel groups: {2,3,4}, {7,8}, {9,10}, {12,13}, {15-23}. Critical path: 1 → 5 → 9 → 11 → 12/13 → 25 → 26.

---

## Task 1: Scaffold directory + vendor mini datasets

**Files:**
- Create: `ml_foundations/post_training/__init__.py` (empty marker)
- Create: `ml_foundations/post_training/data/alpaca_mini.jsonl`
- Create: `ml_foundations/post_training/data/dpo_pairs_mini.jsonl`
- Create: `ml_foundations/post_training/data/README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing
- Produces: `data/alpaca_mini.jsonl` (200 lines, each `{"instruction": str, "input": str, "output": str}`), `data/dpo_pairs_mini.jsonl` (100 lines, each `{"prompt": str, "chosen": str, "rejected": str}`)

- [ ] **Step 1: Create directory tree**

```bash
mkdir -p ml_foundations/post_training/data
touch ml_foundations/post_training/__init__.py
```

- [ ] **Step 2: Generate `alpaca_mini.jsonl` from Alpaca**

Write a one-off generator at `ml_foundations/post_training/data/_gen_alpaca_mini.py`:

```python
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
```

Run once: `python ml_foundations/post_training/data/_gen_alpaca_mini.py`
Expected: file ~80KB, 200 lines.

- [ ] **Step 3: Generate `dpo_pairs_mini.jsonl` from UltraFeedback**

Append to the same generator file or create `_gen_dpo_pairs_mini.py`:

```python
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
```

Run once: `python ml_foundations/post_training/data/_gen_dpo_pairs_mini.py`
Expected: file ~120KB, 100 lines. If network unavailable, fall back to writing a manually authored 10-line stub and note in `data/README.md`.

- [ ] **Step 4: Write `data/README.md`**

```markdown
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
```

- [ ] **Step 5: Update `.gitignore`**

Append to end of `.gitignore`:

```
# Phase 3: Post-training artifacts
ml_foundations/post_training/runs/
ml_foundations/post_training/data/wikitext_cache/
ml_foundations/post_training/data/lm_eval_cache/
*.gguf
```

- [ ] **Step 6: Verify**

```bash
wc -l ml_foundations/post_training/data/alpaca_mini.jsonl
wc -l ml_foundations/post_training/data/dpo_pairs_mini.jsonl
ls -la ml_foundations/post_training/data/
```

Expected: 200 and 100 lines respectively, both files exist.

---

## Task 2: `01_data_construction.py` — SFT data → tokens

**Files:**
- Create: `ml_foundations/post_training/01_data_construction.py` (~200 lines)

**Interfaces:**
- Consumes: `data/alpaca_mini.jsonl` from Task 1
- Produces: nothing (teaching-only); prints `input_ids` / `labels` / `attention_mask` and a Chinese explanation of response-only loss masking

- [ ] **Step 1: Write the demo skeleton**

```python
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
```

- [ ] **Step 2: Run and verify output**

```bash
python ml_foundations/post_training/01_data_construction.py
```

Expected output starts with `✅ 加载 3 条 Alpaca 样本`, shows 3 token tables with MASKED labels for prompt tokens, ends with the 4-line "关键收获" block. Takes < 30 seconds (first run downloads ~1GB tokenizer + model config).

- [ ] **Step 3: Commit (intermediate, will be squashed in Task 26)**

```bash
git add ml_foundations/post_training/01_data_construction.py
git commit -m "wip(phase3): add 01_data_construction"
```

---

## Task 3: `02_multi_turn_chat.py` — multi-turn mask strategies

**Files:**
- Create: `ml_foundations/post_training/02_multi_turn_chat.py` (~180 lines)

**Interfaces:**
- Consumes: nothing (synthetic 4-turn conversation)
- Produces: nothing (teaching-only); prints three mask strategies side-by-side

- [ ] **Step 1: Write the demo**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║  02_multi_turn_chat.py — 多 turn 对话的 loss mask 策略             ║
║                                                                  ║
║  核心问题：4 turn user/assistant 对话，loss 该对谁算？              ║
║  与生产对应：ShareGPT 风格训练为什么能教模型「承上启下」              ║
╚══════════════════════════════════════════════════════════════════╝
"""
import sys
from transformers import AutoTokenizer, set_seed

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

CONVERSATION = [
    {"role": "user", "content": "推荐一本入门 Python 的书"},
    {"role": "assistant", "content": "《Python Crash Course》很适合入门，习题量大、节奏快。"},
    {"role": "user", "content": "如果我已经会 JavaScript 呢？"},
    {"role": "assistant", "content": "那直接看《Fluent Python》第二版，它假设你已经懂动态语言。"},
]


def build_with_strategy(tokenizer, strategy: str) -> tuple[list[int], list[int]]:
    """
    返回 (input_ids, labels)。
    strategy ∈ {"last_turn_only", "all_assistant", "everything"}
    """
    full_str = tokenizer.apply_chat_template(CONVERSATION, tokenize=False)
    full_ids = tokenizer(full_str, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(full_ids)

    if strategy == "everything":
        labels = list(full_ids)
    elif strategy == "all_assistant":
        for end_idx in range(2, len(CONVERSATION) + 1, 2):  # 2, 4
            prefix_str = tokenizer.apply_chat_template(
                CONVERSATION[:end_idx - 1], tokenize=False, add_generation_prompt=True
            )
            prefix_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
            full_so_far = tokenizer.apply_chat_template(
                CONVERSATION[:end_idx], tokenize=False
            )
            full_so_far_ids = tokenizer(full_so_far, add_special_tokens=False)["input_ids"]
            for i in range(len(prefix_ids), len(full_so_far_ids)):
                labels[i] = full_ids[i]
    elif strategy == "last_turn_only":
        prefix_str = tokenizer.apply_chat_template(
            CONVERSATION[:-1], tokenize=False, add_generation_prompt=True
        )
        prefix_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
        for i in range(len(prefix_ids), len(full_ids)):
            labels[i] = full_ids[i]
    else:
        raise ValueError(strategy)
    return full_ids, labels


def visualize(tokenizer, full_ids: list[int], labels: list[int], label: str) -> None:
    print(f"\n── 策略: {label} ──")
    weight_row = "".join("█" if l != -100 else "·" for l in labels)
    print(f"loss-mask: [{weight_row}]   (█=参与 loss, ·=mask)")
    n_active = sum(1 for l in labels if l != -100)
    print(f"参与 loss 的 token 数 = {n_active} / {len(full_ids)} "
          f"({100 * n_active / len(full_ids):.1f}%)")


def main() -> None:
    set_seed(42)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"❌ 下载 tokenizer 失败：{e}")
        sys.exit(1)

    print(f"✅ tokenizer: {MODEL_ID}")
    print(f"对话 turn 数：{len(CONVERSATION)}\n")

    for strategy, label in [
        ("everything", "everything（全部算 loss，含 user）"),
        ("last_turn_only", "last_turn_only（仅最后一个 assistant）"),
        ("all_assistant", "all_assistant（所有 assistant turn）"),
    ]:
        ids, labels = build_with_strategy(tokenizer, strategy)
        visualize(tokenizer, ids, labels, label)

    print("\n=== 关键收获 ===")
    print("1. everything：模型同时学怎么提问，浪费容量；几乎没人用")
    print("2. last_turn_only：单轮 SFT 标准做法，但学不到「承上启下」")
    print("3. all_assistant：ShareGPT 风格，所有 assistant 都参与；多轮主流方案")
    print("4. 选哪个取决于训练数据是不是真实多轮对话还是单轮指令")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```bash
python ml_foundations/post_training/02_multi_turn_chat.py
```

Expected: three mask-strategy blocks, each with a `█/·` ASCII row and a percentage; "全部算 loss" should be 100%, "last_turn_only" should be < 30%, "all_assistant" in between.

- [ ] **Step 3: Commit**

```bash
git add ml_foundations/post_training/02_multi_turn_chat.py
git commit -m "wip(phase3): add 02_multi_turn_chat"
```

---

## Task 4: `04_lora_from_scratch.py` — toy LoRA implementation

**Files:**
- Create: `ml_foundations/post_training/04_lora_from_scratch.py` (~220 lines)

**Interfaces:**
- Consumes: nothing
- Produces: prints param-count comparison and convergence curves for full-FT vs LoRA on a toy regression task

- [ ] **Step 1: Write the demo**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║  04_lora_from_scratch.py — 手写 LoRA 数学                          ║
║                                                                  ║
║  核心问题：W = W₀ + (α/r) BA 为什么 work？r 取多少够用？             ║
║  与生产对应：peft.LoraConfig 后端到底在做什么                       ║
╚══════════════════════════════════════════════════════════════════╝
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)

D_IN, D_OUT = 256, 256
N_TRAIN = 1024
LORA_R = 8
LORA_ALPHA = 16


class LoRALinear(nn.Module):
    """W·x + (α/r) B·A·x，其中 W 冻结，只训 A、B。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # B 初始 0，保证起点等价于 base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (x @ self.A.T @ self.B.T)


def make_toy_data() -> tuple[torch.Tensor, torch.Tensor]:
    """目标：从「base 任务（已学好的线性映射）」迁移到「base + delta」。"""
    W_true_base = torch.randn(D_OUT, D_IN) * 0.1
    delta = torch.randn(D_OUT, LORA_R) @ torch.randn(LORA_R, D_IN) * 0.05  # 低秩扰动
    W_true_new = W_true_base + delta
    X = torch.randn(N_TRAIN, D_IN)
    Y = X @ W_true_new.T
    return X, Y, W_true_base


def count_trainable(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def train_loop(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, steps: int = 200, lr: float = 1e-3) -> list[float]:
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    losses = []
    loader = DataLoader(TensorDataset(X, Y), batch_size=64, shuffle=True)
    it = iter(loader)
    for step in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def ascii_curve(losses: list[float], label: str, width: int = 60) -> None:
    print(f"\n── {label} (final loss = {losses[-1]:.4f}) ──")
    sampled = [losses[int(i * (len(losses) - 1) / (width - 1))] for i in range(width)]
    lo, hi = min(sampled), max(sampled)
    rng = hi - lo if hi > lo else 1.0
    bars = []
    for v in sampled:
        h = int(8 * (1 - (v - lo) / rng))
        bars.append("▁▂▃▄▅▆▇█"[h])
    print("loss: " + "".join(bars))


def main() -> None:
    X, Y, W_base = make_toy_data()
    print(f"任务：从 base 线性映射 → base + 低秩扰动；D={D_IN}→{D_OUT}，N={N_TRAIN}\n")

    # 1) 全参 baseline
    base_full = nn.Linear(D_IN, D_OUT, bias=False)
    with torch.no_grad():
        base_full.weight.copy_(W_base)
    for p in base_full.parameters():
        p.requires_grad = True
    print(f"全参微调：可训练参数 = {count_trainable(base_full):,}")
    losses_full = train_loop(base_full, X, Y)

    # 2) LoRA
    base_lora = nn.Linear(D_IN, D_OUT, bias=False)
    with torch.no_grad():
        base_lora.weight.copy_(W_base)
    lora_model = LoRALinear(base_lora, r=LORA_R, alpha=LORA_ALPHA)
    print(f"LoRA (r={LORA_R}, α={LORA_ALPHA})：可训练参数 = {count_trainable(lora_model):,}")
    losses_lora = train_loop(lora_model, X, Y)

    pct = 100 * count_trainable(lora_model) / count_trainable(base_full)
    print(f"\n参数量比：LoRA / full = {pct:.2f}%")

    ascii_curve(losses_full, "full FT")
    ascii_curve(losses_lora, f"LoRA r={LORA_R}")

    print("\n=== 关键收获 ===")
    print(f"1. r={LORA_R} 时只训原参数的 {pct:.2f}%，最终 loss 与全参接近")
    print("2. B 初始化为 0 → 起点等价于 base，训练过程是「贴补丁」")
    print("3. scaling = α/r 等价于 learning rate 的隐式调节，与 r 解耦")
    print("4. 真实 Transformer 里 LoRA 接在 q/v_proj，原理一模一样")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```bash
python ml_foundations/post_training/04_lora_from_scratch.py
```

Expected: prints "参数量比" around 3% (8·256·2 / 256·256), two ASCII loss curves both descending; takes < 30 seconds on Mac CPU.

- [ ] **Step 3: Commit**

```bash
git add ml_foundations/post_training/04_lora_from_scratch.py
git commit -m "wip(phase3): add 04_lora_from_scratch"
```

---

## Task 5: `03_sft_full.py` — TRL SFTTrainer (critical path)

**Files:**
- Create: `ml_foundations/post_training/03_sft_full.py` (~250 lines)

**Interfaces:**
- Consumes: `data/alpaca_mini.jsonl` from Task 1
- Produces: full-FT SFT adapter saved to `runs/03_sft_full/` (gitignored)

- [ ] **Step 1: Write the demo**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║  03_sft_full.py — TRL SFTTrainer 全参微调 Qwen2.5-0.5B             ║
║                                                                  ║
║  核心问题：base model 怎么变成「会听指令」？SFT 的代价多大？           ║
║  与生产对应：所有 instruction-tuned 模型的第一步                    ║
╚══════════════════════════════════════════════════════════════════╝
"""
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from trl import SFTConfig, SFTTrainer
except ImportError:
    print("❌ 需要 trl：pip install 'trl>=0.11.0'")
    sys.exit(1)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = Path(__file__).parent / "data" / "alpaca_mini.jsonl"
OUT_DIR = Path(__file__).parent / "runs" / "03_sft_full"

GEN_PROMPTS = [
    "用一句话解释什么是注意力机制。",
    "写一首关于秋天的两行小诗。",
    "推荐一道适合初学者的家常菜。",
]


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dataset() -> Dataset:
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


def generate_samples(model, tokenizer, device) -> list[str]:
    outs = []
    model.eval()
    for prompt in GEN_PROMPTS:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt", add_generation_prompt=True
        ).to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        outs.append(f"Q: {prompt}\nA: {text}")
    return outs


def main() -> None:
    set_seed(42)
    device = pick_device()
    print(f"✅ 设备：{device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    except Exception as e:
        print(f"❌ 加载模型失败：{e}")
        print("提示：检查网络，或 export HF_ENDPOINT=https://hf-mirror.com")
        sys.exit(1)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset()
    print(f"✅ 数据集：{len(ds)} 条 Alpaca 样本")

    print("\n=== 训练前生成（base model） ===")
    before = generate_samples(model, tokenizer, device)
    for x in before:
        print(x + "\n")

    config = SFTConfig(
        output_dir=str(OUT_DIR),
        max_steps=50,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        bf16=False,
        fp16=False,
        max_seq_length=512,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    print(f"\n=== 开始训练（50 steps，预计 MPS ~3 min / CPU ~10 min） ===")
    trainer.train()

    print("\n=== 训练后生成（SFT model） ===")
    after = generate_samples(model, tokenizer, device)
    for x in after:
        print(x + "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT_DIR))
    print(f"✅ 模型已保存到 {OUT_DIR}")

    if device.type in {"mps", "cuda"}:
        try:
            if device.type == "mps":
                peak = torch.mps.driver_allocated_memory() / 1024**3
            else:
                peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"内存峰值：{peak:.2f} GB")
        except Exception:
            pass

    print("\n=== 关键收获 ===")
    print("1. 全参 SFT 训完 50 step，Qwen2.5-0.5B 在 alpaca 风格 prompt 上回答更结构化")
    print("2. 0.5B 模型全参微调 MPS/CPU 都能跑；7B 起就必须 LoRA")
    print("3. SFTTrainer 自动套 chat template + mask prompt token")
    print("4. report_to=[] 关闭 wandb；save_strategy=no 不写中间 ckpt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```bash
python ml_foundations/post_training/03_sft_full.py
```

Expected: prints "训练前 / 训练后" 各 3 条生成对比，loss 从 ~2.x 降到 ~1.x；总耗时 MPS < 5min / CPU < 15min；`runs/03_sft_full/` 下出现 `model.safetensors` (~1GB)。

- [ ] **Step 3: Commit**

```bash
git add ml_foundations/post_training/03_sft_full.py
git commit -m "wip(phase3): add 03_sft_full"
```

---

---

## Task 6: `05_lora_peft.py` — PEFT LoRA on Qwen2.5-0.5B

**Files:**
- Create: `ml_foundations/post_training/05_lora_peft.py` (~220 lines)

**Interfaces:**
- Consumes: `data/alpaca_mini.jsonl` from Task 1
- Produces: LoRA adapter saved to `runs/05_lora_peft/` (gitignored)

- [ ] **Step 1: Write the demo**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║  05_lora_peft.py — PEFT 库的工程化 LoRA                            ║
║                                                                  ║
║  核心问题：target_modules 怎么选？adapter 文件多大？merge 怎么做？   ║
║  与生产对应：HuggingFace 微调 90% 用这套                            ║
╚══════════════════════════════════════════════════════════════════╝
"""
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from peft import LoraConfig, PeftModel, get_peft_model
    from trl import SFTConfig, SFTTrainer
except ImportError as e:
    print(f"❌ 缺包：{e}")
    print("提示：pip install 'peft>=0.13.0' 'trl>=0.11.0'")
    sys.exit(1)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = Path(__file__).parent / "data" / "alpaca_mini.jsonl"
OUT_DIR = Path(__file__).parent / "runs" / "05_lora_peft"
MERGED_DIR = Path(__file__).parent / "runs" / "05_lora_peft_merged"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dataset() -> Dataset:
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


def main() -> None:
    set_seed(42)
    device = pick_device()
    print(f"✅ 设备：{device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)

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

    ds = load_dataset()
    config = SFTConfig(
        output_dir=str(OUT_DIR),
        max_steps=50,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,  # LoRA 通常用比全参高 10× 的 lr
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        max_seq_length=512,
        packing=False,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=ds, processing_class=tokenizer)

    print(f"\n=== 开始 LoRA 训练（50 steps，MPS ~3 min / CPU ~10 min） ===")
    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    adapter_size = sum(
        f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()
    ) / 1024**2
    print(f"\n✅ adapter 已保存到 {OUT_DIR}（{adapter_size:.2f} MB）")

    # 重新加载 + merge
    print("\n=== 重新加载 adapter 并 merge_and_unload ===")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    reloaded = PeftModel.from_pretrained(base, str(OUT_DIR))
    merged = reloaded.merge_and_unload()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR))
    print(f"✅ merged model 已保存到 {MERGED_DIR}（与 base 同尺寸，可直接当 base model 用）")

    print("\n=== 关键收获 ===")
    print(f"1. r=8 + target=q,v_proj：adapter 仅 ~{adapter_size:.1f} MB，base model ~1GB")
    print("2. lr=2e-4 比全参 SFT 的 2e-5 高 10×，是 LoRA 的常见经验值")
    print("3. merge_and_unload 把 BA 加回 W₀，输出就是普通模型，可被任意框架加载")
    print("4. target_modules 加入 mlp（gate/up/down_proj）能进一步提效但翻倍参数")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```bash
python ml_foundations/post_training/05_lora_peft.py
```

Expected: `print_trainable_parameters` 显示 ~0.4M (~0.08%)，adapter 大小 ~6MB，merged model ~1GB；MPS < 5min / CPU < 15min。

- [ ] **Step 3: Commit**

```bash
git add ml_foundations/post_training/05_lora_peft.py
git commit -m "wip(phase3): add 05_lora_peft"
```

---

## Task 7: `06_qlora_mlx.py` — Apple MLX 4-bit QLoRA

**Files:**
- Create: `ml_foundations/post_training/06_qlora_mlx.py` (~250 lines)

**Interfaces:**
- Consumes: `data/alpaca_mini.jsonl` from Task 1
- Produces: MLX-format 4-bit base + LoRA adapter under `runs/06_qlora_mlx/` (gitignored)

- [ ] **Step 1: Write the demo**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║  06_qlora_mlx.py — Apple MLX 原生 4-bit QLoRA（Mac 主路径）         ║
║                                                                  ║
║  核心问题：为什么 Mac 上 QLoRA 该用 MLX 而非 bitsandbytes？          ║
║  与生产对应：unified memory + Metal kernel 的优势                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = Path(__file__).parent / "data" / "alpaca_mini.jsonl"
OUT_DIR = Path(__file__).parent / "runs" / "06_qlora_mlx"
MLX_BASE = OUT_DIR / "qwen_mlx_q4"
ADAPTER_DIR = OUT_DIR / "lora_adapter"
MLX_DATA_DIR = OUT_DIR / "mlx_data"


def check_env() -> None:
    if platform.system() != "Darwin":
        print("⚠️ 本 demo 仅在 macOS 上有意义（MLX 是 Apple Silicon 专属）")
        print("   非 Mac 用户请改跑 07_qlora_peft_bnb.py（Colab/CUDA）")
        sys.exit(0)
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("❌ 需要 mlx-lm：pip install 'mlx-lm>=0.20.0'")
        sys.exit(1)


def convert_to_mlx_q4() -> None:
    if MLX_BASE.exists():
        print(f"✅ MLX q4 模型已存在：{MLX_BASE}")
        return
    print(f"=== 步骤 1：量化 {MODEL_ID} → MLX 4-bit ===")
    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", MODEL_ID,
        "--mlx-path", str(MLX_BASE),
        "-q",  # 量化到 4-bit
    ]
    subprocess.run(cmd, check=True)


def prepare_mlx_data() -> None:
    """MLX 的 LoRA 训练接受 jsonl，每行 {"text": "..."}（已套好 chat template）。"""
    if (MLX_DATA_DIR / "train.jsonl").exists():
        print(f"✅ MLX 训练数据已存在：{MLX_DATA_DIR}")
        return
    MLX_DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            user = r["instruction"] + ("\n\n" + r["input"] if r["input"] else "")
            # Qwen 格式 chat template
            text = (
                f"<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n{r['output']}<|im_end|>"
            )
            rows.append({"text": text})
    split = int(0.9 * len(rows))
    with open(MLX_DATA_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows[:split]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(MLX_DATA_DIR / "valid.jsonl", "w", encoding="utf-8") as f:
        for r in rows[split:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ 已写入 train ({split}) / valid ({len(rows) - split}) 到 {MLX_DATA_DIR}")


def train_lora() -> None:
    print(f"\n=== 步骤 2：在 4-bit base 上训 LoRA adapter（200 iters，M1/M2 ~2 min） ===")
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", str(MLX_BASE),
        "--data", str(MLX_DATA_DIR),
        "--train",
        "--iters", "200",
        "--batch-size", "2",
        "--lora-layers", "8",
        "--adapter-path", str(ADAPTER_DIR),
    ]
    subprocess.run(cmd, check=True)


def memory_report() -> None:
    base_size = sum(f.stat().st_size for f in MLX_BASE.rglob("*") if f.is_file()) / 1024**2
    adapter_size = sum(f.stat().st_size for f in ADAPTER_DIR.rglob("*") if f.is_file()) / 1024**2
    print(f"\n=== 体积对比 ===")
    print(f"  Qwen2.5-0.5B fp16 (HF 缓存) ≈ 988 MB")
    print(f"  MLX 4-bit base             = {base_size:.1f} MB")
    print(f"  LoRA adapter               = {adapter_size:.2f} MB")


def main() -> None:
    check_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_to_mlx_q4()
    prepare_mlx_data()
    train_lora()
    memory_report()

    print("\n=== 关键收获 ===")
    print("1. MLX 量化用 group-wise 4-bit，与 NF4 思路类似但实现是 Metal kernel")
    print("2. Apple Silicon unified memory：CPU/GPU 共享内存，无需 to(device) 拷贝")
    print("3. LoRA adapter 体积 < base 1%，可直接 commit 到 Hub")
    print("4. 想用 HF 生态：mlx_lm.fuse 把 adapter merge 回 fp16 模型即可")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```bash
python ml_foundations/post_training/06_qlora_mlx.py
```

Expected on Mac: MLX 量化产物 ~280MB，adapter 几 MB，总耗时 < 5min。非 Mac 直接退出并提示 07。

- [ ] **Step 3: Commit**

```bash
git add ml_foundations/post_training/06_qlora_mlx.py
git commit -m "wip(phase3): add 06_qlora_mlx"
```

---

## Task 8: `07_qlora_peft_bnb.py` — PEFT + bitsandbytes 4-bit (CUDA/Colab)

**Files:**
- Create: `ml_foundations/post_training/07_qlora_peft_bnb.py` (~250 lines)

**Interfaces:**
- Consumes: `data/alpaca_mini.jsonl` from Task 1
- Produces: LoRA adapter under `runs/07_qlora_peft_bnb/` (gitignored, only when on CUDA)

- [ ] **Step 1: Write the demo**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║  07_qlora_peft_bnb.py — PEFT + bitsandbytes 4-bit QLoRA            ║
║                                                                  ║
║  核心问题：HuggingFace 生态的 QLoRA 标准写法                        ║
║  与生产对应：单卡 GPU 训 7B-70B 模型的事实标准                       ║
║  ⚠️ Mac MPS 不支持 bitsandbytes；本 demo 仅在 CUDA / Colab 上跑     ║
╚══════════════════════════════════════════════════════════════════╝
"""
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, set_seed

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = Path(__file__).parent / "data" / "alpaca_mini.jsonl"
OUT_DIR = Path(__file__).parent / "runs" / "07_qlora_peft_bnb"


def check_env() -> None:
    if not torch.cuda.is_available():
        print("❌ 本 demo 需要 CUDA 环境（bitsandbytes 4-bit 在 Mac MPS / CPU 上不工作）")
        print("\n建议替代路径：")
        print("  • Mac 用户 → 跑 06_qlora_mlx.py（Apple MLX 原生）")
        print("  • 想体验 bnb → 上传到 Google Colab（Runtime → T4 GPU）跑本脚本")
        print("\n本脚本接受的 CLI 标志：--cpu-bnb-mock 会在 CPU 上模拟流程（仅打印 API 调用，不真训）")
        if "--cpu-bnb-mock" not in sys.argv:
            sys.exit(0)
    try:
        import bitsandbytes  # noqa: F401
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: F401
        from transformers import BitsAndBytesConfig  # noqa: F401
        from trl import SFTConfig, SFTTrainer  # noqa: F401
    except ImportError as e:
        print(f"❌ 缺包：{e}")
        print("提示：pip install bitsandbytes peft trl")
        sys.exit(1)


def load_dataset() -> Dataset:
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


def main() -> None:
    set_seed(42)
    check_env()
    is_cuda = torch.cuda.is_available()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # double quantization
    )
    print("=== BitsAndBytesConfig ===")
    print(f"  量化类型: NF4（4-bit Normal Float）")
    print(f"  compute dtype: bfloat16（前向反向用 bf16，权重存 4-bit）")
    print(f"  double quant: True（量化常数本身再量化一次，省 ~0.4 bit/param）")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not is_cuda:
        print("\n⚠️ CPU mock 模式：仅展示 API 调用流程，不真训")
        return

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset()
    config = SFTConfig(
        output_dir=str(OUT_DIR),
        max_steps=50, per_device_train_batch_size=4,
        gradient_accumulation_steps=2, learning_rate=2e-4,
        logging_steps=5, save_strategy="no", report_to=[],
        bf16=True, max_seq_length=512, packing=False,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=ds, processing_class=tokenizer)
    print(f"\n=== 开始 QLoRA 训练（50 steps，Colab T4 ~2 min） ===")
    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    print(f"\n✅ 4-bit base + LoRA adapter 已保存到 {OUT_DIR}")

    print("\n=== 关键收获 ===")
    print("1. base model 4-bit 加载，显存比 fp16 省 4×（Qwen2.5-0.5B 从 ~1GB 降到 ~280MB）")
    print("2. compute dtype bf16：4-bit 只存权重，矩阵乘还是高精度，不掉点")
    print("3. prepare_model_for_kbit_training：冻结 4-bit 权重 + cast LayerNorm 到 fp32")
    print("4. target_modules 加入 k/o_proj 比纯 q/v 略好（QLoRA 论文 ablation）")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

On Mac (expected to exit gracefully):
```bash
python ml_foundations/post_training/07_qlora_peft_bnb.py
```
Expected: 提示 "需要 CUDA 环境" 并退出，引导用户去 06 或 Colab。

On Colab T4 / CUDA host:
```bash
python ml_foundations/post_training/07_qlora_peft_bnb.py
```
Expected: 真训 50 steps，adapter 保存到 `runs/07_qlora_peft_bnb/`，总耗时 ~2 min。

- [ ] **Step 3: Commit**

```bash
git add ml_foundations/post_training/07_qlora_peft_bnb.py
git commit -m "wip(phase3): add 07_qlora_peft_bnb"
```

---

## Task 9: `08_dpo_alignment.py` — TRL DPOTrainer (critical path)

**Files:**
- Create: `ml_foundations/post_training/08_dpo_alignment.py` (~250 lines)

**Interfaces:**
- Consumes: `data/dpo_pairs_mini.jsonl` from Task 1; reuses base `Qwen2.5-0.5B-Instruct` (already instruction-tuned, suitable as DPO starting point)
- Produces: DPO-tuned adapter under `runs/08_dpo/` (gitignored)

- [ ] **Step 1: Write the demo**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║  08_dpo_alignment.py — TRL DPOTrainer 偏好对齐                      ║
║                                                                  ║
║  核心问题：为什么 DPO 一招打 PPO 三步？                              ║
║  与生产对应：Llama-3 / Qwen-2 等模型的对齐阶段都跑过类似流程         ║
╚══════════════════════════════════════════════════════════════════╝
"""
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from peft import LoraConfig, get_peft_model
    from trl import DPOConfig, DPOTrainer
except ImportError as e:
    print(f"❌ 缺包：{e}")
    print("提示：pip install 'trl>=0.11.0' 'peft>=0.13.0'")
    sys.exit(1)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = Path(__file__).parent / "data" / "dpo_pairs_mini.jsonl"
OUT_DIR = Path(__file__).parent / "runs" / "08_dpo"

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
    return Dataset.from_list(rows)


def generate(model, tokenizer, device, prompts: list[str]) -> list[str]:
    outs = []
    model.eval()
    for p in prompts:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], return_tensors="pt", add_generation_prompt=True
        ).to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=120, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        outs.append(f"Q: {p}\nA: {text}")
    return outs


def main() -> None:
    set_seed(42)
    device = pick_device()
    print(f"✅ 设备：{device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)

    print("\n=== DPO 前生成（base instruct model） ===")
    before = generate(model, tokenizer, device, EVAL_PROMPTS)
    for x in before:
        print(x + "\n")

    # 用 LoRA 节省 reference model 副本的代价
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_pairs()
    config = DPOConfig(
        output_dir=str(OUT_DIR),
        max_steps=50,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        beta=0.1,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        max_length=768,
        max_prompt_length=384,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT 模型下 ref = disable adapter
        args=config,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    print(f"\n=== 开始 DPO 训练（50 steps，MPS ~5 min / CPU ~15 min） ===")
    trainer.train()

    print("\n=== DPO 后生成 ===")
    after = generate(model, tokenizer, device, EVAL_PROMPTS)
    for x in after:
        print(x + "\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    print(f"✅ DPO adapter 已保存到 {OUT_DIR}")

    print("\n=== 关键收获 ===")
    print("1. DPO 损失：-log σ(β·(log π_θ(c)/π_ref(c) - log π_θ(r)/π_ref(r)))")
    print("2. β 控制对齐强度：β 小 → 接近 SFT，β 大 → 强烈偏好 chosen")
    print("3. ref_model=None 时 PEFT 自动 disable adapter 当 reference（省 1× 显存）")
    print("4. 50 step 在 100 对数据上只能看到风格倾向的微调，工业要 50k+ 对")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```bash
python ml_foundations/post_training/08_dpo_alignment.py
```

Expected: 前后 2 条生成对比，loss 包含 `rewards/chosen` 和 `rewards/rejected` 两列；MPS < 8min / CPU < 20min。

- [ ] **Step 3: Commit**

```bash
git add ml_foundations/post_training/08_dpo_alignment.py
git commit -m "wip(phase3): add 08_dpo_alignment"
```

---

## Task 10: `09_ppo_intro.py` — RLHF/PPO walkthrough (no real train)

**Files:**
- Create: `ml_foundations/post_training/09_ppo_intro.py` (~180 lines)

**Interfaces:**
- Consumes: nothing (teaching demo)
- Produces: ASCII flowchart + pseudo-code + tensor-shape inspection

- [ ] **Step 1: Write the demo**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║  09_ppo_intro.py — RLHF/PPO 全景与工程债务（不真训）                ║
║                                                                  ║
║  核心问题：为什么 InstructGPT 用 PPO，而 Llama-3 改用 DPO？          ║
║  与生产对应：理解 RLHF 工程负担，知道什么时候选 DPO/ORPO/KTO         ║
╚══════════════════════════════════════════════════════════════════╝
"""
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

FLOWCHART = r"""
                    ┌──────────────────────┐
                    │  Pretrained base LLM │
                    └──────────┬───────────┘
                               │ SFT (Task 5)
                               ▼
                    ┌──────────────────────┐
                    │   SFT model π_SFT    │  ◄── DPO 在这里直接闭环（Task 9）
                    └──────────┬───────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │ Reward Model │   │  Reference   │   │   Critic     │
    │  RM(prompt,  │   │  π_ref =     │   │  V_φ(s)      │
    │  response)→r │   │  copy(π_SFT) │   │ value head   │
    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
           │                  │                  │
           └────────┬─────────┴──────────────────┘
                    ▼
            ┌───────────────────────────┐
            │   PPO actor π_θ (clip)    │
            │   loss = -E[min(           │
            │     ratio·A, clip·A)]      │
            │     + β·KL(π_θ‖π_ref)      │
            └───────────────────────────┘
"""


def show_flowchart() -> None:
    print(FLOWCHART)


def show_pseudocode() -> None:
    code = r"""
# PPO 单 step 伪代码（参考 trl.PPOTrainer 实现）
for batch in dataloader:
    # 1. rollout：用当前 actor 采样 response
    response = actor.generate(batch["prompt"])

    # 2. 三个模型同时前向
    logits_actor = actor(prompt + response).logits
    logits_ref   = ref_model(prompt + response).logits
    values       = critic(prompt + response)
    rewards_raw  = reward_model(prompt, response)

    # 3. 计算 KL 惩罚后的 reward
    log_ratio = log_softmax(logits_actor) - log_softmax(logits_ref)
    rewards = rewards_raw - β_kl * log_ratio

    # 4. GAE 计算 advantage
    advantages = compute_gae(rewards, values)

    # 5. PPO clip loss
    new_log_probs = log_softmax(actor(...).logits)
    ratio = exp(new_log_probs - log_ratio.detach())
    surr1 = ratio * advantages
    surr2 = clip(ratio, 1-ε, 1+ε) * advantages
    loss_actor = -min(surr1, surr2).mean()
    loss_critic = ((values - returns) ** 2).mean()
    (loss_actor + 0.5*loss_critic).backward()
"""
    print(code)


def inspect_tensor_shapes() -> None:
    """加载 base + 一个 mock RM，print 一次 forward 的张量形状，体感「4 模型副本」。"""
    set_seed(42)
    print("\n=== 张量形状演示（仅前向一次，不更新参数） ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        actor = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    except Exception as e:
        print(f"⚠️ 模型加载失败：{e}（不影响理解，跳过这一节）")
        return

    # 同样的模型加载 4 份 → 直接 print 占用
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"  Qwen2.5-0.5B 参数量：{n_params:,} ≈ {n_params / 1e6:.0f}M")
    print(f"  PPO 同时需要：actor + reference + reward + critic ≈ 4×{n_params / 1e6:.0f}M")
    print(f"  显存账：fp16 下 ~{4 * n_params * 2 / 1024**3:.2f} GB（仅参数，未含 optimizer state）")

    prompt = "用一句话解释什么是 attention。"
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True
    )
    print(f"\n  prompt input_ids: {tuple(ids.shape)}")
    with torch.no_grad():
        gen = actor.generate(ids, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print(f"  response output_ids: {tuple(gen.shape)}")
    with torch.no_grad():
        out = actor(gen)
    print(f"  logits 形状: {tuple(out.logits.shape)}")
    print(f"  → 每步 PPO 需要这个 logits 算 4 次（actor/ref，加上 RM/critic 的 forward）")


def compare_table() -> None:
    print("\n=== PPO vs DPO 工程对比 ===")
    rows = [
        ("需要 reward model", "是（要单独训练）", "否（直接用 chosen/rejected pair）"),
        ("模型副本", "4（actor/ref/RM/critic）", "2（actor/ref）"),
        ("训练稳定性", "差（reward hacking / KL 爆炸）", "好（监督学习风格）"),
        ("超参数量", "多（β_kl, clip ε, GAE λ, value coef）", "少（仅 β）"),
        ("实现复杂度", "高（rollout / GAE / clip）", "低（单 loss）"),
        ("数据形态", "(prompt, response) + 标量 reward", "(prompt, chosen, rejected)"),
    ]
    print(f"  {'维度':<16} | {'PPO':<32} | DPO")
    print("  " + "─" * 78)
    for r in rows:
        print(f"  {r[0]:<16} | {r[1]:<32} | {r[2]}")


def main() -> None:
    print("# RLHF / PPO 全景（教学型 demo，不真训）\n")
    show_flowchart()
    show_pseudocode()
    inspect_tensor_shapes()
    compare_table()

    print("\n=== 关键收获 ===")
    print("1. PPO 同时要 4 个模型副本，0.5B × 4 在 Mac 16GB 上已经吃力，7B 必须多卡")
    print("2. RM 训练本身需要数十万人工偏好标注，是 OpenAI/Anthropic 的隐性壁垒")
    print("3. DPO 用 chosen/rejected pair 反推「隐式 RM」，省掉 RM 训练 + critic + GAE")
    print("4. 现代变体（ORPO/SimPO/KTO）进一步去 reference model，但思想沿用 DPO")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```bash
python ml_foundations/post_training/09_ppo_intro.py
```

Expected: 完整 ASCII 流程图 + 伪代码 + 张量形状 + 对比表 + 4 条关键收获；总耗时 < 1min（仅加载 base 跑一次 forward）。

- [ ] **Step 3: Commit**

```bash
git add ml_foundations/post_training/09_ppo_intro.py
git commit -m "wip(phase3): add 09_ppo_intro"
```

---

## Task 11: `10_quantization_inference.py` — GGUF + bnb inference comparison

**Files:**
- Create: `ml_foundations/post_training/10_quantization_inference.py` (~280 lines)

**Interfaces:**
- Consumes: nothing (downloads pre-quantized GGUF from HF on demand)
- Produces: 三方案对比表（内存 / 延迟 / 输出质量）

- [ ] **Step 1: Write the demo**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║  10_quantization_inference.py — 训练后量化推理对比                  ║
║                                                                  ║
║  核心问题：GGUF Q4_K_M / bnb 8-bit / fp16，部署时怎么选？           ║
║  与生产对应：llama.cpp / Ollama / LM Studio 的量化格式由来           ║
╚══════════════════════════════════════════════════════════════════╝
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GGUF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
GGUF_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
OUT_DIR = Path(__file__).parent / "runs" / "10_quant"
TEST_PROMPT = "请用一句话解释什么是量化。"


def measure_fp16() -> dict:
    print("\n── 方案 A：fp16 全精度 ──")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": TEST_PROMPT}],
        return_tensors="pt", add_generation_prompt=True
    ).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=64, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    dt = time.perf_counter() - t0
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    del model
    return {"name": "fp16", "memory_mb": mem_mb, "latency_s": dt, "output": text, "params": n}


def have_llama_quantize() -> bool:
    return shutil.which("llama-quantize") is not None


def maybe_quantize_with_llama_cpp() -> None:
    if not have_llama_quantize():
        print("\n⚠️ llama.cpp 未安装，跳过本地量化步骤。")
        print("   安装：brew install llama.cpp（macOS）或 https://github.com/ggerganov/llama.cpp")
        print("   命令示例（安装后可手动尝试）：")
        print(f"     llama-quantize ./qwen.gguf ./qwen-q4_k_m.gguf Q4_K_M")
        return
    print("\n✅ 检测到 llama-quantize，可执行本地量化（本 demo 略过实际跑，因耗时较长）")


def measure_gguf_q4() -> dict | None:
    print("\n── 方案 B：GGUF Q4_K_M（llama-cpp-python 加载预量化模型） ──")
    try:
        from llama_cpp import Llama
    except ImportError:
        print("⚠️ 未装 llama-cpp-python，跳过此方案。")
        print("   安装：pip install llama-cpp-python")
        return None

    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE,
                               local_dir=str(OUT_DIR))
    except Exception as e:
        print(f"⚠️ 下载 GGUF 失败：{e}")
        return None

    mem_mb = Path(path).stat().st_size / 1024**2
    llm = Llama(model_path=path, n_ctx=512, verbose=False)
    t0 = time.perf_counter()
    out = llm(f"<|im_start|>user\n{TEST_PROMPT}<|im_end|>\n<|im_start|>assistant\n",
              max_tokens=64, temperature=0.0, stop=["<|im_end|>"])
    dt = time.perf_counter() - t0
    text = out["choices"][0]["text"].strip()
    return {"name": "GGUF Q4_K_M", "memory_mb": mem_mb, "latency_s": dt, "output": text}


def measure_bnb_8bit() -> dict | None:
    print("\n── 方案 C：bitsandbytes 8-bit ──")
    if not torch.cuda.is_available():
        print("⚠️ bitsandbytes 8-bit 需要 CUDA，Mac/CPU 跳过。")
        print("   降级提示：本路径在 Colab T4 上可直接跑。")
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        print("⚠️ 未装 bitsandbytes，跳过。")
        return None
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                 device_map="auto").eval()
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": TEST_PROMPT}],
        return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=64, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    dt = time.perf_counter() - t0
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return {"name": "bnb 8-bit", "memory_mb": mem_mb, "latency_s": dt, "output": text}


def print_table(results: list[dict]) -> None:
    print("\n=== 三方案对比 ===")
    print(f"  {'方案':<18} | {'权重体积 MB':>14} | {'首 token+64 延迟 s':>20} | 输出前 30 字")
    print("  " + "─" * 92)
    for r in results:
        if r is None:
            continue
        out = (r['output'][:30] + '…') if len(r['output']) > 30 else r['output']
        print(f"  {r['name']:<18} | {r['memory_mb']:>14.1f} | {r['latency_s']:>20.2f} | {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    maybe_quantize_with_llama_cpp()

    results = [measure_fp16(), measure_gguf_q4(), measure_bnb_8bit()]
    print_table(results)

    print("\n=== 关键收获 ===")
    print("1. GGUF Q4_K_M = 约 4.5 bit/param 的非均匀量化（K_M 是 k-quants medium）")
    print("2. bnb NF4 / GPTQ / AWQ / GGUF：训练时 vs 推理时；CUDA only vs 全平台")
    print("3. 量化主要省的是「显存 + 加载时间」，延迟通常并不会更快（除非 GPU 带宽瓶颈）")
    print("4. 部署侧主流：Mac → GGUF + llama.cpp，Linux GPU → AWQ/GPTQ + vLLM")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```bash
python ml_foundations/post_training/10_quantization_inference.py
```

Expected: 三行对比表（Mac 上 fp16 + GGUF 两行有数据，bnb 行打印降级提示）；总耗时 < 5min（含 GGUF 下载 ~350MB）。

- [ ] **Step 3: Commit**

```bash
git add ml_foundations/post_training/10_quantization_inference.py
git commit -m "wip(phase3): add 10_quantization_inference"
```

---

## Task 12: `11_eval_perplexity.py` — PPL eval base/SFT/DPO

**Files:**
- Create: `ml_foundations/post_training/11_eval_perplexity.py` (~180 lines)

**Interfaces:**
- Consumes: `runs/03_sft_full/` (Task 5) and `runs/08_dpo/` (Task 9) if available; falls back to evaluating only base when adapters not found
- Produces: three-model PPL table + side-by-side generation

- [ ] **Step 1: Write the demo**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║  11_eval_perplexity.py — 困惑度评估 base / SFT / DPO                ║
║                                                                  ║
║  核心问题：PPL 真的能衡量「模型变好了」吗？                          ║
║  与生产对应：研究里最常用但最容易误导的单指标                        ║
╚══════════════════════════════════════════════════════════════════╝
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from datasets import load_dataset
    from peft import PeftModel
except ImportError as e:
    print(f"❌ 缺包：{e}")
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
        print(f"⚠️ 下载 WikiText 失败：{e}，使用本地小样本")
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
        ).to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        outs.append(text)
    return outs


def load_variant(name: str, adapter_dir: Path | None, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter_dir is None:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    elif not adapter_dir.exists():
        print(f"⚠️ {name} adapter 不存在（{adapter_dir}），用 base 代替；建议先跑对应 demo")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
    else:
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device)
        # 03_sft_full 保存的是全参，不是 PEFT；尝试两种加载方式
        adapter_cfg = adapter_dir / "adapter_config.json"
        if adapter_cfg.exists():
            model = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(str(adapter_dir),
                                                          torch_dtype=torch.float32).to(device)
    return tokenizer, model


def main() -> None:
    set_seed(42)
    device = pick_device()
    print(f"✅ 设备：{device}")

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
```

- [ ] **Step 2: Run and verify**

```bash
python ml_foundations/post_training/11_eval_perplexity.py
```

Expected: 3 行 PPL 表 + 2 个 prompt × 3 变体的生成对比；总耗时 < 5min。

- [ ] **Step 3: Commit**

```bash
git add ml_foundations/post_training/11_eval_perplexity.py
git commit -m "wip(phase3): add 11_eval_perplexity"
```

---

## Task 13: `12_eval_lm_harness.py` — lm-evaluation-harness subset

**Files:**
- Create: `ml_foundations/post_training/12_eval_lm_harness.py` (~200 lines)

**Interfaces:**
- Consumes: `runs/03_sft_full/` if available
- Produces: arc_easy 小子集分数（base vs SFT）

- [ ] **Step 1: Write the demo**

```python
"""
╔══════════════════════════════════════════════════════════════════╗
║  12_eval_lm_harness.py — lm-evaluation-harness 子集评估             ║
║                                                                  ║
║  核心问题：业界 benchmark（MMLU / ARC 等）实际怎么跑？               ║
║  与生产对应：HF Open LLM Leaderboard / EleutherAI 标配               ║
╚══════════════════════════════════════════════════════════════════╝
"""
import json
import subprocess
import sys
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_DIR = Path(__file__).parent / "runs" / "03_sft_full"
OUT_DIR = Path(__file__).parent / "runs" / "12_lm_eval"

# arc_easy 全集 ~2376 题；20 题足以演示流程
EVAL_TASK = "arc_easy"
LIMIT = 20


def check_env() -> None:
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        print("❌ 需要 lm-eval：pip install 'lm-eval>=0.4.0'")
        sys.exit(1)


def run_eval(model_path: str, label: str) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{label}.json"
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=float32",
        "--tasks", EVAL_TASK,
        "--limit", str(LIMIT),
        "--output_path", str(out_file),
        "--batch_size", "4",
    ]
    print(f"\n=== 跑 {label}（{EVAL_TASK}, limit={LIMIT}） ===")
    print(f"命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ lm-eval 执行失败：\n{result.stderr[-500:]}")
        return {}
    # 找产物 json（lm-eval 会把结果嵌套在子目录里）
    for f in OUT_DIR.rglob("*.json"):
        try:
            data = json.loads(f.read_text())
            if "results" in data:
                return data["results"].get(EVAL_TASK, {})
        except Exception:
            continue
    return {}


def main() -> None:
    check_env()

    print(f"评估任务：{EVAL_TASK}（仅 {LIMIT} 题，演示流程；全集需数小时）\n")

    base_result = run_eval(MODEL_ID, "base")
    sft_result = run_eval(str(SFT_DIR), "sft") if SFT_DIR.exists() else {}

    print("\n=== 子集分数对比 ===")
    print(f"  {'变体':<10} | {'acc':>8} | {'acc_norm':>10}")
    print("  " + "─" * 35)
    print(f"  {'base':<10} | {base_result.get('acc,none', 'N/A'):>8} | "
          f"{base_result.get('acc_norm,none', 'N/A'):>10}")
    if sft_result:
        print(f"  {'SFT':<10} | {sft_result.get('acc,none', 'N/A'):>8} | "
              f"{sft_result.get('acc_norm,none', 'N/A'):>10}")
    else:
        print(f"  {'SFT':<10} | (跳过：未找到 03_sft_full 的产物)")

    print("\n=== 关键收获 ===")
    print(f"1. lm-eval 是 EleutherAI 维护的统一 benchmark 框架，HF leaderboard 用它")
    print(f"2. 20 题子集仅演示流程；正式跑 arc/mmlu 全集需要 GPU + 数小时")
    print(f"3. arc_easy 是 multiple-choice，模型输出概率最高的选项；acc 是准确率")
    print(f"4. 实际选型时看 6 个标准任务：ARC / HellaSwag / MMLU / TruthfulQA / Winogrande / GSM8K")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```bash
python ml_foundations/post_training/12_eval_lm_harness.py
```

Expected: 一张 2 行对比表（如果 03 跑过），总耗时 MPS < 5min。

- [ ] **Step 3: Commit**

```bash
git add ml_foundations/post_training/12_eval_lm_harness.py
git commit -m "wip(phase3): add 12_eval_lm_harness"
```

---

继续 Task 14-26 见下。

---

## Task 14: `KNOWLEDGE.md` — full theory textbook

**Files:**
- Create: `ml_foundations/post_training/KNOWLEDGE.md` (~8000-10000 字)

**Interfaces:**
- Consumes: nothing
- Produces: source-of-truth theory document; docs/ md 章节都从这里精简而来

- [ ] **Step 1: Write the full KNOWLEDGE.md following the §5 outline of the spec**

Use the spec's `§5 KNOWLEDGE.md 大纲` as the chapter skeleton. For each chapter write 1000-1500 字, following these rules:

- 每节先抛"为什么"问题，再讲"是什么 + 怎么做"
- 关键数学（LoRA `W = W₀ + (α/r)BA`、DPO `-log σ(β·(log π_θ(c)/π_ref(c) - log π_θ(r)/π_ref(r)))`、NF4 量化区间）给完整公式 + 直观推导
- 每节给 1-2 个 ASCII 图（LoRA 矩阵分解 / PPO 4 模型协作图 / NF4 量化区间）
- 每节末尾「与生产对应」: 这一节在 transformers / TRL / PEFT 哪个 API 对应
- 关键名词带 emoji 锚点便于站点搜索（如「📌 NF4」「📌 DPO loss」）
- 引用 Phase 2 章节时给 markdown 锚点链接

Chapter 0 (全景) MUST include the four-stage funnel ASCII diagram (Pretrain → SFT → RLHF → 量化部署) with parameter/data/compute orders of magnitude per stage.

Chapter 4 (DPO) MUST include the full derivation from RL form to MLE form (no skipping steps) — this is the chapter that justifies why DPO replaces PPO.

- [ ] **Step 2: Sanity-check character count and structure**

```bash
wc -m ml_foundations/post_training/KNOWLEDGE.md
# Expect: 8000+ 字
grep -c "^## " ml_foundations/post_training/KNOWLEDGE.md
# Expect: ≥ 8 (8 chapters + 2 appendices, some may use ### sub-headers)
```

- [ ] **Step 3: Commit**

```bash
git add ml_foundations/post_training/KNOWLEDGE.md
git commit -m "wip(phase3): add KNOWLEDGE.md"
```

---

## Task 15: `docs/ml-foundations/post-training/index.md`

**Files:**
- Create: `docs/ml-foundations/post-training/index.md` (~600 字)

**Interfaces:**
- Consumes: nothing
- Produces: 章节首页 + 学习地图 + 跳到下一节链接

- [ ] **Step 1: Write following the same pattern as `docs/ml-foundations/transformer-training/index.md`**

Structure:
- Frontmatter (title)
- 一段简介（这一章在整个学习路径中的位置）
- 学习地图表格：12 个 demo + 对应的 7 个理论章节
- 「推荐学习顺序」段落
- 「硬件预算」段落（每个 demo MPS / CPU / Colab 时间表，从 spec §7.2 复制）
- 「下一节」链接 → `./overview.md`

参考已存在的 `docs/ml-foundations/transformer-training/index.md` 的具体写法（在实施期 Read 一下作为模板）。

- [ ] **Step 2: Commit**

```bash
git add docs/ml-foundations/post-training/index.md
git commit -m "wip(phase3): add docs index"
```

---

## Task 16: `docs/ml-foundations/post-training/overview.md`

**Files:**
- Create: `docs/ml-foundations/post-training/overview.md` (~1500 字)

**Interfaces:**
- Consumes: KNOWLEDGE.md chapter 0
- Produces: 网站版的全景章

- [ ] **Step 1: 复用 KNOWLEDGE.md 第 0 章内容，加 frontmatter + 上下页链接**

Add at top:
```markdown
---
title: 全景：从 base model 到 ChatGPT
---

# 0. 全景：从 base model 到 ChatGPT 之间发生了什么

> 上一章：[Transformer 训练实战 · KV cache](../transformer-training/inference.md)
> 下一章：[SFT：让模型学会"听指令"](./sft.md)
```

Then transplant KNOWLEDGE Chapter 0 content. End with:
```markdown
---

[← 上一节](../transformer-training/inference.md) | [下一节：SFT →](./sft.md)
```

- [ ] **Step 2: Commit**

```bash
git add docs/ml-foundations/post-training/overview.md
git commit -m "wip(phase3): add docs overview"
```

---

## Task 17: `docs/ml-foundations/post-training/sft.md`

**Files:**
- Create: `docs/ml-foundations/post-training/sft.md` (~1500-2000 字)

**Interfaces:**
- Consumes: KNOWLEDGE.md chapter 1
- Produces: SFT 章节网站版

- [ ] **Step 1: Transplant KNOWLEDGE Ch.1 + 上下页链接**

```markdown
---
title: SFT：让模型学会"听指令"
---

# 1. SFT：让模型学会"听指令"

> 上一章：[全景](./overview.md)
> 下一章：[PEFT 与 LoRA 数学](./lora.md)
> 对应代码：`ml_foundations/post_training/01_data_construction.py`、`02_multi_turn_chat.py`、`03_sft_full.py`
```

Sections 1.1–1.5 from KNOWLEDGE, ending with:
```markdown
[← 全景](./overview.md) | [下一节：LoRA →](./lora.md)
```

- [ ] **Step 2: Commit**

```bash
git add docs/ml-foundations/post-training/sft.md
git commit -m "wip(phase3): add docs sft"
```

---

## Task 18: `docs/ml-foundations/post-training/lora.md`

**Files:**
- Create: `docs/ml-foundations/post-training/lora.md` (~1500-2000 字)

- [ ] **Step 1: Transplant KNOWLEDGE Ch.2 + 上下页链接**

Frontmatter + 链接结构同 Task 17，对应代码 `04_lora_from_scratch.py` 和 `05_lora_peft.py`，下一章是 `./qlora.md`。

- [ ] **Step 2: Commit**

```bash
git add docs/ml-foundations/post-training/lora.md
git commit -m "wip(phase3): add docs lora"
```

---

## Task 19: `docs/ml-foundations/post-training/qlora.md`

**Files:**
- Create: `docs/ml-foundations/post-training/qlora.md` (~1500-2000 字)

- [ ] **Step 1: Transplant KNOWLEDGE Ch.3 + 上下页链接**

Frontmatter + 链接结构同 Task 17，对应代码 `06_qlora_mlx.py` 和 `07_qlora_peft_bnb.py`，下一章是 `./dpo.md`。本章必须包含「MLX vs bnb 路径选择决策树」ASCII 图。

- [ ] **Step 2: Commit**

```bash
git add docs/ml-foundations/post-training/qlora.md
git commit -m "wip(phase3): add docs qlora"
```

---

## Task 20: `docs/ml-foundations/post-training/dpo.md`

**Files:**
- Create: `docs/ml-foundations/post-training/dpo.md` (~1500-2000 字)

- [ ] **Step 1: Transplant KNOWLEDGE Ch.4 + 上下页链接**

对应代码 `08_dpo_alignment.py` 和 `09_ppo_intro.py`，下一章 `./quantization.md`。本章必须包含完整 DPO 推导（从 RL 形式 → MLE 形式），不跳步骤。

- [ ] **Step 2: Commit**

```bash
git add docs/ml-foundations/post-training/dpo.md
git commit -m "wip(phase3): add docs dpo"
```

---

## Task 21: `docs/ml-foundations/post-training/quantization.md`

**Files:**
- Create: `docs/ml-foundations/post-training/quantization.md` (~1500-2000 字)

- [ ] **Step 1: Transplant KNOWLEDGE Ch.5 + 上下页链接**

对应代码 `10_quantization_inference.py`，下一章 `./evaluation.md`。必须包含「GGUF Q4_K_M vs Q5_K_M vs Q8_0」对比表 + 「训练后 vs 训练中量化」时序图。

- [ ] **Step 2: Commit**

```bash
git add docs/ml-foundations/post-training/quantization.md
git commit -m "wip(phase3): add docs quantization"
```

---

## Task 22: `docs/ml-foundations/post-training/evaluation.md`

**Files:**
- Create: `docs/ml-foundations/post-training/evaluation.md` (~1500-2000 字)

- [ ] **Step 1: Transplant KNOWLEDGE Ch.6 + 上下页链接**

对应代码 `11_eval_perplexity.py` 和 `12_eval_lm_harness.py`，下一章 `./selection.md`。在开篇加 callout：「与 `evaluation/KNOWLEDGE.md` 区别：本章评估『训练好坏』，那一章评估『RAG 答案质量』，互不重叠。」

- [ ] **Step 2: Commit**

```bash
git add docs/ml-foundations/post-training/evaluation.md
git commit -m "wip(phase3): add docs evaluation"
```

---

## Task 23: `docs/ml-foundations/post-training/selection.md`

**Files:**
- Create: `docs/ml-foundations/post-training/selection.md` (~1500-2000 字)

- [ ] **Step 1: Transplant KNOWLEDGE Ch.7 + 上下页链接**

最后一章无下一节，结尾 `[← 评估](./evaluation.md) | [回到学习地图](./index.md)`。必须包含两张决策表：(1) 数据量 vs 方法（200 / 2k / 20k / 200k 各选什么）；(2) 硬件 vs 方法（Mac / 单卡 / 多卡 / 集群）。

- [ ] **Step 2: Commit**

```bash
git add docs/ml-foundations/post-training/selection.md
git commit -m "wip(phase3): add docs selection"
```

---

## Task 24: Integration — sidebar, README, roadmap, gitignore, requirements

**Files:**
- Modify: `docs/.vitepress/config.*` (sidebar)
- Modify: `README.md`
- Modify: `LEARNING_ROADMAP.md`
- Modify: `requirements.txt`
- `.gitignore` already updated in Task 1

**Interfaces:**
- Consumes: docs structure from Tasks 15-23
- Produces: 完整可导航的网站 + 仓库根 README 闭环

- [ ] **Step 1: Locate sidebar config**

```bash
find docs/.vitepress -maxdepth 2 -name "config.*"
```

Read it. Find the `ml-foundations` sidebar block (currently has `零、ML 基础` and `零.5、Transformer 训练实战`).

- [ ] **Step 2: Insert 「零.6、训练后期与对齐」 block after `零.5`**

Pattern (TypeScript or mts):
```typescript
{
  text: '零.6、训练后期与对齐',
  items: [
    { text: '0. 全景', link: '/ml-foundations/post-training/' },
    { text: '0. 全景（独立页）', link: '/ml-foundations/post-training/overview' },
    { text: '1. SFT', link: '/ml-foundations/post-training/sft' },
    { text: '2. LoRA', link: '/ml-foundations/post-training/lora' },
    { text: '3. QLoRA', link: '/ml-foundations/post-training/qlora' },
    { text: '4. DPO', link: '/ml-foundations/post-training/dpo' },
    { text: '5. 量化', link: '/ml-foundations/post-training/quantization' },
    { text: '6. 评估', link: '/ml-foundations/post-training/evaluation' },
    { text: '7. 选型决策', link: '/ml-foundations/post-training/selection' },
  ],
},
```

Adjust syntax to match the actual file (JSON vs TS vs JS). Match the existing `transformer-training` block's style.

- [ ] **Step 3: Build docs and verify**

```bash
cd docs && npm run docs:dev
```

Visit `http://localhost:5173/`, navigate to ML Foundations sidebar, confirm 「零.6、训练后期与对齐」 appears with 9 entries, each link resolves to a page. Stop the dev server (`Ctrl+C`) when verified.

- [ ] **Step 4: Update README.md**

After the existing `### 0.5 Transformer Training from Scratch` block (around line 100), insert:

```markdown
### 0.6 Post-training & Alignment

How a base model becomes ChatGPT-like: SFT, LoRA/QLoRA, DPO, and
post-training quantization. 12 self-contained demos on Qwen2.5-0.5B,
all runnable on Mac CPU/MPS (except the bitsandbytes 4-bit demo,
which is documented for Colab/CUDA).

| Module | Directory | Core Concepts |
|--------|-----------|---------------|
| Post-training & Alignment | `ml_foundations/post_training/` | Data construction, SFT, LoRA, QLoRA (MLX + bnb), DPO, PPO walkthrough, GGUF quantization, perplexity, lm-eval-harness |

Main demos run in **~5 min each on Mac MPS**. See
`docs/ml-foundations/post-training/` for the corresponding theory chapters
covering SFT loss masking, low-rank decomposition math, NF4 quantization,
DPO derivation, GGUF k-quants, and selection decision tables.

**After completing:** You can pick the right fine-tuning approach for a given
dataset size and hardware budget, read PEFT/TRL/llama.cpp source confidently,
and explain why DPO replaced PPO.
```

- [ ] **Step 5: Update LEARNING_ROADMAP.md**

After the existing Phase 2 section, insert a Phase 3 block following the same table style. Include:
- 12 个 demo 表（编号、文件名、核心知识点）
- 推荐学习顺序（按数字顺序 01 → 12）
- 在「推荐学习顺序」总表中插入 `Week 0.6: post_training/01 → 12`

- [ ] **Step 6: Update requirements.txt**

Append to end of `requirements.txt`:

```
# ===== Phase 3: 训练后期与对齐（按需安装）=====
transformers>=4.45.0
peft>=0.13.0
trl>=0.11.0
datasets>=3.0.0
accelerate>=1.0.0

# 可选依赖（不影响其他 demo）:
# bitsandbytes>=0.43.0     # 07_qlora_peft_bnb / 10_quantization 用，仅 CUDA
# llama-cpp-python>=0.3.0  # 10_quantization 用（推理）
# mlx-lm>=0.20.0           # 06_qlora_mlx 用，仅 Apple Silicon
# lm-eval>=0.4.0           # 12_eval_lm_harness 用
```

- [ ] **Step 7: Verify pip install on Mac**

```bash
pip install -r requirements.txt
```

Expected: 所有强依赖（transformers/peft/trl/datasets/accelerate）安装成功，注释中的可选包不被尝试安装。

- [ ] **Step 8: Commit**

```bash
git add docs/.vitepress/ README.md LEARNING_ROADMAP.md requirements.txt \
        docs/ml-foundations/post-training/
git commit -m "wip(phase3): integrate sidebar/README/roadmap/requirements"
```

---

## Task 25: Smoke-test all 12 demos

**Files:**
- Read-only verification; no file changes
- Optional update: `docs/superpowers/specs/2026-06-18-phase3-post-training-design.md` §7.2 表格中的"<X min"占位替换为实测值

**Interfaces:**
- Consumes: 全部 12 个 demo
- Produces: 验证记录（更新 spec 中的 smoke-test 矩阵实测时长）

- [ ] **Step 1: Run each demo in numerical order and record actual wall time**

```bash
for f in ml_foundations/post_training/0*_*.py ml_foundations/post_training/1*_*.py; do
  echo "=== $f ==="
  /usr/bin/time -p python "$f"
done 2>&1 | tee /tmp/phase3_smoke.log
```

For demos that legitimately exit (07 on Mac, 10 if `llama-cpp-python` missing, 12 if `lm-eval` missing), confirm the exit message is friendly (Chinese, points to install command or alternative path).

- [ ] **Step 2: Update spec §7.2 with measured times**

In `docs/superpowers/specs/2026-06-18-phase3-post-training-design.md`, replace each "✅ <Xmin" placeholder in the smoke-test matrix with the actual measured time from `/tmp/phase3_smoke.log`.

- [ ] **Step 3: Verify each demo's key outputs**

Check that each demo's stdout contains:
- ✅ box-art docstring header
- ✅ "关键收获" trailing block with 3-5 lines
- ✅ No Python traceback
- ✅ For training demos: monotonically (mostly) decreasing loss

If any check fails, fix the demo before proceeding.

- [ ] **Step 4: Commit spec update**

```bash
git add docs/superpowers/specs/2026-06-18-phase3-post-training-design.md
git commit -m "wip(phase3): update smoke-test matrix with measured times"
```

---

## Task 26: Squash into single final commit

**Files:**
- Squash all `wip(phase3): ...` commits from Tasks 1-25 into one final commit

**Interfaces:**
- Consumes: all wip commits
- Produces: single commit on master

- [ ] **Step 1: Count wip commits since last non-wip commit**

```bash
git log --oneline | head -40
N_WIP=$(git log --oneline | grep -c "wip(phase3)")
echo "Found $N_WIP wip commits to squash"
```

- [ ] **Step 2: Soft-reset to merge them**

```bash
git reset --soft HEAD~$N_WIP
git status  # 全部 wip 文件应该处于 staged 状态
```

- [ ] **Step 3: Create the final commit**

```bash
git commit -m "$(cat <<'EOF'
feat(ml): add post-training module + docs

Phase 3 covering SFT / LoRA / QLoRA / DPO / quantization / eval on
Qwen2.5-0.5B. 12 demos all runnable on Mac (CPU/MPS), with bnb-CUDA
path documented for cloud. Adds ml_foundations/post_training/, docs
under docs/ml-foundations/post-training/, README + LEARNING_ROADMAP
updates, and requirements pin for transformers/peft/trl/mlx-lm.

Spec:  docs/superpowers/specs/2026-06-18-phase3-post-training-design.md
Plan:  docs/superpowers/plans/2026-06-22-phase3-post-training.md

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify**

```bash
git log --oneline | head -5
git status
git diff HEAD~1 --stat | tail -5
```

Expected: top commit is `feat(ml): add post-training module + docs`, working tree clean, diff stat shows ~20-25 files changed with several thousand insertions.

- [ ] **Step 5: (Optional) Push if user requests**

Do NOT push automatically. Ask user before `git push`.

---

## Self-Review (run by plan author before handoff)

1. **Spec coverage**:
   - §1-2 (motivation, scope): covered by plan header + global constraints ✅
   - §3 (directory structure, 12 demos, naming, data策略): Tasks 1-13 ✅
   - §4 (per-demo specs): Tasks 2-13 each implement one demo ✅
   - §4.3 (failure modes): each demo has friendly-exit branches (check 06/07/10/12 explicitly) ✅
   - §5 (KNOWLEDGE.md): Task 14 ✅
   - §6 (VitePress + sidebar + README + roadmap + requirements + gitignore): Tasks 15-23 (docs) + Task 24 (integration); .gitignore folded into Task 1 ✅
   - §7 (smoke test + single commit): Tasks 25-26 ✅
   - §8 (success criteria): verified by Task 25 ✅

2. **Placeholder scan**: 全部步骤都有完整代码 / 完整命令 / 完整 expected output。所有"Similar to Task N"类引用都展开了。文档章节任务 (17-23) 显式说明"Transplant KNOWLEDGE Ch.X + 上下页链接"并指明 frontmatter 模板，避免"TBD"。

3. **Type consistency**:
   - Demo 间 import: 已确认 0 个跨脚本 import（spec §3.2 决策）✅
   - File paths: 全部用 `ml_foundations/post_training/` 前缀，统一 ✅
   - Adapter 路径：03 用 `runs/03_sft_full/`、05 用 `runs/05_lora_peft/`、08 用 `runs/08_dpo/`，11/12 按这些路径加载 ✅
   - Hyperparameters: 03 用 lr=2e-5（全参），05/08 用 lr=2e-4/5e-5（LoRA），均与 LoRA 论文经验一致 ✅

Plan ready for execution.

