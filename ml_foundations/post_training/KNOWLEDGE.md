# 训练后期与对齐 — 完全手册

> Pretrain 只是起点；真正决定模型"有没有用、好不好用"的，是训练后期（Post-training）的一系列工程。本手册从全景漏斗开始，逐层拆解 SFT → PEFT/LoRA → QLoRA → DPO → PTQ → 评估 → 选型，每一节都回答"为什么这样做"，再讲"是什么 + 怎么做"。

---

## 一、学习路径

本目录 12 个 demo 构成完整的后训练实验室。推荐按下表顺序阅读：

| 序号 | 文件 | 为什么放在这里 |
|------|------|----------------|
| 1 | [`01_data_construction.py`](./01_data_construction.py) | 动手构造 SFT 数据集；理解 chat template 格式 |
| 2 | [`02_multi_turn_chat.py`](./02_multi_turn_chat.py) | 多轮对话的 mask 策略；看清哪些 token 参与 loss |
| 3 | [`03_sft_full.py`](./03_sft_full.py) | Full-parameter SFT；baseline 对比 |
| 4 | [`04_lora_from_scratch.py`](./04_lora_from_scratch.py) | 手写 LoRA 层，验证 W₀ + BA 的数学 |
| 5 | [`05_lora_peft.py`](./05_lora_peft.py) | PEFT 库调用；adapter 保存与加载 |
| 6 | [`06_qlora_mlx.py`](./06_qlora_mlx.py) | Apple MLX 上的 QLoRA（Mac 本机） |
| 7 | [`07_qlora_peft_bnb.py`](./07_qlora_peft_bnb.py) | bitsandbytes NF4 量化 + LoRA（Linux/CUDA） |
| 8 | [`08_dpo_alignment.py`](./08_dpo_alignment.py) | DPO 偏好对齐；直接看 loss 公式落地 |
| 9 | [`09_ppo_intro.py`](./09_ppo_intro.py) | PPO 四模型协作示意；理解工程债务 |
| 10 | [`10_quantization_inference.py`](./10_quantization_inference.py) | PTQ 量化推理；INT4/INT8 精度对比 |
| 11 | [`11_eval_perplexity.py`](./11_eval_perplexity.py) | Perplexity 计算与局限演示 |
| 12 | [`12_eval_lm_harness.py`](./12_eval_lm_harness.py) | lm-evaluation-harness 标准 benchmark |

> **前置知识**：熟悉 Transformer 架构与训练（[Phase 2 KNOWLEDGE.md](../transformer_training/KNOWLEDGE.md)）；了解交叉熵损失与反向传播。

---

## 第 0 章 全景：从 base model 到 ChatGPT 之间发生了什么

### 0.1 为什么 base model 不能直接用

拿到一个预训练完成的语言模型（比如 LLaMA-3 8B base），如果直接问它"帮我写一封请假信"，它大概率会继续补全语料风格的文本，而不是真正"回答"你的问题。原因很简单：预训练的目标是**预测下一个 token**，训练数据是网页/书籍/代码，模型学到的是"世界知识"，但没有学到"如何对话、如何帮助人"的行为模式。

把 base model 变成有用的助手，需要四个阶段：

### 0.2 四阶段流水线与漏斗

```
┌─────────────────────────────────────────────────────────────┐
│                    预训练 Pre-training                        │
│  数据：几万亿 tokens（Common Crawl / Books / Code）           │
│  算力：数千 GPU×月                                            │
│  参数更新：全量（7B 模型 ≈ 28GB fp16 梯度）                   │
│  目标：学习世界知识 + 语言统计规律                             │
└─────────────────────┬───────────────────────────────────────┘
                      │ 万亿 tokens → 数万样本（缩小 10⁸ 倍）
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  指令微调 SFT                                 │
│  数据：1k–200k 条高质量指令-回答对                            │
│  算力：数 GPU×小时（LoRA 可在单卡完成）                        │
│  参数更新：全量或 LoRA（可训练参数降至 0.1%）                  │
│  目标：学会"听指令"的行为模式                                  │
└─────────────────────┬───────────────────────────────────────┘
                      │ 数万条 → 数千条偏好对（再缩小 10 倍）
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               偏好对齐 RLHF / DPO                            │
│  数据：chosen/rejected 对比对（人工或 AI 标注）                │
│  算力：PPO 需 4 模型并行；DPO 等同 SFT 开销                   │
│  目标：对齐人类价值观；拒绝有害输出；更有帮助                   │
└─────────────────────┬───────────────────────────────────────┘
                      │ 对齐后模型 → 量化压缩（体积降 4-8×）
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               量化部署 Quantization + Serving                │
│  技术：GPTQ / AWQ / GGUF / bitsandbytes                     │
│  效果：26GB fp16 → 6GB int4（13B 模型）                      │
│  目标：在消费级硬件上高速推理                                   │
└─────────────────────────────────────────────────────────────┘
```

### 0.3 每阶段量级对照

| 阶段 | 数据规模 | 训练时长（7B 模型） | 显存需求（LoRA） | 产出 |
|------|---------|-----------------|----------------|------|
| 预训练 | 1–10T tokens | 数千 GPU×月 | 数百 GPU | base weights |
| SFT | 1k–200k 对话 | 1–24 GPU×小时 | 单卡 16GB | instruction model |
| DPO | 1k–50k 偏好对 | 2–12 GPU×小时 | 单卡 16GB | aligned model |
| PTQ | 无需训练 | 分钟级 | CPU 可用 | quantized model |

> **关键洞察**：每个阶段的数据量都在缩小，但每条数据的"信息密度"在上升。预训练靠量取胜；SFT 靠质量取胜；DPO 靠对比信号取胜。

**与生产对应** ← 这四个阶段对应 HuggingFace 生态的四个核心库：`transformers`（预训练 / 推理）、`trl.SFTTrainer`（SFT）、`trl.DPOTrainer`（DPO）、`bitsandbytes` / `llama.cpp`（量化）。

---

## 第 1 章 SFT：让模型学会"听指令"

### 1.1 为什么 SFT 有效

预训练语料的格式是连续文本（"The quick brown fox…"），而对话的格式是结构化轮次（`[INST] 你好 [/INST] 你好！`）。SFT 用极少量数据告诉模型"这种格式下我应该如何回应"，模型已有的知识不变，改变的只是**输出行为的条件分布**。

### 1.2 数据形态：Alpaca / ShareGPT / chat template

📌 **chat template** 是不同数据格式的统一抽象。三种主流格式：

**Alpaca 格式**（单轮指令）：
```json
{
  "instruction": "把下面句子翻译成英文",
  "input": "今天天气很好",
  "output": "The weather is great today."
}
```

**ShareGPT 格式**（多轮对话）：
```json
{
  "conversations": [
    {"from": "human", "value": "什么是 LoRA？"},
    {"from": "gpt", "value": "LoRA 是一种参数高效的微调方法…"},
    {"from": "human", "value": "它比全量微调省多少显存？"},
    {"from": "gpt", "value": "通常节省 80-90% 显存…"}
  ]
}
```

**Chat template**（tokenizer 渲染后的格式，以 LLaMA-3 为例）：
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
什么是 LoRA？<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
LoRA 是一种…<|eot_id|>
```

`tokenizer.apply_chat_template()` 自动处理这个渲染过程，不同模型的 special token 格式不同（Mistral 用 `[INST]`，Qwen 用 `<|im_start|>`）。

### 1.3 损失函数：为什么只对 response 算 loss

SFT 的训练目标是让模型学会"如何回应"，而不是"如何提问"。如果对整个序列（包括 user prompt）都算交叉熵 loss，会出现两个问题：

1. 模型会浪费参数容量"记住"用户输入的写法（这部分不可控）
2. prompt 部分的 loss 会稀释有用的梯度信号

解决方案：**对 prompt 部分做 label mask**，只在 assistant 回复的 token 上计算 loss：

```
token 序列: [BOS] [INST] 什么是 LoRA ？ [/INST] LoRA 是 一种 … [EOS]
label mask:  -100  -100  -100 -100 -100  -100  有效  有效 有效  有效
```

PyTorch CrossEntropyLoss 对 label=-100 的位置自动忽略。损失计算：

$$
\mathcal{L}_\text{SFT} = -\frac{1}{|T_\text{response}|} \sum_{t \in T_\text{response}} \log p_\theta(y_t \mid y_{<t}, x)
$$

其中 $T_\text{response}$ 是 assistant 回复的 token 集合，$x$ 是完整对话上下文（包括 prompt）。

### 1.4 多 turn 对话的 mask 策略对比

多轮对话有三种 mask 策略，各有权衡：

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **只掩盖最后一轮 prompt** | 只对最后一个 user turn mask | 简单易实现 | 浪费前几轮 assistant 的监督信号 |
| **掩盖所有 user turns** | 所有 human 发言全部 mask=-100 | 充分利用所有 assistant 回复的监督 | 实现稍复杂 |
| **不掩盖任何内容** | 整个序列都算 loss | 实现最简单 | 模型会"学习提问风格"，效果一般偏差 |

推荐方案：**掩盖所有 user turns**。`trl.SFTTrainer` 默认行为可通过 `DataCollatorForCompletionOnlyLM` 配置。

### 1.5 工程实践：TRL SFTTrainer 在做什么

`SFTTrainer` 本质上是封装了三件事：
1. 调用 `tokenizer.apply_chat_template()` 把对话格式转成 token ids
2. 用 `DataCollatorForCompletionOnlyLM` 自动生成 label mask
3. 标准 `Trainer` 循环（forward → loss → backward → optimizer step）

关键参数：
```python
SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=2048,          # 截断长度
    dataset_text_field="text",    # 已渲染好的对话字段
    peft_config=lora_config,      # 可选：同时启用 LoRA
)
```

### 1.6 SFT 的失败模式

**灾难性遗忘（Catastrophic Forgetting）**：过度 SFT 会使模型"忘记"预训练阶段学到的知识。典型症状：SFT 后模型在通用 benchmark（MMLU / HellaSwag）上得分下降。缓解方法：降低学习率（1e-5 以下）、减少 epoch 数（1-3 轮）、加 KL 惩罚项。

**模板过拟合（Template Overfitting）**：模型学会"说话的格式"但内容空洞。症状：所有回答开头都是"当然！我很乐意帮你…"。原因是 SFT 数据里充斥着 GPT-3.5 生成的模板化回答。解决方法：清洗数据，过滤低信息密度的回答。

**与生产对应** ← `trl.SFTTrainer`；数据格式参考 `datasets.load_dataset("timdettmers/openassistant-guanaco")`；Label mask 实现在 `trl/trainer/sft_trainer.py` 的 `DataCollatorForCompletionOnlyLM` 类。

---

## 第 2 章 PEFT 与 LoRA 数学

### 2.1 为什么需要 PEFT：全量微调的成本墙

假设我们要微调一个 7B 参数的模型：

- 模型本身：7B × 2 bytes (fp16) ≈ **14 GB**
- 优化器状态（Adam）：7B × 2（momentum + variance）× 4 bytes ≈ **56 GB**
- 梯度：7B × 2 bytes ≈ **14 GB**
- 激活值（取决于 batch size）：**数 GB**
- **合计：约 80-100 GB**，需要 2-3 张 A100（80GB）

这对绝大多数研究者和开发者是不可接受的。参数高效微调（**PEFT，Parameter-Efficient Fine-Tuning**）的核心思路：**冻结预训练权重，只训练少量新增参数**。

📌 **LoRA（Low-Rank Adaptation）** 是目前最主流的 PEFT 方法，由 Hu et al. 2021 提出。

### 2.2 LoRA 推导：从观察到公式

**关键观察**：神经网络在适应新任务时，权重的更新矩阵 $\Delta W$ 具有**低内在秩**。即：完整微调时，$W \leftarrow W_0 + \Delta W$，而 $\Delta W$ 的有效秩远小于 $d$。

如果 $\text{rank}(\Delta W) = r \ll d$，那么 $\Delta W$ 可以被分解为两个低秩矩阵的乘积：

$$
\Delta W = BA
$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times d}$，$r \ll d$。

加上缩放系数 $\alpha/r$，📌 **LoRA 的完整公式**为：

$$
W = W_0 + \frac{\alpha}{r} BA
$$

- $W_0$：冻结的预训练权重，不更新
- $B, A$：可训练的低秩矩阵
- $r$：秩（rank），通常取 4–64
- $\alpha$：缩放超参，通常设为 $r$ 的 1–2 倍（默认 $\alpha = r$ 时等效于 $\Delta W = BA$）

**初始化策略**：$A$ 用随机高斯初始化，$B$ 用零初始化。这保证训练开始时 $\Delta W = B \cdot A = 0$，模型行为与原始 $W_0$ 完全相同。

### 2.3 LoRA 矩阵分解图

```
完整权重矩阵（冻结，不反传梯度）：
┌─────────────────────────────────────┐
│                                     │
│         W₀  (d × d)                 │
│         ❄️  FROZEN                   │
│                                     │
└─────────────────────────────────────┘
              +
低秩更新（只有这两个矩阵有梯度）：
┌───────────┐   ┌─────────────────────┐
│           │   │                     │
│  B(d × r) │ × │    A (r × d)        │
│  🔥 train │   │    🔥 train         │
└───────────┘   └─────────────────────┘
  d行 r列           r行 d列

参数量对比：
  W₀ 全量微调:  d × d = d²
  LoRA 训练量:  d×r + r×d = 2dr  （r=8, d=4096 → 节省 256 倍）
```

### 2.4 r 取多少够用？

| 任务类型 | 推荐 r | 说明 |
|---------|--------|------|
| 对话/指令跟随 | 4–16 | 行为调整，秩需求低 |
| 代码生成 | 16–64 | 需要更多新知识迁移 |
| 特定领域知识注入 | 32–128 | 知识差距大时需要更高秩 |
| 完整任务迁移 | 64–256 | 接近全量微调效果 |

经验规律：r=16 在大多数对话任务上效果与 r=64 相差不超过 1-2%，但参数量节省 4 倍。

### 2.5 LoRA 的等价视角：低秩 + 隐式正则化

LoRA 可以视为在 $\Delta W$ 空间上加了一个**秩约束正则化**：强迫权重更新沿低秩流形移动，这天然避免了过拟合（全量微调时的灾难性遗忘部分来自高秩扰动）。

另一个等价视角：LoRA 把原始 $d \times d$ 参数空间的优化问题投影到 $2dr$ 维的子流形上，Adam 在低维空间里收敛更稳定。

### 2.6 target_modules 选择

不是所有矩阵都值得加 LoRA。经验上：

| 模块 | 是否加 LoRA | 原因 |
|------|-----------|------|
| `q_proj` / `k_proj` / `v_proj` | ✅ 必加 | Attention 是行为调整的核心 |
| `o_proj` | ✅ 推荐 | 输出投影同样重要 |
| `gate_proj` / `up_proj` / `down_proj` | 可选 | FFN 层；知识密集任务加 |
| `embed_tokens` / `lm_head` | ❌ 通常不加 | embedding 层秩扰动可能破坏 tokenizer 对齐 |

```python
peft.LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
```

### 2.7 Adapter merge / unmerge 与多 adapter 切换

训练完成后，可以把 LoRA 权重合并回基础模型：

$$
W_\text{merged} = W_0 + \frac{\alpha}{r} BA
$$

合并后模型与普通模型无区别（推理速度不变），但无法再 unmerge。

实际生产中常用**多 adapter 切换**：一个 base model + N 个任务 adapter，按请求类型动态加载，显著降低多任务部署成本。

```python
model.merge_adapter()    # 合并 LoRA 到 W₀
model.unmerge_adapter()  # 分离（仅在未 merge 状态有效）
```

### 2.8 LoRA 变体扫盲

📌 **DoRA（Weight-Decomposed LoRA）**：把权重分解为大小（magnitude）和方向（direction）分量，分别做 LoRA 适配，在某些任务上比 LoRA 高 1-2 个点。

📌 **VeRA（Vector-based RA）**：所有层共享同一对随机矩阵 AB，只训练逐层缩放向量，参数量极低（比 LoRA 再低 10 倍）。

📌 **LoHa / LoKr**：用 Hadamard 积 / Kronecker 积替代矩阵乘法，适合图像生成微调（LoRA for Stable Diffusion）。

📌 **AdaLoRA**：自适应地为不同层分配不同秩预算，根据奇异值大小动态剪枝。

**与生产对应** ← `peft.LoraConfig` / `peft.get_peft_model()`；合并操作 `PeftModel.merge_and_unload()`；demo 参见 [`04_lora_from_scratch.py`](./04_lora_from_scratch.py) 和 [`05_lora_peft.py`](./05_lora_peft.py)。Attention 数学回看 [Phase 2 §3.2](../transformer_training/KNOWLEDGE.md#32-完整公式)。

---

## 第 3 章 QLoRA：量化 + LoRA 的奇迹

### 3.1 为什么需要 QLoRA

LoRA 把可训练参数压缩到 0.1%，但 base model 仍需以 fp16 加载进显存（7B ≈ 14GB）。对于 13B 或 70B 模型，单卡还是放不下。QLoRA（Dettmers et al. 2023）提出：**把 base model 量化到 4-bit 存储，同时在 LoRA adapter 上以 bf16 精度训练**。

### 3.2 NF4 量化原理（4-bit Normal Float）

📌 **NF4（Normal Float 4-bit）** 是 QLoRA 核心创新之一。

普通 INT4 量化把数值范围均匀分成 16 个区间（类似等距刻度尺）。但神经网络权重的分布近似**标准正态分布 $\mathcal{N}(0, 1)$**，在 0 附近密集，在尾部稀疏。均匀量化会浪费大量精度在几乎不存在数据的尾部区间。

NF4 的解决方案：用正态分布的**分位数（quantile）** 作为量化点，使每个区间包含等量的权重值：

$$
q_i = \Phi^{-1}\!\left(\frac{i}{15}\right), \quad i = 0, 1, \ldots, 15
$$

其中 $\Phi^{-1}$ 是标准正态分布的逆 CDF（分位函数）。

```
NF4 的 16 个量化点（信息论最优，非均匀分布）：
                ⬛⬛⬛⬛⬛⬛⬛⬛⬛⬛
             ⬛            ⬛
           ⬛                ⬛
         ⬛                    ⬛
        ⬛                      ⬛
       ⬛                        ⬛
──────┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──────
    -1.0                 0.0                 +1.0

均匀分布（INT4，参考）：
──────┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──────
    -1.0                 0.0                 +1.0

观察：NF4 在 0 附近更密集（8 个点集中在 [-0.5, 0.5]），尾部更稀疏。
这与正态分布权重的真实分布匹配，量化误差更小。
```

**Block-wise 量化**：每 64 个权重一组，各自计算缩放系数（absmax），避免异常值影响整组精度。

### 3.3 Double Quantization 与 Paged Optimizer

**Double Quantization（双重量化）**：上面每 64 个权重需要一个 fp32 缩放系数（32 bits），这本身也占内存。QLoRA 进一步把这些缩放系数再量化（256 个缩放系数一组，再量化为 8-bit），节省约 0.37 bits/parameter。

**Paged Optimizer（分页优化器）**：LoRA adapter 的 Adam 优化器状态（bf16）在显存不足时自动换页到 CPU RAM，训练过程中透明地在 GPU/CPU 间搬运，避免 OOM 崩溃。

### 3.4 为什么 4-bit base + LoRA 不掉点

直觉：LoRA adapter 是以 bf16 精度训练的，梯度计算完全准确。反向传播时，对 4-bit 权重的梯度计算通过先反量化（dequantize）到 bf16 来完成。推理时 4-bit base + LoRA 效果之所以接近 fp16 full fine-tuning，是因为：

1. **NF4 的量化误差极小**（相比 INT4 均匀量化，量化噪声降低约 20-30%）
2. **LoRA 适配层补偿了量化引入的偏差**——adapter 学习中和了一部分量化噪声
3. **预训练权重的大量冗余**使得 4-bit 压缩损失的信息量远小于参数量降低的比例

### 3.5 实现路径：bitsandbytes vs Apple MLX

| 路径 | 硬件 | 库 | 精度 | 速度 |
|------|------|-----|------|------|
| bitsandbytes | NVIDIA GPU (Linux/WSL2) | `bitsandbytes` | NF4 + bf16 LoRA | 最快 |
| Apple MLX | Apple Silicon (M1/M2/M3) | `mlx` / `mlx-lm` | 4-bit + bf16 LoRA | 慢于 A100，但无需 GPU |
| GPTQ + LoRA | NVIDIA GPU | `auto-gptq` + peft | INT4 | 中等 |

```python
# bitsandbytes 路径 (07_qlora_peft_bnb.py)
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
```

### 3.6 内存账本：13B 模型从 26GB → 6GB

| 组件 | fp16 | NF4 QLoRA |
|------|------|-----------|
| 模型权重 | 26 GB | 6.5 GB |
| LoRA adapter (r=16) | — | ~0.1 GB |
| 优化器状态 (bf16 Adam) | 52 GB | 0.4 GB（仅 adapter） |
| 激活值 (batch=4) | ~4 GB | ~4 GB |
| **合计** | **~82 GB（需 2× A100）** | **~11 GB（单张 A100/3090）** |

**与生产对应** ← `transformers.BitsAndBytesConfig` + `peft.LoraConfig`；Apple MLX 路径见 `mlx-lm` 库；demo 见 [`06_qlora_mlx.py`](./06_qlora_mlx.py) 和 [`07_qlora_peft_bnb.py`](./07_qlora_peft_bnb.py)。

---

## 第 4 章 偏好对齐：从 PPO 到 DPO

### 4.1 为什么 SFT 不够：对齐税（Alignment Tax）

SFT 后的模型已经能"听指令"，但还不能保证：
- 拒绝有害请求（不泄露炸弹制造方法）
- 输出诚实、有帮助、无害（HHH 原则）
- 在多个"都对"的回答中选择更好的那个

这需要引入**人类偏好信号**。RLHF（Reinforcement Learning from Human Feedback）是 InstructGPT 和 ChatGPT 的关键技术。

### 4.2 RLHF 三阶段回顾

1. **SFT**：如第 1 章所述，训练出能对话的 base
2. **Reward Model（RM）训练**：给人类标注员两个回答 $(y_w, y_l)$，让他们选更好的；用这些偏好对训练一个奖励模型 $r_\phi(x, y) \in \mathbb{R}$
3. **PPO 强化学习**：用 RM 作为 reward signal，用 PPO 算法最大化期望奖励

### 4.3 PPO 的四模型协作

```
                    ┌────────────────────────────────────────────┐
                    │              PPO 训练时的四个模型             │
                    └────────────────────────────────────────────┘

   ┌──────────────┐   生成回答 y    ┌──────────────────┐
   │   Actor π_θ  │ ──────────────► │  Reward Model    │
   │  (可训练)    │                 │  r_φ(x, y) ∈ ℝ  │
   └──────┬───────┘                 │  (冻结)          │
          │                         └──────────────────┘
          │ log π_θ(y|x)
          ▼
   ┌──────────────┐                 ┌──────────────────┐
   │  Reference   │ ──── KL 惩罚 ──► │   Critic V_ψ    │
   │  π_ref       │                 │  (可训练)        │
   │  (冻结)      │                 │  估计 baseline   │
   └──────────────┘                 └──────────────────┘

  显存占用：4 个 7B 模型 ≈ 4 × 14GB = 56GB（仅权重）
```

### 4.4 PPO 损失函数

PPO 的 clipped surrogate objective：

$$
\mathcal{L}_\text{PPO} = \mathbb{E}\left[\min\left(r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right] - \beta \cdot \text{KL}(\pi_\theta \| \pi_\text{ref})
$$

其中：
- $r_t(\theta) = \pi_\theta(y_t|y_{<t}, x) / \pi_\text{old}(y_t|y_{<t}, x)$：新旧策略的概率比
- $\hat{A}_t$：advantage 估计（RM reward - critic baseline）
- $\epsilon$：clip 范围（通常 0.1–0.2），防止策略更新过大
- $\beta$：KL 惩罚系数，防止模型偏离 SFT 基础太远

### 4.5 PPO 的工程债务

PPO 理论优雅，但工程上噩梦：

1. **4 个模型同时在 GPU**：actor、reference、reward、critic，显存是 SFT 的 4 倍
2. **Reward Hacking**：actor 学会欺骗 reward model（说很多废话但 RM 打高分）
3. **训练不稳定**：clip ratio、KL 系数、advantage 归一化方式的超参极其敏感
4. **工程复杂度**：需要特殊的 rollout 生成 pipeline + 异步 RM 查询

这直接催生了 DPO 的出现。

### 4.6 DPO 推导：从 RL 形式到极大似然形式（完整推导，无跳步）

**动机**：能否绕过显式的 reward model 和 PPO，直接从偏好数据做优化？

**Step 1：Bradley-Terry 偏好模型**

人类偏好可以用 Bradley-Terry 模型建模：给定 prompt $x$ 和两个回答 $y_w$（preferred）、$y_l$（rejected），

$$
p(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))
$$

其中 $r(x, y)$ 是潜在的（隐含的）奖励函数，$\sigma$ 是 sigmoid 函数。

**Step 2：RL 目标**

RLHF 的 RL 优化目标为最大化期望奖励同时约束策略不要偏离 reference 太远：

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)}\left[r(x, y)\right] - \beta \cdot \text{KL}(\pi_\theta(y|x) \| \pi_\text{ref}(y|x))
$$

**Step 3：闭式最优策略**

对上面的 KL 约束 RL 目标，存在解析最优解。对每个 $x$，这是个带 KL 约束的变分优化问题，其闭式解为：

$$
\pi_r(y \mid x) = \frac{1}{Z(x)} \pi_\text{ref}(y \mid x) \exp\!\left(\frac{r(x, y)}{\beta}\right)
$$

其中配分函数 $Z(x) = \sum_y \pi_\text{ref}(y|x) \exp(r(x,y)/\beta)$ 是归一化常数。

**Step 4：反解奖励函数（Re-parameterize）**

从闭式最优策略反解奖励函数 $r(x, y)$：

$$
r(x, y) = \beta \log \frac{\pi_r(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x)
$$

**关键洞察**：我们把未知的 $r(x,y)$ 用 **策略 $\pi_r$ 本身** 重新参数化了！

**Step 5：代入 Bradley-Terry 模型**

将 Step 4 的 $r$ 代入 Step 1 的 Bradley-Terry 公式：

$$
p(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi_r(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi_r(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)
$$

注意：$\beta \log Z(x)$ 项在相减时消掉了！

**Step 6：最大似然损失**

对偏好数据集 $\{(x, y_w, y_l)\}$ 做最大似然估计（最大化人类选 $y_w$ 的概率），取负对数：

$$
\boxed{
\mathcal{L}_\text{DPO} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\!\left(\beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right]\right)\right]
}
$$

这就是 📌 **DPO loss** 的完整形式。

用 LaTeX 展示完整公式：

$$
\mathcal{L}_\text{DPO} = -\log \sigma\!\left(\beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right]\right)
$$

### 4.7 DPO 损失的几何意义与 β 超参

DPO loss 实质上在做：**拉大 preferred 回答的隐含奖励，压低 rejected 回答的隐含奖励**，且整个过程不需要显式的奖励模型。

$\beta$（温度系数）的几何意义：
- **$\beta$ 越小**：策略可以大幅偏离 reference，偏好信号被强化（但容易丢失多样性）
- **$\beta$ 越大**：策略被约束在 reference 附近，保守但稳定
- **典型取值**：0.01–0.5；对话任务常用 0.1–0.2

### 4.8 后 DPO 时代：ORPO / KTO / SimPO / IPO 一览

| 方法 | 核心创新 | 优势 |
|------|---------|------|
| **ORPO** | 在 SFT loss 中直接加 odds ratio 偏好项，无需 reference model | 单阶段训练，省 reference model |
| **KTO** | 只需单条回答的好/坏标签，不需要 paired 数据 | 数据收集更容易 |
| **SimPO** | 用归一化对数似然代替 log ratio，无需 reference model | 更简洁稳定 |
| **IPO** | 直接优化偏好概率，避免 Bradley-Terry 假设的缺陷 | 理论更严密 |

**与生产对应** ← `trl.DPOTrainer`（DPO / IPO / ORPO / KTO 均支持，通过 `loss_type` 参数切换）；PPO 见 `trl.PPOTrainer`；demo 见 [`08_dpo_alignment.py`](./08_dpo_alignment.py) 和 [`09_ppo_intro.py`](./09_ppo_intro.py)。

---

## 第 5 章 训练后量化（PTQ）

### 5.1 为什么量化

对齐好的模型要真正在生产环境服务，还面临一个问题：**推理成本**。fp16 的 7B 模型需要 14GB 显存，每次前向推理的浮点计算量也很大。量化把权重（和/或激活值）从 16/32-bit 压缩到 8/4-bit，显存降低 2-4 倍，推理速度提升 1.5-4 倍。

### 5.2 量化基础

📌 **对称量化**（symmetric）：量化区间关于 0 对称，用 $[-\text{absmax}, +\text{absmax}]$ 均匀分成 $2^b$ 份。

$$
x_q = \text{round}\!\left(\frac{x}{\text{scale}}\right), \quad \text{scale} = \frac{\text{absmax}}{2^{b-1}-1}
$$

📌 **非对称量化**（asymmetric）：用 $[\min, \max]$ 作为量化范围，额外存储 zero-point 参数。适合激活值（ReLU 后的激活值全为正）。

**Per-tensor vs Per-channel**：
- **Per-tensor**：整层权重共享一个 scale，速度快，精度低
- **Per-channel（per-row）**：每行/列独立 scale，精度高（AWQ/GPTQ 默认），稍慢

### 5.3 INT8 / INT4 的精度损失来源

**异常值（Outliers）** 是量化精度下降的主要来源。LLM 激活值中存在少数绝对值极大的维度（比均值大 100 倍），它们迫使 scale 变大，导致正常值的量化粒度变粗。

INT8 量化（8-bit，256 个等级）：精度损失通常 < 1%，可用于生产。
INT4 量化（4-bit，16 个等级）：精度损失 1-5%，需要高级量化算法（GPTQ/AWQ/NF4）来补偿。

### 5.4 GPTQ：基于 Hessian 的逐层量化

GPTQ（Frantar et al. 2022）的核心思路：逐层量化，量化完一层后用该层的 Hessian 矩阵（二阶导数）来**补偿误差**，调整剩余未量化权重来最小化量化后的层输出误差。

关键特点：
- 需要一小批校准数据（~128 样本）来估计 Hessian
- 量化速度：7B 模型约需 1-4 GPU 小时
- 精度：INT4 GPTQ 接近 fp16 baseline（差距 < 2%）

```bash
# AutoGPTQ 示例
python -m auto_gptq.convert --model llama-7b --bits 4 --group_size 128
```

### 5.5 AWQ：Activation-Aware 量化

AWQ（Lin et al. 2023）的观察：权重中只有**约 1% 的权重对激活值的影响最大**（即"重要权重"），对这些权重做高精度量化，其余权重粗糙量化，整体精度超过 GPTQ。

实现：分析激活值分布找到重要权重 → 对重要权重缩放（等价于将其放大再量化，变相提高其量化精度）→ INT4 量化全部权重。AWQ 推理速度比 GPTQ 更快（硬件友好的权重分布）。

### 5.6 GGUF / llama.cpp 量化家族

`llama.cpp` 使用 GGUF 格式，提供多个精度档位：

| 量化类型 | bits/weight | 7B 显存 | 质量 |
|---------|------------|---------|------|
| Q4_K_M | 4.5 | 4.8 GB | ⭐⭐⭐⭐ 推荐 |
| Q5_K_M | 5.5 | 5.7 GB | ⭐⭐⭐⭐⭐ |
| Q6_K | 6.6 | 6.6 GB | 接近 fp16 |
| Q8_0 | 8.0 | 7.7 GB | 几乎无损 |
| Q2_K | 2.6 | 2.9 GB | 质量较差 |

`K` 后缀表示 k-quant（混合精度，重要层用更高精度）；`M` 表示 medium 配置。

### 5.7 量化时机决策表

| 场景 | 推荐方案 | 工具 |
|------|---------|------|
| Mac 本机推理 | Q4_K_M / Q5_K_M GGUF | llama.cpp / Ollama |
| 单卡 CUDA 推理 | NF4 bitsandbytes | transformers BitsAndBytesConfig |
| 生产 GPU 服务 | GPTQ INT4 / AWQ INT4 | vLLM + AutoGPTQ |
| 追求最佳质量 | Q8_0 或 fp16 | 根据显存选择 |
| 微调后再量化 | AWQ（需校准数据） | autoawq |

**与生产对应** ← `bitsandbytes.BitsAndBytesConfig`；`auto_gptq.AutoGPTQForCausalLM`；`awq.AutoAWQForCausalLM`；CPU/Mac 推理用 `llama-cpp-python`；demo 见 [`10_quantization_inference.py`](./10_quantization_inference.py)。

---

## 第 6 章 评估方法学

### 6.1 为什么评估比训练更难

训练有明确的 loss 曲线；评估却没有单一的"正确答案"。"这个回答是否比那个更好"本质上是主观判断。评估方法的选择直接影响我们对模型好坏的认知。

### 6.2 Perplexity 的局限

📌 **Perplexity（困惑度）** 是最古老的语言模型评估指标：

$$
\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{t=1}^N \log p(y_t \mid y_{<t})\right)
$$

PPL 越低，模型对测试文本的预测越好。但 PPL 有两个致命局限：

1. **与下游任务质量相关性弱**：PPL 低的模型不一定在对话、翻译、推理任务上表现好
2. **被 tokenizer 影响**：不同 tokenizer 的 PPL 不可直接比较（词表大小影响每步预测难度）

### 6.3 LLM-as-Judge

用更强的 LLM（如 GPT-4）来给被测模型的输出打分。优点：接近人类判断，自动化；缺点：昂贵，存在偏见（更喜欢自己风格的输出）。

常见范式：
- **Pairwise comparison**：给 GPT-4 两个回答，让它选更好的
- **Absolute scoring**：让 GPT-4 给回答打 1-10 分
- **Criteria-based**：按 helpful / harmless / honest 三维度分别打分

### 6.4 标准 Benchmark

| Benchmark | 内容 | 评估维度 |
|-----------|------|---------|
| **MMLU** | 57 学科多选题（大学水平） | 知识广度 |
| **ARC** | 小学理科多选题 | 常识推理 |
| **HellaSwag** | 句子续写（选最合理的结尾） | 常识理解 |
| **GSM8K** | 小学数学应用题（需要推理步骤） | 数学推理 |
| **HumanEval** | Python 编程填空 | 代码生成 |
| **TruthfulQA** | 常见错误信念测试 | 诚实性 |

### 6.5 lm-evaluation-harness 工程实践

EleutherAI 的 `lm-evaluation-harness` 是运行标准 benchmark 的事实标准工具：

```bash
lm_eval --model hf \
        --model_args pretrained=meta-llama/Llama-3-8B-Instruct \
        --tasks mmlu,arc_challenge,hellaswag,gsm8k \
        --device cuda:0 \
        --batch_size 8
```

`lm_eval` 自动处理 few-shot prompting、loglikelihood 评估、结果汇总，输出标准化分数。

### 6.6 Chatbot Arena / MT-Bench

📌 **Chatbot Arena**：LMSYS 的人工盲测平台。用户与两个匿名模型对话，选出更好的那个，用 Elo 评分系统排名。这是目前最接近"用户真实体验"的评估方法，但成本高、速度慢。

**MT-Bench**：80 道多轮对话问题，用 GPT-4 打分（1-10 分）。分为数学、代码、写作、推理等 8 个类别，可以快速定位模型的弱项。

### 6.7 Benchmark 污染问题

如果模型的预训练数据中包含了 benchmark 的题目和答案，评分就会虚高。这是领域内的严重问题。缓解方法：使用时间戳更新的 benchmark、私有测试集、或检查训练数据是否包含 benchmark 内容的哈希值。

**与生产对应** ← `lm-evaluation-harness`（`pip install lm-eval`）；PPL 计算见 [`11_eval_perplexity.py`](./11_eval_perplexity.py)；harness 调用见 [`12_eval_lm_harness.py`](./12_eval_lm_harness.py)。

---

## 第 7 章 选型决策手册

### 7.1 问题：面对一个新任务，用什么方法？

这是最常见的工程问题。以下从两个维度给出决策树：数据量和硬件。

### 7.2 数据量 vs 方法选择

| 数据规模 | 场景 | 推荐方法 | 理由 |
|---------|------|---------|------|
| **< 200 条** | 极少数据，验证 POC | Prompt Engineering / RAG | 数据不够训练，先试 prompt |
| **200–2k 条** | 小规模定制 | LoRA（r=8, 1-2 epoch） | 低秩适应足够，过拟合风险低 |
| **2k–20k 条** | 典型微调场景 | LoRA（r=16–32）/ QLoRA | 标准做法，可以加 DPO 对齐 |
| **20k–200k 条** | 垂直领域全量微调 | Full SFT + LoRA 混合 | 数据充足，可考虑更高 rank |
| **> 200k 条** | 从头 SFT 大模型 | Full fine-tuning | 接近预训练规模，LoRA 可能欠拟合 |

### 7.3 硬件 vs 方法选择

| 硬件 | 显存 | 可以做什么 |
|------|------|---------|
| Mac M2/M3 | 16–48 GB 统一内存 | QLoRA（MLX）7B 模型；GGUF 推理 70B |
| RTX 3090/4090 | 24 GB | QLoRA 7B–13B；LoRA 7B fp16 |
| A100 40GB | 40 GB | Full SFT 7B；LoRA 13B fp16 |
| A100 80GB | 80 GB | Full SFT 13B；LoRA 70B；DPO 13B |
| 8× A100 | 640 GB | Full SFT 70B；PPO 13B |

### 7.4 方法对比矩阵

| 方法 | 显存（7B） | 训练速度 | 最终质量 | 适用场景 |
|------|-----------|---------|---------|---------|
| Full SFT fp16 | 80 GB | 最快 | 最高 | 数据充足 + 多 GPU |
| LoRA r=16 fp16 | 18 GB | 快 | 高（-1–2%） | 单卡 A100 |
| QLoRA NF4 r=16 | 11 GB | 中 | 中高（-2–4%） | 单卡 3090 |
| DPO（LoRA 基础上） | 18 GB | 中 | 对齐最佳 | 有偏好数据 |
| PTQ INT4 GPTQ | 5 GB（推理） | 推理最快 | 接近 fp16 | 部署优先 |

> **经验法则**：先 RAG，再 LoRA，最后才考虑全量微调。微调是昂贵的，不要把数据问题用算法解决。

**与生产对应** ← 决策过程参考 `peft` 文档的 Performance Guide；多卡训练用 `accelerate` + `deepspeed`；自动选型工具 `optimum`。

---

## 附录 A：与 Phase 2 自训 GPT 的衔接

### A.1 为什么不在 Phase 2 的 GPT 上做 SFT/DPO

Phase 2 的 [`gpt_train.py`](../transformer_training/gpt_train.py) 训练了一个约 3M 参数的字符级 GPT。这个模型无法成为 Phase 3 实验的基础，原因如下：

1. **参数量太小**：3M 参数的模型几乎没有"世界知识"的存储空间。SFT 需要模型已有足够知识，才能学会"如何表达"这些知识。给 3M 模型做 SFT，实质上是从零学，而不是"对齐"。

2. **Chat template 缺失**：Phase 2 用字符级 tokenizer（vocab ≈ 65），没有 `[INST]`、`<|im_start|>` 等 special token，无法表示对话结构。

3. **SFT/DPO 数据稀疏下不收敛**：Alpaca 格式的 SFT 数据对一个 3M 模型来说是严重的 distribution shift，实验显示 loss 不稳定。

### A.2 Phase 2 知识在 Phase 3 的延续

Phase 2 的理论基础在 Phase 3 中完全沿用，只是规模更大：

| Phase 2 概念 | Phase 3 对应 | 参考 |
|-------------|------------|------|
| GPT 的 `CausalSelfAttention` 类 | Qwen/LLaMA 的 `LlamaAttention` 完全相同结构 | [Phase 2 §3](../transformer_training/KNOWLEDGE.md#三自注意力数学推导) |
| $Q K^\top / \sqrt{d_k}$ 注意力公式 | 0.5B–70B 模型的 attention 数学一模一样 | [Phase 2 §3.2](../transformer_training/KNOWLEDGE.md#32-完整公式) |
| Temperature / Top-p 采样 | 生产推理中 `generation_config` 的完全相同参数 | [Phase 2 §6](../transformer_training/KNOWLEDGE.md#六采样策略) |
| KV Cache 原理 | vLLM / TensorRT-LLM 的核心优化仍是 KV Cache | [Phase 2 §7](../transformer_training/KNOWLEDGE.md#七kv-cache) |

**与生产对应**：本附录的衔接关系实际反映在 `ml_foundations/transformer_training/gpt_train.py`（Phase 2 的 ~3M GPT 训练脚本）→ `transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")`（Phase 3 的预训练基座加载）之间。两者的 `nn.MultiheadAttention` / `LayerNorm` / RoPE 实现在结构上完全同构，只是 Phase 3 直接复用 Hugging Face 的 fused kernel。

---

## 附录 B：与上层 LangChain/RAG 章节的接口

### B.1 自己微调的 Adapter 怎么在 LangChain 里加载

完成 QLoRA 微调后，有两种使用方式：

**方式 1：合并后加载（推荐生产环境）**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
model = PeftModel.from_pretrained(base, "./my-lora-adapter")
model = model.merge_and_unload()       # 合并 LoRA 到 base weights
model.save_pretrained("./my-merged-model")

# LangChain 加载
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
pipe = pipeline("text-generation", model="./my-merged-model", ...)
llm = HuggingFacePipeline(pipeline=pipe)
```

**方式 2：PEFT 直接加载（省磁盘，灵活切换 adapter）**
```python
from langchain_community.llms import HuggingFacePipeline
# 保持 PEFT 包装，不 merge
pipe = pipeline("text-generation", model=peft_model, ...)
llm = HuggingFacePipeline(pipeline=pipe)
```

### B.2 微调 vs RAG 的工程权衡

| 维度 | 微调 | RAG |
|------|------|-----|
| 知识来源 | 写入模型权重（静态） | 外部向量库（可实时更新） |
| 更新成本 | 重新训练（高） | 更新向量库（低） |
| 推理延迟 | 无额外延迟 | 有检索延迟（50-200ms） |
| 知识边界 | 训练数据截止 | 可实时扩展 |
| 适用场景 | 行为/风格/格式调整 | 特定文档库问答 |

**"先 RAG 后微调"原则**：如果目标是"让模型知道更多知识"，RAG 几乎总是更快更便宜；如果目标是"改变模型的输出行为风格"或"让模型掌握特定任务格式"，才考虑微调。两者可以组合：先 RAG 检索上下文，再用微调好的模型生成高质量回答。

**与生产对应**：本附录涉及的 API 对照如下：
- LangChain 侧加载 PEFT adapter：`langchain_community.llms.HuggingFacePipeline` + `transformers.pipeline("text-generation", model=...)` + `peft.PeftModel.from_pretrained(base, adapter_dir)`
- 微调-RAG 联动：自己微调的 adapter 通过 `merge_and_unload()` 后直接当作 LangChain `LLM` 注入到 `RetrievalQA` chain 中，与 Project 2/5/17 的 RAG 系统天然组合

---

## 参考文献

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", NeurIPS 2023
- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", NeurIPS 2023
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", ICLR 2023
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", MLSys 2024
- Ouyang et al., "Training language models to follow instructions with human feedback", NeurIPS 2022 (InstructGPT)
