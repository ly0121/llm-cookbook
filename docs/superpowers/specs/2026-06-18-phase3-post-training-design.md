# Phase 3: 训练后期与对齐 — 设计文档

> 在 Phase 2（自训 GPT 全链路）之上，构建从「指令数据 → SFT → LoRA/QLoRA → DPO → 量化部署」的完整训练后期 demo，让读者从理解 LLM 内部 → 学会调一个 LLM。

**Date**: 2026-06-18
**Status**: 待用户 review
**前置依赖**: Phase 2（已完成，commit `63c8ba0`）

---

## 1. 背景与动机

Phase 1 帮零基础读者补齐了"LLM 之前的 ML"。Phase 2 让读者亲手训过一个 ~3M GPT，理解 attention / 位置编码 / 采样 / KV cache。但这两个 phase 留下了一个明显的缺口：**base model 怎么变成 ChatGPT？**

本 phase 用 nanoGPT 风格无法承载的真实指令数据 + 偏好对齐流程，让读者：

- 用 TRL `SFTTrainer` 把 Qwen2.5-0.5B 调成"听话"的指令模型
- 自己手写 LoRA 层，理解 `W = W₀ + (α/r)BA` 不是黑魔法
- 跑通 QLoRA 的两条路（Apple MLX 原生 + PEFT/bnb 通用）
- 用 TRL `DPOTrainer` 体会"DPO 一招打 PPO 三步"
- 跑通 GGUF 量化推理，理解部署侧最后一公里

完成后，再看 PEFT、TRL、llama.cpp 源码，所有零件都认识；面对"我有 200 / 2k / 20k 条数据，硬件 X，要选哪个方法"的工程问题能直接答出来。

---

## 2. 范围决策

### 2.1 已锁定约束（来自 brainstorming Q&A）

| 决策项 | 选择 | 备注 |
|--------|------|------|
| 方向 | Phase 3：SFT + LoRA/QLoRA + DPO + 量化 | Phase 2 spec 中明确预留 |
| 基座模型 | Qwen2.5-0.5B-Instruct（主） / TinyLlama-1.1B-Chat（备） | 现代小模型，含完整 chat template |
| 硬件目标 | Mac CPU/MPS 全部 demo 可跑（QLoRA bnb 路线除外） | 沿用 Phase 1/2 原则 |
| QLoRA 路径 | 双实现：MLX 原生 + PEFT/bnb（标记需 CUDA / Colab） | bitsandbytes 在 Mac MPS 不官方支持 |
| demo 数量 | 12 个（超集，含评估闭环） | |
| 测试 | 不写 `tests/` | 沿用 Phase 1/2 决策 |
| commit 策略 | 单次最终 commit | 沿用 Phase 1/2 决策 |
| 文档形态 | 新建 `docs/ml-foundations/post-training/` 子章节 | 9 个 md（1 index + 8 内容） |
| 侧边栏位置 | "零.6、训练后期与对齐"（插在 Phase 2 之后） | |
| PPO 真训 | 不真训，仅 ASCII 流程图 + 伪代码 + 张量形状演示 | 0.5B + 200 条数据 PPO 训不出有意义结果 |
| 量化路径 | `llama.cpp` GGUF 优先（软依赖），`llama-cpp-python` 推理兜底 | Mac 上 `llama.cpp` 是一等公民 |

### 2.2 显式不做（YAGNI）

- ❌ RLHF (PPO) 真训
- ❌ 多卡分布式（FSDP / DeepSpeed）
- ❌ Constitutional AI / RLAIF
- ❌ 自定义 reward model 训练
- ❌ Speculative decoding / Medusa（属于 Phase 4）
- ❌ 中文垂域语料微调
- ❌ 多模态 SFT（图文 / 语音）
- ❌ MoE 微调
- ❌ ORPO / KTO / SimPO / IPO（仅在 KNOWLEDGE.md 提及）

### 2.3 阶段路线再次声明

- **Phase 1** ✅：经典 ML + DL + NLP（commit `34a0070` + `0f02860`）
- **Phase 2** ✅：Transformer 从零训练（commit `63c8ba0`）
- **Phase 3（本次）**：SFT / LoRA / QLoRA / DPO / 量化
- **Phase 4（未来，独立 spec）**：vLLM / TGI / continuous batching / 推理优化与部署

---

## 3. 代码目录结构

```
ml_foundations/post_training/
├── KNOWLEDGE.md                       # 总览 + 数学推导 + 与 LLM 实践对应
├── data/
│   ├── alpaca_mini.jsonl              # ~200 条指令样本（自构造，演示用）
│   ├── dpo_pairs_mini.jsonl           # ~100 条 chosen/rejected 偏好对
│   └── README.md                      # 数据来源声明 + 重新生成脚本
│
├── 01_data_construction.py            # Alpaca / ShareGPT / chat template
├── 02_multi_turn_chat.py              # 多 turn 对话模板 + loss mask
├── 03_sft_full.py                     # TRL SFTTrainer 全参微调
│
├── 04_lora_from_scratch.py            # 手写 LoRA 层（W = W₀ + BA）
├── 05_lora_peft.py                    # PEFT 库使用（target_modules / r / α）
├── 06_qlora_mlx.py                    # MLX 原生 4-bit QLoRA（Mac 主路径）
├── 07_qlora_peft_bnb.py               # PEFT + bnb 4-bit（Colab/CUDA 路径）
│
├── 08_dpo_alignment.py                # TRL DPOTrainer 偏好对齐
├── 09_ppo_intro.py                    # RLHF/PPO 全景 + 伪代码（不真训）
│
├── 10_quantization_inference.py       # GGUF (llama.cpp) + bnb 4-bit 推理对比
│
├── 11_eval_perplexity.py              # 困惑度 + 人工 side-by-side 对比
└── 12_eval_lm_harness.py              # lm-evaluation-harness 子集集成
```

**总计**：1 子目录 / 12 个 `.py` demo / 1 个 `KNOWLEDGE.md` / 1 个数据子目录

### 3.1 文件命名约定

Phase 3 用数字前缀（`01_xxx.py`），与 Phase 2 不同（Phase 2 因为有跨脚本 `from gpt_train import GPT` 复用所以无前缀）。Phase 3 demo 之间**不互相 import**，所以可以用数字前缀直接体现学习顺序。

### 3.2 模型与 checkpoint 复用约定

- **不**像 Phase 2 那样在脚本之间共享 ckpt。
- 每个 demo 独立从 HuggingFace 下载基座（首次下载 ~1GB 缓存到 `~/.cache/huggingface/`）。
- 训练产物（adapter / merged model）保存到 `ml_foundations/post_training/runs/<demo_name>/`，全部进 `.gitignore`。
- Demo 之间通过**文档**说明衔接（如「跑完 03 后的 SFT 模型可以作为 08 DPO 的起点，但本 demo 为了独立可跑直接用 base」）。

### 3.3 数据集策略

| 文件 | 来源 | 大小 | 是否进 git |
|------|------|------|----------|
| `data/alpaca_mini.jsonl` | 自 stanford-alpaca 抽样 200 条 | ~80KB | ✅ 是 |
| `data/dpo_pairs_mini.jsonl` | 自 UltraFeedback 抽样 100 对 | ~120KB | ✅ 是 |

两个数据集体积都很小，直接进 git，避免运行时下载失败。生成脚本保留在 `data/README.md` 里说明可复现，但用户不需要重跑。

---

## 4. 每个 demo 的实现规范

### 4.1 通用要求（沿用 Phase 1/2）

- 顶部 `╔══...══╗` ASCII box-art docstring（项目名 + 核心问题 + 与生产的关联）
- 中文注释 + 中文 print 输出
- `if __name__ == "__main__": main()` 入口
- `torch.manual_seed(42)` / `transformers.set_seed(42)` 保证可复现
- 首次运行自动检测设备（MPS / CUDA / CPU）并 print 当前后端
- 输出尾部「关键收获」3-5 条，呼应 docstring 核心问题
- 失败优雅：缺包 / 网络下载失败 / 显存不足 → 友好中文提示，不崩
- 每个 demo 独立可跑（除 06/07 标注硬件约束）
- 训练量小到 Mac MPS < 5 分钟可见 loss 下降；CPU < 15 分钟

### 4.2 各 demo 详细规范

#### 01_data_construction.py（~200 行）
**目标**：理解 SFT 数据从原始 → token 的全流程。
**步骤**：(1) 读 Alpaca raw → (2) 用 Qwen2.5 tokenizer 套 chat template → (3) 展示 `input_ids` / `labels` / `attention_mask` 三张量；(4) 对比 instruction 段被 mask 与不被 mask 的 loss 差异。
**关键收获**：为什么 SFT 只对 response token 算 loss。

#### 02_multi_turn_chat.py（~180 行）
**目标**：多 turn 对话的 mask 策略。
**步骤**：(1) 构造 user/assistant/user/assistant 4 turn → (2) 展示三种 loss mask 方案（仅最后 turn / 所有 assistant turn / 全部）→ (3) ASCII 可视化每个位置的 loss 权重。
**关键收获**：为什么 ShareGPT 风格训练能让模型学会"承上启下"。

#### 03_sft_full.py（~250 行）⚠️ 关键路径
**目标**：用 TRL `SFTTrainer` 在 Qwen2.5-0.5B 上跑全参 SFT。
**步骤**：(1) load base + tokenizer → (2) `SFTConfig`（batch=2, lr=2e-5, max_steps=50） → (3) 训练并 print loss 曲线 → (4) 训练前后同 prompt 生成对比。
**预算**：MPS ~3 分钟 / CPU ~10 分钟。
**关键收获**：全参 SFT 的内存/算力代价（监控 GPU/CPU 内存峰值并 print）。

#### 04_lora_from_scratch.py（~220 行）
**目标**：手写 LoRA 数学，理解 `W = W₀ + (α/r) BA`。
**步骤**：(1) `class LoRALinear(nn.Module)` 包装 `nn.Linear` → (2) freeze 原权重，只训 A、B → (3) 在一个 toy 矩阵回归任务上 vs 全参 baseline 对比参数量与收敛 → (4) ASCII 可视化"原矩阵 + 低秩补丁"。
**关键收获**：r=8 时只训原参数的 ~0.4% 也能收敛。

#### 05_lora_peft.py（~220 行）
**目标**：用 PEFT 库的工程化做法替代手写。
**步骤**：(1) `LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"])` → (2) `get_peft_model` → (3) `print_trainable_parameters()` → (4) 训练 + 保存 adapter → (5) 重新加载 + merge_and_unload。
**关键收获**：`target_modules` 选择对效果的影响（仅 attn vs attn+mlp）；adapter 文件 ~6MB vs 全参 ckpt ~1GB。

#### 06_qlora_mlx.py（~250 行）🍎 Mac 主路径
**目标**：用 MLX 原生 4-bit QLoRA。
**步骤**：(1) 用 `mlx_lm.convert` 将 Qwen2.5-0.5B 量化为 4bit MLX 格式 → (2) `mlx_lm.lora` 训练 LoRA adapter → (3) 量化前后内存占用对比。
**预算**：M1/M2 ~2 分钟。
**关键收获**：NF4 与 unified memory 在 Apple Silicon 上的优势。

#### 07_qlora_peft_bnb.py（~250 行）🐧 Linux/CUDA 路径
**目标**：HuggingFace 生态主流写法。
**步骤**：(1) `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` → (2) `prepare_model_for_kbit_training` → (3) PEFT LoRA → (4) 训练。
**Mac 行为**：检测到 MPS 时直接 print 友好提示并退出，引导用户去 06 或 Colab。
**关键收获**：PEFT + bnb + TRL 的标准组合是怎么协作的。

#### 08_dpo_alignment.py（~250 行）⚠️ 关键路径
**目标**：用 TRL `DPOTrainer` 做偏好对齐。
**步骤**：(1) load SFT 后 model（用 03 的产物或重新走一遍） → (2) `DPOConfig(beta=0.1)` → (3) 训 DPO ~50 steps → (4) 同 prompt 生成对比 SFT vs DPO 风格差异。
**预算**：MPS ~5 分钟 / CPU ~15 分钟。
**关键收获**：DPO 损失公式与 PPO 等价但无需 RM 训练。

#### 09_ppo_intro.py（~180 行）📖 教学型，不真训
**目标**：理解 RLHF 全景，知道为什么 DPO 取代了 PPO。
**步骤**：(1) ASCII 流程图：SFT model → RM model → PPO 三步 → (2) 伪代码（reward / advantage / clip ratio） → (3) 调用 `trl.PPOTrainer` 但只 print 一个 step 的中间张量形状不真训 → (4) 表格对比 PPO vs DPO 工程复杂度。
**关键收获**：RLHF 的工程债务在哪里。

#### 10_quantization_inference.py（~280 行）
**目标**：训练后量化的两条主流路径。
**步骤**：
1. **路径 A（首选）**：检测 `llama-quantize` 在 path → 真量化 base 为 GGUF Q4_K_M；不在 path → 提示 `brew install llama.cpp` 并跳过该步。
2. **推理兜底**：用 `llama-cpp-python` 加载预量化 GGUF（HuggingFace 上 `Qwen2.5-0.5B-Instruct-GGUF` 现成），保证零安装也能看到推理对比。
3. **路径 B**：`bnb` 8-bit 加载（Mac 不支持时友好降级到 fp16）。
4. **对比**：fp16 / GGUF Q4_K_M / bnb 8bit 的内存 + 延迟 + 输出质量三方对比表。

**关键收获**：量化在哪一步发生（训练后 vs 推理时）；NF4 / GPTQ / AWQ / GGUF 的差异。

#### 11_eval_perplexity.py（~180 行）
**目标**：最朴素的"模型变好了吗"评估。
**步骤**：(1) 在保留集（WikiText 子集）上算 base / SFT / DPO 三个模型的 perplexity → (2) 同一组 prompt 三个模型并排生成 → (3) 表格 + 人工评判提示。
**关键收获**：perplexity 与生成质量的相关性 / 不相关性。

#### 12_eval_lm_harness.py（~200 行）
**目标**：业界标准 benchmark 集成。
**步骤**：(1) 安装检查 `lm-evaluation-harness` → (2) 跑 `arc_easy` 一个 small subset（20 题）对 base / SFT 两个模型 → (3) 解析 JSON 结果并 print → (4) 链接到完整 leaderboard 工作流。
**预算**：Mac MPS ~3 分钟。
**关键收获**：benchmark 子集 vs 全集；常见污染问题。

### 4.3 失败模式与降级策略

| 场景 | 行为 |
|------|------|
| 模型首次下载失败 | print HF 镜像与代理设置说明，退出 |
| MPS OOM | 自动降级 batch_size=1，提示用户 |
| `bitsandbytes` 在 Mac 不可用（07/10） | print "请在 Colab/CUDA 环境运行" 并退出 |
| `mlx_lm` 未装（06） | print `pip install mlx-lm` 并退出 |
| `llama.cpp` 不在 path（10） | 跳过量化步骤，仅 print 命令；推理改用预量化 GGUF |
| `lm-eval` 未装（12） | print 安装命令并退出 |

---

## 5. KNOWLEDGE.md 大纲

`ml_foundations/post_training/KNOWLEDGE.md`，约 8000-10000 字，结构沿用 Phase 1/2 风格（理论推导 + 工程对应 + 类比 + ASCII 图）。

```
# 训练后期与对齐 — 完全手册

第 0 章 全景：从 base model 到 ChatGPT 之间发生了什么
  - Pretrain → SFT → RLHF → 量化部署 四阶段流水线
  - 每阶段的目标、数据形态、算力量级对照表
  - ASCII 图：参数 / 数据 / 算力的「漏斗」

第 1 章 SFT：让模型学会"听指令"
  1.1 数据形态：Alpaca / ShareGPT / chat template
  1.2 损失函数：为什么只对 response 算 loss（mask 推导）
  1.3 多 turn 对话的 mask 策略对比
  1.4 工程实践：TRL SFTTrainer 在做什么
  1.5 SFT 的失败模式（catastrophic forgetting / 模板过拟合）

第 2 章 PEFT 与 LoRA 数学
  2.1 为什么需要 PEFT：1.7T 模型的 fine-tune 成本
  2.2 LoRA 推导：W = W₀ + (α/r) BA，r 取多少够用
  2.3 LoRA 的等价视角：低秩 + 正则化
  2.4 target_modules 选择：q_proj / v_proj / mlp
  2.5 adapter merge / unmerge 与多 adapter 切换
  2.6 LoRA 变体扫盲：DoRA / VeRA / LoHa / AdaLoRA（仅介绍，不实现）

第 3 章 QLoRA：量化 + LoRA 的奇迹
  3.1 NF4 量化原理（4-bit Normal Float）
  3.2 Double Quantization 与 Paged Optimizer
  3.3 为什么 4-bit base + LoRA 不掉点
  3.4 实现路径：bitsandbytes (Linux/CUDA) vs Apple MLX (Mac)
  3.5 内存账本：13B 模型从 26GB → 6GB 的细节

第 4 章 偏好对齐：从 PPO 到 DPO
  4.1 RLHF 三阶段（SFT → RM → PPO）回顾
  4.2 PPO 损失：clip ratio / advantage / KL 惩罚
  4.3 PPO 的工程债务（4 模型副本 / reward hacking / 不稳定）
  4.4 DPO 推导：从 RL 形式到极大似然形式（含完整数学）
  4.5 DPO 损失公式与 β 超参的几何意义
  4.6 后 DPO 时代：ORPO / KTO / SimPO / IPO 一览

第 5 章 训练后量化（PTQ）
  5.1 量化基础：对称 / 非对称 / per-tensor / per-channel
  5.2 INT8 / INT4 的精度损失来源
  5.3 GPTQ：基于 Hessian 的逐层量化
  5.4 AWQ：activation-aware 量化
  5.5 GGUF / llama.cpp 量化家族（Q4_K_M / Q5_K_M / Q8_0）
  5.6 量化时机决策表：什么时候用什么

第 6 章 评估方法学
  6.1 Perplexity 的局限
  6.2 LLM-as-Judge（与第 9 章评估呼应）
  6.3 标准 benchmark：MMLU / ARC / HellaSwag / GSM8K
  6.4 lm-evaluation-harness 工程实践
  6.5 chatbot arena / MT-Bench：偏好评估
  6.6 benchmark 污染问题

第 7 章 选型决策手册
  7.1 数据量 vs 方法选择（200 / 2k / 20k / 200k 样本各选什么）
  7.2 硬件 vs 方法选择（Mac / 单卡 / 多卡 / 集群）
  7.3 方法对比矩阵（SFT / LoRA / QLoRA / DPO 的资源 × 效果）

附录 A：与 Phase 2 自训 GPT 的衔接
  - 为什么 Phase 3 不在 ~3M 自训 GPT 上做：参数量太小、chat template 缺失、SFT/DPO 数据稀疏下不收敛
  - 但 Phase 2 的 GPT 类 / attention 数学 / 采样策略，到 0.5B 模型上结构完全一致

附录 B：与上层 LangChain/RAG 章节的接口
  - 自己微调出的 adapter 怎么在 langchain 里加载（HuggingFacePipeline + PEFT）
  - 微调 vs RAG 的工程权衡（再次强调"先 RAG 后微调"原则）
```

### 5.1 写作风格沿用

- 每节先抛"为什么"问题，再讲"是什么 + 怎么做"
- 关键数学（LoRA、DPO、NF4）给完整公式 + 直观推导
- 每个理论概念给 1-2 个 ASCII 图（如「LoRA 矩阵分解」「PPO 4 模型协作图」「NF4 量化区间」）
- 每节末尾「与生产对应」: 这一节在 transformers / TRL / PEFT 哪个 API 对应
- 与 Phase 2 KNOWLEDGE.md 的引用钩子（如「attention 数学回看 Phase 2 第 2 章」）

---

## 6. VitePress 文档站集成

### 6.1 新增子章节目录

```
docs/ml-foundations/post-training/
├── index.md                  # 章节导航 + 学习地图（仿 transformer-training/index.md）
├── overview.md               # 全景：pretrain → SFT → RLHF → 量化（KNOWLEDGE 第 0 章）
├── sft.md                    # SFT 数据 / 损失 / 多 turn mask（第 1 章）
├── lora.md                   # PEFT 与 LoRA 数学 + 变体扫盲（第 2 章）
├── qlora.md                  # NF4 / Double Quant / MLX vs bnb（第 3 章）
├── dpo.md                    # PPO 回顾 + DPO 推导 + 后 DPO 变体（第 4 章）
├── quantization.md           # PTQ / GPTQ / AWQ / GGUF（第 5 章）
├── evaluation.md             # PPL / lm-eval-harness / Arena（第 6 章）
└── selection.md              # 选型决策手册（第 7 章）
```

9 个 md（1 个 index 导航 + 8 个内容章节），每个内容章节 1500-2000 字，与 KNOWLEDGE.md 的 0-7 共 8 章一一对应。

### 6.2 与 KNOWLEDGE.md 的关系

沿用 Phase 2 的双轨模式：
- **`KNOWLEDGE.md`**：源代码目录下的"教科书"，单文件长读体验。
- **`docs/ml-foundations/post-training/*.md`**：网站版，分页 + 侧边导航 + 跳转链。
- 两边内容**保持同步但不完全相同**：网站版每页头加 frontmatter，每页尾加「下一节」链接，KNOWLEDGE.md 加 TOC 锚点。

### 6.3 侧边栏插入位置

修改 `docs/.vitepress/config.*`（具体文件名实施时确认），在 ml-foundations 板块下：

```
零、ML 基础（Phase 1）
  - classical-ml
  - deep-learning
  - nlp-foundations

零.5、Transformer 训练实战（Phase 2）
  - transformer-training/...

零.6、训练后期与对齐（Phase 3） ← 新增
  - post-training/index
  - post-training/overview
  - post-training/sft
  - post-training/lora
  - post-training/qlora
  - post-training/dpo
  - post-training/quantization
  - post-training/evaluation
  - post-training/selection

一、LLM 基础（Project 0-1）
  ...
```

### 6.4 README + LEARNING_ROADMAP 更新

- `README.md`：在「0.5 Transformer Training from Scratch」下面新增「**0.6 Post-training & Alignment**」block，仿 Phase 2 写法（一段简介 + 模块表 + 时间预算 + 完成后能做什么）。
- `LEARNING_ROADMAP.md`：在 Phase 2 段后插入 Phase 3 段，含 12 个 demo 表 + 推荐顺序更新（Week 0.6）。
- `requirements.txt`：新增依赖

  ```
  transformers>=4.45.0
  peft>=0.13.0
  trl>=0.11.0
  datasets>=3.0.0
  accelerate>=1.0.0
  bitsandbytes>=0.43.0    # Mac 上装失败也不影响其他 demo
  llama-cpp-python>=0.3.0 # 10 demo 用，可选
  mlx-lm>=0.20.0          # 06 demo 用，仅 Mac
  lm-eval>=0.4.0          # 12 demo 用
  ```

  在 requirements.txt 里把 Mac-only / CUDA-only 的包注释清楚，并在 README 给出"按需安装"建议（避免新读者第一次 `pip install -r` 就在 bnb 上卡半天）。

### 6.5 .gitignore 追加

```
# Phase 3: Post-training artifacts
ml_foundations/post_training/runs/
ml_foundations/post_training/data/wikitext_cache/
ml_foundations/post_training/data/lm_eval_cache/
*.gguf
```

### 6.6 不改动的部分

- 不改 `config.py`（本 phase 不依赖 LLM API key，纯本地训练）
- 不改任何上层应用模块（LangChain / RAG / Agent 等）
- 不改 `pyproject.toml` 的 ruff/black/pytest 配置

---

## 7. 实施顺序与单次最终 commit

### 7.1 任务拓扑

```
A. 脚手架 + 数据                    → Task 1
B. 无 HF 依赖的基础 demo（可并行）    → Tasks 2, 3, 4
   - Task 2: 01_data_construction.py（仅需 tokenizer）
   - Task 3: 02_multi_turn_chat.py（仅需 tokenizer）
   - Task 4: 04_lora_from_scratch.py（纯 toy 矩阵任务，无需 HF model）
C. SFT 关键路径                     → Task 5（03_sft_full.py）
D. PEFT LoRA                       → Task 6（05_lora_peft.py）
E. QLoRA 双路径（独立可并行）         → Tasks 7, 8
   - Task 7: 06_qlora_mlx.py
   - Task 8: 07_qlora_peft_bnb.py
F. DPO + PPO 简介                  → Tasks 9, 10
   - Task 9: 08_dpo_alignment.py
   - Task 10: 09_ppo_intro.py
G. 量化推理                         → Task 11（10_quantization_inference.py）
H. 评估闭环                         → Tasks 12, 13
   - Task 12: 11_eval_perplexity.py
   - Task 13: 12_eval_lm_harness.py
I. KNOWLEDGE.md                    → Task 14
J. VitePress 9 个 md（可并行）       → Tasks 15-23
   - 15: index, 16: overview, 17: sft, 18: lora, 19: qlora,
     20: dpo, 21: quantization, 22: evaluation, 23: selection
K. 侧边栏 + README + roadmap + .gitignore + requirements → Task 24
L. Smoke-test 全部 demo            → Task 25
M. 最终单次 commit                  → Task 26
```

**并行点**：
- Tasks 2/3/4（无 HF 依赖 demo）可并行
- Tasks 7/8（QLoRA 双路径）可并行
- Tasks 9/10（DPO 与 PPO 简介）可并行
- Tasks 12/13（两个评估 demo）可并行
- Tasks 15-23（docs）全部可并行

**关键路径**：1 → 5（SFT） → 9（DPO） → 11（量化） → 12/13（评估） → 25（smoke test） → 26（commit）。

总计 **26 个任务**。

### 7.2 Smoke-test 矩阵（Task 25）

每个 demo 都要在提交前跑通至少一次，记录在最终 spec 末尾的"验证表"：

| Demo | Mac MPS | Mac CPU | Colab T4 | 期望产物 |
|------|---------|---------|----------|---------|
| 01 数据构造 | ✅ | ✅ | ✅ | tokenize 输出 + mask 可视化 |
| 02 多 turn | ✅ | ✅ | ✅ | mask 三方案对比 |
| 03 SFT | ✅ <5min | ✅ <15min | ✅ <2min | loss 下降 + 生成对比 |
| 04 LoRA 手写 | ✅ | ✅ | ✅ | 收敛曲线 |
| 05 LoRA PEFT | ✅ <5min | ✅ <15min | ✅ <2min | adapter ~6MB |
| 06 QLoRA MLX | ✅ <3min | ⚠️ 慢 | ❌ N/A | 4bit MLX 模型 |
| 07 QLoRA bnb | ❌ 提示退出 | ❌ 提示退出 | ✅ <2min | 4bit 训练 |
| 08 DPO | ✅ <5min | ✅ <15min | ✅ <2min | 风格对比 |
| 09 PPO 简介 | ✅ | ✅ | ✅ | 张量 shape + 流程图 |
| 10 量化推理 | ✅ | ✅ | ⚠️ MPS 不可用降级 | 三方案对比表 |
| 11 PPL 评估 | ✅ <3min | ✅ <10min | ✅ <1min | 三模型 PPL 表 |
| 12 lm-eval | ✅ <3min | ⚠️ 慢 | ✅ <2min | arc_easy 子集分数 |

实施期间会在每个 demo 完成后跑一次实测，把上表的"<X min"替换成真实数值。任何"❌ 提示退出"必须验证退出消息友好。

### 7.3 单次最终 commit 内容

最后一条 commit message（仿 Phase 2 风格 `feat(ml): add transformer_training module + docs`）：

```
feat(ml): add post-training module + docs

Phase 3 covering SFT / LoRA / QLoRA / DPO / quantization / eval on
Qwen2.5-0.5B. 12 demos all runnable on Mac (CPU/MPS), with bnb-CUDA
path documented for cloud. Adds ml_foundations/post_training/, docs
under docs/ml-foundations/post-training/, README + LEARNING_ROADMAP
updates, and requirements pin for transformers/peft/trl/mlx-lm.

Spec:  docs/superpowers/specs/2026-06-18-phase3-post-training-design.md
Plan:  docs/superpowers/plans/2026-06-18-phase3-post-training.md
```

不在最终 commit 里跑大型训练产物（已 .gitignore）。

### 7.4 不做的事

- ❌ 不写 unit tests（沿用 Phase 1/2）
- ❌ 不接 CI（项目本身没 CI workflow）
- ❌ 不做版本号 bump（pyproject.toml 维持原版本）
- ❌ 不动现有 21 个应用层模块的任何文件
- ❌ 不动 Phase 1/2 任何文件（除非 README 里有指向 Phase 3 的"下一步"链接需要更新）

---

## 8. 验证

成功标准：

1. **代码可运行**：12 个 demo 在指定硬件上按 §7.2 矩阵全部通过
2. **文档完整**：`KNOWLEDGE.md` 8000+ 字，docs 8 个内容 md 各 1500+ 字（外加 1 个 index 导航）
3. **侧边栏接入**：本地 `npm run docs:dev` 能看到「零.6、训练后期与对齐」节点，所有内部链接可点
4. **依赖清晰**：`pip install -r requirements.txt` 在 Mac 上不卡死（即使 bnb 装失败也只警告不阻塞）
5. **README 闭环**：仓库根 README 能读懂 Phase 3 的位置和价值，新读者能找到入口

### 8.1 后续衔接

完成本 phase 后，下一个自然 phase（独立 spec）选项：

- **Phase 4**：推理优化与部署（vLLM / TGI / continuous batching / PagedAttention 工程版）
- **Phase 5**：评估系统化（自建 LLM 评估基准、人工标注流程、A/B 测试基础设施）
- 或者完全转向 **Capstone 综合项目**：用 Phase 1-3 + 21 个应用层模块整合一个端到端产品

不在本 spec 范围内，留给未来决策。
