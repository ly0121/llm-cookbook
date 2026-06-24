# 零.6、训练后期与对齐

> 从 base model 到 ChatGPT —— 把 SFT / LoRA / QLoRA / DPO / 量化 / 评估完整串起来

---

## 为什么需要这一章？

完成 Phase 2 后，你亲手训了一个能写"伪莎士比亚"的 ~3M 参数小 GPT。但现实中的 ChatGPT、Qwen、LLaMA-Instruct 并不是直接预训练出来的——预训练只给模型装进了"世界知识"，却没有教它"如何帮助人"。

从 base model 到可用的助手，需要一套完整的**训练后期（Post-training）** 工程：指令微调让模型学会对话格式，LoRA 把显存需求压到单卡可用，DPO 注入人类偏好，量化把 26GB 的 fp16 模型压缩到 6GB 可在消费级硬件运行。本章把这四个阶段逐层拆解，12 个可运行 demo 覆盖从数据构造到 benchmark 评估的完整流程。

## 学习地图

| 序号 | Demo 文件 | 对应理论章节 |
|------|-----------|------------|
| 01 | `01_data_construction.py` | 第 1 章 SFT — 数据格式 |
| 02 | `02_multi_turn_chat.py` | 第 1 章 SFT — 多轮 mask |
| 03 | `03_sft_full.py` | 第 1 章 SFT — 全量微调 |
| 04 | `04_lora_from_scratch.py` | 第 2 章 LoRA — 手写推导 |
| 05 | `05_lora_peft.py` | 第 2 章 LoRA — PEFT 库调用 |
| 06 | `06_qlora_mlx.py` | 第 3 章 QLoRA — Apple MLX |
| 07 | `07_qlora_peft_bnb.py` | 第 3 章 QLoRA — bitsandbytes |
| 08 | `08_dpo_alignment.py` | 第 4 章 DPO — 偏好对齐 |
| 09 | `09_ppo_intro.py` | 第 4 章 DPO — PPO 对比 |
| 10 | `10_quantization_inference.py` | 第 5 章 PTQ — 量化推理 |
| 11 | `11_eval_perplexity.py` | 第 6 章 评估 — PPL |
| 12 | `12_eval_lm_harness.py` | 第 6 章 评估 — Benchmark |

## 硬件预算

| Demo | 说明 | Mac MPS | CPU Only | Colab T4 |
|------|------|---------|----------|----------|
| 01–02 | 数据构造 / mask 演示 | < 10s | < 30s | < 10s |
| 03 | Full SFT（Qwen-0.5B） | ~10 min | 不推荐 | ~5 min |
| 04–05 | LoRA 手写 + PEFT | ~5 min | ~15 min | ~3 min |
| 06 | QLoRA MLX（仅 Mac） | ~8 min | N/A | N/A |
| 07 | QLoRA bitsandbytes | N/A | N/A | ~10 min |
| 08–09 | DPO + PPO 示意 | ~10 min | 不推荐 | ~8 min |
| 10 | 量化推理对比 | ~5 min | ~10 min | ~5 min |
| 11 | PPL 计算 | ~3 min | ~8 min | ~3 min |
| 12 | lm-eval-harness | ~20 min | 不推荐 | ~15 min |

## 推荐学习顺序

**第一步**：阅读 [全景章节](./overview)，理解四阶段漏斗——知道 SFT / RLHF / DPO / 量化各自解决什么问题。

**第二步**：运行 demo 01–03，亲手看数据格式与 label mask 的实现，感受全量 SFT 的显存开销。

**第三步**：阅读 [LoRA](./lora) 和 [QLoRA](./qlora) 章节，再运行 demo 04–07，确认 $W = W_0 + \frac{\alpha}{r}BA$ 的公式在代码中的落地。

**第四步**：阅读 [DPO 章节](./dpo)，运行 demo 08–09，对比 PPO 的工程复杂度与 DPO 的简洁。

**第五步**：阅读 [量化](./quantization) 和 [评估](./evaluation) 章节，运行 demo 10–12，拿到自己模型的 benchmark 数字。最后用 [选型手册](./selection) 对照自己的硬件和数据规模，确定最适合的方法。

## 本章包含

- **[全景：从 base model 到 ChatGPT](./overview)** — 四阶段流水线漏斗，每阶段数据量与算力量级对照
- **[SFT：让模型学会听指令](./sft)** — chat template、label mask、TRL SFTTrainer 工程实践
- **[LoRA：参数高效微调数学](./lora)** — 低秩分解推导、LoraConfig 配置、变体扫盲
- **[QLoRA：量化 + LoRA 的奇迹](./qlora)** — NF4 分位量化、双重量化、bitsandbytes vs MLX
- **[DPO：偏好对齐](./dpo)** — PPO 四模型架构、DPO 完整推导、后 DPO 变体一览
- **[PTQ：训练后量化](./quantization)** — GPTQ / AWQ / GGUF 对比、量化时机决策表
- **[评估方法学](./evaluation)** — PPL 局限、标准 benchmark 家族、lm-evaluation-harness 实践
- **[选型决策手册](./selection)** — 数据量决策表、硬件×方法矩阵、经验法则总结

---

::: tip 下一节
→ [全景：从 base model 到 ChatGPT](./overview)
:::
