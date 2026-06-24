---
title: QLoRA：量化 + LoRA 的奇迹
---

# 第 3 章 QLoRA：量化 + LoRA 的奇迹

> 把 13B 模型的训练显存从 82GB 压到 11GB——在单张消费级 GPU 上微调 70B 成为可能

## 为什么需要 QLoRA

LoRA 把可训练参数压缩到 0.1%，但 base model 仍需以 fp16 加载进显存（7B ≈ 14GB）。对于 13B 或 70B 模型，单卡还是放不下。QLoRA（Dettmers et al., NeurIPS 2023）提出：**把 base model 量化到 4-bit 存储，同时在 LoRA adapter 上以 bf16 精度训练**。

核心创新有三：NF4 量化格式、双重量化（Double Quantization）、分页优化器（Paged Optimizer）。

## NF4 量化原理

📌 **NF4（Normal Float 4-bit）** 是 QLoRA 最重要的贡献之一。

普通 INT4 量化把数值范围均匀分成 16 个区间（类似等距刻度尺）。但神经网络权重的分布近似**标准正态分布 $\mathcal{N}(0, 1)$**，在 0 附近密集，在尾部稀疏。均匀量化会浪费大量精度在几乎没有数据的尾部区间。

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

## 双重量化与分页优化器

**Double Quantization（双重量化）**：每 64 个权重需要一个 fp32 缩放系数（32 bits），这本身也占内存。QLoRA 进一步把这些缩放系数再量化（256 个缩放系数一组，再量化为 8-bit），节省约 0.37 bits/parameter。

**Paged Optimizer（分页优化器）**：LoRA adapter 的 Adam 优化器状态（bf16）在显存不足时自动换页到 CPU RAM，训练过程中透明地在 GPU/CPU 间搬运，避免 OOM 崩溃。

## 为什么 4-bit base + LoRA 不掉点

直觉：LoRA adapter 以 bf16 精度训练，梯度计算完全准确。反向传播时，对 4-bit 权重的梯度计算通过先反量化（dequantize）到 bf16 来完成。效果接近 fp16 full fine-tuning 的三个原因：

1. **NF4 的量化误差极小**（相比 INT4 均匀量化，量化噪声降低约 20-30%）
2. **LoRA 适配层补偿了量化引入的偏差**——adapter 学习中和了一部分量化噪声
3. **预训练权重的大量冗余**使得 4-bit 压缩损失的信息量远小于参数量降低的比例

## 实现路径：bitsandbytes vs Apple MLX

| 路径 | 硬件 | 库 | 精度 | 速度 |
|------|------|-----|------|------|
| bitsandbytes | NVIDIA GPU (Linux/WSL2) | `bitsandbytes` | NF4 + bf16 LoRA | 最快 |
| Apple MLX | Apple Silicon (M1/M2/M3) | `mlx` / `mlx-lm` | 4-bit + bf16 LoRA | 慢于 A100，但无需 NVIDIA GPU |
| GPTQ + LoRA | NVIDIA GPU | `auto-gptq` + peft | INT4 | 中等 |

三选一的决策路径如下：

```
              ┌─ 有 NVIDIA GPU? ─┐
              │                  │
             是                  否
              │                  │
              ▼                  ▼
        bitsandbytes      ┌─ Mac Silicon? ─┐
        (CUDA)            │                │
                         是                否
                          │                │
                          ▼                ▼
                      Apple MLX       Colab T4
                      (Metal)         (回到 bnb)
```

注意：MLX 只能用在 Apple Silicon 机器上（Metal 后端），bitsandbytes 只能用在 CUDA 环境中，两者不可互换。

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

## 内存账本：13B 模型从 82GB → 11GB

| 组件 | fp16 全量微调 | NF4 QLoRA |
|------|-------------|-----------|
| 模型权重 | 26 GB | 6.5 GB |
| LoRA adapter (r=16) | — | ~0.1 GB |
| 优化器状态 (bf16 Adam) | 52 GB | 0.4 GB（仅 adapter） |
| 激活值 (batch=4) | ~4 GB | ~4 GB |
| **合计** | **~82 GB（需 2× A100）** | **~11 GB（单张 A100/3090）** |

::: info Mac 用户提示
Apple MLX 路径（`06_qlora_mlx.py`）在 M2/M3 Mac 上可以直接运行，不需要 NVIDIA GPU。统一内存架构使得 GPU 和 CPU 共享同一块内存，16GB 统一内存的 Mac 可以微调 0.5B–1.5B 模型。
:::

## 与生产对应

`transformers.BitsAndBytesConfig` + `peft.LoraConfig`；Apple MLX 路径见 `mlx-lm` 库；合并后推理与普通模型无区别（`PeftModel.merge_and_unload()`）。

::: info 关联 demo
- [`06_qlora_mlx.py`](../../ml_foundations/post_training/06_qlora_mlx.py)：Apple MLX 上的 QLoRA（Mac 本机）
- [`07_qlora_peft_bnb.py`](../../ml_foundations/post_training/07_qlora_peft_bnb.py)：bitsandbytes NF4 量化 + LoRA（Linux/CUDA）
:::

---

::: tip 下一节
→ [DPO：偏好对齐](./dpo)
:::

::: info 上一节
← [LoRA：参数高效微调数学](./lora)
:::
