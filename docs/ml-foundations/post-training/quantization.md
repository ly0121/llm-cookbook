---
title: PTQ：训练后量化
---

# 第 5 章 训练后量化（PTQ）

> 不训练一步，把 26GB 的 fp16 模型压缩到 6GB——在消费级硬件上高速推理

## 为什么量化

对齐好的模型要真正在生产环境服务，还面临推理成本问题：fp16 的 7B 模型需要 14GB 显存，每次前向推理的浮点计算量也很大。量化把权重（和/或激活值）从 16/32-bit 压缩到 8/4-bit，显存降低 2-4 倍，推理速度提升 1.5-4 倍。

PTQ（Post-Training Quantization，训练后量化）无需任何梯度更新，是部署优先场景的首选。

## 量化基础

📌 **对称量化**（symmetric）：量化区间关于 0 对称，用 $[-\text{absmax}, +\text{absmax}]$ 均匀分成 $2^b$ 份：

$$
x_q = \text{round}\!\left(\frac{x}{\text{scale}}\right), \quad \text{scale} = \frac{\text{absmax}}{2^{b-1}-1}
$$

📌 **非对称量化**（asymmetric）：用 $[\min, \max]$ 作为量化范围，额外存储 zero-point 参数。适合激活值（ReLU 后的激活值全为正）。

**Per-tensor vs Per-channel**：
- **Per-tensor**：整层权重共享一个 scale，速度快，精度低
- **Per-channel（per-row）**：每行/列独立 scale，精度高（AWQ/GPTQ 默认），稍慢

**异常值（Outliers）** 是量化精度下降的主要来源。LLM 激活值中存在少数绝对值极大的维度（比均值大 100 倍），它们迫使 scale 变大，导致正常值量化粒度变粗。

## 主流 PTQ 算法对比

| 算法 | 核心思路 | 需要校准数据 | 量化速度 | 精度（INT4）|
|------|---------|-----------|---------|------------|
| **GPTQ** | 逐层量化 + Hessian 误差补偿 | 需要（~128 样本） | 1-4 GPU 小时 | 接近 fp16（< 2% 差距）|
| **AWQ** | 保护重要权重（~1%），缩放后量化 | 需要激活统计 | 更快 | 优于 GPTQ |
| **bitsandbytes** | 逐行动态量化，无需校准 | 不需要 | 即时 | 略低于 GPTQ |
| **GGUF（llama.cpp）** | 混合精度 k-quant，CPU 友好 | 不需要 | 即时 | 取决于档位 |

**GPTQ** 的核心思路：逐层量化，量化完一层后用该层的 Hessian 矩阵（二阶导数）来补偿误差，调整剩余未量化权重以最小化量化后的层输出误差。

**AWQ** 的观察：权重中只有约 1% 的权重对激活值的影响最大（"重要权重"），对这些权重做高精度量化，其余粗糙量化，整体精度超过 GPTQ。AWQ 推理速度比 GPTQ 更快（硬件友好的权重分布）。

## GGUF / llama.cpp 量化家族

### 量化发生在训练管线的哪个阶段

下图说明三种量化策略发生在管线的不同阶段：

```
  时间轴 →

  ┌─────────┐   ┌─────┐   ┌──────┐   ┌─────┐   ┌────────┐
  │ Pretrain│ → │ SFT │ → │ RLHF │ → │ 评估 │ → │ 部署   │
  └─────────┘   └─────┘   └──────┘   └─────┘   └────────┘
      ▲           ▲                                 ▲
      │           │                                 │
     QAT       QLoRA                              PTQ
   (训练中    (训练中     ←─ 训练阶段插入量化 ─┘
    量化感知)  4-bit base
              + LoRA)
                                                 │
                                                 └ 训练完成后
                                                   把权重压成
                                                   int4/int8/GGUF
```

本章 demo 走 PTQ 路线，因为它最简单且零训练成本；QLoRA 见上一节。

`llama.cpp` 使用 GGUF 格式，提供多个精度档位：

| 量化类型 | bits/weight | 7B 显存 | 质量 |
|---------|------------|---------|------|
| Q4_K_M | 4.5 | 4.8 GB | 推荐（精度/显存最佳权衡）|
| Q5_K_M | 5.5 | 5.7 GB | 高质量 |
| Q6_K | 6.6 | 6.6 GB | 接近 fp16 |
| Q8_0 | 8.0 | 7.7 GB | 几乎无损 |
| Q2_K | 2.6 | 2.9 GB | 质量较差，仅显存极限时用 |

`K` 后缀表示 k-quant（混合精度，重要层用更高精度）；`M` 表示 medium 配置。

## 量化时机决策表

| 场景 | 推荐方案 | 工具 |
|------|---------|------|
| Mac 本机推理 | Q4_K_M / Q5_K_M GGUF | llama.cpp / Ollama |
| 单卡 CUDA 推理 | NF4 bitsandbytes | transformers BitsAndBytesConfig |
| 生产 GPU 服务 | GPTQ INT4 / AWQ INT4 | vLLM + AutoGPTQ |
| 追求最佳质量 | Q8_0 或 fp16 | 根据显存选择 |
| 微调后再量化 | AWQ（需校准数据）| autoawq |

::: warning INT4 vs INT8 的精度权衡
INT8 量化（256 个等级）：精度损失通常 < 1%，可直接用于生产。INT4 量化（16 个等级）：精度损失 1-5%，需要 GPTQ/AWQ/NF4 等高级算法来补偿。如果显存允许，优先选 INT8 或 Q6_K。
:::

## 与生产对应

`bitsandbytes.BitsAndBytesConfig`（即时 NF4 量化，无需预处理）；`auto_gptq.AutoGPTQForCausalLM`（GPTQ 量化与推理）；`awq.AutoAWQForCausalLM`（AWQ 量化）；CPU/Mac 推理用 `llama-cpp-python`；生产服务框架 `vLLM` 原生支持 GPTQ / AWQ 推理。

::: info 关联 demo
- [`10_quantization_inference.py`](../../ml_foundations/post_training/10_quantization_inference.py)：PTQ 量化推理；INT4/INT8 精度对比
:::

---

::: tip 下一节
→ [评估方法学](./evaluation)
:::

::: info 上一节
← [DPO：偏好对齐](./dpo)
:::
