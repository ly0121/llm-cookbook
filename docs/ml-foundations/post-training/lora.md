---
title: LoRA：参数高效微调数学
---

# 第 2 章 PEFT 与 LoRA 数学

> 冻结 99.9% 的权重，只训练两个小矩阵——显存需求从 80GB 降到 18GB

## 为什么需要 PEFT

假设要微调一个 7B 参数的模型：

- 模型本身：7B × 2 bytes (fp16) ≈ **14 GB**
- 优化器状态（Adam）：7B × 2（momentum + variance）× 4 bytes ≈ **56 GB**
- 梯度：7B × 2 bytes ≈ **14 GB**
- 激活值（batch size 相关）：**数 GB**
- **合计：约 80-100 GB**，需要 2-3 张 A100（80GB）

这对绝大多数研究者不可接受。参数高效微调（**PEFT，Parameter-Efficient Fine-Tuning**）的核心思路：**冻结预训练权重，只训练少量新增参数**。

📌 **LoRA（Low-Rank Adaptation）** 是目前最主流的 PEFT 方法（Hu et al., ICLR 2022）。

## LoRA 推导：从观察到公式

**关键观察**：神经网络在适应新任务时，权重的更新矩阵 $\Delta W$ 具有**低内在秩**。即：完整微调时 $W \leftarrow W_0 + \Delta W$，而 $\Delta W$ 的有效秩远小于 $d$。

如果 $\text{rank}(\Delta W) = r \ll d$，则 $\Delta W$ 可分解为两个低秩矩阵的乘积：

$$\Delta W = BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times d}$，$r \ll d$。

加上缩放系数 $\alpha/r$，📌 **LoRA 的完整公式**为：

$$W = W_0 + \frac{\alpha}{r} BA$$

- $W_0$：冻结的预训练权重，不更新
- $B, A$：可训练的低秩矩阵
- $r$：秩（rank），通常取 4–64
- $\alpha$：缩放超参，通常设为 $r$ 的 1–2 倍

**初始化策略**：$A$ 用随机高斯初始化，$B$ 用零初始化。这保证训练开始时 $\Delta W = B \cdot A = 0$，模型行为与原始 $W_0$ 完全相同。

## LoRA 矩阵分解图

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

## r 取多少够用？

| 任务类型 | 推荐 r | 说明 |
|---------|--------|------|
| 对话/指令跟随 | 4–16 | 行为调整，秩需求低 |
| 代码生成 | 16–64 | 需要更多新知识迁移 |
| 特定领域知识注入 | 32–128 | 知识差距大时需要更高秩 |
| 完整任务迁移 | 64–256 | 接近全量微调效果 |

经验规律：r=16 在大多数对话任务上效果与 r=64 相差不超过 1-2%，但参数量节省 4 倍。

## target_modules 选择

不是所有矩阵都值得加 LoRA：

| 模块 | 是否加 LoRA | 原因 |
|------|-----------|------|
| `q_proj` / `k_proj` / `v_proj` | 必加 | Attention 是行为调整的核心 |
| `o_proj` | 推荐 | 输出投影同样重要 |
| `gate_proj` / `up_proj` / `down_proj` | 可选 | FFN 层；知识密集任务加 |
| `embed_tokens` / `lm_head` | 通常不加 | embedding 层秩扰动可能破坏 tokenizer 对齐 |

```python
peft.LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
```

## Adapter Merge 与多 Adapter 切换

训练完成后，可以把 LoRA 权重合并回基础模型：

$$W_\text{merged} = W_0 + \frac{\alpha}{r} BA$$

合并后模型与普通模型无区别（推理速度不变），但无法再 unmerge。实际生产中常用**多 adapter 切换**：一个 base model + N 个任务 adapter，按请求类型动态加载，显著降低多任务部署成本。

```python
model.merge_adapter()    # 合并 LoRA 到 W₀
model.unmerge_adapter()  # 分离（仅在未 merge 状态有效）
```

## LoRA 变体扫盲

📌 **DoRA（Weight-Decomposed LoRA）**：把权重分解为大小（magnitude）和方向（direction）分量，分别做 LoRA 适配，在某些任务上比 LoRA 高 1-2 个点。

📌 **VeRA（Vector-based RA）**：所有层共享同一对随机矩阵 AB，只训练逐层缩放向量，参数量比 LoRA 再低约 10 倍。

📌 **AdaLoRA**：自适应地为不同层分配不同秩预算，根据奇异值大小动态剪枝，适合计算预算严格的场景。

📌 **LoHa / LoKr**：用 Hadamard 积 / Kronecker 积替代矩阵乘法，适合图像生成微调（Stable Diffusion LoRA）。

::: tip LoRA 的正则化视角
LoRA 可以视为在 $\Delta W$ 空间上加了一个**秩约束正则化**：强迫权重更新沿低秩流形移动，天然避免了过拟合。全量微调时的灾难性遗忘部分来自高秩扰动，LoRA 的低秩约束天然缓解了这个问题。
:::

## 与生产对应

`peft.LoraConfig` / `peft.get_peft_model()`；合并操作 `PeftModel.merge_and_unload()`；adapter 保存与加载 `model.save_pretrained()` / `PeftModel.from_pretrained()`。

::: info 关联 demo
- [`04_lora_from_scratch.py`](../../ml_foundations/post_training/04_lora_from_scratch.py)：手写 LoRA 层，验证 $W_0 + BA$ 的数学
- [`05_lora_peft.py`](../../ml_foundations/post_training/05_lora_peft.py)：PEFT 库调用；adapter 保存与加载
:::

---

::: tip 下一节
→ [QLoRA：量化 + LoRA 的奇迹](./qlora)
:::

::: info 上一节
← [SFT：让模型学会听指令](./sft)
:::
