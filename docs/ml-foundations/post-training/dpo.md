---
title: DPO：偏好对齐
---

# 第 4 章 偏好对齐：从 PPO 到 DPO

> SFT 让模型"会说话"，DPO 让模型"说好话"——无需显式奖励模型，直接从偏好数据做优化

## 为什么 SFT 不够

SFT 后的模型已经能"听指令"，但还不能保证：
- 拒绝有害请求（不泄露危险信息）
- 输出诚实、有帮助、无害（HHH 原则）
- 在多个"都对"的回答中选择更好的那个

这需要引入**人类偏好信号**。RLHF（Reinforcement Learning from Human Feedback）是 InstructGPT 和 ChatGPT 的关键技术。

## PPO 的四模型协作

RLHF 的完整 PPO 流程需要四个模型同时运作：

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

## PPO 的工程债务

PPO 理论优雅，但工程上代价高昂：

1. **4 个模型同时在 GPU**：actor、reference、reward、critic，显存是 SFT 的 4 倍
2. **Reward Hacking**：actor 学会欺骗 reward model（说很多废话但 RM 打高分）
3. **训练不稳定**：clip ratio、KL 系数、advantage 归一化方式的超参极其敏感
4. **工程复杂度**：需要特殊的 rollout 生成 pipeline + 异步 RM 查询

这直接催生了 DPO 的出现。

## DPO 推导：5 步从 RL 到极大似然

DPO（Rafailov et al., NeurIPS 2023）的核心洞察：**能否绕过显式奖励模型和 PPO，直接从偏好数据做优化？**

以下是推导的关键步骤（详细推导见源码目录 `ml_foundations/post_training/KNOWLEDGE.md` 第 4 章）：

**Step 1**：Bradley-Terry 偏好模型建模人类偏好：给定 prompt $x$ 和两个回答 $y_w$（preferred）、$y_l$（rejected），

$$p(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$

**Step 2**：RLHF 的 RL 目标——最大化期望奖励同时约束策略不偏离 reference 太远：

$$\max_{\pi_\theta} \mathbb{E}\left[r(x, y)\right] - \beta \cdot \text{KL}(\pi_\theta(y|x) \| \pi_\text{ref}(y|x))$$

**Step 3**：上述 KL 约束 RL 目标的闭式最优策略为：

$$\pi_r(y \mid x) = \frac{1}{Z(x)} \pi_\text{ref}(y \mid x) \exp\!\left(\frac{r(x, y)}{\beta}\right)$$

**Step 4（关键）**：反解奖励函数——把未知的 $r(x,y)$ 用策略 $\pi_r$ 本身重新参数化：

$$r(x, y) = \beta \log \frac{\pi_r(y \mid x)}{\pi_\text{ref}(y \mid x)} + \beta \log Z(x)$$

**Step 5**：代入 Bradley-Terry，$\beta \log Z(x)$ 项在相减时消掉，最终对偏好数据集做最大似然估计，得到 📌 **DPO loss**：

$$
\boxed{
\mathcal{L}_\text{DPO} = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log \sigma\!\left(\beta \left[\log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right]\right)\right]
}
$$

## DPO Loss 的几何意义

DPO loss 实质上在做：**拉大 preferred 回答的隐含奖励，压低 rejected 回答的隐含奖励**，且整个过程只需要两个模型（policy model + reference model），不需要显式的奖励模型。

$\beta$（温度系数）的几何意义：
- **$\beta$ 越小**：策略可以大幅偏离 reference，偏好信号被强化（但容易丢失多样性）
- **$\beta$ 越大**：策略被约束在 reference 附近，保守但稳定
- **典型取值**：0.01–0.5；对话任务常用 0.1–0.2

## 后 DPO 时代：变体一览

| 方法 | 核心创新 | 优势 |
|------|---------|------|
| **ORPO** | 在 SFT loss 中直接加 odds ratio 偏好项，无需 reference model | 单阶段训练，省 reference model |
| **KTO** | 只需单条回答的好/坏标签，不需要 paired 数据 | 数据收集更容易 |
| **SimPO** | 用归一化对数似然代替 log ratio，无需 reference model | 更简洁稳定 |
| **IPO** | 直接优化偏好概率，避免 Bradley-Terry 假设的缺陷 | 理论更严密 |

::: tip DPO vs PPO 选择
- 有足够工程资源（多机多卡、专门的 RL 基础设施）→ PPO（效果上限更高）
- 单卡 / 小团队 / 追求快速迭代 → DPO（等同 SFT 的工程复杂度，效果接近 PPO）
:::

## 与生产对应

`trl.DPOTrainer`（支持 DPO / IPO / ORPO / KTO，通过 `loss_type` 参数切换）；PPO 见 `trl.PPOTrainer`；偏好数据格式参考 `Anthropic/hh-rlhf` 数据集（chosen / rejected 字段）。

::: info 关联 demo
- [`08_dpo_alignment.py`](../../ml_foundations/post_training/08_dpo_alignment.py)：DPO 偏好对齐；直接看 loss 公式落地
- [`09_ppo_intro.py`](../../ml_foundations/post_training/09_ppo_intro.py)：PPO 四模型协作示意；理解工程债务
:::

---

::: tip 下一节
→ [PTQ：训练后量化](./quantization)
:::

::: info 上一节
← [QLoRA：量化 + LoRA 的奇迹](./qlora)
:::
