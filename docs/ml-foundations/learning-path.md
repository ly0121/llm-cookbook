# 进阶学习路径

> 从 ML 基础到 LLM 微调的完整路线图

---

## 一、整体路线图

```
                    ┌────────────────────────────┐
                    │   零、ML 基础（本章）        │
                    │   2-3 周，CPU 即可跑        │
                    └────────────────────────────┘
                                  ↓
                    ┌────────────────────────────┐
                    │   一、LLM 基础              │
                    │   Transformer / 推理 / 训练 │
                    └────────────────────────────┘
                                  ↓
                    ┌────────────────────────────┐
                    │   二、LLM 应用（项目主线）   │
                    │   LangChain / RAG / Agent  │
                    └────────────────────────────┘
                                  ↓
                    ┌────────────────────────────┐
                    │   三、LLM 微调（进阶）      │
                    │   SFT / LoRA / DPO / RLHF  │
                    └────────────────────────────┘
                                  ↓
                    ┌────────────────────────────┐
                    │   四、LLM 工程化（生产）     │
                    │   推理优化 / 评测 / 监控    │
                    └────────────────────────────┘
```

---

## 二、本章学完后的下一步

### 2.1 立刻进入 LLM 主线

按以下顺序阅读项目核心文档：

1. **[LLM 基础](/llm/)** —— 总览
2. **[Transformer 架构](/llm-knowledge/transformer-architecture)** —— 看懂"All you need is attention"
3. **[Tokenization](/llm-knowledge/tokenization)** —— 字符串如何变成 token
4. **[训练流程](/llm-knowledge/training)** —— pretrain / SFT / RLHF
5. **[推理与采样](/llm-knowledge/inference)** —— greedy / top-k / temperature
6. **[Scaling Laws](/llm-knowledge/scaling)** —— 大模型为什么"涌现"

### 2.2 实战项目（按难度递增）

| 阶段 | 项目 | 学到什么 | 时间 |
|------|------|---------|------|
| 入门 | [01-langchain-basics](/01-langchain-basics) | LLM API 调用、prompt | 1-2 天 |
| 入门 | [02-prompt-engineering](/02-prompt-engineering) | few-shot、CoT、ReAct | 2-3 天 |
| 进阶 | [04-rag-system](/04-rag-system) | embedding、向量库、检索 | 3-5 天 |
| 进阶 | [09-multi-agent-collab](/09-multi-agent-collab) | LangGraph、Agent | 5-7 天 |
| 高阶 | [13-evaluation-platform](/13-evaluation-platform) | 评测、指标设计 | 5-7 天 |
| 高阶 | [15-stock-trading-system](/15-stock-trading-system) | 端到端复杂系统 | 7-14 天 |

---

## 三、LLM 微调路线（进阶专题）

::: warning 前置要求
- ✅ 完成本章 ML 基础
- ✅ 看懂 Transformer 架构
- ✅ 至少有 1 张能用的 GPU（建议 24GB+，如 RTX 3090/4090 或租 A100）
:::

### 3.1 阶段 A：SFT（监督微调）

**核心问题**：让 LLM 模仿"人类示范"。

**关键概念**：
- Instruction tuning（指令微调）
- Chat template（对话模板：ChatML / Llama-3 / Qwen）
- 数据格式：`{"messages": [{"role": "user", "content": ...}, ...]}`
- Loss masking（只对 assistant 部分算 loss）

**推荐工具**：
- **Hugging Face TRL**：`SFTTrainer`，最易用
- **LLaMA-Factory**：中文社区主流，支持几乎所有开源模型
- **Axolotl**：YAML 配置驱动，适合实验

**入门项目**：
```
1. 准备 200-500 条高质量指令数据（领域专属）
2. 用 LLaMA-Factory 微调 Qwen-7B / Llama-3-8B
3. 对比微调前后在领域任务上的表现
```

### 3.2 阶段 B：参数高效微调（PEFT）

**核心问题**：全参微调太贵（70B 模型要 1000+ GB 显存），怎么省钱？

**关键技术**：

#### LoRA（Low-Rank Adaptation）

```
原权重 W (d×d) 不动
↓
旁路加 ΔW = AB^T，A ∈ R^(d×r)，B ∈ R^(r×d)
↓
只训 A 和 B（r=8 时仅原参数 0.06%）
```

::: tip 与 ML 基础的连接
**LoRA = "低秩约束" 这一正则化思想的应用**：
- L1/L2 正则限制参数**大小**
- LoRA 限制参数**变化方向**（必须在低秩子空间）

学完本章"过拟合"一节，再看 LoRA 就豁然开朗。
:::

#### QLoRA（量化 + LoRA）

把基础模型量化到 4-bit（NF4），LoRA 部分保持 16-bit。
→ 13B 模型可在 12GB 显存上微调。

#### 其他 PEFT 方法

| 方法 | 思想 | 何时用 |
|------|------|--------|
| **LoRA / QLoRA** | 低秩增量 | 默认首选 |
| **Prefix Tuning** | 在每层加可训练前缀 | 早期方法，已少用 |
| **P-Tuning v2** | 改进的 prompt tuning | 中文场景偶尔用 |
| **DoRA** | 分解 magnitude + direction | LoRA 的进阶版 |
| **Adapter** | 在 FFN 中插入小模块 | 历史悠久，效果稍逊 LoRA |

### 3.3 阶段 C：偏好优化（DPO / RLHF）

**核心问题**：SFT 后的模型可能"礼貌但无用"或"有用但有害"，如何对齐人类偏好？

#### RLHF（PPO）—— 复杂但强大

```
1. 训练 reward model（基于人类偏好对）
2. 用 PPO 算法优化 LLM 的输出，最大化 reward
3. KL 约束防止偏离 SFT 模型太远
```

复杂度高，需要工程经验。OpenAI / Anthropic 主力路线。

#### DPO（Direct Preference Optimization）—— 简化版

把 RLHF 数学上等价转化为一个**简单的二分类损失**：

$$
L_{DPO} = -\log\sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)
$$

- $y_w$：用户偏好的回答
- $y_l$：用户不喜欢的回答
- 不需要 reward model，不需要 PPO，几行代码就能跑

**已成为开源社区主流**：Llama-3、Qwen-2 系列都用 DPO。

::: tip 与 ML 基础的连接
**DPO 损失就是个二元逻辑回归**：把"偏好对"作为正负样本，让模型对"好回答"打高分。
学完本章"逻辑回归"，DPO 数学一看就懂。
:::

#### 其他偏好优化方法

| 方法 | 特点 |
|------|------|
| **PPO** | 强大但复杂；OpenAI 主力 |
| **DPO** | 简单直接；社区主流 |
| **KTO** | 不需偏好对，单点反馈即可 |
| **IPO / SimPO** | DPO 改进版，理论更严谨 |
| **GRPO** | DeepSeek 提出，PPO 简化版 |

---

## 四、推理优化（部署进阶）

训练完模型后，如何让它**跑得快、占得少**？

### 4.1 量化（Quantization）

| 精度 | 显存占用 | 速度 | 精度损失 |
|------|---------|------|---------|
| FP16 / BF16 | 1× | 1× | 几乎无损 |
| INT8 | 0.5× | 1.5× | < 1% |
| INT4 (NF4) | 0.25× | 2× | 1-3% |
| INT2 | 0.125× | 3× | 严重损失 |

**主流工具**：
- **GPTQ**：训练后量化，离线
- **AWQ**：激活感知量化，效果更好
- **GGUF (llama.cpp)**：CPU + 消费级 GPU 友好

### 4.2 推理框架

| 框架 | 特点 | 适合场景 |
|------|------|---------|
| **vLLM** | PagedAttention，吞吐之王 | 服务端高并发 |
| **TGI** | HuggingFace 官方 | 与 HF 生态集成 |
| **llama.cpp** | C++，CPU/Metal/Vulkan | Mac / 个人设备 |
| **Ollama** | llama.cpp 套壳 | 个人体验最佳 |
| **TensorRT-LLM** | NVIDIA 极致优化 | NVIDIA GPU 生产环境 |
| **MLX** | Apple Silicon | Mac M 芯片 |

### 4.3 关键优化技术

- **KV Cache**：缓存历史 K/V，避免重算
- **PagedAttention**：把 KV cache 分页管理（vLLM 核心）
- **Continuous Batching**：动态拼 batch，吞吐 ↑
- **Flash Attention**：分块计算 attention，显存 ↓
- **Speculative Decoding**：小模型猜大模型 → 加速 2-3×

---

## 五、评测与监控

### 5.1 通用基准

| Benchmark | 测试什么 |
|-----------|---------|
| **MMLU** | 通识知识（57 个学科） |
| **HumanEval** | 代码生成 |
| **GSM8K / MATH** | 数学推理 |
| **BBH** | 复杂推理 |
| **MT-Bench / Arena** | 对话质量（用 GPT-4 / 人类评） |
| **C-Eval / CMMLU** | 中文知识 |

### 5.2 业务评测（更重要）

通用 benchmark 不代表你的业务场景。一定要：
1. **构建领域评测集**：100-500 条人工标注样本
2. **设计业务指标**：客服满意度、代码可运行率、检索准确率
3. **A/B 测试**：用真实流量验证

### 5.3 在线监控

- **质量监控**：延迟、错误率、用户反馈
- **成本监控**：token 用量、API 费用
- **安全监控**：有害内容、prompt injection、越狱
- **漂移监控**：输入分布变化、效果衰减

---

## 六、推荐学习资源

### 6.1 理论书籍

| 书 | 适合 |
|----|------|
| 周志华《机器学习》 | ML 入门必读 |
| Goodfellow《Deep Learning》 | DL 圣经 |
| Sebastian Raschka *Build a LLM from Scratch* | 从零实现 GPT |
| 黄佳《GPT 图解》 | 可视化讲解 LLM |

### 6.2 视频课程

- **吴恩达 ML & DL Specialization** —— Coursera 经典
- **斯坦福 CS224N** —— NLP 顶级课程
- **Hugging Face NLP Course** —— 实战导向，免费
- **Andrej Karpathy "Zero to Hero"** —— 从 micrograd 到 GPT，必看

### 6.3 博客与社区

- **Lilian Weng's blog** —— 系统综述（attention、prompt、agent）
- **Sebastian Raschka's Magazine** —— 周更 LLM 进展
- **Jay Alammar (illustrated-X)** —— 可视化讲解
- **科学空间（苏剑林）** —— 中文最佳数学解读
- **HuggingFace Daily Papers** —— 跟踪最新论文

### 6.4 实战代码仓库

| 仓库 | 用途 |
|------|------|
| **transformers** | 模型加载与推理 |
| **trl** | SFT / DPO / PPO |
| **peft** | LoRA / QLoRA |
| **LLaMA-Factory** | 一站式微调（中文友好） |
| **vllm** | 高性能推理 |
| **llama.cpp** | CPU/Mac 推理 |
| **DSPy** | Prompt 程序化优化 |

---

## 七、避免的常见弯路

::: warning 常见误区

1. **❌ 上来就读论文** —— 不如先跑通几个项目找感觉
2. **❌ 跳过经典 ML** —— LLM debug 时寸步难行
3. **❌ 只学，不做** —— ML 是手艺活，看 100 篇论文不如训 1 个模型
4. **❌ 追最新模型** —— 基础不变，新模型天天有
5. **❌ 自己从零造轮子** —— 用 trl / LLaMA-Factory，把精力花在数据和评测
6. **❌ 不做评测就上线** —— 没有评测的优化都是凭感觉
7. **❌ 只看通用 benchmark** —— 关键看你自己业务场景
:::

---

## 八、推荐学习节奏

### 8.1 全职学习（约 3 个月）

```
第 1 周  ── ML 基础 + 经典 ML demo
第 2 周  ── 深度学习基础 + PyTorch 实战
第 3 周  ── NLP 基础 + Transformer 论文精读
第 4-5 周 ── LLM 应用入门（LangChain + RAG）
第 6-7 周 ── Agent 与多 Agent 系统
第 8-9 周 ── LoRA 微调实战
第 10-11 周 ── DPO / 评测 / 部署
第 12 周  ── 完整项目（综合实战）
```

### 8.2 兼职学习（约 6 个月）

每周 10-15 小时，按上述节奏 ÷ 2。

### 8.3 速成路径（约 4 周）

如果只想做 LLM 应用、不做底层：

```
周 1：本章速览（不跑代码）+ Transformer 大致看懂
周 2：LangChain + Prompt Engineering
周 3：RAG 与 Agent
周 4：完整项目
```

不做微调，不做训练，**够用**。

---

## 九、最后的话

### 9.1 ML / LLM 的本质

> "All models are wrong, but some are useful." —— George Box

模型只是工具，关键是：
- **理解问题**：什么数据，什么目标，什么约束
- **理解工具**：每种方法的适用场景与局限
- **持续迭代**：评测 → 改进 → 评测

### 9.2 给学习者的最后建议

1. **理解优于记忆**：公式可以查，思想要内化
2. **实战优于理论**：跑通 100 个 demo，胜过读 100 篇论文
3. **质量优于数量**：一个项目从头到尾做完，比浅尝十个有用
4. **耐心优于速成**：ML 是知识密集型领域，没有捷径

---

> **回到主线**：现在开始你的 LLM 之旅 → [一、LLM 基础](/llm/)
>
> 项目所有应用项目入口：[实战项目导航](/01-langchain-basics)
