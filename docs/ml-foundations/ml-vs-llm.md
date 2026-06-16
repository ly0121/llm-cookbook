# ML 与 LLM 的关系

> 一文厘清经典 ML、深度学习、LLM 三者的演进、共性与分工

---

## 一、演进时间轴

```
1950s──┬─ 感知机（Perceptron）
       │  线性分类器，只能解决线性可分问题
       │
1970s──┬─ 决策树 / 朴素贝叶斯
       │  规则与概率方法主导
       │
1990s──┬─ SVM、随机森林、TF-IDF
       │  特征工程时代，统计学习达到经典 ML 巅峰
       │
2006──┬─ 深度信念网络 (Hinton)
       │  深度学习"深度"复兴
       │
2012──┬─ AlexNet 在 ImageNet 大胜
       │  CNN 主宰图像；GPU 训练成为标配
       │
2013──┬─ Word2Vec
       │  词嵌入开启 NLP 表示学习时代
       │
2014──┬─ Seq2Seq + Attention
       │  机器翻译突破
       │
2017──┬─ Transformer ("Attention is All You Need")
       │  扔掉 RNN，纯注意力；并行性 + 长程依赖
       │
2018──┬─ BERT / GPT-1
       │  "预训练 + 微调"范式确立
       │
2020──┬─ GPT-3 (175B)
       │  Scaling Law 验证；in-context learning 涌现
       │
2022──┬─ ChatGPT / RLHF
       │  对齐技术成熟；进入大众视野
       │
2023──┬─ GPT-4 / Llama / 多模态
       │  多模态 + 工具调用 + Agent
       │
2024──┬─ o1 / DeepSeek-V3 / Sora
       │  推理时计算 + MoE + 视频生成
       │
2025──┬─ Claude 4 / GPT-5 / Agentic systems
          自主 agent 大规模落地
```

---

## 二、三者的共性原理

虽然技术形态差别巨大，但**核心原理**一脉相承：

### 2.1 都是"从数据中学映射"

```
经典 ML：       (X, y) → 学  f: X → y
深度学习：      (X, y) → 学  深层 f: X → y
LLM 预训练：   (token序列, next-token) → 学  P(t_n | t_1, ..., t_{n-1})
LLM SFT：      (prompt, response) → 学  P(response | prompt)
LLM RLHF：     (prompt, [response_好, response_差]) → 学  reward → 优化策略
```

### 2.2 都用梯度下降优化

不管是逻辑回归、CNN、还是 GPT-4 的 1.7T 参数：

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$

只是规模、变体（Adam / AdamW）、调度（cosine + warmup）不同。

### 2.3 都面对同一个对手：过拟合

| | 经典 ML | 深度学习 | LLM |
|--|--------|---------|-----|
| 防过拟合手段 | L1/L2 正则、CV、特征选择 | Dropout、BatchNorm、早停 | 海量数据、LayerNorm、LoRA 低秩约束 |
| 偏差-方差 | 严格遵循 | 严格遵循 | 双下降现象（参数远超样本反而泛化好） |

### 2.4 都依赖三大要素

```
        数据
       ╱   ╲
      ╱     ╲
   算力 ── 算法
```

经典 ML：算法主导（数据/算力够用即可）
深度学习：三者均衡，算力开始关键
LLM：**算力 + 数据 主导**，算法相对稳定（Transformer 架构 7 年无大变）

---

## 三、各自的"专长领域"

### 3.1 经典 ML 仍然不可替代

✅ **结构化数据（表格）**：客户流失、信用评分、CTR 预估
- XGBoost / LightGBM 在 Kaggle 表格赛冠军占比 > 80%
- 推理快、可解释、训练成本低

✅ **小样本学习**：医疗、法律小数据集
- LLM 需要海量数据才能预训练；经典 ML 1000 条数据就能跑

✅ **可解释性硬性要求**：金融风控、医疗诊断
- 决策树规则、逻辑回归权重直接可读
- LLM 是"黑盒中的黑盒"

✅ **极低延迟场景**：广告排序、实时反欺诈
- 一次 LR 推理 < 1ms
- LLM 推理 100ms-10s 起步

### 3.2 深度学习（非 LLM）的舞台

✅ **图像**：分类、检测、分割（CNN / ViT）
✅ **语音**：ASR、TTS（Conformer、Whisper）
✅ **专用领域 NLP**：医疗 NER、法律文书分类（小型 BERT）
✅ **结构化时序**：金融预测、需求预测（LSTM / Transformer）

### 3.3 LLM 的舞台

✅ **开放域文本理解与生成**：写作、翻译、摘要
✅ **代码生成与重构**：Cursor / Copilot
✅ **零样本/少样本任务**：用 prompt 解决新任务，无需训练数据
✅ **复杂推理与 Agent**：多步规划、工具调用
✅ **多模态融合**：图文、音视频联合理解

---

## 四、何时选哪个？决策树

```
你的任务是？
│
├─ 结构化数据 + 大量标注？
│   └→ 用 XGBoost / LightGBM（经典 ML）
│
├─ 图像分类 / 检测 / 分割？
│   └→ 用 CNN / ViT（深度学习）
│
├─ 语音识别 / 合成？
│   └→ 用 Whisper / 专用模型
│
├─ 文本任务 + 标注数据 < 1000？
│   ├─ 简单分类？→ TF-IDF + LR/SVM
│   └─ 复杂理解？→ LLM few-shot
│
├─ 文本任务 + 标注数据 >> 1000？
│   ├─ 需要极低延迟？→ 微调小型 BERT
│   └─ 不太敏感？→ LLM API 或 LoRA 微调
│
├─ 开放域问答 / 写作 / 创作？
│   └→ LLM（无替代品）
│
├─ Agent / 多步规划 / 工具调用？
│   └→ LLM + Function Calling
│
└─ 推荐 / 排序 / 表格预测？
    └→ 经典 ML + 深度学习混合（如 DIN、DeepFM）
```

::: warning 反模式：用 LLM 做一切
**常见错误：**
- 把"用户行为预测"塞进 prompt 让 LLM 猜 → 不如简单 LR
- 让 LLM 解析固定格式 JSON → 写正则更可靠
- 用 LLM 做关键词搜索 → BM25 又快又准

**LLM 的优势是"理解 + 生成"，不是"分类 + 排序"。**
:::

---

## 五、对应关系速查表

### 5.1 概念映射

| 经典 ML / DL | LLM 中的对应 | 说明 |
|------------|-------------|------|
| 逻辑回归（softmax） | next-token 预测 | LLM 输出层就是超大词表的 softmax 分类 |
| One-Hot 编码 | Token Embedding | 都是离散 → 数值 |
| Word2Vec | Embedding 模型（BGE） | 同源；后者是 contextual |
| RNN 隐状态 | KV Cache | 都是"压缩历史" |
| FFN（MLP） | Transformer FFN 子层 | 一模一样的 MLP |
| 残差连接 | Transformer sublayer 残差 | 一模一样 |
| BatchNorm | LayerNorm / RMSNorm | 归一化维度不同 |
| Dropout | LLM 微调时常用 | 预训练大模型几乎不用 |
| L2 正则 | LoRA 低秩约束 | 都是"限制参数变化空间" |
| Adam | AdamW | 后者解耦权重衰减 |
| 交叉验证 | Holdout + Benchmark | LLM 不做 CV（成本太高） |
| 集成学习 | MoE（Mixture of Experts） | 多个专家投票 |
| TF-IDF | BM25（在 RAG 中） | 仍在用 |
| 监督学习 | SFT | next-token 监督 |
| 强化学习 | RLHF / DPO | 用奖励模型对齐 |

### 5.2 训练范式映射

```
经典 ML 训练：
   随机初始化 → 喂数据 → 收敛 → 部署
   通常几分钟到几小时

LLM 训练：
   ┌─────────────────────────────────┐
   │ 阶段 1：预训练（自监督，几个月）  │
   │   万亿 token，AdamW，1000 GPU   │
   ├─────────────────────────────────┤
   │ 阶段 2：SFT（监督微调，几小时-几天）│
   │   高质量人工标注 prompt+response │
   ├─────────────────────────────────┤
   │ 阶段 3：RLHF / DPO（对齐，几天）│
   │   人类偏好数据 → 奖励模型 → PPO │
   └─────────────────────────────────┘
```

---

## 六、误区澄清

### 6.1 ❌ "LLM 让经典 ML 过时了"

**真相**：LLM 让**自然语言任务**的入门门槛降低，但：
- 表格数据：经典 ML 仍是 SOTA
- 高频低延迟：经典 ML 不可替代
- 数据极少：经典 ML 反而稳定

很多公司的核心系统（推荐、广告、风控）仍以经典 ML 为主，LLM 是补充。

### 6.2 ❌ "Transformer 取代了 CNN"

**真相**：在**图像领域**，ViT 在大数据下超越 CNN，但：
- 中小数据集：CNN 仍更高效
- 实时推理：CNN 推理更快
- 边缘设备：CNN 仍占主导

最新趋势：**ConvNeXt** 把 Transformer 的设计思想反向移植回 CNN，性能与 ViT 相当。

### 6.3 ❌ "深度学习 = LLM"

**真相**：LLM 是深度学习的一个分支（Transformer + 自回归 + 海量数据）。深度学习还包括：
- CNN（图像）
- RNN/LSTM（序列）
- GAN / Diffusion（生成）
- GNN（图）
- 强化学习（Q-Network / PPO）

### 6.4 ❌ "学了 LLM 就够了，不用学经典 ML"

**真相**：很多 LLM 概念**直接来自经典 ML**：
- 不懂梯度下降 → 不懂 LLM 训练
- 不懂交叉熵 → 不懂 next-token loss
- 不懂正则化 → 不懂 LoRA / 微调过拟合
- 不懂 TF-IDF → 不懂 RAG 混合检索

**经典 ML 是地基，LLM 是高楼。**

---

## 七、未来趋势：边界正在模糊

```
2024-2025 的几个方向：

1. 混合架构
   Mamba / RWKV：用 RNN 的状态空间替代部分 Attention
   性能接近 Transformer，但 O(n) 而非 O(n²)

2. LLM 做经典任务
   "Tabular Transformers"：用 LLM 处理表格数据
   但目前性能仍低于 XGBoost

3. 经典方法做新任务
   FlashAttention：用经典分块技术加速 Attention
   Speculative Decoding：用小模型加速大模型

4. Agent + 工具
   LLM 作为"规划器"，调用经典模型作为"专家工具"
   （e.g., 调用 SQL 引擎、计算器、特定领域分类器）
```

---

## 八、给学习者的建议

### 8.1 你已是 ML 专家

- 跳过本章和"深度学习基础"
- 直接进入 [LLM 基础](/llm/) 与 [Transformer](/llm-knowledge/transformer-architecture)
- 重点关注 LLM 与经典 ML 的差异：scaling law、in-context learning、RLHF

### 8.2 你懂深度学习但不熟 LLM

- 简略浏览本章
- 重点读 [Transformer 架构](/llm-knowledge/transformer-architecture) 与 [LLM 训练](/llm-knowledge/training)
- 实战：跑通 [01-langchain-basics](/01-langchain-basics) 和 [04-rag-system](/04-rag-system)

### 8.3 你完全零基础

- 按本章建议路径走（约 2 周）
- 先跑通 `ml_foundations/` 下所有 demo，**重感受、轻数学**
- 再进入 LLM 主线

### 8.4 你只想做 LLM 应用，不在意原理

- 直接跳到 [LangChain 基础](/01-langchain-basics)
- 用到具体技术（如 RAG）时再回来查对应章节
- **但请至少理解：什么是 embedding、什么是损失函数、什么是过拟合** —— 这是 debug LLM 的最低要求

---

## 九、本章回顾

完成 ML 基础部分后，你应该能：

- ✅ 解释 supervised / unsupervised / RL 三大范式
- ✅ 看懂逻辑回归、决策树、随机森林的代码
- ✅ 理解 TF-IDF、Word2Vec 的原理与公式
- ✅ 描述什么是张量、autograd、反向传播
- ✅ 实现一个最简单的 PyTorch 训练循环
- ✅ 解释为什么 Transformer 取代了 RNN
- ✅ 知道 LayerNorm / Dropout / AdamW 各自作用
- ✅ 在 LLM 概念出现时，能联想到对应的经典 ML 知识

> **下一站**：[进阶学习路径](./learning-path) —— 从 ML 走到 LLM 微调的完整路线
