---
title: Embedding
---

<script setup>
const code1 = `import numpy as np

# 模拟文本向量（实际中由 Embedding 模型生成）
# 这里用简化的 8 维向量演示
vectors = {
    '猫': np.array([0.9, 0.8, 0.2, 0.1, 0.3, 0.7, 0.6, 0.1]),
    '狗': np.array([0.85, 0.75, 0.25, 0.15, 0.35, 0.65, 0.55, 0.15]),
    '汽车': np.array([0.1, 0.2, 0.9, 0.85, 0.7, 0.1, 0.15, 0.8]),
    '自行车': np.array([0.15, 0.25, 0.8, 0.7, 0.65, 0.15, 0.2, 0.75]),
    '苹果': np.array([0.5, 0.6, 0.1, 0.2, 0.1, 0.9, 0.8, 0.3]),
}

def cosine_similarity(a, b):
    '''计算两个向量的余弦相似度'''
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# 计算所有词对之间的相似度
words = list(vectors.keys())
print('余弦相似度矩阵：')
print('=' * 55)

# 打印表头
header = f'{"":6s}'
for w in words:
    header += f'{w:8s}'
print(header)
print('-' * 55)

for w1 in words:
    row = f'{w1:6s}'
    for w2 in words:
        sim = cosine_similarity(vectors[w1], vectors[w2])
        row += f'{sim:.3f}   '
    print(row)

print()
print('关键观察：')
print(f'  猫 vs 狗（同类动物）:   {cosine_similarity(vectors["猫"], vectors["狗"]):.4f}')
print(f'  汽车 vs 自行车（交通工具）: {cosine_similarity(vectors["汽车"], vectors["自行车"]):.4f}')
print(f'  猫 vs 汽车（不同领域）:  {cosine_similarity(vectors["猫"], vectors["汽车"]):.4f}')
print()
print('结论：语义相近的词，向量的余弦相似度更高！')
`

const code2 = `import numpy as np

# 演示向量维度对语义表示能力的影响

def generate_word_vectors(dim, seed=42):
    '''用不同维度生成模拟词向量'''
    np.random.seed(seed)

    # 基础语义类别
    categories = {
        '动物': ['猫', '狗', '鸟'],
        '水果': ['苹果', '香蕉', '橙子'],
        '交通': ['汽车', '飞机', '轮船'],
    }

    vectors = {}
    for cat_idx, (category, words) in enumerate(categories.items()):
        # 同类词共享基础方向 + 小扰动
        base = np.zeros(dim)
        # 在高维空间中为每个类别分配不同区域
        start = cat_idx * (dim // 3)
        end = min(start + dim // 3, dim)
        base[start:end] = 1.0

        for i, word in enumerate(words):
            noise = np.random.randn(dim) * 0.2
            vectors[word] = base + noise
            # 归一化
            vectors[word] = vectors[word] / np.linalg.norm(vectors[word])

    return vectors

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 对比不同维度
dims = [4, 16, 64, 256]
print('向量维度对语义区分能力的影响')
print('=' * 60)

for dim in dims:
    vecs = generate_word_vectors(dim)

    # 计算同类平均相似度
    same_sims = []
    same_sims.append(cosine_sim(vecs['猫'], vecs['狗']))
    same_sims.append(cosine_sim(vecs['苹果'], vecs['香蕉']))
    same_sims.append(cosine_sim(vecs['汽车'], vecs['飞机']))

    # 计算不同类平均相似度
    diff_sims = []
    diff_sims.append(cosine_sim(vecs['猫'], vecs['苹果']))
    diff_sims.append(cosine_sim(vecs['猫'], vecs['汽车']))
    diff_sims.append(cosine_sim(vecs['苹果'], vecs['汽车']))

    avg_same = np.mean(same_sims)
    avg_diff = np.mean(diff_sims)
    gap = avg_same - avg_diff

    print(f'\\n维度 = {dim:3d}:')
    print(f'  同类词平均相似度:   {avg_same:.4f}')
    print(f'  不同类词平均相似度: {avg_diff:.4f}')
    print(f'  区分度 (gap):      {gap:.4f} {"⚠ 区分不足" if gap < 0.3 else "✓ 区分良好" if gap > 0.5 else "△ 一般"}')

print()
print('-' * 60)
print('结论：')
print('  - 维度越高，向量空间越能区分不同语义')
print('  - 但维度过高会增加存储和计算成本')
print('  - 实际模型常用维度：768 / 1024 / 1536 / 3072')
print()
print('常见 Embedding 模型维度对比：')
print(f'  {"模型":<30s} {"维度":<8s} {"最大 tokens"}')
print(f'  {"OpenAI text-embedding-3-small":<30s} {"1536":<8s} {"8191"}')
print(f'  {"OpenAI text-embedding-3-large":<30s} {"3072":<8s} {"8191"}')
print(f'  {"BGE-large-zh":<30s} {"1024":<8s} {"512"}')
print(f'  {"BGE-M3":<30s} {"1024":<8s} {"8192"}')
print(f'  {"Jina-embeddings-v3":<30s} {"1024":<8s} {"8192"}')
`
</script>

# Embedding：词向量与向量表示

## 1. 文本向量化原理

Embedding 是将离散的文本（词、句子、段落）映射到连续的高维向量空间的过程：

```
"人工智能" → Embedding Model → [0.023, -0.156, 0.891, ..., 0.042]
                                        ↑
                                   1536 维浮点数向量
```

核心思想：**语义相近的文本，在向量空间中距离更近**。

| 特性 | 传统表示（One-Hot） | Embedding 表示 |
|------|---------------------|----------------|
| 维度 | 词表大小（数万~数十万） | 固定维度（768~3072） |
| 语义信息 | 无（正交向量） | 有（相似词距离近） |
| 稀疏性 | 极度稀疏 | 稠密向量 |
| 计算效率 | 低 | 高 |
| 可泛化性 | 差 | 好 |

## 2. LLM 中的 Embedding 层详解

### 2.1 Embedding 层是什么？做了什么事？

Embedding 层是 LLM（如 GPT、Claude）的**第一层**，负责将离散的 token ID 转换为连续的稠密向量，供后续 Transformer 层处理。

```
输入文本: "今天天气很好"
    ↓ Tokenizer（分词器）
Token 序列: [今天, 天气, 很, 好]
    ↓ Token → ID 映射
ID 序列: [3456, 7890, 112, 56]
    ↓ Embedding 层（查表）
向量序列: [[0.12, -0.34, ...], [0.56, 0.78, ...], ...]
              ↑ 每个 ID 对应一行向量
```

**核心操作：** 本质就是一次**查表（table lookup）**——用 token ID 作为索引，从 Embedding Matrix 中取出对应的行向量。

### 2.2 Embedding Matrix（嵌入矩阵）

Embedding Matrix 的形状为 `[vocab_size × embedding_dim]`：

```
              embedding_dim (如 4096)
         ┌────────────────────────────────┐
 token 0 │  0.012  -0.034  0.156  ...    │
 token 1 │  0.089   0.045 -0.234  ...    │
 token 2 │ -0.067   0.123  0.078  ...    │
   ...   │         ...                    │
token N  │  0.045  -0.089  0.167  ...    │
         └────────────────────────────────┘
         ↑ vocab_size 行（如 32000~150000）

查表过程：
  token ID = 3456
  → 取出矩阵的第 3456 行 → 得到该 token 的向量表示
```

| 参数 | 含义 | 典型值 |
|------|------|--------|
| vocab_size | 词表大小（所有可能的 token 数量） | 32,000 ~ 150,000 |
| embedding_dim | 每个 token 的向量维度 | 768 / 1024 / 4096 / 8192 |

**关键认知：** 这个矩阵是**训练出来的参数**，不是手动设定的。训练过程中，语义相近的 token 会被优化到向量空间中相近的位置。

### 2.3 Tokenizer（分词器）

Tokenizer 工作在 Embedding 层之前，负责将原始文本切分为 token 序列：

```
Tokenizer 的工作流程：
  原始文本 → 切分为子词 → 映射为 ID

示例（BPE 分词）：
  "unbelievable" → ["un", "believ", "able"] → [432, 8976, 213]
  "今天天气很好"  → ["今天", "天气", "很", "好"] → [3456, 7890, 112, 56]
```

| 算法 | 代表模型 | 特点 |
|------|---------|------|
| BPE (Byte Pair Encoding) | GPT 系列 | 从字符开始合并高频对，平衡词表大小和覆盖率 |
| WordPiece | BERT | 类似 BPE，选择使语言模型概率最大的合并 |
| SentencePiece | Llama, Claude | 语言无关，直接处理原始字符串（含空格） |

**Tokenizer ≠ Embedding：** Tokenizer 只负责「文本 → ID」的映射规则，Embedding 层负责「ID → 向量」的数值表示。

### 2.4 输入表示 = Token Embedding + Position Embedding

LLM 的实际输入向量不只有 token embedding，还需要加入位置信息：

```
最终输入向量 = Token Embedding + Position Embedding

Token Embedding:    [0.12, -0.34, 0.56, ...]  ← "这个词是什么"
Position Embedding: [0.01,  0.02, 0.03, ...]  ← "这个词在第几个位置"
                    ─────────────────────────
最终输入:           [0.13, -0.32, 0.59, ...]  ← 同时编码了"是什么"和"在哪里"
```

**为什么需要位置编码？**

Transformer 的自注意力机制是**无序的**（同时看所有 token），不像 RNN 天然有顺序信息。如果不加位置编码：
- "猫追狗" 和 "狗追猫" 对模型来说完全一样
- 模型无法区分词的先后顺序

| 位置编码方式 | 代表模型 | 特点 |
|------------|---------|------|
| 绝对正弦位置编码 | 原始 Transformer | 固定公式，不可学习 |
| 可学习绝对位置编码 | GPT-2, BERT | 位置编码作为可训练参数 |
| RoPE（旋转位置编码） | Llama, Claude, GPT-4 | 相对位置信息，支持长度外推 |
| ALiBi | BLOOM | 注意力偏置，无需额外参数 |

### 2.5 输出层与 Weight Tying（权重共享）

LLM 的最后一层需要将隐藏状态映射回词表大小的概率分布，这一步和 Embedding 层有密切关系：

```
Embedding 层（输入端）：
  token ID → 查表 → 向量     形状：[vocab_size × dim]

输出层（预测端）：
  隐藏向量 → 线性变换 → logits → softmax → 概率分布
                ↑
        形状也是 [vocab_size × dim]（转置后相乘）
```

**Weight Tying（权重绑定/共享）：**

许多模型让输出层的权重矩阵**直接复用** Embedding Matrix（转置）：

```
Output_logits = Hidden_state × Embedding_Matrix^T

好处：
  1. 减少参数量（省掉一个 [vocab_size × dim] 的大矩阵）
  2. 输入输出语义空间一致（输入时相近的词，输出预测时也倾向相近）
  3. 训练更稳定

使用 Weight Tying 的模型：GPT-2, Llama, T5, ALBERT
不使用的模型：GPT-3（输入输出用独立矩阵）
```

### 2.6 Embedding 的本质：表示而非理解

::: warning 重要区分
Embedding 层只负责**表示**（representation），不负责**理解**（comprehension）。
:::

```
Embedding 层输出的向量：
  - 编码了该 token 的"身份信息"（是哪个词）
  - 携带了统计学习到的语义倾向
  - 但还不包含上下文理解

真正的"理解"发生在：
  Transformer 的多层自注意力计算中
  每一层都在融合上下文信息，逐步构建深层语义

类比：
  Embedding = 每个学生的"档案卡"（静态基本信息）
  Transformer layers = 课堂讨论（学生之间交互、碰撞，产生新理解）
```

### 2.7 余弦相似度计算详解

用一个具体例子展示如何计算两个词向量的相似度：

```
假设简化为 3 维向量：
  猫  = [0.9, 0.8, 0.1]
  狗  = [0.85, 0.75, 0.15]
  汽车 = [0.1, 0.2, 0.9]

计算 cos(猫, 狗)：
  分子（点积）= 0.9×0.85 + 0.8×0.75 + 0.1×0.15
              = 0.765 + 0.6 + 0.015
              = 1.38

  分母 = |猫| × |狗|
       = √(0.81+0.64+0.01) × √(0.7225+0.5625+0.0225)
       = √1.46 × √1.3075
       = 1.208 × 1.143
       = 1.381

  cos(猫, 狗) = 1.38 / 1.381 ≈ 0.999（非常相似！）

计算 cos(猫, 汽车)：
  分子 = 0.9×0.1 + 0.8×0.2 + 0.1×0.9
       = 0.09 + 0.16 + 0.09
       = 0.34

  分母 = 1.208 × √(0.01+0.04+0.81)
       = 1.208 × √0.86
       = 1.208 × 0.927
       = 1.120

  cos(猫, 汽车) = 0.34 / 1.120 ≈ 0.304（不太相似）
```

**结论：** 语义相近的词（猫、狗）余弦相似度接近 1；语义无关的词（猫、汽车）余弦相似度较低。这正是 Embedding 空间的核心性质。

---

## 3. Embedding 模型

常见的 Embedding 模型及其特点：

| 模型 | 提供方 | 维度 | 特点 |
|------|--------|------|------|
| text-embedding-3-small | OpenAI | 1536 | 性价比高，适合通用场景 |
| text-embedding-3-large | OpenAI | 3072 | 精度更高，支持维度裁剪 |
| BGE-large-zh | 智源 | 1024 | 中文表现优秀，开源 |
| BGE-M3 | 智源 | 1024 | 多语言、多粒度、多功能 |
| Jina-embeddings-v3 | Jina AI | 1024 | 支持多任务，开源 |
| GTE-Qwen2 | 阿里 | 1536 | 基于 Qwen2，中英双语 |

::: info OpenAI text-embedding-3 的维度裁剪
text-embedding-3 系列支持 `dimensions` 参数，可以将输出维度从默认值裁剪到更短（如 256、512），在牺牲少量精度的前提下显著降低存储成本。
:::

## 4. 余弦相似度

余弦相似度是衡量两个向量方向相似性的标准方法：

```
                A · B          Σ(ai × bi)
cos(θ) = ─────────── = ───────────────────────
            |A| × |B|     √Σ(ai²) × √Σ(bi²)

取值范围：[-1, 1]
  1  → 方向完全相同（语义最相似）
  0  → 正交（无关）
 -1  → 方向完全相反（语义最不相似）
```

<PythonRunner :browser-runnable="true" :code="code1" />

::: tip 距离度量对比
| 度量方式 | 公式 | 适用场景 |
|---------|------|---------|
| 余弦相似度 | cos(θ) | 文本语义匹配（最常用） |
| 欧氏距离 | √Σ(ai-bi)² | 聚类分析 |
| 点积 | Σ(ai×bi) | 已归一化的向量 |
| 曼哈顿距离 | Σ\|ai-bi\| | 稀疏向量 |
:::

## 5. 向量维度与语义表示

维度越高，向量空间的表达能力越强，但也带来更高的计算和存储开销：

<PythonRunner :browser-runnable="true" :code="code2" />

## 6. 多模态 Embedding

现代 Embedding 不仅限于文本，还可以将图像、音频等映射到同一向量空间：

```
文本: "一只白色的猫"  → Embedding → [0.23, -0.15, ...]  ─┐
                                                          ├─ 相似度高！
图片: 🐱 (白猫照片)   → Embedding → [0.21, -0.14, ...]  ─┘
```

| 模型 | 支持模态 | 典型应用 |
|------|---------|---------|
| CLIP (OpenAI) | 文本 + 图像 | 图文检索、零样本分类 |
| ImageBind (Meta) | 文本 + 图像 + 音频 + 视频 + 深度 + 热力图 | 跨模态理解 |
| Jina CLIP v2 | 文本 + 图像 | 多语言图文检索 |
| BGE-visualized | 文本 + 图像 | 混合模态检索 |

::: warning 多模态对齐的挑战
不同模态的信息密度差异很大。一张图片包含的信息可能需要数百个词来描述，直接对齐可能丢失细节。对比学习（Contrastive Learning）是解决这一问题的主流方法。
:::

## 7. 向量数据库简介

当 Embedding 向量规模达到百万甚至亿级时，需要专门的向量数据库进行高效检索：

| 数据库 | 类型 | 特点 |
|--------|------|------|
| Pinecone | 云服务 | 全托管，开箱即用 |
| Milvus | 开源 | 分布式，支持万亿级向量 |
| Qdrant | 开源 | Rust 编写，高性能 |
| Weaviate | 开源 | 支持混合搜索（向量+关键词） |
| Chroma | 开源 | 轻量级，适合原型开发 |
| FAISS | 库 | Meta 出品，纯计算库（非数据库） |
| pgvector | 扩展 | PostgreSQL 扩展，与现有架构集成 |

### 典型 RAG 检索流程

```
用户问题
    ↓
Embedding Model → 问题向量 [0.12, -0.34, ...]
    ↓
向量数据库: ANN 检索 (近似最近邻)
    ↓
返回 Top-K 最相似的文档片段
    ↓
拼接到 Prompt → 送入 LLM 生成答案
```

::: info ANN（近似最近邻）算法
向量数据库通常不会暴力搜索所有向量，而是使用近似算法：
- **HNSW**（分层可导航小世界图）：精度高，内存占用大
- **IVF**（倒排文件索引）：适合大规模数据，需要训练
- **PQ**（乘积量化）：压缩向量，节省存储
- **ScaNN**（Google）：兼顾速度和精度
:::

---

::: tip 下一步
- [文本生成机制](/llm/generation) — Temperature、Top-P 的数学原理
- [提示工程](/llm/prompt-engineering) — 掌握与 LLM 高效对话的技巧
:::
