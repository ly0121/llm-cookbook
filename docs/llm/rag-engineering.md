---
title: 企业级RAG工程化
---

<script setup>
const code1 = `# BM25 + 向量相似度 混合检索演示
import math
from collections import Counter

# ========== BM25 算法实现 ==========
class BM25:
    """BM25 评分算法实现"""
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.doc_count = len(documents)
        # 分词（简单按字符/空格切分）
        self.doc_tokens = [list(doc) for doc in documents]
        self.avg_dl = sum(len(d) for d in self.doc_tokens) / self.doc_count
        # 计算 IDF
        self.idf = {}
        self._compute_idf()

    def _compute_idf(self):
        """计算逆文档频率"""
        df = Counter()
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] += 1
        for token, freq in df.items():
            # BM25 IDF 公式
            self.idf[token] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, query, doc_idx):
        """计算单个文档的 BM25 分数"""
        doc_tokens = self.doc_tokens[doc_idx]
        doc_len = len(doc_tokens)
        tf = Counter(doc_tokens)
        score = 0.0
        for q_char in query:
            if q_char not in self.idf:
                continue
            term_freq = tf.get(q_char, 0)
            # BM25 TF 归一化
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += self.idf[q_char] * numerator / denominator
        return score

    def search(self, query, top_k=3):
        """返回 top-k 结果"""
        scores = [(i, self.score(query, i)) for i in range(self.doc_count)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

# ========== 简单向量相似度 ==========
def char_vector(text, vocab):
    """基于字符频率的简单向量化"""
    freq = Counter(text)
    return [freq.get(ch, 0) for ch in vocab]

def cosine_similarity(v1, v2):
    """余弦相似度"""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# ========== 混合检索 ==========
def hybrid_search(query, documents, alpha=0.5, top_k=3):
    """
    混合检索: alpha * BM25_norm + (1-alpha) * vector_sim
    alpha: BM25 权重
    """
    # BM25 检索
    bm25 = BM25(documents)
    bm25_scores = [bm25.score(query, i) for i in range(len(documents))]
    # 归一化 BM25 分数
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_norm = [s / max_bm25 for s in bm25_scores]

    # 向量检索
    vocab = sorted(set(''.join(documents) + query))
    query_vec = char_vector(query, vocab)
    vec_scores = [cosine_similarity(query_vec, char_vector(doc, vocab)) for doc in documents]

    # 混合分数
    hybrid_scores = [alpha * b + (1 - alpha) * v for b, v in zip(bm25_norm, vec_scores)]

    results = sorted(enumerate(hybrid_scores), key=lambda x: x[1], reverse=True)[:top_k]
    return results, bm25_norm, vec_scores

# ========== 测试 ==========
documents = [
    "RAG系统通过检索增强来提升大模型的回答质量",
    "向量数据库存储文档的嵌入向量用于语义检索",
    "BM25是经典的基于词频的稀疏检索算法",
    "大语言模型在没有外部知识时容易产生幻觉",
    "混合检索结合了稀疏检索和稠密检索的优势",
    "Transformer架构是现代NLP模型的基础",
]

query = "如何提升检索质量"
print(f"查询: {query}")
print(f"文档数: {len(documents)}")
print("=" * 60)

results, bm25_scores, vec_scores = hybrid_search(query, documents, alpha=0.6)

print(f"\\n{'排名':<4} {'BM25':<8} {'向量':<8} {'混合':<8} 文档内容")
print("-" * 70)
for rank, (idx, hybrid_score) in enumerate(results, 1):
    print(f"{rank:<4} {bm25_scores[idx]:<8.3f} {vec_scores[idx]:<8.3f} {hybrid_score:<8.3f} {documents[idx][:30]}")

print("\\n" + "=" * 60)
print("\\n不同 alpha 值对排序的影响（alpha=BM25权重）:")
print(f"{'alpha':<8} {'Top-1 文档'}")
print("-" * 50)
for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    res, _, _ = hybrid_search(query, documents, alpha=alpha)
    top_idx = res[0][0]
    print(f"{alpha:<8.1f} {documents[top_idx][:35]}")
`

const code2 = `# 文档切片策略对比演示
# 演示不同 chunk_size 和切分方法对检索效果的影响

# ========== 模拟文档 ==========
document = """
人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于研究和开发能够模拟人类智能的系统。
近年来，深度学习技术的突破推动了AI的快速发展。

大语言模型（Large Language Model，LLM）是基于Transformer架构训练的超大规模神经网络模型。
GPT系列、Claude、LLaMA等都是典型的大语言模型。这些模型通过在海量文本数据上进行预训练，
学习到了丰富的语言知识和推理能力。

检索增强生成（Retrieval-Augmented Generation，RAG）是一种结合检索和生成的技术框架。
RAG系统首先从知识库中检索相关文档片段，然后将检索到的内容作为上下文输入给大模型，
从而生成更准确、更有据可查的回答。RAG有效解决了大模型的知识时效性和幻觉问题。

RAG系统的核心组件包括：文档解析、文本切片、向量化、索引构建、检索、重排和生成。
其中文本切片策略直接影响检索的质量。切片太大会引入噪声，切片太小会丢失上下文。
找到合适的切片粒度是RAG工程化中的关键挑战之一。
""".strip()

print(f"原始文档长度: {len(document)} 字符")
print("=" * 60)

# ========== 策略1: 固定大小切分 ==========
def fixed_size_chunking(text, chunk_size, overlap=0):
    """固定大小切分"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ========== 策略2: 按段落切分 ==========
def paragraph_chunking(text):
    """按段落（空行）切分"""
    paragraphs = [p.strip() for p in text.split('\\n\\n') if p.strip()]
    return paragraphs

# ========== 策略3: 递归切分 ==========
def recursive_chunking(text, max_size=150, separators=None):
    """递归切分：按层级分隔符逐级切分"""
    if separators is None:
        separators = ['\\n\\n', '\\n', '。', '，', ' ']

    if len(text) <= max_size:
        return [text]

    chunks = []
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            current_chunk = ""
            for part in parts:
                if len(current_chunk) + len(part) + len(sep) <= max_size:
                    current_chunk += (sep if current_chunk else "") + part
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = part
            if current_chunk:
                chunks.append(current_chunk)
            return chunks

    # 如果没有分隔符可用，强制切分
    return fixed_size_chunking(text, max_size)

# ========== 策略4: 语义切分（模拟） ==========
def semantic_chunking(text, threshold=0.3):
    """语义切分（模拟）：基于句子相似度决定是否切分"""
    sentences = [s.strip() for s in text.replace('\\n', '').split('。') if s.strip()]
    chunks = []
    current_chunk = sentences[0] if sentences else ""

    for i in range(1, len(sentences)):
        # 模拟语义相似度：用字符重叠率
        prev_chars = set(sentences[i-1])
        curr_chars = set(sentences[i])
        overlap = len(prev_chars & curr_chars) / max(len(prev_chars | curr_chars), 1)

        if overlap < threshold:  # 语义差异大，切分
            chunks.append(current_chunk + "。")
            current_chunk = sentences[i]
        else:
            current_chunk += "。" + sentences[i]

    if current_chunk:
        chunks.append(current_chunk + "。")
    return chunks

# ========== 对比不同策略 ==========
print("\\n【策略对比】")
print("-" * 60)

strategies = {
    "固定切分(100字)": fixed_size_chunking(document, 100),
    "固定切分(200字)": fixed_size_chunking(document, 200),
    "固定切分(100字,20重叠)": fixed_size_chunking(document, 100, overlap=20),
    "段落切分": paragraph_chunking(document),
    "递归切分(150字)": recursive_chunking(document, max_size=150),
    "语义切分": semantic_chunking(document),
}

print(f"{'策略':<20} {'块数':<6} {'平均长度':<10} {'最短':<6} {'最长':<6}")
print("-" * 60)
for name, chunks in strategies.items():
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    min_len = min(len(c) for c in chunks)
    max_len = max(len(c) for c in chunks)
    print(f"{name:<20} {len(chunks):<6} {avg_len:<10.1f} {min_len:<6} {max_len:<6}")

# ========== 模拟检索质量评估 ==========
print("\\n" + "=" * 60)
print("\\n【检索质量模拟】")
print("查询: 'RAG系统如何解决幻觉问题'")
print("-" * 60)

query = "RAG系统如何解决幻觉问题"
query_chars = set(query)

def simple_relevance(chunk, query_chars):
    """简单相关性评分：字符重叠"""
    chunk_chars = set(chunk)
    return len(query_chars & chunk_chars) / len(query_chars)

print(f"\\n{'策略':<20} {'Top-1相关度':<12} {'Top-1片段预览'}")
print("-" * 70)
for name, chunks in strategies.items():
    scores = [(i, simple_relevance(c, query_chars)) for i, c in enumerate(chunks)]
    scores.sort(key=lambda x: x[1], reverse=True)
    top_idx, top_score = scores[0]
    preview = chunks[top_idx][:40].replace('\\n', ' ')
    print(f"{name:<20} {top_score:<12.3f} {preview}...")

# 展示段落切分的详细结果
print("\\n" + "=" * 60)
print("\\n【段落切分详细结果】")
para_chunks = paragraph_chunking(document)
for i, chunk in enumerate(para_chunks):
    score = simple_relevance(chunk, query_chars)
    marker = " <-- 最相关" if score == max(simple_relevance(c, query_chars) for c in para_chunks) else ""
    print(f"\\n块{i+1} (长度:{len(chunk)}, 相关度:{score:.3f}){marker}")
    print(f"  {chunk[:60].replace(chr(10), ' ')}...")
`
</script>

# 企业级 RAG 工程化

检索增强生成（RAG）是当前大模型应用落地的核心架构。本文系统介绍 RAG 工程化中的关键技术环节，从文档解析到评估指标，帮助构建生产级 RAG 系统。

## RAG 系统全景架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG 系统架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 文档解析  │───▶│ 文本切片  │───▶│ 向量化   │───▶│ 索引存储  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │                                                │         │
│       ▼                                                ▼         │
│  PDF/Word/HTML                                    向量数据库      │
│  表格/图片/OCR                                                   │
│                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 用户查询  │───▶│ 查询改写  │───▶│ 混合检索  │───▶│  重排序   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                        │         │
│                                                        ▼         │
│                                                  ┌──────────┐   │
│                                                  │ LLM 生成  │   │
│                                                  └──────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. 文档解析与切片策略

### 1.1 文档解析

不同格式的文档需要不同的解析策略：

| 文档类型 | 解析工具 | 难点 |
|---------|---------|------|
| PDF | PyPDF2, PDFPlumber, Unstructured | 跨页表格、双栏布局 |
| Word | python-docx | 嵌套表格、图片标注 |
| HTML | BeautifulSoup, Trafilatura | 噪声去除、正文提取 |
| 表格 | Camelot, Tabula | 合并单元格、跨页表格 |
| 扫描件 | Tesseract, PaddleOCR | 识别精度、版面分析 |

::: tip 实践建议
对于企业级应用，推荐使用 **Unstructured.io** 或 **LlamaParse** 等专业解析工具，它们支持多种格式统一输出，并能保留文档结构信息（标题层级、表格、列表等）。
:::

### 1.2 切片策略

切片策略直接决定检索质量。核心原则：**一个切片应包含一个完整的语义单元**。

```
┌─────────────────────────────────────────────────────┐
│              切片策略决策树                            │
├─────────────────────────────────────────────────────┤
│                                                       │
│  文档类型？                                           │
│  ├── 结构化文档（有标题层级）──▶ 按标题/段落切分       │
│  ├── 长篇连续文本 ──────────▶ 递归切分 + 重叠        │
│  ├── 表格密集型 ───────────▶ 表格单独切片 + 标题关联  │
│  └── 代码/技术文档 ────────▶ 按函数/类/模块切分       │
│                                                       │
│  切片大小建议：                                       │
│  ├── 精确问答场景：128-256 tokens                     │
│  ├── 摘要/综述场景：512-1024 tokens                   │
│  └── 通用场景：256-512 tokens + 10-15% overlap        │
└─────────────────────────────────────────────────────┘
```

**主要切片方法：**

1. **固定大小切分**：简单但容易截断语义
2. **递归切分（LangChain RecursiveCharacterTextSplitter）**：按分隔符层级逐步切分
3. **语义切分**：基于 Embedding 相似度，在语义断点处切分
4. **文档结构切分**：利用标题、段落等文档结构

::: info 表格处理策略
表格是 RAG 中的难点。推荐方案：
- 将表格转为自然语言描述（适合简单表格）
- 表格作为独立切片，附加表头和上下文标题
- 使用多模态模型直接理解表格图片
:::

<PythonRunner :code="code2" title="文档切片策略对比实验" />

---

## 2. 高级检索策略

### 2.1 混合检索（Hybrid Search）

单一检索方式各有局限：

| 检索方式 | 优势 | 劣势 |
|---------|------|------|
| 稀疏检索（BM25） | 精确匹配关键词、无需训练 | 无法理解语义相似性 |
| 稠密检索（向量） | 语义理解、跨语言 | 对精确术语匹配较弱 |
| **混合检索** | **兼具两者优势** | 需要调节权重参数 |

混合检索公式：

```
Score_hybrid = α × Score_BM25_normalized + (1-α) × Score_vector

常用融合策略：
├── 线性加权（Linear Combination）
├── 倒数排名融合（Reciprocal Rank Fusion, RRF）
│   RRF_score = Σ 1/(k + rank_i)，k通常取60
└── 学习融合（Learned Fusion）
```

<PythonRunner :code="code1" title="BM25 + 向量混合检索演示" />

### 2.2 Query 重写与路由

```
┌─────────────────────────────────────────────────┐
│            Query 处理流水线                       │
├─────────────────────────────────────────────────┤
│                                                   │
│  原始 Query                                       │
│      │                                            │
│      ├──▶ Query 重写（消除歧义、补充上下文）        │
│      │    "RAG怎么用" → "如何构建RAG检索增强系统"   │
│      │                                            │
│      ├──▶ Query 分解（复杂问题拆分为子问题）        │
│      │    "对比A和B" → ["A的特点", "B的特点"]      │
│      │                                            │
│      ├──▶ Query 路由（选择最佳检索策略）            │
│      │    事实性问题 → 向量检索                     │
│      │    精确查询 → 关键词检索                     │
│      │    统计问题 → SQL/结构化查询                 │
│      │                                            │
│      └──▶ HyDE（假设性文档嵌入）                   │
│           让 LLM 先生成假设答案，用假设答案检索      │
└─────────────────────────────────────────────────┘
```

### 2.3 HyDE（Hypothetical Document Embeddings）

HyDE 的核心思想：**用 LLM 生成一个假设性回答，然后用这个回答（而非原始 query）去做向量检索**。

```
传统检索:  Query → Embed(Query) → 向量搜索 → 文档

HyDE:      Query → LLM生成假设答案 → Embed(假设答案) → 向量搜索 → 文档
```

::: tip 为什么 HyDE 有效？
假设答案与真实文档在语义空间中更接近（都是"回答式"的文本），而 Query 通常是"问题式"的，与文档形式差异较大。
:::

---

## 3. 重排序（Reranking）

### 3.1 为什么需要重排？

初始检索（粗排）追求**召回率**，返回 Top-50/100 候选文档。重排序追求**精确率**，从候选中挑选最相关的 Top-3/5 送入 LLM。

```
检索流程：
Query → 粗排(Top-100, 毫秒级) → 重排(Top-5, 百毫秒级) → LLM生成

为什么分两阶段？
├── 粗排：速度快（ANN向量搜索），但精度有限
└── 精排：精度高（Cross-Encoder），但速度慢，无法全库扫描
```

### 3.2 重排模型对比

| 模型 | 类型 | 特点 | 适用场景 |
|------|------|------|---------|
| BGE-Reranker-v2 | Cross-Encoder | 中文效果好、多粒度 | 中文RAG首选 |
| Cohere Rerank | API服务 | 多语言、易接入 | 快速集成 |
| bce-reranker | Cross-Encoder | 中英双语优化 | 双语场景 |
| ColBERT | Late Interaction | 速度快、精度较好 | 大规模低延迟 |
| RankGPT | LLM-based | 利用LLM排序能力 | 极高精度需求 |

::: warning 性能提示
Cross-Encoder 重排器对每个 (query, doc) 对做联合编码，计算量为 O(n)。当候选文档数量超过 100 时，延迟会显著增加。建议先用粗排缩小范围到 20-50 篇再重排。
:::

### 3.3 Cross-Encoder vs Bi-Encoder

```
Bi-Encoder（用于粗排/初始检索）:
  Query  →  [Encoder] → query_vec  ─┐
                                      ├─ cosine_sim → score
  Doc    →  [Encoder] → doc_vec   ─┘
  优点: 文档可预计算向量，检索极快
  缺点: query 和 doc 独立编码，交互不够充分

Cross-Encoder（用于精排/重排）:
  [Query + Doc] →  [Encoder] → score
  优点: query 和 doc 充分交互，精度高
  缺点: 无法预计算，每次都要联合编码
```

---

## 4. 向量数据库选型

### 4.1 主流向量数据库对比

| 数据库 | 类型 | 索引算法 | 分布式 | 适用场景 |
|--------|------|---------|--------|---------|
| **Milvus** | 专用向量DB | IVF/HNSW/DiskANN | 原生支持 | 大规模生产环境 |
| **Qdrant** | 专用向量DB | HNSW | 支持 | 中小规模、Rust高性能 |
| **Faiss** | 向量库 | IVF/PQ/HNSW | 不支持 | 研究/单机高性能 |
| **PGVector** | PG扩展 | IVFFlat/HNSW | PG生态 | 已有PG基础设施 |
| **ElasticSearch** | 搜索引擎 | HNSW | 原生支持 | 混合检索、已有ES |
| **Weaviate** | 专用向量DB | HNSW | 支持 | GraphQL生态 |
| **Chroma** | 嵌入式向量DB | HNSW | 不支持 | 原型开发、轻量级 |

### 4.2 选型决策树

```
你的场景是什么？
│
├── 快速原型/POC ──────────────▶ Chroma / Faiss
│
├── 已有 PostgreSQL ──────────▶ PGVector
│
├── 已有 ElasticSearch ───────▶ ES 8.x kNN
│
├── 生产环境（数据量 < 1000万）
│   ├── 追求性能 ─────────────▶ Qdrant
│   └── 需要混合查询 ─────────▶ Milvus
│
└── 大规模生产（> 1亿向量）
    ├── 云原生 ────────────────▶ Milvus / Zilliz Cloud
    └── 需要磁盘索引 ──────────▶ Milvus (DiskANN)
```

::: info 关键性能指标
选择向量数据库时关注：
- **QPS**：每秒查询数（延迟敏感场景）
- **Recall@K**：召回率（精度敏感场景）
- **内存占用**：单向量内存成本
- **写入吞吐**：数据更新频率高时重要
- **过滤性能**：带条件过滤的向量搜索效率
:::

---

## 5. GraphRAG 与 Self-RAG

### 5.1 GraphRAG

GraphRAG 利用知识图谱增强检索，特别适合**需要多跳推理**的复杂问题。

```
传统 RAG:
  Query → 向量搜索 → 独立文档片段 → LLM生成
  问题: 难以回答需要关联多个实体的复杂问题

GraphRAG:
  Query → 实体识别 → 图谱遍历/子图检索 → 关联上下文 → LLM生成

  ┌─────────────────────────────────────┐
  │          知识图谱结构                 │
  │                                       │
  │   [公司A] ──投资──▶ [公司B]           │
  │      │                  │             │
  │    CEO是              产品是           │
  │      ▼                  ▼             │
  │   [人物X] ──毕业于──▶ [大学Y]         │
  │                                       │
  │   查询: "投资公司B的CEO是哪所大学的？"  │
  │   需要: 公司B ← 投资 ← 公司A → CEO    │
  │         → 人物X → 毕业于 → 大学Y       │
  └─────────────────────────────────────┘
```

**GraphRAG 实现要点：**
1. **实体抽取**：从文档中抽取实体和关系（LLM或NER模型）
2. **图谱构建**：构建知识图谱（Neo4j, NetworkX）
3. **社区检测**：对图进行聚类，生成社区摘要
4. **图谱检索**：子图匹配、路径查找、社区搜索

### 5.2 Self-RAG

Self-RAG 让模型**自我判断是否需要检索、检索结果是否有用、回答是否被支持**。

```
Self-RAG 决策流程:
┌─────────────────────────────────────────────┐
│                                               │
│  输入 Query                                   │
│      │                                        │
│      ▼                                        │
│  [Retrieve Token] 是否需要检索？              │
│      ├── No → 直接生成回答                    │
│      └── Yes → 执行检索                       │
│              │                                │
│              ▼                                │
│         [ISREL Token] 检索文档是否相关？       │
│              ├── No → 丢弃，继续检索          │
│              └── Yes → 基于文档生成            │
│                     │                         │
│                     ▼                         │
│            [ISSUP Token] 回答有支撑吗？        │
│                     │                         │
│                     ▼                         │
│            [ISUSE Token] 回答有用吗？          │
│                     │                         │
│                     ▼                         │
│                输出最终答案                     │
└─────────────────────────────────────────────┘
```

::: tip Self-RAG vs 传统RAG
- **传统 RAG**：始终检索，不管是否需要
- **Self-RAG**：按需检索，自我评估质量
- **优势**：减少不必要的检索，避免引入噪声，提升回答质量
- **代价**：需要专门训练的模型（带有反思Token）
:::

---

## 6. RAG 评估指标

### 6.1 评估维度

RAG 系统的评估需要覆盖**检索质量**和**生成质量**两个层面：

```
┌─────────────────────────────────────────────────┐
│              RAG 评估体系                         │
├─────────────────────────────────────────────────┤
│                                                   │
│  检索评估（Context）                              │
│  ├── 召回率（Recall）: 相关文档是否被检索到        │
│  ├── 精确率（Precision）: 检索的文档是否都相关     │
│  ├── MRR（Mean Reciprocal Rank）: 相关文档排名    │
│  └── NDCG: 考虑位置权重的排名质量                 │
│                                                   │
│  生成评估（Answer）                               │
│  ├── 答案相关性（Answer Relevancy）               │
│  ├── 忠实度（Faithfulness）: 答案是否基于上下文    │
│  ├── 答案正确性（Correctness）                    │
│  └── 有害性检测（Harmfulness）                    │
│                                                   │
│  端到端评估                                       │
│  ├── RAGAS 综合分数                               │
│  ├── 人工评估                                     │
│  └── LLM-as-Judge                                │
└─────────────────────────────────────────────────┘
```

### 6.2 核心指标详解

| 指标 | 公式 | 含义 | 目标 |
|------|------|------|------|
| Context Recall | 相关句子数 / 标注答案句子数 | 检索是否覆盖了所需信息 | 越高越好 |
| Context Precision | 相关文档数 / 检索文档总数 | 检索结果的纯净度 | 越高越好 |
| Answer Relevancy | 问题与答案的语义相似度 | 回答是否切题 | 越高越好 |
| Faithfulness | 答案中可验证陈述比例 | 回答是否忠于检索内容 | 越高越好 |

### 6.3 RAGAS 评估框架

RAGAS（Retrieval Augmented Generation Assessment）是目前最流行的 RAG 评估框架：

```python
# RAGAS 评估核心概念（伪代码）
#
# RAGAS Score = Harmonic Mean(
#     Faithfulness,      # 生成内容忠实于检索上下文
#     Answer Relevancy,  # 答案与问题相关
#     Context Precision, # 检索精确率
#     Context Recall     # 检索召回率
# )
```

::: warning 评估注意事项
1. **不要只看单一指标**：高召回率 + 低精确率 = 引入大量噪声
2. **需要标注数据**：Context Recall 需要人工标注的 ground truth
3. **LLM 评估有偏差**：LLM-as-Judge 倾向给长回答更高分
4. **关注实际业务指标**：用户满意度、任务完成率才是最终目标
:::

### 6.4 评估最佳实践

```
构建评估数据集:
├── 收集真实用户问题（至少 100+ 条）
├── 人工标注标准答案和相关文档
├── 覆盖不同难度和类型的问题
└── 定期更新评估集

持续监控:
├── 上线前：离线评估，对比基线
├── 上线后：在线A/B测试
├── 日常：监控检索命中率、用户反馈
└── 迭代：根据 bad case 优化各环节
```

---

## 总结

构建企业级 RAG 系统的关键要素：

| 环节 | 核心要点 | 推荐方案 |
|------|---------|---------|
| 文档解析 | 保留结构信息 | Unstructured + LlamaParse |
| 切片策略 | 语义完整性 | 递归切分 + 语义切分 |
| 检索 | 混合检索 | BM25 + 向量 + RRF融合 |
| 重排 | 精排提质量 | BGE-Reranker-v2 |
| 向量库 | 按规模选型 | Milvus(大) / Qdrant(中) |
| 评估 | 多维度覆盖 | RAGAS + 人工评估 |

::: tip 工程化建议
1. **先跑通再优化**：从最简单的方案开始，逐步迭代
2. **数据质量第一**：好的切片策略 > 好的模型
3. **端到端评估**：每次改动都要有量化指标验证
4. **关注长尾case**：80%的问题都容易解决，难在剩下的20%
:::
