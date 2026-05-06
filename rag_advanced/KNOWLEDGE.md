# RAG 高级检索策略完全知识手册

> 本文档是一份系统性的高级 RAG 技术教科书，覆盖从语义鸿沟问题到自适应 RAG 的全部进阶知识点。
> 配合 `rag_strategies.py` 代码阅读效果更佳。

---

## 目录

1. [语义鸿沟问题](#1-语义鸿沟问题)
2. [HyDE（假设性文档嵌入）](#2-hyde假设性文档嵌入)
3. [Multi-Query（多查询扩展）](#3-multi-query多查询扩展)
4. [Parent-Child 文档结构](#4-parent-child-文档结构)
5. [混合检索（Hybrid Search）](#5-混合检索hybrid-search)
6. [查询改写与扩展](#6-查询改写与扩展)
7. [重排序（Re-ranking）](#7-重排序re-ranking)
8. [自适应 RAG](#8-自适应-rag)
9. [CRAG（Corrective RAG）](#9-cragcorrective-rag)
10. [Self-RAG（自反思 RAG）](#10-self-rag自反思-rag)
11. [Graph RAG（知识图谱增强）](#11-graph-rag知识图谱增强)
12. [多模态 RAG](#12-多模态-rag)
13. [RAG 优化的系统性方法论](#13-rag-优化的系统性方法论)

---

## 1. 语义鸿沟问题

### 1.1 什么是语义鸿沟（Semantic Gap）

```
用户的"问题"和知识库里的"答案"之间存在表述差异：

  用户问: "自动驾驶怎么避免撞人？"          ← 日常口语
  知识库: "感知模块通过多传感器融合实现       ← 技术术语
          行人检测，配合规划模块..."

  ┌─────── 向量空间示意 ───────┐
  │                             │
  │     Q(用户问题)             │
  │        ·                    │    Q 和 D 之间的距离 = 语义鸿沟
  │              (鸿沟)         │
  │                    ·        │
  │               D(知识文档)   │
  │                             │
  └─────────────────────────────┘

为什么 Naive RAG 会在这里失败？
  Embedding 模型虽然能捕捉语义，但当表述差异太大时，
  "撞人" 和 "行人检测/TTC/碰撞风险" 的向量可能距离较远。
```

### 1.2 语义鸿沟的三种表现

```
┌─────────────────┬──────────────────────────────────────────┐
│ 鸿沟类型         │ 示例                                      │
├─────────────────┼──────────────────────────────────────────┤
│ 口语 vs 术语    │ "胡说八道" vs "幻觉(Hallucination)"       │
│ 抽象 vs 具体    │ "怎么让车更聪明" vs "感知算法优化"        │
│ 隐含 vs 显式    │ "这车安全吗" vs "碰撞测试5星/AEB系统"    │
└─────────────────┴──────────────────────────────────────────┘
```

### 1.3 解决语义鸿沟的三种武器

```
┌─────────────── 策略对比 ───────────────┐
│ 策略        │ 解决的问题    │ 代价       │
│─────────────│───────────────│────────────│
│ HyDE        │ 语义鸿沟      │ +1次LLM   │
│ Multi-Query │ 召回率不足    │ +1次LLM   │
│ Parent-Child│ 上下文不完整  │ +存储开销  │
└─────────────────────────────────────────┘
```

---

## 2. HyDE（假设性文档嵌入）

### 2.1 核心思想

```
HyDE = Hypothetical Document Embeddings

传统 Naive RAG:
  Q("怎么避免撞人") ──Embed──→ 搜索 ──→ 可能偏离

HyDE 策略:
  Q("怎么避免撞人")
       │
       ▼ (LLM生成假设性回答)
  H("自动驾驶通过感知模块的行人检测算法，
     结合TTC碰撞时间计算和AEB紧急制动...")
       │
       ▼ (用假设回答做Embedding检索)
  Embed(H) ──→ 搜索 ──→ 更精准！

为什么有效？
  假设回答 H 虽然可能不完全准确，
  但它的"表述风格"和知识库文档一致（都是技术性描述），
  所以在向量空间中更接近真实文档！
```

### 2.2 实现流程（对应 rag_strategies.py Chapter 2）

```python
# Step 1: 构建假设性文档生成 Prompt
hyde_prompt = ChatPromptTemplate.from_template(
    '请针对以下问题，写一段技术性的回答文档。\n'
    '要求：使用专业术语和技术性表述，像教材风格，约100-200字\n'
    '问题: {question}\n'
    '技术文档风格的回答:'
)

hyde_chain = hyde_prompt | llm | StrOutputParser()

# Step 2: 用假设文档做检索
def hyde_retrieve(question, k=3):
    hypothetical_doc = hyde_chain.invoke({'question': question})
    hyde_docs = vectorstore.similarity_search(hypothetical_doc, k=k)
    return hypothetical_doc, hyde_docs
```

### 2.3 适用场景与局限

```
适用场景：
  - 用户口语化提问 + 知识库专业化
  - 跨语言检索（用户中文问，文档英文）
  - 问题描述模糊但领域明确

局限：
  - 多一次 LLM 调用（增加延迟 ~1-3s）
  - 如果 LLM 对领域完全不了解，生成的假设文档质量差
  - 不适合简单的关键词匹配场景
```

---

## 3. Multi-Query（多查询扩展）

### 3.1 核心思想

```
一个问题可能涉及多个知识点，单次检索只能覆盖一个角度：

  原始问题: "自动驾驶怎么避免撞人？"
       │
       ▼  (LLM 改写成多个角度)
  ┌─────────────────────────────┐
  │ Q1: "自动驾驶行人检测技术"    │──→ 检索 ──→ [D1, D2]
  │ Q2: "自动紧急制动AEB原理"    │──→ 检索 ──→ [D2, D3]
  │ Q3: "碰撞预警系统TTC算法"   │──→ 检索 ──→ [D2, D4]
  │ Q4: "感知模块行人轨迹预测"   │──→ 检索 ──→ [D1, D2]
  └─────────────────────────────┘
       │
       ▼  (合并去重)
  最终文档集: [D1, D2, D3, D4]  ← 比单次检索 [D1, D2, D3] 多！

  比喻: 三个侦探从不同方向搜索，总比一个人找得全！
```

### 3.2 实现流程（对应 rag_strategies.py Chapter 3）

```python
# Step 1: 查询扩展 Prompt
multi_query_prompt = ChatPromptTemplate.from_template(
    '请将下面的问题从不同角度改写成4个独立的搜索查询。\n'
    '每个查询关注问题的不同方面或使用不同的技术术语\n'
    '用户问题: {question}\n'
    '4个改写后的搜索查询:'
)

# Step 2: 多次检索 + 合并去重
def multi_query_retrieve(question, k=3):
    raw_queries = multi_query_chain.invoke({'question': question})
    queries = [q.strip() for q in raw_queries.strip().split('\n')]

    all_docs = []
    seen_doc_ids = set()
    for query in queries:
        docs = vectorstore.similarity_search(query, k=k)
        for doc in docs:
            doc_id = doc.metadata.get('doc_id', '')
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                all_docs.append(doc)

    return queries, all_docs
```

### 3.3 Multi-Query vs RAG-Fusion

```
Multi-Query：多次检索 → 合并去重 → 全部传给 LLM
RAG-Fusion：多次检索 → Reciprocal Rank Fusion（RRF）加权排序 → Top-K

RRF 公式：
  RRF_score(d) = Σ 1 / (k + rank_i(d))

  k = 常数（通常 60）
  rank_i(d) = 文档 d 在第 i 次检索中的排名

RAG-Fusion 更优：不仅合并，还根据"被多少次检索命中"来排序。
```

---

## 4. Parent-Child 文档结构

### 4.1 核心思想

```
矛盾：检索精度 vs 上下文完整性

  大块（500字）做 Embedding：
    ✗ 向量被"稀释"，匹配不精准
    ✓ 上下文完整，LLM 能理解全貌

  小块（80字）做 Embedding：
    ✓ 向量集中，匹配精准
    ✗ 缺少上下文，LLM 无法理解

  解决方案：两全其美！
    索引用小块（Child）→ 精确匹配
    返回用大块（Parent）→ 完整上下文

  ┌──────── Parent 文档(完整段落) ────────┐
  │                                        │
  │  ┌─Child 1─┐ ┌─Child 2─┐ ┌─Child 3─┐ │
  │  │ 句子1-2  │ │ 句子3-4  │ │ 句子5-6  │ │
  │  └────┬─────┘ └─────────┘ └─────────┘ │
  │       │                                 │
  └───────│─────────────────────────────────┘
          │ (检索命中 Child 1)
          ▼
    返回整个 Parent 文档给 LLM!
```

### 4.2 实现流程（对应 rag_strategies.py Chapter 4）

```python
# Step 1: 拆分 Parent → Children
def split_into_children(parent_doc, chunk_size=80):
    sentences = split_by_sentence(parent_doc.page_content)
    children = []
    for i in range(0, len(sentences), 2):
        child_doc = Document(
            page_content=''.join(sentences[i:i+2]),
            metadata={
                'parent_id': parent_doc.metadata['doc_id'],
                'child_index': i // 2,
            }
        )
        children.append(child_doc)
    return children

# Step 2: 建立 Child 索引 + Parent 存储
parent_store = {doc.metadata['doc_id']: doc for doc in knowledge_base}
child_vectorstore = FAISS.from_documents(all_children, embeddings)

# Step 3: 检索 Child → 回溯 Parent
def parent_child_retrieve(question, k=3):
    child_hits = child_vectorstore.similarity_search(question, k=k)
    parent_docs = []
    seen = set()
    for child in child_hits:
        pid = child.metadata['parent_id']
        if pid not in seen:
            seen.add(pid)
            parent_docs.append(parent_store[pid])
    return parent_docs
```

### 4.3 变体：Multi-Vector Retriever

```
更通用的思路：为同一个文档生成多种"表示"用于检索

  文档 D → 摘要（Summary）→ Embed → 索引
  文档 D → 关键问题（Questions）→ Embed → 索引
  文档 D → 关键词（Keywords）→ Embed → 索引

  检索命中任一表示 → 返回原始文档 D

  LangChain 提供 MultiVectorRetriever 实现这一模式。
```

---

## 5. 混合检索（Hybrid Search）

### 5.1 BM25 vs 向量检索

```
┌─────────────────────────────────────────────────────────────────┐
│                  BM25（关键词检索）                               │
│                                                                   │
│  原理：基于词频(TF)和逆文档频率(IDF)的匹配                       │
│  公式：BM25(Q,D) = Σ IDF(q_i) × TF(q_i,D) × (k1+1)            │
│                              / (TF(q_i,D) + k1×(1-b+b×|D|/avgdl))│
│                                                                   │
│  优势：精确匹配专有名词、代码、ID 等                             │
│  劣势：无法理解同义词、不同表述                                   │
├─────────────────────────────────────────────────────────────────┤
│                  向量检索（语义检索）                             │
│                                                                   │
│  原理：计算 Query 和 Document 的 Embedding 余弦相似度            │
│  优势：理解语义、同义词、近义表述                                 │
│  劣势：精确匹配弱（"iPhone 15" 可能匹配到 "手机"）              │
├─────────────────────────────────────────────────────────────────┤
│                  混合检索 = BM25 + 向量                          │
│                                                                   │
│  精确匹配（专有名词）+ 语义理解（同义词）= 最佳效果              │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 混合检索的融合策略

```
方法一：线性加权
  final_score = α × BM25_score + (1-α) × vector_score
  α = 0.3~0.5 通常效果较好

方法二：Reciprocal Rank Fusion (RRF)
  RRF_score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_vector(d))
  不需要对齐分数范围，更稳定

方法三：交叉排序
  BM25 Top-20 ∪ Vector Top-20 → Reranker 重排序 → Top-5
```

### 5.3 实现方式

```python
# LangChain 的 EnsembleRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(docs, k=5)
vector = vectorstore.as_retriever(search_kwargs={"k": 5})

# 权重: BM25 占 40%, 向量占 60%
hybrid = EnsembleRetriever(
    retrievers=[bm25, vector],
    weights=[0.4, 0.6],
)
```

---

## 6. 查询改写与扩展

### 6.1 查询改写（Query Rewriting）

```
目的：把用户的模糊/口语化问题改写为适合检索的精确查询

  原始: "这个东西怎么用？"（指代不明）
  改写: "LangChain 的 RecursiveCharacterTextSplitter 使用方法"

实现：用 LLM 改写
  prompt = "请将以下问题改写为更清晰、具体的搜索查询：{question}"
```

### 6.2 查询扩展（Query Expansion）

```
目的：为查询添加同义词、相关概念，扩大检索范围

  原始: "自动驾驶感知"
  扩展: "自动驾驶感知 OR 传感器融合 OR 目标检测 OR LiDAR点云处理"

方法：
  ① 同义词扩展（规则型）
  ② LLM 生成相关术语
  ③ Pseudo-Relevance Feedback（用首轮检索结果中的关键词扩展）
```

### 6.3 Step-Back Prompting

```
思路：先问一个更宏观的问题，获取背景知识，再回答具体问题

  用户问: "Llama-3-70B 的训练需要多少 GPU？"
  Step-back: "大语言模型训练的硬件需求有哪些？"

  先检索 step-back 问题获取背景 → 再结合原始问题的检索结果 → 生成回答
```

---

## 7. 重排序（Re-ranking）

### 7.1 为什么需要重排序

```
类比高考录取：

  第一轮：初试（向量检索）
    从 10 万个文本块中，快速筛出 20 个"大致相关"的候选块。
    方式：计算向量余弦相似度，速度极快（毫秒级）。
    缺点：只看"大致意思"，可能混入不太相关的内容。

  第二轮：复试（重排序）
    对初试选出的 20 个候选块，逐一精细打分，选出 Top-3。
    方式：Cross-Encoder 把"问题+候选块"配对打分。
    优点：准确率远高于初试。
    缺点：速度慢（要逐个打分），只能对少量候选做。
```

### 7.2 Bi-Encoder vs Cross-Encoder

```
┌─────────────────────────────────────────────────────────────────┐
│  Bi-Encoder（双编码器）—— 用于初选                               │
│                                                                   │
│    Query ──→ [Encoder] ──→ q_vec ─┐                             │
│                                     ├── cosine(q_vec, d_vec)     │
│    Doc   ──→ [Encoder] ──→ d_vec ─┘                             │
│                                                                   │
│    特点：Query 和 Doc 独立编码，可以预计算 Doc 向量              │
│    速度：极快（向量已预存，只需算 Query 向量 + 查表）            │
│    精度：中等                                                     │
├─────────────────────────────────────────────────────────────────┤
│  Cross-Encoder（交叉编码器）—— 用于重排序                        │
│                                                                   │
│    [Query + Doc] ──→ [Encoder] ──→ relevance_score               │
│                                                                   │
│    特点：Query 和 Doc 拼接后一起编码，充分交互                   │
│    速度：慢（每对 Query-Doc 都要过一次模型）                     │
│    精度：高（能捕捉细粒度的匹配关系）                            │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 主流重排序方案

| 方案 | 类型 | 特点 | 适用场景 |
|------|------|------|---------|
| BAAI/bge-reranker-base | Cross-Encoder | 中文优秀，本地部署 | 中文 RAG |
| Cohere Reranker | API 服务 | 效果顶级，多语言 | 生产环境 |
| cross-encoder/ms-marco | Cross-Encoder | 英文基准 | 英文 RAG |
| EmbeddingsFilter | Bi-Encoder | 轻量级，无需额外模型 | 快速原型 |
| FlashRank | 本地 | 极速，效果接近 Cohere | 低延迟需求 |

### 7.4 代码实现（对应 advanced_rag.py 第5章）

```python
# 轻量级方案：EmbeddingsFilter（不需要额外模型）
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

relevance_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.3,  # 低于此阈值的文档被丢弃
)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=relevance_filter,
    base_retriever=base_retriever,  # 初选 k=6
)
# 流程：初选(k=6) → 相关性过滤 → 精选结果（通常 < 6 个）
```

---

## 8. 自适应 RAG

### 8.1 核心思想

```
不是所有问题都需要相同的 RAG 策略：

  简单事实查询: "公司成立于哪年？"
    → 直接向量检索即可，不需要 HyDE 或 Multi-Query

  复杂分析问题: "公司的技术路线和竞争对手有什么区别？"
    → 需要 Multi-Query + 多文档综合

  闲聊/不需要检索: "你好，今天天气怎么样？"
    → 根本不需要 RAG，直接让 LLM 回答

自适应 RAG = 根据查询类型动态选择策略
```

### 8.2 查询分类器

```python
# 用 LLM 判断查询类型
classify_prompt = """
判断以下问题属于哪个类型：
A. 简单事实查询（直接检索）
B. 复杂分析问题（Multi-Query + 重排序）
C. 不需要检索（闲聊/常识）

问题: {question}
类型（只输出字母）:
"""

# 根据类型选择策略
if query_type == "A":
    docs = simple_retriever.invoke(question)
elif query_type == "B":
    docs = multi_query_retrieve(question)
else:
    # 直接 LLM 回答，不检索
    answer = llm.invoke(question)
```

---

## 9. CRAG（Corrective RAG）

### 9.1 核心思想

```
Corrective RAG 的关键洞察：
  检索到的文档不一定是正确的！需要"校验"步骤。

流程：
  用户问题 → 检索 → 评估检索质量 → 决定下一步

  ┌─────────────────────────────────────────────────────┐
  │                                                       │
  │  检索文档 → [质量评估器] → 三种情况:                  │
  │                                                       │
  │    ① 文档相关且充分 → 直接生成回答                    │
  │    ② 文档部分相关   → 提取有用部分 + 补充搜索        │
  │    ③ 文档不相关     → 放弃检索结果 + Web搜索补充     │
  │                                                       │
  └─────────────────────────────────────────────────────┘
```

### 9.2 质量评估方法

```
方法一：LLM 评估
  prompt = "以下文档是否能回答该问题？评分1-5"

方法二：相似度阈值
  if max_similarity < 0.6: 判定为不相关

方法三：NLI（自然语言推理）模型
  判断文档和问题之间是否存在蕴含关系
```

---

## 10. Self-RAG（自反思 RAG）

### 10.1 核心思想

```
Self-RAG 让模型在生成过程中"自我反思"：

  普通 RAG:
    检索 → 生成 → 输出（一步到位，不检查质量）

  Self-RAG:
    检索 → 生成 → [自检: 需要检索吗？] → [检索相关吗？]
                  → [回答忠实吗？] → [回答有用吗？]
                  → 如果不满意，重新检索/重新生成

  四个反思标记（Reflection Tokens）：
    [Retrieve]   : 是否需要检索？（判断当前生成是否需要外部知识）
    [IsREL]      : 检索到的文档是否相关？
    [IsSUP]      : 生成的内容是否被文档支持（忠实度）？
    [IsUSE]      : 最终回答是否有用？
```

### 10.2 与标准 RAG 的区别

```
标准 RAG：总是检索 → 总是用检索结果
Self-RAG：按需检索 → 验证后才使用

优势：
  - 不需要检索时不检索（避免噪声引入）
  - 能发现检索结果不相关并纠正
  - 能发现自己的幻觉并修正
```

---

## 11. Graph RAG（知识图谱增强）

### 11.1 为什么需要 Graph RAG

```
向量检索的局限：只能找到"语义相似"的文档

  问: "A公司CEO的母校在哪个城市？"

  向量检索可能找到:
    - "A公司CEO是张三"
    - "张三毕业于北京大学"

  但不能自动推理: CEO = 张三 → 母校 = 北大 → 城市 = 北京

  知识图谱可以！
    [A公司] --CEO--> [张三] --毕业于--> [北京大学] --位于--> [北京]
```

### 11.2 Graph RAG 的实现方式

```
方式一：知识图谱 + 向量混合检索
  - 先用向量检索找到相关实体
  - 再在知识图谱中做关系推理
  - 合并两路结果

方式二：文档图谱
  - 用 LLM 从文档中提取实体和关系
  - 构建文档间的引用/关联图
  - 检索时沿图谱扩展相关文档

方式三：Microsoft GraphRAG
  - 对文档做社区检测（Leiden算法）
  - 为每个社区生成摘要
  - 全局问题用社区摘要回答，局部问题用原始文档
```

---

## 12. 多模态 RAG

### 12.1 多模态 RAG 的挑战

```
传统 RAG 只处理文本，但真实文档包含：
  - 表格（Table）
  - 图片（Image）
  - 图表（Chart）
  - 公式（Formula）

多模态 RAG 需要：
  ① 从非文本元素中提取信息
  ② 将多模态内容统一表示为可检索的形式
  ③ 在生成时综合利用多模态信息
```

### 12.2 实现策略

```
策略一：多模态 Embedding
  - 用 CLIP/SigLIP 等模型把图片也编码为向量
  - 文本和图片在同一向量空间中检索

策略二：多模态 LLM 提取
  - 用 GPT-4V/Gemini 把图片"描述"为文本
  - 然后按标准文本 RAG 流程处理

策略三：结构化提取
  - 表格 → 转为 Markdown/JSON
  - 图表 → OCR + LLM 解读
  - 公式 → LaTeX 转文本描述
```

---

## 13. RAG 优化的系统性方法论

### 13.1 优化层次图

```
┌─────────────────── RAG 系统优化全景图 ───────────────────┐
│                                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Layer 1: 数据层                                    │   │
│  │    · 文档清洗与预处理                               │   │
│  │    · 切分策略优化（chunk_size 调优）                │   │
│  │    · 元数据设计                                     │   │
│  └────────────────────────────────────────────────────┘   │
│                           ↓                                │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Layer 2: 索引层                                    │   │
│  │    · Embedding 模型选型                             │   │
│  │    · 索引结构（Flat/HNSW/IVF）                     │   │
│  │    · Parent-Child / Multi-Vector                    │   │
│  └────────────────────────────────────────────────────┘   │
│                           ↓                                │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Layer 3: 检索层                                    │   │
│  │    · 查询改写 / HyDE / Multi-Query                 │   │
│  │    · 混合检索（BM25 + Vector）                     │   │
│  │    · 元数据过滤                                     │   │
│  └────────────────────────────────────────────────────┘   │
│                           ↓                                │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Layer 4: 后处理层                                  │   │
│  │    · 重排序（Cross-Encoder）                        │   │
│  │    · 去冗余                                         │   │
│  │    · 上下文压缩                                     │   │
│  └────────────────────────────────────────────────────┘   │
│                           ↓                                │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Layer 5: 生成层                                    │   │
│  │    · Prompt 模板优化                                │   │
│  │    · 引用溯源                                       │   │
│  │    · 自反思校验（Self-RAG / CRAG）                 │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 13.2 组合使用建议

```
┌─────────────────── 组合使用建议 ───────────────────┐
│                                                      │
│  推荐组合 1 (高质量):                                │
│    HyDE + Parent-Child + Reranking                  │
│    → 弥合语义鸿沟 + 完整上下文 + 精选结果           │
│                                                      │
│  推荐组合 2 (高召回):                                │
│    Multi-Query + 混合检索 + Reranking               │
│    → 多角度覆盖 + 精确+语义双路 + 精选              │
│                                                      │
│  推荐组合 3 (生产级):                                │
│    自适应RAG + 混合检索 + Reranking + CRAG          │
│    → 按需选策略 + 双路检索 + 质量校验               │
│                                                      │
│  推荐组合 4 (全能型):                                │
│    Multi-Query + HyDE + Parent-Child + Reranking    │
│    → 覆盖所有场景，但成本最高                       │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 13.3 选型决策树

```
你的问题是什么？
  │
  ├─ "用户总用口语问，文档很专业"
  │   → 用 HyDE
  │
  ├─ "一个问题涉及多个知识点"
  │   → 用 Multi-Query
  │
  ├─ "检索到了但回答不完整"
  │   → 用 Parent-Child
  │
  ├─ "专有名词检索不到"
  │   → 加 BM25 混合检索
  │
  ├─ "检索结果中噪声多"
  │   → 加重排序（Reranker）
  │
  ├─ "有时不需要检索"
  │   → 用自适应 RAG
  │
  └─ "回答可能有幻觉"
      → 用 CRAG / Self-RAG
```

---

## 附录 A：代码文件与知识点对应

| 代码文件 | 覆盖的知识点 | 对应本文档章节 |
|---------|-------------|---------------|
| `rag_strategies.py` Chapter 0 | 知识库构建、语义鸿沟演示 | 第1节 |
| `rag_strategies.py` Chapter 1 | Naive RAG 基线与局限 | 第1节 |
| `rag_strategies.py` Chapter 2 | HyDE 完整实现 | 第2节 |
| `rag_strategies.py` Chapter 3 | Multi-Query 完整实现 | 第3节 |
| `rag_strategies.py` Chapter 4 | Parent-Child 完整实现 | 第4节 |

---

> **下一步学习**：阅读 `vectordb/KNOWLEDGE.md` 深入理解向量数据库的索引算法和生产选型，或前往 `agent/KNOWLEDGE.md` 了解如何将 RAG 与 Agent 框架结合。
