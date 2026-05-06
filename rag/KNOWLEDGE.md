# RAG（检索增强生成）完全知识手册

> 本文档是一份系统性的 RAG 技术教科书，从动机原理到实战优化，覆盖检索增强生成的所有核心知识点。
> 配合 `rag_qa.py` 和 `advanced_rag.py` 代码阅读效果更佳。

---

## 目录

1. [RAG 的动机与原理](#1-rag-的动机与原理)
2. [RAG 完整流程](#2-rag-完整流程)
3. [文档加载与预处理](#3-文档加载与预处理)
4. [文本切分策略](#4-文本切分策略)
5. [Embedding 模型原理与选型](#5-embedding-模型原理与选型)
6. [向量相似度计算](#6-向量相似度计算)
7. [检索器（Retriever）类型与配置](#7-检索器retriever类型与配置)
8. [上下文构造与 Prompt 设计](#8-上下文构造与-prompt-设计)
9. [RAG 链的构建](#9-rag-链的构建)
10. [RAG 评估指标](#10-rag-评估指标)
11. [Naive RAG 的问题与改进方向](#11-naive-rag-的问题与改进方向)

---

## 1. RAG 的动机与原理

### 1.1 为什么需要 RAG

LLM 本质上是一个"闭卷考生"——只能依赖训练时记住的知识来回答问题。这带来三个致命缺陷：

```
┌─────────────────────────────────────────────────────────────────┐
│  缺陷一：知识截止（Knowledge Cutoff）                             │
│    GPT-4 的训练数据截止于某日，之后的新闻它一无所知               │
│                                                                   │
│  缺陷二：无法访问私有数据                                         │
│    公司内部文档、个人笔记、专有数据库——模型从未见过               │
│                                                                   │
│  缺陷三：幻觉（Hallucination）                                    │
│    模型"记忆模糊"时会一本正经地编造事实                           │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 RAG 的核心思想

```
RAG = 开卷考试

  闭卷考试（纯 LLM）：
    学生只能凭"脑子里的记忆"答题 → 容易答错、过时、编造

  开卷考试（RAG）：
    学生可以翻阅参考资料，找到相关段落后再答题 → 有据可查、准确可靠

  核心公式：
    RAG = Retrieval（检索相关文档）+ Augmented Generation（基于文档生成回答）
```

### 1.3 RAG vs 其他方案的对比

| 方案 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| 纯 LLM | 简单直接 | 幻觉、过时 | 通用问答 |
| 微调（Fine-tuning）| 深度适配领域 | 成本高、数据难更新 | 固定领域知识 |
| RAG | 知识可实时更新、可溯源 | 依赖检索质量 | 动态知识库、企业文档 |
| RAG + 微调 | 兼得两者优势 | 复杂度最高 | 生产级系统 |

---

## 2. RAG 完整流程

### 2.1 两阶段架构

```
═══════════════ 离线阶段（Indexing）═══════════════

  原始文档（PDF/TXT/网页/数据库）
       ↓
  文档加载（Document Loader）
       ↓
  文本切分（Text Splitter）→ 多个 Chunk
       ↓
  向量化（Embedding Model）→ 每个 Chunk 变成一个向量
       ↓
  存入向量数据库（Vector Store）

═══════════════ 在线阶段（Query）═══════════════

  用户提问
       ↓
  问题向量化（同一个 Embedding Model）
       ↓
  向量相似度检索（Vector Search）→ Top-K 相关 Chunk
       ↓
  上下文构造（拼接 Chunk + 用户问题 → Prompt）
       ↓
  LLM 生成回答
       ↓
  输出（回答 + 可选的引用来源）
```

### 2.2 对应代码映射（rag_qa.py）

```
离线阶段：
  第1章 → 文档加载 + 文本切块（RecursiveCharacterTextSplitter）
  第2章 → 向量化 + FAISS 建索引（FAISS.from_documents）

在线阶段：
  第3章 → 检索演示（retriever.invoke）
  第4章 → 完整 RAG 链（RunnableParallel + prompt + llm）
```

---

## 3. 文档加载与预处理

### 3.1 LangChain Document 对象

```python
# LangChain 用统一的 Document 对象表示一段文档
from langchain_core.documents import Document

doc = Document(
    page_content="文本内容...",      # 实际文字
    metadata={                       # 元数据（来源、页码等）
        "source": "report.pdf",
        "page": 42,
        "section": "财务数据",
    }
)
```

### 3.2 常用文档加载器

| 加载器 | 数据源 | 用法 |
|--------|--------|------|
| TextLoader | .txt 文件 | `TextLoader("file.txt")` |
| PyPDFLoader | PDF 文件 | `PyPDFLoader("doc.pdf")` |
| WebBaseLoader | 网页 URL | `WebBaseLoader("https://...")` |
| DirectoryLoader | 整个目录 | `DirectoryLoader("./docs/")` |
| CSVLoader | CSV 表格 | `CSVLoader("data.csv")` |
| UnstructuredMarkdownLoader | Markdown | 解析标题层级 |

### 3.3 预处理最佳实践

```
文档预处理流水线：

  原始文档 → 编码检测（UTF-8）→ 去除噪声（页眉页脚、水印）
           → 格式统一（换行符、空白）→ 元数据提取（标题、日期）
           → Document 对象列表
```

---

## 4. 文本切分策略

### 4.1 为什么要切分

```
问题：LLM 上下文窗口有限 + Embedding 对长文本效果差

  一篇 3000 字文章直接嵌入 → 向量"稀释"，语义模糊
  切成 500 字小块 → 每块语义集中，检索更精准

  但也不能切太小：
    太大（>1000字）→ 向量稀释，检索不精准
    太小（<100字）→ 缺少上下文，LLM 无法理解
    推荐范围：300-800 字（中文）或 200-500 tokens
```

### 4.2 RecursiveCharacterTextSplitter（推荐）

```python
# rag_qa.py 中使用的切块器
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 每块最多 500 个字符
    chunk_overlap=50,     # 相邻块重叠 50 字符
    length_function=len,  # 用字符数计算长度
)

# "Recursive"的含义：按优先级尝试不同分隔符
#   优先按段落（\n\n）切 → 切不开按换行（\n）→ 再按句号 → 最后按字符
#   尽量保持语义完整性！
```

### 4.3 其他切分策略

```
┌─────────────────┬──────────────────────┬─────────────────────┐
│ 策略             │ 原理                  │ 适用场景             │
├─────────────────┼──────────────────────┼─────────────────────┤
│ 按字符数切       │ 固定字符长度           │ 简单场景             │
│ 递归字符切       │ 按语义边界递归         │ 通用推荐             │
│ 按 Token 切     │ 用 Tokenizer 计数     │ 控制精确 Token 数    │
│ 按语义切        │ Embedding 计算语义边界 │ 高质量需求           │
│ 按文档结构切    │ 利用标题/段落/列表      │ Markdown/HTML        │
│ 按代码结构切    │ 利用函数/类边界         │ 代码文档             │
└─────────────────┴──────────────────────┴─────────────────────┘
```

### 4.4 chunk_overlap 的作用

```
没有重叠时的问题：

  块1: "...图灵在1950年提出了一个重要测试"
  块2: "这个测试要求机器在对话中骗过人类..."

  问"图灵测试的内容是什么？" → 信息被切断！

有重叠时（overlap=50）：

  块1: "...图灵在1950年提出了一个重要测试，这个测试要求机器在对话..."
  块2: "这个测试要求机器在对话中骗过人类..."

  块1 包含完整信息，可以被正确检索！
```

---

## 5. Embedding 模型原理与选型

### 5.1 Embedding 的本质

```
Embedding = 把文本映射到高维向量空间

  "图灵测试"       → [0.23, -0.11, 0.87, 0.04, ...]  (512维)
  "什么是图灵机？"  → [0.21, -0.09, 0.84, 0.06, ...]  (相似！距离近)
  "今天天气不错"    → [-0.45, 0.72, -0.13, 0.33, ...]  (不相关！距离远)

核心性质：语义相似的文本 → 向量空间中距离更近
```

### 5.2 Embedding 模型工作原理

```
  输入文本 → Tokenizer → Token IDs → Transformer Encoder → 最后一层输出
                                                                ↓
                                                          Pooling（平均/CLS）
                                                                ↓
                                                        固定维度的向量 [d维]
```

### 5.3 主流 Embedding 模型选型

| 模型 | 维度 | 语言 | 特点 | 适用场景 |
|------|------|------|------|---------|
| BAAI/bge-small-zh-v1.5 | 512 | 中文 | 小巧快速，中文优秀 | 教学/原型 |
| BAAI/bge-base-zh-v1.5 | 768 | 中文 | 精度更高 | 中文生产 |
| BAAI/bge-large-zh-v1.5 | 1024 | 中文 | 最高精度 | 对质量要求极高 |
| text-embedding-3-small | 1536 | 多语言 | OpenAI API | 多语言通用 |
| text-embedding-3-large | 3072 | 多语言 | OpenAI 最强 | 质量优先 |
| jina-embeddings-v2 | 768 | 多语言 | 支持 8192 token | 长文本 |
| e5-large-v2 | 1024 | 英文 | 微软出品 | 英文场景 |

### 5.4 rag_qa.py 中的选型

```python
# 本地模型，无需 API Key，完全离线运行
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    # 可选：model_kwargs={"device": "mps"}  # Apple Silicon 加速
)

# 归一化选项（推荐开启，提升余弦相似度稳定性）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)
```

---

## 6. 向量相似度计算

### 6.1 三种距离度量

```
设 A、B 为两个向量：

┌─────────────────────────────────────────────────────────────────┐
│ 余弦相似度（Cosine Similarity）— 最常用                          │
│                                                                   │
│   cos(A, B) = (A · B) / (||A|| × ||B||)                         │
│                                                                   │
│   范围：[-1, 1]，1 = 完全相同方向，0 = 正交，-1 = 完全相反      │
│   特点：只看方向，不看长度 → 对文本长度不敏感                     │
│   适用：归一化后的 Embedding（如 bge 模型）                       │
├─────────────────────────────────────────────────────────────────┤
│ 内积（Inner Product / Dot Product）                              │
│                                                                   │
│   IP(A, B) = A · B = Σ(a_i × b_i)                               │
│                                                                   │
│   范围：(-∞, +∞)                                                 │
│   特点：同时考虑方向和长度                                        │
│   适用：向量已归一化时等价于余弦相似度                            │
├─────────────────────────────────────────────────────────────────┤
│ 欧氏距离（L2 Distance）                                          │
│                                                                   │
│   L2(A, B) = √(Σ(a_i - b_i)²)                                   │
│                                                                   │
│   范围：[0, +∞)，0 = 完全相同                                    │
│   特点：值越小越相似（与前两者相反！）                            │
│   适用：FAISS 默认使用 L2                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 三者关系

```
当向量已归一化（||A|| = ||B|| = 1）时：

  L2²(A, B) = 2 - 2 × cos(A, B) = 2 - 2 × IP(A, B)

  即：余弦相似度越大 ↔ L2 距离越小 ↔ 内积越大

  结论：归一化后三者等价，选哪个都一样！
  所以 bge 模型推荐 normalize_embeddings=True
```

---

## 7. 检索器（Retriever）类型与配置

### 7.1 基础向量检索器

```python
# rag_qa.py 中的用法
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # 返回 Top-3 最相似的文档块
)

# 检索
docs = retriever.invoke("用户的问题")
# 返回: [Document, Document, Document]  （最多 k 个）
```

### 7.2 检索器参数配置

```python
# search_type 选项：
retriever = vectorstore.as_retriever(
    search_type="similarity",        # 默认：纯相似度排序
    # search_type="mmr",             # MMR：兼顾相关性和多样性
    # search_type="similarity_score_threshold",  # 带阈值过滤
    search_kwargs={
        "k": 5,                      # 返回数量
        # "score_threshold": 0.7,    # 相似度阈值（仅 threshold 模式）
        # "fetch_k": 20,             # MMR 初选数量
        # "lambda_mult": 0.5,        # MMR 多样性权重
        # "filter": {"year": 2024},  # 元数据过滤
    }
)
```

### 7.3 MMR（最大边际相关性）

```
MMR 解决的问题：
  纯相似度检索可能返回高度重复的内容

  例：问"公司营收" → 返回3块内容几乎一样的财务段落
  MMR：在相关性和多样性之间取平衡

  MMR(D_i) = λ × Sim(Q, D_i) - (1-λ) × max(Sim(D_i, D_j))
                相关性权重              多样性惩罚

  λ=1.0 → 纯相关性（等价于普通检索）
  λ=0.5 → 平衡相关性和多样性
  λ=0.0 → 纯多样性（最大化差异）
```

---

## 8. 上下文构造与 Prompt 设计

### 8.1 RAG Prompt 的核心模式

```python
# rag_qa.py 中的实现
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个严谨的知识库问答助手。
请仅根据下面提供的【参考资料】来回答用户的问题。
如果参考资料中没有相关信息，请直接说"根据现有资料，我无法回答这个问题"，不要猜测或编造。

【参考资料】
{context}"""),
    ("human", "{question}"),
])
```

### 8.2 Prompt 设计要素

```
一个优秀的 RAG Prompt 包含：

┌─────────────────────────────────────────────────────────┐
│ 1. 角色设定：明确 AI 是"基于文档回答的助手"              │
│ 2. 行为约束："只能根据参考资料回答，不能编造"            │
│ 3. 兜底策略："如果资料中没有，说明无法回答"              │
│ 4. 上下文注入：{context} 占位符放入检索到的文档          │
│ 5. 用户问题：{question} 占位符放入原始问题               │
│ 6. 可选格式要求："请用列表形式回答""控制在200字内"       │
└─────────────────────────────────────────────────────────┘
```

### 8.3 上下文格式化函数

```python
# rag_qa.py 中的实现
def format_docs(docs: list) -> str:
    """把多个检索到的文档块拼接成字符串"""
    return "\n\n---\n\n".join(
        f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
        for doc in docs
    )

# 输出示例：
# [来源: ai_history.txt]
# 1950年，图灵提出了图灵测试...
#
# ---
#
# [来源: ai_history.txt]
# 2017年，Transformer 架构...
```

---

## 9. RAG 链的构建

### 9.1 LCEL 语法（LangChain Expression Language）

```python
# rag_qa.py 第4章的核心代码
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,       # 检索 → 格式化
        question=RunnablePassthrough(),         # 原样传递问题
    )
    | rag_prompt    # 填充 Prompt 模板
    | llm           # 调用 LLM
    | parser        # 提取纯文本
)

# 调用：
answer = rag_chain.invoke("谁提出了图灵测试？")
```

### 9.2 数据流详解

```
输入: "谁提出了图灵测试？"（一个字符串）
      │
      ↓
RunnableParallel（并行执行两条支线）：
      ├── context 支线:
      │     "谁提出了图灵测试？"
      │           ↓ retriever.invoke()
      │     [Doc1, Doc2, Doc3]
      │           ↓ format_docs()
      │     "[来源: xx]\n1950年...\\n---\\n..."（字符串）
      │
      └── question 支线:
            "谁提出了图灵测试？"
                  ↓ RunnablePassthrough()
            "谁提出了图灵测试？"（原样传递）
      │
      ↓ 合并为 dict
{"context": "参考资料...", "question": "谁提出了图灵测试？"}
      │
      ↓ rag_prompt（模板填充）
ChatMessage（完整的带上下文的提示词）
      │
      ↓ llm（调用大模型）
AIMessage("图灵在1950年提出了图灵测试...")
      │
      ↓ parser（StrOutputParser）
"图灵在1950年提出了图灵测试..."（纯字符串）
```

### 9.3 RunnablePassthrough 的作用

```
关键理解：

  RunnableParallel 接收输入后，会把输入分别传给每个分支。
  如果没有 RunnablePassthrough()，"question" 键就不存在了！

  RunnablePassthrough() = "我什么都不做，只是把输入原封不动传下去"

  它不是多余的——它确保 question 字段出现在最终的 dict 中。
```

---

## 10. RAG 评估指标

### 10.1 RAG 三维评估框架

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG 评估三角                                    │
│                                                                   │
│              用户问题 (Question)                                  │
│                 / \                                               │
│                /   \                                              │
│    Context    /     \  Answer                                    │
│   Relevance /       \ Relevance                                  │
│            /         \                                            │
│           /           \                                           │
│   检索上下文 ─────────── 生成回答                                 │
│        (Context)  Faithfulness  (Answer)                         │
│                                                                   │
│  三个维度：                                                       │
│    ① Context Relevance：检索到的内容和问题相关吗？               │
│    ② Faithfulness：回答忠于检索到的内容吗？（没有编造？）        │
│    ③ Answer Correctness：最终回答正确吗？                        │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 各指标详解

```
① Context Relevance（上下文相关性）
   评估"检索"环节的质量
   衡量：检索到的 K 个文档中，有多少是真正相关的？
   公式：relevant_chunks / total_chunks
   提升方向：更好的 Embedding、查询改写、过滤策略

② Faithfulness（忠实度）
   评估"生成"环节是否忠于上下文
   衡量：回答中的每个声明（claim）能否在上下文中找到依据？
   公式：supported_claims / total_claims
   提升方向：更强的 Prompt 约束、温度设为 0

③ Answer Correctness（答案正确性）
   评估端到端效果
   衡量：最终回答与标准答案的一致程度
   方法：人工评估 或 LLM-as-Judge
   提升方向：优化检索 + 优化生成
```

### 10.3 常用评估工具

| 工具 | 特点 | 适用场景 |
|------|------|---------|
| RAGAS | 自动化评估框架 | 标准 RAG 评估 |
| LangSmith | LangChain 官方 | 调试 + 评估 |
| TruLens | 可视化面板 | 实时监控 |
| 人工评估 | 金标准 | 最终验收 |

---

## 11. Naive RAG 的问题与改进方向

### 11.1 Naive RAG 的五大痛点

```
┌──────────────────────────────────────────────────────────────┐
│ 痛点                │ 表现                │ 原因               │
├──────────────────────────────────────────────────────────────┤
│ 1. 语义鸿沟         │ 用户口语 vs 文档术语 │ Embedding 局限     │
│ 2. 召回不全         │ 只找到部分相关文档   │ 单查询视角有限     │
│ 3. 上下文不完整     │ 切块丢失前后文       │ 固定大小切分       │
│ 4. 检索不精准       │ 召回无关内容         │ 缺乏过滤和重排序   │
│ 5. 无法溯源         │ 不知道答案出处       │ 丢失元数据         │
└──────────────────────────────────────────────────────────────┘
```

### 11.2 改进方向总览

```
改进方向（详见 rag_advanced/ 和 advanced_rag.py）：

  语义鸿沟 → HyDE（假设性文档嵌入）
  召回不全 → Multi-Query（多查询扩展）
  上下文不完整 → Parent-Child（父子文档）
  检索不精准 → 重排序（Re-ranking）+ 元数据过滤
  无法溯源 → Metadata 追踪（advanced_rag.py 第6章）

  ┌─────────── RAG 优化方向图 ───────────┐
  │                                        │
  │  检索前优化：                           │
  │    · 查询改写 / 扩展                   │
  │    · HyDE 假设文档                     │
  │                                        │
  │  检索中优化：                           │
  │    · 混合检索（BM25 + 向量）           │
  │    · 元数据过滤                         │
  │    · Parent-Child 结构                 │
  │                                        │
  │  检索后优化：                           │
  │    · 重排序（Cross-Encoder）           │
  │    · 去冗余                            │
  │    · 上下文压缩                         │
  │                                        │
  │  生成优化：                             │
  │    · Prompt 工程                       │
  │    · 引用溯源                          │
  │    · 自反思校验                         │
  │                                        │
  └────────────────────────────────────────┘
```

---

## 附录 A：代码文件与知识点对应

| 代码文件 | 覆盖的知识点 | 对应本文档章节 |
|---------|-------------|---------------|
| `rag_qa.py` 第0章 | LLM + Embedding 初始化 | 第5节 |
| `rag_qa.py` 第1章 | 文档加载 + 文本切块 | 第3-4节 |
| `rag_qa.py` 第2章 | 向量化 + FAISS 建索引 | 第5-6节 |
| `rag_qa.py` 第3章 | 检索器使用演示 | 第7节 |
| `rag_qa.py` 第4章 | RAG 链构建（LCEL） | 第8-9节 |
| `advanced_rag.py` 第4章 | 元数据过滤检索 | 第7节 |
| `advanced_rag.py` 第5章 | 重排序（Re-ranking） | 第11节 |
| `advanced_rag.py` 第6章 | 引用溯源 | 第8节 |

---

## 附录 B：关键概念速查表

```
Chunk       = 文本块（文档切分后的一小段）
Embedding   = 向量化（文本→浮点数列表）
Vector Store= 向量数据库（存储和检索向量）
Retriever   = 检索器（封装了检索逻辑的对象）
Top-K       = 返回最相似的 K 个结果
Context     = 上下文（检索到的文档块拼接成的字符串）
RAG Chain   = 检索→构造上下文→LLM生成 的完整流水线
```

---

> **下一步学习**：阅读 `rag_advanced/KNOWLEDGE.md` 了解 HyDE、Multi-Query 等高级检索策略，或前往 `vectordb/KNOWLEDGE.md` 深入理解向量数据库原理。
