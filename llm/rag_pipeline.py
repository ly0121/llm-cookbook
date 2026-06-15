"""
╔══════════════════════════════════════════════════════════════════╗
║         项目：RAG（检索增强生成）工程化实践全流程                  ║
║         从文档切分、向量索引到检索生成的完整 Pipeline              ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题：LLM 为什么需要 RAG？】
═══════════════════════════════════════════════════════════════════

LLM 有两个根本性问题：
  1. 知识截止：训练数据有时效性，无法回答最新问题
  2. 幻觉：面对不确定的问题，LLM 会"一本正经地胡说八道"

RAG（Retrieval-Augmented Generation）的核心思想：
  先从知识库中"检索"出相关文档，再把文档作为上下文"增强"给 LLM 生成回答。

  ┌─────────────────────────────────────────────────────────────┐
  │              RAG 完整工作流程                                  │
  │                                                               │
  │  【离线阶段：建索引】                                          │
  │    原始文档 → 文档切分 → 向量化(Embedding) → 存入向量数据库    │
  │                                                               │
  │  【在线阶段：问答】                                            │
  │    用户提问                                                    │
  │      ↓                                                        │
  │    问题向量化                                                  │
  │      ↓                                                        │
  │    从向量库中检索 Top-K 相关文档块                              │
  │      ↓                                                        │
  │    重排序（Reranking）筛选最相关的                              │
  │      ↓                                                        │
  │    构建 Prompt = 系统指令 + 检索到的文档 + 用户问题             │
  │      ↓                                                        │
  │    LLM 基于上下文生成回答                                      │
  │      ↓                                                        │
  │    返回答案（附带来源引用）                                     │
  └─────────────────────────────────────────────────────────────┘

  形象比喻：
    LLM 像一个"博学但记忆模糊的教授"
    RAG 就像给教授配了一个"图书管理员"
    教授回答问题前，图书管理员先帮他找到相关资料翻到正确页面
    教授就能基于准确资料给出靠谱的回答了

本文件实现一个完整的 RAG Pipeline，从零到一，每一步都可运行。
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：RAG 架构总览 - 索引、检索、生成三阶段
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import math
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import client, MODEL_NAME

print("=" * 60)
print("第 0 章：RAG 架构总览")
print("=" * 60)
print()
print("""
┌──────────────────────────────────────────────────────────────┐
│              RAG 三大阶段                                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  阶段一：索引（Indexing）                                     │
│    目标：把原始文档转变为可高效检索的结构                       │
│    步骤：文档加载 → 切分 → Embedding → 存储                   │
│                                                              │
│  阶段二：检索（Retrieval）                                    │
│    目标：根据用户问题找到最相关的文档片段                       │
│    方式：向量检索 / 关键词检索 / 混合检索                      │
│                                                              │
│  阶段三：生成（Generation）                                   │
│    目标：基于检索到的上下文，让 LLM 生成准确回答               │
│    关键：Prompt 工程 + 来源引用 + 幻觉控制                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
""")

# ── 准备小型知识库（关于人工智能/机器学习的中文文本）──────────────
# 这是我们的"原始文档"，模拟真实场景中从文件加载的内容

KNOWLEDGE_BASE = """
人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于研究和开发能够模拟、延伸和扩展人类智能的理论、方法和技术。人工智能的研究领域包括机器学习、自然语言处理、计算机视觉、语音识别、专家系统等。

机器学习是人工智能的核心技术之一，它使计算机能够从数据中自动学习规律和模式，而无需被显式编程。机器学习的主要方法包括监督学习、无监督学习和强化学习三大类。监督学习通过标注数据训练模型，无监督学习从无标注数据中发现隐藏结构，强化学习通过与环境交互来学习最优策略。

深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的层次化表示。深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。典型的深度学习模型包括卷积神经网络（CNN）、循环神经网络（RNN）和 Transformer 架构。

Transformer 架构由 Google 在 2017 年的论文《Attention is All You Need》中提出，它完全基于自注意力机制，摒弃了传统的循环结构。Transformer 的核心创新是多头自注意力机制，能够并行处理序列中所有位置的信息，大幅提升了训练效率。

大语言模型（Large Language Model，LLM）是基于 Transformer 架构训练的超大规模语言模型，参数量通常在数十亿到数千亿之间。代表性的大语言模型包括 GPT 系列、Claude、LLaMA 等。LLM 通过海量文本数据的预训练，获得了强大的语言理解和生成能力。

自然语言处理（NLP）是人工智能的重要分支，研究如何让计算机理解、解释和生成人类语言。NLP 的核心任务包括文本分类、命名实体识别、情感分析、机器翻译、文本摘要和问答系统等。近年来，预训练语言模型极大地推动了 NLP 技术的发展。

计算机视觉是让计算机能够"看"并理解图像和视频内容的技术。主要任务包括图像分类、目标检测、图像分割和图像生成。卷积神经网络（CNN）是计算机视觉中最常用的深度学习架构，而近年来 Vision Transformer（ViT）也展现出了强大的性能。

强化学习是一种通过试错来学习最优决策策略的方法。智能体（Agent）在环境中采取动作，根据获得的奖励信号来调整策略。强化学习在游戏AI（如AlphaGo）、机器人控制和自动驾驶等领域有广泛应用。

向量数据库是专门为存储和检索高维向量设计的数据库系统。在 RAG 系统中，文本被转换为向量（embedding），存储在向量数据库中。当需要检索时，通过计算查询向量与存储向量之间的相似度（如余弦相似度）来找到最相关的文档。常见的向量数据库包括 Pinecone、Milvus、Weaviate 和 FAISS。

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合检索和生成的技术框架。它通过从外部知识库中检索相关信息来增强大语言模型的回答质量，有效缓解了 LLM 的知识过时和幻觉问题。RAG 的核心流程包括：文档索引、相关性检索、上下文构建和答案生成四个步骤。
"""

print("知识库已加载，共包含关于 AI/ML 的 10 段文本。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：文档切分策略
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 文档切分是 RAG 的第一步，也是最关键的一步。
# 切分策略直接影响检索质量：
#   - 切太大：检索到的块包含过多无关信息，噪声大
#   - 切太小：语义不完整，丢失上下文
#   - 不重叠：切分边界处的信息可能被割裂
#
#   ┌────────────────────────────────────────────────────────┐
#   │  切分策略对比：                                          │
#   │                                                        │
#   │  固定大小：|===100字===|===100字===|===100字===|        │
#   │    优点：简单快速                                       │
#   │    缺点：可能从句子中间切断                             │
#   │                                                        │
#   │  按段落：  |====段落1====|==段落2==|=====段落3=====|    │
#   │    优点：语义完整                                       │
#   │    缺点：段落长度不均匀                                 │
#   │                                                        │
#   │  递归+重叠：|===chunk1===|                              │
#   │                  |===chunk2===|                         │
#   │                       |===chunk3===|                    │
#   │    优点：边界处信息不丢失                               │
#   │    缺点：存储开销稍大                                   │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 1 章：文档切分策略")
print("=" * 60)
print()


# ── 1.1 固定大小切分 ──────────────────────────────────────────
def chunk_by_fixed_size(text: str, chunk_size: int = 100) -> list[str]:
    """
    固定大小切分：每 chunk_size 个字符切一刀。
    最简单的方法，但可能在句子中间断开。
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# ── 1.2 按段落切分 ────────────────────────────────────────────
def chunk_by_paragraph(text: str) -> list[str]:
    """
    按段落切分：以空行（连续换行）为分隔符。
    保持语义完整性，适合结构清晰的文档。
    """
    # 按连续换行符分段
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


# ── 1.3 递归切分（带重叠）────────────────────────────────────
def chunk_recursive_with_overlap(
    text: str, chunk_size: int = 150, overlap: int = 30
) -> list[str]:
    """
    递归切分（带重叠窗口）：
    - 先尝试按段落切分
    - 对过长的段落，按句子切分
    - 相邻块之间有 overlap 个字符的重叠，防止边界信息丢失

    参数：
        text: 原始文本
        chunk_size: 每个块的目标大小（字符数）
        overlap: 相邻块的重叠字符数
    """
    # 第一级：按段落分
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            # 段落不超过 chunk_size，直接作为一个块
            chunks.append(para)
        else:
            # 段落太长，按句号分句后滑动窗口切分
            sentences = para.replace("。", "。\n").split("\n")
            sentences = [s.strip() for s in sentences if s.strip()]

            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) <= chunk_size:
                    current_chunk += sent
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    # 重叠：取上一个块的尾部作为新块的开头
                    if overlap > 0 and current_chunk:
                        current_chunk = current_chunk[-overlap:] + sent
                    else:
                        current_chunk = sent
            if current_chunk:
                chunks.append(current_chunk)

    return chunks


# ── 对比三种切分策略的效果 ────────────────────────────────────
print("── 1.1 固定大小切分（chunk_size=100）────────────────────")
fixed_chunks = chunk_by_fixed_size(KNOWLEDGE_BASE, chunk_size=100)
print(f"  切分结果：共 {len(fixed_chunks)} 个块")
print(f"  第1块: 「{fixed_chunks[0][:60]}...」")
print(f"  第2块: 「{fixed_chunks[1][:60]}...」")
print()

print("── 1.2 按段落切分 ──────────────────────────────────────")
para_chunks = chunk_by_paragraph(KNOWLEDGE_BASE)
print(f"  切分结果：共 {len(para_chunks)} 个块")
for i, chunk in enumerate(para_chunks[:3]):
    print(f"  第{i+1}块({len(chunk)}字): 「{chunk[:50]}...」")
print()

print("── 1.3 递归切分（chunk_size=150, overlap=30）───────────")
recursive_chunks = chunk_recursive_with_overlap(KNOWLEDGE_BASE, chunk_size=150, overlap=30)
print(f"  切分结果：共 {len(recursive_chunks)} 个块")
for i, chunk in enumerate(recursive_chunks[:3]):
    print(f"  第{i+1}块({len(chunk)}字): 「{chunk[:50]}...」")
print()

print("对比总结：")
print(f"  固定大小切分：{len(fixed_chunks)} 块（简单但语义可能不完整）")
print(f"  按段落切分：  {len(para_chunks)} 块（语义完整，块大小不均匀）")
print(f"  递归+重叠：   {len(recursive_chunks)} 块（兼顾语义完整性和大小均匀性）")
print()

# 后续使用按段落切分的结果（语义最完整）
documents = para_chunks
print(f"后续实验使用按段落切分的 {len(documents)} 个文档块。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：构建向量索引
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 向量索引的核心思想：
#   文本 → Embedding 模型 → 高维向量（如 1536 维）
#   语义相近的文本，在向量空间中距离也近
#
#   ┌────────────────────────────────────────────────────────┐
#   │  文本向量化示意：                                        │
#   │                                                        │
#   │  "机器学习" → [0.12, -0.34, 0.56, ..., 0.78]          │
#   │  "深度学习" → [0.15, -0.31, 0.53, ..., 0.75]  ← 很近！│
#   │  "做饭技巧" → [-0.45, 0.67, -0.12, ..., 0.23] ← 很远  │
#   │                                                        │
#   │  相似度计算：                                            │
#   │    余弦相似度 = cos(θ) = (A·B) / (|A| × |B|)          │
#   │    值域：[-1, 1]，越接近 1 越相似                       │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 2 章：构建向量索引")
print("=" * 60)
print()


def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    调用 Embedding API 将文本列表转换为向量数组。

    参数：
        texts: 要向量化的文本列表

    返回：
        numpy 数组，形状为 (len(texts), embedding_dim)
    """
    print(f"  正在向量化 {len(texts)} 段文本...")

    # 调用 OpenAI 兼容的 embedding API
    response = client.embeddings.create(
        model="text-embedding-ada-002",  # 使用 embedding 模型
        input=texts,
    )

    # 提取向量并转为 numpy 数组
    embeddings = np.array([item.embedding for item in response.data])
    print(f"  向量化完成！向量维度: {embeddings.shape[1]}")
    return embeddings


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    计算查询向量与所有文档向量的余弦相似度。

    余弦相似度公式：
        cos(θ) = (A · B) / (||A|| × ||B||)

    参数：
        query_vec: 查询向量，形状 (dim,)
        doc_vecs: 文档向量矩阵，形状 (n_docs, dim)

    返回：
        相似度数组，形状 (n_docs,)
    """
    # 计算点积
    dot_products = np.dot(doc_vecs, query_vec)

    # 计算模长
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)

    # 余弦相似度 = 点积 / (模长之积)
    similarities = dot_products / (query_norm * doc_norms + 1e-8)
    return similarities


# ── 对文档块进行向量化，构建索引 ───────────────────────────────
print("── 2.1 对知识库文档块进行向量化 ────────────────────────")
print()

# 向量化所有文档块
doc_embeddings = get_embeddings(documents)
print(f"  索引构建完成！")
print(f"  文档数量: {doc_embeddings.shape[0]}")
print(f"  向量维度: {doc_embeddings.shape[1]}")
print(f"  索引大小: {doc_embeddings.nbytes / 1024:.1f} KB")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：检索策略
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 检索是 RAG 的核心环节，检索质量直接决定最终回答质量。
# 主要检索策略：
#
#   ┌────────────────────────────────────────────────────────┐
#   │  1. 向量检索（语义检索）                                │
#   │     原理：用 embedding 的余弦相似度找语义最近的文档     │
#   │     优点：能理解语义（"汽车"能匹配到"轿车"）           │
#   │     缺点：对精确关键词匹配不如传统方法                  │
#   │                                                        │
#   │  2. 关键词检索（BM25 / TF-IDF）                        │
#   │     原理：基于词频和逆文档频率计算相关性                │
#   │     优点：精确匹配能力强，对专有名词友好               │
#   │     缺点：无法理解语义相似性                            │
#   │                                                        │
#   │  3. 混合检索（Hybrid Search）                           │
#   │     原理：向量检索 + 关键词检索 的加权融合              │
#   │     优点：兼顾语义理解和精确匹配                        │
#   │     实践：通常效果最好，是工业界主流方案                │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 3 章：检索策略")
print("=" * 60)
print()


# ── 3.1 向量检索 ─────────────────────────────────────────────
def vector_search(query: str, top_k: int = 3) -> list[tuple[int, float, str]]:
    """
    向量检索：将查询向量化，计算与所有文档的余弦相似度，返回 Top-K。

    参数：
        query: 用户查询文本
        top_k: 返回前 K 个最相关的结果

    返回：
        列表，每项为 (文档索引, 相似度分数, 文档内容)
    """
    # 将查询文本向量化
    query_embedding = get_embeddings([query])[0]

    # 计算与所有文档的余弦相似度
    similarities = cosine_similarity(query_embedding, doc_embeddings)

    # 按相似度降序排列，取 Top-K
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append((int(idx), float(similarities[idx]), documents[idx]))

    return results


# ── 3.2 BM25 关键词检索（手写 TF-IDF 近似实现）────────────────
def tokenize_chinese(text: str) -> list[str]:
    """
    简易中文分词：按标点和空格切分，再按字符 bigram 补充。
    生产环境建议使用 jieba 等专业分词工具。
    """
    import re
    # 去除标点，按空格和标点切分
    text = re.sub(r'[，。！？、；：""''（）【】《》\s]+', ' ', text)
    words = text.split()
    # 对每个词生成字符 bigram（模拟分词效果）
    tokens = []
    for word in words:
        tokens.append(word)
        # 生成 bigram
        if len(word) >= 2:
            for i in range(len(word) - 1):
                tokens.append(word[i:i+2])
    return tokens


def compute_bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """
    BM25 关键词检索：基于词频的经典信息检索算法。

    BM25 公式：
        score(Q, D) = Σ IDF(qi) * (tf(qi, D) * (k1 + 1)) / (tf(qi, D) + k1 * (1 - b + b * |D|/avgdl))

    其中：
        IDF(qi) = log((N - df(qi) + 0.5) / (df(qi) + 0.5))
        tf(qi, D) = 词 qi 在文档 D 中的出现次数
        |D| = 文档 D 的长度
        avgdl = 所有文档的平均长度
        N = 文档总数

    参数：
        query: 查询文本
        docs: 文档列表
        k1: 词频饱和参数（通常 1.2~2.0）
        b: 文档长度归一化参数（通常 0.75）

    返回：
        每个文档的 BM25 分数数组
    """
    N = len(docs)

    # 对所有文档分词
    doc_tokens_list = [tokenize_chinese(doc) for doc in docs]
    doc_lengths = [len(tokens) for tokens in doc_tokens_list]
    avgdl = sum(doc_lengths) / N if N > 0 else 1

    # 计算每个词的文档频率（df）
    df = {}
    for doc_tokens in doc_tokens_list:
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            df[token] = df.get(token, 0) + 1

    # 对查询分词
    query_tokens = tokenize_chinese(query)

    # 计算每个文档的 BM25 分数
    scores = np.zeros(N)
    for qi in query_tokens:
        # 计算 IDF
        dfi = df.get(qi, 0)
        idf = math.log((N - dfi + 0.5) / (dfi + 0.5) + 1)

        for doc_idx, doc_tokens in enumerate(doc_tokens_list):
            # 计算词频 tf
            tf = doc_tokens.count(qi)
            # BM25 公式
            dl = doc_lengths[doc_idx]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avgdl)
            scores[doc_idx] += idf * numerator / denominator

    return scores


def keyword_search(query: str, top_k: int = 3) -> list[tuple[int, float, str]]:
    """
    关键词检索：使用 BM25 算法。

    参数：
        query: 用户查询文本
        top_k: 返回前 K 个结果

    返回：
        列表，每项为 (文档索引, BM25分数, 文档内容)
    """
    scores = compute_bm25_scores(query, documents)

    # 按分数降序排列，取 Top-K
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append((int(idx), float(scores[idx]), documents[idx]))

    return results


# ── 3.3 混合检索（加权融合）───────────────────────────────────
def hybrid_search(
    query: str, top_k: int = 3, vector_weight: float = 0.7, keyword_weight: float = 0.3
) -> list[tuple[int, float, str]]:
    """
    混合检索：将向量检索和关键词检索的分数加权融合。

    策略：
        1. 分别用向量检索和 BM25 检索获取分数
        2. 对两种分数进行 Min-Max 归一化到 [0, 1]
        3. 加权求和得到最终分数
        4. 按最终分数排序返回 Top-K

    参数：
        query: 用户查询文本
        top_k: 返回前 K 个结果
        vector_weight: 向量检索权重（默认 0.7）
        keyword_weight: 关键词检索权重（默认 0.3）

    返回：
        列表，每项为 (文档索引, 混合分数, 文档内容)
    """
    # 获取向量检索分数
    query_embedding = get_embeddings([query])[0]
    vector_scores = cosine_similarity(query_embedding, doc_embeddings)

    # 获取 BM25 分数
    bm25_scores = compute_bm25_scores(query, documents)

    # Min-Max 归一化
    def normalize(scores: np.ndarray) -> np.ndarray:
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s < 1e-8:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)

    norm_vector = normalize(vector_scores)
    norm_bm25 = normalize(bm25_scores)

    # 加权融合
    hybrid_scores = vector_weight * norm_vector + keyword_weight * norm_bm25

    # 按分数降序排列，取 Top-K
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append((int(idx), float(hybrid_scores[idx]), documents[idx]))

    return results


# ── 检索效果对比实验 ──────────────────────────────────────────
test_query = "Transformer 是什么？它有什么创新？"

print(f"── 3.1 向量检索（语义匹配）──────────────────────────────")
print(f"  查询: 「{test_query}」")
print()
vector_results = vector_search(test_query, top_k=3)
for rank, (idx, score, doc) in enumerate(vector_results, 1):
    print(f"  Top-{rank} [相似度={score:.4f}] 文档{idx}: {doc[:60]}...")
print()

print(f"── 3.2 关键词检索（BM25）────────────────────────────────")
print(f"  查询: 「{test_query}」")
print()
keyword_results = keyword_search(test_query, top_k=3)
for rank, (idx, score, doc) in enumerate(keyword_results, 1):
    print(f"  Top-{rank} [BM25={score:.4f}] 文档{idx}: {doc[:60]}...")
print()

print(f"── 3.3 混合检索（向量0.7 + 关键词0.3）──────────────────")
print(f"  查询: 「{test_query}」")
print()
hybrid_results = hybrid_search(test_query, top_k=3)
for rank, (idx, score, doc) in enumerate(hybrid_results, 1):
    print(f"  Top-{rank} [混合={score:.4f}] 文档{idx}: {doc[:60]}...")
print()

print("对比观察：")
print("  - 向量检索擅长捕捉语义相似性（即使用词不同）")
print("  - BM25 擅长精确的关键词匹配")
print("  - 混合检索结合两者优势，通常效果最佳")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：重排序（Reranking）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 检索召回的结果可能包含噪声，重排序用更精确的模型进行二次筛选。
#
#   ┌────────────────────────────────────────────────────────┐
#   │  为什么需要重排序？                                      │
#   │                                                        │
#   │  检索阶段：粗筛，追求召回率（宁可多找，不能漏掉）      │
#   │    → 用轻量模型快速从万级文档中找出 Top-20              │
#   │                                                        │
#   │  重排序阶段：精筛，追求准确率（去掉噪声，留下精华）    │
#   │    → 用重量级模型对 Top-20 精细打分，选出 Top-3         │
#   │                                                        │
#   │  检索(粗) → [doc1, doc2, ..., doc20] → 重排序(精)      │
#   │                                           ↓             │
#   │                                    [doc5, doc2, doc11]   │
#   │                                    （重新排列后的 Top-3）│
#   └────────────────────────────────────────────────────────┘
#
# 本章使用 LLM 作为重排序器：让模型给每个文档的相关性打分。

print("=" * 60)
print("第 4 章：重排序（Reranking）")
print("=" * 60)
print()


def rerank_with_llm(query: str, candidates: list[tuple[int, float, str]], top_k: int = 3) -> list[tuple[int, float, str]]:
    """
    使用 LLM 对检索结果进行重排序。

    方法：让 LLM 对每个候选文档与查询的相关性打分（1-10分）。

    参数：
        query: 用户查询
        candidates: 候选文档列表，每项为 (索引, 原始分数, 文档内容)
        top_k: 重排后返回的文档数

    返回：
        重排后的文档列表
    """
    print(f"  正在用 LLM 对 {len(candidates)} 个候选文档重排序...")

    scored_candidates = []

    for idx, original_score, doc_content in candidates:
        # 构建重排序 prompt，让 LLM 打分
        rerank_prompt = f"""请评估以下文档与查询的相关性。

查询：{query}

文档：{doc_content[:200]}

请给出 1-10 的相关性评分（10 分表示完全相关，1 分表示完全不相关）。
只需要回复一个数字，不要其他内容。"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个相关性评估专家。只需回复一个 1-10 的整数评分。"},
                {"role": "user", "content": rerank_prompt},
            ],
            temperature=0.0,
            max_tokens=5,
        )

        # 解析分数
        score_text = response.choices[0].message.content.strip()
        try:
            score = float(score_text)
            score = max(1, min(10, score))  # 限制在 1-10 范围内
        except ValueError:
            # 如果解析失败，使用原始分数
            score = 5.0

        scored_candidates.append((idx, score, doc_content))

    # 按 LLM 打分降序排列
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    # 返回 Top-K
    results = scored_candidates[:top_k]

    print(f"  重排序完成！")
    return results


# ── 重排序实验 ────────────────────────────────────────────────
rerank_query = "RAG 系统是如何工作的？"
print(f"  查询: 「{rerank_query}」")
print()

# 先用混合检索召回 Top-5 候选
print("  步骤1：混合检索召回 Top-5 候选文档")
candidates = hybrid_search(rerank_query, top_k=5)
for rank, (idx, score, doc) in enumerate(candidates, 1):
    print(f"    候选{rank} [混合分={score:.4f}] 文档{idx}: {doc[:50]}...")
print()

# 用 LLM 重排序
print("  步骤2：LLM 重排序")
reranked = rerank_with_llm(rerank_query, candidates, top_k=3)
print()
print("  重排序结果（按 LLM 相关性评分排序）：")
for rank, (idx, score, doc) in enumerate(reranked, 1):
    print(f"    Top-{rank} [LLM评分={score:.1f}/10] 文档{idx}: {doc[:50]}...")
print()

print("  重排序的价值：LLM 能理解深层语义关系，")
print("  把真正最相关的文档排到前面，过滤掉表面相关但实际无关的文档。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 章：完整 RAG Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 将前面所有模块串联成完整的 Pipeline：
#
#   用户提问
#     ↓
#   混合检索（向量 + BM25）召回 Top-5
#     ↓
#   LLM 重排序，精选 Top-3
#     ↓
#   构建 RAG Prompt（系统指令 + 参考文档 + 用户问题）
#     ↓
#   LLM 生成回答
#     ↓
#   返回答案 + 来源引用

print("=" * 60)
print("第 5 章：完整 RAG Pipeline")
print("=" * 60)
print()


def rag_pipeline(
    question: str,
    retrieve_top_k: int = 5,
    rerank_top_k: int = 3,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    use_rerank: bool = True,
) -> str:
    """
    完整的 RAG Pipeline：问题输入 → 检索 → 重排 → 构建 Prompt → 生成回答。

    参数：
        question: 用户问题
        retrieve_top_k: 检索阶段召回数量
        rerank_top_k: 重排后保留数量
        vector_weight: 向量检索权重
        keyword_weight: 关键词检索权重
        use_rerank: 是否使用重排序

    返回：
        LLM 生成的回答文本
    """
    print(f"╭{'─' * 58}╮")
    print(f"│  RAG Pipeline 开始执行                                    │")
    print(f"│  问题: {question[:45]:<45}│")
    print(f"╰{'─' * 58}╯")
    print()

    # ── 步骤1：混合检索 ────────────────────────────────────────
    print("  [步骤1] 混合检索...")
    candidates = hybrid_search(question, top_k=retrieve_top_k,
                                vector_weight=vector_weight,
                                keyword_weight=keyword_weight)
    print(f"    召回 {len(candidates)} 个候选文档")

    # ── 步骤2：重排序 ──────────────────────────────────────────
    if use_rerank:
        print("  [步骤2] LLM 重排序...")
        final_docs = rerank_with_llm(question, candidates, top_k=rerank_top_k)
    else:
        final_docs = candidates[:rerank_top_k]
    print(f"    最终选取 {len(final_docs)} 个文档")
    print()

    # ── 步骤3：构建 RAG Prompt ─────────────────────────────────
    print("  [步骤3] 构建 RAG Prompt...")

    # 组装参考文档上下文
    context_parts = []
    for i, (idx, score, doc) in enumerate(final_docs, 1):
        context_parts.append(f"[参考文档{i}] {doc}")
    context = "\n\n".join(context_parts)

    # RAG 系统提示词（精心设计，控制幻觉）
    system_prompt = """你是一个严谨的知识问答助手。请基于提供的参考文档回答用户问题。

要求：
1. 只基于参考文档中的信息回答，不要编造文档中没有的内容
2. 如果参考文档中的信息不足以回答问题，请明确说明
3. 回答时引用具体来源（如"根据参考文档1"）
4. 回答要准确、简洁、有条理"""

    user_prompt = f"""参考文档：
{context}

用户问题：{question}

请基于以上参考文档回答用户问题。"""

    print(f"    Prompt 总长度: {len(system_prompt) + len(user_prompt)} 字符")
    print()

    # ── 步骤4：LLM 生成回答 ────────────────────────────────────
    print("  [步骤4] LLM 生成回答...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # 低温度，保证回答准确性
        max_tokens=500,
    )

    answer = response.choices[0].message.content.strip()
    print("  [完成] 回答已生成")
    print()

    # ── 输出结果 ───────────────────────────────────────────────
    print(f"╭{'─' * 58}╮")
    print(f"│  回答：                                                    │")
    print(f"╰{'─' * 58}╯")
    print()
    print(f"  {answer}")
    print()
    print(f"  ──────────────────────────────────────────────────────────")
    print(f"  来源文档：")
    for i, (idx, score, doc) in enumerate(final_docs, 1):
        print(f"    [{i}] 文档{idx}（相关性={score:.1f}）: {doc[:40]}...")
    print()

    return answer


# ── 完整 QA 示例 ──────────────────────────────────────────────
print("=" * 60)
print("完整 QA 示例演示")
print("=" * 60)
print()

# 示例问题 1
print("━" * 60)
print("示例问题 1")
print("━" * 60)
answer1 = rag_pipeline("什么是大语言模型？它是基于什么架构的？")
print()

# 示例问题 2
print("━" * 60)
print("示例问题 2")
print("━" * 60)
answer2 = rag_pipeline("向量数据库在 RAG 中起什么作用？有哪些常见的向量数据库？")
print()

# 示例问题 3（测试知识库外的问题，观察 RAG 如何处理）
print("━" * 60)
print("示例问题 3（知识库边界测试）")
print("━" * 60)
answer3 = rag_pipeline("强化学习在哪些领域有应用？")
print()

# ── 总结 ──────────────────────────────────────────────────────
print("=" * 60)
print("全部实验完毕！RAG Pipeline 核心总结")
print("=" * 60)
print("""
  ┌────────────────────────────────────────────────────────────┐
  │  模块           │ 关键技术              │ 本文实现方式       │
  ├────────────────────────────────────────────────────────────┤
  │  文档切分       │ 固定/段落/递归+重叠   │ 三种策略对比       │
  │  向量化        │ Embedding API         │ OpenAI embedding   │
  │  向量检索      │ 余弦相似度            │ numpy 计算         │
  │  关键词检索    │ BM25                  │ 手写 TF-IDF 近似   │
  │  混合检索      │ 分数归一化 + 加权融合  │ 0.7向量 + 0.3关键词│
  │  重排序        │ Cross-encoder / LLM   │ LLM 相关性打分     │
  │  生成          │ 上下文增强 Prompt      │ 结构化 RAG Prompt  │
  └────────────────────────────────────────────────────────────┘

  工程化最佳实践：
  1. 切分策略要根据文档类型调整（代码用AST切分，文章用段落切分）
  2. 混合检索通常优于单一检索方式
  3. 重排序虽然增加延迟，但显著提升检索精度
  4. RAG Prompt 要明确指示模型"只基于上下文回答"以减少幻觉
  5. 生产环境建议使用专业向量数据库（如 Milvus/FAISS）替代 numpy
""")
