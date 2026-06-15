"""
╔══════════════════════════════════════════════════════════════════╗
║         项目：Embedding（词向量）原理与实战全面实验              ║
║         探索文本向量化、相似度计算、语义搜索与 RAG 应用          ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题：计算机怎么"理解"文本的含义？】
═══════════════════════════════════════════════════════════════════

人类理解语言靠语义，计算机理解语言靠数字。
Embedding（词向量）就是把文本映射到高维空间中的一个点（向量），
使得语义相近的文本在空间中距离也近。

  "今天天气真好"  →  [0.12, -0.34, 0.78, ..., 0.56]  (1536维向量)
  "今天阳光明媚"  →  [0.11, -0.32, 0.76, ..., 0.55]  (距离很近！)
  "量子力学很难"  →  [-0.45, 0.67, -0.23, ..., 0.12] (距离很远！)

  ┌─────────────────────────────────────────────────────────────┐
  │  Embedding 的本质：                                          │
  │                                                             │
  │  文本（离散符号）                                            │
  │    ↓                                                        │
  │  Embedding 模型（深度神经网络）                              │
  │    ↓                                                        │
  │  高维向量（连续数值）                                        │
  │    ↓                                                        │
  │  语义相近的文本 → 向量距离近（余弦相似度高）                 │
  │  语义不同的文本 → 向量距离远（余弦相似度低）                 │
  └─────────────────────────────────────────────────────────────┘

  Embedding 的典型应用：
    1. 语义搜索 —— 用向量距离找到最相关的文档
    2. RAG（检索增强生成）—— 先检索再生成，让 LLM 基于事实回答
    3. 文本聚类 —— 把相似文本自动归类
    4. 推荐系统 —— 找到用户可能感兴趣的内容
    5. 异常检测 —— 找到与众不同的文本

本文件通过真实 API 调用，带你亲手体验 Embedding 的强大能力。
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：Embedding 概念总览
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import client, MODEL_NAME

print("=" * 60)
print("第 0 章：Embedding 概念总览")
print("=" * 60)
print()
print("""
┌──────────────────────────────────────────────────────────────┐
│            Embedding（词向量）核心概念图                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  为什么需要 Embedding？                                       │
│                                                              │
│  计算机不认识"文字"，只认识"数字"。                           │
│  传统方法（one-hot编码）：                                    │
│    "猫" → [1,0,0,0,...]   "狗" → [0,1,0,0,...]              │
│    问题：无法表达"猫和狗都是动物"这种语义关系！               │
│                                                              │
│  Embedding 方法：                                             │
│    "猫" → [0.8, 0.3, -0.1, ...]                              │
│    "狗" → [0.7, 0.4, -0.2, ...]  ← 向量接近！               │
│    "汽车" → [-0.5, 0.9, 0.6, ...] ← 向量远离！              │
│                                                              │
│  ┌─────── 二维简化示意（实际是上千维）───────┐                │
│  │         * 猫                              │                │
│  │        * 狗                               │                │
│  │       * 兔子                              │                │
│  │                                           │                │
│  │                     * 汽车                │                │
│  │                    * 飞机                  │                │
│  └───────────────────────────────────────────┘                │
│  语义相近的词在空间中聚在一起！                                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：获取文本 Embedding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 调用 Embedding API 的流程：
#   1. 准备文本（可以是一段话、一个句子、甚至一个词）
#   2. 调用 client.embeddings.create() 接口
#   3. 得到一个高维浮点数向量（通常 1536 维或更多）
#   4. 这个向量就是文本的"数字指纹"——语义的数学表达
#
#   注意事项：
#   - 同一段文本每次获取的 embedding 结果是固定的（确定性的）
#   - 不同 embedding 模型产生的向量维度可能不同
#   - 向量的每一维没有明确的语义含义，但整体编码了语义信息

print("=" * 60)
print("第 1 章：获取文本 Embedding")
print("=" * 60)
print()

# ── 1.1 获取单条文本的 embedding ──────────────────────────
print("── 1.1 获取单条文本的 embedding ──────────────────────")
print()

sample_text = "人工智能正在改变世界"
print(f"  输入文本: 「{sample_text}」")
print()

# 调用 embedding API
response = client.embeddings.create(
    input=sample_text,
    model=MODEL_NAME,  # 使用当前可用的模型
)

# 提取 embedding 向量
embedding_vector = response.data[0].embedding

print(f"  向量维度: {len(embedding_vector)} 维")
print(f"  前 10 个值: {embedding_vector[:10]}")
print(f"  向量类型: {type(embedding_vector)}")
print()

# ── 1.2 批量获取多条文本的 embedding ─────────────────────
print("── 1.2 批量获取多条文本的 embedding ─────────────────")
print()

texts = [
    "今天天气真好",
    "机器学习是人工智能的分支",
    "我喜欢吃火锅",
]

# 批量调用：一次请求获取多个文本的 embedding
response_batch = client.embeddings.create(
    input=texts,
    model=MODEL_NAME,
)

print("  批量处理结果：")
for i, data in enumerate(response_batch.data):
    vec = data.embedding
    print(f"  [{i}] 「{texts[i]}」")
    print(f"      维度: {len(vec)}, 前5个值: {vec[:5]}")
print()

print("  要点：")
print("  - 一次 API 调用可以处理多条文本，比逐条调用更高效")
print("  - 每条文本都会得到一个独立的向量")
print("  - 向量维度取决于模型，常见有 768、1024、1536 等")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：余弦相似度计算
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 有了向量之后，如何衡量两段文本的语义相似程度？
# 最常用的方法是"余弦相似度"（Cosine Similarity）。
#
# 数学公式：
#   cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
#
#   其中：
#     A · B     = 向量点积（对应位置相乘再求和）
#     ||A||     = 向量的 L2 范数（各分量平方和再开根号）
#
# 取值范围：[-1, 1]
#   1   → 方向完全相同（语义最相近）
#   0   → 正交（语义无关）
#   -1  → 方向完全相反（语义相反）
#
# 实际应用中，文本 embedding 的余弦相似度通常在 [0, 1] 范围内。
#
#   ┌────────────────────────────────────────────────────────┐
#   │  形象比喻：                                             │
#   │                                                        │
#   │  两个向量像两根指针：                                   │
#   │    - 指向同一方向 → 余弦相似度 ≈ 1（非常相似）          │
#   │    - 垂直 → 余弦相似度 ≈ 0（不相关）                   │
#   │    - 指向相反方向 → 余弦相似度 ≈ -1（非常不同）         │
#   │                                                        │
#   │         A↗  B↗       A↗              A↗  B↙           │
#   │       相似度≈1        ↑B 相似度≈0      相似度≈-1        │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 2 章：余弦相似度计算")
print("=" * 60)
print()


# ── 手写余弦相似度函数 ────────────────────────────────────
def cosine_similarity(vec_a, vec_b):
    """
    计算两个向量的余弦相似度。

    参数：
        vec_a: 第一个向量（list 或 numpy array）
        vec_b: 第二个向量（list 或 numpy array）

    返回：
        余弦相似度值，范围 [-1, 1]
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    # 计算点积
    dot_product = np.dot(a, b)

    # 计算各自的 L2 范数（模长）
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # 避免除零错误
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # 余弦相似度 = 点积 / (模长之积)
    return dot_product / (norm_a * norm_b)


# ── 2.1 对比相似文本和不相似文本 ──────────────────────────
print("── 2.1 对比相似文本 vs 不相似文本 ────────────────────")
print()

# 准备测试文本对
text_pairs = [
    ("我喜欢吃苹果", "我爱吃水果"),           # 语义相近
    ("今天天气很好", "今天阳光明媚"),           # 语义很相近
    ("机器学习很有趣", "深度学习是AI的前沿"),   # 语义相关
    ("我喜欢吃苹果", "量子物理学很复杂"),       # 语义不相关
    ("今天天气很好", "股票市场今天暴跌"),       # 语义不相关
]

# 获取所有文本的 embedding
all_texts = []
for pair in text_pairs:
    all_texts.extend(pair)

response_all = client.embeddings.create(
    input=all_texts,
    model=MODEL_NAME,
)

# 提取向量
all_embeddings = [data.embedding for data in response_all.data]

# 计算每对的相似度
print("  文本对相似度对比：")
print("  " + "─" * 56)
for i, (text_a, text_b) in enumerate(text_pairs):
    vec_a = all_embeddings[i * 2]
    vec_b = all_embeddings[i * 2 + 1]
    similarity = cosine_similarity(vec_a, vec_b)

    # 用可视化条形图表示相似度
    bar_length = int(similarity * 30)
    bar = "█" * bar_length + "░" * (30 - bar_length)

    print(f"  「{text_a}」")
    print(f"  「{text_b}」")
    print(f"   相似度: {similarity:.4f}  [{bar}]")
    print("  " + "─" * 56)

print()
print("  观察要点：")
print("  - 语义相近的文本对，余弦相似度明显更高（通常 > 0.8）")
print("  - 语义无关的文本对，余弦相似度较低（通常 < 0.5）")
print("  - 这正是 Embedding 的核心价值：把语义距离转化为数值距离")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：语义搜索实现
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 语义搜索的核心思想：
#   传统关键词搜索：看文本中有没有包含搜索词（字面匹配）
#   语义搜索：看文本的含义是否与搜索意图相近（语义匹配）
#
#   例如搜索"如何减肥"：
#     关键词搜索 → 只找到包含"减肥"的文档
#     语义搜索 → 还能找到"控制饮食的方法"、"有氧运动的好处"等
#
#   实现步骤：
#     1. 离线阶段：为知识库中所有文档计算 embedding 并存储
#     2. 在线阶段：用户输入 query → 计算 query embedding
#                 → 与所有文档 embedding 计算相似度
#                 → 返回最相似的 Top-K 结果
#
#   ┌────────────────────────────────────────────────────────┐
#   │                语义搜索流程图                            │
#   │                                                        │
#   │  知识库文档         用户查询                             │
#   │    ↓                  ↓                                │
#   │  [Embedding]      [Embedding]                          │
#   │    ↓                  ↓                                │
#   │  向量数据库  ←── 余弦相似度比较                         │
#   │    ↓                                                   │
#   │  返回 Top-K 最相似文档                                  │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 3 章：语义搜索实现")
print("=" * 60)
print()

# ── 3.1 构建知识库 ────────────────────────────────────────
print("── 3.1 构建知识库（计算所有文档的 embedding）──────────")
print()

# 模拟一个简单的知识库（关于健康和科技的混合话题）
knowledge_base = [
    "Python 是一种简单易学的编程语言，广泛用于数据分析和人工智能。",
    "每天坚持跑步30分钟可以有效提高心肺功能和免疫力。",
    "深度学习通过多层神经网络来学习数据的复杂特征表示。",
    "多吃蔬菜水果，少吃油腻食物，有助于保持身体健康。",
    "Transformer 模型是当前自然语言处理的主流架构。",
    "充足的睡眠对大脑记忆力和注意力有重要影响。",
    "大语言模型（LLM）通过海量文本数据进行预训练，具备强大的语言理解能力。",
    "瑜伽和冥想可以帮助减轻压力，改善心理健康。",
]

print("  知识库内容（共 {} 条）：".format(len(knowledge_base)))
for i, doc in enumerate(knowledge_base):
    print(f"    [{i}] {doc}")
print()

# 为知识库所有文档计算 embedding（离线阶段）
print("  正在为知识库计算 embedding...")
kb_response = client.embeddings.create(
    input=knowledge_base,
    model=MODEL_NAME,
)
kb_embeddings = [data.embedding for data in kb_response.data]
print(f"  完成！共计算 {len(kb_embeddings)} 个向量，每个 {len(kb_embeddings[0])} 维")
print()


# ── 3.2 语义搜索函数 ─────────────────────────────────────
def semantic_search(query, documents, doc_embeddings, top_k=3):
    """
    语义搜索：根据 query 找到最相关的文档。

    参数：
        query: 用户的搜索查询
        documents: 文档列表
        doc_embeddings: 文档对应的 embedding 列表
        top_k: 返回前几个最相关的结果

    返回：
        排序后的 (文档, 相似度) 列表
    """
    # 计算 query 的 embedding
    query_response = client.embeddings.create(
        input=query,
        model=MODEL_NAME,
    )
    query_embedding = query_response.data[0].embedding

    # 计算 query 与每个文档的相似度
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((i, documents[i], sim))

    # 按相似度从高到低排序
    similarities.sort(key=lambda x: x[2], reverse=True)

    # 返回 Top-K
    return similarities[:top_k]


# ── 3.3 测试语义搜索 ─────────────────────────────────────
print("── 3.2 测试语义搜索 ──────────────────────────────────")
print()

# 测试不同的查询
queries = [
    "怎样学编程",
    "如何保持健康",
    "什么是大模型",
]

for query in queries:
    print(f"  查询: 「{query}」")
    print(f"  {'─' * 50}")
    results = semantic_search(query, knowledge_base, kb_embeddings, top_k=3)
    for rank, (idx, doc, score) in enumerate(results, 1):
        print(f"    Top{rank} [相似度:{score:.4f}] {doc}")
    print()

print("  观察要点：")
print("  - '怎样学编程' 能找到 Python 相关文档，即使没出现'编程'这个词的变体")
print("  - '如何保持健康' 能找到运动、饮食、睡眠相关的文档")
print("  - 语义搜索能理解'意思'，而不仅仅是匹配'关键词'")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：Embedding 在 RAG 中的应用
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# RAG（Retrieval-Augmented Generation，检索增强生成）：
#   将"检索"和"生成"结合起来，让 LLM 基于检索到的事实来回答问题。
#
#   为什么需要 RAG？
#   - LLM 的知识有截止日期，无法回答最新问题
#   - LLM 可能"幻觉"（编造不存在的事实）
#   - 企业内部知识不在 LLM 的训练数据中
#
#   RAG 流程：
#     ┌─────────────────────────────────────────────────────┐
#     │                                                     │
#     │  用户问题                                           │
#     │    ↓                                                │
#     │  [1] Embedding: 把问题转为向量                       │
#     │    ↓                                                │
#     │  [2] 检索: 在知识库中找到最相关的文档                │
#     │    ↓                                                │
#     │  [3] 构造 Prompt: 把检索到的文档作为上下文           │
#     │    ↓                                                │
#     │  [4] 生成: LLM 基于上下文回答问题                    │
#     │    ↓                                                │
#     │  最终回答（有事实依据，减少幻觉）                    │
#     │                                                     │
#     └─────────────────────────────────────────────────────┘
#
#   RAG 的关键优势：
#   - 回答有据可查（可追溯到源文档）
#   - 知识可以实时更新（只需更新知识库）
#   - 减少幻觉（LLM 基于给定事实生成）

print("=" * 60)
print("第 4 章：Embedding 在 RAG 中的应用")
print("=" * 60)
print()

# ── 4.1 完整 RAG 流程演示 ────────────────────────────────
print("── 4.1 完整 RAG 流程：检索 → 构造Prompt → 生成 ──────")
print()

# 使用上面已经构建好的知识库
# 模拟用户提出一个问题
user_question = "我想提高身体素质，有什么建议？"
print(f"  用户问题: 「{user_question}」")
print()

# === 步骤1：检索相关文档 ===
print("  [步骤1] 语义检索 —— 找到最相关的知识...")
search_results = semantic_search(user_question, knowledge_base, kb_embeddings, top_k=3)

print("  检索到的相关文档：")
retrieved_docs = []
for rank, (idx, doc, score) in enumerate(search_results, 1):
    print(f"    [{rank}] (相似度:{score:.4f}) {doc}")
    retrieved_docs.append(doc)
print()

# === 步骤2：构造带上下文的 Prompt ===
print("  [步骤2] 构造 Prompt —— 将检索结果作为上下文...")
print()

# 将检索到的文档拼接为上下文
context = "\n".join([f"- {doc}" for doc in retrieved_docs])

# 构造 RAG prompt
rag_prompt = f"""请根据以下参考资料回答用户的问题。
如果参考资料中没有相关信息，请如实说明。

【参考资料】
{context}

【用户问题】
{user_question}

【回答要求】
- 基于参考资料回答，不要编造信息
- 回答要具体、有针对性
- 语言简洁明了
"""

print("  构造的 RAG Prompt：")
print("  " + "─" * 50)
print(f"  {rag_prompt}")
print("  " + "─" * 50)
print()

# === 步骤3：调用 LLM 生成回答 ===
print("  [步骤3] 调用 LLM 生成回答...")
print()

rag_response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "你是一个知识助手，根据提供的参考资料回答问题。回答要准确、有依据。"},
        {"role": "user", "content": rag_prompt},
    ],
    temperature=0.3,  # RAG 场景用较低温度，保证回答忠实于文档
    max_tokens=300,
)

rag_answer = rag_response.choices[0].message.content.strip()
print("  LLM 生成的回答：")
print(f"  {'─' * 50}")
print(f"  {rag_answer}")
print(f"  {'─' * 50}")
print()

# ── 4.2 对比：有 RAG vs 无 RAG ───────────────────────────
print("── 4.2 对比：有 RAG vs 无 RAG 的回答质量 ────────────")
print()

# 无 RAG：直接让 LLM 回答（可能产生幻觉或泛泛而谈）
no_rag_response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "你是一个知识助手，请简洁回答。"},
        {"role": "user", "content": user_question},
    ],
    temperature=0.3,
    max_tokens=300,
)

no_rag_answer = no_rag_response.choices[0].message.content.strip()

print("  [无 RAG] 直接回答（没有参考资料）：")
print(f"  {no_rag_answer}")
print()
print("  [有 RAG] 基于检索结果回答（有参考资料）：")
print(f"  {rag_answer}")
print()

print("  对比要点：")
print("  - 有 RAG 的回答更具体，基于知识库中的实际内容")
print("  - 无 RAG 的回答可能更泛泛，或者包含不在知识库中的信息")
print("  - RAG 让 LLM 的回答'有据可查'，减少了幻觉的风险")
print()


# ── 总结 ──────────────────────────────────────────────────
print("=" * 60)
print("全部实验完毕！核心总结：")
print("=" * 60)
print("""
  ┌────────────────────────────────────────────────────────────┐
  │  概念             │ 说明                                    │
  ├────────────────────────────────────────────────────────────┤
  │  Embedding        │ 将文本映射为高维向量（数字表示语义）     │
  │  余弦相似度       │ 衡量两个向量方向的一致性（-1到1）        │
  │  语义搜索         │ 用向量距离找最相关的文档                 │
  │  RAG              │ 检索+生成，让LLM基于事实回答             │
  └────────────────────────────────────────────────────────────┘

  Embedding 应用链路：
    文本 → Embedding模型 → 向量 → 相似度计算 → 检索/聚类/分类

  实际工程中的进阶方向：
  1. 向量数据库（如 Pinecone、Milvus、Chroma）—— 高效存储和检索百万级向量
  2. 分块策略（Chunking）—— 长文档如何切分为合适大小的片段
  3. 混合搜索（Hybrid Search）—— 结合关键词搜索和语义搜索
  4. 重排序（Reranking）—— 对检索结果做二次精排
  5. 多模态 Embedding —— 图片、音频也能转为向量进行检索
""")
