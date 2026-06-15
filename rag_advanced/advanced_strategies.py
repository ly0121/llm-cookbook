"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     项目 17-B: RAG 高级检索策略 — 混合检索 + 重排序 + CRAG + Self-RAG       ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────── 前置科学知识 ───────────────────────────┐
│                                                                      │
│  【核心问题】为什么 HyDE/Multi-Query/Parent-Child 还不够？           │
│                                                                      │
│  上一篇我们解决了"语义鸿沟"和"召回率"问题。但还有更深层的挑战：       │
│                                                                      │
│    1. 关键词匹配 vs 语义匹配 —— 有时用户就是想精确匹配某个术语       │
│       向量检索会"过度泛化"，反而丢失精确匹配结果                     │
│       → 解决方案: 混合检索(BM25 + 向量 + RRF融合)                   │
│                                                                      │
│    2. 召回了很多文档，但质量参差不齐 —— "找到"不等于"找对"           │
│       → 解决方案: 重排序(Reranking)                                  │
│                                                                      │
│    3. 检索完全失败时怎么办？—— 知识库里压根没有答案                   │
│       → 解决方案: CRAG(纠正性RAG)                                    │
│                                                                      │
│    4. 生成的答案是否真的被文档支持？—— 需要自我反思                   │
│       → 解决方案: Self-RAG(自反思RAG)                                │
│                                                                      │
│  ┌─────────────── 进阶策略全景图 ───────────────┐                   │
│  │                                                │                   │
│  │  用户查询                                      │                   │
│  │    │                                           │                   │
│  │    ▼                                           │                   │
│  │  ┌─────────┐    ┌─────────┐                   │                   │
│  │  │ BM25    │    │ 向量检索 │  ← 混合检索       │                   │
│  │  └────┬────┘    └────┬────┘                   │                   │
│  │       └──────┬───────┘                        │                   │
│  │              ▼                                 │                   │
│  │        RRF 融合排序                            │                   │
│  │              │                                 │                   │
│  │              ▼                                 │                   │
│  │        重排序(Reranking)  ← 精筛              │                   │
│  │              │                                 │                   │
│  │              ▼                                 │                   │
│  │        质量评估(CRAG)     ← 纠错              │                   │
│  │         ╱        ╲                            │                   │
│  │      合格       不合格 → 查询改写/兜底         │                   │
│  │        │                                      │                   │
│  │        ▼                                      │                   │
│  │      生成答案                                  │                   │
│  │        │                                      │                   │
│  │        ▼                                      │                   │
│  │      自反思(Self-RAG)     ← 自检              │                   │
│  │        │                                      │                   │
│  │        ▼                                      │                   │
│  │      最终输出                                  │                   │
│  │                                                │                   │
│  └────────────────────────────────────────────────┘                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

运行方式: python rag_advanced/advanced_strategies.py
依赖: pip install openai numpy
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 0: 初始化 — API配置 + 构建共享知识库
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ┌──────────────────── 本章目标 ────────────────────┐
# │ 1. 初始化 LLM 客户端                              │
# │ 2. 构建共享知识库(8-10篇中文文档)                  │
# │ 3. 实现基础向量化工具(numpy余弦相似度)             │
# └──────────────────────────────────────────────────┘

print("\n" + "=" * 70)
print("Chapter 0: 初始化 — API配置 + 构建共享知识库")
print("=" * 70)

# ─── 0.1 导入依赖 ───
import math
import json
import re
from collections import Counter
from openai import OpenAI
import numpy as np

print("\n[Step 0.1] 所有依赖导入成功")

# ─── 0.2 API 配置 ───
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY, BASE_URL, MODEL_NAME
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
print("[Step 0.2] OpenAI 客户端初始化完成 (model=%s)" % MODEL_NAME)

# ─── 0.3 LLM 调用工具函数 ───
#
# 封装统一的 LLM 调用接口，方便后续各章复用


def call_llm(prompt, temperature=0.7):
    """调用 LLM 并返回文本结果"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def get_embedding(text):
    """
    获取文本的 embedding 向量
    使用 API 的 embedding 接口
    如果不可用，退化为简单的字符频率向量(仅供演示)
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
        )
        return np.array(response.data[0].embedding)
    except Exception:
        # 退化方案: 基于字符频率的简易向量(仅供演示流程)
        chars = list(set(text))
        vec = np.zeros(256)
        for ch in text:
            vec[ord(ch) % 256] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


def cosine_similarity(vec_a, vec_b):
    """计算两个向量的余弦相似度"""
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


print("[Step 0.3] 工具函数定义完成")

# ─── 0.4 构建共享知识库 ───
#
# 设计原则:
#   1. 涵盖自动驾驶/AI/大模型等主题(与rag_strategies.py风格一致)
#   2. 有些文档用学术风格，有些用科普风格(制造语义鸿沟测试场景)
#   3. 包含一些"关键词独特"的文档(用于测试BM25优势)

knowledge_base = [
    {
        "id": "doc_1",
        "title": "自动驾驶感知系统概述",
        "content": (
            "自动驾驶感知系统概述：感知模块是自动驾驶的眼睛和耳朵。"
            "它通过多传感器融合技术（摄像头、激光雷达LiDAR、毫米波雷达）"
            "实现对周围环境的360度全方位感知。其中，摄像头擅长识别交通标志和车道线，"
            "LiDAR擅长精确测距和3D建模，毫米波雷达在恶劣天气下仍能稳定工作。"
        ),
        "chapter": "感知",
    },
    {
        "id": "doc_2",
        "title": "行人检测与AEB系统",
        "content": (
            "行人检测与避障策略：当感知模块检测到行人时，系统会立即评估碰撞风险。"
            "基于深度学习的行人检测算法（如YOLO、CenterNet）可以实时识别行人位置。"
            "系统预测行人未来2-3秒的运动轨迹，并计算TTC(Time-to-Collision)。"
            "如果TTC低于安全阈值，AEB(自动紧急制动)系统作为最后安全防线执行制动。"
        ),
        "chapter": "安全",
    },
    {
        "id": "doc_3",
        "title": "RAG检索增强生成原理",
        "content": (
            "RAG(检索增强生成)通过在回答前先检索相关知识，让大模型基于真实文档回答。"
            "RAG基本流程：Query→Embedding→向量检索→Top-K文档→拼接Prompt→LLM生成。"
            "核心挑战在于检索质量——检索不到正确文档，生成必然出错。"
            "优化策略包括：查询改写、HyDE假设性文档、多路召回、重排序Reranking等。"
        ),
        "chapter": "RAG",
    },
    {
        "id": "doc_4",
        "title": "大语言模型幻觉问题",
        "content": (
            "大语言模型(LLM)存在严重的幻觉问题(Hallucination)。"
            "模型会生成看似合理但实际完全错误的内容，尤其在涉及具体数字、日期、"
            "技术细节时更为突出。幻觉产生的原因包括：训练数据中的噪声、"
            "模型过度自信的概率分布、以及对罕见知识的记忆不牢固。"
            "缓解幻觉的主要手段包括RAG、知识图谱约束、以及自我一致性检查。"
        ),
        "chapter": "幻觉",
    },
    {
        "id": "doc_5",
        "title": "自动驾驶路径规划算法",
        "content": (
            "路径规划模块接收感知结果后，在复杂交通场景中规划安全路径。"
            "常用算法包括A*搜索(全局规划)、Lattice Planner(局部规划)、"
            "MPC模型预测控制(轨迹跟踪)。规划需同时考虑静态障碍物(路缘、护栏)"
            "和动态障碍物(车辆、行人)，并满足车辆运动学约束如最小转弯半径。"
        ),
        "chapter": "规划",
    },
    {
        "id": "doc_6",
        "title": "向量数据库与FAISS",
        "content": (
            "向量数据库是AI应用的重要基础设施。FAISS由Meta开源，支持十亿级向量"
            "的毫秒级检索。常见索引类型包括：Flat(暴力搜索，精确但慢)、"
            "IVF(倒排索引，平衡速度和精度)、HNSW(图索引，高召回率)。"
            "在RAG系统中，向量数据库负责存储文档embedding并提供相似度检索服务。"
            "Milvus和Pinecone是云端向量数据库的代表。"
        ),
        "chapter": "向量DB",
    },
    {
        "id": "doc_7",
        "title": "自动驾驶芯片算力需求",
        "content": (
            "L4级自动驾驶对算力需求极为严苛。感知模块多路摄像头推理需约100 TOPS，"
            "LiDAR点云处理约50 TOPS，规划决策约10 TOPS但延迟要求极高(<100ms)。"
            "主流方案包括NVIDIA Orin(254 TOPS)、地平线征程5(128 TOPS)。"
            "芯片选型需综合考虑算力、功耗、成本和软件生态成熟度。"
        ),
        "chapter": "芯片",
    },
    {
        "id": "doc_8",
        "title": "Transformer注意力机制",
        "content": (
            "Transformer架构的核心是自注意力机制(Self-Attention)。"
            "每个token通过Query、Key、Value三个矩阵计算与其他token的关联度。"
            "多头注意力(Multi-Head Attention)让模型从不同子空间捕获不同类型的依赖关系。"
            "位置编码(Positional Encoding)解决了Transformer无法感知序列顺序的问题。"
            "Flash Attention优化了注意力计算的内存访问模式，大幅加速训练。"
        ),
        "chapter": "Transformer",
    },
    {
        "id": "doc_9",
        "title": "端到端自动驾驶方案",
        "content": (
            "端到端自动驾驶用单一神经网络直接从传感器输入映射到控制输出。"
            "代表方案包括Tesla FSD和UniAD。优势是避免模块间误差累积，"
            "劣势是可解释性差、需海量数据训练。目前业界共识："
            "城市NOA仍需模块化兜底，高速场景端到端已展现优势。"
        ),
        "chapter": "端到端",
    },
    {
        "id": "doc_10",
        "title": "提示工程与思维链",
        "content": (
            "提示工程(Prompt Engineering)是高效使用大模型的关键技术。"
            "思维链(Chain-of-Thought, CoT)通过引导模型逐步推理来提升复杂任务表现。"
            "Few-shot prompting提供示例帮助模型理解任务格式。"
            "ReAct框架结合推理(Reasoning)和行动(Acting)，让模型能调用外部工具。"
            "好的Prompt设计能让同一个模型的表现提升数倍。"
        ),
        "chapter": "提示工程",
    },
]

print("[Step 0.4] 知识库构建完成，共 %d 篇文档" % len(knowledge_base))
print("\n  文档列表:")
for doc in knowledge_base:
    preview = doc["content"][:40] + "..."
    print("    [%s] %s | %s" % (doc["id"], doc["chapter"], preview))

# ─── 0.5 预计算文档 embeddings ───
#
# ┌──────── 向量化流程 ────────┐
# │                              │
# │  文档1 ──→ Embed ──→ vec1  │
# │  文档2 ──→ Embed ──→ vec2  │  → 内存中的向量索引
# │  ...                         │
# │  文档10 ──→ Embed ──→ vec10 │
# │                              │
# └──────────────────────────────┘

print("\n[Step 0.5] 正在预计算文档 embeddings...")
for doc in knowledge_base:
    doc["embedding"] = get_embedding(doc["content"])
print("[Step 0.5] 文档向量化完成!")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 1: 混合检索（Hybrid Search）— BM25 + 向量 + RRF
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ┌──────────────────── 本章目标 ────────────────────┐
# │ 1. 实现 BM25 关键词检索                           │
# │ 2. 实现向量语义检索                               │
# │ 3. 用 RRF(Reciprocal Rank Fusion) 融合两路结果   │
# │ 4. 展示混合检索如何同时捕获精确匹配和语义匹配     │
# └──────────────────────────────────────────────────┘
#
# ┌─────────────── 混合检索原理 ───────────────┐
# │                                              │
# │  单纯向量检索的问题:                         │
# │    查询"NVIDIA Orin" → 可能返回所有芯片文档  │
# │    因为语义上"芯片"都很近，但用户要的是精确  │
# │    匹配"NVIDIA Orin"这个关键词的文档！        │
# │                                              │
# │  单纯关键词检索的问题:                       │
# │    查询"无人车的大脑" → BM25找不到            │
# │    因为文档里写的是"自动驾驶芯片"             │
# │                                              │
# │  混合检索 = 两全其美:                        │
# │                                              │
# │    ┌─── BM25(关键词) ───┐                   │
# │    │ 精确匹配"NVIDIA Orin"│                   │
# │    └─────────┬───────────┘                   │
# │              │                               │
# │              ├──→ RRF 融合 ──→ 最终排序      │
# │              │                               │
# │    ┌─── 向量(语义) ─────┐                   │
# │    │ 语义匹配"芯片算力" │                    │
# │    └─────────┬───────────┘                   │
# │                                              │
# │  比喻: BM25像精确搜索引擎，向量像理解意图     │
# │        的智能助手。两者结合，查全又查准！      │
# │                                              │
# └──────────────────────────────────────────────┘

print("\n\n" + "=" * 70)
print("Chapter 1: 混合检索（Hybrid Search）— BM25 + 向量 + RRF")
print("=" * 70)

# ─── 1.1 实现 BM25 关键词检索 ───
#
# BM25 是信息检索领域的经典算法，基于词频(TF)和逆文档频率(IDF)
# 它不理解语义，只做精确的词汇匹配，但对关键术语检索非常有效
#
# BM25 公式核心思想:
#   score(D, Q) = sum( IDF(q) * TF(q, D) * (k1+1) / (TF(q,D) + k1*(1-b+b*|D|/avgdl)) )
#   - TF: 词在文档中出现的次数越多，越相关
#   - IDF: 词在越少的文档中出现，区分度越高
#   - k1, b: 调节参数

print("\n[Step 1.1] 实现 BM25 关键词检索算法")


def tokenize_chinese(text):
    """
    简易中文分词: 按标点分割后提取连续中文字符和英文单词
    生产环境应使用jieba分词，这里为了减少依赖用简易版本
    """
    # 提取中文双字词(bigram) + 英文单词 + 数字
    tokens = []
    # 英文单词和数字
    tokens.extend(re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text))
    # 中文bigram(相邻两个汉字作为一个token)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    for i in range(len(chinese_chars) - 1):
        tokens.append(chinese_chars[i] + chinese_chars[i + 1])
    return tokens


class BM25:
    """
    BM25 检索器 — 关键词匹配的"老将"

    比喻: 如果向量检索是"理解语义的智者"，
          BM25就是"精确匹配关键词的侦探"，
          它不理解含义，但对精确词汇极度敏感。
    """

    def __init__(self, documents, k1=1.5, b=0.75):
        """
        初始化 BM25 索引

        Args:
            documents: 文档列表，每个文档是 dict(含'content'字段)
            k1: 词频饱和参数(越大，高频词权重越高)
            b: 文档长度归一化参数(越大，短文档优势越明显)
        """
        self.k1 = k1
        self.b = b
        self.documents = documents

        # 对每篇文档分词
        self.doc_tokens = [tokenize_chinese(doc["content"]) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_dl = sum(self.doc_lengths) / len(self.doc_lengths)
        self.n_docs = len(documents)

        # 计算每个词的文档频率(DF)
        self.df = Counter()
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] += 1

    def _idf(self, term):
        """计算逆文档频率 IDF"""
        df = self.df.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def search(self, query, k=5):
        """
        BM25 检索

        Args:
            query: 查询文本
            k: 返回 Top-K 结果

        Returns:
            [(doc, score), ...] 按分数降序排列
        """
        query_tokens = tokenize_chinese(query)
        scores = []

        for i, doc_tokens in enumerate(self.doc_tokens):
            score = 0.0
            tf_counter = Counter(doc_tokens)
            doc_len = self.doc_lengths[i]

            for term in query_tokens:
                if term not in tf_counter:
                    continue
                tf = tf_counter[term]
                idf = self._idf(term)
                # BM25 评分公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avg_dl
                )
                score += idf * numerator / denominator

            scores.append((self.documents[i], score))

        # 按分数降序排列
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]


# 构建 BM25 索引
bm25 = BM25(knowledge_base)
print(
    "  BM25 索引构建完成 (文档数=%d, 平均文档长度=%.0f tokens)"
    % (bm25.n_docs, bm25.avg_dl)
)

# ─── 1.2 实现向量语义检索 ───

print("\n[Step 1.2] 实现向量语义检索")


def vector_search(query, documents, k=5):
    """
    向量语义检索 — 基于余弦相似度的内存检索

    比喻: 向量检索像一个"理解语境的读者"，
          它不在乎你用什么词，只关心你在"说什么意思"。
    """
    query_vec = get_embedding(query)
    scores = []

    for doc in documents:
        sim = cosine_similarity(query_vec, doc["embedding"])
        scores.append((doc, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


print("  向量检索函数定义完成")

# ─── 1.3 实现 RRF (Reciprocal Rank Fusion) ───
#
# ┌─────────── RRF 融合原理 ───────────┐
# │                                      │
# │  RRF_score(d) = sum( 1/(k + rank_i(d)) )  │
# │                                      │
# │  其中 k=60(常数)，rank_i 是文档d     │
# │  在第i路检索中的排名                  │
# │                                      │
# │  优点: 不需要对齐不同检索的分数尺度  │
# │        只看排名，简单而有效           │
# │                                      │
# │  例子:                               │
# │    BM25排名: doc_7=1, doc_1=2        │
# │    向量排名: doc_1=1, doc_7=3        │
# │                                      │
# │    RRF(doc_7) = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323  │
# │    RRF(doc_1) = 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325  │
# │                                      │
# │    → doc_1 综合排名更高!             │
# │                                      │
# └──────────────────────────────────────┘

print("\n[Step 1.3] 实现 RRF (Reciprocal Rank Fusion)")


def rrf_fusion(bm25_results, vector_results, k_param=60):
    """
    RRF 融合 — 将 BM25 和向量检索的排名融合为统一分数

    Args:
        bm25_results: [(doc, score), ...] BM25 检索结果
        vector_results: [(doc, score), ...] 向量检索结果
        k_param: RRF 常数(通常取60)

    Returns:
        [(doc, rrf_score), ...] 融合后按 RRF 分数降序
    """
    rrf_scores = {}  # doc_id → rrf_score

    # BM25 结果贡献
    for rank, (doc, _) in enumerate(bm25_results):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k_param + rank + 1)

    # 向量结果贡献
    for rank, (doc, _) in enumerate(vector_results):
        doc_id = doc["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k_param + rank + 1)

    # 按 RRF 分数排序
    doc_map = {doc["id"]: doc for doc in knowledge_base}
    results = [(doc_map[doc_id], score) for doc_id, score in rrf_scores.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def hybrid_search(query, k=5):
    """
    混合检索: BM25 + 向量 + RRF 融合

    这是完整的混合检索流水线
    """
    bm25_results = bm25.search(query, k=k)
    vec_results = vector_search(query, knowledge_base, k=k)
    fused = rrf_fusion(bm25_results, vec_results)
    return fused[:k], bm25_results[:k], vec_results[:k]


print("  RRF 融合函数定义完成")

# ─── 1.4 对比测试: BM25 vs 向量 vs 混合 ───

print("\n[Step 1.4] 对比测试: BM25 vs 向量 vs 混合检索")
print("=" * 60)

hybrid_test_queries = [
    "NVIDIA Orin芯片的算力是多少？",  # 关键词精确匹配场景(BM25强)
    "无人车怎么理解周围环境？",  # 语义匹配场景(向量强)
    "Flash Attention是什么技术？",  # 混合场景(关键词+语义都有用)
]

for query in hybrid_test_queries:
    print("\n" + "-" * 60)
    print('  查询: "%s"' % query)
    print("-" * 60)

    fused, bm25_res, vec_res = hybrid_search(query, k=3)

    print("\n  【BM25 结果】(关键词匹配):")
    for i, (doc, score) in enumerate(bm25_res[:3]):
        print(
            "    [%d] %s (score=%.3f) | %s..."
            % (i + 1, doc["chapter"], score, doc["content"][:30])
        )

    print("\n  【向量结果】(语义匹配):")
    for i, (doc, score) in enumerate(vec_res[:3]):
        print(
            "    [%d] %s (sim=%.3f) | %s..."
            % (i + 1, doc["chapter"], score, doc["content"][:30])
        )

    print("\n  【RRF 融合结果】(混合检索):")
    for i, (doc, score) in enumerate(fused[:3]):
        print(
            "    [%d] %s (rrf=%.4f) | %s..."
            % (i + 1, doc["chapter"], score, doc["content"][:30])
        )

print("\n[混合检索小结]")
print("  BM25 擅长: 精确术语匹配(NVIDIA Orin, Flash Attention)")
print("  向量 擅长: 语义理解(无人车≈自动驾驶, 理解环境≈感知)")
print("  混合检索: 两全其美，关键词和语义都不遗漏!")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 2: 重排序（Re-ranking）— 从"召回"到"精排"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ┌──────────────────── 本章目标 ────────────────────┐
# │ 1. 理解两阶段检索: 召回(Recall) → 精排(Precision)│
# │ 2. 用 LLM 实现重排序器(Reranker)                 │
# │ 3. 对比重排序前后的结果质量                       │
# └──────────────────────────────────────────────────┘
#
# ┌─────────────── 重排序原理 ───────────────┐
# │                                            │
# │  为什么需要重排序？                        │
# │                                            │
# │  第一阶段(召回): 快速从海量文档中选出候选   │
# │    - BM25/向量检索: 速度快，但粗糙         │
# │    - 类似"海选"，宁可多选也不能漏选        │
# │                                            │
# │  第二阶段(精排): 对候选精细打分             │
# │    - Cross-Encoder: 逐对比较(query, doc)   │
# │    - 类似"复赛"，仔细评判每位选手          │
# │                                            │
# │  ┌─── Bi-Encoder vs Cross-Encoder ───┐   │
# │  │                                     │   │
# │  │ Bi-Encoder(召回阶段):               │   │
# │  │   Query ──→ [向量Q]                 │   │
# │  │   Doc   ──→ [向量D]                 │   │
# │  │   score = cosine(Q, D)              │   │
# │  │   优点: 快(可预计算Doc向量)          │   │
# │  │   缺点: Q和D独立编码，交互不充分     │   │
# │  │                                     │   │
# │  │ Cross-Encoder(精排阶段):            │   │
# │  │   (Query + Doc) ──→ [相关性分数]    │   │
# │  │   Q和D拼接后联合编码                 │   │
# │  │   优点: 充分交互，判断更准确         │   │
# │  │   缺点: 慢(每对都要计算一次)         │   │
# │  │                                     │   │
# │  └─────────────────────────────────────┘   │
# │                                            │
# │  比喻: Bi-Encoder像看简历筛人(快但粗)      │
# │        Cross-Encoder像面试深度考察(慢但准) │
# │                                            │
# └────────────────────────────────────────────┘

print("\n\n" + "=" * 70)
print('Chapter 2: 重排序（Re-ranking）— 从"召回"到"精排"')
print("=" * 70)

# ─── 2.1 实现 LLM-based Reranker ───
#
# 用 LLM 作为 Cross-Encoder 的替代品:
#   输入: (query, document) 对
#   输出: 相关性分数 (0-10)
#
# 优点: 不需要额外训练专用模型
# 缺点: 速度慢、成本高(每个候选文档一次LLM调用)

print("\n[Step 2.1] 实现 LLM-based Reranker")

RERANK_PROMPT_TEMPLATE = (
    "你是一个文档相关性评估专家。\n"
    "请评估以下文档与用户查询的相关程度。\n\n"
    "用户查询: {query}\n\n"
    "文档内容: {document}\n\n"
    "请给出相关性评分(0-10)，其中:\n"
    "- 0分: 完全无关\n"
    "- 5分: 部分相关\n"
    "- 10分: 高度相关，直接回答了用户问题\n\n"
    "请只输出一个数字评分，不要解释。\n"
    "评分:"
)


def llm_rerank(query, documents, top_k=3):
    """
    LLM 重排序器

    对每个候选文档，让 LLM 评估其与查询的相关性
    然后按相关性分数重新排序

    Args:
        query: 用户查询
        documents: 候选文档列表 [(doc, initial_score), ...]
        top_k: 返回 Top-K 结果

    Returns:
        [(doc, rerank_score), ...] 按 LLM 评分降序排列
    """
    reranked = []

    for doc, _ in documents:
        prompt = RERANK_PROMPT_TEMPLATE.format(
            query=query,
            document=doc["content"][:200],  # 截断避免prompt过长
        )
        try:
            response = call_llm(prompt, temperature=0.0)
            # 提取数字分数
            score_match = re.search(r"(\d+(?:\.\d+)?)", response)
            score = float(score_match.group(1)) if score_match else 5.0
            score = min(10.0, max(0.0, score))  # 限制在0-10范围
        except Exception:
            score = 5.0  # 异常时给默认分

        reranked.append((doc, score))

    # 按 LLM 评分降序
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


print("  LLM Reranker 定义完成")

# ─── 2.2 对比测试: 重排序前 vs 重排序后 ───

print("\n[Step 2.2] 对比测试: 重排序前 vs 重排序后")
print("=" * 60)

rerank_query = "如何减少大模型生成错误信息的问题？"
print('\n  查询: "%s"' % rerank_query)

# 第一阶段: 混合检索召回
fused_results, _, _ = hybrid_search(rerank_query, k=5)

print("\n  【第一阶段 — 混合检索召回(Top-5)】:")
for i, (doc, score) in enumerate(fused_results):
    print(
        "    [%d] %s (rrf=%.4f) | %s..."
        % (i + 1, doc["chapter"], score, doc["content"][:40])
    )

# 第二阶段: LLM 重排序
print("\n  【第二阶段 — LLM 重排序(精排)】:")
print("  (正在调用 LLM 逐一评估相关性...)")

reranked_results = llm_rerank(rerank_query, fused_results, top_k=3)

print("\n  重排序后结果:")
for i, (doc, score) in enumerate(reranked_results):
    print(
        "    [%d] %s (relevance=%.1f/10) | %s..."
        % (i + 1, doc["chapter"], score, doc["content"][:40])
    )

print("\n  【效果分析】:")
print('    - 重排序前: 按 RRF 分数排列(仅考虑"位置")')
print("    - 重排序后: LLM 深度理解 query-doc 语义匹配度")
print('    - 与"幻觉"高度相关的 doc_4 应被排到最前面')

print("\n[重排序小结]")
print("  核心优势: 二阶段流水线，召回保量 + 精排保质")
print("  成本代价: 每个候选文档需一次 LLM 调用(N次调用)")
print("  适用场景: 候选文档质量参差不齐，需精确筛选最相关文档")
print("  生产建议: 用专用 Cross-Encoder 模型(如 bge-reranker)替代 LLM")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 3: CRAG（Corrective RAG）— 检索质量自动纠错
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ┌──────────────────── 本章目标 ────────────────────┐
# │ 1. 理解 CRAG 的核心思想: 检索失败时自动纠错       │
# │ 2. 实现文档相关性评估(Grading)                    │
# │ 3. 实现查询改写(Query Rewrite)兜底策略            │
# │ 4. 展示 CRAG 如何自我修正检索质量                 │
# └──────────────────────────────────────────────────┘
#
# ┌─────────────── CRAG 工作流程 ───────────────┐
# │                                               │
# │  传统 RAG 的致命缺陷:                         │
# │    如果检索到的文档和问题无关，LLM 要么幻觉   │
# │    编造答案，要么给出"我不知道"—— 用户体验差  │
# │                                               │
# │  CRAG 的解决方案: "先评估，再决策"            │
# │                                               │
# │  用户查询 ──→ 检索文档                        │
# │                  │                            │
# │                  ▼                            │
# │            ┌───────────┐                      │
# │            │ 评估相关性 │ ← LLM判断: 文档能    │
# │            │ (Grading) │   回答这个问题吗？    │
# │            └─────┬─────┘                      │
# │                  │                            │
# │          ┌───────┼───────┐                    │
# │          │               │                    │
# │       相关(Yes)       不相关(No)              │
# │          │               │                    │
# │          ▼               ▼                    │
# │     直接生成答案    查询改写/兜底             │
# │                          │                    │
# │                          ▼                    │
# │                    重新检索或使用             │
# │                    Web搜索补充                │
# │                                               │
# │  比喻: CRAG像一个谨慎的老师:                  │
# │    先检查参考资料是否对题，                    │
# │    如果发现答非所问，就换一本书再找。          │
# │                                               │
# └───────────────────────────────────────────────┘

print("\n\n" + "=" * 70)
print("Chapter 3: CRAG（Corrective RAG）— 检索质量自动纠错")
print("=" * 70)

# ─── 3.1 实现文档相关性评估(Grading) ───

print("\n[Step 3.1] 实现文档相关性评估器(Grader)")

GRADING_PROMPT_TEMPLATE = (
    "你是一个文档相关性评估专家。\n"
    "请判断以下检索到的文档是否能帮助回答用户的问题。\n\n"
    "用户问题: {query}\n\n"
    "检索到的文档: {document}\n\n"
    "请判断这篇文档是否与用户问题相关且有用。\n"
    '只回答 "yes" 或 "no"。\n'
    "判断:"
)


def grade_document(query, doc_content):
    """
    评估单篇文档是否与查询相关

    Returns:
        'yes' 或 'no'
    """
    prompt = GRADING_PROMPT_TEMPLATE.format(query=query, document=doc_content[:300])
    response = call_llm(prompt, temperature=0.0)
    return "yes" if "yes" in response.lower() else "no"


print("  文档相关性评估器定义完成")

# ─── 3.2 实现查询改写(Query Rewrite) ───

print("\n[Step 3.2] 实现查询改写(Query Rewrite)")

REWRITE_PROMPT_TEMPLATE = (
    "你是一个搜索查询优化专家。\n"
    "用户的原始查询在知识库中没有找到相关结果。\n"
    "请改写这个查询，使其更适合在技术知识库中搜索。\n\n"
    "要求:\n"
    "- 使用更专业的术语\n"
    "- 扩展或变换表述方式\n"
    "- 保持原始意图不变\n\n"
    "原始查询: {query}\n\n"
    "改写后的查询(只输出改写结果，不要解释):"
)


def rewrite_query(query):
    """改写查询，使其更适合检索"""
    prompt = REWRITE_PROMPT_TEMPLATE.format(query=query)
    return call_llm(prompt, temperature=0.3)


print("  查询改写器定义完成")

# ─── 3.3 实现完整 CRAG 流程 ───

print("\n[Step 3.3] 实现完整 CRAG 流程")

ANSWER_PROMPT_TEMPLATE = (
    "你是一个技术助手。请根据以下参考文档回答用户的问题。\n"
    "如果文档信息不足，请如实说明。\n\n"
    "参考文档:\n{context}\n\n"
    "用户问题: {query}\n\n"
    "回答:"
)


def crag_pipeline(query, k=3):
    """
    CRAG 完整流程:
    1. 检索文档
    2. 评估文档相关性
    3. 根据评估结果决定:
       - 相关文档数量足够 → 直接生成
       - 相关文档不足 → 改写查询重新检索
    """
    print("    [CRAG] Step 1: 初始检索...")
    fused_results, _, _ = hybrid_search(query, k=k)
    initial_docs = [doc for doc, _ in fused_results]

    # Step 2: 评估每篇文档的相关性
    print("    [CRAG] Step 2: 评估文档相关性...")
    relevant_docs = []
    for doc in initial_docs:
        grade = grade_document(query, doc["content"])
        status = "PASS" if grade == "yes" else "FAIL"
        print("      %s: %s (%s)" % (doc["id"], status, doc["chapter"]))
        if grade == "yes":
            relevant_docs.append(doc)

    # Step 3: 决策
    if len(relevant_docs) >= 1:
        # 有相关文档，直接生成
        print(
            "    [CRAG] Step 3: 找到 %d 篇相关文档，直接生成答案" % len(relevant_docs)
        )
        context = "\n\n".join(doc["content"] for doc in relevant_docs)
        prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, query=query)
        answer = call_llm(prompt, temperature=0.3)
        return {
            "status": "direct_answer",
            "relevant_docs": relevant_docs,
            "answer": answer,
        }
    else:
        # 没有相关文档，触发纠错: 改写查询重新检索
        print("    [CRAG] Step 3: 未找到相关文档! 触发查询改写...")
        new_query = rewrite_query(query)
        print('    [CRAG] 改写后查询: "%s"' % new_query[:60])

        # 用改写后的查询重新检索
        print("    [CRAG] Step 4: 用改写查询重新检索...")
        fused_results_2, _, _ = hybrid_search(new_query, k=k)
        fallback_docs = [doc for doc, _ in fused_results_2]

        context = "\n\n".join(doc["content"] for doc in fallback_docs[:2])
        prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, query=query)
        answer = call_llm(prompt, temperature=0.3)
        return {
            "status": "corrected",
            "original_query": query,
            "rewritten_query": new_query,
            "fallback_docs": fallback_docs[:2],
            "answer": answer,
        }


# ─── 3.4 CRAG 演示 ───

print("\n[Step 3.4] CRAG 演示")
print("=" * 60)

crag_test_queries = [
    "如何减少AI模型胡说八道的问题？",  # 知识库有相关文档(幻觉)
    "量子计算如何加速机器学习训练？",  # 知识库无相关文档 → 触发纠错
]

for query in crag_test_queries:
    print("\n" + "-" * 60)
    print('  查询: "%s"' % query)
    print("-" * 60)

    result = crag_pipeline(query)

    print("\n  【CRAG 结果】:")
    print("    状态: %s" % result["status"])
    if result["status"] == "direct_answer":
        docs = result["relevant_docs"]
        print("    相关文档: %s" % [d["chapter"] for d in docs])
    else:
        print("    原始查询: %s" % result.get("original_query", ""))
        print("    改写查询: %s" % result.get("rewritten_query", "")[:60])
    print("    回答: %s..." % result["answer"][:100])

print("\n[CRAG 小结]")
print('  核心优势: 检索失败时自动纠错，避免"答非所问"')
print("  成本代价: 每篇文档一次 LLM 评估 + 可能的查询改写")
print("  适用场景: 知识库覆盖不全、用户查询范围广泛的系统")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 4: Self-RAG（自反思 RAG）— 生成后自我评估
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ┌──────────────────── 本章目标 ────────────────────┐
# │ 1. 理解 Self-RAG 三个反思维度                     │
# │ 2. 实现 IsRel / IsSup / IsUse 三个评估 token     │
# │ 3. 展示生成后自我修正的完整流程                   │
# └──────────────────────────────────────────────────┘
#
# ┌─────────────── Self-RAG 工作流程 ───────────────┐
# │                                                   │
# │  Self-RAG 的核心: "边生成，边反思，边修正"        │
# │                                                   │
# │  三个反思 Token:                                  │
# │                                                   │
# │  ┌──────────────────────────────────────────┐    │
# │  │ IsRel (Is Retrieval Relevant?)            │    │
# │  │   → 这个问题需要检索吗？                  │    │
# │  │   → 简单常识题不需要，专业题需要          │    │
# │  │                                          │    │
# │  │ IsSup (Is answer Supported by docs?)     │    │
# │  │   → 生成的答案是否被文档支持？            │    │
# │  │   → 防止 LLM 在有文档时仍然幻觉           │    │
# │  │                                          │    │
# │  │ IsUse (Is answer Useful to user?)        │    │
# │  │   → 答案对用户是否有用？                  │    │
# │  │   → 确保答案质量达标                      │    │
# │  └──────────────────────────────────────────┘    │
# │                                                   │
# │  完整流程:                                        │
# │                                                   │
# │  Query ──→ [IsRel?]                              │
# │              │                                    │
# │        需要检索    不需要                          │
# │              │         │                          │
# │              ▼         ▼                          │
# │          检索文档    直接生成                      │
# │              │                                    │
# │              ▼                                    │
# │          生成答案                                  │
# │              │                                    │
# │              ▼                                    │
# │          [IsSup?] 答案被文档支持吗？              │
# │           │           │                           │
# │         支持        不支持                         │
# │           │           │                           │
# │           ▼           ▼                           │
# │       [IsUse?]    重新检索/重新生成               │
# │           │                                       │
# │           ▼                                       │
# │       输出最终答案                                 │
# │                                                   │
# │  比喻: Self-RAG像一个做完作业后会自己检查的学生:   │
# │    先想"这题需要查资料吗？"                       │
# │    写完答案后问"我的答案有依据吗？"               │
# │    最后确认"这个回答对提问者有帮助吗？"           │
# │                                                   │
# └───────────────────────────────────────────────────┘

print("\n\n" + "=" * 70)
print("Chapter 4: Self-RAG（自反思 RAG）— 生成后自我评估")
print("=" * 70)

# ─── 4.1 实现三个反思 Token ───

print("\n[Step 4.1] 实现三个反思评估器")

# IsRel: 判断是否需要检索
ISREL_PROMPT = (
    "判断以下问题是否需要从知识库检索信息才能准确回答。\n\n"
    '如果是简单常识或闲聊，不需要检索，回答"no"。\n'
    '如果涉及专业知识、具体技术细节、特定事实，需要检索，回答"yes"。\n\n'
    "问题: {query}\n\n"
    "需要检索吗？只回答 yes 或 no:"
)

# IsSup: 判断答案是否被文档支持
ISSUP_PROMPT = (
    "请判断以下回答是否被参考文档中的信息所支持。\n\n"
    "参考文档:\n{context}\n\n"
    "生成的回答:\n{answer}\n\n"
    "评估标准:\n"
    '- "fully_supported": 回答中的关键信息都能在文档中找到依据\n'
    '- "partially_supported": 部分信息有依据，部分是推断或补充\n'
    '- "not_supported": 回答的主要内容在文档中找不到依据\n\n'
    "请只回答 fully_supported / partially_supported / not_supported:"
)

# IsUse: 判断答案是否对用户有用
ISUSE_PROMPT = (
    "请评估以下回答对用户问题的有用程度。\n\n"
    "用户问题: {query}\n\n"
    "生成的回答: {answer}\n\n"
    "评估标准(1-5分):\n"
    "- 1分: 完全没用，答非所问\n"
    "- 3分: 有些相关但不够完整\n"
    "- 5分: 完整回答了问题，信息充分\n\n"
    "请只输出一个数字(1-5):"
)


def check_is_retrieval_needed(query):
    """IsRel: 判断是否需要检索"""
    prompt = ISREL_PROMPT.format(query=query)
    response = call_llm(prompt, temperature=0.0)
    return "yes" in response.lower()


def check_is_supported(context, answer):
    """IsSup: 判断答案是否被文档支持"""
    prompt = ISSUP_PROMPT.format(context=context, answer=answer)
    response = call_llm(prompt, temperature=0.0)
    if "fully_supported" in response.lower():
        return "fully_supported"
    elif "not_supported" in response.lower():
        return "not_supported"
    else:
        return "partially_supported"


def check_is_useful(query, answer):
    """IsUse: 判断答案是否有用"""
    prompt = ISUSE_PROMPT.format(query=query, answer=answer)
    response = call_llm(prompt, temperature=0.0)
    score_match = re.search(r"(\d)", response)
    return int(score_match.group(1)) if score_match else 3


print("  IsRel / IsSup / IsUse 三个反思评估器定义完成")

# ─── 4.2 实现完整 Self-RAG 流程 ───

print("\n[Step 4.2] 实现完整 Self-RAG 流程")


def self_rag_pipeline(query, max_retries=2):
    """
    Self-RAG 完整流程:
    1. IsRel: 判断是否需要检索
    2. 检索(如需要) + 生成答案
    3. IsSup: 检查答案是否被文档支持
    4. 如果不支持 → 重新检索/重新生成(最多重试max_retries次)
    5. IsUse: 最终有用性检查
    """
    print("    [Self-RAG] Step 1: IsRel — 判断是否需要检索...")
    needs_retrieval = check_is_retrieval_needed(query)
    print("      → 需要检索: %s" % ("Yes" if needs_retrieval else "No"))

    context = ""
    retrieved_docs = []

    if needs_retrieval:
        # 检索文档
        print("    [Self-RAG] Step 2: 检索文档...")
        fused_results, _, _ = hybrid_search(query, k=3)
        retrieved_docs = [doc for doc, _ in fused_results[:3]]
        context = "\n\n".join(doc["content"] for doc in retrieved_docs)
        print(
            "      → 检索到 %d 篇文档: %s"
            % (len(retrieved_docs), [d["chapter"] for d in retrieved_docs])
        )

    # 生成答案(可能多次重试)
    for attempt in range(max_retries + 1):
        print(
            "    [Self-RAG] Step 3: 生成答案 (尝试 %d/%d)..."
            % (attempt + 1, max_retries + 1)
        )

        if context:
            prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, query=query)
        else:
            prompt = "请回答以下问题:\n\n%s" % query

        answer = call_llm(prompt, temperature=0.3)
        print("      → 生成答案: %s..." % answer[:60])

        # IsSup 检查(仅在有检索文档时)
        if context:
            print("    [Self-RAG] Step 4: IsSup — 检查答案是否被文档支持...")
            support_level = check_is_supported(context, answer)
            print("      → 支持程度: %s" % support_level)

            if support_level == "not_supported" and attempt < max_retries:
                # 不支持 → 换一批文档重试
                print("      → 答案未被支持，尝试重新检索...")
                # 用不同的方式重新检索(这里简单地改写查询)
                new_query = rewrite_query(query)
                fused_results, _, _ = hybrid_search(new_query, k=3)
                retrieved_docs = [doc for doc, _ in fused_results[:3]]
                context = "\n\n".join(doc["content"] for doc in retrieved_docs)
                continue
        break

    # IsUse 最终检查
    print("    [Self-RAG] Step 5: IsUse — 评估答案有用性...")
    usefulness = check_is_useful(query, answer)
    print("      → 有用性评分: %d/5" % usefulness)

    return {
        "query": query,
        "needs_retrieval": needs_retrieval,
        "retrieved_docs": [d["chapter"] for d in retrieved_docs],
        "answer": answer,
        "support_level": support_level if context else "N/A",
        "usefulness": usefulness,
    }


# ─── 4.3 Self-RAG 演示 ───

print("\n[Step 4.3] Self-RAG 演示")
print("=" * 60)

self_rag_queries = [
    "自动驾驶的行人检测使用了什么算法？",  # 需要检索，文档支持
    "今天天气怎么样？",  # 不需要检索(闲聊)
]

self_rag_results = []
for query in self_rag_queries:
    print("\n" + "-" * 60)
    print('  查询: "%s"' % query)
    print("-" * 60)

    result = self_rag_pipeline(query)
    self_rag_results.append(result)

    print("\n  【Self-RAG 最终结果】:")
    print("    需要检索: %s" % result["needs_retrieval"])
    print("    检索文档: %s" % result["retrieved_docs"])
    print("    支持程度: %s" % result["support_level"])
    print("    有用性: %d/5" % result["usefulness"])
    print("    回答: %s..." % result["answer"][:80])

print("\n[Self-RAG 小结]")
print("  核心优势: 生成后自我反思，发现问题主动修正")
print("  成本代价: 多次 LLM 调用(IsRel + 生成 + IsSup + IsUse)")
print("  适用场景: 对答案准确性要求极高的场景(医疗、法律、金融)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 5: 策略对比实验
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ┌──────────────────── 本章目标 ────────────────────┐
# │ 用同一查询跑过所有策略，直观对比效果               │
# └──────────────────────────────────────────────────┘

print("\n\n" + "=" * 70)
print("Chapter 5: 策略对比实验")
print("=" * 70)

comparison_query = "大模型为什么会生成错误信息？如何解决？"
print('\n  统一测试查询: "%s"' % comparison_query)
print("=" * 60)

# ─── 5.1 Naive RAG (仅向量检索) ───
print("\n  [1/5] Naive RAG (仅向量检索):")
naive_vec_results = vector_search(comparison_query, knowledge_base, k=3)
naive_docs_str = "\n\n".join(doc["content"] for doc, _ in naive_vec_results[:2])
naive_answer = call_llm(
    ANSWER_PROMPT_TEMPLATE.format(context=naive_docs_str, query=comparison_query),
    temperature=0.3,
)
print("    检索: %s" % [doc["chapter"] for doc, _ in naive_vec_results[:3]])
print("    回答: %s..." % naive_answer[:80])

# ─── 5.2 Hybrid Search (BM25 + 向量 + RRF) ───
print("\n  [2/5] Hybrid Search (混合检索):")
hybrid_fused, _, _ = hybrid_search(comparison_query, k=3)
hybrid_docs_str = "\n\n".join(doc["content"] for doc, _ in hybrid_fused[:2])
hybrid_answer = call_llm(
    ANSWER_PROMPT_TEMPLATE.format(context=hybrid_docs_str, query=comparison_query),
    temperature=0.3,
)
print("    检索: %s" % [doc["chapter"] for doc, _ in hybrid_fused[:3]])
print("    回答: %s..." % hybrid_answer[:80])

# ─── 5.3 Hybrid + Rerank ───
print("\n  [3/5] Hybrid + Rerank (混合检索 + 重排序):")
reranked = llm_rerank(comparison_query, hybrid_fused[:5], top_k=2)
rerank_docs_str = "\n\n".join(doc["content"] for doc, _ in reranked)
rerank_answer = call_llm(
    ANSWER_PROMPT_TEMPLATE.format(context=rerank_docs_str, query=comparison_query),
    temperature=0.3,
)
print("    精排后: %s" % [doc["chapter"] for doc, _ in reranked])
print("    回答: %s..." % rerank_answer[:80])

# ─── 5.4 CRAG ───
print("\n  [4/5] CRAG (纠正性RAG):")
crag_result = crag_pipeline(comparison_query)
print("    状态: %s" % crag_result["status"])
print("    回答: %s..." % crag_result["answer"][:80])

# ─── 5.5 Self-RAG ───
print("\n  [5/5] Self-RAG (自反思RAG):")
selfrag_result = self_rag_pipeline(comparison_query)
print(
    "    支持度: %s, 有用性: %d/5"
    % (selfrag_result["support_level"], selfrag_result["usefulness"])
)
print("    回答: %s..." % selfrag_result["answer"][:80])

# ─── 5.6 对比总结 ───

print("\n\n" + "=" * 70)
print("策略对比总结")
print("=" * 70)

print("""
┌──────────────┬────────────────────┬──────────────┬───────────────────┐
│ 策略          │ 核心特点            │ LLM调用次数  │ 推荐场景           │
├──────────────┼────────────────────┼──────────────┼───────────────────┤
│ Naive RAG    │ 纯向量检索+生成     │ 1次          │ 简单问答/原型验证  │
├──────────────┼────────────────────┼──────────────┼───────────────────┤
│ Hybrid Search│ BM25+向量+RRF融合   │ 1次          │ 需要精确术语匹配   │
├──────────────┼────────────────────┼──────────────┼───────────────────┤
│ Hybrid+Rerank│ 混合检索+LLM精排    │ 1+N次        │ 候选文档质量参差   │
├──────────────┼────────────────────┼──────────────┼───────────────────┤
│ CRAG         │ 检索质量评估+纠错   │ 1+N+可能2次  │ 知识库覆盖不完整   │
├──────────────┼────────────────────┼──────────────┼───────────────────┤
│ Self-RAG     │ 生成后自我反思修正  │ 4+次         │ 准确性要求极高     │
└──────────────┴────────────────────┴──────────────┴───────────────────┘

┌─────────────────── 选型建议决策树 ───────────────────┐
│                                                        │
│  你的场景是什么？                                      │
│    │                                                  │
│    ├─ "用户查询含精确术语(型号/算法名/专有名词)"       │
│    │   → Hybrid Search (BM25 抓关键词)               │
│    │                                                  │
│    ├─ "检索结果多但质量不稳定"                         │
│    │   → Hybrid + Rerank (先粗筛再精排)              │
│    │                                                  │
│    ├─ "知识库可能没有答案，需要兜底"                   │
│    │   → CRAG (检索失败时自动纠错)                   │
│    │                                                  │
│    ├─ "对答案准确性要求极高，不能有幻觉"              │
│    │   → Self-RAG (生成后自我校验)                   │
│    │                                                  │
│    └─ "全都要！"                                      │
│        → Hybrid + Rerank + CRAG + Self-RAG 全链路    │
│        → 注意: 成本和延迟会显著增加                   │
│                                                        │
└────────────────────────────────────────────────────────┘

┌─────────────────── 性能 vs 质量权衡 ───────────────────┐
│                                                          │
│  质量 ↑                                                 │
│  │                                                      │
│  │              ★ Self-RAG                              │
│  │          ★ CRAG                                      │
│  │      ★ Hybrid+Rerank                                │
│  │    ★ Hybrid                                          │
│  │  ★ Naive RAG                                         │
│  │                                                      │
│  └──────────────────────────────────── 延迟/成本 →      │
│                                                          │
│  实践建议: 从 Hybrid Search 开始，按需逐步加层          │
│                                                          │
└──────────────────────────────────────────────────────────┘
""")

print("=" * 70)
print("项目 17-B 完成! 高级 RAG 策略: 混合检索 + 重排序 + CRAG + Self-RAG")
print("=" * 70)
