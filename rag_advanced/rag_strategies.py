"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           项目 17: RAG 高级检索策略 — HyDE + Multi-Query + Parent-Child       ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────── 前置科学知识 ───────────────────────────┐
│                                                                      │
│  【核心问题】为什么 Naive RAG 会失败？                                │
│                                                                      │
│  用户的"问题"和知识库里的"答案"之间存在 语义鸿沟(Semantic Gap)：        │
│                                                                      │
│    用户问: "自动驾驶怎么避免撞人？"                                    │
│    知识库: "感知模块通过多传感器融合实现行人检测，配合规划模块..."       │
│                                                                      │
│  问题用"日常口语"表达，答案用"技术术语"描述 —— 向量空间中它们可能      │
│  离得很远！                                                           │
│                                                                      │
│  ┌─────── 向量空间示意 ───────┐                                      │
│  │                             │                                      │
│  │     Q(用户问题)             │                                      │
│  │        ·                    │    Q 和 D 之间的距离 = 语义鸿沟      │
│  │              (鸿沟)         │                                      │
│  │                    ·        │                                      │
│  │               D(知识文档)   │                                      │
│  │                             │                                      │
│  └─────────────────────────────┘                                      │
│                                                                      │
│  【三种武器解决语义鸿沟】                                              │
│                                                                      │
│  1. HyDE: 先让LLM"想象"一个答案 → 用"想象答案"去检索                  │
│     比喻: 你去图书馆找书，不是告诉管理员"我想了解X"，                  │
│           而是先写一段关于X的草稿，让管理员找类似的书                   │
│                                                                      │
│  2. Multi-Query: 把一个问题拆成多个角度 → 合并多次检索结果              │
│     比喻: 三个侦探从不同方向搜索，总比一个人找得全                     │
│                                                                      │
│  3. Parent-Child: 小块精确匹配 + 大块完整上下文                        │
│     比喻: 用书的索引(小块)找到位置，然后给你整章(大块)阅读             │
│                                                                      │
│  ┌─────────────── 策略对比 ───────────────┐                          │
│  │ 策略        │ 解决的问题    │ 代价       │                          │
│  │─────────────│───────────────│────────────│                          │
│  │ Naive RAG   │ 基线          │ 最快       │                          │
│  │ HyDE        │ 语义鸿沟      │ +1次LLM   │                          │
│  │ Multi-Query │ 召回率不足    │ +1次LLM   │                          │
│  │ Parent-Child│ 上下文不完整  │ +存储开销  │                          │
│  └─────────────────────────────────────────┘                          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

运行方式: python rag_advanced/rag_strategies.py
依赖: pip install langchain-openai langchain-huggingface langchain-community faiss-cpu
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 0: 初始化 — LLM + Embeddings + 构建示例知识库
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ┌──────────────────── 本章目标 ────────────────────┐
# │ 1. 初始化 LLM 和 Embedding 模型                   │
# │ 2. 构建一个关于 AI/自动驾驶/技术 的模拟知识库       │
# │ 3. 将知识库文档向量化存入 FAISS                    │
# └──────────────────────────────────────────────────┘
#
# 知识库设计思路:
#   - 10篇相互关联的中文文档，覆盖: 自动驾驶感知/规划/控制、大模型、RAG、芯片
#   - 文档之间有交叉引用关系，模拟真实知识图谱
#   - 有些文档用"学术风格"，有些用"科普风格" —— 制造语义鸿沟

print("\n" + "=" * 70)
print("Chapter 0: 初始化 — LLM + Embeddings + 构建示例知识库")
print("=" * 70)

# ─── 0.1 导入依赖 ───
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

print("\n[Step 0.1] 所有依赖导入成功 ✓")

# ─── 0.2 API 配置 ───
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY, BASE_URL, MODEL_NAME
# ─── 0.3 初始化 LLM ───
#
# temperature=0.7: 生成假设性文档时需要一定创造力
# 但在最终回答时我们会用较低温度确保准确性
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    temperature=0.7,
)
print("[Step 0.3] LLM 初始化完成 (model=%s)" % MODEL_NAME)

# ─── 0.4 初始化 Embedding 模型 ───
#
# BAAI/bge-small-zh-v1.5:
#   - 北京智源出品，中文效果极佳
#   - small版本: 512维，速度快，适合学习/演示
#   - 在 MTEB 中文榜单上排名靠前
#
# ┌──────── Embedding 工作原理 ────────┐
# │                                      │
# │  "自动驾驶" → [0.12, -0.34, ...]    │
# │  "无人驾车" → [0.11, -0.33, ...]    │  ← 语义相近，向量也近！
# │  "今天天气" → [-0.8, 0.56, ...]     │  ← 语义不同，向量远
# │                                      │
# └──────────────────────────────────────┘
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # 归一化，余弦相似度更稳定
)
print("[Step 0.4] Embedding 模型加载完成 (BAAI/bge-small-zh-v1.5, 512维)")

# ─── 0.5 构建示例知识库 ───
#
# 设计原则:
#   1. 文档之间有关联（感知→规划→控制 是完整链路）
#   2. 同一概念用不同表述（制造语义鸿沟的测试场景）
#   3. 有长文档有短文档（为 Parent-Child 做准备）

knowledge_base = [
    # ─── 自动驾驶感知模块 ───
    Document(
        page_content=(
            "自动驾驶感知系统概述：感知模块是自动驾驶的眼睛和耳朵。"
            "它通过多传感器融合技术（摄像头、激光雷达LiDAR、毫米波雷达）"
            "实现对周围环境的360度全方位感知。其中，摄像头擅长识别交通标志和车道线，"
            "LiDAR擅长精确测距和3D建模，毫米波雷达在恶劣天气下仍能稳定工作。"
            "三种传感器的数据通过后融合或前融合算法进行整合，"
            "输出统一的环境感知结果给下游的规划模块。"
        ),
        metadata={"source": "AD感知教材", "chapter": "感知", "doc_id": "doc_1"},
    ),
    Document(
        page_content=(
            "行人检测与避障策略：当感知模块检测到行人时，系统会立即评估碰撞风险。"
            "基于深度学习的行人检测算法（如YOLO、CenterNet）可以实时识别行人位置和运动轨迹。"
            "系统会预测行人未来2-3秒的运动轨迹，并计算TTC(Time-to-Collision)碰撞时间。"
            "如果TTC低于安全阈值，规划模块会触发紧急制动或避让路径规划。"
            "同时，AEB(自动紧急制动)系统作为最后一道安全防线，"
            "可在驾驶员未反应时自主执行制动操作。"
        ),
        metadata={"source": "AD安全手册", "chapter": "安全", "doc_id": "doc_2"},
    ),
    # ─── 自动驾驶规划模块 ───
    Document(
        page_content=(
            "路径规划算法详解：规划模块接收感知模块的环境信息后，"
            "需要在复杂的交通场景中找到一条安全、舒适、高效的行驶路径。"
            "常用算法包括：A*搜索用于全局路径规划，"
            "Lattice Planner用于局部路径规划，"
            "Model Predictive Control(MPC)用于轨迹跟踪。"
            "规划模块需要同时考虑静态障碍物（路缘、护栏）和动态障碍物（车辆、行人），"
            "并满足车辆运动学约束（最小转弯半径、最大加速度等）。"
        ),
        metadata={"source": "AD规划教材", "chapter": "规划", "doc_id": "doc_3"},
    ),
    Document(
        page_content=(
            "自动驾驶决策系统的核心挑战：在十字路口场景中，"
            "决策系统需要处理多个交通参与者的博弈关系。"
            "例如无保护左转时，自车需要判断对向来车的意图、"
            "估计其速度和加速度，然后决定是等待还是先行通过。"
            "这涉及到博弈论和强化学习方法。"
            "当前主流方案包括规则引擎、有限状态机(FSM)、"
            "以及基于学习的端到端决策方法。"
        ),
        metadata={"source": "AD决策论文", "chapter": "决策", "doc_id": "doc_4"},
    ),
    # ─── 大模型与AI ───
    Document(
        page_content=(
            "大语言模型(LLM)的工作原理：以GPT系列为代表的大语言模型"
            "基于Transformer架构，通过海量文本数据预训练学习语言规律。"
            '模型本质是一个"下一个token预测器"——给定前面的文字，'
            "预测后面最可能出现的文字。模型参数量从数十亿到数千亿不等，"
            "参数越多，模型的知识容量和推理能力越强。"
            "但LLM存在幻觉问题(Hallucination)，会生成看似合理但实际错误的内容，"
            "这是RAG技术出现的重要动因。"
        ),
        metadata={"source": "AI基础教材", "chapter": "LLM", "doc_id": "doc_5"},
    ),
    Document(
        page_content=(
            "RAG(检索增强生成)技术原理：RAG通过在生成回答前先检索相关知识，"
            "让大模型基于真实文档而非记忆来回答问题，从而大幅减少幻觉。"
            "RAG的基本流程是：Query→Embedding→向量检索→取Top-K文档→拼接Prompt→LLM生成。"
            "其核心挑战在于检索质量——如果检索不到正确文档，后续生成必然出错。"
            "因此，提升检索召回率和精确度是RAG系统优化的重中之重。"
            "常见优化策略包括：查询改写、HyDE、多路召回、重排序(Reranking)等。"
        ),
        metadata={"source": "RAG技术白皮书", "chapter": "RAG", "doc_id": "doc_6"},
    ),
    # ─── 芯片与硬件 ───
    Document(
        page_content=(
            "自动驾驶芯片算力需求分析：L4级自动驾驶系统对算力的需求极为严苛。"
            "感知模块中，多路摄像头实时推理需要约100 TOPS算力，"
            "LiDAR点云处理需要约50 TOPS，传感器融合约20 TOPS。"
            "规划决策模块虽然计算量较低（约10 TOPS），"
            "但对延迟要求极高（端到端延迟<100ms）。"
            "目前业界主流方案包括NVIDIA Orin(254 TOPS)、"
            "地平线征程5(128 TOPS)等。芯片选型需要综合考虑"
            "算力、功耗、成本和生态成熟度。"
        ),
        metadata={"source": "硬件选型指南", "chapter": "芯片", "doc_id": "doc_7"},
    ),
    # ─── 技术融合 ───
    Document(
        page_content=(
            "大模型在自动驾驶中的应用前景：近年来，研究者开始探索"
            "将大语言模型的推理能力应用于自动驾驶领域。"
            "主要方向包括：(1)用LLM理解复杂交通场景语义，"
            "(2)用LLM进行可解释的驾驶决策，"
            "(3)用多模态大模型(如GPT-4V)直接处理摄像头图像进行端到端驾驶。"
            "但目前LLM的推理延迟（数百毫秒到数秒）仍是实时驾驶场景的瓶颈。"
            "学术界正在研究模型蒸馏、推理加速等方法来缩小这一差距。"
        ),
        metadata={"source": "前沿研究综述", "chapter": "LLM+AD", "doc_id": "doc_8"},
    ),
    Document(
        page_content=(
            "向量数据库在智能驾驶中的应用：随着高精地图和场景数据的积累，"
            "向量数据库成为智能驾驶数据管理的重要工具。"
            "通过将驾驶场景编码为向量，系统可以快速检索历史相似场景，"
            "用于仿真测试、corner case挖掘和在线学习。"
            "常用的向量数据库包括FAISS、Milvus、Pinecone等。"
            "其中FAISS由Meta开源，支持十亿级向量的毫秒级检索，"
            "适合对延迟敏感的在线系统。"
        ),
        metadata={"source": "数据架构设计", "chapter": "向量DB", "doc_id": "doc_9"},
    ),
    Document(
        page_content=(
            "端到端自动驾驶方案解析：传统自动驾驶采用模块化架构"
            "（感知→预测→规划→控制），各模块独立优化。"
            "而端到端方案（如Tesla FSD、UniAD）试图用单一神经网络"
            "直接从传感器输入映射到控制输出，减少信息损失。"
            "端到端方案的优势是避免了模块间的误差累积，"
            "劣势是可解释性差、需要海量数据训练。"
            "目前业界共识是：城市NOA场景仍需模块化兜底，"
            "高速场景端到端已展现优势。"
        ),
        metadata={"source": "技术路线分析", "chapter": "端到端", "doc_id": "doc_10"},
    ),
]

print("[Step 0.5] 知识库构建完成，共 %d 篇文档" % len(knowledge_base))
print("\n  文档列表:")
for i, doc in enumerate(knowledge_base):
    preview = doc.page_content[:40] + "..."
    print("    [%d] %s | %s" % (i + 1, doc.metadata.get("chapter", ""), preview))

# ─── 0.6 向量化并存入 FAISS ───
#
# ┌─────────── FAISS 索引构建流程 ───────────┐
# │                                            │
# │  文档1 ──→ Embed ──→ [0.12, -0.3, ...]   │
# │  文档2 ──→ Embed ──→ [0.08, -0.2, ...]   │──→ FAISS Index
# │  ...                                       │
# │  文档10 ──→ Embed ──→ [-0.1, 0.5, ...]   │
# │                                            │
# └────────────────────────────────────────────┘

print("\n[Step 0.6] 正在向量化文档并构建 FAISS 索引...")
vectorstore = FAISS.from_documents(knowledge_base, embeddings)
print("[Step 0.6] FAISS 索引构建完成! 共索引 %d 个文档向量" % len(knowledge_base))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 1: Naive RAG 基线 — 直接检索 + 生成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ┌──────────────────── 本章目标 ────────────────────┐
# │ 1. 实现最基本的 RAG 流程                          │
# │ 2. 暴露"语义鸿沟"问题 —— 用户口语 vs 文档术语     │
# │ 3. 建立基线，后续章节对比改进效果                  │
# └──────────────────────────────────────────────────┘
#
# Naive RAG 流程:
#
#   用户问题 ──→ Embedding ──→ 向量检索 ──→ Top-K文档 ──→ Prompt ──→ LLM ──→ 答案
#                                  ↑
#                         就这一步有语义鸿沟！
#                    用户说"怎么避免撞人"
#                    文档写"行人检测与避障策略"

print("\n\n" + "=" * 70)
print("Chapter 1: Naive RAG 基线 — 直接检索 + 生成")
print("=" * 70)

# ─── 1.1 构建 Naive RAG Chain ───
naive_rag_prompt = ChatPromptTemplate.from_template(
    "你是一个技术助手。请根据以下参考文档回答用户的问题。\n"
    "如果文档中没有相关信息，请如实说明。\n\n"
    "参考文档:\n{context}\n\n"
    "用户问题: {question}\n\n"
    "回答:"
)


def format_docs(docs):
    """将检索到的文档格式化为字符串"""
    return "\n\n---\n\n".join(
        "[文档%d - %s] %s" % (i + 1, doc.metadata.get("chapter", ""), doc.page_content)
        for i, doc in enumerate(docs)
    )


# ─── 1.2 用口语化问题测试（暴露语义鸿沟） ───
#
# 这些问题故意用"日常口语"表达，和知识库中的"学术术语"有差距
test_questions = [
    "自动驾驶怎么避免撞到人？",  # 口语 vs "行人检测与避障策略"
    "大模型为什么会胡说八道？",  # 口语 vs "幻觉问题(Hallucination)"
    "无人车的大脑芯片要多强？",  # 口语 vs "算力需求分析"
]

print("\n[Step 1.2] 测试 Naive RAG（暴露语义鸿沟问题）")
print("-" * 50)

naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

naive_rag_chain = (
    {"context": naive_retriever | format_docs, "question": RunnablePassthrough()}
    | naive_rag_prompt
    | llm
    | StrOutputParser()
)

# 存储 Naive 结果用于后续对比
naive_results = {}

for q in test_questions:
    print('\n  问题: "%s"' % q)

    # 先单独看检索结果（不经过LLM）
    retrieved_docs = naive_retriever.invoke(q)
    print("  检索到的文档:")
    for i, doc in enumerate(retrieved_docs):
        print(
            "    [%d] %s (相关章节: %s)"
            % (i + 1, doc.page_content[:50] + "...", doc.metadata.get("chapter", ""))
        )

    # 完整 RAG 生成
    answer = naive_rag_chain.invoke(q)
    naive_results[q] = {
        "docs": [d.metadata.get("chapter", "") for d in retrieved_docs],
        "answer": answer[:100],
    }
    print("  Naive RAG 回答: %s..." % answer[:80])
    print()

print("\n[小结] Naive RAG 的局限性:")
print('  - 用户说"撞人"，文档写"行人检测/碰撞风险/TTC"')
print('  - 用户说"胡说八道"，文档写"幻觉(Hallucination)"')
print("  - 口语和术语之间的语义鸿沟导致检索质量下降")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 2: HyDE — 假设性文档嵌入 (Hypothetical Document Embeddings)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ┌──────────────────── 本章目标 ────────────────────┐
# │ 1. 理解 HyDE 的核心思想：用"假设答案"代替"问题"检索 │
# │ 2. 实现完整 HyDE 流程                             │
# │ 3. 对比 HyDE vs Naive 的检索效果                   │
# └──────────────────────────────────────────────────┘
#
# ┌───────────────── HyDE 工作原理 ─────────────────┐
# │                                                   │
# │  传统 Naive RAG:                                  │
# │    Q("怎么避免撞人") ──Embed──→ 搜索 ──→ 可能偏   │
# │                                                   │
# │  HyDE 策略:                                       │
# │    Q("怎么避免撞人")                               │
# │         │                                         │
# │         ▼ (LLM生成假设性回答)                      │
# │    H("自动驾驶通过感知模块的行人检测算法...")       │
# │         │                                         │
# │         ▼ (用假设回答做Embedding)                   │
# │    Embed(H) ──→ 搜索 ──→ 更精准！                 │
# │                                                   │
# │  为什么有效？                                      │
# │    假设回答H虽然可能不准确，但它的"表述风格"        │
# │    和知识库文档一致（都是技术性描述），             │
# │    所以在向量空间中更接近真实文档！                 │
# │                                                   │
# └───────────────────────────────────────────────────┘

print("\n\n" + "=" * 70)
print("Chapter 2: HyDE — 假设性文档嵌入 (Hypothetical Document Embeddings)")
print("=" * 70)

# ─── 2.1 构建假设性文档生成 Prompt ───
#
# 关键: 让 LLM 生成"像知识库文档一样"的回答
# 不需要准确（LLM可能会幻觉），只需要"语言风格"对齐
hyde_prompt = ChatPromptTemplate.from_template(
    "请针对以下问题，写一段技术性的回答文档。\n"
    "要求：\n"
    "- 使用专业术语和技术性表述\n"
    "- 像教材或技术文档的风格\n"
    "- 约100-200字\n"
    "- 不需要完全准确，重点是使用正确的技术词汇\n\n"
    "问题: {question}\n\n"
    "技术文档风格的回答:"
)

hyde_chain = hyde_prompt | llm | StrOutputParser()

print("\n[Step 2.1] HyDE Prompt 模板构建完成")

# ─── 2.2 实现完整 HyDE 检索流程 ───


def hyde_retrieve(question, k=3):
    """
    HyDE 检索流程:
    1. LLM生成假设性文档
    2. 用假设性文档做向量检索（而非原始问题）
    3. 返回检索结果
    """
    # Step 1: 生成假设性文档
    hypothetical_doc = hyde_chain.invoke({"question": question})

    # Step 2: 用假设性文档做向量检索
    # 注意：这里直接用文本检索，FAISS会自动embedding
    hyde_docs = vectorstore.similarity_search(hypothetical_doc, k=k)

    return hypothetical_doc, hyde_docs


# ─── 2.3 对比测试: HyDE vs Naive ───

print("\n[Step 2.3] 对比测试: HyDE vs Naive")
print("=" * 60)

hyde_results = {}

for q in test_questions:
    print("\n" + "-" * 60)
    print('  问题: "%s"' % q)
    print("-" * 60)

    # Naive 检索
    naive_docs = naive_retriever.invoke(q)
    print("\n  【Naive 检索结果】(直接用问题搜索):")
    for i, doc in enumerate(naive_docs):
        print(
            "    [%d] %s... (章节: %s)"
            % (i + 1, doc.page_content[:45], doc.metadata.get("chapter", ""))
        )

    # HyDE 检索
    hypo_doc, hyde_docs = hyde_retrieve(q)
    print("\n  【HyDE 假设性文档】(LLM生成):")
    print('    "%s..."' % hypo_doc[:100])
    print("\n  【HyDE 检索结果】(用假设文档搜索):")
    for i, doc in enumerate(hyde_docs):
        print(
            "    [%d] %s... (章节: %s)"
            % (i + 1, doc.page_content[:45], doc.metadata.get("chapter", ""))
        )

    # 对比
    naive_chapters = [d.metadata.get("chapter", "") for d in naive_docs]
    hyde_chapters = [d.metadata.get("chapter", "") for d in hyde_docs]
    print("\n  【对比】")
    print("    Naive 召回章节: %s" % naive_chapters)
    print("    HyDE  召回章节: %s" % hyde_chapters)

    hyde_results[q] = {
        "hypothetical": hypo_doc[:80],
        "docs": hyde_chapters,
    }

print("\n\n[HyDE 小结]")
print('  核心优势: 假设性文档使用了和知识库相同的"技术语言"')
print("  成本代价: 多了一次 LLM 调用（生成假设性文档）")
print("  适用场景: 用户口语化提问 + 知识库专业化的场景")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 3: Multi-Query — 多角度查询扩展
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ┌──────────────────── 本章目标 ────────────────────┐
# │ 1. 理解多角度查询扩展的动机                        │
# │ 2. 用LLM将一个问题拆成多个子问题                   │
# │ 3. 分别检索、合并去重，提升召回率                  │
# └──────────────────────────────────────────────────┘
#
# ┌─────────────── Multi-Query 工作原理 ───────────────┐
# │                                                      │
# │  原始问题: "自动驾驶怎么避免撞人？"                  │
# │       │                                              │
# │       ▼  (LLM 改写成多个角度)                        │
# │  ┌─────────────────────────────┐                    │
# │  │ Q1: "自动驾驶行人检测技术"    │──→ 检索 ──→ [D1,D2]│
# │  │ Q2: "自动紧急制动AEB原理"    │──→ 检索 ──→ [D2,D3]│
# │  │ Q3: "碰撞预警系统TTC算法"   │──→ 检索 ──→ [D2,D4]│
# │  │ Q4: "感知模块行人轨迹预测"   │──→ 检索 ──→ [D1,D2]│
# │  └─────────────────────────────┘                    │
# │       │                                              │
# │       ▼  (合并去重)                                  │
# │  最终文档集: [D1, D2, D3, D4]  ← 比单次检索多!       │
# │                                                      │
# │  三个侦探从不同方向搜索，总比一个人找得全！           │
# └──────────────────────────────────────────────────────┘

print("\n\n" + "=" * 70)
print("Chapter 3: Multi-Query — 多角度查询扩展")
print("=" * 70)

# ─── 3.1 构建查询扩展 Prompt ───
#
# 让 LLM 从多个角度重新表述问题
# 关键: 每个子问题要关注原始问题的不同方面

multi_query_prompt = ChatPromptTemplate.from_template(
    "你是一个搜索查询优化专家。\n"
    "请将下面的用户问题从不同角度改写成4个独立的搜索查询。\n"
    "要求：\n"
    "- 每个查询关注问题的不同方面或使用不同的技术术语\n"
    "- 查询要具体、适合在技术知识库中搜索\n"
    "- 每行一个查询，不要编号，不要额外解释\n\n"
    "用户问题: {question}\n\n"
    "4个改写后的搜索查询:"
)

multi_query_chain = multi_query_prompt | llm | StrOutputParser()

print("[Step 3.1] Multi-Query Prompt 模板构建完成")

# ─── 3.2 实现多查询检索 + 合并去重 ───


def multi_query_retrieve(question, k=3):
    """
    Multi-Query 检索流程:
    1. LLM 生成多个查询变体
    2. 每个查询独立检索
    3. 合并去重（按 doc_id 去重）
    """
    # Step 1: 生成多个查询
    raw_queries = multi_query_chain.invoke({"question": question})
    queries = [q.strip() for q in raw_queries.strip().split("\n") if q.strip()]

    # Step 2: 逐个检索
    all_docs = []
    seen_doc_ids = set()
    query_doc_map = {}  # 记录每个查询找到了什么

    for query in queries:
        docs = vectorstore.similarity_search(query, k=k)
        query_doc_map[query] = docs
        for doc in docs:
            doc_id = doc.metadata.get("doc_id", doc.page_content[:20])
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                all_docs.append(doc)

    return queries, query_doc_map, all_docs


# ─── 3.3 对比测试: Multi-Query vs Naive ───

print("\n[Step 3.3] 对比测试: Multi-Query vs Naive")
print("=" * 60)

multi_query_results = {}

for q in test_questions:
    print("\n" + "-" * 60)
    print('  原始问题: "%s"' % q)
    print("-" * 60)

    # Naive: 单次检索
    naive_docs = naive_retriever.invoke(q)
    naive_doc_ids = [d.metadata.get("doc_id", "") for d in naive_docs]

    # Multi-Query: 多次检索合并
    queries, query_doc_map, merged_docs = multi_query_retrieve(q)

    print("\n  【LLM 生成的多角度查询】:")
    for i, sub_q in enumerate(queries[:4]):
        sub_docs = query_doc_map.get(sub_q, [])
        chapters = [d.metadata.get("chapter", "") for d in sub_docs]
        print('    Q%d: "%s"' % (i + 1, sub_q[:50]))
        print("        → 召回: %s" % chapters)

    print("\n  【结果对比】:")
    print(
        "    Naive (单次检索): 召回 %d 篇，章节=%s"
        % (len(naive_docs), [d.metadata.get("chapter", "") for d in naive_docs])
    )
    print(
        "    Multi-Query (合并去重): 召回 %d 篇，章节=%s"
        % (len(merged_docs), [d.metadata.get("chapter", "") for d in merged_docs])
    )
    print(
        "    召回率提升: %d → %d 篇 (%.0f%% ↑)"
        % (
            len(naive_docs),
            len(merged_docs),
            (len(merged_docs) - len(naive_docs)) / max(len(naive_docs), 1) * 100,
        )
    )

    multi_query_results[q] = {
        "sub_queries": queries[:4],
        "naive_count": len(naive_docs),
        "multi_count": len(merged_docs),
    }

print("\n\n[Multi-Query 小结]")
print("  核心优势: 多角度覆盖，大幅提升召回率")
print("  成本代价: 多了一次 LLM 调用(查询扩展) + 多次向量检索")
print("  适用场景: 复杂问题（涉及多个知识点）、信息需求广泛的场景")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 4: Parent-Child — 父子文档检索
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ┌──────────────────── 本章目标 ────────────────────┐
# │ 1. 理解 Parent-Child 的动机：精确匹配 vs 完整上下文 │
# │ 2. 把文档拆成小块(child)做检索                     │
# │ 3. 命中后返回完整大块(parent)给 LLM                │
# └──────────────────────────────────────────────────┘
#
# ┌─────────────── Parent-Child 原理 ───────────────┐
# │                                                   │
# │  问题: 普通RAG用整段文档做embedding                │
# │    - 文档太长 → embedding被稀释，匹配不精准        │
# │    - 文档太短 → 缺少上下文，LLM无法理解全貌        │
# │                                                   │
# │  解决: 索引用小块(精确)，返回用大块(完整)          │
# │                                                   │
# │  ┌──────── Parent 文档(完整段落) ────────┐        │
# │  │                                        │        │
# │  │  ┌─Child 1─┐ ┌─Child 2─┐ ┌─Child 3─┐ │        │
# │  │  │ 句子1-2  │ │ 句子3-4  │ │ 句子5-6  │ │        │
# │  │  └────┬─────┘ └─────────┘ └─────────┘ │        │
# │  │       │                                 │        │
# │  └───────│─────────────────────────────────┘        │
# │          │                                          │
# │          ▼ (检索命中Child 1)                        │
# │                                                     │
# │    返回整个 Parent 文档给 LLM!                      │
# │    （书的索引定位 + 整章阅读）                       │
# │                                                     │
# └─────────────────────────────────────────────────────┘

print("\n\n" + "=" * 70)
print("Chapter 4: Parent-Child — 父子文档检索")
print("=" * 70)

# ─── 4.1 准备 Parent 文档 ───
#
# 用前面的知识库作为 parent 文档（完整段落）
# 然后把每个 parent 拆成 2-3 个 child 小块

print("\n[Step 4.1] 构建 Parent-Child 文档结构")


def split_into_children(parent_doc, chunk_size=80):
    """
    将 parent 文档拆成多个 child 小块

    策略: 按句号分割，每个 child 包含2-3个句子
    每个 child 记录其 parent_id，用于检索命中后回溯
    """
    content = parent_doc.page_content
    # 按句号分割（中文句号和英文句号都考虑）
    sentences = []
    current = ""
    for char in content:
        current += char
        if char in ("。", "；") and len(current) > 20:
            sentences.append(current)
            current = ""
    if current:
        sentences.append(current)

    # 每2个句子组成一个 child
    children = []
    for i in range(0, len(sentences), 2):
        child_content = "".join(sentences[i : i + 2])
        if child_content.strip():
            child_doc = Document(
                page_content=child_content,
                metadata={
                    "parent_id": parent_doc.metadata.get("doc_id", ""),
                    "parent_chapter": parent_doc.metadata.get("chapter", ""),
                    "child_index": i // 2,
                    "is_child": True,
                },
            )
            children.append(child_doc)

    return children


# 构建 parent 存储（用 dict 模拟，key=doc_id）
parent_store = {}
all_children = []

for parent_doc in knowledge_base:
    doc_id = parent_doc.metadata.get("doc_id", "")
    parent_store[doc_id] = parent_doc  # 保存完整 parent

    children = split_into_children(parent_doc)
    all_children.extend(children)

print("  Parent 文档: %d 篇" % len(parent_store))
print("  Child 小块:  %d 个" % len(all_children))
print("\n  示例 — Parent doc_2 拆分结果:")
doc2_children = [c for c in all_children if c.metadata.get("parent_id") == "doc_2"]
for i, child in enumerate(doc2_children):
    print('    Child[%d]: "%s..."' % (i, child.page_content[:50]))

# ─── 4.2 构建 Child 向量索引 ───
#
# 注意: 索引的是小块(child)，但最终返回的是大块(parent)

print("\n[Step 4.2] 构建 Child 向量索引...")
child_vectorstore = FAISS.from_documents(all_children, embeddings)
child_retriever = child_vectorstore.as_retriever(search_kwargs={"k": 3})
print("  Child 索引构建完成，共 %d 个小块向量" % len(all_children))

# ─── 4.3 实现 Parent-Child 检索 ───


def parent_child_retrieve(question, k=3):
    """
    Parent-Child 检索流程:
    1. 用问题检索 child 小块（精确匹配）
    2. 根据 child 的 parent_id 找到完整 parent 文档
    3. 返回 parent 文档（去重）
    """
    # Step 1: 检索 child
    child_hits = child_vectorstore.similarity_search(question, k=k)

    # Step 2: 通过 parent_id 回溯到 parent
    seen_parents = set()
    parent_docs = []

    for child in child_hits:
        parent_id = child.metadata.get("parent_id", "")
        if parent_id and parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent_doc = parent_store.get(parent_id)
            if parent_doc:
                parent_docs.append(parent_doc)

    return child_hits, parent_docs


# ─── 4.4 对比测试: Parent-Child vs Naive ───

print("\n[Step 4.4] 对比测试: Parent-Child vs Naive")
print("=" * 60)

# 用一个针对性的问题来演示
pc_test_questions = [
    "AEB自动紧急制动是什么？",  # 这个词只出现在 doc_2 的一小段
    "FAISS向量检索的特点？",  # 这个词只出现在 doc_9 的一小段
    "端到端方案的优势和劣势？",  # 这个出现在 doc_10 的中间部分
]

for q in pc_test_questions:
    print("\n" + "-" * 60)
    print('  问题: "%s"' % q)
    print("-" * 60)

    # Naive: 用整段文档检索
    naive_docs = naive_retriever.invoke(q)

    # Parent-Child: 用小块检索，返回完整段落
    child_hits, parent_docs = parent_child_retrieve(q)

    print("\n  【Naive 检索】(整段文档做 embedding):")
    for i, doc in enumerate(naive_docs[:2]):
        print(
            "    [%d] 章节=%s, 长度=%d字"
            % (i + 1, doc.metadata.get("chapter", ""), len(doc.page_content))
        )
        print('        "%s..."' % doc.page_content[:60])

    print("\n  【Parent-Child 检索】:")
    print("    Step 1 - Child 命中 (小块精确匹配):")
    for i, child in enumerate(child_hits[:3]):
        print('      Child[%d]: "%s..."' % (i, child.page_content[:50]))
        print("               → parent_id=%s" % child.metadata.get("parent_id", ""))

    print("    Step 2 - 返回 Parent (完整上下文):")
    for i, parent in enumerate(parent_docs[:2]):
        print(
            "      Parent[%d]: 章节=%s, 长度=%d字"
            % (i + 1, parent.metadata.get("chapter", ""), len(parent.page_content))
        )
        print('               "%s..."' % parent.page_content[:60])

    print("\n  【关键差异】:")
    print("    - Child 小块更精确地匹配了关键词")
    print("    - 但返回给 LLM 的是完整 Parent，保证上下文连贯")
    naive_total_len = sum(len(d.page_content) for d in naive_docs[:2])
    parent_total_len = sum(len(d.page_content) for d in parent_docs[:2])
    print(
        "    - Naive 上下文: %d字 | Parent-Child 上下文: %d字"
        % (naive_total_len, parent_total_len)
    )

# ─── 4.5 用 Parent-Child 结果做最终生成 ───

print("\n\n[Step 4.5] Parent-Child + LLM 生成示例")
print("-" * 50)

demo_question = "AEB自动紧急制动系统的工作原理是什么？"
print('  问题: "%s"' % demo_question)

child_hits, parent_docs = parent_child_retrieve(demo_question)
context = format_docs(parent_docs)

pc_answer = naive_rag_chain.invoke(demo_question)
print("  回答: %s" % pc_answer[:150])

print("\n[Parent-Child 小结]")
print("  核心优势: 小块精确匹配 + 大块完整上下文，两全其美")
print("  成本代价: 需要维护两层索引(child索引 + parent存储)")
print("  适用场景: 长文档、信息密集型知识库、需要完整段落回答的场景")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary: 策略对比总表
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n\n" + "=" * 70)
print("Summary: RAG 高级检索策略对比总表")
print("=" * 70)

print("""
┌─────────────┬──────────────────┬────────────────────┬──────────────────┐
│ 策略         │ 核心思想          │ 解决的问题          │ 代价             │
├─────────────┼──────────────────┼────────────────────┼──────────────────┤
│ Naive RAG   │ 直接用问题检索    │ (基线)              │ 无额外开销       │
├─────────────┼──────────────────┼────────────────────┼──────────────────┤
│ HyDE        │ LLM先生成假设答案 │ 语义鸿沟            │ +1次LLM调用      │
│             │ 用假设答案检索     │ (口语 vs 术语)      │ (生成假设文档)   │
├─────────────┼──────────────────┼────────────────────┼──────────────────┤
│ Multi-Query │ 一个问题拆多个角度│ 召回率不足          │ +1次LLM调用      │
│             │ 多次检索合并去重   │ (单视角遗漏)        │ +多次向量检索    │
├─────────────┼──────────────────┼────────────────────┼──────────────────┤
│ Parent-Child│ 小块索引+大块返回 │ 上下文不完整        │ +存储开销        │
│             │ 精确匹配+完整上下文│ (检索精度vs上下文)  │ +两层索引维护    │
└─────────────┴──────────────────┴────────────────────┴──────────────────┘

┌─────────────────── 组合使用建议 ───────────────────┐
│                                                      │
│  最佳实践: 这些策略可以叠加使用!                      │
│                                                      │
│  推荐组合 1 (高质量):                                │
│    HyDE + Parent-Child                               │
│    → 假设文档弥合语义鸿沟 + 完整上下文               │
│                                                      │
│  推荐组合 2 (高召回):                                │
│    Multi-Query + Reranking                           │
│    → 多角度提升召回 + 重排序筛选最优                  │
│                                                      │
│  推荐组合 3 (全能型):                                │
│    Multi-Query + HyDE + Parent-Child + Reranking     │
│    → 覆盖所有场景，但成本最高                        │
│                                                      │
└──────────────────────────────────────────────────────┘

┌─────────────────── 选型决策树 ───────────────────┐
│                                                    │
│  你的问题是什么？                                   │
│    │                                               │
│    ├─ "用户总用口语问，文档很专业"                  │
│    │   → 用 HyDE                                   │
│    │                                               │
│    ├─ "一个问题涉及多个知识点"                      │
│    │   → 用 Multi-Query                            │
│    │                                               │
│    ├─ "检索到了但回答不完整"                        │
│    │   → 用 Parent-Child                           │
│    │                                               │
│    └─ "以上都有"                                   │
│        → 组合使用                                   │
│                                                    │
└────────────────────────────────────────────────────┘
""")

print("=" * 70)
print("项目 17 完成! RAG 高级检索策略: HyDE + Multi-Query + Parent-Child")
print("=" * 70)
