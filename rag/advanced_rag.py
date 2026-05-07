"""
╔══════════════════════════════════════════════════════════════════╗
║         项目五：Advanced RAG — 高级检索与 Metadata 溯源            ║
║         元数据过滤 + 混合检索 + 重排序 + 引用来源追踪              ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普一：基础 RAG 的三大痛点——为什么需要 Advanced RAG？】
═══════════════════════════════════════════════════════════════════

回顾项目二，我们实现了一个简单的 RAG 系统。
它能"开卷考试"了，但在真实生产环境中会暴露三个致命问题：

  痛点一：检索不准（召回了不相关的内容）
  ┌─────────────────────────────────────────────────────────────┐
  │  问："公司2024年的营收是多少？"                                │
  │  基础 RAG 召回：2022年的财报、2023年的预测、员工手册……         │
  │  原因：向量相似度只看"语义接近"，不看"时间/页码/章节"         │
  └─────────────────────────────────────────────────────────────┘

  痛点二：无法溯源（用户不知道答案从哪来）
  ┌─────────────────────────────────────────────────────────────┐
  │  用户："你说公司营收 10 亿，依据是什么？"                      │
  │  基础 RAG："呃……我忘了从哪看的了。"                           │
  │  企业场景必须能回答"出处在第几页第几节"！                     │
  └─────────────────────────────────────────────────────────────┘

  痛点三：排序粗糙（向量距离 ≠ 真正相关性）
  ┌─────────────────────────────────────────────────────────────┐
  │  向量检索返回 10 个候选块，但排第一的不一定最相关。            │
  │  原因：Embedding 只捕捉"大致语义"，细粒度匹配能力有限。       │
  │  解法：用"重排序器"对候选结果二次打分，精选 Top-3。           │
  └─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
【前置科普二：Metadata（元数据）——给文本块贴标签】
═══════════════════════════════════════════════════════════════════

想象你去图书馆找一本书：

  没有元数据的图书馆：
    所有书堆在一个大仓库里，没有编号、没有分类。
    你问管理员："帮我找关于2024年财报的内容"
    管理员只能把所有书都翻一遍……效率极低。

  有元数据的图书馆：
    每本书都有标签：
      📖 书名 = "年度报告"
      📄 页码 = 42
      📂 章节 = "财务数据"
      📅 年份 = 2024

    管理员可以直接说："只搜第三章、2024年的内容"，
    瞬间把搜索范围缩小 10 倍！

  在 RAG 中：
    每个文本块（chunk）都可以附带一个 metadata 字典：
    {
        "source": "annual_report_2024.pdf",
        "page": 42,
        "section": "第三章：财务数据",
        "year": 2024
    }

    检索时可以加过滤条件：
      "只在 page > 30 且 section == '财务数据' 的块中搜索"

═══════════════════════════════════════════════════════════════════
【前置科普三：重排序（Re-ranking）——初试 + 复试】
═══════════════════════════════════════════════════════════════════

类比高考录取：

  ┌─────────────────────────────────────────────────────────────┐
  │  第一轮：初试（向量检索）                                     │
  │    从 10 万个文本块中，快速筛出 20 个"大致相关"的候选块。     │
  │    方式：计算向量余弦相似度，速度极快（毫秒级）。              │
  │    缺点：只看"大致意思"，可能混入不太相关的内容。             │
  │                                                             │
  │  第二轮：复试（重排序）                                       │
  │    对初试选出的 20 个候选块，逐一精细打分，选出 Top-3。        │
  │    方式：用更强的模型（Cross-Encoder）把"问题+候选块"配对打分。│
  │    优点：准确率远高于初试，能精确找出最相关的内容。            │
  │    缺点：速度慢（要逐个打分），所以只能对少量候选做。          │
  └─────────────────────────────────────────────────────────────┘

  为什么不直接用复试？
    因为对 10 万个文本块逐一精细打分太慢了！
    所以先用"快但粗"的初试缩小范围，再用"慢但准"的复试精选。

  在代码中：
    初试 = vectorstore.similarity_search(query, k=20)
    复试 = reranker.compress_documents(初试结果, query)  → Top-3
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# LangChain 聊天模型
from langchain_openai import ChatOpenAI

# 本地向量化模型
from langchain_huggingface import HuggingFaceEmbeddings

# 输出解析器 + 提示词模板
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# LCEL 并行与直传组件
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Document 对象：LangChain 统一的文档表示
from langchain_core.documents import Document

# 文本切块器
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS 向量数据库
from langchain_community.vectorstores import FAISS

# 用于实现轻量级重排序的工具
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化 LLM 和 Embeddings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 0 章：初始化 LLM 和 Embeddings")
print("=" * 60)

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
CHAT_MODEL = "kivy-kimi-k2_5"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

# RAG 场景 temperature=0：精确回答
llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.0,
)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

print("✅ LLM 初始化完成")
print(f"   聊天模型: {CHAT_MODEL}")
print("✅ Embeddings 初始化完成")
print(f"   向量化模型: {EMBEDDING_MODEL}")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：构造带丰富元数据的文档
# 目标：模拟一份多章节的企业报告，每个段落都带有 page/section 元数据
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：构造带丰富元数据的文档")
print("=" * 60)
print()

# ── 为什么要模拟"页码"和"章节"？────────────────────────────
#
# 真实场景：
#   用 PDFLoader 加载一份 100 页的 PDF，每页自动带上 page 元数据。
#   用 DirectoryLoader 加载多个文件，每个文件带上 source 元数据。
#
# 教学场景：
#   我们手动构造文档，显式给每个段落标注 page 和 section，
#   这样你能更清晰地看到"元数据是怎么附加的"以及"检索时怎么用"。
#
# ⚠️ 避坑指南：元数据设计原则
#   ① 只存"检索时用得上"的信息（page、section、source、date）
#   ② 不要存大段文本（那是 page_content 的事）
#   ③ 字段名统一用英文小写（方便 filter 查询）
#   ④ 数值型字段（page）用 int，方便做范围过滤

# ── 模拟一份"智驾科技公司2024年度报告" ───────────────────

REPORT_PAGES = [
    # ═══ 第一章：公司概况（第1-2页）═══
    {
        "content": """智驾科技成立于2018年，总部位于北京，是一家专注于自动驾驶技术研发的高科技企业。
公司拥有员工3200人，其中研发人员占比65%。截至2024年底，公司已获得自动驾驶相关专利487项，
在全国12个城市开展了L4级自动驾驶测试，累计测试里程超过2000万公里。
公司核心产品包括：智驾大脑（感知决策一体化平台）、高精地图引擎、车路协同系统。""",
        "page": 1,
        "section": "第一章：公司概况",
        "year": 2024,
    },
    {
        "content": """公司发展历程：
2018年：公司成立，获得天使轮融资5000万元。
2019年：发布第一代感知算法，获得北京自动驾驶路测牌照。
2020年：完成A轮融资3亿元，团队扩张至800人。
2021年：L4级自动驾驶系统在限定区域内实现商业化运营。
2022年：与三家主流车企达成量产合作协议。
2023年：年营收突破15亿元，完成C轮融资20亿元。
2024年：年营收达到28亿元，同比增长87%，实现首次盈利。""",
        "page": 2,
        "section": "第一章：公司概况",
        "year": 2024,
    },
    # ═══ 第二章：财务数据（第3-4页）═══
    {
        "content": """2024年度财务概要：
总营收：28.3亿元人民币（同比增长87%）
毛利润：11.2亿元（毛利率39.6%）
净利润：2.1亿元（首次实现全年盈利）
研发投入：9.8亿元（占营收34.6%）

营收构成：
- 智驾解决方案（OEM合作）：16.5亿元，占比58%
- 高精地图服务：5.2亿元，占比18%
- 自动驾驶运营服务（Robotaxi）：4.1亿元，占比15%
- 技术授权与咨询：2.5亿元，占比9%""",
        "page": 3,
        "section": "第二章：财务数据",
        "year": 2024,
    },
    {
        "content": """2024年各季度营收趋势：
Q1：5.2亿元（同比+62%）
Q2：6.8亿元（同比+78%）
Q3：7.5亿元（同比+91%）
Q4：8.8亿元（同比+112%）

增长驱动因素分析：
Q4营收增速显著加快，主要得益于：
1. 与理想汽车的量产合作在Q3开始交付，Q4进入放量阶段
2. Robotaxi业务在成都、深圳两城新增运营区域
3. 高精地图业务获得政府新基建项目大单（1.2亿元）""",
        "page": 4,
        "section": "第二章：财务数据",
        "year": 2024,
    },
    # ═══ 第三章：技术研发（第5-6页）═══
    {
        "content": """核心技术突破——感知系统：
2024年，公司自研的"天眼"多模态感知系统实现重大升级：
- 纯视觉方案（不依赖激光雷达）识别准确率从92%提升至97.3%
- 恶劣天气（暴雨/大雾）感知能力提升40%，得益于全新的Rain-Robust算法
- 首次实现对"鬼探头"场景（突然出现的行人）的100ms内响应
- 夜间场景识别准确率达到95%，接近白天水平

技术对比（与行业标杆Waymo对比）：
| 指标 | 智驾科技 | Waymo |
| 日间感知准确率 | 97.3% | 98.1% |
| 夜间感知准确率 | 95.0% | 96.2% |
| 恶劣天气表现 | 89.5% | 91.0% |
| 响应延迟 | 85ms | 72ms |

虽与Waymo仍有差距，但差距已从2023年的5-8%缩小至1-3%。""",
        "page": 5,
        "section": "第三章：技术研发",
        "year": 2024,
    },
    {
        "content": """核心技术突破——决策规划系统：
2024年发布的"智脑3.0"决策引擎，采用大模型+强化学习混合架构：

1. 端到端自动驾驶：
   告别传统的"感知→预测→规划"三段式流水线，
   实现从传感器输入到控制指令输出的一体化决策。
   复杂路口通过率从78%提升至93%。

2. 世界模型（World Model）：
   内部训练了一个"驾驶世界模拟器"，
   能预测未来5秒内周围交通参与者的行为轨迹。
   预测准确率：直行场景98%、左转场景91%、无保护左转85%。

3. 安全兜底机制：
   双冗余系统设计——主系统失效时，
   备份系统在200ms内接管车辆控制。
   2024年全年零安全事故（2000万公里测试）。""",
        "page": 6,
        "section": "第三章：技术研发",
        "year": 2024,
    },
    # ═══ 第四章：市场与竞争（第7-8页）═══
    {
        "content": """市场格局分析：
2024年中国自动驾驶市场规模约580亿元，同比增长45%。
主要竞争对手：
1. 百度Apollo：L4商业化运营最广，覆盖11个城市，日均订单超10万
2. 小马智行（Pony.ai）：获得出租车运营牌照最多，技术路线侧重激光雷达
3. 文远知行（WeRide）：已在美国上市，Robotaxi业务拓展至中东市场
4. 华为ADS：依托车企合作，2024年装车量超50万台，走量产路线

智驾科技的差异化竞争优势：
- 纯视觉方案成本优势：硬件成本比激光雷达方案低60%
- 端到端架构性能优势：复杂场景处理效率比传统架构高35%
- 车路协同生态优势：与12个城市政府签署智慧交通合作协议""",
        "page": 7,
        "section": "第四章：市场与竞争",
        "year": 2024,
    },
    {
        "content": """2025年战略规划：
目标营收：50亿元（同比增长77%）

核心战略方向：
1. 量产合作扩大：
   - 新增3家车企OEM合作，目标全年交付智驾系统100万套
   - 推出面向20万以下车型的低成本方案"智驾Lite"

2. Robotaxi规模化：
   - 新增5个城市运营，目标日均订单突破2万单
   - 与出行平台（滴滴/高德）深度合作，接入流量入口

3. 出海计划：
   - Q2在新加坡启动路测
   - Q4在沙特利雅得开展商业化试运营
   - 组建50人海外团队

4. 技术目标：
   - "天眼"系统日间准确率目标：99%
   - "智脑4.0"发布：首个千亿参数自动驾驶基础模型
   - L4全无人驾驶（无安全员）在3个城市获批""",
        "page": 8,
        "section": "第四章：市场与竞争",
        "year": 2024,
    },
]

# ── 将原始数据转换为 Document 对象 ─────────────────────────
#
# 关键操作：每个 Document 除了 page_content，还带上丰富的 metadata！
# 这些 metadata 就是"标签"，后续检索时可以按标签过滤。

documents = []
for page_data in REPORT_PAGES:
    doc = Document(
        page_content=page_data["content"],
        metadata={
            "source": "智驾科技2024年度报告.pdf",
            "page": page_data["page"],  # 📄 页码（int，支持范围过滤）
            "section": page_data["section"],  # 📂 章节（str，支持精确匹配）
            "year": page_data["year"],  # 📅 年份（int）
        },
    )
    documents.append(doc)

print(f"  ✅ 已构建 {len(documents)} 个带元数据的 Document 对象")
print()
print("  【Document 元数据示例（第1页）】")
print(f"    page_content: '{documents[0].page_content[:50]}...'")
print(f"    metadata: {documents[0].metadata}")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：智能切块 + 元数据继承
# 目标：切块时保留并继承原始文档的元数据
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：智能切块 + 元数据继承")
print("=" * 60)
print()

# ── 元数据继承的关键机制 ──────────────────────────────────
#
# 当 RecursiveCharacterTextSplitter.split_documents() 切块时：
#   ① 一个 Document 可能被切成多个小块
#   ② 每个小块会自动继承原始 Document 的 metadata！
#
# 例如第3页（财务数据）如果太长，被切成 chunk_A 和 chunk_B：
#   chunk_A.metadata = {"page": 3, "section": "第二章：财务数据", ...}
#   chunk_B.metadata = {"page": 3, "section": "第二章：财务数据", ...}
#
# 这就是 LangChain 的设计巧思：切块不丢元数据！

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # 较小的 chunk_size，制造更多切块（便于演示过滤效果）
    chunk_overlap=30,  # 相邻块重叠 30 字，防止句子被切断
    length_function=len,
)

chunks = text_splitter.split_documents(documents)

print(f"  原始 {len(documents)} 页文档 → 切成 {len(chunks)} 个块")
print()

# ── 打印切块结果，重点展示元数据继承 ──────────────────────

print("  【切块后的元数据分布】")
print("  ┌──────┬──────────────────────────┬──────────────────┐")
print("  │ 块号 │ 页码(page)               │ 章节(section)    │")
print("  ├──────┼──────────────────────────┼──────────────────┤")
for i, chunk in enumerate(chunks):
    page = chunk.metadata["page"]
    section = chunk.metadata["section"]
    content_preview = chunk.metadata.get("source", "")[:15]
    print(
        f"  │ {i:4d} │ 第 {page} 页{' ' * (20 - len(str(page)))}│ {section[:16]:16s} │"
    )
print("  └──────┴──────────────────────────┴──────────────────┘")
print()
print("  💡 观察：每个 chunk 都保留了来源页码和章节信息！")
print("     即使文本被切碎，我们仍然知道它来自哪里。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：向量化 + FAISS 建索引
# 目标：把切块后的文档存入向量数据库（和项目二类似，但带元数据）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：向量化 + FAISS 建索引")
print("=" * 60)
print()
print("  【正在向量化所有文本块...】")

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)

print(f"  ✅ FAISS 索引建立完成！")
print(f"     已存入向量数量: {vectorstore.index.ntotal}")
print(f"     每个向量都关联着对应 chunk 的 metadata")
print()

# ── FAISS 中元数据的存储方式 ──────────────────────────────
#
# FAISS 本身只存向量（float 数组），不存文本和元数据！
# LangChain 的 FAISS 包装器做了额外工作：
#   在 FAISS 索引旁边，用 Python dict 维护了 {向量ID: Document} 的映射。
#   检索时：FAISS 返回相似向量的 ID → LangChain 再用 ID 查找对应的 Document。
#
# 这意味着：metadata 完整保留在 Document 对象中，检索后可以直接访问！


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：Metadata Filtering（元数据过滤检索）
# 目标：演示如何在检索时用 metadata 缩小搜索范围
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 4 章：Metadata Filtering（元数据过滤检索）")
print("=" * 60)
print()

# ── 元数据过滤的原理 ─────────────────────────────────────
#
# 普通检索：在所有 chunk 中搜索
#   vectorstore.similarity_search("营收", k=3)
#
# 带过滤的检索：先按 metadata 筛选，再在子集中搜索
#   vectorstore.similarity_search("营收", k=3, filter={"section": "第二章：财务数据"})
#
# FAISS 的 filter 实现方式：
#   FAISS 包装器先在 Python 层按 metadata 过滤出符合条件的 Document ID，
#   然后只在这些 ID 对应的向量中做相似度搜索。
#   （注意：这是"先过滤后搜索"，对大数据集效率较低；
#    生产环境推荐 Pinecone/Weaviate 等原生支持过滤的向量数据库）
#
# ⚠️ 避坑指南：FAISS 的 filter 只支持精确匹配！
#   filter={"page": 3}        ✅ 精确匹配
#   filter={"page": {"$gt": 3}} ❌ FAISS 不支持范围查询
#   如需范围查询，使用 Chroma/Pinecone 等向量数据库


def demo_metadata_filter(query: str, metadata_filter: dict, description: str):
    """演示带元数据过滤的检索"""
    print(f"  🔍 问题：「{query}」")
    print(f"     过滤条件：{metadata_filter}")
    print(f"     场景说明：{description}")
    print()

    # 带 filter 的检索
    filtered_results = vectorstore.similarity_search(query, k=3, filter=metadata_filter)

    # 不带 filter 的检索（对比）
    unfiltered_results = vectorstore.similarity_search(query, k=3)

    print(f"  【不带过滤】召回的块来自：")
    for i, doc in enumerate(unfiltered_results):
        print(f"    [{i + 1}] 第{doc.metadata['page']}页 - {doc.metadata['section']}")
        print(f"        内容预览：{doc.page_content[:60]}...")

    print()
    print(f"  【带过滤 filter={metadata_filter}】召回的块来自：")
    for i, doc in enumerate(filtered_results):
        print(f"    [{i + 1}] 第{doc.metadata['page']}页 - {doc.metadata['section']}")
        print(f"        内容预览：{doc.page_content[:60]}...")

    print()
    print(f"  💡 对比：过滤后的结果更精准，避免了无关章节的干扰！")
    print()
    return filtered_results


# ── 演示一：只搜"财务数据"章节 ──────────────────────────

print("━" * 60)
print("【演示一：只在财务章节中搜索营收信息】")
print("━" * 60)

demo_metadata_filter(
    query="公司2024年的总营收是多少？",
    metadata_filter={"section": "第二章：财务数据"},
    description="用户问财务问题，只在财务章节中搜索",
)

# ── 演示二：只搜技术章节 ─────────────────────────────────

print("━" * 60)
print("【演示二：只在技术章节中搜索感知系统信息】")
print("━" * 60)

demo_metadata_filter(
    query="感知系统的识别准确率是多少？",
    metadata_filter={"section": "第三章：技术研发"},
    description="用户问技术问题，只在技术章节中搜索",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 章：重排序（Re-ranking）——用 Embedding 相似度二次精选
# 目标：对初步检索的结果做二次打分，过滤掉低相关性的噪声
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 5 章：重排序（Re-ranking）——二次精选")
print("=" * 60)
print()

# ── 重排序方案选择 ─────────────────────────────────────────
#
# 方案一（生产级）：Cohere Reranker / BGE-Reranker
#   需要额外的 API Key 或下载专用模型（Cross-Encoder），
#   效果最好，但本教学环境不方便配置。
#
# 方案二（轻量级，本项目采用）：EmbeddingsFilter
#   利用已有的 Embedding 模型做二次相似度打分，
#   设定一个阈值（如 similarity_threshold=0.5），
#   低于阈值的文档块直接丢弃。
#
#   虽然不如 Cross-Encoder 精确，但：
#   ① 不需要额外模型/API
#   ② 能有效过滤"语义距离较远"的噪声块
#   ③ 原理完全一致——都是对初选结果做二次打分
#
# 方案三（本项目也演示）：EmbeddingsRedundantFilter
#   去重过滤器：如果候选结果中有两个块内容高度相似（冗余），
#   只保留其中一个，避免提供给 LLM 重复的信息。
#
# ⚠️ 避坑指南：重排序 ≠ 重新搜索！
#   重排序只在已有的候选结果（如 20 个块）中做二次筛选，
#   不会去向量数据库中重新搜索。它的输入是"初选结果列表"。

# ── 构建重排序管道 ─────────────────────────────────────────

# 过滤器一：去冗余（去掉内容高度相似的重复块）
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)

# 过滤器二：相关性阈值过滤（低于阈值的块直接丢弃）
# similarity_threshold：0-1 之间，越高越严格
# 0.3 是一个比较宽松的阈值（bge 模型的相似度分布偏低），实际使用时需要根据模型调整
relevance_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.3,
)

# 把多个过滤器组成一个"管道"（Pipeline）
# 数据流：初选结果 → 去冗余 → 相关性过滤 → 最终精选结果
compressor_pipeline = DocumentCompressorPipeline(
    transformers=[redundant_filter, relevance_filter]
)

# ── 构建"带重排序的检索器" ─────────────────────────────────
#
# ContextualCompressionRetriever 把"基础检索器 + 重排序器"组合：
#   ① 先用 base_retriever 做初选（返回 k=6 个候选）
#   ② 再用 base_compressor 做二次过滤（去冗余 + 阈值筛选）
#   ③ 最终返回精选后的结果（通常 < 6 个）

base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 6}  # 初选：取 6 个候选（比最终需要的多，留给重排序筛选）
)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor_pipeline,
    base_retriever=base_retriever,
)

print("  ✅ 重排序检索器构建完成！")
print("     流程：初选(k=6) → 去冗余 → 相关性过滤 → 精选结果")
print()

# ── 演示重排序效果 ────────────────────────────────────────

print("━" * 60)
print("【演示：重排序前后的对比】")
print("━" * 60)
print()

demo_query = "公司的自动驾驶技术和Waymo相比如何？"
print(f"  🔍 问题：「{demo_query}」")
print()

# 初选结果（不经过重排序）
raw_results = base_retriever.invoke(demo_query)
print(f"  【初选结果】共 {len(raw_results)} 个块：")
for i, doc in enumerate(raw_results):
    print(f"    [{i + 1}] 第{doc.metadata['page']}页 | {doc.metadata['section']}")
    print(f"        {doc.page_content[:70]}...")
print()

# 重排序后的结果
reranked_results = reranking_retriever.invoke(demo_query)
print(f"  【重排序后】精选为 {len(reranked_results)} 个块：")
for i, doc in enumerate(reranked_results):
    print(f"    [{i + 1}] 第{doc.metadata['page']}页 | {doc.metadata['section']}")
    print(f"        {doc.page_content[:70]}...")
print()
print("  💡 重排序去掉了冗余和低相关性的块，结果更精准！")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 6 章：完整 Advanced RAG 问答 + 引用溯源
# 目标：实现"回答 + 参考来源"的完整输出
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 6 章：完整 Advanced RAG 问答 + 引用溯源")
print("=" * 60)
print()

# ── 构建带溯源的 RAG 链 ──────────────────────────────────
#
# 与项目二的区别：
#   项目二：只返回答案文本
#   项目五：返回答案 + 引用来源（哪几页、哪个章节）
#
# 实现思路：
#   ① 检索阶段：保留完整的 Document 对象（含 metadata）
#   ② 生成阶段：只把 page_content 喂给 LLM
#   ③ 输出阶段：把 metadata 中的 page/section 作为引用来源展示
#
# 为什么不让 LLM 自己标注来源？
#   因为 LLM 可能会"编造"引用！
#   正确做法：由程序逻辑（而非 LLM）负责追踪引用来源。

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是智驾科技公司的内部知识库问答助手。
请严格根据下面提供的【参考资料】来回答用户的问题。
如果参考资料中没有相关信息，请直接说"根据现有资料，我无法回答这个问题"。
不要编造或猜测参考资料中没有的内容。

回答要求：
1. 准确引用数据（数字、百分比等）
2. 回答简洁清晰，控制在 200 字以内
3. 如果涉及对比，请列出具体数据

【参考资料】
{context}""",
        ),
        ("human", "{question}"),
    ]
)

parser = StrOutputParser()


def advanced_rag_query(question: str, use_filter: dict = None) -> None:
    """
    执行完整的 Advanced RAG 问答，输出：
      ① 检索召回的文档块（含来源信息）
      ② LLM 的回答
      ③ 引用来源追踪（页码 + 章节）

    参数：
      question   : 用户的问题
      use_filter : 可选的元数据过滤条件
    """
    print("╔" + "═" * 58 + "╗")
    print(f"║  ❓ 问题：{question[:46]}")
    print("╚" + "═" * 58 + "╝")
    print()

    # ── 步骤一：检索（带重排序 + 可选的元数据过滤）──────────
    if use_filter:
        # 带元数据过滤的检索
        print(f"  📂 元数据过滤条件：{use_filter}")
        # 先用 filter 做初选
        filtered_docs = vectorstore.similarity_search(question, k=6, filter=use_filter)
        # 再用重排序管道做二次过滤
        retrieved_docs = compressor_pipeline.compress_documents(filtered_docs, question)
    else:
        # 不带过滤，直接用重排序检索器
        retrieved_docs = reranking_retriever.invoke(question)

    print(f"  🔍 检索完成：召回 {len(retrieved_docs)} 个相关文本块")
    print()

    # ── 步骤二：展示检索到的文本块（含元数据）──────────────
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │ 📚 检索召回的参考资料                                 │")
    print("  ├─────────────────────────────────────────────────────┤")
    for i, doc in enumerate(retrieved_docs, 1):
        page = doc.metadata["page"]
        section = doc.metadata["section"]
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  │ [{i}] 📄 第{page}页 | {section}")
        print(f"  │     「{preview}...」")
    print("  └─────────────────────────────────────────────────────┘")
    print()

    # ── 步骤三：拼接上下文，调用 LLM 生成回答 ─────────────
    # 把检索到的文档块拼成字符串，作为 LLM 的"参考资料"
    context_text = "\n\n---\n\n".join(
        f"[来源: 第{doc.metadata['page']}页, {doc.metadata['section']}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    # 调用 RAG 链
    chain = rag_prompt | llm | parser
    answer = chain.invoke(
        {
            "context": context_text,
            "question": question,
        }
    )

    # ── 步骤四：输出回答 + 引用来源 ──────────────────────
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │ 🤖 AI 回答                                           │")
    print("  ├─────────────────────────────────────────────────────┤")
    # 分行打印答案
    for line in answer.split("\n"):
        print(f"  │   {line}")
    print("  └─────────────────────────────────────────────────────┘")
    print()

    # ── 步骤五：引用来源追踪（⭐ 核心亮点！）─────────────
    #
    # 这一步是 Advanced RAG 的精华：
    # 不是让 LLM 告诉你"出处"，而是由程序直接从 metadata 中提取！
    # 这样引用来源 100% 可靠，不会有 LLM 幻觉的问题。

    # 收集所有引用来源（去重）
    sources = []
    seen_pages = set()
    for doc in retrieved_docs:
        page = doc.metadata["page"]
        if page not in seen_pages:
            seen_pages.add(page)
            sources.append(
                {
                    "page": page,
                    "section": doc.metadata["section"],
                    "source": doc.metadata["source"],
                }
            )

    # 按页码排序
    sources.sort(key=lambda x: x["page"])

    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │ 📖 引用来源（Metadata 溯源）                          │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  文档：{sources[0]['source'] if sources else '未知'}")
    page_list = ", ".join(f"第{s['page']}页" for s in sources)
    print(f"  │  引用页码：{page_list}")
    print(f"  │  涉及章节：")
    for s in sources:
        print(f"  │    • 第{s['page']}页 → {s['section']}")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  💡 以上引用来源由程序从 metadata 中直接提取，")
    print("     不依赖 LLM 生成，100% 可靠可追溯！")
    print()
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 7 章：运行三轮问答演示
# 目标：展示 Advanced RAG 的完整能力
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 7 章：运行问答演示")
print("=" * 60)
print()

# ── 问答一：财务问题（带元数据过滤）─────────────────────
print("━" * 60)
print("【问答一：财务问题 + 元数据过滤】")
print("━" * 60)

advanced_rag_query(
    question="公司2024年的总营收和净利润分别是多少？同比增长了多少？",
    use_filter={"section": "第二章：财务数据"},
)

# ── 问答二：技术问题（不带过滤，纯重排序）────────────────
print("━" * 60)
print("【问答二：技术问题 + 重排序精选】")
print("━" * 60)

advanced_rag_query(
    question="智驾科技的感知系统和Waymo相比，在哪些指标上有差距？",
)

# ── 问答三：战略问题（跨章节）─────────────────────────────
print("━" * 60)
print("【问答三：综合问题 + 跨章节检索】")
print("━" * 60)

advanced_rag_query(
    question="公司2025年有哪些出海计划？目标营收是多少？",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 总结
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("🎉 项目五学习完毕！")
print("=" * 60)
print()
print("💡 核心升级对比（项目二 vs 项目五）：")
print()
print("  ┌─────────────────┬──────────────────────────────────┐")
print("  │  项目二（基础RAG）│  项目五（Advanced RAG）           │")
print("  ├─────────────────┼──────────────────────────────────┤")
print("  │  无元数据         │  page/section/year 标签          │")
print("  │  全量搜索         │  Metadata Filtering 精确过滤     │")
print("  │  单次检索         │  初选 + 重排序（二次精选）        │")
print("  │  无法溯源         │  引用来源 100% 可追踪             │")
print("  │  可能召回噪声     │  去冗余 + 相关性阈值过滤          │")
print("  └─────────────────┴──────────────────────────────────┘")
print()
print("💡 生产环境进阶方向：")
print("   ① 向量数据库换 Pinecone/Weaviate → 原生支持范围过滤")
print("   ② 重排序换 BGE-Reranker/Cohere → Cross-Encoder 精度更高")
print("   ③ 混合检索：BM25（关键词）+ Vector（语义）双路召回")
print("   ④ 查询改写：用 LLM 改写用户问题，提升检索命中率")
print("   ⑤ 多文档溯源：支持多份文档的交叉引用和来源标注")
print("=" * 60)
