"""
╔══════════════════════════════════════════════════════════════════╗
║         项目十二：向量数据库持久化存储                              ║
║         FAISS 本地存储 + Chroma 持久化 + 增量更新                  ║
╚══════════════════════════════════════════════════════════════════╝

【学前必读：为什么需要持久化？】

回顾一下我们之前的向量数据库用法：

  docs = [Document(...), ...]
  db = FAISS.from_documents(docs, embeddings)  ← 每次运行都重新算！

这有一个巨大的问题：
  每次启动程序，都要把所有文档重新"向量化"一遍。
  如果你有 10 万篇文章，每次启动都要等几十分钟，还要花钱调 embedding API！

【持久化 vs 内存存储】

  内存存储（之前的用法）：
  ┌────────────────────────────────────────┐
  │  程序启动 → 算向量 → 存内存 → 程序退出 → 数据消失  │
  └────────────────────────────────────────┘

  持久化存储（本章的用法）：
  ┌────────────────────────────────────────────────────┐
  │  首次运行：算向量 → 存硬盘                             │
  │  后续运行：直接从硬盘加载（秒级！不用重新计算！）          │
  └────────────────────────────────────────────────────┘

【主流向量数据库选型指南】

  ┌──────────┬──────────────┬────────────────┬─────────────────┐
  │  数据库   │  存储方式     │  适合场景       │  特点            │
  ├──────────┼──────────────┼────────────────┼─────────────────┤
  │  FAISS   │  本地文件     │  中小规模离线    │  快、无服务器     │
  │  Chroma  │  本地/远程    │  开发/生产均可   │  支持元数据过滤   │
  │  Pinecone│  云服务       │  大规模生产      │  托管、高可用     │
  │  Weaviate│  本地/云      │  企业级         │  支持混合搜索     │
  └──────────┴──────────────┴────────────────┴─────────────────┘

【增量更新的意义】

  想象你维护一个公司知识库：
  - 今天新增了 5 篇政策文件 → add_documents()，不用重建整个库
  - 某篇文件过期了 → delete()，精确删除
  - 某篇文件内容更新了 → upsert，先删后加
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# os：检查文件/目录是否存在，决定是"首次建库"还是"加载已有库"
import os

# shutil：用于清理测试产生的临时文件（演示结束后清理现场）
import shutil

# HuggingFaceEmbeddings：本地运行的 embedding 模型，无需 API Key，完全离线
# BAAI/bge-small-zh-v1.5 是专门为中文优化的小型高效模型
from langchain_huggingface import HuggingFaceEmbeddings

# Document：LangChain 的文档对象
# page_content = 文本内容，metadata = 附加属性（来源、分类、日期等）
from langchain_core.documents import Document

# FAISS：Facebook AI Research 的向量库，本地文件持久化
# save_local() / load_local() 是它的持久化接口
from langchain_community.vectorstores import FAISS

# Chroma：功能更强的向量数据库，支持元数据过滤、按 ID 删除等高级操作
# persist_directory 参数直接指定持久化目录，无需手动调用保存
from langchain_chroma import Chroma


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化 Embedding 模型 & 准备测试文档
# 目标：了解本地 embedding 模型的加载方式，准备演示数据
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 0 章：初始化 Embedding 模型")
print("=" * 60)

# ⚠️ 避坑指南：本地模型 vs API 模型
#
# OpenAI text-embedding-ada-002（API 模型）：
#   优点：效果好，无需本地算力
#   缺点：每次调用都要花钱、需要网络、有速率限制
#
# BAAI/bge-small-zh-v1.5（本地模型）：
#   优点：完全免费、离线可用、速度稳定
#   缺点：首次运行需要下载模型文件（~100MB）
#
# 对于持久化场景，本地模型更合适：
#   首次建库时算一次，之后加载时完全不需要重新算 embedding！

EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

print(f"正在加载 Embedding 模型：{EMBEDDING_MODEL}")
print("（首次运行会下载模型文件，约 100MB，请耐心等待...）")
print()

# model_kwargs={'device': 'cpu'} 指定在 CPU 上运行（无 GPU 也能跑）
# encode_kwargs={'normalize_embeddings': True} 对向量归一化，提升余弦相似度计算精度
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("Embedding 模型加载完成！")
print()

# ─── 准备测试文档 ─────────────────────────────────────────

# 模拟一个"智驾科技公司"的知识库文档
# metadata 字段很重要：它们可以用于后续的过滤检索
docs_initial = [
    Document(
        page_content="智驾科技成立于2020年，专注于自动驾驶感知算法研发，"
                     "总部位于北京，员工规模500人。",
        metadata={"id": "doc_001", "category": "公司简介", "year": 2020},
    ),
    Document(
        page_content="智驾科技的核心产品是视觉感知系统VisionCore，"
                     "已搭载于国内多家主流汽车品牌，累计行驶里程超过10亿公里。",
        metadata={"id": "doc_002", "category": "产品介绍", "year": 2023},
    ),
    Document(
        page_content="智驾科技2023年完成B轮融资，融资金额5亿元人民币，"
                     "投资方包括红杉资本和多家汽车主机厂战略投资。",
        metadata={"id": "doc_003", "category": "融资信息", "year": 2023},
    ),
    Document(
        page_content="智驾科技技术团队由前百度、华为自动驾驶部门核心成员组成，"
                     "拥有发明专利200余项，覆盖感知、规划、控制全栈。",
        metadata={"id": "doc_004", "category": "团队介绍", "year": 2022},
    ),
]

print("【测试文档准备完毕】")
for doc in docs_initial:
    print(f"  [{doc.metadata['id']}] {doc.metadata['category']}: "
          f"{doc.page_content[:30]}...")
print()

# 持久化目录配置
FAISS_PATH = "./vectordb/faiss_store"
CHROMA_PATH = "./vectordb/chroma_store"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：FAISS 本地持久化
# 目标：掌握 save_local / load_local，理解"建库"vs"加载"的逻辑
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：FAISS 本地持久化")
print("=" * 60)
print()

# ─── 核心逻辑：判断是"建库"还是"加载" ────────────────────

# 这个判断是持久化的关键模式！
# os.path.exists() 检查目录是否存在（即历史上是否建过库）
if os.path.exists(FAISS_PATH):
    # ── 情况 A：库已存在，直接加载（毫秒级！）──
    print(f"检测到已有 FAISS 库：{FAISS_PATH}")
    print("直接从磁盘加载，无需重新计算向量...")

    # load_local() 参数说明：
    #   folder_path  = 存储目录
    #   embeddings   = 必须传入，用于后续新查询的向量化（不是用来重算已有向量）
    #   allow_dangerous_deserialization = True（FAISS 使用 pickle，需要显式允许）
    faiss_db = FAISS.load_local(
        folder_path=FAISS_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    print("FAISS 库加载成功！")
else:
    # ── 情况 B：首次运行，建库并保存 ──
    print(f"未检测到已有 FAISS 库，首次建库中...")
    print("正在对文档进行向量化（这一步会调用 embedding 模型）...")

    # from_documents() 做了两件事：
    # ① 对每个 doc.page_content 调用 embeddings.embed_documents() 得到向量
    # ② 把文本 + 向量存入内存中的 FAISS 索引
    faiss_db = FAISS.from_documents(docs_initial, embeddings)

    # save_local() 把内存中的 FAISS 索引序列化到磁盘
    # 会生成两个文件：index.faiss（向量索引）和 index.pkl（文档内容+元数据）
    faiss_db.save_local(FAISS_PATH)
    print(f"FAISS 库已保存到：{FAISS_PATH}")
    print(f"  生成文件：{FAISS_PATH}/index.faiss")
    print(f"  生成文件：{FAISS_PATH}/index.pkl")

print()

# ─── 演示：向已有 FAISS 库追加文档 ──────────────────────

print("【演示：向 FAISS 库追加新文档】")

new_doc_faiss = Document(
    page_content="智驾科技于2024年推出第二代感知芯片DriveChip X2，"
                 "算力达到200TOPS，支持8路摄像头并行处理。",
    metadata={"id": "doc_005", "category": "产品介绍", "year": 2024},
)

# add_documents() 只追加新文档，不会重算已有文档的向量
faiss_db.add_documents([new_doc_faiss])

# ⚠️ 避坑指南：add_documents 后必须重新 save_local！
# FAISS 是"内存优先"的：add_documents 只更新内存，不会自动写磁盘。
# 如果不重新 save_local，下次加载时新文档会丢失！
faiss_db.save_local(FAISS_PATH)
print(f"新文档已追加并重新保存到磁盘")
print()

# ─── 演示：检索测试 ────────────────────────────────────

print("【FAISS 检索测试】")
query = "智驾科技的融资情况"
faiss_results = faiss_db.similarity_search(query, k=2)
print(f"查询：'{query}'")
print(f"Top-2 结果：")
for i, doc in enumerate(faiss_results, 1):
    print(f"  {i}. [{doc.metadata.get('id', 'N/A')}] {doc.page_content[:50]}...")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：Chroma 向量数据库
# 目标：掌握 Chroma 的持久化方式，理解 collection 和 document ID
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：Chroma 向量数据库")
print("=" * 60)
print()

print("【FAISS vs Chroma 的核心区别】")
print("""
  FAISS：
    - Facebook 出品的纯向量检索库
    - 不是数据库，没有"删除某条记录"的原生支持
    - 持久化需要手动调用 save_local()
    - 优势：检索速度极快，适合亿级向量

  Chroma：
    - 专为 LLM 应用设计的向量数据库
    - 支持按 ID 精确删除、更新（upsert）
    - persist_directory 自动持久化（写入即落盘）
    - 支持 metadata 过滤（$eq, $in, $gt 等操作符）
    - 适合中小规模、需要灵活增删改的场景
""")

# ─── 初始化 Chroma（同样判断是否首次建库）─────────────────

# 为文档准备显式的 ID 列表
# ⚠️ 避坑指南：如果不指定 ids，Chroma 会自动生成 UUID
# 但自动生成的 UUID 你无法知道，也就无法后续做定向删除/更新！
# 最佳实践：业务层面生成有意义的 ID，例如文档名、数据库主键等
doc_ids = [doc.metadata["id"] for doc in docs_initial]

if os.path.exists(CHROMA_PATH):
    print(f"检测到已有 Chroma 库：{CHROMA_PATH}")
    print("直接加载已有库...")

    # Chroma 的加载方式：只需提供相同的 persist_directory 和 collection_name
    # Chroma 会自动恢复上次的状态
    chroma_db = Chroma(
        collection_name="knowledge_base",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    print("Chroma 库加载成功！")
else:
    print(f"未检测到已有 Chroma 库，首次建库中...")

    # from_documents() + persist_directory：建库并自动持久化
    # collection_name：一个 Chroma 实例可以有多个 collection（类似数据库的"表"）
    # ids：显式指定每个文档的唯一标识符
    chroma_db = Chroma.from_documents(
        documents=docs_initial,
        embedding=embeddings,
        collection_name="knowledge_base",
        persist_directory=CHROMA_PATH,
        ids=doc_ids,
    )
    print(f"Chroma 库已持久化到：{CHROMA_PATH}")

print()

# ─── 查看库中文档数量 ─────────────────────────────────────

# _collection.count() 返回当前 collection 中的文档总数
count = chroma_db._collection.count()
print(f"【当前 Chroma 库文档数量】：{count} 条")
print()

# ─── 检索测试 ─────────────────────────────────────────────

print("【Chroma 基础检索测试】")
query = "自动驾驶核心产品"
chroma_results = chroma_db.similarity_search(query, k=2)
print(f"查询：'{query}'")
print(f"Top-2 结果：")
for i, doc in enumerate(chroma_results, 1):
    print(f"  {i}. [{doc.metadata.get('id', 'N/A')}] {doc.page_content[:50]}...")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：增量操作——增、删、改
# 目标：掌握 add_documents / delete / upsert 三个核心操作
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：增量操作——增、删、改")
print("=" * 60)
print()

# ─── 操作一：新增文档（add_documents）────────────────────

print("【操作 1：新增文档】")

new_docs = [
    Document(
        page_content="智驾科技与多家城市签署智慧交通合作协议，"
                     "计划在2025年实现10座城市的智能网联覆盖。",
        metadata={"id": "doc_006", "category": "战略合作", "year": 2024},
    ),
]

# add_documents() 追加文档；ids 参数确保 ID 可控
chroma_db.add_documents(new_docs, ids=["doc_006"])

count_after_add = chroma_db._collection.count()
print(f"新增 doc_006 后，文档数量：{count_after_add} 条")
print()

# ─── 操作二：删除文档（delete）───────────────────────────

print("【操作 2：删除文档（按 ID 精确删除）】")
print("删除 doc_004（团队介绍文档）...")

# delete() 接受一个 ID 列表，支持批量删除
# ⚠️ 避坑指南：这里的 ID 是建库时传入的 ids，不是 metadata 里的 "id" 字段！
#   两者在这个例子里值相同，但概念不同：
#   - Chroma 内部 ID：建库时通过 ids 参数设置，是 Chroma 索引的 key
#   - metadata["id"]：文档的业务 ID，只是一个普通属性，无法直接用于 delete()
chroma_db.delete(ids=["doc_004"])

count_after_delete = chroma_db._collection.count()
print(f"删除 doc_004 后，文档数量：{count_after_delete} 条")
print()

# ─── 操作三：更新/替换文档（upsert）──────────────────────

print("【操作 3：更新文档（upsert = 先删后增）】")
print("更新 doc_001（公司简介）的内容...")

# Chroma 原生没有"update"操作，标准做法是"先 delete，再 add"
# 有些版本的 Chroma 支持 update() 方法，但 upsert 模式更通用
updated_doc = Document(
    page_content="智驾科技成立于2020年，专注于自动驾驶感知算法研发，"
                 "总部位于北京，员工规模已扩展至800人，在上海、深圳设有研发中心。",
    metadata={"id": "doc_001", "category": "公司简介", "year": 2024},
)

# 先删除旧版本
chroma_db.delete(ids=["doc_001"])
# 再添加新版本（保持相同 ID）
chroma_db.add_documents([updated_doc], ids=["doc_001"])

count_after_update = chroma_db._collection.count()
print(f"更新 doc_001 后，文档数量：{count_after_update} 条（总数不变）")
print()

# 验证更新是否生效
print("验证 doc_001 的最新内容：")
verify_results = chroma_db.similarity_search("智驾科技员工规模", k=1)
if verify_results:
    print(f"  {verify_results[0].page_content[:60]}...")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：元数据过滤检索
# 目标：掌握 filter 参数，实现精确的条件检索
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 4 章：元数据过滤检索")
print("=" * 60)
print()

print("【为什么需要元数据过滤？】")
print("""
  纯向量检索的问题：
    query "公司产品" → 可能返回任意类别的文档

  元数据过滤的价值：
    query "公司产品" + filter category="产品介绍"
    → 只在"产品介绍"类文档中做向量检索，精准！

  类比 SQL：
    SELECT * FROM docs WHERE category='产品介绍'
    ORDER BY cosine_similarity(vector, query_vector) DESC
    LIMIT 2;
""")

# ─── 过滤示例 1：按 category 过滤 ─────────────────────────

print('【过滤示例 1：只检索"产品介绍"类别】')

# filter 参数使用 Chroma 的 Where 语法
# $eq = equals（等于）
product_results = chroma_db.similarity_search(
    query="技术参数",
    k=2,
    filter={"category": {"$eq": "产品介绍"}},
)
print(f"查询：'技术参数'（仅限产品介绍类别）")
print(f"结果数量：{len(product_results)}")
for i, doc in enumerate(product_results, 1):
    print(f"  {i}. [{doc.metadata.get('id')}] [{doc.metadata.get('category')}] "
          f"{doc.page_content[:45]}...")
print()

# ─── 过滤示例 2：按 year 范围过滤 ─────────────────────────

print("【过滤示例 2：只检索 2023 年及以后的文档】")

# $gte = greater than or equal（大于等于）
# Chroma 支持的比较操作符：$eq, $ne, $gt, $gte, $lt, $lte
# 逻辑操作符：$and, $or
recent_results = chroma_db.similarity_search(
    query="最新进展",
    k=3,
    filter={"year": {"$gte": 2023}},
)
print(f"查询：'最新进展'（仅限 2023 年及以后）")
print(f"结果数量：{len(recent_results)}")
for i, doc in enumerate(recent_results, 1):
    print(f"  {i}. [{doc.metadata.get('id')}] [year={doc.metadata.get('year')}] "
          f"{doc.page_content[:45]}...")
print()

# ─── 过滤示例 3：组合过滤（AND 条件）─────────────────────

print("【过滤示例 3：组合过滤（产品介绍 AND 2023年以后）】")

# $and 操作符：所有条件都必须满足
combined_results = chroma_db.similarity_search(
    query="芯片算力",
    k=2,
    filter={
        "$and": [
            {"category": {"$eq": "产品介绍"}},
            {"year": {"$gte": 2023}},
        ]
    },
)
print(f"查询：'芯片算力'（产品介绍 AND 2023年以后）")
print(f"结果数量：{len(combined_results)}")
for i, doc in enumerate(combined_results, 1):
    print(f"  {i}. [{doc.metadata.get('id')}] {doc.page_content[:45]}...")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 章：重启验证——持久化的终极考验
# 目标：模拟"程序重启"，验证数据真的存在磁盘上
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 5 章：重启验证")
print("=" * 60)
print()

print("【模拟程序重启：销毁内存中的对象】")
print("执行 del chroma_db 和 del faiss_db，内存数据全部释放...")
print()

# 删除内存中的对象，模拟程序退出后重启的场景
del chroma_db
del faiss_db

print("内存对象已销毁！")
print()

# ─── 重新从磁盘加载 ────────────────────────────────────

print("【从磁盘重新加载向量库】")
print()

# 重新加载 FAISS
print("重新加载 FAISS...")
faiss_reloaded = FAISS.load_local(
    folder_path=FAISS_PATH,
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)
print("FAISS 重新加载成功！")

# 验证 FAISS 数据完整性
faiss_verify = faiss_reloaded.similarity_search("融资", k=1)
print(f"  检索验证：查询'融资' → [{faiss_verify[0].metadata.get('id')}] "
      f"{faiss_verify[0].page_content[:40]}...")
print()

# 重新加载 Chroma
print("重新加载 Chroma...")
chroma_reloaded = Chroma(
    collection_name="knowledge_base",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)
print("Chroma 重新加载成功！")

# 验证 Chroma 数据完整性
count_reloaded = chroma_reloaded._collection.count()
print(f"  文档总数验证：{count_reloaded} 条")

# 验证第 3 章的操作是否持久化：
# - doc_004 应该已被删除（不应出现在结果中）
# - doc_001 应该是更新后的版本（包含"800人"）
chroma_verify = chroma_reloaded.similarity_search("员工规模", k=1)
print(f"  更新验证（doc_001）：{chroma_verify[0].page_content[:50]}...")

# 验证 doc_004 确实被删除了
doc_004_check = chroma_reloaded.similarity_search("百度华为自动驾驶团队", k=1)
print(f"  删除验证（查询团队信息，doc_004 应已删除）：")
print(f"    最相似文档：[{doc_004_check[0].metadata.get('id')}] "
      f"{doc_004_check[0].page_content[:40]}...")
print()

print("持久化验证完成！所有操作（增删改）均已正确写入磁盘。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 总结
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("总结：向量数据库持久化选型指南")
print("=" * 60)
print("""
  ┌─────────────────┬─────────────────────┬─────────────────────┐
  │      特性        │        FAISS        │       Chroma        │
  ├─────────────────┼─────────────────────┼─────────────────────┤
  │  持久化方式      │  手动 save_local()  │  自动（写入即落盘）  │
  │  按 ID 删除      │  不支持             │  支持 delete(ids=[])│
  │  元数据过滤      │  不支持             │  支持 filter 参数   │
  │  检索速度        │  极快（亿级优化）   │  快（千万级）        │
  │  部署复杂度      │  零依赖             │  需要 chromadb 库   │
  │  适合场景        │  离线、只读、大规模  │  开发、需增删改      │
  └─────────────────┴─────────────────────┴─────────────────────┘

  推荐决策树：
    需要删除/更新单条记录？    → 选 Chroma
    需要元数据过滤？           → 选 Chroma
    数据只读、追求极致性能？   → 选 FAISS
    生产环境、需要高可用？     → 选 Pinecone / Weaviate（云服务）
""")

# ─── 清理演示产生的临时文件 ──────────────────────────────

print("清理演示产生的临时文件...")

# ⚠️ 关键修复说明：
# 错误写法（会删除脚本自身所在的目录！）：
#   shutil.rmtree("./vectordb")  ← 危险！persistent_store.py 就在 ./vectordb/ 下
#
# 正确写法（只清理各自的子目录）：
shutil.rmtree("./vectordb/faiss_store", ignore_errors=True)
shutil.rmtree("./vectordb/chroma_store", ignore_errors=True)

print("  已清理：./vectordb/faiss_store")
print("  已清理：./vectordb/chroma_store")
print()
print("演示完成！")
