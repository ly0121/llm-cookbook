"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 项目二十: Document ETL Pipeline (真实文档处理)                ║
║                                                                              ║
║  ETL = Extract + Transform + Load                                            ║
║  这是数据工程里最核心的概念之一，也是 RAG 系统的"地基工程"。                  ║
║                                                                              ║
║  类比: 盖房子之前要打地基                                                     ║
║  ┌─────────────────────────────────────────────────────────────────────┐     ║
║  │  原始文档(乱糟糟)  →  清洗整理(干干净净)  →  入库备查(随时能找到)   │     ║
║  │       PDF/HTML/DB         去噪/分块/结构化       向量化/索引/元数据   │     ║
║  └─────────────────────────────────────────────────────────────────────┘     ║
║                                                                              ║
║  为什么需要 ETL?                                                             ║
║  ─────────────────                                                           ║
║  想象你有一个图书馆，书是乱丢的:                                              ║
║  - 有些书缺了封面 (元数据丢失)                                               ║
║  - 有些书页面粘在一起 (格式混乱)                                             ║
║  - 有些书重复了三本 (数据冗余)                                               ║
║  - 有些书太厚找不到关键段落 (未分块)                                          ║
║                                                                              ║
║  ETL 就是把这个乱图书馆变成一个:                                              ║
║  - 每本书都有完整目录索引 ✓                                                  ║
║  - 每个章节独立可检索 ✓                                                      ║
║  - 没有重复冗余 ✓                                                            ║
║  - 能通过"意思相近"来查找 ✓ (向量检索)                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 导入区
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import re
import time
import os
from datetime import datetime

# LangChain 核心组件
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API 配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "kivy-kimi-k2_5"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Chapter 0: ETL 概念科普 + 完整 Pipeline 总览                               ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║                                                                              ║
# ║  ETL Pipeline 的完整数据流:                                                  ║
# ║                                                                              ║
# ║  ┌──────────┐    ┌──────────────────┐    ┌─────────────────┐                ║
# ║  │ Extract  │    │    Transform     │    │      Load       │                ║
# ║  │──────────│    │──────────────────│    │─────────────────│                ║
# ║  │ PDF      │    │ 1. 清洗去噪      │    │ 1. Embedding    │                ║
# ║  │ HTML     │───▶│ 2. 格式统一      │───▶│ 2. FAISS索引    │                ║
# ║  │ Database │    │ 3. 智能分块      │    │ 3. 元数据关联    │                ║
# ║  │ Markdown │    │ 4. 元数据补充    │    │ 4. 去重入库      │                ║
# ║  └──────────┘    └──────────────────┘    └─────────────────┘                ║
# ║                                                                              ║
# ║  三个阶段的核心任务:                                                         ║
# ║  ─────────────────                                                           ║
# ║  E (Extract 抽取):                                                           ║
# ║    - 从各种数据源读取原始内容                                                ║
# ║    - 保留原始格式和元数据                                                    ║
# ║    - 处理编码、格式兼容性问题                                                ║
# ║                                                                              ║
# ║  T (Transform 转换):                                                         ║
# ║    - 去除噪音 (HTML标签、页眉页脚、乱码)                                    ║
# ║    - 统一格式 (空白字符、换行符)                                             ║
# ║    - 智能分块 (让每个chunk语义完整)                                          ║
# ║    - 补充元数据 (来源、时间、类型)                                           ║
# ║                                                                              ║
# ║  L (Load 加载):                                                              ║
# ║    - 向量化: 把文本变成高维数字向量                                          ║
# ║    - 入库: 存入向量数据库 (FAISS)                                            ║
# ║    - 索引: 建立高效的相似度检索索引                                          ║
# ║    - 去重: 避免重复内容浪费存储和干扰检索                                    ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("=" * 70)
print("Chapter 0: ETL Pipeline 概念科普")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    Document ETL Pipeline 全景图                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   [原始数据源]          [转换处理]            [向量数据库]            │
│                                                                       │
│   ┌─────┐             ┌─────────┐           ┌──────────┐            │
│   │ PDF │─┐           │ 清洗去噪 │           │          │            │
│   └─────┘  │          └────┬────┘           │  FAISS   │            │
│   ┌─────┐  │   Extract     │    Transform   │  向量库   │            │
│   │HTML │──┼──────────▶ 格式统一 ──────────▶│          │            │
│   └─────┘  │               │                │ [索引+   │            │
│   ┌─────┐  │          ┌────┴────┐           │  元数据] │            │
│   │ DB  │──┘          │ 智能分块 │           │          │            │
│   └─────┘             └─────────┘           └──────────┘            │
│                                                                       │
│   关键指标: 文档数 → chunk数 → 向量数 → 检索质量                     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

为什么 Transform 是最关键的一步?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

想象你在搜索"电动汽车的续航里程":
  - 如果chunk太大(整篇文章): 检索到了，但有用信息被噪音淹没
  - 如果chunk太小(单个句子): 上下文丢失，回答不完整
  - 如果有HTML标签残留: 向量化时噪音干扰语义
  - 如果有重复chunk: 浪费存储，检索结果冗余

所以，Transform 决定了最终 RAG 的质量上限!
""")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Chapter 1: Extract — 多种文档加载方式演示                                  ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║                                                                              ║
# ║  在真实场景中，文档来源五花八门:                                             ║
# ║  - 网页爬虫抓的 HTML (带各种标签噪音)                                       ║
# ║  - PDF 解析出来的文本 (页眉页脚、换行碎片)                                  ║
# ║  - 数据库导出的 JSON/CSV (结构化但需要文本化)                                ║
# ║  - Markdown 笔记 (带格式标记)                                               ║
# ║                                                                              ║
# ║  这里我们用 Document 类手动构造"脏数据"来模拟真实场景，                      ║
# ║  这样脚本完全自包含，不依赖任何外部文件!                                     ║
# ║                                                                              ║
# ║  Document 的结构:                                                            ║
# ║  ┌────────────────────────────────────────┐                                  ║
# ║  │ Document                                │                                  ║
# ║  │ ├── page_content: str  (正文内容)       │                                  ║
# ║  │ └── metadata: dict     (元数据)         │                                  ║
# ║  │     ├── source: str                     │                                  ║
# ║  │     ├── page: int                       │                                  ║
# ║  │     └── ... (自定义字段)                │                                  ║
# ║  └────────────────────────────────────────┘                                  ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n")
print("=" * 70)
print('Chapter 1: Extract — 从各种"数据源"抽取原始文档')
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 1.1 模拟"从网页抓取"的文档 (含 HTML 标签噪音)
# ─────────────────────────────────────────────────────────────────────────────
# 真实场景: 用 BeautifulSoup/Scrapy 抓取网页后，常常残留大量 HTML 标签
# 这些标签对于语义理解是"噪音"，但又混杂在有用文本中

web_docs = [
    Document(
        page_content='<div class="article-body"><h1>新能源汽车发展报告</h1>'
        "<p>2024年，中国新能源汽车销量突破<strong>900万辆</strong>，"
        '同比增长<span style="color:red">35%</span>。</p>'
        "<p>其中，纯电动车占比约65%，插电混动占比约35%。</p>"
        '<div class="ad-banner">广告位招租</div>'
        "<p>电池技术方面，磷酸铁锂电池因成本优势继续主导市场，"
        "但固态电池研发取得重大突破。</p></div>",
        metadata={"source": "https://auto.news.cn/ev-report-2024", "doc_type": "web"},
    ),
    Document(
        page_content="<html><body><nav>首页 | 新闻 | 科技</nav>"
        "<article><h2>智能驾驶技术路线对比</h2>"
        "<p>目前主流的智能驾驶技术路线分为两派：</p>"
        "<ul><li>纯视觉方案：以特斯拉为代表，依靠摄像头+AI算法</li>"
        "<li>多传感器融合：以Waymo为代表，激光雷达+摄像头+毫米波雷达</li></ul>"
        "<p>两种方案各有优劣，纯视觉成本低但安全冗余不足，"
        "融合方案安全性高但成本居高不下。</p>"
        "<footer>版权所有 &copy; 2024</footer></article></body></html>",
        metadata={
            "source": "https://tech.blog.com/autonomous-driving",
            "doc_type": "web",
        },
    ),
    Document(
        page_content="<div><p>动力电池回收是新能源汽车产业链的<b>最后一环</b>。"
        "</p><br/><br/><p>截至2024年底，中国累计退役动力电池超过<em>30万吨</em>，"
        "预计到2030年将达到350万吨。</p>"
        '<script>console.log("tracking")</script>'
        "<p>回收利用主要有两条路径：梯次利用（用于储能电站）和拆解回收（提取锂、钴等金属）。</p></div>",
        metadata={
            "source": "https://green.energy.cn/battery-recycle",
            "doc_type": "web",
        },
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 1.2 模拟"从 PDF 提取"的文档 (含页眉页脚、换行符碎片)
# ─────────────────────────────────────────────────────────────────────────────
# 真实场景: PDF 解析工具 (PyPDF2/pdfplumber) 提取出来的文本通常:
# - 每行末尾有硬换行 (原本是排版断行，不是语义断句)
# - 页眉页脚反复出现
# - 表格数据格式错乱

pdf_docs = [
    Document(
        page_content="第3页 / 共15页\n"
        "━━━━━━━━━━━━━━━━━━━━━━━\n"
        "中国智能网联汽车技术白皮书\n"
        "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "第二章 车路协同技术\n\n"
        "车路协同(V2X)是指车辆与周围环\n"
        "境(包括其他车辆、道路基础设施、\n"
        "行人等)之间的信息交互技术。该技\n"
        "术能够显著提升道路安全性和交通\n"
        "效率。\n\n"
        "V2X包含四个子类:\n"
        "- V2V (车与车通信)\n"
        "- V2I (车与基础设施通信)\n"
        "- V2P (车与行人通信)\n"
        "- V2N (车与网络通信)\n\n"
        "━━━ 中国汽车工程学会 ━━━",
        metadata={"source": "v2x_whitepaper.pdf", "page": 3, "doc_type": "pdf"},
    ),
    Document(
        page_content="第7页 / 共15页\n"
        "━━━━━━━━━━━━━━━━━━━━━━━\n"
        "中国智能网联汽车技术白皮书\n"
        "━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "第五章 高精地图\n\n"
        "高精地图是L3+自动驾驶的核心依\n"
        "赖之一。与普通导航地图不同，高\n"
        "精地图的精度达到厘米级，包含车\n"
        "道级拓扑信息、道路曲率、坡度、\n"
        "交通标志标线等丰富信息。\n\n"
        "国内高精地图主要厂商:\n"
        "1. 四维图新\n"
        "2. 百度地图\n"
        "3. 高德地图\n\n"
        '但随着"轻地图"方案兴起(如华为\n'
        "ADS 2.0)，行业对高精地图的依赖\n"
        "正在减弱。\n\n"
        "━━━ 中国汽车工程学会 ━━━",
        metadata={"source": "v2x_whitepaper.pdf", "page": 7, "doc_type": "pdf"},
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 1.3 模拟"从数据库导出"的结构化文档 (含JSON字段)
# ─────────────────────────────────────────────────────────────────────────────
# 真实场景: 从 MySQL/MongoDB 导出的数据，通常是结构化的 JSON
# 需要将结构化字段"文本化"才能送入 RAG 系统

db_docs = [
    Document(
        page_content='{"brand": "理想汽车", "model": "L9 Max", "year": 2024, '
        '"range_km": 1315, "battery_kwh": 44.5, '
        '"drivetrain": "增程式", "price_wan": 45.98, '
        '"features": ["空气悬架", "激光雷达", "HUD", "冰箱彩电大沙发"], '
        '"review_summary": "家庭用户首选的豪华SUV，空间巨大，智能化程度高"}',
        metadata={"source": "car_database", "table": "models", "doc_type": "database"},
    ),
    Document(
        page_content='{"brand": "小米汽车", "model": "SU7 Max", "year": 2024, '
        '"range_km": 800, "battery_kwh": 101, '
        '"drivetrain": "纯电四驱", "price_wan": 29.99, '
        '"features": ["Nidec电机", "800V平台", "CDC减震", "小米生态"], '
        '"review_summary": "性价比极高的纯电轿跑，加速3.78秒，智能座舱体验优秀"}',
        metadata={"source": "car_database", "table": "models", "doc_type": "database"},
    ),
    Document(
        page_content='{"brand": "蔚来", "model": "ET7", "year": 2024, '
        '"range_km": 1000, "battery_kwh": 150, '
        '"drivetrain": "纯电四驱", "price_wan": 44.80, '
        '"features": ["换电", "Aquila超感系统", "Adam超算平台", "PanoCinema"], '
        '"review_summary": "换电模式解决补能焦虑，豪华行政轿车定位，NIO Pilot体验出色"}',
        metadata={"source": "car_database", "table": "models", "doc_type": "database"},
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 1.4 模拟"从 Markdown"的文档 (含标题层级)
# ─────────────────────────────────────────────────────────────────────────────
# 真实场景: 技术文档、Wiki 笔记通常是 Markdown 格式

md_docs = [
    Document(
        page_content="# 充电桩建设指南\n\n"
        "## 1. 选址要求\n\n"
        "充电桩选址需要考虑以下因素：\n"
        "- 电网容量：单个快充桩功率120kW-480kW，需确保变压器容量充足\n"
        "- 车流量：优先选择停车场、商圈、高速服务区\n"
        "- 安全距离：距离加油站不少于8米，距离建筑物不少于2米\n\n"
        "## 2. 设备规格\n\n"
        "| 类型 | 功率 | 充电时间(30%-80%) |\n"
        "|------|------|-------------------|\n"
        "| 慢充 | 7kW | 6-8小时 |\n"
        "| 快充 | 120kW | 30分钟 |\n"
        "| 超充 | 480kW | 10分钟 |\n\n"
        "## 3. 运营建议\n\n"
        "运营商应关注利用率指标，一般认为利用率超过8%即可盈亏平衡。",
        metadata={"source": "charging_guide.md", "doc_type": "markdown"},
    ),
    Document(
        page_content="# OTA升级技术解析\n\n"
        "## 什么是OTA\n\n"
        "OTA(Over-The-Air)即空中下载技术，允许车辆通过无线网络接收软件更新。\n\n"
        "## OTA的两种类型\n\n"
        "### FOTA (Firmware OTA)\n"
        "固件级别的升级，可以更新ECU固件，涉及底盘、动力等核心系统。\n"
        "风险较高，需要完整的A/B分区方案保证升级失败可回滚。\n\n"
        "### SOTA (Software OTA)\n"
        "软件级别的升级，主要更新车机系统、导航、娱乐等应用层功能。\n"
        "风险较低，类似手机App更新。\n\n"
        "## 安全挑战\n\n"
        "OTA升级的安全性至关重要，需要确保：\n"
        "1. 传输加密(TLS 1.3)\n"
        "2. 固件签名验证\n"
        "3. 安全启动链\n"
        "4. 升级失败回滚机制",
        metadata={"source": "ota_explained.md", "doc_type": "markdown"},
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# 汇总所有原始文档
# ─────────────────────────────────────────────────────────────────────────────
all_raw_docs = web_docs + pdf_docs + db_docs + md_docs

print(f"\n📦 Extract 阶段完成! 共加载 {len(all_raw_docs)} 个原始文档")
print(f"   - 网页文档: {len(web_docs)} 个")
print(f"   - PDF文档: {len(pdf_docs)} 个")
print(f"   - 数据库文档: {len(db_docs)} 个")
print(f"   - Markdown文档: {len(md_docs)} 个")

print('\n--- 原始文档预览 (展示"脏数据") ---')
for i, doc in enumerate(all_raw_docs):
    print(
        f"\n[Doc {i}] 来源: {doc.metadata.get('source', 'unknown')} | 类型: {doc.metadata.get('doc_type', 'unknown')}"
    )
    # 只显示前120字符，避免刷屏
    preview = doc.page_content[:120].replace("\n", "\\n")
    print(f"  内容预览: {preview}...")
    print(f"  原始长度: {len(doc.page_content)} 字符")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Chapter 2: Transform — 文本清洗 + 智能分块                                 ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║                                                                              ║
# ║  Transform 是 ETL 中最精细、最关键的环节。                                   ║
# ║                                                                              ║
# ║  数据清洗流水线 (Pipeline):                                                  ║
# ║  ┌────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐             ║
# ║  │ 原始   │   │ 去HTML标签  │   │ 统一空白   │   │ 去页眉     │             ║
# ║  │ 文本   │──▶│            │──▶│ 字符       │──▶│ 页脚       │──┐          ║
# ║  └────────┘   └────────────┘   └────────────┘   └────────────┘  │          ║
# ║                                                                    │          ║
# ║       ┌──────────────────────────────────────────────────────────┘          ║
# ║       │                                                                      ║
# ║       ▼                                                                      ║
# ║  ┌────────────┐   ┌────────────────┐   ┌────────────────┐                  ║
# ║  │ 合并断行   │   │ 文本分块       │   │ 添加元数据     │                  ║
# ║  │            │──▶│ (多策略可选)   │──▶│ (chunk_id等)   │                  ║
# ║  └────────────┘   └────────────────┘   └────────────────┘                  ║
# ║                                                                              ║
# ║  分块策略对比:                                                               ║
# ║  ┌─────────────────────────────────────────────────────────────────┐        ║
# ║  │ 策略           │ 原理              │ 优点      │ 缺点           │        ║
# ║  │────────────────│───────────────────│───────────│────────────────│        ║
# ║  │ 固定长度       │ 每N字符切一刀     │ 简单      │ 可能切断句子    │        ║
# ║  │ 递归分割       │ 按\n\n→\n→.→空格 │ 语义完整  │ chunk大小不均   │        ║
# ║  │ 语义分块       │ 按段落/标题分割   │ 最自然    │ 实现复杂        │        ║
# ║  └─────────────────────────────────────────────────────────────────┘        ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n\n")
print("=" * 70)
print("Chapter 2: Transform — 文本清洗 + 智能分块")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 2.1 清洗函数定义
# ─────────────────────────────────────────────────────────────────────────────


def remove_html_tags(text: str) -> str:
    """
    去除HTML标签，保留纯文本内容。

    原理:
    - 用正则匹配 <...> 模式并替换为空
    - 特别处理 <script>/<style> 标签: 连同内容一起删除
    - 处理HTML实体 (&copy; &amp; 等)

    注意: 这是简化版。生产环境建议用 BeautifulSoup 的 get_text()
    """
    # 先删除 script/style 标签及其内容
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    # 删除所有 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 处理常见 HTML 实体
    text = text.replace("&copy;", "(C)")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&nbsp;", " ")
    return text


def normalize_whitespace(text: str) -> str:
    """
    统一空白字符。

    处理:
    - 多个连续空格 → 单个空格
    - 多个连续空行 → 最多两个换行
    - 去除行首行尾多余空格
    """
    # 每行去除首尾空格
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # 多个连续空行合并为两个换行(保留段落间距)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 行内多个空格合并为一个
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def remove_headers_footers(text: str) -> str:
    """
    去除页眉页脚。

    常见模式:
    - "第X页 / 共Y页"
    - 重复出现的标题行
    - 分隔线 + 机构名
    """
    # 去除 "第X页 / 共Y页" 模式
    text = re.sub(r"第\d+页\s*/\s*共\d+页", "", text)
    # 去除 "━━━ XXX ━━━" 模式的页眉页脚
    text = re.sub(r"━+\s*.*?\s*━+", "", text)
    return text


def merge_broken_lines(text: str) -> str:
    """
    合并被错误换行拆断的句子。

    PDF提取的一大痛点: 原文排版时每行固定宽度，到行末就换行，
    但这个换行不是语义上的断句。

    判断规则:
    - 如果一行不以句号/问号/叹号/冒号结尾，且下一行不是空行、不以特殊符号开头
      → 说明是"被拆断的行"，应该拼接
    """
    lines = text.split("\n")
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 如果当前行非空，且不以常见句末标点结尾，且下一行也非空且不以特殊符号开头
        if (
            line
            and not re.search(r"[。！？：;\n]$", line)
            and not re.match(r"^[-\d#*•|>]", line)
            and i + 1 < len(lines)
            and lines[i + 1]
            and not re.match(r"^[-\d#*•|>]", lines[i + 1])
            and not lines[i + 1].startswith(" ")
        ):
            # 拼接下一行
            merged.append(line + lines[i + 1])
            i += 2
        else:
            merged.append(line)
            i += 1
    return "\n".join(merged)


def convert_db_record_to_text(text: str) -> str:
    """
    将数据库JSON记录转换为自然语言文本。

    原因: 向量化模型理解自然语言远比理解JSON好。
    "brand": "小米" → "品牌: 小米"
    """
    import json

    try:
        data = json.loads(text)
        parts = []
        field_map = {
            "brand": "品牌",
            "model": "型号",
            "year": "年份",
            "range_km": "续航(km)",
            "battery_kwh": "电池容量(kWh)",
            "drivetrain": "驱动形式",
            "price_wan": "售价(万元)",
            "features": "亮点配置",
            "review_summary": "用户评价",
        }
        for key, label in field_map.items():
            if key in data:
                val = data[key]
                if isinstance(val, list):
                    val = "、".join(str(v) for v in val)
                parts.append(f"{label}: {val}")
        return "\n".join(parts)
    except (json.JSONDecodeError, TypeError):
        return text


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 对每种类型的文档执行清洗
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- 2.1 文本清洗演示 ---")

cleaned_docs = []

for doc in all_raw_docs:
    original_text = doc.page_content
    doc_type = doc.metadata.get("doc_type", "unknown")

    # 根据文档类型选择清洗流程
    if doc_type == "web":
        # 网页: 去HTML → 统一空白
        cleaned = remove_html_tags(original_text)
        cleaned = normalize_whitespace(cleaned)
    elif doc_type == "pdf":
        # PDF: 去页眉页脚 → 合并断行 → 统一空白
        cleaned = remove_headers_footers(original_text)
        cleaned = merge_broken_lines(cleaned)
        cleaned = normalize_whitespace(cleaned)
    elif doc_type == "database":
        # 数据库: JSON → 自然语言
        cleaned = convert_db_record_to_text(original_text)
    elif doc_type == "markdown":
        # Markdown: 保留大部分格式，只统一空白
        cleaned = normalize_whitespace(original_text)
    else:
        cleaned = normalize_whitespace(original_text)

    cleaned_docs.append(Document(page_content=cleaned, metadata=doc.metadata.copy()))

# 展示清洗效果对比
print("\n┌─ 清洗前后对比 (选取3个典型文档) ─┐")

compare_indices = [0, 3, 6]  # web, pdf, database 各取一个
for idx in compare_indices:
    if idx < len(all_raw_docs):
        print(f"\n  [Doc {idx}] 类型: {all_raw_docs[idx].metadata['doc_type']}")
        print(f"  来源: {all_raw_docs[idx].metadata['source']}")
        before = all_raw_docs[idx].page_content[:100].replace("\n", "\\n")
        after = cleaned_docs[idx].page_content[:100].replace("\n", "\\n")
        print(f"  清洗前: {before}...")
        print(f"  清洗后: {after}...")
        print(
            f"  长度变化: {len(all_raw_docs[idx].page_content)} → {len(cleaned_docs[idx].page_content)} 字符"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 2.3 分块策略对比
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n--- 2.2 分块策略对比 ---")

# 用一段较长的清洗后文本来演示分块效果
demo_text = (
    cleaned_docs[0].page_content
    + "\n\n"
    + cleaned_docs[1].page_content
    + "\n\n"
    + cleaned_docs[2].page_content
)
print(f"\n演示文本总长度: {len(demo_text)} 字符")

# 策略1: 固定长度分块
print("\n┌─ 策略1: CharacterTextSplitter (固定长度) ─┐")
fixed_splitter = CharacterTextSplitter(
    separator="\n",  # 按换行符分割
    chunk_size=150,  # 每块最大150字符
    chunk_overlap=20,  # 重叠20字符(保证上下文连贯)
)
fixed_chunks = fixed_splitter.split_text(demo_text)
print(f"  分块数: {len(fixed_chunks)}")
for i, chunk in enumerate(fixed_chunks[:3]):
    print(f"  Chunk {i} ({len(chunk)}字符): {chunk[:60].replace(chr(10), ' ')}...")
if len(fixed_chunks) > 3:
    print(f"  ... 还有 {len(fixed_chunks) - 3} 个块")

# 策略2: 递归分块
print("\n┌─ 策略2: RecursiveCharacterTextSplitter (递归分割) ─┐")
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        "。",
        "，",
        " ",
    ],  # 分割优先级: 段落 > 换行 > 句号 > 逗号 > 空格
    chunk_size=150,
    chunk_overlap=20,
)
recursive_chunks = recursive_splitter.split_text(demo_text)
print(f"  分块数: {len(recursive_chunks)}")
for i, chunk in enumerate(recursive_chunks[:3]):
    print(f"  Chunk {i} ({len(chunk)}字符): {chunk[:60].replace(chr(10), ' ')}...")
if len(recursive_chunks) > 3:
    print(f"  ... 还有 {len(recursive_chunks) - 3} 个块")

# 策略3: 按语义分块 (自定义: 按段落/标题分割)
print("\n┌─ 策略3: 自定义语义分块 (按段落/标题) ─┐")


def semantic_split(text: str, max_chunk_size: int = 300) -> list:
    """
    按语义边界分块:
    1. 先按 "段落" (双换行) 分割
    2. 如果单个段落太长，再按句号分割
    3. 合并太短的段落到相邻块中
    """
    paragraphs = re.split(r"\n\n+", text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 如果当前chunk加上新段落不超过限制，就合并
        if len(current_chunk) + len(para) + 2 <= max_chunk_size:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        else:
            # 保存当前chunk
            if current_chunk:
                chunks.append(current_chunk)
            # 如果段落本身太长，按句号分割
            if len(para) > max_chunk_size:
                sentences = re.split(r"(?<=[。！？])", para)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) <= max_chunk_size:
                        current_chunk += sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


semantic_chunks = semantic_split(demo_text, max_chunk_size=200)
print(f"  分块数: {len(semantic_chunks)}")
for i, chunk in enumerate(semantic_chunks[:3]):
    print(f"  Chunk {i} ({len(chunk)}字符): {chunk[:60].replace(chr(10), ' ')}...")
if len(semantic_chunks) > 3:
    print(f"  ... 还有 {len(semantic_chunks) - 3} 个块")

# 对比总结
print("\n┌─ 三种策略对比总结 ─┐")
print(
    f"  固定长度: {len(fixed_chunks)} 块, 平均 {sum(len(c) for c in fixed_chunks) // max(len(fixed_chunks), 1)} 字符/块"
)
print(
    f"  递归分割: {len(recursive_chunks)} 块, 平均 {sum(len(c) for c in recursive_chunks) // max(len(recursive_chunks), 1)} 字符/块"
)
print(
    f"  语义分块: {len(semantic_chunks)} 块, 平均 {sum(len(c) for c in semantic_chunks) // max(len(semantic_chunks), 1)} 字符/块"
)
print('  结论: 递归分割在"语义完整性"和"实现简单性"之间取得最佳平衡')

# ─────────────────────────────────────────────────────────────────────────────
# 2.4 对所有清洗后的文档进行正式分块
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n--- 2.3 正式分块: 使用 RecursiveCharacterTextSplitter ---")

# 生产推荐参数:
# - chunk_size: 200-500 (中文建议偏小，因为信息密度高)
# - chunk_overlap: chunk_size 的 10%-20%
final_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "；", "，", " "],
    chunk_size=200,
    chunk_overlap=30,
)

all_chunks = []
for doc in cleaned_docs:
    chunks = final_splitter.split_documents([doc])
    all_chunks.extend(chunks)

# 为每个chunk添加丰富的元数据
for i, chunk in enumerate(all_chunks):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["word_count"] = len(chunk.page_content)
    chunk.metadata["created_at"] = datetime.now().isoformat()

print(f"\n  输入: {len(cleaned_docs)} 个文档")
print(f"  输出: {len(all_chunks)} 个 chunks")
print(
    f"  平均chunk大小: {sum(len(c.page_content) for c in all_chunks) // max(len(all_chunks), 1)} 字符"
)
print(f"  最小chunk: {min(len(c.page_content) for c in all_chunks)} 字符")
print(f"  最大chunk: {max(len(c.page_content) for c in all_chunks)} 字符")

print("\n  前5个chunk预览:")
for i in range(min(5, len(all_chunks))):
    chunk = all_chunks[i]
    print(
        f"    Chunk {i}: [{chunk.metadata['doc_type']}] {chunk.page_content[:50].replace(chr(10), ' ')}..."
    )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Chapter 3: Load — 向量化 + 入库 + 元数据管理                               ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║                                                                              ║
# ║  向量化 (Embedding) 是什么?                                                  ║
# ║  ─────────────────────────                                                   ║
# ║  把一段文本变成一个固定长度的数字数组(向量)，使得:                            ║
# ║  - 语义相近的文本 → 向量距离近                                              ║
# ║  - 语义无关的文本 → 向量距离远                                              ║
# ║                                                                              ║
# ║  例如:                                                                       ║
# ║  "新能源汽车" → [0.23, -0.15, 0.87, ..., 0.42]  (512维)                    ║
# ║  "电动车"    → [0.21, -0.14, 0.85, ..., 0.40]  (距离很近!)                 ║
# ║  "今天天气好" → [0.98, 0.54, -0.32, ..., -0.11] (距离很远)                 ║
# ║                                                                              ║
# ║  FAISS (Facebook AI Similarity Search):                                      ║
# ║  ────────────────────────────────────                                        ║
# ║  - Meta 开源的高效向量相似度搜索库                                           ║
# ║  - 支持百万级向量的毫秒级检索                                               ║
# ║  - 内存索引，速度极快                                                       ║
# ║  - 支持多种索引类型 (Flat精确搜索, IVF近似搜索, HNSW图搜索)                 ║
# ║                                                                              ║
# ║  去重策略:                                                                   ║
# ║  ─────────                                                                   ║
# ║  ┌──────────┐   计算余弦相似度    ┌──────────┐                              ║
# ║  │ 新chunk  │──────────────────▶ │ 已有chunk │                              ║
# ║  └──────────┘                     └──────────┘                              ║
# ║       │                                 │                                    ║
# ║       └──── 相似度 > 0.95 ────────────▶ 跳过(重复)                          ║
# ║       └──── 相似度 < 0.95 ────────────▶ 入库(新内容)                        ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n\n")
print("=" * 70)
print("Chapter 3: Load — 向量化 + 入库 + 元数据管理")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# 3.1 初始化 Embedding 模型
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 3.1 初始化 Embedding 模型 ---")
print("  使用模型: BAAI/bge-small-zh-v1.5")
print("  特点: 中文优化、体积小(~90MB)、效果好")
print("  向量维度: 512")

# bge-small-zh-v1.5 是北京智源研究院开源的中文Embedding模型
# 参数量小但中文效果优秀，非常适合本地学习使用
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},  # CPU推理(学习用，不需要GPU)
    encode_kwargs={"normalize_embeddings": True},  # 归一化，使余弦相似度计算更准确
)

# 测试 embedding
test_text = "新能源汽车的续航里程"
test_vector = embedding_model.embed_query(test_text)
print(f'\n  测试文本: "{test_text}"')
print(f"  向量维度: {len(test_vector)}")
print(f"  向量前5个值: {[round(v, 4) for v in test_vector[:5]]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.2 构建 FAISS 向量库
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 3.2 构建 FAISS 向量库 ---")

start_time = time.time()

# FAISS.from_documents 一步完成: 向量化 + 建索引
vectorstore = FAISS.from_documents(
    documents=all_chunks,
    embedding=embedding_model,
)

build_time = time.time() - start_time
print(f"  向量库构建完成!")
print(f"  文档chunk数: {len(all_chunks)}")
print(f"  构建耗时: {build_time:.2f} 秒")
print(f"  索引类型: FlatL2 (精确搜索，适合小规模数据)")

# 验证: 做一次相似度检索
print("\n  验证检索功能:")
query = "电动汽车的续航能力"
results = vectorstore.similarity_search_with_score(query, k=3)
print(f'  查询: "{query}"')
for i, (doc, score) in enumerate(results):
    print(
        f"  Top{i + 1} (距离={score:.4f}): {doc.page_content[:50].replace(chr(10), ' ')}..."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 3.3 元数据管理
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 3.3 元数据管理 ---")
print("  每个chunk都携带以下元数据:")
sample = all_chunks[0].metadata
for key, val in sample.items():
    print(f"    {key}: {val}")

# 展示元数据过滤检索
print("\n  元数据过滤检索示例:")
print("  (只在 web 来源的文档中搜索)")

# FAISS 不原生支持元数据过滤，我们手动实现
web_chunks = [c for c in all_chunks if c.metadata.get("doc_type") == "web"]
print(f"  web类型chunk数: {len(web_chunks)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3.4 增量更新演示
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 3.4 增量更新: 新文档加入已有索引 ---")

# 模拟一批新文档到来
new_docs = [
    Document(
        page_content="2025年，固态电池量产取得突破性进展。宁德时代宣布其第一代固态电池能量密度达到500Wh/kg，"
        "是目前三元锂电池的两倍以上。固态电池的优势包括：安全性高(不易燃烧)、能量密度大、"
        "低温性能好、循环寿命长。预计2026年将开始大规模装车。",
        metadata={
            "source": "https://battery.news/solid-state-2025",
            "doc_type": "web",
            "is_new": True,
        },
    ),
]

# 清洗新文档
new_cleaned = []
for doc in new_docs:
    cleaned = remove_html_tags(doc.page_content)
    cleaned = normalize_whitespace(cleaned)
    new_cleaned.append(Document(page_content=cleaned, metadata=doc.metadata.copy()))

# 分块
new_chunks = final_splitter.split_documents(new_cleaned)
for i, chunk in enumerate(new_chunks):
    chunk.metadata["chunk_id"] = len(all_chunks) + i
    chunk.metadata["word_count"] = len(chunk.page_content)
    chunk.metadata["created_at"] = datetime.now().isoformat()

print(f"  新文档数: {len(new_docs)}")
print(f"  新chunk数: {len(new_chunks)}")

# 增量添加到已有向量库
vectorstore.add_documents(new_chunks)
print(f"  增量添加完成! 向量库现在共有 {len(all_chunks) + len(new_chunks)} 个向量")

# ─────────────────────────────────────────────────────────────────────────────
# 3.5 去重: 检测重复/高度相似的chunk
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 3.5 去重: 检测高度相似的chunk ---")


def deduplicate_chunks(chunks: list, embedding_model, threshold: float = 0.95) -> list:
    """
    基于向量相似度的去重。

    原理:
    1. 对每个chunk计算embedding
    2. 新chunk与已保留的chunk计算余弦相似度
    3. 如果最高相似度 > threshold，认为是重复，跳过
    4. 否则保留

    注意: 这是 O(n^2) 的朴素实现。
    生产环境可用 LSH (局部敏感哈希) 加速。
    """
    import numpy as np

    if not chunks:
        return []

    # 批量计算所有chunk的embedding
    texts = [c.page_content for c in chunks]
    embeddings = embedding_model.embed_documents(texts)
    embeddings = np.array(embeddings)

    kept_indices = [0]  # 第一个总是保留

    for i in range(1, len(embeddings)):
        # 计算与所有已保留chunk的余弦相似度
        kept_embs = embeddings[kept_indices]
        # 由于embedding已经归一化，余弦相似度 = 点积
        similarities = np.dot(kept_embs, embeddings[i])
        max_sim = similarities.max()

        if max_sim < threshold:
            kept_indices.append(i)

    return kept_indices


# 构造测试场景: 故意加入一些重复内容
test_chunks_for_dedup = all_chunks[:5] + [
    # 这是 all_chunks[0] 的轻微改写版(应该被判为重复)
    Document(
        page_content=all_chunks[0].page_content.replace("。", "。 "),
        metadata={"source": "duplicate_test", "doc_type": "test"},
    ),
]

print(f"  去重前chunk数: {len(test_chunks_for_dedup)}")
kept_indices = deduplicate_chunks(
    test_chunks_for_dedup, embedding_model, threshold=0.95
)
print(f"  去重后chunk数: {len(kept_indices)}")
print(f"  去除了 {len(test_chunks_for_dedup) - len(kept_indices)} 个重复chunk")

for idx in kept_indices:
    chunk = test_chunks_for_dedup[idx]
    print(f"    保留 Chunk {idx}: {chunk.page_content[:40].replace(chr(10), ' ')}...")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Chapter 4: 完整 Pipeline 串联 + 端到端演示                                 ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║                                                                              ║
# ║  把前面三章的所有步骤封装成一个 Pipeline 类:                                 ║
# ║  ┌─────────────────────────────────────────────────────────────────┐        ║
# ║  │  DocumentETLPipeline                                             │        ║
# ║  │  ├── extract(raw_docs)        → List[Document]                   │        ║
# ║  │  ├── transform(docs)          → List[Document] (chunks)          │        ║
# ║  │  ├── load(chunks)             → FAISS vectorstore                │        ║
# ║  │  └── run(raw_docs)            → FAISS vectorstore (一键执行)     │        ║
# ║  └─────────────────────────────────────────────────────────────────┘        ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n\n")
print("=" * 70)
print("Chapter 4: 完整 Pipeline 串联 + 端到端演示")
print("=" * 70)


class DocumentETLPipeline:
    """
    文档 ETL Pipeline 类。

    职责:
    - 封装 Extract → Transform → Load 的完整流程
    - 提供进度追踪
    - 记录统计信息
    """

    def __init__(self, embedding_model, chunk_size: int = 200, chunk_overlap: int = 30):
        """
        初始化 Pipeline。

        Args:
            embedding_model: 向量化模型
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
        """
        self.embedding_model = embedding_model
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "；", "，", " "],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vectorstore = None
        self.stats = {
            "total_docs": 0,
            "total_chunks": 0,
            "total_time": 0,
            "extract_time": 0,
            "transform_time": 0,
            "load_time": 0,
        }

    def extract(self, raw_docs: list) -> list:
        """
        Extract 阶段: 接收原始文档，返回原样(模拟从各种源读取)。
        在真实场景中，这里会调用各种 DocumentLoader。
        """
        print(f"\n  [Extract] 接收 {len(raw_docs)} 个原始文档")
        for i, doc in enumerate(raw_docs):
            doc_type = doc.metadata.get("doc_type", "unknown")
            print(
                f"    [{i + 1}/{len(raw_docs)}] {doc_type}: {doc.metadata.get('source', '?')} ({len(doc.page_content)}字符)"
            )
        return raw_docs

    def transform(self, docs: list) -> list:
        """
        Transform 阶段: 清洗 + 分块。
        """
        print(f"\n  [Transform] 开始处理 {len(docs)} 个文档...")

        # 清洗
        cleaned = []
        for i, doc in enumerate(docs):
            doc_type = doc.metadata.get("doc_type", "unknown")
            text = doc.page_content

            if doc_type == "web":
                text = remove_html_tags(text)
                text = normalize_whitespace(text)
            elif doc_type == "pdf":
                text = remove_headers_footers(text)
                text = merge_broken_lines(text)
                text = normalize_whitespace(text)
            elif doc_type == "database":
                text = convert_db_record_to_text(text)
            else:
                text = normalize_whitespace(text)

            cleaned.append(Document(page_content=text, metadata=doc.metadata.copy()))
            print(
                f"    [{i + 1}/{len(docs)}] 清洗完成: {len(doc.page_content)} → {len(text)} 字符"
            )

        # 分块
        all_chunks = []
        for doc in cleaned:
            chunks = self.splitter.split_documents([doc])
            all_chunks.extend(chunks)

        # 补充元数据
        for i, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["word_count"] = len(chunk.page_content)
            chunk.metadata["created_at"] = datetime.now().isoformat()

        print(f"    分块完成: {len(cleaned)} 文档 → {len(all_chunks)} chunks")
        return all_chunks

    def load(self, chunks: list):
        """
        Load 阶段: 向量化 + 入库。
        """
        print(f"\n  [Load] 开始向量化 {len(chunks)} 个chunks...")

        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
        )

        print(f"    向量库构建完成!")
        return self.vectorstore

    def run(self, raw_docs: list):
        """
        一键执行完整 ETL 流程。
        """
        print("\n" + "─" * 50)
        print("  ETL Pipeline 开始运行...")
        print("─" * 50)

        total_start = time.time()

        # Extract
        t0 = time.time()
        docs = self.extract(raw_docs)
        self.stats["extract_time"] = time.time() - t0

        # Transform
        t0 = time.time()
        chunks = self.transform(docs)
        self.stats["transform_time"] = time.time() - t0

        # Load
        t0 = time.time()
        vs = self.load(chunks)
        self.stats["load_time"] = time.time() - t0

        self.stats["total_time"] = time.time() - total_start
        self.stats["total_docs"] = len(raw_docs)
        self.stats["total_chunks"] = len(chunks)

        print("\n" + "─" * 50)
        print("  ETL Pipeline 运行完成!")
        print("─" * 50)

        return vs

    def print_stats(self):
        """打印统计信息。"""
        print("\n┌─ Pipeline 运行统计 ─┐")
        print(f"  输入文档数: {self.stats['total_docs']}")
        print(f"  输出chunk数: {self.stats['total_chunks']}")
        print(f"  向量维度: {len(self.embedding_model.embed_query('test'))}")
        print(f"  Extract 耗时: {self.stats['extract_time']:.3f}s")
        print(f"  Transform 耗时: {self.stats['transform_time']:.3f}s")
        print(f"  Load 耗时: {self.stats['load_time']:.3f}s")
        print(f"  总耗时: {self.stats['total_time']:.3f}s")
        print(
            f"  平均每文档处理时间: {self.stats['total_time'] / max(self.stats['total_docs'], 1):.3f}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4.1 运行完整 Pipeline
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- 4.1 运行完整 Pipeline ---")

pipeline = DocumentETLPipeline(
    embedding_model=embedding_model,
    chunk_size=200,
    chunk_overlap=30,
)

# 用所有原始文档跑一次完整流程
final_vectorstore = pipeline.run(all_raw_docs)

# 打印统计
pipeline.print_stats()

# ─────────────────────────────────────────────────────────────────────────────
# 4.2 端到端问答验证
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n--- 4.2 端到端问答验证: 用处理好的向量库回答问题 ---")

# 初始化 LLM
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    temperature=0,
)

# 构建简单的 RAG 问答
qa_prompt = ChatPromptTemplate.from_template(
    "根据以下参考资料回答用户问题。如果资料中没有相关信息，请诚实地说不知道。\n\n"
    "参考资料:\n{context}\n\n"
    "用户问题: {question}\n\n"
    "回答:"
)

# 测试问题列表
test_questions = [
    "智能驾驶有哪些技术路线?",
    "小米SU7的售价和续航是多少?",
    "什么是OTA升级?有哪些类型?",
]

for q in test_questions:
    print(f"\n  问题: {q}")

    # 检索相关chunk
    retrieved_docs = final_vectorstore.similarity_search(q, k=3)
    context = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    print(f"  检索到 {len(retrieved_docs)} 个相关chunk:")
    for i, doc in enumerate(retrieved_docs):
        print(
            f"    [{i + 1}] ({doc.metadata.get('doc_type', '?')}): {doc.page_content[:40].replace(chr(10), ' ')}..."
        )

    # 调用 LLM 生成回答
    chain = qa_prompt | llm
    response = chain.invoke({"context": context, "question": q})
    print(f"  回答: {response.content[:200]}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Summary: ETL 最佳实践 + 生产注意事项                                       ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║                                                                              ║
# ║  1. Extract 最佳实践:                                                        ║
# ║     - 保留原始格式用于调试                                                   ║
# ║     - 记录数据血缘(来源、时间、版本)                                        ║
# ║     - 并行加载提升吞吐量                                                    ║
# ║                                                                              ║
# ║  2. Transform 最佳实践:                                                      ║
# ║     - chunk_size 根据下游模型的 context window 调整                          ║
# ║     - overlap 保证语义连贯(通常 10%-20%)                                    ║
# ║     - 清洗规则要针对数据源定制，没有万能方案                                 ║
# ║     - 保留 metadata 用于后续过滤检索                                        ║
# ║                                                                              ║
# ║  3. Load 最佳实践:                                                           ║
# ║     - 小规模(<10万条): FAISS Flat 即可                                      ║
# ║     - 中规模(10万-1000万): FAISS IVF + PQ 量化                             ║
# ║     - 大规模(>1000万): Milvus / Qdrant / Pinecone                          ║
# ║     - 一定要做去重! 重复数据是 RAG 的大敌                                   ║
# ║                                                                              ║
# ║  4. 生产注意事项:                                                            ║
# ║     - 增量更新 > 全量重建 (避免每次重新Embedding所有文档)                    ║
# ║     - 监控 Embedding 质量退化 (模型/数据分布变化时需重建)                    ║
# ║     - 持久化: vectorstore.save_local() / load_local()                       ║
# ║     - 异步处理大批量文档 (避免阻塞主线程)                                   ║
# ║     - 定期清理过期/失效文档                                                  ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

print("\n\n")
print("=" * 70)
print("Summary: Document ETL Pipeline 学习总结")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                     ETL Pipeline 全流程回顾                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Chapter 0: 概念 — ETL 就是"脏数据清洗入库"的标准流程                │
│  Chapter 1: Extract — 从多种源加载文档(保留原始格式)                  │
│  Chapter 2: Transform — 清洗(去噪) + 分块(语义完整)                  │
│  Chapter 3: Load — 向量化 + FAISS建索引 + 去重                       │
│  Chapter 4: Pipeline — 封装成类，端到端可复用                        │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  核心收获:                                                            │
│  1. 数据质量决定 RAG 质量上限 (Garbage In, Garbage Out)              │
│  2. 分块策略要根据业务场景调优 (没有银弹)                            │
│  3. 元数据是检索过滤的利器 (doc_type, source, date...)              │
│  4. 去重避免冗余检索结果                                             │
│  5. 增量更新 > 全量重建 (生产效率)                                   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
""")

print("Done! ETL Pipeline 学习完成。")
