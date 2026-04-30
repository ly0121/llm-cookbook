"""
╔══════════════════════════════════════════════════════════════════╗
║         项目二：本地文档 RAG 知识库（外挂大脑）                       ║
║         文档切块 + 向量化 + FAISS 检索 + LLM 回答                   ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普：开卷考试——RAG 的完整工作流是什么？】
═══════════════════════════════════════════════════════════════════

想象一场特殊的考试：

  普通 LLM 回答 = 闭卷考试
    考生（LLM）只能依靠"记在脑子里的知识"（训练数据）作答。
    缺点：
      ① 知识有截止日期（训练数据不包含最新信息）
      ② 不了解你的私有文档（公司内部文件、个人笔记）
      ③ 容易"记忆出错"——幻觉（Hallucination）

  RAG 回答 = 开卷考试
    考生（LLM）可以打开参考书（你的文档），
    先翻到相关章节，再基于原文作答。
    优点：答案有据可查，不易幻觉，知识可随时更新！

RAG 的四步工作流：

  第一步：切块（Chunking）
    把长文档切成一段段小块（chunk），因为 LLM 上下文窗口有限。
    ┌──────────────────────────────────────────────┐
    │  一篇 3000 字文章  →  切成若干块，每块约 500 字  │
    └──────────────────────────────────────────────┘

  第二步：向量化（Embedding）
    把每个文本块变成一个"数字向量"（一串浮点数）。
    含义相似的文本，在向量空间里距离会更近。
    ┌──────────────────────────────────────────────┐
    │  "图灵测试是什么？" → [0.23, -0.11, 0.87, ...]  │
    │  "什么是图灵机？"   → [0.21, -0.09, 0.84, ...]  │
    │  （两者相似，向量接近）                          │
    └──────────────────────────────────────────────┘

  第三步：检索（Retrieval）
    用户提问时，先把问题也向量化，
    然后在向量数据库里找出"最相似"的几个文本块。
    ┌──────────────────────────────────────────────┐
    │  问：谁发明了深度学习？                         │
    │  检索 → 找到包含"深度学习""Geoffrey Hinton"的块  │
    └──────────────────────────────────────────────┘

  第四步：生成（Generation）
    把检索到的文本块 + 用户问题一起发给 LLM：
    "请根据以下参考资料回答用户的问题：[文本块] 问题：[用户提问]"
    LLM 基于这些原文来回答，减少幻觉。

  整体流程：
    文档 → 切块 → 向量化 → 存入向量数据库
    用户提问 → 向量化 → 相似度检索 → 召回相关块 → 喂给 LLM → 得到回答
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# LangChain OpenAI 集成：同时提供聊天模型和向量化模型
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 输出解析器：把 AIMessage 对象转成纯字符串
from langchain_core.output_parsers import StrOutputParser

# 提示词模板
from langchain_core.prompts import ChatPromptTemplate

# LCEL 并行运行和直传组件
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Document：LangChain 统一的文档对象（包含 page_content + metadata）
from langchain_core.documents import Document

# 文本切块器：递归按语义边界切割
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS 向量数据库（内存版，轻量高速）
# ⚠️ 避坑指南：需要安装 faiss-cpu 包，不要装 faiss-gpu（会冲突）
#   pip install faiss-cpu
from langchain_community.vectorstores import FAISS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化 LLM 和 Embeddings 客户端
# 目标：同时初始化两个客户端——一个用来聊天，一个用来向量化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 0 章：初始化 LLM 和 Embeddings 客户端")
print("=" * 60)

# ── API 配置（和 langchain/chatbot.py 保持一致）──────────────
API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
CHAT_MODEL = "kivy-kimi-k2_5"

# ⚠️ 避坑指南：Embedding 模型名
#
# 不同 Gateway 支持的 Embedding 模型名不同，常见的有：
#   "text-embedding-3-small"  → OpenAI 新版，性能更好（推荐）
#   "text-embedding-ada-002"  → OpenAI 旧版，更多 Gateway 支持
#
# 如果运行时报 404 或 422 错误，说明这个模型名不对，换另一个试试！
EMBEDDING_MODEL = "text-embedding-3-small"

# 初始化聊天 LLM（和项目一完全一样）
# temperature=0.0：RAG 场景用 0，要精确回答，不要创意发散
llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.0,
)

# 初始化 Embeddings 客户端
# OpenAIEmbeddings 和 ChatOpenAI 一样，支持 base_url 指向任意兼容接口
# 它的作用：把文本字符串 → 数字向量（一个 float 列表）
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
)

print("✅ LLM 初始化完成")
print(f"   聊天模型: {CHAT_MODEL}")
print("✅ Embeddings 初始化完成")
print(f"   向量化模型: {EMBEDDING_MODEL}")
print()
