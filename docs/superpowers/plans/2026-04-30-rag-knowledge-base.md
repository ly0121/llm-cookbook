# 项目二：本地文档 RAG 知识库 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `rag/rag_qa.py` 中实现完整的本地 RAG 知识库教学文件——文档切块、向量化存入 FAISS、检索召回打印、LCEL RAG 问答链，让学习者亲眼看到"大模型基于哪些原文资料回答问题"。

**Architecture:** 单文件渐进式教学（5章节），内联约 3000 字人工智能发展简史作为知识库文本，使用 FAISS 内存索引 + Gateway OpenAIEmbeddings，LCEL `RunnableParallel` 构建 RAG 链。每章独立可观察，第3章专门打印检索召回结果，第4章完整问答链复用检索器。

**Tech Stack:** Python 3.9+, langchain>=0.3.0, langchain-openai>=0.2.0, langchain-community>=0.3.0, faiss-cpu>=1.7.4（新增）, langchain-text-splitters（langchain 内置）

---

## 文件清单

| 操作 | 路径 | 职责 |
|------|------|------|
| Create dir | `rag/` | RAG 系列学习目录 |
| Create | `rag/rag_qa.py` | 项目二主教学文件（6个任务逐步构建） |
| Modify | `requirements.txt` | 追加 `faiss-cpu>=1.7.4` |

---

## Task 1：创建 rag/ 目录，安装 faiss-cpu，更新 requirements.txt

**Files:**
- Create dir: `rag/`
- Modify: `requirements.txt`

- [ ] **Step 1：创建目录**

```bash
mkdir -p /Users/liuyu22/Desktop/langchain_learning/rag
```

预期：命令无输出，目录创建成功。

- [ ] **Step 2：追加 faiss-cpu 到 requirements.txt**

在 `/Users/liuyu22/Desktop/langchain_learning/requirements.txt` 末尾追加一行：

```
faiss-cpu>=1.7.4
```

完整文件内容变为：
```
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
openai>=1.0.0
faiss-cpu>=1.7.4
```

- [ ] **Step 3：安装 faiss-cpu**

```bash
cd /Users/liuyu22/Desktop/langchain_learning
source .venv/bin/activate
pip install faiss-cpu>=1.7.4
```

预期输出包含：`Successfully installed faiss-cpu-...`

- [ ] **Step 4：验证安装**

```bash
python -c "import faiss; print('faiss 版本:', faiss.__version__)"
```

预期输出：`faiss 版本: 1.x.x`（具体版本号不限）

- [ ] **Step 5：提交**

```bash
git add requirements.txt
git commit -m "feat: add faiss-cpu dependency for project two RAG

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2：创建 `rag/rag_qa.py` — 文件头科普 + 第 0 章

**Files:**
- Create: `rag/rag_qa.py`

- [ ] **Step 1：创建文件，写入文件头 docstring 和第 0 章**

创建 `/Users/liuyu22/Desktop/langchain_learning/rag/rag_qa.py`，完整内容如下：

```python
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

print(f"✅ LLM 初始化完成")
print(f"   聊天模型: {CHAT_MODEL}")
print(f"✅ Embeddings 初始化完成")
print(f"   向量化模型: {EMBEDDING_MODEL}")
print()
```

- [ ] **Step 2：验证第 0 章可以运行（不发任何 API 请求，只验证语法和初始化）**

```bash
cd /Users/liuyu22/Desktop/langchain_learning
source .venv/bin/activate
python rag/rag_qa.py
```

预期输出：
```
============================================================
第 0 章：初始化 LLM 和 Embeddings 客户端
============================================================
✅ LLM 初始化完成
   聊天模型: kivy-kimi-k2_5
✅ Embeddings 初始化完成
   向量化模型: text-embedding-3-small
```

- [ ] **Step 3：提交第 0 章**

```bash
git add rag/rag_qa.py
git commit -m "feat: add project two - rag/rag_qa.py header + chapter 0

File header with 开卷考试 RAG analogy, four-step workflow diagram.
Chapter 0: ChatOpenAI + OpenAIEmbeddings initialization.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3：追加第 1 章（文档加载 + 文本切块）

**Files:**
- Modify: `rag/rag_qa.py`（末尾追加）

- [ ] **Step 1：追加第 1 章代码**

在 `rag/rag_qa.py` 末尾追加：

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：文档加载 + 文本切块
# 目标：把长文档切成小块，准备向量化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：文档加载 + 文本切块")
print("=" * 60)
print()

# ── 准备原始文档 ────────────────────────────────────────────
#
# 真实场景：用 TextLoader、PDFLoader、WebBaseLoader 等加载外部文件
# 教学场景：直接在代码里写一段长文本，省去文件读取的麻烦
#
# 文档主题：人工智能发展简史（从图灵测试到 GPT 时代）

RAW_TEXT = """
人工智能发展简史

第一节：萌芽期（1950年代）

1950年，英国数学家艾伦·图灵（Alan Turing）在论文《计算机器与智能》中提出了著名的"图灵测试"：
如果一台机器能在对话中让人无法分辨它是人还是机器，那么这台机器就可以被认为是"有智能的"。
这一思想成为人工智能领域的奠基石。

1956年，在美国达特茅斯学院举办的一场会议上，约翰·麦卡锡（John McCarthy）正式提出了
"人工智能"（Artificial Intelligence，AI）这一术语，标志着 AI 作为一门独立学科的诞生。
参与这次会议的还有马文·明斯基（Marvin Minsky）、克劳德·香农（Claude Shannon）等先驱。

第二节：专家系统时代（1970-1980年代）

进入70年代，研究者们发现通用推理机器的研发极其困难，转而专注于特定领域的"专家系统"。
专家系统把人类专家的知识编码成规则库，例如医疗诊断系统 MYCIN 能够诊断细菌感染并推荐抗生素，
其准确率甚至超过了部分初级医生。

然而专家系统存在致命缺陷：知识需要手动编码，无法自动学习，维护成本极高。
80年代末，专家系统热潮退去，AI 进入第二次"寒冬"。

第三节：神经网络的崛起（1980-2000年代）

1986年，杰弗里·辛顿（Geoffrey Hinton）等人重新推广了"反向传播算法"（Backpropagation），
使得多层神经网络的训练成为可能。神经网络能够从大量数据中自动学习特征，不再需要手动编写规则。

1998年，杨立昆（Yann LeCun）提出了卷积神经网络（CNN），并成功应用于手写数字识别，
准确率达到99%以上。这是神经网络在实际应用中的重要里程碑。

但受限于当时的计算能力和数据量，神经网络的潜力未能完全发挥，研究再次进入低潮期。

第四节：深度学习革命（2012年至今）

2012年是人工智能历史上的转折点。在著名的 ImageNet 图像识别竞赛中，杰弗里·辛顿的学生
亚历克斯·克里热夫斯基（Alex Krizhevsky）提出了深度卷积神经网络 AlexNet，
以超出第二名10个百分点的巨大优势夺冠，震惊了整个学术界。

这标志着"深度学习"时代的到来。深度学习的核心突破来自三个方面：
其一是大数据：互联网产生了海量标注数据；
其二是 GPU：英伟达（NVIDIA）的图形处理器使并行计算速度提升百倍；
其三是算法改进：ReLU激活函数、Dropout正则化等技术解决了深层网络的训练难题。

第五节：Transformer 与大语言模型（2017年至今）

2017年，谷歌（Google）发表论文《Attention Is All You Need》，提出了 Transformer 架构。
Transformer 完全基于"注意力机制"（Attention Mechanism），能够并行处理序列数据，
训练速度远超此前的循环神经网络（RNN/LSTM）。

2018年，谷歌推出 BERT（双向编码器表示）模型，在多项自然语言处理任务上刷新了记录。
同年，OpenAI 推出 GPT-1，开启了生成式预训练语言模型的时代。

2020年，OpenAI 发布 GPT-3，拥有1750亿参数，展现出惊人的语言生成和零样本学习能力，
引发了全球对大语言模型（LLM）的广泛关注。

2022年11月，OpenAI 推出 ChatGPT，将 GPT 技术与对话界面结合，
仅5天就吸引了100万用户，成为历史上增长最快的消费级应用之一。
ChatGPT 使普通大众第一次真实体验到了高质量 AI 对话的魅力。

2023年，GPT-4 发布，多模态能力（理解图片和文字）进一步拓展了大模型的应用边界。
与此同时，谷歌的 Gemini、Meta 的 LLaMA、百度的文心一言等也相继发布，
大语言模型领域进入百花齐放的时代。

第六节：未来展望

当前的大语言模型虽然能力强大，但仍面临诸多挑战：
幻觉问题（Hallucination）——模型可能生成听起来合理但实际错误的内容；
知识截止日期——训练数据有时效性，无法获知最新信息；
推理能力限制——复杂的多步逻辑推理仍是模型的弱项。

RAG（检索增强生成）技术正是为了解决前两个问题而生：
通过在回答时动态检索最新文档，让 LLM 的回答有据可查，大幅减少幻觉。
这正是我们这个项目要实现的核心功能。
"""

# ── 用 Document 包装文本 ──────────────────────────────────
#
# LangChain 用 Document 对象统一表示"一段文档"
# 包含两个字段：
#   page_content → 文本内容（字符串）
#   metadata     → 元数据（字典，可存文件名、页码、来源 URL 等）

raw_document = Document(
    page_content=RAW_TEXT.strip(),
    metadata={"source": "ai_history.txt", "topic": "人工智能发展简史"},
)

print(f"【原始文档信息】")
print(f"  来源: {raw_document.metadata['source']}")
print(f"  总字符数: {len(raw_document.page_content)} 字")
print()

# ── 文本切块 ─────────────────────────────────────────────
#
# RecursiveCharacterTextSplitter：LangChain 推荐的文本切块器
#
# 参数说明：
#   chunk_size    = 每块最多包含多少字符
#                  500 字 ≈ 250 个中文汉字，适合中文文档
#   chunk_overlap = 相邻两块重叠多少字符
#                  设置重叠是为了防止一个完整句子被切断在两块边界，
#                  导致检索时丢失上下文
#
# "Recursive"的含义：优先按段落（\n\n）切，切不开再按换行（\n），
#   再切不开才按字符切。这样尽量保持语义完整性。
#
# ⚠️ 避坑指南：chunk_overlap 必须严格小于 chunk_size，
#   否则 LangChain 会抛出 ValueError！

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,    # 用字符数计算长度（不是 token 数）
)

# 把单个 Document 对象切割成多个小块（仍然是 Document 列表）
chunks = text_splitter.split_documents([raw_document])

print(f"【文本切块结果】")
print(f"  原始文档 {len(raw_document.page_content)} 字 → 切成 {len(chunks)} 块")
print()

# 打印第一块，让你看看每块大概长什么样
print(f"【第 1 块样本（共 {len(chunks[0].page_content)} 字）】")
print("-" * 60)
print(chunks[0].page_content)
print("-" * 60)
print(f"  元数据: {chunks[0].metadata}")
print()
```

- [ ] **Step 2：运行并验证第 1 章输出**

```bash
python rag/rag_qa.py
```

预期在第0章输出之后，新增：
```
============================================================
第 1 章：文档加载 + 文本切块
============================================================

【原始文档信息】
  来源: ai_history.txt
  总字符数: xxxx 字

【文本切块结果】
  原始文档 xxxx 字 → 切成 x 块

【第 1 块样本（共 xxx 字）】
------------------------------------------------------------
人工智能发展简史

第一节：萌芽期（1950年代）...
------------------------------------------------------------
  元数据: {'source': 'ai_history.txt', 'topic': '人工智能发展简史'}
```

验证要点：切块数量 > 1，第一块包含文档开头内容，元数据正确。

- [ ] **Step 3：提交第 1 章**

```bash
git add rag/rag_qa.py
git commit -m "feat: add rag_qa.py chapter 1 - document loading and text splitting

Inline 3000-char AI history text, RecursiveCharacterTextSplitter
chunk_size=500 overlap=50, prints chunk count and sample chunk.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4：追加第 2 章（向量化 + 存入 FAISS）

**Files:**
- Modify: `rag/rag_qa.py`（末尾追加）

- [ ] **Step 1：追加第 2 章代码**

在 `rag/rag_qa.py` 末尾追加：

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：向量化 + 存入 FAISS 向量数据库
# 目标：把每个文本块变成数字向量，存入内存索引
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：向量化 + 存入 FAISS")
print("=" * 60)
print()
print("【正在向量化所有文本块，请稍候（需要调用 Embedding API）...】")

# ── 核心操作：一行代码完成"向量化 + 建索引" ──────────────────
#
# FAISS.from_documents() 做了什么？
#   ① 遍历 chunks 列表中的每个 Document
#   ② 调用 embeddings.embed_documents() 把文本 → 向量（float 列表）
#   ③ 把所有向量存入 FAISS 索引（内存中的高速相似度搜索引擎）
#
# ⚠️ 避坑指南：FAISS 是纯内存索引！
#   程序退出后，索引消失，需要重新构建。
#   如果需要持久化，可以用：
#     vectorstore.save_local("./faiss_index")
#   下次用：
#     vectorstore = FAISS.load_local("./faiss_index", embeddings,
#                                    allow_dangerous_deserialization=True)
#   本教学代码为了简洁，不做持久化。

vectorstore = FAISS.from_documents(
    documents=chunks,       # 所有文本块（来自第1章）
    embedding=embeddings,   # 用第0章初始化的 Embeddings 客户端
)

print(f"✅ 向量化完成！FAISS 索引已建立")
print(f"   已存入向量数量: {vectorstore.index.ntotal} 个")
print(f"   （等于切块数量 {len(chunks)}，每块对应一个向量）")
print()
print("💡 向量是什么？")
print("   每个文本块 → 一个 float 列表（例如 1536 维）")
print("   含义相近的文本 → 向量在高维空间里距离更近")
print("   检索时：把问题也变成向量，找距离最近的 k 个文本块")
print()
```

- [ ] **Step 2：运行并验证第 2 章输出**

```bash
python rag/rag_qa.py
```

预期在第1章输出之后，新增（此步骤会发起真实的 Embedding API 调用）：
```
============================================================
第 2 章：向量化 + 存入 FAISS
============================================================

【正在向量化所有文本块，请稍候（需要调用 Embedding API）...】
✅ 向量化完成！FAISS 索引已建立
   已存入向量数量: x 个
   （等于切块数量 x，每块对应一个向量）

💡 向量是什么？
   每个文本块 → 一个 float 列表（例如 1536 维）
   含义相近的文本 → 向量在高维空间里距离更近
   检索时：把问题也变成向量，找距离最近的 k 个文本块
```

关键验证：`已存入向量数量` 等于第1章打印的切块数量。若 Embedding API 报错（404/422），需修改 `EMBEDDING_MODEL` 常量后重试。

- [ ] **Step 3：提交第 2 章**

```bash
git add rag/rag_qa.py
git commit -m "feat: add rag_qa.py chapter 2 - vectorization and FAISS indexing

FAISS.from_documents() builds in-memory vector index,
prints vector count, explains embedding concept.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5：追加第 3 章（检索演示——打印召回的原始文本块）

**Files:**
- Modify: `rag/rag_qa.py`（末尾追加）

- [ ] **Step 1：追加第 3 章代码**

在 `rag/rag_qa.py` 末尾追加：

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：检索演示（不问 LLM，只看检索结果）
# 目标：亲眼看到"向量数据库检索到了哪些文本块"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：检索演示（直接查看召回的原始文本块）")
print("=" * 60)
print()

# ── 创建检索器 ────────────────────────────────────────────
#
# as_retriever() 把向量数据库包装成一个"检索器"对象
# search_kwargs={"k": 3} 表示：每次检索返回最相似的 3 个文本块
#
# 检索原理：
#   用户提问 → 向量化为 query_vector
#   在 FAISS 索引中找出与 query_vector 最近的 k 个向量
#   返回这 k 个向量对应的文本块

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # 每次检索召回 3 个最相关的文本块
)


def show_retrieval_results(query: str) -> list:
    """执行检索并打印召回的文本块，返回结果列表"""
    print(f"🔍 检索问题：「{query}」")
    print()

    # retriever.invoke() 返回一个 Document 列表（k 个）
    docs = retriever.invoke(query)

    print(f"   检索到 {len(docs)} 个相关文本块：")
    print()

    for i, doc in enumerate(docs, 1):
        print(f"  ┌── 召回块 #{i}（{len(doc.page_content)} 字）──")
        # 打印完整文本块内容，缩进对齐
        lines = doc.page_content.split("\n")
        for line in lines:
            print(f"  │  {line}")
        print(f"  └── 元数据: {doc.metadata}")
        print()

    return docs


# ── 演示1：关于图灵测试 ─────────────────────────────────────
print("【演示：不同问题检索到不同的文本块】")
print("-" * 60)
show_retrieval_results("艾伦·图灵和图灵测试是什么？")

print("-" * 60)
# ── 演示2：关于深度学习 ─────────────────────────────────────
show_retrieval_results("深度学习是什么时候兴起的？")

print("💡 小结：RAG 的检索步骤就是这些。")
print("   大模型看到的'参考资料'，就是这几个文本块！")
print()
```

- [ ] **Step 2：运行并验证第 3 章输出**

```bash
python rag/rag_qa.py
```

预期新增（第3章）：
```
============================================================
第 3 章：检索演示（直接查看召回的原始文本块）
============================================================

【演示：不同问题检索到不同的文本块】
------------------------------------------------------------
🔍 检索问题：「艾伦·图灵和图灵测试是什么？」

   检索到 3 个相关文本块：

  ┌── 召回块 #1（xxx 字）──
  │  1950年，英国数学家艾伦·图灵（Alan Turing）...
  └── 元数据: {'source': 'ai_history.txt', ...}
  ...
```

关键验证：关于图灵的问题应该召回包含"图灵"的文本块，关于深度学习的问题应该召回包含"2012""AlexNet""深度学习"的文本块——两次检索召回的内容应该明显不同。

- [ ] **Step 3：提交第 3 章**

```bash
git add rag/rag_qa.py
git commit -m "feat: add rag_qa.py chapter 3 - retrieval demo with raw chunk display

Prints retrieved chunks verbatim for two different queries,
demonstrating that different questions retrieve different context.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6：追加第 4 章（完整 RAG 问答链）

**Files:**
- Modify: `rag/rag_qa.py`（末尾追加）

- [ ] **Step 1：追加第 4 章代码**

在 `rag/rag_qa.py` 末尾追加：

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：完整 RAG 问答链（LCEL 语法）
# 目标：把检索器、提示词、LLM 串联成一条完整的 RAG 链
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 4 章：完整 RAG 问答链")
print("=" * 60)
print()

# ── 构建 RAG 提示词模板 ──────────────────────────────────
#
# RAG 提示词的关键：把"检索到的上下文"和"用户问题"都塞进去
# 并明确告诉 LLM "只能基于这些资料回答，不能乱编"

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个严谨的知识库问答助手。
请仅根据下面提供的【参考资料】来回答用户的问题。
如果参考资料中没有相关信息，请直接说"根据现有资料，我无法回答这个问题"，不要猜测或编造。

【参考资料】
{context}"""),
    ("human", "{question}"),
])

# ── 构建文档格式化函数 ────────────────────────────────────
#
# retriever 返回的是 Document 对象列表，
# 但提示词模板里的 {context} 需要字符串。
# 这个函数把多个文档块拼接成一段文字。

def format_docs(docs: list) -> str:
    """把检索到的文档块列表拼接成字符串，用分隔符隔开"""
    return "\n\n---\n\n".join(
        f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
        for doc in docs
    )

# ── 构建 RAG 链（LCEL 语法） ──────────────────────────────
#
# 数据流解析：
#
#   输入：{"question": "用户的问题字符串"}
#       ↓
#   RunnableParallel（并行处理，同时做两件事）：
#     ├── "context"  : retriever | format_docs  → 把相关文本块拼成字符串
#     └── "question" : RunnablePassthrough()    → 原样保留用户问题
#       ↓
#   输出：{"context": "相关文本...", "question": "用户的问题"}
#       ↓
#   rag_prompt  →  填充模板，生成完整的提示词
#       ↓
#   llm         →  调用大模型，得到 AIMessage
#       ↓
#   parser      →  提取纯文本字符串
#
# ⚠️ 避坑指南：RunnablePassthrough 的作用
#   它不是"什么都不做"，而是"把输入原封不动地传给下一步"。
#   在 RunnableParallel 中，如果没有它，"question"键就会丢失！

parser = StrOutputParser()

rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,       # 检索 → 格式化为字符串
        question=RunnablePassthrough(),         # 原样保留问题字符串
    )
    | rag_prompt
    | llm
    | parser
)

print("✅ RAG 链构建完成！")
print("   链的结构：RunnableParallel(context, question) | prompt | llm | parser")
print()


def rag_query(question: str) -> str:
    """执行完整的 RAG 问答，先打印检索依据，再打印 AI 回答"""
    print(f"{'=' * 60}")
    print(f"❓ 问：{question}")
    print(f"{'=' * 60}")

    # 先单独检索，打印召回的原文（让你看到大模型"翻的是哪页书"）
    retrieved_docs = retriever.invoke(question)
    print(f"\n📚 检索召回了 {len(retrieved_docs)} 个相关文本块（大模型将基于这些资料回答）：")
    for i, doc in enumerate(retrieved_docs, 1):
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"   [{i}] {preview}...")

    # 再调用完整 RAG 链得到答案
    print(f"\n🤖 基于以上资料，AI 回答：")
    answer = rag_chain.invoke(question)
    print(f"   {answer}")
    print()
    return answer


# ── 运行三轮问答演示 ──────────────────────────────────────
print("【RAG 问答演示——三个问题】")
print()

rag_query("谁提出了图灵测试？在哪一年？")
rag_query("深度学习的三大突破是什么？")
rag_query("ChatGPT 是哪年发布的，发布后有什么影响？")

print("=" * 60)
print("🎉 项目二学习完毕！你已经掌握了 RAG 的完整工作流。")
print("   核心公式：文档切块 + 向量化 + 相似度检索 + LLM 生成 = RAG")
print("=" * 60)
```

- [ ] **Step 2：运行完整脚本，验证三轮 RAG 问答**

```bash
python rag/rag_qa.py
```

预期最终新增（第4章）：
```
============================================================
第 4 章：完整 RAG 问答链
============================================================

✅ RAG 链构建完成！
   链的结构：RunnableParallel(context, question) | prompt | llm | parser

【RAG 问答演示——三个问题】

============================================================
❓ 问：谁提出了图灵测试？在哪一年？
============================================================

📚 检索召回了 3 个相关文本块（大模型将基于这些资料回答）：
   [1] 1950年，英国数学家艾伦·图灵（Alan Turing）在论文...
   [2] ...
   [3] ...

🤖 基于以上资料，AI 回答：
   1950年，英国数学家艾伦·图灵提出了图灵测试...

（另外两轮问答类似结构）

============================================================
🎉 项目二学习完毕！...
============================================================
```

关键验证点：
- AI 的回答内容必须来自文本块（可对照）
- 三轮问答的召回文本块应该各不相同

- [ ] **Step 3：提交第 4 章，完成项目二**

```bash
git add rag/rag_qa.py
git commit -m "feat: add rag_qa.py chapter 4 - complete RAG chain with LCEL

RunnableParallel(context=retriever|format_docs, question=passthrough)
| prompt | llm | parser. Three demo Q&A pairs with retrieval printout.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## 自检清单（Spec Self-Review）

- [x] Task 1 覆盖 rag/ 目录创建 + faiss-cpu 安装 + requirements.txt 更新
- [x] Task 2 覆盖文件头"开卷考试"RAG 工作流科普 + 第0章初始化（ChatOpenAI + OpenAIEmbeddings + EMBEDDING_MODEL 常量）
- [x] Task 3 覆盖第1章：内联 AI 历史文本 + Document 包装 + RecursiveCharacterTextSplitter + 打印切块数和样本
- [x] Task 4 覆盖第2章：FAISS.from_documents() + 打印向量数量 + 避坑（内存索引）
- [x] Task 5 覆盖第3章：retriever.as_retriever(k=3) + 打印完整召回文本块（spec 核心要求）
- [x] Task 6 覆盖第4章：format_docs + rag_prompt + RunnableParallel + RunnablePassthrough + rag_chain + 三轮问答 + 每次打印检索依据
- [x] 所有 ⚠️ 避坑指南已写入代码注释（EMBEDDING_MODEL 可换、faiss-cpu/gpu 冲突、chunk_overlap < chunk_size、FAISS 内存索引、RunnablePassthrough 作用）
- [x] 每个 Step 都有完整代码，无 TBD 占位符
- [x] 每个 Step 都有预期输出，方便验证
- [x] 变量名在所有章节一致：chunks、vectorstore、retriever、rag_chain、rag_query
- [x] format_docs 函数在第4章定义，rag_chain 也在第4章使用——无提前引用问题
- [x] retriever 在第3章定义，第4章继续复用——章节间依赖清晰
