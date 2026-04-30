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
