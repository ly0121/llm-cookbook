"""
╔══════════════════════════════════════════════════════════════════╗
║         项目十六：Memory 进阶 — 摘要记忆 + 向量长期记忆             ║
║         让 AI 像人一样：记住要点、遗忘细节、回忆相关经历            ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普一：项目一的记忆有什么问题？】
═══════════════════════════════════════════════════════════════════

项目一（chatbot.py）用的是"完整历史记忆"：

  ┌─────────────────────────────────────────────────────────────┐
  │  完整历史记忆的问题：                                        │
  │                                                             │
  │  用户聊了 100 轮对话 → 100 轮全部塞进 Prompt               │
  │                                                             │
  │  问题一：Token 爆炸 💸                                      │
  │    100轮对话 ≈ 5万token → 超出上下文窗口 → 报错！           │
  │    即使不超，也浪费大量 token 费用                           │
  │                                                             │
  │  问题二：信息稀释 📉                                        │
  │    第 1 轮说了"我叫张三"，到第 100 轮被海量历史淹没         │
  │    LLM 注意力被分散，忘了重要信息                           │
  │                                                             │
  │  问题三：跨会话失忆 🧠                                      │
  │    用户昨天聊的内容，今天开新会话就全忘了                    │
  └─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
【前置科普二：三种进阶记忆策略】
═══════════════════════════════════════════════════════════════════

  ┌────────────────────┬──────────────────────────────────────────┐
  │  策略               │  类比                                     │
  ├────────────────────┼──────────────────────────────────────────┤
  │  窗口记忆          │  只记最近 K 轮（短期记忆）               │
  │  Window Memory     │  像人只记得今天发生的事                   │
  ├────────────────────┼──────────────────────────────────────────┤
  │  摘要记忆          │  把长历史压缩成一段摘要（工作记忆）       │
  │  Summary Memory    │  像人回忆：不记原话，但记得大意           │
  ├────────────────────┼──────────────────────────────────────────┤
  │  向量记忆          │  把历史向量化，按相关性检索（长期记忆）   │
  │  Vector Memory     │  像人回忆：遇到相关话题时想起过去的事    │
  └────────────────────┴──────────────────────────────────────────┘

  本项目实现全部三种，并演示"组合使用"的效果。

═══════════════════════════════════════════════════════════════════
【前置科普三：人类记忆 vs AI 记忆的对应关系】
═══════════════════════════════════════════════════════════════════

  人类：
    感觉记忆（<1秒）→ 短期记忆（几分钟）→ 长期记忆（永久）

  AI 对应：
    当前输入    → 窗口记忆（最近K轮） → 向量记忆（全部历史）
                  摘要记忆（压缩版）

  人类"想起来"= 当前话题触发了长期记忆中的相关片段
  AI "想起来" = 当前问题向量搜索到了历史对话中的相关片段
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("项目十六：Memory 进阶 — 摘要记忆 + 向量长期记忆")
print("=" * 60)
print()

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "kivy-kimi-k2_5"

llm = ChatOpenAI(model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, temperature=0.7)
summary_llm = ChatOpenAI(
    model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, temperature=0.0
)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

print("✅ LLM + Embeddings 初始化完成")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：窗口记忆（Window Memory）
# 目标：只保留最近 K 轮对话，超过的自动丢弃
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：窗口记忆（Window Memory）")
print("=" * 60)
print()

# ── 窗口记忆原理 ──────────────────────────────────────────
#
# 最简单的"防爆"策略：只保留最近 K 轮。
# 优点：Token 消耗固定，不会超限。
# 缺点：超过 K 轮的内容彻底丢失。
#
# 实现方式：维护一个 message 列表，超过 2K 条（问+答=2）就删头部。


class WindowMemory:
    """窗口记忆：只保留最近 K 轮对话"""

    def __init__(self, k: int = 3):
        self.k = k
        self.messages = []  # 存储 HumanMessage 和 AIMessage

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        self.messages.append(AIMessage(content=content))

    def get_messages(self) -> list:
        """返回最近 K 轮的消息（K轮 = 2K条消息）"""
        return self.messages[-(self.k * 2) :]

    @property
    def size(self) -> int:
        return len(self.messages) // 2


# ── 演示窗口记忆 ──────────────────────────────────────────

window_mem = WindowMemory(k=2)  # 只记最近 2 轮

window_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个友好的助手，回答简洁（30字以内）。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

window_chain = window_prompt | llm | StrOutputParser()

# 模拟 5 轮对话
CONVERSATIONS = [
    "我叫张三，今年28岁",
    "我是一名 Python 开发者",
    "我在北京工作",
    "我喜欢打篮球",
    "请问我叫什么名字？",  # 测试：第1轮的信息还记得吗？
]

print("【窗口记忆演示（k=2，只记最近2轮）】")
print()

for i, question in enumerate(CONVERSATIONS, 1):
    history = window_mem.get_messages()
    answer = window_chain.invoke({"question": question, "history": history})

    window_mem.add_user_message(question)
    window_mem.add_ai_message(answer)

    retained = "✅ 在窗口内" if i > 3 or i <= 2 else "⚠️"
    print(f"  [{i}] 用户：{question}")
    print(f"      AI：{answer}")
    print(f"      （窗口内消息：{len(window_mem.get_messages())}条）")
    print()

print("  💡 观察：第5轮问'我叫什么'时，第1轮的信息已滑出窗口！")
print("     窗口记忆的代价：早期信息会永久丢失。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：摘要记忆（Summary Memory）
# 目标：把超出窗口的历史压缩成摘要，保留关键信息
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：摘要记忆（Summary Memory）")
print("=" * 60)
print()

# ── 摘要记忆原理 ──────────────────────────────────────────
#
# 不是直接丢弃旧消息，而是让 LLM 把旧消息"压缩"成一段摘要。
#
# 流程：
#   对话历史超过 K 轮 → 把最旧的几轮发给 LLM → LLM 生成摘要
#   → 用摘要替代原始对话 → 节省 Token 但保留关键信息
#
# 类比：你不记得昨天每一句话，但记得"昨天和小明讨论了项目进度，
#       他说下周能交付"。这就是"摘要"。

SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """请将以下对话历史压缩成一段简洁的摘要。
要求：
1. 保留关键事实（人名、数字、偏好、重要决定）
2. 去除寒暄和重复
3. 用第三人称描述
4. 控制在100字以内""",
        ),
        ("human", "对话历史：\n{conversation}\n\n请生成摘要："),
    ]
)

summarize_chain = SUMMARIZE_PROMPT | summary_llm | StrOutputParser()


class SummaryMemory:
    """摘要记忆：超出窗口的部分自动压缩为摘要"""

    def __init__(self, k: int = 2):
        self.k = k  # 窗口大小
        self.messages = []  # 当前窗口内的消息
        self.summary = ""  # 压缩后的历史摘要

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str):
        self.messages.append(AIMessage(content=content))
        # 检查是否需要压缩
        if len(self.messages) > self.k * 2:
            self._compress()

    def _compress(self):
        """将最旧的对话压缩进摘要"""
        # 取出要压缩的部分（保留最近 k 轮）
        to_compress = self.messages[:2]  # 压缩最旧的 1 轮
        self.messages = self.messages[2:]  # 保留剩余

        # 构造对话文本
        conv_text = ""
        for msg in to_compress:
            role = "用户" if isinstance(msg, HumanMessage) else "AI"
            conv_text += f"{role}：{msg.content}\n"

        # 如果已有摘要，一起压缩
        if self.summary:
            conv_text = f"[之前的摘要] {self.summary}\n[新对话]\n{conv_text}"

        # 调用 LLM 生成新摘要
        self.summary = summarize_chain.invoke({"conversation": conv_text})

    def get_context(self) -> tuple:
        """返回 (摘要, 最近消息列表)"""
        return self.summary, self.messages


# ── 演示摘要记忆 ──────────────────────────────────────────

summary_mem = SummaryMemory(k=2)

summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个友好的助手，回答简洁（30字以内）。

{summary_context}""",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

summary_chain_full = summary_prompt | llm | StrOutputParser()

print("【摘要记忆演示（k=2，超出部分自动摘要）】")
print()

CONVERSATIONS_2 = [
    "我叫李明，今年32岁，是一名算法工程师",
    "我擅长 PyTorch 和 TensorFlow，最近在研究大模型微调",
    "我的目标是今年内发一篇顶会论文",
    "我目前在做 LoRA 方面的研究",
    "请问我的名字和研究方向是什么？",  # 测试：早期信息是否被摘要保留
]

for i, question in enumerate(CONVERSATIONS_2, 1):
    summary, history = summary_mem.get_context()
    summary_context = f"[历史摘要] {summary}" if summary else ""

    answer = summary_chain_full.invoke(
        {
            "question": question,
            "history": history,
            "summary_context": summary_context,
        }
    )

    summary_mem.add_user_message(question)
    summary_mem.add_ai_message(answer)

    print(f"  [{i}] 用户：{question}")
    print(f"      AI：{answer}")
    if summary_mem.summary:
        print(f"      📝 当前摘要：{summary_mem.summary[:50]}...")
    print()

print("  💡 观察：虽然早期对话被压缩了，但摘要保留了'李明''算法工程师'等关键信息！")
print("     摘要记忆 = 窗口记忆 + 不丢失关键事实")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：向量长期记忆（Vector Memory）
# 目标：把所有对话向量化，按语义相关性检索"回忆"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：向量长期记忆（Vector Memory）")
print("=" * 60)
print()

# ── 向量记忆原理 ──────────────────────────────────────────
#
# 把每一轮对话向量化存入向量库。
# 当用户提新问题时，用问题去向量库中检索最相关的历史对话。
#
# 优势：
#   ① 无限容量（向量库不怕多）
#   ② 按相关性召回（不是按时间顺序）
#   ③ 跨会话可用（持久化后可以跨会话记忆）
#
# 类比：人类的长期记忆不是"按时间倒放"的，
#       而是"遇到相关刺激时触发回忆"。


class VectorMemory:
    """向量长期记忆：按语义相关性检索历史对话"""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []  # 存储所有对话 Document
        self.vectorstore = None

    def add_interaction(self, user_msg: str, ai_msg: str, metadata: dict = None):
        """记录一轮对话到向量库"""
        # 把一轮对话组成一个 Document
        content = f"用户问：{user_msg}\nAI答：{ai_msg}"
        meta = metadata or {}
        meta["user_message"] = user_msg
        doc = Document(page_content=content, metadata=meta)
        self.documents.append(doc)

        # 重建向量索引（生产中应增量添加）
        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)

    def recall(self, query: str, k: int = 2) -> list:
        """根据当前问题，回忆最相关的历史对话"""
        if not self.vectorstore:
            return []
        results = self.vectorstore.similarity_search(query, k=k)
        return results

    @property
    def size(self) -> int:
        return len(self.documents)


# ── 演示向量长期记忆 ───────────────────────────────────────

vector_mem = VectorMemory(embeddings)

# 预置一些"历史对话"（模拟跨会话记忆）
PAST_CONVERSATIONS = [
    ("我最喜欢吃四川火锅", "好的，记住了！你喜欢四川火锅。"),
    ("我下周要去上海出差", "祝出差顺利！需要我推荐上海的餐厅吗？"),
    ("我家的猫叫小橘，是只橘猫", "小橘是个可爱的名字！橘猫通常很亲人。"),
    ("我的生日是3月15日", "记住了，3月15日是你的生日！"),
    ("我正在学习 LangChain 框架", "LangChain 是很好的 LLM 应用开发框架。"),
    ("我对自动驾驶技术很感兴趣", "自动驾驶是个很有前景的领域。"),
]

print("  ⏳ 预置 6 条历史对话到向量记忆...")
for user_msg, ai_msg in PAST_CONVERSATIONS:
    vector_mem.add_interaction(user_msg, ai_msg)
print(f"  ✅ 向量记忆中有 {vector_mem.size} 条记录")
print()

# 测试：用新问题"激活"相关记忆
TEST_QUERIES = [
    "推荐一家好吃的餐厅",  # 应该回忆起"喜欢四川火锅"
    "我的宠物怎么样了",  # 应该回忆起"猫叫小橘"
    "有什么技术值得学",  # 应该回忆起"LangChain""自动驾驶"
]

print('【向量记忆检索演示——按语义相关性"回忆"】')
print()

for query in TEST_QUERIES:
    recalled = vector_mem.recall(query, k=2)
    print(f"  ❓ 当前问题：{query}")
    print(f"  💭 触发的回忆：")
    for j, doc in enumerate(recalled, 1):
        print(f"     [{j}] {doc.page_content[:50]}...")
    print()

# ── 完整的"带向量记忆的对话"演示 ─────────────────────────

print("【完整演示：向量记忆 + 对话】")
print()

vector_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个有记忆力的 AI 助手。

以下是你对用户的"回忆"（来自之前的对话）：
{recalled_memories}

利用这些回忆来个性化你的回答。回答简洁（50字以内）。""",
        ),
        ("human", "{question}"),
    ]
)

vector_chain = vector_prompt | llm | StrOutputParser()

FINAL_QUESTIONS = [
    "今晚吃什么好？",
    "送我什么生日礼物好？",
]

for question in FINAL_QUESTIONS:
    # 检索相关记忆
    memories = vector_mem.recall(question, k=2)
    memory_text = (
        "\n".join(doc.page_content for doc in memories) if memories else "暂无相关回忆"
    )

    answer = vector_chain.invoke(
        {
            "question": question,
            "recalled_memories": memory_text,
        }
    )

    print(f"  ❓ 用户：{question}")
    print(
        f"  💭 激活记忆：{memories[0].metadata.get('user_message', '')[:30] if memories else '无'}..."
    )
    print(f"  🤖 AI：{answer}")
    print()

print("  💡 向量记忆的威力：")
print("     问'今晚吃什么' → 想起'你喜欢四川火锅' → 推荐火锅")
print("     问'送什么礼物' → 想起'你的生日是3月15日' → 个性化建议")
print("     这就是'像人一样的记忆'！")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：组合记忆（摘要 + 向量 = 最强记忆体系）
# 目标：短期用摘要，长期用向量，两者结合
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 4 章：组合记忆架构")
print("=" * 60)
print()

# ── 组合策略 ──────────────────────────────────────────────
#
# 短期（当前会话）：窗口记忆 或 摘要记忆 → 保持上下文连贯
# 长期（跨会话）  ：向量记忆 → 按相关性召回过去的重要信息
#
# Prompt 构造：
#   [系统提示]
#   [向量召回的长期记忆]（2-3条最相关的）
#   [当前会话摘要]（如果有）
#   [最近几轮消息]
#   [用户当前问题]

print("【组合记忆架构示意】")
print()
print("  ┌─────────────────────────────────────────────────────────┐")
print("  │  Prompt 组装顺序（从上到下）：                           │")
print("  │                                                         │")
print("  │  ① System Prompt（角色设定）                             │")
print("  │  ② 长期记忆（向量检索的 top-K 相关历史）                │")
print("  │  ③ 会话摘要（当前会话的压缩版）                         │")
print("  │  ④ 最近 K 轮原始对话（窗口内）                          │")
print("  │  ⑤ 当前用户问题                                         │")
print("  │                                                         │")
print("  │  效果：短期记忆保持连贯 + 长期记忆提供个性化             │")
print("  └─────────────────────────────────────────────────────────┘")
print()
print("  💡 Token 预算分配建议（4K上下文为例）：")
print("     System Prompt   ：~200 token")
print("     长期记忆（向量） ：~500 token（2-3条）")
print("     会话摘要        ：~200 token")
print("     最近对话窗口    ：~800 token（2-3轮）")
print("     用户问题+回答   ：~300 token")
print("     ─────────────────────────────")
print("     总计            ：~2000 token（留一半给生成）")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 总结
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("🎉 项目十六学习完毕！")
print("=" * 60)
print()
print("💡 记忆策略选型：")
print()
print("  ┌──────────────────┬──────────────┬────────────────────────┐")
print("  │  策略             │  适用场景     │  特点                   │")
print("  ├──────────────────┼──────────────┼────────────────────────┤")
print("  │  窗口记忆        │  简单对话     │  实现最简单，但会丢失   │")
print("  │  摘要记忆        │  长会话       │  保留关键信息，额外LLM调用 │")
print("  │  向量记忆        │  跨会话/个性化│  无限容量，按相关性召回 │")
print("  │  组合记忆        │  生产环境     │  短期+长期，效果最好    │")
print("  └──────────────────┴──────────────┴────────────────────────┘")
print()
print("💡 生产进阶：")
print("   ① 向量记忆持久化：存入 Chroma/Milvus，重启不丢失")
print("   ② 记忆衰减：老记忆随时间降低权重，模拟人类遗忘")
print("   ③ 实体记忆：提取'张三→年龄28→职业开发者'知识图谱")
print("   ④ 多用户隔离：每个用户有独立的记忆空间")
print("=" * 60)
