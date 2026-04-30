"""
╔══════════════════════════════════════════════════════════════════╗
║         项目零：LLM 原生 API 核心调用与控制                          ║
║         完全不使用 LangChain，纯用 OpenAI Python SDK               ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普一：剧组拍戏——System、User、Assistant 到底是什么？】
═══════════════════════════════════════════════════════════════════

想象你正在拍一部电影：

  ┌─────────────────────────────────────────────────────────┐
  │  🎬 导演（System）                                        │
  │  在开机前，导演把演员叫进小黑屋，悄悄交代：               │
  │  "你今天扮演一个冷静严肃的侦探，只说中文，不能跑题。"     │
  │  这些话观众永远听不到，但演员会铭记在心并照单全收。       │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  🎤 观众提问（User）                                      │
  │  这是观众（你）每一轮向 AI 提出的问题或指令。             │
  │  AI 会根据导演指令（System）的框架来回答你的问题。        │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │  🤖 AI 演员（Assistant）                                  │
  │  这是 AI 的回复。在多轮对话中，每一条历史回复都会原样     │
  │  追加到 messages 数组里，让 AI 能"记住"之前说过什么。    │
  └─────────────────────────────────────────────────────────┘

发给 OpenAI API 的数据结构长这样：

  messages = [
      {"role": "system",    "content": "你是冷静的侦探..."},     ← 导演指令
      {"role": "user",      "content": "案发现场有什么线索？"},   ← 观众提问
      {"role": "assistant", "content": "门口有一个泥脚印..."},   ← AI 上一轮回复
      {"role": "user",      "content": "嫌疑人是谁？"},           ← 观众新问题
  ]

  关键点：每次调用 API，都要把"完整对话历史"全部发过去！
  因为 LLM 本身没有记忆，它只能通过读取 messages 来"理解"上下文。
  这也是 langchain/chatbot.py 里 RunnableWithMessageHistory 帮你做的事。

═══════════════════════════════════════════════════════════════════
【前置科普二：Temperature 是什么？——AI 的"脑洞大小旋钮"】
═══════════════════════════════════════════════════════════════════

Temperature（温度）控制 AI 回答的"随机性"和"创造力"：

  temperature = 0.0  →  "学霸模式"
    AI 每次都选概率最高的下一个词，答案非常确定、保守、可复现。
    适合场景：代码生成、数学计算、需要精确答案的问答。
    缺点：容易"一本正经地重复废话"，答案可能机械。

  temperature = 0.7  →  "黄金平衡点"（本文件使用此值）
    在确定性和创造力之间取得平衡，大多数对话场景的默认推荐值。
    答案既不会太离谱，也有足够的自然感。

  temperature = 1.0+  →  "艺术家模式"
    AI 会大胆选择低概率的词，答案天马行空、充满创意。
    适合场景：创意写作、头脑风暴、生成多样化内容。
    缺点：可能"一本正经地胡说八道"——听起来流畅但内容不准确！

  ⚠️ 记住：高温度 ≠ 更聪明，低温度 ≠ 更笨
     温度只影响"选词的随机性"，不影响模型的知识量。
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：OpenAI 客户端初始化
# 目标：用最少的代码建立与 LLM 的连接
# 对比：和 langchain/chatbot.py 的 ChatOpenAI(...)对比——
#       那是 LangChain 封装版，这是原生版，底层原理完全一样
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 只需要导入一个东西！这就是"纯原生"的魅力——没有任何框架依赖
from openai import OpenAI

print("=" * 60)
print("第 0 章：OpenAI 客户端初始化")
print("=" * 60)

# ── API 连接配置 ──────────────────────────────────────────
#
# 这三个参数，你在 langchain/chatbot.py（LangChain 版）里也见过：
#   ChatOpenAI(model=..., api_key=..., base_url=...)
# 原生 SDK 一模一样，只是类名不同：OpenAI(...)
#
# base_url：告诉 SDK "不要连 OpenAI 官方，连这个兼容接口"
#            只要目标接口遵循 OpenAI 的 API 格式，就能无缝切换

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "kivy-kimi-k2_5"

# 创建 OpenAI 客户端实例
# 这个 client 对象是你与 LLM 通信的"电话机"，后续所有调用都通过它
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

print(f"✅ OpenAI 原生客户端初始化完成")
print(f"   模型: {MODEL_NAME}")
print(f"   接口: {BASE_URL}")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：阻塞式调用 + 原始数据包解剖
# 目标：看清楚 OpenAI API 返回的完整原始对象长什么样
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：阻塞式调用 + 原始数据包解剖")
print("=" * 60)
print()

# ── 构造 messages 数组 ────────────────────────────────────
#
# messages 是发给 API 的核心数据，本质是一个 JSON 数组。
# 每个元素是一个对象，有两个字段：
#   "role"    → 角色，只能是 "system" / "user" / "assistant"
#   "content" → 这条消息的具体内容（字符串）
#
# 发给 OpenAI 服务器时，它长这样（JSON 格式）：
# [
#   {"role": "system",  "content": "你是..."},
#   {"role": "user",    "content": "..."}
# ]

messages = [
    {
        "role": "system",
        # 导演指令：给 AI 设定角色和约束
        # 这条消息用户看不到，但 AI 会严格遵守
        "content": "你是一个严谨的科普作家，用通俗易懂的语言解释复杂概念，回答控制在60字以内。",
    },
    {
        "role": "user",
        # 用户提问：这一轮我们问的问题
        "content": "黑洞是什么？",
    },
]

print("【发送给 API 的 messages 数组】")
for i, msg in enumerate(messages):
    print(f"  [{i}] role={msg['role']!r:12s}  content={msg['content'][:40]!r}...")
print()

# ── 发起阻塞式 API 调用 ───────────────────────────────────
#
# "阻塞式"的意思：代码执行到这一行后会"暂停"，
# 等待 API 服务器处理完毕并返回完整结果后，才继续向下执行。
# 就像打电话：你说完话，必须等对方把整段话都说完才挂断。

print("【正在调用 API（阻塞中，请稍候...）】")

response = client.chat.completions.create(
    model=MODEL_NAME,       # 指定模型
    messages=messages,      # 发送对话历史
    temperature=0.7,        # 温度：0.7 = 黄金平衡点（见文件头科普）
    max_tokens=200,         # 最多生成 200 个 token（防止回复过长）
    # stream=False          # 默认就是 False（阻塞式），注释掉是为了和第2章对比
)

# ── 解剖原始返回对象 ──────────────────────────────────────
#
# response 是一个 ChatCompletion 对象，不是普通字符串！
# LangChain 的 StrOutputParser 做的事情，就是把这个对象"剥开"
# 只取出 content 字符串。现在我们自己动手剥一遍，更有感觉。

print()
print("【API 返回的完整原始对象（这就是 OpenAI 服务器发回来的数据）】")
print("-" * 60)
print(response)  # 打印整个对象，让你看到所有字段
print("-" * 60)
print()

# ── 逐层解剖每个关键字段 ──────────────────────────────────

print("【逐层解剖关键字段】")
print()

# choices：一个列表，通常只有一个元素（除非你设置了 n>1 要求多个备选）
# choices[0] 就是 AI 的第一条（也是唯一一条）回复
print(f"  ① response.choices  （类型: {type(response.choices)}，长度: {len(response.choices)}）")
print(f"     → choices 是一个列表，每个元素是一个候选回复（通常只有 1 个）")
print()

# message：这一轮的完整消息对象（包含 role 和 content）
print(f"  ② response.choices[0].message  （类型: {type(response.choices[0].message).__name__}）")
print(f"     → role: {response.choices[0].message.role!r}")
print(f"     → content（AI 实际说的话）：")
print(f"       【{response.choices[0].message.content}】")
print()

# finish_reason：AI 为什么停止生成？
# "stop"           → 自然结束（最常见，AI 认为回答完整了）
# "length"         → 达到了 max_tokens 限制被截断（增大 max_tokens 可解决）
# "content_filter" → 内容被安全过滤器拦截
print(f"  ③ response.choices[0].finish_reason: {response.choices[0].finish_reason!r}")
print(f"     → 'stop' = AI 自然结束  'length' = 被 max_tokens 截断")
print()

# usage：这次调用消耗了多少 token（计费的依据！）
# prompt_tokens     = 你发过去的 messages 用了多少 token
# completion_tokens = AI 回复用了多少 token
# total_tokens      = 两者之和（就是这次调用的计费量）
print(f"  ④ response.usage（Token 消耗统计）：")
print(f"     → prompt_tokens（你的输入）:    {response.usage.prompt_tokens}")
print(f"     → completion_tokens（AI 输出）: {response.usage.completion_tokens}")
print(f"     → total_tokens（本次总消耗）:   {response.usage.total_tokens}")
print()
print("💡 小结：LangChain 的 StrOutputParser 做的事，等价于：")
print("   response.choices[0].message.content")
print()
