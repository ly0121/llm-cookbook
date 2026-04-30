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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：流式打字机效果（Streaming）
# 目标：实现 ChatGPT 那种"一个字一个字蹦出来"的效果
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：流式打字机效果（Streaming）")
print("=" * 60)
print()
print("【阻塞式 vs 流式的原理区别】")
print("""
  阻塞式（stream=False，第1章）：
    你 → 发请求 → 等待 → 等待 → 等待 → 收到完整回复
    体验：等一段时间，然后文字"一下子全出来"

  流式（stream=True，本章）：
    你 → 发请求 → 立刻开始收到碎片 → 碎片 → 碎片 → 结束
    体验：文字像打字机一样"一个字一个字蹦出来"

  流式的好处：
    ① 用户体验更好，感觉更"实时"
    ② 不用等 AI 全部生成完才看到内容
    ③ 可以在 AI 还在"说话"时就开始处理前面的内容
""")

# ── 构造 messages（和第1章结构一模一样，内容不同） ─────────

messages_stream = [
    {
        "role": "system",
        "content": "你是一位有趣的科普作家，擅长用生动的比喻解释自然现象，回答在100字左右。",
    },
    {
        "role": "user",
        "content": "为什么天空是蓝色的？",
    },
]

# ── 发起流式 API 调用 ─────────────────────────────────────
#
# 和第1章的代码对比，唯一的区别就是加了：stream=True
# 加了这个参数后，API 不等 AI 说完，而是每生成一点就立刻发一个"碎片"给你

print("【流式打字机效果（注意：下面的字会一个个蹦出来）】")
print("AI：", end="", flush=True)  # 先打印"AI："前缀，不换行

stream = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages_stream,
    temperature=0.7,
    max_tokens=300,
    stream=True,  # 🔑 关键参数！开启流式输出
)

# ── 用 for 循环接收每一个碎片（chunk）────────────────────
#
# stream 是一个迭代器，每次循环拿到一个"碎片"（chunk）
# 每个碎片里只有一小段文字（可能是一个字、几个字，甚至是标点）
#
# chunk 的结构和第1章的 response 类似，但有一个关键区别：
#   第1章：response.choices[0].message.content  ← 完整文本
#   本章：   chunk.choices[0].delta.content      ← 文字碎片
#
# "delta"（增量）：每个 chunk 只包含这一小步新增的内容，不是全文

full_response = ""  # 用来累积完整回复（如果你之后需要处理完整文本）

for chunk in stream:
    # 取出这个碎片的文字内容
    # ── 防御：处理两种不同的"结束信号" ─────────────────────────
    #
    # 情况一（标准 OpenAI 协议）：最后一个有效 chunk 的 choices 非空，
    #   但 delta.content = None，finish_reason = "stop"
    #   → 用 "or ''" 处理 None，这行 delta_text 会是空字符串，无害
    #
    # 情况二（本 Gateway 特有）：Gateway 在标准结束后再发一个额外的
    #   "哨兵 chunk"，它的 choices 是空列表 []
    #   → 不做处理会触发 IndexError，所以先 skip
    if not chunk.choices:
        continue
    delta_text = chunk.choices[0].delta.content or ""

    # 打字机效果的魔法：
    #   end=""    → 不在每个碎片后面换行（默认 print 会换行）
    #   flush=True → 立刻把缓冲区里的内容冲到屏幕，不等攒满再显示
    #               （如果没有 flush=True，字符可能会一次性全跳出来，打字机效果失败！）
    print(delta_text, end="", flush=True)

    full_response += delta_text  # 累积完整回复

# 所有碎片接收完毕后，打印一个换行（因为上面所有 print 都没有换行）
print()
print()

# ── 展示累积的完整回复 ────────────────────────────────────
print(f"【流式接收完毕！完整回复共 {len(full_response)} 个字符】")
print()

print("💡 小结：流式 vs 阻塞式，代码差异只有两处：")
print("   1. create(..., stream=True)")
print("   2. for chunk in stream:  →  chunk.choices[0].delta.content")
print("      （delta = 增量碎片，而非 message = 完整文本）")
print()
print("=" * 60)
print("🎉 项目零学习完毕！你已经掌握了 LLM 最底层的调用方式。")
print("   下一站：langchain/chatbot.py —— 看看 LangChain 帮你封装了什么。")
print("=" * 60)
