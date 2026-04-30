"""
╔══════════════════════════════════════════════════════════════════╗
║         项目三：基于 ReAct 框架的单体 Agent（全能助手）              ║
║         工具定义 + Agent 构建 + ReAct 思维链可视化                  ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普一：老板、员工、工具箱——Agent 究竟是什么？】
═══════════════════════════════════════════════════════════════════

先来对比两种场景：

  场景一：普通 LLM 对话（只能用脑子的员工）
  ┌─────────────────────────────────────────────────────────┐
  │  老板：北京今天天气怎么样？                                │
  │  员工：抱歉，我不知道，我的知识有截止日期，               │
  │        而且我没有办法上网查询实时信息。                    │
  └─────────────────────────────────────────────────────────┘
  问题：LLM 的知识是静态的，它无法主动获取外部信息。

  场景二：Agent（配备了工具箱的员工）
  ┌─────────────────────────────────────────────────────────┐
  │  老板：北京今天天气怎么样？                                │
  │  员工（心里想）：老板问天气，我得用天气查询工具。           │
  │  员工（拿起电话）：[调用 get_weather("北京")]              │
  │  工具返回：晴，28°C                                       │
  │  员工：北京今天天气晴，气温 28 摄氏度。                    │
  └─────────────────────────────────────────────────────────┘
  关键区别：Agent 可以主动调用外部函数（工具），LLM 只是决策大脑。

  ┌──────────────────────────────────────────────────────────┐
  │  工具箱（Tools）= 一组可以被 LLM 主动调用的 Python 函数   │
  │  每个工具都有：                                           │
  │    name        → 工具名称（LLM 用这个名字调用工具）        │
  │    description → 工具说明（LLM 靠这个决定"要不要用它"）   │
  │    function    → 实际执行的 Python 函数                   │
  └──────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
【前置科普二：ReAct 框架——大模型是怎么"思考"的？】
═══════════════════════════════════════════════════════════════════

ReAct = Reasoning（推理）+ Acting（行动）

每次 Agent 处理一个问题，内部会进行若干轮"思考-行动-观察"循环：

  ┌─────────────────────────────────────────────────────────┐
  │  Thought（思考）                                         │
  │    LLM 的内心独白："我现在知道了什么，下一步该做什么？"   │
  │    这一步是纯文本推理，不调用任何工具。                   │
  ├─────────────────────────────────────────────────────────┤
  │  Action（行动）                                          │
  │    LLM 决定：调用哪个工具？传什么参数？                   │
  │    例如：Action: get_weather                             │
  │          Action Input: 北京                              │
  ├─────────────────────────────────────────────────────────┤
  │  Observation（观察）                                     │
  │    工具函数被执行，返回结果喂给 LLM。                     │
  │    例如：Observation: 晴，28°C                           │
  └─────────────────────────────────────────────────────────┘

  循环直到 LLM 认为信息足够，输出：
    Final Answer: [最终回答给用户的内容]

  verbose=True 时，AgentExecutor 会把这整个过程打印到控制台，
  让你亲眼看到 LLM 的每一步思考！
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# LangChain Agent 核心组件
# create_structured_chat_agent：支持多参数工具，Action Input 以 JSON blob 传递
from langchain.agents import create_structured_chat_agent, AgentExecutor

# @tool 装饰器：把普通 Python 函数变成 Agent 可以调用的工具
from langchain_core.tools import tool

# Prompt 模板：Structured Chat Agent 使用 ChatPromptTemplate + MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 聊天模型（和前几个项目完全一样）
from langchain_openai import ChatOpenAI


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化 LLM
# 目标：建立与大模型的连接（和项目零/一/二完全一致）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 0 章：初始化 LLM")
print("=" * 60)

# 教学用硬编码；生产环境请改用环境变量：os.environ["OPENAI_API_KEY"]
API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "kivy-kimi-k2_5"

# Agent 场景建议 temperature=0：需要精确推理和格式遵循，不要创意发散
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.0,
)

print("✅ LLM 初始化完成")
print(f"   模型: {MODEL_NAME}")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：用 @tool 装饰器定义工具
# 目标：把普通 Python 函数变成 Agent 可以调用的工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：工具定义（@tool 装饰器）")
print("=" * 60)
print()

# ── Mock 天气数据 ─────────────────────────────────────────
#
# 真实场景：调用 OpenWeatherMap 等天气 API
# 教学场景：用固定字典模拟，省去 API Key 和网络请求

WEATHER_DATA = {
    "北京": {"weather": "晴", "temperature": 28},
    "上海": {"weather": "多云", "temperature": 24},
    "广州": {"weather": "阵雨", "temperature": 32},
    "成都": {"weather": "阴", "temperature": 20},
}


# ── 工具一：天气查询 ──────────────────────────────────────
#
# @tool 装饰器做了三件事：
#   ① 把函数名 → tool.name（Agent 用这个名字调用工具）
#   ② 把函数 docstring → tool.description（⭐ 最重要！）
#   ③ 把函数参数类型注解 → tool.args_schema（LLM 靠这个知道怎么传参数）
#
# ⚠️ 避坑指南：description 写不清楚后果很严重！
#   description 是 LLM 判断"要不要用这个工具"的唯一依据。
#   如果写成"查天气"——太模糊，LLM 可能不知道什么时候该调用它。
#   应该写清楚：① 这个工具做什么 ② 输入格式 ③ 返回什么
#
# ⚠️ 避坑指南：参数必须有类型注解！
#   @tool 依赖类型注解（city: str）生成 args_schema，
#   缺少类型注解时 LLM 不知道应该传什么类型的参数。

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气信息，返回天气状况和温度（摄氏度）。
    输入城市名（中文），例如"北京"、"上海"、"广州"、"成都"。
    当用户询问某个城市的天气、温度、气候时使用此工具。"""
    data = WEATHER_DATA.get(city)
    if data:
        return f"{city}：{data['weather']}，温度 {data['temperature']}°C"
    return f"暂无 {city} 的天气数据，目前支持：{'、'.join(WEATHER_DATA.keys())}"


# ── 工具二：幂次计算 ──────────────────────────────────────
#
# 这个工具做真实计算（不是 mock），结果可以验证。
# 演示"工具可以是任何 Python 函数"——查询、计算、文件操作……

@tool
def calculate_power(base: int, exponent: int) -> str:
    """计算一个整数的幂次方（base 的 exponent 次方）。
    例如：base=2, exponent=10 → 返回 "2 的 10 次方 = 1024"。
    当用户需要精确的数学幂次计算时使用此工具。
    注意：只接受整数输入。"""
    result = base ** exponent
    return f"{base} 的 {exponent} 次方 = {result}"


# ── 打印工具元数据，让你看清楚 @tool 做了什么 ────────────────

tools = [get_weather, calculate_power]

print("【@tool 装饰器为每个工具生成的元数据】")
print()
for t in tools:
    print(f"  工具名称 (name):        {t.name}")
    print(f"  工具描述 (description): {t.description[:60]}...")
    print(f"  参数结构 (args_schema): {t.args_schema.schema()['properties']}")
    print()

print("💡 LLM 接收到的工具信息就是上面这些。")
print("   它靠 description 决定用哪个工具，靠 args_schema 知道怎么传参数。")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：构建 ReAct Agent
# 目标：把 LLM + 工具 + Prompt 组装成一个 Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：构建 ReAct Agent")
print("=" * 60)
print()

# ── Structured Chat Agent Prompt 模板 ─────────────────────
#
# create_structured_chat_agent 需要一个 ChatPromptTemplate，
# 必须包含三个占位符：
#
#   {tools}            → 所有工具的名称 + description + args 拼接文本
#                        AgentExecutor 会自动填入，LLM 靠这个认识工具
#   {tool_names}       → 工具名称列表（逗号分隔）
#   {agent_scratchpad} → MessagesPlaceholder，存放历史推理消息
#
# 与 create_react_agent 的关键区别：
#   create_react_agent       使用 ReActSingleInputOutputParser，
#                            Action Input 是纯文本字符串，只能传单参数。
#   create_structured_chat_agent 使用 JSONAgentOutputParser，
#                            Action Input 是 JSON 对象，天然支持多参数！
#
# ⚠️ 注意：这里使用 ChatPromptTemplate（不是 PromptTemplate）
#   因为 Structured Chat Agent 用消息角色（system/human）区分提示区域，
#   agent_scratchpad 用 MessagesPlaceholder 存放 AIMessage/ToolMessage 列表。

SYSTEM_PROMPT = """你是一个严谨、有条理的 AI 助手。你可以使用以下工具来回答用户的问题：

{tools}

回答问题时，请严格按照以下格式逐行输出：

Question: 你需要回答的问题
Thought: 分析当前情况，决定下一步行动
Action:
```
{{
  "action": "工具名称（必须是 {tool_names} 中的一个）",
  "action_input": 传给工具的参数（单参数直接写值，多参数写 JSON 对象）
}}
```
Observation: （工具返回的结果，由系统填入）
...（以上 Thought/Action/Observation 可以重复多次）
Thought: 我现在已经有足够的信息来回答问题了
Action:
```
{{
  "action": "Final Answer",
  "action_input": "对用户原始问题的完整回答"
}}
```

现在开始！"""

HUMAN_PROMPT = """{input}

{agent_scratchpad}"""

react_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])

print("【Structured Chat Agent Prompt 模板已定义】")
print("   system 模板含占位符：{tools} {tool_names}")
print("   human 模板含占位符：{input} {agent_scratchpad}")
print()

# ── 创建 Agent ─────────────────────────────────────────────
#
# create_structured_chat_agent 把三个组件组装成一个"决策单元"（Agent）：
#   llm    → 负责推理：读取 Thought，决定下一步 Action（JSON blob 格式）
#   tools  → 可调用的工具列表
#   prompt → Structured Chat 格式的提示词模板
#
# 与 create_react_agent 的本质区别：
#   输出解析器从 ReActSingleInputOutputParser（纯文本 Action Input）
#   升级为 JSONAgentOutputParser（JSON blob Action Input）。
#   这使得多参数工具（如 calculate_power）可以正确接收 {"base":2,"exponent":10}。

agent = create_structured_chat_agent(llm, tools, react_prompt)

# ── 创建 AgentExecutor ────────────────────────────────────
#
# AgentExecutor 是 Agent 的"运行引擎"，负责：
#   ① 调用 agent 做推理，解析 Action（JSON blob）和 action_input
#   ② 找到对应的工具函数并执行
#   ③ 把 Observation（工具返回值）传回给 agent 继续推理
#   ④ 循环直到 agent 输出 Final Answer
#
# 关键参数说明：
#   verbose=True         → 打印每一步 Thought/Action/Observation（教学必备！）
#   handle_parsing_errors→ LLM 输出格式不标准时不 crash，让 Agent 自我纠正
#   max_iterations=5     → 最多循环 5 次，防止 Agent 陷入死循环
#
# ⚠️ 避坑指南：handle_parsing_errors=True 一定要加！
#   LLM 偶尔会输出格式不标准的内容（比如 JSON 前多了空格），
#   不加这个参数时会直接抛出 OutputParserException，导致程序崩溃。

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,               # 打印完整 ReAct 循环（Thought/Action/Observation）
    handle_parsing_errors=True, # 格式错误时自我纠正，而不是崩溃
    max_iterations=5,           # 最多循环 5 次，防止死循环
)

print("✅ Agent 构建完成！")
print("   agent_executor 已准备好，调用 agent_executor.invoke({'input': '...'}) 即可运行")
print()
print("💡 小结：三个组件的分工")
print("   create_structured_chat_agent(llm, tools, prompt) → 决策单元（知道怎么想）")
print("   AgentExecutor(agent, tools, verbose=True) → 执行引擎（负责跑循环）")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：运行演示——亲眼看到 ReAct 循环
# 目标：观察 Agent 如何思考、调用工具、得出答案
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：ReAct Agent 演示")
print("=" * 60)
print()
print("【提示】下方输出中：")
print("  > Entering new AgentExecutor chain... → Agent 开始运行")
print("  Thought: ...  → LLM 的推理过程")
print("  Action: ...   → LLM 决定调用哪个工具")
print("  Observation:  → 工具返回的结果")
print("  Final Answer: → Agent 的最终回答")
print()


def run_demo(title: str, question: str) -> None:
    """运行一轮 Agent 演示，打印标题和问题后调用 agent_executor"""
    print("━" * 60)
    print(f"【演示：{title}】")
    print(f"❓ 问题：{question}")
    print("━" * 60)
    result = agent_executor.invoke({"input": question})
    print()
    print(f"✅ 最终答案：{result['output']}")
    print()


# ── 演示一：单工具调用（天气）────────────────────────────
#
# 期望 Agent 行为：
#   Thought: 用户问天气，我需要用 get_weather 工具
#   Action: get_weather
#   Action Input: 北京
#   Observation: 北京：晴，温度 28°C
#   Final Answer: 北京今天天气晴，气温 28 摄氏度。

run_demo("单工具调用（天气查询）", "北京今天天气怎么样？")

# ── 演示二：单工具调用（计算）────────────────────────────
#
# 期望 Agent 行为：
#   Thought: 用户要计算 2 的 10 次方，我需要用 calculate_power 工具
#   Action: calculate_power
#   Action Input: 2, 10  （或 JSON 格式，取决于 LLM）
#   Observation: 2 的 10 次方 = 1024
#   Final Answer: 2 的 10 次方是 1024。

run_demo("单工具调用（幂次计算）", "2 的 10 次方是多少？")

# ── 演示三：多步推理（工具链式调用）─────────────────────
#
# 这是最精彩的演示！Agent 需要：
#   第一步：调用 get_weather("北京") → 获得温度 28
#   第二步：用温度值 28 作为 base，调用 calculate_power(28, 2) → 784
#   Final Answer: 温度是 28°C，28 的 2 次方是 784
#
# 这展示了 ReAct 的核心能力：
#   上一步的 Observation 成为下一步的推理依据！

run_demo(
    "多步推理（链式工具调用）",
    "北京今天的温度是多少度？如果把这个温度值作为底数、2 作为指数，结果是多少？",
)

print("=" * 60)
print("🎉 项目三学习完毕！你已经掌握了 ReAct Agent 的核心机制。")
print("   核心公式：LLM（大脑）+ Tools（工具箱）+ ReAct 循环 = Agent")
print("=" * 60)
