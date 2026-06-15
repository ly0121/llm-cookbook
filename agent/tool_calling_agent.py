"""
╔══════════════════════════════════════════════════════════════════╗
║         项目七：Tool Calling Agent（函数调用 Agent）                ║
║         告别纯文本 ReAct，拥抱结构化 Function Calling             ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普一：ReAct Agent 的致命缺陷——为什么要升级？】
═══════════════════════════════════════════════════════════════════

回顾项目三的 create_react_agent，它的工具调用方式：

  老版 ReAct（纯文本协议）：
  ┌─────────────────────────────────────────────────────────────┐
  │  LLM 输出一段纯文本：                                        │
  │    Thought: 我需要查天气                                     │
  │    Action: get_weather                                       │
  │    Action Input: 北京                                        │
  │                                                             │
  │  然后 LangChain 用正则表达式从文本中解析出：                  │
  │    工具名 = "get_weather"                                    │
  │    参数   = "北京"（注意：这是一个纯字符串！）                │
  └─────────────────────────────────────────────────────────────┘

  问题：
    ① Action Input 只能是一个字符串 → 多参数必须自己解析"2,10"
    ② 靠正则解析 → LLM 格式稍微偏一点就 crash
    ③ 无法传复杂类型 → 没法传列表、字典、嵌套对象
    ④ 不稳定 → handle_parsing_errors=True 是在"擦屁股"

═══════════════════════════════════════════════════════════════════
【前置科普二：Tool Calling（函数调用）——OpenAI 原生能力】
═══════════════════════════════════════════════════════════════════

2023年6月，OpenAI 推出了 Function Calling（后改名 Tool Calling）：

  新版 Tool Calling（结构化协议）：
  ┌─────────────────────────────────────────────────────────────┐
  │  LLM 不再输出纯文本，而是输出一个结构化的 JSON 对象：         │
  │  {                                                          │
  │    "tool_calls": [{                                         │
  │      "function": {                                          │
  │        "name": "get_weather",                               │
  │        "arguments": {"city": "北京", "unit": "celsius"}     │
  │      }                                                      │
  │    }]                                                       │
  │  }                                                          │
  │                                                             │
  │  参数是 JSON 对象，天然支持多参数、嵌套类型、列表等！        │
  └─────────────────────────────────────────────────────────────┘

  对比：
  ┌───────────────────┬───────────────────────────────────────┐
  │  ReAct 纯文本      │  Tool Calling 结构化                  │
  ├───────────────────┼───────────────────────────────────────┤
  │  Action Input: 北京│  {"city": "北京", "unit": "celsius"}  │
  │  单字符串参数      │  多参数 JSON 对象                     │
  │  正则解析（脆弱）  │  JSON 解析（稳定）                    │
  │  LLM 容易格式错   │  LLM 原生支持（几乎不出错）           │
  │  需手动写 Prompt   │  Schema 自动注入                      │
  └───────────────────┴───────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
【前置科普三：create_tool_calling_agent vs create_react_agent】
═══════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────┐
  │  create_react_agent（项目三，已过时）                          │
  │    • 使用 PromptTemplate（纯文本）                           │
  │    • LLM 输出纯文本 → 正则解析 Action/Action Input           │
  │    • 单字符串参数，多参数要自己拼"逗号分隔"                  │
  │    • 需要手写 ReAct 格式的详细 Prompt                        │
  ├──────────────────────────────────────────────────────────────┤
  │  create_tool_calling_agent（本项目，生产推荐）                │
  │    • 使用 ChatPromptTemplate（消息模板）                     │
  │    • LLM 原生输出 tool_calls JSON → 直接解析                 │
  │    • 多参数自然支持（每个参数是 JSON 的一个 key）             │
  │    • Prompt 只需定义角色，不需要写 ReAct 格式说明            │
  │    • 内置 MessagesPlaceholder("agent_scratchpad")            │
  └──────────────────────────────────────────────────────────────┘

  一句话总结：
    ReAct Agent  = 让 LLM "写作文"来调用工具（容易写错格式）
    Tool Calling = 让 LLM "填表格"来调用工具（格式天然正确）
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# create_tool_calling_agent：生产级 Agent 构建器（替代 create_react_agent）
# AgentExecutor：Agent 执行引擎（和项目三一样，但搭配不同 Agent）
from langchain.agents import create_tool_calling_agent, AgentExecutor

# @tool 装饰器：把 Python 函数变成 Agent 可调用的工具
from langchain_core.tools import tool

# ChatPromptTemplate + MessagesPlaceholder
# 注意：Tool Calling Agent 使用 ChatPromptTemplate，不是 PromptTemplate！
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Pydantic：用于定义工具参数的结构化 Schema
from pydantic import BaseModel, Field

# 聊天模型
from langchain_openai import ChatOpenAI

# 类型注解
from typing import Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化 LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 0 章：初始化 LLM")
print("=" * 60)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY, BASE_URL, MODEL_NAME
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
# 第 1 章：定义多参数工具（对比项目三的单字符串参数）
# 目标：演示 Tool Calling 如何天然支持多参数、复杂类型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：定义多参数工具")
print("=" * 60)
print()

# ── 核心对比：项目三 vs 本项目的参数传递方式 ───────────────
#
# 项目三（ReAct）：
#   @tool
#   def calculate_power(base_and_exponent: str) -> str:
#       """..."""
#       parts = base_and_exponent.split(",")  ← 自己手动解析字符串！
#       base = int(parts[0])
#       exponent = int(parts[1])
#
# 本项目（Tool Calling）：
#   @tool
#   def calculate_power(base: int, exponent: int) -> str:
#       """..."""
#       result = base ** exponent  ← 直接拿到正确类型的多个参数！
#
# 这就是 Tool Calling 的核心优势：LLM 原生输出 JSON 参数，
# LangChain 自动解析并传入函数，无需手动字符串拆分！


# ── 工具一：天气查询（多参数 + 可选参数）──────────────────

WEATHER_DATA = {
    "北京": {"weather": "晴", "temp_c": 28, "humidity": 45},
    "上海": {"weather": "多云", "temp_c": 24, "humidity": 72},
    "广州": {"weather": "阵雨", "temp_c": 32, "humidity": 85},
    "成都": {"weather": "阴", "temp_c": 20, "humidity": 60},
    "东京": {"weather": "晴", "temp_c": 26, "humidity": 55},
}


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """查询指定城市的实时天气信息。

    Args:
        city: 城市名称（中文或英文），例如"北京"、"上海"、"东京"
        unit: 温度单位，"celsius"（摄氏度）或"fahrenheit"（华氏度），默认摄氏度
    """
    # ⚠️ 注意看：city 和 unit 是两个独立参数！
    # Tool Calling 模式下，LLM 会直接传入 {"city": "北京", "unit": "celsius"}
    # 不需要像项目三那样从 "北京,celsius" 字符串里自己拆分！

    data = WEATHER_DATA.get(city)
    if not data:
        return f"暂无 {city} 的天气数据，支持的城市：{'、'.join(WEATHER_DATA.keys())}"

    temp = data["temp_c"]
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32, 1)
        unit_label = "°F"
    else:
        unit_label = "°C"

    return (
        f"{city}：{data['weather']}，温度 {temp}{unit_label}，湿度 {data['humidity']}%"
    )


# ── 工具二：幂次计算（多参数，直接传 int）─────────────────
#
# 对比项目三的 calculate_power(base_and_exponent: str)：
#   项目三：LLM 传 "2,10" → 函数自己 split 解析
#   本项目：LLM 传 {"base": 2, "exponent": 10} → 直接拿到 int


@tool
def calculate_power(base: int, exponent: int) -> str:
    """计算幂次方（base 的 exponent 次方）。

    Args:
        base: 底数（整数）
        exponent: 指数（整数）
    """
    result = base**exponent
    return f"{base} 的 {exponent} 次方 = {result}"


# ── 工具三：多城市对比（演示列表参数）───────────────────
#
# 这是 ReAct Agent 完全做不到的！
# 列表参数在 Tool Calling 中天然支持。


@tool
def compare_cities_weather(cities: list[str]) -> str:
    """对比多个城市的天气信息，返回表格式对比结果。

    Args:
        cities: 需要对比的城市名称列表，例如 ["北京", "上海", "广州"]
    """
    results = []
    results.append("| 城市 | 天气 | 温度 | 湿度 |")
    results.append("|------|------|------|------|")
    for city in cities:
        data = WEATHER_DATA.get(city)
        if data:
            results.append(
                f"| {city} | {data['weather']} | {data['temp_c']}°C | {data['humidity']}% |"
            )
        else:
            results.append(f"| {city} | 暂无数据 | - | - |")
    return "\n".join(results)


# ── 工具四：行程规划（演示复杂多参数）────────────────────


@tool
def plan_trip(
    origin: str,
    destination: str,
    days: int,
    budget: Optional[int] = None,
) -> str:
    """根据出发地、目的地、天数和预算，生成简要行程建议。

    Args:
        origin: 出发城市
        destination: 目的地城市
        days: 旅行天数
        budget: 预算金额（人民币元），可选
    """
    budget_info = f"，预算 {budget} 元" if budget else ""
    return (
        f"行程建议：从 {origin} 出发前往 {destination}，"
        f"共 {days} 天{budget_info}。\n"
        f"建议安排：\n"
        f"  Day 1: 抵达{destination}，市区游览\n"
        f"  Day 2-{max(2, days - 1)}: 景点深度游\n"
        f"  Day {days}: 返回{origin}"
    )


# ── 打印工具元数据对比 ─────────────────────────────────────

tools = [get_weather, calculate_power, compare_cities_weather, plan_trip]

print("【工具列表及参数结构（Tool Calling 模式）】")
print()
for t in tools:
    print(f"  📌 {t.name}")
    print(f"     描述: {t.description[:60]}...")
    # 获取参数 Schema
    schema = t.args_schema.model_json_schema() if t.args_schema else {}
    props = schema.get("properties", {})
    required = schema.get("required", [])
    for param_name, param_info in props.items():
        param_type = param_info.get("type", param_info.get("anyOf", "unknown"))
        is_required = "必填" if param_name in required else "可选"
        print(f"     • {param_name}: {param_type} ({is_required})")
    print()

print("💡 对比项目三：每个工具都有明确的多参数定义，")
print("   LLM 通过 JSON Schema 知道每个参数的名称、类型、是否必填。")
print("   不再需要 '2,10' 这种字符串拼接 hack！")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：构建 Tool Calling Agent
# 目标：用 create_tool_calling_agent 替代 create_react_agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：构建 Tool Calling Agent")
print("=" * 60)
print()

# ── Prompt 模板的关键区别 ──────────────────────────────────
#
# 项目三 ReAct Agent 的 Prompt：
#   需要手写一大段 ReAct 格式说明（Thought/Action/Observation...）
#   使用 PromptTemplate（纯文本）
#   必须包含 {tools}、{tool_names}、{agent_scratchpad}
#
# 本项目 Tool Calling Agent 的 Prompt：
#   只需要定义角色和任务，不需要写格式说明！
#   使用 ChatPromptTemplate（消息格式）
#   必须包含 MessagesPlaceholder("agent_scratchpad")
#   不需要 {tools}、{tool_names}（Schema 自动通过 function calling 注入）
#
# ⚠️ 避坑指南：
#   MessagesPlaceholder(variable_name="agent_scratchpad") 是必须的！
#   AgentExecutor 用这个位置存放"工具调用和返回"的中间消息。
#   如果没有这个占位符，Agent 循环会报错。

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个智能旅行和生活助手。你可以：
1. 查询城市天气（支持指定温度单位）
2. 进行数学幂次计算
3. 对比多个城市的天气
4. 规划旅行行程

请根据用户的问题，选择合适的工具来回答。
如果用户的问题不需要调用工具，直接用你的知识回答即可。
回答请使用中文。""",
        ),
        # MessagesPlaceholder：Agent 中间推理过程的存放位置
        # AgentExecutor 每轮循环会把：
        #   ① LLM 的 tool_calls 消息
        #   ② 工具返回的 ToolMessage
        # 都追加到这个位置，让 LLM 在下一轮能"看到"之前的工具结果
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("human", "{input}"),
    ]
)

print("【Prompt 模板结构】")
print("  [system]  → 角色定义 + 能力说明")
print("  [agent_scratchpad] → Agent 中间推理（自动管理）")
print("  [human]   → 用户输入 {input}")
print()
print("  💡 对比项目三：不需要写 ReAct 格式说明（Thought/Action...），")
print("     因为 Tool Calling 是 LLM 的原生能力，不需要 Prompt 引导格式。")
print()

# ── 创建 Tool Calling Agent ──────────────────────────────
#
# create_tool_calling_agent 内部做了什么？
#   ① 把工具列表的 Schema 绑定到 LLM（通过 llm.bind_tools(tools)）
#   ② 创建一个 Agent Runnable，输出为 AgentAction 或 AgentFinish
#   ③ Agent 解析 LLM 的 tool_calls 字段（JSON），不是从纯文本解析
#
# 对比 create_react_agent：
#   create_react_agent → 输出纯文本 → 用正则解析 Action/Action Input
#   create_tool_calling_agent → 输出 tool_calls JSON → 直接解析

agent = create_tool_calling_agent(llm, tools, prompt)

# ── 创建 AgentExecutor ───────────────────────────────────
#
# 和项目三一样，AgentExecutor 负责运行"推理 → 执行 → 推理"循环。
# 但底层的"推理"方式不同：
#   项目三：解析纯文本（脆弱）
#   本项目：解析 JSON tool_calls（稳定）

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 打印完整推理过程
    handle_parsing_errors=True,  # 格式错误时自我纠正
    max_iterations=8,  # 最多循环次数
)

print("✅ Tool Calling Agent 构建完成！")
print()
print("  对比项目三 create_react_agent：")
print("  ┌────────────────────┬──────────────────────────────┐")
print("  │  项目三             │  本项目                      │")
print("  ├────────────────────┼──────────────────────────────┤")
print("  │  PromptTemplate    │  ChatPromptTemplate          │")
print("  │  需写 ReAct 格式   │  只定义角色，格式自动        │")
print("  │  {tools}{tool_names}│  不需要，Schema 自动注入     │")
print("  │  正则解析输出       │  JSON 原生解析               │")
print("  │  单字符串参数       │  多参数/复杂类型             │")
print("  └────────────────────┴──────────────────────────────┘")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：运行演示——观察 Tool Calling 的推理过程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：运行演示")
print("=" * 60)
print()
print("【提示】verbose=True 下的输出格式：")
print("  > Entering new AgentExecutor chain...")
print("  Invoking: `tool_name` with `{JSON参数}`")
print("  → 注意：参数是 JSON 对象，不是纯文本字符串！")
print()


def run_demo(title: str, question: str) -> str:
    """运行一轮 Agent 演示"""
    print("━" * 60)
    print(f"【演示：{title}】")
    print(f"❓ 问题：{question}")
    print("━" * 60)
    result = agent_executor.invoke({"input": question})
    print()
    print(f"✅ 最终答案：{result['output']}")
    print()
    return result["output"]


# ── 演示一：多参数工具调用 ─────────────────────────────────
#
# 期望：LLM 输出 tool_calls: [{"function": {"name": "get_weather",
#        "arguments": {"city": "东京", "unit": "fahrenheit"}}}]
# 对比项目三：Action Input 只能传一个字符串"东京"

run_demo(
    "多参数工具调用（指定温度单位）",
    "东京现在多少华氏度？",
)

# ── 演示二：原生多参数（不需要逗号拼接）────────────────────
#
# 期望：LLM 传 {"base": 3, "exponent": 15}
# 项目三只能传 "3,15" 然后自己 split

run_demo(
    "多参数计算（直接传 int，无需字符串拼接）",
    "3 的 15 次方等于多少？",
)

# ── 演示三：列表参数（ReAct 做不到！）─────────────────────
#
# 期望：LLM 传 {"cities": ["北京", "上海", "广州"]}
# 这在 ReAct 纯文本模式下几乎不可能稳定实现

run_demo(
    "列表参数（对比多城市天气）",
    "帮我对比一下北京、上海和广州的天气",
)

# ── 演示四：复杂多参数 + 可选参数 ─────────────────────────
#
# 期望：LLM 传 {"origin": "北京", "destination": "成都",
#              "days": 5, "budget": 8000}
#
# ⚠️ 避坑指南：LLM 可能重复调用工具！
#   如果工具返回的结果过于简单（比如只有几句话），
#   LLM 可能认为"信息不够"而再次调用同一个工具。
#   这是 LLM 推理的不确定性，解决方案：
#     ① 让工具返回更详细的结果
#     ② 在 Prompt 中加入"每个工具最多调用一次"的指令
#     ③ max_iterations 是安全网，防止无限循环

run_demo(
    "复杂多参数 + 可选参数（行程规划）",
    "我想从北京去成都玩5天，预算8000元，帮我规划一下",
)

# ── 演示五：多步推理 + 多工具串联 ─────────────────────────
#
# 期望 Agent 行为：
#   第一步：调用 get_weather("成都") → 获得温度 20°C
#   第二步：调用 calculate_power(20, 3) → 8000
#   第三步：综合回答

run_demo(
    "多步推理（串联调用多个工具）",
    "成都今天的温度是多少度？把这个温度的3次方算出来。",
)

# ── 演示六：无需工具的直接回答 ────────────────────────────
#
# 期望：Agent 判断不需要工具，直接用 LLM 知识回答
# 这验证了 Agent 不会"过度调用工具"

run_demo(
    "不需要工具的问题（直接回答）",
    "Python 的 GIL 是什么？一句话解释。",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：用 Pydantic 定义工具参数 Schema（高级用法）
# 目标：为工具参数添加更丰富的描述和验证
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 4 章：Pydantic args_schema（高级工具定义）")
print("=" * 60)
print()

# ── 为什么需要 args_schema？──────────────────────────────
#
# 用 @tool 装饰器时，LangChain 会从函数签名自动推断参数类型。
# 但有时候自动推断不够精确，你想要：
#   ① 给每个参数加详细描述（帮助 LLM 理解参数含义）
#   ② 限制参数取值范围（如 enum）
#   ③ 添加示例值
#
# 方法：定义一个 Pydantic 模型作为 args_schema，
# 用 Field(description=...) 给每个参数加说明。
#
# 和项目六的 Structured Output 呼应：
# 项目六用 Pydantic 定义"LLM 输出的结构"
# 本章用 Pydantic 定义"LLM 传给工具的参数结构"
# 底层原理完全一样：都是把 JSON Schema 发给 LLM！


class CurrencyConvertInput(BaseModel):
    """货币转换工具的参数定义"""

    amount: float = Field(description="要转换的金额数值")
    from_currency: str = Field(
        description="源货币代码，如 CNY（人民币）、USD（美元）、JPY（日元）、EUR（欧元）"
    )
    to_currency: str = Field(
        description="目标货币代码，如 CNY（人民币）、USD（美元）、JPY（日元）、EUR（欧元）"
    )


# 模拟汇率数据
EXCHANGE_RATES = {
    ("CNY", "USD"): 0.14,
    ("USD", "CNY"): 7.24,
    ("CNY", "JPY"): 21.5,
    ("JPY", "CNY"): 0.047,
    ("CNY", "EUR"): 0.13,
    ("EUR", "CNY"): 7.85,
    ("USD", "JPY"): 155.0,
    ("JPY", "USD"): 0.0065,
    ("USD", "EUR"): 0.92,
    ("EUR", "USD"): 1.09,
}


@tool(args_schema=CurrencyConvertInput)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """将一种货币转换为另一种货币，支持 CNY/USD/JPY/EUR。"""
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    if from_currency == to_currency:
        return f"{amount} {from_currency} = {amount} {to_currency}（同币种无需转换）"

    rate = EXCHANGE_RATES.get((from_currency, to_currency))
    if rate is None:
        return f"暂不支持 {from_currency} → {to_currency} 的转换"

    converted = round(amount * rate, 2)
    return f"{amount} {from_currency} = {converted} {to_currency}（汇率: 1 {from_currency} = {rate} {to_currency}）"


# 打印这个工具的详细 Schema
print("【使用 args_schema 定义的工具参数】")
print()
print(f"  工具名: {convert_currency.name}")
print(f"  描述: {convert_currency.description}")
print()
schema = convert_currency.args_schema.model_json_schema()
print("  参数 Schema（发送给 LLM 的）：")
for param, info in schema.get("properties", {}).items():
    print(f"    • {param}:")
    print(f"        type: {info.get('type', 'unknown')}")
    print(f"        description: {info.get('description', '无')}")
print()
print("  💡 args_schema 让 LLM 看到更丰富的参数描述，")
print("     从而更准确地理解每个参数应该填什么值。")
print()

# ── 用新工具重新构建 Agent 并演示 ─────────────────────────

all_tools = tools + [convert_currency]

agent2 = create_tool_calling_agent(llm, all_tools, prompt)
agent_executor2 = AgentExecutor(
    agent=agent2,
    tools=all_tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=8,
)

print("━" * 60)
print("【演示：Pydantic args_schema 工具（货币转换）】")
print("━" * 60)

result = agent_executor2.invoke(
    {"input": "1000 人民币等于多少日元？另外 500 美元等于多少欧元？"}
)
print()
print(f"✅ 最终答案：{result['output']}")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 章：直接绑定工具到 LLM（不经过 AgentExecutor）
# 目标：展示 Tool Calling 的底层机制
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 5 章：底层机制——bind_tools 与手动调用")
print("=" * 60)
print()

# ── AgentExecutor 是可选的！──────────────────────────────
#
# AgentExecutor 只是一个"便利包装"，帮你做了循环调用。
# 如果你想更细粒度地控制工具调用过程，可以：
#   ① llm.bind_tools(tools) → 绑定工具
#   ② 调用 LLM → 检查返回中是否有 tool_calls
#   ③ 如果有，手动执行工具函数
#   ④ 把工具结果喂回 LLM
#
# 这在以下场景很有用：
#   - 你想在工具调用前做人工审批
#   - 你想记录每一步的详细日志
#   - 你想自定义错误处理逻辑

# 绑定工具（不经过 AgentExecutor）
llm_with_tools = llm.bind_tools([get_weather, calculate_power])

print("【底层调用：llm.bind_tools() 直接看 LLM 的原始输出】")
print()

# 直接调用 LLM
response = llm_with_tools.invoke("上海天气怎么样？")

print("  📨 LLM 原始返回的 AIMessage：")
print(f"     content: {response.content!r}")
print(f"     tool_calls: {response.tool_calls}")
print()

if response.tool_calls:
    print("  📋 解析 tool_calls 结构：")
    for i, tc in enumerate(response.tool_calls):
        print(f"     [{i}] name: {tc['name']!r}")
        print(f"         args: {tc['args']}")
        print(f"         id:   {tc['id']!r}")
    print()
    print("  💡 观察：")
    print("     • content 为空（LLM 选择调用工具而非直接回复）")
    print("     • tool_calls 包含工具名和 JSON 参数")
    print("     • args 是 dict，不是字符串！无需手动解析")
    print()

    # 手动执行工具
    tool_call = response.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # 找到对应的工具函数并执行
    tool_map = {t.name: t for t in [get_weather, calculate_power]}
    tool_result = tool_map[tool_name].invoke(tool_args)

    print(f"  🔧 手动执行工具：{tool_name}({tool_args})")
    print(f"     返回结果：{tool_result}")
    print()
    print("  💡 这就是 AgentExecutor 内部循环的核心逻辑：")
    print("     检查 tool_calls → 执行工具 → 把结果喂回 LLM → 循环")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 总结
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("🎉 项目七学习完毕！")
print("=" * 60)
print()
print("💡 核心升级（项目三 → 项目七）：")
print()
print("  ┌────────────────────┬──────────────────────────────────┐")
print("  │  项目三 ReAct       │  项目七 Tool Calling             │")
print("  ├────────────────────┼──────────────────────────────────┤")
print("  │  纯文本协议         │  JSON 结构化协议                 │")
print("  │  正则解析（脆弱）   │  原生 JSON 解析（稳定）          │")
print("  │  单字符串参数       │  多参数 + 复杂类型               │")
print("  │  手写 ReAct Prompt  │  只定义角色，格式自动            │")
print("  │  PromptTemplate    │  ChatPromptTemplate              │")
print("  │  容易格式出错       │  几乎不出错                      │")
print("  │  不支持列表参数     │  list/dict/Optional 全支持       │")
print("  └────────────────────┴──────────────────────────────────┘")
print()
print("💡 生产环境建议：")
print("   ① 永远优先用 create_tool_calling_agent（只要 LLM 支持）")
print("   ② 用 Pydantic args_schema 给参数加详细描述")
print("   ③ 需要精细控制时，用 bind_tools + 手动循环")
print("   ④ create_react_agent 只在 LLM 不支持 function calling 时使用")
print("=" * 60)
