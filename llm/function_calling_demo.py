"""
╔══════════════════════════════════════════════════════════════════╗
║         项目：Function Calling / Tool Use 完整实战教程           ║
║         探索模型如何感知工具、调用函数、构建 Agent 循环            ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题：LLM 怎么"使用工具"？它真的会执行代码吗？】
═══════════════════════════════════════════════════════════════════

答案是：LLM 本身不会执行任何代码！它只是"告诉你它想调用什么函数"。

整个 Function Calling 的流程如下：

  用户提问 → [模型推理] → 模型输出 tool_calls（函数名 + 参数）
                                     ↓
                          开发者在本地执行对应函数
                                     ↓
                          把函数执行结果返回给模型
                                     ↓
                          模型根据结果生成最终回答

  ┌─────────────────────────────────────────────────────────────┐
  │  关键认知：                                                   │
  │                                                               │
  │  1. 模型不执行函数，它只"决定"要调用哪个函数、传什么参数       │
  │  2. 函数的定义通过 JSON Schema 告诉模型（像给模型一本说明书）  │
  │  3. 模型的输出是结构化的 tool_calls，而非自然语言             │
  │  4. 开发者负责真正执行函数，并把结果喂回模型                   │
  │  5. 这就是 AI Agent 的核心机制——思考+行动的循环               │
  └─────────────────────────────────────────────────────────────┘

  形象比喻：
    模型 = 一个聪明但没有手的"大脑"
    工具 = 大脑可以指挥的"手"
    开发者 = 连接大脑和手的"神经系统"

本文件通过真实 API 调用，带你从零掌握 Function Calling 的全部流程。
"""

import json

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：Function Calling 概念总览
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import client, MODEL_NAME

print("=" * 60)
print("第 0 章：Function Calling 概念总览")
print("=" * 60)
print()
print("""
┌──────────────────────────────────────────────────────────────┐
│           Function Calling 完整交互流程                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: 开发者定义工具（JSON Schema）                        │
│    → 告诉模型"你有哪些工具可以用，每个工具需要什么参数"       │
│                                                              │
│  Step 2: 用户发送消息                                         │
│    → "北京今天天气怎么样？"                                   │
│                                                              │
│  Step 3: 模型决定是否调用工具                                 │
│    → 如果需要外部信息，模型输出 tool_calls                    │
│    → 如果不需要，模型直接回答                                 │
│                                                              │
│  Step 4: 开发者执行函数                                       │
│    → 解析 tool_calls，调用本地函数 get_weather("北京")        │
│                                                              │
│  Step 5: 把结果返回模型                                       │
│    → role="tool" 的消息，包含函数执行结果                     │
│                                                              │
│  Step 6: 模型生成最终回答                                     │
│    → "北京今天晴天，气温 25 度，适合出行。"                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘

与普通对话的区别：
  普通对话：  用户 → 模型 → 回答（一步到位）
  工具调用：  用户 → 模型 → tool_calls → 执行 → 结果 → 模型 → 回答
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：定义工具（JSON Schema）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 工具定义的结构：
#   每个工具是一个字典，包含：
#     - type: 固定为 "function"
#     - function: 具体函数描述
#       - name: 函数名（模型会用这个名字来"调用"）
#       - description: 函数功能描述（非常重要！模型靠这个判断何时使用）
#       - parameters: JSON Schema 格式的参数定义
#
#   ┌────────────────────────────────────────────────────────┐
#   │  description 写得好不好，直接影响模型能否正确使用工具！  │
#   │                                                        │
#   │  好的 description：                                     │
#   │    "查询指定城市的实时天气，包括温度、湿度、天气状况"    │
#   │                                                        │
#   │  差的 description：                                     │
#   │    "天气函数"                                           │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 1 章：定义工具（JSON Schema）")
print("=" * 60)
print()

# ── 定义工具 1：天气查询函数 ─────────────────────────────────
# 这是一个模拟函数，在真实场景中你会调用天气 API
def get_weather(city: str, unit: str = "celsius") -> str:
    """
    模拟天气查询函数。
    在真实项目中，这里会调用天气 API（如 OpenWeatherMap）。
    """
    # 模拟数据
    weather_data = {
        "北京": {"temp": 25, "humidity": 40, "condition": "晴天"},
        "上海": {"temp": 28, "humidity": 75, "condition": "多云"},
        "广州": {"temp": 32, "humidity": 85, "condition": "雷阵雨"},
        "深圳": {"temp": 30, "humidity": 80, "condition": "阴天"},
    }

    if city in weather_data:
        data = weather_data[city]
        temp = data["temp"]
        if unit == "fahrenheit":
            temp = temp * 9 / 5 + 32
            unit_str = "华氏度"
        else:
            unit_str = "摄氏度"
        return json.dumps({
            "city": city,
            "temperature": f"{temp}{unit_str}",
            "humidity": f"{data['humidity']}%",
            "condition": data["condition"],
        }, ensure_ascii=False)
    else:
        return json.dumps({"error": f"暂不支持查询{city}的天气"}, ensure_ascii=False)


# ── 定义工具 2：计算器函数 ───────────────────────────────────
# 模型本身做数学计算容易出错，计算器工具可以保证精确
def calculator(expression: str) -> str:
    """
    安全计算数学表达式。
    支持基本运算：加减乘除、乘方、取余等。
    """
    try:
        # 使用 eval 计算（生产环境建议用更安全的方式如 sympy）
        # 限制只允许数学运算，防止代码注入
        allowed_chars = set("0123456789+-*/.() %")
        if not all(c in allowed_chars for c in expression.replace("**", "")):
            return json.dumps({"error": "表达式包含非法字符"}, ensure_ascii=False)
        result = eval(expression)
        return json.dumps({
            "expression": expression,
            "result": result,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"计算失败: {str(e)}"}, ensure_ascii=False)


# ── 用 JSON Schema 定义工具的"说明书"，提供给模型 ─────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气信息，包括温度、湿度、天气状况。当用户询问天气相关问题时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "要查询天气的城市名称，例如：北京、上海、广州",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，celsius 为摄氏度，fahrenheit 为华氏度，默认摄氏度",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式的结果。当用户需要进行数学计算时使用此工具。支持加减乘除、乘方、取余等运算。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如：'2 + 3 * 4' 或 '(100 - 20) / 4'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]

print("已定义 2 个工具：")
print()
for tool in tools:
    func = tool["function"]
    print(f"  工具名: {func['name']}")
    print(f"  描述:   {func['description']}")
    params = func["parameters"]["properties"]
    param_names = list(params.keys())
    print(f"  参数:   {param_names}")
    print()

print("这些工具定义会通过 tools 参数传给 API，模型就'知道'自己有哪些工具可用了。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：单轮 Function Calling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 完整的单轮函数调用流程：
#
#   ┌─────────┐      ┌─────────┐      ┌─────────┐
#   │ 用户提问 │ ───→ │ 模型推理 │ ───→ │tool_calls│
#   └─────────┘      └─────────┘      └────┬────┘
#                                           │
#                                           ↓
#   ┌─────────┐      ┌─────────┐      ┌─────────┐
#   │ 最终回答 │ ←─── │ 模型总结 │ ←─── │ 执行函数 │
#   └─────────┘      └─────────┘      └─────────┘
#
# 注意：模型的 finish_reason 会是 "tool_calls"（而非 "stop"）
# 这告诉我们模型想要调用函数，而不是直接回答。

print("=" * 60)
print("第 2 章：单轮 Function Calling")
print("=" * 60)
print()

# ── 2.1 发送消息，让模型决定是否调用工具 ─────────────────────
print("── 2.1 发送用户问题，观察模型是否调用工具 ──────────────")
print()

messages = [
    {"role": "system", "content": "你是一个有用的助手，可以查询天气和进行数学计算。"},
    {"role": "user", "content": "北京今天天气怎么样？"},
]

print(f"  用户问题: {messages[-1]['content']}")
print()

# 第一次调用：模型决定是否需要工具
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto 表示让模型自己决定是否调用工具
)

assistant_message = response.choices[0].message
print(f"  模型 finish_reason: {response.choices[0].finish_reason}")
print(f"  模型是否调用了工具: {assistant_message.tool_calls is not None}")
print()

# ── 2.2 解析 tool_calls 返回 ────────────────────────────────
print("── 2.2 解析模型返回的 tool_calls ──────────────────────")
print()

if assistant_message.tool_calls:
    for tool_call in assistant_message.tool_calls:
        print(f"  工具调用 ID:  {tool_call.id}")
        print(f"  函数名称:     {tool_call.function.name}")
        print(f"  函数参数:     {tool_call.function.arguments}")
        print()

# ── 2.3 执行本地函数 ─────────────────────────────────────────
print("── 2.3 在本地执行对应的函数 ──────────────────────────")
print()

# 建立函数名到实际函数的映射（函数调度表）
available_functions = {
    "get_weather": get_weather,
    "calculator": calculator,
}

# 把模型的回复加入消息历史（重要！不能跳过这一步）
messages.append(assistant_message)

# 执行每个工具调用
if assistant_message.tool_calls:
    for tool_call in assistant_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        print(f"  正在执行: {function_name}({function_args})")

        # 调用实际函数
        function_response = available_functions[function_name](**function_args)
        print(f"  执行结果: {function_response}")
        print()

        # 把函数结果作为 tool 消息加入对话历史
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,  # 必须与 tool_call 的 id 对应
            "content": function_response,
        })

# ── 2.4 把结果返回给模型，获得最终回答 ───────────────────────
print("── 2.4 将函数结果返回模型，获得最终自然语言回答 ──────")
print()

# 第二次调用：模型根据函数结果生成自然语言回答
final_response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    tools=tools,
)

final_answer = final_response.choices[0].message.content
print(f"  模型最终回答: {final_answer}")
print()
print("  观察: 模型把结构化的天气数据转换成了人类友好的自然语言！")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：多轮对话中的工具调用
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 在多轮对话中，模型可以：
#   1. 根据上下文判断何时需要调用工具
#   2. 在同一轮中调用多个工具（parallel function calling）
#   3. 有时直接回答，有时调用工具——自主决策
#
#   对话 1: "你好" → 模型直接回答（不需要工具）
#   对话 2: "北京天气" → 模型调用 get_weather
#   对话 3: "那上海呢？" → 模型调用 get_weather（理解了上下文）
#   对话 4: "两地温差是多少？" → 模型调用 calculator

print("=" * 60)
print("第 3 章：多轮对话中的工具调用")
print("=" * 60)
print()

# 定义一个辅助函数，处理单轮对话（包含可能的工具调用）
def chat_with_tools(messages: list, user_input: str) -> str:
    """
    处理一轮对话，自动处理工具调用。
    返回模型的最终文本回答。
    """
    # 添加用户消息
    messages.append({"role": "user", "content": user_input})

    # 调用模型
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    assistant_msg = response.choices[0].message

    # 如果模型想调用工具
    if assistant_msg.tool_calls:
        # 将 assistant 消息加入历史
        messages.append(assistant_msg)

        # 执行所有工具调用
        for tool_call in assistant_msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"    [工具调用] {func_name}({func_args})")

            # 执行函数
            result = available_functions[func_name](**func_args)
            print(f"    [执行结果] {result}")

            # 把结果加入消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # 再次调用模型，让它根据工具结果生成回答
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
        )
        assistant_msg = response.choices[0].message

    # 将最终回复加入历史
    messages.append(assistant_msg)
    return assistant_msg.content


# ── 开始多轮对话演示 ─────────────────────────────────────────
multi_turn_messages = [
    {"role": "system", "content": "你是一个有用的助手，可以查询天气和进行数学计算。回答简洁明了。"},
]

# 多轮对话的问题序列
conversation_turns = [
    "你好，你能做什么？",                    # 不需要工具
    "帮我查一下北京的天气",                  # 需要 get_weather
    "那上海呢？",                            # 需要 get_weather（理解上下文）
    "帮我算一下 (25 + 28) / 2 等于多少？",   # 需要 calculator
]

for i, user_input in enumerate(conversation_turns, 1):
    print(f"── 第 {i} 轮对话 {'─' * 45}")
    print(f"  用户: {user_input}")
    answer = chat_with_tools(multi_turn_messages, user_input)
    print(f"  助手: {answer}")
    print()

print("观察要点：")
print("  - 第1轮：模型直接回答，不调用任何工具")
print("  - 第2轮：模型调用 get_weather 查询北京天气")
print("  - 第3轮：模型理解'那上海呢'指的是天气，自动调用 get_weather")
print("  - 第4轮：模型调用 calculator 进行精确计算")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：简单 ReAct Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ReAct = Reasoning + Acting（推理 + 行动）
#
# Agent 的核心循环：
#
#   ┌──────────────────────────────────────────────────────┐
#   │                                                      │
#   │   用户问题                                            │
#   │      ↓                                               │
#   │   ┌─────────────────────────────────────┐            │
#   │   │  模型思考：我需要什么信息？          │            │
#   │   │  → 决定调用工具                     │ ←─── 循环  │
#   │   └─────────────┬───────────────────────┘     │      │
#   │                 ↓                              │      │
#   │   ┌─────────────────────────────────────┐     │      │
#   │   │  执行工具，获得观察结果              │     │      │
#   │   └─────────────┬───────────────────────┘     │      │
#   │                 ↓                              │      │
#   │   ┌─────────────────────────────────────┐     │      │
#   │   │  模型判断：信息足够了吗？            │ ────┘      │
#   │   │  → 不够：继续调用工具               │            │
#   │   │  → 够了：生成最终回答               │            │
#   │   └─────────────────────────────────────┘            │
#   │                                                      │
#   └──────────────────────────────────────────────────────┘
#
# 与简单的 Function Calling 的区别：
#   - Function Calling：模型调用一次工具就结束
#   - ReAct Agent：模型可以多次调用工具，直到获得足够信息
#   - Agent 有一个显式的"循环"，不断思考和行动

print("=" * 60)
print("第 4 章：简单 ReAct Agent")
print("=" * 60)
print()


def react_agent(user_question: str, max_iterations: int = 5) -> str:
    """
    简单的 ReAct Agent 实现。

    核心逻辑：
      1. 接收用户问题
      2. 进入循环：模型思考 → 调用工具 → 观察结果
      3. 当模型不再调用工具时，返回最终答案
      4. 设置最大迭代次数防止无限循环

    参数：
        user_question: 用户的问题
        max_iterations: 最大循环次数（防止死循环）

    返回：
        模型的最终回答
    """
    # 初始化消息列表
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个智能助手，拥有天气查询和数学计算工具。"
                "请一步一步思考用户的问题，必要时调用工具获取信息。"
                "如果一个问题需要多步操作（例如先查询再计算），请逐步完成。"
            ),
        },
        {"role": "user", "content": user_question},
    ]

    print(f"  用户问题: {user_question}")
    print(f"  最大迭代: {max_iterations} 次")
    print()

    for iteration in range(1, max_iterations + 1):
        print(f"  ── 迭代 {iteration} {'─' * 40}")

        # 调用模型
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        assistant_msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # 如果模型不调用工具，说明它已经准备好回答了
        if not assistant_msg.tool_calls:
            print(f"  模型决定：信息足够，生成最终回答")
            print(f"  最终回答: {assistant_msg.content}")
            messages.append(assistant_msg)
            return assistant_msg.content

        # 模型想调用工具——执行所有工具调用
        messages.append(assistant_msg)

        for tool_call in assistant_msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"  [思考→行动] 调用 {func_name}({func_args})")

            # 执行函数
            result = available_functions[func_name](**func_args)
            print(f"  [观察结果]  {result}")

            # 把结果加入消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        print()

    # 如果达到最大迭代次数，强制获取最终回答
    print(f"  已达最大迭代次数 {max_iterations}，强制生成回答...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    final_answer = response.choices[0].message.content
    print(f"  最终回答: {final_answer}")
    return final_answer


# ── 4.1 简单问题：只需一次工具调用 ───────────────────────────
print("── 4.1 简单问题（单次工具调用）──────────────────────")
print()
react_agent("广州现在的天气如何？")
print()

# ── 4.2 复合问题：需要多次工具调用 ───────────────────────────
print("── 4.2 复合问题（多次工具调用）──────────────────────")
print()
react_agent("帮我查一下北京和上海的天气，然后计算它们的平均温度。")
print()

# ── 4.3 不需要工具的问题 ────────────────────────────────────
print("── 4.3 不需要工具的问题 ────────────────────────────")
print()
react_agent("请用一句话解释什么是人工智能。")
print()


# ── 总结 ────────────────────────────────────────────────────
print("=" * 60)
print("全部实验完毕！核心总结：")
print("=" * 60)
print("""
  ┌────────────────────────────────────────────────────────────┐
  │  概念             │ 要点                                    │
  ├────────────────────────────────────────────────────────────┤
  │  工具定义         │ 用 JSON Schema 描述函数名、描述、参数    │
  │  tool_choice      │ auto=模型自主决定, required=强制调用     │
  │  tool_calls       │ 模型输出的函数调用请求（名称+参数）      │
  │  role="tool"      │ 函数结果回传给模型的消息角色             │
  │  tool_call_id     │ 关联请求和结果的唯一标识                 │
  │  ReAct Agent      │ 思考→行动→观察的循环，直到问题解决       │
  └────────────────────────────────────────────────────────────┘

  最佳实践：
  1. description 写清楚——模型靠它判断何时使用工具
  2. 参数用 required 标注必填项，用 enum 限制取值范围
  3. 始终把 assistant 的 tool_calls 消息加入历史（不能跳过）
  4. tool_call_id 必须正确对应，否则 API 会报错
  5. Agent 循环要设置 max_iterations 防止无限调用
  6. 生产环境中函数执行要做异常处理和超时控制
""")
