---
title: 函数调用与工具使用
---

<script setup>
const code1 = `# Function Calling 模拟：模型如何决定调用函数
import json

# === 1. 定义工具的 JSON Schema ===
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索知识库中的相关文档",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    }
]

print("=== 已注册的工具 ===")
for tool in tools:
    fn = tool["function"]
    print(f"  - {fn['name']}: {fn['description']}")
print()

# === 2. 模拟模型的函数调用决策 ===
def simulate_model_decision(user_message, tools):
    """模拟模型根据用户输入决定是否调用函数"""
    # 简化的关键词匹配逻辑（真实模型使用深度理解）
    keywords_map = {
        "天气": "get_weather",
        "温度": "get_weather",
        "搜索": "search_knowledge",
        "查找": "search_knowledge",
        "文档": "search_knowledge",
    }

    for keyword, func_name in keywords_map.items():
        if keyword in user_message:
            return func_name, user_message
    return None, None

# === 3. 模拟函数执行 ===
def execute_function(name, arguments):
    """模拟函数的实际执行"""
    mock_results = {
        "get_weather": {"city": "北京", "temp": 22, "condition": "晴", "humidity": 45},
        "search_knowledge": {"results": ["文档1: LLM概述", "文档2: Transformer架构", "文档3: 注意力机制"]}
    }
    return mock_results.get(name, {"error": "未知函数"})

# === 4. 完整流程演示 ===
test_messages = [
    "北京今天天气怎么样？",
    "帮我搜索关于注意力机制的文档",
    "你好，今天过得怎么样？",
]

print("=== Function Calling 决策流程 ===")
for msg in test_messages:
    print(f"\\n用户: {msg}")
    func_name, _ = simulate_model_decision(msg, tools)

    if func_name:
        # 模型决定调用函数
        call = {"name": func_name, "arguments": {"city": "北京"} if "天气" in msg else {"query": "注意力机制"}}
        print(f"  -> 模型决策: 调用 {func_name}")
        print(f"  -> 调用参数: {json.dumps(call['arguments'], ensure_ascii=False)}")

        # 执行函数
        result = execute_function(func_name, call["arguments"])
        print(f"  -> 执行结果: {json.dumps(result, ensure_ascii=False)}")
        print(f"  -> 模型回复: 基于工具结果生成自然语言回答")
    else:
        print(f"  -> 模型决策: 直接回答（无需工具）")
        print(f"  -> 模型回复: 你好！我今天很好，谢谢关心。")
`

const code2 = `# ReAct 循环实现：思考-行动-观察
import json

# === 定义可用工具 ===
class ToolBox:
    """工具箱：管理所有可用工具"""

    def __init__(self):
        self.tools = {}

    def register(self, name, func, description):
        self.tools[name] = {"func": func, "description": description}

    def execute(self, name, **kwargs):
        if name not in self.tools:
            return f"错误: 工具 '{name}' 不存在"
        return self.tools[name]["func"](**kwargs)

    def list_tools(self):
        return {name: info["description"] for name, info in self.tools.items()}

# === 模拟工具实现 ===
def calculator(expression):
    """安全计算器"""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return {"result": eval(expression)}
        return {"error": "不安全的表达式"}
    except Exception as e:
        return {"error": str(e)}

def search(query):
    """模拟搜索引擎"""
    mock_db = {
        "地球半径": "地球平均半径约为 6371 公里",
        "光速": "光速约为 299,792,458 米/秒",
        "圆周率": "圆周率 π ≈ 3.14159265358979",
    }
    for key, value in mock_db.items():
        if key in query:
            return {"result": value}
    return {"result": f"未找到关于 '{query}' 的信息"}

def unit_convert(value, from_unit, to_unit):
    """单位换算"""
    conversions = {
        ("km", "m"): 1000,
        ("m", "km"): 0.001,
        ("kg", "g"): 1000,
        ("g", "kg"): 0.001,
    }
    factor = conversions.get((from_unit, to_unit))
    if factor:
        return {"result": f"{value} {from_unit} = {value * factor} {to_unit}"}
    return {"error": f"不支持 {from_unit} 到 {to_unit} 的转换"}

# === 初始化工具箱 ===
toolbox = ToolBox()
toolbox.register("calculator", calculator, "数学计算，参数: expression")
toolbox.register("search", search, "搜索知识库，参数: query")
toolbox.register("unit_convert", unit_convert, "单位换算，参数: value, from_unit, to_unit")

print("=== 可用工具 ===")
for name, desc in toolbox.list_tools().items():
    print(f"  - {name}: {desc}")
print()

# === ReAct 循环 ===
def react_loop(question, toolbox, max_steps=5):
    """
    ReAct (Reasoning + Acting) 循环
    每一步: Thought -> Action -> Observation
    """
    print(f"问题: {question}")
    print("=" * 50)

    # 预定义的推理步骤（模拟模型的思考过程）
    # 真实场景中这些由 LLM 生成
    reasoning_plans = {
        "地球周长是多少公里？": [
            {
                "thought": "用户问地球周长。我需要先查找地球半径，然后用公式 C = 2πr 计算。",
                "action": "search",
                "action_input": {"query": "地球半径"}
            },
            {
                "thought": "地球半径是 6371 公里。圆周长公式 C = 2 * π * r = 2 * 3.14159 * 6371",
                "action": "calculator",
                "action_input": {"expression": "2 * 3.14159 * 6371"}
            },
            {
                "thought": "计算得到约 40030 公里。我现在可以回答用户了。",
                "action": "FINISH",
                "action_input": {"answer": "地球周长约为 40,030 公里（通过 2πr 计算，其中 r = 6371 km）。"}
            }
        ]
    }

    steps = reasoning_plans.get(question, [])
    context = []

    for i, step in enumerate(steps, 1):
        print(f"\\n--- 步骤 {i} ---")

        # Thought（思考）
        print(f"💭 Thought: {step['thought']}")

        # Action（行动）
        action = step["action"]
        action_input = step["action_input"]

        if action == "FINISH":
            print(f"✅ Final Answer: {action_input['answer']}")
            return action_input["answer"]

        print(f"🔧 Action: {action}({json.dumps(action_input, ensure_ascii=False)})")

        # Observation（观察）
        observation = toolbox.execute(action, **action_input)
        print(f"👁 Observation: {json.dumps(observation, ensure_ascii=False)}")

        context.append({
            "step": i,
            "thought": step["thought"],
            "action": action,
            "observation": observation
        })

    return "推理未完成"

# === 执行演示 ===
print("\\n" + "=" * 50)
print("ReAct 循环演示")
print("=" * 50)
result = react_loop("地球周长是多少公里？", toolbox)

print("\\n\\n" + "=" * 50)
print("ReAct 模式总结")
print("=" * 50)
print("""
循环结构:
  1. Thought - 模型思考当前状态和下一步计划
  2. Action  - 选择并调用合适的工具
  3. Observation - 获取工具返回的结果
  4. 重复直到能给出最终答案 (FINISH)

关键优势:
  - 可解释性: 每步推理过程透明可见
  - 可靠性: 通过工具获取准确信息而非幻觉
  - 灵活性: 可根据中间结果动态调整策略
""")
`
</script>

# 函数调用与工具使用（Function Calling / Tool Use）

Function Calling 是让 LLM 从"纯文本生成器"进化为"能力执行者"的关键技术。通过工具调用，模型可以访问实时数据、执行计算、操作外部系统。

## 1. Function Calling 核心原理

### 模型如何感知工具？

LLM 本身只能生成文本，但通过特殊的提示格式，模型可以输出**结构化的函数调用指令**，由外部系统解析执行。

```
用户消息 + 工具定义 → LLM 推理 → 输出函数调用 JSON → 执行 → 结果回传 → 生成最终回答
```

::: info 核心洞察
Function Calling 的本质是**约束解码**：模型被训练为在特定上下文下，输出符合 JSON Schema 的结构化文本，而非自由文本。
:::

### 决策流程

| 阶段 | 描述 | 输出 |
|------|------|------|
| 1. 感知 | 模型读取用户意图和可用工具列表 | 理解任务需求 |
| 2. 匹配 | 判断是否需要工具，选择最合适的工具 | 工具名称 |
| 3. 参数构建 | 从上下文中提取参数并填充 JSON | 结构化参数 |
| 4. 执行 | 外部系统调用实际 API | 原始结果 |
| 5. 整合 | 模型基于结果生成自然语言回答 | 最终回复 |

## 2. JSON Schema 定义工具

工具通过 JSON Schema 描述其**名称、功能和参数结构**，模型根据这些信息决定调用方式。

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取指定城市的当前天气信息",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "城市名称"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"]
        }
      },
      "required": ["city"]
    }
  }
}
```

::: tip 工具定义最佳实践
- **description** 要清晰具体，这是模型判断是否使用该工具的关键依据
- **参数描述**不可省略，模型依赖它来正确提取参数
- 使用 `enum` 约束参数取值范围，减少错误调用
- `required` 字段确保关键参数不被遗漏
:::

### 交互式演示：Function Calling 决策过程

<PythonRunner :code="code1" />

## 3. Agent 核心原理

### 感知-推理-行动循环

Agent（智能体）是 Function Calling 的高级形态。它不仅调用单个工具，还能**多步推理、动态规划**。

```
         ┌─────────────────────────────┐
         │        Agent Loop           │
         │                             │
         │   ┌─────────┐              │
         │   │ Thought │ ← 推理当前状态│
         │   └────┬────┘              │
         │        ↓                    │
         │   ┌─────────┐              │
         │   │ Action  │ ← 选择工具    │
         │   └────┬────┘              │
         │        ↓                    │
         │   ┌─────────────┐          │
         │   │ Observation │ ← 获取结果│
         │   └──────┬──────┘          │
         │          ↓                  │
         │    完成？─── 否 → 回到 Thought│
         │     │                       │
         │     是                       │
         │     ↓                       │
         │  最终回答                    │
         └─────────────────────────────┘
```

### ReAct 框架

ReAct（Reasoning + Acting）是最经典的 Agent 范式：

| 组件 | 作用 | 示例 |
|------|------|------|
| Thought | 分析当前状态，规划下一步 | "我需要先查找地球半径..." |
| Action | 选择并调用工具 | `search(query="地球半径")` |
| Observation | 接收工具返回结果 | "地球半径约 6371 公里" |
| Final Answer | 综合所有信息给出答案 | "地球周长约 40030 公里" |

::: info 为什么需要 Thought？
没有显式思考步骤时，模型容易"冲动行动"——调用错误的工具或遗漏步骤。Thought 强制模型先推理再行动，显著提升任务成功率。
:::

### 交互式演示：ReAct 循环

<PythonRunner :code="code2" />

## 4. 主流实现对比

| 特性 | OpenAI | Claude (Anthropic) | 开源模型 |
|------|--------|-------------------|----------|
| 调用格式 | `tool_calls` 字段 | `tool_use` content block | 各异（多为文本格式） |
| 并行调用 | 支持（多个 tool_calls） | 支持（多个 tool_use blocks） | 部分支持 |
| 强制调用 | `tool_choice: required` | `tool_choice: any` | 通常不支持 |
| 流式输出 | 支持 | 支持 | 部分支持 |
| 嵌套调用 | 支持多轮 | 支持多轮 | 需自行实现 |
| Schema 严格性 | 高（strict mode） | 高 | 中等 |

### OpenAI 格式

```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京天气如何？"}],
    tools=tools,  # JSON Schema 定义
    tool_choice="auto"
)
```

### Claude 格式

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "北京天气如何？"}],
    tools=tools,  # 类似但格式略有不同
)
# 结果在 response.content 中作为 tool_use block
```

### 开源模型（如 Qwen、GLM）

```python
# 通常通过特殊 token 标记工具调用
# <|tool_call|>{"name": "get_weather", "arguments": {"city": "北京"}}<|/tool_call|>
```

## 5. 实践：工具链设计与错误处理

### 工具链设计原则

| 原则 | 说明 |
|------|------|
| 单一职责 | 每个工具只做一件事，避免"万能工具" |
| 明确边界 | 工具之间职责不重叠，减少模型选择困难 |
| 优雅降级 | 工具失败时提供有意义的错误信息 |
| 幂等设计 | 同样的调用多次执行结果一致 |
| 超时控制 | 设置合理超时，避免阻塞整个对话 |

### 错误处理策略

```python
# 推荐的错误处理模式
def safe_tool_call(tool_fn, **kwargs):
    try:
        result = tool_fn(**kwargs)
        return {"status": "success", "data": result}
    except ValidationError as e:
        return {"status": "error", "type": "invalid_params", "message": str(e)}
    except TimeoutError:
        return {"status": "error", "type": "timeout", "message": "工具调用超时"}
    except Exception as e:
        return {"status": "error", "type": "unknown", "message": str(e)}
```

::: tip 错误处理最佳实践
1. **结构化错误返回**：让模型能理解错误类型并决定重试或换策略
2. **重试机制**：对于网络类错误，允许有限次重试
3. **回退方案**：工具不可用时，模型应能给出近似回答
4. **日志记录**：记录所有工具调用的输入输出，便于调试
:::

### 常见陷阱

| 陷阱 | 问题 | 解决方案 |
|------|------|----------|
| 工具描述模糊 | 模型频繁误调用 | 加入正例/反例说明 |
| 参数缺乏约束 | 产生无效参数 | 使用 enum、pattern 等约束 |
| 无限循环 | Agent 反复调用同一工具 | 设置最大步数限制 |
| 结果过长 | 超出上下文窗口 | 截断或摘要处理 |
| 权限泄露 | 工具执行危险操作 | 实施严格的权限控制和沙箱 |

## 总结

| 概念 | 要点 |
|------|------|
| Function Calling | 模型输出结构化 JSON 调用外部函数 |
| JSON Schema | 描述工具的接口契约 |
| ReAct Agent | 思考-行动-观察的多步推理循环 |
| 工具链设计 | 单一职责、明确边界、优雅降级 |
| 错误处理 | 结构化返回、重试机制、回退方案 |

Function Calling 让 LLM 从"知识库"变为"执行者"，而 Agent 范式让它从"单次调用"升级为"自主规划"。掌握这些技术是构建实用 AI 应用的关键基础。
