# Agent（智能体）完全知识手册

> 本文档是一份系统性的 AI Agent 技术教科书，从基础概念到多 Agent 系统，覆盖智能体的所有核心知识点。
> 配合 `react_agent.py` 和 `tool_calling_agent.py` 代码阅读效果更佳。

---

## 目录

1. [Agent 的定义与核心思想](#1-agent-的定义与核心思想)
2. [Agent 架构模式](#2-agent-架构模式)
3. [ReAct 框架详解](#3-react-框架详解)
4. [Tool Calling / Function Calling 机制](#4-tool-calling--function-calling-机制)
5. [工具定义与描述的最佳实践](#5-工具定义与描述的最佳实践)
6. [AgentExecutor 运行时机制](#6-agentexecutor-运行时机制)
7. [多步推理与工具链调用](#7-多步推理与工具链调用)
8. [Agent 记忆与状态管理](#8-agent-记忆与状态管理)
9. [多 Agent 协作模式](#9-多-agent-协作模式)
10. [Agent 可靠性与错误处理](#10-agent-可靠性与错误处理)
11. [Agent 评估方法](#11-agent-评估方法)
12. [从 Agent 到 Agentic Systems 的演进](#12-从-agent-到-agentic-systems-的演进)

---

## 1. Agent 的定义与核心思想

### 1.1 什么是 Agent

**Agent（智能体）** 是一个能够感知环境、进行推理、并采取行动的自主系统。在 LLM 时代，Agent = LLM（大脑）+ Tools（工具箱）+ 循环控制（执行引擎）。

```
核心本质：Agent 是一个"会用工具的 LLM"

普通 LLM：
  用户提问 → LLM 回答（只能用训练时的知识）

Agent：
  用户提问 → LLM 思考 → 调用工具 → 观察结果 → 再思考 → ... → 最终回答

关键区别：Agent 可以主动获取外部信息、执行外部操作！
```

### 1.2 感知-推理-行动循环（Perception-Reasoning-Action Loop）

```
              ┌─────────────────────────────────────────┐
              │         Agent 核心循环                    │
              │                                         │
              │   ┌──────────┐                          │
              │   │ 感知      │ ← 接收用户输入/工具结果  │
              │   │ Perceive │                          │
              │   └────┬─────┘                          │
              │        ↓                                │
              │   ┌──────────┐                          │
              │   │ 推理      │ ← LLM 分析当前状况      │
              │   │ Reason   │   决定下一步行动         │
              │   └────┬─────┘                          │
              │        ↓                                │
              │   ┌──────────┐                          │
              │   │ 行动      │ ← 调用工具/输出答案     │
              │   │ Act      │                          │
              │   └────┬─────┘                          │
              │        │                                │
              │        └──── 循环直到任务完成 ──────────→│
              └─────────────────────────────────────────┘
```

### 1.3 Agent vs 普通 LLM 对话

| 维度 | 普通 LLM | Agent |
|------|----------|-------|
| 知识 | 静态（训练时截止） | 动态（可实时查询） |
| 能力 | 纯文本生成 | 调用任意工具 |
| 推理 | 单次回答 | 多步迭代推理 |
| 自主性 | 被动问答 | 主动规划执行 |
| 可靠性 | 可能幻觉 | 可验证（工具返回真实数据） |

---

## 2. Agent 架构模式

### 2.1 主流架构对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent 架构模式全景                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ① ReAct（本项目实现）                                          │
│     Thought → Action → Observation → ... → Final Answer        │
│     特点：推理和行动交替进行，每步都有思考                       │
│                                                                 │
│  ② Plan-and-Execute                                            │
│     先制定完整计划 → 逐步执行 → 根据结果调整计划               │
│     特点：先全局规划，再逐步落地                                │
│                                                                 │
│  ③ REWOO (Reasoning Without Observation)                       │
│     一次性规划所有步骤 → 批量执行 → 最终综合                   │
│     特点：减少 LLM 调用次数，降低成本                           │
│                                                                 │
│  ④ Tool Calling（本项目实现）                                   │
│     LLM 原生输出 JSON tool_calls → 执行 → 反馈 → 继续         │
│     特点：结构化、稳定、生产首选                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 ReAct vs Plan-and-Execute

```
ReAct（边想边做）：
  "我先查查天气... 哦是28度... 那我再算个平方..."
  每一步看到结果后才决定下一步。

Plan-and-Execute（先规划后执行）：
  "我的计划：1.查天气 2.取温度值 3.计算平方"
  先制定完整计划，然后逐步执行。

  ┌────────┐     ┌─────────────────────────────┐
  │ Planner│────→│ Step1 → Step2 → Step3 → ... │
  │ (规划) │     │        Executor (执行)       │
  └────┬───┘     └─────────────────────────────┘
       ↑                      │
       └──── 执行结果反馈 ─────┘  (可选：根据结果修正计划)

适用场景对比：
  ReAct：简单任务、步骤未知、需要灵活应变
  Plan-and-Execute：复杂任务、步骤可预见、需要全局一致性
```

### 2.3 REWOO 架构

```
REWOO 的核心优势：减少 LLM 调用次数

传统 ReAct（每步都调用 LLM）：
  LLM调用1 → 工具1 → LLM调用2 → 工具2 → LLM调用3 → 答案
  共 3 次 LLM 调用

REWOO（一次规划，批量执行）：
  LLM调用1(规划所有步骤) → 工具1 → 工具2 → LLM调用2(综合) → 答案
  共 2 次 LLM 调用

  成本可降低 30-50%，但灵活性不如 ReAct
```

---

## 3. ReAct 框架详解

### 3.1 ReAct = Reasoning + Acting

ReAct 的核心创新：让 LLM 在"推理"和"行动"之间交替进行。

```
对应 react_agent.py 中的实现：

  Thought（推理）：LLM 的内心独白
    "用户问北京天气，我需要用 get_weather 工具"

  Action（行动）：选择并调用工具
    "Action: get_weather"
    "Action Input: 北京"

  Observation（观察）：工具返回的结果
    "Observation: 北京：晴，温度 28°C"

  ... 循环 ...

  Final Answer（最终答案）：
    "北京今天天气晴，气温28摄氏度。"
```

### 3.2 ReAct Prompt 模板结构

```
对应 react_agent.py 第2章的 REACT_PROMPT_TEMPLATE：

必须包含四个占位符：
  {tools}            → 工具列表描述（自动填入）
  {tool_names}       → 工具名列表（如 get_weather, calculate_power）
  {input}            → 用户的问题
  {agent_scratchpad} → 历史推理（Thought/Action/Observation 累积）

格式要求极其严格：
  Question: ...
  Thought: ...
  Action: 工具名（必须在列表中）
  Action Input: 参数（纯文本字符串）
  Observation: ...（系统填入）
  Final Answer: ...
```

### 3.3 ReAct 的局限性

```
对应 tool_calling_agent.py 前置科普：

  ① Action Input 只能是单字符串 → 多参数需手动解析 "2,10"
  ② 依赖正则表达式解析 LLM 输出 → 格式稍偏就 crash
  ③ 无法传递复杂类型 → 列表、字典、嵌套对象都不行
  ④ 需要 handle_parsing_errors=True 兜底

结论：ReAct 适合教学和不支持 Function Calling 的模型，
      生产环境应使用 Tool Calling Agent。
```

---

## 4. Tool Calling / Function Calling 机制

### 4.1 协议原理

```
对应 tool_calling_agent.py 前置科普二：

Tool Calling 是 LLM 的原生能力（2023年6月 OpenAI 推出）：

  开发者定义工具的 JSON Schema → 发送给 LLM →
  LLM 判断需要调用工具时，输出结构化 JSON：

  {
    "tool_calls": [{
      "id": "call_abc123",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"北京\", \"unit\": \"celsius\"}"
      }
    }]
  }

  关键：arguments 是 JSON 字符串，不是纯文本！
       支持多参数、嵌套类型、可选参数。
```

### 4.2 JSON Schema 工具描述

```
LLM 接收到的工具定义格式（OpenAI 协议）：

{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "查询指定城市的实时天气信息",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "城市名称，如北京、上海"
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
}

LLM 根据 description 决定何时调用，根据 parameters 决定传什么参数。
```

### 4.3 ReAct vs Tool Calling 对比

| 维度 | ReAct 纯文本 | Tool Calling 结构化 |
|------|-------------|-------------------|
| 参数格式 | 单字符串 "北京" | JSON {"city":"北京"} |
| 解析方式 | 正则表达式（脆弱） | JSON 解析（稳定） |
| 多参数 | 手动拼接 "2,10" | 天然支持多个 key |
| 复杂类型 | 不支持 | list/dict/Optional |
| Prompt | 需写 ReAct 格式说明 | 只定义角色即可 |
| 可靠性 | 容易格式出错 | 几乎不出错 |

---

## 5. 工具定义与描述的最佳实践

### 5.1 @tool 装饰器的三个作用

```
对应 react_agent.py 第1章：

@tool 装饰器为函数生成三要素：
  ① tool.name        ← 函数名（Agent 用这个名字调用）
  ② tool.description ← 函数 docstring（最重要！）
  ③ tool.args_schema ← 函数参数类型注解

示例：
  @tool
  def get_weather(city: str) -> str:
      """查询指定城市的实时天气信息，返回天气状况和温度。
      输入城市名（中文），例如"北京"、"上海"。
      当用户询问某个城市的天气、温度时使用此工具。"""
```

### 5.2 Description 写作指南

```
description 是 LLM 判断"要不要用这个工具"的唯一依据！

❌ 差的描述：
  "查天气"  → 太模糊

✅ 好的描述应包含三要素：
  ① 功能：这个工具做什么
  ② 输入：接受什么格式的参数
  ③ 场景：什么情况下应该使用

示例：
  "查询指定城市的实时天气信息，返回天气状况和温度（摄氏度）。
   输入城市名（中文），例如'北京'、'上海'。
   当用户询问某个城市的天气、温度、气候时使用此工具。"
```

### 5.3 Pydantic args_schema（高级用法）

```
对应 tool_calling_agent.py 第4章：

用 Pydantic BaseModel 为参数添加更丰富的描述：

class CurrencyConvertInput(BaseModel):
    amount: float = Field(description="要转换的金额数值")
    from_currency: str = Field(
        description="源货币代码，如 CNY、USD、JPY"
    )
    to_currency: str = Field(
        description="目标货币代码，如 CNY、USD、JPY"
    )

@tool(args_schema=CurrencyConvertInput)
def convert_currency(amount, from_currency, to_currency) -> str:
    ...

效果：LLM 看到更精确的参数描述，调用更准确。
```

---

## 6. AgentExecutor 运行时机制

### 6.1 AgentExecutor 的职责

```
对应 react_agent.py 第2章：

AgentExecutor 是 Agent 的"运行引擎"，负责循环：

  ┌──────────────────────────────────────────────────────┐
  │                AgentExecutor 循环                     │
  │                                                      │
  │  ① 调用 Agent（LLM）做推理                           │
  │       ↓                                              │
  │  ② 解析输出：是 AgentAction 还是 AgentFinish？       │
  │       ↓                                              │
  │  ③ 如果是 Action → 执行工具 → 把结果传回 ① 继续     │
  │     如果是 Finish → 返回 Final Answer，循环结束      │
  │                                                      │
  │  安全机制：                                           │
  │    max_iterations=5   → 防止死循环                   │
  │    handle_parsing_errors=True → 格式错误时自我纠正   │
  └──────────────────────────────────────────────────────┘
```

### 6.2 关键参数

```
agent_executor = AgentExecutor(
    agent=agent,              # 决策单元（Agent Runnable）
    tools=tools,              # 可调用的工具列表
    verbose=True,             # 打印推理过程（教学/调试必备）
    handle_parsing_errors=True,  # LLM 输出格式错误时不 crash
    max_iterations=5,         # 最多循环次数（安全网）
    return_intermediate_steps=False,  # 是否返回中间步骤
)
```

---

## 7. 多步推理与工具链调用

### 7.1 链式调用示例

```
对应 react_agent.py 第3章演示三：

问题："北京今天温度多少？把温度值作为底数、2为指数，结果是？"

Agent 推理过程：
  Thought: 先查北京温度
  Action: get_weather
  Action Input: 北京
  Observation: 北京：晴，温度 28°C

  Thought: 温度是28，现在计算 28 的 2 次方
  Action: calculate_power
  Action Input: 28,2
  Observation: 28 的 2 次方 = 784

  Thought: 信息足够了
  Final Answer: 北京今天温度28°C，28的2次方是784。

核心能力：上一步的 Observation 成为下一步的推理依据！
```

### 7.2 并行工具调用

```
Tool Calling 模式下，LLM 可以一次输出多个 tool_calls：

{
  "tool_calls": [
    {"function": {"name": "get_weather", "arguments": {"city": "北京"}}},
    {"function": {"name": "get_weather", "arguments": {"city": "上海"}}}
  ]
}

AgentExecutor 会并行执行这些工具调用，然后把所有结果一起传回 LLM。
这在"对比多城市天气"等场景中自动发生。
```

---

## 8. Agent 记忆与状态管理

### 8.1 短期记忆（Scratchpad）

```
Agent 的 scratchpad 就是短期记忆：

  ReAct 模式：agent_scratchpad 累积所有历史 Thought/Action/Observation
  Tool Calling 模式：MessagesPlaceholder 存放 tool_calls + ToolMessage

每轮循环，LLM 都能"看到"之前所有的推理过程和工具结果。
但这些记忆只在单次任务执行期间存在。
```

### 8.2 长期记忆

```
跨任务的记忆需要外部存储：

  ① 对话历史（ConversationBufferMemory）
     存储所有历史消息，下次对话时传入

  ② 摘要记忆（ConversationSummaryMemory）
     LLM 自动总结历史对话，减少 token 占用

  ③ 向量记忆（VectorStoreRetrieverMemory）
     将历史片段向量化，按相关性检索

  ④ 检查点记忆（LangGraph Checkpoint）
     保存整个图的状态，支持断点续传
     → 详见 langgraph_advanced/KNOWLEDGE.md
```

---

## 9. 多 Agent 协作模式

### 9.1 协作模式分类

```
┌─────────────────────────────────────────────────────────────┐
│              多 Agent 协作模式                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ① 辩论模式（Debate）                                       │
│     Agent A 提出观点 → Agent B 反驳 → 多轮辩论 → 综合结论   │
│     适用：需要多角度分析的决策问题                            │
│                                                             │
│  ② 分工模式（Division of Labor）                            │
│     研究员→ 写手 → 主编（流水线）                            │
│     适用：任务可明确拆分为子任务                             │
│     → 对应 langgraph/media_studio.py 的实现                 │
│                                                             │
│  ③ 层级模式（Hierarchical）                                 │
│     Manager Agent 分配任务给 Worker Agents                  │
│     Manager 负责规划和协调，Worker 负责执行                  │
│     适用：复杂项目管理、大规模任务分解                       │
│                                                             │
│  ④ 投票模式（Voting / Ensemble）                            │
│     多个 Agent 独立解决同一问题 → 多数投票选最优             │
│     适用：需要高可靠性的推理任务                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 多 Agent 通信机制

```
Agent 之间如何传递信息？

  ① 共享状态（Shared State）
     所有 Agent 读写同一个 State 对象
     → LangGraph 的 TypedDict 方式

  ② 消息传递（Message Passing）
     Agent A 发消息给 Agent B
     → 基于消息队列或直接函数调用

  ③ 黑板模式（Blackboard）
     所有 Agent 往"黑板"上写信息，其他 Agent 自取
     适合松耦合的异步协作
```

---

## 10. Agent 可靠性与错误处理

### 10.1 常见错误类型

```
① 输出格式错误
   LLM 输出不符合预期格式（如 ReAct 格式偏差）
   解决：handle_parsing_errors=True + 重试

② 工具调用错误
   传入错误参数、调用不存在的工具
   解决：工具函数内部 try-except + 友好错误提示

③ 死循环
   Agent 反复调用同一工具但得不到想要的结果
   解决：max_iterations 限制 + 循环检测

④ 幻觉
   Agent 编造工具不存在的功能或参数
   解决：严格的工具 description + Schema 校验

⑤ 工具超时
   外部 API 响应慢或不可用
   解决：设置超时 + 降级策略
```

### 10.2 防御式设计模式

```
@tool
def get_weather(city: str) -> str:
    """..."""
    try:
        data = WEATHER_DATA.get(city)
        if data:
            return f"{city}：{data['weather']}，温度 {data['temperature']}°C"
        # 返回友好错误信息而非抛异常——让 Agent 可以自我纠正
        return f"暂无 {city} 的天气数据，支持：{'、'.join(WEATHER_DATA.keys())}"
    except Exception as e:
        return f"查询失败：{str(e)}，请稍后重试"
```

---

## 11. Agent 评估方法

### 11.1 评估维度

```
┌─────────────────────────────────────────────────────────┐
│  维度          │ 指标                    │ 说明          │
├─────────────────────────────────────────────────────────┤
│ 任务完成率     │ 成功完成任务的比例       │ 最核心指标    │
│ 步骤效率       │ 完成任务的平均步骤数     │ 越少越好      │
│ 工具选择准确率 │ 选对工具的比例           │ 工具多时重要  │
│ 参数正确率     │ 工具参数正确的比例       │ Schema 设计   │
│ 延迟           │ 端到端响应时间           │ 用户体验      │
│ 成本           │ Token 消耗 / API 调用数  │ 生产关键      │
│ 鲁棒性         │ 错误恢复能力             │ 稳定性        │
└─────────────────────────────────────────────────────────┘
```

### 11.2 评估方法

```
① 基准测试（Benchmark）
   - 预定义问题集 + 标准答案
   - 自动计算通过率

② 轨迹评估（Trajectory Evaluation）
   - 检查 Agent 的每一步推理和行动是否合理
   - 不仅看结果，还看过程

③ 人类评估（Human Evaluation）
   - 人工判断回答质量
   - 适合开放式任务

④ LLM-as-Judge
   - 用另一个 LLM 评估 Agent 输出
   - 成本低、可规模化
```

---

## 12. 从 Agent 到 Agentic Systems 的演进

### 12.1 发展阶段

```
阶段1: 单工具 Agent（2023初）
  LLM + 1个工具，简单问答
  → react_agent.py 的 get_weather 示例

阶段2: 多工具 Agent（2023中）
  LLM + 多个工具 + 工具选择
  → tool_calling_agent.py 的多工具示例

阶段3: 多 Agent 系统（2024）
  多个 Agent 协作完成复杂任务
  → langgraph/media_studio.py（研究员+写手+主编）

阶段4: Agentic Systems（2025+）
  自主规划 + 自我反思 + 长期记忆 + 动态工具创建
  Agent 不仅使用工具，还能创造工具
```

### 12.2 Agentic Systems 的特征

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  传统 Agent          │  Agentic System                      │
│  ─────────────       │  ──────────────                      │
│  固定工具集          │  动态发现/创建工具                    │
│  单次任务执行        │  长期运行（天/周级）                  │
│  无自我反思          │  执行后自我评估、改进策略              │
│  无学习能力          │  从历史经验中学习                     │
│  人类完全控制        │  高自主性 + 关键节点人类审批          │
│                      │  → langgraph_advanced 的 HITL 模式   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 附录 A：本文件夹代码与知识点对应

| 代码文件 | 覆盖的知识点 | 对应本文档章节 |
|---------|-------------|---------------|
| `react_agent.py` | ReAct 框架、@tool 装饰器、AgentExecutor、多步推理 | 第3、5、6、7节 |
| `tool_calling_agent.py` | Tool Calling 协议、多参数工具、Pydantic Schema、bind_tools | 第4、5节 |

---

## 附录 B：推荐学习路径

```
入门（1-2天）：
  第1-3节 → 理解 Agent 概念和 ReAct 框架
  运行 react_agent.py → 观察 Thought/Action/Observation 循环

进阶（2-3天）：
  第4-7节 → 掌握 Tool Calling 和多步推理
  运行 tool_calling_agent.py → 对比两种模式差异

高级（1周）：
  第8-12节 → 理解多 Agent 系统和生产实践
  阅读 langgraph/media_studio.py → 体验多 Agent 协作
```

---

> **下一步学习**：前往 `langgraph/KNOWLEDGE.md` 了解如何用图结构编排多 Agent 工作流，然后阅读 `langgraph_advanced/KNOWLEDGE.md` 掌握人机协作和检查点机制。
