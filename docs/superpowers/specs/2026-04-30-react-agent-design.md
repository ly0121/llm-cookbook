# 项目三：基于 ReAct 框架的单体 Agent — 设计文档

**日期：** 2026-04-30
**项目：** 项目三：ReAct Agent（老板、员工、工具箱）
**目标读者：** LLM 开发新手

---

## 项目目标

用 LangChain 实现一个基于 ReAct 框架的单体 Agent：
通过 `@tool` 装饰器定义 mock 工具（天气查询 + 幂次计算），
用 `create_react_agent` + `AgentExecutor(verbose=True)` 构建并运行 Agent，
在控制台打印完整的 Thought → Action → Observation 循环，
让学习者亲眼看到"大模型是怎么一步步思考、决策、调用工具的"。

---

## 目录结构

```
langchain_learning/
├── llm/native_api.py         （已有，项目零）
├── langchain/chatbot.py      （已有，项目一）
├── rag/rag_qa.py             （已有，项目二）
├── agent/                    ← 新建
│   └── react_agent.py        ← 项目三主教学文件（新建）
└── requirements.txt          ← 无需新增依赖
```

---

## `agent/react_agent.py` 内部章节

| 章节 | 内容 | 核心学习点 |
|------|------|-----------|
| 文件头（docstring） | "老板、员工、工具箱"比喻；Agent vs 普通 LLM 的本质区别；ReAct 三步循环底层机制 | 什么是 Agent；Tool Calling 原理；Thought/Action/Observation 循环 |
| 第 0 章 | LLM 初始化（和前几个项目完全一致） | 复用已学 ChatOpenAI 配置 |
| 第 1 章 | `@tool` 装饰器定义两个 mock 工具：`get_weather` 和 `calculate_power`；打印工具元数据 | `@tool` 用法；`description` 的关键作用；`name`/`args_schema` |
| 第 2 章 | 内联中文 ReAct prompt；`create_react_agent(llm, tools, prompt)`；`AgentExecutor(verbose=True)` | `create_react_agent`；`AgentExecutor`；ReAct prompt 占位符 |
| 第 3 章 | 三轮演示：①单工具（天气）②单工具（计算）③多步推理（天气+计算链式调用） | 多步 ReAct 循环；工具链式调用 |

---

## 文件头科普内容

### 比喻一：老板、员工、工具箱

```
普通 LLM 对话 = 只能用脑子的员工
  你问什么，他只能根据脑子里记住的知识回答。
  无法上网查、无法打电话、无法用计算器。

Agent = 配备了工具箱的员工
  员工（LLM）接到任务后，会自己思考：
    "我需要查天气" → 拿起电话（调用 get_weather 工具）
    "我需要算一个数" → 拿起计算器（调用 calculate_power 工具）
    "信息够了，可以给老板汇报了" → 返回最终答案

  工具箱（Tools） = 一组可以被 LLM 主动调用的函数
```

### 比喻二：ReAct 框架的底层循环

```
ReAct = Reasoning（推理）+ Acting（行动）

每一轮循环分三步：
  Thought   → LLM 内心独白："我现在知道了什么，下一步应该做什么"
  Action    → LLM 决定调用哪个工具，传什么参数
  Observation → 工具执行后返回的结果，LLM 看到结果后进入下一轮

循环直到 LLM 决定"我信息够了，可以给出 Final Answer"
```

---

## 工具设计

### `get_weather(city: str) -> str`

Mock 天气查询工具。返回城市 → 天气/温度的固定字典查询结果。

```python
WEATHER_DATA = {
    "北京": {"weather": "晴", "temperature": 28},
    "上海": {"weather": "多云", "temperature": 24},
    "广州": {"weather": "阵雨", "temperature": 32},
    "成都": {"weather": "阴", "temperature": 20},
}
```

找不到城市时返回"暂无数据"提示字符串。

**Description 示例：**
```
查询指定城市的实时天气信息，返回天气状况和温度。
输入城市名（中文），例如"北京"、"上海"。
```

### `calculate_power(base: int, exponent: int) -> str`

真实计算 `base ** exponent`，返回结果字符串。

**Description 示例：**
```
计算一个整数的幂次方（base 的 exponent 次方）。
例如：base=2, exponent=10 → 1024。
适合需要精确数学计算的场景。
```

---

## ReAct Prompt 设计

内联中文 ReAct prompt（不依赖 `hub.pull`，完全离线），包含四个必要占位符：

```
{tools}           → 所有工具的名称 + description（LLM 靠这个决定用哪个工具）
{tool_names}      → 工具名列表（格式约束用）
{input}           → 用户的实际问题
{agent_scratchpad}→ Agent 的历史 Thought/Action/Observation 记录
```

Prompt 包含三部分：
1. System 角色设定（严谨的助手，有工具可用）
2. 工具列表展示（`{tools}` 填入）
3. ReAct 格式规范（告诉 LLM 必须用 `Thought:/Action:/Observation:/Final Answer:` 格式）

---

## 技术选型

| 决策点 | 选择 | 原因 |
|--------|------|------|
| Agent 构建 API | `create_react_agent` + `AgentExecutor` | LangChain 现有包内置，无需额外安装；`verbose=True` 开箱打印 ReAct 循环 |
| ReAct Prompt | 内联中文 prompt（非 `hub.pull`） | 离线可运行；学习者可以直接看到 LLM 接收的完整提示词格式 |
| 工具数量 | 2 个（天气 + 幂次计算） | 足以演示单工具和多工具链式调用，不过度复杂 |
| 工具实现 | Mock 数据（天气）+ 真实计算（幂次） | 避免外部 API 依赖；计算工具有真实反馈，结果可验证 |
| verbose 模式 | `AgentExecutor(verbose=True)` | 零额外代码打印完整 Thought/Action/Observation |
| LLM 类 | `ChatOpenAI`（和前三个项目一致） | 共用已验证的接口配置 |
| 新增依赖 | 无 | `langchain.agents` 已在现有 `langchain>=0.3.0` 中 |

---

## 演示场景设计

| 轮次 | 问题 | 期望 Agent 行为 |
|------|------|----------------|
| ①   | "北京今天天气怎么样？" | Thought → Action: get_weather("北京") → Observation → Final Answer |
| ②   | "2 的 10 次方是多少？" | Thought → Action: calculate_power(2, 10) → Observation → Final Answer |
| ③   | "北京今天温度是多少度？如果把这个温度值作为底数、2 作为指数，结果是多少？" | 多步：get_weather → 提取温度 → calculate_power → Final Answer（验证多步推理） |

---

## ⚠️ 避坑指南（文件内需包含的注释）

1. **`description` 写不清楚**：LLM 会选错工具或不知道该用哪个，必须描述"什么时候用"和"参数格式"
2. **`handle_parsing_errors=True`**：LLM 偶尔输出格式不标准，不加这个参数会直接抛异常
3. **ReAct prompt 三个必填占位符**：`{tools}`、`{tool_names}`、`{agent_scratchpad}` 缺一不可，否则 Agent 无法工作
4. **`max_iterations` 防死循环**：Agent 可能反复调用工具无法收敛，建议设置 `max_iterations=5`
5. **工具函数必须有类型注解**：`@tool` 装饰器依赖类型注解生成 `args_schema`，缺少类型注解会影响 LLM 理解参数格式

---

## 学习目标

读完 `agent/react_agent.py` 后，学习者应能理解：
1. Agent 和普通 LLM 聊天的本质区别（主动调用工具 vs 只能凭记忆回答）
2. `@tool` 装饰器的用法以及 `description` 参数为何是 Agent 的"工具说明书"
3. `create_react_agent` + `AgentExecutor` 的构建流程
4. ReAct 框架的 Thought → Action → Observation 三步循环如何运作
5. 如何在控制台观察 Agent 的内部思维过程，验证 Agent 是否正确理解了任务
