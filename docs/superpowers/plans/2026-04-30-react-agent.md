# 项目三：ReAct Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `agent/react_agent.py` 中实现完整的 ReAct Agent 教学文件——用 `@tool` 装饰器定义 mock 工具，用 `create_react_agent + AgentExecutor(verbose=True)` 构建 Agent，在控制台打印完整 Thought→Action→Observation 循环。

**Architecture:** 单文件渐进式教学（4章节），内联老板/员工/工具箱科普 + ReAct 原理说明，两个 mock 工具（天气查询 + 幂次计算），内联中文 ReAct prompt（不依赖 hub.pull），三轮演示（单工具 × 2 + 多步推理）。

**Tech Stack:** Python 3.9+, langchain>=0.3.0, langchain-openai>=0.2.0（已安装，无需新增依赖）

---

## 文件清单

| 操作 | 路径 | 职责 |
|------|------|------|
| Create dir | `agent/` | Agent 系列学习目录 |
| Create | `agent/react_agent.py` | 项目三主教学文件（4个任务逐步构建） |

---

## Task 1：创建 `agent/` 目录，写入文件头 + 第 0 章

**Files:**
- Create dir: `agent/`
- Create: `agent/react_agent.py`

- [ ] **Step 1：创建目录**

```bash
mkdir -p /Users/liuyu22/Desktop/langchain_learning/agent
```

预期：命令无输出，目录创建成功。

- [ ] **Step 2：创建文件，写入完整内容**

创建 `/Users/liuyu22/Desktop/langchain_learning/agent/react_agent.py`，内容如下：

```python
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
from langchain.agents import create_react_agent, AgentExecutor

# @tool 装饰器：把普通 Python 函数变成 Agent 可以调用的工具
from langchain_core.tools import tool

# Prompt 模板：ReAct Agent 需要特定格式的 PromptTemplate
from langchain_core.prompts import PromptTemplate

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
```

- [ ] **Step 3：验证第 0 章可以运行（只验证语法和初始化，不发 API 请求）**

```bash
cd /Users/liuyu22/Desktop/langchain_learning && source .venv/bin/activate && python agent/react_agent.py
```

预期输出：
```
============================================================
第 0 章：初始化 LLM
============================================================
✅ LLM 初始化完成
   模型: kivy-kimi-k2_5
```

- [ ] **Step 4：提交**

```bash
cd /Users/liuyu22/Desktop/langchain_learning && git add agent/react_agent.py && git commit -m "$(cat <<'EOF'
feat: add project three - agent/react_agent.py header + chapter 0

File header with 老板/员工/工具箱 Agent analogy and ReAct
Thought/Action/Observation loop explanation. Chapter 0: LLM init.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2：追加第 1 章（工具定义）

**Files:**
- Modify: `agent/react_agent.py`（末尾追加）

- [ ] **Step 1：追加第 1 章代码**

在 `agent/react_agent.py` 末尾追加：

```python
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
    例如：base=2, exponent=10 → 返回 1024。
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
```

- [ ] **Step 2：运行并验证第 1 章输出**

```bash
cd /Users/liuyu22/Desktop/langchain_learning && source .venv/bin/activate && python agent/react_agent.py
```

预期新增输出：
```
============================================================
第 1 章：工具定义（@tool 装饰器）
============================================================

【@tool 装饰器为每个工具生成的元数据】

  工具名称 (name):        get_weather
  工具描述 (description): 查询指定城市的实时天气信息，返回天气状况和温度（摄氏度）。...
  参数结构 (args_schema): {'city': {'title': 'City', 'type': 'string'}}

  工具名称 (name):        calculate_power
  工具描述 (description): 计算一个整数的幂次方（base 的 exponent 次方）。...
  参数结构 (args_schema): {'base': {...}, 'exponent': {...}}

💡 LLM 接收到的工具信息就是上面这些。
   它靠 description 决定用哪个工具，靠 args_schema 知道怎么传参数。
```

- [ ] **Step 3：提交**

```bash
cd /Users/liuyu22/Desktop/langchain_learning && git add agent/react_agent.py && git commit -m "$(cat <<'EOF'
feat: add react_agent.py chapter 1 - tool definitions with @tool

Mock get_weather (city dict lookup) and calculate_power (real math).
Prints tool metadata showing name/description/args_schema.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3：追加第 2 章（构建 Agent）

**Files:**
- Modify: `agent/react_agent.py`（末尾追加）

- [ ] **Step 1：追加第 2 章代码**

在 `agent/react_agent.py` 末尾追加：

```python
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：构建 ReAct Agent
# 目标：把 LLM + 工具 + Prompt 组装成一个 Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：构建 ReAct Agent")
print("=" * 60)
print()

# ── ReAct Prompt 模板 ─────────────────────────────────────
#
# create_react_agent 需要一个包含四个占位符的 PromptTemplate：
#
#   {tools}            → 所有工具的名称 + description 拼接文本
#                        AgentExecutor 会自动填入，LLM 靠这个认识工具
#   {tool_names}       → 工具名称列表（逗号分隔）
#                        用于格式约束：告诉 LLM "Action 只能是这些名字之一"
#   {input}            → 用户的实际问题
#   {agent_scratchpad} → Agent 的历史推理记录（上一轮的 Thought/Action/Observation）
#                        LLM 靠这个"记住"之前做了什么，避免重复调用工具
#
# ⚠️ 避坑指南：四个占位符缺一不可！
#   如果 prompt 里漏掉任何一个，create_react_agent 会抛出 ValueError。
#
# 注意：这里使用 PromptTemplate（不是 ChatPromptTemplate）
#   因为 ReAct 格式是纯文本协议，不区分 system/human 角色。

REACT_PROMPT_TEMPLATE = """你是一个严谨、有条理的 AI 助手。你可以使用以下工具来回答用户的问题：

{tools}

回答问题时，请严格按照以下格式逐行输出（不要跳过任何步骤）：

Question: 你需要回答的问题
Thought: 分析当前情况，决定下一步行动
Action: 选择要使用的工具，必须是 [{tool_names}] 中的一个
Action Input: 传给工具的参数（直接写值，不要加引号或 JSON 格式）
Observation: （工具返回的结果，由系统填入，你不需要编写这行）
...（以上 Thought/Action/Action Input/Observation 可以重复多次）
Thought: 我现在已经有足够的信息来回答问题了
Final Answer: 对用户原始问题的完整回答

现在开始！

Question: {input}
Thought:{agent_scratchpad}"""

react_prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

print("【ReAct Prompt 模板已定义】")
print("   包含四个必填占位符：{tools} {tool_names} {input} {agent_scratchpad}")
print()

# ── 创建 Agent ─────────────────────────────────────────────
#
# create_react_agent 把三个组件组装成一个"决策单元"（Agent）：
#   llm    → 负责推理：读取 Thought，决定下一步 Action
#   tools  → 可调用的工具列表
#   prompt → ReAct 格式的提示词模板
#
# 返回的 agent 本身是一个 Runnable，不能直接 invoke——
# 需要包在 AgentExecutor 里才能运行完整的 ReAct 循环。

agent = create_react_agent(llm, tools, react_prompt)

# ── 创建 AgentExecutor ────────────────────────────────────
#
# AgentExecutor 是 Agent 的"运行引擎"，负责：
#   ① 调用 agent 做推理，解析 Action 和 Action Input
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
#   LLM 偶尔会输出格式不标准的内容（比如 Action 前多了空格），
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
print("   create_react_agent(llm, tools, prompt) → 决策单元（知道怎么想）")
print("   AgentExecutor(agent, tools, verbose=True) → 执行引擎（负责跑循环）")
print()
```

- [ ] **Step 2：运行并验证第 2 章输出（不发 API 请求，只验证构建）**

```bash
cd /Users/liuyu22/Desktop/langchain_learning && source .venv/bin/activate && python agent/react_agent.py
```

预期新增输出：
```
============================================================
第 2 章：构建 ReAct Agent
============================================================

【ReAct Prompt 模板已定义】
   包含四个必填占位符：{tools} {tool_names} {input} {agent_scratchpad}

✅ Agent 构建完成！
   agent_executor 已准备好，调用 agent_executor.invoke({'input': '...'}) 即可运行

💡 小结：三个组件的分工
   create_react_agent(llm, tools, prompt) → 决策单元（知道怎么想）
   AgentExecutor(agent, tools, verbose=True) → 执行引擎（负责跑循环）
```

- [ ] **Step 3：提交**

```bash
cd /Users/liuyu22/Desktop/langchain_learning && git add agent/react_agent.py && git commit -m "$(cat <<'EOF'
feat: add react_agent.py chapter 2 - ReAct prompt and agent construction

Inline Chinese ReAct prompt with {tools}/{tool_names}/{input}/
{agent_scratchpad} placeholders. AgentExecutor verbose=True,
handle_parsing_errors=True, max_iterations=5.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4：追加第 3 章（三轮演示）

**Files:**
- Modify: `agent/react_agent.py`（末尾追加）

- [ ] **Step 1：追加第 3 章代码**

在 `agent/react_agent.py` 末尾追加：

```python
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
```

- [ ] **Step 2：运行完整脚本，观察三轮 ReAct 循环（发起真实 LLM API 调用）**

```bash
cd /Users/liuyu22/Desktop/langchain_learning && source .venv/bin/activate && python agent/react_agent.py
```

预期最终新增输出（结构示意，AI 文本会有差异）：
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【演示：单工具调用（天气查询）】
❓ 问题：北京今天天气怎么样？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> Entering new AgentExecutor chain...
Thought: 用户想了解北京的天气，我应该使用 get_weather 工具查询。
Action: get_weather
Action Input: 北京
Observation: 北京：晴，温度 28°C
Thought: 我已经获得了北京的天气信息，可以回答用户了。
Final Answer: 北京今天天气晴，气温 28 摄氏度。

> Finished chain.

✅ 最终答案：北京今天天气晴，气温 28 摄氏度。
```

关键验证点：
- 演示一：出现 `Action: get_weather` 和 `Observation: 北京：晴`
- 演示二：出现 `Action: calculate_power` 和 `Observation: 2 的 10 次方 = 1024`
- 演示三：出现两次 Action（先 get_weather 再 calculate_power），展示多步推理

- [ ] **Step 3：提交第 3 章，完成项目三**

```bash
cd /Users/liuyu22/Desktop/langchain_learning && git add agent/react_agent.py && git commit -m "$(cat <<'EOF'
feat: add react_agent.py chapter 3 - three ReAct demo scenarios

Single-tool demos (weather, power) and multi-step chained
tool call demo showing Observation feeding next Thought.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## 自检清单（Spec Self-Review）

- [x] Task 1 覆盖 agent/ 目录创建 + 文件头（老板/员工/工具箱 + ReAct 循环科普）+ 第0章 LLM 初始化
- [x] Task 2 覆盖第1章：`@tool` 装饰器、两个工具（get_weather + calculate_power）、打印工具元数据、description 重要性注释、类型注解要求注释
- [x] Task 3 覆盖第2章：内联中文 ReAct prompt（四占位符）、`create_react_agent`、`AgentExecutor(verbose=True, handle_parsing_errors=True, max_iterations=5)`
- [x] Task 4 覆盖第3章：三轮演示（单工具天气、单工具计算、多步链式调用）
- [x] 所有 ⚠️ 避坑指南写入代码注释（description 重要性、类型注解、四占位符、handle_parsing_errors、max_iterations）
- [x] 每个 Step 都有完整代码，无 TBD 占位符
- [x] 每个 Step 都有预期输出，方便验证
- [x] 变量名一致：`llm`、`tools`、`agent`、`agent_executor`、`react_prompt`
- [x] `WEATHER_DATA` 字典在第1章定义，`get_weather` 函数使用它——无提前引用问题
- [x] 演示三的多步推理设计合理：28（北京温度）→ calculate_power(28, 2) = 784，结果可验证
