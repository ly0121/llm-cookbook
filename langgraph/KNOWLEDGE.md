# LangGraph（图结构工作流编排）完全知识手册

> 本文档是一份系统性的 LangGraph 技术教科书，从图计算基础到多 Agent 编排，覆盖 LangGraph 的所有核心知识点。
> 配合 `media_studio.py` 代码阅读效果更佳。

---

## 目录

1. [LangGraph 设计理念](#1-langgraph-设计理念)
2. [有向图基础概念](#2-有向图基础概念)
3. [StateGraph 与 TypedDict 状态定义](#3-stategraph-与-typeddict-状态定义)
4. [节点函数设计](#4-节点函数设计)
5. [普通边 vs 条件边](#5-普通边-vs-条件边)
6. [编译与运行图](#6-编译与运行图)
7. [多 Agent 工作流编排](#7-多-agent-工作流编排)
8. [循环与反馈机制](#8-循环与反馈机制)
9. [并行节点执行](#9-并行节点执行)
10. [子图（Subgraph）](#10-子图subgraph)
11. [错误处理与重试](#11-错误处理与重试)
12. [与 LangChain 的关系和区别](#12-与-langchain-的关系和区别)

---

## 1. LangGraph 设计理念

### 1.1 为什么需要 LangGraph

```
对应 media_studio.py 前置科普一：

AgentExecutor 的局限：
  ① 只有单个 LLM 在循环——无法多角色分工
  ② 固定的 ReAct 循环——无法自定义流程分支
  ③ 黑盒运行——中间状态不可观测
  ④ 无法"打回重做"——缺乏循环/重试机制

LangGraph 的解决方案：把 AI 工作流建模为"有向图"
  节点 = 工作站（每个节点做一件事）
  边   = 传送带（数据从一个节点流向另一个）
  状态 = 在传送带上流动的数据包
```

### 1.2 图计算 vs 链式计算

```
链式计算（LangChain LCEL）：
  A → B → C → D
  数据严格线性流动，中间无法分支或循环。

图计算（LangGraph）：
         ┌→ B ─┐
  A → ───┤     ├→ D → E
         └→ C ─┘     ↑
              │       │
              └── 不合格 ──┘ (循环！)

  支持：
  ① 条件分支（根据状态选择路径）
  ② 循环（打回重做）
  ③ 并行（多节点同时执行）
  ④ 子图（嵌套复杂逻辑）
```

### 1.3 核心设计原则

```
① 状态驱动（State-Driven）
   一切决策基于当前 State，不依赖隐式上下文

② 声明式定义（Declarative）
   先声明图结构（节点+边），再编译执行

③ 可观测性（Observability）
   每一步的 State 变化都可追踪和调试

④ 可组合性（Composability）
   图可以嵌套、组合，构建复杂工作流
```

---

## 2. 有向图基础概念

### 2.1 三要素

```
对应 media_studio.py 前置科普二：

┌────────────────────────────────────────────────────────────┐
│  ① Node（节点）= 一个"工作站"                              │
│     每个节点是一个 Python 函数                              │
│     输入：当前 State                                       │
│     输出：要更新的 State 字段（dict）                       │
│                                                            │
│  ② Edge（边）= 节点之间的"传送带"                          │
│     普通边：A → B（无条件）                                │
│     条件边：A → B 或 C（根据 State 判断）                  │
│                                                            │
│  ③ State（状态）= 在图上流动的"数据包"                     │
│     所有节点共享同一个 State（TypedDict）                   │
│     每个节点通过返回 dict 来更新 State 字段                │
└────────────────────────────────────────────────────────────┘
```

### 2.2 特殊节点

```
START：图的入口点（不是真正的节点，而是标记起始位置）
END：图的终结点（到达这里表示执行完毕）

使用：
  from langgraph.graph import StateGraph, START, END

  workflow.add_edge(START, "first_node")    # 图从 first_node 开始
  workflow.add_edge("last_node", END)       # last_node 结束后图完成
```

### 2.3 图结构示例

```
对应 media_studio.py 前置科普三的工作室流程：

          START
            │
            ▼
      ┌──────────┐
      │ researcher│  生成大纲素材
      └─────┬────┘
            │
            ▼
      ┌──────────┐
      │  writer  │◀──── 不合格（打回重写）
      └─────┬────┘          │
            │               │
            ▼               │
      ┌──────────┐          │
      │  editor  │──────────┘
      └─────┬────┘
            │ 合格
            ▼
           END
```

---

## 3. StateGraph 与 TypedDict 状态定义

### 3.1 State 定义

```
对应 media_studio.py 第1章：

用 TypedDict 定义 State 的结构：

class StudioState(TypedDict):
    topic: str          # 用户给定的主题（全程不变）
    outline: str        # 研究员产出的大纲
    draft: str          # 写手的文章初稿
    feedback: str       # 主编的修改意见
    is_approved: bool   # 是否通过审核
    revision: int       # 当前修改轮次
    final: str          # 最终定稿

好处：
  ① 类型提示：IDE 自动补全字段名
  ② 文档化：一眼看清数据流转
  ③ 运行时检查：节点返回非法 key 时报错
```

### 3.2 状态更新语义

```
默认行为：覆盖（Override）

  节点 A 返回 {"draft": "版本1"}
  节点 B 返回 {"draft": "版本2"}
  → State 中 draft = "版本2"（后者覆盖前者）

追加行为：使用 Annotated + reducer

  from typing import Annotated
  import operator

  class ChatState(TypedDict):
      messages: Annotated[list, operator.add]  # 追加模式！
      step: str                                 # 覆盖模式

  节点 A 返回 {"messages": ["消息1"]}
  节点 B 返回 {"messages": ["消息2"]}
  → State 中 messages = ["消息1", "消息2"]（拼接！）

何时用追加：
  - 消息历史（越来越长）
  - 日志记录（累积）
  - 结果收集（多个节点各贡献一部分）
```

### 3.3 创建 StateGraph

```
# 传入 State 类型，告诉 LangGraph 数据结构
workflow = StateGraph(StudioState)

# 此时 workflow 是一个"空图"，需要添加节点和边
```

---

## 4. 节点函数设计

### 4.1 节点函数规则

```
对应 media_studio.py 第2章：

每个节点函数必须遵循：
  ① 接受一个参数：state（类型是你定义的 TypedDict）
  ② 返回一个 dict：key 对应 State 中要更新的字段
  ③ 只返回需要更新的字段（不需要返回完整 State）

示例：
def researcher_node(state: StudioState) -> dict:
    # 读取 state 中的数据
    topic = state["topic"]

    # 做任何处理（调用 LLM、API、计算...）
    outline = call_llm(topic)

    # 返回要更新的字段
    return {"outline": outline}
```

### 4.2 节点内部可以做什么

```
节点函数内部可以执行任意 Python 代码：

  ① 调用 LLM（最常见）
     chain = prompt | llm | parser
     result = chain.invoke({"topic": state["topic"]})

  ② 调用外部 API
     response = requests.get(url)

  ③ 读写文件/数据库
     data = db.query(...)

  ④ 纯逻辑计算
     if len(draft) < 150: return {"is_approved": False}

  ⑤ 组合以上所有操作

LangGraph 不关心节点内部做了什么，只关心它返回了什么。
```

### 4.3 节点设计模式

```
模式一：LLM 调用节点（最常见）
  读取 State → 构造 Prompt → 调用 LLM → 解析输出 → 返回更新

模式二：路由/判断节点
  读取 State → 做逻辑判断 → 返回标志位
  （配合条件边实现分支）

模式三：聚合节点
  读取 State 中多个字段 → 综合处理 → 返回汇总结果

模式四：副作用节点
  发送邮件、写数据库、调用第三方 API
  （注意幂等性设计）
```

---

## 5. 普通边 vs 条件边

### 5.1 普通边（Normal Edge）

```
对应 media_studio.py 第4章：

add_edge(from_node, to_node)：无条件跳转

  workflow.add_edge(START, "researcher")      # 入口 → 研究员
  workflow.add_edge("researcher", "writer")   # 研究员 → 写手
  workflow.add_edge("writer", "editor")       # 写手 → 主编

  "做完 A 一定做 B"——没有任何条件判断。
```

### 5.2 条件边（Conditional Edge）

```
对应 media_studio.py 第3章：

add_conditional_edges(source_node, route_function, path_map)

  source_node    → 条件判断发生在哪个节点之后
  route_function → 路由函数（读取 State，返回目标节点名）
  path_map       → 可选映射 {返回值: 目标节点}

路由函数：
  def should_continue(state: StudioState) -> str:
      if state["is_approved"]:
          return END        # 审核通过 → 结束
      else:
          return "writer"   # 不通过 → 打回写手

添加条件边：
  workflow.add_conditional_edges(
      "editor",           # 主编节点之后
      should_continue,    # 路由函数
  )

路由函数返回值必须是：
  - 图中已定义的节点名称（如 "writer"）
  - 特殊值 END（结束图执行）
```

### 5.3 条件边 vs if-else 的区别

```
为什么不直接在节点里写 if-else？

  节点内 if-else：逻辑内聚在一个节点里，图结构不可见
  条件边：         路由逻辑显式声明在图结构中，可视化友好

条件边的优势：
  ① 图的结构是"声明式"的，编译时可校验
  ② 图可以可视化（用 .get_graph().draw_mermaid()）
  ③ 路由逻辑与业务逻辑分离，易测试
  ④ 支持动态路由到多个目标（不止两个分支）
```

---

## 6. 编译与运行图

### 6.1 编译（compile）

```
对应 media_studio.py 第4章步骤五：

app = workflow.compile()

compile() 做了什么？
  ① 验证图结构合法性（所有节点可达、没有孤立节点）
  ② 检查边的完整性（每个节点都有出边）
  ③ 生成 CompiledGraph 对象（是一个 Runnable）

编译可选参数：
  app = workflow.compile(
      checkpointer=memory,         # 检查点存储（持久化）
      interrupt_before=["review"], # 在哪些节点前暂停（HITL）
  )
  → 详见 langgraph_advanced/KNOWLEDGE.md
```

### 6.2 运行（invoke）

```
对应 media_studio.py 第5章：

final_state = app.invoke({
    "topic": "为什么程序员应该学习AI",
    "revision": 0,
    "is_approved": False,
    ...
})

invoke() 的执行流程：
  ① 把初始 State 传给第一个节点
  ② 节点处理完后，用返回值更新 State
  ③ 沿着边把更新后的 State 传给下一个节点
  ④ 遇到条件边时，调用路由函数决定下一站
  ⑤ 到达 END 时，返回最终的完整 State

只需提供起始节点需要的字段：
  研究员只需要 topic → 初始 State 只需提供 topic
  其他字段会在执行过程中被各节点填充
```

### 6.3 流式执行

```
# 逐步获取每个节点的输出
for event in app.stream(initial_state):
    print(event)
    # 每个 event 包含：节点名 → 该节点的输出

# 也可以使用 astream 异步版本
async for event in app.astream(initial_state):
    ...
```

---

## 7. 多 Agent 工作流编排

### 7.1 角色分工模式

```
对应 media_studio.py 的完整实现：

三个 AI 角色，各司其职：

  研究员（Researcher）
    输入：topic
    输出：outline（大纲素材）
    Prompt：资深内容研究员，生成有逻辑的大纲

  写手（Writer）
    输入：topic + outline（+ feedback，如果是重写）
    输出：draft（文章初稿）
    Prompt：才华横溢的写手，生动有趣

  主编（Editor）
    输入：draft
    输出：is_approved + feedback/final
    Prompt：严格的主编，审核质量

每个角色有独立的 System Prompt 和职责边界。
```

### 7.2 状态设计原则

```
好的 State 设计应该：

  ① 明确"谁写谁读"
     topic     → 用户写入，所有节点读取
     outline   → 研究员写入，写手读取
     draft     → 写手写入，主编读取
     feedback  → 主编写入，写手读取（循环时）
     final     → 主编写入，最终输出

  ② 包含流程控制字段
     revision  → 防止无限循环（最多修改 N 次）
     is_approved → 条件路由的依据

  ③ 字段粒度适中
     太粗：一个 result 字段装所有内容 → 难以路由
     太细：每步中间变量都放 State → 状态臃肿
```

---

## 8. 循环与反馈机制

### 8.1 循环实现

```
对应 media_studio.py 的"主编打回"机制：

实现方式：条件边 + 指回之前的节点

  writer → editor → (条件边) → writer（循环！）
                             → END（结束）

代码：
  workflow.add_edge("writer", "editor")
  workflow.add_conditional_edges("editor", should_continue)

  def should_continue(state):
      if state["is_approved"]:
          return END       # 通过 → 结束
      return "writer"      # 不通过 → 回到 writer
```

### 8.2 防止无限循环

```
对应 media_studio.py 中 editor_node 的 MAX_REVISIONS：

方法一：计数器
  MAX_REVISIONS = 3
  if not is_approved and revision >= MAX_REVISIONS:
      is_approved = True  # 强制通过

方法二：在路由函数中判断
  def should_continue(state):
      if state["revision"] >= 5:
          return END  # 超过5次，强制结束
      ...

方法三：图编译时设置 recursion_limit
  app = workflow.compile()
  app.invoke(state, config={"recursion_limit": 20})
```

### 8.3 反馈传递

```
主编 → 写手的反馈通过 State 传递：

editor_node 返回：
  {"is_approved": False, "feedback": "字数不够，请扩充"}

writer_node 读取：
  feedback = state.get("feedback", "")
  if feedback:
      # 根据反馈重写（使用不同的 Prompt）
      prompt = "根据以下反馈修改文章：{feedback}"
```

---

## 9. 并行节点执行

### 9.1 扇出（Fan-out）

```
多个节点从同一个节点出发，并行执行：

          ┌→ 研究员A（调研技术方面）
  START → ┼→ 研究员B（调研市场方面）
          └→ 研究员C（调研竞品方面）
                     │
                     ▼
                   聚合节点（合并三个研究结果）

实现方式：一个节点连出多条普通边
  workflow.add_edge("start_node", "researcher_a")
  workflow.add_edge("start_node", "researcher_b")
  workflow.add_edge("start_node", "researcher_c")

状态合并：使用 Annotated[list, operator.add]
  三个研究员各自往 results 列表追加自己的结果
```

### 9.2 扇入（Fan-in）

```
多个并行节点汇聚到同一个节点：

  workflow.add_edge("researcher_a", "aggregator")
  workflow.add_edge("researcher_b", "aggregator")
  workflow.add_edge("researcher_c", "aggregator")

  LangGraph 会等待所有前驱节点完成后，才执行聚合节点。
  此时 State 中已经包含所有并行节点的更新。
```

### 9.3 注意事项

```
并行节点写同一个字段时的冲突解决：

  ❌ 错误：两个节点都返回 {"result": "..."}
     后执行的会覆盖先执行的！

  ✅ 正确：使用 Annotated + reducer
     results: Annotated[list, operator.add]
     每个节点返回 {"results": [自己的结果]}
     → 最终 results = [结果A, 结果B, 结果C]
```

---

## 10. 子图（Subgraph）

### 10.1 子图概念

```
子图 = 图中嵌套的图

场景：一个节点本身是一个复杂的子工作流

  主图：
    START → 需求分析 → [开发子图] → 测试 → 部署 → END

  开发子图（展开）：
    START → 架构设计 → 编码 → Code Review → END

实现：
  # 定义子图
  sub_workflow = StateGraph(SubState)
  sub_workflow.add_node(...)
  sub_graph = sub_workflow.compile()

  # 在主图中作为节点使用
  main_workflow.add_node("development", sub_graph)
```

### 10.2 子图状态映射

```
子图可以有自己的 State 定义，与主图不同：

  主图 State：{topic, sub_result, final}
  子图 State：{input_text, draft, review_result}

LangGraph 会自动处理同名字段的映射。
如果字段名不同，需要手动定义输入/输出映射。
```

---

## 11. 错误处理与重试

### 11.1 节点内部错误处理

```
def writer_node(state: StudioState) -> dict:
    try:
        draft = call_llm(state["topic"])
        return {"draft": draft}
    except Exception as e:
        # 返回错误信息到 State，让后续节点处理
        return {"draft": "", "error": str(e)}
```

### 11.2 重试机制

```
方法一：节点内部重试
  from tenacity import retry, stop_after_attempt

  @retry(stop=stop_after_attempt(3))
  def call_llm_with_retry(prompt):
      return llm.invoke(prompt)

方法二：图级别重试（通过条件边实现）
  如果节点输出包含 error → 路由回自身 → 重新执行
  （需要配合计数器防止无限重试）

方法三：Fallback 节点
  主节点失败 → 条件边 → Fallback 节点（降级处理）
```

### 11.3 超时处理

```
节点执行超时的处理策略：

  ① 异步超时
     import asyncio
     result = await asyncio.wait_for(node_func(), timeout=30)

  ② 配合 Checkpoint（推荐）
     即使超时，之前执行的节点结果已保存
     修复问题后可从中断点继续
     → 详见 langgraph_advanced/KNOWLEDGE.md
```

---

## 12. 与 LangChain 的关系和区别

### 12.1 架构关系

```
┌─────────────────────────────────────────────────────────────┐
│                     技术栈层次                                │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  LangGraph（编排层）                                  │   │
│  │  图结构、状态管理、条件路由、并行、循环、HITL        │   │
│  └──────────────────────────┬──────────────────────────┘   │
│                              │ 使用                         │
│  ┌──────────────────────────┴──────────────────────────┐   │
│  │  LangChain Core（基础组件层）                        │   │
│  │  ChatModel、Prompt、OutputParser、Tool、Runnable     │   │
│  └──────────────────────────┬──────────────────────────┘   │
│                              │ 调用                         │
│  ┌──────────────────────────┴──────────────────────────┐   │
│  │  LLM Provider（模型服务层）                          │   │
│  │  OpenAI、Anthropic、本地模型                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

LangGraph 建立在 LangChain 之上，但独立于 LangChain 的"链"概念。
```

### 12.2 LangChain vs LangGraph 对比

```
┌───────────────────┬────────────────────────────────────────┐
│    LangChain      │           LangGraph                    │
├───────────────────┼────────────────────────────────────────┤
│  LCEL 链式调用    │  图结构编排                            │
│  A | B | C        │  节点 + 边 + 条件路由                 │
│  线性流水线       │  支持分支、循环、并行                  │
│  无状态管理       │  TypedDict 显式状态                    │
│  适合简单管道     │  适合复杂工作流                        │
│  AgentExecutor    │  StateGraph + 自定义循环               │
│  单 Agent         │  多 Agent 协作                         │
│  不支持 HITL      │  interrupt_before 人机协作             │
│  不支持持久化     │  Checkpoint 断点续传                   │
└───────────────────┴────────────────────────────────────────┘

何时用 LangChain LCEL：
  - 简单的 prompt → LLM → parser 管道
  - 单步 RAG 检索问答
  - 线性数据处理流水线

何时用 LangGraph：
  - 需要条件分支或循环
  - 多 Agent 协作
  - 需要人工审批（HITL）
  - 需要状态持久化/断点续传
  - 复杂工作流（超过3个步骤且有分支）
```

### 12.3 在 LangGraph 中使用 LangChain 组件

```
对应 media_studio.py 中节点函数的实现：

LangGraph 的节点内部使用 LangChain 组件：

def researcher_node(state: StudioState) -> dict:
    # LangChain 的 Prompt 模板
    prompt = ChatPromptTemplate.from_messages([...])

    # LangChain 的 LCEL 链
    chain = prompt | llm | StrOutputParser()

    # LangChain 的 invoke
    outline = chain.invoke({"topic": state["topic"]})

    # 返回给 LangGraph 的状态更新
    return {"outline": outline}

关系：LangGraph 负责"编排"，LangChain 负责"执行"。
```

---

## 附录 A：本文件夹代码与知识点对应

| 代码文件 | 覆盖的知识点 | 对应本文档章节 |
|---------|-------------|---------------|
| `media_studio.py` | StateGraph、TypedDict、节点函数、条件边、循环、多Agent协作 | 第2-8节 |

---

## 附录 B：完整图构建代码模板

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. 定义 State
class MyState(TypedDict):
    input: str
    result: str
    is_done: bool

# 2. 定义节点函数
def process_node(state: MyState) -> dict:
    return {"result": f"处理了: {state['input']}"}

def check_node(state: MyState) -> dict:
    return {"is_done": len(state["result"]) > 10}

# 3. 定义路由函数
def router(state: MyState) -> str:
    return END if state["is_done"] else "process"

# 4. 构建图
workflow = StateGraph(MyState)
workflow.add_node("process", process_node)
workflow.add_node("check", check_node)
workflow.add_edge(START, "process")
workflow.add_edge("process", "check")
workflow.add_conditional_edges("check", router)

# 5. 编译并运行
app = workflow.compile()
result = app.invoke({"input": "hello", "result": "", "is_done": False})
```

---

## 附录 C：推荐学习路径

```
入门（1天）：
  第1-2节 → 理解图计算思想
  第3-6节 → 掌握基本构建流程
  运行 media_studio.py → 观察多 Agent 协作

进阶（2-3天）：
  第7-9节 → 多 Agent 编排与循环
  第10-11节 → 子图和错误处理
  修改 media_studio.py → 添加新角色/新分支

高级：
  第12节 → 理解架构全景
  前往 langgraph_advanced/KNOWLEDGE.md → 学习 HITL 和 Checkpoint
```

---

> **下一步学习**：前往 `langgraph_advanced/KNOWLEDGE.md` 了解人机协作（Human-in-the-Loop）和检查点持久化机制——这是构建生产级 AI Agent 的关键特性。
