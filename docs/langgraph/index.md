---
title: LangGraph 图编排
---

# LangGraph（图结构工作流编排）

LangGraph 将 AI 工作流建模为有向图，支持条件分支、循环、并行执行，是构建复杂 Agent 系统的核心框架。

## 1. 为什么需要 LangGraph

AgentExecutor 的局限：
- 只有单个 LLM 循环，无法多角色分工
- 固定 ReAct 循环，无法自定义分支
- 无法"打回重做"（循环/重试）
- 不支持人工审批（HITL）

## 2. 三要素

| 要素 | 含义 | 对应 |
|------|------|------|
| **Node** | 工作站，一个 Python 函数 | 做一件事 |
| **Edge** | 传送带，数据流转 | 普通边/条件边 |
| **State** | 数据包，在图上流动 | TypedDict |

## 3. State 定义

```python
from typing import TypedDict

class StudioState(TypedDict):
    topic: str          # 主题（全程不变）
    outline: str        # 研究员产出
    draft: str          # 写手初稿
    feedback: str       # 主编意见
    is_approved: bool   # 是否通过
    revision: int       # 修改轮次
```

**更新语义：** 默认覆盖；使用 `Annotated[list, operator.add]` 可切换为追加模式。

## 4. 节点函数

```python
def researcher_node(state: StudioState) -> dict:
    topic = state["topic"]
    chain = prompt | llm | StrOutputParser()
    outline = chain.invoke({"topic": topic})
    return {"outline": outline}  # 只返回需更新的字段
```

## 5. 普通边 vs 条件边

```python
# 普通边：无条件跳转
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")

# 条件边：根据 State 路由
def should_continue(state: StudioState) -> str:
    if state["is_approved"]:
        return END
    return "writer"  # 打回重写

workflow.add_conditional_edges("editor", should_continue)
```

## 6. 图结构示例

```
      START
        │
        ▼
   [researcher] → 生成大纲
        │
        ▼
    [writer] ◀─── 不合格（打回重写）
        │              │
        ▼              │
    [editor] ──────────┘
        │ 合格
        ▼
       END
```

## 7. 编译与运行

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(StudioState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)
# ... add edges ...

app = workflow.compile()
result = app.invoke({"topic": "AI 学习方法", "revision": 0})
```

## 8. 循环与防死循环

条件边指回之前的节点即实现循环。通过计数器或 `recursion_limit` 防止无限循环：

```python
MAX_REVISIONS = 3
if not is_approved and revision >= MAX_REVISIONS:
    is_approved = True  # 强制通过
```

## 9. 并行执行

扇出（Fan-out）+ 扇入（Fan-in）：

```python
# 多节点从同一节点出发并行执行
workflow.add_edge("start", "researcher_a")
workflow.add_edge("start", "researcher_b")
# 汇聚到同一节点
workflow.add_edge("researcher_a", "aggregator")
workflow.add_edge("researcher_b", "aggregator")
```

使用 `Annotated[list, operator.add]` 避免并行写同字段冲突。

## 10. LangChain vs LangGraph

| LangChain LCEL | LangGraph |
|---------------|-----------|
| 线性管道 A\|B\|C | 图：分支+循环+并行 |
| 无状态管理 | TypedDict 显式状态 |
| 单 Agent | 多 Agent 协作 |
| 不支持 HITL | interrupt_before 人机协作 |

::: warning 需要本地运行
完整实现见 `langgraph/media_studio.py`（多 Agent 内容工作室）。
:::

---

::: tip 下一步
- [LangGraph 高级](/langgraph/advanced) — 人机协作、检查点、断点续传
- [Agent 基础](/agent/) — 复习 Agent 核心概念
:::
