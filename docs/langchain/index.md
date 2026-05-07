---
title: LangChain 框架
---

# LangChain 框架

LangChain 是一个 LLM 应用开发框架，核心使命是让开发者能用标准化的组件快速构建 LLM 应用。

## 1. 设计哲学

LangChain 的三大设计原则：

| 原则 | 说明 | 体现 |
|------|------|------|
| 组合性 | 每个组件是独立的积木块 | `prompt \| llm \| parser` |
| 可观测性 | 输入输出可追踪调试 | Callbacks + LangSmith |
| 可互换性 | 换模型只需改一行 | `ChatOpenAI → ChatAnthropic` |

## 2. LCEL（表达式语言）

LCEL 用管道符 `|` 将组件串联，就像 Linux 命令行管道：

```python
# Linux:     cat file | grep "key" | sort
# LangChain: prompt   | llm       | parser

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{topic}专家"),
    ("human", "{question}"),
])

chain = prompt | ChatOpenAI() | StrOutputParser()
result = chain.invoke({"topic": "Python", "question": "什么是列表？"})
```

**数据流：** `dict → ChatPromptValue → AIMessage → str`

## 3. Prompt Templates

```python
# ChatPromptTemplate：聊天模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}专家"),      # → SystemMessage
    ("human", "{question}"),           # → HumanMessage
])

# MessagesPlaceholder：为历史消息留位置
from langchain_core.prompts import MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
```

## 4. Output Parsers

| 解析器 | 输出类型 | 支持流式 | 适用场景 |
|--------|---------|---------|---------|
| StrOutputParser | str | 是 | 纯文本 |
| JsonOutputParser | dict | 是(部分) | 灵活 JSON |
| PydanticOutputParser | Pydantic 对象 | 否 | 强类型校验 |

## 5. Memory 机制

LLM 本身无状态，RunnableWithMessageHistory 实现跨轮对话记忆：

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_memory = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_session_fn,
    input_messages_key="input",
    history_messages_key="history",
)
```

## 6. Chains vs Agents

| 维度 | Chain（链） | Agent（智能体） |
|------|-----------|---------------|
| 流程控制 | 开发者预定义 | LLM 动态决策 |
| 可预测性 | 高 | 低 |
| 工具使用 | 不使用/固定 | 按需选择 |
| 适用场景 | 流程固定的任务 | 需要推理的复杂任务 |

**选型原则：** 能用 Chain 解决的就不要用 Agent。

## 7. Runnable 统一接口

所有 LangChain 组件实现 Runnable 接口：

```python
class Runnable:
    def invoke(self, input)    # 同步调用
    def ainvoke(self, input)   # 异步调用
    def stream(self, input)    # 流式输出
    def batch(self, inputs)    # 批量调用
```

高级组合模式：
- **RunnableParallel** — 并行执行多条链
- **RunnableBranch** — 条件路由
- **RunnableLambda** — 包装自定义函数
- **RunnablePassthrough** — 原样透传

## 8. 生态系统

| 组件 | 用途 |
|------|------|
| LangSmith | 可观测性/调试平台 |
| LangServe | 一行代码部署 REST API |
| LangGraph | 图结构 Agent 编排 |
| langchain-community | 社区集成 |

::: warning 需要本地运行
完整代码示例见 `langchain/chatbot.py`，包含 LCEL 管道、Prompt 模板、记忆机制的完整实现。
:::

---

::: tip 下一步
- [RAG 检索增强生成](/rag/) — 让 LLM 基于外部知识回答
- [流式输出](/engineering/streaming) — LangChain 的流式接口详解
- [Agent 智能体](/agent/) — 从 Chain 进阶到 Agent
:::
