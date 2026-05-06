# LangChain 框架完全知识手册

> 本文档是 LangChain 框架的系统性技术教科书，从设计哲学到核心组件，覆盖所有关键知识点。
> 配合 `chatbot.py` 代码阅读效果更佳。

---

## 目录

1. [LangChain 框架概述与设计哲学](#1-langchain-框架概述与设计哲学)
2. [LCEL 链式语法详解](#2-lcel-链式语法详解)
3. [Prompt Templates 提示词模板](#3-prompt-templates-提示词模板)
4. [Output Parsers 输出解析器](#4-output-parsers-输出解析器)
5. [Memory 机制](#5-memory-机制)
6. [Chains vs Agents 的区别](#6-chains-vs-agents-的区别)
7. [Document Loaders 文档加载器](#7-document-loaders-文档加载器)
8. [Text Splitters 文本切分器](#8-text-splitters-文本切分器)
9. [Callbacks 回调系统](#9-callbacks-回调系统)
10. [Runnable 接口与组合模式](#10-runnable-接口与组合模式)
11. [LangChain 生态](#11-langchain-生态)

---

## 1. LangChain 框架概述与设计哲学

### 1.1 什么是 LangChain

```
LangChain 是一个 LLM 应用开发框架，核心使命：

  "让开发者能用标准化的组件快速构建 LLM 应用"

用一个餐厅比喻（对应 chatbot.py 开头的注释）：

  ┌──────────────────────────────────────────────────┐
  │  开餐厅 = 开发 LLM 应用                          │
  │                                                  │
  │  食材（用户输入/数据）                            │
  │      ↓                                           │
  │  厨房流程（LangChain 框架）                       │
  │      ↓                                           │
  │  菜品（格式化的 AI 输出）                         │
  │                                                  │
  │  没有 LangChain：每次做菜都从磨刀、生火开始      │
  │  有 LangChain：标准化厨房，专注创新菜品           │
  └──────────────────────────────────────────────────┘
```

### 1.2 设计哲学

```
LangChain 的三大设计原则：

① 组合性（Composability）
   每个组件是独立的"积木块"，用 | 管道符自由组合。
   prompt | llm | parser  ← 三个积木拼成一条链

② 可观测性（Observability）
   每个组件的输入输出都可追踪、可调试。
   → 对应 Callbacks 系统和 LangSmith 平台

③ 可互换性（Interchangeability）
   换模型只需改一行代码，其余逻辑不变。
   ChatOpenAI → ChatAnthropic → ChatOllama  ← 接口一致
```

### 1.3 包结构（v0.2+）

```
LangChain 拆分为多个包，各司其职：

  langchain-core       核心抽象（Runnable, Prompt, Parser）
  langchain            高层封装（Chains, Agents）
  langchain-openai     OpenAI 集成
  langchain-anthropic  Anthropic 集成
  langchain-community  社区贡献的集成
  langgraph            图结构 Agent 框架
  langsmith            可观测性平台

对应 chatbot.py 中的导入：
  from langchain_openai import ChatOpenAI       # langchain-openai
  from langchain_core.prompts import ...        # langchain-core
  from langchain_core.output_parsers import ... # langchain-core
  from langchain_community.chat_message_histories import ... # langchain-community
```

---

## 2. LCEL 链式语法详解

### 2.1 LCEL 是什么

```
LCEL = LangChain Expression Language（LangChain 表达式语言）

核心思想：用 | 管道符将组件串联，就像 Linux 命令行管道：

  Linux:     cat file | grep "key" | sort | head -10
  LangChain: prompt   | llm       | parser

对应 chatbot.py 第2章：
  simple_chain = demo_prompt | llm | parser
```

### 2.2 数据流动原理

```
每个组件实现 Runnable 接口，定义了 invoke(input) → output：

  用户输入 {"topic": "Python", "question": "什么是列表？"}
      │
      ↓  prompt.invoke(input)
  ChatPromptValue([SystemMessage(...), HumanMessage(...)])
      │
      ↓  llm.invoke(messages)
  AIMessage(content="列表是...", response_metadata={...})
      │
      ↓  parser.invoke(ai_message)
  "列表是..."  ← 纯字符串

数据类型的变化：
  dict → ChatPromptValue → AIMessage → str
  每一步的输出类型 = 下一步的输入类型（类型必须匹配！）
```

### 2.3 LCEL 的高级组合

```
除了线性管道，LCEL 还支持：

① 并行（RunnableParallel）：
  from langchain_core.runnables import RunnableParallel

  parallel = RunnableParallel(
      summary=summary_chain,
      translation=translate_chain,
  )
  # 两条链并行执行，结果合并为 dict

② 条件分支（RunnableBranch）：
  from langchain_core.runnables import RunnableBranch

  branch = RunnableBranch(
      (lambda x: "代码" in x["question"], code_chain),
      (lambda x: "翻译" in x["question"], translate_chain),
      default_chain,  # 默认分支
  )

③ 传递（RunnablePassthrough）：
  from langchain_core.runnables import RunnablePassthrough

  # 原样传递输入，常用于保留原始数据
  chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
```

### 2.4 为什么 | 能工作

```
Python 魔法方法重载：

  class Runnable:
      def __or__(self, other):
          return RunnableSequence(self, other)

  prompt | llm  等价于  RunnableSequence(prompt, llm)

对应 chatbot.py 的注释：
  "这里的 | 不是'或'运算符！LangChain 重载了 | 运算符，
   让它变成了'管道'语义。"
```

---

## 3. Prompt Templates 提示词模板

### 3.1 为什么要用模板

```
对比原始 f-string：

  # 方式一：f-string（有问题）
  prompt = f"你是{role}专家，回答：{question}"
  # 问题：变量和模板混杂、无法序列化、无法复用

  # 方式二：PromptTemplate（推荐）
  prompt = ChatPromptTemplate.from_messages([
      ("system", "你是{role}专家"),
      ("human", "{question}"),
  ])
  # 优势：可复用、可序列化、可版本管理、与 LCEL 集成

对应 chatbot.py 第1章的详细解释。
```

### 3.2 ChatPromptTemplate

```
聊天场景最常用的模板，理解三种角色：

  ("system", "...")   → SystemMessage    AI 的"人设"，用户看不到
  ("human", "...")    → HumanMessage     用户说的话
  ("ai", "...")       → AIMessage        AI 说的话（用于 few-shot）

对应 chatbot.py：
  demo_prompt = ChatPromptTemplate.from_messages([
      ("system", "你是{topic}领域专家"),
      ("human", "{question}"),
  ])

模板变量用 {variable_name} 表示，invoke 时传入 dict 填充。
```

### 3.3 MessagesPlaceholder

```
在模板中为"消息列表"留位置，专门用于插入历史对话：

  chat_prompt = ChatPromptTemplate.from_messages([
      ("system", "你是助手"),
      MessagesPlaceholder(variable_name="history"),  ← 历史消息的"洞"
      ("human", "{input}"),
  ])

运行时，history 会被替换为实际的消息列表：
  [HumanMessage("你好"), AIMessage("你好！"), HumanMessage("天气"), ...]

位置规则（对应 chatbot.py 的避坑指南）：
  system → history → 最新 human
  ✗ 不要把 history 放在 system 之前
  ✗ 不要把 history 放在最新 human 之后
```

### 3.4 FewShotPromptTemplate

```
用于动态注入少样本示例：

  from langchain_core.prompts import FewShotChatMessagePromptTemplate

  examples = [
      {"input": "高兴", "output": "happy"},
      {"input": "悲伤", "output": "sad"},
  ]

  example_prompt = ChatPromptTemplate.from_messages([
      ("human", "{input}"),
      ("ai", "{output}"),
  ])

  few_shot = FewShotChatMessagePromptTemplate(
      example_prompt=example_prompt,
      examples=examples,
  )

  final_prompt = ChatPromptTemplate.from_messages([
      ("system", "你是翻译器"),
      few_shot,
      ("human", "{input}"),
  ])

高级用法：动态选择示例（ExampleSelector）
  根据用户输入的相似度，动态选择最相关的 few-shot 示例
```

---

## 4. Output Parsers 输出解析器

### 4.1 为什么需要 Parser

```
LLM 返回的是 AIMessage 对象：
  AIMessage(content="你好", response_metadata={...}, id="run-xxx")

但我们通常只需要里面的文本，甚至需要结构化数据。

对应 chatbot.py 第3章的对比演示：
  不带 parser → 得到 AIMessage 对象
  带 parser   → 得到纯字符串 "你好"
```

### 4.2 StrOutputParser

```
最简单的解析器：提取 AIMessage.content 文本

  from langchain_core.output_parsers import StrOutputParser

  chain = prompt | llm | StrOutputParser()
  result = chain.invoke(...)  # 返回 str 类型

对应 chatbot.py 中 parser = StrOutputParser() 的用法。
```

### 4.3 PydanticOutputParser

```
将 LLM 输出解析为 Pydantic 结构化对象：

  from langchain_core.output_parsers import PydanticOutputParser
  from pydantic import BaseModel, Field

  class MovieReview(BaseModel):
      title: str = Field(description="电影名称")
      rating: float = Field(description="评分 1-10")
      summary: str = Field(description="一句话总结")

  parser = PydanticOutputParser(pydantic_object=MovieReview)

  # parser.get_format_instructions() 会生成格式说明，
  # 告诉 LLM 必须输出什么样的 JSON
  prompt = ChatPromptTemplate.from_messages([
      ("system", "按要求输出。{format_instructions}"),
      ("human", "评价电影：{movie}"),
  ]).partial(format_instructions=parser.get_format_instructions())

  chain = prompt | llm | parser
  result = chain.invoke({"movie": "盗梦空间"})
  # result 是 MovieReview 对象：result.title, result.rating, result.summary
```

### 4.4 JsonOutputParser

```
比 PydanticOutputParser 更灵活，直接输出 dict：

  from langchain_core.output_parsers import JsonOutputParser

  parser = JsonOutputParser(pydantic_object=MovieReview)  # 可选 schema
  # 或者不传 schema，让 LLM 自由输出 JSON

支持流式解析（边接收边解析部分 JSON），
PydanticOutputParser 必须等完整输出后才能解析。
```

### 4.5 解析器对比

```
┌──────────────────────┬───────────────┬─────────────┬────────────────┐
│ 解析器               │ 输出类型      │ 是否支持流式 │ 适用场景       │
├──────────────────────┼───────────────┼─────────────┼────────────────┤
│ StrOutputParser      │ str           │ 是          │ 纯文本输出     │
│ JsonOutputParser     │ dict          │ 是(部分)    │ 灵活 JSON      │
│ PydanticOutputParser │ Pydantic 对象 │ 否          │ 强类型校验     │
│ CommaSeparatedList   │ List[str]     │ 否          │ 列表输出       │
│ XMLOutputParser      │ dict          │ 是          │ XML 格式       │
└──────────────────────┴───────────────┴─────────────┴────────────────┘
```

---

## 5. Memory 机制

### 5.1 为什么需要 Memory

```
LLM 本身是无状态的（对应 chatbot.py 第4章核心概念）：

  第一次调用: "我叫小明"    → AI: "你好小明！"
  第二次调用: "我叫什么？"  → AI: "我不知道你叫什么"  ← 忘了！

  原因：每次 API 调用都是独立的，LLM 不会自动记住之前的对话。

解决方案：手动把历史消息一起发送给 LLM：
  第二次调用时实际发送：
  [
    HumanMessage("我叫小明"),
    AIMessage("你好小明！"),
    HumanMessage("我叫什么？"),    ← 连带历史一起发
  ]
```

### 5.2 RunnableWithMessageHistory（推荐方案）

```
LangChain 0.2+ 官方推荐的记忆方案，核心思路是"分离关注点"：

  基础链（只负责：接收输入 → 调用 LLM → 返回输出）
       ↓  被包装
  RunnableWithMessageHistory（自动：取历史 → 注入 → 调用 → 存结果）

对应 chatbot.py：
  chain_with_memory = RunnableWithMessageHistory(
      runnable=base_chain,           # 基础链
      get_session_history=...,       # 从哪取历史
      input_messages_key="input",    # 输入变量名
      history_messages_key="history",# 历史占位符名
  )

调用时通过 config 传入 session_id：
  chain_with_memory.invoke(
      {"input": "你好"},
      config={"configurable": {"session_id": "user_001"}},
  )
```

### 5.3 历史存储方案

```
chatbot.py 使用内存字典（教学用）：
  store: dict[str, BaseChatMessageHistory] = {}

生产环境选择：
┌─────────────────────┬──────────────────────────────────┐
│ 存储方案            │ 适用场景                          │
├─────────────────────┼──────────────────────────────────┤
│ ChatMessageHistory  │ 内存存储，教学/原型（重启即丢失）  │
│ RedisChatHistory    │ 高性能，支持 TTL 自动过期         │
│ MongoDBChatHistory  │ 持久化，支持复杂查询              │
│ SQLChatMessageHistory│ 关系数据库存储                   │
│ FileChatMessageHistory│ 文件存储，简单但不适合高并发    │
└─────────────────────┴──────────────────────────────────┘
```

### 5.4 历史管理策略

```
随着对话变长，历史消息会占满上下文窗口。解决方案：

① 截断（Trim）：只保留最近 N 轮
  from langchain_core.messages import trim_messages
  trimmer = trim_messages(max_tokens=1000, strategy="last")

② 摘要压缩：用 LLM 总结旧对话
  "之前的对话总结：用户是一名 Python 开发者，正在学习 LangChain..."

③ Token 计数限制：保留最多 K 个 token 的历史
  按 token 数精确截断，比按轮次更精确
```

### 5.5 已废弃的 Memory 类（避坑）

```
⚠️ 以下类在 LangChain 0.2+ 已废弃，不要使用：
  - ConversationBufferMemory      → 用 RunnableWithMessageHistory
  - ConversationSummaryMemory     → 用 trim_messages + 自定义摘要
  - ConversationTokenBufferMemory → 用 trim_messages(max_tokens=...)

对应 chatbot.py 的避坑指南：
  "不要用老版本的 ConversationBufferMemory，它已被官方标为废弃！"
```

---

## 6. Chains vs Agents 的区别

### 6.1 核心区别

```
Chain（链）：固定流程，开发者预定义每一步

  用户输入 → 步骤A → 步骤B → 步骤C → 输出

  流程是确定的，不会变。就像工厂流水线。

Agent（智能体）：动态决策，LLM 自己决定下一步

  用户输入 → LLM 思考 → 需要查天气？→ 调用天气工具
                      → 需要计算？  → 调用计算器
                      → 信息够了？  → 生成最终回答

  流程是不确定的，取决于 LLM 的判断。就像有经验的员工。
```

### 6.2 对比表

```
┌───────────────┬─────────────────────┬─────────────────────────┐
│ 维度          │ Chain               │ Agent                    │
├───────────────┼─────────────────────┼─────────────────────────┤
│ 流程控制      │ 开发者预定义        │ LLM 动态决策             │
│ 可预测性      │ 高（确定性流程）    │ 低（LLM 可能走不同路径） │
│ 工具使用      │ 不使用/固定使用     │ 按需选择使用             │
│ 实现复杂度    │ 低                  │ 中~高                    │
│ 调试难度      │ 低                  │ 高（路径不确定）         │
│ 适用场景      │ 流程固定的任务      │ 需要推理和工具的复杂任务 │
│ 对应 LCEL     │ prompt | llm | parser│ create_tool_calling_agent│
│ 流式输出      │ 逐 token            │ 逐步骤（默认）           │
└───────────────┴─────────────────────┴─────────────────────────┘
```

### 6.3 选型指南

```
你的任务需要 Agent 吗？问自己：

  Q1: 任务流程是否固定？
      是 → 用 Chain
      否 → 继续判断

  Q2: 是否需要根据情况调用不同工具？
      是 → 用 Agent
      否 → 用 Chain

  Q3: 是否需要多步推理？
      是 → 用 Agent（或 LangGraph）
      否 → 用 Chain

经验法则：
  "能用 Chain 解决的就不要用 Agent"
  Agent 引入了不确定性和额外成本（多次 LLM 调用）。
```

---

## 7. Document Loaders 文档加载器

### 7.1 概念

```
Document Loader 的职责：把各种格式的文件加载为统一的 Document 对象

  PDF / Word / HTML / CSV / 数据库 / 网页 / ...
       ↓ Document Loader
  List[Document(page_content="...", metadata={...})]

每个 Document 包含：
  - page_content: str   文本内容
  - metadata: dict      元信息（来源、页码、作者...）
```

### 7.2 常用加载器

```
┌────────────────────────────┬──────────────────────────────────────┐
│ 加载器                     │ 用途                                  │
├────────────────────────────┼──────────────────────────────────────┤
│ PyPDFLoader                │ PDF 文件（按页分割）                  │
│ Docx2txtLoader             │ Word 文档                            │
│ CSVLoader                  │ CSV 文件（每行一个 Document）         │
│ WebBaseLoader              │ 网页爬取                              │
│ DirectoryLoader            │ 批量加载文件夹内所有文件              │
│ UnstructuredFileLoader     │ 自动识别格式的通用加载器              │
│ TextLoader                 │ 纯文本文件                            │
│ JSONLoader                 │ JSON 文件（支持 jq 表达式选取字段）    │
└────────────────────────────┴──────────────────────────────────────┘

使用示例：
  from langchain_community.document_loaders import PyPDFLoader

  loader = PyPDFLoader("report.pdf")
  documents = loader.load()  # List[Document]

  for doc in documents:
      print(f"页 {doc.metadata['page']}: {doc.page_content[:100]}...")
```

---

## 8. Text Splitters 文本切分器

### 8.1 为什么要切分

```
LLM 有上下文窗口限制，一个长文档不能直接全部塞进去。
需要把长文档切成合适大小的"块"（chunk）。

  ┌─────────────────────────────────────────────┐
  │  100 页 PDF（50000 tokens）                  │
  │  ↓ Text Splitter                            │
  │  [chunk_1(500t), chunk_2(500t), ... chunk_100(500t)]│
  │  ↓ Embedding + 向量检索                     │
  │  只把相关的 3-5 个 chunk 发给 LLM            │
  └─────────────────────────────────────────────┘
```

### 8.2 切分策略对比

```
┌─────────────────────────────┬──────────────────────────────────┐
│ Splitter                    │ 策略                              │
├─────────────────────────────┼──────────────────────────────────┤
│ CharacterTextSplitter       │ 按字符数切分（最简单）            │
│ RecursiveCharacterSplitter  │ 递归按分隔符切分（最常用）        │
│ TokenTextSplitter           │ 按 token 数切分（更精确）         │
│ MarkdownHeaderSplitter      │ 按 Markdown 标题层级切分          │
│ HTMLHeaderTextSplitter      │ 按 HTML 标题层级切分              │
│ SemanticChunker             │ 按语义相似度切分（最智能）        │
└─────────────────────────────┴──────────────────────────────────┘

最常用：RecursiveCharacterTextSplitter
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,        # 每块最大字符数
      chunk_overlap=200,      # 相邻块重叠字符数（保持上下文连续性）
      separators=["\n\n", "\n", "。", " ", ""],  # 优先按段落切
  )
  chunks = splitter.split_documents(documents)
```

### 8.3 chunk_overlap 的作用

```
没有重叠：信息可能被切断
  chunk_1: "...量子纠缠是一种"
  chunk_2: "物理现象，两个粒子..."
  → 检索到 chunk_2 时，缺少"量子纠缠"这个关键信息

有重叠（overlap=200）：
  chunk_1: "...量子纠缠是一种物理现象，两个粒子会..."
  chunk_2: "量子纠缠是一种物理现象，两个粒子会产生关联..."
  → 重叠部分确保上下文不断裂
```

---

## 9. Callbacks 回调系统

### 9.1 回调机制

```
Callbacks 让你能"监听"链运行过程中的每一个事件：

  chain.invoke(input)
      │
      ├── on_chain_start(inputs)     链开始
      │     ├── on_llm_start(...)    LLM 开始
      │     ├── on_llm_new_token(t)  LLM 产出 token（流式）
      │     ├── on_llm_end(output)   LLM 结束
      │     ├── on_tool_start(...)   工具开始
      │     └── on_tool_end(...)     工具结束
      └── on_chain_end(output)       链结束

用途：日志记录、计费统计、性能监控、流式输出通知
```

### 9.2 自定义回调处理器

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[日志] LLM 开始推理...")

    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        print(f"\n[日志] LLM 推理完毕，用时 {response.llm_output}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[日志] 工具调用: {serialized['name']}")

# 使用：
chain.invoke(input, config={"callbacks": [MyHandler()]})
```

### 9.3 内置回调处理器

```
┌──────────────────────────┬────────────────────────────────┐
│ 处理器                   │ 用途                            │
├──────────────────────────┼────────────────────────────────┤
│ StdOutCallbackHandler    │ 打印到控制台（verbose=True）    │
│ StreamingStdOutHandler   │ 流式打印 LLM 输出              │
│ LangChainTracer          │ 发送到 LangSmith 平台          │
│ WandbTracer              │ 发送到 Weights & Biases        │
│ FileCallbackHandler      │ 写入日志文件                    │
└──────────────────────────┴────────────────────────────────┘
```

---

## 10. Runnable 接口与组合模式

### 10.1 Runnable 统一接口

```
LangChain 中所有组件都实现 Runnable 接口，这是框架的基石：

  class Runnable:
      def invoke(self, input)           # 同步调用
      def ainvoke(self, input)          # 异步调用
      def stream(self, input)           # 同步流式
      def astream(self, input)          # 异步流式
      def batch(self, inputs)           # 批量调用
      def abatch(self, inputs)          # 异步批量

  实现了 Runnable 的组件：
    ChatPromptTemplate  ✓
    ChatOpenAI          ✓
    StrOutputParser     ✓
    RunnableSequence    ✓ （管道组合后的链）
    RunnableLambda      ✓ （自定义函数包装）
    ...

这意味着：任何组件都能用 .invoke()、.stream() 等方法调用！
```

### 10.2 组合模式

```
① RunnableSequence（顺序执行）：
   chain = a | b | c       # a 的输出喂给 b，b 的输出喂给 c

② RunnableParallel（并行执行）：
   parallel = RunnableParallel(key1=chain_a, key2=chain_b)
   # 同时执行 chain_a 和 chain_b，结果合并为 {"key1": ..., "key2": ...}

③ RunnableLambda（自定义函数）：
   from langchain_core.runnables import RunnableLambda

   def custom_fn(input_str):
       return input_str.upper()

   chain = prompt | llm | parser | RunnableLambda(custom_fn)

④ RunnablePassthrough（透传）：
   # 保持输入不变地传递下去
   chain = {"original": RunnablePassthrough(), "processed": some_chain} | ...

⑤ RunnableBranch（条件路由）：
   branch = RunnableBranch(
       (condition_fn_1, chain_1),
       (condition_fn_2, chain_2),
       default_chain,
   )
```

### 10.3 RunnableLambda 实战

```
将任意 Python 函数包装为 Runnable，使其能加入管道：

  from langchain_core.runnables import RunnableLambda

  # 包装同步函数
  def add_metadata(text: str) -> dict:
      return {"text": text, "length": len(text), "timestamp": "2024-01-01"}

  chain = prompt | llm | parser | RunnableLambda(add_metadata)

  # 包装异步函数
  async def async_process(text: str) -> str:
      await asyncio.sleep(0.1)
      return text.strip()

  chain = prompt | llm | parser | RunnableLambda(async_process)
```

---

## 11. LangChain 生态

### 11.1 LangSmith — 可观测性平台

```
LangSmith 是 LangChain 官方的可观测性和调试平台：

  功能：
  ┌────────────────────────────────────────────────┐
  │ ① Tracing（链路追踪）                          │
  │   每次调用的完整执行树：输入、输出、延迟、成本  │
  │                                                │
  │ ② Evaluation（评估）                           │
  │   自动化评估 LLM 输出质量                       │
  │                                                │
  │ ③ Monitoring（监控）                           │
  │   实时监控：成功率、延迟分布、成本趋势          │
  │                                                │
  │ ④ Playground（调试台）                         │
  │   可视化修改 Prompt 并即时看到效果              │
  │                                                │
  │ ⑤ Dataset（数据集）                            │
  │   管理测试用例，做回归测试                      │
  └────────────────────────────────────────────────┘

启用方式：
  export LANGCHAIN_TRACING_V2=true
  export LANGCHAIN_API_KEY="ls-xxx"
  # 代码不用改！LangChain 自动发送 trace 数据
```

### 11.2 LangServe — API 部署

```
一行代码将 LangChain 链部署为 REST API：

  from langserve import add_routes
  from fastapi import FastAPI

  app = FastAPI()
  add_routes(app, my_chain, path="/my-chain")

  # 自动生成：
  # POST /my-chain/invoke       → 阻塞调用
  # POST /my-chain/stream       → 流式调用
  # POST /my-chain/batch        → 批量调用
  # GET  /my-chain/playground   → 可视化调试界面

注意：LangServe 适合快速原型，生产环境建议用 FastAPI 自己封装，
     以获得更细粒度的控制（认证、限流、自定义错误处理等）。
```

### 11.3 LangGraph — 图结构 Agent

```
LangGraph 是 LangChain 团队推出的"下一代 Agent 框架"：

  传统 Agent（AgentExecutor）：
    LLM → 工具 → LLM → 工具 → ... → 最终回答
    线性循环，难以实现复杂流程

  LangGraph Agent：
    用"图"（Graph）定义 Agent 的状态和流转：

        ┌──────────┐
        │  START   │
        └────┬─────┘
             ↓
        ┌──────────┐     ┌──────────┐
        │ 推理节点  │────→│ 工具节点  │
        └──────────┘←────└──────────┘
             │
             ↓ (完成)
        ┌──────────┐
        │   END    │
        └──────────┘

  优势：支持循环、条件分支、人工审核、持久化状态
  → 对应项目 langgraph/ 文件夹
```

---

## 附录 A：本文件夹代码与知识点对应

| 代码文件 | 覆盖的知识点 | 对应本文档章节 |
|---------|-------------|---------------|
| `chatbot.py` | LCEL管道、PromptTemplate、StrOutputParser、RunnableWithMessageHistory | 第2-5节 |

---

## 附录 B：常见问题与避坑指南

```
Q1: ConversationBufferMemory 还能用吗？
A: 已废弃！请用 RunnableWithMessageHistory。参见第5.5节。

Q2: | 管道符报错 "unsupported operand type"？
A: 确保两侧都是 Runnable 对象。普通函数需要用 RunnableLambda 包装。

Q3: 历史消息太长导致超出上下文窗口？
A: 使用 trim_messages 截断，或实现摘要压缩策略。参见第5.4节。

Q4: stream() 输出类型不对？
A: 带 StrOutputParser → chunk 是 str
   不带 → chunk 是 AIMessageChunk
   参见 chatbot.py 第3章。

Q5: Agent 和 Chain 怎么选？
A: 能用 Chain 就别用 Agent。参见第6.3节选型指南。
```

---

> **下一步学习**：前往 `streaming/KNOWLEDGE.md` 了解流式输出的完整实现方案，然后阅读 `agent/` 和 `langgraph/` 进入 Agent 世界。
