# LLM Cookbook — 大模型应用开发实战手册

> 从零基础到企业级 LLM 应用开发，21 个渐进式实战项目

---

## 学习路线总览

```
基础篇 (项目 0-1)        → 掌握 LLM 调用和 LangChain 核心组件
RAG 篇 (项目 2-5)        → 从基础 RAG 到高级检索策略
Agent 篇 (项目 3-4)      → ReAct Agent + 多 Agent 协作
应用篇 (项目 6-8)        → 结构化输出、工具调用、流式处理
工程篇 (项目 9-15)       → 评估、API服务、异步、持久化、安全、监控、容错
进阶篇 (项目 16-18)      → 高级记忆、高级RAG、人机协作
```

---

## 项目清单

### 一、基础篇 — 入门 LLM 应用开发

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 0 | 原生 API 调用 | `llm/native_api.py` | OpenAI SDK、阻塞/流式调用、Response 结构解析 |
| 1 | LangChain 核心 | `langchain/chatbot.py` | ChatOpenAI、PromptTemplate、LCEL 管道、OutputParser、MessageHistory |

**学完你能：** 用 Python 调用 LLM，理解 Prompt→Model→Output 的完整链路

---

### 二、RAG 篇 — 检索增强生成

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 2 | 基础 RAG | `rag/rag_qa.py` | 文档加载、文本分块、Embedding、FAISS 向量检索、RAG Chain |
| 5 | 高级 RAG | `rag/advanced_rag.py` | 元数据过滤、来源追踪、多文档检索、上下文压缩 |
| 17 | RAG 进阶策略 | `rag_advanced/rag_strategies.py` | HyDE 假设性文档、Multi-Query 多角度扩展、Parent-Child 父子文档 |

**学完你能：** 构建生产级知识问答系统，解决语义鸿沟、召回率不足等实际问题

---

### 三、Agent 篇 — 自主决策的 AI

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 3 | ReAct Agent | `agent/react_agent.py` | @tool 定义工具、ReAct 思维链、Agent 执行器 |
| 4 | 多 Agent 协作 | `langgraph/media_studio.py` | LangGraph 状态图、多节点协作、条件路由 |
| 7 | Tool Calling Agent | `agent/tool_calling_agent.py` | 原生函数调用、工具绑定、结构化工具参数 |

**学完你能：** 让 AI 自主调用工具、拆解任务、多角色协作完成复杂目标

---

### 四、应用篇 — 实用功能模块

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 6 | 结构化输出 | `structured_output/extraction.py` | Pydantic 模型、with_structured_output、实体提取 |
| 8 | 流式处理 | `streaming/stream_demo.py` | stream()、astream_events、逐 token 输出、流式 RAG |

**学完你能：** 让 LLM 输出结构化 JSON，实现打字机效果的实时响应

---

### 五、工程篇 — 生产级 AI 应用必备

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 9 | RAG 评估 | `evaluation/rag_eval.py` | 忠实度、相关性、完整性、LLM-as-Judge 评估框架 |
| 10 | API 服务化 | `api_service/fastapi_server.py` | FastAPI、LangServe、REST API、流式端点 |
| 11 | 异步并发 | `async_concurrent/async_demo.py` | async/await、asyncio.gather、Semaphore 限流、超时控制 |
| 12 | 向量库持久化 | `vectordb/persistent_store.py` | FAISS save/load、Chroma CRUD、元数据过滤、持久化策略 |
| 13 | 安全护栏 | `guardrails/safety_guard.py` | 输入防注入、System Prompt 加固、输出脱敏、三层防护链 |
| 14 | 可观测性 | `observability/tracing_demo.py` | 自定义 Callback、链路追踪、JSON 结构化日志、Token 成本监控 |
| 15 | 容错机制 | `error_handling/retry_demo.py` | .with_retry()、指数退避、.with_fallbacks()、完整容错包装器 |

**学完你能：** 将 AI 应用部署到生产环境，具备监控、安全、容错、评估能力

---

### 六、进阶篇 — 高级架构与模式

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 16 | Memory 进阶 | `memory_advanced/memory_strategies.py` | 窗口记忆、摘要记忆、向量长期记忆、组合记忆架构 |
| 17 | RAG 进阶策略 | `rag_advanced/rag_strategies.py` | HyDE、Multi-Query、Parent-Child、策略组合 |
| 18 | LangGraph 进阶 | `langgraph_advanced/hitl_checkpoint.py` | Checkpointing 状态持久化、Human-in-the-Loop、interrupt_before |

**学完你能：** 构建有长期记忆的 AI、高质量检索系统、需要人工审批的 AI 工作流

---

## 推荐学习顺序

```
Week 1:  项目 0 → 1 → 2           (基础：API → LangChain → RAG)
Week 2:  项目 3 → 4 → 5           (进阶：Agent → 多Agent → 高级RAG)
Week 3:  项目 6 → 7 → 8           (应用：结构化输出 → 工具调用 → 流式)
Week 4:  项目 9 → 10 → 11 → 12    (工程：评估 → API → 异步 → 持久化)
Week 5:  项目 13 → 14 → 15        (生产：安全 → 监控 → 容错)
Week 6:  项目 16 → 17 → 18        (高阶：记忆 → RAG策略 → HITL)
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| LLM 框架 | LangChain >= 0.3.0、LangGraph >= 0.6.0 |
| 模型调用 | langchain-openai (兼容 OpenAI API 的网关) |
| 向量化 | HuggingFace Embeddings (BAAI/bge-small-zh-v1.5) |
| 向量数据库 | FAISS (内存)、Chroma (持久化) |
| API 服务 | FastAPI + LangServe + Uvicorn |
| Python | 3.9+ |

---

## 运行方式

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行任意项目（每个文件独立可运行）
python llm/native_api.py          # 项目 0
python langchain/chatbot.py       # 项目 1
python rag/rag_qa.py              # 项目 2
# ... 以此类推
```

每个 `.py` 文件都是**自包含**的：
- 顶部有详细的前置科普（概念解释 + 类比 + ASCII 图）
- 代码逐章执行，print 输出展示每一步的数据流
- 从上到下顺序运行即可，无需额外配置

---

## 项目特色

1. **保姆级注释** — 每个概念都有"为什么→是什么→怎么做"三段式讲解
2. **类比教学** — 用生活比喻解释抽象概念（LLM=厨师、RAG=开卷考试）
3. **可视化输出** — 大量 print + ASCII 表格，运行即可看到完整数据流
4. **渐进式设计** — 每个项目复用前面学到的知识，难度逐步递增
5. **生产导向** — 工程篇覆盖真实生产环境需要的所有基础设施
