# LLM Cookbook — 大模型应用开发完全手册

> 从零基础到企业级 LLM 应用开发，系统性掌握 LLM/Agent/RAG 全技术栈
> 每个模块 = KNOWLEDGE.md（理论知识） + .py（实战代码）

---

## 学习路线总览

```
基础篇 (项目 0-1)        → 掌握 LLM 原理、API 调用和 LangChain 核心组件
RAG 篇 (项目 2-5)        → 从基础 RAG 到高级检索策略（CRAG、Self-RAG）
Agent 篇 (项目 3-4)      → ReAct Agent + 多 Agent 协作 + LangGraph 工作流
应用篇 (项目 6-8)        → 结构化输出、工具调用、流式处理
工程篇 (项目 9-15)       → 评估、API服务、异步、持久化、安全、监控、容错
进阶篇 (项目 16-18)      → 高级记忆、高级RAG、人机协作
```

---

## 项目结构说明

每个文件夹包含两部分：

```
folder/
├── KNOWLEDGE.md        ← 📖 完整知识文档（教科书级，覆盖该主题所有知识点）
├── xxx.py              ← 💻 实战代码（可独立运行，带详细注释）
└── (yyy.py)            ← 💻 补充代码（覆盖更多实战场景）
```

**建议学习方式：** 先读 KNOWLEDGE.md 建立完整知识体系 → 再运行 .py 代码动手实践

---

## 项目清单

### 一、基础篇 — LLM 原理与开发入门

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 0 | LLM 原理与 API | `llm/` | Transformer 架构、注意力机制、生成机制、微调、对齐、提示工程 |

**代码文件：**
| 文件 | 内容 |
|------|------|
| `llm/native_api.py` | OpenAI SDK、阻塞/流式调用、Response 结构解析 |
| `llm/prompt_engineering.py` | Few-shot、CoT 思维链、角色设定、格式控制 |
| `llm/tokenization_demo.py` | BPE 算法、tiktoken、Token 计数与成本计算 |
| `llm/generation_strategies.py` | Temperature、Top-P、频率惩罚、停止条件实验 |

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 1 | LangChain 核心 | `langchain/` | LCEL 管道、PromptTemplate、OutputParser、Memory、Callbacks |

**代码文件：**
| 文件 | 内容 |
|------|------|
| `langchain/chatbot.py` | ChatOpenAI、LCEL 链式语法、MessageHistory 多轮对话 |

**学完你能：** 理解 LLM 底层原理（Transformer → 生成 → 对齐），用 Python 调用 LLM，掌握提示工程核心技巧

---

### 二、RAG 篇 — 检索增强生成

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 2 | 基础 RAG | `rag/` | RAG 完整流程、Embedding、向量检索、上下文构造 |

**代码文件：**
| 文件 | 内容 |
|------|------|
| `rag/rag_qa.py` | 文档加载、文本分块、FAISS 向量检索、RAG Chain |
| `rag/advanced_rag.py` | 元数据过滤、来源追踪、上下文压缩 |

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 5 | RAG 高级策略 | `rag_advanced/` | 语义鸿沟、HyDE、Multi-Query、CRAG、Self-RAG、混合检索 |

**代码文件：**
| 文件 | 内容 |
|------|------|
| `rag_advanced/rag_strategies.py` | HyDE 假设性文档、Multi-Query、Parent-Child 父子文档 |
| `rag_advanced/advanced_strategies.py` | 混合检索(BM25+Vector)、重排序、CRAG、Self-RAG |

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 12 | 向量数据库 | `vectordb/` | 索引算法(HNSW/IVF)、FAISS/Chroma 对比、持久化、CRUD |

**代码文件：**
| 文件 | 内容 |
|------|------|
| `vectordb/persistent_store.py` | FAISS save/load、Chroma CRUD、元数据过滤 |

**学完你能：** 构建生产级知识问答系统，解决语义鸿沟、召回率不足等实际问题

---

### 三、Agent 篇 — 自主决策的 AI

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 3 | Agent 系统 | `agent/` | ReAct 框架、Tool Calling、多 Agent 协作、Agentic Systems |

**代码文件：**
| 文件 | 内容 |
|------|------|
| `agent/react_agent.py` | @tool 定义工具、ReAct 思维链、AgentExecutor |
| `agent/tool_calling_agent.py` | 原生函数调用、工具绑定、结构化工具参数 |

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 4 | LangGraph 工作流 | `langgraph/` | 图计算、StateGraph、条件路由、循环反馈、多 Agent 编排 |

**代码文件：**
| 文件 | 内容 |
|------|------|
| `langgraph/media_studio.py` | 状态图、多节点协作、条件路由、反馈循环 |

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 18 | LangGraph 进阶 | `langgraph_advanced/` | HITL 人机协作、检查点、时间旅行、动态路由 |

**代码文件：**
| 文件 | 内容 |
|------|------|
| `langgraph_advanced/hitl_checkpoint.py` | Checkpointing、Human-in-the-Loop、interrupt_before |

**学完你能：** 让 AI 自主调用工具、拆解任务、多角色协作，构建需要人工审批的工作流

---

### 四、应用篇 — 实用功能模块

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 6 | 结构化输出 | `structured_output/` | Pydantic 模型、Function Calling、嵌套类型、验证器 |

**代码文件：**
| 文件 | 内容 |
|------|------|
| `structured_output/extraction.py` | with_structured_output、实体提取、复杂嵌套结构 |

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 8 | 流式处理 | `streaming/` | SSE 协议、异步流、事件驱动、前端集成 |

**代码文件：**
| 文件 | 内容 |
|------|------|
| `streaming/stream_demo.py` | stream()、astream_events、LangGraph 流式模式 |

**学完你能：** 让 LLM 输出结构化 JSON，实现打字机效果的实时响应

---

### 五、工程篇 — 生产级 AI 应用必备

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 9 | RAG 评估 | `evaluation/` | BLEU/ROUGE、LLM-as-Judge、RAGAS、A/B 测试 |
| 10 | API 服务化 | `api_service/` | FastAPI、SSE 流式端点、认证鉴权、负载均衡、部署 |
| 11 | 异步并发 | `async_concurrent/` | asyncio、并发调用、信号量限流、速率限制、批处理 |
| 13 | 安全护栏 | `guardrails/` | 提示注入防御、越狱检测、PII 脱敏、红队测试 |
| 14 | 可观测性 | `observability/` | 三支柱、LangSmith/LangFuse、OpenTelemetry、成本监控 |
| 15 | 容错机制 | `error_handling/` | 重试策略、断路器、降级方案、Tenacity |
| 19 | 缓存优化 | `caching/` | 精确/语义缓存、Redis、Prompt Caching、成本节省 |

**代码文件：**
| 文件 | 内容 |
|------|------|
| `evaluation/rag_eval.py` | 忠实度、相关性、完整性评估框架 |
| `api_service/fastapi_server.py` | FastAPI LLM 服务、流式端点 |
| `api_service/test_client.py` | API 客户端测试 |
| `async_concurrent/async_demo.py` | async/await、asyncio.gather、Semaphore 限流 |
| `guardrails/safety_guard.py` | 输入防注入、System Prompt 加固、输出脱敏 |
| `observability/tracing_demo.py` | 自定义 Callback、链路追踪、Token 成本监控 |
| `error_handling/retry_demo.py` | .with_retry()、指数退避、.with_fallbacks() |
| `caching/cache_demo.py` | InMemoryCache、语义缓存、缓存策略 |

**学完你能：** 将 AI 应用部署到生产环境，具备监控、安全、容错、评估、缓存能力

---

### 六、进阶篇 — 高级架构与模式

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 16 | Memory 进阶 | `memory_advanced/` | 窗口/摘要/向量/实体记忆、多层架构、MemGPT |
| 20 | 自我反思 | `self_reflection/` | Reflexion、自评估、迭代改进、元认知 |

**代码文件：**
| 文件 | 内容 |
|------|------|
| `memory_advanced/memory_strategies.py` | 窗口记忆、摘要记忆、向量长期记忆、组合架构 |
| `self_reflection/reflection_agent.py` | 生成-批评-改进循环、质量自评 |

**学完你能：** 构建有长期记忆的 AI、能自我改进的 Agent

---

## 知识文档索引

每个 KNOWLEDGE.md 都是一份独立的教科书级文档，可单独阅读：

| 文件 | 页数(约) | 核心主题 |
|------|---------|---------|
| `llm/KNOWLEDGE.md` | 40+ | Transformer、注意力、位置编码、生成机制、微调(LoRA/QLoRA)、对齐(RLHF/DPO)、提示工程 |
| `langchain/KNOWLEDGE.md` | 25+ | LCEL 语法、Prompt Templates、Output Parsers、Runnable 接口、生态系统 |
| `streaming/KNOWLEDGE.md` | 20+ | SSE 协议、WebSocket 对比、异步流、LangGraph 流式、前端实现 |
| `rag/KNOWLEDGE.md` | 25+ | RAG 流程、Embedding 原理、切分策略、检索器、评估指标 |
| `rag_advanced/KNOWLEDGE.md` | 30+ | HyDE、Multi-Query、混合检索、Reranking、CRAG、Self-RAG、Graph RAG |
| `vectordb/KNOWLEDGE.md` | 25+ | HNSW/IVF/PQ 索引、FAISS/Chroma/Pinecone 对比、量化、分布式 |
| `agent/KNOWLEDGE.md` | 25+ | ReAct、Tool Calling 协议、多 Agent 协作、Agentic Systems 演进 |
| `langgraph/KNOWLEDGE.md` | 25+ | 图计算范式、StateGraph、条件边、并行执行、子图 |
| `langgraph_advanced/KNOWLEDGE.md` | 25+ | HITL、检查点持久化、时间旅行、动态路由、LangGraph Platform |
| `memory_advanced/KNOWLEDGE.md` | 25+ | 记忆分类学、多层架构、检索策略、MemGPT、生产方案 |
| `self_reflection/KNOWLEDGE.md` | 20+ | Reflexion、元认知、多 Agent 互评、停止条件 |
| `evaluation/KNOWLEDGE.md` | 25+ | BLEU/ROUGE/BERTScore、LLM-as-Judge、RAGAS、EDD |
| `guardrails/KNOWLEDGE.md` | 25+ | 三层防护、提示注入、越狱、PII、红队测试、合规 |
| `api_service/KNOWLEDGE.md` | 25+ | FastAPI、认证鉴权、限流、负载均衡、Docker/K8s 部署 |
| `async_concurrent/KNOWLEDGE.md` | 20+ | asyncio、并发模式、信号量、速率限制、生产者-消费者 |
| `caching/KNOWLEDGE.md` | 20+ | 精确/语义缓存、Redis、TTL/LRU、Prompt Caching |
| `error_handling/KNOWLEDGE.md` | 20+ | 重试策略、断路器、降级方案、Tenacity、幂等性 |
| `observability/KNOWLEDGE.md` | 25+ | 三支柱、LangSmith/LangFuse、OpenTelemetry、成本监控 |
| `structured_output/KNOWLEDGE.md` | 20+ | Function Calling 机制、Pydantic、验证器、流式提取 |

---

## 推荐学习顺序

```
Week 1:  llm/ → langchain/                (基础：LLM 原理 → 框架使用)
Week 2:  rag/ → vectordb/                 (检索：RAG 流程 → 向量存储)
Week 3:  rag_advanced/                     (进阶检索：HyDE → CRAG → Self-RAG)
Week 4:  agent/ → langgraph/              (Agent：ReAct → 图工作流)
Week 5:  structured_output/ → streaming/   (应用：结构化输出 → 流式处理)
Week 6:  evaluation/ → guardrails/         (质量：评估 → 安全)
Week 7:  api_service/ → async_concurrent/  (部署：服务化 → 并发)
Week 8:  error_handling/ → caching/ → observability/  (运维：容错 → 缓存 → 监控)
Week 9:  memory_advanced/ → self_reflection/ → langgraph_advanced/  (高阶)
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| LLM 框架 | LangChain >= 0.3.0、LangGraph >= 0.6.0 |
| 模型调用 | OpenAI SDK (兼容 OpenAI API 格式的网关) |
| 向量化 | HuggingFace Embeddings (BAAI/bge-small-zh-v1.5) |
| 向量数据库 | FAISS (内存)、Chroma (持久化) |
| API 服务 | FastAPI + Uvicorn |
| 分词工具 | tiktoken |
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
python llm/native_api.py              # API 基础调用
python llm/prompt_engineering.py      # 提示工程实战
python llm/tokenization_demo.py       # 分词与 Token
python llm/generation_strategies.py   # 生成策略实验
python langchain/chatbot.py           # LangChain 对话
python rag/rag_qa.py                  # 基础 RAG
python rag_advanced/rag_strategies.py # HyDE/Multi-Query
python rag_advanced/advanced_strategies.py  # CRAG/Self-RAG
# ... 以此类推
```

每个 `.py` 文件都是**自包含**的：
- 顶部有详细的前置科普（概念解释 + 类比 + ASCII 图）
- 代码逐章执行，print 输出展示每一步的数据流
- 从上到下顺序运行即可，无需额外配置

---

## 项目特色

1. **理论+实战双轨制** — 每个模块都有 KNOWLEDGE.md（完整理论）+ .py（可运行代码）
2. **教科书级深度** — 从数学公式到工程实现，系统性覆盖所有知识点
3. **保姆级注释** — 每个概念都有"为什么→是什么→怎么做"三段式讲解
4. **类比教学** — 用生活比喻解释抽象概念（Transformer=全班同学互相看、RAG=开卷考试）
5. **可视化输出** — 大量 ASCII 图表 + print 输出，运行即可看到完整数据流
6. **渐进式设计** — 每个项目复用前面学到的知识，难度逐步递增
7. **生产导向** — 工程篇覆盖真实生产环境需要的所有基础设施
8. **全栈覆盖** — 从 Transformer 底层原理到 K8s 部署，一个项目掌握全部
