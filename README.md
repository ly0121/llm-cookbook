# LLM Cookbook — 大模型应用开发实战手册

> 从零基础到企业级 LLM 应用开发，21 个渐进式实战项目

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.0+-green.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6.0+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

---

## 🎯 项目简介

这是一个系统化的 LLM 应用开发学习项目，通过 21 个渐进式实战项目，带你从零基础掌握大模型应用开发的核心技能，直至具备构建企业级 AI 应用的能力。

**核心理念：** 不只是调用 API，而是掌握构建生产级 AI 应用的完整工程能力。

---

## 📚 学习路线总览

```
基础篇 (项目 0-1)        → 掌握 LLM 调用和 LangChain 核心组件
RAG 篇 (项目 2-5)        → 从基础 RAG 到高级检索策略
Agent 篇 (项目 3-4)      → ReAct Agent + 多 Agent 协作
应用篇 (项目 6-8)        → 结构化输出、工具调用、流式处理
工程篇 (项目 9-15)       → 评估、API服务、异步、持久化、安全、监控、容错
进阶篇 (项目 16-18)      → 高级记忆、高级RAG、人机协作
```

---

## 🚀 快速开始

### 环境准备

```bash
# 克隆项目
git clone https://github.com/ly0121/llm-cookbook.git
cd llm-cookbook

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置 API 密钥

```bash
# 设置环境变量
export OPENAI_API_KEY="your-api-key-here"
# 或创建 .env 文件
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

---

## 📖 项目清单

### 一、基础篇 — 入门 LLM 应用开发

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 0 | 原生 API 调用 | `llm/native_api.py` | OpenAI SDK、阻塞/流式调用、Response 结构解析 |
| 1 | LangChain 核心 | `langchain/chatbot.py` | ChatOpenAI、PromptTemplate、LCEL 管道、OutputParser、MessageHistory |

**学完你能：** 用 Python 调用 LLM，理解 Prompt→Model→Output 的完整链路

### 二、RAG 篇 — 检索增强生成

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 2 | 基础 RAG | `rag/rag_qa.py` | 文档加载、文本分块、Embedding、FAISS 向量检索、RAG Chain |
| 5 | 高级 RAG | `rag/advanced_rag.py` | 元数据过滤、来源追踪、多文档检索、上下文压缩 |
| 17 | RAG 进阶策略 | `rag_advanced/rag_strategies.py` | HyDE 假设性文档、Multi-Query 多角度扩展、Parent-Child 父子文档 |

**学完你能：** 构建生产级知识问答系统，解决语义鸿沟、召回率不足等实际问题

### 三、Agent 篇 — 自主决策的 AI

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 3 | ReAct Agent | `agent/react_agent.py` | @tool 定义工具、ReAct 思维链、Agent 执行器 |
| 4 | 多 Agent 协作 | `langgraph/media_studio.py` | LangGraph 状态图、多节点协作、条件路由 |
| 7 | Tool Calling Agent | `agent/tool_calling_agent.py` | 原生函数调用、工具绑定、结构化工具参数 |

**学完你能：** 让 AI 自主调用工具、拆解任务、多角色协作完成复杂目标

### 四、应用篇 — 实用功能模块

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 6 | 结构化输出 | `structured_output/extraction.py` | Pydantic 模型、with_structured_output、实体提取 |
| 8 | 流式处理 | `streaming/stream_demo.py` | stream()、astream_events、逐 token 输出、流式 RAG |

**学完你能：** 让 LLM 输出结构化 JSON，实现打字机效果的实时响应

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

### 六、进阶篇 — 高级架构与模式

| # | 项目 | 目录 | 核心知识点 |
|---|------|------|-----------|
| 16 | Memory 进阶 | `memory_advanced/memory_strategies.py` | 窗口记忆、摘要记忆、向量长期记忆、组合记忆架构 |
| 17 | RAG 进阶策略 | `rag_advanced/rag_strategies.py` | HyDE、Multi-Query、Parent-Child、策略组合 |
| 18 | LangGraph 进阶 | `langgraph_advanced/hitl_checkpoint.py` | Checkpointing 状态持久化、Human-in-the-Loop、interrupt_before |

**学完你能：** 构建有长期记忆的 AI、高质量检索系统、需要人工审批的 AI 工作流

---

## 🗓️ 推荐学习顺序

```
Week 1:  项目 0 → 1 → 2           (基础：API → LangChain → RAG)
Week 2:  项目 3 → 4 → 5           (进阶：Agent → 多Agent → 高级RAG)
Week 3:  项目 6 → 7 → 8           (应用：结构化 → 工具调用 → 流式)
Week 4:  项目 9 → 10 → 11         (工程：评估 → API服务 → 异步)
Week 5:  项目 12 → 13 → 14 → 15   (工程：持久化 → 安全 → 监控 → 容错)
Week 6:  项目 16 → 17 → 18        (进阶：记忆 → RAG策略 → LangGraph)
```

---

## 📁 项目结构

```
llm-cookbook/
├── llm/                    # 原生 LLM API 调用
├── langchain/              # LangChain 核心组件
├── rag/                    # RAG 基础与高级
├── rag_advanced/           # RAG 进阶策略
├── agent/                  # Agent 开发
├── langgraph/              # LangGraph 多 Agent 协作
├── langgraph_advanced/     # LangGraph 进阶特性
├── structured_output/      # 结构化输出
├── streaming/              # 流式处理
├── evaluation/             # RAG 评估
├── api_service/            # API 服务化
├── async_concurrent/       # 异步并发
├── vectordb/               # 向量库持久化
├── guardrails/             # 安全护栏
├── observability/          # 可观测性
├── error_handling/         # 容错机制
├── memory_advanced/        # 高级记忆策略
├── document_etl/           # 文档 ETL 处理
├── self_reflection/        # 自我反思
├── docs/                   # 文档资料
├── requirements.txt        # 项目依赖
└── LEARNING_ROADMAP.md     # 详细学习路线图
```

---

## 🔧 技术栈

- **核心框架：** LangChain, LangGraph
- **LLM 接口：** OpenAI, Azure OpenAI
- **向量数据库：** FAISS, Chroma
- **Embeddings：** OpenAI, HuggingFace
- **API 框架：** FastAPI, LangServe
- **部署：** Uvicorn, SSE-Starlette

---

## 🤝 贡献

欢迎提交 Issue 和 PR！如果你有好的实战案例或改进建议，请随时分享。

---

## 📄 许可证

[MIT License](LICENSE.txt)

---

## 🙏 致谢

- [LangChain](https://python.langchain.com/) - 强大的 LLM 应用开发框架
- [LangGraph](https://langchain-ai.github.io/langgraph/) - 复杂 Agent 工作流编排
- [OpenAI](https://openai.com/) - 提供强大的大模型能力

---

> **Happy Coding! 🚀** 让我们一起探索大模型应用开发的无限可能！
