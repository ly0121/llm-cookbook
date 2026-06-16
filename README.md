# LLM Cookbook — Practical Guide for Large Language Model Application Development

> From Zero to Production-Grade LLM Applications: 21 Progressive Hands-On Projects

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.0+-green.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6.0+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview

This is a systematic LLM application development learning project. Through 21 progressive hands-on projects, you'll master the core skills of large language model application development from scratch to building enterprise-grade AI applications.

**Core Philosophy:** It's not just about calling APIs, but mastering the complete engineering capabilities required to build production-ready AI applications.

---

## 📚 Learning Path Overview

```
ML Foundations (Optional)      → Classical ML, deep learning, classic NLP — the prerequisites
Fundamentals (Projects 0-1)    → Master LLM APIs and LangChain core components
RAG (Projects 2-5)             → From basic RAG to advanced retrieval strategies
Agent (Projects 3-4)           → ReAct Agent + Multi-Agent collaboration
Application (Projects 6-8)     → Structured output, tool calling, streaming
Engineering (Projects 9-15)    → Evaluation, API services, async, persistence, security, monitoring, fault tolerance
Advanced (Projects 16-18)      → Advanced memory, advanced RAG, human-in-the-loop
```

---

## 🚀 Quick Start

### Environment Setup

```bash
# Clone the project
git clone https://github.com/ly0121/llm-cookbook.git
cd llm-cookbook

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# Or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configure API Key

The project uses a centralized `config.py` that loads credentials from environment
variables (or a local `.env` file via `python-dotenv`). Every demo imports the
shared `client` and `MODEL_NAME` from there — you only need to configure once.

```bash
# 1. Copy the template
cp .env.example .env

# 2. Edit .env and fill in your real key
#    LLM_API_KEY=your_key_here
#    LLM_BASE_URL=https://...           (optional; default points to internal gateway)
#    LLM_MODEL_NAME=aws-claude-sonnet-4-6 (optional)
#    LLM_EMBEDDING_MODEL=...             (optional)
#    LLM_JUDGE_MODEL=...                 (optional, for LLM-as-Judge eval)
```

The `.env` file is `.gitignore`-d, so your key never enters version control.
Pure-local demos (e.g. `llm/tokenization_demo.py`) work without any key —
`config.py` only validates the key when you actually call the API.

In any script under a sub-directory, the unified import pattern is:

```python
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import client, MODEL_NAME
```

---

## 📖 Project Checklist

### 0. ML Foundations — Prerequisites for LLM (Optional)

Before diving into LLM application development, this section provides a foundation in classical machine learning, deep learning, and classic NLP. **Skip if you already have ML background.**

| Module | Directory | Core Concepts |
|--------|-----------|---------------|
| Classical ML | `ml_foundations/classical/` | Logistic Regression, Decision Tree, Random Forest, Ridge/Lasso, K-Means, Pipeline + Feature Engineering |
| Deep Learning | `ml_foundations/deep_learning/` | PyTorch tensors & autograd, MLP from scratch, MLP/CNN on MNIST, Char-level LSTM language model |
| NLP Foundations | `ml_foundations/nlp_foundations/` | Chinese/English preprocessing (jieba/NLTK), TF-IDF + Naive Bayes/SVM classification, Word2Vec + analogical reasoning |

All demos run on **Mac CPU within 3 minutes**, no GPU or LLM API key required. See `docs/ml-foundations/` for the corresponding theory chapters covering the bridge from classical ML to LLM internals (e.g. *why MLP = Transformer FFN sublayer*, *why LoRA is a low-rank regularizer*, *why BM25 still matters in RAG*).

**After completing:** You can confidently read LLM literature, debug fine-tuning pipelines, and understand why Transformer replaced RNN.

### 1. Fundamentals — Getting Started with LLM Application Development

| # | Project | Directory | Core Concepts |
|---|---------|-----------|---------------|
| 0 | Native API Calls | `llm/native_api.py` | OpenAI SDK, blocking/streaming calls, Response structure parsing |
| 1 | LangChain Core | `langchain/chatbot.py` | ChatOpenAI, PromptTemplate, LCEL pipeline, OutputParser, MessageHistory |

**After completing:** You can call LLMs with Python and understand the complete Prompt→Model→Output pipeline.

### 2. RAG — Retrieval-Augmented Generation

| # | Project | Directory | Core Concepts |
|---|---------|-----------|---------------|
| 2 | Basic RAG | `rag/rag_qa.py` | Document loading, text chunking, Embedding, FAISS vector retrieval, RAG Chain |
| 5 | Advanced RAG | `rag/advanced_rag.py` | Metadata filtering, source tracking, multi-document retrieval, context compression |
| 17 | Advanced RAG Strategies | `rag_advanced/rag_strategies.py` | HyDE hypothetical documents, Multi-Query expansion, Parent-Child chunking |

**After completing:** You can build production-grade Q&A systems, solving semantic gap and recall rate issues.

### 3. Agent — Autonomous Decision-Making AI

| # | Project | Directory | Core Concepts |
|---|---------|-----------|---------------|
| 3 | ReAct Agent | `agent/react_agent.py` | @tool decorator, ReAct reasoning chain, Agent executor |
| 4 | Multi-Agent Collaboration | `langgraph/media_studio.py` | LangGraph state graph, multi-node collaboration, conditional routing |
| 7 | Tool Calling Agent | `agent/tool_calling_agent.py` | Native function calling, tool binding, structured tool parameters |

**After completing:** You can make AI autonomously call tools, break down tasks, and collaborate across multiple roles to achieve complex goals.

### 4. Application — Practical Feature Modules

| # | Project | Directory | Core Concepts |
|---|---------|-----------|---------------|
| 6 | Structured Output | `structured_output/extraction.py` | Pydantic models, with_structured_output, entity extraction |
| 8 | Streaming | `streaming/stream_demo.py` | stream(), astream_events, token-by-token output, streaming RAG |

**After completing:** You can make LLMs output structured JSON and implement typewriter-effect real-time responses.

### 5. Engineering — Production-Ready AI Application Essentials

| # | Project | Directory | Core Concepts |
|---|---------|-----------|---------------|
| 9 | RAG Evaluation | `evaluation/rag_eval.py` | Faithfulness, relevance, completeness, LLM-as-Judge evaluation framework |
| 10 | API Service | `api_service/fastapi_server.py` | FastAPI, LangServe, REST API, streaming endpoints |
| 11 | Async Concurrency | `async_concurrent/async_demo.py` | async/await, asyncio.gather, Semaphore rate limiting, timeout control |
| 12 | Vector Store Persistence | `vectordb/persistent_store.py` | FAISS save/load, Chroma CRUD, metadata filtering, persistence strategies |
| 13 | Safety Guardrails | `guardrails/safety_guard.py` | Input injection prevention, System Prompt hardening, output sanitization, three-layer protection |
| 14 | Observability | `observability/tracing_demo.py` | Custom Callbacks, tracing, JSON structured logging, token cost monitoring |
| 15 | Fault Tolerance | `error_handling/retry_demo.py` | .with_retry(), exponential backoff, .with_fallbacks(), complete fault tolerance wrapper |

**After completing:** You can deploy AI applications to production environments with monitoring, security, fault tolerance, and evaluation capabilities.

### 6. Advanced — Advanced Architecture and Patterns

| # | Project | Directory | Core Concepts |
|---|---------|-----------|---------------|
| 16 | Advanced Memory | `memory_advanced/memory_strategies.py` | Window memory, summary memory, vector long-term memory, composite memory architecture |
| 17 | Advanced RAG Strategies | `rag_advanced/rag_strategies.py` | HyDE, Multi-Query, Parent-Child, strategy composition |
| 18 | Advanced LangGraph | `langgraph_advanced/hitl_checkpoint.py` | Checkpointing state persistence, Human-in-the-Loop, interrupt_before |

**After completing:** You can build AI with long-term memory, high-quality retrieval systems, and AI workflows requiring human approval.

---

## 🗓️ Recommended Learning Order

```
Week 1:  Projects 0 → 1 → 2         (Fundamentals: API → LangChain → RAG)
Week 2:  Projects 3 → 4 → 5         (Advanced: Agent → Multi-Agent → Advanced RAG)
Week 3:  Projects 6 → 7 → 8         (Application: Structured → Tool Calling → Streaming)
Week 4:  Projects 9 → 10 → 11       (Engineering: Evaluation → API Service → Async)
Week 5:  Projects 12 → 13 → 14 → 15 (Engineering: Persistence → Security → Monitoring → Fault Tolerance)
Week 6:  Projects 16 → 17 → 18      (Advanced: Memory → RAG Strategies → LangGraph)
```

---

## 📁 Project Structure

```
llm-cookbook/
├── config.py                # Shared LLM client + model config (loaded from .env)
├── .env.example             # Template for environment variables
├── ml_foundations/          # ML prerequisites: classical ML, deep learning, classic NLP
│                            # (sklearn / PyTorch / gensim / jieba — all CPU-runnable)
├── llm/                     # LLM fundamentals: native API, tokenization, generation,
│                            # transformer, embedding, RAG pipeline, function calling,
│                            # evaluation, observability, security, data engineering
├── langchain/               # LangChain core components
├── rag/                     # RAG basics and advanced
├── rag_advanced/            # Advanced RAG strategies (HyDE, Multi-Query, etc.)
├── agent/                   # Agent development (ReAct, tool calling)
├── langgraph/               # LangGraph multi-agent collaboration
├── langgraph_advanced/      # Checkpointing, Human-in-the-Loop
├── structured_output/       # Pydantic-driven structured output
├── streaming/               # Streaming processing
├── evaluation/              # RAG evaluation framework
├── api_service/             # FastAPI + LangServe deployment
├── async_concurrent/        # async/await, semaphore, timeout
├── vectordb/                # Vector store persistence (FAISS/Chroma)
├── guardrails/              # Safety guardrails (3-layer protection)
├── observability/           # Tracing, structured logging, cost monitoring
├── error_handling/          # Retry, fallback, fault tolerance
├── memory_advanced/         # Window/summary/vector memory strategies
├── self_reflection/         # Self-reflection / critic loop
├── caching/                 # Response/embedding caching, document ETL
├── docs/                    # VitePress documentation site
├── pyproject.toml           # Project metadata, ruff/black/pytest config
├── requirements.txt         # Pinned runtime dependencies
└── LEARNING_ROADMAP.md      # Detailed learning roadmap
```

---

## 🔧 Tech Stack

- **Core Framework:** LangChain, LangGraph
- **LLM APIs:** OpenAI, Azure OpenAI
- **Vector Databases:** FAISS, Chroma
- **Embeddings:** OpenAI, HuggingFace
- **API Framework:** FastAPI, LangServe
- **Deployment:** Uvicorn, SSE-Starlette

---

## 🤝 Contributing

Issues and PRs are welcome! If you have good practical cases or improvement suggestions, please feel free to share.

---

## 📄 License

[MIT License](LICENSE)

---

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) - Powerful LLM application development framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Complex agent workflow orchestration
- [OpenAI](https://openai.com/) - Providing powerful large language model capabilities

---

> **Happy Coding! 🚀** Let's explore the infinite possibilities of large language model application development together!