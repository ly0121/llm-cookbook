---
title: 快速开始
---

# 快速开始

## 在线阅读

直接浏览左侧导航，选择感兴趣的主题开始学习。带有 `🐍 Python (浏览器运行)` 标记的代码块可以直接点击运行。

## 本地运行

如果你想运行所有代码（包括需要 API 调用的部分），请在本地克隆项目：

```bash
# 克隆项目
git clone https://github.com/liuyu22/llm-cookbook.git
cd llm-cookbook

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 运行任意项目
python llm/native_api.py
python llm/prompt_engineering.py
python rag/rag_qa.py
```

## 项目结构

```
llm-cookbook/
├── llm/                  ← LLM 基础（API 调用、提示工程、分词、生成策略）
├── langchain/            ← LangChain 框架
├── rag/                  ← RAG 基础
├── rag_advanced/         ← RAG 高级策略（CRAG、Self-RAG）
├── agent/                ← Agent 系统
├── langgraph/            ← LangGraph 工作流
├── langgraph_advanced/   ← HITL、检查点
├── streaming/            ← 流式处理
├── structured_output/    ← 结构化输出
├── async_concurrent/     ← 异步并发
├── caching/              ← 缓存
├── error_handling/       ← 错误处理
├── observability/        ← 可观测性
├── api_service/          ← API 服务化
├── guardrails/           ← 安全护栏
├── evaluation/           ← 评估体系
├── memory_advanced/      ← 记忆系统
├── self_reflection/      ← 自我反思
└── vectordb/             ← 向量数据库
```

## 代码类型说明

本站代码块分为两种类型：

### 🌐 浏览器可运行

基于 Pyodide（浏览器端 Python），支持：
- 纯 Python 逻辑和算法
- NumPy 数学计算
- 简化版的 BPE 分词、向量检索演示

### 🖥️ 需要本地环境

需要克隆项目到本地运行：
- OpenAI / LLM API 调用
- LangChain / LangGraph 完整链
- FAISS / Chroma 向量数据库
- tiktoken 分词器

## 推荐学习顺序

1. **Week 1-2**: LLM 基础 → LangChain
2. **Week 3-4**: RAG 基础 → RAG 高级
3. **Week 5-6**: Agent → LangGraph
4. **Week 7-8**: 工程实践（流式、异步、缓存、错误处理）
5. **Week 9**: 生产部署 + 高级主题
