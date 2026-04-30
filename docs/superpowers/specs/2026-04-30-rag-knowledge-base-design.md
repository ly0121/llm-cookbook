# 项目二：本地文档 RAG 知识库 — 设计文档

**日期：** 2026-04-30
**项目：** 项目二：本地文档 RAG 知识库（外挂大脑）
**目标读者：** LLM 开发新手

---

## 项目目标

用 LangChain 实现一个完整的本地 RAG（Retrieval-Augmented Generation）知识库：
将一段长文本向量化存入本地 FAISS，用户提问时先检索相关文本块，再让 LLM 基于检索结果回答，
并在控制台打印出检索召回的原始文本块，让学习者亲眼看到"大模型是在基于什么资料回答问题"。

---

## 目录结构

```
langchain_learning/
├── llm/native_api.py         （已有，项目零）
├── langchain/chatbot.py      （已有，项目一）
├── rag/                      ← 新建
│   └── rag_qa.py             ← 项目二主教学文件（新建）
└── requirements.txt          ← 追加 faiss-cpu>=1.7.4
```

---

## 样本文档

代码内直接模拟一段约 3000 字的中文技术科普文章，主题：**"人工智能发展简史"**
（从 1950 年代图灵测试 → 专家系统 → 神经网络 → 深度学习 → Transformer → GPT 时代）。

问答示例：
- "谁提出了图灵测试？" → 答案在文本里：艾伦·图灵
- "深度学习是什么时候兴起的？" → 答案在文本里：2012年 AlexNet
- "GPT 系列是哪家公司开发的？" → 答案在文本里：OpenAI

---

## `rag/rag_qa.py` 内部章节

| 章节 | 内容 | 核心学习点 |
|------|------|-----------|
| 文件头（docstring） | "开卷考试"比喻讲透 RAG 工作流 | 切块→向量化→检索→回答 全流程；RAG 解决什么问题（知识截止日期/私有文档） |
| 第 0 章 | 初始化 LLM + Embeddings 客户端 | `ChatOpenAI` + `OpenAIEmbeddings(base_url=..., model=EMBEDDING_MODEL)`；EMBEDDING_MODEL 常量可配置 |
| 第 1 章 | 文档加载 + 文本切块 | 内联原始文本用 `Document`；`RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`；打印切块总数和第一块样本 |
| 第 2 章 | 向量化 + 存入 FAISS | `FAISS.from_documents(docs, embeddings)`；打印索引中向量数量；解释 FAISS 是纯内存索引 |
| 第 3 章 | 检索演示（不问 LLM） | `retriever = vectorstore.as_retriever(search_kwargs={"k": 3})`；`retriever.invoke("...")`；**打印检索召回的每块原始文本**；解释相似度检索原理 |
| 第 4 章 | 完整 RAG 问答链（LCEL） | `RunnableParallel(context=retriever, question=RunnablePassthrough()) \| prompt \| llm \| parser`；打印最终回答；对比"不用 RAG 的回答"（LLM 可能直接拒绝或幻觉） |

---

## 技术选型

| 决策点 | 选择 | 原因 |
|--------|------|------|
| 向量数据库 | `FAISS`（内存，不持久化） | 教学场景轻量，原理透明（两个文件），安装简单 |
| Embedding 客户端 | `OpenAIEmbeddings(base_url=GATEWAY_URL, api_key=API_KEY, model=EMBEDDING_MODEL)` | 与 LLM 共用同一 Gateway，零额外配置 |
| Embedding 模型 | `EMBEDDING_MODEL = "text-embedding-3-small"`（顶部常量，注释说明可换 `text-embedding-ada-002`） | 如 Gateway 不支持，用户只需改一个常量 |
| 文本切块 | `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)` | 递归按语义边界切割，比固定长度切割效果更好 |
| RAG 链 | LCEL `RunnableParallel` + `RunnablePassthrough` | 最新官方推荐语法，无废弃 API |
| 问答 Prompt | 中文 System Prompt，要求"仅根据上下文回答，不知道就说不知道" | 防止 LLM 用自身知识绕过 RAG |
| LLM 类 | `ChatOpenAI`（和 langchain/chatbot.py 一致） | 共用已验证的接口配置 |

---

## `requirements.txt` 变更

追加：
```
faiss-cpu>=1.7.4
```

（`langchain-openai` 已包含 `OpenAIEmbeddings`，无需额外包）

---

## 学习目标

读完 `rag/rag_qa.py` 后，学习者应能理解：
1. RAG 解决什么问题（LLM 知识截止日期、私有文档问答）
2. "切块→向量化→检索→生成"四步流水线的每步作用
3. `RecursiveCharacterTextSplitter` 的参数含义（chunk_size / chunk_overlap）
4. FAISS 向量数据库的工作原理（把文本变成数字向量，用距离衡量相似度）
5. LCEL `RunnableParallel` 如何同时传入"检索结果"和"原始问题"
6. 如何看懂检索召回的原始文本块，验证 RAG 是否工作正确

---

## ⚠️ 避坑指南（文件内需包含的内容）

1. **Embedding 模型名不对**：如 Gateway 不支持 `text-embedding-3-small`，报 404/422，改 `EMBEDDING_MODEL` 常量即可
2. **faiss-cpu 和 faiss-gpu 不能同时装**：教学环境只装 `faiss-cpu`
3. **chunk_overlap 必须小于 chunk_size**：否则 LangChain 报 ValueError
4. **FAISS 是内存索引**：程序退出后索引消失，每次运行重新构建（教学可接受，生产用持久化）
5. **RunnablePassthrough 的作用**：不是"什么都不做"，而是"原封不动地把输入传到下一步"，和 `RunnableParallel` 配合时尤为重要
