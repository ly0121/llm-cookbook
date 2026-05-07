---
title: 向量数据库
---

# 向量数据库

向量数据库是 RAG 系统的核心基础设施，提供高维向量的高效存储和最近邻检索能力。

## 1. 为什么需要向量数据库

| 对比 | 传统数据库 | 向量数据库 |
|------|-----------|-----------|
| 查询方式 | `WHERE name LIKE '%手机%'` | `search("手机", top_k=5)` |
| 匹配能力 | 精确/模糊 | 语义理解 |
| "手机"查询 | 找不到 "iPhone" | 能找到 "iPhone"、"移动电话" |

## 2. 向量索引算法

### Flat（暴力搜索）

```
每次查询遍历所有向量，计算相似度
精度: 100%（完美）
速度: O(n)，百万级别不可用
适用: 数据量 < 1 万
```

### IVF（倒排索引）

```
预先将向量聚类为 N 个桶
查询时只在最近的 K 个桶中搜索

精度: ~95%
速度: O(n/N × K)
适用: 十万~百万级
```

### HNSW（分层可导航小世界图）

```
构建多层跳表式图结构
从顶层粗搜到底层精搜

精度: >95%
速度: O(log n)
适用: 百万~千万级（推荐）
```

### 对比

| 算法 | 构建速度 | 查询速度 | 精度 | 内存 |
|------|---------|---------|------|------|
| Flat | 无需构建 | 慢 | 100% | 低 |
| IVF | 快 | 中 | 95% | 低 |
| HNSW | 慢 | 快 | >95% | 高 |

## 3. FAISS

Meta 开源的向量检索库（非数据库），纯内存、极快：

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")  # 保存到磁盘
```

适用：原型开发、中小规模、对持久化要求不高。

## 4. Chroma

轻量级向量数据库，开发者友好：

```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    docs, embeddings,
    persist_directory="./chroma_db"
)
```

适用：本地开发、中小项目。

## 5. 主流对比

| 数据库 | 类型 | 特点 | 适用 |
|--------|------|------|------|
| FAISS | 库 | 纯内存，极快 | 原型/中小规模 |
| Chroma | 轻量DB | 简单易用 | 本地开发 |
| Milvus | 分布式DB | 高性能，支持十亿级 | 大规模生产 |
| Pinecone | 云服务 | 全托管，开箱即用 | 快速上线 |
| Weaviate | DB | GraphQL API，多模态 | 灵活场景 |
| Qdrant | DB | Rust 实现，高性能 | 性能敏感 |
| pgvector | 扩展 | PostgreSQL 插件 | 已有 PG 基础设施 |

## 6. CRUD 操作

```python
# Create - 添加文档
vectorstore.add_documents(new_docs)

# Read - 相似度搜索
results = vectorstore.similarity_search("查询", k=5)

# Update - 通常删除后重新添加
vectorstore.delete(ids=["doc_1"])
vectorstore.add_documents([updated_doc])

# Delete
vectorstore.delete(ids=["doc_1", "doc_2"])
```

## 7. Metadata 过滤

```python
# 混合查询：向量相似度 + 元数据过滤
results = vectorstore.similarity_search(
    "查询",
    k=5,
    filter={"year": 2024, "category": "技术"}
)
```

## 8. 生产选型指南

```
数据量 < 10 万 → FAISS / Chroma
数据量 10 万~1000 万 → Milvus / Qdrant
数据量 > 1000 万 → Milvus 分布式
已有 PostgreSQL → pgvector
不想运维 → Pinecone (云服务)
```

::: warning 需要本地运行
完整实现见 `vectordb/persistent_store.py`，包含 Chroma 持久化和 CRUD 操作。
:::

---

::: tip 下一步
- [RAG 基础](/rag/) — 向量数据库在 RAG 中的应用
- [RAG 高级](/rag/advanced) — 混合检索和重排序
:::
