---
title: RAG 检索增强生成
---

<script setup>
const code1 = `import numpy as np

# 余弦相似度计算演示
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 模拟 3 个文档的 Embedding（简化为 5 维）
np.random.seed(42)
query = np.array([0.8, 0.6, 0.1, -0.2, 0.3])       # 用户问题向量
doc_relevant = np.array([0.75, 0.55, 0.15, -0.1, 0.25])  # 相关文档
doc_partial = np.array([0.4, 0.3, 0.5, 0.2, -0.1])       # 部分相关
doc_irrelevant = np.array([-0.5, 0.1, 0.8, 0.6, -0.3])   # 不相关

docs = {
    '相关文档': doc_relevant,
    '部分相关': doc_partial,
    '不相关文档': doc_irrelevant,
}

print('=== 余弦相似度检索结果 ===')
print(f'Query 向量: {query}')
print()

results = []
for name, vec in docs.items():
    sim = cosine_similarity(query, vec)
    results.append((name, sim))

results.sort(key=lambda x: -x[1])

for rank, (name, sim) in enumerate(results, 1):
    bar = '█' * int(sim * 30) if sim > 0 else ''
    print(f'  #{rank} {name:8s}  相似度={sim:.4f}  {bar}')

print()
print('结论: 相似度最高的文档被检索返回给 LLM 作为上下文')
print()

# L2 距离对比
print('=== L2 距离（越小越相似）===')
for name, vec in docs.items():
    l2 = np.linalg.norm(query - vec)
    print(f'  {name:8s}  L2={l2:.4f}')
`
</script>

# RAG（检索增强生成）

RAG = Retrieval-Augmented Generation，核心思想是让 LLM 从"闭卷考试"变成"开卷考试"。

## 1. 为什么需要 RAG

LLM 的三大缺陷驱动了 RAG 的诞生：

| 缺陷 | 表现 | RAG 如何解决 |
|------|------|-------------|
| 知识截止 | 不知道最新信息 | 实时检索外部文档 |
| 无私有数据 | 没见过公司内部文档 | 接入私有知识库 |
| 幻觉 | 一本正经地编造 | 基于真实文档生成，可溯源 |

## 2. RAG 完整流程

```
═══ 离线阶段（Indexing）═══
  文档 → 加载 → 切分 → Embedding → 存入向量数据库

═══ 在线阶段（Query）═══
  用户提问 → Embedding → 向量检索 → Top-K 文档
           → 构造 Prompt → LLM 生成回答
```

## 3. 文本切分

使用 RecursiveCharacterTextSplitter 按语义边界递归切分：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", " ", ""],
)
```

**chunk_overlap** 确保相邻块的信息不断裂。

## 4. Embedding 与向量相似度

文本通过 Embedding 模型映射为高维向量，语义相似的文本在向量空间中距离更近。

**三种距离度量：**

| 度量 | 公式 | 特点 |
|------|------|------|
| 余弦相似度 | cos(A,B) = A·B / (\|\|A\|\| × \|\|B\|\|) | 最常用，不受长度影响 |
| 内积 | A·B | 归一化后等价余弦 |
| L2 距离 | sqrt(sum((a-b)^2)) | 值越小越相似 |

<PythonRunner :browser-runnable="true" :code="code1" />

## 5. RAG 链构建

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("谁提出了图灵测试？")
```

## 6. RAG 评估

RAG 系统从三个维度评估：

| 维度 | 评估什么 | 提升方向 |
|------|---------|---------|
| Context Relevance | 检索到的内容相关吗 | 更好的 Embedding、查询改写 |
| Faithfulness | 回答忠于检索内容吗 | 更强的 Prompt 约束 |
| Answer Correctness | 最终答案正确吗 | 优化检索+生成 |

## 7. Naive RAG 的局限

| 痛点 | 改进方向 |
|------|---------|
| 语义鸿沟（口语 vs 术语） | HyDE 假设文档 |
| 召回不全 | Multi-Query 多查询扩展 |
| 上下文不完整 | Parent-Child 父子文档 |
| 检索不精准 | 重排序（Re-ranking） |

::: warning 需要本地运行
完整 RAG 实现见 `rag/rag_qa.py`，包含文档加载、FAISS 索引、检索和完整 RAG 链。
:::

---

::: tip 下一步
- [RAG 高级策略](/rag/advanced) — HyDE、Multi-Query、重排序等进阶技术
- [向量数据库](/advanced/vectordb) — 深入理解向量索引和生产选型
:::
