---
title: RAG 高级策略
---

# RAG 高级检索策略

Naive RAG 面临语义鸿沟、召回不全等问题，本章介绍系统性的进阶优化方案。

## 1. 语义鸿沟问题

用户的口语化提问与知识库的专业术语之间存在表述差异：

```
用户问: "自动驾驶怎么避免撞人？"     ← 日常口语
知识库: "感知模块通过多传感器融合..."   ← 技术术语
```

三种解决武器：

| 策略 | 解决的问题 | 代价 |
|------|-----------|------|
| HyDE | 语义鸿沟 | +1 次 LLM 调用 |
| Multi-Query | 召回率不足 | +1 次 LLM 调用 |
| Parent-Child | 上下文不完整 | +存储开销 |

## 2. HyDE（假设性文档嵌入）

核心思想：用 LLM 生成一个"假设性回答"，再用它做检索。

```
传统: Q("怎么避免撞人") → Embed → 搜索 → 可能偏离
HyDE:  Q → LLM生成假设回答H → Embed(H) → 搜索 → 更精准
```

假设回答的"表述风格"与知识库文档一致，向量空间中更接近真实文档。

```python
hyde_prompt = ChatPromptTemplate.from_template(
    '请针对以下问题，写一段技术性的回答文档（100-200字）\n'
    '问题: {question}\n技术文档风格的回答:'
)
hyde_chain = hyde_prompt | llm | StrOutputParser()
```

## 3. Multi-Query（多查询扩展）

一个问题从多个角度改写，多次检索后合并去重：

```
原始: "自动驾驶怎么避免撞人？"
  → Q1: "自动驾驶行人检测技术"   → [D1, D2]
  → Q2: "自动紧急制动AEB原理"    → [D2, D3]
  → Q3: "碰撞预警系统TTC算法"    → [D2, D4]
  → 合并去重: [D1, D2, D3, D4]   ← 比单次检索覆盖更全
```

## 4. Parent-Child 文档结构

解决"检索精度 vs 上下文完整"的矛盾：

- **索引用小块（Child）** — 向量集中，匹配精准
- **返回用大块（Parent）** — 上下文完整，LLM 能理解

```
┌──── Parent (完整段落) ────┐
│ [Child 1] [Child 2] [Child 3] │
└───────────────────────────────┘
  检索命中 Child 1 → 返回整个 Parent
```

## 5. 混合检索（Hybrid Search）

BM25（关键词）+ 向量（语义）双路检索：

| 方式 | 优势 | 劣势 |
|------|------|------|
| BM25 | 精确匹配专有名词 | 无法理解同义词 |
| 向量 | 理解语义近义 | 精确匹配弱 |
| 混合 | 兼得两者 | 需融合排序 |

融合策略推荐 RRF（Reciprocal Rank Fusion）：
```
RRF_score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_vector(d))
```

## 6. 重排序（Re-ranking）

两阶段检索：初选（Bi-Encoder，快但粗）→ 精排（Cross-Encoder，慢但准）。

| 方案 | 类型 | 适用场景 |
|------|------|---------|
| bge-reranker-base | Cross-Encoder | 中文 RAG |
| Cohere Reranker | API 服务 | 多语言生产 |
| FlashRank | 本地 | 低延迟需求 |

## 7. 自适应 RAG / CRAG / Self-RAG

**自适应 RAG**：根据查询类型动态选择策略（简单→直接检索，复杂→Multi-Query，闲聊→不检索）

**CRAG（Corrective RAG）**：检索后评估质量，不相关则放弃+补充搜索

**Self-RAG（自反思 RAG）**：生成时自我检查"是否需要检索？检索相关吗？回答忠实吗？"

## 8. 优化组合建议

| 组合 | 适用场景 |
|------|---------|
| HyDE + Parent-Child + Reranking | 高质量 |
| Multi-Query + 混合检索 + Reranking | 高召回 |
| 自适应 + 混合 + CRAG | 生产级 |

::: warning 需要本地运行
完整实现见 `rag_advanced/rag_strategies.py`，包含 HyDE、Multi-Query、Parent-Child 的代码。
:::

---

::: tip 下一步
- [向量数据库](/advanced/vectordb) — 深入索引算法和生产选型
- [Agent 智能体](/agent/) — 将 RAG 与 Agent 结合
:::
