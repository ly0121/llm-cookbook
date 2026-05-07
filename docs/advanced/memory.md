---
title: 记忆系统
---

# LLM 记忆系统

LLM 本身无状态，每次调用都"失忆"。记忆系统让 AI 在多轮对话中保持上下文连贯。

## 1. 核心问题

```
第1次: "我叫张三" → "你好张三！"
第2次: "我叫什么？" → "我不知道。"  ← 完全不记得！
```

所有"记忆"必须塞进上下文窗口，带来 Token 爆炸、费用激增、注意力稀释三重问题。

## 2. 记忆分类

| 类型 | 说明 | 适用 |
|------|------|------|
| 窗口记忆 | 保留最近 N 轮 | 简单对话 |
| 摘要记忆 | LLM 总结旧对话 | 长对话 |
| 向量记忆 | 按相关性检索历史 | 信息密集 |
| 实体记忆 | 提取关键实体和关系 | 个性化 |

## 3. 窗口记忆

最简单：只保留最近 K 轮对话。

```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=2000,
    strategy="last",  # 保留最新的
)
```

优点：实现简单、Token 可控。缺点：丢失早期重要信息。

## 4. 摘要记忆

用 LLM 总结旧对话为一段摘要：

```
原始历史(5000 tokens):
  用户讨论了Python学习...然后问了Docker...又聊了K8s...

压缩为摘要(200 tokens):
  "用户是后端开发者，正在学习云原生技术栈(Docker+K8s)"
```

## 5. 向量记忆

将历史对话片段向量化存储，按相关性检索：

```python
# 存储
memory_store.add(conversation_chunk, metadata={"timestamp": now})

# 检索（只取相关的历史）
relevant_history = memory_store.search(current_question, top_k=3)
```

适合信息密集的多主题对话。

## 6. 多层记忆架构

```
┌─────────────────────────────────┐
│  短期记忆 (最近 3 轮)            │ ← 完整保留
├─────────────────────────────────┤
│  工作记忆 (摘要 + 关键实体)      │ ← LLM 压缩
├─────────────────────────────────┤
│  长期记忆 (向量数据库)           │ ← 按需检索
└─────────────────────────────────┘
```

## 7. Token 预算管理

```python
TOTAL_BUDGET = 4000  # 总 Token 预算

system_prompt = 500    # 固定
recent_history = 1500  # 最近几轮
retrieved_memory = 1000 # 检索的相关记忆
user_input = 500       # 当前输入
reserved_output = 500  # 留给模型输出
```

## 8. 跨会话持久化

| 存储 | 特点 | 适用 |
|------|------|------|
| Redis | 高性能，支持 TTL | 在线服务 |
| PostgreSQL | 持久化，复杂查询 | 长期记忆 |
| 向量数据库 | 语义检索 | 相关性记忆 |

## 9. MemGPT 思想

将 LLM 的上下文窗口类比为操作系统的"内存"：

- 主内存（上下文窗口）容量有限
- 外部存储（向量库）容量无限
- 操作系统（控制器）决定何时换入换出

::: warning 需要本地运行
完整实现见 `memory_advanced/memory_strategies.py`。
:::

---

::: tip 下一步
- [自我反思](/advanced/self-reflection) — 让 AI 评估和改进自身输出
- [向量数据库](/advanced/vectordb) — 记忆存储的底层支撑
:::
