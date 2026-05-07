---
title: 缓存策略
---

# LLM 缓存（Caching）

LLM 调用成本高、延迟高、重复多。缓存可节省 70% 成本，将响应时间从秒级降到毫秒级。

## 1. 为什么需要缓存

| 痛点 | 数据 |
|------|------|
| 成本高 | GPT-4o: $2.50/1M input tokens |
| 延迟高 | 每次 1-5 秒 |
| 重复浪费 | 客服场景 70% 问题重复 |

**缓存效果：** 1000 次请求，70% 命中 → 只需 300 次 API 调用，节省 70% 费用。

## 2. 缓存层次

```
┌─────────────┐
│ API 层缓存   │ ← Anthropic Prompt Caching（供应商提供）
├─────────────┤
│ 应用层缓存   │ ← InMemoryCache / SQLite / Redis（精确匹配）
├─────────────┤
│ 语义层缓存   │ ← SemanticCache（向量相似度，命中率最高）
└─────────────┘
```

## 3. 精确匹配缓存

输入完全相同时命中：

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
# 第二次相同请求直接返回缓存结果
```

| 方案 | 持久性 | 适用 |
|------|--------|------|
| InMemoryCache | 重启丢失 | 开发 |
| SQLiteCache | 文件持久 | 单机 |
| RedisCache | 分布式 | 生产 |

## 4. 语义缓存

问题不完全相同但语义相近时命中：

```
"Python怎么读文件" ≈ "用Python打开文件" → 命中！
```

```python
from langchain.cache import RedisSemanticCache

set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=embeddings,
    score_threshold=0.9,
))
```

## 5. 缓存键设计

好的缓存键 = 完整的影响因素：

```python
cache_key = hash(
    model_name +
    system_prompt +
    user_input +
    temperature +
    str(max_tokens)
)
```

## 6. 缓存失效策略

| 策略 | 说明 | 适用 |
|------|------|------|
| TTL | 固定时间过期 | 通用 |
| LRU | 最久未使用淘汰 | 内存有限 |
| 手动失效 | 数据变更时清除 | 知识库更新 |
| 版本号 | Prompt 改版时全部失效 | Prompt 迭代 |

## 7. Prompt Caching（API 层）

Anthropic / OpenAI 提供的服务端缓存，相同前缀的 Prompt 自动复用计算：

```
长 System Prompt (2000 tokens) + 用户问题 (50 tokens)
→ System Prompt 部分被缓存，后续请求只计算用户问题部分
→ 节省 90% 输入 token 费用
```

## 8. 成本节省分析

```
场景: 日活 10 万客服机器人
无缓存: 100,000 × $0.005 = $500/天
有缓存(70%命中): 30,000 × $0.005 = $150/天
节省: $350/天 = $10,500/月
```

::: warning 需要本地运行
完整实现见 `caching/cache_demo.py`，包含多种缓存策略的性能对比。
:::

---

::: tip 下一步
- [错误处理](/engineering/error-handling) — 重试与降级策略
- [可观测性](/engineering/observability) — 监控缓存命中率
:::
