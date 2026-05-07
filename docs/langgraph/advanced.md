---
title: LangGraph 高级
---

# LangGraph 高级篇（HITL + Checkpoint）

人机协作（Human-in-the-Loop）和检查点持久化是构建生产级 AI Agent 系统的基石。

## 1. 人机协作（HITL）

AI 在执行不可逆操作前暂停，等待人类确认：

```
场景: 发送邮件、部署代码、删除数据、付费操作
流程: AI 生成方案 → ⚡暂停 → 人类审批 → 继续/终止
```

**核心 API：**

```python
# 1. 编译时设置暂停点
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["execute_action"]
)

# 2. 执行到暂停点
result = app.invoke(input, config=config)

# 3. 查看暂停状态
state = app.get_state(config)

# 4. 注入人类决策
app.update_state(config, {"approved": True}, as_node="generate")

# 5. 继续执行
final = app.invoke(None, config=config)
```

## 2. Checkpoint 持久化

每执行完一个节点，自动保存状态快照：

| Checkpointer | 存储 | 适用场景 |
|--------------|------|---------|
| MemorySaver | 内存 | 开发/测试 |
| SqliteSaver | SQLite | 单机/小规模 |
| PostgresSaver | PostgreSQL | 生产环境 |

**thread_id** 标识独立执行线程，类似 session_id：

```python
config = {"configurable": {"thread_id": "user_001_session_1"}}
```

## 3. 断点续传

```python
# 暂停 → 跨时间恢复（可以是小时/天后）
result = app.invoke(input, config)        # 执行到暂停点
# ... 人类审核 ...
app.update_state(config, values, as_node="node_name")
final = app.invoke(None, config)          # 从暂停处继续
```

`as_node` 告诉 LangGraph"假装这个更新来自哪个节点"，决定了从哪条边继续。

## 4. 时间旅行

Checkpoint 保存每步快照，支持回到过去任意一步：

```python
# 获取历史状态
history = list(app.get_state_history(config))
# 从某个历史状态恢复
old_config = history[2].config
app.invoke(None, config=old_config)
```

应用场景：调试、A/B 测试、用户后悔回退。

## 5. 工具审批流程

按风险级别选择性审批：

```python
HIGH_RISK_TOOLS = {"delete_file", "send_email", "deploy"}

def tool_node(state):
    if state["tool_name"] in HIGH_RISK_TOOLS:
        return {"needs_approval": True}  # 触发暂停
    else:
        return {"result": execute(state["tool_name"])}
```

## 6. 多租户

不同 thread_id 完全隔离，同一个 app 实例可同时服务多用户：

```python
# 用户A和用户B的流程互不影响
app.invoke(input_a, config={"configurable": {"thread_id": "user_a"}})
app.invoke(input_b, config={"configurable": {"thread_id": "user_b"}})
```

## 7. 流式输出

```python
# 逐节点流式
for event in app.stream(input, config):
    for node_name, output in event.items():
        print(f"[{node_name}] → {output}")

# Token 级流式
async for event in app.astream_events(input, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

## 8. 生产最佳实践

| 维度 | 建议 |
|------|------|
| State 设计 | 最小化，大数据用引用（文件路径/ID） |
| interrupt 位置 | 放在不可逆操作之前 |
| 超时 | 设置人类审批超时策略 |
| 错误处理 | LLM 超时重试 → 降级 → 人工介入 |
| 监控 | LangSmith 集成 + 结构化日志 |

::: warning 需要本地运行
完整实现见 `langgraph_advanced/hitl_checkpoint.py`。
:::

---

::: tip 下一步
- [流式输出](/engineering/streaming) — 流式与 HITL 的结合
- [API 服务](/production/api-service) — 部署图工作流为 REST API
:::
