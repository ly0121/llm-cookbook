---
title: Agent 智能体
---

# Agent（智能体）

Agent = LLM（大脑）+ Tools（工具箱）+ 循环控制（执行引擎）。与普通 LLM 不同，Agent 可以主动调用工具获取外部信息。

## 1. 核心概念

```
普通 LLM: 提问 → 回答（只能用训练时的知识）
Agent:    提问 → 思考 → 调用工具 → 观察 → 再思考 → ... → 最终回答
```

| 维度 | 普通 LLM | Agent |
|------|----------|-------|
| 知识 | 静态（训练截止） | 动态（实时查询） |
| 能力 | 纯文本生成 | 调用任意工具 |
| 推理 | 单次回答 | 多步迭代 |

## 2. 架构模式

| 模式 | 特点 | 适用场景 |
|------|------|---------|
| **ReAct** | 推理和行动交替 | 教学、灵活任务 |
| **Tool Calling** | LLM 原生 JSON 输出 | 生产首选 |
| **Plan-and-Execute** | 先全局规划再逐步执行 | 复杂项目 |
| **REWOO** | 一次规划批量执行 | 降低成本 |

## 3. ReAct 框架

Thought → Action → Observation 循环：

```
Thought: 用户问北京天气，我需要用 get_weather 工具
Action: get_weather
Action Input: 北京
Observation: 北京：晴，28°C

Thought: 信息够了
Final Answer: 北京今天晴，28°C。
```

## 4. Tool Calling（生产首选）

LLM 原生输出结构化 JSON 调用工具，比 ReAct 文本解析更稳定：

```python
@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气信息。
    输入城市名（中文），如"北京"。
    当用户询问天气时使用此工具。"""
    ...
```

**ReAct vs Tool Calling 对比：**

| 维度 | ReAct | Tool Calling |
|------|-------|-------------|
| 参数格式 | 单字符串 | JSON 多参数 |
| 解析方式 | 正则（脆弱） | JSON（稳定） |
| 可靠性 | 容易格式出错 | 几乎不出错 |

## 5. 多步推理

Agent 可以链式调用工具，上一步结果作为下一步的推理依据：

```
问: "北京温度多少？把温度作为底数，2为指数，结果是？"

Step 1: get_weather("北京") → 28°C
Step 2: calculate_power(28, 2) → 784
Final: 北京28°C，28的2次方是784
```

## 6. 工具描述最佳实践

description 是 LLM 判断"用不用这个工具"的唯一依据：

```
❌ "查天气"                    — 太模糊
✅ "查询指定城市的实时天气信息，
   返回天气状况和温度（摄氏度）。
   输入城市名（中文）。
   当用户询问天气时使用。"      — 包含功能+输入+场景
```

## 7. 错误处理

```python
@tool
def get_weather(city: str) -> str:
    """..."""
    try:
        data = WEATHER_DATA.get(city)
        if data:
            return f"{city}：{data['weather']}，{data['temperature']}°C"
        return f"暂无 {city} 数据，支持：{'、'.join(WEATHER_DATA.keys())}"
    except Exception as e:
        return f"查询失败：{str(e)}"
```

返回友好错误信息让 Agent 自我纠正，而非抛异常。

## 8. 多 Agent 协作模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| 辩论 | A 提观点 → B 反驳 → 综合 | 多角度决策 |
| 分工 | 研究员 → 写手 → 主编 | 流水线任务 |
| 层级 | Manager 分配 → Worker 执行 | 大规模项目 |

::: warning 需要本地运行
完整实现见 `agent/react_agent.py` 和 `agent/tool_calling_agent.py`。
:::

---

::: tip 下一步
- [LangGraph 编排](/langgraph/) — 用图结构编排多 Agent 工作流
- [LangGraph 高级](/langgraph/advanced) — 人机协作和检查点机制
:::
