---
title: 结构化输出
---

# 结构化输出（Structured Output）

让 LLM 输出类型安全的结构化数据（而非自由文本），是信息提取、自动化流程的基础。

## 1. 为什么需要结构化输出

```
非结构化: "候选人叫张三，28岁，会Python" → 需要正则提取 → 易出错
结构化:   ResumeInfo(name="张三", age=28, skills=["Python"]) → 直接访问
```

## 2. 三种方式对比

| 方式 | 机制 | 可靠性 | 适用场景 |
|------|------|--------|---------|
| JSON Mode | 强制输出 JSON | 中 | 简单结构 |
| Function Calling | 工具调用协议 | 高 | 多参数提取 |
| with_structured_output | LangChain 封装 | 最高 | 推荐方案 |

## 3. Pydantic 定义 Schema

```python
from pydantic import BaseModel, Field
from typing import Optional

class ResumeInfo(BaseModel):
    name: str = Field(description="候选人姓名")
    age: Optional[int] = Field(description="年龄")
    skills: list[str] = Field(description="技术技能列表")
    expected_salary: Optional[int] = Field(description="期望月薪(元)")
```

## 4. with_structured_output

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(ResumeInfo)

result = structured_llm.invoke("张三，28岁，Python/Java 开发，期望25k")
# result.name = "张三"
# result.age = 28
# result.skills = ["Python", "Java"]
```

## 5. 嵌套模型与复杂类型

```python
from enum import Enum

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class LineItem(BaseModel):
    name: str = Field(description="商品名称")
    quantity: int = Field(description="数量")
    unit_price: float = Field(description="单价")

class Invoice(BaseModel):
    invoice_no: str = Field(description="发票编号")
    date: str = Field(description="开票日期")
    items: list[LineItem] = Field(description="商品明细")
    total: float = Field(description="总金额")
    priority: Priority = Field(description="处理优先级")
```

## 6. 解析失败处理

| 策略 | 做法 | 适用场景 |
|------|------|---------|
| 重试 | 自动重试 1-2 次 | 偶发格式错误 |
| Prompt 增强 | 加入格式示例 | 复杂 Schema |
| 降级 | 返回原始文本 | 非关键场景 |
| 自动修复 | LLM 修正自身输出 | 高可靠需求 |

## 7. 性能优化建议

- Field description 写清楚：直接影响提取准确率
- 避免过深嵌套：2-3 层为佳
- Optional 用好：不确定的字段标 Optional
- 枚举约束：有限选项用 Enum 而非 str

::: warning 需要本地运行
完整实现见 `structured_output/extraction.py`，包含简历提取、发票解析等完整案例。
:::

---

::: tip 下一步
- [错误处理](/engineering/error-handling) — 输出解析失败的容错机制
- [API 服务](/production/api-service) — 在 API 中使用结构化输出
:::
