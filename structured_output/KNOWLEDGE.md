# LLM 结构化输出（Structured Output）完全知识手册

> 本文档系统性地覆盖 LLM 结构化输出的所有核心知识点。
> 配合 `extraction.py` 代码阅读效果更佳。

---

## 目录

1. [为什么需要结构化输出](#1-为什么需要结构化输出)
2. [JSON Mode vs Function Calling vs Structured Output](#2-json-mode-vs-function-calling-vs-structured-output)
3. [Pydantic 模型定义基础](#3-pydantic-模型定义基础)
4. [with_structured_output() 方法详解](#4-with_structured_output-方法详解)
5. [Field 描述与约束](#5-field-描述与约束)
6. [嵌套模型与复杂类型](#6-嵌套模型与复杂类型)
7. [Optional 字段与默认值](#7-optional-字段与默认值)
8. [列表类型与枚举类型](#8-列表类型与枚举类型)
9. [自定义验证器](#9-自定义验证器pydantic-validator)
10. [输出解析失败的处理策略](#10-输出解析失败的处理策略)
11. [信息提取实战](#11-信息提取实战)
12. [与 RAG 结合的结构化提取](#12-与-rag-结合的结构化提取)
13. [流式结构化输出](#13-流式结构化输出)
14. [性能与准确性优化](#14-性能与准确性优化)
15. [多模型结构化输出对比](#15-多模型结构化输出对比)

---

## 1. 为什么需要结构化输出

### 1.1 非结构化 vs 结构化

```
对应 extraction.py 前置科普一

非结构化输出 (纯文本):
  LLM 返回: "这位候选人叫张三，28岁，会Python和Java，期望薪资25k"
  → 你需要正则/NLP 从中提取信息 → 容易出错、难维护

结构化输出 (类型安全的对象):
  LLM 返回: ResumeInfo(name="张三", age=28, skills=["Python","Java"], salary=25000)
  → 直接用 .name .age 访问 → 100% 类型安全

  ┌──────────────────────────────────────────────────────────┐
  │  非结构化文本           结构化输出                        │
  │  ──────────────        ──────────────                    │
  │  "张三, 28岁"    →    {"name": "张三", "age": 28}       │
  │                                                          │
  │  后续处理:                                               │
  │  非结构化: 正则提取 → 容易失败                           │
  │  结构化:   result.name → 永远不会错                      │
  └──────────────────────────────────────────────────────────┘
```

### 1.2 应用场景

```
  ┌──────────────────────────────────────────────────────────┐
  │  场景                  │ 需要提取的结构                    │
  ├──────────────────────────────────────────────────────────┤
  │  简历解析              │ 姓名/年龄/技能/薪资              │
  │  发票识别              │ 编号/日期/金额/商品明细          │
  │  合同分析              │ 甲乙方/条款/金额/日期            │
  │  客服工单              │ 类别/优先级/情感/关键信息        │
  │  用户评论              │ 情感/评分/关键词/改进建议        │
  │  新闻分类              │ 类别/实体/时间/地点              │
  │  医疗记录              │ 症状/诊断/用药/注意事项          │
  └──────────────────────────────────────────────────────────┘

对应 extraction.py 第3章的简历提取和第4章的发票提取
```

---

## 2. JSON Mode vs Function Calling vs Structured Output

### 2.1 三种方式对比

```
  ┌─────────────────┬──────────────────────────────────────────┐
  │ 方式             │ 机制与特点                                │
  ├─────────────────┼──────────────────────────────────────────┤
  │ JSON Mode       │ 只保证输出是合法 JSON                     │
  │                 │ 不保证符合特定 Schema                     │
  │                 │ 需要在 Prompt 中描述期望结构              │
  │                 │ 可靠性: 中                                │
  ├─────────────────┼──────────────────────────────────────────┤
  │ Function Calling│ 通过 tools/functions 参数定义 Schema      │
  │                 │ LLM 被强制输出符合 Schema 的 JSON         │
  │                 │ 可靠性: 高                                │
  │                 │ 返回的是 function call 格式               │
  ├─────────────────┼──────────────────────────────────────────┤
  │ Structured      │ LangChain 的封装 (底层用 Function Calling)│
  │ Output          │ 输入: Pydantic 模型                      │
  │                 │ 输出: Pydantic 实例 (不是 dict!)          │
  │                 │ 可靠性: 最高 (类型验证)                   │
  └─────────────────┴──────────────────────────────────────────┘
```

### 2.2 可靠性递进

```
  可靠性:  StrOutputParser < JSON Mode < Function Calling < with_structured_output

  StrOutputParser:
    LLM 输出什么就是什么，完全无结构保证

  JSON Mode (response_format={"type": "json_object"}):
    保证输出是合法 JSON，但字段名/类型不保证
    可能输出 {"answer": "..."} 而你期望 {"name": "..."}

  Function Calling:
    保证输出符合 JSON Schema，字段名和类型都正确
    但返回的是 dict，没有 Python 类型验证

  with_structured_output():
    Function Calling + Pydantic 验证
    返回的是 Python 对象，类型 100% 安全

对应 extraction.py 第5章的对比总结
```

---

## 3. Pydantic 模型定义基础

### 3.1 BaseModel 基础

```python
# 对应 extraction.py 第1章

from pydantic import BaseModel, Field
from typing import Optional

class ResumeInfo(BaseModel):
    """简历信息数据模型"""
    name: str                          # 必填字符串
    age: int                           # 必填整数
    skills: list[str]                  # 字符串列表
    expected_salary: int               # 必填整数
    is_remote_ok: bool = False         # 带默认值的布尔
    years_of_experience: Optional[int] = None  # 可选整数

# 使用:
info = ResumeInfo(name="张三", age=28, skills=["Python"], expected_salary=25000)
print(info.name)         # "张三" (类型安全的属性访问)
print(info.model_dump()) # 转为 dict
print(info.model_json_schema())  # 查看 JSON Schema
```

### 3.2 Pydantic 的类型验证

```
Pydantic 自动做类型转换和验证:

  age: int
    输入 28     → 通过 (int)
    输入 "28"   → 通过 (str 自动转 int)
    输入 "abc"  → 报错! (无法转为 int)

  skills: list[str]
    输入 ["Python", "Java"]  → 通过
    输入 "Python"            → 报错! (不是 list)
    输入 [1, 2, 3]           → 通过 (int 自动转 str)

这就是"类型安全"的威力:
  错误数据在进入系统时就被拦截!
```

---

## 4. with_structured_output() 方法详解

### 4.1 工作原理

```
对应 extraction.py 前置科普二

  ┌────────────────────────────────────────────────────────────┐
  │  with_structured_output(PydanticModel) 的内部流程:         │
  │                                                            │
  │  Step 1: 读取 PydanticModel 的 JSON Schema                 │
  │    → 字段名、类型、描述、约束                               │
  │                                                            │
  │  Step 2: 转换为 OpenAI Function Calling 格式               │
  │    → tools=[{"function": {"name": "...", "parameters": schema}}] │
  │                                                            │
  │  Step 3: LLM 生成符合 Schema 的 JSON                       │
  │    → {"name": "张三", "age": 28, ...}                      │
  │                                                            │
  │  Step 4: Pydantic 解析 JSON → Python 对象                  │
  │    → ResumeInfo(name="张三", age=28, ...)                  │
  │                                                            │
  │  你拿到的: 类型安全的 Pydantic 实例!                        │
  └────────────────────────────────────────────────────────────┘
```

### 4.2 基本用法

```python
# 对应 extraction.py 第2章

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 创建结构化输出 LLM
structured_llm = llm.with_structured_output(ResumeInfo)

# 调用: 输入是 messages，输出是 Pydantic 对象 (不是字符串!)
result = structured_llm.invoke("我叫张三，28岁，会Python")
# result 类型: ResumeInfo
# result.name → "张三"
# result.age → 28

# 与 Prompt 组合成链
chain = prompt | structured_llm
result = chain.invoke({"text": "简历文本..."})
```

### 4.3 method 参数

```python
# with_structured_output 支持不同的底层实现方式

# 方式一: function_calling (默认, 最可靠)
structured_llm = llm.with_structured_output(MyModel, method="function_calling")

# 方式二: json_mode (部分模型不支持 function calling 时用)
structured_llm = llm.with_structured_output(MyModel, method="json_mode")
# 需要在 prompt 中明确要求输出 JSON

# 方式三: json_schema (OpenAI 新特性, 严格模式)
structured_llm = llm.with_structured_output(MyModel, method="json_schema", strict=True)
# 100% 保证输出符合 schema (比 function_calling 更严格)
```

---

## 5. Field 描述与约束

### 5.1 description 的重要性

```
对应 extraction.py 前置科普三

Field(description=...) 是给大模型看的"说明书":

  差的描述:
    name: str = Field(description="名字")
    → 大模型: "填姓？还是全名？还是英文名？"

  好的描述:
    name: str = Field(description="候选人的全名（包括姓和名），英文名也完整提取")
    → 大模型: "张三" / "John Smith" (精准!)

  description 写作原则:
    1. 明确告诉大模型"这个字段要填什么"
    2. 给出转换规则 ("25k" → 25000)
    3. 说明边界情况 ("如果未提及则为 null")
    4. 给出示例 (可选但有帮助)
```

### 5.2 常见约束

```python
from pydantic import Field

class Product(BaseModel):
    # 基本描述
    name: str = Field(description="产品名称")

    # 数值约束
    price: float = Field(description="价格", gt=0, le=1000000)
    # gt=0: 必须大于0
    # le=1000000: 必须小于等于100万

    # 字符串约束
    sku: str = Field(description="SKU编码", min_length=6, max_length=12)

    # 列表约束
    tags: list[str] = Field(description="标签", min_length=1, max_length=10)
    # 至少1个标签，最多10个

    # 默认值
    is_active: bool = Field(default=True, description="是否在售")
```

---

## 6. 嵌套模型与复杂类型

### 6.1 模型嵌套

```python
# 对应 extraction.py 第4章

class InvoiceItem(BaseModel):
    """发票中的单个商品"""
    item_name: str = Field(description="商品名称")
    quantity: int = Field(description="数量")
    unit_price: float = Field(description="单价")

class InvoiceInfo(BaseModel):
    """发票信息（嵌套结构）"""
    invoice_number: str = Field(description="发票号码")
    date: str = Field(description="日期 YYYY-MM-DD")
    items: list[InvoiceItem] = Field(description="商品明细列表")  # ← 嵌套!
    total_amount: float = Field(description="总金额")

# LLM 能理解并生成嵌套结构:
# {
#   "invoice_number": "INV-001",
#   "items": [
#     {"item_name": "GPU服务器", "quantity": 2, "unit_price": 15000},
#     {"item_name": "咨询服务", "quantity": 1, "unit_price": 8000}
#   ],
#   "total_amount": 38000
# }
```

### 6.2 嵌套深度建议

```
  ┌─────────────────────────────────────────────────────────┐
  │  嵌套层级    │ 可靠性    │ 建议                          │
  ├─────────────────────────────────────────────────────────┤
  │  1层(扁平)   │ 极高      │ 优先使用                      │
  │  2层嵌套     │ 高        │ 常见且可靠 (如发票+明细)       │
  │  3层嵌套     │ 中        │ 可用，注意 description 清晰   │
  │  4层+        │ 低        │ 不推荐，考虑拆分为多次提取    │
  └─────────────────────────────────────────────────────────┘

  嵌套越深，LLM 出错概率越高
  优化策略: 复杂结构拆分为多次简单提取
```

---

## 7. Optional 字段与默认值

### 7.1 Optional 类型

```python
from typing import Optional

class PersonInfo(BaseModel):
    name: str                            # 必填: 必须有值
    age: int                             # 必填: 必须有值
    email: Optional[str] = None          # 可选: 可以为 None
    phone: Optional[str] = None          # 可选: 可以为 None
    years_exp: Optional[int] = None      # 可选: 可以为 None

# Optional[int] 的含义:
#   "这个字段的值可以是 int，也可以是 None"
#   如果文本中没有提到相关信息，LLM 应该返回 null

# 对应 extraction.py 中:
years_of_experience: Optional[int] = Field(
    default=None,
    description="工作年限，如果文本中未提及则为null"
)
```

### 7.2 默认值设计原则

```
  字段类型          默认值建议              原因
  ────────────────────────────────────────────────────
  bool             False                  "未提及"通常等于"否"
  Optional[str]    None                   "未提及"用 None 表示
  Optional[int]    None                   不要用 0（0 有语义）
  list[str]        [] (空列表)            没提到技能 ≠ null
  str              不设默认值 (必填)       关键字段必须有

  反面教材:
    age: int = 0     ← 0岁有意义! 应该用 Optional[int] = None
    salary: int = -1 ← 魔法数字! 应该用 Optional[int] = None
```

---

## 8. 列表类型与枚举类型

### 8.1 列表类型

```python
from typing import Literal
from enum import Enum

class SkillAnalysis(BaseModel):
    # 字符串列表
    skills: list[str] = Field(description="技能列表")
    # → ["Python", "Java", "SQL"]

    # 嵌套对象列表
    experiences: list[WorkExperience] = Field(description="工作经历列表")
    # → [WorkExperience(...), WorkExperience(...)]

    # 限制列表长度
    top_skills: list[str] = Field(description="最擅长的前3个技能", max_length=3)
```

### 8.2 枚举类型

```python
# 方式一: Literal (推荐, 简洁)
class TicketInfo(BaseModel):
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="工单优先级"
    )
    category: Literal["技术问题", "账户问题", "计费问题", "其他"] = Field(
        description="工单类别"
    )

# 方式二: Enum (更正式)
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TicketInfo(BaseModel):
    priority: Priority = Field(description="工单优先级")

# 效果: LLM 只能输出枚举中的值，否则 Pydantic 会报错
# 枚举的 description 会告诉 LLM 可选值是什么
```

### 8.3 Literal vs Enum 选择

```
  ┌──────────────┬─────────────────────────────────────────┐
  │  Literal     │ 简单场景，值固定，不需要复杂操作        │
  │              │ 例: priority: Literal["low","high"]      │
  ├──────────────┼─────────────────────────────────────────┤
  │  Enum        │ 需要枚举值的方法/属性/跨模块复用        │
  │              │ 例: Priority.HIGH.value                  │
  └──────────────┴─────────────────────────────────────────┘
```

---

## 9. 自定义验证器（Pydantic Validator）

### 9.1 字段验证器

```python
from pydantic import BaseModel, Field, field_validator

class ResumeInfo(BaseModel):
    name: str
    age: int
    salary: int

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        """年龄必须在合理范围内"""
        if v < 16 or v > 80:
            raise ValueError(f"年龄 {v} 不在合理范围 (16-80)")
        return v

    @field_validator("salary")
    @classmethod
    def validate_salary(cls, v):
        """薪资标准化: 如果看起来像月薪(k)，转为元"""
        if v < 100:  # 可能是"25"代表25k
            return v * 1000
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        """姓名不能为空"""
        if not v.strip():
            raise ValueError("姓名不能为空")
        return v.strip()
```

### 9.2 模型验证器（跨字段验证）

```python
from pydantic import model_validator

class DateRange(BaseModel):
    start_date: str
    end_date: str

    @model_validator(mode="after")
    def validate_dates(self):
        """结束日期必须晚于开始日期"""
        if self.end_date < self.start_date:
            raise ValueError("结束日期不能早于开始日期")
        return self
```

### 9.3 验证器与 LLM 输出

```
注意: 验证器在 Pydantic 解析时运行

  LLM 输出 JSON → Pydantic 解析 → 验证器执行
                                         │
                                    验证通过 → 返回对象
                                    验证失败 → 抛出 ValidationError

  如果验证失败:
    方案1: 重试 (让 LLM 重新生成)
    方案2: OutputFixingParser (让 LLM 修复)
    方案3: 放宽验证条件
```

---

## 10. 输出解析失败的处理策略

### 10.1 常见失败原因

```
  原因一: LLM 输出不是合法 JSON
    LLM: "好的，以下是结果：{"name": ...}"  ← 多了前缀

  原因二: 字段类型不匹配
    期望 age: int，LLM 返回 age: "二十八"

  原因三: 缺少必填字段
    LLM 没有返回 name 字段

  原因四: 枚举值不匹配
    期望 priority: "low"|"high"，LLM 返回 "中等"

  原因五: 嵌套结构错误
    items 应该是数组，LLM 返回了单个对象
```

### 10.2 处理策略

```python
from langchain.output_parsers import OutputFixingParser, RetryWithErrorOutputParser

# 策略一: 自动修复 (用 LLM 修复格式错误)
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=llm
)
# 流程: 解析失败 → 把错误信息+原输出发给LLM → LLM修复 → 再次解析

# 策略二: 带错误重试 (重新生成)
retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=base_parser,
    llm=llm
)
# 流程: 解析失败 → 把原始prompt+错误信息发给LLM → LLM重新生成

# 策略三: 多次尝试
def robust_structured_call(chain, input_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        except (ValidationError, OutputParserException) as e:
            if attempt == max_retries - 1:
                raise
            # 可以调整 temperature 重试
            continue
```

---

## 11. 信息提取实战

### 11.1 简历信息提取

```python
# 对应 extraction.py 第3章

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的简历信息提取助手。
提取规则：
1. 未明确提及的信息用 null
2. 薪资统一转为元/月 ("25k"=25000)
3. 技能细分不合并 ("Python和Java" → ["Python", "Java"])
4. 中文数字转阿拉伯数字"""),
    ("human", "提取以下文本中的候选人信息：\n\n{text}"),
])

chain = extraction_prompt | llm.with_structured_output(ResumeInfo)
result = chain.invoke({"text": "我叫张三，28岁，会Python..."})
```

### 11.2 发票信息提取

```python
# 对应 extraction.py 第4章

class InvoiceItem(BaseModel):
    item_name: str = Field(description="商品名称")
    quantity: int = Field(description="数量")
    unit_price: float = Field(description="单价(元)")

class InvoiceInfo(BaseModel):
    invoice_number: str = Field(description="发票号码")
    date: str = Field(description="开票日期 YYYY-MM-DD")
    seller: str = Field(description="销售方公司名称")
    buyer: str = Field(description="购买方公司名称")
    items: list[InvoiceItem] = Field(description="商品明细")
    total_amount: float = Field(description="总金额(元)")
    has_tax: bool = Field(description="是否含税")
```

### 11.3 合同关键条款提取

```python
class ContractInfo(BaseModel):
    party_a: str = Field(description="甲方名称")
    party_b: str = Field(description="乙方名称")
    contract_type: Literal["销售", "服务", "租赁", "合作", "其他"]
    start_date: str = Field(description="合同开始日期")
    end_date: str = Field(description="合同结束日期")
    total_value: Optional[float] = Field(description="合同总金额")
    payment_terms: str = Field(description="付款条件摘要")
    key_obligations: list[str] = Field(description="核心义务条款(前5条)")
    termination_clause: Optional[str] = Field(description="终止条款摘要")
```

---

## 12. 与 RAG 结合的结构化提取

### 12.1 RAG + 结构化输出

```
  传统 RAG:
    检索文档 → 拼接 Prompt → LLM 生成自由文本

  结构化 RAG:
    检索文档 → 拼接 Prompt → LLM 生成结构化数据

  应用场景:
    "从知识库中提取所有产品的规格参数表"
    "基于检索到的文档，提取关键事实并标注来源"
```

### 12.2 带来源引用的结构化提取

```python
class ExtractedFact(BaseModel):
    fact: str = Field(description="提取的事实")
    source: str = Field(description="信息来源文档标题")
    confidence: Literal["high", "medium", "low"] = Field(
        description="置信度: high=明确陈述, medium=可推断, low=不确定"
    )

class StructuredAnswer(BaseModel):
    answer: str = Field(description="综合回答")
    facts: list[ExtractedFact] = Field(description="支撑事实及来源")
    unanswerable_parts: Optional[str] = Field(
        description="无法从文档中回答的部分"
    )

# RAG + 结构化输出链
rag_chain = (
    retriever
    | format_docs
    | structured_prompt
    | llm.with_structured_output(StructuredAnswer)
)
```

---

## 13. 流式结构化输出

### 13.1 挑战

```
流式输出 + 结构化 = 矛盾?

  普通流式: 逐 token 输出 → 用户立即看到文字
  结构化: 必须等完整 JSON 才能解析 → 无法逐步展示

解决方案:
  1. 部分解析 (Partial Parsing): 在 JSON 不完整时尝试解析已有字段
  2. 流式事件: 按字段逐步返回 (field1完成 → 发送, field2完成 → 发送)
```

### 13.2 LangChain 流式结构化输出

```python
# 使用 .stream() 获取部分结果
structured_llm = llm.with_structured_output(ResumeInfo)

for partial in structured_llm.stream("简历文本..."):
    # partial 是 Pydantic 对象，但可能只有部分字段
    print(partial)  # ResumeInfo(name="张三", age=None, skills=None, ...)
    # 随着 token 生成，字段逐步被填充

# 注意: 不是所有模型都支持流式结构化输出
# 目前支持: OpenAI (gpt-4o+), Anthropic (Claude 3+)
```

---

## 14. 性能与准确性优化

### 14.1 提高准确性的技巧

```
  ┌─────────────────────────────────────────────────────────┐
  │  技巧                     │ 效果                         │
  ├─────────────────────────────────────────────────────────┤
  │  description 写得清晰具体 │ 最重要! 大模型靠这个理解字段 │
  │  temperature=0            │ 确定性输出，减少随机错误     │
  │  给出提取规则             │ 统一转换标准 (25k→25000)     │
  │  给出示例 (Few-shot)      │ 复杂格式时显著提升准确率     │
  │  Literal 限制枚举值       │ 避免输出意外的值             │
  │  拆分复杂为多步简单       │ 嵌套太深时分步提取           │
  └─────────────────────────────────────────────────────────┘
```

### 14.2 性能优化

```
  优化一: 减少 Schema 复杂度
    字段越少 → token 越少 → 速度越快 → 成本越低
    只定义真正需要的字段!

  优化二: 批量提取
    10 份简历一个一个提取 → 10 次 API 调用
    10 份简历一次性提取 → 1 次 API 调用 (如果总 token 允许)

  优化三: 缓存
    相同输入的提取结果缓存
    (→ caching/KNOWLEDGE.md)

  优化四: 模型选择
    简单提取 (姓名/日期): GPT-3.5-turbo 够用
    复杂提取 (合同条款): 需要 GPT-4o
    批量提取 (成本敏感): GPT-4o-mini
```

---

## 15. 多模型结构化输出对比

### 15.1 各模型支持情况

```
  ┌─────────────────────┬──────────┬───────────────────────────┐
  │ 模型                │ 支持方式  │ 注意事项                   │
  ├─────────────────────┼──────────┼───────────────────────────┤
  │ GPT-4o / GPT-4o-mini│ 原生支持  │ 支持 strict mode          │
  │ GPT-3.5-turbo       │ 原生支持  │ 复杂结构偶有错误          │
  │ Claude 3.5 Sonnet   │ 工具调用  │ 通过 tool_use 实现        │
  │ Llama 3 (本地)      │ 有限支持  │ 需要 Outlines/Guidance    │
  │ Qwen 2.5            │ 工具调用  │ 中文结构化能力强          │
  │ Gemini 2.0          │ 原生支持  │ responseSchema 参数       │
  └─────────────────────┴──────────┴───────────────────────────┘
```

### 15.2 准确性对比（经验值）

```
  任务: 从中文简历提取 6 个字段

  模型              准确率(全部字段正确)    备注
  ─────────────────────────────────────────────────
  GPT-4o            95-98%                 最稳定
  GPT-4o-mini       90-95%                 性价比之王
  Claude 3.5 Sonnet 93-97%                 中文理解好
  GPT-3.5-turbo     80-90%                 复杂场景下降
  Llama 3 70B       85-92%                 需要好的 prompt
  Qwen 2.5 72B      90-95%                 中文场景推荐
```

### 15.3 选型建议

```
  优先级一: 准确性 → GPT-4o / Claude 3.5 Sonnet
  优先级二: 成本   → GPT-4o-mini / Qwen 2.5
  优先级三: 隐私   → Llama 3 / Qwen (本地部署)
  优先级四: 速度   → GPT-3.5-turbo / GPT-4o-mini

  对应 extraction.py 中使用 temperature=0 确保输出确定性
```

---

## 附录 A：本文件夹代码与知识点对应

| 代码位置 | 覆盖知识点 | 对应本文档章节 |
|---------|-----------|---------------|
| `extraction.py` 前置科普 | 结构化输出原理、对比正则 | 第1-2节 |
| `extraction.py` 第1章 | Pydantic 模型、Field、Schema | 第3节、第5节 |
| `extraction.py` 第2章 | with_structured_output、链构建 | 第4节 |
| `extraction.py` 第3章 | 简历提取实战、类型验证 | 第11节 |
| `extraction.py` 第4章 | 嵌套模型、发票提取 | 第6节、第11节 |
| `extraction.py` 第5章 | 三种输出方式对比 | 第2节 |

---

## 附录 B：推荐学习路径

```
入门（1天）：
  第1-2节 → 理解为什么需要结构化输出
  第3-4节 → 运行 extraction.py 第1-2章
  第5节 → 掌握 Field description 写法

进阶（2-3天）：
  第6-8节 → 嵌套模型、枚举、列表
  第9节 → Pydantic 验证器
  第11节 → 运行 extraction.py 第3-4章

生产（1周）：
  第10节 → 解析失败处理
  第12-13节 → RAG + 结构化、流式
  第14-15节 → 性能优化、模型选型
```

---

> **下一步学习**：前往 `error_handling/KNOWLEDGE.md` 了解结构化输出解析失败时的容错处理，或前往 `observability/KNOWLEDGE.md` 学习监控结构化输出的准确率和延迟。
