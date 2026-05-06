"""
╔══════════════════════════════════════════════════════════════════╗
║         项目六：Structured Output — 结构化输出与实体提取            ║
║         让大模型 100% 稳定地输出严格 JSON，告别正则地狱            ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普一：正则地狱 vs 大模型结构化输出——降维打击】
═══════════════════════════════════════════════════════════════════

想象你是一家 HR 公司，每天收到 1000 份格式各异的简历：

  传统做法（正则表达式）= 用镊子从垃圾堆里挑零件
  ┌─────────────────────────────────────────────────────────────┐
  │  输入："我叫张三，今年28岁，会Python和Java，期望月薪25k"       │
  │                                                             │
  │  提取姓名：r"我叫(.+?)，"           → 能匹配                │
  │  但如果写成"本人张三"呢？           → 正则失效 💥            │
  │  如果写成"张三是我的名字"呢？       → 又要加规则 💥          │
  │  如果是英文名"I'm Tom"呢？          → 完全不匹配 💥         │
  │                                                             │
  │  问题：人类表达方式千变万化，正则是"死规则"，永远追不上！    │
  └─────────────────────────────────────────────────────────────┘

  大模型做法（Structured Output）= 请一个理解人类语言的助手帮你整理
  ┌─────────────────────────────────────────────────────────────┐
  │  输入："我叫张三，今年28岁，会Python和Java，期望月薪25k"       │
  │                                                             │
  │  你告诉大模型：                                              │
  │    "请把这段话整理成这个格式：                               │
  │     {name: 字符串, age: 整数, skills: 列表, salary: 整数}"   │
  │                                                             │
  │  大模型输出：                                                │
  │    {"name": "张三", "age": 28,                              │
  │     "skills": ["Python", "Java"], "salary": 25000}          │
  │                                                             │
  │  无论输入是中文、英文、口语、书面语，大模型都能理解并提取！   │
  └─────────────────────────────────────────────────────────────┘

  核心优势对比：
  ┌──────────────┬─────────────────────────────────────────────┐
  │   正则表达式  │  Structured Output（结构化输出）             │
  ├──────────────┼─────────────────────────────────────────────┤
  │  死规则       │  理解语义（"会"="掌握"="精通"=技能）        │
  │  格式敏感     │  格式无关（口语/书面/中英文都行）            │
  │  维护噩梦     │  零维护（模型自带语言理解能力）              │
  │  返回字符串   │  直接返回 Python 对象（类型安全！）          │
  │  容易漏提取   │  字段定义清晰，缺什么一目了然               │
  └──────────────┴─────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
【前置科普二：with_structured_output() 的工作原理】
═══════════════════════════════════════════════════════════════════

LangChain 的 with_structured_output(PydanticModel) 做了什么？

  第一步：读取你定义的 Pydantic 模型，提取字段名、类型、描述
         ↓
  第二步：把这些信息转换成 OpenAI 的 "function calling" 格式
         （告诉 LLM："你的输出必须是一个严格符合这个 JSON Schema 的对象"）
         ↓
  第三步：LLM 生成符合 Schema 的 JSON 字符串
         ↓
  第四步：LangChain 自动用 Pydantic 解析 JSON → Python 对象
         （如果格式不对，Pydantic 会报错，保证类型安全）

  整个过程对你透明：你只需要定义一个 Pydantic 类，
  调用 llm.with_structured_output(YourClass).invoke(...)，
  就能直接拿到一个类型安全的 Python 对象！

═══════════════════════════════════════════════════════════════════
【前置科普三：Pydantic Field(description=...) 的魔力】
═══════════════════════════════════════════════════════════════════

大模型怎么知道每个字段应该填什么？靠的就是 Field(description=...)！

  class ResumeInfo(BaseModel):
      name: str = Field(description="候选人的全名")
      age: int = Field(description="候选人的年龄，单位为岁")

  LangChain 会把这些 description 发送给大模型，
  大模型读取描述后，知道：
    "name 字段要填候选人的全名"
    "age 字段要填整数类型的年龄"

  ⚠️ 关键：description 写得越清晰，大模型提取越准确！
  坏例子：description="名字"        → 大模型可能只填姓
  好例子：description="候选人的全名，包括姓和名"  → 大模型精准提取

  这就像你给实习生发任务时：
  "帮我把名字提出来"       → 实习生："张？还是张三？"
  "帮我提取候选人的全名"   → 实习生："张三"（准确！）
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Pydantic：Python 生态最强的数据验证库
# BaseModel：所有 Pydantic 模型的基类
# Field    ：字段定义器，可以添加描述、默认值、验证规则
from pydantic import BaseModel, Field

# typing 模块：提供类型注解
from typing import Optional

# LangChain 聊天模型
from langchain_openai import ChatOpenAI

# 提示词模板
from langchain_core.prompts import ChatPromptTemplate

# json 模块：用于美化打印 JSON
import json


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化 LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 0 章：初始化 LLM")
print("=" * 60)

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "kivy-kimi-k2_5"

# 结构化输出场景必须 temperature=0：要求精确、确定性的输出
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.0,
)

print("✅ LLM 初始化完成")
print(f"   模型: {MODEL_NAME}")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：用 Pydantic 定义数据模型
# 目标：定义"简历信息"的严格数据结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：用 Pydantic 定义数据模型")
print("=" * 60)
print()

# ── Pydantic BaseModel 是什么？──────────────────────────
#
# Pydantic 是 Python 最流行的"数据验证 + 序列化"库。
# 用 BaseModel 定义一个类，就相当于定义了一份"数据合同"：
#   ① 字段名是什么
#   ② 每个字段的类型是什么（str? int? list?）
#   ③ 哪些字段是必填的，哪些是可选的
#   ④ 每个字段的含义是什么（description）
#
# 当你把数据塞进这个模型时，Pydantic 会自动验证：
#   age = "二十八"  → 报错！因为 age 类型是 int，不接受字符串
#   age = 28        → 通过！
#
# 这就是"类型安全"的威力——错误在"进门"时就被拦截！


class ResumeInfo(BaseModel):
    """
    简历信息数据模型。

    ⭐ 大模型如何使用这个类？
    LangChain 会把这个类的 JSON Schema 发送给大模型，包含：
      - 字段名（name, age, skills...）
      - 字段类型（string, integer, array...）
      - 字段描述（Field 的 description 参数）

    大模型看到的信息大致是：
    {
      "name": {"type": "string", "description": "候选人的全名..."},
      "age":  {"type": "integer", "description": "候选人的年龄..."},
      ...
    }

    大模型根据这些描述，从输入文本中精准提取对应信息。
    description 越清晰 → 提取越准确！
    """

    # ── 字段一：姓名（str 类型）────────────────────────────
    #
    # Field(description=...) 中的描述文本会被发送给大模型！
    # 大模型靠这段描述理解"这个字段要填什么"。
    name: str = Field(
        description="候选人的全名（包括姓和名），如果是英文名也要完整提取"
    )

    # ── 字段二：年龄（int 类型）────────────────────────────
    #
    # 注意类型是 int，不是 str！
    # 即使输入中写的是"二十八岁"，大模型也要转换成整数 28。
    # Pydantic 会做类型验证：如果大模型返回了字符串"28"，
    # Pydantic 会尝试转换为 int；如果返回"二十八"则直接报错。
    age: int = Field(
        description="候选人的年龄，必须是整数（如果文本中是中文数字如'二十八'，转换为阿拉伯数字28）"
    )

    # ── 字段三：技能列表（list[str] 类型）─────────────────
    #
    # list[str] 表示"字符串数组"——一个列表，里面每个元素是字符串。
    # 大模型会把文本中提到的所有技能提取出来，放进这个数组。
    # 例如："会 Python 和 Java" → ["Python", "Java"]
    skills: list[str] = Field(
        description="候选人掌握的技术技能列表，每个技能作为数组中的一个独立元素"
    )

    # ── 字段四：期望月薪（int 类型）───────────────────────
    #
    # 统一转换为整数（元/月）。
    # 大模型需要做单位转换："25k" → 25000，"2万5" → 25000
    expected_salary: int = Field(
        description="候选人的期望月薪，单位为人民币元（如'25k'转换为25000，'2万'转换为20000）"
    )

    # ── 字段五：是否接受远程（bool 类型）──────────────────
    #
    # bool 类型只有 True/False 两个值。
    # 大模型需要从文本中推断候选人的意愿：
    #   "我希望能远程办公" → True
    #   "我更喜欢坐班"     → False
    #   未提及             → False（默认值）
    is_remote_ok: bool = Field(
        default=False,
        description="候选人是否接受远程工作（True=接受远程, False=不接受或未提及）"
    )

    # ── 字段六：工作年限（Optional[int] 类型）─────────────
    #
    # Optional[int] = 可以是 int，也可以是 None（表示"未提供"）。
    # 这是"可选字段"的标准写法。
    # 如果文本中没有提到工作年限，大模型应该返回 None 而不是瞎猜。
    years_of_experience: Optional[int] = Field(
        default=None,
        description="候选人的工作年限（整数），如果文本中未提及则为null"
    )


# ── 打印 Pydantic 模型的 JSON Schema ─────────────────────
#
# 这就是大模型实际看到的"任务说明书"！
# LangChain 把这个 Schema 作为 function calling 的参数发送给 LLM。

print('【ResumeInfo 的 JSON Schema（大模型看到的"任务说明书"）】')
print("-" * 60)
schema = ResumeInfo.model_json_schema()
print(json.dumps(schema, indent=2, ensure_ascii=False))
print("-" * 60)
print()
print("💡 大模型通过阅读上面的 Schema（特别是每个字段的 description），")
print("   来决定从输入文本中提取什么信息、填到哪个字段里。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：构建结构化输出链
# 目标：用 with_structured_output() 让 LLM 直接返回 Pydantic 对象
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：构建结构化输出链")
print("=" * 60)
print()

# ── with_structured_output() 的用法 ──────────────────────
#
# llm.with_structured_output(PydanticModel) 返回一个新的 Runnable：
#   输入：普通的 messages/prompt
#   输出：PydanticModel 的实例（不是字符串！是真正的 Python 对象！）
#
# 内部流程：
#   ① LangChain 把 PydanticModel 的 Schema 转成 OpenAI function calling 格式
#   ② 发送给 LLM 时，附带 tools=[{function definition}]
#   ③ LLM 被强制以 function call 格式输出 JSON
#   ④ LangChain 拿到 JSON 后，用 PydanticModel.model_validate() 解析
#   ⑤ 返回给你的是一个 PydanticModel 实例（类型安全！）
#
# ⚠️ 避坑指南：不是所有 LLM 都支持 structured_output！
#   支持的前提：LLM 必须支持 function calling（OpenAI、兼容接口）。
#   如果你的 LLM 不支持 function calling，需要用 PydanticOutputParser
#   （靠 prompt 引导 LLM 输出 JSON，然后手动解析——不如 structured_output 可靠）。

structured_llm = llm.with_structured_output(ResumeInfo)

print("  ✅ 结构化输出 LLM 创建完成！")
print("     llm.with_structured_output(ResumeInfo)")
print("     → 输出类型：ResumeInfo 实例（不是字符串）")
print()

# ── 构建提取 Prompt ─────────────────────────────────────
#
# Prompt 的作用：告诉 LLM "你是一个信息提取助手"。
# 注意：不需要在 Prompt 里重复描述字段含义！
# 因为 with_structured_output 已经通过 Schema 告诉 LLM 每个字段的含义了。
# Prompt 只需要说明任务和要求即可。

extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的简历信息提取助手。
你的任务是从用户提供的非结构化文本中，精准提取候选人的关键信息。

提取规则：
1. 如果某项信息在文本中没有明确提及，对应字段使用 null（可选字段）或合理默认值
2. 薪资统一转换为人民币元/月（"25k"=25000, "2万5"=25000）
3. 技能列表要细分，不要合并（"Python和Java" → ["Python", "Java"]）
4. 年龄如果是中文数字要转为阿拉伯数字"""),
    ("human", "请从以下文本中提取候选人信息：\n\n{text}"),
])

# 用 LCEL 管道把 prompt 和 structured_llm 串联
extraction_chain = extraction_prompt | structured_llm

print("  ✅ 提取链构建完成：extraction_prompt | structured_llm")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：实战演示——从混乱口语中提取结构化数据
# 目标：输入各种风格的自我介绍，验证结构化输出的稳定性
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：实战演示——从混乱口语中提取结构化数据")
print("=" * 60)
print()

# 准备多个风格迥异的输入文本，模拟真实场景中千变万化的表达方式
TEST_INPUTS = [
    {
        "title": "口语化自我介绍（中文）",
        "text": """嗨你好！我叫李明，今年28岁。之前在字节跳动干了3年后端开发，
主要用的Python和Go，也会一些Java。Redis和MySQL用得很熟。
最近在学Kubernetes。我期望薪资大概在35k左右吧，
最好能远程办公，毕竟通勤太累了哈哈。""",
    },
    {
        "title": "碎片化信息（夹杂英文）",
        "text": """本人王芳，female，刚过完26岁生日。
技术栈的话，前端React/Vue/TypeScript都OK，
后端Node.js也能写。CSS嘛，虽然不太喜欢但也算会吧。
salary expectation是28k一个月，坐班remote都行。
对了忘了说，我有4年经验了。""",
    },
    {
        "title": "极简风格（信息缺失多）",
        "text": """张伟，35，全栈。会Python、JavaScript、SQL。
要求月薪四万五。""",
    },
]


def run_extraction(title: str, text: str) -> ResumeInfo:
    """
    执行一轮信息提取，打印完整的输入→处理→输出→验证过程。
    """
    print("━" * 60)
    print(f"【{title}】")
    print("━" * 60)
    print()

    # 打印输入
    print("  📝 原始输入文本：")
    print("  ┌" + "─" * 54 + "┐")
    for line in text.strip().split("\n"):
        print(f"  │  {line.strip()}")
    print("  └" + "─" * 54 + "┘")
    print()

    # 调用提取链
    print("  ⏳ 正在调用 LLM 进行结构化提取...")
    result = extraction_chain.invoke({"text": text})
    print("  ✅ 提取完成！")
    print()

    # ━━━ 打印结构化输出结果 ━━━
    print("  📦 结构化输出结果：")
    print("  ┌" + "─" * 54 + "┐")
    print(f"  │  name               : {result.name!r}")
    print(f"  │  age                : {result.age}")
    print(f"  │  skills             : {result.skills}")
    print(f"  │  expected_salary    : {result.expected_salary}")
    print(f"  │  is_remote_ok       : {result.is_remote_ok}")
    print(f"  │  years_of_experience: {result.years_of_experience}")
    print("  └" + "─" * 54 + "┘")
    print()

    # ━━━ 打印 JSON 格式 ━━━
    #
    # model_dump() 把 Pydantic 对象转成 Python 字典
    # json.dumps() 把字典转成格式化的 JSON 字符串
    result_dict = result.model_dump()
    print("  📋 JSON 格式（可直接存入数据库/发送给前端）：")
    print(json.dumps(result_dict, indent=4, ensure_ascii=False))
    print()

    # ━━━ 严格类型验证（⭐ 关键！证明返回的是对象不是字符串）━━━
    #
    # 这一步非常重要：用代码证明 with_structured_output 返回的
    # 确实是类型安全的 Python 对象，而不是一段需要手动解析的字符串！

    print("  🔬 类型安全验证：")
    print(f"     result 的类型      : {type(result).__name__}")
    print(f"     是 ResumeInfo 实例? : {isinstance(result, ResumeInfo)}")
    print(f"     是 BaseModel 实例? : {isinstance(result, BaseModel)}")
    print(f"     result.name 的类型 : {type(result.name).__name__}")
    print(f"     result.age 的类型  : {type(result.age).__name__}")
    print(f"     result.skills 的类型: {type(result.skills).__name__}")
    print(f"     skills[0] 的类型   : {type(result.skills[0]).__name__ if result.skills else 'N/A'}")
    print()

    # 用 assert 断言验证（如果类型不对会直接报错，保证严格性）
    assert isinstance(result, ResumeInfo), "返回值不是 ResumeInfo 实例！"
    assert isinstance(result.name, str), "name 不是字符串！"
    assert isinstance(result.age, int), "age 不是整数！"
    assert isinstance(result.skills, list), "skills 不是列表！"
    assert isinstance(result.expected_salary, int), "expected_salary 不是整数！"
    assert isinstance(result.is_remote_ok, bool), "is_remote_ok 不是布尔值！"
    assert all(isinstance(s, str) for s in result.skills), "skills 中有非字符串元素！"

    print("  ✅ 全部 assert 断言通过！类型 100% 正确。")
    print("     → 返回的是真正的 Python 对象，不是字符串！")
    print("     → 可以直接 result.name 访问属性，无需 json.loads()！")
    print()

    return result


# ── 运行三轮提取演示 ──────────────────────────────────────

results = []
for test_case in TEST_INPUTS:
    result = run_extraction(test_case["title"], test_case["text"])
    results.append(result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：进阶——批量提取 + 嵌套模型
# 目标：演示更复杂的数据结构（嵌套对象、数组中的对象）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 4 章：进阶——嵌套模型（从发票中提取结构化数据）")
print("=" * 60)
print()

# ── 嵌套 Pydantic 模型 ──────────────────────────────────
#
# 真实场景中，数据结构往往不是扁平的，而是嵌套的：
#   发票 → 包含多个商品行 → 每行有名称、单价、数量
#
# Pydantic 完美支持嵌套：模型 A 的字段类型可以是模型 B！
# 大模型也能理解嵌套结构——Schema 中嵌套对象会被展开成完整描述。


class InvoiceItem(BaseModel):
    """发票中的单个商品/服务项目"""
    item_name: str = Field(description="商品或服务的名称")
    quantity: int = Field(description="数量")
    unit_price: float = Field(description="单价（人民币元）")


class InvoiceInfo(BaseModel):
    """
    发票信息数据模型——演示嵌套结构。

    包含：基本信息 + 商品行列表（list[InvoiceItem]）
    大模型会看到嵌套的 Schema 定义，自动提取多层结构。
    """
    invoice_number: str = Field(description="发票号码")
    date: str = Field(description="开票日期，格式为 YYYY-MM-DD")
    seller: str = Field(description="销售方/开票方的公司名称")
    buyer: str = Field(description="购买方的公司名称")
    items: list[InvoiceItem] = Field(description="发票中的商品/服务明细列表")
    total_amount: float = Field(description="发票总金额（人民币元）")
    has_tax: bool = Field(description="是否含税（True=含税发票, False=不含税）")


# ── 构建发票提取链 ────────────────────────────────────────

invoice_llm = llm.with_structured_output(InvoiceInfo)

invoice_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的发票信息提取助手。
请从用户提供的发票文本或描述中，精准提取所有关键信息。
日期统一转换为 YYYY-MM-DD 格式。
金额统一为人民币元。"""),
    ("human", "请从以下内容中提取发票信息：\n\n{text}"),
])

invoice_chain = invoice_prompt | invoice_llm

# ── 测试发票提取 ──────────────────────────────────────────

INVOICE_TEXT = """
收到一张发票，编号是 INV-2024-0815，开票日期是2024年8月15日。
卖方是"北京智能科技有限公司"，买方是"上海数据服务集团"。
上面列了三项：
1. GPU服务器租赁 2台，每台15000元/月
2. 技术咨询服务 1次，8000元
3. 数据标注 5000条，单价0.5元/条
这是一张含税发票，总额40500元。
"""

print("━" * 60)
print("【发票信息提取演示——嵌套结构】")
print("━" * 60)
print()

print("  📝 原始发票文本：")
for line in INVOICE_TEXT.strip().split("\n"):
    print(f"     {line}")
print()

print("  ⏳ 正在提取...")
invoice_result = invoice_chain.invoke({"text": INVOICE_TEXT})
print("  ✅ 提取完成！")
print()

# 打印嵌套结构
print("  📦 结构化输出结果（嵌套对象）：")
print("  ┌" + "─" * 54 + "┐")
print(f"  │  invoice_number : {invoice_result.invoice_number!r}")
print(f"  │  date           : {invoice_result.date!r}")
print(f"  │  seller         : {invoice_result.seller!r}")
print(f"  │  buyer          : {invoice_result.buyer!r}")
print(f"  │  has_tax        : {invoice_result.has_tax}")
print(f"  │  total_amount   : {invoice_result.total_amount}")
print(f"  │  items（嵌套列表）:")
for i, item in enumerate(invoice_result.items, 1):
    print(f"  │    [{i}] {item.item_name} × {item.quantity} @ ¥{item.unit_price}")
print("  └" + "─" * 54 + "┘")
print()

# JSON 格式输出
print("  📋 完整 JSON：")
print(json.dumps(invoice_result.model_dump(), indent=4, ensure_ascii=False))
print()

# 类型验证
print("  🔬 嵌套对象类型验证：")
print(f"     invoice_result 类型         : {type(invoice_result).__name__}")
print(f"     invoice_result.items 类型   : {type(invoice_result.items).__name__}")
print(f"     invoice_result.items[0] 类型: {type(invoice_result.items[0]).__name__}")
print(f"     是 InvoiceItem 实例?        : {isinstance(invoice_result.items[0], InvoiceItem)}")
print()

assert isinstance(invoice_result, InvoiceInfo)
assert isinstance(invoice_result.items, list)
assert all(isinstance(item, InvoiceItem) for item in invoice_result.items)
assert isinstance(invoice_result.has_tax, bool)
assert isinstance(invoice_result.total_amount, (int, float))

print("  ✅ 嵌套模型断言全部通过！")
print("     → 即使是复杂的嵌套结构，大模型也能精准提取！")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 章：对比总结——三种输出方式
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 5 章：三种输出方式对比总结")
print("=" * 60)
print()
print("  ┌───────────────────┬─────────────────────────────────────┐")
print("  │  方式             │  特点                               │")
print("  ├───────────────────┼─────────────────────────────────────┤")
print("  │  StrOutputParser  │  返回纯字符串，最灵活但无结构保证   │")
print("  │  JsonOutputParser │  返回 dict，靠 prompt 引导格式      │")
print("  │  with_structured_ │  返回 Pydantic 对象，类型 100% 安全 │")
print("  │  output()         │  靠 function calling 强制格式       │")
print("  └───────────────────┴─────────────────────────────────────┘")
print()
print("  💡 推荐策略：")
print("     需要自由文本回答 → StrOutputParser")
print("     需要固定格式数据 → with_structured_output()（首选！）")
print()
print()
print("=" * 60)
print("🎉 项目六学习完毕！")
print("=" * 60)
print()
print("💡 核心公式：")
print("   Pydantic 模型（定义结构）")
print("   + with_structured_output()（强制 LLM 输出符合 Schema）")
print("   + Field(description=...)（指导 LLM 理解每个字段）")
print("   = 100% 类型安全的结构化信息提取")
print()
print("💡 实际应用场景：")
print("   ① 简历/名片解析 → 候选人信息结构化入库")
print("   ② 发票/合同 OCR → 关键条款自动提取")
print("   ③ 客服工单分类 → 自动打标签、分优先级")
print("   ④ 日志/报警解析 → 结构化监控数据")
print("   ⑤ 用户评论分析 → 提取情感、主题、评分")
print("=" * 60)
