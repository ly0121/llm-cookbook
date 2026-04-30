"""
╔══════════════════════════════════════════════════════════════════╗
║         项目三：基于 ReAct 框架的单体 Agent（全能助手）              ║
║         工具定义 + Agent 构建 + ReAct 思维链可视化                  ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普一：老板、员工、工具箱——Agent 究竟是什么？】
═══════════════════════════════════════════════════════════════════

先来对比两种场景：

  场景一：普通 LLM 对话（只能用脑子的员工）
  ┌─────────────────────────────────────────────────────────┐
  │  老板：北京今天天气怎么样？                                │
  │  员工：抱歉，我不知道，我的知识有截止日期，               │
  │        而且我没有办法上网查询实时信息。                    │
  └─────────────────────────────────────────────────────────┘
  问题：LLM 的知识是静态的，它无法主动获取外部信息。

  场景二：Agent（配备了工具箱的员工）
  ┌─────────────────────────────────────────────────────────┐
  │  老板：北京今天天气怎么样？                                │
  │  员工（心里想）：老板问天气，我得用天气查询工具。           │
  │  员工（拿起电话）：[调用 get_weather("北京")]              │
  │  工具返回：晴，28°C                                       │
  │  员工：北京今天天气晴，气温 28 摄氏度。                    │
  └─────────────────────────────────────────────────────────┘
  关键区别：Agent 可以主动调用外部函数（工具），LLM 只是决策大脑。

  ┌──────────────────────────────────────────────────────────┐
  │  工具箱（Tools）= 一组可以被 LLM 主动调用的 Python 函数   │
  │  每个工具都有：                                           │
  │    name        → 工具名称（LLM 用这个名字调用工具）        │
  │    description → 工具说明（LLM 靠这个决定"要不要用它"）   │
  │    function    → 实际执行的 Python 函数                   │
  └──────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
【前置科普二：ReAct 框架——大模型是怎么"思考"的？】
═══════════════════════════════════════════════════════════════════

ReAct = Reasoning（推理）+ Acting（行动）

每次 Agent 处理一个问题，内部会进行若干轮"思考-行动-观察"循环：

  ┌─────────────────────────────────────────────────────────┐
  │  Thought（思考）                                         │
  │    LLM 的内心独白："我现在知道了什么，下一步该做什么？"   │
  │    这一步是纯文本推理，不调用任何工具。                   │
  ├─────────────────────────────────────────────────────────┤
  │  Action（行动）                                          │
  │    LLM 决定：调用哪个工具？传什么参数？                   │
  │    例如：Action: get_weather                             │
  │          Action Input: 北京                              │
  ├─────────────────────────────────────────────────────────┤
  │  Observation（观察）                                     │
  │    工具函数被执行，返回结果喂给 LLM。                     │
  │    例如：Observation: 晴，28°C                           │
  └─────────────────────────────────────────────────────────┘

  循环直到 LLM 认为信息足够，输出：
    Final Answer: [最终回答给用户的内容]

  verbose=True 时，AgentExecutor 会把这整个过程打印到控制台，
  让你亲眼看到 LLM 的每一步思考！
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# LangChain Agent 核心组件
from langchain.agents import create_react_agent, AgentExecutor

# @tool 装饰器：把普通 Python 函数变成 Agent 可以调用的工具
from langchain_core.tools import tool

# Prompt 模板：ReAct Agent 需要特定格式的 PromptTemplate
from langchain_core.prompts import PromptTemplate

# 聊天模型（和前几个项目完全一样）
from langchain_openai import ChatOpenAI


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化 LLM
# 目标：建立与大模型的连接（和项目零/一/二完全一致）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 0 章：初始化 LLM")
print("=" * 60)

# 教学用硬编码；生产环境请改用环境变量：os.environ["OPENAI_API_KEY"]
API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "kivy-kimi-k2_5"

# Agent 场景建议 temperature=0：需要精确推理和格式遵循，不要创意发散
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
# 第 1 章：用 @tool 装饰器定义工具
# 目标：把普通 Python 函数变成 Agent 可以调用的工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：工具定义（@tool 装饰器）")
print("=" * 60)
print()

# ── Mock 天气数据 ─────────────────────────────────────────
#
# 真实场景：调用 OpenWeatherMap 等天气 API
# 教学场景：用固定字典模拟，省去 API Key 和网络请求

WEATHER_DATA = {
    "北京": {"weather": "晴", "temperature": 28},
    "上海": {"weather": "多云", "temperature": 24},
    "广州": {"weather": "阵雨", "temperature": 32},
    "成都": {"weather": "阴", "temperature": 20},
}


# ── 工具一：天气查询 ──────────────────────────────────────
#
# @tool 装饰器做了三件事：
#   ① 把函数名 → tool.name（Agent 用这个名字调用工具）
#   ② 把函数 docstring → tool.description（⭐ 最重要！）
#   ③ 把函数参数类型注解 → tool.args_schema（LLM 靠这个知道怎么传参数）
#
# ⚠️ 避坑指南：description 写不清楚后果很严重！
#   description 是 LLM 判断"要不要用这个工具"的唯一依据。
#   如果写成"查天气"——太模糊，LLM 可能不知道什么时候该调用它。
#   应该写清楚：① 这个工具做什么 ② 输入格式 ③ 返回什么
#
# ⚠️ 避坑指南：参数必须有类型注解！
#   @tool 依赖类型注解（city: str）生成 args_schema，
#   缺少类型注解时 LLM 不知道应该传什么类型的参数。

@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气信息，返回天气状况和温度（摄氏度）。
    输入城市名（中文），例如"北京"、"上海"、"广州"、"成都"。
    当用户询问某个城市的天气、温度、气候时使用此工具。"""
    data = WEATHER_DATA.get(city)
    if data:
        return f"{city}：{data['weather']}，温度 {data['temperature']}°C"
    return f"暂无 {city} 的天气数据，目前支持：{'、'.join(WEATHER_DATA.keys())}"


# ── 工具二：幂次计算 ──────────────────────────────────────
#
# 这个工具做真实计算（不是 mock），结果可以验证。
# 演示"工具可以是任何 Python 函数"——查询、计算、文件操作……

@tool
def calculate_power(base: int, exponent: int) -> str:
    """计算一个整数的幂次方（base 的 exponent 次方）。
    例如：base=2, exponent=10 → 返回 "2 的 10 次方 = 1024"。
    当用户需要精确的数学幂次计算时使用此工具。
    注意：只接受整数输入。"""
    result = base ** exponent
    return f"{base} 的 {exponent} 次方 = {result}"


# ── 打印工具元数据，让你看清楚 @tool 做了什么 ────────────────

tools = [get_weather, calculate_power]

print("【@tool 装饰器为每个工具生成的元数据】")
print()
for t in tools:
    print(f"  工具名称 (name):        {t.name}")
    print(f"  工具描述 (description): {t.description[:60]}...")
    print(f"  参数结构 (args_schema): {t.args_schema.schema()['properties']}")
    print()

print("💡 LLM 接收到的工具信息就是上面这些。")
print("   它靠 description 决定用哪个工具，靠 args_schema 知道怎么传参数。")
print()
