"""
╔══════════════════════════════════════════════════════════════════╗
║         项目一：LangChain 核心组件打样（地基）                       ║
║         带历史记忆的控制台聊天机器人                                  ║
╚══════════════════════════════════════════════════════════════════╝

【学前必读：LangChain 是什么？】

用一个比喻来理解：

  假设你要开一家餐厅（LLM 应用）：
  - 食材（原始数据/用户输入）需要加工
  - 厨师（LLM）需要按食谱（Prompt）来做菜
  - 最后摆盘（OutputParser）才能端给客人

  LangChain 就是这家餐厅的"标准化厨房流程系统"，
  它帮你把：食材处理 → 厨师烹饪 → 摆盘上菜，
  这一整条"流水线"组装起来，让你不用每次都从头搭建。

【LCEL 语法（管道 | ）是什么？】

  LCEL = LangChain Expression Language（LangChain 表达式语言）

  你用过 Linux 的命令行管道吗？
    cat file.txt | grep "关键词" | sort

  LCEL 的 | 符号和这个完全一样的思想：
    prompt | llm | output_parser

  意思是："把 prompt 的输出，喂给 llm；把 llm 的输出，喂给 output_parser"
  每个组件只做一件事，组合起来就是完整的处理链。

  这种设计的好处：
  ① 可读性极强，一眼看出数据流向
  ② 任何一个组件都可以单独替换（换个 LLM 只需改一行）
  ③ 支持流式输出、并行处理等高级特性
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】把需要的"工具"从工具箱里拿出来
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# os 模块：用来读取操作系统的环境变量（存放 API Key 的推荐方式）
import os

# ChatOpenAI：LangChain 封装好的"OpenAI 兼容接口"客户端
# 它不只能连 OpenAI，只要接口兼容 OpenAI 格式，都能用（比如我们的 DeepSeek）
from langchain_openai import ChatOpenAI

# ChatPromptTemplate：聊天场景专用的提示词模板
# 它理解"system/human/ai"这三种角色，而不是简单的字符串拼接
from langchain_core.prompts import ChatPromptTemplate

# MessagesPlaceholder：一个特殊的"占位符"，专门用来在模板中留一个位置给历史消息
# 想象成在提示词里打了一个"洞"，运行时把历史消息塞进去
from langchain_core.prompts import MessagesPlaceholder

# StrOutputParser：输出解析器，把 LLM 返回的 AIMessage 对象解析成普通字符串
# 因为 LLM 直接返回的是一个"消息对象"，不是纯文本
from langchain_core.output_parsers import StrOutputParser

# RunnableWithMessageHistory：LangChain 0.2+ 官方推荐的历史记忆解决方案
# ⚠️ 避坑指南：不要用老版本的 ConversationBufferMemory，它已被官方标为废弃！
# 新版本的思路是：把记忆管理从 Chain 中分离出来，作为独立的"包装层"
from langchain_core.runnables.history import RunnableWithMessageHistory

# ChatMessageHistory：内存中的聊天历史存储
# 它就是一个简单的列表，存着 [HumanMessage, AIMessage, HumanMessage, AIMessage...]
from langchain_community.chat_message_histories import ChatMessageHistory

# BaseMessage：消息基类，用于类型标注
from langchain_core.chat_history import BaseChatMessageHistory


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：环境初始化
# 目标：配置好 LLM 连接，让代码能和 AI 对话
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 0 章：初始化 LLM 连接")
print("=" * 60)

# ⚠️ 避坑指南：API Key 的正确处理方式
#
# 方式一（教学用，不推荐上生产）：直接写在代码里
#   api_key = "ak-infer-xxxxx"
#
# 方式二（推荐）：放在环境变量里
#   终端执行：export LIXIANG_API_KEY="ak-infer-xxxxx"
#   代码里读：api_key = os.environ.get("LIXIANG_API_KEY")
#
# 方式三（团队协作推荐）：放在 .env 文件里，用 python-dotenv 加载
#   pip install python-dotenv
#   from dotenv import load_dotenv; load_dotenv()
#
# 这里为了教学方便，直接写死，生产环境请务必用方式二或三！

# 你的 API Key（从 agent.py 复用）
API_KEY = "ak-infer-ZGV2ZWxvcC1kYXRhLWFuYWx5c2lzOnB0ZXB5cjpDMDAwMjEzOmppbnl1YW5odWlAbGl4aWFuZy5jb206aW5mZXI_ZjdkYWEwZmEtYzg4ZS00MmUxLWJmMzYtMjRiNWE3MzZiZGVi"

# 理想内部 LLM 接口地址（OpenAI 兼容格式）
BASE_URL = "https://lpai-llm.lixiang.com/inference/deepseek-ai/deepseek-v3/v1/"

# 使用的模型名称
MODEL_NAME = "deepseek-v3-0324"

# 初始化 ChatOpenAI 客户端
# 关键点：LangChain 的 ChatOpenAI 类通过 base_url 参数支持任何 OpenAI 兼容接口
# temperature=0.7 控制"创造力"，0=死板精确，1=天马行空，0.7是个平衡点
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.7,
)

print(f"✅ LLM 初始化完成")
print(f"   模型: {MODEL_NAME}")
print(f"   接口: {BASE_URL}")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：PromptTemplate（提示词模板）
# 目标：理解为什么要用模板，而不是直接拼字符串
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：PromptTemplate 演示")
print("=" * 60)

# 为什么要用 PromptTemplate，而不是直接 f-string 拼字符串？
#
# 想象你是一个翻译公司，每天要翻译不同语言的文件：
#   "请把以下{source_lang}文本翻译成{target_lang}：\n{text}"
#
# 用 f-string 的问题：
#   ① 变量和模板混在一起，难以维护
#   ② 无法复用（每次都要重写）
#   ③ 无法做类型检查（变量名写错了不会报错）
#
# PromptTemplate 解决了这些问题，并且还能：
#   ① 序列化（保存到文件、数据库）
#   ② 版本管理
#   ③ 与 LCEL 管道无缝集成

# 创建一个聊天提示词模板
# from_messages 方法接受一个消息列表，每个消息是 (角色, 内容) 的元组
# 角色有三种：
#   "system"  - 系统指令，设定 AI 的"人设"（用户看不到，但 AI 会遵守）
#   "human"   - 用户说的话
#   "ai"      - AI 说的话（用于少样本示例，few-shot）
demo_prompt = ChatPromptTemplate.from_messages([
    # system 消息：设定 AI 的角色和规则
    # {topic} 是一个变量，花括号括起来的都是变量
    ("system", "你是一位专业的{topic}领域专家，请用通俗易懂的语言回答问题。"),
    # human 消息：用户的提问
    # {question} 也是变量
    ("human", "{question}"),
])

# 打印模板的"原始样子"——让你看清楚模板的内部结构
print("【模板原始结构】")
print(demo_prompt)
print()

# 用 .invoke() 方法填充变量，生成真正的提示词
# 这一步叫"模板渲染"，就像把合同模板填写成具体合同
filled_prompt = demo_prompt.invoke({
    "topic": "Python 编程",
    "question": "什么是列表推导式？"
})

# 打印填充后的提示词，让你看到 LLM 实际收到的内容
print("【填充变量后的提示词（LLM 将收到这个）】")
for message in filled_prompt.messages:
    # message.type 是消息类型（system/human/ai）
    # message.content 是消息内容
    print(f"  [{message.type.upper()}] {message.content}")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：LCEL 管道（ | 语法）
# 目标：理解"链式调用"的核心思想
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：LCEL 管道演示")
print("=" * 60)

# 创建输出解析器实例
# StrOutputParser 的作用：把 LLM 返回的 AIMessage 对象 → 纯字符串
# 为什么需要它？
#   LLM 返回的是：AIMessage(content="你好", response_metadata={...}, id="...")
#   我们想要的是：纯字符串 "你好"
parser = StrOutputParser()

# ⚠️ 避坑指南：LCEL 管道的 | 运算符
#
# 这里的 | 不是"或"运算符（Python 原生的位运算）！
# LangChain 重载了 | 运算符，让它变成了"管道"语义。
# 每个 LangChain 组件都继承自 Runnable 接口，Runnable 实现了 __or__ 方法。
#
# 数据流向：
#   demo_prompt  →  llm  →  parser
#      ↓              ↓        ↓
#   生成消息列表   调用AI   提取纯文本

# 用 | 把三个组件串联成一条"链"
# 这一行就是 LCEL 的精华！
simple_chain = demo_prompt | llm | parser

print("【LCEL 链已创建】")
print(f"  链的结构：demo_prompt | llm | parser")
print(f"  链的类型：{type(simple_chain).__name__}")
print()

# 调用链！只需要提供模板中需要的变量
print("【正在调用链，请稍候...】")
result = simple_chain.invoke({
    "topic": "Python 编程",
    "question": "用一句话解释什么是列表推导式？"
})

print(f"【链的最终输出（字符串）】")
print(f"  {result}")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：OutputParser 的作用对比
# 目标：亲眼看到"解析前"和"解析后"的区别
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：OutputParser 对比演示")
print("=" * 60)

# 创建一条"不带 parser"的链，直接拿 LLM 的原始输出
chain_without_parser = demo_prompt | llm

print("【不带 OutputParser 的原始输出】")
raw_output = chain_without_parser.invoke({
    "topic": "Python 编程",
    "question": "Python 中的 None 是什么？请用10字以内回答。"
})
# 打印原始 AIMessage 对象的类型和全部信息
print(f"  类型: {type(raw_output)}")
print(f"  内容: {raw_output}")
print()

# 创建一条"带 parser"的链
chain_with_parser = demo_prompt | llm | parser

print("【带 StrOutputParser 的解析后输出】")
parsed_output = chain_with_parser.invoke({
    "topic": "Python 编程",
    "question": "Python 中的 None 是什么？请用10字以内回答。"
})
# 现在拿到的是干净的字符串
print(f"  类型: {type(parsed_output)}")
print(f"  内容: {parsed_output}")
print()
print("💡 结论：StrOutputParser 把 AIMessage 对象剥开，只留下 .content 文本")
print()
