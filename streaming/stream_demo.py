"""
╔══════════════════════════════════════════════════════════════════╗
║         项目八：Streaming 全链路流式输出                             ║
║         从原生 SDK 到 Chain/Agent/LangGraph 的完整流式方案         ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普一：为什么流式输出在生产环境中是必须的？】
═══════════════════════════════════════════════════════════════════

回顾项目零：我们用原生 SDK 实现了"打字机效果"（stream=True）。
但从项目一开始，所有代码都用的是 .invoke()（阻塞式调用）。

生产环境中的体感差异：

  阻塞式（.invoke）：
  ┌─────────────────────────────────────────────────────────────┐
  │  用户点击"发送" → 等3秒 → 等5秒 → 等8秒... → 突然出现一大段   │
  │  用户感受："卡死了？是不是挂了？要不要刷新？"                 │
  └─────────────────────────────────────────────────────────────┘

  流式（.stream / .astream_events）：
  ┌─────────────────────────────────────────────────────────────┐
  │  用户点击"发送" → 0.3秒后开始逐字显示 → 一个字一个字蹦出来    │
  │  用户感受："哇，它在思考呢！好快！"                           │
  └─────────────────────────────────────────────────────────────┘

  同样等待8秒，流式输出让用户感觉"快了10倍"！
  这就是为什么 ChatGPT、Claude 等产品都用流式输出。

═══════════════════════════════════════════════════════════════════
【前置科普二：LangChain 的四种流式 API】
═══════════════════════════════════════════════════════════════════

LangChain 提供了从简单到复杂的四种流式接口：

  ┌───────────────────┬────────────────────────────────────────┐
  │  API              │  适用场景                               │
  ├───────────────────┼────────────────────────────────────────┤
  │  .stream()        │  最简单，逐 chunk 输出文本              │
  │                   │  适合：简单 Chain 的流式输出             │
  ├───────────────────┼────────────────────────────────────────┤
  │  .astream()       │  .stream() 的异步版本                  │
  │                   │  适合：FastAPI 等异步框架               │
  ├───────────────────┼────────────────────────────────────────┤
  │  .astream_events()│  最强大！能看到链中每个组件的事件       │
  │                   │  适合：Agent/复杂链的全链路监控         │
  │                   │  能区分"LLM 在生成"还是"工具在执行"   │
  ├───────────────────┼────────────────────────────────────────┤
  │  .astream_log()   │  流式输出 JSON Patch 日志               │
  │                   │  适合：前端实时展示链的运行状态          │
  └───────────────────┴────────────────────────────────────────┘

  本项目重点演示：.stream()（同步）和 .astream_events()（异步全链路）

═══════════════════════════════════════════════════════════════════
【前置科普三：同步 vs 异步——什么时候用 async？】
═══════════════════════════════════════════════════════════════════

  同步（sync）：代码一行一行执行，前一行没完成后一行不能开始。
    for chunk in chain.stream(...):  ← 同步迭代器
        print(chunk)

  异步（async）：代码可以"暂停等待"，让出 CPU 给其他任务。
    async for event in chain.astream_events(...):  ← 异步迭代器
        print(event)

  什么时候必须用 async？
    ① 你的应用是 FastAPI/Starlette（天然异步框架）
    ② 你需要同时处理多个用户请求
    ③ 你要用 astream_events（只有异步版本！）

  本教学中：
    .stream() → 用普通 for 循环（简单直观）
    .astream_events() → 用 asyncio.run() 包装（因为我们是脚本环境）
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import asyncio

# LangChain 核心
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

# Agent
from langchain.agents import create_tool_calling_agent, AgentExecutor

# LangGraph
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化 LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 0 章：初始化 LLM")
print("=" * 60)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY, BASE_URL, MODEL_NAME
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.7,
    # ⚠️ 流式输出不需要额外设置 streaming=True！
    # LangChain 的 .stream() / .astream() 会自动开启流式。
    # streaming=True 只影响 .invoke() 时的回调行为，通常不需要手动设。
)

print("✅ LLM 初始化完成")
print(f"   模型: {MODEL_NAME}")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：Chain 的流式输出（.stream()）
# 目标：让 LCEL 链逐 token 输出，实现打字机效果
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：Chain 的流式输出（.stream）")
print("=" * 60)
print()

# ── .invoke() vs .stream() ──────────────────────────────
#
# .invoke(input)  → 返回完整结果（等 LLM 全部生成完）
# .stream(input)  → 返回一个迭代器，逐 chunk 产出碎片
#
# 对比项目零：
#   项目零用的是原生 SDK 的 stream=True
#   这里用的是 LangChain 的 .stream() 方法
#   效果一样，但 .stream() 能在整条 LCEL 链上工作！
#
# ⚠️ 避坑指南：.stream() 的输出格式
#   对于 prompt | llm | StrOutputParser 这条链：
#     每个 chunk 是一小段字符串（可能是一个字、几个字、一个标点）
#   对于 prompt | llm（没有 parser）：
#     每个 chunk 是一个 AIMessageChunk 对象

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位简洁的科普作家，用生动的比喻解释概念。回答控制在100字以内。",
        ),
        ("human", "{question}"),
    ]
)

parser = StrOutputParser()
chain = prompt | llm | parser

# ── 演示一：普通 .invoke()（阻塞，对比用）─────────────────

print("【对比：.invoke() 阻塞式调用】")
print("  （等待中...直到全部生成完才显示）")
result = chain.invoke({"question": "量子纠缠是什么？"})
print(f"  结果：{result}")
print()

# ── 演示二：.stream() 流式调用 ────────────────────────────

print("【.stream() 流式调用——逐 token 输出】")
print("  AI：", end="", flush=True)

chunk_count = 0
full_text = ""

for chunk in chain.stream({"question": "量子纠缠是什么？"}):
    # chunk 是一小段字符串（因为有 StrOutputParser）
    print(chunk, end="", flush=True)
    full_text += chunk
    chunk_count += 1

print()  # 换行
print()
print(f"  📊 统计：共收到 {chunk_count} 个 chunk，总长度 {len(full_text)} 字")
print()

# ── 演示三：不带 parser 的 .stream()（看原始 chunk 结构）────

print("【不带 StrOutputParser 的原始 chunk 结构】")
raw_chain = prompt | llm  # 没有 parser

print("  前 5 个 chunk 的类型和内容：")
for i, chunk in enumerate(raw_chain.stream({"question": "太阳为什么是圆的？"})):
    if i < 5:
        print(f"    [{i}] 类型: {type(chunk).__name__}, content: {chunk.content!r}")
    elif i == 5:
        print(f"    ... 后续 chunk 省略")
        # 消费完迭代器
for _ in raw_chain.stream({"question": ""}):
    pass  # 只是为了不留悬挂的迭代器

print()
print("  💡 观察：不带 parser 时，chunk 是 AIMessageChunk 对象，")
print("     带 parser 时，chunk 是纯字符串。parser 帮你做了 .content 提取。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：Agent 的流式输出
# 目标：让 Agent 在推理和工具调用过程中也能流式输出
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：Agent 的流式输出")
print("=" * 60)
print()

# ── Agent 流式的特殊性 ───────────────────────────────────
#
# 普通 Chain 的流式很简单：逐 token 输出文本。
# Agent 的流式更复杂，因为 Agent 有多个阶段：
#   ① LLM 推理（生成 Thought / 决定调用工具）
#   ② 工具执行（调用函数，等待结果）
#   ③ LLM 生成最终回答
#
# AgentExecutor.stream() 输出的是"步骤事件"：
#   每完成一个步骤（工具调用/最终回答），产出一个 dict。
#   不是逐 token 的！而是逐步骤的。
#
# 如果你要 Agent 最终回答也逐 token，需要用 astream_events()（第3章）。


@tool
def get_population(city: str) -> str:
    """查询城市的人口数据。

    Args:
        city: 城市名称（中文）
    """
    data = {
        "北京": "2189万（2023年）",
        "上海": "2487万（2023年）",
        "深圳": "1766万（2023年）",
    }
    return data.get(city, f"暂无 {city} 的人口数据")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式的结果。

    Args:
        expression: 数学表达式字符串，如 "2189 + 2487"
    """
    try:
        # 安全计算：只允许基本数学运算
        allowed = set("0123456789+-*/.(). ")
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        return "表达式包含不支持的字符"
    except Exception as e:
        return f"计算错误: {e}"


agent_tools = [get_population, calculate]

agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个数据分析助手，可以查询城市人口并进行计算。回答简洁。"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ("human", "{input}"),
    ]
)

agent = create_tool_calling_agent(llm, agent_tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=agent_tools,
    verbose=False,  # 关闭 verbose，我们自己打印流式事件
)

# ── Agent .stream() 演示 ─────────────────────────────────

print("【Agent .stream() —— 逐步骤输出】")
print()
print("  ❓ 问题：北京和上海的人口加起来是多少？")
print()
print("  📡 流式事件：")

for event in agent_executor.stream({"input": "北京和上海的人口加起来是多少？"}):
    # AgentExecutor.stream() 产出的事件类型：
    #   {"actions": [...], "messages": [...]}  → Agent 决定调用工具
    #   {"steps": [...], "messages": [...]}    → 工具执行完毕
    #   {"output": "...", "messages": [...]}   → 最终回答

    if "actions" in event:
        for action in event["actions"]:
            print(f"    🔧 调用工具: {action.tool}({action.tool_input})")
    elif "steps" in event:
        for step in event["steps"]:
            print(f"    📥 工具返回: {step.observation}")
    elif "output" in event:
        print(f"    ✅ 最终回答: {event['output']}")

print()
print('  💡 观察：Agent .stream() 输出的是"步骤级"事件，')
print("     不是逐 token 的。每个事件代表一个完整的步骤。")
print("     如果需要最终回答的逐 token 流式，请看第3章 astream_events。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：astream_events() — 全链路事件流（最强大）
# 目标：获取链/Agent 运行过程中每个组件的细粒度事件
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：astream_events() — 全链路事件流")
print("=" * 60)
print()

# ── astream_events 是什么？────────────────────────────────
#
# .stream() 只能看到"最外层"的输出。
# astream_events() 能看到链中"每一层"的事件：
#
#   事件类型            含义
#   on_chain_start     链开始执行
#   on_chain_end       链执行完毕
#   on_llm_start       LLM 开始生成
#   on_llm_new_token   LLM 生成了一个新 token（⭐ 逐字流式！）
#   on_llm_end         LLM 生成完毕
#   on_tool_start      工具开始执行
#   on_tool_end        工具执行完毕
#
# 这意味着：即使是 Agent 的最终回答，你也能逐 token 接收！
#
# ⚠️ 注意：astream_events 只有异步版本（需要 async for）
#   在脚本中用 asyncio.run() 包装即可。
#   在 FastAPI 中天然是异步环境，直接用即可。


async def demo_chain_astream_events():
    """演示 Chain 的 astream_events"""
    print("【Chain 的 astream_events — 逐 token + 事件类型】")
    print()
    print("  ❓ 问题：什么是黑洞？")
    print("  📡 事件流（只显示关键事件）：")
    print()

    token_count = 0
    full_answer = ""

    async for event in chain.astream_events(
        {"question": "什么是黑洞？用一句话回答。"},
        version="v2",  # 使用 v2 版本的事件格式（推荐）
    ):
        kind = event["event"]

        if kind == "on_chain_start":
            # 链开始
            if event["name"] == "RunnableSequence":
                print("    ⏳ [on_chain_start] 链开始执行")

        elif kind == "on_chat_model_stream":
            # LLM 生成新 token（⭐ 逐字流式的核心！）
            content = event["data"]["chunk"].content
            if content:
                if token_count == 0:
                    print("    💬 [on_chat_model_stream] LLM 开始生成：", end="")
                print(content, end="", flush=True)
                full_answer += content
                token_count += 1

        elif kind == "on_chain_end":
            if event["name"] == "RunnableSequence":
                if token_count > 0:
                    print()  # 换行
                print(f"    ✅ [on_chain_end] 链执行完毕")

    print()
    print(f"  📊 统计：LLM 共产出 {token_count} 个 token")
    print(f"  📝 完整回答：{full_answer}")
    print()


async def demo_agent_astream_events():
    """演示 Agent 的 astream_events — 看到工具调用 + 最终回答的逐 token"""
    print("【Agent 的 astream_events — 全链路细粒度事件】")
    print()
    print("  ❓ 问题：深圳的人口是多少？")
    print("  📡 全链路事件流：")
    print()

    final_answer = ""

    async for event in agent_executor.astream_events(
        {"input": "深圳的人口是多少？"},
        version="v2",
    ):
        kind = event["event"]

        # 工具调用开始
        if kind == "on_tool_start":
            tool_name = event["name"]
            tool_input = event["data"].get("input", "")
            print(f"    🔧 [on_tool_start] 调用工具: {tool_name}")
            print(f"       参数: {tool_input}")

        # 工具调用结束
        elif kind == "on_tool_end":
            tool_output = event["data"].get("output", "")
            print(f"    📥 [on_tool_end] 工具返回: {tool_output}")
            print()

        # LLM 生成 token（最终回答的逐字流式！）
        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content and not event["data"]["chunk"].tool_calls:
                # 只打印非工具调用的文本（即最终回答）
                if not final_answer:
                    print("    💬 [最终回答 - 逐 token] ", end="")
                print(content, end="", flush=True)
                final_answer += content

    if final_answer:
        print()  # 换行
    print()
    print(f"  📝 完整回答：{final_answer}")
    print()
    print("  💡 核心价值：astream_events 让你能同时看到：")
    print("     ① 工具何时被调用、传了什么参数、返回了什么")
    print("     ② 最终回答的每一个 token（真正的逐字流式！）")
    print("     在前端开发中，你可以据此实现：")
    print("     • 工具调用时显示 loading 动画")
    print("     • 最终回答时显示打字机效果")
    print()


# 运行异步演示
asyncio.run(demo_chain_astream_events())
asyncio.run(demo_agent_astream_events())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：LangGraph 的流式输出
# 目标：让多智能体图也能流式输出每个节点的执行过程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 4 章：LangGraph 的流式输出")
print("=" * 60)
print()

# ── LangGraph 的 .stream() 模式 ──────────────────────────
#
# LangGraph 编译后的图支持 .stream()，有两种模式：
#
#   stream_mode="values"（默认）：
#     每次一个节点执行完，产出当前完整的 State。
#     适合：你想看到每个节点执行后 State 的全貌。
#
#   stream_mode="updates"：
#     每次一个节点执行完，只产出该节点更新的字段。
#     适合：你只关心"哪个节点改了什么"。
#
# 两者的区别用比喻理解：
#   "values" = 每次拍一张全景照片（看到所有人的状态）
#   "updates" = 每次只拍变化的人（谁动了拍谁）


class MiniState(TypedDict):
    """简化的状态，用于演示 LangGraph 流式"""

    topic: str
    summary: str
    expanded: str


def summarize_node(state: MiniState) -> dict:
    """摘要节点：用 LLM 生成主题的一句话摘要"""
    result = chain.invoke({"question": f"用一句话概括：{state['topic']}"})
    return {"summary": result}


def expand_node(state: MiniState) -> dict:
    """扩展节点：基于摘要进行详细展开"""
    result = chain.invoke({"question": f"请详细解释：{state['summary']}"})
    return {"expanded": result}


# 构建简单图
graph = StateGraph(MiniState)
graph.add_node("summarize", summarize_node)
graph.add_node("expand", expand_node)
graph.add_edge(START, "summarize")
graph.add_edge("summarize", "expand")
graph.add_edge("expand", END)
app = graph.compile()

# ── 演示：stream_mode="values" ────────────────────────────

print('【LangGraph .stream(stream_mode="values") — 每步输出完整 State】')
print()
print("  ❓ 主题：人工智能")
print("  📡 每个节点执行后的完整 State：")
print()

for i, state_snapshot in enumerate(
    app.stream(
        {"topic": "人工智能", "summary": "", "expanded": ""},
        stream_mode="values",
    )
):
    print(f"  ┌── 快照 #{i} ──────────────────────────────────────")
    print(f"  │ topic:    {state_snapshot.get('topic', '')!r}")
    summary = state_snapshot.get("summary", "")
    print(f"  │ summary:  {summary[:50] + '...' if len(summary) > 50 else summary!r}")
    expanded = state_snapshot.get("expanded", "")
    print(
        f"  │ expanded: {expanded[:50] + '...' if len(expanded) > 50 else expanded!r}"
    )
    print(f"  └──────────────────────────────────────────────────")
    print()

# ── 演示：stream_mode="updates" ───────────────────────────

print('【LangGraph .stream(stream_mode="updates") — 只看每个节点的增量更新】')
print()
print("  ❓ 主题：区块链")
print("  📡 每个节点更新的字段：")
print()

for update in app.stream(
    {"topic": "区块链", "summary": "", "expanded": ""},
    stream_mode="updates",
):
    # update 格式：{node_name: {updated_fields}}
    for node_name, node_output in update.items():
        print(f"  🔄 节点 [{node_name}] 更新了：")
        for key, value in node_output.items():
            display = value[:60] + "..." if len(str(value)) > 60 else value
            print(f"     • {key} = {display!r}")
    print()

print("  💡 stream_mode='updates' 更轻量，适合前端只关心变化的场景。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 章：生产实践——SSE 格式输出（模拟 FastAPI 场景）
# 目标：演示如何把流式输出格式化为 Server-Sent Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 5 章：生产实践——SSE 格式输出")
print("=" * 60)
print()

# ── SSE（Server-Sent Events）是什么？─────────────────────
#
# SSE 是浏览器原生支持的"服务器推送"协议：
#   服务器不断向客户端发送事件，客户端实时接收。
#   比 WebSocket 简单得多，单向推送就够了。
#
# SSE 的格式规范：
#   每个事件由 "data: ..." 开头，以两个换行结束：
#     data: {"token": "你"}
#     \n
#     data: {"token": "好"}
#     \n
#     data: [DONE]
#     \n
#
# 在 FastAPI 中的用法：
#   from fastapi.responses import StreamingResponse
#   return StreamingResponse(generate_sse(), media_type="text/event-stream")
#
# 本章模拟这个过程，展示如何把 .stream() 的输出转换为 SSE 格式。

print("【模拟 SSE 输出格式（FastAPI 场景）】")
print()
print("  前端会收到这样的事件流：")
print("  " + "-" * 50)

import json as json_module

for chunk in chain.stream({"question": "什么是深度学习？一句话。"}):
    if chunk:
        # 格式化为 SSE event
        sse_data = json_module.dumps({"token": chunk}, ensure_ascii=False)
        print(f"  data: {sse_data}")

# 发送结束信号
print("  data: [DONE]")
print("  " + "-" * 50)
print()
print("  💡 前端 JS 代码（EventSource）：")
print("     const es = new EventSource('/api/chat/stream');")
print("     es.onmessage = (e) => {")
print("       if (e.data === '[DONE]') { es.close(); return; }")
print("       const {token} = JSON.parse(e.data);")
print("       document.getElementById('output').textContent += token;")
print("     };")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 总结
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("🎉 项目八学习完毕！")
print("=" * 60)
print()
print("💡 流式 API 选择指南：")
print()
print("  ┌────────────────────┬────────────────────────────────────┐")
print("  │  场景               │  推荐 API                         │")
print("  ├────────────────────┼────────────────────────────────────┤")
print("  │  简单 Chain 打字机  │  .stream()（同步，最简单）         │")
print("  │  FastAPI 异步服务   │  .astream()（异步版 stream）       │")
print("  │  Agent 全链路监控   │  .astream_events()（细粒度事件）   │")
print("  │  LangGraph 节点追踪 │  .stream(stream_mode='updates')   │")
print("  │  前端 SSE 推送      │  .stream() + SSE 格式化            │")
print("  └────────────────────┴────────────────────────────────────┘")
print()
print("💡 关键记忆点：")
print("   ① .stream() 是同步的，用 for 循环")
print("   ② .astream_events() 是异步的，用 async for")
print("   ③ Agent 的 .stream() 是步骤级的（不是逐 token）")
print("   ④ 要 Agent 逐 token → 必须用 astream_events")
print("   ⑤ LangGraph 用 stream_mode 控制输出粒度")
print("   ⑥ 生产环境用 SSE 协议推送到前端")
print("=" * 60)
