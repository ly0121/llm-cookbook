"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  项目 18: LangGraph 高级篇 — Human-in-the-Loop + Checkpointing            ║
║                                                                              ║
║  "AI 不是要取代人类, 而是要和人类协作"                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
前置科普: Human-in-the-Loop (HITL) 与 Checkpointing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎬 电影比喻: 想象你是一部电影的导演 (Human), AI 是演员。

  普通模式 (无 HITL):
    导演喊"开始" → 演员自己演完整场戏 → 导演只能看结果

  HITL 模式:
    导演喊"开始" → 演员演到关键情节 → 暂停! → 导演审核 →
    导演说"继续/重来" → 演员继续表演

  Checkpointing (存档点):
    就像游戏里的"存档"功能!
    - 每走一步, 游戏自动存档
    - 如果角色死了, 可以从最近的存档点继续
    - 不需要从头开始!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心概念图:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────────────────────────────────────────────┐
  │                    LangGraph 执行流程                            │
  │                                                                 │
  │   START ──→ [Node A] ──→ [Node B] ──→ [Node C] ──→ END        │
  │                              │                                  │
  │                              │  ← interrupt_before              │
  │                              ▼                                  │
  │                        ┌──────────┐                            │
  │                        │  暂停!   │                            │
  │                        │  等待人类 │                            │
  │                        │  决策...  │                            │
  │                        └──────────┘                            │
  │                              │                                  │
  │                    人类: "同意" / "拒绝"                        │
  │                              │                                  │
  │                              ▼                                  │
  │                     继续执行 / 终止                             │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │                    Checkpointing 机制                           │
  │                                                                 │
  │   执行 Node A → 💾 存档1                                       │
  │   执行 Node B → 💾 存档2  ← (中断在这里)                      │
  │   ...程序关闭...                                                │
  │   ...程序重启...                                                │
  │   从 存档2 恢复 → 继续执行 Node C → END                       │
  │                                                                 │
  │   关键: thread_id 是存档的"文件名"                             │
  │         同一个 thread_id = 同一个对话/流程                     │
  └─────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
技术要点:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. MemorySaver: 内存中的 checkpoint 存储 (适合开发/演示)
  2. interrupt_before: 在指定节点执行"之前"暂停
  3. thread_id: 标识一个独立的执行线程 (类似 session_id)
  4. graph.get_state(): 查看当前暂停状态
  5. graph.update_state(): 注入人类决策, 修改状态
  6. 再次 invoke: 从暂停处继续执行

依赖: pip install langchain-openai langgraph langchain-core
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 0: 初始化 — LLM 配置 + HITL 概念详解
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print(" Chapter 0: 初始化 — LLM 配置 + HITL 概念详解")
print("=" * 70)

# ─── API 配置 ───────────────────────────────────────────────────────────────
# 与之前项目一致, 使用统一的 LLM Gateway
API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "kivy-kimi-k2_5"

# ─── 创建 LLM 实例 ─────────────────────────────────────────────────────────
# temperature=0.7: 给 AI 一些创造力, 适合写作场景
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    temperature=0.7,
)

print("\n[✓] LLM 初始化完成")
print(f"    模型: {MODEL_NAME}")
print(f"    网关: {BASE_URL}")

# ─── HITL 概念解释 ─────────────────────────────────────────────────────────
print("\n" + "-" * 50)
print(" Human-in-the-Loop 核心思想:")
print("-" * 50)
print("""
  为什么需要 HITL?
  ─────────────────
  场景1: AI 要发送一封重要邮件 → 发送前让人类确认
  场景2: AI 生成了一段代码要部署 → 部署前让人类审核
  场景3: AI 要花真金白银调用付费API → 扣费前让人类批准
  场景4: AI 写了一篇文章要发布 → 发布前让人类审阅

  共同点: AI 在执行"不可逆"或"高风险"操作前, 暂停等待人类确认

  LangGraph 的实现方式:
  ─────────────────────
  1. 定义图的节点 (Nodes) 和边 (Edges)
  2. 编译时指定 interrupt_before=["需要暂停的节点名"]
  3. 运行时, 到达该节点前会自动暂停
  4. 开发者通过 get_state() 查看当前状态
  5. 开发者通过 update_state() 注入人类决策
  6. 再次 invoke() 从暂停处继续
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 1: LangGraph 基础回顾 — 简单图的构建与执行
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print(" Chapter 1: LangGraph 基础回顾 — 简单图的构建与执行")
print("=" * 70)

print("""
  本章目标: 先用一个简单的 3 节点图, 回顾 LangGraph 的基本用法

  图结构:
  ┌───────┐     ┌───────────┐     ┌───────────┐
  │ START │────→│ 收集信息  │────→│  分析处理 │────→ END
  └───────┘     └───────────┘     └───────────┘
                  (node_1)           (node_2)
""")

# ─── 第一步: 定义 State (状态) ──────────────────────────────────────────────
# State 就是在图中流动的"数据包"
# TypedDict 让我们明确定义状态包含哪些字段
# Annotated[list, operator.add] 表示: 这个字段是 list 类型,
# 当多个节点都往里写数据时, 用 operator.add (拼接) 合并


class BasicState(TypedDict):
    """基础状态定义 — 图中流动的数据"""

    # messages: 消息列表, 使用 operator.add 做 reducer (追加合并)
    messages: Annotated[list, operator.add]
    # current_step: 当前步骤标记, 方便观察流程
    current_step: str


print("[State 定义完成]")
print("  - messages: list (追加模式, 用 operator.add 合并)")
print("  - current_step: str (覆盖模式, 只保留最新值)")


# ─── 第二步: 定义节点函数 ──────────────────────────────────────────────────
# 每个节点是一个函数, 接收 state, 返回要更新的字段
def collect_info(state: BasicState) -> dict:
    """节点1: 收集信息 — 模拟收集用户需求"""
    print("\n  [Node: collect_info] 正在执行...")
    print(f"    输入状态 messages 长度: {len(state['messages'])}")

    # 返回值会和当前 state 合并:
    # - messages 用 operator.add → 追加到列表末尾
    # - current_step 直接覆盖
    result = {"messages": ["[收集] 用户需求已收集完毕"], "current_step": "collected"}
    print(f"    输出: {result}")
    return result


def analyze_process(state: BasicState) -> dict:
    """节点2: 分析处理 — 模拟对信息进行分析"""
    print("\n  [Node: analyze_process] 正在执行...")
    print(f"    输入状态 messages: {state['messages']}")
    print(f"    输入状态 current_step: {state['current_step']}")

    result = {"messages": ["[分析] 分析完成, 生成结论"], "current_step": "analyzed"}
    print(f"    输出: {result}")
    return result


# ─── 第三步: 构建图 ────────────────────────────────────────────────────────
print("\n[构建图...]")

# StateGraph 需要传入 State 类型, 它会根据类型注解来决定如何合并状态
graph_builder = StateGraph(BasicState)

# 添加节点: add_node("节点名", 节点函数)
graph_builder.add_node("collect_info", collect_info)
graph_builder.add_node("analyze_process", analyze_process)

# 添加边: add_edge(起点, 终点)
# START 是特殊常量, 表示图的入口
graph_builder.add_edge(START, "collect_info")
graph_builder.add_edge("collect_info", "analyze_process")
graph_builder.add_edge("analyze_process", END)

# 编译图 — 没有 checkpointer, 最简单的模式
simple_graph = graph_builder.compile()

print("[✓] 图构建完成! 节点: collect_info → analyze_process → END")

# ─── 第四步: 执行图 ────────────────────────────────────────────────────────
print("\n[执行图...]")
print("─" * 50)

# invoke() 传入初始状态, 图会从 START 开始执行
initial_state = {"messages": ["[开始] 用户发起请求"], "current_step": "init"}
print(f"  初始状态: {initial_state}")

result = simple_graph.invoke(initial_state)

print("─" * 50)
print("\n[✓] 图执行完成!")
print(f"  最终状态:")
print(f"    messages: {result['messages']}")
print(f"    current_step: {result['current_step']}")

print("""
  观察要点:
  ─────────
  1. messages 从 1 条 → 3 条 (每个节点追加了一条)
  2. current_step 从 "init" → "collected" → "analyzed" (每次覆盖)
  3. 节点按顺序执行: collect_info → analyze_process
  4. 状态在节点间流动, 每个节点都能看到前面的累积结果
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 2: Checkpointing — 给图加上"存档功能"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print(' Chapter 2: Checkpointing — 给图加上"存档功能"')
print("=" * 70)

print("""
  Checkpointing 就是"自动存档":
  ─────────────────────────────

  ┌────────┐   💾    ┌────────┐   💾    ┌────────┐   💾
  │ Node A │──存档──→│ Node B │──存档──→│ Node C │──存档──→ END
  └────────┘         └────────┘         └────────┘

  每执行完一个节点, 状态就会被保存到 checkpointer 中。

  MemorySaver: 最简单的 checkpointer, 存在内存里。
  - 优点: 零配置, 开箱即用, 速度快
  - 缺点: 程序关闭后数据丢失
  - 适用: 开发测试、演示、短生命周期应用

  生产环境可以用: SqliteSaver, PostgresSaver 等持久化方案。

  thread_id 的作用:
  ─────────────────
  一个 checkpointer 可以保存多个"对话"的状态,
  通过 thread_id 来区分不同的对话/流程。
  就像游戏有多个存档槽位: 存档1, 存档2, 存档3...
""")

# ─── 第一步: 创建 MemorySaver ──────────────────────────────────────────────
print("[创建 MemorySaver...]")
memory = MemorySaver()
print("[✓] MemorySaver 创建完成 (内存中的 checkpoint 存储)")

# ─── 第二步: 重新构建图, 但这次加上 checkpointer ──────────────────────────
print("\n[重新构建图, 加入 checkpointer...]")


class ChatState(TypedDict):
    """带消息历史的状态"""

    messages: Annotated[list, operator.add]
    step_count: int


def step_one(state: ChatState) -> dict:
    """步骤一: 打招呼"""
    print("\n  [Node: step_one] 执行中...")
    count = state.get("step_count", 0) + 1
    return {"messages": [f"[Step {count}] 你好! 我是步骤一的输出"], "step_count": count}


def step_two(state: ChatState) -> dict:
    """步骤二: 处理"""
    print("\n  [Node: step_two] 执行中...")
    count = state.get("step_count", 0) + 1
    return {"messages": [f"[Step {count}] 我是步骤二, 处理完毕!"], "step_count": count}


# 构建图
ckpt_builder = StateGraph(ChatState)
ckpt_builder.add_node("step_one", step_one)
ckpt_builder.add_node("step_two", step_two)
ckpt_builder.add_edge(START, "step_one")
ckpt_builder.add_edge("step_one", "step_two")
ckpt_builder.add_edge("step_two", END)

# 编译时传入 checkpointer=memory — 这就是唯一的区别!
ckpt_graph = ckpt_builder.compile(checkpointer=memory)

print("[✓] 带 checkpoint 的图构建完成!")

# ─── 第三步: 第一次执行 (thread_id = "demo-thread-1") ─────────────────────
print("\n" + "-" * 50)
print(' 第一次执行 (thread_id = "demo-thread-1")')
print("-" * 50)

# config 中必须包含 configurable.thread_id
# 这是 checkpointer 用来区分不同"对话"的标识
config_1 = {"configurable": {"thread_id": "demo-thread-1"}}

result_1 = ckpt_graph.invoke(
    {"messages": ["用户: 第一次对话开始"], "step_count": 0}, config=config_1
)

print(f"\n  [执行结果]")
print(f"    messages: {result_1['messages']}")
print(f"    step_count: {result_1['step_count']}")

# ─── 第四步: 查看 checkpoint 状态 ─────────────────────────────────────────
print("\n[查看 checkpoint 状态...]")
state_snapshot = ckpt_graph.get_state(config_1)
print(f'  state.values["messages"] 长度: {len(state_snapshot.values["messages"])}')
print(f'  state.values["step_count"]: {state_snapshot.values["step_count"]}')
print(f"  state.next: {state_snapshot.next}")
print("  (next 为空 tuple 表示图已执行完毕)")

# ─── 第五步: 同一个 thread_id 再次执行 — 状态累积! ───────────────────────
print("\n" + "-" * 50)
print(" 同一个 thread_id 再次执行 — 观察状态累积")
print("-" * 50)

result_2 = ckpt_graph.invoke(
    {"messages": ["用户: 第二次对话继续"], "step_count": 0}, config=config_1
)

print(f"\n  [第二次执行结果]")
print(f"    messages 数量: {len(result_2['messages'])}")
print(f"    所有 messages:")
for i, msg in enumerate(result_2["messages"]):
    print(f"      [{i}] {msg}")

print("""
  观察要点:
  ─────────
  1. 第二次执行时, messages 累积了! (之前的消息还在)
  2. 这就是 checkpoint 的威力 — 状态跨调用持久化
  3. thread_id 相同 = 继续同一个"对话"
  4. thread_id 不同 = 开启全新的"对话"
""")

# ─── 第六步: 不同 thread_id — 全新对话 ───────────────────────────────────
print("-" * 50)
print(" 不同 thread_id — 全新对话")
print("-" * 50)

config_2 = {"configurable": {"thread_id": "demo-thread-2"}}

result_3 = ckpt_graph.invoke(
    {"messages": ["用户: 这是一个全新的对话"], "step_count": 0}, config=config_2
)

print(f"\n  [新 thread 执行结果]")
print(f"    messages 数量: {len(result_3['messages'])}")
print(f"    (只有 3 条, 因为是全新的 thread, 没有历史累积)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 3: Human-in-the-Loop — interrupt_before 实现人工审批
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print(" Chapter 3: Human-in-the-Loop — interrupt_before 实现人工审批")
print("=" * 70)

print("""
  本章核心: 用 interrupt_before 让图在关键节点前暂停, 等待人类决策

  场景: AI 生成一个方案, 人类审核后决定是否继续

  图结构:
  ┌───────┐     ┌──────────┐     ┌──────────┐
  │ START │────→│ AI 生成  │────→│ 执行操作 │────→ END
  └───────┘     └──────────┘     └──────────┘
                 (generate)    ⚡ (execute)
                                  ↑
                          interrupt_before!
                          (这里暂停, 等人类审核)

  工作流程:
  ─────────
  1. invoke() → AI 生成方案 → 到达 execute 前暂停
  2. get_state() → 查看 AI 生成了什么
  3. 人类决策: 批准/拒绝
  4. update_state() → 注入人类决策
  5. invoke(None) → 从暂停处继续执行
""")


# ─── 第一步: 定义带审批的状态 ──────────────────────────────────────────────
class ApprovalState(TypedDict):
    """带人工审批的状态"""

    messages: Annotated[list, operator.add]
    proposal: str  # AI 生成的方案
    human_approved: bool  # 人类是否批准
    final_result: str  # 最终结果


# ─── 第二步: 定义节点 ──────────────────────────────────────────────────────
def generate_proposal(state: ApprovalState) -> dict:
    """AI 节点: 生成方案"""
    print("\n  [Node: generate_proposal] AI 正在生成方案...")

    # 模拟 AI 生成一个方案 (实际项目中会调用 LLM)
    proposal = "建议将服务器从 2 核升级到 8 核, 预计费用增加 3000 元/月"

    print(f"    生成的方案: {proposal}")
    return {
        "messages": [f"[AI] 方案已生成: {proposal}"],
        "proposal": proposal,
        "human_approved": False,  # 默认未批准
    }


def execute_action(state: ApprovalState) -> dict:
    """执行节点: 根据人类决策执行或取消"""
    print("\n  [Node: execute_action] 检查人类决策...")
    print(f"    human_approved = {state['human_approved']}")

    if state["human_approved"]:
        result = "操作已执行! 服务器升级工单已提交。"
        print(f"    [✓] 人类批准, 执行操作: {result}")
    else:
        result = "操作已取消。人类拒绝了该方案。"
        print(f"    [✗] 人类拒绝, 取消操作")

    return {
        "messages": [f"[执行] {result}"],
        "final_result": result,
    }


# ─── 第三步: 构建带 interrupt 的图 ────────────────────────────────────────
print("\n[构建带 interrupt_before 的图...]")

hitl_memory = MemorySaver()

hitl_builder = StateGraph(ApprovalState)
hitl_builder.add_node("generate_proposal", generate_proposal)
hitl_builder.add_node("execute_action", execute_action)

hitl_builder.add_edge(START, "generate_proposal")
hitl_builder.add_edge("generate_proposal", "execute_action")
hitl_builder.add_edge("execute_action", END)

# 关键! interrupt_before=["execute_action"]
# 意思: 在执行 execute_action 节点之前, 暂停图的执行!
hitl_graph = hitl_builder.compile(
    checkpointer=hitl_memory,
    interrupt_before=["execute_action"],  # ← 这是 HITL 的核心!
)

print("[✓] HITL 图构建完成!")
print('    interrupt_before=["execute_action"]')
print("    → 图会在 execute_action 执行前暂停")

# ─── 第四步: 第一次 invoke — 图会暂停! ───────────────────────────────────
print("\n" + "-" * 50)
print(" 演示: 图在 execute_action 前暂停")
print("-" * 50)

hitl_config = {"configurable": {"thread_id": "hitl-demo-1"}}

print("\n[第一次 invoke — 图会自动暂停...]")
result_hitl = hitl_graph.invoke(
    {
        "messages": ["用户: 请帮我升级服务器"],
        "proposal": "",
        "human_approved": False,
        "final_result": "",
    },
    config=hitl_config,
)

print(f"\n  [invoke 返回]")
print(f"    messages: {result_hitl['messages']}")
print(f"    proposal: {result_hitl['proposal']}")
print(f"    human_approved: {result_hitl['human_approved']}")

# ─── 第五步: 查看暂停状态 ──────────────────────────────────────────────────
print("\n[查看暂停状态 — get_state()]")
paused_state = hitl_graph.get_state(hitl_config)
print(f"  state.next: {paused_state.next}")
print(f'  ↑ 显示下一个要执行的节点是 "execute_action"')
print(f"  ↑ 但它被 interrupt_before 拦住了, 还没执行!")

print("""
  ┌─────────────────────────────────────────────────────────┐
  │  当前状态:                                              │
  │                                                         │
  │  [generate_proposal] ──✓──→ ⚡暂停⚡ → [execute_action] │
  │                              ↑ 我们在这里!             │
  │                              等待人类审核...            │
  └─────────────────────────────────────────────────────────┘
""")

# ─── 第六步: 模拟人类审核 — 批准场景 ─────────────────────────────────────
print("-" * 50)
print(" 模拟人类审核: 批准!")
print("-" * 50)

# 在实际应用中, 这里会是一个 web 界面或命令行交互
# 人类看到方案后做出决策
human_decision = True  # 人类决定: 批准!
print(f"\n  [人类决策] approved = {human_decision}")
print("  (实际应用中, 这可能来自 Web UI、Slack 消息、命令行输入等)")

# update_state() 注入人类的决策到图的状态中
# as_node 参数指定"假装这个更新来自哪个节点"
# 这很重要! 因为图需要知道从哪里继续执行
hitl_graph.update_state(
    hitl_config,
    values={"human_approved": human_decision},
    as_node="generate_proposal",  # 假装是 generate_proposal 节点产生的更新
)

print("  [✓] update_state() 完成, human_approved 已设为 True")

# 验证状态已更新
updated_state = hitl_graph.get_state(hitl_config)
print(
    f'  验证 state.values["human_approved"]: {updated_state.values["human_approved"]}'
)

# ─── 第七步: 继续执行 — invoke(None) ─────────────────────────────────────
print("\n[继续执行 — invoke(None, config)...]")
print('  (传入 None 表示"不添加新输入, 从暂停处继续")')

final_result = hitl_graph.invoke(None, config=hitl_config)

print(f"\n  [最终结果]")
print(f"    messages: {final_result['messages']}")
print(f"    final_result: {final_result['final_result']}")
print(f"    human_approved: {final_result['human_approved']}")

# ─── 第八步: 演示拒绝场景 ──────────────────────────────────────────────────
print("\n" + "-" * 50)
print(" 演示: 人类拒绝的场景")
print("-" * 50)

# 新的 thread — 全新的流程
hitl_config_reject = {"configurable": {"thread_id": "hitl-demo-reject"}}

print("\n[invoke — 生成方案并暂停...]")
result_reject = hitl_graph.invoke(
    {
        "messages": ["用户: 请帮我删除所有数据"],
        "proposal": "",
        "human_approved": False,
        "final_result": "",
    },
    config=hitl_config_reject,
)
print(f"  方案: {result_reject['proposal']}")

# 人类拒绝!
human_decision_reject = False
print(f"\n  [人类决策] approved = {human_decision_reject} (拒绝!)")

hitl_graph.update_state(
    hitl_config_reject,
    values={"human_approved": human_decision_reject},
    as_node="generate_proposal",
)

# 继续执行
print("\n[继续执行...]")
final_reject = hitl_graph.invoke(None, config=hitl_config_reject)
print(f"  最终结果: {final_reject['final_result']}")

print("""
  总结 HITL 工作流:
  ─────────────────
  1. compile(interrupt_before=["节点名"]) — 设置暂停点
  2. invoke(初始输入, config) — 执行到暂停点停下
  3. get_state(config) — 查看当前状态 (state.next 显示暂停在哪)
  4. update_state(config, values, as_node) — 注入人类决策
  5. invoke(None, config) — 从暂停处继续执行
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 4: 实战 — AI 写作助手 (需要人工审核)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print(" Chapter 4: 实战 — AI 写作助手 (需要人工审核)")
print("=" * 70)

print("""
  场景: 一个 AI 写作助手, 流程如下:

  ┌───────┐    ┌────────┐    ┌────────┐    ┌────────┐
  │ START │───→│ 起草   │───→│ 审核   │───→│ 修改   │───→...
  └───────┘    │ (AI)   │    │ (人类) │    │ (AI)   │
               └────────┘    └────────┘    └────────┘
                  draft     ⚡interrupt     revise
                                │
                                ├─→ 通过 ───→ [发布] → END
                                │              publish
                                └─→ 需修改 ──→ [修改] → [审核] (循环)

  特点:
  1. AI 起草内容
  2. 人类审核 — 暂停等待 (interrupt_before)
  3. 人类可以: 直接通过 → 发布 / 要求修改 → AI 修改后重新审核
  4. 整个过程有 checkpoint, 随时可以中断和恢复
""")


# ─── 定义状态 ──────────────────────────────────────────────────────────────
class WritingState(TypedDict):
    """AI 写作助手状态"""

    messages: Annotated[list, operator.add]  # 流程日志
    topic: str  # 写作主题
    draft: str  # 当前草稿
    feedback: str  # 人类反馈 (空=通过, 有内容=需修改)
    is_approved: bool  # 是否通过审核
    revision_count: int  # 修改次数
    final_output: str  # 最终发布内容


# ─── 定义节点 ──────────────────────────────────────────────────────────────
def draft_node(state: WritingState) -> dict:
    """起草节点: AI 根据主题生成草稿"""
    print("\n  [Node: draft] AI 正在起草...")
    topic = state["topic"]
    revision_count = state.get("revision_count", 0)

    if revision_count == 0:
        # 第一次起草 — 调用 LLM
        print(f"    主题: {topic}")
        print("    调用 LLM 生成初稿...")

        response = llm.invoke(
            [
                SystemMessage(
                    content="你是一个专业的文案写手。请根据主题写一段简短的宣传文案(50字以内)。"
                ),
                HumanMessage(content=f"主题: {topic}"),
            ]
        )
        draft_text = response.content
    else:
        # 修改 — 根据反馈修改
        feedback = state.get("feedback", "")
        old_draft = state.get("draft", "")
        print(f"    第 {revision_count} 次修改")
        print(f"    原草稿: {old_draft}")
        print(f"    人类反馈: {feedback}")
        print("    调用 LLM 根据反馈修改...")

        response = llm.invoke(
            [
                SystemMessage(
                    content="你是一个专业的文案写手。请根据反馈修改文案(50字以内)。"
                ),
                HumanMessage(
                    content=f"原文案: {old_draft}\n\n修改要求: {feedback}\n\n请输出修改后的文案:"
                ),
            ]
        )
        draft_text = response.content

    print(f"    草稿内容: {draft_text}")

    return {
        "messages": [f"[起草] 第 {revision_count + 1} 版草稿已完成"],
        "draft": draft_text,
        "is_approved": False,
    }


def review_node(state: WritingState) -> dict:
    """审核节点: 这个节点本身只是一个"通过点"

    真正的审核逻辑在 interrupt + update_state 中完成。
    当这个节点执行时, 说明人类已经做出了决策。
    """
    print("\n  [Node: review] 处理人类审核结果...")
    print(f"    is_approved: {state['is_approved']}")
    print(f"    feedback: {state.get('feedback', '')}")

    if state["is_approved"]:
        return {"messages": ["[审核] 人类批准, 准备发布!"]}
    else:
        return {
            "messages": [f"[审核] 人类要求修改: {state.get('feedback', '')}"],
            "revision_count": state.get("revision_count", 0) + 1,
        }


def publish_node(state: WritingState) -> dict:
    """发布节点: 输出最终结果"""
    print("\n  [Node: publish] 发布最终内容!")
    final = state["draft"]
    print(f"    发布内容: {final}")

    return {
        "messages": ["[发布] 内容已发布!"],
        "final_output": final,
    }


def revise_node(state: WritingState) -> dict:
    """修改节点: 根据反馈让 AI 修改"""
    print("\n  [Node: revise] AI 根据反馈修改中...")
    feedback = state.get("feedback", "")
    old_draft = state.get("draft", "")
    revision_count = state.get("revision_count", 0)

    print(f"    当前修改次数: {revision_count}")
    print(f"    原草稿: {old_draft}")
    print(f"    反馈: {feedback}")

    response = llm.invoke(
        [
            SystemMessage(
                content="你是一个专业的文案写手。请根据反馈修改文案(50字以内)。直接输出修改后的文案。"
            ),
            HumanMessage(content=f"原文案: {old_draft}\n\n修改要求: {feedback}"),
        ]
    )
    new_draft = response.content
    print(f"    修改后: {new_draft}")

    return {
        "messages": [f"[修改] 第 {revision_count} 次修改完成"],
        "draft": new_draft,
        "is_approved": False,
    }


# ─── 定义条件路由 ──────────────────────────────────────────────────────────
def after_review_router(state: WritingState) -> str:
    """审核后的路由: 通过 → 发布, 不通过 → 修改"""
    if state["is_approved"]:
        return "publish"
    else:
        return "revise"


# ─── 构建图 ────────────────────────────────────────────────────────────────
print("\n[构建 AI 写作助手图...]")

writing_memory = MemorySaver()

writing_builder = StateGraph(WritingState)

# 添加节点
writing_builder.add_node("draft", draft_node)
writing_builder.add_node("review", review_node)
writing_builder.add_node("revise", revise_node)
writing_builder.add_node("publish", publish_node)

# 添加边
writing_builder.add_edge(START, "draft")
writing_builder.add_edge("draft", "review")  # 起草完 → 审核

# 条件路由: 审核后根据结果决定下一步
writing_builder.add_conditional_edges(
    "review", after_review_router, {"publish": "publish", "revise": "revise"}
)

# 修改完 → 回到审核 (循环!)
writing_builder.add_edge("revise", "review")

# 发布 → 结束
writing_builder.add_edge("publish", END)

# 编译 — interrupt_before review 节点!
# 意思: AI 起草完后, 到审核前暂停, 等人类决策
writing_graph = writing_builder.compile(
    checkpointer=writing_memory,
    interrupt_before=["review"],  # 审核前暂停!
)

print("[✓] AI 写作助手图构建完成!")
print("""
  节点: draft → (interrupt!) → review → publish/revise
  审核前会暂停, 等待人类:
    - 设置 is_approved=True → 走 publish 路径
    - 设置 is_approved=False + feedback → 走 revise 路径
""")

# ─── 运行完整流程 ──────────────────────────────────────────────────────────
print("\n" + "─" * 70)
print(" 完整演示: AI 写作助手流程")
print("─" * 70)

writing_config = {"configurable": {"thread_id": "writing-demo-1"}}

# === 阶段 1: AI 起草 ===
print("\n" + "~" * 40)
print(" 阶段 1: AI 起草初稿")
print("~" * 40)

result_w1 = writing_graph.invoke(
    {
        "messages": ["[系统] 写作任务开始"],
        "topic": "新能源汽车的未来",
        "draft": "",
        "feedback": "",
        "is_approved": False,
        "revision_count": 0,
        "final_output": "",
    },
    config=writing_config,
)

print(f"\n  [图已暂停]")
print(f"    当前草稿: {result_w1['draft']}")

# 查看暂停状态
w_state = writing_graph.get_state(writing_config)
print(f"    next: {w_state.next} (在 review 前暂停)")

# === 阶段 2: 人类审核 — 要求修改 ===
print("\n" + "~" * 40)
print(" 阶段 2: 人类审核 — 要求修改")
print("~" * 40)

# 模拟人类: 看了草稿后, 觉得需要修改
human_feedback = "语气太正式了, 改得更活泼一些, 加入一些年轻人喜欢的表达"
print(f"\n  [人类反馈] {human_feedback}")
print("  [人类决策] is_approved = False (需要修改)")

# 注入人类决策
writing_graph.update_state(
    writing_config,
    values={
        "is_approved": False,
        "feedback": human_feedback,
    },
    as_node="draft",  # 假装来自 draft 节点的输出
)

# 继续执行 — 会执行 review → revise, 然后在下一次 review 前又暂停!
print("\n[继续执行 — review → revise → (暂停在下一次 review 前)...]")
result_w2 = writing_graph.invoke(None, config=writing_config)

print(f"\n  [图再次暂停]")
print(f"    修改后草稿: {result_w2['draft']}")
print(f"    修改次数: {result_w2.get('revision_count', 0)}")

w_state_2 = writing_graph.get_state(writing_config)
print(f"    next: {w_state_2.next} (又在 review 前暂停了!)")

# === 阶段 3: 人类审核 — 通过! ===
print("\n" + "~" * 40)
print(" 阶段 3: 人类审核 — 通过!")
print("~" * 40)

print("\n  [人类决策] is_approved = True (批准发布!)")

# 注入: 批准
writing_graph.update_state(
    writing_config,
    values={
        "is_approved": True,
        "feedback": "",
    },
    as_node="draft",  # 假装来自 draft 的输出
)

# 继续执行 — review → publish → END
print("\n[继续执行 — review → publish → END...]")
result_w3 = writing_graph.invoke(None, config=writing_config)

print(f"\n  [图执行完毕!]")
print(f"    最终发布内容: {result_w3['final_output']}")
print(f"    总修改次数: {result_w3.get('revision_count', 0)}")
print(f"    完整流程日志:")
for i, msg in enumerate(result_w3["messages"]):
    print(f"      [{i}] {msg}")

# ─── 流程总结 ──────────────────────────────────────────────────────────────
print("""
  完整流程回顾:
  ─────────────
  1. invoke(初始输入) → AI 起草 → 暂停在 review 前
  2. 人类看草稿 → "不行, 改一下" → update_state(feedback + is_approved=False)
  3. invoke(None) → review(走revise路径) → revise → 暂停在 review 前
  4. 人类看修改版 → "可以了!" → update_state(is_approved=True)
  5. invoke(None) → review(走publish路径) → publish → END

  这就是一个完整的 Human-in-the-Loop 写作工作流!
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary: 总结与最佳实践
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print(" Summary: 总结与最佳实践")
print("=" * 70)

print("""
  ┌──────────────────────────────────────────────────────────────────┐
  │                 HITL + Checkpointing 知识总结                    │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  一、什么时候需要 Human-in-the-Loop?                            │
  │  ─────────────────────────────────────                          │
  │  - 执行不可逆操作前 (删除数据、发送邮件、付款)                 │
  │  - 高风险决策 (代码部署、权限变更)                              │
  │  - 内容发布 (文章、广告、公告)                                  │
  │  - 需要领域专家判断的环节                                       │
  │  - 法律/合规要求必须有人工审批                                  │
  │                                                                  │
  │  二、Checkpointing 模式                                          │
  │  ────────────────────                                           │
  │  - MemorySaver: 开发/测试用, 内存存储, 重启即丢失              │
  │  - SqliteSaver: 单机持久化, 适合小规模                          │
  │  - PostgresSaver: 生产级, 分布式, 高可用                        │
  │                                                                  │
  │  三、关键 API                                                    │
  │  ──────────                                                     │
  │  - compile(checkpointer=xxx, interrupt_before=[...])            │
  │  - invoke(input, config={"configurable":{"thread_id":"xx"}})    │
  │  - get_state(config) → 查看状态, .next 看暂停位置              │
  │  - update_state(config, values, as_node="xx") → 注入决策       │
  │  - invoke(None, config) → 从暂停处继续                         │
  │                                                                  │
  │  四、设计原则                                                    │
  │  ──────────                                                     │
  │  1. interrupt 放在"不可逆节点"之前, 不要放太早                 │
  │  2. 状态设计要清晰: 哪些字段是 AI 填的, 哪些是人类填的        │
  │  3. as_node 要选对: 决定了图从哪条边继续                       │
  │  4. 考虑超时: 人类可能几小时/几天后才审批                      │
  │  5. 提供上下文: 暂停时给人类足够信息做决策                     │
  │                                                                  │
  │  五、架构图                                                      │
  │  ─────────                                                      │
  │                                                                  │
  │  [前端 UI]                                                       │
  │      │                                                           │
  │      ▼                                                           │
  │  [API Server] ←──→ [LangGraph + Checkpointer]                  │
  │      │                          │                                │
  │      │                          ▼                                │
  │      │                    [PostgreSQL]                           │
  │      │                    (存储 checkpoints)                     │
  │      ▼                                                           │
  │  [通知服务] → 邮件/Slack/企微 → 人类审批 → 回调 API            │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

  恭喜! 你已经掌握了 LangGraph 的两个高级特性:
  - Checkpointing: 让图有"记忆", 跨调用保持状态
  - Human-in-the-Loop: 让 AI 在关键节点暂停, 等待人类决策

  这两个特性组合在一起, 就是构建生产级 AI Agent 的基石! 🎉
""")

print("\n[项目 18 全部执行完毕]")
