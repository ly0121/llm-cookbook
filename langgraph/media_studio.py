"""
╔══════════════════════════════════════════════════════════════════╗
║         项目四：基于 LangGraph 的多智能体协作系统                      ║
║         模拟"自媒体工作室"——研究员 → 写手 → 主编 流水线             ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普一：为什么 AgentExecutor 不够用了？】
═══════════════════════════════════════════════════════════════════

回顾项目三：我们用 AgentExecutor 构建了一个"单体 Agent"。
它的工作模式是：

  用户提问 → LLM 思考 → 调用工具 → 观察结果 → 再思考 → ... → Final Answer

问题来了——如果你想做这样的事情：

  ① 让不同的 AI 角色分工合作（研究员 + 写手 + 主编）
  ② 让某个步骤可以"打回重做"（主编不满意 → 写手重写）
  ③ 让流程有明确的"阶段"，而不是一锅乱炖

AgentExecutor 做不到！因为它只有一个 LLM 在"想"，
没有"多角色协作"和"可控流程分支"的概念。

═══════════════════════════════════════════════════════════════════
【前置科普二：LangGraph 是什么？——用"图"来编排 AI 工作流】
═══════════════════════════════════════════════════════════════════

LangGraph 的核心思想：把 AI 工作流看成一张"有向图"（Directed Graph）。

  ┌────────────────────────────────────────────────────────────┐
  │  图的三要素：                                                │
  │                                                            │
  │  ① Node（节点）= 一个"工作站"                              │
  │     每个节点是一个 Python 函数，做一件具体的事：             │
  │     查资料、写文章、审核文章……                              │
  │                                                            │
  │  ② Edge（边）= 节点之间的"传送带"                          │
  │     数据从一个节点流向另一个节点。                           │
  │     普通边：A → B（A 做完，无条件交给 B）                   │
  │     条件边：A → B 或 A → C（根据条件选择下一站）            │
  │                                                            │
  │  ③ State（状态）= 在传送带上流动的"数据包"                 │
  │     所有节点共享同一个 State（TypedDict），                  │
  │     每个节点可以读取、修改 State 中的字段。                  │
  │     这就是节点之间"传递信息"的方式！                        │
  └────────────────────────────────────────────────────────────┘

对比 AgentExecutor vs LangGraph：

  ┌─────────────────────┬────────────────────────────┐
  │     AgentExecutor   │        LangGraph           │
  ├─────────────────────┼────────────────────────────┤
  │  单个 LLM 循环      │  多个节点（多角色）协作     │
  │  ReAct 固定循环     │  自定义图结构（灵活）       │
  │  难以控制流程       │  条件边精确控制路由         │
  │  黑盒运行          │  每一步状态透明可观测       │
  │  无法"打回重做"    │  条件边可实现循环/重试      │
  └─────────────────────┴────────────────────────────┘

═══════════════════════════════════════════════════════════════════
【前置科普三：本项目的图结构——"自媒体工作室"数据流转图】
═══════════════════════════════════════════════════════════════════

              ┌──────────────┐
              │   START      │   用户输入主题
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │  研究员       │   根据主题生成大纲和素材
              │  (researcher)│
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │  写手         │   根据素材写出短文初稿
              │  (writer)    │◀─────────────────┐
              └──────┬───────┘                  │
                     │                          │
                     ▼                          │ 不合格
              ┌──────────────┐                  │ (打回重写)
              │  主编/审核    │──────────────────┘
              │  (editor)    │
              └──────┬───────┘
                     │ 合格
                     ▼
              ┌──────────────┐
              │    END       │   输出最终文章
              └──────────────┘

  State（状态数据包）在这张图上流转：
    topic      → 用户给定的主题（全程不变）
    outline    → 研究员产出的大纲素材
    draft      → 写手产出的文章初稿
    feedback   → 主编给出的修改意见
    revision   → 当前是第几次修改（防止无限循环）
    final      → 最终定稿的文章
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# typing 模块：TypedDict 和 Annotated 用于定义 State 的结构
from typing import TypedDict, Annotated

# LangGraph 核心组件
# StateGraph  ：用来定义图结构（添加节点、边）
# START / END ：特殊节点，标记图的起点和终点
from langgraph.graph import StateGraph, START, END

# operator.add：用作 Annotated 的 reducer，当多个节点写同一个字段时的合并策略
# 本项目不需要合并策略（每个字段只有一个节点在写），所以不用 operator
# 但了解它的存在很重要：在并行节点场景下，reducer 决定如何合并冲突

# LangChain 聊天模型
from langchain_openai import ChatOpenAI

# 提示词模板
from langchain_core.prompts import ChatPromptTemplate

# 输出解析器
from langchain_core.output_parsers import StrOutputParser


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化 LLM
# 目标：建立与大模型的连接（和前几个项目完全一致）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 0 章：初始化 LLM")
print("=" * 60)

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "kivy-kimi-k2_5"

# 多智能体场景建议 temperature=0.7：每个角色需要一定创造力
# 如果需要更确定的输出（比如审核角色），可以在节点内部单独设置
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.7,
)

print("✅ LLM 初始化完成")
print(f"   模型: {MODEL_NAME}")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：定义 State（状态数据结构）
# 目标：用 TypedDict 定义在图中流转的"数据包"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：定义 State（状态数据结构）")
print("=" * 60)
print()

# ── State 是什么？──────────────────────────────────────────
#
# State 就是在图上所有节点之间共享的"数据包"。
# 每个节点：
#   ① 接收当前 State 作为输入参数
#   ② 返回一个 dict，其中的 key 会更新 State 对应的字段
#
# 用 TypedDict 定义 State 的好处：
#   ① 类型提示：IDE 能自动补全字段名
#   ② 文档化：一眼看清楚图中流转哪些数据
#   ③ 运行时检查：LangGraph 会校验节点返回的 key 是否合法
#
# ⚠️ 避坑指南：State 字段更新是"覆盖"语义（除非用 Annotated + reducer）
#   节点 A 返回 {"draft": "版本1"}
#   节点 B 返回 {"draft": "版本2"}
#   → State 中 draft 最终是 "版本2"（后者覆盖前者）
#
#   如果你需要"追加"语义（比如消息列表越加越长），用：
#     messages: Annotated[list, operator.add]
#   这样每个节点返回的 list 会被 append 到已有 list 后面。
#   本项目的字段都是"覆盖"语义，不需要 reducer。


class StudioState(TypedDict):
    """自媒体工作室的状态数据包——在所有节点之间流转"""

    # ── 输入字段 ──
    topic: str  # 用户给定的创作主题（全程不变）

    # ── 研究员产出 ──
    outline: str  # 研究员生成的大纲和素材

    # ── 写手产出 ──
    draft: str  # 写手的文章初稿（可能被多次覆盖）

    # ── 主编产出 ──
    feedback: str  # 主编的审核意见（不合格时有具体修改建议）
    is_approved: bool  # 主编是否批准（True=合格，False=打回）

    # ── 流程控制 ──
    revision: int  # 当前是第几轮修改（从 0 开始，用于防止无限循环）
    final: str  # 最终定稿的文章（只有通过审核后才有值）


print("【StudioState 字段定义】")
print("  ┌────────────────────────────────────────────────────┐")
print("  │  topic       : str   ← 用户输入的主题              │")
print("  │  outline     : str   ← 研究员产出的大纲            │")
print("  │  draft       : str   ← 写手的文章初稿              │")
print("  │  feedback    : str   ← 主编的修改意见              │")
print("  │  is_approved : bool  ← 主编是否批准                │")
print("  │  revision    : int   ← 第几轮修改                  │")
print("  │  final       : str   ← 最终定稿                    │")
print("  └────────────────────────────────────────────────────┘")
print()
print("💡 所有节点共享同一个 State 实例，通过返回 dict 来更新字段。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：定义节点函数（三个 AI 角色）
# 目标：每个角色是一个 Python 函数，读取 State → 调用 LLM → 返回更新
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：定义节点函数（三个 AI 角色）")
print("=" * 60)
print()

# ── 节点函数的规则 ─────────────────────────────────────────
#
# 每个节点函数必须：
#   ① 接受一个参数：state（类型是 StudioState）
#   ② 返回一个 dict：其中的 key 对应 State 中要更新的字段
#
# 节点函数可以做任何事情：调用 LLM、调用 API、读写文件、纯逻辑计算……
# LangGraph 不关心节点内部做了什么，只关心它返回了什么。

# ── 创建输出解析器（所有节点共用）──
parser = StrOutputParser()


# ═══════════════════════════════════════════════════════════
# 节点一：研究员（Researcher）
# ═══════════════════════════════════════════════════════════


def researcher_node(state: StudioState) -> dict:
    """
    研究员节点：根据主题生成大纲和素材。

    输入：state["topic"]（用户给定的主题）
    输出：更新 state["outline"]（大纲素材）
    """
    print()
    print("┌" + "─" * 58 + "┐")
    print("│  🔬 【研究员】正在工作...                                   │")
    print("└" + "─" * 58 + "┘")
    print(f"  📥 收到的任务主题：「{state['topic']}」")

    # 构造研究员专属的 Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一位资深的自媒体内容研究员。
你的任务是为给定主题生成文章大纲和关键素材。

请输出：
1. 一个 3-5 点的文章大纲（每点一句话概括）
2. 每个大纲点对应的 2-3 个关键素材/论据/数据

注意：大纲要有逻辑递进关系，素材要具体、有说服力。
请直接输出大纲内容，不要加多余的客套话。""",
            ),
            ("human", "请为以下主题生成文章大纲和素材：{topic}"),
        ]
    )

    # 用 LCEL 链调用 LLM
    chain = prompt | llm | parser
    outline = chain.invoke({"topic": state["topic"]})

    print(f"  📤 研究员产出的大纲：")
    print(f"  {'─' * 50}")
    # 缩进打印大纲内容
    for line in outline.split("\n"):
        print(f"  │ {line}")
    print(f"  {'─' * 50}")
    print()

    # 返回要更新的 State 字段
    return {"outline": outline}


# ═══════════════════════════════════════════════════════════
# 节点二：写手（Writer）
# ═══════════════════════════════════════════════════════════


def writer_node(state: StudioState) -> dict:
    """
    写手节点：根据大纲素材写出文章初稿。

    输入：state["topic"]（主题）+ state["outline"]（大纲）
          如果是重写，还会参考 state["feedback"]（主编的修改意见）
    输出：更新 state["draft"]（文章初稿）+ state["revision"]（修改轮次+1）
    """
    current_revision = state.get("revision", 0)
    feedback = state.get("feedback", "")

    print()
    print("┌" + "─" * 58 + "┐")
    if current_revision == 0:
        print("│  ✍️  【写手】正在创作初稿...                                │")
    else:
        print(
            f"│  ✍️  【写手】正在第 {current_revision + 1} 次修改...                           │"
        )
    print("└" + "─" * 58 + "┘")
    print(f"  📥 主题：「{state['topic']}」")
    print(f"  📥 大纲长度：{len(state.get('outline', ''))} 字")
    if feedback:
        print(f"  📥 主编反馈：「{feedback[:80]}...」")

    # 根据是否有反馈，构造不同的 Prompt
    if feedback:
        # 重写模式：参考之前的稿子和主编反馈
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一位才华横溢的自媒体写手。
你之前写的文章被主编打回了，现在需要根据反馈修改。

要求：
1. 文章字数必须在 150 字以上
2. 语言生动有趣，使用比喻和故事
3. 逻辑清晰，层层递进
4. 必须认真参考主编的修改意见进行改进

请直接输出修改后的文章正文，不要加标题前缀。""",
                ),
                (
                    "human",
                    """原始主题：{topic}

大纲素材：
{outline}

你上一稿的内容：
{draft}

主编的修改意见：
{feedback}

请根据以上反馈，重新写一篇改进后的文章：""",
                ),
            ]
        )
        draft = (prompt | llm | parser).invoke(
            {
                "topic": state["topic"],
                "outline": state["outline"],
                "draft": state.get("draft", ""),
                "feedback": feedback,
            }
        )
    else:
        # 首次创作模式
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一位才华横溢的自媒体写手。
根据研究员提供的大纲和素材，写出一篇引人入胜的短文。

要求：
1. 文章字数必须在 150 字以上
2. 语言生动有趣，使用比喻和故事
3. 逻辑清晰，层层递进
4. 开头要有吸引力（hook），结尾要有力量

请直接输出文章正文，不要加标题前缀。""",
                ),
                (
                    "human",
                    """主题：{topic}

大纲素材：
{outline}

请根据以上素材，写出一篇生动有趣的短文：""",
                ),
            ]
        )
        draft = (prompt | llm | parser).invoke(
            {
                "topic": state["topic"],
                "outline": state["outline"],
            }
        )

    print(f"  📤 写手产出的文章（{len(draft)} 字）：")
    print(f"  {'─' * 50}")
    for line in draft.split("\n"):
        print(f"  │ {line}")
    print(f"  {'─' * 50}")
    print()

    # 返回更新：新的稿件 + 修改轮次+1
    return {
        "draft": draft,
        "revision": current_revision + 1,
    }


# ═══════════════════════════════════════════════════════════
# 节点三：主编/审核（Editor）
# ═══════════════════════════════════════════════════════════


def editor_node(state: StudioState) -> dict:
    """
    主编节点：审核文章质量，决定通过或打回。

    输入：state["draft"]（写手的文章）
    输出：更新 state["is_approved"]（是否通过）
          更新 state["feedback"]（修改意见，不通过时）
          更新 state["final"]（最终稿，通过时）

    审核标准：
      ① 字数是否达到 150 字以上
      ② LLM 判断内容是否生动有趣、逻辑清晰
    """
    draft = state["draft"]
    revision = state.get("revision", 1)

    print()
    print("┌" + "─" * 58 + "┐")
    print("│  📋 【主编】正在审核...                                     │")
    print("└" + "─" * 58 + "┘")
    print(f"  📥 收到写手第 {revision} 版稿件（{len(draft)} 字）")

    # ── 硬性检查：字数是否达标 ──
    MIN_WORDS = 150
    if len(draft) < MIN_WORDS:
        feedback = f"字数不达标！当前仅 {len(draft)} 字，要求至少 {MIN_WORDS} 字。请扩充内容，增加更多细节和例子。"
        print(f"  ❌ 硬性不通过：字数不达标（{len(draft)} < {MIN_WORDS}）")
        print(f"  📤 反馈：「{feedback}」")
        print()
        return {
            "is_approved": False,
            "feedback": feedback,
        }

    # ── 软性检查：让 LLM 判断内容质量 ──
    #
    # 这里用 temperature=0 让审核结果更稳定
    # ⚠️ 避坑指南：让 LLM 做"通过/不通过"判断时，
    #   prompt 必须要求它先给出明确结论（PASS/FAIL），再给原因。
    #   否则 LLM 可能写一大堆分析但不给出明确结论。

    review_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一位严格的自媒体主编。请审核以下文章的质量。

审核标准：
1. 内容是否有趣、生动（有比喻或故事）
2. 逻辑是否清晰、层层递进
3. 是否有吸引读者的开头

请严格按以下格式回复（必须第一行就是结论）：
结论：PASS 或 FAIL
原因：一句话说明理由
修改建议：如果 FAIL，给出具体修改建议；如果 PASS，写"无"。""",
            ),
            ("human", "请审核以下文章：\n\n{draft}"),
        ]
    )

    review_result = (review_prompt | llm | parser).invoke({"draft": draft})

    print(f"  📝 主编审核意见：")
    for line in review_result.split("\n"):
        print(f"     {line}")

    # ── 解析 LLM 的审核结论 ──
    #
    # 简单的字符串检查：如果输出里包含"PASS"就通过
    # 生产环境中可以用更结构化的 OutputParser（如 PydanticOutputParser）
    is_approved = "PASS" in review_result.upper()

    # ── 防止无限循环：最多修改 3 次 ──
    MAX_REVISIONS = 3
    if not is_approved and revision >= MAX_REVISIONS:
        print(f"  ⚠️  已达最大修改次数（{MAX_REVISIONS}），强制通过！")
        is_approved = True

    if is_approved:
        print(f"  ✅ 审核通过！文章定稿。")
        print()
        return {
            "is_approved": True,
            "feedback": "",
            "final": draft,
        }
    else:
        # 提取修改建议作为 feedback
        feedback = review_result
        print(f"  ❌ 审核不通过，打回给写手修改。")
        print()
        return {
            "is_approved": False,
            "feedback": feedback,
        }


print("✅ 三个节点函数定义完成：")
print("   researcher_node → 研究员（生成大纲素材）")
print("   writer_node     → 写手（撰写/修改文章）")
print("   editor_node     → 主编（审核质量、打回/通过）")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：定义条件边路由函数
# 目标：根据主编的审核结果，决定下一站是"END"还是"writer"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：定义条件边路由函数")
print("=" * 60)
print()

# ── 条件边是什么？────────────────────────────────────────
#
# 普通边（Normal Edge）：A → B，无条件，A 做完一定交给 B。
#
# 条件边（Conditional Edge）：A → ？
#   需要一个"路由函数"来决定下一站是谁。
#   路由函数读取当前 State，返回一个字符串（节点名称）。
#
# 在本项目中：
#   主编审核后 → 路由函数检查 is_approved：
#     True  → 返回 "end"（图结束）
#     False → 返回 "writer"（打回重写）


def should_continue(state: StudioState) -> str:
    """
    条件边路由函数：决定主编审核后的下一站。

    读取 state["is_approved"]：
      True  → 返回 END（结束流程，输出最终文章）
      False → 返回 "writer"（打回给写手重写）

    返回值必须是图中已定义的节点名称，或者特殊值 END。
    """
    if state["is_approved"]:
        print("  🔀 路由决策：审核通过 → 流程结束（END）")
        return END
    else:
        print(
            f"  🔀 路由决策：审核不通过 → 打回给写手（第 {state.get('revision', 0) + 1} 次修改）"
        )
        return "writer"


print("【条件边路由函数 should_continue】")
print("  if is_approved == True  → END（输出最终文章）")
print("  if is_approved == False → 'writer'（打回重写）")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：构建图（StateGraph 组装）
# 目标：把节点、边、条件边组装成一张完整的工作流图
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 4 章：构建图（StateGraph 组装）")
print("=" * 60)
print()

# ── 步骤一：创建 StateGraph 实例 ──────────────────────────
#
# StateGraph 的参数是 State 类型（我们定义的 TypedDict）
# 这告诉 LangGraph："图中流转的数据长什么样"

workflow = StateGraph(StudioState)

# ── 步骤二：添加节点 ─────────────────────────────────────
#
# add_node(name, function)：
#   name     → 节点的名称（字符串），后面连边时用这个名字引用
#   function → 节点函数（我们在第2章定义的那些函数）
#
# ⚠️ 避坑指南：节点名称必须唯一！
#   如果你添加两个同名节点，后者会覆盖前者。

workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", editor_node)

print("  ✅ 添加节点：researcher, writer, editor")

# ── 步骤三：添加边（连接节点）────────────────────────────
#
# add_edge(from_node, to_node)：普通边，无条件跳转
#
# 特殊节点 START：图的入口点。
# 用 add_edge(START, "researcher") 表示"图开始时先执行 researcher"。

workflow.add_edge(START, "researcher")  # 入口 → 研究员
workflow.add_edge("researcher", "writer")  # 研究员 → 写手

print("  ✅ 添加普通边：START → researcher → writer")

# ── 步骤四：添加条件边 ───────────────────────────────────
#
# add_conditional_edges(source_node, route_function, path_map)：
#
#   source_node    → 条件判断发生在哪个节点之后
#   route_function → 路由函数（读取 State，返回目标节点名称）
#   path_map       → 可选，显式映射 {路由返回值: 目标节点名}
#                    如果路由函数直接返回节点名称，可以省略
#
# 这里 should_continue 返回 END 或 "writer"，
# LangGraph 会自动找到对应的节点。

workflow.add_conditional_edges(
    "editor",  # 条件边的起点：主编节点之后
    should_continue,  # 路由函数：决定去 END 还是 writer
)

print("  ✅ 添加条件边：editor → [END 或 writer]（由 should_continue 决定）")

# 写手完成后无条件交给主编
workflow.add_edge("writer", "editor")

print("  ✅ 添加普通边：writer → editor")
print()

# ── 步骤五：编译图 ───────────────────────────────────────
#
# .compile() 做什么？
#   ① 验证图的结构是否合法（所有节点都可达、没有孤立节点）
#   ② 生成可执行的 CompiledGraph 对象
#   ③ 编译后的图是一个 Runnable，可以用 .invoke() 调用
#
# ⚠️ 避坑指南：编译前必须确保所有节点都有出边！
#   如果某个节点没有出边，compile() 会报错。

app = workflow.compile()

print("  ✅ 图编译完成！")
print()
print("  【最终图结构】")
print("  START → researcher → writer ⇄ editor → END")
print("                         ↑                │")
print("                         └── 不合格打回 ──┘")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 章：运行工作流！
# 目标：用一个主题触发整个"自媒体工作室"，观察状态流转
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 5 章：运行工作流！")
print("=" * 60)
print()

# ── 输入：只需要提供 State 中的初始字段 ──────────────────
#
# 调用 app.invoke(initial_state) 时：
#   你只需要提供"起始节点需要的字段"，
#   其他字段会在图运行过程中被各节点填充。
#
# 我们的图从 researcher 开始，它只需要 topic，
# 所以初始 State 只提供 topic 即可。

TOPIC = "为什么程序员应该学习 AI？从工具到思维的转变"

print(f"🎬 启动自媒体工作室！")
print(f"   输入主题：「{TOPIC}」")
print()
print("━" * 60)
print("  ⬇️  数据开始在图中流转...")
print("━" * 60)

# ── 调用编译后的图 ─────────────────────────────────────────
#
# app.invoke() 做了什么？
#   ① 把初始 State 传给第一个节点（researcher）
#   ② researcher 处理完后，用返回值更新 State
#   ③ 沿着边把更新后的 State 传给下一个节点（writer）
#   ④ writer 处理完后传给 editor
#   ⑤ editor 处理完后，条件边决定继续还是结束
#   ⑥ 如果打回 writer，循环继续……
#   ⑦ 到达 END 时，返回最终的完整 State

final_state = app.invoke(
    {
        "topic": TOPIC,
        "revision": 0,  # 初始修改轮次为 0
        "is_approved": False,
        "feedback": "",
        "outline": "",
        "draft": "",
        "final": "",
    }
)

# ── 打印最终结果 ──────────────────────────────────────────

print()
print("━" * 60)
print("  🏁 工作流执行完毕！最终 State 状态：")
print("━" * 60)
print()
print(f"  topic       : 「{final_state['topic']}」")
print(f"  outline     : （{len(final_state.get('outline', ''))} 字，省略）")
print(f"  draft       : （{len(final_state.get('draft', ''))} 字）")
print(f"  revision    : 共经历 {final_state.get('revision', 0)} 轮写作")
print(f"  is_approved : {final_state.get('is_approved')}")
print(f"  feedback    : 「{final_state.get('feedback', '无')}」")
print()
print("═" * 60)
print("📰 【最终定稿文章】")
print("═" * 60)
print()
print(final_state.get("final", final_state.get("draft", "（无输出）")))
print()
print("═" * 60)
print()
print("🎉 项目四学习完毕！你已经掌握了 LangGraph 多智能体协作的核心机制。")
print()
print("💡 核心公式：")
print("   State（共享数据） + Nodes（角色函数） + Edges（流转路径）")
print("   + Conditional Edges（条件路由） = 可控多智能体工作流")
print()
print("💡 对比回顾：")
print("   项目三 AgentExecutor：单个 LLM 的 ReAct 循环，无法多角色协作")
print("   项目四 LangGraph    ：多节点图结构，角色分工 + 条件分支 + 循环重试")
print()
print("💡 进阶方向：")
print("   ① 加入更多节点（配图师、SEO 优化师）")
print("   ② 用 Annotated[list, operator.add] 实现消息历史追加")
print("   ③ 并行节点：多个研究员同时调研不同子话题")
print("   ④ 人机协作：加入 interrupt_before 让人类在关键节点审批")
print("=" * 60)
