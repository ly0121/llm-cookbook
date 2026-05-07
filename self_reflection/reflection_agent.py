"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 项目 21: Self-Reflection Agent (自我反思 Agent)               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  【核心科学概念: 自我反思 (Self-Reflection)】                                  ║
║                                                                              ║
║  什么是自我反思?                                                              ║
║  ─────────────                                                               ║
║  人类写作文的过程:                                                            ║
║    写初稿 → 重读一遍 → 发现问题 → 修改 → 再检查 → 满意了 → 定稿              ║
║                                                                              ║
║  LLM 的自我反思也是同样的道理:                                                ║
║    生成回答 → 自我评估 → 发现不足 → 改进回答 → 再评估 → 满意 → 输出          ║
║                                                                              ║
║                                                                              ║
║  为什么 LLM 需要反思?                                                        ║
║  ───────────────────                                                         ║
║  1. 一次生成的质量不稳定 — 有时好有时差                                       ║
║  2. LLM 自己"知道"什么是好的回答 — 它能评判质量                              ║
║  3. 给它机会修改，质量可以大幅提升                                            ║
║  4. 类似于人类的"系统2思维" — 慢思考、深思考                                 ║
║                                                                              ║
║                                                                              ║
║  业界经典方法:                                                                ║
║  ────────────                                                                ║
║  - Reflexion (2023): Agent 从失败经验中学习，记住教训                         ║
║  - Self-Refine (2023): 生成 → 反馈 → 精炼 的迭代框架                        ║
║  - Constitutional AI: 用原则约束自我修正                                      ║
║  - Reflection in LangGraph: 把反思流程图形化                                  ║
║                                                                              ║
║                                                                              ║
║  核心流程 (ASCII 图):                                                        ║
║  ─────────────────────                                                       ║
║                                                                              ║
║      ┌──────────┐     ┌──────────┐     ┌──────────┐                         ║
║      │ Generate │────>│ Critique │────>│  Refine  │                          ║
║      │ (生成)   │     │ (评估)   │     │ (改进)   │                          ║
║      └──────────┘     └──────────┘     └────┬─────┘                         ║
║                              ^               │                               ║
║                              │               │                               ║
║                              └───────────────┘                               ║
║                            (循环直到满意为止)                                  ║
║                                                                              ║
║                                                                              ║
║  与普通 Agent 的区别:                                                        ║
║  ──────────────────                                                          ║
║  普通 Agent:  用户问 → LLM答 → 结束                                         ║
║  反思 Agent:  用户问 → LLM答 → LLM自评 → 不满意? → LLM改 → 再评 → 满意!    ║
║                                                                              ║
║  成本考量:                                                                   ║
║  ─────────                                                                   ║
║  每次反思 = 额外的 LLM 调用 = 额外的时间和费用                               ║
║  所以要设置最大迭代次数，防止无限循环                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 导入部分
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import re
import operator
from typing import Annotated, List

from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 0: 初始化 + 自我反思概念科普
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 【类比: 写作文的过程】
#
# 小学生写作文:
#   第一遍: 想到啥写啥，写完交上去 → 质量参差不齐
#
# 优秀学生写作文:
#   第一遍: 先写初稿
#   第二遍: 自己读一遍，发现"这里逻辑不通"、"那里用词不当"
#   第三遍: 针对问题修改
#   第四遍: 再检查一遍，满意了，定稿
#
# LLM 的自我反思就是让 AI 学会"优秀学生"的做法:
#   不是一次回答就完事，而是 生成 → 评估 → 改进 → 再评估 → ... → 定稿
#
# 【为什么这有效?】
# LLM 在"评价"模式下，往往比"生成"模式更准确
# 就像你写完一篇文章可能不完美，但你很容易发现别人文章里的问题
# 同一个 LLM，扮演不同角色(作者 vs 评审)，能产生自我改进的效果
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 70)
print("Chapter 0: 初始化 — 自我反思 Agent 的基础设施")
print("=" * 70)

# ── API 配置 ──────────────────────────────────────────────────────────────
API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "kivy-kimi-k2_5"

# ── 创建 LLM 实例 ────────────────────────────────────────────────────────
# temperature=0.7: 生成时需要一些创造力
# max_tokens=1024: 限制输出长度，避免超时
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    temperature=0.7,
    max_tokens=1024,
)

print(f"[init] LLM 实例已创建: model={MODEL_NAME}")
print(f"[init] base_url={BASE_URL}")
print()

# ── 创建输出解析器 ────────────────────────────────────────────────────────
parser = StrOutputParser()

print("[init] StrOutputParser 就绪")
print()
print("反思 Agent 的核心思想:")
print("  ┌──────────┐     ┌──────────┐     ┌──────────┐")
print("  │ Generate │────>│ Critique │────>│  Refine  │")
print("  │ (生成)   │     │ (评估)   │     │ (改进)   │")
print("  └──────────┘     └──────────┘     └────┬─────┘")
print("                          ^               │")
print("                          │               │")
print("                          └───────────────┘")
print("                        (循环直到满意为止)")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 1: 基础反思 — Generate + Critique 两步法
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 【两步法原理】
#
# 这是自我反思最简单的形式，只有两步:
#
#   第一步 — Generate (生成):
#     给 LLM 一个任务，让它生成初始回答
#     这就是普通的 LLM 调用，没什么特别的
#
#   第二步 — Critique (评估):
#     把初始回答喂给 LLM (可以是同一个 LLM)
#     让它从多个维度评估这个回答:
#       - 准确性: 信息是否正确?
#       - 完整性: 是否遗漏了重要内容?
#       - 逻辑性: 论述是否连贯?
#       - 清晰度: 表达是否清晰易懂?
#     输出一个分数 (1-10) 和具体的改进建议
#
# 【为什么分成两步?】
# 因为 LLM 在"评价者"角色下更客观
# 就像你让同一个人"先写作文"再"当老师批改"，
# 批改时他能发现写作时没注意到的问题
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print()
print("=" * 70)
print("Chapter 1: 基础反思 — Generate + Critique 两步法")
print("=" * 70)
print()

# ── 第一步: 定义"生成"Prompt ──────────────────────────────────────────────
# 任务: 让 LLM 写一小段关于"Python装饰器"的技术说明
# 注意: 我们故意限制字数，让第一版可能不够完善，从而体现反思的价值

generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位技术写作专家。请用简洁的中文回答，控制在150字以内。"),
        ("human", "{task}"),
    ]
)

# 构建生成链: prompt → LLM → 解析为字符串
generate_chain = generate_prompt | llm | parser

print("[Chapter 1] 第一步: 生成初始回答...")
print()

# 执行生成
task = "请简要解释Python装饰器是什么，以及它的一个实际用途。"
print(f"[任务] {task}")
print()

initial_response = generate_chain.invoke({"task": task})
print(f"[初始回答]\n{initial_response}")
print()

# ── 第二步: 定义"评估"Prompt ──────────────────────────────────────────────
# 评估 Prompt 的关键设计:
# 1. 明确告诉 LLM 它的角色是"评审专家"
# 2. 给出评估维度
# 3. 要求输出格式: 分数 + 改进建议
# 4. 用 "分数: X" 的格式方便后续解析

critique_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位严格的技术文章评审专家。
请从以下4个维度评估这段文字，每个维度1-10分:
1. 准确性 - 技术信息是否正确
2. 完整性 - 是否覆盖了关键点
3. 逻辑性 - 论述是否连贯
4. 清晰度 - 是否易于理解

请按如下格式输出(务必严格遵守):
准确性: X分
完整性: X分
逻辑性: X分
清晰度: X分
总分数: X
改进建议: (简要列出1-2条最重要的改进建议)""",
        ),
        ("human", "请评估以下回答:\n\n任务: {task}\n\n回答: {response}"),
    ]
)

# 构建评估链
critique_chain = critique_prompt | llm | parser

print("[Chapter 1] 第二步: 评估初始回答...")
print()

# 执行评估
critique_result = critique_chain.invoke(
    {
        "task": task,
        "response": initial_response,
    }
)

print(f"[评估结果]\n{critique_result}")
print()

# ── 解析分数 ──────────────────────────────────────────────────────────────
# 从评估结果中提取"总分数"
# 使用正则表达式匹配 "总分数: X" 或 "总分数: X分" 的模式


def extract_score(critique_text):
    """从评估文本中提取总分数，返回整数"""
    # 尝试匹配 "总分数: 8" 或 "总分数: 8分" 或 "总分数：8"
    patterns = [
        r"总分数[：:]\s*(\d+)",
        r"总分[：:]\s*(\d+)",
        r"综合[分得][：:]\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, critique_text)
        if match:
            return int(match.group(1))
    # 如果都没匹配到，返回一个保守的默认值
    print("[警告] 无法从评估结果中提取分数，使用默认值 6")
    return 6


score = extract_score(critique_result)
print(f"[解析] 提取到的总分数: {score}/10")
print()
print("[Chapter 1 总结]")
print(f"  初始回答长度: {len(initial_response)} 字")
print(f"  评估分数: {score}/10")
print("  这就是最基础的两步反思: 生成 + 评估")
print('  接下来 Chapter 2 会加入"修改"步骤，形成完整的迭代改进循环')
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 2: Iterative Refinement — 迭代改进
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 【迭代改进的核心思想】
#
# Chapter 1 只有 生成 + 评估，但没有"改"的动作
# Chapter 2 加入"修改"步骤，形成完整循环:
#
#   生成初稿
#      ↓
#   评估 (打分 + 建议)
#      ↓
#   分数 >= 8? ─── 是 ──→ 输出最终版本 ✓
#      │
#      否
#      ↓
#   根据建议修改
#      ↓
#   回到"评估"步骤 (循环)
#
# 【关键设计】
# - max_iterations = 3: 防止无限循环 (即使分数始终不达标，最多改3次就停)
# - 阈值分数 = 8: 8分以上认为质量足够好
# - 每轮都传入上一版本 + 评估建议，让 LLM 有针对性地改进
#
# 【为什么需要限制迭代次数?】
# 1. 每次迭代 = 2次 LLM 调用 (评估 + 修改)，成本会累积
# 2. 实践中，2-3次迭代通常能获得大部分改进
# 3. 超过3次，改进幅度递减，但成本线性增长
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print()
print("=" * 70)
print("Chapter 2: Iterative Refinement — 迭代改进")
print("=" * 70)
print()

# ── 定义"修改"Prompt ──────────────────────────────────────────────────────
# 修改 Prompt 的关键: 同时传入原文 + 评估建议
# 让 LLM 知道"哪里不好"，才能有针对性地改进

refine_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位技术写作专家。请根据评审意见改进以下文章。
要求:
1. 保留原文的优点
2. 针对评审意见中指出的问题进行改进
3. 控制在200字以内
4. 直接输出改进后的文章，不要输出其他内容""",
        ),
        ("human", "原文:\n{draft}\n\n评审意见:\n{critique}\n\n请输出改进后的版本:"),
    ]
)

# 构建修改链
refine_chain = refine_prompt | llm | parser

# ── 迭代改进循环 ──────────────────────────────────────────────────────────
MAX_ITERATIONS = 3  # 最大迭代次数
THRESHOLD_SCORE = 8  # 分数阈值: >=8 认为质量达标

print(f"[配置] 最大迭代次数: {MAX_ITERATIONS}")
print(f"[配置] 质量阈值分数: {THRESHOLD_SCORE}/10")
print(f"[配置] 任务: {task}")
print()

# 使用 Chapter 1 的初始回答作为起点
current_draft = initial_response
iteration_history = []  # 记录每轮的版本和分数

print(f"[第0版 - 初始稿]\n{current_draft}")
print()

for iteration in range(1, MAX_ITERATIONS + 1):
    print(f"--- 第 {iteration} 轮迭代 ---")
    print()

    # 步骤1: 评估当前版本
    print(f"[轮次{iteration}] 评估中...")
    critique_text = critique_chain.invoke(
        {
            "task": task,
            "response": current_draft,
        }
    )
    current_score = extract_score(critique_text)

    print(f"[轮次{iteration}] 评估分数: {current_score}/10")
    print(f"[轮次{iteration}] 评审意见:\n{critique_text}")
    print()

    # 记录历史
    iteration_history.append(
        {
            "iteration": iteration,
            "score": current_score,
            "draft": current_draft,
        }
    )

    # 步骤2: 检查是否达标
    if current_score >= THRESHOLD_SCORE:
        print(
            f"[轮次{iteration}] 分数 {current_score} >= {THRESHOLD_SCORE}，质量达标! 停止迭代。"
        )
        break

    # 步骤3: 未达标，执行修改
    print(f"[轮次{iteration}] 分数 {current_score} < {THRESHOLD_SCORE}，需要改进...")
    print()

    current_draft = refine_chain.invoke(
        {
            "draft": current_draft,
            "critique": critique_text,
        }
    )

    print(f"[轮次{iteration}] 修改后版本:\n{current_draft}")
    print()

else:
    # 如果循环正常结束(没有break)，说明达到最大迭代次数
    print(f"[注意] 已达到最大迭代次数 {MAX_ITERATIONS}，停止迭代。")

# ── 展示进化过程 ──────────────────────────────────────────────────────────
print()
print("=" * 50)
print("迭代改进总结 — 文章质量进化过程:")
print("=" * 50)
print(f"  初始版本 → 分数: (未评估，直接进入迭代)")
for record in iteration_history:
    print(f"  第{record['iteration']}轮评估 → 分数: {record['score']}/10")
print(f"  最终版本字数: {len(current_draft)}")
print()
print(f"[最终版本]\n{current_draft}")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 3: LangGraph 实现 Reflection Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 【为什么要用 LangGraph?】
#
# Chapter 2 用 Python for 循环实现了迭代改进
# 但这种方式有几个问题:
#   1. 流程不够直观 — 看代码才能理解逻辑
#   2. 不易扩展 — 想加新节点(如"人工审核")很麻烦
#   3. 状态管理混乱 — 变量散落在各处
#
# LangGraph 把反思流程变成一张"图":
#   - 节点 (Node): 每个处理步骤 (生成、评估、修改)
#   - 边 (Edge): 步骤之间的连接
#   - 条件边 (Conditional Edge): 根据条件决定走哪条路
#   - 状态 (State): 统一管理所有数据
#
# 【LangGraph 版本的流程图】
#
#   START
#     │
#     ▼
#   ┌──────────┐
#   │ generate │  生成初始回答
#   └────┬─────┘
#        │
#        ▼
#   ┌──────────┐
#   │ evaluate │  评估当前回答
#   └────┬─────┘
#        │
#        ▼
#   ┌─────────────────┐     score >= 8
#   │ should_continue? │────────────────→ END
#   └────┬────────────┘
#        │ score < 8
#        ▼
#   ┌──────────┐
#   │  refine  │  根据评估改进
#   └────┬─────┘
#        │
#        └──→ evaluate (回到评估)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print()
print("=" * 70)
print("Chapter 3: LangGraph 实现 Reflection Agent")
print("=" * 70)
print()

# ── 第一步: 定义 State (状态) ─────────────────────────────────────────────
# TypedDict 定义了图中流转的数据结构
# 所有节点共享这个 State，读取和更新其中的字段


class ReflectionState(TypedDict):
    task: str  # 用户的原始任务
    draft: str  # 当前草稿
    critique: str  # 最新的评估意见
    score: int  # 最新的评估分数
    iteration: int  # 当前迭代轮次
    max_iterations: int  # 最大迭代次数
    history: List[str]  # 每轮草稿的历史记录


print("[State 定义]")
print("  task: 用户任务")
print("  draft: 当前草稿")
print("  critique: 评估意见")
print("  score: 评估分数")
print("  iteration: 当前轮次")
print("  max_iterations: 最大轮次")
print("  history: 历史版本列表")
print()

# ── 第二步: 定义节点函数 ──────────────────────────────────────────────────
# 每个节点函数接收 State，返回更新后的字段 (部分更新)


def generate_node(state: ReflectionState) -> dict:
    """生成节点: 根据任务生成初始回答"""
    print("  [Node: generate] 正在生成初始回答...")

    result = generate_chain.invoke({"task": state["task"]})

    print(f"  [Node: generate] 生成完成，长度: {len(result)} 字")
    return {
        "draft": result,
        "iteration": 1,
        "history": [result],
    }


def evaluate_node(state: ReflectionState) -> dict:
    """评估节点: 对当前草稿打分并给出建议"""
    print(f"  [Node: evaluate] 正在评估第 {state['iteration']} 版草稿...")

    critique_text = critique_chain.invoke(
        {
            "task": state["task"],
            "response": state["draft"],
        }
    )

    score = extract_score(critique_text)
    print(f"  [Node: evaluate] 评估完成，分数: {score}/10")

    return {
        "critique": critique_text,
        "score": score,
    }


def refine_node(state: ReflectionState) -> dict:
    """修改节点: 根据评估意见改进草稿"""
    print(f"  [Node: refine] 正在根据建议改进第 {state['iteration']} 版...")

    new_draft = refine_chain.invoke(
        {
            "draft": state["draft"],
            "critique": state["critique"],
        }
    )

    new_iteration = state["iteration"] + 1
    new_history = state["history"] + [new_draft]

    print(f"  [Node: refine] 改进完成，进入第 {new_iteration} 版")

    return {
        "draft": new_draft,
        "iteration": new_iteration,
        "history": new_history,
    }


# ── 第三步: 定义条件路由函数 ──────────────────────────────────────────────
# 这个函数决定评估后走哪条边: 结束 or 继续修改


def should_continue(state: ReflectionState) -> str:
    """条件路由: 决定是结束还是继续迭代"""
    score = state["score"]
    iteration = state["iteration"]
    max_iter = state["max_iterations"]

    if score >= THRESHOLD_SCORE:
        print(f"  [Router] 分数 {score} >= {THRESHOLD_SCORE}，质量达标 → END")
        return "end"
    elif iteration >= max_iter:
        print(f"  [Router] 已达最大迭代 {iteration}/{max_iter} → END")
        return "end"
    else:
        print(
            f"  [Router] 分数 {score} < {THRESHOLD_SCORE}，轮次 {iteration}/{max_iter} → refine"
        )
        return "refine"


# ── 第四步: 构建 StateGraph ───────────────────────────────────────────────
print("[构建 LangGraph 状态图]")
print()

# 创建图
workflow = StateGraph(ReflectionState)

# 添加节点
workflow.add_node("generate", generate_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("refine", refine_node)

print("  节点已添加: generate, evaluate, refine")

# 添加边
workflow.add_edge(START, "generate")  # 入口 → 生成
workflow.add_edge("generate", "evaluate")  # 生成 → 评估
workflow.add_edge("refine", "evaluate")  # 修改 → 评估 (形成循环)

# 添加条件边: 评估之后根据分数决定走向
workflow.add_conditional_edges(
    "evaluate",  # 从 evaluate 节点出发
    should_continue,  # 用这个函数判断
    {
        "end": END,  # 返回 'end' → 结束
        "refine": "refine",  # 返回 'refine' → 进入修改节点
    },
)

print("  边已添加: START→generate→evaluate→(条件)→refine/END")
print("  条件边: score>=8 或 达到最大轮次 → END")
print("  条件边: score<8 且 未达最大轮次 → refine → evaluate (循环)")
print()

# 编译图
graph = workflow.compile()
print("[图已编译] 准备执行...")
print()

# ── 第五步: 执行图 ────────────────────────────────────────────────────────
print("[执行 LangGraph 反思 Agent]")
print("-" * 50)

# 准备初始状态
initial_state = {
    "task": "请简要解释什么是RESTful API，以及它的核心设计原则。",
    "draft": "",
    "critique": "",
    "score": 0,
    "iteration": 0,
    "max_iterations": 3,
    "history": [],
}

print(f"任务: {initial_state['task']}")
print()

# 执行图
final_state = graph.invoke(initial_state)

# 展示结果
print()
print("-" * 50)
print("[LangGraph 执行完成]")
print(f"  总迭代轮次: {final_state['iteration']}")
print(f"  最终分数: {final_state['score']}/10")
print(f"  历史版本数: {len(final_state['history'])}")
print()
print(f"[最终输出]\n{final_state['draft']}")
print()

# 打印版本对比
if len(final_state["history"]) > 1:
    print("[版本进化对比]")
    for i, version in enumerate(final_state["history"]):
        label = "初始版" if i == 0 else f"第{i}次改进"
        print(
            f"  [{label}] {version[:80]}..."
            if len(version) > 80
            else f"  [{label}] {version}"
        )
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chapter 4: 实战 — 代码生成 + 自我反思修复
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 【场景说明】
#
# 自我反思最实用的场景之一: 代码生成 + 自动审查
#
# 普通代码生成:
#   用户: "写个斐波那契函数"
#   LLM: 写出代码 → 交付 (可能有 bug、缺少边界检查)
#
# 反思式代码生成:
#   用户: "写个斐波那契函数"
#   LLM (作者角色): 写出初版代码
#   LLM (审查角色): 检查边界条件、异常处理、命名规范、注释
#   LLM (作者角色): 根据审查意见修改
#   LLM (审查角色): 再次检查 → 满意 → 交付
#
# 【代码审查的维度】
# 1. 边界条件: 空输入、负数、超大数值
# 2. 异常处理: 类型错误、值错误
# 3. 命名规范: 变量名是否有意义
# 4. 注释完整性: 函数文档、关键逻辑注释
# 5. 代码风格: PEP 8 规范
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print()
print("=" * 70)
print("Chapter 4: 实战 — 代码生成 + 自我反思修复")
print("=" * 70)
print()

# ── 定义代码生成 Prompt ───────────────────────────────────────────────────
code_generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位Python开发者。请根据需求编写代码。
要求: 直接输出Python代码，不要加markdown代码块标记，不要多余解释。代码要简洁。""",
        ),
        ("human", "{task}"),
    ]
)

code_generate_chain = code_generate_prompt | llm | parser

# ── 定义代码审查 Prompt ───────────────────────────────────────────────────
# 这个 Prompt 是反思的核心 — 模拟资深开发者的 Code Review
code_review_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位资深Python代码审查专家。请从以下维度审查代码:
1. 边界条件 - 是否处理了空输入、负数、异常类型等边界情况
2. 异常处理 - 是否有适当的try/except或参数验证
3. 命名规范 - 变量名和函数名是否清晰有意义
4. 注释完整性 - 是否有docstring和关键注释
5. 代码风格 - 是否符合PEP 8

请按如下格式输出:
边界条件: X分 (1-10)
异常处理: X分 (1-10)
命名规范: X分 (1-10)
注释完整性: X分 (1-10)
代码风格: X分 (1-10)
总分数: X
改进建议: (列出最重要的2-3条改进建议)""",
        ),
        ("human", "需求: {task}\n\n代码:\n{code}"),
    ]
)

code_review_chain = code_review_prompt | llm | parser

# ── 定义代码修改 Prompt ───────────────────────────────────────────────────
code_refine_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位Python开发者。请根据代码审查意见改进代码。
要求:
1. 保留原有功能
2. 针对审查意见进行改进
3. 直接输出改进后的Python代码，不要加markdown代码块标记
4. 不要输出多余解释""",
        ),
        ("human", "原始代码:\n{code}\n\n审查意见:\n{critique}\n\n请输出改进后的代码:"),
    ]
)

code_refine_chain = code_refine_prompt | llm | parser

# ── 执行代码生成 + 反思循环 ───────────────────────────────────────────────
code_task = "写一个计算斐波那契数列第n项的Python函数fibonacci(n)，要求健壮可靠。"

print(f"[代码任务] {code_task}")
print()

# 第一步: 生成初始代码
print("[Step 1] 生成初始代码...")
print()

current_code = code_generate_chain.invoke({"task": code_task})
print(f"[初始代码 - 第1版]\n{current_code}")
print()

# 迭代审查 + 修改
CODE_MAX_ITERATIONS = 3
CODE_THRESHOLD = 8

for code_iter in range(1, CODE_MAX_ITERATIONS + 1):
    print(f"--- 代码审查第 {code_iter} 轮 ---")
    print()

    # 审查
    print(f"[轮次{code_iter}] 代码审查中...")
    review_result = code_review_chain.invoke(
        {
            "task": code_task,
            "code": current_code,
        }
    )
    code_score = extract_score(review_result)

    print(f"[轮次{code_iter}] 审查分数: {code_score}/10")
    print(f"[轮次{code_iter}] 审查意见:\n{review_result}")
    print()

    # 检查是否达标
    if code_score >= CODE_THRESHOLD:
        print(
            f"[轮次{code_iter}] 代码质量达标 (分数 {code_score} >= {CODE_THRESHOLD})，停止迭代。"
        )
        break

    # 修改
    print(f"[轮次{code_iter}] 代码需要改进，根据审查意见修改中...")
    current_code = code_refine_chain.invoke(
        {
            "code": current_code,
            "critique": review_result,
        }
    )
    print(f"[轮次{code_iter}] 改进后代码 - 第{code_iter + 1}版:\n{current_code}")
    print()

print()
print("=" * 50)
print("[最终代码]")
print("=" * 50)
print(current_code)
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary: 自我反思 Agent 总结
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print()
print("=" * 70)
print("Summary: 自我反思 Agent 知识总结")
print("=" * 70)
print()
print("【何时使用反思?】")
print("  适合:")
print("  - 需要高质量输出的任务 (写作、代码生成、方案设计)")
print("  - 一次生成质量不稳定的场景")
print("  - 有明确评估标准的任务 (可以量化打分)")
print()
print("  不适合:")
print("  - 简单问答 (一次就能答好)")
print("  - 实时性要求高的场景 (反思需要多次调用，耗时长)")
print("  - 没有明确评估标准的开放性任务")
print()
print("【反思的成本】")
print("  - 每轮反思 = 2次额外 LLM 调用 (评估 + 修改)")
print("  - 3轮迭代 = 1(生成) + 6(3轮x2次) = 7次 LLM 调用")
print("  - 时间: 是单次调用的 5-7 倍")
print("  - 费用: 是单次调用的 5-7 倍")
print()
print("【与 Human-in-the-Loop (HITL) 的结合】")
print("  - 自动反思: 全自动，适合批量处理")
print('  - HITL 反思: 人类代替"评估节点"，质量更高但效率更低')
print("  - 混合模式: 自动反思到一定分数 → 人类最终审核")
print()
print("【本项目的 4 个 Chapter 回顾】")
print("  Ch1: 两步法 (生成 + 评估) — 最简反思")
print("  Ch2: 迭代改进 (生成 → 评估 → 修改 → 循环) — 完整反思")
print("  Ch3: LangGraph 实现 — 工程化反思")
print("  Ch4: 代码审查实战 — 反思的实际应用")
print()
print('反思是让 LLM 从"一次性输出"进化到"迭代优化"的关键技术!')
print()
print("=" * 70)
print("项目 21 全部完成!")
print("=" * 70)
