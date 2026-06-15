"""
╔══════════════════════════════════════════════════════════════════╗
║         项目九：RAG 评估体系 — 用指标量化你的 AI 到底答得好不好      ║
║         忠实度 + 相关性 + 答案质量 = 自动化质量门禁               ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普一：为什么"感觉还不错"不够？——评估的必要性】
═══════════════════════════════════════════════════════════════════

没有评估的 RAG 开发，就像没有考试的学校：

  ┌─────────────────────────────────────────────────────────────┐
  │  没有评估时的开发流程（盲人摸象）：                           │
  │                                                             │
  │  改了 chunk_size → "试了几个问题，感觉还行" → 上线          │
  │  换了 Embedding 模型 → "好像回答更准了？" → 上线             │
  │  调了 Prompt → "看起来没变差" → 上线                         │
  │                                                             │
  │  结果：上线后用户投诉"回答不准"，但你不知道哪里退步了！      │
  └─────────────────────────────────────────────────────────────┘

  有评估时的开发流程（科学实验）：
  ┌─────────────────────────────────────────────────────────────┐
  │  准备评估集（50个标准问答对）                                 │
  │  ↓                                                          │
  │  改了 chunk_size → 跑评估 → 忠实度 82%→85% ✅ 上线          │
  │  换了 Embedding   → 跑评估 → 相关性 78%→72% ❌ 回滚          │
  │  调了 Prompt      → 跑评估 → 整体 +3% ✅ 上线               │
  │                                                             │
  │  每次改动都有数字说话，退步立刻能发现！                      │
  └─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
【前置科普二：RAG 评估的三大核心指标】
═══════════════════════════════════════════════════════════════════

用一个考试的比喻来理解：

  ┌─────────────────────────────────────────────────────────────┐
  │  指标一：检索相关性（Context Relevance）                      │
  │  比喻：考试时翻到的参考书页面对不对？                        │
  │                                                             │
  │  问："公司2024年营收？"                                      │
  │  检索到了财务数据页 → 相关性高 ✅                            │
  │  检索到了员工手册页 → 相关性低 ❌                            │
  │                                                             │
  │  评估方法：判断检索到的文档块和问题是否相关                  │
  ├─────────────────────────────────────────────────────────────┤
  │  指标二：忠实度（Faithfulness / Groundedness）               │
  │  比喻：回答是不是"照着书抄的"？有没有自己编？               │
  │                                                             │
  │  参考资料说"营收28亿"，回答也说"28亿" → 忠实 ✅            │
  │  参考资料说"营收28亿"，回答说"50亿"   → 幻觉 ❌             │
  │                                                             │
  │  评估方法：回答中的每个事实是否都能在检索文档中找到依据      │
  ├─────────────────────────────────────────────────────────────┤
  │  指标三：答案正确性（Answer Correctness）                     │
  │  比喻：最终答案和标准答案对不对得上？                        │
  │                                                             │
  │  标准答案："营收28.3亿"                                      │
  │  RAG回答："营收约28亿"  → 基本正确 ✅                       │
  │  RAG回答："营收15亿"    → 错误 ❌                            │
  │                                                             │
  │  评估方法：将 RAG 回答与人工标注的标准答案对比               │
  └─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
【前置科普三：LLM-as-Judge（用大模型当评委）】
═══════════════════════════════════════════════════════════════════

传统评估方法（BLEU、ROUGE）只看"字面重合度"，
不理解语义（"28.3亿"和"约28亿"字面不同但语义相同）。

现代做法：让另一个 LLM 当"阅卷老师"：
  ① 给 LLM 评分标准（rubric）
  ② 给 LLM 参考答案 + 学生回答
  ③ LLM 打分（1-5 分）+ 给出理由

这就是 "LLM-as-Judge" 模式，
LangSmith、RAGAS、DeepEval 等工具的底层都是这个原理。
本项目用纯 LangChain 手动实现，让你理解底层机制。
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
import json


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化 + 构建一个待评估的 RAG 系统
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 0 章：初始化 + 构建待评估的 RAG 系统")
print("=" * 60)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY, BASE_URL, MODEL_NAME
# RAG 用的 LLM（temperature=0，精确回答）
llm = ChatOpenAI(model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, temperature=0.0)

# 评估用的 LLM（同一个模型，但作为"评委"角色）
# 生产环境中建议用更强的模型当评委（如 GPT-4 评估 GPT-3.5 的输出）
eval_llm = ChatOpenAI(
    model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, temperature=0.0
)

# Embedding 模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

print("✅ LLM 和 Embeddings 初始化完成")
print()

# ── 构建一个简单的 RAG 知识库（复用项目五的数据子集）─────

RAG_DOCUMENTS = [
    Document(
        page_content="智驾科技2024年总营收28.3亿元人民币，同比增长87%。净利润2.1亿元，首次实现全年盈利。研发投入9.8亿元，占营收34.6%。",
        metadata={"page": 3, "section": "财务数据"},
    ),
    Document(
        page_content="公司拥有员工3200人，其中研发人员占比65%。截至2024年底，已获得自动驾驶相关专利487项，在12个城市开展L4级自动驾驶测试。",
        metadata={"page": 1, "section": "公司概况"},
    ),
    Document(
        page_content="2024年纯视觉感知系统识别准确率达97.3%，恶劣天气感知能力提升40%。与Waymo对比：日间准确率差距从5%缩小至0.8%。",
        metadata={"page": 5, "section": "技术研发"},
    ),
    Document(
        page_content="2025年目标营收50亿元，同比增长77%。计划新增5个城市Robotaxi运营，Q2在新加坡启动路测，Q4在沙特利雅得商业化试运营。",
        metadata={"page": 8, "section": "战略规划"},
    ),
    Document(
        page_content="主要竞争对手包括百度Apollo、小马智行、文远知行、华为ADS。智驾科技差异化优势：纯视觉方案成本低60%，端到端架构效率高35%。",
        metadata={"page": 7, "section": "市场竞争"},
    ),
]

# 向量化并建索引
vectorstore = FAISS.from_documents(RAG_DOCUMENTS, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# RAG 链
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """根据以下参考资料回答问题。只使用参考资料中的信息，不要编造。
如果资料中没有相关信息，说"根据现有资料无法回答"。

参考资料：
{context}""",
        ),
        ("human", "{question}"),
    ]
)

parser = StrOutputParser()

print("✅ RAG 系统构建完成（5个文档块，k=2检索）")
print()


def run_rag(question: str) -> dict:
    """执行 RAG 问答，返回问题、检索文档、回答"""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = rag_prompt | llm | parser
    answer = chain.invoke({"context": context, "question": question})
    return {
        "question": question,
        "contexts": [doc.page_content for doc in docs],
        "answer": answer,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：定义评估数据集（Ground Truth）
# 目标：准备"标准答案"，让评估有据可依
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：定义评估数据集（Ground Truth）")
print("=" * 60)
print()

# ── 评估数据集的构成 ─────────────────────────────────────
#
# 一条评估数据包含：
#   question       → 测试问题
#   ground_truth   → 人工标注的标准答案（"参考答案"）
#   contexts       → 理想情况下应该检索到的文档（可选）
#
# ⚠️ 避坑指南：评估集的质量决定评估的价值！
#   ① 覆盖多种问题类型（事实型、对比型、推理型）
#   ② 标准答案要精确（数字、名称不能含糊）
#   ③ 数量建议：至少 30-50 条，覆盖核心业务场景
#   ④ 定期更新：知识库变了，评估集也要跟着变

EVAL_DATASET = [
    {
        "question": "智驾科技2024年的总营收是多少？",
        "ground_truth": "28.3亿元人民币",
        "eval_type": "事实提取",
    },
    {
        "question": "公司2024年的净利润是多少？同比增长情况如何？",
        "ground_truth": "净利润2.1亿元，首次实现全年盈利",
        "eval_type": "事实提取",
    },
    {
        "question": "公司研发人员占比多少？有多少专利？",
        "ground_truth": "研发人员占比65%，拥有487项自动驾驶相关专利",
        "eval_type": "多事实提取",
    },
    {
        "question": "感知系统和Waymo的差距是多少？",
        "ground_truth": "日间准确率差距从5%缩小至0.8%",
        "eval_type": "对比型",
    },
    {
        "question": "2025年的海外计划有哪些？",
        "ground_truth": "Q2在新加坡启动路测，Q4在沙特利雅得商业化试运营",
        "eval_type": "规划型",
    },
    {
        "question": "公司相比竞争对手的核心优势是什么？",
        "ground_truth": "纯视觉方案成本低60%，端到端架构效率高35%",
        "eval_type": "分析型",
    },
]

print(f"  ✅ 评估数据集准备完成：{len(EVAL_DATASET)} 条测试用例")
print()
print("  【评估集概览】")
print("  ┌────┬──────────────────────────────────┬──────────┐")
print("  │ #  │ 问题                              │ 类型     │")
print("  ├────┼──────────────────────────────────┼──────────┤")
for i, item in enumerate(EVAL_DATASET, 1):
    q = item["question"][:30]
    print(f"  │ {i:2d} │ {q:32s} │ {item['eval_type']:8s} │")
print("  └────┴──────────────────────────────────┴──────────┘")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：实现三大评估指标（LLM-as-Judge）
# 目标：用大模型对 RAG 的回答进行自动打分
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：实现三大评估指标（LLM-as-Judge）")
print("=" * 60)
print()

# ── 评估结果的结构化输出 ─────────────────────────────────
#
# 用 Pydantic 定义评分结果（和项目六的 Structured Output 呼应！）
# 这保证评委 LLM 返回的是可解析的结构化评分，而不是自由文本。


class EvalScore(BaseModel):
    """单个评估指标的结果"""

    score: int = Field(description="评分，1-5分（1=最差，5=最好）")
    reasoning: str = Field(description="评分理由，一句话解释为什么给这个分数")


# ── 指标一：检索相关性 ───────────────────────────────────

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位严格的检索质量评估专家。
请评估：给定用户的问题，检索到的文档块是否相关。

评分标准（1-5分）：
5分：检索到的文档完全包含回答问题所需的信息
4分：检索到的文档大部分相关，包含关键信息
3分：检索到的文档部分相关，但缺少一些关键信息
2分：检索到的文档与问题关系不大
1分：检索到的文档完全不相关""",
        ),
        (
            "human",
            """用户问题：{question}

检索到的文档内容：
{contexts}

请评分：""",
        ),
    ]
)


# ── 指标二：忠实度（Faithfulness）────────────────────────

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位严格的事实核查专家。
请评估：AI的回答是否忠实于提供的参考资料？是否有编造/幻觉？

评分标准（1-5分）：
5分：回答中所有事实都能在参考资料中找到依据，零幻觉
4分：回答基本忠实，有极少量推断但合理
3分：回答部分忠实，有一些无法验证的内容
2分：回答有明显的编造内容，但也有正确部分
1分：回答大量编造，严重偏离参考资料""",
        ),
        (
            "human",
            """参考资料：
{contexts}

AI的回答：
{answer}

请评估回答的忠实度：""",
        ),
    ]
)


# ── 指标三：答案正确性 ───────────────────────────────────

CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一位严格的答案质量评估专家。
请评估：AI的回答与标准答案相比，正确性如何？

评分标准（1-5分）：
5分：完全正确，包含标准答案的所有关键信息
4分：基本正确，覆盖了大部分关键信息
3分：部分正确，有遗漏但无明显错误
2分：有错误信息，或关键信息严重遗漏
1分：完全错误或答非所问

注意：表述方式不同但含义相同应视为正确（如"28.3亿"和"约28亿"）。""",
        ),
        (
            "human",
            """问题：{question}
标准答案：{ground_truth}
AI的回答：{answer}

请评分：""",
        ),
    ]
)


# ── 构建评估链（结构化输出）────────────────────────────────

eval_structured = eval_llm.with_structured_output(EvalScore)


def evaluate_relevance(question: str, contexts: list[str]) -> EvalScore:
    """评估检索相关性"""
    chain = RELEVANCE_PROMPT | eval_structured
    return chain.invoke(
        {
            "question": question,
            "contexts": "\n\n".join(contexts),
        }
    )


def evaluate_faithfulness(contexts: list[str], answer: str) -> EvalScore:
    """评估忠实度"""
    chain = FAITHFULNESS_PROMPT | eval_structured
    return chain.invoke(
        {
            "contexts": "\n\n".join(contexts),
            "answer": answer,
        }
    )


def evaluate_correctness(question: str, ground_truth: str, answer: str) -> EvalScore:
    """评估答案正确性"""
    chain = CORRECTNESS_PROMPT | eval_structured
    return chain.invoke(
        {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
        }
    )


print("  ✅ 三大评估指标定义完成：")
print("     • evaluate_relevance()    → 检索相关性（1-5）")
print("     • evaluate_faithfulness() → 忠实度（1-5）")
print("     • evaluate_correctness()  → 答案正确性（1-5）")
print()
print("  💡 底层原理：LLM-as-Judge")
print("     用一个 LLM 当'阅卷老师'，按 rubric 评分标准打分。")
print("     通过 with_structured_output 保证返回可解析的结构化评分。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：运行完整评估流程
# 目标：对所有测试用例跑 RAG + 评估，输出评估报告
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：运行完整评估流程")
print("=" * 60)
print()
print("  ⏳ 对每个测试用例执行：RAG问答 → 三维度评分...")
print()

# 存储所有评估结果
eval_results = []

for i, item in enumerate(EVAL_DATASET, 1):
    question = item["question"]
    ground_truth = item["ground_truth"]

    print(f"  ━━━ 用例 #{i}/{len(EVAL_DATASET)} ━━━")
    print(f"  ❓ 问题：{question}")

    # 步骤1：运行 RAG 问答
    rag_result = run_rag(question)
    answer = rag_result["answer"]
    contexts = rag_result["contexts"]

    print(f"  🤖 RAG回答：{answer[:60]}...")
    print(f"  📖 标准答案：{ground_truth}")

    # 步骤2：三维度评估
    relevance = evaluate_relevance(question, contexts)
    faithfulness = evaluate_faithfulness(contexts, answer)
    correctness = evaluate_correctness(question, ground_truth, answer)

    # 记录结果
    result = {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "relevance": {"score": relevance.score, "reason": relevance.reasoning},
        "faithfulness": {"score": faithfulness.score, "reason": faithfulness.reasoning},
        "correctness": {"score": correctness.score, "reason": correctness.reasoning},
    }
    eval_results.append(result)

    print(
        f"  📊 评分：相关性={relevance.score}/5  忠实度={faithfulness.score}/5  正确性={correctness.score}/5"
    )
    print()

print("  ✅ 全部评估完成！")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：评估报告生成 + 指标分析
# 目标：汇总统计，生成可读的评估报告
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 4 章：评估报告")
print("=" * 60)
print()

# ── 计算统计指标 ─────────────────────────────────────────

relevance_scores = [r["relevance"]["score"] for r in eval_results]
faithfulness_scores = [r["faithfulness"]["score"] for r in eval_results]
correctness_scores = [r["correctness"]["score"] for r in eval_results]

avg_relevance = sum(relevance_scores) / len(relevance_scores)
avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
avg_correctness = sum(correctness_scores) / len(correctness_scores)
overall_avg = (avg_relevance + avg_faithfulness + avg_correctness) / 3

# ── 打印评估报告 ─────────────────────────────────────────

print("╔══════════════════════════════════════════════════════════╗")
print("║                   RAG 系统评估报告                       ║")
print("╠══════════════════════════════════════════════════════════╣")
print(f"║  评估样本数：{len(eval_results)} 条")
print(f"║  RAG 配置：k=2, chunk_size=无切块（直接文档块）")
print(f"║  模型：{MODEL_NAME}")
print("╠══════════════════════════════════════════════════════════╣")
print("║                    综合评分                              ║")
print("╠══════════════════════════════════════════════════════════╣")


# 用 ASCII 柱状图可视化
def bar(score, max_score=5, width=20):
    filled = int(score / max_score * width)
    return "█" * filled + "░" * (width - filled)


print(f"║  检索相关性  {bar(avg_relevance)}  {avg_relevance:.2f}/5")
print(f"║  忠实度      {bar(avg_faithfulness)}  {avg_faithfulness:.2f}/5")
print(f"║  答案正确性  {bar(avg_correctness)}  {avg_correctness:.2f}/5")
print(f"║  {'─' * 52}")
print(f"║  综合均分    {bar(overall_avg)}  {overall_avg:.2f}/5")
print("╠══════════════════════════════════════════════════════════╣")
print("║                    逐题详情                              ║")
print("╠══════════════════════════════════════════════════════════╣")

for i, r in enumerate(eval_results, 1):
    q_short = r["question"][:28]
    rel = r["relevance"]["score"]
    fai = r["faithfulness"]["score"]
    cor = r["correctness"]["score"]
    avg = (rel + fai + cor) / 3
    status = "✅" if avg >= 4 else "⚠️" if avg >= 3 else "❌"
    print(f"║  {status} #{i} {q_short:28s}  {rel} | {fai} | {cor}  均{avg:.1f}")

print("╠══════════════════════════════════════════════════════════╣")
print("║                    评分理由摘要                          ║")
print("╠══════════════════════════════════════════════════════════╣")

# 找出得分最低的案例，展示评分理由
lowest = min(
    eval_results,
    key=lambda x: (
        x["relevance"]["score"] + x["faithfulness"]["score"] + x["correctness"]["score"]
    ),
)
print(f"║  📉 最低分案例：{lowest['question'][:40]}")
print(f"║     相关性理由：{lowest['relevance']['reason'][:45]}")
print(f"║     忠实度理由：{lowest['faithfulness']['reason'][:45]}")
print(f"║     正确性理由：{lowest['correctness']['reason'][:45]}")

print("╚══════════════════════════════════════════════════════════╝")
print()

# ── 质量门禁判断 ─────────────────────────────────────────
#
# 生产环境中，评估分数可以作为"门禁"：
# 分数低于阈值 → 阻止部署 / 发出告警

THRESHOLD = 3.5
print(f"  🚦 质量门禁（阈值：{THRESHOLD}/5）：")
if overall_avg >= THRESHOLD:
    print(f"     ✅ 通过！综合均分 {overall_avg:.2f} ≥ {THRESHOLD}")
    print("     → 可以部署上线")
else:
    print(f"     ❌ 未通过！综合均分 {overall_avg:.2f} < {THRESHOLD}")
    print("     → 需要优化后重新评估")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 章：A/B 对比评估（改了配置后效果变好了还是变差了？）
# 目标：演示如何对比两种 RAG 配置的评估结果
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 5 章：A/B 对比评估")
print("=" * 60)
print()

# ── 为什么需要 A/B 对比？──────────────────────────────────
#
# 场景：你想把 k=2 改成 k=3（检索更多文档块）
# 问题："改了之后效果变好了吗？"
#
# 方法：
#   配置 A（k=2）→ 跑评估 → 记录分数
#   配置 B（k=3）→ 跑评估 → 记录分数
#   对比两组分数 → 量化差异

# 配置 B：k=3 检索器
retriever_b = vectorstore.as_retriever(search_kwargs={"k": 3})


def run_rag_b(question: str) -> dict:
    """配置 B 的 RAG"""
    docs = retriever_b.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    chain = rag_prompt | llm | parser
    answer = chain.invoke({"context": context, "question": question})
    return {
        "question": question,
        "contexts": [doc.page_content for doc in docs],
        "answer": answer,
    }


print("  ⏳ 对比实验：配置A (k=2) vs 配置B (k=3)")
print()

# 只跑前 3 条用例做对比（节省时间）
ab_results = []
for item in EVAL_DATASET[:3]:
    q = item["question"]
    gt = item["ground_truth"]

    # 配置 A
    result_a = run_rag(q)
    correctness_a = evaluate_correctness(q, gt, result_a["answer"])

    # 配置 B
    result_b = run_rag_b(q)
    correctness_b = evaluate_correctness(q, gt, result_b["answer"])

    ab_results.append(
        {
            "question": q[:30],
            "score_a": correctness_a.score,
            "score_b": correctness_b.score,
        }
    )

print("  【A/B 对比结果（答案正确性维度）】")
print("  ┌────────────────────────────────┬────────┬────────┬────────┐")
print("  │ 问题                            │ A(k=2) │ B(k=3) │ 差异   │")
print("  ├────────────────────────────────┼────────┼────────┼────────┤")

total_a, total_b = 0, 0
for r in ab_results:
    diff = r["score_b"] - r["score_a"]
    diff_str = f"+{diff}" if diff > 0 else str(diff)
    arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
    print(
        f"  │ {r['question']:30s} │   {r['score_a']}    │   {r['score_b']}    │  {diff_str} {arrow}  │"
    )
    total_a += r["score_a"]
    total_b += r["score_b"]

avg_a = total_a / len(ab_results)
avg_b = total_b / len(ab_results)
diff_avg = avg_b - avg_a
print("  ├────────────────────────────────┼────────┼────────┼────────┤")
diff_str = f"+{diff_avg:.1f}" if diff_avg > 0 else f"{diff_avg:.1f}"
print(
    f"  │ 平均                            │  {avg_a:.1f}   │  {avg_b:.1f}   │ {diff_str:>5s}  │"
)
print("  └────────────────────────────────┴────────┴────────┴────────┘")
print()

if diff_avg > 0:
    print(f"  📈 结论：配置B (k=3) 比配置A (k=2) 平均高 {diff_avg:.1f} 分，建议采用。")
elif diff_avg < 0:
    print(
        f"  📉 结论：配置B (k=3) 比配置A (k=2) 平均低 {abs(diff_avg):.1f} 分，不建议切换。"
    )
else:
    print("  📊 结论：两种配置效果相当，保持现状即可。")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 总结
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("🎉 项目九学习完毕！")
print("=" * 60)
print()
print("💡 RAG 评估核心公式：")
print()
print("  评估数据集（问题+标准答案）")
print("  + 三大指标（相关性 / 忠实度 / 正确性）")
print("  + LLM-as-Judge（大模型当评委打分）")
print("  + 质量门禁（分数 < 阈值则阻止上线）")
print("  = 科学、可量化的 RAG 质量保障体系")
print()
print("💡 生产环境进阶：")
print("   ① LangSmith 平台：自动跑评估 + 历史趋势图 + 回归检测")
print("   ② RAGAS 框架：开源评估框架，指标更丰富（上下文精确率/召回率）")
print("   ③ CI/CD 集成：PR 合并前自动跑评估，分数下降则阻止合并")
print("   ④ 人工抽检：定期抽样人工评估，校准 LLM 评委的偏差")
print("   ⑤ 多评委共识：用多个 LLM 评分取平均，减少单模型偏见")
print("=" * 60)
