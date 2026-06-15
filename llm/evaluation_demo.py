"""
╔══════════════════════════════════════════════════════════════════╗
║         项目：LLM 评测与测试（Evaluation & Testing）全面实验      ║
║         探索客观题评测、RAG评测、LLM裁判、批量评测框架            ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题：我们怎么知道 LLM 的回答是好是坏？】
═══════════════════════════════════════════════════════════════════

在实际应用中，LLM 的输出质量直接决定产品体验。但"好不好"不能靠直觉，
需要一套系统化的评测方法。评测体系的构建是 LLM 工程化落地的关键环节。

  ┌─────────────────────────────────────────────────────────────┐
  │  LLM 评测体系总览                                            │
  │                                                             │
  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
  │  │  离线评测    │    │  在线评测    │    │  人工评测    │     │
  │  │             │    │             │    │             │     │
  │  │ - 客观题    │    │ - A/B测试   │    │ - 众包标注  │     │
  │  │ - RAG评测   │    │ - 用户反馈  │    │ - 专家评审  │     │
  │  │ - 自动打分  │    │ - 指标监控  │    │ - 偏好对比  │     │
  │  └─────────────┘    └─────────────┘    └─────────────┘     │
  │                                                             │
  │  本文件聚焦于【离线自动化评测】，包括：                       │
  │    1. 客观题评测（选择题自动判分）                            │
  │    2. RAG 评测（忠实性 + 相关性）                            │
  │    3. LLM-as-a-Judge（模型当裁判打分）                       │
  │    4. 批量评测框架（工程化封装）                              │
  └─────────────────────────────────────────────────────────────┘

本文件通过真实 API 调用，演示如何系统化地评估 LLM 的输出质量。
"""

import json
import time
from typing import Any

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：评测体系总览
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import client, MODEL_NAME

print("=" * 60)
print("第 0 章：评测体系总览")
print("=" * 60)
print()
print("""
┌──────────────────────────────────────────────────────────────┐
│           为什么需要评测？                                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 模型选型：哪个模型更适合我的场景？                        │
│  2. Prompt 优化：改了 prompt 后效果变好还是变差？             │
│  3. 回归测试：新版本上线前是否有能力退化？                    │
│  4. 质量监控：线上服务是否稳定输出高质量结果？                │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│           评测的分类                                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  【离线评测】在部署前，用固定测试集批量评估                   │
│    - 优点：可重复、可对比、成本可控                           │
│    - 方法：客观题、自动评分、LLM裁判                          │
│                                                              │
│  【在线评测】在部署后，用真实用户数据评估                     │
│    - 优点：反映真实场景、捕捉长尾问题                        │
│    - 方法：A/B测试、用户点赞/踩、人工抽检                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：客观题评测
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 客观题评测是最简单直接的方法：
#   - 给模型一道选择题，让它选 A/B/C/D
#   - 自动对比标准答案，判对错
#   - 批量跑完计算准确率
#
# 这就是 MMLU、C-Eval 等基准测试的核心思路。
#
#   ┌────────────────────────────────────────────────────────┐
#   │  MMLU 风格评测流程：                                    │
#   │                                                        │
#   │  测试集 → 逐题喂给模型 → 提取答案 → 对比正确答案       │
#   │                                           ↓            │
#   │                                     统计准确率          │
#   │                                                        │
#   │  关键挑战：                                             │
#   │    1. 如何让模型稳定输出 A/B/C/D（而不是一大段解释）    │
#   │    2. 如何从模型输出中可靠地提取答案选项                │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 1 章：客观题评测（MMLU 风格）")
print("=" * 60)
print()

# ── 1.1 构造测试题库 ──────────────────────────────────────────
# 模拟一组 MMLU 风格的选择题
EXAM_QUESTIONS = [
    {
        "question": "光合作用主要发生在植物细胞的哪个结构中？",
        "choices": {"A": "线粒体", "B": "叶绿体", "C": "细胞核", "D": "内质网"},
        "answer": "B",
    },
    {
        "question": "HTTP 协议默认使用的端口号是？",
        "choices": {"A": "21", "B": "22", "C": "80", "D": "443"},
        "answer": "C",
    },
    {
        "question": "以下哪位是《红楼梦》的作者？",
        "choices": {"A": "施耐庵", "B": "罗贯中", "C": "曹雪芹", "D": "吴承恩"},
        "answer": "C",
    },
    {
        "question": "地球上最大的洋是？",
        "choices": {"A": "大西洋", "B": "印度洋", "C": "北冰洋", "D": "太平洋"},
        "answer": "D",
    },
    {
        "question": "Python 中，list 和 tuple 最主要的区别是？",
        "choices": {"A": "list 有序 tuple 无序", "B": "list 可变 tuple 不可变", "C": "list 更快", "D": "tuple 不能嵌套"},
        "answer": "B",
    },
]


def extract_answer(response_text: str) -> str:
    """
    从模型回复中提取选项字母。
    策略：寻找第一个出现的 A/B/C/D 字母。
    """
    # 优先匹配常见的答案格式
    import re
    # 匹配 "答案是X"、"选X"、"X." 、"X、" 等格式
    patterns = [
        r"答案[是为：:]\s*([A-D])",
        r"选\s*([A-D])",
        r"^([A-D])[.、\s）)]",
        r"([A-D])",
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text.strip())
        if match:
            return match.group(1)
    return ""


def run_objective_eval(questions: list[dict]) -> dict:
    """
    运行客观题评测，返回评测结果。
    """
    correct = 0
    total = len(questions)
    results = []

    for i, q in enumerate(questions):
        # 构造 prompt，要求模型只输出答案字母
        choices_text = "\n".join([f"  {k}. {v}" for k, v in q["choices"].items()])
        prompt = f"""请回答以下选择题，只需要输出答案选项字母（A/B/C/D），不要解释。

题目：{q['question']}
选项：
{choices_text}

答案："""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个考试答题机器人，只需要输出正确答案的选项字母，不要任何解释。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )

        model_output = response.choices[0].message.content.strip()
        extracted = extract_answer(model_output)
        is_correct = (extracted == q["answer"])
        if is_correct:
            correct += 1

        results.append({
            "题号": i + 1,
            "模型输出": model_output,
            "提取答案": extracted,
            "正确答案": q["answer"],
            "判定": "正确" if is_correct else "错误",
        })

    accuracy = correct / total if total > 0 else 0
    return {"results": results, "accuracy": accuracy, "correct": correct, "total": total}


print("【实验】对模型进行 5 道客观题测试...")
print()

eval_result = run_objective_eval(EXAM_QUESTIONS)

for r in eval_result["results"]:
    status = "✓" if r["判定"] == "正确" else "✗"
    print(f"  [{status}] 第{r['题号']}题 | 模型输出: {r['模型输出']:<6} | 提取: {r['提取答案']} | 正确: {r['正确答案']} | {r['判定']}")

print()
print(f"  总计：{eval_result['correct']}/{eval_result['total']} 正确")
print(f"  准确率：{eval_result['accuracy']:.1%}")
print()
print("  提示：真实评测中，题目数量通常在数百到数千道，覆盖多个学科领域")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：RAG 评测（RAGAS 风格）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# RAG（Retrieval Augmented Generation）系统的评测，核心关注：
#   1. Faithfulness（忠实性）：答案是否基于给定的上下文？
#      - 有没有"幻觉"——编造上下文中没有的信息？
#   2. Answer Relevancy（答案相关性）：答案是否回答了用户的问题？
#      - 有没有答非所问？
#
#   ┌────────────────────────────────────────────────────────┐
#   │  RAGAS 评测框架核心指标：                                │
#   │                                                        │
#   │  用户问题 ──┐                                          │
#   │             ├─→ Answer Relevancy（答案和问题相关吗？）  │
#   │  模型答案 ──┤                                          │
#   │             ├─→ Faithfulness（答案忠于上下文吗？）      │
#   │  检索上下文─┘                                          │
#   │                                                        │
#   │  评测方法：用另一个 LLM 来评判                          │
#   │    - 将答案分解为多个陈述（claims）                     │
#   │    - 逐个检查每个陈述是否能从上下文中推导出              │
#   │    - 计算"忠实陈述数 / 总陈述数"作为忠实度分数         │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 2 章：RAG 评测（RAGAS 风格）")
print("=" * 60)
print()

# ── 2.1 Faithfulness 评测（忠实性） ────────────────────────────
print("── 2.1 Faithfulness（忠实性）评测 ────────────────────────")
print()
print("  评测目标：答案是否完全基于提供的上下文？有没有编造信息？")
print()

# 准备测试数据
RAG_TEST_CASES = [
    {
        "question": "CHJ公司的成立时间是什么？",
        "context": "CHJ公司成立于2015年7月，总部位于北京市顺义区。公司主要从事智能电动汽车的研发与制造，目前已推出多款量产车型。",
        "answer": "CHJ公司成立于2015年7月，是一家总部位于北京的智能电动汽车公司。",
        "label": "忠实（答案完全来自上下文）",
    },
    {
        "question": "CHJ公司的成立时间是什么？",
        "context": "CHJ公司成立于2015年7月，总部位于北京市顺义区。公司主要从事智能电动汽车的研发与制造，目前已推出多款量产车型。",
        "answer": "CHJ公司成立于2015年7月，创始人是李想，公司估值超过200亿美元。",
        "label": "不忠实（编造了创始人和估值信息）",
    },
]


def evaluate_faithfulness(question: str, context: str, answer: str) -> dict:
    """
    评测答案对上下文的忠实性。
    方法：让 LLM 将答案分解为独立陈述，然后逐个验证是否来自上下文。
    """
    # 步骤1：将答案分解为独立陈述
    decompose_prompt = f"""请将以下答案分解为独立的事实陈述（每个陈述只包含一个事实点）。
用 JSON 数组格式输出，每个元素是一个陈述字符串。

答案：{answer}

请直接输出 JSON 数组，不要其他内容："""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个文本分析工具，将文本分解为独立的事实陈述。只输出JSON数组。"},
            {"role": "user", "content": decompose_prompt},
        ],
        temperature=0.0,
        max_tokens=300,
    )

    # 解析陈述列表
    claims_text = response.choices[0].message.content.strip()
    try:
        # 尝试提取 JSON 数组
        import re
        json_match = re.search(r'\[.*\]', claims_text, re.DOTALL)
        if json_match:
            claims = json.loads(json_match.group())
        else:
            claims = [claims_text]
    except json.JSONDecodeError:
        claims = [claims_text]

    # 步骤2：逐个验证每个陈述是否能从上下文推导
    verified_count = 0
    verification_details = []

    for claim in claims:
        verify_prompt = f"""请判断以下陈述是否能从给定的上下文中推导出来。

上下文：{context}
陈述：{claim}

请只回答"是"或"否"：
- "是"表示该陈述可以从上下文中推导出来
- "否"表示该陈述包含上下文中没有的信息

判断："""

        verify_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个严格的事实验证工具。只回答'是'或'否'。"},
                {"role": "user", "content": verify_prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )

        verdict = verify_response.choices[0].message.content.strip()
        is_supported = "是" in verdict
        if is_supported:
            verified_count += 1
        verification_details.append({"陈述": claim, "是否支持": "是" if is_supported else "否"})

    # 计算忠实度分数
    faithfulness_score = verified_count / len(claims) if claims else 0

    return {
        "陈述列表": claims,
        "验证详情": verification_details,
        "忠实度分数": faithfulness_score,
        "支持数/总数": f"{verified_count}/{len(claims)}",
    }


for i, case in enumerate(RAG_TEST_CASES):
    print(f"  测试用例 {i+1}（预期：{case['label']}）")
    print(f"  问题：{case['question']}")
    print(f"  上下文：{case['context'][:50]}...")
    print(f"  答案：{case['answer']}")

    result = evaluate_faithfulness(case["question"], case["context"], case["answer"])

    print(f"  陈述分解：{result['陈述列表']}")
    print(f"  忠实度分数：{result['忠实度分数']:.2f}（{result['支持数/总数']}）")
    for detail in result["验证详情"]:
        print(f"    - [{detail['是否支持']}] {detail['陈述']}")
    print()


# ── 2.2 Answer Relevancy 评测（答案相关性） ─────────────────────
print("── 2.2 Answer Relevancy（答案相关性）评测 ─────────────────")
print()
print("  评测目标：答案是否真正回答了用户的问题？")
print()

RELEVANCY_TEST_CASES = [
    {
        "question": "Python 中如何读取 JSON 文件？",
        "answer": "在 Python 中，可以使用 json 模块的 load() 函数读取 JSON 文件。示例：import json; with open('data.json') as f: data = json.load(f)",
        "label": "高相关（直接回答了问题）",
    },
    {
        "question": "Python 中如何读取 JSON 文件？",
        "answer": "Python 是一种广泛使用的编程语言，由 Guido van Rossum 于 1991 年发布。它支持多种编程范式。",
        "label": "低相关（答非所问）",
    },
]


def evaluate_relevancy(question: str, answer: str) -> dict:
    """
    评测答案与问题的相关性。
    方法：让 LLM 根据答案反推可能的问题，然后比较原始问题和反推问题的相似度。
    """
    # 让 LLM 直接评分
    eval_prompt = f"""请评估以下答案与问题的相关性。

问题：{question}
答案：{answer}

评分标准（1-5分）：
  1分：完全不相关，答非所问
  2分：略有关联但没有回答问题
  3分：部分回答了问题
  4分：基本回答了问题，但不够完整
  5分：完全回答了问题，高度相关

请按以下 JSON 格式输出：
{{"score": <1-5的整数>, "reason": "<简短理由>"}}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一个答案质量评估工具。只输出JSON格式的评分结果。"},
            {"role": "user", "content": eval_prompt},
        ],
        temperature=0.0,
        max_tokens=100,
    )

    result_text = response.choices[0].message.content.strip()
    try:
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"score": 0, "reason": "解析失败"}
    except json.JSONDecodeError:
        result = {"score": 0, "reason": "解析失败"}

    return result


for i, case in enumerate(RELEVANCY_TEST_CASES):
    print(f"  测试用例 {i+1}（预期：{case['label']}）")
    print(f"  问题：{case['question']}")
    print(f"  答案：{case['answer'][:60]}...")

    result = evaluate_relevancy(case["question"], case["answer"])
    print(f"  相关性评分：{result.get('score', 'N/A')}/5")
    print(f"  评分理由：{result.get('reason', 'N/A')}")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：LLM-as-a-Judge（模型当裁判）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# LLM-as-a-Judge 是目前最流行的自动评测方法之一：
#   - 用一个强大的模型充当"裁判"
#   - 对另一个模型的输出进行多维度打分
#   - 相比人工评测，成本低、速度快、可大规模执行
#
#   ┌────────────────────────────────────────────────────────┐
#   │  LLM-as-a-Judge 工作流程：                              │
#   │                                                        │
#   │  原始问题 ─┐                                           │
#   │            ├─→ 裁判模型 ─→ 多维度评分                  │
#   │  模型回答 ─┘        │                                  │
#   │                     ↓                                  │
#   │              ┌─────────────────┐                       │
#   │              │ 准确性：8/10     │                       │
#   │              │ 完整性：7/10     │                       │
#   │              │ 流畅性：9/10     │                       │
#   │              │ 总评：8.0/10    │                       │
#   │              └─────────────────┘                       │
#   │                                                        │
#   │  注意事项：                                             │
#   │    - 裁判模型自身也有偏差（位置偏差、冗长偏差等）       │
#   │    - 最好用比被评测模型更强的模型当裁判                  │
#   │    - 评分标准要详细明确，减少主观性                      │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 3 章：LLM-as-a-Judge（模型当裁判）")
print("=" * 60)
print()

# ── 3.1 多维度评分实现 ─────────────────────────────────────────

JUDGE_DIMENSIONS = {
    "准确性": "信息是否正确，有无事实错误",
    "完整性": "是否全面回答了问题的各个方面",
    "流畅性": "语言是否通顺自然，逻辑是否清晰",
}


def llm_judge(question: str, answer: str, dimensions: dict[str, str] = JUDGE_DIMENSIONS) -> dict:
    """
    让 LLM 充当裁判，对回答进行多维度评分。

    参数：
        question: 原始问题
        answer: 待评测的回答
        dimensions: 评分维度及其描述

    返回：
        包含各维度分数和总评的字典
    """
    # 构造评分维度说明
    dim_text = "\n".join([f"  - {name}（{desc}）：1-10分" for name, desc in dimensions.items()])

    judge_prompt = f"""你是一位严格公正的评测专家。请对以下问答对进行多维度评分。

【问题】
{question}

【回答】
{answer}

【评分维度】（每个维度 1-10 分，10分最好）
{dim_text}

请按以下 JSON 格式输出评分结果：
{{
  "scores": {{
    "准确性": <分数>,
    "完整性": <分数>,
    "流畅性": <分数>
  }},
  "overall": <三个维度的平均分，保留一位小数>,
  "comment": "<总体评价，一句话>"
}}

请直接输出 JSON，不要其他内容："""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一位公正严格的AI评测专家。只输出JSON格式的评分。"},
            {"role": "user", "content": judge_prompt},
        ],
        temperature=0.0,
        max_tokens=200,
    )

    result_text = response.choices[0].message.content.strip()
    try:
        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = {"scores": {}, "overall": 0, "comment": "解析失败"}
    except json.JSONDecodeError:
        result = {"scores": {}, "overall": 0, "comment": "解析失败"}

    return result


# ── 3.2 生成待评测的回答并打分 ──────────────────────────────────
print("【实验】先让模型回答问题，再用模型当裁判打分")
print()

JUDGE_TEST_QUESTIONS = [
    "请解释什么是机器学习中的过拟合（overfitting），以及如何避免它？",
    "请用简单的语言解释区块链技术的工作原理。",
]

for i, question in enumerate(JUDGE_TEST_QUESTIONS):
    print(f"── 问题 {i+1}：{question}")
    print()

    # 先让模型回答
    answer_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一位知识渊博的技术专家，请用中文简洁回答。"},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=200,
    )
    answer = answer_response.choices[0].message.content.strip()
    print(f"  模型回答：{answer[:100]}...")
    print()

    # 用 LLM 裁判打分
    judge_result = llm_judge(question, answer)
    print(f"  裁判评分：")
    scores = judge_result.get("scores", {})
    for dim_name, score in scores.items():
        print(f"    {dim_name}：{score}/10")
    print(f"    综合分：{judge_result.get('overall', 'N/A')}/10")
    print(f"    总评：{judge_result.get('comment', 'N/A')}")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：批量评测框架
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 在实际项目中，我们需要一个工程化的评测框架：
#   - 支持批量运行多个测试用例
#   - 自动汇总统计结果
#   - 输出结构化报告
#   - 支持多种评测方法的组合
#
#   ┌────────────────────────────────────────────────────────┐
#   │  批量评测框架设计：                                      │
#   │                                                        │
#   │  EvalFramework                                         │
#   │    ├── add_test_case(question, expected, ...)          │
#   │    ├── run_all()                                       │
#   │    │     ├── 逐个运行测试用例                           │
#   │    │     ├── 调用评测方法（客观题/LLM裁判/...）         │
#   │    │     └── 收集结果                                  │
#   │    └── report()                                        │
#   │          ├── 统计汇总（平均分、通过率）                 │
#   │          └── 输出结构化报告                             │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 4 章：批量评测框架")
print("=" * 60)
print()


class EvalFramework:
    """
    简单的 LLM 批量评测框架。
    支持添加测试用例、批量运行、汇总报告。
    """

    def __init__(self, model_name: str = MODEL_NAME, judge_dimensions: dict = None):
        """
        初始化评测框架。

        参数：
            model_name: 被评测的模型名称
            judge_dimensions: LLM裁判的评分维度
        """
        self.model_name = model_name
        self.judge_dimensions = judge_dimensions or JUDGE_DIMENSIONS
        self.test_cases: list[dict] = []
        self.results: list[dict] = []

    def add_test_case(self, question: str, system_prompt: str = "",
                      expected_answer: str = "", eval_type: str = "judge"):
        """
        添加一个测试用例。

        参数：
            question: 测试问题
            system_prompt: 系统提示词
            expected_answer: 期望答案（用于对比）
            eval_type: 评测类型，"judge"=LLM裁判, "exact"=精确匹配, "contains"=包含匹配
        """
        self.test_cases.append({
            "question": question,
            "system_prompt": system_prompt,
            "expected_answer": expected_answer,
            "eval_type": eval_type,
        })

    def _get_model_response(self, question: str, system_prompt: str) -> str:
        """调用模型获取回答"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    def _evaluate_single(self, case: dict, model_answer: str) -> dict:
        """对单个测试用例进行评测"""
        eval_type = case["eval_type"]

        if eval_type == "exact":
            # 精确匹配
            passed = model_answer.strip() == case["expected_answer"].strip()
            return {"passed": passed, "score": 1.0 if passed else 0.0, "detail": "精确匹配"}

        elif eval_type == "contains":
            # 包含匹配：答案中是否包含关键信息
            keywords = case["expected_answer"].split("|")
            matched = sum(1 for kw in keywords if kw.strip() in model_answer)
            score = matched / len(keywords) if keywords else 0
            return {
                "passed": score >= 0.8,
                "score": score,
                "detail": f"关键词匹配 {matched}/{len(keywords)}",
            }

        elif eval_type == "judge":
            # LLM 裁判评分
            judge_result = llm_judge(case["question"], model_answer, self.judge_dimensions)
            overall = judge_result.get("overall", 0)
            # 7分以上算通过
            try:
                overall_float = float(overall)
            except (TypeError, ValueError):
                overall_float = 0
            return {
                "passed": overall_float >= 7.0,
                "score": overall_float / 10.0,
                "detail": judge_result.get("comment", ""),
                "dimension_scores": judge_result.get("scores", {}),
            }

        return {"passed": False, "score": 0, "detail": "未知评测类型"}

    def run_all(self) -> list[dict]:
        """运行所有测试用例"""
        self.results = []
        total = len(self.test_cases)

        print(f"  开始批量评测，共 {total} 个测试用例...")
        print()

        for i, case in enumerate(self.test_cases):
            print(f"  [{i+1}/{total}] 正在评测：{case['question'][:30]}...")

            # 获取模型回答
            model_answer = self._get_model_response(case["question"], case["system_prompt"])

            # 进行评测
            eval_result = self._evaluate_single(case, model_answer)

            self.results.append({
                "case_index": i + 1,
                "question": case["question"],
                "model_answer": model_answer,
                "eval_type": case["eval_type"],
                **eval_result,
            })

            status = "通过" if eval_result["passed"] else "未通过"
            print(f"       结果：{status}（得分：{eval_result['score']:.2f}）")

        print()
        return self.results

    def report(self) -> dict:
        """输出汇总报告"""
        if not self.results:
            print("  暂无评测结果，请先运行 run_all()")
            return {}

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        avg_score = sum(r["score"] for r in self.results) / total

        # 按评测类型分组统计
        type_stats = {}
        for r in self.results:
            et = r["eval_type"]
            if et not in type_stats:
                type_stats[et] = {"total": 0, "passed": 0, "scores": []}
            type_stats[et]["total"] += 1
            if r["passed"]:
                type_stats[et]["passed"] += 1
            type_stats[et]["scores"].append(r["score"])

        # 输出报告
        print("  ┌────────────────────────────────────────────────┐")
        print("  │          评测汇总报告                           │")
        print("  ├────────────────────────────────────────────────┤")
        print(f"  │  被评测模型：{self.model_name:<32}│")
        print(f"  │  测试用例数：{total:<32}│")
        print(f"  │  通过数量：  {passed}/{total} ({passed/total:.1%}){' '*(25-len(f'{passed}/{total} ({passed/total:.1%})'))}│")
        print(f"  │  平均得分：  {avg_score:.3f}{' '*27}│")
        print("  ├────────────────────────────────────────────────┤")
        print("  │  按评测类型分组：                               │")
        for et, stats in type_stats.items():
            et_avg = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
            print(f"  │    {et:<10} 通过率: {stats['passed']}/{stats['total']:<4} 平均分: {et_avg:.2f}  │")
        print("  └────────────────────────────────────────────────┘")
        print()

        # 列出未通过的用例
        failed = [r for r in self.results if not r["passed"]]
        if failed:
            print("  未通过的用例：")
            for r in failed:
                print(f"    - [{r['case_index']}] {r['question'][:40]}... (得分: {r['score']:.2f})")
            print()

        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total,
            "avg_score": avg_score,
            "type_stats": type_stats,
        }


# ── 4.1 使用评测框架运行批量测试 ───────────────────────────────
print("── 4.1 使用评测框架运行批量测试 ──────────────────────────")
print()

# 创建评测框架实例
framework = EvalFramework(model_name=MODEL_NAME)

# 添加测试用例 —— 包含匹配类型
framework.add_test_case(
    question="中国的首都是哪里？",
    system_prompt="用一个词回答。",
    expected_answer="北京",
    eval_type="contains",
)

framework.add_test_case(
    question="1+1等于几？",
    system_prompt="只回答数字。",
    expected_answer="2",
    eval_type="contains",
)

# 添加测试用例 —— LLM 裁判类型
framework.add_test_case(
    question="请解释什么是梯度下降算法，以及它在机器学习中的作用。",
    system_prompt="你是一位机器学习教授，请用简洁的语言回答。",
    eval_type="judge",
)

framework.add_test_case(
    question="请列举三种常见的排序算法，并简述它们的时间复杂度。",
    system_prompt="你是一位计算机科学教授。",
    eval_type="judge",
)

framework.add_test_case(
    question="请解释TCP和UDP的区别。",
    system_prompt="你是一位网络工程师，请简洁回答。",
    expected_answer="可靠|连接|无连接",
    eval_type="contains",
)

# 运行所有测试
framework.run_all()

# 输出汇总报告
print("── 4.2 评测汇总报告 ─────────────────────────────────────")
print()
framework.report()


# ── 总结 ────────────────────────────────────────────────────
print("=" * 60)
print("全部实验完毕！核心总结：")
print("=" * 60)
print("""
  ┌────────────────────────────────────────────────────────────┐
  │  评测方法         │ 适用场景              │ 优缺点          │
  ├────────────────────────────────────────────────────────────┤
  │  客观题评测       │ 知识能力评估           │ 简单可靠但覆    │
  │                   │ （MMLU/C-Eval）        │ 盖面有限        │
  ├────────────────────────────────────────────────────────────┤
  │  RAG 评测         │ 检索增强生成质量       │ 针对性强，需    │
  │  (RAGAS风格)      │ 忠实性/相关性          │ 要设计测试集    │
  ├────────────────────────────────────────────────────────────┤
  │  LLM-as-a-Judge   │ 开放式问答、创意      │ 灵活通用但有    │
  │                   │ 生成等主观任务         │ 评判偏差        │
  ├────────────────────────────────────────────────────────────┤
  │  批量评测框架     │ 工程化落地、回归      │ 可自动化、可    │
  │                   │ 测试、持续监控         │ 重复、可扩展    │
  └────────────────────────────────────────────────────────────┘

  实践建议：
  1. 先定义清楚"好的回答"的标准，再设计评测方案
  2. 客观题 + LLM裁判 组合使用，兼顾覆盖面和评测深度
  3. 关键场景建立回归测试集，每次改 prompt/模型都要跑一遍
  4. 注意 LLM 裁判的偏差：位置偏差、冗长偏差、自我偏好
  5. 评测结果要结合人工抽检进行校验
""")
