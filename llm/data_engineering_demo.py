"""
╔══════════════════════════════════════════════════════════════════╗
║         项目：生产级数据工程（Data Engineering for LLM）          ║
║         从数据清洗到合成数据生成的完整工程实践                     ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题：为什么说"数据质量决定 LLM 应用的上限"？】
═══════════════════════════════════════════════════════════════════

无论是微调（SFT/RLHF）还是 RAG 检索增强，数据质量都是第一生产力：

  原始数据 → [清洗去噪] → [质量评估] → [格式化] → [合成扩展] → 高质量训练集
                ↓              ↓            ↓            ↓
           去HTML/去重    多维度打分    Alpaca/ShareGPT  Seed→批量生成

  ┌─────────────────────────────────────────────────────────────┐
  │  数据工程的四大支柱：                                         │
  │                                                             │
  │  1. 数据清洗（Cleaning）                                     │
  │     去除噪声、统一格式、消除重复                              │
  │                                                             │
  │  2. 质量评估（Quality Scoring）                              │
  │     长度/多样性/信息密度 + LLM 辅助打分                       │
  │                                                             │
  │  3. 格式标准化（Formatting）                                 │
  │     将原始文本转为 Alpaca / ShareGPT 等训练格式               │
  │                                                             │
  │  4. 合成数据（Synthetic Data）                               │
  │     用 LLM 从少量种子样本批量生成高质量训练数据               │
  └─────────────────────────────────────────────────────────────┘

本文件通过真实代码 + API 调用，演示完整的数据工程流水线。
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：LLM 数据工程总览
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import json
import re
import hashlib
from collections import Counter

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import client, MODEL_NAME

print("=" * 60)
print("第 0 章：LLM 数据工程总览")
print("=" * 60)
print()
print("""
┌──────────────────────────────────────────────────────────────┐
│           数据在 LLM 应用中的角色（Data's Role in LLM）        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  场景一：微调（Fine-tuning / SFT）                            │
│    原始语料 → 清洗 → QA对构造 → Alpaca格式 → 训练             │
│    数据决定：模型学到什么知识、什么风格                        │
│                                                              │
│  场景二：RAG（检索增强生成）                                   │
│    文档库 → 清洗 → 切片 → 向量化 → 检索 → 拼接Prompt         │
│    数据决定：检索的准确率和回答的可靠性                        │
│                                                              │
│  场景三：评估（Evaluation）                                   │
│    评测集 → 格式化 → 模型推理 → 自动/人工打分                 │
│    数据决定：评估是否公平、全面、有区分度                      │
│                                                              │
│  核心原则：Garbage In, Garbage Out                            │
│  无论模型多强大，低质量数据只会产出低质量结果                  │
└──────────────────────────────────────────────────────────────┘
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：数据清洗
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 数据清洗是数据工程的第一步，也是最容易被忽视的一步。
# 常见清洗操作：
#   1. 去除 HTML 标签 — 从网页抓取的文本中移除标签
#   2. 统一标点符号 — 全角/半角统一，避免同义不同形
#   3. 去除多余空格 — 连续空格合并，首尾空格去除
#   4. 文本去重 — 基于指纹算法去除近似重复文本
#
#   ┌────────────────────────────────────────────────────────┐
#   │  清洗流水线：                                           │
#   │                                                        │
#   │  原始文本                                               │
#   │    ↓ strip_html()     去除<p><b>等HTML标签              │
#   │    ↓ normalize_punct() 全角逗号→半角，统一引号          │
#   │    ↓ clean_whitespace() 多个空格→一个，去首尾空格       │
#   │    ↓ deduplicate()     SimHash指纹去重                  │
#   │  干净文本                                               │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 1 章：数据清洗")
print("=" * 60)
print()


# ── 1.1 文本清洗函数 ─────────────────────────────────────────
print("── 1.1 文本清洗函数 ──────────────────────────────────")
print()


def strip_html(text: str) -> str:
    """去除 HTML 标签，保留纯文本内容"""
    # 去除所有 HTML 标签
    clean = re.sub(r"<[^>]+>", "", text)
    # 处理常见 HTML 实体
    clean = clean.replace("&nbsp;", " ")
    clean = clean.replace("&amp;", "&")
    clean = clean.replace("&lt;", "<")
    clean = clean.replace("&gt;", ">")
    clean = clean.replace("&quot;", '"')
    return clean


def normalize_punct(text: str) -> str:
    """统一标点符号：全角转半角，统一引号"""
    # 全角标点转半角
    replacements = {
        "，": ", ",
        "。": ". ",
        "！": "! ",
        "？": "? ",
        "；": "; ",
        "：": ": ",
        "\u201c": '"',  # 左双引号
        "\u201d": '"',  # 右双引号
        "\u2018": "'",  # 左单引号
        "\u2019": "'",  # 右单引号
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_whitespace(text: str) -> str:
    """清理多余空白字符"""
    # 将多个连续空格合并为一个
    text = re.sub(r" {2,}", " ", text)
    # 将多个连续换行合并为一个
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去除首尾空白
    text = text.strip()
    return text


def clean_text(text: str) -> str:
    """完整清洗流水线：HTML去除 → 标点统一 → 空白清理"""
    text = strip_html(text)
    text = normalize_punct(text)
    text = clean_whitespace(text)
    return text


# 演示清洗效果
raw_samples = [
    '<p>这是一段<b>带HTML标签</b>的文本。</p><br>&nbsp;&nbsp;包含多余空格。',
    '人工智能（AI）正在改变世界，  它的应用范围非常广泛。   从医疗到金融，无所不在。',
    '这是\u201c一段带中文引号\u201d的文本，还有全角逗号\uff0c和感叹号\uff01',
]

print("  清洗前后对比：")
print()
for i, raw in enumerate(raw_samples, 1):
    cleaned = clean_text(raw)
    print(f"  样本{i} 原始: {raw[:60]}...")
    print(f"  样本{i} 清洗: {cleaned[:60]}...")
    print()


# ── 1.2 基于 SimHash 的文本去重 ─────────────────────────────
print("── 1.2 基于 SimHash 的文本去重 ──────────────────────")
print()


def get_simhash(text: str, hash_bits: int = 64) -> int:
    """
    计算文本的 SimHash 指纹。
    SimHash 是一种局部敏感哈希，相似文本的哈希值也相似。
    """
    # 分词（简单按字符 n-gram 切分）
    n = 3  # 3-gram
    tokens = [text[i:i+n] for i in range(len(text) - n + 1)]

    # 初始化向量
    v = [0] * hash_bits

    for token in tokens:
        # 对每个 token 计算哈希
        token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        for i in range(hash_bits):
            bit = (token_hash >> i) & 1
            if bit:
                v[i] += 1
            else:
                v[i] -= 1

    # 生成最终指纹
    fingerprint = 0
    for i in range(hash_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def hamming_distance(hash1: int, hash2: int) -> int:
    """计算两个哈希值的汉明距离"""
    xor = hash1 ^ hash2
    distance = bin(xor).count("1")
    return distance


def deduplicate(texts: list, threshold: int = 10) -> list:
    """
    基于 SimHash 去重。
    汉明距离 <= threshold 的文本视为近似重复。
    """
    unique_texts = []
    fingerprints = []

    for text in texts:
        fp = get_simhash(text)
        is_duplicate = False

        for existing_fp in fingerprints:
            if hamming_distance(fp, existing_fp) <= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_texts.append(text)
            fingerprints.append(fp)

    return unique_texts


# 演示去重效果
corpus = [
    "大语言模型是人工智能领域的重要突破，它能够理解和生成自然语言。",
    "大语言模型是AI领域的重要突破，它可以理解和生成自然语言文本。",  # 近似重复
    "深度学习在计算机视觉方面取得了巨大成功，尤其是图像识别任务。",
    "深度学习在CV领域取得了巨大进步，特别是在图像识别方面。",  # 近似重复
    "强化学习通过奖励信号来训练智能体，使其学会在环境中做出最优决策。",
]

print(f"  去重前文本数量: {len(corpus)}")
unique_corpus = deduplicate(corpus, threshold=10)
print(f"  去重后文本数量: {len(unique_corpus)}")
print()
print("  保留的文本:")
for i, text in enumerate(unique_corpus, 1):
    print(f"    {i}. {text}")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：数据质量评估
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 数据质量评估帮助我们筛选出高质量的训练样本。
# 评估维度：
#   1. 长度得分 — 太短信息不足，太长可能有噪声
#   2. 多样性得分 — 词汇丰富度，避免重复堆砌
#   3. 信息密度 — 有效信息占比，排除"水文"
#   4. LLM 辅助打分 — 用大模型从可读性/准确性/完整性维度打分
#
#   ┌────────────────────────────────────────────────────────┐
#   │  质量评估流水线：                                       │
#   │                                                        │
#   │  文本样本                                               │
#   │    ↓ length_score()      长度是否在合理范围             │
#   │    ↓ diversity_score()   词汇多样性(TTR指标)            │
#   │    ↓ density_score()     信息密度(停用词占比)           │
#   │    ↓ llm_quality_score() LLM多维度打分                 │
#   │  综合质量分数 → 决定是否纳入训练集                      │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 2 章：数据质量评估")
print("=" * 60)
print()


# ── 2.1 多维度质量打分函数 ───────────────────────────────────
print("── 2.1 多维度质量打分 ──────────────────────────────────")
print()

# 常见中文停用词（简化版）
STOP_WORDS = set("的了是在我有和人这中大为上个国地到以说时要就出会也年对生能而学下自可她与里那"
                 "将后作来用我们为着去之过家十要之所到对于子二三被不多么同现当没动面起")


def length_score(text: str, min_len: int = 20, max_len: int = 2000) -> float:
    """
    长度得分：文本长度在合理范围内得高分。
    太短（<min_len）或太长（>max_len）会被惩罚。
    """
    n = len(text)
    if n < min_len:
        return n / min_len  # 线性衰减
    elif n > max_len:
        return max(0.0, 1.0 - (n - max_len) / max_len)
    else:
        return 1.0


def diversity_score(text: str) -> float:
    """
    多样性得分：基于 Type-Token Ratio (TTR)。
    TTR = 不同字符数 / 总字符数，越高表示词汇越丰富。
    """
    chars = [c for c in text if c.strip()]  # 去除空白
    if not chars:
        return 0.0
    unique_chars = set(chars)
    # TTR 通常在 0.3~0.8 之间，归一化到 0~1
    ttr = len(unique_chars) / len(chars)
    return min(1.0, ttr / 0.7)  # 0.7 以上视为满分


def density_score(text: str) -> float:
    """
    信息密度得分：非停用词占比越高，信息密度越大。
    纯"水文"停用词占比高，有效信息少。
    """
    chars = [c for c in text if c.strip()]
    if not chars:
        return 0.0
    non_stop = [c for c in chars if c not in STOP_WORDS]
    ratio = len(non_stop) / len(chars)
    return ratio


def compute_quality_scores(text: str) -> dict:
    """计算文本的多维度质量分数"""
    scores = {
        "长度得分": round(length_score(text), 3),
        "多样性得分": round(diversity_score(text), 3),
        "信息密度得分": round(density_score(text), 3),
    }
    # 综合分 = 加权平均
    scores["综合得分"] = round(
        scores["长度得分"] * 0.3 +
        scores["多样性得分"] * 0.4 +
        scores["信息密度得分"] * 0.3,
        3
    )
    return scores


# 演示质量打分
quality_samples = [
    "好的好的好的好的好的好的好的好的好的好的",  # 低质量：重复堆砌
    "AI 是一种技术。",  # 低质量：太短
    "大语言模型通过海量文本数据进行预训练，学习语言的统计规律和语义表示。在微调阶段，通过特定任务的标注数据进一步优化模型参数，使其能够更好地完成下游任务，如问答、摘要、翻译等。",  # 高质量
]

print("  多维度质量评分结果：")
print()
for i, sample in enumerate(quality_samples, 1):
    scores = compute_quality_scores(sample)
    print(f"  样本{i}: {sample[:40]}...")
    print(f"    {json.dumps(scores, ensure_ascii=False, indent=6)}")
    print()


# ── 2.2 用 LLM 给样本质量打分 ───────────────────────────────
print("── 2.2 用 LLM 给样本质量打分 ──────────────────────────")
print()


def llm_quality_score(text: str) -> dict:
    """
    用 LLM 对文本质量进行多维度打分。
    返回可读性、准确性、完整性的分数（1-5分）。
    """
    prompt = f"""请对以下文本进行质量评估，从三个维度打分（1-5分）：
1. 可读性：文本是否通顺、易读
2. 信息量：文本是否包含有价值的信息
3. 完整性：文本表达是否完整、不残缺

文本内容：
\"\"\"{text}\"\"\"

请严格按以下JSON格式返回，不要包含其他内容：
{{"可读性": 分数, "信息量": 分数, "完整性": 分数, "评语": "一句话点评"}}"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一位数据质量评估专家，请严格按JSON格式输出评估结果。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    result_text = response.choices[0].message.content.strip()
    # 尝试解析 JSON
    try:
        # 提取 JSON 部分（处理可能的 markdown 包裹）
        json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    return {"原始返回": result_text}


# 用 LLM 对样本打分
llm_eval_samples = [
    "好好好好好好好好好好",
    "大语言模型通过自注意力机制捕捉长距离依赖关系，使用Transformer架构进行并行化训练，在自然语言理解和生成任务上取得了突破性进展。",
]

print("  LLM 质量评估结果：")
print()
for i, sample in enumerate(llm_eval_samples, 1):
    print(f"  样本{i}: {sample[:50]}...")
    score_result = llm_quality_score(sample)
    print(f"    评分: {json.dumps(score_result, ensure_ascii=False, indent=6)}")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：SFT 数据格式化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 微调数据需要特定格式，常见两种：
#
#   1. Alpaca 格式（Stanford Alpaca 提出）
#      适合单轮指令跟随任务
#      {"instruction": "...", "input": "...", "output": "..."}
#
#   2. ShareGPT 格式（多轮对话）
#      适合对话式微调
#      {"conversations": [{"from": "human", "value": "..."},
#                          {"from": "gpt", "value": "..."}]}
#
#   ┌────────────────────────────────────────────────────────┐
#   │  Alpaca 格式：                                         │
#   │  {                                                     │
#   │    "instruction": "将以下文本翻译成英文",               │
#   │    "input": "今天天气很好",                             │
#   │    "output": "The weather is nice today"               │
#   │  }                                                     │
#   │                                                        │
#   │  ShareGPT 格式：                                       │
#   │  {                                                     │
#   │    "conversations": [                                  │
#   │      {"from": "human", "value": "你好"},               │
#   │      {"from": "gpt", "value": "你好！有什么可以帮你？"} │
#   │    ]                                                   │
#   │  }                                                     │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 3 章：SFT 数据格式化")
print("=" * 60)
print()


# ── 3.1 Alpaca 格式数据构造 ──────────────────────────────────
print("── 3.1 Alpaca 格式数据构造 ────────────────────────────")
print()


def create_alpaca_sample(instruction: str, input_text: str, output: str) -> dict:
    """构造 Alpaca 格式的训练样本"""
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


# 构造示例数据
alpaca_samples = [
    create_alpaca_sample(
        instruction="对以下文本进行情感分析，判断是正面还是负面。",
        input_text="这家餐厅的菜品非常美味，服务也很周到，下次还会再来！",
        output="正面情感。文本中'非常美味'、'很周到'、'还会再来'等表述表明用户体验良好。"
    ),
    create_alpaca_sample(
        instruction="将以下段落总结为一句话。",
        input_text="深度学习是机器学习的一个分支，它通过多层神经网络来学习数据的表示。近年来，随着计算能力的提升和大规模数据集的可用性，深度学习在图像识别、自然语言处理等领域取得了显著突破。",
        output="深度学习是利用多层神经网络学习数据表示的机器学习分支，近年来在视觉和语言等领域取得突破。"
    ),
    create_alpaca_sample(
        instruction="解释以下概念。",
        input_text="",
        output="Transformer是一种基于自注意力机制的神经网络架构，由Vaswani等人于2017年提出，它摒弃了传统的循环结构，通过并行计算大幅提升了训练效率。"
    ),
]

print("  Alpaca 格式样本示例：")
print()
for i, sample in enumerate(alpaca_samples, 1):
    print(f"  样本{i}:")
    print(f"    {json.dumps(sample, ensure_ascii=False, indent=4)}")
    print()


# ── 3.2 ShareGPT 格式数据构造 ────────────────────────────────
print("── 3.2 ShareGPT 格式数据构造 ──────────────────────────")
print()


def create_sharegpt_sample(conversations: list) -> dict:
    """
    构造 ShareGPT 格式的训练样本。
    conversations: [(role, content), ...] 其中 role 为 "human" 或 "gpt"
    """
    return {
        "conversations": [
            {"from": role, "value": content}
            for role, content in conversations
        ]
    }


# 构造多轮对话样本
sharegpt_samples = [
    create_sharegpt_sample([
        ("human", "什么是机器学习？"),
        ("gpt", "机器学习是人工智能的一个子领域，它让计算机能够从数据中自动学习规律，而无需被显式编程。常见的方法包括监督学习、无监督学习和强化学习。"),
        ("human", "监督学习和无监督学习的区别是什么？"),
        ("gpt", "主要区别在于是否使用标注数据：监督学习使用带标签的数据进行训练（如分类、回归），而无监督学习处理无标签数据，旨在发现数据中的隐藏结构（如聚类、降维）。"),
    ]),
    create_sharegpt_sample([
        ("human", "帮我写一个Python冒泡排序"),
        ("gpt", "```python\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n```\n这个冒泡排序的时间复杂度为O(n^2)，空间复杂度为O(1)。"),
    ]),
]

print("  ShareGPT 格式样本示例：")
print()
for i, sample in enumerate(sharegpt_samples, 1):
    print(f"  样本{i}:")
    print(f"    {json.dumps(sample, ensure_ascii=False, indent=4)}")
    print()


# ── 3.3 用 LLM 从原始文本生成 QA 对 ────────────────────────
print("── 3.3 用 LLM 从原始文本生成 QA 对 ──────────────────")
print()


def generate_qa_pairs(text: str, num_pairs: int = 3) -> list:
    """
    用 LLM 从原始文本自动生成问答对。
    适合将知识文档转化为 SFT 训练数据。
    """
    prompt = f"""请基于以下文本，生成{num_pairs}个高质量的问答对。
要求：
1. 问题应该多样化（事实性问题、理解性问题、应用性问题）
2. 答案应该准确、完整，基于原文内容
3. 严格按JSON数组格式返回

原始文本：
\"\"\"{text}\"\"\"

请按以下格式返回：
[{{"question": "问题1", "answer": "答案1"}}, {{"question": "问题2", "answer": "答案2"}}, ...]"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一位教育领域的数据工程师，擅长从文本中提取知识点并生成高质量问答对。请严格按JSON格式输出。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=800,
    )
    result_text = response.choices[0].message.content.strip()
    # 解析 JSON
    try:
        json_match = re.search(r"\[.*\]", result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    return [{"原始返回": result_text}]


# 从一段知识文本生成 QA 对
source_text = """
Transformer模型由编码器和解码器组成。编码器通过多头自注意力机制和前馈神经网络处理输入序列，
解码器在此基础上增加了交叉注意力层来关注编码器的输出。模型使用位置编码来注入序列顺序信息，
因为自注意力机制本身不具有位置感知能力。训练时使用教师强制策略，推理时使用自回归方式逐步生成。
"""

print(f"  原始文本: {source_text.strip()[:80]}...")
print()
print("  生成的 QA 对：")
qa_pairs = generate_qa_pairs(source_text.strip(), num_pairs=3)
for i, qa in enumerate(qa_pairs, 1):
    print(f"    QA{i}: {json.dumps(qa, ensure_ascii=False, indent=6)}")
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：合成数据生成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 合成数据（Synthetic Data）是用 LLM 批量生成训练数据的技术。
# 核心思路：从少量高质量"种子样本"出发，让 LLM 模仿风格批量扩展。
#
# 典型流程：
#   1. 人工编写 3-5 个种子样本（Seed Examples）
#   2. 将种子样本作为 Few-shot 示例放入 Prompt
#   3. 指定目标领域和变化维度
#   4. LLM 批量生成新样本
#   5. 质量过滤 + 去重
#
#   ┌────────────────────────────────────────────────────────┐
#   │  合成数据流水线：                                       │
#   │                                                        │
#   │  种子样本(3-5个)                                        │
#   │    ↓ 构造 Few-shot Prompt                              │
#   │    ↓ 指定领域/场景/难度                                 │
#   │    ↓ LLM 批量生成                                      │
#   │    ↓ 质量过滤(打分>阈值)                                │
#   │    ↓ 去重(SimHash)                                     │
#   │  最终训练数据集                                         │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 4 章：合成数据生成")
print("=" * 60)
print()


# ── 4.1 定义种子样本 ─────────────────────────────────────────
print("── 4.1 种子样本定义（金融领域示例）──────────────────────")
print()

# 金融领域种子样本
finance_seeds = [
    {
        "instruction": "解释什么是市盈率（P/E Ratio），并说明如何使用它评估股票。",
        "output": "市盈率（P/E Ratio）是股票价格与每股收益的比值，计算公式为：P/E = 股价 / 每股收益(EPS)。它反映投资者愿意为每一元利润支付多少价格。一般来说，P/E较低可能意味着股票被低估，但也需要结合行业平均水平和公司成长性综合判断。成长型公司通常有较高的P/E，因为投资者预期未来盈利增长。"
    },
    {
        "instruction": "什么是资产负债表？它包含哪些主要部分？",
        "output": "资产负债表是反映企业在某一特定日期财务状况的会计报表。它遵循'资产 = 负债 + 所有者权益'的基本等式，包含三大部分：1）资产（流动资产如现金、应收账款，非流动资产如固定资产、无形资产）；2）负债（流动负债如应付账款，长期负债如长期借款）；3）所有者权益（实收资本、资本公积、留存收益等）。"
    },
    {
        "instruction": "解释央行加息对经济的影响。",
        "output": "央行加息会产生多方面影响：1）借贷成本上升，企业融资和个人贷款（如房贷）变贵，抑制投资和消费；2）存款利率上升，鼓励储蓄，减少市场流动性；3）通常会抑制通货膨胀，因为消费需求减少；4）可能导致股市承压，因为企业盈利预期下降且资金从股市流向固定收益产品；5）本币可能升值，因为高利率吸引外资流入。"
    },
]

print("  金融领域种子样本（共3个）：")
for i, seed in enumerate(finance_seeds, 1):
    print(f"    种子{i}: {seed['instruction'][:40]}...")
print()


# ── 4.2 批量合成数据生成 ─────────────────────────────────────
print("── 4.2 批量合成数据生成 ────────────────────────────────")
print()


def generate_synthetic_data(
    seeds: list,
    domain: str,
    num_samples: int = 5,
    difficulty: str = "中等",
) -> list:
    """
    基于种子样本批量生成合成训练数据。

    参数：
        seeds: 种子样本列表，每个样本包含 instruction 和 output
        domain: 目标领域（如"金融"、"医疗"）
        num_samples: 需要生成的样本数量
        difficulty: 难度级别（简单/中等/困难）
    """
    # 构造 Few-shot 示例
    seed_examples = ""
    for i, seed in enumerate(seeds, 1):
        seed_examples += f"""
示例{i}:
{{"instruction": "{seed['instruction']}", "output": "{seed['output']}"}}
"""

    prompt = f"""你是一位{domain}领域的数据工程师。请参考以下种子样本的风格和质量，生成{num_samples}个新的训练样本。

要求：
1. 内容必须属于{domain}领域
2. 难度级别：{difficulty}
3. 指令要多样化，覆盖不同的知识点
4. 回答要准确、专业、有深度
5. 不要重复种子样本的内容

种子样本参考：
{seed_examples}

请严格按以下JSON数组格式返回{num_samples}个新样本：
[{{"instruction": "...", "output": "..."}}, ...]"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": f"你是{domain}领域的资深专家，同时也是数据工程师。请生成高质量的训练数据，确保内容专业准确。严格按JSON格式输出。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=2000,
    )
    result_text = response.choices[0].message.content.strip()

    # 解析 JSON
    try:
        json_match = re.search(r"\[.*\]", result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass
    return [{"原始返回": result_text}]


# 生成金融领域合成数据
print("  正在生成金融领域合成数据...")
print()
finance_synthetic = generate_synthetic_data(
    seeds=finance_seeds,
    domain="金融",
    num_samples=3,
    difficulty="中等",
)

print("  生成的金融领域合成数据：")
print()
for i, sample in enumerate(finance_synthetic, 1):
    print(f"  合成样本{i}:")
    print(f"    {json.dumps(sample, ensure_ascii=False, indent=4)}")
    print()


# ── 4.3 医疗领域合成数据 ─────────────────────────────────────
print("── 4.3 医疗领域合成数据 ────────────────────────────────")
print()

# 医疗领域种子样本
medical_seeds = [
    {
        "instruction": "解释什么是血红蛋白，以及它在人体中的功能。",
        "output": "血红蛋白(Hemoglobin, Hb)是红细胞中的含铁蛋白质，其主要功能是运输氧气。它由4个亚基组成，每个亚基含有一个血红素基团，可与氧分子可逆结合。在肺部，血红蛋白与氧结合形成氧合血红蛋白；在组织中，由于氧分压降低，氧被释放供细胞使用。正常成人血红蛋白浓度：男性120-160g/L，女性110-150g/L。"
    },
    {
        "instruction": "什么是2型糖尿病？它的主要风险因素有哪些？",
        "output": "2型糖尿病是一种以胰岛素抵抗和相对胰岛素分泌不足为特征的慢性代谢疾病，导致血糖持续升高。主要风险因素包括：1）肥胖（尤其是腹型肥胖）；2）缺乏运动；3）家族遗传史；4）年龄（45岁以上风险增加）；5）高血压和血脂异常；6）妊娠期糖尿病史。管理方式包括饮食控制、规律运动、口服降糖药和必要时使用胰岛素。"
    },
]

print("  正在生成医疗领域合成数据...")
print()
medical_synthetic = generate_synthetic_data(
    seeds=medical_seeds,
    domain="医疗健康",
    num_samples=3,
    difficulty="中等",
)

print("  生成的医疗领域合成数据：")
print()
for i, sample in enumerate(medical_synthetic, 1):
    print(f"  合成样本{i}:")
    print(f"    {json.dumps(sample, ensure_ascii=False, indent=4)}")
    print()


# ── 4.4 完整流水线：合成 + 质量过滤 + 去重 ───────────────────
print("── 4.4 完整流水线：合成 → 质量过滤 → 去重 ──────────────")
print()


def full_synthetic_pipeline(
    seeds: list,
    domain: str,
    num_samples: int = 5,
    quality_threshold: float = 0.6,
) -> list:
    """
    完整的合成数据流水线：
    1. 用 LLM 生成合成数据
    2. 对每条数据进行质量打分
    3. 过滤低质量样本
    4. SimHash 去重
    """
    print(f"    [步骤1] 生成 {num_samples} 条合成数据...")
    synthetic_data = generate_synthetic_data(seeds, domain, num_samples)
    print(f"    [步骤1] 完成，实际生成 {len(synthetic_data)} 条")

    # 质量过滤
    print(f"    [步骤2] 质量评估与过滤（阈值={quality_threshold}）...")
    high_quality = []
    for sample in synthetic_data:
        if "output" in sample:
            scores = compute_quality_scores(sample["output"])
            if scores["综合得分"] >= quality_threshold:
                sample["quality_score"] = scores["综合得分"]
                high_quality.append(sample)
    print(f"    [步骤2] 过滤后剩余 {len(high_quality)} 条")

    # 去重
    print(f"    [步骤3] SimHash 去重...")
    texts = [s.get("output", "") for s in high_quality]
    unique_texts = deduplicate(texts, threshold=10)
    final_data = [s for s in high_quality if s.get("output", "") in unique_texts]
    print(f"    [步骤3] 去重后最终 {len(final_data)} 条")
    print()

    return final_data


# 运行完整流水线
print("  运行完整合成数据流水线（金融领域）：")
print()
final_dataset = full_synthetic_pipeline(
    seeds=finance_seeds,
    domain="金融",
    num_samples=5,
    quality_threshold=0.5,
)

print("  最终数据集：")
for i, sample in enumerate(final_dataset, 1):
    instruction = sample.get("instruction", "N/A")
    quality = sample.get("quality_score", "N/A")
    print(f"    {i}. [质量={quality}] {instruction[:50]}...")
print()


# ── 总结 ──────────────────────────────────────────────────────
print("=" * 60)
print("全部实验完毕！核心总结：")
print("=" * 60)
print("""
  ┌────────────────────────────────────────────────────────────┐
  │  阶段           │ 关键技术              │ 产出              │
  ├────────────────────────────────────────────────────────────┤
  │  数据清洗       │ 正则/HTML解析/SimHash  │ 干净无重复文本    │
  │  质量评估       │ TTR/信息密度/LLM打分   │ 质量分数         │
  │  格式化         │ Alpaca/ShareGPT       │ 标准训练格式      │
  │  合成生成       │ Few-shot/Seed扩展      │ 批量训练数据      │
  └────────────────────────────────────────────────────────────┘

  生产级数据工程的核心原则：
  1. 数据质量 > 数据数量 — 100条高质量数据胜过10000条噪声数据
  2. 流水线自动化 — 清洗→评估→过滤→格式化应该是一键运行的
  3. 版本管理 — 数据集应该像代码一样做版本控制
  4. 持续迭代 — 根据模型表现反馈，不断优化数据配比和质量标准
""")
