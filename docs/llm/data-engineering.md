---
title: 生产级数据工程
---

<script setup>
const code1 = `# MinHash 近似去重检测
import hashlib
import random

def get_shingles(text, k=3):
    """将文本切分为 k-gram shingles"""
    shingles = set()
    for i in range(len(text) - k + 1):
        shingles.add(text[i:i+k])
    return shingles

def minhash_signature(shingles, num_hashes=100):
    """生成 MinHash 签名向量"""
    signature = []
    for seed in range(num_hashes):
        min_hash = float('inf')
        for shingle in shingles:
            # 使用不同种子生成不同哈希函数
            h = int(hashlib.md5(f"{seed}_{shingle}".encode()).hexdigest(), 16)
            min_hash = min(min_hash, h)
        signature.append(min_hash)
    return signature

def jaccard_estimate(sig1, sig2):
    """通过 MinHash 签名估算 Jaccard 相似度"""
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

# 测试数据：模拟近重复文本
documents = [
    "大语言模型通过海量数据训练获得强大的文本生成能力",
    "大语言模型通过海量数据的训练获得了强大的文本生成能力",  # 近似重复
    "强化学习从人类反馈中优化模型的输出质量",
    "强化学习根据人类反馈来优化模型输出的质量",  # 近似重复
    "Transformer架构是现代NLP的基础组件",
]

print("=" * 60)
print("MinHash 近似去重检测")
print("=" * 60)

# 计算每个文档的 MinHash 签名
num_hashes = 50
signatures = []
for doc in documents:
    shingles = get_shingles(doc, k=2)
    sig = minhash_signature(shingles, num_hashes)
    signatures.append(sig)
    print(f"\\n文档: \\"{doc[:20]}...\\"")
    print(f"  Shingles 数量: {len(shingles)}")

# 两两比较相似度
print("\\n" + "-" * 60)
print("相似度矩阵（阈值 > 0.5 判定为近重复）:")
print("-" * 60)

duplicates = []
threshold = 0.5

for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        sim = jaccard_estimate(signatures[i], signatures[j])
        status = "⚠ 近重复!" if sim > threshold else "✓ 不同"
        if sim > threshold:
            duplicates.append((i, j, sim))
        print(f"  文档{i} vs 文档{j}: 相似度={sim:.2f} [{status}]")

print(f"\\n去重结果: 发现 {len(duplicates)} 对近重复文档")
print(f"原始文档数: {len(documents)}")
print(f"去重后文档数: {len(documents) - len(duplicates)}")
`

const code2 = `# 数据质量多维度评分系统
import math

def score_length(text, min_len=10, max_len=2000, ideal_min=50, ideal_max=500):
    """长度维度评分：过短或过长都扣分"""
    length = len(text)
    if length < min_len or length > max_len:
        return 0.0
    if ideal_min <= length <= ideal_max:
        return 1.0
    if length < ideal_min:
        return length / ideal_min
    return max(0.3, 1.0 - (length - ideal_max) / (max_len - ideal_max))

def score_diversity(text):
    """词汇多样性评分：基于字符种类丰富度"""
    if len(text) == 0:
        return 0.0
    unique_chars = len(set(text))
    total_chars = len(text)
    # Type-Token Ratio (TTR)
    ttr = unique_chars / total_chars
    # 中文文本 TTR 通常较高，归一化到 0-1
    score = min(1.0, ttr * 2)
    return score

def score_coherence(text):
    """连贯性评分：基于标点和结构特征"""
    if len(text) == 0:
        return 0.0
    score = 1.0
    # 检查是否有基本标点结构
    punctuation = set('。！？，、；：""''（）')
    punct_count = sum(1 for c in text if c in punctuation)
    punct_ratio = punct_count / len(text)

    # 标点密度过低或过高都扣分
    if punct_ratio < 0.02:
        score -= 0.3  # 缺乏标点
    elif punct_ratio > 0.15:
        score -= 0.2  # 标点过多

    # 检查是否有段落结构
    if '。' in text or '！' in text or '？' in text:
        score += 0.1  # 有句子结构

    # 检查重复模式
    chunks = [text[i:i+10] for i in range(0, len(text)-10, 10)]
    if len(chunks) > 1:
        unique_chunks = len(set(chunks))
        repetition_ratio = unique_chunks / len(chunks)
        if repetition_ratio < 0.5:
            score -= 0.4  # 高度重复

    return max(0.0, min(1.0, score))

def score_information_density(text):
    """信息密度评分：基于信息熵"""
    if len(text) == 0:
        return 0.0
    # 计算字符级信息熵
    freq = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    entropy = 0
    for count in freq.values():
        p = count / len(text)
        entropy -= p * math.log2(p)
    # 归一化（中文文本熵值通常在 4-8 之间）
    normalized = min(1.0, entropy / 7.0)
    return normalized

def evaluate_quality(text, weights=None):
    """综合质量评分"""
    if weights is None:
        weights = {'length': 0.2, 'diversity': 0.25, 'coherence': 0.3, 'density': 0.25}

    scores = {
        'length': score_length(text),
        'diversity': score_diversity(text),
        'coherence': score_coherence(text),
        'density': score_information_density(text),
    }
    total = sum(scores[k] * weights[k] for k in scores)
    scores['total'] = total
    return scores

# 测试样本
samples = [
    ("高质量样本", "大语言模型（LLM）是基于Transformer架构的深度学习模型，通过在海量文本数据上进行预训练，学习语言的统计规律和语义表示。这类模型在自然语言理解、文本生成、代码编写等任务上展现出了强大的能力。"),
    ("低质量-过短", "模型很好"),
    ("低质量-重复", "训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练训练"),
    ("中等质量", "LLM可以做很多事比如写代码写文章做翻译回答问题总结文档等等功能很强大"),
]

print("=" * 60)
print("数据质量多维度评分系统")
print("=" * 60)

for name, text in samples:
    scores = evaluate_quality(text)
    print(f"\\n{'─' * 50}")
    print(f"样本: [{name}]")
    print(f"文本: \\"{text[:40]}...\\"" if len(text) > 40 else f"文本: \\"{text}\\"")
    print(f"  长度评分:   {scores['length']:.2f} {'█' * int(scores['length'] * 10)}")
    print(f"  多样性评分: {scores['diversity']:.2f} {'█' * int(scores['diversity'] * 10)}")
    print(f"  连贯性评分: {scores['coherence']:.2f} {'█' * int(scores['coherence'] * 10)}")
    print(f"  信息密度:   {scores['density']:.2f} {'█' * int(scores['density'] * 10)}")
    print(f"  ────────────────────────────")
    print(f"  综合得分:   {scores['total']:.2f} {'█' * int(scores['total'] * 10)}")

    # 质量等级判定
    if scores['total'] >= 0.8:
        level = "A (优质)"
    elif scores['total'] >= 0.6:
        level = "B (合格)"
    elif scores['total'] >= 0.4:
        level = "C (待改进)"
    else:
        level = "D (不合格)"
    print(f"  质量等级:   {level}")

print(f"\\n{'═' * 60}")
print("评分标准说明:")
print("  A级 (>=0.8): 直接用于训练")
print("  B级 (>=0.6): 轻度清洗后可用")
print("  C级 (>=0.4): 需要改写或补充")
print("  D级 (<0.4):  建议丢弃")
`
</script>

# 生产级数据工程

在大模型训练和微调过程中，**数据质量直接决定模型能力上限**。本章系统介绍生产环境中数据工程的核心流程与最佳实践。

::: tip 核心原则
数据工程遵循 "Garbage In, Garbage Out" 原则。高质量的 10K 数据往往优于低质量的 100K 数据。
:::

## 数据工程全景流程

```
┌─────────────────────────────────────────────────────────┐
│                    数据工程流水线                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  数据采集 → 数据清洗 → 质量评估 → 格式转换 → 数据增强    │
│     │          │          │          │          │       │
│     ▼          ▼          ▼          ▼          ▼       │
│  爬虫/API   去重/过滤   打分/筛选   SFT格式   合成数据    │
│  公开数据集  规范化      人工审核    对话格式   回译增强    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 一、数据清洗流程

### 1.1 去重策略

数据去重是数据清洗的第一步，需要处理精确重复和近似重复两种情况：

| 去重方法 | 适用场景 | 复杂度 | 精确度 |
|---------|---------|--------|--------|
| MD5/SHA256 哈希 | 精确去重 | O(n) | 100% |
| MinHash + LSH | 近似去重（长文本） | O(n log n) | ~95% |
| SimHash | 近似去重（短文本） | O(n) | ~90% |
| N-gram 重叠率 | 段落级去重 | O(n²) | ~85% |
| 编辑距离 | 小规模精确比对 | O(n²m²) | 100% |

<PythonRunner :code="code1" title="MinHash 近似去重检测" />

### 1.2 质量过滤规则

```
过滤规则优先级（从高到低）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 有害内容过滤（安全红线）
2. 语言检测（剔除乱码/非目标语言）
3. 长度过滤（过短无信息量，过长噪声多）
4. 重复率检查（段落/句子级别）
5. 特殊字符比例（HTML标签、控制字符）
6. 困惑度过滤（PPL过高表示质量差）
```

### 1.3 格式统一

::: info 格式统一检查清单
- 统一编码为 UTF-8
- 移除多余空白字符和控制字符
- 标准化标点符号（全角/半角统一）
- 统一换行符（`\r\n` → `\n`）
- URL、邮箱、电话号码脱敏处理
- 数字格式标准化
:::

## 二、合成数据生成

### 2.1 用 GPT-4 生成垂直领域数据

合成数据是解决垂直领域数据稀缺的核心方法：

```
合成数据生成流程：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  种子数据 (50-100条人工标注)
       │
       ▼
  Prompt 模板设计
  (包含领域知识、格式要求、质量标准)
       │
       ▼
  GPT-4 批量生成
  (控制 temperature, 多样性采样)
       │
       ▼
  质量过滤 + 人工抽检 (10-20%)
       │
       ▼
  迭代优化 Prompt → 重新生成
```

### 2.2 合成数据 Prompt 模板示例

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| Self-Instruct | 让模型自己生成指令-回答对 | 通用能力数据 |
| Evol-Instruct | 逐步增加指令复杂度 | 复杂推理数据 |
| 回译增强 | 翻译成其他语言再翻回来 | 多样性增强 |
| 角色扮演 | 设定不同专家角色生成 | 领域知识数据 |
| 对抗生成 | 让模型找自己回答的漏洞 | 安全对齐数据 |

::: warning 注意事项
合成数据容易产生"模型幻觉循环"——用模型 A 生成数据训练模型 B，再用 B 生成数据训练 C，质量会逐代退化。需要人工数据锚定。
:::

## 三、SFT 数据格式

### 3.1 Alpaca 格式

最简单的单轮指令微调格式：

```json
{
  "instruction": "解释什么是反向传播算法",
  "input": "",
  "output": "反向传播（Backpropagation）是训练神经网络的核心算法..."
}
```

### 3.2 ShareGPT 格式

多轮对话格式，适合对话模型微调：

```json
{
  "conversations": [
    {"from": "system", "value": "你是一个AI助手"},
    {"from": "human", "value": "什么是Transformer？"},
    {"from": "gpt", "value": "Transformer是一种基于注意力机制的..."},
    {"from": "human", "value": "它和RNN有什么区别？"},
    {"from": "gpt", "value": "主要区别有三点：1. 并行性..."}
  ]
}
```

### 3.3 格式对比

| 特性 | Alpaca | ShareGPT | OpenAI Messages |
|------|--------|----------|----------------|
| 多轮对话 | 不支持 | 支持 | 支持 |
| System Prompt | 不支持 | 支持 | 支持 |
| 工具调用 | 不支持 | 部分支持 | 完整支持 |
| 生态兼容性 | LLaMA系 | 广泛 | OpenAI系 |
| 适合场景 | 简单指令 | 对话微调 | 函数调用 |

## 四、数据质量评估指标

### 4.1 评分维度

<PythonRunner :code="code2" title="数据质量多维度评分系统" />

### 4.2 自动化评估指标体系

| 维度 | 指标 | 计算方法 | 阈值建议 |
|------|------|---------|---------|
| 长度 | 字符数/Token数 | 直接统计 | 50-2000字符 |
| 多样性 | TTR / Distinct-N | 唯一N-gram占比 | TTR > 0.3 |
| 连贯性 | 困惑度(PPL) | 语言模型评分 | PPL < 100 |
| 信息量 | 信息熵 | Shannon Entropy | > 4.0 bits |
| 相关性 | 余弦相似度 | 与领域种子集比对 | > 0.6 |
| 安全性 | 有害分类器 | 安全模型打分 | 安全概率 > 0.95 |

### 4.3 人工评估与自动评估结合

```
评估流水线：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  全量数据 → 自动评分 → 分桶
                         │
            ┌────────────┼────────┐
            ▼            ▼        ▼
        高质量桶      中等桶    低质量桶
        (自动通过)  (抽样人审)  (自动丢弃)
            │            │
            ▼            ▼
         合并为最终训练集
```

## 五、数据标注工具与流程

### 5.1 常用标注工具

| 工具 | 类型 | 特点 | 适用场景 |
|------|------|------|---------|
| Label Studio | 开源 | 支持多模态、可自部署 | 中小团队 |
| Doccano | 开源 | 轻量级、NLP专用 | 文本标注 |
| Prodigy | 商业 | 主动学习、高效 | 专业NLP团队 |
| Scale AI | 平台 | 人力外包+质检 | 大规模标注 |
| Argilla | 开源 | LLM反馈专用 | RLHF数据 |

### 5.2 标注质量控制

::: tip 标注一致性保障
1. **标注指南**：编写详细的标注规范文档（含正反例）
2. **试标注**：正式开始前进行小批量试标注校准
3. **双人标注**：关键数据由两人独立标注后对比
4. **Kappa 系数**：计算标注者间一致性（目标 > 0.8）
5. **定期校准**：每周会议讨论争议样本
:::

### 5.3 标注流程管理

```
标注管理流程：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  任务拆分 → 分配 → 标注 → 质检 → 汇总
     │         │      │      │      │
     ▼         ▼      ▼      ▼      ▼
  按难度分级  负载均衡  规范遵守 抽检+仲裁 版本管理
  标注指南   专长匹配  实时反馈  一致性分析 变更追踪
```

## 六、领域数据治理

### 6.1 数据生命周期管理

| 阶段 | 关键活动 | 产出物 |
|------|---------|--------|
| 规划 | 需求分析、数据源调研 | 数据采集方案 |
| 采集 | 爬虫/API/合作获取 | 原始数据集 |
| 处理 | 清洗、去重、标注 | 清洗后数据集 |
| 存储 | 版本化、元数据管理 | 数据仓库 |
| 使用 | 训练、评估、迭代 | 模型产出 |
| 归档 | 过期数据处理、合规审计 | 审计报告 |

### 6.2 数据版本管理

::: info 推荐实践
- 使用 DVC (Data Version Control) 管理大规模数据集版本
- 每次数据变更记录：变更原因、影响范围、负责人
- 保留数据血缘关系（lineage）：从原始数据到最终训练集的完整链路
- 建立数据卡片（Data Card）：记录数据集的统计特征和已知偏差
:::

### 6.3 数据合规与安全

```
数据治理检查点：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[√] 个人信息脱敏（姓名、手机、身份证）
[√] 版权合规审查（训练数据授权）
[√] 数据偏差评估（性别、地域、年龄）
[√] 有害内容筛查（暴力、歧视、违法）
[√] 数据访问权限控制（最小权限原则）
[√] 审计日志完整性（谁在何时访问了什么）
```

## 总结

| 环节 | 核心目标 | 关键指标 |
|------|---------|---------|
| 数据清洗 | 消除噪声和冗余 | 去重率、过滤率 |
| 合成数据 | 补充稀缺领域数据 | 多样性、真实性 |
| 格式标准化 | 适配训练框架 | 格式合规率 |
| 质量评估 | 量化数据价值 | 综合质量分 |
| 标注管理 | 保证标注一致性 | Kappa系数 |
| 数据治理 | 合规+可追溯 | 审计通过率 |

::: tip 实战建议
1. 先建立数据质量基线，再做增量优化
2. 自动化流水线优先，人工审核兜底
3. 小批量快速迭代，避免一次性大规模清洗
4. 建立数据飞轮：模型输出 → 筛选 → 回流训练集
:::
