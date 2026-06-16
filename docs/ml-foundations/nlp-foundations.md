# NLP 经典基础

> 分词、TF-IDF、Word2Vec —— LLM 之前，文本是这样被处理的

---

## 一、为什么 NLP 比图像难？

```
图像                            文本
─────                          ─────
原始数据 = 像素矩阵              原始数据 = 字符序列
天然连续 / 数值化                天然离散 / 符号化
平移不变（猫平移仍是猫）          位置敏感（"狗咬人" ≠ "人咬狗"）
全局信息容易聚合                  长程依赖普遍且重要
```

**核心难点**：把离散的"符号"映射到模型能处理的"数值表示"，且保留语义信息。

LLM 出现之前，这条路走了 30 年：**词袋 → TF-IDF → Word2Vec → ELMo → BERT → GPT**。

---

## 二、文本预处理流水线

### 2.1 英文流水线

```
原文：The cats were running quickly.
   ↓ 小写化
the cats were running quickly.
   ↓ 分词（tokenization）
["the", "cats", "were", "running", "quickly", "."]
   ↓ 去标点
["the", "cats", "were", "running", "quickly"]
   ↓ 去停用词（the/a/is/...）
["cats", "running", "quickly"]
   ↓ 词干化（stemming，粗暴切后缀）
["cat", "run", "quickli"]      ← Porter Stemmer
   ↓ 词形还原（lemmatization，基于词典）
["cat", "run", "quickly"]      ← WordNet Lemmatizer
```

| 操作 | 词干化 | 词形还原 |
|------|--------|---------|
| **方法** | 规则后缀剥离 | 词典查询 + 词性 |
| **速度** | 快 | 慢 |
| **质量** | 粗糙（"quickli"） | 准确（"quickly"） |
| **典型工具** | PorterStemmer | WordNetLemmatizer |

### 2.2 中文流水线（关键挑战：分词）

```
原文：机器学习是人工智能的分支
   ↓ 分词（无空格，必须靠算法）
["机器学习", "是", "人工智能", "的", "分支"]
   ↓ 去停用词
["机器学习", "人工智能", "分支"]
```

::: warning 中文分词的歧义难题
- "**结合上海大学**的研究" → "结合 / 上海大学" 还是 "结合 / 上海 / 大学"？
- "**南京市长江大桥**" → "南京市 / 长江大桥" 还是 "南京 / 市长 / 江大桥"？
- "**他从马上下来**" → "从 / 马上 / 下来" 还是 "从 / 马 / 上下来"？

**jieba** 用 HMM + 词典联合方法，处理大多数常见歧义。学术界更准确的工具：THULAC / pkuseg / LAC。
:::

::: tip LLM 视角
**BPE（Byte Pair Encoding）/ WordPiece 直接学子词单元，绕过了"中文分词"这个传统难题。**

```
传统中文 NLP：    "机器学习" → jieba → ["机器学习", "是", ...]
LLM tokenizer：  "机器学习" → BPE → ["机器", "学习"] 或更细
```

但中文 RAG 的 BM25 检索、关键词提取、命名实体识别仍然需要传统分词。
:::

---

## 三、词袋（Bag of Words）

最朴素的表示：**统计每个词出现了多少次**，丢弃语序。

```
文档1: "I love cats and dogs"
文档2: "I love dogs"

词表: [I, love, cats, and, dogs]
文档1: [1, 1, 1, 1, 1]
文档2: [1, 1, 0, 0, 1]
```

**优点**：简单、可解释
**缺点**：
- 维度爆炸（词表大）
- 矩阵极度稀疏
- 完全丢失语序与语义

---

## 四、TF-IDF：词袋的进化

### 4.1 直觉

不是所有词都同等重要：
- **the / a / is** —— 在所有文档都出现 → 没区分力
- **量子纠缠** —— 只在物理论文出现 → 强区分力

### 4.2 公式

$$
\text{TF}(t, d) = \frac{t \text{ 在 } d \text{ 中出现次数}}{d \text{ 总词数}}
$$

$$
\text{IDF}(t) = \log\frac{N}{|\{d : t \in d\}|}
$$

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)
$$

含义：
- **TF** = 在本文档里多重要
- **IDF** = 在整个语料里多稀有
- 相乘 = 在**这篇**里出现很多但**其他篇**里少 → 最有判别力

### 4.3 sklearn 用法

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

vec = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),  # uni + bigram
    max_features=10000,
    min_df=2,
)
X = vec.fit_transform(corpus)
clf = MultinomialNB().fit(X, y)
```

### 4.4 BM25：TF-IDF 的工业级改进

BM25 在 TF-IDF 基础上做了两点关键改动：

1. **TF 饱和**：词出现 100 次不应该比出现 10 次重要 10 倍
2. **文档长度归一化**：长文档天然词多，要做惩罚

$$
\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\overline{|d|}})}
$$

::: tip LLM 关联：现代 RAG 的混合检索
**BM25 远未被淘汰**，反而是工业级 RAG 的标配组件：

```
                  ┌─────── BM25 (稀疏) ───────┐
查询 query ───────┤                            ├──→ Reranker → top-k
                  └─── 向量检索 (密集) ──────┘
```

**为什么需要 BM25？**
- **专有名词**：人名、产品 ID、API 名 —— 向量模型常常学不准
- **精确匹配**：用户搜"GPT-4-turbo-2024-04-09" —— 向量召回不如关键词
- **可解释性**：能解释"为什么召回了这条"

实际部署：Elasticsearch / Qdrant / Weaviate 都内置 BM25 + 向量的混合检索。
:::

---

## 五、Word2Vec：从离散到稠密

### 5.1 核心思想：分布假设

> "你认识一个词，是因为你认识它的邻居。" —— J.R. Firth, 1957

意思相近的词出现在相似的上下文里：
- "苹果" 和 "橙子" 的上下文（颜色、味道、水果店）重合度高
- 所以它们的向量也应该接近

### 5.2 两种训练目标

```
Skip-Gram（更常用）：给中心词，预测上下文
   "机器 [学习] 是 人工智能"
            ↑
         中心词       目标：学习→机器，学习→是，等

CBOW：给上下文，预测中心词
   "机器 ___ 是 人工智能"  →  预测 [学习]
```

### 5.3 损失函数

简化形式（softmax 版本）：

$$
L = -\log P(o \mid c) = -\log \frac{\exp(v_o^\top v_c)}{\sum_w \exp(v_w^\top v_c)}
$$

**问题**：分母对整个词表（10万+）求和 → 计算爆炸。

**解决**：负采样（Negative Sampling）—— 只对少量（5-20 个）负样本计算。

### 5.4 神奇现象：语义算术

训练完后会发现：

```
king - man + woman ≈ queen
Paris - France + Italy ≈ Rome
walking - walk + swim ≈ swimming
```

::: tip 为什么会这样？
词向量在某些方向上**自动编码**了语义关系：
- 一个方向 ≈ "性别"
- 另一个方向 ≈ "国家 → 首都"
- 还有方向 ≈ "动词时态"

这是**表示学习的"涌现"**：你只让它做"预测上下文"，但它学到了远不止于此的结构。

这种"涌现"在 LLM 中变得更夸张 —— GPT-3 涌现出推理、翻译、代码生成等能力，而它的训练目标依然只是"next-token prediction"。
:::

### 5.5 静态 vs 动态嵌入

```
Word2Vec（静态）：
   "I deposited money in the bank"
   "I sat by the river bank"
        ↓
   两个 "bank" 的向量完全相同 —— 错！

LLM（动态/contextual）：
   两个 "bank" 经过 Transformer 后，向量完全不同
   （注意力让每个 token 看到自己的上下文）
```

::: tip 一脉相承的关系
- **Word2Vec embedding** = LLM embedding 层的精神祖先
- **LLM 的 token embedding** = 学到的可训练查表
- **LLM 的隐层输出** = "动态 embedding"，解决了 Word2Vec 的一词多义问题

你今天用的 OpenAI `text-embedding-3` / BGE 模型，本质上是 BERT 的 [CLS] 输出 → 但其训练目标依然继承自 Word2Vec 的"分布假设"。
:::

---

## 六、传统文本分类：TF-IDF + 朴素贝叶斯/SVM

### 6.1 朴素贝叶斯

基于贝叶斯定理的"朴素"假设：**特征之间条件独立**。

$$
P(c \mid d) \propto P(c) \prod_{t \in d} P(t \mid c)
$$

虽然"独立"假设在文本里**完全不成立**，但实际效果出奇地好，原因：
- 决策只看 argmax，而不是精确概率
- 文本类别差异往往主要由**词出现/不出现**主导

### 6.2 性能对比（20-newsgroups 4 类）

| 模型 | 准确率 | 训练时间 | 推理速度 |
|------|--------|---------|---------|
| MultinomialNB | ~85% | 极快 | 极快 |
| LinearSVC | ~88% | 快 | 快 |
| LogisticRegression | ~88% | 中 | 快 |
| **BERT 微调** | ~93% | 慢（GPU） | 慢 |
| **GPT-4 zero-shot** | ~94% | 无 | 慢（API） |

::: warning 现实选型建议
**不要无脑上 LLM**：
- 数据量大、类别清晰 → TF-IDF + 线性分类器 性价比最高
- 数据少、类别细 → BERT 系列微调
- 数据极少、需要快速验证 → LLM zero/few-shot

很多生产环境的"看似 LLM"系统，底层依然是 BM25 + 简单分类器。
:::

---

## 七、文本相似度

### 7.1 余弦相似度（最常用）

$$
\cos(\theta) = \frac{a \cdot b}{\|a\| \|b\|}
$$

```
两个向量
  ↗
 ╱       cos = 1：完全相同方向（最相似）
╱        cos = 0：垂直（无关）
        cos = -1：相反方向（最不相似）
```

为什么不用欧氏距离？
- 文档长度不同 → 向量大小不同
- 余弦只看**方向**，对长度不敏感

### 7.2 Jaccard 相似度（集合）

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

适合：词集合重叠度（去重、相似文档检测）。

### 7.3 编辑距离（字符级）

把字符串 A 变成 B 最少需要的"插入/删除/替换"次数。

适合：拼写检查、模糊匹配、DNA 序列对齐。

---

## 八、信息检索（IR）的演进

```
布尔检索 (1960s)
   "AND OR NOT" → 精确但太死板
        ↓
向量空间模型 + TF-IDF (1970s-)
   余弦相似度排序 → 至今仍是 baseline
        ↓
BM25 (1990s)
   TF 饱和 + 长度归一化 → 工业 IR 标准
        ↓
词嵌入检索 (2013-)
   Word2Vec/GloVe 平均 → 简单但有效
        ↓
密集向量检索 (2019-)
   BERT/Sentence-BERT 编码 → 现代 RAG 主力
        ↓
混合检索 (2023-)
   BM25 + 向量 + Reranker → 工业最佳实践
```

::: tip RAG 的真相
现代 RAG 不是"纯向量检索"，而是**老办法 + 新办法 + 排序模型**的组合：

```
查询
  ↓
┌────── BM25 (top 100) ──────┐
│                             │
└─ 向量检索 (top 100) ───────┤
                              ↓
                        Cross-encoder Reranker
                              ↓
                          top 10
                              ↓
                          LLM 生成答案
```

每一层都用最适合的技术 —— 这就是为什么本章经典 NLP 内容仍然重要。
:::

---

## 九、配套代码

| 文件 | 演示主题 |
|------|---------|
| [`text_preprocessing.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/nlp_foundations/text_preprocessing.py) | 中英文预处理流水线 + 中文分词歧义 |
| [`tfidf_classification.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/nlp_foundations/tfidf_classification.py) | TF-IDF + 朴素贝叶斯/SVM 新闻分类 |
| [`word2vec_demo.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/nlp_foundations/word2vec_demo.py) | Word2Vec 训练 + 类比推理 + PCA 可视化 |

---

## 十、延伸阅读

- Jurafsky & Martin *Speech and Language Processing* —— NLP 圣经
- 宗成庆《统计自然语言处理》—— 中文经典
- [Stanford CS224N](https://web.stanford.edu/class/cs224n/) —— 从 Word2Vec 到 Transformer
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) —— 可视化讲解

> **下一站**：[ML 与 LLM 的关系](./ml-vs-llm) —— 站在更高的视角看演进
