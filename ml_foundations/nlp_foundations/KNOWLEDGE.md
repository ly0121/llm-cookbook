# 经典 NLP 基础（NLP Foundations）

> 在 Transformer 之前，自然语言处理是怎么做的？这些方法仍是 LLM 时代的有用工具

---

## 一、经典 NLP 流水线

在 GPT / Llama 出现之前，主流 NLP 应用要经过这条流水线：

```
原始文本
   ↓
分词 (Tokenization)
   ↓
预处理 (大小写 / 停用词 / 词干化 / 词形还原)
   ↓
特征化 (词袋 / TF-IDF / Word2Vec)
   ↓
分类器 (SVM / 朴素贝叶斯 / 逻辑回归)
   ↓
预测 (情感 / 主题 / 实体 ...)
```

LLM 时代很多步骤被合并到端到端模型里了，但**理解每一步在解决什么问题**对调好 RAG / 提示工程 / 数据清洗仍然必要。

---

## 二、文本表示：从词袋到词向量

### 2.1 词袋模型（Bag of Words, BoW）

最简单的文本数值化：

```
"The cat sat on the mat"
词表 = [the, cat, sat, on, mat]
向量 = [2, 1, 1, 1, 1]
```

**问题**：
- 完全忽略词序（"猫 追 老鼠" = "老鼠 追 猫"）
- 没有语义信息（"高兴" 和 "开心" 完全无关）
- 维度爆炸（词表 10 万 → 向量 10 万维）

### 2.2 TF-IDF（Term Frequency – Inverse Document Frequency）

BoW 的"加权升级版"。核心想法：**在某个文档中出现多但在所有文档中出现少的词，就是关键词**。

$$
\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}}
$$

$$
\text{IDF}(t) = \log\frac{N}{|\{d : t \in d\}|}
$$

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)
$$

**举例**：在 1000 篇科技新闻里：
- "the" 在每篇都出现 → IDF ≈ 0 → 权重低
- "transformer" 只在 50 篇出现 → IDF ≈ 3 → 权重高

> **LLM 时代的应用**：BM25（TF-IDF 的改进版）在 RAG 中作为"稀疏检索"和"密集检索（向量）"互补使用。**混合检索（hybrid search）= BM25 + 向量** 已经是工业级 RAG 的标配。

### 2.3 Word2Vec（2013）：词向量的革命

**核心思想**："你认识一个词，是因为你认识它的邻居"（distributional hypothesis）。

#### Skip-Gram 训练目标

给定中心词，预测上下文词：

```
"机器 学习 是 人工智能 的 一个 分支"
                ↑
            中心词
预测: 学习, 是, 的, 一个   (window=2)
```

通过最大化条件概率，学出每个词的稠密向量（通常 100-300 维）。

#### 神奇的语义算术

```
king - man + woman ≈ queen
Paris - France + Italy ≈ Rome
```

**为什么？** 因为词向量在某些方向上编码了语义关系（性别、国家-首都）。

#### 与 LLM Embedding 的关系

| | Word2Vec (2013) | LLM Embedding (2024) |
|---|---|---|
| 训练目标 | 预测上下文 | 多任务（next-token / 对比学习） |
| 上下文 | 固定窗口 | 整个序列（自注意力） |
| 静态/动态 | **静态**：每个词一个向量 | **动态**：同一个词在不同句子里向量不同 |
| 维度 | 100-300 | 768-3072 |

> **关键洞察**：BERT/GPT 的 token embedding **延续了 Word2Vec 的精神**，只是把"固定窗口共现"换成了"任意位置 attention"。

---

## 三、文本预处理（中英文都需要）

### 3.1 英文预处理

| 步骤 | 例子 |
|------|------|
| 小写化 | `"Apple"` → `"apple"` |
| 分词 | `"don't go!"` → `["do", "n't", "go", "!"]` |
| 去停用词 | 删除 `the, is, a, of, ...` |
| 词干化（Stemming） | `running, ran, runs` → `run`（粗暴砍后缀） |
| 词形还原（Lemmatization） | `better` → `good`（基于词典） |

### 3.2 中文预处理

中文最大的挑战：**没有空格分词**。

```
"我爱北京天安门"
   ↓ 分词
"我 / 爱 / 北京 / 天安门"
```

主流分词工具：
- **jieba**：开源最广，支持 HMM + 用户词典
- **THULAC / pkuseg**：学术派，准确率更高
- **LAC**（百度）：工业级

> **LLM 时代变化**：BPE / WordPiece tokenizer 直接学子词单元，**不需要中文分词**了！但中文搜索 / 信息提取等场景，分词仍然重要。

### 3.3 工具速查

| 任务 | 英文 | 中文 |
|------|------|------|
| 分词 | nltk.word_tokenize | jieba.cut |
| 去停用词 | nltk.corpus.stopwords | 自定义停用词表 |
| 词干化 | nltk.PorterStemmer | （中文无对应概念） |
| 词形还原 | nltk.WordNetLemmatizer | （中文用同义词词典） |
| POS 标注 | nltk.pos_tag | jieba.posseg |

---

## 四、文本分类：经典任务

### 4.1 朴素贝叶斯（Naive Bayes）

基于贝叶斯定理 + 特征独立假设：

$$
P(c | x) \propto P(c) \prod_{i} P(x_i | c)
$$

为什么对文本特别有效？
- 训练快（统计每个词在每类中的频率即可）
- 即使"特征独立"假设不成立，效果仍很不错
- 适合高维稀疏特征（TF-IDF 向量）

### 4.2 SVM（支持向量机）

寻找一个超平面，使两类样本到超平面的最小距离最大：

```
       .  o  o
     .  .  o  o      → SVM 找到这条线让间隔最大
   .  .       o  o
 .                 o  o
```

文本分类时通常用线性核（特征已经在高维空间）。

### 4.3 LLM 时代的文本分类

```python
# 经典方法
sklearn 的 TF-IDF + LogisticRegression  → 5 行代码,1 秒训练

# LLM 方法
prompt = f"判断以下文本的情感(正面/负面): {text}"
response = openai.chat.completions.create(...)  → 0 行训练,0.5 秒推理
```

**何时用经典方法？**
- 数据量大（百万条以上），LLM 推理太慢/太贵
- 类别固定且简单（垃圾邮件、情感二分类）
- 需要可解释性（哪些词导致分类）

**何时用 LLM？**
- 数据少（few-shot）
- 类别复杂或需要理解上下文
- 类别动态变化

---

## 五、命名实体识别（NER）的演进

```
"Apple released the iPhone in 2007"
  ↓
[Apple|ORG] released the [iPhone|PRODUCT] in [2007|DATE]
```

**演进路线**：

| 时代 | 方法 | 特点 |
|------|------|------|
| 1990s | 规则 + 词典 | 写不完，泛化差 |
| 2000s | HMM / CRF | 统计模型，需特征工程 |
| 2010s | BiLSTM + CRF | 端到端，但训练慢 |
| 2018+ | BERT 微调 | 预训练 + 任务头，成为标杆 |
| 2023+ | LLM zero-shot | 直接 prompt 提取 |

> **现状**：通用领域用 LLM；专业领域（医疗、法律）仍倾向 BERT 微调（精度更高）。

---

## 六、信息检索：从 BM25 到向量检索

### 6.1 BM25（TF-IDF 的改进）

加入：
- 文档长度归一化（避免长文档总是赢）
- TF 饱和（出现 1000 次和 100 次的"机器学习"重要性差不多）

仍然是稀疏检索的事实标准（ElasticSearch / Lucene 默认）。

### 6.2 稠密向量检索

```
查询 → embedding 模型 → 768 维向量 → 在向量库中找最近邻
```

优点：能找到**语义相似但字面不同**的结果。

缺点：对**关键词精确匹配**反而不如 BM25。

### 6.3 现代 RAG 的做法：混合检索

```
查询 ──┬── BM25 → 候选集 1
       └── 向量 → 候选集 2
            ↓
         融合（RRF / 线性加权）→ Reranker → 最终结果
```

**这就是为什么本章 TF-IDF 仍然重要** —— 它是混合检索的一半。

---

## 七、本目录 demo 速查

| 文件 | 主题 | 关键 API |
|------|------|---------|
| `text_preprocessing.py` | 中英文预处理对比 | jieba.cut, nltk.word_tokenize, PorterStemmer |
| `tfidf_classification.py` | 20-newsgroups 子集分类 | TfidfVectorizer, MultinomialNB, LinearSVC |
| `word2vec_demo.py` | gensim 训练 + 类比推理 | Word2Vec, most_similar, similarity |

---

## 八、与 LLM 的衔接

完成本章后，你能更深刻理解：

| 经典 NLP 概念 | LLM 中的对应 |
|--------------|--------------|
| Tokenization | BPE / WordPiece / SentencePiece |
| 词袋 → 上下文向量 | static embedding → contextual embedding |
| TF-IDF 文档检索 | BM25 + dense vector hybrid search |
| Word2Vec 语义算术 | LLM embedding 仍保留这种线性结构 |
| NER 任务头 | LLM in-context extraction（prompt 提取） |
| 朴素贝叶斯先验 | LLM 的 "system prompt" 可视为先验 |

---

## 九、延伸阅读

- Jurafsky & Martin *Speech and Language Processing*（NLP 圣经）
- Manning et al. *Introduction to Information Retrieval*
- Mikolov 2013 Word2Vec 原始论文
- BM25 原论文：Robertson & Zaragoza 2009
- jieba 文档：https://github.com/fxsjy/jieba
- nltk 教程：https://www.nltk.org/book/

> **下一站**：阅读 `../../llm/tokenization_demo.py` 看 LLM 时代的 BPE 分词，然后 `../../rag/rag_qa.py` 看 TF-IDF 思想如何在 RAG 中被使用。
