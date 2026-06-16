# 经典机器学习

> 监督学习、无监督学习、特征工程、模型评估 —— ML 的"基本盘"

---

## 一、什么是机器学习？

> **机器学习**：让计算机从数据中"学习"规律，而不是显式地写规则。

### 1.1 范式转换

```
传统编程                       机器学习
─────────────                  ─────────────
输入 + 规则 → 输出              输入 + 输出 → 规则（模型）

写垃圾邮件过滤：                 训练垃圾邮件过滤：
if 含"中奖" then 垃圾            给 1 万封标注邮件，让模型自己学
if 含"链接" then 垃圾
if 来自陌生人 then 垃圾
... 写不完的规则
```

### 1.2 三大范式

| 范式 | 数据形式 | 目标 | 例子 | LLM 中的对应 |
|------|---------|------|------|--------------|
| **监督学习** | (X, y) 有标签 | 学 X→y 的映射 | 房价预测、垃圾邮件分类 | next-token prediction (SFT) |
| **无监督学习** | 只有 X | 发现内在结构 | 用户分群、异常检测 | LLM 预训练（自监督） |
| **强化学习** | 状态-动作-奖励 | 学最优策略 | AlphaGo、机器人 | RLHF、PPO/DPO |

LLM 训练横跨三大范式：**预训练（自监督）→ SFT（监督）→ RLHF（强化）**。

---

## 二、监督学习：分类

### 2.1 逻辑回归（Logistic Regression）

最简单也最重要的分类器。

#### 数学形式

线性模型 + sigmoid 把输出压到 [0, 1]：

$$
P(y=1 \mid x) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}
$$

#### 损失函数：二元交叉熵

$$
L = -\frac{1}{N}\sum_{i=1}^{N}\Big[y_i \log p_i + (1-y_i)\log(1-p_i)\Big]
$$

::: tip LLM 视角
**LLM 的 next-token prediction 本质上是一个超大规模的多分类逻辑回归**：

- 输入：上下文 token 序列经过 Transformer 编码后的最后一个隐状态 $h$
- 输出：词表大小（V ≈ 5万-15万）的概率分布
- 形式：$P(\text{token}_t \mid \text{context}) = \text{softmax}(W \cdot h)$
- 损失：交叉熵

数学上完全相同，只是规模放大了几个数量级。
:::

#### 代码示例

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# 看权重了解每个特征的重要性
print(model.coef_)
```

### 2.2 决策树（Decision Tree）

逐层用一个特征切分数据，直到叶节点足够"纯"。

```
                 收入 > 50K?
                /          \
              是            否
            /    \         /    \
        年龄>30  ...    教育=本科  ...
       /     \
    买房    租房
```

#### 信息增益（用于选择切分特征）

$$
\text{Gain}(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)
$$

其中 $H(S) = -\sum_i p_i \log p_i$ 是熵。

#### 优缺点

✅ **优点**：可解释（人类能读懂规则）、不需要特征缩放、能处理类别特征
❌ **缺点**：容易过拟合、对训练数据扰动敏感

### 2.3 随机森林（Random Forest）

100 棵决策树投票。每棵树用：
1. **Bagging**：从训练集中有放回抽样
2. **特征随机**：每个分裂节点只考虑随机子集

#### 为什么 work？

单棵树过拟合（高方差），但**独立**的多棵树误差互相抵消 → 集成后方差大幅降低。

```
单棵树误差：       ████████████░░  方差大
100 棵投票：       ███░░░░░░░░░░░  方差小
```

::: tip LLM 关联
MoE（Mixture of Experts）架构 —— 一个 token 由多个"专家网络"投票产生 —— 与随机森林精神一致：**多个弱专家组合成强系统**。

GPT-4、Mixtral 8x7B、DeepSeek-V3 都用 MoE。
:::

### 2.4 三模型对比

| 模型 | 训练速度 | 推理速度 | 可解释性 | 精度 | 适用场景 |
|------|---------|---------|---------|------|---------|
| 逻辑回归 | 极快 | 极快 | 高（看权重） | 中 | baseline、特征工程主导 |
| 决策树 | 快 | 快 | 高（看规则） | 中 | 可解释性需求 |
| 随机森林 | 中 | 中 | 中（特征重要性） | 高 | 通用 baseline |
| **XGBoost / LightGBM** | 中 | 中 | 中 | **极高** | Kaggle、表格数据冠军 |

> **现实**：在结构化数据（表格）上，**梯度提升树（XGBoost / LightGBM）至今仍打败深度学习**。深度学习的优势主要在非结构化数据（图像、文本、音频）。

---

## 三、监督学习：回归

### 3.1 线性回归（OLS）

$$
\hat{y} = w^\top x + b, \quad L = \frac{1}{N}\sum_i (y_i - \hat{y}_i)^2
$$

闭式解：$w = (X^\top X)^{-1} X^\top y$。

**问题**：当特征 > 样本，或特征间高度相关，$X^\top X$ 不可逆 → 解不稳定 → 需要正则化。

### 3.2 岭回归（Ridge / L2）

$$
L = \|y - Xw\|^2 + \lambda \|w\|_2^2
$$

L2 惩罚让所有权重"小一点"，但不会归零。

### 3.3 Lasso（L1）

$$
L = \|y - Xw\|^2 + \lambda \|w\|_1
$$

L1 惩罚让部分权重**直接归零** → 自动特征选择。

### 3.4 几何直觉

```
       Ridge (L2)             Lasso (L1)
       约束=圆               约束=菱形

         ╭──╮                  ╱╲
        │ ●○│                 ╱○●╲     ← ● 是无约束最优
        │○ │                 ╲   ╱
         ╰──╯                  ╲╱
                              （角点更易被命中 → 系数稀疏）

  约束让最优解被"拉向圆心/菱形"
```

::: tip LLM 关联
**LoRA（Low-Rank Adaptation）** 微调：在原权重 $W$ 旁加一个低秩增量 $\Delta W = AB^\top$（$A \in \mathbb{R}^{d \times r}$，$r \ll d$）。

本质是**用"低秩约束"代替"L1/L2 约束"**，限制更新方向，防止过拟合。

```
原权重 W ∈ R^(d×d)         约 d² 个参数（不动）
LoRA 增量 AB^T              仅 2dr 个参数（可训练）
```

例：Llama-7B 全参微调要训 70 亿参数，LoRA r=8 时只训 ~400 万参数。
:::

---

## 四、模型评估

### 4.1 分类指标

```
                 预测正        预测负
真实正    [TP]            [FN]
真实负    [FP]            [TN]
```

| 指标 | 公式 | 含义 | 何时用 |
|------|------|------|--------|
| **Accuracy** | (TP+TN)/全部 | 整体正确率 | 类别均衡时 |
| **Precision** | TP/(TP+FP) | 预测为正的里有多少真的正 | 关心"误报" |
| **Recall** | TP/(TP+FN) | 真实为正的里有多少被找回 | 关心"漏报" |
| **F1** | 2PR/(P+R) | 调和平均 | 综合考虑 |
| **AUC-ROC** | 曲线下面积 | 排序质量 | 不依赖阈值 |

::: warning 类别不平衡陷阱
癌症筛查：99% 健康 + 1% 患病。
- 模型："**全部预测健康**" → accuracy = 99% 🎉
- 但 recall = 0% —— 漏掉了所有患者！

**永远要看 precision/recall/F1，不要只看 accuracy。**
:::

### 4.2 回归指标

| 指标 | 公式 | 特点 |
|------|------|------|
| **MSE** | $\frac{1}{N}\sum(y-\hat{y})^2$ | 对大误差敏感（平方放大） |
| **MAE** | $\frac{1}{N}\sum\|y-\hat{y}\|$ | 更鲁棒 |
| **R²** | $1 - \frac{\text{SSE}}{\text{SST}}$ | 解释方差比例（≤1，越大越好） |

### 4.3 交叉验证（k-fold CV）

避免"单次划分"的偶然性：

```
[Fold1][Fold2][Fold3][Fold4][Fold5]
 train  train  test   train  train  → score₁
 train  test   train  train  train  → score₂
 ...                                  → score_K

最终得分 = mean(score_1, ..., score_K)
```

常用 k = 5 或 10。

::: tip LLM 视角
LLM **不用 CV**（训练成本太高），而是用 **holdout** + benchmark：
- 训练集：万亿 token
- 验证集：固定 holdout（监控 loss）
- 测试集：MMLU、HumanEval、BBH 等公开 benchmark

但这带来 **数据污染（contamination）** 问题：测试集可能已经在预训练数据里。
:::

---

## 五、聚类（无监督学习）

### 5.1 K-Means

```
1. 随机选 k 个聚类中心
2. 把每个样本分给最近的中心
3. 重新计算每个簇的均值，作为新的中心
4. 重复 2-3 直到收敛
```

#### 目标函数

$$
J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2
$$

#### 选 k 的方法

**肘部法**：J 关于 k 的曲线"肘部"位置

```
J ↑
  *
   *
    *
     *_____   ← 肘部，k=3 是合理选择
          *___
              ___
  └──────────────→ k
  1  2  3  4  5  6
```

**轮廓系数（Silhouette）**：

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]
$$

- $a(i)$：i 到本簇其他点的平均距离
- $b(i)$：i 到最近"非本簇"的平均距离
- 越大越好

::: tip LLM 关联
**Embedding 聚类** 是 RAG 的雏形：把语义相似的文本块聚到一起。

但现代 RAG 用 **ANN（Approximate Nearest Neighbor）** —— FAISS / HNSW —— 而不是 K-Means，因为：
- ANN 是"在线"检索（给查询找最近邻）
- K-Means 是"离线"分组（找数据的中心）

K-Means 仍用于：**embedding 主题发现** / **训练数据去重** / **冷启动用户分群**。
:::

---

## 六、特征工程

> 经典 ML 时代 80% 时间花在特征工程上。LLM 时代被 embedding 大幅取代，但 **prompt 设计本质上仍是"语义特征工程"**。

### 6.1 数值特征

| 处理 | 何时用 | 公式/方法 |
|------|--------|---------|
| **标准化（StandardScaler）** | 大多数线性模型/神经网络 | $z = (x-\mu)/\sigma$ |
| **归一化（MinMaxScaler）** | 需要 [0,1] 范围 | $(x - x_{\min}) / (x_{\max} - x_{\min})$ |
| **对数变换** | 长尾分布 | $\log(1+x)$ |
| **分箱（Binning）** | 引入非线性 | 把年龄分成 [0-18, 19-35, ...] |

### 6.2 类别特征

| 处理 | 何时用 |
|------|--------|
| **One-Hot Encoding** | 无序类别且基数低（< 50） |
| **Label Encoding** | 有序类别（如学历 = 高中/本科/硕士） |
| **Target Encoding** | 高基数类别（邮编、商品 ID） |
| **Embedding** | 神经网络中的高基数类别 |

::: tip LLM 视角
**所有类别特征本质上都可以被 embedding 替代**。Tokenizer 把每个 token 映射到 embedding，正是 "category → vector" 的极致版（词表大小可达 15 万）。
:::

### 6.3 sklearn Pipeline（工业级最佳实践）

把所有预处理 + 模型串成一条管道，**避免数据泄漏**：

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

# 自动按顺序处理：fit 用训练集统计量，transform 用同一组统计量
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)
```

::: warning 常见陷阱：数据泄漏
错误做法：

```python
# ❌ 错：在切分前对全数据 fit
X_all = scaler.fit_transform(X_all)
X_train, X_test = train_test_split(X_all, ...)
# 测试集统计量泄漏到训练
```

正确做法：

```python
# ✅ 对：先切分,再用 Pipeline
X_train, X_test = train_test_split(X, y, ...)
pipeline.fit(X_train, y_train)  # 只用训练集 fit
pipeline.score(X_test, y_test)
```
:::

---

## 七、过拟合：所有 ML 的"魔鬼"

### 7.1 三种状态

```
欠拟合              恰到好处             过拟合
high bias          good                high variance

训练 ↑              训练 ↓              训练 ↓↓
测试 ↑              测试 ↓              测试 ↑

模型太简单         模型容量合适         模型太复杂
```

### 7.2 七种缓解武器

1. **更多数据**（最有效）
2. **正则化**（L1/L2/Dropout）
3. **降低模型复杂度**（减层 / 减维）
4. **早停**（监控验证集 loss）
5. **交叉验证 + 网格搜索**（超参调优）
6. **数据增强**（图像翻转、文本同义替换）
7. **集成学习**（Bagging / Boosting）

::: tip LLM 视角：双下降现象
经典统计学说："参数 > 样本 必然过拟合"。

但大模型时代发现：**当模型足够大时，过参数化反而能减少过拟合**（double descent）。

```
test error
   ↑      ╱╲
   │     ╱  ╲      ╱─────  ← 极大模型 + 大数据
   │    ╱    ╲    ╱
   │   ╱      ╲  ╱
   │  ╱        ╲╱
   └──────────────────→ 模型容量
   小模型     甜蜜点    临界点    超大模型
```

但**微调阶段**（数据量小）依然受过拟合困扰，所以 LoRA 是低秩约束 + 正则。
:::

---

## 八、配套代码

| 文件 | 演示主题 |
|------|---------|
| [`ml_foundations/classical/classification.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/classical/classification.py) | 三模型分类对比（iris） |
| [`ml_foundations/classical/regression.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/classical/regression.py) | OLS / Ridge / Lasso 正则化路径 |
| [`ml_foundations/classical/clustering.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/classical/clustering.py) | K-Means + 肘部法 + 轮廓系数 |
| [`ml_foundations/classical/feature_engineering.py`](https://github.com/ly0121/llm-cookbook/blob/master/ml_foundations/classical/feature_engineering.py) | 完整 sklearn Pipeline + 特征选择 |

---

## 九、延伸阅读

- 周志华《机器学习》（"西瓜书"）—— 中文最经典入门
- Hastie et al. *The Elements of Statistical Learning* —— 数学严谨
- Andrew Ng [CS229 lecture notes](https://cs229.stanford.edu/) —— 公式推导清晰
- sklearn [官方文档](https://scikit-learn.org/) —— API 与示例最权威

> **下一站**：[深度学习基础](./deep-learning) —— 把"线性 + 非线性"堆叠起来，就是神经网络
