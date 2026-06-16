# 经典机器学习（Classical ML）

> 监督学习、无监督学习、特征工程、模型评估 —— LLM 之前必须先理解的底层范式

---

## 一、什么是机器学习？

> **机器学习**：让计算机从数据中"学习"规律，而不是显式地写规则。

```
传统编程：     输入 + 规则     →  输出
机器学习：     输入 + 输出     →  规则（模型）
```

举例：
- **传统编程**：人手写 100 条 `if 邮件包含"中奖" then 垃圾邮件`
- **机器学习**：给 1 万封标注好的邮件，让模型自己学出"哪些特征 = 垃圾"

### 1.1 三大范式

| 范式 | 数据形式 | 目标 | 例子 |
|------|---------|------|------|
| **监督学习** | (X, y) 有标签 | 学 X → y 的映射 | 房价预测、垃圾邮件分类 |
| **无监督学习** | 只有 X | 发现数据内在结构 | 用户分群、异常检测 |
| **强化学习** | 状态-动作-奖励 | 学最优策略 | 游戏 AI、机器人 |

LLM 训练几乎全部走监督 + 强化（SFT + RLHF），但根基是**统计学习**。

### 1.2 与 LLM 的衔接

| 经典 ML 概念 | LLM 中的对应 |
|--------------|--------------|
| 损失函数（MSE / 交叉熵） | 自回归交叉熵（next-token prediction loss） |
| 过拟合 / 正则化 | LLM 用 dropout + weight decay + 早停 |
| 训练/验证/测试集划分 | LLM 用 holdout + benchmark（MMLU / HumanEval） |
| 特征工程 | LLM 时代被 embedding 取代，但 prompt 仍然是"特征工程" |
| 偏差-方差权衡 | 大模型仍在这条曲线上 |

---

## 二、监督学习

### 2.1 分类（Classification）

**目标**：把样本归入若干离散类别。

#### 逻辑回归（Logistic Regression）

最简单也最重要的分类器。线性模型 + sigmoid 把输出压到 [0,1]：

$$
P(y=1 | x) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}
$$

**损失函数**（二元交叉熵）：

$$
L = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]
$$

> **关键洞察**：LLM 的 next-token prediction 本质上是一个 **"在 V 个 token 中选一个"的多分类逻辑回归**（softmax + 交叉熵），只是分类头特别大（V ≈ 5 万~15 万）。

#### 决策树（Decision Tree）

逐层用一个特征切分数据，直到叶节点足够"纯"：

```
                  收入 > 50K?
                 /          \
              是              否
            /    \          /    \
        年龄>30  ...    教育=本科  ...
```

**信息增益**（ID3 算法）：

$$
\text{Gain}(S, A) = H(S) - \sum_{v} \frac{|S_v|}{|S|} H(S_v)
$$

其中 $H(S) = -\sum p_i \log p_i$ 是熵。

#### 随机森林（Random Forest）

100 棵决策树投票，每棵树用：
1. 随机抽样的训练样本（bagging）
2. 随机抽取的特征子集

**为什么 work**：单棵树过拟合，但**独立**误差互相抵消。这是"集成学习"的核心思想。

> **LLM 关联**：MoE（Mixture of Experts）架构中"多个专家投票"的思想，与随机森林精神一致。

### 2.2 回归（Regression）

**目标**：预测连续数值。

#### 线性回归（OLS）

$$
\hat{y} = w^\top x + b, \quad L = \frac{1}{N}\sum (y_i - \hat{y}_i)^2
$$

闭式解（normal equation）：$w = (X^\top X)^{-1} X^\top y$。

但当特征维度 > 样本数时 $X^\top X$ 不可逆 → 需要正则化。

#### 岭回归（Ridge / L2）

$$
L = \|y - Xw\|^2 + \lambda \|w\|^2_2
$$

惩罚大权重，让所有 $w_i$ 都"小一点"，但**不会变 0**。

#### Lasso（L1）

$$
L = \|y - Xw\|^2 + \lambda \|w\|_1
$$

L1 范数让部分 $w_i$ **直接归零** → 自动特征选择。

| 方法 | 几何形状 | 效果 |
|------|---------|------|
| Ridge | 圆 | 系数缩小 |
| Lasso | 菱形 | 系数稀疏 |

> **LLM 关联**：LoRA 微调本质上是"低秩 + 正则"的思想；模型剪枝（pruning）就是 Lasso 的极致版。

### 2.3 模型评估

#### 分类指标

```
                 预测正        预测负
真实正    [TP]            [FN]
真实负    [FP]            [TN]
```

| 指标 | 公式 | 含义 |
|------|------|------|
| Accuracy | (TP+TN)/全部 | 整体正确率，**类别不平衡时具有误导性** |
| Precision | TP/(TP+FP) | 预测为正的里有多少真的正 |
| Recall | TP/(TP+FN) | 真实为正的里有多少被找回 |
| F1 | 2PR/(P+R) | 调和平均 |
| AUC-ROC | 曲线下面积 | 排序质量 |

#### 回归指标

- **MSE**：$\frac{1}{N}\sum(y-\hat{y})^2$，对大误差敏感
- **MAE**：$\frac{1}{N}\sum|y-\hat{y}|$，更鲁棒
- **R²**：$1 - \frac{\text{SSE}}{\text{SST}}$，解释方差比例

#### 交叉验证（k-fold CV）

```
[Fold1][Fold2][Fold3][Fold4][Fold5]
 train  train  test   train  train  → score1
 train  test   train  train  train  → score2
 ...                                  → scoreK
最终得分 = mean(scores)
```

**目的**：避免单次划分的偶然性。常用 k=5 或 k=10。

> **LLM 关联**：LLM 评测用 holdout（如 MMLU 测试集 + leakage 检查），不用 CV，因为训练成本太高。

---

## 三、无监督学习

### 3.1 K-Means 聚类

```
1. 随机初始化 k 个聚类中心
2. 把每个点分配给最近的中心
3. 重新计算每个簇的中心（均值）
4. 重复 2-3 直到收敛
```

**目标函数**：

$$
J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2
$$

#### 选择 k：肘部法

绘制 k vs J 曲线，找"肘部"拐点：

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

#### 轮廓系数（Silhouette）

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$

- $a(i)$：i 到本簇其他点的平均距离
- $b(i)$：i 到最近的"非本簇"的平均距离
- $s \in [-1, 1]$，越大越好

> **LLM 关联**：embedding 聚类是 RAG 中"语义检索"的雏形，但现代 RAG 用 ANN（FAISS/HNSW）而非 K-Means。

### 3.2 层次聚类（Hierarchical）

不需要预设 k，自底向上合并最近的两簇，输出树状图（dendrogram）。可在任意层级"切一刀"得到不同 k。

---

## 四、特征工程

> 经典 ML 时代 80% 的精力花在特征工程上。LLM 时代被 embedding 大幅取代，但 **prompt 设计本质上仍是"语义特征工程"**。

### 4.1 数值特征

| 处理 | 何时用 | 公式 |
|------|--------|------|
| **标准化**（StandardScaler） | 大多数线性模型/神经网络 | $z = (x-\mu)/\sigma$ |
| **归一化**（MinMax） | 需要 [0,1] 范围 | $(x-x_{min})/(x_{max}-x_{min})$ |
| **对数变换** | 长尾分布（房价、收入） | $\log(1+x)$ |
| **分箱**（Binning） | 非线性关系 | 把年龄分成 [0-18, 19-35, ...] |

### 4.2 类别特征

| 处理 | 何时用 |
|------|--------|
| **One-Hot** | 无序类别且基数低（< 50） |
| **Label Encoding** | 有序类别（如学历） |
| **Target Encoding** | 高基数类别（如邮编） |
| **Embedding** | 神经网络中的类别特征 |

> **LLM 时代的视角**：所有类别特征本质上都可以被 embedding 替代。Tokenizer 把每个 token 映射到 embedding，正是 "category → vector" 的极致版。

### 4.3 特征选择

- **过滤法**：相关系数、卡方、互信息
- **包裹法**：递归特征消除（RFE）
- **嵌入法**：L1 正则、树模型的 feature_importance

### 4.4 sklearn Pipeline

把所有预处理 + 模型串成一条管道，避免数据泄漏：

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])
pipeline.fit(X_train, y_train)  # 自动按顺序处理
```

`ColumnTransformer` 可以对不同列做不同处理（数值列标准化，类别列 OneHot）。

---

## 五、过拟合与偏差-方差权衡

### 5.1 三种状态

```
欠拟合          恰到好处         过拟合
high bias      good           high variance
训练 ↑          训练 ↓           训练 ↓↓
测试 ↑          测试 ↓           测试 ↑

模型太简单     模型容量合适       模型太复杂
```

### 5.2 缓解过拟合的"七种武器"

1. **更多数据**（最有效）
2. **正则化**（L1/L2/Dropout）
3. **降低模型复杂度**（减层/减维）
4. **早停**（监控验证集 loss）
5. **交叉验证 + 网格搜索**（HP 调优）
6. **数据增强**（图像翻转、文本同义替换）
7. **集成学习**（Bagging / Boosting）

> **LLM 关联**：大模型时代发现"够大 + 够多数据"反而不容易过拟合（**double descent** 现象）。但微调阶段（数据量小）依然受过拟合困扰，所以 LoRA 是低秩约束 + 正则。

---

## 六、常见陷阱

| 陷阱 | 症状 | 解决 |
|------|------|------|
| 数据泄漏（Leakage） | 训练集精度极高，线上崩 | 用 Pipeline，划分后再做特征工程 |
| 类别不平衡 | accuracy 高但 recall 低 | class_weight、过采样（SMOTE）、阈值调优 |
| 多重共线性 | 系数不稳定 | VIF 检测、PCA、L2 正则 |
| 维度灾难 | 特征 >> 样本 | 特征选择、降维、Lasso |
| 不一致编码 | 训练集见过的类别测试集没有 | OneHotEncoder(handle_unknown="ignore") |

---

## 七、本目录 demo 速查

| 文件 | 主题 | 数据集 | 关键 API |
|------|------|--------|---------|
| `classification.py` | 三模型分类对比 | iris | LogisticRegression, DecisionTree, RandomForest |
| `regression.py` | OLS / Ridge / Lasso | california_housing | Ridge, Lasso, alpha 网格 |
| `clustering.py` | K-Means + 评估 | 合成 blob | KMeans, silhouette_score, 肘部法 |
| `feature_engineering.py` | 完整 Pipeline | 泰坦尼克微缩 | ColumnTransformer, SelectKBest |

---

## 八、延伸阅读

- 周志华《机器学习》（西瓜书）—— 中文最经典入门
- Hastie et al. *The Elements of Statistical Learning* —— 数学严谨
- Andrew Ng *CS229* lecture notes —— 公式推导清晰
- sklearn 官方文档 —— API 与示例最权威

> **下一站**：把"线性 + 非线性变换"堆叠起来，就是神经网络 → 见 `../deep_learning/KNOWLEDGE.md`
