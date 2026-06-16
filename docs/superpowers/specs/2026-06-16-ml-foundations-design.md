# ML Foundations 板块设计文档（阶段 1）

- **日期**：2026-06-16
- **作者**：Claude (brainstorming)
- **范围**：阶段 1 —— 经典 ML + 深度学习基础 + NLP 经典基础
- **状态**：待用户审阅

---

## 1. 背景与动机

`llm-cookbook` 现有 21 个进阶项目全部聚焦在 **"LLM 应用层"**：调用现成大模型 + 工程化封装（LangChain、RAG、Agent、可观测性等）。整个仓库 **不包含任何机器学习基础内容**：

- `requirements.txt` 中无 `torch`、`scikit-learn`、`numpy`（核心 ML 库）
- 0 处出现 `fit()`、`backward`、`optimizer`、`loss` 等训练相关代码
- 唯一接近 ML 的 `llm/transformer_demo.py` 是**纯前向推理**演示

这对于一本完整的"大模型应用开发手册"是显著缺口：读者若没有 ML/DL 背景，无法真正理解 Transformer、Embedding、微调等概念的本质。

本设计旨在补齐 ML 前置内容，并为后续阶段（Transformer 训练、LLM 微调）打地基。

## 2. 范围决策

### 2.1 已锁定约束（来自 brainstorming Q&A）

| 约束 | 决策 |
|------|------|
| 终极方向 | 全栈式 ML 基础 → LLM 微调 |
| **本次范围** | **阶段 1：经典 ML + DL 基础 + NLP 基础** |
| 硬件 | Mac CPU/MPS 本地可跑（小数据集） |
| 代码风格 | `KNOWLEDGE.md` + `demo.py`（与现有 21 个项目一致） |
| 依赖管理 | 全部进 `requirements.txt`，`pyproject.toml` 同步加 `ml` extra |
| docs 位置 | 放在最前作为 **"零、ML 基础（前置补课）"** |
| 测试 | **本次不补 `tests/`** 目录 |

### 2.2 阶段路线（仅声明，不实现）

- **阶段 1（本次）**：经典 ML + DL 基础 + NLP 基础
- **阶段 2（未来）**：Transformer 训练、tokenizer 训练、from-scratch 预训练
- **阶段 3（未来）**：SFT、LoRA/QLoRA、DPO/RLHF、量化部署

阶段 2/3 会在 `ml_foundations/` 下继续添加 `transformer_training/`、`llm_finetuning/` 等子目录，**不影响阶段 1 的结构**。

### 2.3 显式不做（YAGNI）

- ❌ 单元测试（保持 `tests/` 不动）
- ❌ 中文 `README.zh-CN.md`
- ❌ Jupyter Notebook（坚持 `.py`）
- ❌ Docker / CI 改动
- ❌ 阶段 2/3 任何代码

## 3. 代码目录结构

```
ml_foundations/
├── classical/
│   ├── KNOWLEDGE.md            # 监督/无监督/评估/特征工程原理
│   ├── classification.py       # 鸢尾花：LR + DT + RF 三模型对比
│   ├── regression.py           # 加州房价：线性 + 岭 + Lasso
│   ├── clustering.py           # 客户分群：K-Means + 肘部法 + 轮廓系数
│   └── feature_engineering.py  # Pipeline + ColumnTransformer + 特征选择
├── deep_learning/
│   ├── KNOWLEDGE.md            # 张量/反传/优化器/正则化
│   ├── pytorch_basics.py       # 张量运算 + autograd
│   ├── mlp_from_scratch.py     # 纯 NumPy MLP（手算前向 + 反传）
│   ├── mlp_pytorch.py          # PyTorch MLP，MNIST 子集（1000 张）
│   ├── cnn_mnist.py            # 简化 LeNet
│   └── rnn_lstm.py             # 字符级语言模型
└── nlp_foundations/
    ├── KNOWLEDGE.md            # NLP 流水线 + 文本表示演进
    ├── text_preprocessing.py   # jieba 中文 + nltk 英文
    ├── tfidf_classification.py # 20-newsgroups + TF-IDF + NB/SVM
    └── word2vec_demo.py        # gensim 训练 + 类比推理
```

**总计**：3 子目录 / 12 个 `.py` demo / 3 个 `KNOWLEDGE.md`

## 4. 每个 demo 的实现规范

### 4.1 通用要求

- **顶部 docstring**：复用现有 `llm/transformer_demo.py` 的 ASCII 框图风格，含核心问题、原理推导、信息流图
- **执行体量**：所有 demo 在 M1 Mac CPU 上 **<3 分钟跑完**
- **离线运行**：不依赖 `config.py`，不调 LLM API，无网络依赖（数据集用 sklearn/torchvision 内置或本地合成）
- **可视化**：matplotlib 输出保存为 `figures/<demo_name>.png`，路径加入 `.gitignore`
- **入口**：`if __name__ == "__main__": main()` 标准结构

### 4.2 各 demo 内容要点

#### `classical/`

| 文件 | 核心内容 |
|------|---------|
| `classification.py` | 加载 iris；逻辑回归 + 决策树 + 随机森林对比；混淆矩阵；交叉验证 |
| `regression.py` | 加州房价（california_housing）；OLS / Ridge / Lasso；正则化系数对系数稀疏性的影响曲线 |
| `clustering.py` | 合成 3 簇 blob 数据；K-Means + 层次聚类；肘部法 + 轮廓系数选 K |
| `feature_engineering.py` | 泰坦尼克微缩数据集；ColumnTransformer 处理数值/类别；SelectKBest 特征选择；完整 Pipeline |

#### `deep_learning/`

| 文件 | 核心内容 |
|------|---------|
| `pytorch_basics.py` | 张量创建/变形/广播；`requires_grad` + `.backward()`；手动验证梯度 |
| `mlp_from_scratch.py` | 2 层 MLP，纯 NumPy；前向传播逐步推导；反向传播链式法则展开；用 XOR 验证收敛 |
| `mlp_pytorch.py` | 同样 2 层 MLP，用 `nn.Module`；MNIST 子集训练；展示训练循环模板 |
| `cnn_mnist.py` | 2 层卷积 + 池化 + FC；MNIST 子集；可视化卷积核 |
| `rnn_lstm.py` | 字符级语言模型；用《狂人日记》前 1000 字训练；生成新文本展示 |

#### `nlp_foundations/`

| 文件 | 核心内容 |
|------|---------|
| `text_preprocessing.py` | 中文 jieba 分词；英文 nltk 分词 + 词干化 + 词形还原；停用词处理；对比展示 |
| `tfidf_classification.py` | 20-newsgroups 4 类子集；TF-IDF 向量化；朴素贝叶斯 vs SVM 对比 |
| `word2vec_demo.py` | gensim 在小型语料上训练 Skip-Gram；展示 `king - man + woman ≈ queen`；可视化词向量降维 |

### 4.3 KNOWLEDGE.md 内容大纲

每个 KNOWLEDGE.md ~500–700 行，包含：

1. **核心问题**：本主题解决什么问题
2. **原理推导**：关键公式 + 直觉解释
3. **算法对比表**：不同方法的适用场景
4. **常见陷阱**：过拟合、维度灾难、类别不平衡等
5. **与 LLM 的衔接**：本节内容如何为后续 LLM 学习铺垫
6. **延伸阅读**：经典论文/教材链接

## 5. 文档站集成（VitePress）

### 5.1 新增文件

```
docs/ml-foundations/
├── index.md                  # 板块总览 + 学习路径
├── classical-ml.md           # 经典 ML 原理详解
├── deep-learning.md          # DL 基础详解
├── nlp-foundations.md        # NLP 经典基础详解
├── ml-vs-llm.md              # ML 与 LLM 的关系/演进/分工
└── learning-path.md          # 从 ML 到 LLM 微调的进阶路径
```

每个 .md 文件 500–1500 行（视主题复杂度而定，VitePress 文档比 KNOWLEDGE.md 更详细，含图表、代码片段、对比表）。

### 5.2 `docs/.vitepress/config.ts` 改动

**A. sidebar 顶部新增（在"开始"之后、"一、LLM 基础"之前）：**

```ts
{
  text: "零、ML 基础（前置补课）",
  collapsed: false,
  items: [
    { text: "板块总览", link: "/ml-foundations/" },
    { text: "经典机器学习", link: "/ml-foundations/classical-ml" },
    { text: "深度学习基础", link: "/ml-foundations/deep-learning" },
    { text: "NLP 经典基础", link: "/ml-foundations/nlp-foundations" },
    { text: "ML 与 LLM 的关系", link: "/ml-foundations/ml-vs-llm" },
    { text: "进阶学习路径", link: "/ml-foundations/learning-path" },
  ],
},
```

**B. 顶部 nav 的"知识模块"菜单首位插入：**

```ts
{ text: "ML 基础（前置）", link: "/ml-foundations/" },
```

## 6. 依赖更新

### 6.1 `requirements.txt` 末尾追加

```
# ===== ML 基础（阶段 1） =====
numpy>=1.24.0
scikit-learn>=1.3.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
torch>=2.0.0                    # CPU 版即可
torchvision>=0.15.0             # MNIST 数据集
gensim>=4.3.0                   # Word2Vec
nltk>=3.8.0                     # 英文 NLP 预处理
jieba>=0.42.1                   # 中文分词
```

### 6.2 `pyproject.toml` 新增 extra

在 `[project.optional-dependencies]` 中新增：

```toml
ml = [
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "pandas>=2.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "gensim>=4.3.0",
    "nltk>=3.8.0",
    "jieba>=0.42.1",
]
```

### 6.3 `.gitignore` 新增

```
# ML 板块生成的可视化与数据缓存
ml_foundations/**/figures/
ml_foundations/**/data/
ml_foundations/**/runs/
~/.cache/torch/
~/nltk_data/
```

## 7. README 更新

在 README "Project Checklist" 末尾（"6. Advanced" 之后）追加新章节，与现有英文风格保持一致：

```markdown
### 0. ML Foundations — Prerequisite Crash Course (not numbered)

| Sub-directory | Topics |
|---------------|--------|
| `ml_foundations/classical/` | sklearn classical ML: classification / regression / clustering / feature engineering |
| `ml_foundations/deep_learning/` | PyTorch DL: tensors / MLP / CNN / RNN |
| `ml_foundations/nlp_foundations/` | Classical NLP: preprocessing / TF-IDF / Word2Vec |

**Purpose:** Optional prerequisite for readers without ML/DL background. All demos run offline on CPU, no LLM API required.
```

在 "Quick Start" 段加一行说明（英文）：

> ML Foundation demos under `ml_foundations/` require no API key and run fully offline — an optional crash course before diving into LLM content.

## 8. 提交策略

**单次提交**：

```
feat(ml): add ml_foundations module + docs

- New ml_foundations/ with 3 sub-modules: classical, deep_learning, nlp_foundations
- 12 demos, all CPU-runnable on Mac in <3 min each
- KNOWLEDGE.md per sub-module + 6 VitePress docs under docs/ml-foundations/
- Add new top-level "零、ML 基础（前置补课）" chapter to docs sidebar
- Append ML deps to requirements.txt; add [ml] extra in pyproject.toml
- Update README with new section
```

## 9. 验收标准

实现完成后必须满足：

1. **代码层**
   - [ ] `ml_foundations/` 目录创建，含 3 子目录
   - [ ] 12 个 `.py` 文件全部存在，每个有完整 docstring + `main()`
   - [ ] 每个 demo 在干净 Python 3.10+ 环境（按 `requirements.txt` 安装后）能运行至 `main()` 结束，无未捕获异常
   - [ ] 3 个 `KNOWLEDGE.md` 文件存在，结构遵循 §4.3
2. **文档层**
   - [ ] `docs/ml-foundations/` 6 个 .md 文件存在
   - [ ] `docs/.vitepress/config.ts` sidebar 已加 "零、ML 基础" 章节，位置在"开始"之后
   - [ ] 顶部 nav "知识模块" 已加 ML 基础入口
3. **配置层**
   - [ ] `requirements.txt` 末尾已加 ML 依赖区块
   - [ ] `pyproject.toml` 已加 `ml` extra
   - [ ] `.gitignore` 已加 figures/data/runs 排除规则
4. **元数据层**
   - [ ] README 已加新章节
   - [ ] 单次 commit 推送至当前分支

## 10. 已知风险与缓解

| 风险 | 缓解 |
|------|------|
| `torch` CPU 版在某些 Mac 上首次导入慢 | demo 中提示首次运行可能下载模型/数据 |
| MNIST 数据集首次下载需联网 | 在 KNOWLEDGE.md 注明，并告知数据缓存位置 |
| nltk 需下载 stopwords/punkt 数据包 | demo 中 try/except 调用 `nltk.download()` |
| jieba 词典加载首次慢 | 在 demo 顶部注释提示 |
| 不同 Python 版本 numpy / sklearn 兼容性 | 依赖只用 `>=`，不锁死小版本 |
| 阶段 2/3 未来扩展可能要重构 | 目录命名 `ml_foundations/` 已预留扩展空间，子目录扁平不嵌套 |

## 11. 工作量估算

- 12 个 demo × 平均 150 行 ≈ 1800 行 Python
- 3 个 KNOWLEDGE.md × 平均 600 行 ≈ 1800 行 Markdown
- 6 个 docs/ml-foundations/*.md × 平均 1000 行 ≈ 6000 行 Markdown
- 配置/README/sidebar 改动：小

总计约 **9600 行新增内容**。

## 12. 后续计划（不在本次范围）

完成阶段 1 后，建议：

1. 用户实际跑一遍所有 demo，反馈体验
2. 观察 docs 站构建是否正常（`pnpm dev`）
3. 进入阶段 2 设计：Transformer 训练板块（独立 spec）
