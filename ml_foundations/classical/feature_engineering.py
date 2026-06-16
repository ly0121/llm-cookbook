"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:特征工程完整 Pipeline                                ║
║         数值/类别混合特征 + 缺失值 + 特征选择                     ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题:真实数据脏乱差,如何用 sklearn 做工业级预处理?】
═══════════════════════════════════════════════════════════════════

  数据集:微缩泰坦尼克(本地合成,12 列,200 行)
    数值列: Age, Fare
    类别列: Sex, Pclass, Embarked
    标签:   Survived (0/1)
    挑战:   Age 有缺失,Embarked 有缺失,类别不平衡

  ┌─────────────────────────────────────────────────────────────┐
  │   完整 Pipeline:                                              │
  │                                                             │
  │   原始数据                                                    │
  │      ↓                                                       │
  │   ColumnTransformer(对不同列做不同处理)                       │
  │      ├── 数值列: SimpleImputer(均值) → StandardScaler       │
  │      └── 类别列: SimpleImputer(众数) → OneHotEncoder         │
  │      ↓                                                       │
  │   SelectKBest(挑选最重要的 k 个特征)                          │
  │      ↓                                                       │
  │   LogisticRegression                                         │
  │                                                             │
  │   关键:用 Pipeline 把所有步骤串起来,                          │
  │         避免"先全量预处理再切分"导致的数据泄漏!                 │
  └─────────────────────────────────────────────────────────────┘

  与 LLM 的关联:
    经典 ML 时代 80% 时间花在特征工程。LLM 时代被 embedding
    替代了大部分,但 prompt 设计 / few-shot examples 选择
    本质上仍是"语义特征工程"。
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(42)


def generate_titanic_like():
    """生成微缩泰坦尼克风格数据(本地合成,无需下载)。"""
    n = 200
    rng = np.random.default_rng(42)

    # Pclass: 1/2/3 等舱
    pclass = rng.choice([1, 2, 3], size=n, p=[0.2, 0.3, 0.5])
    # Sex
    sex = rng.choice(["male", "female"], size=n, p=[0.65, 0.35])
    # Age: 含缺失
    age = rng.normal(30, 14, size=n)
    age = np.clip(age, 0.5, 80)
    age_mask = rng.random(n) < 0.20  # 20% 缺失
    age = np.where(age_mask, np.nan, age)
    # Fare: 与 Pclass 相关 + 噪声
    fare_base = np.where(pclass == 1, 80, np.where(pclass == 2, 25, 10))
    fare = fare_base + rng.exponential(15, size=n)
    # Embarked: 含少量缺失
    emb = rng.choice(["S", "C", "Q"], size=n, p=[0.7, 0.2, 0.1])
    emb_mask = rng.random(n) < 0.05
    emb = np.where(emb_mask, None, emb)

    # Survived: 受 Sex / Pclass / Age 影响
    p_survive = 0.1
    p_survive += 0.4 * (sex == "female")
    p_survive += 0.15 * (pclass == 1)
    p_survive += 0.05 * (pclass == 2)
    p_survive -= 0.10 * (np.nan_to_num(age, nan=30) > 50)
    p_survive = np.clip(p_survive, 0.02, 0.95)
    survived = (rng.random(n) < p_survive).astype(int)

    df = pd.DataFrame({
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "Fare": fare,
        "Embarked": emb,
        "Survived": survived,
    })
    return df


def show_data_quality(df: pd.DataFrame):
    """打印数据质量报告。"""
    print("─" * 60)
    print("数据质量诊断:")
    print("─" * 60)
    print(f"  总样本: {len(df)}")
    print(f"  类别平衡: 存活={df['Survived'].sum()}  遇难={(df['Survived']==0).sum()}")
    print("\n  缺失值:")
    for col in df.columns:
        n_null = df[col].isna().sum()
        if n_null > 0:
            print(f"    {col:12s}  {n_null}/{len(df)}  ({n_null/len(df)*100:.1f}%)")
    print("\n  特征类型:")
    for col, dt in df.dtypes.items():
        print(f"    {col:12s}  {dt}")
    print()


def build_pipeline(numeric_features, categorical_features, k_best=5):
    """构建完整 Pipeline:预处理 + 特征选择 + 模型。"""
    # 数值列处理
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    # 类别列处理
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # 合并:对不同列做不同处理
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])

    # 完整管道:预处理 → 特征选择 → 分类器
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("select", SelectKBest(score_func=f_classif, k=k_best)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    return pipeline


def show_selected_features(pipeline, numeric_features, categorical_features):
    """展示 SelectKBest 选中了哪些特征。"""
    pre = pipeline.named_steps["preprocess"]
    sel = pipeline.named_steps["select"]

    # 重建特征名
    onehot = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(onehot.get_feature_names_out(categorical_features))
    all_names = list(numeric_features) + cat_names

    mask = sel.get_support()
    scores = sel.scores_

    print("\n  ──── 特征选择结果(F-score 越大越相关) ────")
    paired = sorted(
        [(n, s, m) for n, s, m in zip(all_names, scores, mask)],
        key=lambda x: -x[1],
    )
    for name, score, selected in paired:
        marker = "  ★ 入选" if selected else ""
        bar = "█" * min(40, int(score / 5))
        print(f"    {name:25s}  F={score:7.2f}  {bar}{marker}")


def main():
    print("\n" + "█" * 60)
    print("█" + " " * 16 + "特征工程 + Pipeline" + " " * 23 + "█")
    print("█" * 60)

    # 1. 数据
    df = generate_titanic_like()
    show_data_quality(df)

    # 2. 切分
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    numeric_features = ["Age", "Fare"]
    categorical_features = ["Pclass", "Sex", "Embarked"]

    # 3. 构建并训练 Pipeline
    print("═" * 60)
    print("  完整 Pipeline 训练:")
    print("═" * 60)
    pipeline = build_pipeline(numeric_features, categorical_features, k_best=5)
    pipeline.fit(X_train, y_train)

    # 4. 评估
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)

    print(f"  训练集准确率: {train_acc:.4f}")
    print(f"  测试集准确率: {test_acc:.4f}")
    print(f"  5-fold CV: mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}")

    # 5. 分类报告
    y_pred = pipeline.predict(X_test)
    print("\n  分类报告:")
    for line in classification_report(
        y_test, y_pred, target_names=["遇难", "存活"], digits=3
    ).split("\n"):
        print(f"    {line}")

    # 6. 特征选择详情
    show_selected_features(pipeline, numeric_features, categorical_features)

    # 7. 不同 k_best 的对比
    print("\n" + "═" * 60)
    print("  不同 k_best 对性能的影响(网格搜索的简化版):")
    print("═" * 60)
    for k in [1, 2, 3, 4, 5, 7]:
        p = build_pipeline(numeric_features, categorical_features, k_best=k)
        scores = cross_val_score(p, X_train, y_train, cv=5)
        bar = "█" * int(scores.mean() * 50)
        print(f"  k={k}  cv_acc={scores.mean():.4f} ± {scores.std():.4f}  {bar}")

    # 8. 总结
    print("\n" + "═" * 60)
    print("  关键收获:")
    print("═" * 60)
    print("  ✓ 用 Pipeline 把预处理和模型绑定 → 防止数据泄漏")
    print("  ✓ 用 ColumnTransformer 对不同列做不同处理")
    print("  ✓ SimpleImputer 处理缺失值 → 数值用均值,类别用众数")
    print("  ✓ SelectKBest 挑出最有区分度的特征")
    print("  ✓ 整个流水线对训练集和测试集自动一致处理\n")


if __name__ == "__main__":
    main()
