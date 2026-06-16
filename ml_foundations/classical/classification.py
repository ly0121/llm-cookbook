"""
╔══════════════════════════════════════════════════════════════════╗
║         项目：经典分类算法对比（Classification）                  ║
║         逻辑回归 vs 决策树 vs 随机森林                            ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题：给定特征 X,如何预测类别 y?】
═══════════════════════════════════════════════════════════════════

  数据集:鸢尾花(iris) — 经典三分类
    特征 X(4维):花萼长/宽、花瓣长/宽
    标签 y(3类):setosa / versicolor / virginica

  ┌─────────────────────────────────────────────────────────────┐
  │   三种思路:                                                   │
  │                                                             │
  │   1. 逻辑回归    →  线性边界 + sigmoid                       │
  │      P(y=k|x) = softmax(W_k·x + b_k)                       │
  │                                                             │
  │   2. 决策树      →  逐步切分特征空间                          │
  │      if 花瓣长 < 2.45: setosa                              │
  │      else if 花瓣宽 > 1.75: virginica                       │
  │      else: versicolor                                       │
  │                                                             │
  │   3. 随机森林    →  100 棵决策树投票                          │
  │      Bagging + 特征随机 = 降低方差                           │
  └─────────────────────────────────────────────────────────────┘

  与 LLM 的关联:
    LLM 的 next-token prediction 在最后一层就是
    一个超大规模的多分类 softmax(V≈15万),loss 是交叉熵 ——
    与逻辑回归本质上是同一个数学结构。
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)


def load_and_split():
    """加载 iris 并做训练/测试切分。"""
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("─" * 60)
    print(f"数据集: iris  |  样本: {len(X)}  |  特征: {X.shape[1]}  |  类别: {len(target_names)}")
    print(f"训练: {len(X_train)} 条  |  测试: {len(X_test)} 条")
    print(f"特征名: {feature_names}")
    print(f"类别名: {list(target_names)}")
    print("─" * 60)
    return X_train, X_test, y_train, y_test, target_names


def evaluate_model(name, model, X_train, X_test, y_train, y_test, target_names):
    """通用评估:训练 + 测试 + 交叉验证 + 混淆矩阵。"""
    print(f"\n{'='*60}")
    print(f"  模型: {name}")
    print("=" * 60)

    # 训练
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 单次切分得分
    acc = accuracy_score(y_test, y_pred)
    print(f"  测试集准确率: {acc:.4f}")

    # 5 折交叉验证(更稳健的得分)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"  5-fold CV: mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}")
    print(f"            scores={[f'{s:.3f}' for s in cv_scores]}")

    # 分类报告(precision / recall / f1)
    print("\n  分类报告:")
    report = classification_report(y_test, y_pred, target_names=target_names, digits=3)
    for line in report.split("\n"):
        print(f"    {line}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("  混淆矩阵 (行=真实, 列=预测):")
    header = "        " + "  ".join(f"{n[:8]:>8}" for n in target_names)
    print(header)
    for i, row in enumerate(cm):
        cells = "  ".join(f"{v:>8}" for v in row)
        print(f"    {target_names[i][:6]:>6}  {cells}")

    return acc


def show_logistic_regression_details(model, feature_names, target_names):
    """展示逻辑回归的可解释性:权重和偏置。"""
    print("\n  ──── 逻辑回归可解释性 ────")
    print("  每个类别的权重(正→促进,负→抑制):")
    coefs = model.coef_  # (n_classes, n_features)
    for cls_idx, cls_name in enumerate(target_names):
        print(f"\n  类别 [{cls_name}] 权重:")
        for f_idx, f_name in enumerate(feature_names):
            w = coefs[cls_idx, f_idx]
            bar = "+" * int(abs(w) * 3) if w > 0 else "-" * int(abs(w) * 3)
            print(f"    {f_name:25s} {w:+.3f}  {bar}")


def show_decision_tree_rules(model, feature_names):
    """展示决策树学到的 if-else 规则。"""
    from sklearn.tree import export_text

    print("\n  ──── 决策树规则 ────")
    rules = export_text(model, feature_names=feature_names, max_depth=3)
    for line in rules.split("\n"):
        print(f"    {line}")


def show_feature_importance(model, feature_names, model_name):
    """随机森林的特征重要性。"""
    print(f"\n  ──── {model_name} 特征重要性 ────")
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    for rank, idx in enumerate(order, 1):
        bar = "█" * int(importances[idx] * 40)
        print(f"  {rank}. {feature_names[idx]:25s} {importances[idx]:.4f}  {bar}")


def main():
    print("\n" + "█" * 60)
    print("█" + " " * 18 + "经典分类算法对比" + " " * 24 + "█")
    print("█" * 60)

    # 1. 加载数据
    X_train, X_test, y_train, y_test, target_names = load_and_split()
    iris = load_iris()
    feature_names = iris.feature_names

    # 2. 标准化(对逻辑回归很重要,树模型可以省略)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 3. 三个模型
    results = {}

    # —— 逻辑回归 ——
    lr = LogisticRegression(max_iter=1000, multi_class="multinomial")
    results["LogisticRegression"] = evaluate_model(
        "逻辑回归 (LogisticRegression)",
        lr,
        X_train_s,
        X_test_s,
        y_train,
        y_test,
        target_names,
    )
    show_logistic_regression_details(lr, feature_names, target_names)

    # —— 决策树 ——
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    results["DecisionTree"] = evaluate_model(
        "决策树 (DecisionTreeClassifier, max_depth=4)",
        dt,
        X_train,
        X_test,
        y_train,
        y_test,
        target_names,
    )
    show_decision_tree_rules(dt, feature_names)

    # —— 随机森林 ——
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    results["RandomForest"] = evaluate_model(
        "随机森林 (RandomForest, n_estimators=100)",
        rf,
        X_train,
        X_test,
        y_train,
        y_test,
        target_names,
    )
    show_feature_importance(rf, feature_names, "随机森林")

    # 4. 总结
    print("\n" + "═" * 60)
    print("  最终对比:")
    print("═" * 60)
    for name, acc in results.items():
        bar = "█" * int(acc * 50)
        print(f"  {name:25s}  acc={acc:.4f}  {bar}")

    print("\n  💡 经验法则:")
    print("    - iris 这种小且线性可分的数据集,三个模型差不多")
    print("    - 真实世界:随机森林通常 baseline 最稳;深度学习需要更多数据")
    print("    - 可解释性需求 → 逻辑回归 / 决策树;追求精度 → 集成方法\n")


if __name__ == "__main__":
    main()
