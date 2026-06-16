"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:线性回归 + 正则化(Ridge / Lasso)                    ║
║         加州房价预测 + 正则化系数对系数稀疏性的影响               ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题:特征很多时,普通线性回归会过拟合,如何缓解?】
═══════════════════════════════════════════════════════════════════

  数据集:加州房价(california_housing)
    8 个特征 → 1 个连续目标(房价中位数)

  ┌─────────────────────────────────────────────────────────────┐
  │   三个模型的损失函数对比:                                       │
  │                                                             │
  │   OLS    : L = ||y - Xw||²                                  │
  │   Ridge  : L = ||y - Xw||² + α·||w||²₂   ← L2 惩罚          │
  │   Lasso  : L = ||y - Xw||² + α·||w||₁    ← L1 惩罚          │
  │                                                             │
  │   几何直觉:                                                   │
  │     L2 → 圆形约束 → 系数都被"压扁"但不会变 0                  │
  │     L1 → 菱形约束 → 角点更容易被命中 → 系数稀疏化             │
  └─────────────────────────────────────────────────────────────┘

  与 LLM 的关联:
    LoRA 微调 = 在原权重 W 上加一个低秩增量 ΔW = AB^T,
    本质是"把更新限制在低维子空间" — 与 Ridge / Lasso 一样,
    都是用"正则约束"来防止过拟合。
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


def load_data():
    """加载 California 房价数据。"""
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 标准化(线性模型 + 正则化必须做)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("─" * 60)
    print(f"数据集: california_housing  |  特征 {X.shape[1]} 维")
    print(f"训练: {len(X_train)} 条  |  测试: {len(X_test)} 条")
    print(f"目标: 房价中位数(单位: 10万美元)")
    print(f"特征: {list(feature_names)}")
    print("─" * 60)
    return X_train, X_test, y_train, y_test, feature_names


def evaluate(name, model, X_train, X_test, y_train, y_test):
    """训练并报告 RMSE / MAE / R²。"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  {name:30s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return rmse, mae, r2, model.coef_


def show_coefficients(coefs, feature_names, model_name):
    """打印权重大小和方向。"""
    print(f"\n  ──── {model_name} 权重 ────")
    order = np.argsort(np.abs(coefs))[::-1]
    for f_idx in order:
        w = coefs[f_idx]
        if w > 0:
            bar = "+" * min(40, int(abs(w) * 50))
        else:
            bar = "-" * min(40, int(abs(w) * 50))
        zero_marker = "  ← =0(被 L1 剔除)" if abs(w) < 1e-6 else ""
        print(f"    {feature_names[f_idx]:12s}  {w:+.4f}  {bar}{zero_marker}")


def alpha_path_analysis(X_train, X_test, y_train, y_test, feature_names):
    """绘制系数随正则化强度 α 的变化(文本版)。"""
    print("\n" + "═" * 60)
    print("  正则化路径:α 增大,系数被压缩或归零")
    print("═" * 60)

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

    print("\n  Lasso(L1)系数随 α 变化:")
    print(f"  {'α':>8s}  " + "".join(f"{n[:8]:>9s}" for n in feature_names))
    for a in alphas:
        m = Lasso(alpha=a, max_iter=10000)
        m.fit(X_train, y_train)
        coefs_str = "".join(
            f"{c:>9.3f}" if abs(c) > 1e-6 else f"{'.':>9s}" for c in m.coef_
        )
        n_zeros = int(np.sum(np.abs(m.coef_) < 1e-6))
        print(f"  α={a:<6.3f}{coefs_str}    [{n_zeros}/8 系数=0]")

    print("\n  Ridge(L2)系数随 α 变化:")
    print(f"  {'α':>8s}  " + "".join(f"{n[:8]:>9s}" for n in feature_names))
    for a in alphas:
        m = Ridge(alpha=a)
        m.fit(X_train, y_train)
        coefs_str = "".join(f"{c:>9.3f}" for c in m.coef_)
        print(f"  α={a:<6.3f}{coefs_str}")

    print("\n  💡 观察:")
    print("    - L2 让所有系数同步缩小,但不会归零")
    print("    - L1 在 α 较大时把不重要特征的系数直接压到 0(自动特征选择)")


def main():
    print("\n" + "█" * 60)
    print("█" + " " * 14 + "线性回归 + 正则化对比" + " " * 24 + "█")
    print("█" * 60)

    X_train, X_test, y_train, y_test, feature_names = load_data()

    # 三模型对比
    print("\n" + "═" * 60)
    print("  三模型测试集表现对比:")
    print("═" * 60)
    rmse_ols, _, _, coefs_ols = evaluate(
        "OLS (无正则)", LinearRegression(), X_train, X_test, y_train, y_test
    )
    rmse_ridge, _, _, coefs_ridge = evaluate(
        "Ridge (α=1.0)", Ridge(alpha=1.0), X_train, X_test, y_train, y_test
    )
    rmse_lasso, _, _, coefs_lasso = evaluate(
        "Lasso (α=0.1)",
        Lasso(alpha=0.1, max_iter=10000),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # 权重对比
    show_coefficients(coefs_ols, feature_names, "OLS")
    show_coefficients(coefs_ridge, feature_names, "Ridge α=1.0")
    show_coefficients(coefs_lasso, feature_names, "Lasso α=0.1")

    # α 路径分析
    alpha_path_analysis(X_train, X_test, y_train, y_test, feature_names)

    # 总结
    print("\n" + "═" * 60)
    print("  总结:")
    print("═" * 60)
    print("  - OLS / Ridge / Lasso 在小 α 下性能接近")
    print("  - Ridge 适合所有特征都有用的场景")
    print("  - Lasso 适合稀疏解(只有少数特征真正重要)")
    print("  - 实际中常用 ElasticNet = α₁·L1 + α₂·L2 综合两者\n")


if __name__ == "__main__":
    main()
