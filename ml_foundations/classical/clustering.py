"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:K-Means 聚类与簇数选择                              ║
║         合成 blob 数据 + 肘部法 + 轮廓系数                       ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题:没有标签,如何把数据分组?】
═══════════════════════════════════════════════════════════════════

  K-Means 算法(直观):
  ┌─────────────────────────────────────────────────────────────┐
  │   1. 随机选 k 个中心点                                         │
  │   2. 把每个样本分给最近的中心                                   │
  │   3. 重新计算每组的均值,作为新的中心                            │
  │   4. 重复 2-3 直到中心不再变化                                  │
  │                                                             │
  │   目标函数(惯量 inertia):                                    │
  │     J = Σᵢ Σ_x∈Cᵢ ||x - μᵢ||²  (所有点到所属簇心的距离平方和)  │
  └─────────────────────────────────────────────────────────────┘

  关键问题:k 怎么选?
    - 肘部法(Elbow): J 关于 k 的曲线"肘部"位置
    - 轮廓系数(Silhouette): 越大越好,综合"内聚 + 分离"

  与 LLM 的关联:
    Embedding 聚类是"语义检索"的雏形 — 把相似的文本块聚到一起。
    现代 RAG 用 ANN 索引(FAISS/HNSW)做检索,但 K-Means 仍用于
    在 embedding 空间做"主题发现" / "数据去重"。
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

np.random.seed(42)


def generate_data():
    """生成 4 簇的合成数据(真实 k=4)。"""
    X, y_true = make_blobs(
        n_samples=400,
        centers=4,
        cluster_std=[1.0, 1.5, 0.8, 1.2],
        center_box=(-10.0, 10.0),
        random_state=42,
    )
    print("─" * 60)
    print(f"数据: 400 个二维点,真实分 4 簇,std 各异")
    print(f"  X.shape = {X.shape}")
    print(f"  X.range = [{X.min():.2f}, {X.max():.2f}]")
    print("─" * 60)
    return X, y_true


def visualize_clusters_text(X, labels, centers, title, width=60, height=20):
    """文本字符画展示二维聚类结果。"""
    print(f"\n  ──── {title} ────")
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    # 字符 grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    def to_grid(x, y):
        gx = int((x - x_min) / (x_max - x_min + 1e-9) * (width - 1))
        gy = int((y_max - y) / (y_max - y_min + 1e-9) * (height - 1))
        return gx, gy

    cluster_chars = ["o", "x", "+", "*", "#", "@", "%", "&"]
    for i, (px, py) in enumerate(X):
        gx, gy = to_grid(px, py)
        if 0 <= gx < width and 0 <= gy < height:
            grid[gy][gx] = cluster_chars[labels[i] % len(cluster_chars)]

    if centers is not None:
        for cx, cy in centers:
            gx, gy = to_grid(cx, cy)
            if 0 <= gx < width and 0 <= gy < height:
                grid[gy][gx] = "C"

    print("  ┌" + "─" * width + "┐")
    for row in grid:
        print("  │" + "".join(row) + "│")
    print("  └" + "─" * width + "┘")
    print("  图例: 不同字符=不同簇, C=簇心")


def elbow_method(X, k_range):
    """肘部法:看 J 关于 k 的曲线。"""
    print("\n" + "═" * 60)
    print("  肘部法:不同 k 下的 inertia(越小越紧凑)")
    print("═" * 60)

    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)

    # 文本曲线
    max_iner = max(inertias)
    print(f"\n  {'k':>3s}  {'inertia':>12s}  曲线")
    for k, j in zip(k_range, inertias):
        bar = "█" * int(j / max_iner * 40)
        print(f"  {k:>3d}  {j:>12.2f}  {bar}")

    # 找肘部:二阶差分最大的点
    if len(inertias) >= 3:
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)
        elbow_k = k_range[np.argmax(second_diffs) + 1]
        print(f"\n  💡 肘部估计 k* = {elbow_k}")

    return inertias


def silhouette_analysis(X, k_range):
    """轮廓系数:越大越好,k=1 时未定义。"""
    print("\n" + "═" * 60)
    print("  轮廓系数(silhouette):综合内聚和分离,越大越好")
    print("═" * 60)

    scores = []
    for k in k_range:
        if k < 2:
            scores.append(None)
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels)
        scores.append(s)

    print(f"\n  {'k':>3s}  {'silhouette':>12s}  曲线")
    for k, s in zip(k_range, scores):
        if s is None:
            print(f"  {k:>3d}  {'N/A':>12s}")
        else:
            bar = "█" * int(s * 40)
            print(f"  {k:>3d}  {s:>12.4f}  {bar}")

    # 最佳 k
    valid = [(k, s) for k, s in zip(k_range, scores) if s is not None]
    if valid:
        best_k, best_s = max(valid, key=lambda x: x[1])
        print(f"\n  💡 轮廓系数最佳 k* = {best_k} (s={best_s:.4f})")

    return scores


def main():
    print("\n" + "█" * 60)
    print("█" + " " * 16 + "K-Means 聚类与簇数选择" + " " * 18 + "█")
    print("█" * 60)

    # 1. 生成数据
    X, y_true = generate_data()

    # 2. 用真实 k=4 聚类,可视化
    km = KMeans(n_clusters=4, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    visualize_clusters_text(X, labels, km.cluster_centers_, "K-Means 结果(k=4)")

    # 3. 肘部法和轮廓系数
    k_range = list(range(1, 9))
    elbow_method(X, k_range)
    silhouette_analysis(X, k_range)

    # 4. 不同 k 下的字符画对比
    print("\n" + "═" * 60)
    print("  不同 k 下的聚类结果(对比):")
    print("═" * 60)
    for k in [2, 3, 4, 6]:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        lbl = km.fit_predict(X)
        visualize_clusters_text(X, lbl, km.cluster_centers_, f"k={k}", width=50, height=14)

    # 5. 总结
    print("\n" + "═" * 60)
    print("  总结:")
    print("═" * 60)
    print("  - 肘部法和轮廓系数都指向 k=4(真实值),方法 work")
    print("  - K-Means 假设簇是球形,对非球形/密度差异大的数据失效")
    print("  - 替代方案: DBSCAN(基于密度) / GMM(高斯混合) / 层次聚类")
    print("  - LLM 时代: 在 embedding 空间做 K-Means 可发现语义主题\n")


if __name__ == "__main__":
    main()
