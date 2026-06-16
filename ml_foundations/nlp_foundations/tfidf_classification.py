"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:TF-IDF + 经典文本分类                                ║
║         20-newsgroups 新闻分类:朴素贝叶斯 vs 线性 SVM            ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题:在 LLM 出现之前,文本分类的标准做法是什么?】
═══════════════════════════════════════════════════════════════════

  数据集:20-newsgroups 子集(4 个主题)
    sci.space          (太空)
    rec.sport.baseball (棒球)
    talk.politics.guns (政治-枪支)
    comp.graphics      (计算机图形)

  ┌─────────────────────────────────────────────────────────────┐
  │   流水线:                                                     │
  │                                                             │
  │   原始新闻文本                                                │
  │      ↓                                                       │
  │   TfidfVectorizer  → 稀疏矩阵 (n_docs, vocab_size)           │
  │      ↓                                                       │
  │   分类器(朴素贝叶斯 / 线性 SVM)                              │
  │      ↓                                                       │
  │   预测主题                                                    │
  │                                                             │
  │   TF-IDF 公式:                                              │
  │     TF(t,d) = t 在 d 中出现次数 / d 总词数                  │
  │     IDF(t) = log(N / 包含 t 的文档数)                       │
  │     TF-IDF(t,d) = TF · IDF                                  │
  └─────────────────────────────────────────────────────────────┘

  与 LLM 的关联:
    BM25(TF-IDF 改进版)在现代 RAG 中作为"稀疏检索"组件,
    与"密集向量检索"形成混合检索 — 工业级 RAG 标配。
    本文件展示了"为什么 BM25 仍未被淘汰"的底层原因。
"""

import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


CATEGORIES = [
    "sci.space",
    "rec.sport.baseball",
    "talk.politics.guns",
    "comp.graphics",
]
CATEGORY_CN = {
    "sci.space": "太空科学",
    "rec.sport.baseball": "棒球",
    "talk.politics.guns": "政治-枪支",
    "comp.graphics": "计算机图形",
}


def load_data():
    """从 sklearn 拉取 20-newsgroups 子集。"""
    print("─" * 60)
    print("  加载 20-newsgroups 数据(首次运行需下载,约 14MB)")
    print("─" * 60)

    train = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),  # 去掉元信息防止泄漏
        random_state=42,
    )
    test = fetch_20newsgroups(
        subset="test",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )

    print(f"  训练集: {len(train.data)} 篇")
    print(f"  测试集: {len(test.data)} 篇")
    print(f"  类别 ({len(train.target_names)}):")
    for i, name in enumerate(train.target_names):
        n_train = (train.target == i).sum()
        cn = CATEGORY_CN.get(name, "")
        print(f"    [{i}] {name:30s} {cn:15s}  训练 {n_train} 篇")
    return train, test


def show_sample(train, idx=0):
    """打印一篇样本看看长什么样。"""
    print("\n  ──── 样本预览 ────")
    label = train.target_names[train.target[idx]]
    text = train.data[idx]
    print(f"  类别: {label} ({CATEGORY_CN.get(label, '')})")
    print(f"  文本(前 300 字):")
    preview = text[:300].replace("\n", "\n         ")
    print(f"         {preview}{'...' if len(text) > 300 else ''}")


def show_top_features_per_class(vectorizer, classifier, target_names, top_k=8):
    """对线性分类器,展示每类最有判别力的关键词。"""
    print("\n  ──── 每类的 top 关键词(基于线性权重) ────")
    feature_names = vectorizer.get_feature_names_out()

    # 获取系数:线性 SVM / 逻辑回归是 (n_classes, n_features)
    if hasattr(classifier, "coef_"):
        coefs = classifier.coef_
        for cls_idx, cls_name in enumerate(target_names):
            top_idx = coefs[cls_idx].argsort()[-top_k:][::-1]
            top_words = [feature_names[i] for i in top_idx]
            cn = CATEGORY_CN.get(cls_name, "")
            print(f"  [{cls_name}] ({cn}):")
            print(f"    {', '.join(top_words)}")


def evaluate_classifier(name, classifier, train, test, vectorizer):
    """训练 + 测试 + 报告。"""
    print(f"\n{'═'*60}")
    print(f"  分类器: {name}")
    print("═" * 60)

    # 流水线:vectorizer → classifier
    pipe = Pipeline([("tfidf", vectorizer), ("clf", classifier)])

    t0 = time.time()
    pipe.fit(train.data, train.target)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = pipe.predict(test.data)
    pred_time = time.time() - t0

    acc = accuracy_score(test.target, y_pred)
    print(f"  训练耗时: {train_time:.2f}s")
    print(f"  推理耗时: {pred_time:.2f}s ({len(test.data)} 篇 → {pred_time/len(test.data)*1000:.2f}ms/篇)")
    print(f"  测试集准确率: {acc:.4f}")

    # 分类报告
    print("\n  分类报告:")
    report = classification_report(
        test.target, y_pred, target_names=test.target_names, digits=3
    )
    for line in report.split("\n"):
        print(f"    {line}")

    # 混淆矩阵
    cm = confusion_matrix(test.target, y_pred)
    print("  混淆矩阵 (行=真实, 列=预测):")
    short = [n.split(".")[-1][:8] for n in test.target_names]
    print("        " + "  ".join(f"{n:>8s}" for n in short))
    for i, row in enumerate(cm):
        cells = "  ".join(f"{v:>8d}" for v in row)
        print(f"    {short[i]:>6s}  {cells}")

    # 关键词
    show_top_features_per_class(
        pipe.named_steps["tfidf"],
        pipe.named_steps["clf"],
        test.target_names,
    )

    return acc, train_time


def vocab_growth_demo():
    """展示 max_features 对性能的影响。"""
    print("\n" + "═" * 60)
    print("  词表大小 vs 性能(展示稀疏特征的特点)")
    print("═" * 60)

    train = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )
    test = fetch_20newsgroups(
        subset="test",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )

    print(f"\n  {'max_features':>13s}  {'实际词表':>10s}  {'测试 acc':>8s}  {'耗时':>6s}")
    for max_feat in [500, 1000, 5000, 10000, None]:
        vec = TfidfVectorizer(
            max_features=max_feat,
            stop_words="english",
            ngram_range=(1, 1),
            min_df=2,
        )
        clf = MultinomialNB()
        pipe = Pipeline([("tfidf", vec), ("clf", clf)])
        t0 = time.time()
        pipe.fit(train.data, train.target)
        acc = pipe.score(test.data, test.target)
        elapsed = time.time() - t0
        actual = len(pipe.named_steps["tfidf"].vocabulary_)
        cap = "无上限" if max_feat is None else str(max_feat)
        print(f"  {cap:>13s}  {actual:>10d}  {acc:>8.4f}  {elapsed:>5.2f}s")


def main():
    print("\n" + "█" * 60)
    print("█" + " " * 12 + "TF-IDF + 朴素贝叶斯 / 线性 SVM" + " " * 13 + "█")
    print("█" * 60)

    train, test = load_data()
    show_sample(train, idx=0)

    # 共用一个 vectorizer 模板
    def make_vectorizer():
        return TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            min_df=2,           # 去掉只出现 1 次的词
            max_df=0.95,        # 去掉太常见的词
            ngram_range=(1, 2),  # uni + bigram
        )

    # 朴素贝叶斯
    nb_acc, nb_time = evaluate_classifier(
        "MultinomialNB",
        MultinomialNB(),
        train,
        test,
        make_vectorizer(),
    )

    # 线性 SVM
    svm_acc, svm_time = evaluate_classifier(
        "LinearSVC",
        LinearSVC(C=1.0, random_state=42),
        train,
        test,
        make_vectorizer(),
    )

    # 逻辑回归
    lr_acc, lr_time = evaluate_classifier(
        "LogisticRegression",
        LogisticRegression(max_iter=1000, random_state=42),
        train,
        test,
        make_vectorizer(),
    )

    # 总结
    print("\n" + "═" * 60)
    print("  三模型对比:")
    print("═" * 60)
    print(f"  {'模型':30s}  {'测试 acc':>9s}  {'训练时间':>9s}")
    print("  " + "─" * 56)
    print(f"  {'MultinomialNB(朴素贝叶斯)':30s}  {nb_acc:>9.4f}  {nb_time:>8.2f}s")
    print(f"  {'LinearSVC(线性 SVM)':30s}  {svm_acc:>9.4f}  {svm_time:>8.2f}s")
    print(f"  {'LogisticRegression(逻辑回归)':30s}  {lr_acc:>9.4f}  {lr_time:>8.2f}s")

    vocab_growth_demo()

    print("\n" + "═" * 60)
    print("  关键收获:")
    print("═" * 60)
    print("  ✓ TF-IDF + 线性分类器在 4 类新闻上能轻松达 80%+ acc")
    print("  ✓ 朴素贝叶斯训练最快,适合极大规模文本")
    print("  ✓ Bigram 提供了微弱的局部语序信息(优于纯 unigram)")
    print("  ✓ 模型可解释:每类的 top 关键词直接可读")
    print("  ✓ LLM 时代:这套方法仍是 RAG 中 BM25 检索的精神基础\n")


if __name__ == "__main__":
    main()
