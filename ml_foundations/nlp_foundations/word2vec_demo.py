"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:Word2Vec 词向量训练与语义算术                        ║
║         理解"king - man + woman ≈ queen"的奇迹是怎么发生的         ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题:词向量为什么能做"语义算术"?】
═══════════════════════════════════════════════════════════════════

  分布假设(Distributional Hypothesis):
    "你认识一个词,是因为你认识它的邻居"
    意思相近的词,出现在相似的上下文里。
    → "苹果"和"橙子"的邻居词重合度高 → 它们的向量也接近。

  ┌─────────────────────────────────────────────────────────────┐
  │   Word2Vec 两种训练目标:                                       │
  │                                                             │
  │   Skip-Gram   :  给中心词,预测上下文                           │
  │     "机器 [学习] 是 人工智能 的 一个 分支"                     │
  │                  ↑                                          │
  │                中心词       目标:学习→机器,学习→是,等         │
  │                                                             │
  │   CBOW         :  给上下文,预测中心词                         │
  │     "机器 ___ 是 人工智能"  →  预测 [学习]                    │
  │                                                             │
  │   损失函数(简化版):                                          │
  │     L = -log P(o|c) = -log [exp(v_o·v_c) / Σ exp(v_w·v_c)]  │
  │     用负采样(negative sampling)避免遍历整个词表               │
  └─────────────────────────────────────────────────────────────┘

  神奇现象:
    king - man + woman ≈ queen
    Paris - France + Italy ≈ Rome
    walking - walk + swim ≈ swimming

  为什么?
    因为词向量在某些方向上编码了"性别"、"国家-首都"、"动词时态"
    这些语义关系。这是"在大规模文本中学习"的副产品。

  与 LLM 的关联:
    LLM 的 token embedding 沿用了 Word2Vec 的精神,但:
      Word2Vec → 静态:同一个词永远一个向量
      LLM      → 动态:同一个词在不同句子里向量不同(由 attention 决定)
    后者解决了一词多义问题(bank=银行 vs 河岸)。
"""

import sys

# ─────────────────────────────────────────────────────────────
# 训练语料(中英混合,展示双语都能学)
# ─────────────────────────────────────────────────────────────
CORPUS_EN = [
    "the cat sat on the mat",
    "the dog sat on the floor",
    "cats and dogs are pets",
    "the king ruled the kingdom with wisdom",
    "the queen was loved by the people",
    "the man walked into the palace",
    "the woman walked into the palace",
    "the boy played with the toy",
    "the girl played with the doll",
    "paris is the capital of france",
    "london is the capital of england",
    "rome is the capital of italy",
    "berlin is the capital of germany",
    "tokyo is the capital of japan",
    "beijing is the capital of china",
    "machine learning is a branch of artificial intelligence",
    "deep learning uses neural networks",
    "transformer is a kind of neural network architecture",
    "neural networks have many layers",
    "the king and the queen lived in the palace",
    "the man loved the woman",
    "the boy loved the girl",
    "kings and queens rule countries",
    "men and women are equal",
    "boys and girls go to school",
    "cats chase mice",
    "dogs chase cats",
    "the model learns from data",
    "the algorithm processes input",
    "training requires a lot of data",
    "language models predict the next word",
    "embeddings are vectors representing words",
    "vectors encode semantic information",
    "similar words have similar vectors",
] * 20  # 重复扩充语料

CORPUS_CN = [
    "机器 学习 是 人工智能 的 分支",
    "深度 学习 使用 神经网络 处理 数据",
    "国王 统治 王国",
    "女王 被 人民 爱戴",
    "男人 走进 宫殿",
    "女人 走进 宫殿",
    "男孩 喜欢 玩具",
    "女孩 喜欢 玩具",
    "巴黎 是 法国 的 首都",
    "伦敦 是 英国 的 首都",
    "罗马 是 意大利 的 首都",
    "北京 是 中国 的 首都",
    "东京 是 日本 的 首都",
    "国王 和 女王 住在 宫殿",
    "男人 和 女人 都 在 工作",
    "猫 喜欢 老鼠",
    "狗 喜欢 骨头",
    "模型 从 数据 学习",
    "算法 处理 输入",
    "训练 需要 大量 数据",
    "Transformer 是 神经网络 架构",
    "注意力 机制 是 核心",
    "向量 编码 语义 信息",
    "相似 的 词 有 相似 的 向量",
] * 25


# ─────────────────────────────────────────────────────────────
# 训练 Word2Vec
# ─────────────────────────────────────────────────────────────
def train_word2vec(corpus, vector_size=50, window=3, sg=1, min_count=1, epochs=30):
    """sg=1 用 Skip-Gram, sg=0 用 CBOW。"""
    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("  ❌ gensim 未安装,请运行 pip install gensim")
        sys.exit(1)

    sentences = [s.split() for s in corpus]
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        sg=sg,
        min_count=min_count,
        epochs=epochs,
        workers=2,
        seed=42,
    )
    return model


# ─────────────────────────────────────────────────────────────
# 探索:相似词、类比推理、向量大小
# ─────────────────────────────────────────────────────────────
def show_vocab_and_dim(model, name):
    print(f"\n  {name} 词表大小: {len(model.wv.key_to_index)}")
    print(f"  词向量维度: {model.wv.vector_size}")


def show_similar_words(model, words, topn=5):
    print("\n  ──── 相似词查询 ────")
    for w in words:
        if w not in model.wv.key_to_index:
            print(f"  '{w}' 不在词表里")
            continue
        sims = model.wv.most_similar(w, topn=topn)
        sim_str = ", ".join(f"{x[0]}({x[1]:.3f})" for x in sims)
        print(f"  与 '{w}' 最相似: {sim_str}")


def analogy(model, a, b, c, topn=3):
    """a:b :: c:?  (即 b - a + c ≈ ?)"""
    if not all(w in model.wv.key_to_index for w in (a, b, c)):
        missing = [w for w in (a, b, c) if w not in model.wv.key_to_index]
        return f"  ❌ 词表缺失: {missing}"
    result = model.wv.most_similar(positive=[b, c], negative=[a], topn=topn)
    res_str = ", ".join(f"{x[0]}({x[1]:.3f})" for x in result)
    return f"  '{a}' : '{b}' :: '{c}' : ?  →  {res_str}"


def show_analogies(model, pairs):
    print("\n  ──── 语义算术(类比推理) ────")
    print("  形式: a : b :: c : ?    (b - a + c ≈ ?)")
    for a, b, c in pairs:
        print(analogy(model, a, b, c))


def visualize_vectors_2d(model, words, name):
    """把若干词向量降到 2 维,字符画展示。"""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return
    valid = [w for w in words if w in model.wv.key_to_index]
    if len(valid) < 2:
        return
    vecs = [model.wv[w] for w in valid]
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(vecs)

    print(f"\n  ──── {name} 词向量 PCA 二维投影 ────")
    width, height = 50, 14
    xs, ys = reduced[:, 0], reduced[:, 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for w, (x, y) in zip(valid, reduced):
        gx = int((x - x_min) / (x_max - x_min) * (width - 1))
        gy = int((y_max - y) / (y_max - y_min) * (height - 1))
        gy = max(0, min(height - 1, gy))
        gx = max(0, min(width - 1, gx))
        # 放置词的首字
        if grid[gy][gx] == " ":
            grid[gy][gx] = w[0]

    print("  ┌" + "─" * width + "┐")
    for row in grid:
        print("  │" + "".join(row) + "│")
    print("  └" + "─" * width + "┘")
    print("  (每个字符是对应词的首字,位置反映语义距离)")


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "█" * 60)
    print("█" + " " * 14 + "Word2Vec 词向量训练与语义算术" + " " * 14 + "█")
    print("█" * 60)

    # 1. 英文 Word2Vec
    print("\n" + "═" * 60)
    print("  ※ 英文 Skip-Gram Word2Vec ※")
    print("═" * 60)
    en_model = train_word2vec(CORPUS_EN, vector_size=50, window=3, sg=1, epochs=30)
    show_vocab_and_dim(en_model, "英文模型")

    # 相似词
    show_similar_words(en_model, ["king", "queen", "machine", "paris"])

    # 类比推理
    show_analogies(en_model, [
        ("king", "queen", "man"),       # → 期望 "woman"
        ("man", "woman", "boy"),        # → 期望 "girl"
        ("paris", "france", "rome"),    # → 期望 "italy"
        ("paris", "france", "tokyo"),   # → 期望 "japan"
    ])

    # 词向量可视化
    visualize_vectors_2d(en_model, [
        "king", "queen", "man", "woman", "boy", "girl",
        "paris", "rome", "london", "tokyo", "beijing",
        "cat", "dog", "machine", "neural", "model",
    ], "英文")

    # 2. 中文 Word2Vec(语料较小,效果会差一些)
    print("\n" + "═" * 60)
    print("  ※ 中文 Skip-Gram Word2Vec(小语料,仅作演示) ※")
    print("═" * 60)
    cn_model = train_word2vec(CORPUS_CN, vector_size=50, window=3, sg=1, epochs=50)
    show_vocab_and_dim(cn_model, "中文模型")

    show_similar_words(cn_model, ["国王", "巴黎", "机器", "学习"])

    show_analogies(cn_model, [
        ("国王", "女王", "男人"),
        ("巴黎", "法国", "罗马"),
        ("巴黎", "法国", "东京"),
    ])

    visualize_vectors_2d(cn_model, [
        "国王", "女王", "男人", "女人", "男孩", "女孩",
        "巴黎", "罗马", "伦敦", "东京", "北京",
        "机器", "学习", "深度", "数据",
    ], "中文")

    # 3. 数学层面:展示一个具体的向量算术
    print("\n" + "═" * 60)
    print("  深入:king - man + woman 的向量算术(英文)")
    print("═" * 60)
    if all(w in en_model.wv.key_to_index for w in ("king", "man", "woman", "queen")):
        v_king = en_model.wv["king"]
        v_man = en_model.wv["man"]
        v_woman = en_model.wv["woman"]
        v_queen = en_model.wv["queen"]
        result = v_king - v_man + v_woman

        # 余弦相似度
        import numpy as np
        def cos(a, b):
            return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

        print(f"  cos(king - man + woman, queen) = {cos(result, v_queen):.4f}")
        print(f"  cos(king - man + woman, king)  = {cos(result, v_king):.4f}")
        print(f"  cos(queen, king)                = {cos(v_queen, v_king):.4f}")
        print("  (越接近 1 越相似)")

    print("\n" + "═" * 60)
    print("  关键收获:")
    print("═" * 60)
    print("  ✓ Word2Vec 通过\"分布假设\"学出稠密词向量")
    print("  ✓ 向量在某些方向编码了语义关系(性别、国家-首都)")
    print("  ✓ 类比推理 = 向量加减,这是表示学习的\"涌现\"现象")
    print("  ✓ 静态词向量的局限:同一个词永远一个向量(无法处理一词多义)")
    print("  ✓ LLM 用动态 contextual embedding 解决了这个问题\n")


if __name__ == "__main__":
    main()
