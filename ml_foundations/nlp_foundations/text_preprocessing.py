"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:中英文文本预处理对比                                  ║
║         分词 / 停用词 / 词干化 / 词形还原                          ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题:把"原始文本"变成"模型能吃的特征"需要哪些步骤?】
═══════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────┐
  │   英文流水线:                                                  │
  │     原文 → 小写化 → 分词 → 去停用词 → 词干化/词形还原           │
  │                                                             │
  │   中文流水线:                                                  │
  │     原文 → 分词(关键挑战!) → 去停用词                          │
  │     (中文没有"词干"概念)                                       │
  │                                                             │
  │   关键差异:                                                   │
  │     英文 — 单词由空格自然分开                                   │
  │     中文 — 必须靠算法分词("结合上海大学"歧义?)                  │
  └─────────────────────────────────────────────────────────────┘

  与 LLM 的关联:
    BPE / WordPiece tokenizer 直接学子词单元,绕过了"中文分词"
    这个传统难题。但中文 RAG / 关键词搜索 / 信息提取仍然需要分词。
"""

import re
import string
from collections import Counter

# ─────────────────────────────────────────────────────────────
# 测试样本
# ─────────────────────────────────────────────────────────────
ENGLISH_TEXTS = [
    "Machine learning algorithms are running on the GPUs to train large language models.",
    "The cats were chasing the mice running across the rooftops in the early morning.",
    "OpenAI released GPT-4 which has amazing reasoning capabilities for complex tasks.",
]

CHINESE_TEXTS = [
    "机器学习算法正在 GPU 上训练大语言模型,这是当前人工智能的核心方向。",
    "结合上海大学的研究成果,我们发现 Transformer 架构在多模态任务上表现优异。",
    "深度学习需要大量数据和算力,但模型性能也随之大幅提升。",
]


# ─────────────────────────────────────────────────────────────
# 英文预处理
# ─────────────────────────────────────────────────────────────
def english_pipeline(text):
    """完整的英文预处理流水线。"""
    print("\n" + "─" * 60)
    print(f"  原文: {text}")
    print("─" * 60)

    # 1. 小写化
    lower = text.lower()
    print(f"  ① 小写化:")
    print(f"     {lower}")

    # 2. 分词:用 nltk(若不可用则降级到正则)
    try:
        import nltk
        try:
            tokens = nltk.word_tokenize(lower)
        except LookupError:
            print("     (首次运行需下载 nltk punkt 包,自动下载中...)")
            nltk.download("punkt", quiet=True)
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass
            tokens = nltk.word_tokenize(lower)
    except Exception as e:
        # 降级:简单正则分词
        print(f"     (nltk 分词不可用: {e},降级到正则)")
        tokens = re.findall(r"[a-zA-Z]+|[0-9]+|[^\w\s]", lower)
    print(f"  ② 分词({len(tokens)} 个 token):")
    print(f"     {tokens}")

    # 3. 去标点
    no_punct = [t for t in tokens if t not in string.punctuation and t.strip()]
    print(f"  ③ 去标点:")
    print(f"     {no_punct}")

    # 4. 去停用词
    try:
        from nltk.corpus import stopwords
        try:
            sw = set(stopwords.words("english"))
        except LookupError:
            import nltk as _nltk
            _nltk.download("stopwords", quiet=True)
            sw = set(stopwords.words("english"))
    except Exception:
        # fallback 极简停用词
        sw = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
              "to", "of", "in", "on", "at", "and", "or", "but", "for", "with",
              "this", "that", "these", "those", "which"}
    no_stop = [t for t in no_punct if t not in sw]
    print(f"  ④ 去停用词(剩 {len(no_stop)} 个):")
    print(f"     {no_stop}")

    # 5. 词干化(Stemming)
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        stemmed = [stemmer.stem(t) for t in no_stop]
    except Exception:
        # 极简后缀剥离
        def _stem(w):
            for suf in ("ing", "ed", "es", "s", "ly"):
                if w.endswith(suf) and len(w) > len(suf) + 2:
                    return w[: -len(suf)]
            return w
        stemmed = [_stem(t) for t in no_stop]
    print(f"  ⑤ 词干化(Porter):")
    print(f"     {stemmed}")

    # 6. 词形还原(Lemmatization,基于词典更准确)
    try:
        from nltk.stem import WordNetLemmatizer
        try:
            lemmatizer = WordNetLemmatizer()
            lemmatized = [lemmatizer.lemmatize(t, pos="v") for t in no_stop]
        except LookupError:
            import nltk as _nltk
            _nltk.download("wordnet", quiet=True)
            lemmatizer = WordNetLemmatizer()
            lemmatized = [lemmatizer.lemmatize(t, pos="v") for t in no_stop]
    except Exception:
        lemmatized = stemmed  # fallback
    print(f"  ⑥ 词形还原(WordNet):")
    print(f"     {lemmatized}")

    return lemmatized


# ─────────────────────────────────────────────────────────────
# 中文预处理
# ─────────────────────────────────────────────────────────────
CHINESE_STOPWORDS = {
    "的", "了", "是", "在", "和", "我", "你", "他", "她", "它", "们",
    "也", "都", "就", "上", "下", "不", "有", "为", "与", "或", "并", "及",
    "这", "那", "这个", "那个", "因为", "所以", "但是", "而且", "如果",
    "于", "对", "把", "被", "让", "使", "可以", "可能", "需要",
    ",", "。", ";", ":", "?", "!", """, """, "'", "'", "(", ")",
    "（", "）", "—", "·",
}


def chinese_pipeline(text):
    """中文预处理流水线。"""
    print("\n" + "─" * 60)
    print(f"  原文: {text}")
    print("─" * 60)

    # 1. 分词:用 jieba
    try:
        import jieba
        jieba.setLogLevel(60)  # 关闭 jieba 的日志
        tokens = list(jieba.cut(text))
    except ImportError:
        # fallback:按字符切
        print("     (jieba 不可用,降级到逐字切分)")
        tokens = [c for c in text if c.strip()]

    print(f"  ① jieba 分词({len(tokens)} 个 token):")
    print(f"     {tokens}")

    # 2. 去停用词
    no_stop = [t for t in tokens if t.strip() and t not in CHINESE_STOPWORDS]
    print(f"  ② 去停用词(剩 {len(no_stop)} 个):")
    print(f"     {no_stop}")

    # 3. 词性标注(POS tagging)
    try:
        import jieba.posseg as pseg
        pos_tags = [(w, f) for w, f in pseg.cut(text) if w.strip()]
        print(f"  ③ 词性标注(前 10):")
        for w, f in pos_tags[:10]:
            tag_explain = {
                "n": "名词", "v": "动词", "a": "形容词", "d": "副词",
                "p": "介词", "c": "连词", "u": "助词", "m": "数词",
                "r": "代词", "x": "其他", "nr": "人名", "ns": "地名",
                "nt": "机构名", "eng": "英文", "nz": "其他专名",
            }
            tag_cn = tag_explain.get(f, f)
            print(f"     {w:8s}  {f:4s} ({tag_cn})")
    except Exception as e:
        print(f"  ③ 词性标注:跳过({e})")

    return no_stop


# ─────────────────────────────────────────────────────────────
# 中文分词的"歧义切分"演示
# ─────────────────────────────────────────────────────────────
def chinese_ambiguity_demo():
    """jieba 处理歧义的能力。"""
    print("\n" + "═" * 60)
    print("  中文分词的歧义挑战")
    print("═" * 60)

    cases = [
        "结合上海大学的研究成果",        # 上海大学 vs 上海/大学?
        "南京市长江大桥",                # 南京/市长/江大桥? vs 南京市/长江大桥?
        "我喜欢吃苹果",                  # 苹果(水果 or 公司)?
        "他从马上下来",                  # 从马上/下来 vs 从马/上下来?
    ]
    try:
        import jieba
        jieba.setLogLevel(60)
        for s in cases:
            print(f"\n  '{s}'")
            print(f"     → 默认: {' / '.join(jieba.cut(s))}")
            print(f"     → 全模式: {' / '.join(jieba.cut(s, cut_all=True))}")
    except ImportError:
        print("  jieba 不可用,跳过此节")

    print("\n  💡 jieba 默认用 HMM + 词典联合,能处理大多数常见歧义")
    print("     学术界更准确的工具:THULAC / pkuseg / LAC")


# ─────────────────────────────────────────────────────────────
# 词频统计
# ─────────────────────────────────────────────────────────────
def show_token_distribution(all_tokens, lang_label):
    print(f"\n  {lang_label}:Top 10 高频词")
    counter = Counter(all_tokens)
    for w, c in counter.most_common(10):
        bar = "█" * c
        print(f"    {w:15s}  count={c}  {bar}")


def main():
    print("\n" + "█" * 60)
    print("█" + " " * 16 + "中英文文本预处理对比" + " " * 21 + "█")
    print("█" * 60)

    print("\n" + "═" * 60)
    print("  ※ 英文流水线 ※")
    print("═" * 60)
    en_all = []
    for txt in ENGLISH_TEXTS:
        tokens = english_pipeline(txt)
        en_all.extend(tokens)
    show_token_distribution(en_all, "英文")

    print("\n" + "═" * 60)
    print("  ※ 中文流水线 ※")
    print("═" * 60)
    cn_all = []
    for txt in CHINESE_TEXTS:
        tokens = chinese_pipeline(txt)
        cn_all.extend(tokens)
    show_token_distribution(cn_all, "中文")

    chinese_ambiguity_demo()

    print("\n" + "═" * 60)
    print("  关键收获:")
    print("═" * 60)
    print("  ✓ 英文流水线:小写 → 分词 → 去标点 → 去停用词 → 词干化")
    print("  ✓ 中文流水线:分词 → 去停用词(无词干化概念)")
    print("  ✓ 中文分词处理歧义靠 HMM+词典(jieba 是事实标准)")
    print("  ✓ LLM 时代用 BPE/WordPiece 绕过分词,但 RAG 关键词检索仍需要")
    print("  ✓ 词形还原比词干化更准确,但需要词典(慢一些)\n")


if __name__ == "__main__":
    main()
