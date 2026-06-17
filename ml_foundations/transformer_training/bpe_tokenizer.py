"""
╔══════════════════════════════════════════════════════════════════╗
║         项目:从零实现 BPE Tokenizer                                ║
║         GPT 用的子词分词器是怎么造出来的                            ║
╚══════════════════════════════════════════════════════════════════╝

【核心问题:tokenizer 是怎么从原始字符学到 "the"、"ing" 这类常见子词的?】

  BPE (Byte-Pair Encoding) 算法:
    1. 初始词表 = 所有字符
    2. 统计语料中相邻 token 对出现频次
    3. 合并最高频的 pair → 新 token
    4. 重复 N 次,得到最终词表

  与 LLM 的关联:
    GPT-2: ~50K BPE 词表
    GPT-4 (cl100k_base): ~100K
    LLaMA-3: ~128K
    本 demo 训 ~265 词表,展示算法骨架,与生产 tokenizer 数学完全一致。
"""

import re
from collections import Counter
from pathlib import Path

class BPETokenizer:
    def __init__(self):
        self.merges = []           # list of (pair, new_token_id)
        self.vocab = {}            # int -> bytes
        self.token_to_id = {}      # bytes -> int

    def _get_stats(self, ids_list):
        """统计相邻 pair 频次。"""
        counts = Counter()
        for ids in ids_list:
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] += 1
        return counts

    def _merge(self, ids_list, pair, new_id):
        out = []
        for ids in ids_list:
            new_ids, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                    new_ids.append(new_id); i += 2
                else:
                    new_ids.append(ids[i]); i += 1
            out.append(new_ids)
        return out

    def train(self, text, num_merges=200, verbose_first_n=30):
        # init: each byte -> int (0..255)
        ids_list = [list(s.encode("utf-8")) for s in text.split("\n") if s.strip()]
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []

        for step in range(num_merges):
            stats = self._get_stats(ids_list)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + step
            ids_list = self._merge(ids_list, pair, new_id)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges.append((pair, new_id))
            if step < verbose_first_n:
                merged_str = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"  step {step:3d}  merge {pair} → {new_id:3d}  '{merged_str}'  (count={stats[pair]})")

        self.token_to_id = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        for pair, new_id in self.merges:
            ids = self._merge([ids], pair, new_id)[0]
        return ids

    def decode(self, ids):
        out = b"".join(self.vocab[i] for i in ids)
        return out.decode("utf-8", errors="replace")


def main():
    print("\n" + "█" * 60)
    print("█" + " " * 18 + "BPE Tokenizer 从零训练" + " " * 18 + "█")
    print("█" * 60)

    corpus_path = Path(__file__).parent / "data" / "tiny_shakespeare.txt"
    if not corpus_path.exists():
        print(f"  ❌ 语料文件未找到: {corpus_path}")
        print("     请运行: curl -L -o {corpus_path} \\")
        print("       https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        return
    text = corpus_path.read_text()[:50_000]  # 前 50KB 训练演示
    print(f"\n  语料: tiny_shakespeare.txt 前 {len(text)} 字符")

    tok = BPETokenizer()
    print("\n  ──── 训练 200 轮合并(展示前 30 轮) ────")
    tok.train(text, num_merges=200, verbose_first_n=30)
    print(f"\n  最终词表大小: {len(tok.vocab)}  (256 字节 + {len(tok.merges)} 合并)")

    sample = "ROMEO: But soft, what light through yonder window breaks?"
    char_count = len(sample.encode("utf-8"))
    bpe_ids = tok.encode(sample)
    print(f"\n  ──── 编码示例 ────")
    print(f"  原文({char_count} 字节): {sample}")
    print(f"  BPE  ({len(bpe_ids)} tokens): {bpe_ids[:20]}{'...' if len(bpe_ids) > 20 else ''}")
    print(f"  压缩比: {char_count/len(bpe_ids):.2f}x")
    print(f"  解码回原文: {tok.decode(bpe_ids)}")

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        gpt4_ids = enc.encode(sample)
        print(f"\n  ──── 与 GPT-4 (cl100k_base, ~100K 词表) 对比 ────")
        print(f"  本 demo: {len(bpe_ids)} tokens")
        print(f"  GPT-4 : {len(gpt4_ids)} tokens (词表大 ~400 倍 → token 更短)")
    except ImportError:
        pass

    print("\n  关键收获:")
    print("  ✓ BPE = 反复合并最高频字符对")
    print("  ✓ 词表越大压缩越好,但 embedding 矩阵也越大")
    print("  ✓ 字节级初始化保证可处理任意 Unicode")

if __name__ == "__main__":
    main()
