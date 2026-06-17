# BPE Tokenization

> 模型不认识字符串 —— 在第一行训练代码运行之前，语料必须先变成整数序列

---

## 一、为什么需要 tokenization？

神经网络的输入是**数字**，不是字符。把原始文本转换为整数 id 序列，就是 tokenization 要做的事。

```
"Hello, world!"
      ↓  tokenizer
[9906, 11, 1917, 0]
      ↓  embedding lookup
[[0.2, -0.1, ...], ...]   ← 模型真正处理的张量
```

这一步看起来平凡，但它直接决定：
- 序列有多长（影响 attention 的 $O(T^2)$ 开销）
- 词表有多大（影响 embedding 矩阵参数量）
- 是否能处理新词、新语言、emoji

### 1.1 三种粒度的对比

| 方案 | 词表大小 | 序列长度 | OOV 问题 | 典型应用 |
|------|---------|---------|---------|---------|
| **字符级（char-level）** | ~100 | 长 | 无 | nanoGPT 教学、本 demo |
| **词级（word-level）** | 数十万 | 短 | 严重 | 经典 NLP（Word2Vec 时代） |
| **子词（subword / BPE）** | 1 万–10 万 | 中 | 几乎无 | 现代所有 LLM |

::: tip LLM 视角
字符级在教学场景很受欢迎（零依赖、直观），但生产 LLM 全部用子词。原因很简单：字符级序列太长，同样的文本内容要占用多倍的 context window，attention 开销随长度平方增长。

本目录的 `gpt_train.py` 用字符级（vocab≈65）是为了"5 分钟看到 loss 收敛"，而本节的 `bpe_tokenizer.py` 才是真正的 LLM tokenizer 原型。
:::

---

## 二、字节级、字符级与 subword

### 2.1 字符级 vs 字节级

**字符级**：把文本拆分为 Unicode 字符（码点），中文每个汉字是一个 token，emoji 是一个 token。

**字节级（byte-level）**：把文本 UTF-8 编码后，以**字节**（0–255）为最小单元。

```
"你好"
  字符级: ['你', '好']           → 需要汉字在词表里
  字节级: [0xe4, 0xbd, 0xa0,    → 三个字节，一定在 0–255 范围内
            0xe5, 0xa5, 0xbd]
```

Unicode 有超过 140,000 个码点，覆盖全部需要大词表；而字节级初始词表恒定 256 个，**任何 Unicode 输入都能处理，无需 OOV 处理**。

### 2.2 GPT-2 为什么选字节级 BPE

GPT-2 的 tokenizer 基础是 byte-level BPE（Radford 2019）：

```
初始词表 = 256 个字节 token
↓ 在字节序列上做 BPE 合并
↓ 最终词表 ≈ 50,257（256 + 50,000 次合并 + 1 个特殊 token）
```

**好处**：
- 任何输入（emoji、代码、任意语言）都能编码，压根不存在 unknown token
- 中文等非 ASCII 语言虽然效率低（一个汉字需 2–3 个 token），但至少能处理

::: tip LLM 视角
cl100k_base（GPT-4 用的编码）同样是 byte-level BPE，词表约 10 万，对中文的单字 token 覆盖更好：很多常见汉字能对应一个 token 而非多个字节。这也是词表越来越大的动机之一。
:::

---

## 三、BPE 算法逐步推演

### 3.1 算法思想

BPE（Byte-Pair Encoding）最初是一种数据压缩算法，Sennrich 2016 把它引入 NLP：

> 反复找到语料中最高频的相邻 token 对，把它们合并成一个新 token。重复 $N$ 次。

伪代码：

```
vocab = {每个字节 0..255}
corpus = 语料按行字节化为 id 序列

for step in range(N):
    pairs = count_adjacent_pairs(corpus)   # 统计所有相邻 (a, b) 的频次
    best = argmax(pairs)                   # 找出频次最高的 pair
    new_id = 256 + step                    # 分配新 token id
    corpus = replace_all(corpus, best, new_id)
    vocab[new_id] = vocab[best[0]] + vocab[best[1]]

# 最终词表大小 = 256 + N
```

### 3.2 手算示例：经典 5 词语料

用 Sennrich 2016 论文中的经典例子演示前 3 轮合并：

**初始语料**（词频标注）：

```
"low"  × 5,  "lower" × 2,  "newest" × 6,  "wider" × 3,  "new" × 2
```

字符级初始化后，将每个词视为字符序列（带词尾标记 `</w>`）：

```
初始 corpus token 序列（词频展开）：
l o w </w>          ×5
l o w e r </w>      ×2
n e w e s t </w>    ×6
w i d e r </w>      ×3
n e w </w>          ×2
```

**第 1 轮**：统计所有相邻 pair 频次

| pair | 频次 |
|------|------|
| `(e, s)` | 6（来自 newest ×6） |
| `(e, r)` | 5（lower ×2 + wider ×3） |
| `(n, e)` | 8（newest ×6 + new ×2） |
| `(e, w)` | 8（newest ×6 + new ×2） |
| ... | ... |

最高频：`(e, s)` = 6 或 `(e, w)` = 8（视统计顺序，这里取 `(e, w)` = 8）

合并 `e + w` → `ew`，语料变为：

```
l o w </w>           ×5
l o w e r </w>       ×2
n ew e s t </w>      ×6
w i d e r </w>       ×3
n ew </w>            ×2
```

**第 2 轮**：统计更新后的 pair 频次，最高频为 `(n, ew)` = 8

合并 `n + ew` → `new`：

```
l o w </w>           ×5
l o w e r </w>       ×2
new e s t </w>       ×6
w i d e r </w>       ×3
new </w>             ×2
```

**第 3 轮**：最高频变为 `(l, o)` = 7（low ×5 + lower ×2）

合并 `l + o` → `lo`：

```
lo w </w>            ×5
lo w e r </w>        ×2
new e s t </w>       ×6
w i d e r </w>       ×3
new </w>             ×2
```

经过 3 轮，高频的 `ew`、`new`、`lo` 已经成为独立 token。继续迭代下去，`low`、`newest` 等整词最终也会变成单个 token。

### 3.3 代码实现：`bpe_tokenizer.py` 的核心函数

`_get_stats`：统计相邻 pair 频次

```python
def _get_stats(self, ids_list):
    counts = Counter()
    for ids in ids_list:
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += 1
    return counts
```

`_merge`：把语料中所有 `pair` 替换为 `new_id`

```python
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
```

`train` 主循环：

```python
for step in range(num_merges):
    stats = self._get_stats(ids_list)
    pair = max(stats, key=stats.get)      # 最高频 pair
    new_id = 256 + step
    ids_list = self._merge(ids_list, pair, new_id)
    self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
    self.merges.append((pair, new_id))
```

结构极简：三个函数，核心逻辑不到 30 行。

---

## 四、用 demo 实操

### 4.1 运行结果

语料：Tiny Shakespeare 前 50KB，跑 200 轮合并。

```
初始词表：65 个字符（Tiny Shakespeare 的 unique chars）
         ↓ bpe_tokenizer.py 使用字节级初始化
实际初始：256 个字节 token
训练后：256 + 200 = 456 个 token
```

合并轨迹的前几步（实际输出片段）：

```
  step   0  merge (101, 32) → 256  'e '  (count=2394)
  step   1  merge (116, 104) → 257  'th'  (count=2178)
  step   2  merge (105, 110) → 258  'in'  (count=1765)
  step   3  merge (97, 110) → 259  'an'  (count=1712)
  step   4  merge (257, 101) → 260  'the'  (count=1598)
  step   5  merge (111, 117) → 261  'ou'  (count=1512)
  step   6  merge (104, 97) → 262  'ha'  (count=1412)
  step   7  merge (101, 114) → 263  'er'  (count=1350)
  step   8  merge (32, 119) → 264  ' w'  (count=1287)
  step   9  merge (114, 101) → 265  're'  (count=1244)
```

**规律**：最先被合并的是高频的字母 bigram（空格+字母、常见英文音节）。`th`、`in`、`an` 在英文中极其常见，前 10 步就被吸收。

### 4.2 编码示例

```
原文（55 字节）：ROMEO: But soft, what light through yonder window breaks?
BPE  （~35 tokens）：[82, 79, 77, 69, 79, 58, 32, 66, ...]
压缩比：1.58×
解码回原文：ROMEO: But soft, what light through yonder window breaks?
```

压缩比 1.58× 意味着：原文 55 个字节，经过 200 次合并学到的 BPE 词表只需约 35 个 token 就能表示。序列长度缩短，attention 计算量（$O(T^2)$）随之大幅降低。

::: tip LLM 视角
200 次合并只是演示，词表仅 456。GPT-4 的 cl100k_base 进行了约 10 万次合并，词表 ~100,256。同样的 ROMEO 那句话，GPT-4 大约只需 14–16 个 token，压缩比更高，因为更多完整英文单词已经有了自己的 token id。算法完全一样，规模不同而已。
:::

### 4.3 encode / decode 接口

```python
from bpe_tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train(text, num_merges=200)

ids = tok.encode("ROMEO: But soft")
# → [82, 79, 77, 69, 79, 58, 32, ...]

text_back = tok.decode(ids)
# → "ROMEO: But soft"
```

`encode` 的实现：先把输入字节化，然后按训练时记录的合并顺序依次应用所有 merge，贪心地把 pair 合并。

---

## 五、词表大小的权衡

### 5.1 两端的代价

$$
\text{embedding 参数} = V \times d_{\text{model}}
$$

词表越大，序列越短（压缩比越高），但 embedding 矩阵越大。

**词表小（如本 demo 的 456）**：
- embedding 矩阵很小（$456 \times d$，几乎忽略不计）
- 但序列长，attention 的 $O(T^2)$ 开销大
- 常见词没有专属 token，需要拼多个子词

**词表大（如 cl100k_base 的 100,256）**：
- embedding 矩阵大（$100{,}256 \times d_{\text{model}}$），在小模型里占比高
- 序列短，推理快
- 多语言、代码等长尾词有更好的覆盖

### 5.2 主流 LLM 词表对比

| 模型 | 词表大小 | 编码方案 | embedding 参数占比（参考） |
|------|---------|---------|------------------------|
| GPT-2 small | 50,257 | byte-level BPE | ~30%（768 维 × 50K）|
| LLaMA-1/2 | 32,000 | SentencePiece BPE | ~10%（4096 维 × 32K） |
| LLaMA-3 | 128,256 | tiktoken BPE | ~10%（4096 维 × 128K） |
| GPT-4（cl100k_base） | ~100,256 | byte-level BPE | — |
| Qwen 系列 | 152,064 | tiktoken BPE | — |
| 本 demo | 456 | byte-level BPE | < 1%（教学用） |

::: tip LLM 视角
为什么现代 LLM 词表越来越大？

1. **多语言**：中文、日文、阿拉伯文等非 ASCII 语言在小词表下一个字符要拆成 2–4 个 token，效率极低；大词表能为高频汉字分配专属 token。
2. **代码**：Python 关键字、缩进空格在大词表下可以整块处理。
3. **效率**：同样的语料，大词表序列更短，相同 context window 能"看"更多内容。

但词表增大有边际收益递减：100K 到 200K 的提升远不如 30K 到 100K 明显。
:::

### 5.3 GPT-2 small 的 embedding 参数占比计算

$$
\text{embedding 参数} = 50{,}257 \times 768 \approx 38.6M
$$

GPT-2 small 总参数 124M，embedding 占约 **31%**。加上 lm_head（GPT-2 权重共享，两者是同一个矩阵），实际参与计算的 attention + FFN 只有 ~85M。这也是为什么"参数量"在词表很大时不完全反映模型的表示能力。

---

## 六、生产实现：tiktoken / SentencePiece

### 6.1 tiktoken（OpenAI）

byte-level BPE + Rust 实现，速度极快（比纯 Python 快 10–100 倍）：

```python
import tiktoken

# GPT-4 使用的编码
enc = tiktoken.encoding_for_model("gpt-4")
ids = enc.encode("Hello, 你好")
# → [9906, 11, 220, 19526, 53901]

text = enc.decode(ids)
# → "Hello, 你好"

print(enc.n_vocab)  # 100256
```

cl100k_base 是 GPT-3.5 / GPT-4 / text-embedding-ada-002 共用的编码方案。

### 6.2 SentencePiece（Google）

LLaMA / Qwen / T5 等 Google 系模型的标准选择，支持两种模式：

| 模式 | 算法 | 特点 |
|------|------|------|
| **BPE** | 与本节相同的合并逻辑 | 贪心，确定性 |
| **Unigram LM** | 维护概率词表，训练时剪枝 | 可计算最优分词 |

LLaMA-1/2 使用 SentencePiece 的 BPE 模式，词表 32K；LLaMA-3 改用 tiktoken，词表扩到 128K。

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("llama2_tokenizer.model")
ids = sp.Encode("Hello, 你好")
sp.Decode(ids)
```

### 6.3 本 demo 与生产实现的对比

| 维度 | `bpe_tokenizer.py`（本 demo） | tiktoken / SentencePiece |
|------|------------------------------|--------------------------|
| 语言 | 纯 Python | Rust / C++ 核心 |
| 训练速度 | 慢（教学用，~几十秒） | 快（并行、SIMD） |
| 训练数据 | 50KB Tiny Shakespeare | 万亿 token 多语料 |
| 合并次数 | 200 | ~5 万–10 万 |
| 词表大小 | 456 | 32K–152K |
| 算法 | **完全一致** | **完全一致** |

::: warning 教学版 vs 生产版
本 demo 是教学工具，**不要**在生产中使用它。生产场景请直接用 `tiktoken`（pip install tiktoken）或 HuggingFace `tokenizers` 库。

但理解了本 demo，你就理解了所有主流 tokenizer 的核心算法。
:::

---

## 七、特殊 token

### 7.1 常见特殊 token

| Token | 含义 | 典型使用方 |
|-------|------|---------|
| `<\|endoftext\|>` | 文档边界 | GPT 系列 |
| `<bos>` / `<s>` | 序列开始（Begin of Sequence） | LLaMA |
| `<eos>` / `</s>` | 序列结束（End of Sequence） | LLaMA |
| `<pad>` | padding，填充到相同长度 | BERT / 批处理 |
| `<\|im_start\|>` / `<\|im_end\|>` | 消息开始/结束（ChatML 格式） | Qwen、GPT-4 |
| `[INST]` / `[/INST]` | instruction 标记 | LLaMA-2 chat |

### 7.2 对话格式是怎么用特殊 token 拼成的

ChatGPT / Claude 的多轮对话，在 tokenizer 层面其实是把各角色的消息拼接成一个长序列，用特殊 token 分隔：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi, how can I help?<|im_end|>
```

这整个字符串被 tokenize 成一个 id 序列，一次性喂给模型。模型从 `<|im_start|>assistant` 之后开始生成，直到产生 `<|im_end|>` 停止。

### 7.3 安全：prompt injection 的 tokenizer 层面

如果用户输入中包含 `<|im_start|>system` 这样的字符串，tokenizer 会把它编码成真正的特殊 token id，可能欺骗模型切换角色。OpenAI 在处理用户输入时会对特殊 token 进行**转义**（不允许用户输入被解析为特殊 token），这是 production tokenizer 需要额外处理的边界情况。

---

## 八、与 LLM 训练的衔接

### 8.1 训练数据的预处理流程

```
原始文本（TB 级）
      ↓
全量 tokenize（通常离线完成，保存为 .bin 文件）
      ↓
token id 序列，存为 uint16 / int32 二进制
      ↓
训练时 mmap 读取，切成 block_size 大小的片段
      ↓
喂给模型
```

**为什么要预先 tokenize 并存二进制**：

- tokenize 本身有 CPU 开销，训练时每个 epoch 重复 tokenize 浪费时间
- 二进制存储比文本紧凑：1 个 token id 只需 2 字节（uint16），而原文可能要 4–6 字节

### 8.2 token 数量 = 训练规模的关键指标

Chinchilla 定律（Hoffmann et al. 2022）给出了模型参数量与最优训练 token 数的关系：

$$
\text{最优训练 tokens} \approx 20 \times \text{参数量}
$$

| 模型 | 参数量 | 训练 tokens（Chinchilla 最优） | 实际训练 tokens |
|------|--------|------------------------------|----------------|
| GPT-3 | 175B | ~3.5T | 300B（训练不足） |
| LLaMA-1 7B | 7B | ~140B | 1T（过训练） |
| LLaMA-3 8B | 8B | ~160B | 15T（大幅过训练） |

::: tip LLM 视角
"训了多少 token"是衡量预训练规模最直接的数字，比"用了多少 GPU 小时"更通用。Chinchilla 揭示了 GPT-3 是"参数大、数据少"（训练不足），LLaMA 系列则刻意用更多数据训练相对小的模型——得到的模型在推理时更高效（参数少，速度快），同时效果更好。

token 数量也是 API 计费的基础单位：OpenAI / Anthropic 的 API 按 input tokens + output tokens 收费，这里的"token"就是 tokenizer 的输出单元。
:::

### 8.3 token 数量估算

粗估：**1M token ≈ 700KB 英文文本**（GPT-4 tokenizer）

Tiny Shakespeare 全文约 1MB，≈ 1.4M tokens。本 demo 取前 50KB，≈ 70K tokens，200 轮合并后进一步压缩到 ~44K tokens。

---

## 九、配套代码

| 文件 | 主题 | 运行命令 |
|------|------|---------|
| `ml_foundations/transformer_training/bpe_tokenizer.py` | BPE 训练 + encode / decode | `python bpe_tokenizer.py` |

运行前需要下载语料：

```bash
mkdir -p ml_foundations/transformer_training/data
curl -L -o ml_foundations/transformer_training/data/tiny_shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

::: tip 与生产 LLM 对比
| | 本 demo | GPT-4 cl100k_base |
|--|---------|-------------------|
| 训练数据 | 50KB Tiny Shakespeare | ~万亿 token 多语言语料 |
| 合并次数 | 200 | ~100,000 |
| 词表大小 | 456 | 100,256 |
| 压缩比（英文） | 1.58× | ~4–5× |
| 实现语言 | 纯 Python | Rust |

算法完全相同，规模和工程优化天差地别。理解了这 128 行代码，你就理解了 cl100k_base 的数学核心。
:::

---

## 十、延伸阅读

- **Sennrich et al. 2016**，"Neural Machine Translation of Rare Words with Subword Units"——BPE 引入 NLP 的原始论文，算法描述即本节所述
- **Radford et al. 2019**，"Language Models are Unsupervised Multitask Learners"——GPT-2 论文，byte-level BPE 的提出
- **Karpathy，"Let's build the GPT Tokenizer"（2024）**——minBPE YouTube 视频，约 2 小时，从零实现并与 tiktoken 对比，本节的最佳配套视频
- **tiktoken**（[github.com/openai/tiktoken](https://github.com/openai/tiktoken)）——OpenAI 开源的 Rust BPE 实现，5 行代码即可使用
- **HuggingFace tokenizers 库**——支持 BPE、WordPiece、Unigram，API 统一，生产首选

---

> **下一站**：[自注意力机制](./attention) —— token id 进入 embedding 层之后，Transformer 是怎么让它们互相"看到"彼此的？
