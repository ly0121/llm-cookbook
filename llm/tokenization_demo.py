"""
╔══════════════════════════════════════════════════════════════════╗
║         Tokenization 深度剖析：从文本到 Token 的完整旅程          ║
║         纯本地运行，无需任何 API 调用                              ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题：计算机为什么不能直接"读懂"文字？】
═══════════════════════════════════════════════════════════════════

计算机只认识数字（0 和 1），不认识"你好"或"Hello"。
所以我们需要一个"翻译官"——Tokenizer，把人类文字翻译成数字序列。

但为什么不直接用 ASCII/Unicode 编码呢？
因为那样效率太低了！"机器学习" 4个字 = 12 字节 = 12 个数字，
但它表达的是一个概念，应该被当作 1-2 个 token 处理。

Tokenization 就是找到"恰到好处的粒度"——
不太细（不是每个字节），也不太粗（不是每个句子），
而是像搭乐高一样，用合理大小的"积木块"拼出任意文本。

═══════════════════════════════════════════════════════════════════
【全景流水线：一段文字是怎么变成 AI 能理解的东西的？】
═══════════════════════════════════════════════════════════════════

  ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
  │  原始文本  │ ──→ │  Tokens   │ ──→ │ Token IDs │ ──→ │Embeddings│
  │ "我爱AI"  │      │["我","爱" │      │[12345,    │      │[[0.1,0.3 │
  │           │      │ ,"AI"]   │      │ 678, 15]  │      │  ...]]   │
  └──────────┘      └──────────┘      └──────────┘      └──────────┘
    人类可读          分词结果           数字索引            向量表示
                   (Tokenizer)       (词汇表查找)        (Embedding层)

  本文件专注前两步：文本 → Tokens → Token IDs
  后面的 Embedding 是模型内部的事，我们管不着。
"""

import tiktoken

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：什么是 Tokenization
# 目标：用直觉理解"分词"这件事
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 0 章：什么是 Tokenization")
print("=" * 60)
print()

# ── 类比：Tokenization 就像"切寿司" ────────────────────────
#
# 想象一条长长的寿司卷（= 你的文本），
# 厨师需要把它切成一块块方便入口的小段（= tokens）。
#
# 切太细（每个字符一块）→ 筷子夹不住，效率低
# 切太粗（整句话一块）→ 嘴塞不下，灵活性差
# 切得刚好（常见词组一块）→ 吃起来优雅又高效
#
# BPE（Byte Pair Encoding）就是那个聪明的厨师，
# 它通过统计学习，找到"最佳切法"。

print("【直觉演示：同一句话的不同切法】")
print()
text_demo = "机器学习很有趣"
print(f"  原始文本：{text_demo}")
print(f"  按字切分：['机', '器', '学', '习', '很', '有', '趣']  → 7 个 token")
print(f"  按词切分：['机器学习', '很', '有趣']                   → 3 个 token")
print(f"  BPE 切分：['机器', '学习', '很有', '趣']               → 4 个 token（实际结果）")
print()
print("  BPE 的切法是通过大量文本训练出来的，不需要词典，")
print("  它根据字符对出现的频率自动学习合并规则。")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：BPE 算法演示
# 目标：从零实现一个迷你 BPE，看懂合并过程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：BPE 算法演示（从零实现）")
print("=" * 60)
print()

# ── BPE 核心思想 ──────────────────────────────────────────
#
# 第1步：把文本拆成最小单元（字符级别）
# 第2步：统计所有"相邻对"的出现次数
# 第3步：把出现次数最多的一对合并成新 token
# 第4步：重复第2-3步，直到达到预设的词汇表大小
#
# 就像拼图：先把所有碎片摊开，然后不断把最常配对的碎片粘在一起

print("【迷你 BPE 训练过程】")
print()

def train_bpe(text, num_merges=5):
    """
    一个简化的 BPE 训练算法。
    text: 训练文本
    num_merges: 要执行多少次合并操作
    """
    # 第1步：初始化——把文本拆成字符列表
    tokens = list(text)
    print(f"  初始 tokens: {tokens}")
    print()

    for step in range(num_merges):
        # 第2步：统计所有相邻 token 对的频率
        pair_counts = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            break

        # 第3步：找到频率最高的 pair
        best_pair = max(pair_counts, key=pair_counts.get)
        best_count = pair_counts[best_pair]

        # 第4步：把这个 pair 合并成一个新 token
        new_token = best_pair[0] + best_pair[1]
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        tokens = merged

        print(f"  第{step + 1}轮合并: {best_pair[0]!r} + {best_pair[1]!r} → {new_token!r}  (出现{best_count}次)")
        print(f"  合并后: {tokens}")
        print()

    return tokens

# 用一个简单的英文示例演示 BPE
sample_text = "low lower lowest low lower"
print(f"  训练文本: {sample_text!r}")
print()
final_tokens = train_bpe(sample_text, num_merges=5)
print(f"  最终 tokens: {final_tokens}")
print(f"  从 {len(sample_text)} 个字符压缩到 {len(final_tokens)} 个 token")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：tiktoken 实战
# 目标：用 OpenAI 官方 tokenizer 进行真实分词
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：tiktoken 实战")
print("=" * 60)
print()

# ── 获取编码器 ────────────────────────────────────────────
#
# tiktoken 提供多种编码方案：
#   cl100k_base → GPT-4、GPT-3.5-turbo 使用
#   o200k_base  → GPT-4o 使用（更新、更高效）
#
# "cl100k" 意思是词汇表大小约 10万 个 token
# "o200k" 意思是词汇表大小约 20万 个 token（对中文更友好）

enc = tiktoken.get_encoding("cl100k_base")
print(f"  编码器: cl100k_base (GPT-4 使用)")
print(f"  词汇表大小: {enc.n_vocab} 个 token")
print()

# ── 编码与解码 ────────────────────────────────────────────
#
# encode: 文本 → token ID 列表
# decode: token ID 列表 → 文本

print("【编码/解码演示】")
print()

examples = [
    "Hello, world!",
    "你好，世界！",
    "Machine Learning is amazing",
    "机器学习非常神奇",
]

for text in examples:
    token_ids = enc.encode(text)
    # 把每个 token ID 解码回文字碎片，看看实际切分
    token_strs = [enc.decode([tid]) for tid in token_ids]
    print(f"  文本: {text!r}")
    print(f"  Token IDs: {token_ids}")
    print(f"  Token 切分: {token_strs}")
    print(f"  Token 数量: {len(token_ids)}")
    print()

# ── 中英文效率对比 ────────────────────────────────────────
#
# 同样的意思，中文和英文所需的 token 数量不同！
# 这直接影响 API 调用成本和上下文窗口利用率。

print("【中英文 Token 效率对比】")
print()
comparisons = [
    ("Hello, how are you?", "你好，你好吗？"),
    ("Artificial intelligence will change the world", "人工智能将改变世界"),
    ("The quick brown fox jumps over the lazy dog", "那只敏捷的棕色狐狸跳过了懒狗"),
]

print(f"  {'英文':<45s} | tokens | {'中文':<20s} | tokens")
print(f"  {'-' * 45}-+--------+{'-' * 22}+-------")
for en, zh in comparisons:
    en_tokens = len(enc.encode(en))
    zh_tokens = len(enc.encode(zh))
    print(f"  {en:<45s} | {en_tokens:>4d}   | {zh:<20s} | {zh_tokens:>4d}")
print()
print("  结论：中文在 cl100k_base 下通常比英文需要更多 token，")
print("        因为该 tokenizer 主要基于英文语料训练。")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：Token 计数与成本计算
# 目标：学会预估 API 调用成本
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：Token 计数与成本计算")
print("=" * 60)
print()

# ── 消息数组的 Token 计数 ─────────────────────────────────
#
# OpenAI API 的计费不只是纯文本 token！
# 每条消息有额外的"结构开销"：
#   - 每条消息固定 +4 tokens（<|im_start|>role\ncontent<|im_end|>\n）
#   - 整个请求末尾 +3 tokens（assistant回复的起始标记）
#
# 这个函数模拟 OpenAI 官方的 token 计数逻辑

def count_message_tokens(messages, model="gpt-4"):
    """
    计算一个 messages 数组的总 token 数（包含结构开销）。
    这是 OpenAI 官方推荐的计算方式。
    """
    encoding = tiktoken.encoding_for_model(model)

    # 每条消息的固定开销
    tokens_per_message = 4  # <|im_start|>{role}\n{content}<|im_end|>\n
    tokens_per_name = -1    # 如果有 name 字段，role 会被省略

    total = 0
    for message in messages:
        total += tokens_per_message
        for key, value in message.items():
            total += len(encoding.encode(value))
    total += 3  # 每个请求末尾的 assistant 回复前缀
    return total


# ── 演示：不同 system prompt 长度的影响 ────────────────────

print("【消息 Token 计数演示】")
print()

messages_short = [
    {"role": "system", "content": "你是助手。"},
    {"role": "user", "content": "什么是Python？"},
]

messages_long = [
    {"role": "system", "content": "你是一个资深的 Python 编程专家，拥有20年开发经验。"
     "你的回答要包含代码示例、最佳实践、常见陷阱和性能优化建议。"
     "回答使用中文，代码注释也用中文。格式要清晰，使用 markdown。"},
    {"role": "user", "content": "什么是Python？"},
]

tokens_short = count_message_tokens(messages_short)
tokens_long = count_message_tokens(messages_long)

print(f"  短 system prompt: {tokens_short} tokens")
print(f"  长 system prompt: {tokens_long} tokens")
print(f"  差异: {tokens_long - tokens_short} tokens（每次调用都要多付这么多）")
print()

# ── 成本计算函数 ──────────────────────────────────────────
#
# GPT-4 定价（举例，实际价格请查 OpenAI 官网）：
#   输入: $0.03 / 1K tokens
#   输出: $0.06 / 1K tokens

def estimate_cost(input_tokens, output_tokens, model="gpt-4"):
    """
    估算单次 API 调用成本（美元）。
    价格基于 2024 年 GPT-4 标准定价。
    """
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},        # $ per 1K tokens
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }
    price = pricing.get(model, pricing["gpt-4"])
    input_cost = (input_tokens / 1000) * price["input"]
    output_cost = (output_tokens / 1000) * price["output"]
    return input_cost + output_cost


print("【成本估算演示】")
print()
print(f"  假设：输入 {tokens_long} tokens，输出 500 tokens")
print()
for model_name in ["gpt-4", "gpt-4o", "gpt-3.5-turbo"]:
    cost = estimate_cost(tokens_long, 500, model_name)
    print(f"  {model_name:<15s}: ${cost:.6f} / 次")
print()
print("  一天调用 1000 次的月成本：")
for model_name in ["gpt-4", "gpt-4o", "gpt-3.5-turbo"]:
    cost = estimate_cost(tokens_long, 500, model_name) * 1000 * 30
    print(f"  {model_name:<15s}: ${cost:.2f} / 月")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：Token 限制与上下文管理
# 目标：学会在有限的上下文窗口内管理对话历史
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 4 章：Token 限制与上下文管理")
print("=" * 60)
print()

# ── 上下文窗口：AI 的"工作记忆" ────────────────────────────
#
# 每个模型有一个固定的上下文窗口（context window）：
#   GPT-3.5-turbo: 16K tokens
#   GPT-4:         8K / 32K / 128K tokens
#   GPT-4o:        128K tokens
#
# 这个窗口必须容纳：输入（messages）+ 输出（AI 回复）
# 超出就会报错！所以我们需要"上下文管理策略"。
#
#   ┌─────────────────────────────────────────┐
#   │           上下文窗口 (如 8K tokens)        │
#   │  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
#   │  │ System   │ │ 对话历史  │ │ 输出预留 │ │
#   │  │ (固定)   │ │ (可压缩)  │ │ (固定)  │ │
#   │  └──────────┘ └──────────┘ └─────────┘ │
#   └─────────────────────────────────────────┘

print("【上下文窗口管理策略】")
print()

def manage_conversation(messages, max_tokens=4096, reserved_for_output=500):
    """
    管理对话历史，确保不超出 token 限制。

    策略：保留 system 消息 + 最新的对话轮次，裁剪较早的历史。
    这就像一个滑动窗口，永远只保留最近的对话。

    参数：
        messages: 完整对话历史
        max_tokens: 模型的上下文窗口大小
        reserved_for_output: 预留给 AI 回复的 token 数
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    budget = max_tokens - reserved_for_output  # 输入可用的 token 预算

    # 分离 system 消息和对话消息
    system_msgs = [m for m in messages if m["role"] == "system"]
    conversation = [m for m in messages if m["role"] != "system"]

    # 计算 system 消息占用的 token
    system_tokens = sum(len(encoding.encode(m["content"])) + 4 for m in system_msgs)
    remaining_budget = budget - system_tokens - 3  # 减去请求末尾开销

    # 从最新的消息开始，往回添加，直到预算用完
    kept_messages = []
    current_tokens = 0

    for msg in reversed(conversation):
        msg_tokens = len(encoding.encode(msg["content"])) + 4
        if current_tokens + msg_tokens <= remaining_budget:
            kept_messages.insert(0, msg)
            current_tokens += msg_tokens
        else:
            break  # 预算不够了，更早的消息被裁剪

    trimmed_count = len(conversation) - len(kept_messages)
    final_messages = system_msgs + kept_messages

    return final_messages, trimmed_count


# ── 演示：模拟一个很长的对话 ──────────────────────────────

long_conversation = [
    {"role": "system", "content": "你是一个友好的AI助手。"},
]
# 模拟 20 轮对话
for i in range(1, 21):
    long_conversation.append({"role": "user", "content": f"这是第{i}个问题，请详细解释一下相关概念和背景知识。"})
    long_conversation.append({"role": "assistant", "content": f"好的，关于第{i}个问题，让我详细解释..." * 5})

total_tokens = count_message_tokens(long_conversation)
print(f"  原始对话: {len(long_conversation)} 条消息, 约 {total_tokens} tokens")

# 使用一个很小的窗口来演示裁剪效果
managed, trimmed = manage_conversation(long_conversation, max_tokens=2000, reserved_for_output=500)
managed_tokens = count_message_tokens(managed)
print(f"  管理后:   {len(managed)} 条消息, 约 {managed_tokens} tokens")
print(f"  裁剪了:   {trimmed} 条消息")
print()
print("  策略说明：保留 system prompt + 最新的对话轮次，")
print("  丢弃较早的历史。这是最简单有效的上下文管理方式。")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 章：不同 Tokenizer 对比
# 目标：理解不同编码方案对效率的影响
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 5 章：不同 Tokenizer 对比")
print("=" * 60)
print()

# ── 加载不同的编码器 ──────────────────────────────────────
#
# cl100k_base: GPT-4 / GPT-3.5-turbo 使用，词汇表约 100K
# o200k_base:  GPT-4o 使用，词汇表约 200K，对中文更优化
#
# 词汇表越大 → 常见词组更容易被"一口吞下" → token 数更少 → 更省钱

enc_cl100k = tiktoken.get_encoding("cl100k_base")
enc_o200k = tiktoken.get_encoding("o200k_base")

print(f"  cl100k_base 词汇表大小: {enc_cl100k.n_vocab}")
print(f"  o200k_base  词汇表大小: {enc_o200k.n_vocab}")
print()

# ── 对比测试 ──────────────────────────────────────────────

test_texts = [
    ("英文短句", "The quick brown fox jumps over the lazy dog."),
    ("中文短句", "敏捷的棕色狐狸跳过了那只懒狗。"),
    ("代码片段", "def hello_world():\n    print('Hello, World!')"),
    ("中文长句", "大规模语言模型通过自注意力机制处理输入序列中的每个token。"),
    ("混合文本", "GPT-4的context window支持128K tokens，比GPT-3.5大8倍。"),
    ("数学公式", "E = mc^2, where m is mass and c is speed of light"),
    ("日常对话", "今天天气真好，我们一起去公园散步吧！"),
]

print("【不同编码器的 Token 数量对比】")
print()
print(f"  {'类型':<8s} | {'cl100k':>6s} | {'o200k':>6s} | {'节省':>5s} | 文本")
print(f"  {'-' * 8}-+{'-' * 8}+{'-' * 8}+{'-' * 7}+{'-' * 30}")

for label, text in test_texts:
    n_cl100k = len(enc_cl100k.encode(text))
    n_o200k = len(enc_o200k.encode(text))
    saving = f"{((n_cl100k - n_o200k) / n_cl100k * 100):.0f}%" if n_cl100k > n_o200k else "-"
    print(f"  {label:<8s} | {n_cl100k:>6d} | {n_o200k:>6d} | {saving:>5s} | {text[:28]}")

print()
print("  结论：o200k_base（GPT-4o）对中文的 tokenization 效率明显更高，")
print("        同样的中文内容通常可以节省 20-40% 的 token。")
print("        这意味着用 GPT-4o 处理中文时，成本更低，上下文窗口利用率更高。")
print()

# ── 深入对比：同一个中文句子的切分细节 ─────────────────────

print("【切分细节对比】")
print()
detail_text = "人工智能正在改变世界"
print(f"  文本: {detail_text!r}")
print()

ids_cl100k = enc_cl100k.encode(detail_text)
ids_o200k = enc_o200k.encode(detail_text)

tokens_cl100k = [enc_cl100k.decode([tid]) for tid in ids_cl100k]
tokens_o200k = [enc_o200k.decode([tid]) for tid in ids_o200k]

print(f"  cl100k_base ({len(ids_cl100k)} tokens): {tokens_cl100k}")
print(f"  o200k_base  ({len(ids_o200k)} tokens): {tokens_o200k}")
print()
print("  可以看到 o200k 能把更多中文字符合并为一个 token，")
print("  这是因为它的词汇表更大，包含了更多中文词组。")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 总结
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("学习完毕！Tokenization 核心要点回顾：")
print("=" * 60)
print("""
  1. Tokenization 是 LLM 的"入口翻译官"，文本必须先变成 token 才能被处理
  2. BPE 算法通过统计字符对频率，自动学习最优的分词规则
  3. 中文在 cl100k_base 下比英文消耗更多 token（成本更高）
  4. o200k_base (GPT-4o) 对中文效率更高，能节省 20-40% token
  5. 上下文管理是实际应用中的核心问题——对话太长必须裁剪
  6. Token 数量直接决定 API 成本，system prompt 越长每次调用越贵

  下一步建议：
    - 运行本文件观察实际输出
    - 尝试修改测试文本，观察不同内容的 token 效率
    - 在实际项目中使用 count_message_tokens() 监控成本
""")
