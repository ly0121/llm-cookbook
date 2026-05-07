---
title: Tokenization
---

<script setup>
const code1 = `# 简化版 BPE 训练算法
def train_bpe(corpus, num_merges):
    '''从字符级别逐步合并高频字符对'''
    # 初始化：每个字符是一个 token
    tokens = list(corpus)
    merges = []

    print(f'初始 tokens ({len(set(tokens))} 种): {tokens[:20]}...')
    print()

    for step in range(num_merges):
        # 统计相邻 token 对的频率
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pairs[pair] = pairs.get(pair, 0) + 1

        if not pairs:
            break

        # 找到最高频的对
        best_pair = max(pairs, key=pairs.get)
        freq = pairs[best_pair]
        merged = best_pair[0] + best_pair[1]
        merges.append((best_pair, merged))

        # 执行合并
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

        print(f'第{step+1}轮: 合并 {best_pair} → "{merged}" (频率:{freq})')
        print(f'  tokens: {tokens[:15]}...')
        print()

    return tokens, merges

# 示例语料
corpus = 'abababcdababcdabcd'
print(f'语料: "{corpus}"')
print('=' * 50)
final_tokens, merge_rules = train_bpe(corpus, 4)
print(f'最终 tokens: {final_tokens}')
print(f'词表: {sorted(set(final_tokens))}')
`

const code2 = `# 模拟 Token 计数和成本计算（简化版）

# 简化的分词规则（模拟 GPT-4 的 cl100k）
def estimate_tokens(text, lang='zh'):
    '''粗略估算 token 数量'''
    if lang == 'zh':
        # 中文：平均 1.5 个字符 = 1 token
        return int(len(text) / 1.5)
    else:
        # 英文：平均 4 个字符 = 1 token
        return int(len(text) / 4)

# 对比中英文的 Token 效率
texts = {
    '中文': '大语言模型是人工智能领域的重要突破，它通过海量数据训练获得了强大的文本生成能力。',
    '英文': 'Large language models are a major breakthrough in AI, trained on massive data to gain powerful text generation capabilities.',
}

print('Token 计数估算：')
print('-' * 60)

for lang, text in texts.items():
    tokens = estimate_tokens(text, 'zh' if lang == '中文' else 'en')
    print(f'{lang}: "{text[:30]}..."')
    print(f'  字符数: {len(text)}, 估算 tokens: {tokens}')
    print()

# 成本计算
print('\\n成本估算 (以 GPT-4o 为例):')
print('=' * 60)
input_price = 2.50   # $/1M tokens
output_price = 10.00  # $/1M tokens

# 一次典型对话
input_tokens = 500
output_tokens = 200

input_cost = input_tokens * input_price / 1_000_000
output_cost = output_tokens * output_price / 1_000_000
total = input_cost + output_cost

print(f'输入 {input_tokens} tokens × \${input_price}/M = \${input_cost:.6f}')
print(f'输出 {output_tokens} tokens × \${output_price}/M = \${output_cost:.6f}')
print(f'单次对话总成本: \${total:.4f} ≈ ¥{total*7.2:.3f}')
print(f'\\n每天 100 次对话: \${total*100:.2f} ≈ ¥{total*100*7.2:.1f}')
print(f'每月 3000 次对话: \${total*3000:.2f} ≈ ¥{total*3000*7.2:.0f}')
`
</script>

# Tokenization：文本如何变成数字

## 1. 为什么需要 Tokenization

计算机只能处理数字，不能直接处理文字。Tokenization 是把文字转换成数字序列的过程：

```
"我爱编程" → Tokenizer → [6311, 101, 32586, 8949]
                               ↓
                       Embedding → 向量序列
                               ↓
                       送入 Transformer 计算
```

## 2. BPE 算法（Byte Pair Encoding）

GPT 系列使用的分词算法。核心思想：从字符级别逐步合并高频对。

<PythonRunner :browser-runnable="true" :code="code1" />

## 3. Token 计数与成本

不同语言的 Token 效率差异很大：

<PythonRunner :browser-runnable="true" :code="code2" />

## 4. 上下文窗口

上下文窗口 = 模型一次能"看到"的最大 Token 数：

```
上下文窗口 = 输入 tokens + 输出 tokens

各模型对比：
  GPT-4o:     128K tokens
  Claude 3.5: 200K tokens
  Gemini 2.0: 2M tokens  (最长！)
  Llama 3:    128K tokens
```

::: warning 超过上下文窗口
如果输入超过模型的上下文窗口，API 会报错。解决方案：
1. 截断（丢弃最早的对话历史）
2. 摘要压缩（用模型总结旧对话）
3. RAG（不把所有内容放 prompt，需要时再检索）
:::

---

::: tip 下一步
- [文本生成机制](/llm/generation) — Temperature、Top-P 的数学原理
- [提示工程](/llm/prompt-engineering) — 掌握与 LLM 高效对话的技巧
:::
