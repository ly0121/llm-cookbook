---
title: 文本生成策略
---

<script setup>
const code1 = `import numpy as np

np.random.seed(42)

# 模拟词表和 logits
vocab = ['好', '不错', '热', '冷', '晴', '糟糕', '一般', '可以']
logits = np.array([2.0, 1.5, 0.8, 0.3, 0.1, -0.5, -0.2, 0.6])

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# 原始概率分布
probs = softmax(logits)
print('=== 原始概率分布 ===')
for w, p in sorted(zip(vocab, probs), key=lambda x: -x[1]):
    bar = '█' * int(p * 40)
    print(f'  {w:4s} {p:.3f} {bar}')

# Top-K 采样 (K=3)
K = 3
top_k_indices = np.argsort(logits)[-K:][::-1]
top_k_logits = logits[top_k_indices]
top_k_probs = softmax(top_k_logits)

print(f'\\n=== Top-{K} 采样 ===')
print(f'候选词: {[vocab[i] for i in top_k_indices]}')
print(f'重新归一化概率: {np.round(top_k_probs, 3)}')

# 模拟采样 10 次
samples = np.random.choice(top_k_indices, size=10, p=top_k_probs)
print(f'\\n采样10次结果: {[vocab[s] for s in samples]}')
print(f'注意: 低概率词（冷、晴等）被完全排除!')
`

const code2 = `import numpy as np

np.random.seed(123)

vocab = ['好', '不错', '热', '冷', '晴', '糟糕', '一般', '可以']
logits = np.array([2.0, 1.5, 0.8, 0.3, 0.1, -0.5, -0.2, 0.6])

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

probs = softmax(logits)

# Top-P 采样 (P=0.8)
P = 0.8
sorted_indices = np.argsort(probs)[::-1]
sorted_probs = probs[sorted_indices]
cumulative_probs = np.cumsum(sorted_probs)

print('=== Top-P 采样过程 (P=0.8) ===')
print(f'{"词":>4s}  {"概率":>6s}  {"累积":>6s}  选入?')
print('-' * 36)

nucleus_indices = []
for i, idx in enumerate(sorted_indices):
    included = '✓' if cumulative_probs[i] <= P or i == 0 else ('✓' if cumulative_probs[i-1] < P else '✗')
    print(f'{vocab[idx]:>4s}  {sorted_probs[i]:.3f}   {cumulative_probs[i]:.3f}   {included}')
    if cumulative_probs[i] <= P or (i == 0):
        nucleus_indices.append(idx)
    elif cumulative_probs[i-1] < P:
        nucleus_indices.append(idx)

# 重新归一化并采样
nucleus_probs = probs[nucleus_indices]
nucleus_probs = nucleus_probs / nucleus_probs.sum()
print(f'\\n核心词集: {[vocab[i] for i in nucleus_indices]}')
print(f'归一化概率: {np.round(nucleus_probs, 3)}')

# 对比 Top-K vs Top-P
print(f'\\n=== Top-K vs Top-P 对比 ===')
print(f'Top-3 固定选 3 个词，不管概率分布形状')
print(f'Top-P=0.8 动态选词，尖锐分布选少、平坦分布选多')
`

const code3 = `import numpy as np

vocab = ['好', '不错', '热', '冷', '晴']
logits = np.array([2.0, 1.5, 0.8, 0.3, 0.1])

def softmax_with_temperature(logits, T):
    scaled = logits / T
    exp_x = np.exp(scaled - np.max(scaled))
    return exp_x / exp_x.sum()

print('不同 Temperature 下的概率分布：')
print(f'{"词":>4s}  T=0.3(保守)  T=1.0(正常)  T=2.0(创意)')
print('-' * 50)

temps = [0.3, 1.0, 2.0]
all_probs = {t: softmax_with_temperature(logits, t) for t in temps}

for i, w in enumerate(vocab):
    print(f'{w:>4s}  {all_probs[0.3][i]:.3f}        {all_probs[1.0][i]:.3f}        {all_probs[2.0][i]:.3f}')

print()
print('规律总结：')
print('  T→0: 趋向 Greedy（只选最高概率词）')
print('  T=1: 原始分布不变')
print('  T→∞: 趋向均匀分布（完全随机）')
print()
print('实践建议：')
print('  代码生成/事实问答: T=0.0~0.3')
print('  通用对话:          T=0.7~1.0')
print('  创意写作/头脑风暴: T=1.0~1.5')
`
</script>

# 文本生成策略（Decoding Strategies）

LLM 通过自回归方式逐词生成文本。每一步模型输出词表上的概率分布，如何从中"选词"就是解码策略的核心问题。

## 1. Greedy Decoding（贪心解码）

每一步选择概率最高的 token，简单但可能陷入重复。

```
概率分布: 好(0.35) 不错(0.25) 热(0.15) 冷(0.10) ...
Greedy 选择: "好" (概率最高)

优点: 确定性输出、适合代码生成
缺点: 容易重复、缺乏多样性
```

## 2. Top-K Sampling（Top-K 采样）

只从概率最高的 K 个 token 中采样，过滤掉低概率噪声：

<PythonRunner :browser-runnable="true" :code="code1" />

## 3. Top-P Sampling（Nucleus Sampling）

动态选取累积概率达到 P 的最小词集，比 Top-K 更灵活：

<PythonRunner :browser-runnable="true" :code="code2" />

## 4. Temperature 效果

Temperature 控制概率分布的"尖锐程度"：

<PythonRunner :browser-runnable="true" :code="code3" />

## 5. 策略对比总览

| 策略 | 确定性 | 多样性 | 适用场景 |
|------|--------|--------|---------|
| Greedy | 最高 | 最低 | 代码生成、数学计算 |
| Top-K | 中等 | 中等 | 通用对话（K=40~100） |
| Top-P | 中等 | 自适应 | 创意写作（P=0.9~0.95） |
| Temperature | 可调 | 可调 | 配合其他策略使用 |

::: info 实际应用
生产中通常组合使用：例如 `temperature=0.7 + top_p=0.9` 是常见的通用配置。
:::

---

::: tip 下一步
- [提示工程](/llm/prompt-engineering) — 掌握与 LLM 高效对话的技巧
- [LangChain 框架](/langchain/) — 将生成策略与应用框架结合
:::
