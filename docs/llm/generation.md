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

## 5. Temperature 与 Top-P 深度对比

### Temperature 的本质：调整"贫富差距"

Temperature 改变的是概率分布的**形状**——所有词都还在候选名单上，只是概率比例变了。

```
假设原始分布：
  "月亮" 40%   ████████
  "明月" 25%   █████
  "圆月" 15%   ███
  "银盘" 10%   ██
  "玉兔"  5%   █
  "冰轮"  3%   ░
  "蟾宫"  2%   ░

Temperature=0.3（低温，分布变尖锐，强者更强）：
  "月亮" 65%   █████████████
  "明月" 20%   ████
  "圆月"  8%   ██
  "银盘"  4%   █
  "玉兔"  2%   ░
  "冰轮"  1%   ░
  → 高概率词被"放大"，低概率词被"压缩"，几乎总是选"月亮"

Temperature=1.0（高温，分布变平坦，雨露均沾）：
  "月亮" 22%   ████
  "明月" 18%   ████
  "圆月" 16%   ███
  "银盘" 15%   ███
  "玉兔" 12%   ██
  "冰轮" 10%   ██
  → 所有词的概率被"拉平"，谁都有机会被选中
```

Temperature 就像调音量旋钮——把差距放大或缩小，但**所有词都还有被选中的可能**。

### Top-P 的本质：设置"入围门槛"

Top-P 不改变概率分布的形状，而是**直接踢掉不够格的词**。

```
top_p=0.65（概率不变，只是砍掉尾巴）：
  "月亮" 40%   ████████  ✓ 累计 40%
  "明月" 25%   █████     ✓ 累计 65% ≥ 0.65，到此为止
  ─────────────────────────────────── 入围线
  "圆月" 15%   ███       ✗ 淘汰
  "银盘" 10%   ██        ✗ 淘汰
  "玉兔"  5%   █         ✗ 淘汰
  → 只从"月亮"和"明月"中选，其他词直接没资格
  → 留下来的词之间概率比例不变（40:25 → 62%:38%）
```

Top-P 就像画一条线——线以上的留下，线以下的直接淘汰。

### 核心区别

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Temperature：调整"贫富差距"                                 │
│    所有人都在，但分配变了                                     │
│    小 → 差距大（确定性高）                                   │
│    大 → 差距小（随机性高）                                   │
│                                                              │
│  Top-P：设置"入围门槛"                                      │
│    不够格的直接淘汰，留下的比例不变                           │
│    小 → 门槛高，只留头部几个词（确定性高）                    │
│    大 → 门槛低，几乎都能入围（随机性高）                     │
│                                                              │
│  共同点：越小越保守，越大越奔放                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 参数大小速记

| 参数 | 越小(→0) | 越大(→1) |
|------|----------|----------|
| **Temperature** | 确定、保守，几乎总选概率最高的词 | 随机、多样，低概率词也有机会 |
| **Top-P** | 候选池极小，只从头部几个词选 | 候选池大，所有词都有资格 |

### 实际使用建议

| 场景 | 推荐设置 | 原因 |
|------|----------|------|
| 代码生成、数据提取 | temperature=0 | 要确定性，不能乱来 |
| 日常对话 | temperature=0.7 | 自然但不跑偏 |
| 写诗、头脑风暴 | temperature=1.0 | 要创意和惊喜 |

::: warning Claude 使用注意
Claude 模型的 temperature 范围是 `0~1`（不支持 >1），且 **temperature 和 top_p 不能同时设置**，只能选其一。一般建议直接用 temperature 控制即可。
:::

## 6. 策略对比总览

| 策略 | 确定性 | 多样性 | 适用场景 |
|------|--------|--------|---------|
| Greedy | 最高 | 最低 | 代码生成、数学计算 |
| Top-K | 中等 | 中等 | 通用对话（K=40~100） |
| Top-P | 中等 | 自适应 | 创意写作（P=0.9~0.95） |
| Temperature | 可调 | 可调 | 配合其他策略使用 |

::: info 实际应用
生产中通常组合使用：例如 `temperature=0.7 + top_p=0.9` 是常见的通用配置。但注意 Claude 不支持同时设置两者，只能选其一。
:::

---

::: tip 下一步
- [提示工程](/llm/prompt-engineering) — 掌握与 LLM 高效对话的技巧
- [LangChain 框架](/langchain/) — 将生成策略与应用框架结合
:::
