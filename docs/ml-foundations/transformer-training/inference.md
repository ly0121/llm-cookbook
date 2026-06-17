# 推理优化与 KV Cache

> 从 O(N³) 到 O(N²)——为什么 ChatGPT 的第二个 token 比第一个快很多

---

## 一、为什么推理优化重要？

训练和推理是两种根本不同的工作负载：

```
训练                              推理
─────────────────────────────    ──────────────────────────────
跑一次（或少数几次）               无限次，每个用户每个 query 都跑
forward + backward                只有 forward
可以离线批量                       低延迟要求（< 1 秒）
成本一次性摊销                     成本按请求计费
```

**一个实际的成本结构**：训练 GPT-4 的花费估计在 $5,000 万美元量级，但这是一次性的。真正烧钱的是推理——每天服务数亿次请求，每次都要跑一遍 forward。Anthropic/OpenAI 的推理服务器成本远高于训练集群的折旧摊销。

**ChatGPT 的"第二个 token 快"现象**：

```
用户看到的感知：
  第 1 个 token：等了约 300ms（prefill 整个 prompt）
  第 2 个 token：50ms
  第 3 个 token：50ms
  ...后续都很快

原因：KV Cache 在工作
```

**长 context 带来的内存危机**：GPT-3.5 的 4K 上下文时代，KV cache 还可以接受。到 GPT-4 Turbo 的 128K 上下文，单请求的 cache 可达数 GB——如何管理这些显存变成一个工程难题，催生了 vLLM/PagedAttention 等专门的推理引擎。

::: tip LLM 视角
推理优化是 LLM 产品化的核心工程挑战。一个"在研究环境下运行正确"的模型，和一个"能每秒服务 10 万请求、P99 延迟 < 2 秒"的生产系统，中间有巨大的工程鸿沟。KV cache 是其中最基础的一块砖。
:::

---

## 二、生成的两个阶段：Prefill vs Decode

自回归生成不是一个均匀的过程，它分为两个截然不同的阶段：

### 2.1 Prefill（预填充）

将整段 prompt 一次性 forward，建立初始 KV cache：

```
prompt: "ROMEO: But soft, what light"
           ↓
  [一次完整的 Transformer forward]
           ↓
  每层: 计算全部 T 个 token 的 K, V 并缓存
  最后一个位置的 logit → 生成第 1 个新 token
```

**计算特征**：compute-bound（算力受限）。可以把整个 prompt 做成一个大矩阵乘法，GPU 利用率高。

### 2.2 Decode（解码）

逐 token 生成，每次只输入上一步生成的 1 个 token：

```
step 1: 输入 token[t]   → 查询历史 KV cache → 生成 token[t+1]
step 2: 输入 token[t+1] → 查询历史 KV cache → 生成 token[t+2]
step 3: 输入 token[t+2] → 查询历史 KV cache → 生成 token[t+3]
...
```

**计算特征**：memory-bound（内存带宽受限）。每步只有 1 个 token 的矩阵乘法，GPU 矩阵乘法单元大量空转，真正的瓶颈是从显存搬数据（KV cache + 权重）的速度。

### 2.3 两阶段的性能含义

| | Prefill | Decode |
|---|---|---|
| 输入规模 | T 个 token（一次） | 1 个 token（每步） |
| 瓶颈 | 计算（FLOPs） | 内存带宽（GB/s） |
| 延迟体现 | TTFT（首 token 延迟） | TPOT（每 token 延迟） |
| GPU 利用率 | 高 | 低 |

::: tip LLM 视角
这就是为什么大家说"第一个 token 慢，后续 token 快"——prefill 要处理整段 prompt（几十到几千 token），decode 每步只处理 1 个 token。生产系统会分别优化这两个阶段，有时甚至用不同的 GPU 实例来跑 prefill 和 decode（prefill/decode disaggregation）。
:::

---

## 三、KV Cache 原理

### 3.1 关键观察：历史 K/V 不变

生成第 $t$ 个新 token 时，我们需要对所有历史 token $0 \ldots t-1$ 计算 attention。但注意：

> **前 $t-1$ 个 token 的 K、V 向量，在生成每一步时完全相同。**

原因：K、V 由 `W_K · x` 和 `W_V · x` 算出，而权重 `W_K, W_V` 固定，历史 token 的表示 `x` 也不变（自回归生成只在末尾追加，不修改历史）。

### 3.2 朴素生成 vs KV Cache

**朴素生成**：每步都对整个序列跑完整 forward：

```
生成 token[1]: forward([p0, p1, ..., p_{L-1}])          → 算 L 次 K/V
生成 token[2]: forward([p0, ..., p_{L-1}, t1])           → 算 L+1 次 K/V
生成 token[3]: forward([p0, ..., p_{L-1}, t1, t2])       → 算 L+2 次 K/V
...
```

**KV Cache**：prefill 一次算完所有历史，之后每步只算新 token：

```
prefill: forward([p0, ..., p_{L-1}])    → 缓存每层的 K_cache, V_cache
decode:
  token[1]: 只算 t1 的 K_t1, V_t1
            concat → K_cache = [K_cache; K_t1]
            query: q_t1 @ K_cache.T → 复用历史 V
  token[2]: 只算 t2 的 K_t2, V_t2
            concat → K_cache = [K_cache; K_t2]
  ...
```

### 3.3 复杂度分析

设生成 $N$ 个新 token，序列总长度约为 $T = L_{prompt} + N$：

**朴素生成**（每步 forward 整个序列）：

$$\text{cost}(t) = O(t \cdot d), \quad \text{总 cost} = \sum_{t=1}^{N} O(t \cdot d) = O(N^2 \cdot d)$$

（这里已经是单步 attention 的线性近似；含完整 attention 矩阵时实际是 $O(N^3 \cdot d)$ 量级。）

**KV Cache**（每步只算新 token）：

$$\text{cost}(t) = O(d), \quad \text{总 cost} = O(N \cdot d)$$

对长序列，KV cache 的加速接近 $O(N)$ 倍——序列越长，收益越显著。

::: tip LLM 视角
这个复杂度差异在短序列（N=10）时感觉不大，但在 8K context 的生产场景下，朴素生成每步要做 8000 个 token 的 attention，而 KV cache 版本只做 1 个 token 的查询。理论收益是 8000×。实测受限于内存带宽，但仍有 3-10× 的实际加速。
:::

---

## 四、KV Cache 内存估算

### 4.1 内存公式

每个样本（batch size = 1）的 KV cache 占用：

$$\text{cache\_bytes} = 2 \times L \times H \times T \times d_h \times \text{bytes\_per\_float}$$

其中：
- **2** = K 和 V 各一份
- **L** = Transformer 层数
- **H** = attention head 数
- **T** = 当前序列长度（随生成增长）
- **$d_h$** = head 维度（$d_h = d_\text{model} / H$）
- **bytes_per_float** = fp32: 4，fp16/bf16: 2，int8: 1

### 4.2 不同规模模型的 cache 大小

| 模型 | 配置 (L×H×$d_h$) | T=4096 cache (fp16) | 备注 |
|------|------------------|----------------------|------|
| 本 demo | 6×6×32 | ~1 MB | 教学用，极小 |
| GPT-2 small | 12×12×64 | ~75 MB | 可接受 |
| LLaMA-7B | 32×32×128 | ~1 GB | 单卡勉强 |
| LLaMA-70B | 80×64×128 | ~10 GB | 需要多卡 |

> 当 batch=32（服务 32 个并发请求），LLaMA-70B 的 KV cache 达到 **320 GB**，远超单卡显存。这是 PagedAttention 解决的核心问题之一。

### 4.3 我们 demo 的实测 cache 大小

`kv_cache.py` 实测（n_new=100，序列到 cache 末尾时）：

```
cache 大小: 900 KB
cache 形状(每层): K=(1, 6, 100, 32), V=(1, 6, 100, 32)
```

即 `batch=1, heads=6, seq_len=100, head_dim=32`。与公式吻合：

$$2 \times 6 \times 6 \times 100 \times 32 \times 4\ \text{bytes(fp32)} = 921,600\ \text{bytes} \approx 900\ \text{KB}$$

---

## 五、我们 demo 的实现

### 5.1 CausalSelfAttention 的 kv_cache 参数

`gpt_train.py` 中 `CausalSelfAttention.forward` 支持可选的 `kv_cache` 参数：

```python
def forward(self, x, kv_cache=None):
    B, T, C = x.shape
    q, k, v = self.qkv(x).split(C, dim=2)
    # 分头
    q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    if kv_cache is not None:
        past_k, past_v = kv_cache
        k = torch.cat([past_k, k], dim=2)   # 追加新 token 的 K
        v = torch.cat([past_v, v], dim=2)   # 追加新 token 的 V
        new_cache = (k, v)
    else:
        new_cache = None

    att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    if new_cache is None:   # 训练阶段才需要 causal mask
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
    att = F.softmax(att, dim=-1)
    ...
```

**两个关键设计**：
1. 训练时（`kv_cache=None`）：正常计算 + causal mask，整个序列一起处理
2. 推理时（传入 cache）：concat 历史 K/V，新 token 的 query 可以看到所有历史，**不需要 mask**（因为新 token 天然在末尾，自回归已经保证了因果性）

### 5.2 GPT.generate 接口

`gpt_train.py` 中 `generate` 方法的 `use_cache` 开关：

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=False):
    self.eval()
    kv_caches = [None] * self.cfg.n_layer if use_cache else None
    for _ in range(max_new_tokens):
        if use_cache and kv_caches[0] is not None:
            idx_in = idx[:, -1:]          # decode 阶段：只输入最新 1 个 token
        else:
            idx_in = idx[:, -self.cfg.block_size:]  # prefill 或无 cache：输入整段
        ...
```

### 5.3 kv_cache.py 对比测试

`kv_cache.py` 中的 `gen_no_cache` 和 `gen_with_cache` 构成了完整的基准对比：

```python
# 无 cache：每步输入完整序列
def gen_no_cache(model, prompt_ids, max_new):
    idx = torch.tensor([prompt_ids], ...)
    for _ in range(max_new):
        idx_in = idx[:, -cfg.block_size:]
        logits, _, _ = model(idx_in)      # 每次 forward 整个序列
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)

# 有 cache：decode 阶段每步只输入 1 个 token
def gen_with_cache(model, prompt_ids, max_new):
    # prefill
    logits, _, _ = model(idx)
    ...
    # decode loop
    for _ in range(max_new - 2):
        logits, _, kv_caches = model(next_id, kv_caches=kv_caches)
        ...
```

**实测结果**（n_new=100，prompt="ROMEO:"）：

```
生成 100 个 token:
  无 cache: 加速比分母
  有 cache: 更快
  加速比  : 1.88×
  cache 大小: 900 KB
  cache 形状(每层): K=(1, 6, 100, 32), V=(1, 6, 100, 32)
```

::: warning 为什么我们的 demo 加速只有 1.88×？

有两层原因需要如实解释：

**原因 1：规模效应**

本 demo 是 ~3M 参数的极小模型，在 MPS/CPU 上，单次矩阵乘法耗时极短，Python 循环开销和设备调度延迟占据了相当比例。KV cache 节省的矩阵计算时间被这些固定开销"稀释"了。

真实的 LLaMA-7B 在 H100 上生成 1K token 时，实测加速通常在 **5-10×** 以上——规模越大，矩阵计算占比越高，cache 收益越显著。

**原因 2：position embedding 的硬限制（真实 bug，请注意）**

本 demo 使用学习式位置编码：

```python
self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # block_size = 128
```

若 `prompt 长度 + 生成长度 > 128`，position id 越界：
- **CPU 上**：抛出 IndexError，程序崩溃，错误明显
- **MPS 上（Apple Silicon GPU）**：越界静默发生，产生垃圾 embedding，模型继续"生成"无意义输出，**没有任何报错**

这是 `gpt_train.py` 的一个已知简化，出于教学目的保留（实现简单，限制在 demo 规模内不影响实验）。生产 LLM 使用 RoPE，位置编码按公式动态计算，不存在上界限制。
:::

---

## 六、注意力可视化：理解模型在做什么

KV cache 解决了"如何高效计算"的问题；注意力可视化帮助我们理解"模型在计算什么"。

### 6.1 实现技巧：Monkey-Patch

`attention_visualization.py` 的核心是 `patch_attn_to_record`——在不修改原始模型代码的前提下，把注意力权重"偷"出来：

```python
def patch_attn_to_record(model):
    """Monkey-patch CausalSelfAttention.forward 让它把权重存到 self.last_attn。"""
    def forward(self, x, kv_cache=None):
        ...
        att = F.softmax(att, dim=-1)
        self.last_attn = att.detach()      # ← 保存 softmax 后的权重
        ...
    for block in model.blocks:
        block.attn.forward = forward.__get__(block.attn, CausalSelfAttention)
```

关键点：在 `F.softmax` 之后、`dropout` 之前存储——此时权重已归一化为概率分布（和为 1），可直接映射到 ASCII 灰度级。

### 6.2 ASCII Heatmap

`show_attn_grid` 使用 4 级灰度字符 `" ·∙○●"` 显示注意力矩阵：

```python
chars = " ·∙○●"
# 每个权重值映射到对应字符
chars[min(int(v * len(chars)), len(chars)-1)]
```

对 prompt `"ROMEO: But soft!"` 的示意输出（实际形状由模型决定）：

```
── Layer 0, Head 0 ──（浅层：关注邻近 token）
       R  O  M  E  O  :     B  u  t
  R   ●●●···············
  O   ···●●●············
  M   ·····●●●··········
  E   ·······●●●········
  O   ·········●●●······
  :   ···········●●●····
      ·············●●●··
  B   ···············●●●
  u   ●·············●●●·
  t   ···············●●●

── Layer 5, Head 5 ──（深层：长程语义关联）
       R  O  M  E  O  :     B  u  t
  R   ●●●○···○··●·······
  O   ·●●●····○·····●···
  M   ·○·●●●·····○·····○
  ...（注意力分散到远端 token）
```

### 6.3 各层注意力模式

| 层深度 | 典型模式 | 解释 |
|--------|---------|------|
| 浅层（Layer 0-1） | 对角线强，短程依赖 | 类似 n-gram，聚合邻近字符/词 |
| 中层（Layer 2-3） | 开始出现跨距依赖 | 句法结构（主谓、修饰关系） |
| 深层（Layer 4-5） | 长程、非局部注意力 | 语义关联、主题词汇（"ROMEO" ↔ "love"） |

::: tip LLM 视角
Anthropic 的机制可解释性研究中，**induction heads** 是最著名的发现之一：某些特定的 head 专门实现"如果我见过 A→B 模式，现在遇到 A，就预测 B"这种 in-context learning 能力。这类头在中层最常见。

另一类 **retrieval heads** 负责从长 context 中精准召回特定信息（如"文章第 3 段提到的数字是多少"）。理解这些机制是提升模型 long-context 能力的基础。
:::

---

## 七、生产推理引擎

从 demo 到服务数百万用户，需要专门的推理引擎。

### 7.1 主要框架

| 引擎 | 来源 | 核心特点 |
|------|------|---------|
| **vLLM** | UC Berkeley → LMSys | PagedAttention；高吞吐；已商业化 |
| **TensorRT-LLM** | NVIDIA | 深度融合 CUDA kernel；fp8/int4 量化 |
| **TGI**（Text Generation Inference） | HuggingFace | 直接对接 transformers 模型；部署门槛低 |
| **llama.cpp** | 社区（ggerganov） | CPU/Metal/CUDA；Q4/Q5 量化；可在 Mac 运行 70B |
| **SGLang** | Stanford | 结构化输出；高并发多轮对话优化 |

### 7.2 关键优化技术汇总

| 技术 | 解决的问题 | 典型收益 |
|------|-----------|---------|
| **KV Cache** | 消除重复 K/V 计算 | decode 从 $O(N^2)$ 降到 $O(N)$ |
| **PagedAttention** | KV cache 内存碎片 | 吞吐 ↑ 2-4×，内存利用率 ↑ |
| **Continuous Batching** | 不同长度请求动态拼 batch | GPU 利用率大幅提升 |
| **FlashAttention** | HBM 访问次数过多 | attention 速度 ↑ 2-4×，内存 $O(N)$ |
| **Speculative Decoding** | decode 阶段 GPU 空转 | 端到端延迟 ↓ 2-3×（视命中率） |
| **量化（int8/int4）** | 权重 + KV cache 显存占用 | 显存 ↓ 50-75%，速度视硬件而定 |
| **GQA / MQA** | KV cache 过大 | cache 大小 ↓ H/G 倍 |

::: tip LLM 视角
这些技术不是互斥的——生产系统会叠加使用：FlashAttention（更快的 attention kernel）+ KV cache（消除重复计算）+ PagedAttention（管理 cache 内存）+ Continuous batching（提高 GPU 利用率）+ int8 量化（压缩显存）。vLLM 在一个系统里集成了其中大部分。
:::

---

## 八、FlashAttention：改实现，不改算法

### 8.1 核心思想

标准 attention 实现的内存瓶颈：

```
标准实现的数据流：
  GPU SRAM (快，小)          GPU HBM (慢，大)
  ─────────────────         ─────────────────
  计算 Q @ K.T     ──写──→   存储整个 (N×N) score 矩阵
  读取 score 矩阵  ←──读──   score (N×N)
  计算 softmax     ──写──→   存储 softmax weights (N×N)
  读取 weights     ←──读──   weights (N×N)
  计算 weights @ V ──写──→   输出
```

对于序列长度 $N=8192$，score 矩阵是 $8192 \times 8192 \approx 268M$ 个 float，占用 ~1 GB 显存，且要被反复读写。

**FlashAttention**（Dao et al. 2022）的解法：把 Q/K/V 分成小 tile，每次只加载一个 tile 到 SRAM 里，在 SRAM 内完成整个 attention 计算，**永远不把中间的 $N \times N$ 矩阵写回 HBM**：

```
FlashAttention 数据流：
  GPU SRAM (快)              GPU HBM (慢)
  ─────────────────         ─────────────────
  加载 Q tile, K tile, V tile ←── 只读小块
  在 SRAM 内：
    score = Q_tile @ K_tile.T
    softmax（在线算法，增量更新）
    output += softmax @ V_tile
  写出 output tile      ──→  只写小块
```

### 8.2 性能特征

- **速度**：attention 计算 2-4× 加速（取决于序列长度和硬件）
- **内存**：中间 score 矩阵从 $O(N^2)$ 降到 $O(N)$（tile 大小固定）
- **精度**：数值等价于标准实现（在线 softmax 算法保证精度）
- **算法复杂度**：仍然是 $O(N^2)$（该算的还是要算），改的是 IO

FlashAttention-2（2023）进一步优化了 warp-level 并行和线程块的工作划分；FlashAttention-3（2024）针对 H100 的 Tensor Core 做了异步流水线优化。

::: tip LLM 视角
FlashAttention 现在几乎是"默认开启"的基础设施——PyTorch 2.0 的 `F.scaled_dot_product_attention` 在检测到合适条件时会自动调用 FlashAttention kernel。训练和推理都受益。本 demo 没有使用（为了教学清晰），生产代码请务必开启。
:::

---

## 九、PagedAttention：像操作系统管理内存一样管理 KV Cache

### 9.1 问题：KV Cache 的内存碎片

传统 KV cache 的分配方式：为每个请求**预先分配最大长度**的连续显存块。

```
显存布局（传统方式）：
┌─────────────────────────────────────────────────────┐
│ 请求 A: [已用 500 tokens ████████████████░░░░░░░░░░░] │  ← 预分配 max_len=1000
│ 请求 B: [已用 100 tokens ████░░░░░░░░░░░░░░░░░░░░░░░] │  ← 预分配 max_len=1000
│ 请求 C: [已用 999 tokens ████████████████████████████] │
│ [碎片]  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
└─────────────────────────────────────────────────────┘
```

问题：
- 请求 A 预分配了 max_len=1000 的显存，但实际只用了 500，另外 500 浪费
- 无法预知请求实际会生成多长，若分少了会 OOM，分多了会浪费
- 不同长度的请求难以高效共用显存

### 9.2 PagedAttention 的解法

灵感直接来自操作系统的**虚拟内存分页**：

```
PagedAttention 显存布局：
┌─────────────────────────────────────────────────────┐
│ 物理 Block 0: [请求 A, token 0-15]                    │  ← 固定大小 block (16 tokens)
│ 物理 Block 1: [请求 B, token 0-15]                    │
│ 物理 Block 2: [请求 A, token 16-31]                   │
│ 物理 Block 3: [请求 C, token 0-15]                    │
│ 物理 Block 4: [请求 A, token 32-47]                   │
│ 物理 Block 5: [请求 B, token 16-31]                   │
│ ...（按需分配，不连续但通过 block table 索引）           │
└─────────────────────────────────────────────────────┘

block table (逻辑 → 物理):
  请求 A: [block 0 → 物理 0, block 1 → 物理 2, block 2 → 物理 4]
  请求 B: [block 0 → 物理 1, block 1 → 物理 5]
  请求 C: [block 0 → 物理 3]
```

**核心收益**：
- 显存利用率从 ~55% 提升到 ~90%+（只有最后一个 block 可能有内部碎片）
- 支持多请求共享相同的 prefix cache（如 system prompt）
- 不需要预分配最大长度，按实际生成动态扩展

Kwon et al. 2023 发表的 vLLM 论文报告：与 HuggingFace 默认推理相比，吞吐量提升 **2-4×**（视 batch size 和序列长度）。

---

## 十、配套代码

| 文件 | 内容 | 关键函数 |
|------|------|---------|
| `kv_cache.py` | 朴素 vs cache 加速对比 | `gen_no_cache`, `gen_with_cache`, `cache_size_bytes` |
| `attention_visualization.py` | 注意力 heatmap 调试工具 | `patch_attn_to_record`, `show_attn_grid` |
| `gpt_train.py` | 内置 use_cache + KV cache 接口 | `CausalSelfAttention.forward(kv_cache=)`, `generate(use_cache=)` |

::: tip 跑一遍

```bash
cd ml_foundations/transformer_training

# 先训练（生成 checkpoint，约 30 秒 MPS / 5 分钟 CPU）
python gpt_train.py

# KV cache 加速对比
python kv_cache.py

# 注意力热图可视化
python attention_visualization.py
```

预期输出（kv_cache.py）：

```
生成 100 个 token:
  无 cache: ~Xs   (~Y tok/s)
  有 cache: ~Xs   (~Y tok/s)
  加速比  : 1.88×
  cache 大小: 900.0 KB
  cache 形状(每层): K=(1, 6, 100, 32), V=(1, 6, 100, 32)
```

:::

::: warning 运行前注意 position embedding 限制

`gpt_train.py` 的 `block_size=128`，即位置编码只支持最多 128 个 token。

若 prompt 较长 + 生成 token 数超过 128，**在 MPS 设备上会静默生成垃圾输出**（CPU 上会报 IndexError）。

`kv_cache.py` 默认 prompt="ROMEO:"（5 tokens）+ 生成 100 tokens = 105 tokens，在安全范围内。若要实验更长序列，请先降低 n_new 或缩短 prompt。
:::

---

## 十一、本章总结

你已经完成了整个 Transformer 从零训练的旅程：

```
BPE 分词        → 理解数据如何进入模型
自注意力        → 理解 Transformer 的核心计算
位置编码        → 理解序列顺序如何编码
GPT 训练        → 亲手跑完整的训练循环
采样策略        → 理解如何从概率分布"说话"
注意力可视化    → 打开黑盒，看模型关注什么
KV Cache       → 理解推理加速的核心机制    ← 你在这里
```

**关键结论**：
- KV cache 把 decode 成本从 $O(N^2)$ 降到 $O(N)$，是所有生产 LLM 推理的基础
- 代价是线性增长的显存——长 context 时 cache 内存成为主要瓶颈
- PagedAttention（vLLM）、FlashAttention、GQA 是管理这一代价的工程方案
- 小 demo 的 1.88× 加速与生产系统的 5-10× 差距，主要来自规模效应（矩阵计算占比）

**下一站建议**：

| 路径 | 推荐资源 |
|------|---------|
| 深入理解代码 | Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT)，架构与本 demo 一脉相承 |
| 微调已有模型 | HuggingFace `transformers` + PEFT 库，从 LLaMA/Qwen 起步 |
| 理解工业推理 | 阅读 vLLM 源码，重点看 `worker.py` 和 `block_manager.py` |
| 机制可解释性 | Anthropic 的 [Transformer Circuits](https://transformer-circuits.pub/) 系列文章 |

::: tip LLM 视角
这个 demo 里的 3M 参数模型和 LLaMA-70B 在**架构骨架上完全一致**——都是 Transformer block 堆叠，都有 multi-head attention + FFN，都用 KV cache 推理。区别只是参数量（70B vs 3M）、训练数据量（数万亿 token vs 1MB）、和一些现代化改进（RoPE、SwiGLU、GQA）。

你已经亲手实现了这个骨架的每一块。
:::

---

## 十二、延伸阅读

### 核心论文

- Dao et al. **"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"** (2022) — attention kernel 的革命性优化
- Dao et al. **"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"** (2023) — 进一步提升并行效率
- Kwon et al. **"Efficient Memory Management for Large Language Model Serving with PagedAttention"** (2023, vLLM 论文) — KV cache 分页管理
- Pope et al. **"Efficiently Scaling Transformer Inference"** (Google, 2022) — 系统性分析大模型推理的瓶颈
- Leviathan et al. **"Fast Inference from Transformers via Speculative Decoding"** (2023) — 投机解码
- Chen et al. **"Accelerating Large Language Model Decoding with Speculative Sampling"** (2023) — 同期投机采样工作

### 博客与工具

- Anthropic **"In-context Learning and Induction Heads"** — 机制可解释性，理解 attention head 功能分工
- Lilian Weng **"Large Transformer Model Inference Optimization"** — 推理优化全景综述
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp) — 在消费级硬件（Mac M 系、普通 CPU）运行 LLM 的参考实现，支持 Q4/Q5 量化
- [vLLM GitHub](https://github.com/vllm-project/vllm) — 生产级推理引擎，阅读源码了解 PagedAttention 工程实现
- [BertViz](https://github.com/jessevig/bertviz) — Transformer 注意力可视化工具，比本 demo 的 ASCII 图更精细
