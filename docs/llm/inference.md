---
title: 生产级推理部署与加速
---

<script setup>
const code1 = `import numpy as np

np.random.seed(42)

# ========================================
# 模型量化效果模拟
# ========================================
# 模拟一个 Transformer 层的权重矩阵 (768 x 768)
print("=== 模型量化原理演示 ===\\n")

# 生成模拟权重 (正态分布，模拟真实模型权重)
weights_fp32 = np.random.randn(768, 768).astype(np.float32)
print(f"原始权重矩阵形状: {weights_fp32.shape}")
print(f"原始权重范围: [{weights_fp32.min():.4f}, {weights_fp32.max():.4f}]")
print(f"原始精度 (FP32): {weights_fp32.nbytes / 1024 / 1024:.2f} MB")

# FP16 量化
weights_fp16 = weights_fp32.astype(np.float16)
fp16_error = np.abs(weights_fp32 - weights_fp16.astype(np.float32)).mean()
print(f"\\n--- FP16 量化 ---")
print(f"模型大小: {weights_fp16.nbytes / 1024 / 1024:.2f} MB (压缩比 2x)")
print(f"平均精度损失: {fp16_error:.8f}")

# INT8 量化 (对称量化)
def quantize_int8(tensor):
    scale = tensor.max() / 127.0
    quantized = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
    return quantized, scale

def dequantize_int8(quantized, scale):
    return quantized.astype(np.float32) * scale

weights_int8, scale_int8 = quantize_int8(weights_fp32)
weights_int8_reconstructed = dequantize_int8(weights_int8, scale_int8)
int8_error = np.abs(weights_fp32 - weights_int8_reconstructed).mean()
print(f"\\n--- INT8 量化 (对称量化) ---")
print(f"scale factor: {scale_int8:.6f}")
print(f"模型大小: {weights_int8.nbytes / 1024 / 1024:.2f} MB (压缩比 4x)")
print(f"平均精度损失: {int8_error:.6f}")

# INT4 量化 (分组量化, group_size=128)
def quantize_int4_grouped(tensor, group_size=128):
    """模拟 INT4 分组量化 (GPTQ/AWQ 风格)"""
    rows, cols = tensor.shape
    # 按列分组
    n_groups = cols // group_size
    quantized = np.zeros_like(tensor, dtype=np.int8)
    scales = np.zeros((rows, n_groups), dtype=np.float32)

    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        group = tensor[:, start:end]
        # 每组独立计算 scale
        group_scale = np.abs(group).max(axis=1, keepdims=True) / 7.0  # INT4: -8~7
        scales[:, g] = group_scale.squeeze()
        group_scale[group_scale == 0] = 1.0
        quantized[:, start:end] = np.clip(np.round(group / group_scale), -8, 7)

    return quantized, scales

def dequantize_int4_grouped(quantized, scales, group_size=128):
    rows, cols = quantized.shape
    n_groups = cols // group_size
    result = np.zeros((rows, cols), dtype=np.float32)
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size
        result[:, start:end] = quantized[:, start:end] * scales[:, g:g+1]
    return result

weights_int4, scales_int4 = quantize_int4_grouped(weights_fp32)
weights_int4_reconstructed = dequantize_int4_grouped(weights_int4, scales_int4)
int4_error = np.abs(weights_fp32 - weights_int4_reconstructed).mean()

# INT4 实际只用 4 bit，但存储为 int8，实际大小为一半
int4_size = weights_int4.nbytes / 2 + scales_int4.nbytes  # 4bit数据 + scales
print(f"\\n--- INT4 分组量化 (group_size=128) ---")
print(f"模型大小: {int4_size / 1024 / 1024:.2f} MB (压缩比 ~8x)")
print(f"平均精度损失: {int4_error:.6f}")

# 汇总对比
print(f"\\n{'='*50}")
print(f"{'精度':<8} {'大小(MB)':<12} {'压缩比':<10} {'平均误差':<12}")
print(f"{'-'*50}")
fp32_size = weights_fp32.nbytes / 1024 / 1024
print(f"{'FP32':<8} {fp32_size:<12.2f} {'1.0x':<10} {'0 (基准)':<12}")
print(f"{'FP16':<8} {weights_fp16.nbytes/1024/1024:<12.2f} {'2.0x':<10} {fp16_error:<12.8f}")
print(f"{'INT8':<8} {weights_int8.nbytes/1024/1024:<12.2f} {'4.0x':<10} {int8_error:<12.6f}")
print(f"{'INT4':<8} {int4_size/1024/1024:<12.2f} {'~8.0x':<10} {int4_error:<12.6f}")
print(f"{'='*50}")

# 精度损失分布可视化
print(f"\\n=== INT8 量化误差分布 ===")
errors = np.abs(weights_fp32.flatten()[:1000] - weights_int8_reconstructed.flatten()[:1000])
bins = [0, 0.001, 0.005, 0.01, 0.02, 0.05, float('inf')]
labels = ['<0.001', '0.001-0.005', '0.005-0.01', '0.01-0.02', '0.02-0.05', '>0.05']
for i in range(len(bins)-1):
    count = np.sum((errors >= bins[i]) & (errors < bins[i+1]))
    pct = count / len(errors) * 100
    bar = '█' * int(pct / 2)
    print(f"  {labels[i]:<12} {pct:5.1f}% {bar}")
`

const code2 = `import numpy as np

np.random.seed(42)

# ========================================
# KV Cache 内存计算 & PagedAttention 模拟
# ========================================

print("=== KV Cache 内存计算 ===\\n")

# 模型参数配置 (类似 LLaMA-7B)
n_layers = 32
n_heads = 32
head_dim = 128
dtype_bytes = 2  # FP16

# 单个 token 的 KV Cache 大小
kv_per_token = 2 * n_layers * n_heads * head_dim * dtype_bytes  # K 和 V
print(f"模型配置: {n_layers}层, {n_heads}头, head_dim={head_dim}, FP16")
print(f"单个 token KV Cache: {kv_per_token} bytes = {kv_per_token/1024:.1f} KB")

# 不同序列长度的 KV Cache
print(f"\\n--- 单请求 KV Cache 内存占用 ---")
seq_lengths = [512, 2048, 4096, 8192, 32768, 131072]
for seq_len in seq_lengths:
    mem = kv_per_token * seq_len
    print(f"  seq_len={seq_len:>6d}: {mem/1024/1024:>8.1f} MB ({mem/1024/1024/1024:.2f} GB)")

# 批处理场景
print(f"\\n--- 批处理场景 (seq_len=2048) ---")
batch_sizes = [1, 8, 16, 32, 64, 128]
seq_len = 2048
for bs in batch_sizes:
    mem = kv_per_token * seq_len * bs
    print(f"  batch_size={bs:>3d}: {mem/1024/1024/1024:>6.2f} GB")

# ========================================
# PagedAttention 块分配模拟
# ========================================
print(f"\\n{'='*55}")
print(f"=== PagedAttention 块分配模拟 ===\\n")

# PagedAttention 参数
block_size = 16  # 每个 block 存储 16 个 token 的 KV
block_mem = block_size * kv_per_token  # 单个 block 的内存大小
total_gpu_mem = 24 * 1024 * 1024 * 1024  # 24 GB GPU
model_mem = 14 * 1024 * 1024 * 1024  # 模型本身占 14 GB
available_mem = total_gpu_mem - model_mem
total_blocks = int(available_mem / block_mem)

print(f"GPU 显存: 24 GB")
print(f"模型权重: 14 GB")
print(f"可用于 KV Cache: {available_mem/1024/1024/1024:.1f} GB")
print(f"Block size: {block_size} tokens")
print(f"每个 Block 内存: {block_mem/1024:.1f} KB")
print(f"总可用 Blocks: {total_blocks}")

# 模拟多个请求的 block 分配
class PagedKVCache:
    def __init__(self, total_blocks):
        self.total_blocks = total_blocks
        self.free_blocks = list(range(total_blocks))
        self.allocated = {}  # request_id -> [block_ids]

    def allocate(self, request_id, n_tokens):
        """为请求分配 blocks"""
        n_blocks_needed = (n_tokens + block_size - 1) // block_size
        if len(self.free_blocks) < n_blocks_needed:
            return False
        blocks = [self.free_blocks.pop(0) for _ in range(n_blocks_needed)]
        self.allocated[request_id] = blocks
        return True

    def free(self, request_id):
        """释放请求的 blocks"""
        if request_id in self.allocated:
            self.free_blocks.extend(self.allocated[request_id])
            del self.allocated[request_id]

    def utilization(self):
        used = self.total_blocks - len(self.free_blocks)
        return used / self.total_blocks * 100

# 模拟动态请求处理
cache = PagedKVCache(total_blocks)
print(f"\\n--- 模拟请求调度 ---")
print(f"{'操作':<25} {'已用Blocks':<12} {'利用率':<10} {'碎片率':<10}")
print(f"{'-'*55}")

# 模拟不同长度的请求到来
requests = [
    ("请求A: 1024 tokens", "A", 1024),
    ("请求B: 512 tokens", "B", 512),
    ("请求C: 2048 tokens", "C", 2048),
    ("请求D: 256 tokens", "D", 256),
    ("释放请求B", None, 0),
    ("请求E: 768 tokens", "E", 768),
    ("释放请求A", None, 0),
    ("请求F: 1500 tokens", "F", 1500),
]

for desc, req_id, n_tokens in requests:
    if req_id is None:
        # 找到要释放的请求
        free_id = desc.replace("释放请求", "").strip()
        cache.free(free_id)
    else:
        cache.allocate(req_id, n_tokens)

    used = cache.total_blocks - len(cache.free_blocks)
    util = cache.utilization()
    # 计算碎片率 (非连续空闲块比例)
    free_sorted = sorted(cache.free_blocks)
    fragments = 0
    for i in range(1, len(free_sorted)):
        if free_sorted[i] != free_sorted[i-1] + 1:
            fragments += 1
    frag_rate = fragments / max(len(free_sorted), 1) * 100
    print(f"  {desc:<23} {used:<12d} {util:<10.1f}% {frag_rate:<10.1f}%")

print(f"\\n--- PagedAttention 优势 ---")
# 对比传统预分配方式
max_seq_len = 2048
traditional_per_req = max_seq_len  # 预分配最大长度
actual_tokens = [1024, 512, 2048, 256, 768, 1500]
traditional_total = sum([max_seq_len] * len(actual_tokens))
paged_total = sum([(t + block_size - 1) // block_size * block_size for t in actual_tokens])
actual_total = sum(actual_tokens)

print(f"实际使用 tokens: {actual_total}")
print(f"传统预分配 (max_len={max_seq_len}): {traditional_total} tokens 空间")
print(f"PagedAttention 分配: {paged_total} tokens 空间")
print(f"传统方式内存浪费: {(traditional_total-actual_total)/traditional_total*100:.1f}%")
print(f"PagedAttention 浪费: {(paged_total-actual_total)/paged_total*100:.1f}%")
print(f"内存效率提升: {traditional_total/paged_total:.2f}x")
`
</script>

# 生产级推理部署与加速

大模型从训练到生产，推理部署是最关键的"最后一公里"。如何在有限的硬件资源下实现高吞吐、低延迟的推理服务，是每个 AI 工程师必须掌握的核心技能。

## 模型量化原理

量化（Quantization）是将模型权重从高精度浮点数转换为低精度表示的技术，是降低推理成本的最直接手段。

### 数值精度对比

| 精度格式 | 位宽 | 数值范围 | 典型用途 |
|---------|------|---------|---------|
| FP32 | 32 bit | ±3.4×10³⁸ | 训练基准 |
| FP16 | 16 bit | ±65504 | 混合精度训练 |
| BF16 | 16 bit | ±3.4×10³⁸ | 训练/推理（范围大） |
| INT8 | 8 bit | -128~127 | 推理量化 |
| INT4 | 4 bit | -8~7 | 极致压缩推理 |

### 量化方法分类

```
量化技术
├── 训练后量化 (PTQ - Post-Training Quantization)
│   ├── 对称量化: scale = max(|w|) / (2^(n-1) - 1)
│   ├── 非对称量化: scale + zero_point
│   └── 分组量化: 按 group_size 分组，每组独立 scale
│
├── GPTQ (逐层最优量化)
│   ├── 基于 Hessian 矩阵的最优量化
│   ├── 逐列量化，最小化输出误差
│   └── 支持 INT4/INT3，精度损失小
│
└── AWQ (Activation-aware Weight Quantization)
    ├── 保护重要权重通道 (salience-based)
    ├── 基于激活值分布确定重要性
    └── 不需要反向传播，速度快
```

::: tip GPTQ vs AWQ 选择建议
- **GPTQ**: 量化质量更高，适合对精度要求严格的场景，但量化过程较慢
- **AWQ**: 量化速度快，推理速度略优，适合快速部署场景
- 两者在 INT4 下的 perplexity 差异通常 < 0.5
:::

### 交互示例：量化效果模拟

<PythonRunner :code="code1" />

## 推理引擎对比

| 特性 | vLLM | TensorRT-LLM | Ollama | TGI |
|------|------|-------------|--------|-----|
| **开发方** | UC Berkeley | NVIDIA | Ollama | Hugging Face |
| **核心技术** | PagedAttention | FP8/INT4 + 自定义 Kernel | llama.cpp | Flash Attention |
| **硬件支持** | NVIDIA GPU | NVIDIA GPU (专属) | CPU + GPU | NVIDIA GPU |
| **易用性** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **吞吐量** | 极高 | 最高 | 中等 | 高 |
| **延迟** | 低 | 最低 | 中等 | 低 |
| **量化支持** | GPTQ/AWQ/FP8 | FP8/INT4/INT8 | GGUF (多种) | GPTQ/AWQ |
| **分布式** | Tensor Parallel | TP + PP | 单机 | TP |
| **适用场景** | 生产服务 | 极致性能 | 本地开发 | HF 生态集成 |

::: info 选择建议
- **本地开发/实验**: Ollama（一键运行，资源占用小）
- **生产服务**: vLLM（社区活跃，功能完善）
- **极致性能**: TensorRT-LLM（NVIDIA 深度优化）
- **HF 生态**: TGI（与 Transformers 无缝集成）
:::

## 核心加速技术

### PagedAttention

传统推理中，每个请求需要预分配最大序列长度的连续内存，造成严重浪费。PagedAttention 借鉴操作系统虚拟内存的分页思想：

```
传统方式 (连续分配):
┌─────────────────────────────────────────────────┐
│ 请求A: [实际1024 tokens][     浪费 1024 空间     ]│  预分配 max_len=2048
├─────────────────────────────────────────────────┤
│ 请求B: [实际512][          浪费 1536 空间        ]│
└─────────────────────────────────────────────────┘
  内存利用率: ~37.5%

PagedAttention (分页分配):
┌──────┬──────┬──────┬──────┬──────┬──────┐
│ A-b0 │ A-b1 │ B-b0 │ A-b2 │ B-b1 │ A-b3 │  Block Pool
└──────┴──────┴──────┴──────┴──────┴──────┘
  - Block 大小固定 (如 16 tokens)
  - 非连续分配，按需增长
  - 内存利用率: >95%
```

**关键优势**：
- 内存利用率从 ~50% 提升到 >95%
- 支持更大 batch size，提高吞吐量
- 支持 Copy-on-Write，实现 beam search 共享前缀

### Continuous Batching

传统 Static Batching 中，batch 内所有请求必须等最长的完成才能释放资源：

```
Static Batching:
时间 ──────────────────────────────────────────>
请求A: [████████████████████████████████]  (长请求)
请求B: [████████████]___等待A完成_________   (短请求被阻塞)
请求C: [██████]_________等待A完成_________   (更短的请求)
              ↑ B/C 已完成但无法释放 slot

Continuous Batching:
时间 ──────────────────────────────────────────>
请求A: [████████████████████████████████]
请求B: [████████████]
请求D:              [██████████████████]     ← B完成后立即插入
请求C: [██████]
请求E:        [████████████████]             ← C完成后立即插入
```

**效果**：吞吐量提升 2-5x，GPU 利用率显著提高。

### KV Cache 管理

KV Cache 是推理加速的核心——缓存已计算的 Key/Value，避免重复计算：

```
自回归生成过程:

Step 1: "今天"     → 计算 K₁, V₁, 存入 Cache
Step 2: "今天天气" → 从 Cache 读取 K₁V₁, 只计算新 token 的 K₂V₂
Step 3: "今天天气很" → 从 Cache 读取 K₁V₁K₂V₂, 只计算 K₃V₃
...
  → 计算复杂度从 O(n²) 降为 O(n) per step
```

::: warning KV Cache 内存瓶颈
对于 LLaMA-70B (80层, 64头, head_dim=128, FP16):
- 单 token KV Cache = 2 × 80 × 64 × 128 × 2 bytes = **2.5 MB**
- 4096 长度序列 = **10 GB** / 请求
- 这就是为什么大模型推理的主要瓶颈是**内存**而非计算
:::

### 交互示例：KV Cache 与 PagedAttention

<PythonRunner :code="code2" />

## Speculative Decoding（推测解码）

利用小模型"猜测"多个 token，大模型一次性验证，加速自回归生成：

```
传统自回归 (1 token/step):
大模型: [预测t1] → [预测t2] → [预测t3] → [预测t4]
延迟:    100ms      100ms      100ms      100ms    = 400ms

Speculative Decoding (多 token/step):
小模型: [猜测 t1, t2, t3, t4, t5]  ← 5ms (并行猜测)
大模型: [验证: ✓t1, ✓t2, ✓t3, ✗t4] ← 110ms (一次前向)
结果:   接受 3 个 token                 = 115ms (省 ~3x)
```

**关键要素**：
- **Draft Model（草稿模型）**: 同系列小模型，如用 LLaMA-7B 辅助 LLaMA-70B
- **接受率**: 小模型猜对的比例，通常 70-90%
- **加速比**: 通常 2-3x，取决于接受率和 draft 长度
- **无损**: 数学上保证与大模型单独生成的分布完全一致

::: tip 适用场景
- 大模型推理延迟高、计算密集的场景
- Draft model 与 target model 分布越接近，加速比越高
- 不适合 batch size 很大的场景（此时瓶颈是内存带宽而非计算）
:::

## 部署模式

### 单机部署

```
单 GPU 部署 (适合 7B-13B 模型):
┌─────────────────────────────────┐
│           GPU (24-80 GB)         │
│  ┌─────────────────────────────┐│
│  │   完整模型权重 + KV Cache    ││
│  └─────────────────────────────┘│
└─────────────────────────────────┘
```

### Tensor Parallelism（张量并行）

将单层的矩阵运算切分到多个 GPU：

```
Tensor Parallel (TP=4):
         输入 X
           │
    ┌──────┼──────┬──────┐
    ▼      ▼      ▼      ▼
┌──────┐┌──────┐┌──────┐┌──────┐
│GPU 0 ││GPU 1 ││GPU 2 ││GPU 3 │
│W[:,0]││W[:,1]││W[:,2]││W[:,3]│  ← 权重按列切分
└──┬───┘└──┬───┘└──┬───┘└──┬───┘
   │       │       │       │
   └───────┴───AllReduce───┘      ← 通信开销
           │
         输出 Y
```

- **优势**: 降低单 GPU 显存需求，延迟最优
- **劣势**: 需要高速互联（NVLink），通信开销随 TP 数增加
- **适用**: 同一节点内的多 GPU

### Pipeline Parallelism（流水线并行）

将不同层分配到不同 GPU：

```
Pipeline Parallel (PP=4, 32层模型):
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  GPU 0   │──▶│  GPU 1   │──▶│  GPU 2   │──▶│  GPU 3   │
│Layer 0-7 │   │Layer 8-15│   │Layer16-23│   │Layer24-31│
└──────────┘   └──────────┘   └──────────┘   └──────────┘
     ↑                                              │
     └──────────────── 输出 ◀───────────────────────┘
```

- **优势**: 跨节点通信量小（仅传激活值）
- **劣势**: Pipeline Bubble（流水线气泡）降低效率
- **适用**: 跨节点部署超大模型

### 混合并行策略

| 模型规模 | 推荐策略 | 硬件需求 |
|---------|---------|---------|
| 7B | 单 GPU / TP=2 | 1-2× A100 80GB |
| 13B | TP=2 | 2× A100 80GB |
| 34B | TP=4 | 4× A100 80GB |
| 70B | TP=8 / TP=4+PP=2 | 8× A100 80GB |
| 405B | TP=8+PP=4+ | 32× A100/H100 |

## 性能指标

### 关键指标定义

| 指标 | 全称 | 含义 | 优化方向 |
|------|------|------|---------|
| **TTFT** | Time To First Token | 首 token 延迟 | Prefill 阶段优化 |
| **TPOT** | Time Per Output Token | 每 token 生成时间 | Decode 阶段优化 |
| **Tokens/s** | Throughput per request | 单请求生成速度 | 1/TPOT |
| **QPS** | Queries Per Second | 服务吞吐量 | Batching + 并行 |
| **总吞吐量** | Total Tokens/s | 系统总 token 产出 | QPS × avg_len |

### 延迟分解

```
用户请求 ──▶ [网络] ──▶ [排队] ──▶ [Prefill] ──▶ [Decode × N] ──▶ 响应完成
                                      │                │
                                    TTFT            TPOT × N
                                 (计算密集)        (内存带宽密集)
```

::: info Prefill vs Decode 的不同瓶颈
- **Prefill 阶段**: 一次处理所有输入 token，计算密集（compute-bound）
  - 优化方向: Flash Attention、Tensor Parallel、更快的 GPU
- **Decode 阶段**: 每步只生成 1 个 token，内存带宽密集（memory-bound）
  - 优化方向: 量化降低内存占用、更高内存带宽、Speculative Decoding
:::

### 典型性能参考（A100 80GB）

| 模型 | 精度 | TTFT | Tokens/s | 并发吞吐 |
|------|------|------|----------|---------|
| LLaMA-7B | FP16 | ~50ms | ~100 t/s | ~2000 t/s (batch=32) |
| LLaMA-13B | FP16 | ~80ms | ~70 t/s | ~1500 t/s (batch=16) |
| LLaMA-70B | INT4 | ~200ms | ~40 t/s | ~800 t/s (TP=4) |
| Mixtral-8x7B | FP16 | ~100ms | ~60 t/s | ~1200 t/s (TP=2) |

## 总结

```
生产级推理部署决策树:

延迟敏感？──是──▶ TensorRT-LLM + INT4/FP8 + TP
    │
    否
    │
吞吐优先？──是──▶ vLLM + Continuous Batching + PagedAttention
    │
    否
    │
本地开发？──是──▶ Ollama + GGUF 量化
    │
    否
    │
HF生态？───是──▶ TGI + Flash Attention
```

::: tip 优化优先级建议
1. **量化** (效果最大): FP16 → INT8 → INT4，立竿见影
2. **PagedAttention**: 提升显存利用率，支撑更大并发
3. **Continuous Batching**: 提高吞吐量
4. **Tensor Parallel**: 解决单卡放不下的问题
5. **Speculative Decoding**: 进一步降低延迟
:::
