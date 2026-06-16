# Phase 2: Transformer 从零训练 — 设计文档

> 在 Phase 1（经典 ML / DL / NLP 基础）之上，构建从 tokenizer 到推理优化的完整 Transformer 训练全链路 demo。

**Date**: 2026-06-16
**Status**: 待用户 review
**前置依赖**: Phase 1（已完成，commit `34a0070` + `0f02860`）

---

## 1. 背景与动机

Phase 1 帮零基础读者补齐了"LLM 之前的 ML"。但要真正理解 LLM **内部如何工作**，必须亲手训过一次 Transformer。本 phase 用 nanoGPT 风格的小模型（~5M 参数）让读者：

- 自己实现 BPE tokenizer，理解 LLM 词表是怎么来的
- 自己写 self-attention，理解 `softmax(QKᵀ/√d)V` 不是黑魔法
- 跑通完整训练循环，看 loss 从 4.x 降到 1.x，模型学会模仿莎士比亚
- 实验采样策略，理解 temperature/top-p 在 OpenAI API 里到底干什么
- 实现 KV cache，理解为什么 ChatGPT "第二个 token 比第一个快"

完成后，再看 GPT-4 / LLaMA / Qwen 的源码或论文，所有零件都认识。

---

## 2. 范围决策

### 2.1 已锁定约束（来自 brainstorming Q&A）

| 决策项 | 选择 | 备注 |
|--------|------|------|
| 方向 | 阶段 2：Transformer 从零训练 | 跳过阶段 3（暂不做 LoRA/DPO） |
| 训练规模 | Small：~5M 参数，BPE token 级 | 4 主 demo 训 5 分钟可见效果 |
| demo 数量 | 7 个（增强版） | 含注意力可视化 + KV cache |
| 训练语料 | Tiny Shakespeare（英文） | ~1MB 内置打包，无需下载 |
| 文档形态 | 新建 `docs/ml-foundations/transformer-training/` 子章节 | 6 个 md |
| 侧边栏位置 | "零.5、Transformer 训练实战" | 插在零和一之间 |
| 硬件目标 | Mac CPU 全部可跑；MPS 加速主训练 | 自动检测设备 |
| 测试 | 不写 `tests/` | 沿用 Phase 1 决策 |
| commit 策略 | 单次最终 commit | 沿用 Phase 1 决策 |

### 2.2 显式不做（YAGNI）

- ❌ 多 GPU 分布式训练（DDP/FSDP）
- ❌ FlashAttention / xFormers 集成
- ❌ HuggingFace transformers 库的封装演示
- ❌ 中文语料训练（仅 Tiny Shakespeare 英文）
- ❌ 阶段 3 内容（SFT/LoRA/DPO/量化）
- ❌ 动态 batch / continuous batching（vLLM 那套）
- ❌ Sliding window / GQA 等现代变体（仅在 docs 里提及）

### 2.3 阶段路线（再次声明）

- **阶段 1**（已完成）：经典 ML + DL 基础 + NLP 基础
- **阶段 2**（本次）：Transformer 训练全链路
- **阶段 3**（未来，独立 spec）：SFT、LoRA/QLoRA、DPO、量化部署

---

## 3. 代码目录结构

```
ml_foundations/transformer_training/
├── KNOWLEDGE.md                       # 总览 + 数学推导 + 与 LLM 对应关系
├── data/
│   └── tiny_shakespeare.txt           # ~1.1MB 内置语料
├── bpe_tokenizer.py                   # 从零实现 BPE
├── attention_from_scratch.py          # 自注意力 NumPy + PyTorch 双实现
├── positional_encoding.py             # 绝对/学习/RoPE 对比
├── gpt_train.py                       # 主 demo：完整 GPT 训练（导出 GPT 类）
├── sampling_strategies.py             # 加载 ckpt，4 种采样对比
├── attention_visualization.py         # 加载 ckpt，注意力 ASCII 热点图
└── kv_cache.py                        # 加载 ckpt，KV cache 推理加速对比
```

**总计**：1 子目录 / 7 个 `.py` demo / 1 个 `KNOWLEDGE.md` / 1 个数据文件

> 文件名不带数字前缀（Python 模块名约束）；学习顺序在 `KNOWLEDGE.md` 与 docs `index.md` 中明确列出。

### 3.1 模型/Checkpoint 复用约定

- `gpt_train.py` 同时充当**主训练脚本**与**模型类的定义模块**：
  - 顶层导出 `class GPT`、`class GPTConfig`、`encode`、`decode`、`load_checkpoint`、`generate`
  - `if __name__ == "__main__":` 启动训练
- `sampling_strategies.py` / `attention_visualization.py` / `kv_cache.py` 通过 `from gpt_train import GPT, load_checkpoint, encode, decode` 复用
- Checkpoint 默认保存到 `data/ckpt.pt`，被 `.gitignore` 排除
- 如果 `data/ckpt.pt` 不存在，依赖脚本提示用户先跑 `python gpt_train.py`

---

## 4. 每个 demo 的实现规范

### 4.1 通用要求（沿用 Phase 1）

- 顶部 `╔══...══╗` ASCII box-art docstring（包含项目名 + 核心问题 + 与 LLM 的关联）
- 中文注释 + 中文 print 输出
- `if __name__ == "__main__": main()` 入口
- 优先 `random_state=42` / `torch.manual_seed(42)` 保证可复现
- 输出尾部"关键收获"3-5 条，呼应 docstring 的核心问题
- 失败优雅：缺包 → 友好提示，不崩
- 每个 demo 独立可跑，不依赖外部下载（除 04 训练保存 ckpt 供 05-07 复用）

### 4.2 数据集打包

`data/tiny_shakespeare.txt`：
- 来源：Andrej Karpathy nanoGPT 经典数据，公有领域
- 大小：~1.1MB
- 是否进 git：**是**（数据集本身就是莎士比亚著作，公有领域）
- 备选下载逻辑：如果文件不存在，从 `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt` 拉取（带超时 + 提示）

### 4.3 各 demo 详细规范

#### bpe_tokenizer.py（~250 行）

**目标**：从零实现 Byte-Pair Encoding，理解 GPT/LLaMA 的 tokenizer 怎么造出来的。

**步骤**：
1. 加载 Tiny Shakespeare（前 10K 字符做演示，全语料训练费时）
2. 初始化：每个字符作为一个 token
3. 迭代合并最高频字符对，重复 N=200 次
4. 展示前 30 次合并轨迹（"e ", "th", "the", ...）
5. 用训练出的 tokenizer 编码一个示例句子
6. 对比：直接字符级 vs BPE 的 token 数差异

**关键 print**：
- 词表大小（初始 65 → 合并后 265）
- 合并历史前 30 条
- 测试句子的字符级 / BPE 切分对比
- 压缩比

**LLM 关联**：与 GPT-4 的 cl100k_base（10万词表）对比，给出 ratio 直觉。

#### attention_from_scratch.py（~300 行）

**目标**：用 NumPy 和 PyTorch 双实现 self-attention，理解每一步矩阵运算。

**步骤**：
1. 输入：随机 5 个 token，每个 d=8 维 embedding
2. NumPy 版本：手算 Q = X@Wq, K = X@Wk, V = X@Wv → scores = QKᵀ/√d → softmax → @V
3. 因果 mask（causal mask）：上三角填 -inf，演示效果
4. 多头版本：拆 heads，每头独立算，最后 concat
5. PyTorch 版本：用 `nn.Linear` + `torch.matmul`
6. 用 `torch.nn.functional.scaled_dot_product_attention` 验证手写实现一致

**关键可视化**：
- 注意力矩阵（5×5）的 softmax 概率热度（ASCII 灰度）
- 因果 mask 应用前后对比

**LLM 关联**：解释为什么除以 √d（防止 softmax 进饱和区）；为什么多头（不同子空间）。

#### positional_encoding.py（~250 行）

**目标**：对比三种位置编码，看出 RoPE 为什么成现代主流。

**步骤**：
1. 实现三种 PE：
   - 绝对正余弦（原版 Transformer）
   - 学习式（GPT-2 / BERT）
   - RoPE（LLaMA / Qwen）
2. 各自对一个长度 32 的序列生成位置向量
3. 计算位置 i 和 j 的位置编码的内积，画 ASCII 热度图
4. 验证 RoPE 的"相对位置"性质：dot(rope(q,m), rope(k,n)) 只依赖 m-n

**关键洞察**：
- 绝对 PE 长度外推差
- 学习式 PE 完全无法外推
- RoPE 天然相对，外推性最佳（虽然也有上限）

**LLM 关联**：列举 Llama-3 用 RoPE + base=500K 来支持长上下文。

#### gpt_train.py（~500 行，最重）

**目标**：完整训练一个 ~3M 参数的 decoder-only Transformer，看 loss 下降，生成莎士比亚风文本。

**模型配置**（已校准目标参数量）：
```python
@dataclass
class GPTConfig:
    block_size: int = 128       # 上下文长度
    vocab_size: int = 1024      # BPE 后的词表
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192           # head_dim = 32
    dropout: float = 0.1
```
参数量估算（不含 lm_head 共享 embedding）：
- Embedding: 1024 × 192 ≈ 0.2M
- 每个 Block: 12 × 192² ≈ 0.44M（4× attention + 8× FFN）
- 6 个 Block: ~2.6M
- **总计：~2.9M 参数**

具体数字以代码 dry-run 为准；如训练时间偏离目标过多，调小到 n_layer=4。

**模块**：
- `class CausalSelfAttention(nn.Module)`：含因果 mask
- `class MLP(nn.Module)`：FFN 子层
- `class Block(nn.Module)`：LayerNorm → Attention → 残差 → LayerNorm → FFN → 残差
- `class GPT(nn.Module)`：embedding + 多个 Block + final LayerNorm + lm_head

**训练循环**：
- 优化器：AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
- 调度器：cosine 衰减 + warmup（前 100 step）
- 数据加载：`get_batch()` 随机切片
- 步数：~2000 step（CPU 5min / MPS 30s）
- 每 200 step 评估一次（train/val loss），打印
- loss 曲线 ASCII 简易绘图（终端友好）

**保存**：
- `data/ckpt.pt`：模型 state_dict + config + tokenizer 信息
- `data/loss_log.json`：用于 generate-time 复绘

**生成**：训练完成后立即生成一段 200 token 文本作为 sanity check

**关键 print**：
- 设备检测（CPU/MPS/CUDA）
- 模型参数量
- 每 200 步：step / lr / train loss / val loss
- 训练完后：生成示例 + ASCII loss 曲线

**导出 API**（供 05/06/07 import）：
```python
GPT, GPTConfig, encode, decode, load_checkpoint, generate
```

#### sampling_strategies.py（~250 行）

**目标**：从 ckpt 加载模型，用同一个 prompt + 随机种子，对比 4 种采样输出。

**实现**：
1. 加载 `gpt_train.py` 保存的 ckpt
2. 同一 prompt = `"ROMEO:"`
3. 4 种采样：
   - greedy（temperature=0）
   - temperature=0.8
   - top-k（k=40）
   - top-p（p=0.95）
4. 各生成 100 token
5. 显示 4 段对比 + 简短解读

**进阶**：
- 演示 temperature 极值：T=0.1（机械重复）vs T=2.0（噪音）
- 演示 repetition_penalty 效果

**关键 print**：每种策略的特点（"greedy 重复严重"、"top-p 多样性最佳"）。

#### attention_visualization.py（~250 行）

**目标**：加载 ckpt，对一个 prompt 画出每层每头的注意力分布。

**实现**：
1. 加载模型 + 拦截 attention weights（hook 或修改 forward）
2. 跑一个示例 prompt，收集每层每头的 (T, T) 注意力矩阵
3. 选 2-3 个有代表性的层 × 头组合
4. ASCII 灰度图打印（' '/'·'/'+'/'■' 四级）
5. 解读：浅层倾向"邻近 token"，深层倾向"语义 token"

**关键洞察**：让读者看到"注意力不是均匀的"，每个头学到了不同的功能。

#### kv_cache.py（~300 行）

**目标**：实现 KV cache，对比有无 KV cache 的推理速度。

**实现**：
1. 修改 `GPT.generate()` 接受 `use_cache` 参数
2. 没有 cache：每生成一个 token，重新 forward 整个序列
3. 有 cache：每层 attention 缓存 K/V，新 token 仅 append
4. 生成 200 token，分别计时
5. 对比 token/秒
6. 打印 cache 占用的内存大小

**关键 print**：
- 无 cache：~5 token/s
- 有 cache：~30 token/s（提升 6 倍左右）
- cache 内存：~MB 级，与序列长度成正比

**LLM 关联**：解释为什么生产 LLM 推理必须开 KV cache；LLaMA-70B 的 cache 大小为何能轻松到几 GB。

### 4.4 KNOWLEDGE.md 内容大纲（~900 行）

```
# Transformer 从零训练

1. 为什么训一个 Transformer
2. BPE Tokenization
   - 算法步骤、词表大小权衡、与 GPT-4 cl100k_base 对比
3. 自注意力机制
   - QKᵀ/√d 数学推导、为什么除以 √d
   - 多头注意力、因果 mask
   - 复杂度 O(T²d) 与长上下文挑战
4. 位置编码三流派
   - 绝对正余弦、学习式、RoPE
   - 长度外推性对比
5. Transformer 训练
   - AdamW、cosine warmup、梯度裁剪
   - LayerNorm 位置（pre-LN vs post-LN）
   - 参数量估算公式
6. 文本生成
   - 采样策略：greedy / temperature / top-k / top-p
   - repetition_penalty
   - beam search 为啥在 LLM 时代被弃用
7. 推理优化
   - KV cache 原理
   - 内存占用估算
   - 引子：FlashAttention / PagedAttention（vLLM）
8. 与现代 LLM 的差距
   - GQA / SwiGLU / RMSNorm
   - 训练数据规模、context length
9. 配套代码索引
```

### 4.5 模块名 / import 复用约定

文件名不带数字前缀（Python 模块名不能以数字开头）；学习顺序通过 `KNOWLEDGE.md` 与 docs 中显式列出。模块间复用：
```python
# sampling_strategies.py / attention_visualization.py / kv_cache.py
from gpt_train import GPT, GPTConfig, encode, decode, load_checkpoint
```

`gpt_train.py` 用 `if __name__ == "__main__":` 守卫训练入口，仅当直接执行时启动训练；被 import 时不执行。

---

## 5. docs 章节内容

### 5.1 目录结构

```
docs/ml-foundations/transformer-training/
├── index.md                # ~250 行，总览
├── tokenization.md         # ~600 行，BPE 深度
├── attention.md            # ~800 行，自注意力数学
├── positional-encoding.md  # ~500 行，PE 三流派
├── training.md             # ~700 行，训练循环
├── generation.md           # ~600 行，采样策略
└── inference.md            # ~600 行，KV cache + 注意力可视化
```

总计 ~4000 行 Markdown。

### 5.2 内容设计原则

- 每章配 LLM-tip 块说明对应到生产 LLM 的细节（用 VitePress `:::tip` 容器）
- 每章末尾链接到对应 demo
- 数学公式用 KaTeX
- ASCII 示意图配合说明
- 与 `docs/ml-foundations/` 的现有章节风格一致

### 5.3 侧边栏与导航

`docs/.vitepress/config.ts`：
```ts
{
  text: "零.5、Transformer 训练实战",
  collapsed: true,
  items: [
    { text: "本章导读", link: "/ml-foundations/transformer-training/" },
    { text: "BPE Tokenization", link: "/ml-foundations/transformer-training/tokenization" },
    { text: "自注意力机制", link: "/ml-foundations/transformer-training/attention" },
    { text: "位置编码", link: "/ml-foundations/transformer-training/positional-encoding" },
    { text: "完整训练流程", link: "/ml-foundations/transformer-training/training" },
    { text: "文本生成与采样", link: "/ml-foundations/transformer-training/generation" },
    { text: "推理优化与 KV Cache", link: "/ml-foundations/transformer-training/inference" },
  ],
}
```

插入在"零、ML 基础（前置补课）"之后、"一、LLM 基础"之前。

顶部 nav 不动（避免膨胀）。

---

## 6. 依赖

无新增依赖。Phase 1 已经包含：
- `numpy`、`torch`（核心）
- `matplotlib`、`seaborn`（如需绘图，但优先 ASCII）

不引入：
- `tiktoken`（已有，用于对比 cl100k_base）—— 已在 requirements.txt
- `transformers`（不需要）

---

## 7. README 改动

在"0. ML Foundations"小节后追加 Phase 2 描述：

```markdown
### 0.5 Transformer Training from Scratch

A self-contained walkthrough of building a small GPT-style model:
BPE tokenizer → attention from scratch → positional encoding (sin/learned/RoPE)
→ ~3M-parameter GPT training on Tiny Shakespeare → sampling strategies
(greedy/temp/top-k/top-p) → attention heatmap visualization → KV cache
inference optimization.

| Module | Directory | Core Concepts |
|--------|-----------|---------------|
| Transformer Training | `ml_foundations/transformer_training/` | 7 demos covering tokenization, attention, position encoding, training, generation, visualization, KV cache |

Main training (`gpt_train.py`) runs in **~5 min on Mac CPU / ~30 s on MPS**;
all other demos run in under 1 min.
```

项目结构图新增：
```
├── ml_foundations/
│   ├── classical/
│   ├── deep_learning/
│   ├── nlp_foundations/
│   └── transformer_training/    # NEW: BPE, attention, GPT-mini training, KV cache
```

---

## 8. .gitignore 改动

追加：
```
ml_foundations/transformer_training/data/ckpt.pt
ml_foundations/transformer_training/data/loss_log.json
ml_foundations/transformer_training/data/runs/
```

`tiny_shakespeare.txt` **不**被忽略（语料随仓库分发）。

---

## 9. 验收标准

代码层面：
- ✅ 所有 7 个 demo 在 Mac CPU 跑通，无 import / runtime 错误
- ✅ `gpt_train.py` 在 CPU < 6 分钟、MPS < 1 分钟跑完
- ✅ `gpt_train.py` 训练后能生成出"看起来像英文"的文本（不是乱码）
- ✅ `kv_cache.py` 显示 KV cache 加速 ≥ 3×
- ✅ 所有 `.py` 通过 `python -m py_compile`
- ✅ 无新增依赖，仅复用 Phase 1 已加入的 numpy/torch

文档层面：
- ✅ docs 子章节 7 篇完成
- ✅ VitePress 侧边栏正确插入
- ✅ README 添加 Phase 2 描述
- ✅ 所有内部链接指向有效文件

工程层面：
- ✅ 单次最终 commit，message 形如 `feat(ml): add transformer_training module + docs`
- ✅ 不破坏现有 Phase 1 任何文件

---

## 10. 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 5M 参数 5 分钟 CPU 训练效果可能仍是 garbage | 实测后调小到 3M / 缩短上下文，或在 docs 中说明"这是 nanoGPT 体量，效果弱但能看出趋势" |
| Mac MPS 在 attention 上偶有 NaN | 默认 fp32；MPS 不稳就 fallback CPU |
| RoPE 实现踩坑（complex / paired-real 表示） | 用 paired-real 写法，照搬 LLaMA 代码风格，注释充分 |
| KV cache 实现复杂度高 | 提供两段实现：朴素全重算 vs 增量 cache，用 contextmanager 切换 |
| Tiny Shakespeare 1MB 进 git 让仓库膨胀 | 1MB 可接受；如有顾虑改为首次运行下载 |
| 文件命名不带数字会让"学习顺序"不清晰 | 在 KNOWLEDGE.md 和 docs/index.md 里明确列序 |

---

## 11. 工作量估算

- 7 个 demo × 平均 300 行 ≈ 2100 行 Python
- 1 个 KNOWLEDGE.md ≈ 900 行 Markdown
- 7 个 docs/.../*.md × 平均 600 行 ≈ 4200 行 Markdown
- 配置/README/sidebar 改动：小

总计约 **7200 行新增内容**。比 Phase 1 (~9600) 略小，但代码密度更高。

---

## 12. 后续计划（不在本次范围）

完成阶段 2 后：

1. 用户实际跑一遍 `gpt_train.py` + 后续 demo，反馈训练时长与生成质量
2. 观察 docs 站构建是否正常（`pnpm dev`）
3. 进入阶段 3 设计：LLM 微调实战（独立 spec），主题：HF transformers 加载预训练模型 + LoRA + DPO + 量化

---

## 13. 实施 phase 划分（待 writing-plans 细化）

1. **Phase A**：代码骨架 + 数据
   - 拷入 tiny_shakespeare.txt
   - bpe_tokenizer.py
   - attention_from_scratch.py
   - positional_encoding.py
2. **Phase B**：主训练 demo
   - gpt_train.py（含 GPT 类、训练循环、保存 ckpt）
3. **Phase C**：依赖 ckpt 的 demo
   - sampling_strategies.py
   - attention_visualization.py
   - kv_cache.py
4. **Phase D**：知识文档
   - KNOWLEDGE.md
   - 7 个 VitePress md
5. **Phase E**：集成
   - sidebar / README / .gitignore
6. **Phase F**：验证
   - 跑通所有 demo（含 MPS）
   - py_compile 全检查
7. **Phase G**：单次 commit
