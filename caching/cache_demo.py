"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                 项目 19: LLM 响应缓存 (Response Caching)                     ║
║                                                                              ║
║  "同样的问题问100遍，难道要付100遍的钱？"                                      ║
║                                                                              ║
║  本文件从零到一，手把手带你理解 LLM 缓存的核心原理与实战应用                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎓 前置科学: 为什么 LLM 需要缓存？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

想象一个客服场景:

  用户A: "你们的退货政策是什么？"     ──→ API调用 ──→ 花费 0.01元 + 2秒延迟
  用户B: "你们的退货政策是什么？"     ──→ API调用 ──→ 花费 0.01元 + 2秒延迟
  用户C: "你们的退货政策是什么？"     ──→ API调用 ──→ 花费 0.01元 + 2秒延迟
  ...
  用户100: "你们的退货政策是什么？"   ──→ API调用 ──→ 花费 0.01元 + 2秒延迟

  总计: 100次调用 = 1.00元 + 200秒总延迟

加了缓存之后:

  用户A: "你们的退货政策是什么？"     ──→ API调用 ──→ 花费 0.01元 + 2秒延迟
  用户B: "你们的退货政策是什么？"     ──→ 缓存命中 ──→ 花费 0元 + 0.001秒
  用户C: "你们的退货政策是什么？"     ──→ 缓存命中 ──→ 花费 0元 + 0.001秒
  ...
  用户100: "你们的退货政策是什么？"   ──→ 缓存命中 ──→ 花费 0元 + 0.001秒

  总计: 1次调用 + 99次缓存 = 0.01元 + ~2.1秒总延迟
  节省: 99% 成本 + 99% 延迟！

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

缓存的工作原理 (key-value 存储):

    ┌─────────────────────────────────────────────────────────────┐
    │                      Cache Store                             │
    │                                                             │
    │   Key (问题的哈希/向量)         Value (LLM的回答)            │
    │   ─────────────────────        ─────────────────            │
    │   hash("退货政策是什么")  ──→   "我们的退货政策是..."         │
    │   hash("怎么注册账号")    ──→   "注册步骤如下..."            │
    │   hash("价格多少")        ──→   "我们的定价方案..."          │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘

    查询流程:
    ┌────────┐    ┌───────────┐    ┌─────────┐
    │ 用户问题 │──→│ 查缓存     │──→│ 命中？   │
    └────────┘    └───────────┘    └─────────┘
                                      │    │
                                    Yes    No
                                      │    │
                                      ▼    ▼
                              ┌──────────┐  ┌──────────┐
                              │ 返回缓存  │  │ 调用LLM  │
                              │ (极快)    │  │ + 存缓存  │
                              └──────────┘  └──────────┘

三种缓存类型对比:

    ┌──────────────┬──────────────┬──────────────────────────────┐
    │ InMemoryCache│ SQLiteCache  │ SemanticCache                │
    ├──────────────┼──────────────┼──────────────────────────────┤
    │ 存在内存里    │ 存在磁盘文件 │ 用向量相似度匹配              │
    │ 重启就丢失    │ 重启不丢失   │ 语义相近也能命中              │
    │ 速度最快      │ 速度快       │ 速度稍慢(要算向量)            │
    │ 精确匹配      │ 精确匹配     │ 模糊匹配                     │
    └──────────────┴──────────────┴──────────────────────────────┘
"""

import os
import time
import numpy as np

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 环境准备: 确保缓存目录存在
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# mkdir -p 等效操作: 如果目录不存在就创建，存在也不报错
os.makedirs('./caching', exist_ok=True)
print('=' * 70)
print('项目 19: LLM 响应缓存 (Response Caching)')
print('=' * 70)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chapter 0: 初始化 + 为什么需要缓存                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print('\n')
print('━' * 70)
print('Chapter 0: 初始化 + 为什么需要缓存')
print('━' * 70)

# ──────────────────────────────────────────────────────────────────────────
# Step 0.1: API 配置
# ──────────────────────────────────────────────────────────────────────────
# 这里我们使用统一的 API 网关配置
# 所有项目共享同一套配置，方便管理

API_KEY = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY'
BASE_URL = 'https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1'
MODEL_NAME = 'kivy-kimi-k2_5'

print(f'[配置] API Base URL: {BASE_URL}')
print(f'[配置] Model: {MODEL_NAME}')

# ──────────────────────────────────────────────────────────────────────────
# Step 0.2: 初始化 LLM
# ──────────────────────────────────────────────────────────────────────────
# 使用 LangChain 的 ChatOpenAI 作为我们的 LLM 引擎
# temperature=0 确保相同问题得到相同回答（对缓存友好！）

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    temperature=0,  # 温度为0: 同一问题总是得到相同答案，缓存效果最佳
)

print('[初始化] ChatOpenAI 实例创建完成')
print('[初始化] temperature=0 (确定性输出，缓存友好)')

# ──────────────────────────────────────────────────────────────────────────
# Step 0.3: 成本对比演算
# ──────────────────────────────────────────────────────────────────────────
# 让我们用数字说话: 缓存能省多少钱？

print('\n--- 成本对比演算 ---')
print()
print('假设场景: 客服机器人，每天收到 1000 个问题，其中 70% 是重复问题')
print()

daily_questions = 1000
repeat_rate = 0.70
cost_per_call = 0.01  # 假设每次 API 调用 0.01 元
latency_per_call = 2.0  # 假设每次 API 调用 2 秒

# 无缓存
no_cache_cost = daily_questions * cost_per_call
no_cache_time = daily_questions * latency_per_call
print(f'[无缓存] 每日成本: {daily_questions} 次 x {cost_per_call}元 = {no_cache_cost:.2f}元')
print(f'[无缓存] 每日总延迟: {daily_questions} 次 x {latency_per_call}秒 = {no_cache_time:.0f}秒')

# 有缓存
unique_questions = daily_questions * (1 - repeat_rate)
cached_questions = daily_questions * repeat_rate
with_cache_cost = unique_questions * cost_per_call
with_cache_time = unique_questions * latency_per_call + cached_questions * 0.001
print()
print(f'[有缓存] 唯一问题: {unique_questions:.0f} 次 API 调用')
print(f'[有缓存] 重复问题: {cached_questions:.0f} 次缓存命中 (接近零成本)')
print(f'[有缓存] 每日成本: {unique_questions:.0f} x {cost_per_call}元 = {with_cache_cost:.2f}元')
print(f'[有缓存] 每日总延迟: {with_cache_time:.1f}秒')

savings_pct = (1 - with_cache_cost / no_cache_cost) * 100
print()
print(f'[节省] 成本节省: {savings_pct:.0f}%  ({no_cache_cost:.2f}元 → {with_cache_cost:.2f}元)')
print(f'[节省] 这就是缓存的威力!')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chapter 1: InMemoryCache (内存缓存)                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print('\n\n')
print('━' * 70)
print('Chapter 1: InMemoryCache (内存缓存)')
print('━' * 70)

# ──────────────────────────────────────────────────────────────────────────
# 什么是 InMemoryCache？
# ──────────────────────────────────────────────────────────────────────────
#
# InMemoryCache 就是一个 Python 字典:
#
#   cache = {}
#   cache[hash(问题)] = 答案
#
# 优点: 极快 (纳秒级)
# 缺点: 程序重启就没了 (内存是易失性存储)
#
# 适用场景:
#   - 开发调试 (避免重复调用浪费钱)
#   - 短期运行的脚本
#   - 不需要持久化的场景
# ──────────────────────────────────────────────────────────────────────────

print('\n--- Step 1.1: 设置 InMemoryCache ---')

from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

# set_llm_cache() 是全局设置:
# 一旦设置，所有 LangChain LLM 调用都会自动走缓存
set_llm_cache(InMemoryCache())
print('[缓存] InMemoryCache 已设置为全局缓存')
print('[缓存] 底层就是一个 Python dict，存在进程内存里')

# ──────────────────────────────────────────────────────────────────────────
# Step 1.2: 第一次调用 (Cache Miss - 缓存未命中)
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 1.2: 第一次调用 (Cold Call) ---')
print('[说明] 第一次问这个问题，缓存里没有，必须真正调用 API')

test_question = '用一句话解释什么是缓存？'
print(f'[问题] {test_question}')

# 计时: 记录 API 调用耗时
start_time = time.time()
response_1 = llm.invoke(test_question)
elapsed_1 = time.time() - start_time

print(f'[回答] {response_1.content}')
print(f'[耗时] {elapsed_1:.3f} 秒 (真正的 API 网络往返)')
print(f'[状态] Cache MISS - 答案已被缓存起来，下次就不用再调 API 了')

# ──────────────────────────────────────────────────────────────────────────
# Step 1.3: 第二次调用 (Cache Hit - 缓存命中!)
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 1.3: 第二次调用 (Cache Hit!) ---')
print('[说明] 同样的问题第二次问，这次应该直接从缓存返回')

start_time = time.time()
response_2 = llm.invoke(test_question)
elapsed_2 = time.time() - start_time

print(f'[回答] {response_2.content}')
print(f'[耗时] {elapsed_2:.6f} 秒 (从内存缓存直接返回)')
print(f'[状态] Cache HIT - 几乎是瞬时的!')

# ──────────────────────────────────────────────────────────────────────────
# Step 1.4: 速度对比
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 1.4: 速度对比 ---')

if elapsed_2 > 0:
    speedup = elapsed_1 / elapsed_2
    print(f'[对比] 第一次 (API调用): {elapsed_1:.3f} 秒')
    print(f'[对比] 第二次 (缓存命中): {elapsed_2:.6f} 秒')
    print(f'[对比] 加速比: {speedup:.0f}x 倍!')
else:
    print(f'[对比] 第一次 (API调用): {elapsed_1:.3f} 秒')
    print(f'[对比] 第二次 (缓存命中): 接近 0 秒 (太快了无法测量)')

# ──────────────────────────────────────────────────────────────────────────
# Step 1.5: 验证 - 不同问题不会命中缓存
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 1.5: 不同问题 = 缓存未命中 ---')
print('[说明] InMemoryCache 是精确匹配，问题文字必须完全一样才能命中')

different_question = '用一句话解释什么是数据库？'
print(f'[问题] {different_question}')

start_time = time.time()
response_3 = llm.invoke(different_question)
elapsed_3 = time.time() - start_time

print(f'[回答] {response_3.content}')
print(f'[耗时] {elapsed_3:.3f} 秒 (新问题，必须调用 API)')
print(f'[状态] Cache MISS - 问题不同，无法命中缓存')
print()
print('[重要] InMemoryCache 的局限性:')
print('  - "什么是缓存" 和 "缓存是什么" 是两个不同的 key')
print('  - 即使语义完全相同，文字不同就无法命中')
print('  - 这个问题在 Chapter 3 的语义缓存中解决!')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chapter 2: SQLiteCache (持久化缓存)                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print('\n\n')
print('━' * 70)
print('Chapter 2: SQLiteCache (持久化缓存)')
print('━' * 70)

# ──────────────────────────────────────────────────────────────────────────
# 什么是 SQLiteCache？
# ──────────────────────────────────────────────────────────────────────────
#
# SQLiteCache 把缓存存到 SQLite 数据库文件里:
#
#   ┌──────────────────────────────────────────┐
#   │  .cache.db (SQLite 文件)                  │
#   │                                          │
#   │  TABLE full_llm_cache:                   │
#   │  ┌─────────┬──────────┬────────────┐    │
#   │  │ prompt  │ llm_str  │ response   │    │
#   │  ├─────────┼──────────┼────────────┤    │
#   │  │ "退货"  │ "gpt-4"  │ "退货需..." │    │
#   │  │ "注册"  │ "gpt-4"  │ "步骤..." │     │
#   │  └─────────┴──────────┴────────────┘    │
#   └──────────────────────────────────────────┘
#
# 优点: 程序重启后缓存仍在！
# 缺点: 比内存稍慢 (磁盘IO)，但仍远快于 API 调用
#
# 适用场景:
#   - 长期运行的服务
#   - 需要在重启后保留缓存的场景
#   - 单机部署
# ──────────────────────────────────────────────────────────────────────────

print('\n--- Step 2.1: 设置 SQLiteCache ---')

from langchain_community.cache import SQLiteCache

# 数据库文件路径 - 放在 caching 子目录下
CACHE_DB_PATH = './caching/.cache.db'
print(f'[路径] 缓存数据库: {CACHE_DB_PATH}')

# 切换全局缓存为 SQLiteCache
set_llm_cache(SQLiteCache(database_path=CACHE_DB_PATH))
print('[缓存] SQLiteCache 已设置为全局缓存')
print('[缓存] 底层是 SQLite 数据库文件，数据持久化到磁盘')

# ──────────────────────────────────────────────────────────────────────────
# Step 2.2: 写入缓存 (第一次调用)
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 2.2: 第一次调用 - 写入 SQLite 缓存 ---')

sqlite_question = '什么是 SQLite 数据库？请用一句话回答。'
print(f'[问题] {sqlite_question}')

start_time = time.time()
response_sqlite_1 = llm.invoke(sqlite_question)
elapsed_sqlite_1 = time.time() - start_time

print(f'[回答] {response_sqlite_1.content}')
print(f'[耗时] {elapsed_sqlite_1:.3f} 秒 (API调用 + 写入SQLite)')
print(f'[状态] Cache MISS → 答案已写入 {CACHE_DB_PATH}')

# 验证文件确实创建了
if os.path.exists(CACHE_DB_PATH):
    file_size = os.path.getsize(CACHE_DB_PATH)
    print(f'[验证] 数据库文件已创建，大小: {file_size} bytes')
else:
    print('[警告] 数据库文件未找到，可能路径有问题')

# ──────────────────────────────────────────────────────────────────────────
# Step 2.3: 从 SQLite 缓存读取 (第二次调用)
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 2.3: 第二次调用 - 从 SQLite 读取 ---')

start_time = time.time()
response_sqlite_2 = llm.invoke(sqlite_question)
elapsed_sqlite_2 = time.time() - start_time

print(f'[回答] {response_sqlite_2.content}')
print(f'[耗时] {elapsed_sqlite_2:.6f} 秒 (从SQLite读取)')
print(f'[状态] Cache HIT - 从磁盘文件读取，依然极快!')

if elapsed_sqlite_2 > 0:
    speedup_sqlite = elapsed_sqlite_1 / elapsed_sqlite_2
    print(f'[对比] 加速比: {speedup_sqlite:.0f}x 倍!')

# ──────────────────────────────────────────────────────────────────────────
# Step 2.4: 模拟重启 - SQLite 的持久化优势
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 2.4: 模拟"重启" - 验证持久化 ---')
print('[说明] 我们重新创建一个 SQLiteCache 实例 (模拟程序重启)')
print('[说明] InMemoryCache 重启后就没了，但 SQLiteCache 还在!')

# 重新设置 SQLiteCache (模拟重启)
set_llm_cache(SQLiteCache(database_path=CACHE_DB_PATH))
print('[操作] 重新创建 SQLiteCache 实例 (连接同一个 .db 文件)')

start_time = time.time()
response_sqlite_3 = llm.invoke(sqlite_question)
elapsed_sqlite_3 = time.time() - start_time

print(f'[回答] {response_sqlite_3.content}')
print(f'[耗时] {elapsed_sqlite_3:.6f} 秒')
print(f'[结论] "重启"后缓存依然有效! 这就是持久化缓存的价值!')

# ──────────────────────────────────────────────────────────────────────────
# Step 2.5: 清理 + 切回 InMemoryCache
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 2.5: 清理 SQLite 缓存文件 ---')

# 重要: 先切换回 InMemoryCache，释放 SQLite 文件锁
# 否则后续章节可能遇到文件锁问题
set_llm_cache(InMemoryCache())
print('[操作] 已切回 InMemoryCache (释放 SQLite 文件锁)')

# 删除数据库文件
if os.path.exists(CACHE_DB_PATH):
    os.remove(CACHE_DB_PATH)
    print(f'[清理] 已删除 {CACHE_DB_PATH}')
else:
    print(f'[清理] {CACHE_DB_PATH} 不存在，无需删除')

print('[完成] SQLiteCache 演示结束，环境已清理')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chapter 3: 自定义语义缓存 (Semantic Cache)                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print('\n\n')
print('━' * 70)
print('Chapter 3: 自定义语义缓存 (Semantic Cache)')
print('━' * 70)

# ──────────────────────────────────────────────────────────────────────────
# 什么是语义缓存？
# ──────────────────────────────────────────────────────────────────────────
#
# 传统缓存 (精确匹配):
#   "今天天气怎样？" → 命中
#   "今天天气如何？" → 未命中 ❌ (文字不同！)
#
# 语义缓存 (相似度匹配):
#   "今天天气怎样？" → 命中
#   "今天天气如何？" → 命中 ✓ (语义相似！)
#
# 核心思想:
#   1. 把问题转成向量 (Embedding)
#   2. 新问题也转成向量
#   3. 计算两个向量的余弦相似度
#   4. 相似度 > 阈值 → 视为同一个问题 → 返回缓存的答案
#
#   ┌─────────────────────────────────────────────────────────────────┐
#   │                   Semantic Cache 工作流程                        │
#   │                                                                 │
#   │  "今天天气如何？"                                                │
#   │       │                                                         │
#   │       ▼                                                         │
#   │  [Embedding Model] ──→ [0.12, 0.87, -0.34, ...]  (向量)        │
#   │       │                                                         │
#   │       ▼                                                         │
#   │  [FAISS 向量搜索] ──→ 找到最相似的已缓存问题                     │
#   │       │                                                         │
#   │       ▼                                                         │
#   │  相似度 = 0.95 > 阈值 0.85 ──→ 命中! 返回缓存答案               │
#   └─────────────────────────────────────────────────────────────────┘
# ──────────────────────────────────────────────────────────────────────────

print('\n--- Step 3.1: 加载 Embedding 模型 ---')
print('[说明] 我们用 BAAI/bge-small-zh-v1.5 中文向量模型')
print('[说明] 这个模型把文本转成 512 维向量，专门为中文优化')

from langchain_huggingface import HuggingFaceEmbeddings

# 加载中文 embedding 模型
# bge-small-zh-v1.5: 轻量级中文向量模型，效果好速度快
embedding_model = HuggingFaceEmbeddings(
    model_name='BAAI/bge-small-zh-v1.5',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},  # 归一化: 方便计算余弦相似度
)
print('[加载] Embedding 模型加载完成: BAAI/bge-small-zh-v1.5')

# 验证向量维度
test_vec = embedding_model.embed_query('测试')
print(f'[验证] 向量维度: {len(test_vec)}')

# ──────────────────────────────────────────────────────────────────────────
# Step 3.2: 实现 SemanticCache 类
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 3.2: 实现 SemanticCache 类 ---')


class SemanticCache:
    """
    语义缓存: 基于向量相似度的智能缓存

    工作原理:
    1. put(question, answer):
       - 将 question 转为向量
       - 存入 FAISS 索引 + 对应答案

    2. get(question, threshold=0.85):
       - 将 question 转为向量
       - 在 FAISS 中搜索最相似的已缓存问题
       - 如果相似度 > threshold，返回缓存答案
       - 否则返回 None (缓存未命中)
    """

    def __init__(self, embedding_model, threshold=0.85):
        """
        初始化语义缓存

        Args:
            embedding_model: 用于将文本转为向量的模型
            threshold: 相似度阈值，大于此值视为命中 (0~1)
        """
        import faiss

        self.embedding_model = embedding_model
        self.threshold = threshold

        # 获取向量维度
        sample_vec = embedding_model.embed_query('init')
        self.dimension = len(sample_vec)

        # FAISS 索引: 使用 Inner Product (因为向量已归一化，IP = 余弦相似度)
        # IndexFlatIP: 暴力搜索，适合小规模数据
        self.index = faiss.IndexFlatIP(self.dimension)

        # 存储答案的列表 (与 FAISS 索引中的向量一一对应)
        self.answers = []
        # 存储原始问题 (用于展示命中了哪个缓存条目)
        self.questions = []

        print(f'  [SemanticCache] 初始化完成')
        print(f'  [SemanticCache] 向量维度: {self.dimension}')
        print(f'  [SemanticCache] 相似度阈值: {self.threshold}')

    def put(self, question: str, answer: str):
        """
        存入缓存: question → 向量 → FAISS 索引, answer → 答案列表

        Args:
            question: 问题文本
            answer: 对应的答案
        """
        # 将问题转为向量
        vec = self.embedding_model.embed_query(question)
        vec_array = np.array([vec], dtype=np.float32)

        # 添加到 FAISS 索引
        self.index.add(vec_array)

        # 存储答案和原始问题
        self.answers.append(answer)
        self.questions.append(question)

        print(f'  [PUT] 已缓存: "{question}" (当前缓存条目数: {self.index.ntotal})')

    def get(self, question: str, threshold: float = None) -> dict:
        """
        查询缓存: question → 向量 → FAISS 搜索 → 判断是否命中

        Args:
            question: 查询问题
            threshold: 可选，覆盖默认阈值

        Returns:
            dict: {
                'hit': bool,           # 是否命中
                'answer': str or None, # 命中时返回答案
                'similarity': float,   # 最高相似度
                'matched_question': str or None  # 命中的原始问题
            }
        """
        if threshold is None:
            threshold = self.threshold

        # 如果缓存为空，直接返回未命中
        if self.index.ntotal == 0:
            return {
                'hit': False,
                'answer': None,
                'similarity': 0.0,
                'matched_question': None,
            }

        # 将查询问题转为向量
        vec = self.embedding_model.embed_query(question)
        vec_array = np.array([vec], dtype=np.float32)

        # FAISS 搜索: 找最相似的 1 个向量
        # distances: 相似度分数 (因为用的是 IP，已归一化后就是余弦相似度)
        # indices: 对应的索引位置
        distances, indices = self.index.search(vec_array, 1)

        similarity = float(distances[0][0])
        best_idx = int(indices[0][0])

        if similarity >= threshold:
            return {
                'hit': True,
                'answer': self.answers[best_idx],
                'similarity': similarity,
                'matched_question': self.questions[best_idx],
            }
        else:
            return {
                'hit': False,
                'answer': None,
                'similarity': similarity,
                'matched_question': self.questions[best_idx] if best_idx >= 0 else None,
            }


print('[完成] SemanticCache 类定义完毕')

# ──────────────────────────────────────────────────────────────────────────
# Step 3.3: 创建语义缓存实例并填充数据
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 3.3: 创建语义缓存并填充示例数据 ---')

# 创建语义缓存实例
semantic_cache = SemanticCache(
    embedding_model=embedding_model,
    threshold=0.85,
)

# 模拟: 用 LLM 回答几个问题并缓存
# (实际场景中，这些答案是 LLM 生成的，这里直接用预设答案演示原理)
print()
print('[填充] 向语义缓存中存入示例数据...')

cache_entries = [
    ('今天天气怎样', '今天天气晴朗，气温适宜，适合外出活动。'),
    ('Python 是什么编程语言', 'Python 是一种高级解释型编程语言，以简洁易读著称。'),
    ('如何学习机器学习', '学习机器学习建议从数学基础开始，然后学习经典算法，最后做实战项目。'),
]

for question, answer in cache_entries:
    semantic_cache.put(question, answer)

print(f'\n[完成] 缓存中共 {semantic_cache.index.ntotal} 个条目')

# ──────────────────────────────────────────────────────────────────────────
# Step 3.4: 演示三种情况
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 3.4: 演示语义缓存的三种情况 ---')

# 情况1: 精确命中 (完全相同的问题)
print('\n[情况1] 精确命中 - 问题完全相同')
print('-' * 50)
query_1 = '今天天气怎样'
print(f'  查询: "{query_1}"')
result_1 = semantic_cache.get(query_1)
print(f'  命中: {result_1["hit"]}')
print(f'  相似度: {result_1["similarity"]:.4f}')
print(f'  匹配到: "{result_1["matched_question"]}"')
print(f'  答案: {result_1["answer"]}')

# 情况2: 语义命中 (问题不同，但意思相近)
print('\n[情况2] 语义命中 - 问题不同但意思相近')
print('-' * 50)
query_2 = '今天天气如何'
print(f'  查询: "{query_2}"')
print(f'  (注意: 缓存中存的是"今天天气怎样"，不是"今天天气如何")')
result_2 = semantic_cache.get(query_2)
print(f'  命中: {result_2["hit"]}')
print(f'  相似度: {result_2["similarity"]:.4f}')
print(f'  匹配到: "{result_2["matched_question"]}"')
if result_2['hit']:
    print(f'  答案: {result_2["answer"]}')
    print(f'  [精彩] 虽然文字不同，但语义相似度 {result_2["similarity"]:.4f} > 阈值 0.85，命中!')
else:
    print(f'  [说明] 相似度 {result_2["similarity"]:.4f} 未达到阈值 0.85')

# 再试一个语义相近的
print()
query_2b = '怎样学好机器学习'
print(f'  查询: "{query_2b}"')
print(f'  (缓存中存的是"如何学习机器学习")')
result_2b = semantic_cache.get(query_2b)
print(f'  命中: {result_2b["hit"]}')
print(f'  相似度: {result_2b["similarity"]:.4f}')
print(f'  匹配到: "{result_2b["matched_question"]}"')
if result_2b['hit']:
    print(f'  答案: {result_2b["answer"]}')

# 情况3: 未命中 (问题完全不相关)
print('\n[情况3] 未命中 - 问题完全不相关')
print('-' * 50)
query_3 = '量子计算机的工作原理是什么'
print(f'  查询: "{query_3}"')
result_3 = semantic_cache.get(query_3)
print(f'  命中: {result_3["hit"]}')
print(f'  相似度: {result_3["similarity"]:.4f} (低于阈值 0.85)')
print(f'  最接近的: "{result_3["matched_question"]}"')
print(f'  [结论] 相似度太低，缓存未命中，需要调用 LLM 获取答案')

# ──────────────────────────────────────────────────────────────────────────
# Step 3.5: 完整的带 LLM 的语义缓存使用流程
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 3.5: 完整工作流 (SemanticCache + LLM) ---')


def ask_with_semantic_cache(question: str, cache: SemanticCache, llm_instance) -> str:
    """
    带语义缓存的问答函数:
    1. 先查缓存
    2. 命中 → 直接返回
    3. 未命中 → 调用 LLM → 存入缓存 → 返回
    """
    print(f'\n  [查询] "{question}"')

    # Step 1: 查缓存
    start_time = time.time()
    cache_result = cache.get(question)
    cache_check_time = time.time() - start_time

    if cache_result['hit']:
        print(f'  [缓存] HIT! 相似度={cache_result["similarity"]:.4f}')
        print(f'  [缓存] 匹配到: "{cache_result["matched_question"]}"')
        print(f'  [耗时] {cache_check_time:.4f} 秒 (仅向量搜索)')
        return cache_result['answer']
    else:
        print(f'  [缓存] MISS (最高相似度={cache_result["similarity"]:.4f})')
        # Step 2: 调用 LLM
        start_time = time.time()
        response = llm_instance.invoke(question)
        llm_time = time.time() - start_time
        answer = response.content
        print(f'  [LLM] 调用完成，耗时 {llm_time:.3f} 秒')

        # Step 3: 存入缓存
        cache.put(question, answer)
        return answer


# 演示完整流程
print('[演示] 问一个缓存中没有的新问题:')
answer = ask_with_semantic_cache(
    '什么是深度学习？请简要说明。',
    semantic_cache,
    llm,
)
print(f'  [答案] {answer}')

print('\n[演示] 用语义相近的方式再问一次:')
answer = ask_with_semantic_cache(
    '请简单介绍一下深度学习',
    semantic_cache,
    llm,
)
print(f'  [答案] {answer}')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chapter 4: 缓存策略选型 + 生产实践                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print('\n\n')
print('━' * 70)
print('Chapter 4: 缓存策略选型 + 生产实践')
print('━' * 70)

# ──────────────────────────────────────────────────────────────────────────
# Step 4.1: 缓存方案对比表
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 4.1: 缓存方案对比表 ---')
print()
print('┌──────────────┬────────────┬────────────┬────────────┬──────────────┐')
print('│ 特性         │ InMemory   │ SQLite     │ Redis      │ Semantic     │')
print('├──────────────┼────────────┼────────────┼────────────┼──────────────┤')
print('│ 持久化       │ ❌ 否      │ ✅ 是      │ ✅ 是      │ 看底层存储   │')
print('│ 速度         │ 极快(ns)   │ 快(ms)     │ 快(ms)     │ 稍慢(10ms+) │')
print('│ 匹配方式     │ 精确匹配   │ 精确匹配   │ 精确匹配   │ 语义相似度   │')
print('│ 分布式       │ ❌ 否      │ ❌ 否      │ ✅ 是      │ 看底层存储   │')
print('│ 命中率       │ 低         │ 低         │ 低         │ 高           │')
print('│ 实现复杂度   │ 极低       │ 低         │ 中         │ 高           │')
print('│ 适用场景     │ 开发调试   │ 单机服务   │ 分布式服务 │ 智能问答     │')
print('└──────────────┴────────────┴────────────┴────────────┴──────────────┘')

# ──────────────────────────────────────────────────────────────────────────
# Step 4.2: 缓存失效策略
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 4.2: 缓存失效策略 ---')
print()
print('缓存不能永远不变! 以下是常见的失效策略:')
print()
print('1. TTL (Time To Live) - 过期时间')
print('   cache.set(key, value, ttl=3600)  # 1小时后过期')
print('   适用: 天气、新闻等时效性内容')
print()
print('2. LRU (Least Recently Used) - 最近最少使用')
print('   缓存满了时，淘汰最久没被访问的条目')
print('   适用: 内存有限的场景')
print()
print('3. 手动清除')
print('   当知识库更新时，主动清除相关缓存')
print('   适用: RAG 系统知识库更新后')
print()
print('4. 版本控制')
print('   key = f"{model_version}:{prompt_hash}"')
print('   模型升级时，旧缓存自动失效')

# ──────────────────────────────────────────────────────────────────────────
# Step 4.3: Token 节省计算
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 4.3: Token 节省计算示例 ---')
print()

# 模拟计算
daily_requests = 10000
cache_hit_rate = 0.65  # 65% 命中率
avg_input_tokens = 100
avg_output_tokens = 500
cost_per_1k_tokens = 0.002  # 假设 $0.002 / 1K tokens

total_tokens_no_cache = daily_requests * (avg_input_tokens + avg_output_tokens)
total_tokens_with_cache = daily_requests * (1 - cache_hit_rate) * (avg_input_tokens + avg_output_tokens)
tokens_saved = total_tokens_no_cache - total_tokens_with_cache

cost_no_cache = total_tokens_no_cache / 1000 * cost_per_1k_tokens
cost_with_cache = total_tokens_with_cache / 1000 * cost_per_1k_tokens
cost_saved = cost_no_cache - cost_with_cache

print(f'场景: 日请求量 {daily_requests:,}, 缓存命中率 {cache_hit_rate*100:.0f}%')
print(f'平均每次: 输入 {avg_input_tokens} tokens + 输出 {avg_output_tokens} tokens')
print()
print(f'[无缓存] 日消耗 tokens: {total_tokens_no_cache:,}')
print(f'[无缓存] 日成本: ${cost_no_cache:.2f}')
print()
print(f'[有缓存] 日消耗 tokens: {total_tokens_with_cache:,.0f}')
print(f'[有缓存] 日成本: ${cost_with_cache:.2f}')
print()
print(f'[节省] 每日节省 tokens: {tokens_saved:,.0f}')
print(f'[节省] 每日节省成本: ${cost_saved:.2f}')
print(f'[节省] 每月节省成本: ${cost_saved * 30:.2f}')

# ──────────────────────────────────────────────────────────────────────────
# Step 4.4: 什么时候不该缓存
# ──────────────────────────────────────────────────────────────────────────
print('\n--- Step 4.4: 什么时候不该用缓存 ---')
print()
print('以下场景不适合缓存 (或需要非常短的 TTL):')
print()
print('1. 创意生成类任务')
print('   - "帮我写一首诗" → 每次应该不同!')
print('   - 如果缓存了，用户每次得到一样的诗，体验很差')
print()
print('2. 实时数据查询')
print('   - "现在的股票价格是多少" → 数据实时变化')
print('   - 缓存会返回过时信息')
print()
print('3. 个性化回答')
print('   - 同一问题对不同用户应该有不同回答')
print('   - 需要把用户 ID 也加入缓存 key')
print()
print('4. 多轮对话 (依赖上下文)')
print('   - "继续" → 含义取决于之前的对话')
print('   - 缓存 key 必须包含完整上下文')
print()
print('5. temperature > 0 的场景')
print('   - 高温度意味着希望多样性')
print('   - 缓存与多样性目标矛盾')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary: 总结对比
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n\n')
print('━' * 70)
print('SUMMARY: LLM 缓存全景总结')
print('━' * 70)
print()
print('本项目覆盖的缓存方案:')
print()
print('┌─────┬───────────────┬──────────────────────────────────────────────┐')
print('│ Ch  │ 方案           │ 核心特点                                     │')
print('├─────┼───────────────┼──────────────────────────────────────────────┤')
print('│  1  │ InMemoryCache │ 最简单，重启丢失，适合开发调试                 │')
print('│  2  │ SQLiteCache   │ 持久化到文件，重启不丢失，适合单机             │')
print('│  3  │ SemanticCache │ 语义相似也能命中，命中率最高，实现较复杂        │')
print('│  4  │ 生产实践       │ 选型指南 + 失效策略 + 成本计算                │')
print('└─────┴───────────────┴──────────────────────────────────────────────┘')
print()
print('选型决策树:')
print()
print('  需要缓存吗？')
print('    │')
print('    ├─ 只是开发调试 ──→ InMemoryCache (最简单)')
print('    │')
print('    ├─ 单机 + 需要持久化 ──→ SQLiteCache')
print('    │')
print('    ├─ 分布式多节点 ──→ Redis Cache')
print('    │')
print('    └─ 想要高命中率 ──→ SemanticCache (向量相似度)')
print()
print('=' * 70)
print('项目 19 完成! LLM 缓存从入门到实践，全部搞定。')
print('=' * 70)
