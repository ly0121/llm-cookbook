"""
╔══════════════════════════════════════════════════════════════════╗
║         项目：LLM 生成策略（Decoding Strategies）全面实验         ║
║         探索 Temperature、Top-P、Penalty、Stop 等参数的效果       ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心问题：LLM 是怎么"一个字一个字"生成回答的？】
═══════════════════════════════════════════════════════════════════

LLM 生成文本的过程，本质上是一个"逐步选词"的循环：

  输入 prompt → [模型前向计算] → 得到下一个词的概率分布 → 选一个词
                                        ↓
                              把选出的词追加到输入 → 重复上述过程
                                        ↓
                              直到遇到结束标记或达到 max_tokens

关键：每一步都有成千上万个候选词，模型给每个词打了一个分数（logit）。
"生成策略"就是决定"从这些候选词中怎么选"的规则。

  ┌─────────────────────────────────────────────────────────────┐
  │  Logits（原始分数）                                           │
  │    ↓                                                         │
  │  Softmax（转换为概率分布）                                    │
  │    ↓                                                         │
  │  Temperature（拉平/尖锐化概率分布）                           │
  │    ↓                                                         │
  │  Top-P / Top-K（截断候选池）                                  │
  │    ↓                                                         │
  │  Frequency/Presence Penalty（惩罚已出现的词）                 │
  │    ↓                                                         │
  │  采样（从最终分布中随机选一个词）                             │
  └─────────────────────────────────────────────────────────────┘

本文件通过真实 API 调用，让你亲眼看到每个参数如何影响生成结果。
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化与生成策略总览
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from openai import OpenAI

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "kivy-kimi-k2_5"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print("=" * 60)
print("第 0 章：生成策略总览")
print("=" * 60)
print()
print("""
┌──────────────────────────────────────────────────────────────┐
│              LLM 文本生成管道（Generation Pipeline）            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  "今天天气"  →  模型计算  →  Logits: [很:5.2, 好:4.8, ...]   │
│                                 ↓                            │
│                         Temperature 缩放                      │
│                  （温度高→概率拉平，温度低→概率尖锐）          │
│                                 ↓                            │
│                         Top-P 截断                            │
│                  （只保留累积概率达到 P 的词）                  │
│                                 ↓                            │
│                         Penalty 惩罚                          │
│                  （降低已出现词的概率，减少重复）              │
│                                 ↓                            │
│                         采样/贪婪选择                          │
│                  （从最终分布中选出下一个 token）              │
│                                 ↓                            │
│                         输出: "很"                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘

想象你在点菜：
  - Temperature = 你愿不愿意尝试新菜（高温=喜欢冒险，低温=只点老菜）
  - Top-P = 菜单上可选的范围（低P=只看前几道热门菜，高P=整本菜单都看）
  - Penalty = 不想重复点的惩罚（"上次吃过了，这次换一个"）
  - Max Tokens = 最多吃几道菜（到了数量就停）
""")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：Temperature 实验
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Temperature 的数学原理：
#   对 logits 除以 temperature 后再做 softmax：
#     P(token_i) = exp(logit_i / T) / sum(exp(logit_j / T))
#
#   T → 0：分布趋近于 one-hot（只有最大概率的词被选中）→ 确定性输出
#   T = 1：保持模型原始概率分布不变
#   T > 1：分布被"拉平"，低概率的词也有更多机会被选中 → 更随机
#
#   形象比喻：
#     T=0  像"开卷考试只抄标准答案"——每次都一样
#     T=1  像"正常发挥"——有随机性但整体合理
#     T=1.5 像"灵感爆发的诗人"——天马行空但可能跑偏

print("=" * 60)
print("第 1 章：Temperature 实验")
print("=" * 60)
print()
print("【实验设计】同一个 prompt，不同 temperature，各调用 3 次")
print("  观察：temperature=0 时 3 次结果是否一样？高温度时呢？")
print()

TEMP_PROMPT = "用一句话描述月亮。"
TEMPERATURES = [0.0, 0.5, 1.0, 1.5]

for temp in TEMPERATURES:
    print(f"── Temperature = {temp} {'─' * 40}")
    for trial in range(1, 4):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一位诗意的文学家，回答控制在20字以内。"},
                {"role": "user", "content": TEMP_PROMPT},
            ],
            temperature=temp,
            max_tokens=60,
        )
        text = response.choices[0].message.content.strip()
        print(f"  第{trial}次: {text}")
    print()

print("💡 观察要点：")
print("   - temperature=0 时，3 次结果应该完全一样（贪婪解码，确定性输出）")
print("   - temperature 越高，3 次结果差异越大（随机性增加）")
print("   - temperature=1.5 时可能出现不通顺的表达（概率分布过于平坦）")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：Top-P（Nucleus Sampling）实验
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# Top-P 的原理：
#   1. 将所有候选词按概率从高到低排列
#   2. 从最高概率开始累加，直到累积概率 >= top_p
#   3. 只从这些"入围"的词中进行采样，其他词直接排除
#
#   形象比喻：
#     top_p=0.1  → 只看"前几名优等生"（候选池极小，回答保守）
#     top_p=0.5  → 看"前半数同学"
#     top_p=0.9  → 看"绝大多数同学"（候选池大，回答多样）
#     top_p=1.0  → 看"全班所有人"（不做截断，等价于关闭 top_p）
#
#   ┌────────────────────────────────────────────────────────┐
#   │  候选词概率分布（从高到低排列）：                         │
#   │                                                        │
#   │  ████████  "月亮" (40%)                                │
#   │  █████     "明月" (25%)                                │
#   │  ███       "圆月" (15%)                                │
#   │  ██        "银盘" (10%)                                │
#   │  █         "玉兔" (5%)                                 │
#   │  ░         "冰轮" (3%)                                 │
#   │  ░         "蟾宫" (2%)                                 │
#   │                                                        │
#   │  top_p=0.65 → 只保留前3个词（累积=40+25=65%）          │
#   │  top_p=0.9  → 保留前4个词（累积=40+25+15+10=90%）      │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 2 章：Top-P（Nucleus Sampling）实验")
print("=" * 60)
print()

TOP_P_PROMPT = "请列举三种你喜欢的水果，并各写一个形容词。"
TOP_P_VALUES = [0.1, 0.5, 0.9, 1.0]

# 固定 temperature=0.8，这样可以单独看 top_p 的效果
for top_p in TOP_P_VALUES:
    print(f"── top_p = {top_p} (temperature=0.8) {'─' * 30}")
    for trial in range(1, 3):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个活泼的美食博主，回答简洁有趣。"},
                {"role": "user", "content": TOP_P_PROMPT},
            ],
            temperature=0.8,
            top_p=top_p,
            max_tokens=100,
        )
        text = response.choices[0].message.content.strip()
        print(f"  第{trial}次: {text}")
    print()

print("💡 观察要点：")
print("   - top_p=0.1 时回答非常保守，用词重复率高")
print("   - top_p=0.9 时回答更多样，出现更有创意的形容词")
print("   - top_p 和 temperature 同时控制多样性，通常只调其中一个")
print("   - OpenAI 官方建议：调 temperature 就把 top_p 设为 1，反之亦然")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：Frequency & Presence Penalty
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 这两个参数都用来减少重复，但作用机制不同：
#
# frequency_penalty（频率惩罚）：
#   每个 token 已出现的次数越多，惩罚越重
#   公式：logit -= frequency_penalty * token出现次数
#   效果：像"你已经说了5遍'非常'了，少说点！"
#   范围：-2.0 ~ 2.0（正值惩罚重复，负值鼓励重复）
#
# presence_penalty（存在惩罚）：
#   只要 token 出现过（不管几次），就施加固定惩罚
#   公式：logit -= presence_penalty * (1 if token出现过 else 0)
#   效果：像"这个词你用过了，换个新词试试"
#   范围：-2.0 ~ 2.0
#
#   ┌────────────────────────────────────────────────────────┐
#   │  区别比喻：                                             │
#   │                                                        │
#   │  frequency_penalty = 记仇的老师                         │
#   │    "你迟到了 1 次扣 1 分，迟到了 5 次扣 5 分"           │
#   │    → 出现越多，惩罚越重                                 │
#   │                                                        │
#   │  presence_penalty = 鼓励探索的导师                       │
#   │    "你提过的话题我都不想再听了，说点新的"                │
#   │    → 只看"有没有出现过"，不看出现几次                   │
#   │    → 更适合鼓励 AI 探索新主题                           │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 3 章：Frequency & Presence Penalty")
print("=" * 60)
print()

# ── 3.1 展示无惩罚时的重复问题 ──────────────────────────────
print("── 3.1 无惩罚：观察 AI 的重复倾向 ──────────────────")
print()

REPEAT_PROMPT = "请写一段关于春天的散文，至少100字。"

response_no_penalty = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "你是一位散文作家。"},
        {"role": "user", "content": REPEAT_PROMPT},
    ],
    temperature=0.7,
    max_tokens=200,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)
print("  [无惩罚] frequency_penalty=0, presence_penalty=0:")
print(f"  {response_no_penalty.choices[0].message.content.strip()}")
print()

# ── 3.2 frequency_penalty 的效果 ────────────────────────────
print("── 3.2 添加 frequency_penalty=1.5 ──────────────────")
print()

response_freq = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "你是一位散文作家。"},
        {"role": "user", "content": REPEAT_PROMPT},
    ],
    temperature=0.7,
    max_tokens=200,
    frequency_penalty=1.5,
    presence_penalty=0.0,
)
print("  [频率惩罚] frequency_penalty=1.5:")
print(f"  {response_freq.choices[0].message.content.strip()}")
print()

# ── 3.3 presence_penalty 的效果 ─────────────────────────────
print("── 3.3 添加 presence_penalty=1.5 ──────────────────")
print()

response_pres = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "你是一位散文作家。"},
        {"role": "user", "content": REPEAT_PROMPT},
    ],
    temperature=0.7,
    max_tokens=200,
    frequency_penalty=0.0,
    presence_penalty=1.5,
)
print("  [存在惩罚] presence_penalty=1.5:")
print(f"  {response_pres.choices[0].message.content.strip()}")
print()

print("💡 观察要点：")
print("   - 无惩罚时，AI 可能重复使用'春天'、'万物'等高频词")
print("   - frequency_penalty 会让重复用词的频率降低")
print("   - presence_penalty 会鼓励 AI 引入更多新的意象和词汇")
print("   - 设置过高（>2）可能导致文本变得不自然")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：Max Tokens 与停止条件
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# max_tokens：限制 AI 最多生成多少个 token
#   - 1 个中文字 ≈ 1-2 个 token
#   - 如果 AI 还没说完就到了 max_tokens，会被强制截断
#   - 此时 finish_reason = "length"（而非正常结束的 "stop"）
#
# stop：自定义停止序列
#   - 当 AI 生成的文本中出现指定字符串时，立即停止生成
#   - 停止序列本身不会包含在输出中
#   - 适合场景：提取结构化数据、限制输出格式
#
#   ┌────────────────────────────────────────────────────────┐
#   │  比喻：max_tokens 像"计时器"                            │
#   │    不管说到哪里，时间到了就必须停                        │
#   │                                                        │
#   │  比喻：stop 像"安全词"                                  │
#   │    一旦说出这个词，对话立刻结束                          │
#   └────────────────────────────────────────────────────────┘

print("=" * 60)
print("第 4 章：Max Tokens 与停止条件")
print("=" * 60)
print()

# ── 4.1 max_tokens 截断实验 ─────────────────────────────────
print("── 4.1 max_tokens 截断实验 ──────────────────────────")
print()

response_short = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "你是一位历史老师。"},
        {"role": "user", "content": "请详细介绍一下唐朝的历史。"},
    ],
    temperature=0.7,
    max_tokens=30,  # 故意设很小，观察截断
)

text_short = response_short.choices[0].message.content
finish = response_short.choices[0].finish_reason

print(f"  max_tokens=30 的输出：")
print(f"  【{text_short}】")
print(f"  finish_reason = {finish!r}")
print()

if finish == "length":
    print("  ⚠️ 检测到截断！finish_reason='length' 意味着 AI 还没说完就被截断了")
    print("     解决方案：增大 max_tokens，或在业务逻辑中检测并续写")
elif finish == "stop":
    print("  AI 自然结束，没有被截断")
print()

# ── 4.2 stop 停止序列实验 ───────────────────────────────────
print("── 4.2 stop 停止序列实验 ──────────────────────────────")
print()
print("  让 AI 列举内容，但遇到'3.'时自动停止（只要前2条）")
print()

response_stop = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "你是一位百科全书，请用编号列表格式回答。"},
        {"role": "user", "content": "列举中国四大发明。"},
    ],
    temperature=0.0,
    max_tokens=200,
    stop=["3."],  # 遇到 "3." 就停止
)

text_stop = response_stop.choices[0].message.content
finish_stop = response_stop.choices[0].finish_reason

print(f"  stop=['3.'] 的输出：")
print(f"  【{text_stop}】")
print(f"  finish_reason = {finish_stop!r}")
print()
print("  注意：stop 序列 '3.' 本身不会出现在输出中")
print("  finish_reason='stop' 可能表示自然结束或命中了停止序列")
print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 章：组合策略与最佳实践
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# 不同任务需要不同的参数组合，就像不同菜需要不同的火候：
#
#   ┌────────────────────────────────────────────────────────┐
#   │  任务类型        │ temperature │ top_p │ penalty       │
#   ├────────────────────────────────────────────────────────┤
#   │  代码生成        │    0.0      │  1.0  │  0.0          │
#   │  数据提取        │    0.0      │  1.0  │  0.0          │
#   │  日常对话        │    0.7      │  0.9  │  0.0          │
#   │  创意写作        │    0.9      │  0.95 │  0.6(pres)    │
#   │  头脑风暴        │    1.0      │  0.95 │  1.0(pres)    │
#   └────────────────────────────────────────────────────────┘
#
#   原则：
#   1. 需要确定性输出 → temperature=0（贪婪解码）
#   2. 需要多样性 → 调高 temperature 或降低 top_p
#   3. temperature 和 top_p 通常只调一个，另一个设为默认值
#   4. penalty 主要用于长文本生成（短回答一般不需要）

print("=" * 60)
print("第 5 章：组合策略与最佳实践")
print("=" * 60)
print()


# ── 封装：根据任务类型自动设置最佳参数 ──────────────────────
def generate_with_strategy(prompt: str, task_type: str, system_prompt: str = "") -> str:
    """
    根据任务类型自动选择最佳生成策略。

    参数：
        prompt: 用户输入
        task_type: 任务类型，可选值：
            - "code"      代码生成（精确、确定性）
            - "creative"  创意写作（多样、有想象力）
            - "extract"   数据提取（精确、结构化）
            - "chat"      日常对话（自然、平衡）
            - "brainstorm" 头脑风暴（发散、探索性）
        system_prompt: 系统提示词（可选）

    返回：
        AI 生成的文本
    """
    # 根据任务类型选择参数组合
    strategies = {
        "code": {
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 500,
        },
        "creative": {
            "temperature": 0.9,
            "top_p": 0.95,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.6,
            "max_tokens": 300,
        },
        "extract": {
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 200,
        },
        "chat": {
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "max_tokens": 200,
        },
        "brainstorm": {
            "temperature": 1.0,
            "top_p": 0.95,
            "frequency_penalty": 0.5,
            "presence_penalty": 1.0,
            "max_tokens": 300,
        },
    }

    if task_type not in strategies:
        raise ValueError(f"未知任务类型: {task_type}，可选: {list(strategies.keys())}")

    params = strategies[task_type]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        **params,
    )

    return response.choices[0].message.content.strip()


# ── 5.1 代码生成策略演示 ────────────────────────────────────
print("── 5.1 代码生成（temperature=0, 精确确定性）─────────")
print()

code_result = generate_with_strategy(
    prompt="写一个 Python 函数，计算斐波那契数列的第 n 项（递归实现）。",
    task_type="code",
    system_prompt="你是一位 Python 专家，只输出代码，不要解释。",
)
print(f"  {code_result}")
print()

# ── 5.2 创意写作策略演示 ────────────────────────────────────
print("── 5.2 创意写作（temperature=0.9, 富有想象力）────────")
print()

creative_result = generate_with_strategy(
    prompt="用一段话描述一个从未存在过的颜色。",
    task_type="creative",
    system_prompt="你是一位充满想象力的诗人。",
)
print(f"  {creative_result}")
print()

# ── 5.3 数据提取策略演示 ────────────────────────────────────
print("── 5.3 数据提取（temperature=0, 精确结构化）──────────")
print()

extract_result = generate_with_strategy(
    prompt="从以下文本中提取人名和地点：'张三昨天去了北京，在那里遇到了李四。'请用JSON格式输出。",
    task_type="extract",
    system_prompt="你是一个信息提取工具，只输出JSON，不要任何额外解释。",
)
print(f"  {extract_result}")
print()

# ── 5.4 头脑风暴策略演示 ────────────────────────────────────
print("── 5.4 头脑风暴（temperature=1.0, 发散探索）──────────")
print()

brainstorm_result = generate_with_strategy(
    prompt="给一个卖雨伞的店铺想5个有创意的名字。",
    task_type="brainstorm",
    system_prompt="你是一位创意总监，擅长取名。",
)
print(f"  {brainstorm_result}")
print()

# ── 总结 ────────────────────────────────────────────────────
print("=" * 60)
print("全部实验完毕！核心总结：")
print("=" * 60)
print("""
  ┌────────────────────────────────────────────────────────────┐
  │  参数             │ 作用                │ 关键取值          │
  ├────────────────────────────────────────────────────────────┤
  │  temperature      │ 控制随机性           │ 0=确定, 0.7=平衡  │
  │  top_p            │ 截断候选词池         │ 0.1=保守, 1=不截断 │
  │  frequency_penalty│ 惩罚高频重复词       │ 0~2, 越大越不重复  │
  │  presence_penalty │ 鼓励使用新词/新主题  │ 0~2, 越大越多样    │
  │  max_tokens       │ 限制输出长度         │ 按需设置           │
  │  stop             │ 自定义停止条件       │ 字符串列表          │
  └────────────────────────────────────────────────────────────┘

  黄金法则：
  1. 需要精确答案 → temperature=0, top_p=1
  2. 需要创意多样 → temperature=0.7~1.0, presence_penalty=0.5~1.0
  3. temperature 和 top_p 只调一个，另一个保持默认
  4. 先从默认值开始，根据实际输出效果微调
""")
