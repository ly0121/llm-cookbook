"""
╔══════════════════════════════════════════════════════════════════╗
║         提示工程（Prompt Engineering）核心技巧演示               ║
║         用真实 API 调用展示：如何"说话"决定了 AI 的表现           ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【核心理念：提示工程是什么？——给 AI 写"说明书"的艺术】
═══════════════════════════════════════════════════════════════════

想象你雇了一个超级聪明但"完全没有背景信息"的新员工：

  ┌─────────────────────────────────────────────────────────┐
  │  你说："帮我写个东西。"                                   │
  │  员工："写什么？给谁看？多长？什么风格？什么格式？"       │
  │                                                           │
  │  你说："你是一位面向5岁儿童的科普老师，请用比喻解释        │
  │        什么是递归，用3个要点，每个要点不超过20字。"         │
  │  员工：完美输出！                                          │
  └─────────────────────────────────────────────────────────┘

提示工程的本质：你给的指令越精确，AI 的输出就越接近你的期望。

  模糊指令 ──→ 模糊输出（AI 自由发挥，结果不可控）
  精确指令 ──→ 精确输出（AI 按照模板执行，结果可复现）

本文件通过 6 个章节，展示最实用的提示工程技巧。
每个技巧都有"对比实验"——同一问题，不同提示，天壤之别。
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 初始化：OpenAI 客户端配置（和 native_api.py 完全一致）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from openai import OpenAI

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "kivy-kimi-k2_5"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ── 工具函数：简化重复的 API 调用代码 ───────────────────────
def chat(messages, temperature=0.7, max_tokens=500):
    """封装 API 调用，返回纯文本回复"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def show_result(label, prompt_desc, result):
    """格式化打印结果"""
    print(f"\n  【{label}】")
    print(f"  提示策略：{prompt_desc}")
    print(f"  AI 回复：")
    for line in result.strip().split("\n"):
        print(f"    {line}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：角色设定（Role Prompting）
# 核心原理：同一问题，不同"人设"→ 完全不同的回答风格和深度
#
#   就像同一个问题问幼儿园老师和大学教授，
#   你期望得到的答案是截然不同的。
#   System Prompt 就是"角色扮演指令"。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 0 章：角色设定（Role Prompting）")
print("=" * 60)
print()
print("【实验】同一个问题"什么是递归？"，分别问两种角色：")

# ── 角色 A：幼儿园老师 ──────────────────────────────────────
# 设定：用最简单的比喻，面向完全不懂技术的5岁小朋友
role_a = "你是一位幼儿园老师，擅长用讲故事的方式给5岁小朋友解释概念。用简单的比喻，不要用任何专业术语，回答控制在80字以内。"

result_a = chat([
    {"role": "system", "content": role_a},
    {"role": "user", "content": "什么是递归？"},
])

show_result("角色A：幼儿园老师", "简单比喻、面向儿童", result_a)

# ── 角色 B：计算机科学教授 ──────────────────────────────────
# 设定：严谨学术风格，面向有编程基础的学生
role_b = "你是一位计算机科学教授，回答需要包含准确的技术定义、时间复杂度分析和代码示例。面向有编程基础的大学生，回答控制在150字以内。"

result_b = chat([
    {"role": "system", "content": role_b},
    {"role": "user", "content": "什么是递归？"},
])

show_result("角色B：计算机科学教授", "学术严谨、含代码示例", result_b)

print()
print("  💡 启示：System Prompt 中的角色设定，决定了回答的")
print("     深度、风格、用词和结构。这是最基础也最有效的技巧。")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：格式控制
# 核心原理：明确告诉 AI 输出的"形状"，它就能严格遵循
#
#   AI 就像一个超级听话的排版员：
#   你说"用 JSON"，它就输出 JSON；
#   你说"用表格"，它就输出表格。
#   但如果你不说，它就随心所欲。
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 1 章：格式控制")
print("=" * 60)
print()
print("【实验】要求 AI 用不同格式回答同一问题：")

# ── 格式 A：要求 JSON 输出 ─────────────────────────────────
# 技巧：明确指定字段名和结构，AI 就会严格遵循
format_prompt_json = """请列出3种常见的排序算法，用严格的 JSON 数组格式输出。
每个元素包含 "name"（算法名）、"time_complexity"（时间复杂度）、"stable"（是否稳定）三个字段。
只输出 JSON，不要任何额外文字。"""

result_json = chat([
    {"role": "system", "content": "你是一个数据接口，只输出格式化数据，不输出任何多余文字。"},
    {"role": "user", "content": format_prompt_json},
], temperature=0.0)  # temperature=0 让输出更确定，适合结构化数据

show_result("格式A：JSON 输出", "明确指定字段名 + 温度0", result_json)

# ── 格式 B：要求 Markdown 要点列表 ─────────────────────────
format_prompt_md = """请用 Markdown 格式解释 HTTP 状态码的分类，要求：
- 使用二级标题(##)分隔每个类别
- 每个类别下用无序列表列出2个代表性状态码
- 每个状态码后用一句话解释含义
回答控制在150字以内。"""

result_md = chat([
    {"role": "system", "content": "你是一位技术文档作者，严格按照用户要求的格式输出。"},
    {"role": "user", "content": format_prompt_md},
])

show_result("格式B：Markdown 结构化", "指定标题级别+列表格式", result_md)

print()
print("  💡 启示：格式控制的关键是"具体"——不要说"用列表"，")
print("     而要说"用 Markdown 无序列表，每项不超过20字"。")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：Few-Shot Prompting（少样本提示）
# 核心原理：通过几个示例"教会" AI 一种新的任务模式
#
#   类比：你不用解释规则，只需要做几遍示范，
#   聪明的学生就能从示例中"归纳"出模式并举一反三。
#
#   ┌───────────────────────────────────────────┐
#   │  Zero-shot：没有示例，只有指令             │
#   │  Few-shot： 3~5 个输入→输出示例           │
#   │  效果：    Few-shot 通常显著优于 Zero-shot │
#   └───────────────────────────────────────────┘
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 2 章：Few-Shot Prompting（少样本提示）")
print("=" * 60)
print()
print("【实验】用 Few-Shot 构建自定义情感分析器：")
print("  目标：让 AI 学会一种特殊的标注格式（情感+强度分）")

# ── Few-Shot 消息构造 ──────────────────────────────────────
# 关键技巧：用 user/assistant 消息对模拟"示例"
# 每个 user 消息是输入，对应的 assistant 消息是期望输出
# AI 会从这些示例中"学到"你想要的输出模式

few_shot_messages = [
    {"role": "system", "content": "你是一个情感分析器。对用户输入的文本进行情感分析，严格按照示例格式输出。"},
    # ── 示例1 ──
    {"role": "user", "content": "分析：这家餐厅的菜太好吃了，下次还来！"},
    {"role": "assistant", "content": "情感：积极 | 强度：4/5 | 关键词：好吃、还来"},
    # ── 示例2 ──
    {"role": "user", "content": "分析：快递又迟了三天，客服态度还很差。"},
    {"role": "assistant", "content": "情感：消极 | 强度：4/5 | 关键词：迟了、态度差"},
    # ── 示例3 ──
    {"role": "user", "content": "分析：今天天气还行，没什么特别的。"},
    {"role": "assistant", "content": "情感：中性 | 强度：2/5 | 关键词：还行、没什么特别"},
    # ── 真正的任务（AI 需要按照上面的模式回答）──
    {"role": "user", "content": "分析：这部电影的剧情太拖沓了，但是演员演技真的绝了，值得一看。"},
]

result_fewshot = chat(few_shot_messages, temperature=0.0)

show_result("Few-Shot 情感分析", "3个示例教会AI自定义格式", result_fewshot)

print()
print("  💡 启示：Few-Shot 的威力在于"模式归纳"——")
print("     AI 不需要你解释规则，它从示例中自动学会了：")
print("     格式（情感|强度|关键词）、评分标准、输出长度。")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：Chain-of-Thought（思维链）
# 核心原理：让 AI "出声思考"，显著提升逻辑推理准确率
#
#   类比：你让一个学生直接写答案，他可能跳步犯错；
#   但你说"请写出解题过程"，他就会一步步验证，正确率大增。
#
#   ┌─────────────────────────────────────────────┐
#   │  直接回答：  问题 ──→ 答案（容易出错）       │
#   │  思维链：    问题 ──→ 步骤1 → 步骤2 → 答案  │
#   └─────────────────────────────────────────────┘
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 3 章：Chain-of-Thought（思维链）")
print("=" * 60)
print()

# 一道需要逻辑推理的问题
logic_problem = "一个农场有鸡和兔，共有35个头和94只脚。请问有多少只鸡和多少只兔？"

print(f"【逻辑题】{logic_problem}")
print()

# ── 方式 A：直接要求答案（容易出错或跳步）────────────────
result_direct = chat([
    {"role": "system", "content": "直接给出最终答案，不需要过程，用一句话回答。"},
    {"role": "user", "content": logic_problem},
], temperature=0.0)

show_result("方式A：直接回答", "不要求思考过程", result_direct)

# ── 方式 B：零样本思维链（Zero-shot CoT）─────────────────
# 魔法咒语："请一步步思考"（Let's think step by step）
# 只需加这一句话，推理准确率就能显著提升！
result_cot = chat([
    {"role": "system", "content": "你是一个严谨的数学老师。"},
    {"role": "user", "content": logic_problem + "\n\n请一步步思考，列出每一步的推理过程，最后给出答案。"},
], temperature=0.0, max_tokens=800)

show_result("方式B：思维链（CoT）", "加了'请一步步思考'", result_cot)

print()
print("  💡 启示：'请一步步思考'是提示工程中最强大的6个字。")
print("     它迫使 AI 展开推理链条，而不是凭直觉猜答案。")
print("     对数学、逻辑、代码调试类问题效果尤为显著。")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：System Prompt 设计模板
# 核心原理：一个好的 System Prompt 应该包含5个要素
#
#   ┌────────────────────────────────────────────┐
#   │  ① 角色定义：你是谁？                      │
#   │  ② 任务描述：你要做什么？                  │
#   │  ③ 输出格式：结果长什么样？                │
#   │  ④ 约束条件：什么不能做？                  │
#   │  ⑤ 示例/补充：给一个参考                   │
#   └────────────────────────────────────────────┘
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 4 章：System Prompt 设计模板")
print("=" * 60)
print()
print("【对比实验】模糊 vs 精确的 System Prompt")

# ── 模糊版本：几乎没有有用信息 ─────────────────────────────
vague_system = "你是一个助手，帮我翻译东西。"

result_vague = chat([
    {"role": "system", "content": vague_system},
    {"role": "user", "content": "翻译：We need to refactor the legacy codebase to improve maintainability."},
])

show_result("模糊 System Prompt", vague_system, result_vague)

# ── 精确版本：包含5要素的完整模板 ──────────────────────────
detailed_system = """# 角色
你是一位资深的技术文档翻译官，专注于软件工程领域。

# 任务
将用户提供的英文技术文本翻译为中文。

# 输出格式
严格按以下格式输出：
- 译文：（翻译结果）
- 术语表：（列出专业术语的翻译对照）

# 约束
- 专业术语保留英文原文并在括号中给出中文翻译
- 不要添加任何解释或评论
- 语气正式、简洁"""

result_detailed = chat([
    {"role": "system", "content": detailed_system},
    {"role": "user", "content": "翻译：We need to refactor the legacy codebase to improve maintainability."},
])

show_result("精确 System Prompt（5要素模板）", "角色+任务+格式+约束+示例", result_detailed)

print()
print("  💡 启示：模糊的 System Prompt 像'帮我做点事'，")
print("     精确的 System Prompt 像一份详细的工作说明书。")
print("     投入5分钟写好模板，节省无数次重试。")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 章：提示工程反模式（常见错误 & 修复）
# 核心原理：知道"什么不该做"，和"什么该做"同等重要
#
#   ┌────────────────────────────────────────────────────┐
#   │  反模式1：否定指令 → AI 更容易犯错                  │
#   │  反模式2：信息过载 → AI 迷失重点                    │
#   │  修复方法：正面指令 + 结构化约束                    │
#   └────────────────────────────────────────────────────┘
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("第 5 章：提示工程反模式（常见错误 & 修复）")
print("=" * 60)
print()
print("【反模式1】否定指令 vs 正面指令")
print("  问题：'不要提到X' 反而会让 AI 更容易提到 X")

# ── 反模式：否定指令（"不要..."）─────────────────────────
negative_prompt = "介绍Python语言。不要提到缩进，不要提到动态类型，不要超过100字。"

result_negative = chat([
    {"role": "system", "content": "你是一位编程语言专家。"},
    {"role": "user", "content": negative_prompt},
], temperature=0.3)

show_result("反模式：否定指令", "不要X、不要Y、不要Z", result_negative)

# ── 正确做法：正面指令（"只关注..."）────────────────────
positive_prompt = "介绍Python语言。只关注以下3个方面：生态系统丰富、社区活跃、应用领域广。每个方面用一句话，总共不超过80字。"

result_positive = chat([
    {"role": "system", "content": "你是一位编程语言专家。"},
    {"role": "user", "content": positive_prompt},
], temperature=0.3)

show_result("正确做法：正面指令", "只关注X、Y、Z", result_positive)

print()
print("【反模式2】一次塞太多任务 vs 分步拆解")
print("  问题：一个提示包含多个不相关任务，AI 容易顾此失彼")

# ── 反模式：一次性多任务 ───────────────────────────────────
overloaded_prompt = "帮我：1)写一首关于春天的诗 2)翻译成英文 3)分析它的修辞手法 4)打分"

result_overloaded = chat([
    {"role": "user", "content": overloaded_prompt},
], max_tokens=600)

show_result("反模式：一次性多任务", "4个不相关任务塞一起", result_overloaded)

# ── 正确做法：分步执行（每步一个 API 调用）────────────────
# 第一步：先完成核心任务
step1_result = chat([
    {"role": "system", "content": "你是一位中国古典诗词大师。只输出诗歌本身，不加任何解释。"},
    {"role": "user", "content": "写一首关于春天的五言绝句。"},
], temperature=0.8)

# 第二步：基于上一步结果继续
step2_result = chat([
    {"role": "system", "content": "你是一位文学评论家。用一句话分析下面这首诗的核心修辞手法。"},
    {"role": "user", "content": f"请分析这首诗的修辞手法：\n{step1_result}"},
], temperature=0.3)

print()
print("  【正确做法：分步拆解】")
print(f"    步骤1（写诗）：{step1_result.strip()}")
print(f"    步骤2（分析）：{step2_result.strip()}")

print()
print("  💡 启示：")
print("     1. 用正面指令（'只做X'）替代否定指令（'不要做Y'）")
print("     2. 复杂任务拆成多步，每步一个清晰目标")
print("     3. 上一步的输出作为下一步的输入（管道模式）")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 总结：提示工程速查表
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 60)
print("总结：提示工程速查表")
print("=" * 60)
print("""
  ┌──────────────────┬─────────────────────────────────────┐
  │  技巧             │  一句话总结                          │
  ├──────────────────┼─────────────────────────────────────┤
  │  角色设定         │  System Prompt 定义"AI是谁"          │
  │  格式控制         │  明确说"用JSON/表格/列表"输出        │
  │  Few-Shot         │  给3-5个示例，AI自动学会模式         │
  │  思维链(CoT)      │  加"请一步步思考"，推理力飙升        │
  │  5要素模板        │  角色+任务+格式+约束+示例            │
  │  正面指令         │  说"只做X"而非"不要做Y"             │
  └──────────────────┴─────────────────────────────────────┘

  下一步学习建议：
  - 尝试组合多种技巧（如 角色设定 + Few-Shot + CoT）
  - 对同一问题反复调整提示，观察输出变化
  - 建立自己的 System Prompt 模板库
""")
print("=" * 60)
print("提示工程学习完毕！记住：好的提示 = 好的输出。")
print("=" * 60)
