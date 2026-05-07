---
title: 提示工程
---

<script setup>
const code1 = `# Prompt Template 模式演示（纯 Python，无需 API）

class PromptTemplate:
    '''简易提示词模板'''
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

# 定义模板
qa_template = PromptTemplate(
    template='''你是一位{role}领域的专家。

请根据以下背景知识回答问题：
背景：{context}

问题：{question}

要求：{requirements}'''
)

# 使用模板
prompt = qa_template.format(
    role='人工智能',
    context='Transformer 架构由 Google 在 2017 年提出，是现代 LLM 的基础。',
    question='Transformer 是什么时候提出的？',
    requirements='用一句话简洁回答'
)

print('=== 生成的完整 Prompt ===')
print(prompt)
print()
print('=== 模板变量 ===')
print(f'模板中的占位符数量: 4 (role, context, question, requirements)')
print(f'生成的 Prompt 长度: {len(prompt)} 字符')
`
</script>

# 提示工程（Prompt Engineering）

提示工程是与大语言模型高效交互的核心技能。通过精心设计提示词，可以显著提升模型输出的质量和可控性。

## 1. Few-Shot Prompting（少样本提示）

Few-Shot 的核心思想：通过提供几个示例，让模型"学会"你期望的输出模式。

```
无示例（Zero-Shot）：
  用户: 翻译成英文：高兴
  AI:   happy

有示例（Few-Shot）：
  用户: 翻译成英文，格式为"中文 -> 英文"
        示例：
        高兴 -> happy
        悲伤 -> sad
        翻译：愤怒
  AI:   愤怒 -> angry
```

**为什么 Few-Shot 有效？**

| 维度 | Zero-Shot | Few-Shot |
|------|-----------|----------|
| 格式控制 | 模型自由发挥 | 严格遵循示例格式 |
| 输出稳定性 | 不可预测 | 高度一致 |
| 适用场景 | 简单任务 | 复杂格式/风格要求 |

## 2. Chain-of-Thought（思维链）

让模型"一步步思考"，而非直接给出答案：

```
直接回答（容易出错）：
  问: 一个商店有 23 个苹果，卖了 17 个，又进了 12 个，还有几个？
  答: 18（正确答案是 18，但复杂问题会出错）

思维链（CoT）：
  问: 一个商店有 23 个苹果，卖了 17 个，又进了 12 个，还有几个？
      请一步步思考。
  答: 1. 初始: 23 个
      2. 卖了 17 个: 23 - 17 = 6 个
      3. 进了 12 个: 6 + 12 = 18 个
      答案是 18 个。
```

**CoT 的变体：**

- **Zero-Shot CoT**：仅添加"请一步步思考"
- **Manual CoT**：手写推理示例
- **Auto-CoT**：让模型自动生成推理链
- **Tree-of-Thought**：多条推理路径并行探索

## 3. Prompt Template 模式

在实际开发中，我们用模板化的方式管理提示词。下面是一个纯 Python 实现的提示模板模式：

<PythonRunner :browser-runnable="true" :code="code1" />

## 4. 提示工程最佳实践

| 技巧 | 说明 | 示例 |
|------|------|------|
| 角色设定 | 给模型一个专业身份 | "你是一位资深 Python 开发者" |
| 格式约束 | 明确输出格式 | "请用 JSON 格式输出" |
| 示例引导 | 提供 Few-Shot 样例 | 给出 2-3 个输入输出对 |
| 分步引导 | Chain-of-Thought | "请分步骤分析" |
| 约束边界 | 限制回答范围 | "仅基于提供的资料回答" |

## 5. 完整 API 示例

::: warning 需要本地运行
完整的 LLM API 调用示例（包括 OpenAI / Anthropic 接口调用、流式输出、多轮对话等）请参考源代码文件：

```
llm/prompt_engineering.py
```

该文件包含：系统提示设计、Few-Shot 实现、Chain-of-Thought 完整流程、输出格式控制等。
:::

---

::: tip 下一步
- [文本生成机制](/llm/generation) — 深入了解 Greedy / Top-K / Top-P 等解码策略
- [LangChain 框架](/langchain/) — 用框架管理复杂的提示词模板
:::
