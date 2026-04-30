# 项目零：LLM 原生 API 核心调用与控制 — 设计文档

**日期：** 2026-04-30
**项目：** 项目零：LLM 原生 API 核心调用（纯 OpenAI SDK，无 LangChain）
**目标读者：** LLM 开发新手

---

## 项目目标

完全不使用 LangChain 等上层框架，纯用原生 OpenAI Python SDK 与大模型进行底层交互，
掌握 Messages 三角色、基础参数控制（Temperature/Max_Tokens）、阻塞式调用与流式打字机效果。

---

## 目录结构变更

### 变更前

```
langchain_learning/
├── chatbot.py
├── requirements.txt
└── docs/
```

### 变更后

```
langchain_learning/
├── llm/                       ← 新建（纯原生 OpenAI SDK 系列）
│   └── native_api.py          ← 项目零（新建）
├── langchain/                 ← 新建（LangChain 框架系列）
│   └── chatbot.py             ← 从根目录移入
├── requirements.txt           ← 根目录共用，openai>=1.0.0 已包含
└── docs/
```

未来可扩展：
```
├── rag/                       ← 项目二：RAG（待建）
├── agent/                     ← 项目三：Agent（待建）
```

---

## 文件设计：`llm/native_api.py`

### 内部章节

| 章节 | 内容 | 核心学习点 |
|------|------|-----------|
| 文件头（docstring） | 剧组比喻科普 + Temperature 大白话解释 | System/User/Assistant 三角色定位；Temperature 控制 AI "发散度" |
| 第 0 章 | OpenAI 客户端初始化 | `OpenAI(base_url=..., api_key=...)` 直连 OpenAI 兼容接口；Messages 数组结构说明 |
| 第 1 章 | 阻塞式调用 + 原始数据包解剖 | 打印完整 response 对象；展示 `choices`/`finish_reason`/`usage` 的位置 |
| 第 2 章 | 流式打字机效果 | `stream=True`；`for chunk in stream` 循环；`print(end="", flush=True)` 逐 token 输出 |

### 章节详细说明

**文件头（docstring）**
- 用"剧组拍戏"比喻解释三角色：
  - System（导演）：在开机前给 AI 下达"角色人设"和"行为准则"
  - User（演员/观众的问题）：每一轮用户的实际提问
  - Assistant（AI 演员）：AI 的回复，多轮时历史回复也会追加进 messages
- 用大白话解释 Temperature：
  - 类比"脑洞大小旋钮"，0 = 每次给最保守答案，1+ = 天马行空，0.7 = 黄金平衡点
  - 说明为什么创意写作用高温度，代码生成用低温度

**第 0 章**
- 仅用 `from openai import OpenAI` 一行导入
- 明确标注与 chatbot.py（LangChain 版）的对比：同样的 API，少了所有封装层
- 打印初始化成功信息

**第 1 章（阻塞式）**
- 构造 messages 数组，每条消息展示 JSON 格式注释
- 调用 `client.chat.completions.create()`，接收完整 response
- 用 `pprint` 或格式化打印展示完整原始对象
- 逐行解释 `response.choices[0].message.content`、`finish_reason`、`usage.prompt_tokens`/`completion_tokens`/`total_tokens`

**第 2 章（流式打字机）**
- 同样的 messages，追加 `stream=True`
- 用 `for chunk in stream:` 循环
- 用 `print(chunk.choices[0].delta.content or "", end="", flush=True)` 实现打字机
- 解释 `delta.content` vs `message.content` 的区别（流式碎片 vs 完整文本）
- 循环结束后换行

---

## 技术选型

| 决策点 | 选择 | 原因 |
|--------|------|------|
| SDK | `openai>=1.0.0` | 已在 requirements.txt，无需新增依赖 |
| 接口 | 内部 `chj.cloud` LLM Gateway | 与 chatbot.py 共用已验证的接口，零配置直接运行 |
| 模型 | `kivy-kimi-k2_5` | 与现有项目一致 |
| Temperature 演示 | 第1章用 0.7，第2章用 0.7 | 教学中保持一致，用注释说明如何修改 |
| 文件风格 | 单文件渐进式 | 与 chatbot.py 风格一致，学习曲线平滑 |

---

## 学习目标

读完 `llm/native_api.py` 后，学习者应能理解：
1. Messages 数组的结构和三种角色（System/User/Assistant）的职责分工
2. Temperature 参数如何控制 AI 输出的确定性
3. OpenAI API 原始响应对象的完整结构（choices/finish_reason/usage）
4. 流式输出与阻塞式输出的代码区别和原理差异
5. 纯原生 SDK 调用与 LangChain 封装的关系（理解框架帮你省了什么）
