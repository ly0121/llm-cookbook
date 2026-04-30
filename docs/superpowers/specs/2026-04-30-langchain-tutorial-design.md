# LangChain 核心组件教学项目 - 设计文档

**日期：** 2026-04-30
**项目：** 项目一：LangChain 核心组件打样（地基）
**目标读者：** LLM 开发新手

---

## 项目位置

```
~/Desktop/langchain_learning/
├── chatbot.py          # 主教学文件（单文件渐进式，5个章节）
└── requirements.txt    # 专属依赖列表
```

独立于 livis 主项目，使用独立 venv 环境。

---

## 文件结构：chatbot.py 内部章节

| 章节 | 内容 | 核心学习点 |
|------|------|-----------|
| 第 0 章 | 环境初始化 | ChatOpenAI + base_url 接入 lixiang 接口 |
| 第 1 章 | PromptTemplate 演示 | 模板定义、变量填充、打印效果 |
| 第 2 章 | LCEL 管道演示 | `prompt | llm | parser` 三级管道语法 |
| 第 3 章 | OutputParser 演示 | AIMessage 原始对象 vs 解析后字符串对比 |
| 第 4 章 | 带记忆的完整聊天机器人 | RunnableWithMessageHistory + session_id |

---

## 技术选型

| 决策点 | 选择 | 原因 |
|--------|------|------|
| LLM 类 | `ChatOpenAI` | 支持 `base_url`，兼容 lixiang.com DeepSeek-V3 接口 |
| 记忆方案 | `RunnableWithMessageHistory` + `ChatMessageHistory` | LangChain 0.2+ 官方推荐，旧 ConversationBufferMemory 已废弃 |
| 输出解析 | `StrOutputParser` | 最常用，AIMessage → 纯字符串 |
| 历史存储 | 内存 dict `store = {}` | 教学轻量无依赖，生产可换 Redis |
| API Key | 代码内配置 + 注释说明环境变量正确姿势 | 降低新手门槛 |

---

## requirements.txt 依赖

```
langchain>=0.3.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
openai>=1.0.0
```

---

## 学习目标

读完 chatbot.py 后，学习者应能理解：
1. LangChain 是什么，解决什么问题（流水线组装 LLM 应用）
2. LCEL 管道 `|` 的本质（Unix pipe 思想在 LLM 中的应用）
3. 如何用 RunnableWithMessageHistory 实现有状态的多轮对话
4. 如何对接 OpenAI 兼容的第三方接口
