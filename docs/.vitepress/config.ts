import { defineConfig } from "vitepress";

export default defineConfig({
  title: "LLM Cookbook",
  description:
    "大模型应用开发完全手册 — 从原理到实战，系统掌握 LLM/Agent/RAG 全技术栈",
  lang: "zh-CN",
  base: "/llm-cookbook/",

  head: [["link", { rel: "icon", href: "/llm-cookbook/favicon.ico" }]],

  themeConfig: {
    logo: "/logo.svg",

    nav: [
      { text: "首页", link: "/" },
      { text: "快速开始", link: "/guide/getting-started" },
      { text: "学习路线", link: "/guide/roadmap" },
      {
        text: "知识模块",
        items: [
          { text: "ML 基础（前置）", link: "/ml-foundations/" },
          { text: "LLM 基础", link: "/llm/" },
          { text: "LangChain", link: "/langchain/" },
          { text: "RAG 检索增强", link: "/rag/" },
          { text: "Agent 智能体", link: "/agent/" },
          { text: "LangGraph", link: "/langgraph/" },
        ],
      },
    ],

    sidebar: {
      "/": [
        {
          text: "开始",
          items: [
            { text: "快速开始", link: "/guide/getting-started" },
            { text: "学习路线", link: "/guide/roadmap" },
          ],
        },
        {
          text: "零、ML 基础（前置补课）",
          collapsed: true,
          items: [
            { text: "本章导读", link: "/ml-foundations/" },
            { text: "经典机器学习", link: "/ml-foundations/classical-ml" },
            { text: "深度学习基础", link: "/ml-foundations/deep-learning" },
            { text: "NLP 经典基础", link: "/ml-foundations/nlp-foundations" },
            { text: "ML 与 LLM 的关系", link: "/ml-foundations/ml-vs-llm" },
            { text: "进阶学习路径", link: "/ml-foundations/learning-path" },
          ],
        },
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
        },
        {
          text: "零.6、训练后期与对齐",
          collapsed: true,
          items: [
            { text: "0. 全景", link: "/ml-foundations/post-training/" },
            { text: "0. 全景（独立页）", link: "/ml-foundations/post-training/overview" },
            { text: "1. SFT", link: "/ml-foundations/post-training/sft" },
            { text: "2. LoRA", link: "/ml-foundations/post-training/lora" },
            { text: "3. QLoRA", link: "/ml-foundations/post-training/qlora" },
            { text: "4. DPO", link: "/ml-foundations/post-training/dpo" },
            { text: "5. 量化", link: "/ml-foundations/post-training/quantization" },
            { text: "6. 评估", link: "/ml-foundations/post-training/evaluation" },
            { text: "7. 选型决策", link: "/ml-foundations/post-training/selection" },
          ],
        },
        {
          text: "一、LLM 基础",
          collapsed: false,
          items: [
            { text: "什么是大语言模型", link: "/llm/" },
            { text: "NLP 技术演进史", link: "/llm/nlp-evolution" },
            { text: "Transformer 架构详解", link: "/llm/transformer" },
            { text: "提示工程", link: "/llm/prompt-engineering" },
            { text: "Tokenization", link: "/llm/tokenization" },
            { text: "文本生成机制", link: "/llm/generation" },
            { text: "Embedding 词向量", link: "/llm/embedding" },
            { text: "Function Calling", link: "/llm/function-calling" },
            { text: "开源模型生态", link: "/llm/open-source-models" },
          ],
        },
        {
          text: "一（续）、LLM 生产级知识",
          collapsed: true,
          items: [
            { text: "推理部署与加速", link: "/llm/inference" },
            { text: "企业级 RAG 工程化", link: "/llm/rag-engineering" },
            { text: "LLM 评测与测试", link: "/llm/evaluation" },
            { text: "可观测性与 LLMOps", link: "/llm/observability" },
            { text: "安全合规与护栏", link: "/llm/security" },
            { text: "生产级数据工程", link: "/llm/data-engineering" },
            { text: "进阶方向与前沿技术", link: "/llm/advanced-topics" },
          ],
        },
        {
          text: "二、LangChain 框架",
          collapsed: true,
          items: [{ text: "框架概述与核心组件", link: "/langchain/" }],
        },
        {
          text: "三、RAG 检索增强",
          collapsed: true,
          items: [
            { text: "RAG 基础", link: "/rag/" },
            { text: "高级策略", link: "/rag/advanced" },
          ],
        },
        {
          text: "四、Agent 智能体",
          collapsed: true,
          items: [{ text: "Agent 架构", link: "/agent/" }],
        },
        {
          text: "五、LangGraph 工作流",
          collapsed: true,
          items: [
            { text: "图计算基础", link: "/langgraph/" },
            { text: "进阶：HITL 与持久化", link: "/langgraph/advanced" },
          ],
        },
        {
          text: "六、工程实践",
          collapsed: true,
          items: [
            { text: "流式处理", link: "/engineering/streaming" },
            { text: "结构化输出", link: "/engineering/structured-output" },
            { text: "异步并发", link: "/engineering/async" },
            { text: "缓存策略", link: "/engineering/caching" },
            { text: "错误处理", link: "/engineering/error-handling" },
            { text: "可观测性", link: "/engineering/observability" },
          ],
        },
        {
          text: "七、生产部署",
          collapsed: true,
          items: [
            { text: "API 服务化", link: "/production/api-service" },
            { text: "安全护栏", link: "/production/guardrails" },
            { text: "评估体系", link: "/production/evaluation" },
          ],
        },
        {
          text: "八、高级主题",
          collapsed: true,
          items: [
            { text: "记忆系统", link: "/advanced/memory" },
            { text: "自我反思", link: "/advanced/self-reflection" },
            { text: "向量数据库", link: "/advanced/vectordb" },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: "github", link: "https://github.com/liuyu22/llm-cookbook" },
    ],

    search: {
      provider: "local",
    },

    outline: {
      level: [2, 3],
      label: "本页目录",
    },

    docFooter: {
      prev: "上一篇",
      next: "下一篇",
    },

    lastUpdated: {
      text: "最后更新",
    },

    footer: {
      // message: "用 VitePress 构建 | 代码可在浏览器中运行",
      copyright: "lmillion",
    },
  },

  markdown: {
    lineNumbers: true,
  },

  vite: {
    optimizeDeps: {
      exclude: ["pyodide"],
    },
  },
});
