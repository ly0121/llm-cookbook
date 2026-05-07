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
          text: "一、LLM 基础",
          collapsed: false,
          items: [
            { text: "什么是大语言模型", link: "/llm/" },
            { text: "提示工程", link: "/llm/prompt-engineering" },
            { text: "Tokenization", link: "/llm/tokenization" },
            { text: "文本生成机制", link: "/llm/generation" },
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
