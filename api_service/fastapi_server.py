"""
╔══════════════════════════════════════════════════════════════════╗
║         项目十：API 服务化 — FastAPI + LangServe                    ║
║         把你的 LLM 应用从"脚本"变成"可调用的 HTTP 服务"            ║
╚══════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════
【前置科普一：脚本 vs 服务——为什么必须"服务化"？】
═══════════════════════════════════════════════════════════════════

当前我们所有项目都是"跑一次就结束"的脚本：
  python rag_eval.py → 跑完 → 退出

但企业中，LLM 应用需要"随时待命"：

  ┌─────────────────────────────────────────────────────────────┐
  │  脚本模式（当前）：                                          │
  │                                                             │
  │  运行 → 干活 → 结束                                         │
  │  （每次使用都要手动跑一次，只有你能用）                      │
  ├─────────────────────────────────────────────────────────────┤
  │  服务模式（生产）：                                          │
  │                                                             │
  │  启动服务器 → 一直在线监听                                   │
  │  前端/App/其他服务 → 发 HTTP 请求 → 拿到响应                │
  │  （任何人、任何程序都能通过 URL 调用你的 AI 能力）           │
  └─────────────────────────────────────────────────────────────┘

  真实场景：
    ① 手机 App 调用你的"智能客服"接口
    ② 前端网页通过 fetch() 调用你的"文档问答"接口
    ③ 其他微服务通过 HTTP 调用你的"信息提取"接口
    ④ 定时任务 cron 调用你的"报告生成"接口

═══════════════════════════════════════════════════════════════════
【前置科普二：FastAPI——Python 最快的 Web 框架】
═══════════════════════════════════════════════════════════════════

FastAPI 是 Python 生态中性能最强、开发最快的 Web 框架：

  ┌───────────────────┬────────────────────────────────────────┐
  │  特性             │  说明                                   │
  ├───────────────────┼────────────────────────────────────────┤
  │  速度快           │  基于 Starlette/ASGI，性能接近 Go/Node │
  │  类型安全         │  基于 Pydantic（和项目六呼应！）        │
  │  自动文档         │  自动生成 Swagger UI（/docs 页面）      │
  │  原生异步         │  天然支持 async/await                   │
  │  流式支持         │  StreamingResponse（和项目八呼应！）    │
  └───────────────────┴────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
【前置科普三：LangServe——一行代码把 Chain 变成 API】
═══════════════════════════════════════════════════════════════════

LangServe 是 LangChain 官方的"一键服务化"工具：
  它接收一个 LCEL Chain，自动为你生成：
    ① POST /invoke   → 阻塞调用，返回完整结果
    ② POST /stream   → 流式调用，返回 SSE 事件流
    ③ POST /batch    → 批量调用，一次处理多个输入
    ④ GET  /input_schema  → 查看输入格式
    ⑤ GET  /output_schema → 查看输出格式
    ⑥ Playground UI       → 内置调试页面

  本项目演示两种方式：
    方式一：纯 FastAPI 手动实现（理解底层）
    方式二：LangServe 一行代码（生产推荐）

═══════════════════════════════════════════════════════════════════
【运行方式】
═══════════════════════════════════════════════════════════════════

  启动服务器：
    cd api_service
    python fastapi_server.py

  然后用浏览器或 curl 测试：
    浏览器打开：http://localhost:8000/docs （Swagger UI）
    curl 测试：见第 4 章的客户端代码

  ⚠️ 依赖安装：
    pip install fastapi uvicorn sse-starlette langserve
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 【导入区】
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FastAPI 框架核心
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

# Pydantic：请求/响应的数据模型（和项目六的 Structured Output 是同一个库！）
from pydantic import BaseModel, Field

# LangChain 核心
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangServe：一行代码把 Chain 变成 API
from langserve import add_routes

# 标准库
import json
import uvicorn


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 0 章：初始化 LLM + 构建 Chain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API_KEY = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJBUkh6SlZ6Rm9ZZkZXZGdTTDF0Y292MGliRk5YU1J4WiJ9.MEUVU99Rh6CCLsHw4Fu4XcTSJURtbLDNFYxHERnW5qY"
BASE_URL = "https://llm-gateway-proxy.inner.chj.cloud/llm-gateway/v1"
MODEL_NAME = "kivy-kimi-k2_5"

llm = ChatOpenAI(model=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, temperature=0.7)

# ── 构建两条 Chain（模拟不同的 AI 能力接口）──────────────

# Chain 1：通用问答
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个简洁的 AI 助手，回答控制在 100 字以内。"),
    ("human", "{question}"),
])
qa_chain = qa_prompt | llm | StrOutputParser()

# Chain 2：翻译
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个翻译专家。将用户输入的文本翻译为{target_language}。只输出翻译结果，不要额外解释。"),
    ("human", "{text}"),
])
translate_chain = translate_prompt | llm | StrOutputParser()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 1 章：创建 FastAPI 应用
# 目标：搭建 HTTP 服务框架
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── FastAPI() 做了什么？─────────────────────────────────
#
# 创建一个 ASGI 应用实例。ASGI = 异步网关接口，
# 是 Python 异步 Web 应用的标准协议。
#
# 类比：FastAPI() 就像"开了一家餐厅"，
#   后面的 @app.get/post 就是"菜单上的每道菜"。

app = FastAPI(
    title="LangChain AI 服务",
    description="项目十：将 LLM Chain 以 REST API 形式对外提供服务",
    version="1.0.0",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 2 章：手动实现 API 端点（理解底层原理）
# 目标：用纯 FastAPI 手动把 Chain 包装成 HTTP 接口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 请求/响应模型（Pydantic）─────────────────────────────
#
# FastAPI 用 Pydantic 模型定义 API 的输入输出格式：
#   ① 自动验证请求参数（类型不对直接报 422 错误）
#   ② 自动生成 API 文档（Swagger UI 上能看到每个字段的说明）
#   ③ 自动序列化响应（Python 对象 → JSON）
#
# 这和项目六的 Structured Output 原理一模一样！
# 区别是：项目六用 Pydantic 约束 LLM 的输出，
#         这里用 Pydantic 约束 HTTP 请求的输入输出。


class QuestionRequest(BaseModel):
    """问答接口的请求体"""
    question: str = Field(description="用户的问题", examples=["什么是量子计算？"])


class QuestionResponse(BaseModel):
    """问答接口的响应体"""
    question: str = Field(description="原始问题")
    answer: str = Field(description="AI 的回答")
    model: str = Field(description="使用的模型名称")


class TranslateRequest(BaseModel):
    """翻译接口的请求体"""
    text: str = Field(description="要翻译的文本", examples=["Hello, world!"])
    target_language: str = Field(
        default="中文",
        description="目标语言",
        examples=["中文", "English", "日本語"],
    )


class TranslateResponse(BaseModel):
    """翻译接口的响应体"""
    original: str = Field(description="原始文本")
    translated: str = Field(description="翻译结果")
    target_language: str = Field(description="目标语言")


# ── 端点一：问答（POST /api/qa）───────────────────────────
#
# @app.post 装饰器：
#   "/api/qa"     → 这个接口的 URL 路径
#   response_model → 响应的数据结构（自动校验 + 文档生成）
#
# 函数参数 request: QuestionRequest：
#   FastAPI 看到参数类型是 Pydantic 模型，就知道从请求 Body 中解析 JSON。
#   如果客户端发的 JSON 格式不对，FastAPI 自动返回 422 错误。

@app.post("/api/qa", response_model=QuestionResponse, tags=["手动实现"])
async def qa_endpoint(request: QuestionRequest):
    """
    通用问答接口（手动实现版）。

    接收一个问题，调用 LangChain Chain，返回回答。
    """
    try:
        # 调用 Chain（注意：这里用 ainvoke 异步调用，第11章会深入讲）
        answer = await qa_chain.ainvoke({"question": request.question})
        return QuestionResponse(
            question=request.question,
            answer=answer,
            model=MODEL_NAME,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM 调用失败: {str(e)}")


# ── 端点二：翻译（POST /api/translate）────────────────────

@app.post("/api/translate", response_model=TranslateResponse, tags=["手动实现"])
async def translate_endpoint(request: TranslateRequest):
    """
    翻译接口（手动实现版）。

    接收文本和目标语言，返回翻译结果。
    """
    try:
        translated = await translate_chain.ainvoke({
            "text": request.text,
            "target_language": request.target_language,
        })
        return TranslateResponse(
            original=request.text,
            translated=translated,
            target_language=request.target_language,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"翻译失败: {str(e)}")


# ── 端点三：流式问答（GET /api/qa/stream）────────────────
#
# 流式接口用 StreamingResponse 返回 SSE 格式数据。
# 前端用 EventSource 接收（和项目八的 SSE 章节呼应！）。
#
# 为什么流式重要？
#   普通接口：用户等 5 秒才看到结果（体验差）
#   流式接口：0.3 秒开始逐字显示（体验好）

@app.post("/api/qa/stream", tags=["手动实现"])
async def qa_stream_endpoint(request: QuestionRequest):
    """
    流式问答接口（SSE 格式）。

    返回 Server-Sent Events 流，前端可逐 token 接收。
    """
    async def generate_sse():
        """异步生成器：逐 token 产出 SSE 事件"""
        try:
            async for chunk in qa_chain.astream({"question": request.question}):
                if chunk:
                    # SSE 格式：每行以 "data: " 开头，以两个换行结束
                    data = json.dumps({"token": chunk}, ensure_ascii=False)
                    yield f"data: {data}\n\n"
            # 发送结束信号
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 3 章：LangServe 一键服务化（生产推荐方式）
# 目标：用 add_routes() 一行代码自动生成全套 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── LangServe add_routes() 做了什么？──────────────────────
#
# add_routes(app, chain, path="/xxx") 自动为 chain 创建：
#   POST /xxx/invoke         → 阻塞调用
#   POST /xxx/batch          → 批量调用
#   POST /xxx/stream         → 流式调用（SSE）
#   POST /xxx/stream_log     → 流式日志
#   GET  /xxx/input_schema   → 输入格式说明
#   GET  /xxx/output_schema  → 输出格式说明
#   GET  /xxx/playground     → 内置调试 UI 页面
#
# 你不需要写任何端点代码！一行 add_routes 搞定一切！
#
# ⚠️ 前提：Chain 的输入必须是 dict（LCEL 链天然满足）

add_routes(
    app,
    qa_chain,
    path="/langserve/qa",
    # enabled_endpoints 可以控制只暴露哪些端点（安全考虑）
    # enabled_endpoints=["invoke", "stream"],
)

add_routes(
    app,
    translate_chain,
    path="/langserve/translate",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 4 章：健康检查 + 元数据接口
# 目标：生产环境必备的运维端点
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 为什么需要健康检查？──────────────────────────────────
#
# 在 Kubernetes / Docker / 负载均衡环境中：
#   探针会定期请求 /health，判断服务是否存活。
#   如果返回非 200，自动重启容器或摘除流量。
#
# 这是生产部署的"必备标配"接口。

@app.get("/health", tags=["运维"])
async def health_check():
    """
    健康检查接口。

    K8s liveness/readiness probe 会调用此接口。
    返回 200 表示服务正常。
    """
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "endpoints": [
            "/api/qa",
            "/api/translate",
            "/api/qa/stream",
            "/langserve/qa",
            "/langserve/translate",
        ],
    }


@app.get("/", tags=["运维"])
async def root():
    """根路径，展示服务信息"""
    return {
        "service": "LangChain AI 服务",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "api_endpoints": {
            "手动实现（理解底层）": {
                "问答": "POST /api/qa",
                "翻译": "POST /api/translate",
                "流式问答": "POST /api/qa/stream",
            },
            "LangServe（一键生成）": {
                "问答-invoke": "POST /langserve/qa/invoke",
                "问答-stream": "POST /langserve/qa/stream",
                "问答-playground": "GET /langserve/qa/playground",
                "翻译-invoke": "POST /langserve/translate/invoke",
            },
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第 5 章：启动服务
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── uvicorn 是什么？─────────────────────────────────────
#
# uvicorn 是 ASGI 服务器——负责"监听端口 + 接收请求 + 转发给 FastAPI"。
# 类比：
#   FastAPI = 厨师（处理业务逻辑）
#   uvicorn = 服务员（接待客人、传菜）
#
# 生产环境通常：
#   开发：uvicorn app:app --reload（自动重载，方便调试）
#   生产：gunicorn -k uvicorn.workers.UvicornWorker -w 4 app:app
#         （多 worker 进程，充分利用多核 CPU）

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 项目十：FastAPI + LangServe AI 服务")
    print("=" * 60)
    print()
    print("  服务启动中...")
    print()
    print("  📡 可用端点：")
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │  手动实现（理解底层）：                           │")
    print("  │    POST /api/qa          → 通用问答              │")
    print("  │    POST /api/translate    → 文本翻译              │")
    print("  │    POST /api/qa/stream    → 流式问答（SSE）       │")
    print("  ├──────────────────────────────────────────────────┤")
    print("  │  LangServe（一键生成）：                          │")
    print("  │    POST /langserve/qa/invoke     → 问答           │")
    print("  │    POST /langserve/qa/stream     → 流式问答       │")
    print("  │    GET  /langserve/qa/playground  → 调试 UI       │")
    print("  ├──────────────────────────────────────────────────┤")
    print("  │  运维：                                           │")
    print("  │    GET  /health          → 健康检查              │")
    print("  │    GET  /docs            → Swagger UI 文档       │")
    print("  └──────────────────────────────────────────────────┘")
    print()
    print("  💡 测试命令：")
    print('     curl -X POST http://localhost:8000/api/qa \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"question": "什么是量子计算？"}\'')
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
