# LLM 错误处理与重试（Error Handling）完全知识手册

> 本文档系统性地覆盖 LLM 应用中容错机制的所有核心知识点。
> 配合 `retry_demo.py` 代码阅读效果更佳。

---

## 目录

1. [LLM API 调用的错误分类](#1-llm-api-调用的错误分类)
2. [可重试错误 vs 不可重试错误](#2-可重试错误-vs-不可重试错误)
3. [重试策略](#3-重试策略)
4. [Tenacity 库使用详解](#4-tenacity-库使用详解)
5. [断路器模式](#5-断路器模式circuit-breaker)
6. [降级方案](#6-降级方案fallback)
7. [超时控制](#7-超时控制)
8. [速率限制错误的特殊处理](#8-速率限制错误429的特殊处理)
9. [输出解析错误与自动修复](#9-输出解析错误与自动修复)
10. [流式调用的错误处理](#10-流式调用的错误处理)
11. [幂等性设计](#11-幂等性设计)
12. [错误监控与告警](#12-错误监控与告警)
13. [全局异常处理架构](#13-全局异常处理架构)
14. [生产环境容错最佳实践](#14-生产环境容错最佳实践)

---

## 1. LLM API 调用的错误分类

### 1.1 错误全景图

```
对应 retry_demo.py 前置科普一

  ┌─────────────────────────────────────────────────────────────┐
  │                  LLM API 错误分类                             │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  网络层错误                                                  │
  │  ├── ConnectionError    DNS 解析失败/连接被拒               │
  │  ├── TimeoutError       连接超时/读取超时                   │
  │  └── SSLError           证书问题                            │
  │                                                             │
  │  API 层错误（HTTP 状态码）                                   │
  │  ├── 400 Bad Request    请求格式错误                        │
  │  ├── 401 Unauthorized   API Key 无效                        │
  │  ├── 403 Forbidden      权限不足                            │
  │  ├── 404 Not Found      模型/端点不存在                     │
  │  ├── 429 Rate Limited   请求频率超限                        │
  │  ├── 500 Internal Error 服务端内部错误                      │
  │  ├── 502 Bad Gateway    网关错误                            │
  │  └── 503 Unavailable    服务暂时不可用                      │
  │                                                             │
  │  业务层错误                                                  │
  │  ├── ContentFilterError   内容安全过滤                      │
  │  ├── ContextLengthError   超过上下文窗口                    │
  │  ├── OutputParserError    输出格式解析失败                  │
  │  └── InvalidResponseError 响应结构异常                      │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
```

### 1.2 错误频率分布（典型生产环境）

```
  错误类型          占比        处理优先级
  ─────────────────────────────────────────
  429 Rate Limit    40-50%      高（最常见）
  Timeout           20-30%      高
  500/502/503       15-20%      中
  网络错误          5-10%       中
  解析错误          3-5%        低
  认证错误          <1%         低（配置问题）
```

---

## 2. 可重试错误 vs 不可重试错误

### 2.1 判断标准

```
核心原则：暂时性错误 → 重试；永久性错误 → 不重试

  ┌──────────────────┬────────────┬──────────────────────────┐
  │  错误类型         │ 是否重试   │ 原因                      │
  ├──────────────────┼────────────┼──────────────────────────┤
  │  429 Rate Limit  │ ✅ 重试    │ 等一会儿就好了            │
  │  500 Server Err  │ ✅ 重试    │ 服务端可能自动恢复        │
  │  502 Bad Gateway │ ✅ 重试    │ 暂时性网关问题            │
  │  503 Unavailable │ ✅ 重试    │ 服务暂时过载              │
  │  ConnectionError │ ✅ 重试    │ 网络抖动                  │
  │  Timeout         │ ✅ 重试    │ 可能是暂时过载            │
  ├──────────────────┼────────────┼──────────────────────────┤
  │  400 Bad Request │ ❌ 不重试  │ 请求本身有问题，重试也没用 │
  │  401 Unauthorized│ ❌ 不重试  │ 密钥错了，重试 100 次也错 │
  │  403 Forbidden   │ ❌ 不重试  │ 权限不足                  │
  │  404 Not Found   │ ❌ 不重试  │ 端点/模型不存在           │
  │  ContentFilter   │ ❌ 不重试  │ 内容违规                  │
  │  ContextTooLong  │ ❌ 不重试  │ 需要截断输入              │
  └──────────────────┴────────────┴──────────────────────────┘
```

### 2.2 代码判断逻辑

```python
RETRYABLE_STATUS_CODES = {429, 500, 502, 503}
RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, IOError)

def should_retry(error):
    """判断是否应该重试"""
    if isinstance(error, RETRYABLE_EXCEPTIONS):
        return True
    if hasattr(error, 'status_code'):
        return error.status_code in RETRYABLE_STATUS_CODES
    return False
```

---

## 3. 重试策略

### 3.1 固定间隔重试

```
策略: 每次失败后等待固定时间再重试

  失败 → 等 2s → 重试 → 失败 → 等 2s → 重试 → 成功

优点: 实现简单
缺点: 所有客户端同时重试（雷群效应）
适用: 错误不频繁、客户端数量少的场景
```

### 3.2 指数退避（Exponential Backoff）

```
策略: 等待时间按指数增长

  delay = base_delay × 2^(attempt-1)

  第1次失败 → 等 1s  → 重试
  第2次失败 → 等 2s  → 重试
  第3次失败 → 等 4s  → 重试
  第4次失败 → 等 8s  → 重试
  第5次失败 → 放弃

对应 retry_demo.py 中 ResilientInvoker:
  delay = self.base_delay * (2 ** (attempt - 1))

优点: 给服务端恢复时间，避免持续施压
缺点: 仍有雷群效应（所有客户端在同一时刻重试）
```

### 3.3 指数退避 + 抖动（Jitter）

```
策略: 在指数退避基础上加随机偏移

  delay = base_delay × 2^(attempt-1) + random(0, jitter)

对应 retry_demo.py:
  delay = self.base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)

  时间轴:
  客户端A:  ─────┤ 1.2s ├──────┤ 2.4s ├──────
  客户端B:  ─────┤ 1.0s ├────────┤ 2.8s ├────
  客户端C:  ─────┤ 1.5s ├──┤ 2.1s ├──────────

  重试时刻被打散 → 避免雷群效应（Thundering Herd）

对应 LangChain:
  chain.with_retry(wait_exponential_jitter=True)
```

### 3.4 三种策略对比

```
  ┌───────────────┬──────────┬──────────┬──────────────────┐
  │ 策略          │ 复杂度   │ 效果     │ 适用场景          │
  ├───────────────┼──────────┼──────────┼──────────────────┤
  │ 固定间隔      │ 低       │ 一般     │ 单客户端开发环境  │
  │ 指数退避      │ 中       │ 好       │ 多客户端生产环境  │
  │ 指数退避+抖动 │ 中       │ 最好     │ 高并发生产环境    │
  └───────────────┴──────────┴──────────┴──────────────────┘
```

---

## 4. Tenacity 库使用详解

### 4.1 基础用法

```python
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)
import logging

logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),                    # 最多重试3次
    wait=wait_exponential(multiplier=1, max=10),   # 指数退避，最大10s
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),  # 只重试这些
    before_sleep=before_sleep_log(logger, logging.WARNING),  # 重试前记日志
)
def call_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### 4.2 高级配置

```python
from tenacity import (
    retry, stop_after_attempt, stop_after_delay,
    wait_exponential_jitter, retry_if_exception,
    RetryCallState
)

def is_retryable(exception):
    """自定义重试判断函数"""
    if hasattr(exception, 'status_code'):
        return exception.status_code in {429, 500, 502, 503}
    return isinstance(exception, (ConnectionError, TimeoutError))

def custom_before_sleep(retry_state: RetryCallState):
    """重试前的回调: 记录日志 + 告警"""
    attempt = retry_state.attempt_number
    error = retry_state.outcome.exception()
    logger.warning(f"第{attempt}次失败: {error}, 即将重试...")

@retry(
    stop=(stop_after_attempt(5) | stop_after_delay(30)),  # 5次或30秒
    wait=wait_exponential_jitter(initial=1, max=10, jitter=2),
    retry=retry_if_exception(is_retryable),
    before_sleep=custom_before_sleep,
    reraise=True,  # 最终失败时抛出原始异常
)
def robust_llm_call(prompt):
    ...
```

### 4.3 与 LangChain .with_retry() 的对比

```
┌────────────────────┬──────────────────────────────────────┐
│ LangChain          │ Tenacity                             │
│ .with_retry()      │ @retry 装饰器                         │
├────────────────────┼──────────────────────────────────────┤
│ 简单，3行配置      │ 功能强大，高度可定制                  │
│ 只支持 Runnable    │ 任何函数都能用                        │
│ 配置项较少         │ 配置项丰富（stop/wait/retry/callback）│
│ 内置到链中         │ 独立库，需额外安装                    │
└────────────────────┴──────────────────────────────────────┘

对应 retry_demo.py 第1章的 .with_retry() 用法:
  chain.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
```

---

## 5. 断路器模式（Circuit Breaker）

### 5.1 原理

```
灵感来源: 家里的电路保险丝——电流过大时自动断开，保护整个电路

  ┌──────────────────────────────────────────────────────────┐
  │                 断路器三种状态                             │
  │                                                          │
  │   ┌────────┐   失败率>阈值   ┌────────┐                 │
  │   │ CLOSED │ ─────────────→ │  OPEN  │                  │
  │   │ (正常) │                 │ (熔断) │                  │
  │   └────┬───┘ ←───────────── └────┬───┘                  │
  │        │       成功重置           │                      │
  │        │                    超时后尝试                    │
  │        │                         ↓                      │
  │        │                   ┌──────────┐                  │
  │        └───── 成功 ←────── │HALF-OPEN │                  │
  │                            │ (试探)   │                  │
  │                            └──────────┘                  │
  │                                 │                        │
  │                            失败 → 回到 OPEN              │
  └──────────────────────────────────────────────────────────┘

  CLOSED: 正常工作，请求正常通过
  OPEN:   熔断状态，所有请求直接返回错误（不调用 API）
  HALF-OPEN: 允许少量请求通过，测试服务是否恢复
```

### 5.2 实现示例

```python
import time

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = "CLOSED"
        self.last_failure_time = 0

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"  # 尝试恢复
            else:
                raise CircuitBreakerOpen("服务暂时不可用，请稍后重试")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
```

### 5.3 应用场景

```
什么时候用断路器:
  - LLM 服务完全宕机（连续失败 5+ 次）
  - 避免无意义的重试浪费用户时间
  - 保护下游服务不被雪崩式请求压垮

与重试的关系:
  重试: 处理暂时性故障（1-3次失败）
  断路器: 处理持续性故障（5+次连续失败）

  请求 → 断路器判断 → 如果 CLOSED → 重试逻辑 → 调用 API
                    → 如果 OPEN → 直接降级响应
```

---

## 6. 降级方案（Fallback）

### 6.1 降级策略层次

```
对应 retry_demo.py 第3章

  ┌────────────────────────────────────────────────────────────┐
  │  降级策略优先级（从高到低）                                  │
  │                                                            │
  │  Level 1: 备用模型                                         │
  │    GPT-4 不可用 → 切换 GPT-3.5-turbo                      │
  │    延迟/成本更低，但能力也更弱                              │
  │                                                            │
  │  Level 2: 缓存回答                                         │
  │    如果缓存中有相似问题的答案 → 直接返回                    │
  │    答案可能不够精确，但总比没有好                            │
  │                                                            │
  │  Level 3: 规则引擎响应                                     │
  │    预定义的关键词→回答映射                                  │
  │    "退货" → "请联系客服热线 400-xxx-xxxx"                  │
  │                                                            │
  │  Level 4: 兜底消息                                         │
  │    "抱歉，服务暂时不可用，请稍后重试"                       │
  │    最后的保底，确保用户不会看到报错                          │
  └────────────────────────────────────────────────────────────┘
```

### 6.2 LangChain .with_fallbacks() 用法

```python
# 对应 retry_demo.py 第3章

from langchain_openai import ChatOpenAI

# 主力模型
primary = ChatOpenAI(model="gpt-4o", temperature=0.7)

# 备用模型
backup = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 组装降级链
chain = prompt | primary.with_fallbacks([backup]) | parser

# 执行顺序:
#   primary 成功 → 返回结果
#   primary 失败 → 自动尝试 backup → 返回结果
#   backup 也失败 → 抛出异常
```

### 6.3 完整容错调用器

```python
# 对应 retry_demo.py 第4章 ResilientInvoker

class ResilientInvoker:
    """重试 → 降级 → 兜底 的完整容错方案"""

    def invoke(self, input_dict):
        # 阶段一: 重试主链 (指数退避)
        for attempt in range(max_retries):
            try:
                return primary_chain.invoke(input_dict)
            except RetryableError:
                wait(exponential_backoff + jitter)

        # 阶段二: 尝试降级方案
        try:
            return fallback_chain.invoke(input_dict)
        except Exception:
            pass

        # 阶段三: 返回兜底响应
        return "抱歉，服务暂时不可用"
```

---

## 7. 超时控制

### 7.1 两种超时

```
  ┌──────────────────────────────────────────────────────────┐
  │  连接超时（Connection Timeout）                           │
  │    定义: 建立 TCP 连接的最大等待时间                       │
  │    典型值: 5-10 秒                                       │
  │    超过说明: 网络不通 / DNS 失败 / 服务端端口未开          │
  │                                                          │
  │  读取超时（Read Timeout）                                 │
  │    定义: 连接建立后，等待响应数据的最大时间                │
  │    典型值: 30-120 秒（LLM 生成较慢）                      │
  │    超过说明: LLM 生成时间过长 / 服务端过载                │
  └──────────────────────────────────────────────────────────┘

  时间轴:
  ├─── 连接超时 ───┤─────────── 读取超时 ────────────────────┤
  [发起请求]      [连接成功]                           [收到响应]
```

### 7.2 设置方式

```python
# OpenAI Python SDK
client = OpenAI(
    timeout=httpx.Timeout(
        connect=5.0,     # 连接超时 5s
        read=60.0,       # 读取超时 60s
        write=10.0,      # 写入超时 10s
        pool=10.0,       # 连接池超时 10s
    )
)

# LangChain
llm = ChatOpenAI(
    request_timeout=60,  # 总超时 60s
)

# 异步超时控制
import asyncio
try:
    result = await asyncio.wait_for(
        llm.ainvoke(prompt),
        timeout=30.0
    )
except asyncio.TimeoutError:
    # 超时处理
    result = fallback_response()
```

### 7.3 超时设置建议

```
  场景                  连接超时    读取超时
  ──────────────────────────────────────────
  短文本生成(< 200t)    5s         30s
  长文本生成(> 1000t)   5s         120s
  流式输出              5s         120s (首 token)
  Embedding 计算        5s         15s
  Function Calling      5s         60s
```

---

## 8. 速率限制错误（429）的特殊处理

### 8.1 理解速率限制

```
API 提供商的限制维度:
  RPM (Requests Per Minute):  每分钟请求数
  TPM (Tokens Per Minute):    每分钟 Token 数
  RPD (Requests Per Day):     每日请求数

  超过任一限制 → 返回 429 Too Many Requests

响应头中的关键信息:
  Retry-After: 5              # 建议等待 5 秒后重试
  X-RateLimit-Remaining: 0    # 剩余配额为 0
  X-RateLimit-Reset: 1234567  # 配额重置时间戳
```

### 8.2 处理策略

```python
import time

def handle_rate_limit(error):
    """专门处理 429 错误"""
    # 方案一: 使用 Retry-After 头
    retry_after = getattr(error, 'headers', {}).get('Retry-After')
    if retry_after:
        time.sleep(float(retry_after))
        return

    # 方案二: 指数退避
    # 429 通常需要等更长时间（比普通错误）
    time.sleep(min(60, base_delay * 2 ** attempt))

# 方案三: 请求队列 + 令牌桶限流
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=50, period=60)  # 每分钟最多 50 次
def rate_limited_call(prompt):
    return llm.invoke(prompt)
```

### 8.3 预防策略

```
1. 预估用量，选择合适的 Tier
2. 实现客户端限流（不要依赖服务端限流）
3. 批量请求合并（多个小请求 → 一个大请求）
4. 错峰调用（非高峰期预计算）
5. 多 API Key 轮换（不同 key 有独立配额）
```

---

## 9. 输出解析错误与自动修复

### 9.1 常见解析错误

```
场景: LLM 输出不符合预期格式

  期望 JSON:  {"name": "张三", "age": 28}
  实际输出1:  ```json\n{"name": "张三", "age": 28}\n```  ← 多了代码块标记
  实际输出2:  {"name": "张三", "age": "二十八"}  ← 类型错误
  实际输出3:  好的，以下是结果：{"name":...}     ← 多了前缀文本
  实际输出4:  {"name": "张三", "age": 28,}       ← 多了逗号（非法JSON）
```

### 9.2 自动修复策略

```python
import json
import re

def robust_json_parse(text):
    """健壮的 JSON 解析，处理常见格式问题"""

    # 策略1: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 策略2: 提取代码块中的 JSON
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    # 策略3: 提取第一个 {...} 或 [...]
    match = re.search(r'[\{\[].*[\}\]]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    # 策略4: 用 LLM 修复（OutputFixingParser 的思路）
    # 把错误的输出和报错信息发给 LLM，让它修复
    raise ParseError(f"无法解析: {text[:100]}")
```

### 9.3 LangChain OutputFixingParser

```python
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser

# 原始 parser
base_parser = PydanticOutputParser(pydantic_object=MyModel)

# 包装成自修复 parser
fixing_parser = OutputFixingParser.from_llm(
    parser=base_parser,
    llm=llm,  # 用 LLM 修复格式错误
)

# 工作流程:
#   LLM 输出 → base_parser 解析
#     成功 → 返回结果
#     失败 → 把输出+错误信息发给 LLM → LLM 修复 → 再次解析
```

---

## 10. 流式调用的错误处理

### 10.1 流式特殊性

```
非流式: 一次性返回完整结果，错误在调用时就能捕获
流式:   逐 token 返回，错误可能在中途出现

  ┌──────────────────────────────────────────────┐
  │  非流式:                                      │
  │  请求 ──→ [等待] ──→ 完整响应 或 错误         │
  │                                              │
  │  流式:                                        │
  │  请求 ──→ token1 → token2 → ... → 中途断开!  │
  │                                              │
  │  问题: 已经输出了一半给用户，突然断了怎么办？  │
  └──────────────────────────────────────────────┘
```

### 10.2 处理策略

```python
def safe_stream(prompt, max_retries=3):
    """安全的流式调用，支持断点续传"""
    collected_text = ""

    for attempt in range(max_retries):
        try:
            stream = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    collected_text += content
                    yield content  # 逐步返回给用户
            return  # 正常结束

        except Exception as e:
            if attempt < max_retries - 1:
                # 从断点续传: 把已有内容作为上下文继续生成
                prompt = f"请继续以下内容（从断点处继续）:\n{collected_text}"
                yield "\n[连接中断，正在恢复...]\n"
            else:
                yield "\n[生成中断，请重试]"
```

---

## 11. 幂等性设计

### 11.1 什么是幂等性

```
幂等性: 同一操作执行多次，结果和执行一次相同

  非幂等操作:
    "发送邮件" → 重试 3 次 = 发了 3 封邮件 (灾难!)

  幂等操作:
    "查询余额" → 重试 3 次 = 得到同一个余额 (安全)

LLM 场景:
  查询类 (天然幂等):  "什么是AI?" → 重试安全
  操作类 (需要设计):  "帮我发一封邮件" → 重试可能重复发送!
```

### 11.2 幂等性设计方案

```python
import hashlib
from datetime import datetime

class IdempotentLLMCaller:
    """幂等的 LLM 调用器"""

    def __init__(self):
        self.executed_requests = {}  # request_id → result

    def call(self, prompt, request_id=None):
        """
        幂等调用: 同一 request_id 只执行一次
        """
        # 生成幂等键
        if request_id is None:
            request_id = hashlib.md5(prompt.encode()).hexdigest()

        # 检查是否已执行过
        if request_id in self.executed_requests:
            return self.executed_requests[request_id]

        # 首次执行
        result = llm.invoke(prompt)
        self.executed_requests[request_id] = result
        return result
```

---

## 12. 错误监控与告警

### 12.1 关键监控指标

```
  ┌─────────────────────────────────────────────────────────┐
  │  指标            │ 正常范围    │ 告警阈值              │
  ├─────────────────────────────────────────────────────────┤
  │  错误率          │ < 1%       │ > 5% (P1告警)         │
  │  429 频率        │ < 10/min   │ > 50/min              │
  │  平均重试次数    │ < 0.1      │ > 0.5                 │
  │  断路器打开次数  │ 0          │ > 0 (立即告警)        │
  │  降级触发次数    │ < 5/hour   │ > 20/hour             │
  │  超时率          │ < 2%       │ > 10%                 │
  │  P99 延迟        │ < 5s       │ > 15s                 │
  └─────────────────────────────────────────────────────────┘
```

### 12.2 告警分级

```
P1 (立即处理):
  - 断路器打开 (服务完全不可用)
  - 错误率 > 50%
  - 所有降级方案都失败

P2 (30分钟内处理):
  - 错误率 > 5%
  - 429 频率持续飙升
  - 降级触发频繁

P3 (下一个工作日):
  - 错误率 > 1%
  - 偶发超时
  - 解析错误增多
```

---

## 13. 全局异常处理架构

### 13.1 分层异常处理

```
  ┌─────────────────────────────────────────────────────────┐
  │  Layer 4: 全局异常处理器 (兜底)                          │
  │    try: ... except Exception: return default_response   │
  ├─────────────────────────────────────────────────────────┤
  │  Layer 3: 断路器 (防雪崩)                               │
  │    连续失败 5 次 → 熔断 30s → 半开试探                  │
  ├─────────────────────────────────────────────────────────┤
  │  Layer 2: 重试 + 降级 (处理暂时性故障)                   │
  │    重试 3 次 → 降级到备用模型 → 缓存回答                │
  ├─────────────────────────────────────────────────────────┤
  │  Layer 1: 输入验证 (预防错误)                            │
  │    检查 token 长度、格式合法性、敏感词过滤              │
  └─────────────────────────────────────────────────────────┘
```

### 13.2 错误处理决策树

```
  收到错误
    │
    ├─ 是 4xx 客户端错误？
    │    ├─ 401/403 → 告警 + 检查配置
    │    ├─ 400 → 记录日志 + 返回错误提示
    │    └─ 429 → 读取 Retry-After → 等待后重试
    │
    ├─ 是 5xx 服务端错误？
    │    └─ 指数退避重试 (最多 3 次)
    │         ├─ 成功 → 返回结果
    │         └─ 失败 → 降级方案
    │
    ├─ 是网络错误？
    │    └─ 立即重试 1 次 → 指数退避重试
    │
    ├─ 是超时？
    │    └─ 增加超时时间重试 → 降级到更快的模型
    │
    └─ 是解析错误？
         └─ OutputFixingParser → 重新生成 → 返回原始文本
```

---

## 14. 生产环境容错最佳实践

### 14.1 完整容错架构

```
  用户请求
      ↓
  ┌──────────────────────────────────────────┐
  │  输入验证 & 预处理                        │
  │  - Token 长度检查                         │
  │  - 敏感词过滤                             │
  │  - 请求去重 (幂等性)                      │
  └─────────────────────┬────────────────────┘
                        ↓
  ┌──────────────────────────────────────────┐
  │  断路器检查                               │
  │  - OPEN → 直接降级                        │
  │  - CLOSED → 继续                          │
  └─────────────────────┬────────────────────┘
                        ↓
  ┌──────────────────────────────────────────┐
  │  带重试的 LLM 调用                        │
  │  - 指数退避 + 抖动                        │
  │  - 最多 3 次                              │
  │  - 超时 30s                               │
  └─────────────────────┬────────────────────┘
                   成功 ↓ ↓ 失败
              返回结果   ↓
  ┌──────────────────────────────────────────┐
  │  降级方案                                 │
  │  1. 备用模型                              │
  │  2. 缓存回答                              │
  │  3. 规则响应                              │
  │  4. 兜底消息                              │
  └──────────────────────────────────────────┘
```

### 14.2 生产要点清单

```
对应 retry_demo.py 总结部分

  ┌──────────────────────────────────────────────────────────┐
  │  生产容错清单                                             │
  │                                                          │
  │  □ 只对暂时性错误重试（429/500/网络错误）                 │
  │  □ 使用指数退避 + 抖动，避免雷群效应                      │
  │  □ 设置最大重试次数（通常 3 次）+ 最大等待时间            │
  │  □ 429 错误读取 Retry-After 头                           │
  │  □ 实现断路器，防止雪崩                                   │
  │  □ 准备多级降级方案（备用模型→缓存→兜底）                 │
  │  □ 设置合理的超时（连接 5s + 读取 60s）                   │
  │  □ 记录所有错误和重试日志                                 │
  │  □ 监控错误率，设置告警                                   │
  │  □ 对操作类请求实现幂等性                                 │
  │  □ 定期演练故障恢复（Chaos Engineering）                  │
  └──────────────────────────────────────────────────────────┘
```

---

## 附录 A：本文件夹代码与知识点对应

| 代码位置 | 覆盖知识点 | 对应本文档章节 |
|---------|-----------|---------------|
| `retry_demo.py` 前置科普 | 错误分类、重试策略 | 第1节、第3节 |
| `retry_demo.py` 第1章 | .with_retry() 用法 | 第3节、第4节 |
| `retry_demo.py` 第2章 | 模拟错误、重试可视化 | 第3节 |
| `retry_demo.py` 第3章 | .with_fallbacks() 降级 | 第6节 |
| `retry_demo.py` 第4章 | ResilientInvoker 完整方案 | 第6节、第14节 |

---

## 附录 B：推荐学习路径

```
入门（1天）：
  第1-2节 → 理解错误分类和重试判断
  第3节 → 运行 retry_demo.py 第1-2章
  第6节 → 理解降级方案

进阶（2-3天）：
  第4节 → 学习 Tenacity 库
  第5节 → 理解断路器模式
  第7-8节 → 超时和限流处理

生产（1周）：
  第11-14节 → 幂等性、监控、全局架构
  搭建完整的容错框架
```

---

> **下一步学习**：前往 `observability/KNOWLEDGE.md` 了解如何监控这些错误和重试行为，或前往 `caching/KNOWLEDGE.md` 学习用缓存减少 API 调用失败的概率。
