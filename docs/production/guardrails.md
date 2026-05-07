---
title: 安全护栏
---

# Guardrails（AI 安全护栏）

LLM 应用面临提示注入、越狱攻击、信息泄露等安全风险，三层防护架构是生产必备。

## 1. 安全风险全景

| 攻击面 | 风险类型 |
|--------|---------|
| 输入侧 | 提示注入、越狱、对抗样本、超长输入 DoS |
| 模型侧 | 幻觉、偏见、知识泄露 |
| 输出侧 | 有害内容、PII 泄露、虚假信息 |
| 系统侧 | API Key 泄露、权限提升 |

## 2. OWASP LLM Top 10（2024）

| 排名 | 风险 | 说明 |
|------|------|------|
| 1 | Prompt Injection | 通过输入操纵模型行为 |
| 2 | Insecure Output | 未过滤输出导致 XSS |
| 3 | Data Poisoning | 训练数据被投毒 |
| 4 | Model DoS | 消耗资源致服务不可用 |
| 5 | Supply Chain | 第三方组件漏洞 |
| 6 | Info Disclosure | 泄露敏感信息 |

## 3. 三层防护架构

```
用户输入 → [输入防护层] → LLM → [输出防护层] → 用户
              ↓                      ↓
         过滤/清洗              内容审查
         注入检测              PII 脱敏
         长度限制              话题边界
```

## 4. 提示注入防御

| 攻击方式 | 防御策略 |
|---------|---------|
| 直接注入："忽略上面的指令" | System Prompt 强化 + 输入检测 |
| 间接注入：文档中嵌入指令 | 内容隔离 + 可信源标记 |
| 越狱：角色扮演绕过 | 多层检测 + 输出审查 |

```python
# 输入检测示例
INJECTION_PATTERNS = [
    r"忽略.*指令", r"ignore.*instruction",
    r"你现在是", r"假装你是",
    r"system prompt", r"DAN",
]

def check_injection(user_input: str) -> bool:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, user_input, re.IGNORECASE):
            return True
    return False
```

## 5. PII 检测与脱敏

```python
PII_PATTERNS = {
    "phone": r"1[3-9]\d{9}",
    "id_card": r"\d{17}[\dXx]",
    "email": r"[\w.-]+@[\w.-]+\.\w+",
}

def mask_pii(text: str) -> str:
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[{pii_type}_MASKED]", text)
    return text
```

## 6. 话题边界控制

```python
ALLOWED_TOPICS = ["产品咨询", "技术支持", "订单查询"]

topic_check_prompt = """判断用户问题是否属于以下允许话题：
{topics}
如果不属于，返回 "off_topic"。
用户问题：{question}"""
```

## 7. 红队测试

系统上线前进行对抗性测试：

| 测试类型 | 目标 |
|---------|------|
| 注入测试 | 尝试绕过 System Prompt |
| 越狱测试 | 让模型输出禁止内容 |
| 泄露测试 | 提取系统内部信息 |
| 边界测试 | 超长输入、特殊字符 |

## 8. 生产安全清单

- [ ] 输入长度限制（防 DoS）
- [ ] 注入模式检测
- [ ] PII 脱敏（输入+输出）
- [ ] 输出内容审查
- [ ] 话题边界限制
- [ ] API 认证 + 限流
- [ ] 日志审计（不含敏感信息）
- [ ] 定期红队测试

::: warning 需要本地运行
完整实现见 `guardrails/safety_guard.py`。
:::

---

::: tip 下一步
- [评估体系](/production/evaluation) — 评估安全防护效果
- [API 服务](/production/api-service) — 在 API 层集成护栏
:::
