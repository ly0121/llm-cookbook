"""
╔══════════════════════════════════════════════════════════════════╗
║         项目十（附）：API 客户端测试脚本                              ║
║         用 requests/httpx 调用 FastAPI 服务，验证所有端点            ║
╚══════════════════════════════════════════════════════════════════╝

运行方式：
  1. 先启动服务器：python fastapi_server.py
  2. 另开终端运行：python test_client.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """测试健康检查接口"""
    print("=" * 60)
    print("测试一：健康检查 GET /health")
    print("=" * 60)
    resp = requests.get(f"{BASE_URL}/health")
    print(f"  状态码: {resp.status_code}")
    print(f"  响应: {json.dumps(resp.json(), indent=2, ensure_ascii=False)}")
    print()


def test_qa():
    """测试问答接口"""
    print("=" * 60)
    print("测试二：问答 POST /api/qa")
    print("=" * 60)
    resp = requests.post(
        f"{BASE_URL}/api/qa",
        json={"question": "什么是量子计算？"},
    )
    print(f"  状态码: {resp.status_code}")
    data = resp.json()
    print(f"  问题: {data['question']}")
    print(f"  回答: {data['answer']}")
    print(f"  模型: {data['model']}")
    print()


def test_translate():
    """测试翻译接口"""
    print("=" * 60)
    print("测试三：翻译 POST /api/translate")
    print("=" * 60)
    resp = requests.post(
        f"{BASE_URL}/api/translate",
        json={"text": "人工智能正在改变世界", "target_language": "English"},
    )
    print(f"  状态码: {resp.status_code}")
    data = resp.json()
    print(f"  原文: {data['original']}")
    print(f"  译文: {data['translated']}")
    print(f"  目标语言: {data['target_language']}")
    print()


def test_stream():
    """测试流式问答接口"""
    print("=" * 60)
    print("测试四：流式问答 POST /api/qa/stream")
    print("=" * 60)
    resp = requests.post(
        f"{BASE_URL}/api/qa/stream",
        json={"question": "用一句话解释黑洞"},
        stream=True,
    )
    print(f"  状态码: {resp.status_code}")
    print("  流式输出: ", end="")
    for line in resp.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            data = json.loads(data_str)
            print(data.get("token", ""), end="", flush=True)
    print()
    print()


def test_langserve_invoke():
    """测试 LangServe invoke 接口"""
    print("=" * 60)
    print("测试五：LangServe POST /langserve/qa/invoke")
    print("=" * 60)
    resp = requests.post(
        f"{BASE_URL}/langserve/qa/invoke",
        json={"input": {"question": "什么是深度学习？"}},
    )
    print(f"  状态码: {resp.status_code}")
    data = resp.json()
    print(f"  输出: {data.get('output', data)}")
    print()


def test_langserve_batch():
    """测试 LangServe batch 接口"""
    print("=" * 60)
    print("测试六：LangServe POST /langserve/qa/batch")
    print("=" * 60)
    resp = requests.post(
        f"{BASE_URL}/langserve/qa/batch",
        json={"inputs": [
            {"question": "什么是 GPU？"},
            {"question": "什么是 CPU？"},
        ]},
    )
    print(f"  状态码: {resp.status_code}")
    data = resp.json()
    outputs = data.get("output", data)
    if isinstance(outputs, list):
        for i, out in enumerate(outputs, 1):
            display = out[:60] + "..." if len(str(out)) > 60 else out
            print(f"  [{i}] {display}")
    else:
        print(f"  响应: {outputs}")
    print()


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         API 客户端测试（确保服务器已启动！）              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    try:
        test_health()
        test_qa()
        test_translate()
        test_stream()
        test_langserve_invoke()
        test_langserve_batch()
        print("=" * 60)
        print("🎉 所有测试完成！")
        print("=" * 60)
    except requests.ConnectionError:
        print("❌ 连接失败！请先启动服务器：python fastapi_server.py")
