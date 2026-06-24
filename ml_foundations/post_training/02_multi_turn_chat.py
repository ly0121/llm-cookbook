"""
╔══════════════════════════════════════════════════════════════════╗
║  02_multi_turn_chat.py — 多 turn 对话的 loss mask 策略             ║
║                                                                  ║
║  核心问题：4 turn user/assistant 对话，loss 该对谁算？              ║
║  与生产对应：ShareGPT 风格训练为什么能教模型「承上启下」              ║
╚══════════════════════════════════════════════════════════════════╝
"""
import sys

try:
    import torch
except ImportError:
    print("❌ 缺少依赖 torch，请先运行：pip install torch")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, set_seed
except ImportError:
    print("❌ 缺少依赖 transformers，请先运行：pip install transformers")
    sys.exit(1)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

CONVERSATION = [
    {"role": "user", "content": "推荐一本入门 Python 的书"},
    {"role": "assistant", "content": "《Python Crash Course》很适合入门，习题量大、节奏快。"},
    {"role": "user", "content": "如果我已经会 JavaScript 呢？"},
    {"role": "assistant", "content": "那直接看《Fluent Python》第二版，它假设你已经懂动态语言。"},
]


def build_with_strategy(tokenizer, strategy: str) -> tuple[list[int], list[int]]:
    """
    返回 (input_ids, labels)。
    strategy ∈ {"last_turn_only", "all_assistant", "everything"}
    """
    full_str = tokenizer.apply_chat_template(CONVERSATION, tokenize=False)
    full_ids = tokenizer(full_str, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(full_ids)

    if strategy == "everything":
        labels = list(full_ids)
    elif strategy == "all_assistant":
        for end_idx in range(2, len(CONVERSATION) + 1, 2):  # 2, 4
            prefix_str = tokenizer.apply_chat_template(
                CONVERSATION[:end_idx - 1], tokenize=False, add_generation_prompt=True
            )
            prefix_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
            full_so_far = tokenizer.apply_chat_template(
                CONVERSATION[:end_idx], tokenize=False
            )
            full_so_far_ids = tokenizer(full_so_far, add_special_tokens=False)["input_ids"]
            for i in range(len(prefix_ids), len(full_so_far_ids)):
                labels[i] = full_ids[i]
    elif strategy == "last_turn_only":
        prefix_str = tokenizer.apply_chat_template(
            CONVERSATION[:-1], tokenize=False, add_generation_prompt=True
        )
        prefix_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
        for i in range(len(prefix_ids), len(full_ids)):
            labels[i] = full_ids[i]
    else:
        raise ValueError(strategy)
    return full_ids, labels


def visualize(tokenizer, full_ids: list[int], labels: list[int], label: str) -> None:
    print(f"\n── 策略: {label} ──")
    weight_row = "".join("█" if l != -100 else "·" for l in labels)
    print(f"loss-mask: [{weight_row}]   (█=参与 loss, ·=mask)")
    n_active = sum(1 for l in labels if l != -100)
    print(f"参与 loss 的 token 数 = {n_active} / {len(full_ids)} "
          f"({100 * n_active / len(full_ids):.1f}%)")


def main() -> None:
    set_seed(42)
    torch.manual_seed(42)

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"❌ 下载 tokenizer 失败：{e}")
        print("提示：如网络不通，可设置环境变量 HF_ENDPOINT=https://hf-mirror.com 后重试")
        sys.exit(1)

    print(f"✅ tokenizer: {MODEL_ID}")
    print(f"对话 turn 数：{len(CONVERSATION)}\n")

    for strategy, label in [
        ("everything", "everything（全部算 loss，含 user）"),
        ("last_turn_only", "last_turn_only（仅最后一个 assistant）"),
        ("all_assistant", "all_assistant（所有 assistant turn）"),
    ]:
        ids, labels = build_with_strategy(tokenizer, strategy)
        visualize(tokenizer, ids, labels, label)

    print("\n=== 关键收获 ===")
    print("1. everything：模型同时学怎么提问，浪费容量；几乎没人用")
    print("2. last_turn_only：单轮 SFT 标准做法，但学不到「承上启下」")
    print("3. all_assistant：ShareGPT 风格，所有 assistant 都参与；多轮主流方案")
    print("4. 选哪个取决于训练数据是不是真实多轮对话还是单轮指令")


if __name__ == "__main__":
    main()
