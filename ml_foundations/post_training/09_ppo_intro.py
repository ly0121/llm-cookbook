"""
╔══════════════════════════════════════════════════════════════════╗
║  09_ppo_intro.py — RLHF/PPO 全景与工程债务（不真训）                ║
║                                                                  ║
║  核心问题：为什么 InstructGPT 用 PPO，而 Llama-3 改用 DPO？          ║
║  与生产对应：理解 RLHF 工程负担，知道什么时候选 DPO/ORPO/KTO         ║
╚══════════════════════════════════════════════════════════════════╝
"""
import sys

# ── 保护性导入 ──────────────────────────────────────────────────────

try:
    import torch
except ImportError:
    print("缺少依赖 torch，请先安装：pip install torch")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
except ImportError:
    print("缺少依赖 transformers，请先安装：pip install transformers")
    sys.exit(1)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

FLOWCHART = r"""
                    ┌──────────────────────┐
                    │  Pretrained base LLM │
                    └──────────┬───────────┘
                               │ SFT (Task 5)
                               ▼
                    ┌──────────────────────┐
                    │   SFT model π_SFT    │  ◄── DPO 在这里直接闭环（Task 9）
                    └──────────┬───────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │ Reward Model │   │  Reference   │   │   Critic     │
    │  RM(prompt,  │   │  π_ref =     │   │  V_φ(s)      │
    │  response)→r │   │  copy(π_SFT) │   │ value head   │
    └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
           │                  │                  │
           └────────┬─────────┴──────────────────┘
                    ▼
            ┌───────────────────────────┐
            │   PPO actor π_θ (clip)    │
            │   loss = -E[min(           │
            │     ratio·A, clip·A)]      │
            │     + β·KL(π_θ‖π_ref)      │
            └───────────────────────────┘
"""


def show_flowchart() -> None:
    print(FLOWCHART)


def show_pseudocode() -> None:
    code = r"""
# PPO 单 step 伪代码（参考 trl.PPOTrainer 实现）
for batch in dataloader:
    # 1. rollout：用当前 actor 采样 response
    response = actor.generate(batch["prompt"])

    # 2. 三个模型同时前向
    logits_actor = actor(prompt + response).logits
    logits_ref   = ref_model(prompt + response).logits
    values       = critic(prompt + response)
    rewards_raw  = reward_model(prompt, response)

    # 3. 计算 KL 惩罚后的 reward
    log_ratio = log_softmax(logits_actor) - log_softmax(logits_ref)
    rewards = rewards_raw - β_kl * log_ratio

    # 4. GAE 计算 advantage
    advantages = compute_gae(rewards, values)

    # 5. PPO clip loss
    new_log_probs = log_softmax(actor(...).logits)
    ratio = exp(new_log_probs - log_ratio.detach())
    surr1 = ratio * advantages
    surr2 = clip(ratio, 1-ε, 1+ε) * advantages
    loss_actor = -min(surr1, surr2).mean()
    loss_critic = ((values - returns) ** 2).mean()
    (loss_actor + 0.5*loss_critic).backward()
"""
    print(code)


def inspect_tensor_shapes() -> None:
    """加载 base + 一个 mock RM，print 一次 forward 的张量形状，体感「4 模型副本」。"""
    set_seed(42)
    torch.manual_seed(42)
    print("\n=== 张量形状演示（仅前向一次，不更新参数） ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        actor = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    except Exception as e:
        print(f"⚠️ 模型加载失败：{e}（不影响理解，跳过这一节）")
        print("提示：如在网络受限地区，可设置 HF_ENDPOINT=https://hf-mirror.com 后重试")
        return

    # 同样的模型加载 4 份 → 直接 print 占用
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"  Qwen2.5-0.5B 参数量：{n_params:,} ≈ {n_params / 1e6:.0f}M")
    print(f"  PPO 同时需要：actor + reference + reward + critic ≈ 4×{n_params / 1e6:.0f}M")
    print(f"  显存账：fp16 下 ~{4 * n_params * 2 / 1024**3:.2f} GB（仅参数，未含 optimizer state）")

    prompt = "用一句话解释什么是 attention。"
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True
    )
    # newer transformers may return BatchEncoding; extract tensor
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    print(f"\n  prompt input_ids: {tuple(ids.shape)}")
    with torch.no_grad():
        gen = actor.generate(ids, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print(f"  response output_ids: {tuple(gen.shape)}")
    with torch.no_grad():
        out = actor(gen)
    print(f"  logits 形状: {tuple(out.logits.shape)}")
    print(f"  → 每步 PPO 需要这个 logits 算 4 次（actor/ref，加上 RM/critic 的 forward）")


def compare_table() -> None:
    print("\n=== PPO vs DPO 工程对比 ===")
    rows = [
        ("需要 reward model", "是（要单独训练）", "否（直接用 chosen/rejected pair）"),
        ("模型副本", "4（actor/ref/RM/critic）", "2（actor/ref）"),
        ("训练稳定性", "差（reward hacking / KL 爆炸）", "好（监督学习风格）"),
        ("超参数量", "多（β_kl, clip ε, GAE λ, value coef）", "少（仅 β）"),
        ("实现复杂度", "高（rollout / GAE / clip）", "低（单 loss）"),
        ("数据形态", "(prompt, response) + 标量 reward", "(prompt, chosen, rejected)"),
    ]
    print(f"  {'维度':<16} | {'PPO':<32} | DPO")
    print("  " + "─" * 78)
    for r in rows:
        print(f"  {r[0]:<16} | {r[1]:<32} | {r[2]}")


def main() -> None:
    print("# RLHF / PPO 全景（教学型 demo，不真训）\n")
    show_flowchart()
    show_pseudocode()
    inspect_tensor_shapes()
    compare_table()

    print("\n=== 关键收获 ===")
    print("1. PPO 同时要 4 个模型副本，0.5B × 4 在 Mac 16GB 上已经吃力，7B 必须多卡")
    print("2. RM 训练本身需要数十万人工偏好标注，是 OpenAI/Anthropic 的隐性壁垒")
    print("3. DPO 用 chosen/rejected pair 反推「隐式 RM」，省掉 RM 训练 + critic + GAE")
    print("4. 现代变体（ORPO/SimPO/KTO）进一步去 reference model，但思想沿用 DPO")


if __name__ == "__main__":
    main()
