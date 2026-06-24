"""
╔══════════════════════════════════════════════════════════════════╗
║  12_eval_lm_harness.py — lm-evaluation-harness 子集评估             ║
║                                                                  ║
║  核心问题：业界 benchmark（MMLU / ARC 等）实际怎么跑？               ║
║  与生产对应：HF Open LLM Leaderboard / EleutherAI 标配               ║
╚══════════════════════════════════════════════════════════════════╝
"""
import json
import subprocess
import sys
import time
from pathlib import Path

# 守护导入：torch（仅用于设置随机种子，保持公约）
try:
    import torch
except ImportError:
    print("缺少依赖 torch，请先安装：pip install torch")
    sys.exit(1)

# 守护导入：transformers（仅用于 set_seed，保持公约）
try:
    from transformers import set_seed
except ImportError:
    print("缺少依赖 transformers，请先安装：pip install transformers")
    sys.exit(1)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SFT_DIR = Path(__file__).parent / "runs" / "03_sft_full"
OUT_DIR = Path(__file__).parent / "runs" / "12_lm_eval"

# arc_easy 全集 ~2376 题；20 题足以演示流程
EVAL_TASK = "arc_easy"
LIMIT = 20


def check_env() -> None:
    """检查 lm-eval 是否已安装；未安装则打印提示并退出。"""
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        print("❌ 需要 lm-eval：pip install 'lm-eval>=0.4.0'")
        print("安装后再次运行本脚本即可。")
        sys.exit(1)


def run_eval(model_path: str, label: str) -> dict:
    """
    调用 lm_eval CLI 子进程评估指定模型，返回 arc_easy 的结果字典。

    使用基于 mtime 的过滤确保只读取本次运行新生成的 JSON 结果文件。
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{label}.json"
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=float32",
        "--tasks", EVAL_TASK,
        "--limit", str(LIMIT),
        "--output_path", str(out_file),
        "--batch_size", "4",
        "--seed", "42",
    ]
    print(f"\n=== 跑 {label}（{EVAL_TASK}, limit={LIMIT}） ===")
    print(f"命令: {' '.join(cmd)}")
    print("（lm-eval 子进程运行中，无输出至完成；通常 < 2min）")

    # 记录启动时间，用于 mtime 过滤，避免读取遗留 JSON
    t_start = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"⚠️ lm-eval 执行失败：\n{result.stderr[-500:]}")
        print("如下载失败，可尝试: export HF_ENDPOINT=https://hf-mirror.com")
        return {}

    # 通过 mtime 过滤只取本次运行生成的 JSON 文件，避免多次运行后读取旧结果
    for f in OUT_DIR.rglob("*.json"):
        if f.stat().st_mtime < t_start:
            continue  # 跳过早于本次运行的文件
        try:
            data = json.loads(f.read_text())
            if "results" in data:
                return data["results"].get(EVAL_TASK, {})
        except Exception:
            continue
    return {}


def main() -> None:
    # 保持种子公约（此脚本不直接使用 torch/transformers，但保持一致性）
    set_seed(42)
    torch.manual_seed(42)

    check_env()

    print(f"评估任务：{EVAL_TASK}（仅 {LIMIT} 题，演示流程；全集需数小时）\n")

    base_result = run_eval(MODEL_ID, "base")
    sft_result = run_eval(str(SFT_DIR), "sft") if SFT_DIR.exists() else {}

    print("\n=== 子集分数对比 ===")
    print(f"  {'变体':<10} | {'acc':>8} | {'acc_norm':>10}")
    print("  " + "─" * 35)
    print(f"  {'base':<10} | {base_result.get('acc,none', 'N/A'):>8} | "
          f"{base_result.get('acc_norm,none', 'N/A'):>10}")
    if sft_result:
        print(f"  {'SFT':<10} | {sft_result.get('acc,none', 'N/A'):>8} | "
              f"{sft_result.get('acc_norm,none', 'N/A'):>10}")
    else:
        print(f"  {'SFT':<10} | (跳过：未找到 03_sft_full 的产物)")

    print("\n=== 关键收获 ===")
    print("1. lm-eval 是 EleutherAI 维护的统一 benchmark 框架，HF leaderboard 用它")
    print("2. 20 题子集仅演示流程；正式跑 arc/mmlu 全集需要 GPU + 数小时")
    print("3. arc_easy 是 multiple-choice，模型输出概率最高的选项；acc 是准确率")
    print("4. 实际选型时看 6 个标准任务：ARC / HellaSwag / MMLU / TruthfulQA / Winogrande / GSM8K")


if __name__ == "__main__":
    main()
