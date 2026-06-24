"""
╔══════════════════════════════════════════════════════════════════╗
║  06_qlora_mlx.py — Apple MLX 原生 4-bit QLoRA（Mac 主路径）         ║
║                                                                  ║
║  核心问题：为什么 Mac 上 QLoRA 该用 MLX 而非 bitsandbytes？          ║
║  与生产对应：unified memory + Metal kernel 的优势                  ║
╚══════════════════════════════════════════════════════════════════╝

与 Task 6（05_lora_peft.py）对比：
  - 同样是 LoRA，但底层量化引擎完全不同
  - bitsandbytes NF4 针对 CUDA；MLX group-wise 4-bit 针对 Apple Silicon
  - Mac 用户唯一的原生 QLoRA 路径：mlx_lm.convert + mlx_lm.lora

非 Mac 用户（Linux/CUDA）请改跑 07_qlora_peft_bnb.py。
"""
import json
import platform
import subprocess
import sys
from pathlib import Path

# ── 路径常量 ──────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_HERE = Path(__file__).parent
DATA_PATH = _HERE / "data" / "alpaca_mini.jsonl"
OUT_DIR = _HERE / "runs" / "06_qlora_mlx"
MLX_BASE = OUT_DIR / "qwen_mlx_q4"
ADAPTER_DIR = OUT_DIR / "lora_adapter"
MLX_DATA_DIR = OUT_DIR / "mlx_data"


# ── 步骤 0：环境检测 ────────────────────────────────────────────────────────

def check_env() -> None:
    """检查平台与依赖，不满足则友好退出。"""
    # 必须是 macOS
    if platform.system() != "Darwin":
        print("本 demo 仅在 macOS 上有意义（MLX 是 Apple Silicon 专属）")
        print("非 Mac 用户请改跑 07_qlora_peft_bnb.py（Colab/CUDA）")
        sys.exit(0)

    # 区分 Intel Mac（x86_64）vs Apple Silicon（arm64）
    if platform.machine() != "arm64":
        print("MLX requires Apple Silicon (arm64).")
        print("Intel Mac 无法使用 MLX——请改跑 07_qlora_peft_bnb.py（CUDA 路径）。")
        sys.exit(0)

    # 检测 mlx_lm 是否已安装
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print("缺少依赖 mlx-lm，请先安装：pip install 'mlx-lm>=0.20.0'")
        print("安装完成后重新运行本脚本。")
        sys.exit(0)

    print(f"环境检测通过：macOS arm64 + mlx_lm 已安装")


# ── 步骤 1：量化 base model → MLX 4-bit ─────────────────────────────────────

def convert_to_mlx_q4() -> None:
    """用 mlx_lm.convert 把 HuggingFace 模型量化为 MLX group-wise 4-bit。"""
    if MLX_BASE.exists() and any(MLX_BASE.iterdir()):
        print(f"MLX q4 模型已存在，跳过：{MLX_BASE}")
        return

    print(f"\n=== 步骤 1：量化 {MODEL_ID} → MLX 4-bit ===")
    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", MODEL_ID,
        "--mlx-path", str(MLX_BASE),
        "-q",                        # 量化到 4-bit（默认 group_size=64）
    ]
    print("运行命令：", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"mlx_lm.convert 失败（returncode={result.returncode}）")
        print("提示：如遇网络问题，可设置 HF_ENDPOINT=https://hf-mirror.com 后重试。")
        sys.exit(result.returncode)
    print(f"量化完成 → {MLX_BASE}")


# ── 步骤 2：准备 MLX-LM 期望的 JSONL 训练数据 ──────────────────────────────

def prepare_mlx_data() -> None:
    """
    mlx_lm.lora 接受每行 {"text": "..."} 格式（已套好 chat template）。
    从 alpaca_mini.jsonl 转换并分 train / valid。
    """
    train_file = MLX_DATA_DIR / "train.jsonl"
    if train_file.exists():
        print(f"MLX 训练数据已存在，跳过：{MLX_DATA_DIR}")
        return

    if not DATA_PATH.exists():
        print(f"未找到训练数据：{DATA_PATH}")
        print("请先运行 01_data_construction.py 生成 alpaca_mini.jsonl。")
        sys.exit(1)

    MLX_DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # 拼接 instruction + input（如有）
            user_msg = r.get("instruction", "").strip()
            inp = r.get("input", "").strip()
            if inp:
                user_msg = f"{user_msg}\n\n{inp}"
            output = r.get("output", "").strip()
            # Qwen chat template（<|im_start|> / <|im_end|>）
            text = (
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n{output}<|im_end|>"
            )
            rows.append({"text": text})

    if not rows:
        print("alpaca_mini.jsonl 为空，请检查文件内容。")
        sys.exit(1)

    split = max(1, int(0.9 * len(rows)))
    train_rows, valid_rows = rows[:split], rows[split:]

    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(MLX_DATA_DIR / "valid.jsonl", "w", encoding="utf-8") as f:
        for r in valid_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"训练数据写入完毕：train={len(train_rows)} 条，"
        f"valid={len(valid_rows)} 条 → {MLX_DATA_DIR}"
    )


# ── 步骤 3：在 4-bit base 上训练 LoRA adapter ─────────────────────────────

def train_lora() -> None:
    """调用 mlx_lm.lora CLI 训练 200 步（M1/M2 约 2-3 分钟）。"""
    print("\n=== 步骤 2：在 4-bit base 上训 LoRA adapter（200 iters，M-series ~2 min）===")
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", str(MLX_BASE),
        "--seed", "42",
        "--data", str(MLX_DATA_DIR),
        "--train",
        "--iters", "200",
        "--batch-size", "2",
        "--lora-layers", "8",
        "--adapter-path", str(ADAPTER_DIR),
    ]
    print("运行命令：", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"mlx_lm.lora 训练失败（returncode={result.returncode}）")
        sys.exit(result.returncode)
    print(f"LoRA adapter 训练完成 → {ADAPTER_DIR}")


# ── 步骤 4：体积对比报告 ────────────────────────────────────────────────────

def memory_report() -> None:
    """打印量化前后体积对比（fp16 ≈ 988 MB，4-bit ≈ 280 MB）。"""

    def dir_size_mb(d: Path) -> float:
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1024 ** 2

    base_mb = dir_size_mb(MLX_BASE) if MLX_BASE.exists() else 0.0
    adapter_mb = dir_size_mb(ADAPTER_DIR) if ADAPTER_DIR.exists() else 0.0

    print("\n=== 体积对比 ===")
    print(f"  Qwen2.5-0.5B fp16（HuggingFace 缓存参考值）≈ 988 MB")
    print(f"  MLX 4-bit base（本次量化产物）              = {base_mb:.1f} MB")
    print(f"  压缩比                                       ≈ {988 / base_mb:.1f}x" if base_mb > 0 else "  （base 目录不存在，跳过压缩比）")
    print(f"  LoRA adapter                                 = {adapter_mb:.2f} MB")
    if base_mb > 0:
        print(f"  adapter / base                               ≈ {adapter_mb / base_mb * 100:.2f}%")


# ── 主入口 ──────────────────────────────────────────────────────────────────

def main() -> None:
    check_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 子进程 mlx_lm.lora 的 RNG 由 --seed 42 控制（见下方 cmd）
    import random as _random
    _random.seed(42)

    convert_to_mlx_q4()
    prepare_mlx_data()
    train_lora()
    memory_report()

    print("\n=== 关键收获 ===")
    print("1. MLX 量化采用 group-wise 4-bit，与 NF4 思路相近但底层是 Metal kernel，Mac 原生加速。")
    print("2. Apple Silicon unified memory：CPU/GPU 共享物理内存，无需 to(device) 数据拷贝。")
    print("3. LoRA adapter 体积 < base 的 1%，可单独保存并随时插拔——无需重新量化 base。")
    print("4. 想回 HuggingFace 生态：mlx_lm.fuse 可将 adapter merge 回 fp16 safetensors。")
    print("5. bitsandbytes 官方不支持 Apple MPS，Mac 上的 QLoRA 唯一成熟路径就是 MLX。")


if __name__ == "__main__":
    main()
