"""
╔══════════════════════════════════════════════════════════════════╗
║  10_quantization_inference.py — 训练后量化推理对比                  ║
║                                                                  ║
║  核心问题：GGUF Q4_K_M / bnb 8-bit / fp16，部署时怎么选？           ║
║  与生产对应：llama.cpp / Ollama / LM Studio 的量化格式由来           ║
╚══════════════════════════════════════════════════════════════════╝
"""
import shutil
import sys
import time
from pathlib import Path

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
GGUF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
GGUF_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
OUT_DIR = Path(__file__).parent / "runs" / "10_quant"
TEST_PROMPT = "请用一句话解释什么是量化。"


def measure_fp16() -> dict:
    print("\n── 方案 A：fp16 全精度 ──")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # transformers 5.x renamed torch_dtype → dtype
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16
    ).to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    # transformers 5.x: apply_chat_template may return BatchEncoding, not Tensor
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": TEST_PROMPT}],
        return_tensors="pt", add_generation_prompt=True
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    ids = ids.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=64, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    dt = time.perf_counter() - t0
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    del model
    return {"name": "fp16", "memory_mb": mem_mb, "latency_s": dt, "output": text, "params": n}


def have_llama_quantize() -> bool:
    return shutil.which("llama-quantize") is not None


def maybe_quantize_with_llama_cpp() -> None:
    if not have_llama_quantize():
        print("\n⚠️ llama.cpp 未安装，跳过本地量化步骤。")
        print("   安装：brew install llama.cpp（macOS）或 https://github.com/ggerganov/llama.cpp")
        print("   命令示例（安装后可手动尝试）：")
        print(f"     llama-quantize ./qwen.gguf ./qwen-q4_k_m.gguf Q4_K_M")
        return
    print("\n✅ 检测到 llama-quantize，可执行本地量化（本 demo 略过实际跑，因耗时较长）")


def measure_gguf_q4():  # -> dict | None
    print("\n── 方案 B：GGUF Q4_K_M（llama-cpp-python 加载预量化模型） ──")
    try:
        from llama_cpp import Llama
    except ImportError:
        print("⚠️ 未装 llama-cpp-python，跳过此方案。")
        print("   安装：pip install llama-cpp-python")
        return None

    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=GGUF_REPO, filename=GGUF_FILE,
                               local_dir=str(OUT_DIR))
    except Exception as e:
        print(f"⚠️ 下载 GGUF 失败：{e}")
        print("   网络受限提示：export HF_ENDPOINT=https://hf-mirror.com")
        return None

    mem_mb = Path(path).stat().st_size / 1024**2
    llm = Llama(model_path=path, n_ctx=512, verbose=False)
    t0 = time.perf_counter()
    out = llm(f"<|im_start|>user\n{TEST_PROMPT}<|im_end|>\n<|im_start|>assistant\n",
              max_tokens=64, temperature=0.0, stop=["<|im_end|>"])
    dt = time.perf_counter() - t0
    text = out["choices"][0]["text"].strip()
    return {"name": "GGUF Q4_K_M", "memory_mb": mem_mb, "latency_s": dt, "output": text}


def measure_bnb_8bit():  # -> dict | None
    print("\n── 方案 C：bitsandbytes 8-bit ──")
    if not torch.cuda.is_available():
        print("⚠️ bitsandbytes 8-bit 需要 CUDA，Mac/CPU 跳过。")
        print("   降级提示：本路径在 Colab T4 上可直接跑。")
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        print("⚠️ 未装 bitsandbytes，跳过。")
        return None
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb,
                                                 device_map="auto").eval()
    mem_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    # transformers 5.x: apply_chat_template may return BatchEncoding, not Tensor
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": TEST_PROMPT}],
        return_tensors="pt", add_generation_prompt=True
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    ids = ids.to(model.device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=64, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    dt = time.perf_counter() - t0
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return {"name": "bnb 8-bit", "memory_mb": mem_mb, "latency_s": dt, "output": text}


def print_table(results):  # list[dict]
    print("\n=== 三方案对比 ===")
    print(f"  {'方案':<18} | {'权重体积 MB':>14} | {'首 token+64 延迟 s':>20} | 输出前 30 字")
    print("  " + "─" * 92)
    for r in results:
        if r is None:
            continue
        out = (r['output'][:30] + '…') if len(r['output']) > 30 else r['output']
        print(f"  {r['name']:<18} | {r['memory_mb']:>14.1f} | {r['latency_s']:>20.2f} | {out}")


def main() -> None:
    set_seed(42)
    torch.manual_seed(42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    maybe_quantize_with_llama_cpp()

    results = [measure_fp16(), measure_gguf_q4(), measure_bnb_8bit()]
    print_table(results)

    print("\n=== 关键收获 ===")
    print("1. GGUF Q4_K_M = 约 4.5 bit/param 的非均匀量化（K_M 是 k-quants medium）")
    print("2. bnb NF4 / GPTQ / AWQ / GGUF：训练时 vs 推理时；CUDA only vs 全平台")
    print("3. 量化主要省的是「显存 + 加载时间」，延迟通常并不会更快（除非 GPU 带宽瓶颈）")
    print("4. 部署侧主流：Mac → GGUF + llama.cpp，Linux GPU → AWQ/GPTQ + vLLM")


if __name__ == "__main__":
    main()
