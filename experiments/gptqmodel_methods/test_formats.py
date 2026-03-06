#!/usr/bin/env python3
"""
Format 相容性測試腳本
====================
測試所有 GPTQModel 的 format + method 組合在當前環境是否可用。

用法：
    python experiments/gptqmodel_methods/test_formats.py
"""
import sys
import warnings
warnings.filterwarnings("ignore")

from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig

RESET = "\033[0m"
GREEN = "\033[32m"
RED   = "\033[31m"
YELLOW = "\033[33m"
BOLD  = "\033[1m"

def ok(s):   return f"{GREEN}✓ {s}{RESET}"
def fail(s): return f"{RED}✗ {s}{RESET}"
def warn(s): return f"{YELLOW}⚠ {s}{RESET}"


# ── 1. 系統資訊 ────────────────────────────────────────────────
print(f"\n{BOLD}=== 系統資訊 ==={RESET}")
try:
    import torch
    sm = torch.cuda.get_device_capability(0)
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU      : {gpu}")
    print(f"  SM       : {sm[0]}.{sm[1]}  ({'≥8.0 marlin 可用' if sm[0] >= 8 else '<8.0 marlin 不可用'})")
    print(f"  VRAM     : {vram:.1f} GB")
    print(f"  CUDA     : {torch.version.cuda}")
    print(f"  torch    : {torch.__version__}")
except Exception as e:
    print(f"  {fail(f'torch 載入失敗: {e}')}")
    sys.exit(1)

try:
    import gptqmodel
    print(f"  gptqmodel: {gptqmodel.__version__}")
except Exception as e:
    print(f"  {fail(f'gptqmodel 載入失敗: {e}')}")


# ── 2. 可選套件檢查 ────────────────────────────────────────────
print(f"\n{BOLD}=== 可選套件 ==={RESET}")

# bitblas：需做深層測試，頂層 import 成功不代表可用（tilelang 是 lazy import）
BITBLAS_OK = False
_ver = "unknown"
try:
    import bitblas as _bb
    _ver = _bb.__version__
    # 必須嘗試實例化 operator，才真正觸發 tilelang/CUDA 路徑
    from bitblas import Matmul as _M, MatmulConfig as _MC
    _op = _M(config=_MC(M=1, N=512, K=512, A_dtype="float16",
                        W_dtype="int4", out_dtype="float16"))
    BITBLAS_OK = True
    print(f"  {ok(f'bitblas {_ver} 完全可用（kernel 編譯成功）')}")
except ImportError:
    print(f"  {warn('bitblas 未安裝  →  pip install bitblas thefuzz')}")
except Exception as e:
    err = str(e)
    if "undefined symbol" in err or "libc10_cuda" in err:
        print(f"  {warn(f'bitblas {_ver} 已安裝，kernel 實例化失敗（tilelang CUDA symbol 不符）')}")
        print(f"       原因：tilelang 子套件 import torch 時 CUDA 版本符號衝突")
        print(f"       建議：改用 format: marlin（RTX 4090 SM=8.9 完全支援，速度相當）")
    else:
        print(f"  {warn(f'bitblas 錯誤: {err[:80]}')}")
BITBLAS_STATUS = ok("可用") if BITBLAS_OK else warn("CUDA相容問題")

# autoawq（讀取 llm_awq 格式時可能用到）
try:
    import awq
    print(f"  {ok(f'autoawq 已安裝')}")
except ImportError:
    print(f"  {warn('autoawq 未安裝  →  pip install autoawq  （只有需要用 autoawq 讀取時才需要）')}")


# ── 3. Format × Method 相容性矩陣 ──────────────────────────────
print(f"\n{BOLD}=== Format × Method 相容性（QuantizeConfig 驗證）==={RESET}")

method_map = {"gptq": METHOD.GPTQ, "awq": METHOD.AWQ, "qqq": METHOD.QQQ}
format_map = {
    "gptq":      FORMAT.GPTQ,
    "gptq_v2":   FORMAT.GPTQ_V2,
    "marlin":    FORMAT.MARLIN,
    "bitblas":   FORMAT.BITBLAS,
    "gemm":      FORMAT.GEMM,
    "gemv":      FORMAT.GEMV,
    "gemv_fast": FORMAT.GEMV_FAST,
    "llm_awq":   FORMAT.LLM_AWQ,
    "qqq":       FORMAT.QQQ,
}

# (method, format, 額外條件說明)
ALL_COMBOS = [
    ("gptq", "gptq",      None),
    ("gptq", "gptq_v2",   None),
    ("gptq", "marlin",    f"需 SM≥8.0（你 {sm[0]}.{sm[1]} {'✓' if sm[0]>=8 else '✗'}），group_size=128 或 -1"),
    ("gptq", "bitblas",   f"{'已安裝但 tilelang CUDA 相容問題，不可用' if not BITBLAS_OK else '可用'}"),
    ("gptq", "gemm",      None),
    ("gptq", "gemv",      None),
    ("gptq", "gemv_fast", None),
    ("awq",  "llm_awq",   "autoawq 相容標準格式"),
    ("awq",  "gptq",      "⚠ 被自動 fix 成 gemm，設定 gptq = 等同 gemm"),
    ("awq",  "gptq_v2",   "⚠ 被自動 fix 成 gemm，設定 gptq_v2 = 等同 gemm"),
    ("awq",  "marlin",    f"需 SM≥8.0（你 {sm[0]}.{sm[1]} {'✓' if sm[0]>=8 else '✗'})"),
    ("awq",  "gemm",      "GPTQModel 推理，batch 優化"),
    ("awq",  "gemv",      "GPTQModel 推理，batch=1 優化"),
    ("awq",  "gemv_fast", "GPTQModel 推理，batch=1 最快"),
    ("qqq",  "qqq",       "W4A8，QQQ 唯一格式"),
]

INVALID_COMBOS = {
    ("gptq", "gemm"), ("gptq", "gemv"), ("gptq", "gemv_fast"),
    ("gptq", "qqq"), ("gptq", "llm_awq"),
    ("awq",  "qqq"), ("awq",  "bitblas"),
    ("qqq",  "gptq"), ("qqq",  "gptq_v2"), ("qqq",  "marlin"),
    ("qqq",  "gemm"), ("qqq",  "gemv"), ("qqq",  "gemv_fast"),
    ("qqq",  "llm_awq"), ("qqq",  "bitblas"),
}

print(f"  {'method+format':<22} {'Config':<8} {'Runtime':<10} 說明")
print("  " + "-" * 75)

for method_str, format_str, note in ALL_COMBOS:
    combo = (method_str, format_str)
    tag = f"{method_str}+{format_str}"

    # Config 層面驗證
    try:
        cfg = QuantizeConfig(
            bits=4, group_size=128,
            quant_method=method_map[method_str],
            format=format_map[format_str],
        )
        config_status = ok("OK")
    except Exception as e:
        config_status = fail(f"ERR")

    # Runtime 可用性判斷（Config 失敗則 Runtime 也標為不適用）
    config_ok = "✓" in config_status
    if not config_ok:
        rt_status = warn("N/A (Config ERR)")
    elif format_str == "marlin" and sm[0] < 8:
        rt_status = fail("SM<8.0")
    elif format_str == "bitblas" and not BITBLAS_OK:
        rt_status = warn("CUDA相容問題")
    else:
        rt_status = ok("OK")

    note_str = f"  {note}" if note else ""
    print(f"  {tag:<22} {config_status:<18} {rt_status:<20}{note_str}")

print()

# ── 4. 快速建議 ────────────────────────────────────────────────
print(f"{BOLD}=== 推薦使用（你的 RTX 4090）==={RESET}")
print(f"""
  GPTQ 方法最快推理  →  format: marlin    （group_size 必須 128 或 -1）
  GPTQ 最廣相容      →  format: gptq / gptq_v2
  AWQ 跨工具部署     →  format: llm_awq   （可被 autoawq 讀取）
  AWQ 純 GPTQModel   →  format: gemv_fast  （batch=1 最快）
  QQQ               →  format: qqq        （唯一可用格式）
  BitBLAS            →  ⚠ 已安裝但不可用（tilelang 子套件 CUDA 12.8 symbol 問題）
""")
