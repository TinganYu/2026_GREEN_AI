#!/usr/bin/env python3
"""
測試 AWQ / QQQ 支援哪些 group_size
用 subprocess 隔離每個測試，避免 kernel thread 崩潰影響整體流程
"""
import sys
import time
import shutil
import subprocess
import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_BASE = ROOT / "quantized_test" / "group_size_test"

GROUP_SIZES = [-1, 16, 32, 64, 128, 256]
METHODS = [
    ("awq", "gemm", False, 0.05),
    ("qqq", "qqq",  True,  0.005),
]

# ── 單一測試 worker（被 subprocess 呼叫）────────────────────────────────────
WORKER_CODE = '''
import sys, gc, json
from pathlib import Path
ROOT = Path("{root}")
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv()

method  = "{method}"
gs      = {gs}
fmt     = "{fmt}"
desc_act= {desc_act}
damp    = {damp}
out_dir = "{out_dir}"

def make_calib():
    base = ("The quick brown fox jumps over the lazy dog. "
            "Quantization reduces model size by representing weights with fewer bits. "
            "Large language models benefit from post-training quantization. ")
    return [base * 4 + f" Sample {{i}}." for i in range(64)]

try:
    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import QuantizeConfig, FORMAT, METHOD
    import torch

    METHOD_MAP = {{"awq": METHOD.AWQ, "qqq": METHOD.QQQ}}
    FORMAT_MAP  = {{"gemm": FORMAT.GEMM, "qqq": FORMAT.QQQ}}

    qc = QuantizeConfig(
        quant_method=METHOD_MAP[method],
        bits=4,
        group_size=gs,
        format=FORMAT_MAP[fmt],
        desc_act=desc_act,
        damp_percent=damp,
    )
    print(f"[CONFIG OK] {{method.upper()}} g={{gs}}", flush=True)

    calib = make_calib()
    model = GPTQModel.load("{model}", qc)
    print(f"[LOAD OK]", flush=True)

    model.quantize(calib, batch_size=1)
    print(f"[QUANT OK]", flush=True)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_quantized(out_dir)

    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(json.dumps({{"ok": True, "error": ""}}))

except Exception as e:
    print(json.dumps({{"ok": False, "error": str(e)[:200]}}))
'''


def run_single(method: str, fmt: str, gs: int, desc_act: bool, damp: float) -> tuple[bool, str]:
    out_dir = str(OUTPUT_BASE / f"{method}_g{gs}")
    code = WORKER_CODE.format(
        root=str(ROOT),
        method=method,
        gs=gs,
        fmt=fmt,
        desc_act=str(desc_act),
        damp=damp,
        out_dir=out_dir,
        model=BASE_MODEL,
    )

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(ROOT),
        )

        stdout_lines = []
        # Real-time stdout
        for line in proc.stdout:
            line = line.rstrip()
            stdout_lines.append(line)
            # 過濾掉 gptqmodel 的 banner 和進度條
            if line and not line.startswith('_') and '\\' not in line:
                print(f"    {line}", flush=True)

        proc.wait()
        elapsed = time.time() - t0

        # 最後一行應該是 JSON 結果
        result_line = next(
            (l for l in reversed(stdout_lines) if l.startswith('{')),
            None
        )
        if result_line:
            r = json.loads(result_line)
            return r["ok"], r.get("error", "")
        else:
            # 沒有 JSON → 崩潰
            stderr = proc.stderr.read()[:200]
            return False, f"crash (exit={proc.returncode}): {stderr}"

    except Exception as e:
        return False, str(e)[:200]


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    results = []

    for method, fmt, desc_act, damp in METHODS:
        print(f"\n{'='*65}", flush=True)
        print(f"  {method.upper()} — 測試 group_sizes: {GROUP_SIZES}", flush=True)
        print(f"{'='*65}", flush=True)

        for gs in GROUP_SIZES:
            print(f"\n  → {method.upper()} group_size={gs}", flush=True)
            ok, err = run_single(method, fmt, gs, desc_act, damp)
            results.append({"method": method, "group_size": gs, "ok": ok, "error": err})

            status = "✓ OK  " if ok else "✗ FAIL"
            note = f"  ({err[:70]})" if not ok else ""
            print(f"  {status}  {method.upper():<5} group_size={gs:>4}{note}", flush=True)

    # 彙總
    print("\n" + "=" * 65, flush=True)
    print("  FINAL SUMMARY", flush=True)
    print("=" * 65, flush=True)
    for method, _, _, _ in METHODS:
        ok_gs   = sorted(r["group_size"] for r in results if r["method"] == method and r["ok"])
        fail_gs = sorted(r["group_size"] for r in results if r["method"] == method and not r["ok"])
        print(f"  {method.upper():<5}  支援: {ok_gs}", flush=True)
        print(f"         不支援: {fail_gs}", flush=True)
    print("=" * 65, flush=True)

    # 結果存檔
    result_path = OUTPUT_BASE.parent / "group_size_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果已存: {result_path}", flush=True)

    # 清理量化模型
    shutil.rmtree(OUTPUT_BASE, ignore_errors=True)
    print("測試模型已清理完成", flush=True)


if __name__ == "__main__":
    main()
