#!/usr/bin/env python3
"""
GPTQ 參數搜索實驗腳本
=====================
量化 + 評估多個 GPTQ 配置變體，比較精度、GPU 記憶體、推理速度。

實驗變體（全部基於 Llama-3.2-1B-Instruct, bits=4, damp=0.05）：
  baseline         : format=gptq,    desc_act=false, group=128          (重用現有模型)
  gptq_v2          : format=gptq_v2, desc_act=false, group=128          (測試格式相容性)
  marlin           : format=marlin,  desc_act=false, group=128          (測試推理速度)
  desc_act         : format=gptq,    desc_act=true,  group=128          (精度提升測試)
  rotation_hadamard: format=gptq,    desc_act=false, group=128, hadamard (精度提升測試)
  group64          : format=gptq,    desc_act=false, group=64           (更細粒度精度)

用法：
    # 含已有基準模型（跳過重新量化）：
    python experiments/gptqmodel_methods/run_param_sweep.py \\
        --baseline quantized_test/Llama-3.2-1B-Instruct-gptq-4bit-20260302_024452

    # 背景執行：
    nohup python experiments/gptqmodel_methods/run_param_sweep.py \\
        --baseline quantized_test/Llama-3.2-1B-Instruct-gptq-4bit-20260302_024452 \\
        > sweep.log 2>&1 &

    # 只執行部分實驗：
    python experiments/gptqmodel_methods/run_param_sweep.py \\
        --baseline quantized_test/... --experiments baseline gptq_v2 marlin

    # 只評估已量化的模型（不重新量化）：
    python experiments/gptqmodel_methods/run_param_sweep.py --eval-only \\
        quantized_test/model-a quantized_test/model-b
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("ParamSweep")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
logger.propagate = False

# ============================================================================
# 實驗定義
# ============================================================================

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CALIB_CONFIG = {
    "dataset": "openai/gsm8k",
    "dataset_config": "main",
    "split": "train",
    "num_samples": 256,
    "text_column": "question+answer",
    "batch_size": 1,
}
BASE_QUANT = {
    "quant_method": "gptq",
    "bits": 4,
    "sym": True,
    "true_sequential": True,
    "lm_head": False,
    "pack_dtype": "int32",
    "offload_to_disk": True,
    "damp_percent": 0.05,
    "mse": 0.0,
}

EXPERIMENTS = [
    {
        "name": "baseline",
        "desc": "GPTQ v1, group=128, desc_act=false  [基準]",
        "quant_overrides": {"format": "gptq", "group_size": 128, "desc_act": False},
    },
    {
        "name": "gptq_v2",
        "desc": "GPTQ v2 format（有號零點，精度略優）",
        "quant_overrides": {"format": "gptq_v2", "group_size": 128, "desc_act": False},
    },
    {
        "name": "marlin",
        "desc": "Marlin format（RTX 4090 SM=8.9，推理 2-4x 更快）",
        "quant_overrides": {"format": "marlin", "group_size": 128, "desc_act": False},
    },
    {
        "name": "desc_act",
        "desc": "desc_act=true（降序激活，改善量化誤差）",
        "quant_overrides": {"format": "gptq", "group_size": 128, "desc_act": True},
    },
    {
        "name": "rotation_hadamard",
        "desc": "Hadamard rotation（改善 outlier 分佈）",
        "quant_overrides": {"format": "gptq", "group_size": 128, "desc_act": False, "rotation": "hadamard"},
    },
    {
        "name": "group64",
        "desc": "group_size=64（更細粒度縮放，精度更高）",
        "quant_overrides": {"format": "gptq", "group_size": 64, "desc_act": False},
    },
]

EVAL_DATASETS = ["gsm8k"]
DEFAULT_NUM_SAMPLES = 200
SWEEP_RESULTS_DIR = "experiments/gptqmodel_methods/sweep_results"


# ============================================================================
# YAML 生成
# ============================================================================

def make_experiment_yaml(quant_overrides: dict) -> dict:
    quant = {**BASE_QUANT, **quant_overrides}
    return {
        "model": {"name": BASE_MODEL, "output_dir": None, "output_base": "quantized_test"},
        "calibration": CALIB_CONFIG,
        "quantization": quant,
    }


def write_variant_yaml(config_dict: dict, name: str) -> str:
    variants_dir = ROOT / "experiments" / "gptqmodel_methods" / "config" / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = str(variants_dir / f"sweep_{name}.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)
    logger.info(f"Config written: {yaml_path}")
    return yaml_path


# ============================================================================
# 執行函數
# ============================================================================

def run_quantization(yaml_path: str) -> str | None:
    """執行量化，返回量化模型的輸出目錄。"""
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "gptqmodel_methods" / "run_experiment.py"),
        "--config", yaml_path,
    ]
    logger.info(f"Quantizing with: {yaml_path}")
    start = time.time()

    output_lines = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(ROOT)
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)
    proc.wait()

    elapsed = time.time() - start
    logger.info(f"Quantization done in {elapsed/60:.1f} min (exit={proc.returncode})")

    if proc.returncode != 0:
        logger.error("Quantization failed")
        return None

    # 解析輸出目錄：找包含路徑的行（最後出現的）
    for line in reversed(output_lines):
        stripped = line.strip()
        for keyword in ("Model saved to:", "Done. Output:", "Saved to", "Experiment complete"):
            if keyword in stripped:
                # 取 keyword 之後的部分
                after = stripped.split(keyword, 1)[-1].strip().rstrip(".")
                # 移除 ANSI 顏色碼
                clean = ""
                skip = False
                for ch in after:
                    if ch == "\x1b":
                        skip = True
                    elif skip and ch == "m":
                        skip = False
                    elif not skip:
                        clean += ch
                clean = clean.strip()
                if clean and ("quantized_test" in clean or Path(clean).exists()):
                    return clean

    # Fallback: 找 quantized_test/ 下最新的目錄
    qt_dir = ROOT / "quantized_test"
    if qt_dir.exists():
        dirs = sorted(qt_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if dirs:
            logger.warning(f"Falling back to latest quantized_test dir: {dirs[0].name}")
            return str(dirs[0])

    logger.error("Cannot determine quantization output directory")
    return None


def run_evaluation(model_path: str, datasets: list, num_samples: int) -> dict:
    """執行評估，返回各資料集的結果 dict。"""
    cmd = [
        sys.executable,
        str(ROOT / "experiments" / "gptqmodel_methods" / "run_eval.py"),
        "--model", model_path,
        "--datasets", ",".join(datasets),
        "--num-samples", str(num_samples),
        "--output-dir", str(ROOT / SWEEP_RESULTS_DIR),
    ]
    logger.info(f"Evaluating: {Path(model_path).name}")
    start = time.time()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(ROOT)
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()

    elapsed = time.time() - start
    logger.info(f"Evaluation done in {elapsed/60:.1f} min (exit={proc.returncode})")

    if proc.returncode != 0:
        return {"error": f"eval exit code {proc.returncode}"}

    return _read_results(model_path, datasets)


def _read_results(model_path: str, datasets: list) -> dict:
    """從 sweep_results 目錄讀取 JSON 結果。"""
    model_name = Path(model_path).name
    results = {}
    for ds in datasets:
        result_path = ROOT / SWEEP_RESULTS_DIR / model_name / f"{ds}_results.json"
        if result_path.exists():
            with open(result_path) as f:
                results[ds] = json.load(f)
        else:
            results[ds] = {"error": f"result file not found: {result_path}"}
    return results


def has_existing_results(model_path: str, datasets: list) -> bool:
    model_name = Path(model_path).name
    return all(
        (ROOT / SWEEP_RESULTS_DIR / model_name / f"{ds}_results.json").exists()
        for ds in datasets
    )


# ============================================================================
# 結果輸出
# ============================================================================

def print_comparison_table(all_results: list, num_samples: int):
    print("\n" + "=" * 105)
    print("  GPTQ 參數搜索結果比較")
    print("=" * 105)

    for ds in EVAL_DATASETS:
        print(f"\n  【{ds.upper()}】（{num_samples} samples）")
        header = f"  {'實驗名稱':<22} {'Accuracy':>10} {'Correct/Total':>14} {'GPU Peak MB':>12} {'Throughput t/s':>15}  說明"
        print(header)
        print("  " + "-" * 100)

        baseline_acc = None
        for exp in all_results:
            ds_result = exp.get("results", {}).get(ds, {})
            name = exp["name"]
            desc = exp["desc"]

            if "error" in ds_result:
                err = ds_result["error"][:50]
                print(f"  {name:<22} {'FAILED':<10}  {err}")
                continue

            acc = ds_result.get("accuracy", 0)
            correct = ds_result.get("correct", 0)
            total = ds_result.get("total", 0)
            gpu = ds_result.get("gpu_peak_mb", "N/A")
            tput = ds_result.get("throughput_tokens_per_sec", "N/A")

            gpu_str = f"{gpu:.0f}" if isinstance(gpu, float) else str(gpu)
            tput_str = f"{tput:.1f}" if isinstance(tput, float) else str(tput)
            ct_str = f"{correct}/{total}"

            if name == "baseline":
                baseline_acc = acc
                delta_str = "[基準]  "
            elif baseline_acc is not None:
                delta = (acc - baseline_acc) * 100
                sign = "+" if delta >= 0 else ""
                delta_str = f"({sign}{delta:.1f}%)"
            else:
                delta_str = ""

            print(
                f"  {name:<22} {acc:>9.4f}  {ct_str:>14} "
                f"{gpu_str:>12} {tput_str:>15}  "
                f"{delta_str} {desc}"
            )

    print("\n" + "=" * 105)


def save_comparison_json(all_results: list):
    out_dir = ROOT / SWEEP_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Comparison JSON saved: {out_path}")


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPTQ 參數搜索：量化 + 評估多個變體")
    parser.add_argument(
        "--baseline", default=None,
        help="現有基準模型路徑（跳過重新量化）。"
             "例如：quantized_test/Llama-3.2-1B-Instruct-gptq-4bit-20260302_024452",
    )
    parser.add_argument(
        "--eval-only", nargs="+", default=None,
        help="只評估指定模型，不量化。提供模型路徑列表。",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="只量化，不評估",
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="只執行指定實驗名稱。可選：baseline gptq_v2 marlin desc_act rotation_hadamard group64",
    )
    parser.add_argument(
        "--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
        help=f"每個資料集的評估樣本數（預設 {DEFAULT_NUM_SAMPLES}）",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    (ROOT / SWEEP_RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # --eval-only 模式
    if args.eval_only:
        logger.info(f"eval-only 模式：{len(args.eval_only)} 個模型")
        all_results = []
        for model_path in args.eval_only:
            if has_existing_results(model_path, EVAL_DATASETS):
                results = _read_results(model_path, EVAL_DATASETS)
                logger.info(f"使用已有結果：{Path(model_path).name}")
            else:
                results = run_evaluation(model_path, EVAL_DATASETS, args.num_samples)
            all_results.append({
                "name": Path(model_path).name,
                "desc": model_path,
                "model_path": model_path,
                "results": results,
            })
        print_comparison_table(all_results, args.num_samples)
        save_comparison_json(all_results)
        return

    # 決定要執行哪些實驗
    experiments_to_run = EXPERIMENTS
    if args.experiments:
        names = set(args.experiments)
        experiments_to_run = [e for e in EXPERIMENTS if e["name"] in names]
        if not experiments_to_run:
            logger.error(f"No matching experiments. Available: {[e['name'] for e in EXPERIMENTS]}")
            sys.exit(1)

    logger.info(f"執行 {len(experiments_to_run)} 個實驗：{[e['name'] for e in experiments_to_run]}")

    all_results = []

    for i, exp in enumerate(experiments_to_run):
        name = exp["name"]
        logger.info(f"\n{'='*70}")
        logger.info(f"[{i+1}/{len(experiments_to_run)}] {name}：{exp['desc']}")
        logger.info(f"{'='*70}")

        model_path = None

        # ── 量化 ──────────────────────────────────────────────
        if name == "baseline" and args.baseline:
            model_path = args.baseline
            logger.info(f"重用現有基準模型：{model_path}")
        else:
            yaml_path = write_variant_yaml(
                make_experiment_yaml(exp["quant_overrides"]), name
            )
            model_path = run_quantization(yaml_path)

        if model_path is None:
            logger.error(f"實驗 {name} 量化失敗，跳過")
            all_results.append({
                "name": name, "desc": exp["desc"],
                "model_path": None,
                "results": {ds: {"error": "quantization failed"} for ds in EVAL_DATASETS},
            })
            continue

        # ── 評估 ──────────────────────────────────────────────
        if args.skip_eval:
            all_results.append({"name": name, "desc": exp["desc"], "model_path": model_path, "results": {}})
            continue

        if has_existing_results(model_path, EVAL_DATASETS):
            logger.info("發現已有評估結果，直接讀取（刪除結果目錄可強制重新評估）")
            results = _read_results(model_path, EVAL_DATASETS)
        else:
            results = run_evaluation(model_path, EVAL_DATASETS, args.num_samples)

        all_results.append({"name": name, "desc": exp["desc"], "model_path": model_path, "results": results})

        # 每個實驗結束後即時輸出
        gsm = results.get("gsm8k", {})
        if "error" not in gsm:
            logger.info(f"  ✓ gsm8k  accuracy={gsm.get('accuracy', 0):.4f}  gpu={gsm.get('gpu_peak_mb', 'N/A')} MB")

    print_comparison_table(all_results, args.num_samples)
    save_comparison_json(all_results)
    logger.info(f"完成！結果已儲存至 {SWEEP_RESULTS_DIR}/comparison.json")


if __name__ == "__main__":
    main()
