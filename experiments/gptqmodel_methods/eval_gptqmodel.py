"""
GPTQModel 統一評估腳本
=====================

用 GPTQModel.from_quantized() 載入所有 GPTQModel 量化的模型
（包括 GPTQ、AWQ、QQQ method），然後注入到現有 evaluator 執行評估。

用法:
    python experiments/gptqmodel_methods/evaluate.py \
        --model quantized_test/Llama-3.2-1B-Instruct-awq-4bit-20260302_032645 \
        --datasets gsm8k \
        --num-samples null
"""

import argparse
import logging
import os
import sys
import traceback

from dotenv import load_dotenv

# 確保能從專案根目錄 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tmp.evals import (
    GSM8KEvaluator,
    TruthfulQAEvaluator,
    CommonsenseQAEvaluator,
    HumanEvalEvaluator,
    BBHEvaluator,
)

logger = logging.getLogger("GPTQModelEval")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
logger.propagate = False

EVALUATOR_MAP = {
    "gsm8k": GSM8KEvaluator,
    "truthfulqa": TruthfulQAEvaluator,
    "commonsenseqa": CommonsenseQAEvaluator,
    "humaneval": HumanEvalEvaluator,
    "bbh": BBHEvaluator,
}


def load_model_gptqmodel(model_path: str):
    """用 GPTQModel.from_quantized() 統一載入模型和 tokenizer"""
    from gptqmodel import GPTQModel
    from transformers import AutoTokenizer

    logger.info(f"Loading model with GPTQModel: {model_path}")
    model = GPTQModel.from_quantized(
        model_path,
        device_map={"": "cuda:0"},
    )

    logger.info(f"Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded: {type(model).__name__}, device={model.device}")
    return model, tokenizer


def build_eval_config(model_path: str, output_dir: str, num_samples=None):
    """建立 evaluator 所需的 dict config（模擬 model_config.yaml 結構）"""
    config = {
        "model": {
            "name": model_path,
        },
        "dataset": {
            "config_file": "tmp/config/dataset_config.yaml",
            "override": {},
        },
    }
    if num_samples is not None:
        config["dataset"]["override"]["num_samples"] = num_samples
    if output_dir:
        config["dataset"]["override"]["output_dir"] = output_dir
    return config


def run_evaluation(model_path: str, datasets: list, num_samples=None, output_dir: str = "results"):
    """載入模型一次，對多個 dataset 執行評估"""
    # 載入模型
    model, tokenizer = load_model_gptqmodel(model_path)

    # 建立 config dict
    config = build_eval_config(model_path, output_dir, num_samples)

    all_results = {}
    for dataset_name in datasets:
        logger.info(f"{'=' * 60}")
        logger.info(f"Evaluating: {dataset_name.upper()}")
        logger.info(f"{'=' * 60}")

        if dataset_name not in EVALUATOR_MAP:
            logger.error(f"Unknown dataset: {dataset_name}. Available: {list(EVALUATOR_MAP.keys())}")
            all_results[dataset_name] = {"error": f"Unknown dataset: {dataset_name}"}
            continue

        try:
            evaluator_cls = EVALUATOR_MAP[dataset_name]
            evaluator = evaluator_cls(config)

            # 注入 GPTQModel 載入的 model 和 tokenizer
            evaluator.model = model
            evaluator.tokenizer = tokenizer
            evaluator._use_vllm = False
            evaluator._setup_generation_pipeline()

            # 執行評估
            results = evaluator.evaluate()
            evaluator.save_results(results)
            all_results[dataset_name] = results

            accuracy = results.get("accuracy", 0)
            total = results.get("total", 0)
            logger.info(f"{dataset_name.upper()} done: accuracy={accuracy:.4f}, total={total}")

        except Exception as e:
            logger.error(f"{dataset_name} evaluation failed: {e}")
            traceback.print_exc()
            all_results[dataset_name] = {"error": str(e)}

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GPTQModel-quantized models (GPTQ/AWQ/QQQ) using existing evaluators"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to GPTQModel-quantized model directory"
    )
    parser.add_argument(
        "--datasets", default="gsm8k",
        help="Comma-separated list of datasets to evaluate (default: gsm8k)"
    )
    parser.add_argument(
        "--num-samples", default="null",
        help="Number of samples per dataset (null = all, default: null)"
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for results (default: results)"
    )
    args = parser.parse_args()

    load_dotenv()

    datasets = [d.strip() for d in args.datasets.split(",")]
    num_samples = None if args.num_samples == "null" else int(args.num_samples)

    logger.info(f"Model: {args.model}")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Samples: {num_samples or 'ALL'}")
    logger.info(f"Output: {args.output_dir}")

    all_results = run_evaluation(
        model_path=args.model,
        datasets=datasets,
        num_samples=num_samples,
        output_dir=args.output_dir,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("Evaluation Summary")
    print(f"{'=' * 60}")
    for dataset_name, result in all_results.items():
        if "error" in result:
            print(f"  {dataset_name:<15} FAILED: {result['error'][:60]}")
        else:
            acc = result.get("accuracy", 0)
            total = result.get("total", 0)
            gpu = result.get("gpu_peak_mb", "N/A")
            print(f"  {dataset_name:<15} accuracy={acc:.4f}  total={total}  gpu_peak={gpu} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
