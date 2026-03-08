import logging
import gc
import shutil
import torch
import json
from datetime import datetime
from pathlib import Path
import argparse
from .llm_client import LLMDecisionMaker
from .executors import run_asvd, run_sparse, run_quantization, run_evaluation
from .utils import get_pareto_frontier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModularOrchestrator")

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def _make_trial_name(i: int, suggestion) -> str:
    """生成 trial 目錄名稱，包含編號、方法與關鍵參數"""
    parts = [f"trial_{i:03d}"]
    mode = suggestion.mode

    if mode in ("sparse_only", "hybrid"):
        struct = suggestion.sparsity_structure or "unstructured"
        is_structured = struct != "unstructured"
        has_ratio = suggestion.sparsity_ratio and suggestion.sparsity_ratio > 0
        if is_structured or has_ratio:
            if is_structured:
                # "2:4" → "sparse_2x4"、"4:8" → "sparse_4x8"
                parts.append(f"sparse_{struct.replace(':', 'x')}")
            else:
                ratio_pct = int(suggestion.sparsity_ratio * 100)
                parts.append(f"sparse_{ratio_pct}pct")

    if mode in ("asvd_only", "hybrid") and suggestion.alpha is not None:
        ratio_str = f"{int((suggestion.param_ratio_target or 0.9) * 100):03d}"
        alpha_str = f"{int((suggestion.alpha or 0.5) * 100):02d}"
        parts.append(f"asvd_r{ratio_str}_a{alpha_str}")

    if mode in ("quant_only", "hybrid") and suggestion.quant_method != "none":
        m = suggestion.quant_method
        b = suggestion.quant_bits
        g = suggestion.quant_group_size or 128
        fmt = suggestion.quant_format or "gptq"
        if m == "gptq":
            parts.append(f"gptq_{b}bit_g{g}_{fmt}")
        elif m == "awq":
            parts.append(f"awq_{b}bit_g{g}")
        elif m == "qqq":
            parts.append(f"qqq_4bit_g{g}")
        elif m == "bnb":
            parts.append(f"bnb_{b}bit")

    return "_".join(parts)


class OptimizationOrchestrator:
    def __init__(self, model_id: str, task: str, max_iterations: int = 10,
                 weights: dict = None, num_samples: int = None,
                 cleanup: bool = True, keep_best: bool = True):
        self.model_id = model_id
        self.task = task
        self.max_iterations = max_iterations
        self.base_model_path = model_id
        self.num_samples = num_samples
        self.cleanup = cleanup
        self.keep_best = keep_best
        self.trial_history = []
        self.llm = LLMDecisionMaker(model_id, task, max_iterations)
        self.best_score = -float('inf')
        self.best_result = None
        self.weights = weights or {"acc": 0.5, "lat": 0.1, "vram": 0.2, "emit": 0.2}
        self.baseline_metrics = None

        # 實驗目錄：tuning_results/exp_{model}_{task}_{timestamp}/
        model_name = Path(model_id).name
        task_str = task.replace(",", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = _ROOT_DIR / "tuning_results" / f"exp_{model_name}_{task_str}_{ts}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"實驗目錄: {self.exp_dir}")

        # baseline 快取目錄：tuning_results/baselines/{model_name}_{task}.json
        self._baseline_cache_dir = _ROOT_DIR / "tuning_results" / "baselines"
        self._baseline_cache_dir.mkdir(parents=True, exist_ok=True)
        self._baseline_cache_path = self._baseline_cache_dir / f"{model_name}_{task_str}.json"

    def _load_or_run_baseline(self) -> dict:
        """載入快取的 baseline，若不存在則重新評估並快取。"""
        if self._baseline_cache_path.exists():
            with open(self._baseline_cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            # 快取的 num_samples 必須 >= 當前設定才可重用
            cached_samples = cached.get("num_samples")
            if cached_samples is None or self.num_samples is None or cached_samples >= self.num_samples:
                logger.info(f"使用快取 baseline: {self._baseline_cache_path}")
                logger.info(f"  accuracy={cached['accuracy']:.4f}, latency={cached['latency']:.4f}s, "
                            f"vram={cached['vram']:.4f}GB, emissions={cached['emissions']:.6f}kg CO2")
                return cached

        logger.info("--- 計算基線指標（無快取，重新評估）---")
        results = run_evaluation(
            self.base_model_path, self.task,
            weights=self.weights, baseline_metrics=None,
            num_samples=self.num_samples,
            output_dir=str(self._baseline_cache_dir),
        )
        # 存入快取
        cache_entry = {
            "model_id": self.model_id,
            "task": self.task,
            "num_samples": self.num_samples,   # 用於判斷快取是否足夠
            "accuracy": results["accuracy"],
            "latency": results["latency"],
            "vram": results["vram"],
            "emissions": results["emissions"],
        }
        with open(self._baseline_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_entry, f, indent=2, ensure_ascii=False)
        logger.info(f"基線已快取至: {self._baseline_cache_path}")
        return results

    def optimize(self):
        self.baseline_metrics = self._load_or_run_baseline()
        logger.info(f"基線建立完成: {self.baseline_metrics}")

        # 儲存實驗設定
        self._save_experiment_config()


        trial_dirs = []  # 追蹤所有生成的 trial 目錄

        for i in range(1, self.max_iterations + 1):
            logger.info(f"\n===== Iteration {i} =====")

            # Step 1: LLM 決策
            pareto = get_pareto_frontier(self.trial_history)
            suggestion, llm_output = self.llm.get_suggestion(i, self.trial_history, pareto=pareto,
                                                              weights=self.weights)
            logger.info(f"建議: {suggestion.mode} | 理由: {suggestion.reasoning}")

            # Step 2: 建立 trial 目錄
            trial_name = _make_trial_name(i, suggestion)
            trial_dir = str(self.exp_dir / trial_name)
            Path(trial_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Trial 目錄: {trial_dir}")

            # Step 3: 執行壓縮
            # 最後一個壓縮步驟的輸出存到 trial_dir（讓評估器結果路徑正確）
            # 中間步驟存到子目錄以便偵錯
            current_model = self.base_model_path

            _struct = suggestion.sparsity_structure or "unstructured"
            has_sparse = suggestion.mode in ["sparse_only", "hybrid"] and (
                _struct != "unstructured" or bool(suggestion.sparsity_ratio)
            )
            has_asvd = suggestion.mode in ["asvd_only", "hybrid"] and suggestion.alpha is not None
            has_quant = suggestion.mode in ["quant_only", "hybrid"] and suggestion.quant_method != "none"

            try:
                if has_sparse:
                    # 若後面還有其他步驟，存到子目錄；否則直接存到 trial_dir
                    sparse_out = trial_dir if (not has_asvd and not has_quant) else str(Path(trial_dir) / "sparse")
                    current_model = run_sparse(current_model, suggestion, output_dir=sparse_out)

                if has_asvd:
                    asvd_out = trial_dir if not has_quant else str(Path(trial_dir) / "asvd")
                    current_model = run_asvd(current_model, suggestion, output_dir=asvd_out)

                if has_quant:
                    current_model = run_quantization(current_model, suggestion, output_dir=trial_dir)

            except Exception as e:
                logger.error(f"壓縮失敗 (iteration {i}): {e}")
                import traceback; traceback.print_exc()
                self.trial_history.append({
                    "iteration": i, "config": suggestion.to_log_dict(),
                    "suggestion": llm_output,
                    "metrics": {"score": 0.0}, "model_path": None, "error": str(e),
                    "trial_name": trial_name,
                })
                self.save_history()
                continue

            trial_dirs.append(trial_dir)

            # 確保評估時 current_model 的名稱 == trial_name
            # 否則評估器會用錯誤子目錄名（例如 BNB on-the-fly 後 current_model 仍是 trial_dir/asvd）
            if Path(current_model).resolve() != Path(trial_dir).resolve():
                src = Path(current_model)
                dst = Path(trial_dir)
                if src.is_dir() and src.parent.resolve() == dst.resolve():
                    # current_model 是 trial_dir 的直接子目錄，將內容搬上來
                    import shutil as _shutil
                    for item in src.iterdir():
                        _shutil.move(str(item), str(dst / item.name))
                    src.rmdir()
                    current_model = trial_dir
                    logger.info(f"已將模型搬移至 trial_dir: {trial_dir}")

            # Step 4: 評估
            # output_dir=exp_dir，評估器內部會 append Path(current_model).name = trial_name
            # → 結果存到 exp_dir/trial_name/ = trial_dir/
            metrics = run_evaluation(
                current_model, self.task,
                weights=self.weights,
                baseline_metrics=self.baseline_metrics,
                num_samples=self.num_samples,
                output_dir=str(self.exp_dir),
            )

            trial_data = {
                "iteration": i,
                "trial_name": trial_name,
                "config": suggestion.to_log_dict(),
                "suggestion": llm_output,
                "metrics": metrics,
                "model_path": str(current_model),
                "trial_dir": trial_dir,
            }
            self.trial_history.append(trial_data)
            self._update_best(trial_data)
            self.save_history()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 最終報告
        self._print_summary()

        # 清理 trial 模型
        if self.cleanup:
            self._cleanup_trials(trial_dirs)

    def _cleanup_trials(self, trial_dirs: list):
        best_dir = str(self.best_result["trial_dir"]) if self.best_result else None
        deleted, kept = 0, 0
        for d in trial_dirs:
            if self.keep_best and d == best_dir:
                logger.info(f"保留最佳 trial: {Path(d).name}")
                kept += 1
                continue
            try:
                shutil.rmtree(d, ignore_errors=True)
                logger.info(f"已刪除: {Path(d).name}")
                deleted += 1
            except Exception as e:
                logger.warning(f"刪除失敗 {d}: {e}")
        logger.info(f"清理完成：刪除 {deleted} 個 trial，保留 {kept} 個")

    def _print_summary(self):
        logger.info("\n" + "=" * 50)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 50)
        if self.best_result:
            logger.info(f"Best trial : {self.best_result['trial_name']}")
            logger.info(f"Score      : {self.best_score:.4f}")
            logger.info(f"Accuracy   : {self.best_result['metrics']['accuracy']:.4f}")
            logger.info(f"Latency    : {self.best_result['metrics']['latency']:.4f}s")
            logger.info(f"VRAM       : {self.best_result['metrics']['vram']:.4f} GB")
            logger.info(f"Config     : {self.best_result['config']}")
            logger.info(f"Model path : {self.best_result['model_path']}")
        logger.info(f"Results    : {self.exp_dir}")
        logger.info("=" * 50 + "\n")

    def _save_experiment_config(self):
        """儲存實驗設定（weights、baseline、參數）至 experiment_config.json。"""
        config = {
            "model_id": self.model_id,
            "task": self.task,
            "max_iterations": self.max_iterations,
            "num_samples": self.num_samples,
            "weights": self.weights,
            "cleanup": self.cleanup,
            "keep_best": self.keep_best,
            "exp_dir": str(self.exp_dir),
            "baseline": {
                "accuracy": self.baseline_metrics.get("accuracy"),
                "latency": self.baseline_metrics.get("latency"),
                "vram": self.baseline_metrics.get("vram"),
                "emissions": self.baseline_metrics.get("emissions"),
            },
        }
        path = self.exp_dir / "experiment_config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"實驗設定已儲存: {path}")

    def save_history(self):
        output_path = self.exp_dir / "optimization_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self._make_serializable(self.trial_history), f, indent=2, ensure_ascii=False)
        logger.info(f"歷史已儲存: {output_path}")

    def _make_serializable(self, data):
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(v) for v in data]
        elif isinstance(data, (Path, torch.device)):
            return str(data)
        return data

    def _update_best(self, result):
        if result['metrics']['score'] > self.best_score:
            self.best_score = result['metrics']['score']
            self.best_result = result
            logger.info(f"新最佳結果！Score: {self.best_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global Tuner v2 Orchestrator")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--max_iterations", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=None,
                        help="每個 dataset 的評估樣本數（None = 全部）")
    parser.add_argument("--acc_weight", type=float, default=0.7)
    parser.add_argument("--lat_weight", type=float, default=0.1)
    parser.add_argument("--vram_weight", type=float, default=0.2)
    parser.add_argument("--emit_weight", type=float, default=0.0)
    parser.add_argument("--no-cleanup", dest="cleanup", action="store_false",
                        help="跑完後不刪除 trial 模型（預設：刪除）")
    parser.add_argument("--no-keep-best", dest="keep_best", action="store_false",
                        help="刪除時連最佳 trial 也刪（預設：保留最佳）")
    parser.set_defaults(cleanup=True, keep_best=True)

    args = parser.parse_args()
    weights = {
        "acc": args.acc_weight, "lat": args.lat_weight,
        "vram": args.vram_weight, "emit": args.emit_weight,
    }
    orchestrator = OptimizationOrchestrator(
        args.model_id, args.task, args.max_iterations, weights,
        num_samples=args.num_samples,
        cleanup=args.cleanup, keep_best=args.keep_best,
    )
    orchestrator.optimize()
