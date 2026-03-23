"""
Systematic Tuner Orchestrator
以 Optuna（TPE / NSGA-II / Random）搜尋最佳量化配置
"""

import logging
import gc
import shutil
import traceback
import torch
import json
import argparse
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import sys
_ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT_DIR))

from .executors import run_asvd, run_sparse, run_quantization, run_evaluation


# ── Process Isolation（從 Global_Tuner_v2 移植）────────────────────────────
def _worker(queue, func, *args, **kwargs):
    import torch
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = func(*args, **kwargs)
        peak_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if torch.cuda.is_available() else 0.0
        )
        queue.put({"status": "success", "result": result, "peak_vram_mb": peak_mb})
    except Exception as e:
        queue.put({"status": "error", "error": str(e), "traceback": traceback.format_exc()})

def run_isolated(func, *args, **kwargs):
    """
    在獨立的 spawn 子進程中執行函式。
    子進程結束後 OS 保證 100% 釋放所有 VRAM，根本解決記憶體殘留問題。

    Returns:
        (result, peak_vram_mb): 函式回傳值 + 子進程內的 GPU peak（MB）
    """
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(queue, func) + args, kwargs=kwargs)
    p.start()
    p.join()

    if not queue.empty():
        res = queue.get()
        if res["status"] == "success":
            return res["result"], res.get("peak_vram_mb", 0.0)
        else:
            raise RuntimeError(f"Isolated process failed:\n{res['error']}\n{res['traceback']}")
    else:
        raise RuntimeError("子進程異常終止（可能是 OOM 被 OS 砍掉）。")
# ────────────────────────────────────────────────────────────────────────────
from .optuna_searcher import OptunaSearcher
from .search_space import ALL_MODES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SystematicOrchestrator")


# ──────────────────────────────────────────────────────────────────────────────
def _make_trial_name(i: int, suggestion) -> str:
    parts = [f"trial_{i:03d}"]
    mode = suggestion.mode

    if mode in ("sparse_only", "hybrid"):
        struct = suggestion.sparsity_structure or "unstructured"
        is_structured = struct != "unstructured"
        has_ratio = suggestion.sparsity_ratio and suggestion.sparsity_ratio > 0
        if is_structured or has_ratio:
            if is_structured:
                parts.append(f"sparse_{struct.replace(':', 'x')}")
            else:
                parts.append(f"sparse_{int(suggestion.sparsity_ratio * 100)}pct")

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


# ──────────────────────────────────────────────────────────────────────────────
class SystematicOrchestrator:
    """
    以 Optuna（TPE / NSGA-II / Random）逐一嘗試量化配置。

    sampler: "tpe" (預設) | "nsga2" | "random"
    modes:   要搜尋的模式子集，None = 全部（見 search_space.ALL_MODES）
    """

    def __init__(
        self,
        model_id: str,
        task: str,
        max_iterations: int = 20,
        weights: dict = None,
        num_samples: int = None,
        cleanup: bool = True,
        keep_best: bool = True,
        optuna_sampler: str = "tpe",
        modes: list = None,
        seed: int = None,
        n_startup_trials: int = 10,
        population_size: int = 50,
        pen_t: float = 0.15,
        pen_a: float = 10.0,
    ):
        self.model_id = model_id
        self.task = task
        self.search_method = "optuna"
        self.max_iterations = max_iterations
        self.num_samples = num_samples
        self.cleanup = cleanup
        self.keep_best = keep_best
        self.weights = weights or {"acc": 3.0, "lat": 1.0, "vram": 1.0, "emit": 1.0}
        self.pen_t = pen_t
        self.pen_a = pen_a
        self.trial_history = []
        self.best_score = -float("inf")
        self.best_result = None
        self.baseline_metrics = None
        self._seen_configs: set = set()

        self.searcher = OptunaSearcher(
            sampler=optuna_sampler, modes=modes, seed=seed,
            n_startup_trials=n_startup_trials,
            population_size=population_size,
        )
        logger.info(f"Optuna 模式 (sampler={optuna_sampler}, iterations={max_iterations})")

        # 實驗目錄
        model_name = Path(model_id).name
        task_str = task.replace(",", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = (
            _ROOT_DIR / "systematic_results"
            / f"{optuna_sampler}_{model_name}_{task_str}_{ts}"
        )
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"實驗目錄: {self.exp_dir}")

        # baseline 快取目錄
        self._baseline_cache_dir = _ROOT_DIR / "systematic_results" / "baselines"
        self._baseline_cache_dir.mkdir(parents=True, exist_ok=True)
        self._baseline_cache_path = (
            self._baseline_cache_dir / f"{model_name}_{task_str}.json"
        )

    # ──────────────────────────────────────────────────────────────────────────
    def _load_or_run_baseline(self) -> dict:
        if self._baseline_cache_path.exists():
            with open(self._baseline_cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_samples = cached.get("num_samples")
            if (
                cached_samples is None
                or self.num_samples is None
                or cached_samples >= self.num_samples
            ):
                logger.info(f"使用快取 baseline: {self._baseline_cache_path}")
                logger.info(
                    f"  accuracy={cached['accuracy']:.4f}, "
                    f"latency={cached['latency']:.4f}s, "
                    f"vram={cached['vram']:.4f}GB, "
                    f"emissions={cached['emissions']:.6f}kg CO2"
                )
                return cached

        logger.info("--- 計算基線指標（無快取，重新評估）---")
        results, baseline_vram_mb = run_isolated(
            run_evaluation,
            self.model_id, self.task,
            weights=self.weights, baseline_metrics=None,
            num_samples=self.num_samples,
            output_dir=str(self._baseline_cache_dir),
            pen_t=self.pen_t, pen_a=self.pen_a,
        )
        logger.info(f"Baseline 壓縮過程 GPU peak: {baseline_vram_mb:.1f} MB")
        cache_entry = {
            "model_id": self.model_id,
            "task": self.task,
            "num_samples": self.num_samples,
            "accuracy":   results["accuracy"],
            "latency":    results["latency"],
            "vram":       results["vram"],
            "emissions":  results["emissions"],
        }
        with open(self._baseline_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_entry, f, indent=2, ensure_ascii=False)
        logger.info(f"基線已快取至: {self._baseline_cache_path}")
        return results

    # ──────────────────────────────────────────────────────────────────────────
    def optimize(self):
        self.baseline_metrics = self._load_or_run_baseline()
        logger.info(f"基線建立完成: {self.baseline_metrics}")
        self._save_experiment_config()

        trial_dirs = []

        total_iters = self.max_iterations

        _MAX_DUP_RETRIES = 5  # Optuna 重複配置最大重試次數

        for i in range(1, total_iters + 1):
            logger.info(f"\n===== Iteration {i}/{total_iters} =====")

            # Step 1: 取得不重複的建議
            suggestion, raw_output = None, None
            for _retry in range(_MAX_DUP_RETRIES):
                try:
                    _s, _raw = self.searcher.get_suggestion(i, self.trial_history)
                except StopIteration:
                    logger.info("Grid search 已完成所有配置")
                    break

                _fp = self._config_fingerprint(_s)
                if _fp not in self._seen_configs:
                    suggestion, raw_output = _s, _raw
                    self._seen_configs.add(_fp)
                    break

                logger.warning(
                    f"[去重] 重複配置，跳過 (retry {_retry + 1}/{_MAX_DUP_RETRIES}): "
                    f"{_fp}"
                )
                if self.search_method == "optuna":
                    self.searcher.report_failure()
            else:
                logger.warning(f"Iteration {i}: 達到去重重試上限，跳過本輪")

            if suggestion is None:
                continue

            logger.info(f"配置: {suggestion.mode} | {suggestion.to_log_dict()}")

            # Step 2: 建立 trial 目錄
            trial_name = _make_trial_name(i, suggestion)
            trial_dir = str(self.exp_dir / trial_name)
            Path(trial_dir).mkdir(parents=True, exist_ok=True)

            # Step 3: 執行壓縮
            current_model = self.model_id
            _struct = suggestion.sparsity_structure or "unstructured"
            has_sparse = suggestion.mode in ("sparse_only", "hybrid") and (
                _struct != "unstructured" or bool(suggestion.sparsity_ratio)
            )
            has_asvd  = suggestion.mode in ("asvd_only", "hybrid") and suggestion.alpha is not None
            has_quant = suggestion.mode in ("quant_only", "hybrid") and suggestion.quant_method != "none"

            compression_vram = {}
            try:
                if has_sparse:
                    sparse_out = trial_dir if not (has_asvd or has_quant) else str(Path(trial_dir) / "sparse")
                    current_model, _mb = run_isolated(run_sparse, current_model, suggestion, output_dir=sparse_out)
                    compression_vram["sparse_mb"] = round(_mb, 1)
                    logger.info(f"Sparse GPU peak: {_mb:.1f} MB")

                if has_asvd:
                    asvd_out = trial_dir if not has_quant else str(Path(trial_dir) / "asvd")
                    current_model, _mb = run_isolated(run_asvd, current_model, suggestion, output_dir=asvd_out)
                    compression_vram["asvd_mb"] = round(_mb, 1)
                    logger.info(f"ASVD GPU peak: {_mb:.1f} MB")

                if has_quant:
                    current_model, _mb = run_isolated(run_quantization, current_model, suggestion, output_dir=trial_dir)
                    compression_vram["quant_mb"] = round(_mb, 1)
                    logger.info(f"Quantization GPU peak: {_mb:.1f} MB")

            except Exception as e:
                logger.error(f"壓縮失敗 (iteration {i}): {e}")
                traceback.print_exc()
                # 回報失敗給 Optuna
                if self.search_method == "optuna":
                    self.searcher.report_failure()
                self.trial_history.append({
                    "iteration": i, "config": suggestion.to_log_dict(),
                    "suggestion": raw_output,
                    "metrics": {"score": 0.0}, "model_path": None,
                    "error": str(e), "trial_name": trial_name,
                })
                self.save_history()
                continue

            trial_dirs.append(trial_dir)

            # 確保模型位於 trial_dir
            if Path(current_model).resolve() != Path(trial_dir).resolve():
                src = Path(current_model)
                dst = Path(trial_dir)
                if src.is_dir() and src.parent.resolve() == dst.resolve():
                    for item in src.iterdir():
                        shutil.move(str(item), str(dst / item.name))
                    src.rmdir()
                    current_model = trial_dir
                    logger.info(f"已將模型搬移至 trial_dir: {trial_dir}")

            # Step 4: 評估（在獨立子進程執行，VRAM 由 OS 保證完整釋放）
            metrics, eval_vram_mb = run_isolated(
                run_evaluation,
                current_model, self.task,
                weights=self.weights,
                baseline_metrics=self.baseline_metrics,
                num_samples=self.num_samples,
                output_dir=str(self.exp_dir),
                pen_t=self.pen_t, pen_a=self.pen_a,
            )
            compression_vram["eval_mb"] = round(eval_vram_mb, 1)
            logger.info(f"Evaluation GPU peak: {eval_vram_mb:.1f} MB")

            self.searcher.report_score(metrics["score"])

            trial_data = {
                "iteration":  i,
                "trial_name": trial_name,
                "config":     suggestion.to_log_dict(),
                "suggestion": raw_output,
                "metrics":    metrics,
                "compression_vram_mb": compression_vram,
                "model_path": str(current_model),
                "trial_dir":  trial_dir,
            }
            self.trial_history.append(trial_data)
            self._update_best(trial_data)
            self.save_history()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._print_summary()

        if self.cleanup:
            self._cleanup_trials(trial_dirs)

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _config_fingerprint(suggestion) -> str:
        """將 StrategySuggestion 轉為可比較的字串 fingerprint（排除 reasoning）"""
        d = suggestion.to_log_dict()
        return json.dumps(d, sort_keys=True, ensure_ascii=False)

    def _update_best(self, result: dict):
        if result["metrics"]["score"] > self.best_score:
            self.best_score = result["metrics"]["score"]
            self.best_result = result
            logger.info(f"新最佳結果！Score: {self.best_score:.4f}")

    def _cleanup_trials(self, trial_dirs: list):
        best_dir = str(self.best_result["trial_dir"]) if self.best_result else None
        deleted = kept = 0
        for d in trial_dirs:
            if self.keep_best and d == best_dir:
                logger.info(f"保留最佳 trial: {Path(d).name}")
                kept += 1
                continue
            try:
                shutil.rmtree(d, ignore_errors=True)
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
        config = {
            "model_id":       self.model_id,
            "task":           self.task,
            "search_method":  self.search_method,
            "sampler":        self.searcher.sampler,
            "max_iterations": self.max_iterations,
            "num_samples":    self.num_samples,
            "weights":        self.weights,
            "cleanup":        self.cleanup,
            "keep_best":      self.keep_best,
            "exp_dir":        str(self.exp_dir),
            "baseline": {
                "accuracy":  self.baseline_metrics.get("accuracy"),
                "latency":   self.baseline_metrics.get("latency"),
                "vram":      self.baseline_metrics.get("vram"),
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
            json.dump(self._make_serializable(self.trial_history), f,
                      indent=2, ensure_ascii=False)

    def _make_serializable(self, data):
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._make_serializable(v) for v in data]
        if isinstance(data, (Path, torch.device)):
            return str(data)
        return data


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Systematic Tuner")
    parser.add_argument("--model_id",    type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--task",        type=str, default="gsm8k")
    parser.add_argument("--max_iterations", type=int, default=20)
    parser.add_argument("--optuna_sampler", type=str, default="tpe",
                        choices=["tpe", "nsga2", "random"])
    parser.add_argument("--modes",       type=str, nargs="*", default=None,
                        metavar="MODE",
                        help=f"要搜尋的模式子集，可選: {ALL_MODES}")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--acc_weight",  type=float, default=3.0)
    parser.add_argument("--lat_weight",  type=float, default=1.0)
    parser.add_argument("--vram_weight", type=float, default=1.0)
    parser.add_argument("--emit_weight", type=float, default=1.0)
    parser.add_argument("--pen_t",       type=float, default=0.15,
                        help="Accuracy penalty 容忍量（絕對值，掉幅超過此值才扣分）")
    parser.add_argument("--pen_a",       type=float, default=10.0,
                        help="Accuracy penalty 放大倍率")
    parser.add_argument("--seed",             type=int,   default=None,
                        help="亂數種子（不指定則每次隨機，避免重複採樣）")
    parser.add_argument("--n_startup_trials", type=int,   default=10,
                        help="TPE 前幾輪純隨機探索再開始學習（tpe only）")
    parser.add_argument("--population_size",  type=int,   default=50,
                        help="遺傳演算法族群大小，建議 <= max_iterations/2（nsga2 only）")
    parser.add_argument("--no-cleanup",  dest="cleanup",    action="store_false")
    parser.add_argument("--no-keep-best",dest="keep_best",  action="store_false")
    parser.set_defaults(cleanup=True, keep_best=True)

    args = parser.parse_args()
    weights = {
        "acc":  args.acc_weight,
        "lat":  args.lat_weight,
        "vram": args.vram_weight,
        "emit": args.emit_weight,
    }

    orchestrator = SystematicOrchestrator(
        model_id=args.model_id,
        task=args.task,
        max_iterations=args.max_iterations,
        weights=weights,
        num_samples=args.num_samples,
        cleanup=args.cleanup,
        keep_best=args.keep_best,
        optuna_sampler=args.optuna_sampler,
        modes=args.modes,
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
        population_size=args.population_size,
        pen_t=args.pen_t,
        pen_a=args.pen_a,
    )
    orchestrator.optimize()
