import logging
import gc
import shutil
import torch
import json
import yaml
from datetime import datetime
from pathlib import Path
import argparse
from llm_client import LLMDecisionMaker
from executors import run_asvd, run_sparse, run_quantization, run_evaluation
from utils import get_pareto_frontier
from Evals.base_evaluator import BaseEvaluator
import multiprocessing as mp
import traceback
import statistics
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModularOrchestrator")

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# ============================================================================
# PROCESS ISOLATION WRAPPER
# ============================================================================
def _worker(queue, func, *args, **kwargs):
    """Worker function that executes the target function and captures the result."""
    try:
        result = func(*args, **kwargs)
        queue.put({"status": "success", "result": result})
    except Exception as e:
        queue.put({"status": "error", "error": str(e), "traceback": traceback.format_exc()})

def run_isolated(func, *args, **kwargs):
    """
    Runs a function in a completely isolated process using the 'spawn' context.
    This guarantees that the OS will wipe 100% of the allocated VRAM when the function finishes.
    """
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(queue, func) + args, kwargs=kwargs)
    p.start()
    p.join()

    if not queue.empty():
        res = queue.get()
        if res["status"] == "success":
            return res["result"]
        else:
            raise RuntimeError(f"Isolated process failed: {res['error']}\n{res['traceback']}")
    else:
        raise RuntimeError("Process died unexpectedly (likely killed by OS Out-Of-Memory).")
# ============================================================================

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
                 cleanup: bool = True, keep_best: bool = True, memory_type: str = "full"):
        self.model_id = model_id
        self.task = task
        self.max_iterations = max_iterations
        self.base_model_path = model_id
        self.num_samples = num_samples
        self.cleanup = cleanup
        self.keep_best = keep_best
        self.memory_type = memory_type
        self.trial_history = []
        self._seen_configs: set = set()
        self.llm = LLMDecisionMaker(model_id, task, max_iterations, memory_type=self.memory_type)
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

    def _parse_requested_tasks(self) -> list:
        if isinstance(self.task, str):
            return [t.strip() for t in self.task.split(",") if t.strip()]
        return [str(t).strip() for t in self.task if str(t).strip()]

    def _is_same_num_samples(self, cached_samples) -> bool:
        """baseline 可重用條件：num_samples 必須完全相同（含 None）。"""
        return cached_samples == self.num_samples

    def _aggregate_details(self, requested_tasks: list, details: dict) -> dict:
        n = len(requested_tasks)
        if n == 0:
            return {"accuracy": 0.0, "latency": 0.0, "vram": 0.0, "emissions": 0.0}

        total_acc = sum(details[t].get("accuracy", 0.0) for t in requested_tasks)
        total_lat = sum(details[t].get("latency", 0.0) for t in requested_tasks)
        total_emit = sum(details[t].get("emissions", 0.0) for t in requested_tasks)
        max_vram = max(details[t].get("vram", 0.0) for t in requested_tasks)

        return {
            "accuracy": total_acc / n,
            "latency": total_lat / n,
            "vram": max_vram,
            "emissions": total_emit / n,
        }

    def _load_task_detail_from_result_file(self, task_name: str) -> dict:
        """嘗試從 baseline 單一 task 結果檔讀取可重用指標。"""
        model_name = Path(self.model_id).name
        result_path = self._baseline_cache_dir / model_name / f"{task_name}_results.json"
        if not result_path.exists():
            return None

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.warning(f"讀取 baseline task 檔失敗 {result_path}: {e}")
            return None

        cached_samples = raw.get("num_samples")
        if self.num_samples is None:
            default_samples = None
            config_path = _ROOT_DIR / "Evals" / "config" / "dataset_config.yaml"
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    dataset_cfg = yaml.safe_load(f) or {}
                task_cfg = dataset_cfg.get(task_name, {})
                default_samples = task_cfg.get("num_samples")
            except Exception as e:
                logger.warning(f"讀取 dataset_config.yaml 失敗 {config_path}: {e}")

            if cached_samples != default_samples:
                logger.info(
                    f"baseline task {task_name} 樣本數不符 dataset 預設，"
                    f"cached={cached_samples}, default={default_samples}"
                )
                return None
        elif not self._is_same_num_samples(cached_samples):
            return None

        return {
            "accuracy": raw.get("accuracy", raw.get("pass@1", 0.0)),
            "latency": raw.get("total_generation_time_sec", 0.0),
            "vram": raw.get("gpu_peak_mb", 0.0) / 1024.0,
            "emissions": raw.get("emissions_kg_co2", 0.0),
        }

    def _load_or_run_baseline(self) -> dict:
        """載入快取的 baseline，若不存在則重新評估並快取。"""
        requested_tasks = self._parse_requested_tasks()
        reused_details = {}

        if self._baseline_cache_path.exists():
            with open(self._baseline_cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)

            cached_samples = cached.get("num_samples")
            cached_details = cached.get("details") if isinstance(cached.get("details"), dict) else {}

            if self._is_same_num_samples(cached_samples):
                reused_details = {
                    task: cached_details[task]
                    for task in requested_tasks
                    if task in cached_details
                }

                if len(reused_details) == len(requested_tasks):
                    logger.info(f"使用快取 baseline: {self._baseline_cache_path}")
                    logger.info(f"  accuracy={cached['accuracy']:.4f}, latency={cached['latency']:.4f}s, "
                                f"vram={cached['vram']:.4f}GB, emissions={cached['emissions']:.6f}kg CO2")
                    return cached
            else:
                logger.info("baseline 快取 num_samples 不一致，將只重用可匹配的 task 檔並補算缺少 task。")

        # 若完整 cache 不足，嘗試從既有 task 結果檔補齊（可支援新增 task 的情境）
        missing_tasks = [task for task in requested_tasks if task not in reused_details]
        recovered = 0
        for task in list(missing_tasks):
            detail = self._load_task_detail_from_result_file(task)
            if detail is not None:
                reused_details[task] = detail
                recovered += 1

        missing_tasks = [task for task in requested_tasks if task not in reused_details]
        if recovered:
            logger.info(f"已從既有 baseline task 結果重用 {recovered} 個 task。")

        if missing_tasks:
            logger.info(f"baseline 尚缺 task，將僅評估: {missing_tasks}")
            # eval_results = run_evaluation(
            #     self.base_model_path,
            #     ",".join(missing_tasks),
            #     weights=self.weights,
            #     baseline_metrics=None,
            #     num_samples=self.num_samples,
            #     output_dir=str(self._baseline_cache_dir),
            # )
            eval_results = run_isolated(
                run_evaluation,
                self.base_model_path,
                ",".join(missing_tasks),
                weights=self.weights,
                baseline_metrics=None,
                num_samples=self.num_samples,
                output_dir=str(self._baseline_cache_dir),
            )
            reused_details.update(eval_results.get("details", {}))
        else:
            logger.info("baseline 所有 task 都可重用，無需重新評估。")

        aggregate = self._aggregate_details(requested_tasks, reused_details)
        cache_entry = {
            "model_id": self.model_id,
            "task": self.task,
            "num_samples": self.num_samples,
            "accuracy": aggregate["accuracy"],
            "latency": aggregate["latency"],
            "vram": aggregate["vram"],
            "emissions": aggregate["emissions"],
            "details": {task: reused_details[task] for task in requested_tasks},
        }
        with open(self._baseline_cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_entry, f, indent=2, ensure_ascii=False)
        logger.info(f"基線已快取至: {self._baseline_cache_path}")
        return cache_entry

    def optimize(self):
        self.baseline_metrics = self._load_or_run_baseline()
        logger.info(f"基線建立完成: {self.baseline_metrics}")

        # 儲存實驗設定
        self._save_experiment_config()


        trial_dirs = []  # 追蹤所有生成的 trial 目錄

        for i in range(1, self.max_iterations + 1):
            logger.info(f"\n===== Iteration {i} =====")
            
            # Check available memory before each iteration
            import psutil
            import time
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                logger.warning(f"⚠️  Memory usage high ({mem.percent}%), cleaning up...")
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(5)

            # Step 1: LLM 決策（含去重重試）
            pareto = get_pareto_frontier(self.trial_history)
            _MAX_DUP_RETRIES = 3
            suggestion, llm_output = None, None
            for _retry in range(_MAX_DUP_RETRIES):
                _s, _raw = self.llm.get_suggestion(i, self.trial_history, pareto=pareto, weights=self.weights)
                _fp = self._config_fingerprint(_s)
                if _fp not in self._seen_configs:
                    suggestion, llm_output = _s, _raw
                    self._seen_configs.add(_fp)
                    break
                logger.warning(f"[去重] LLM 建議重複 config (retry {_retry+1}/{_MAX_DUP_RETRIES})")
            else:
                logger.warning(f"Iteration {i}: LLM 無法產生新 config，跳過")
                self.trial_history.append({
                    "iteration": i, "config": None,
                    "suggestion": None,
                    "metrics": {"score": 0.0}, "model_path": None,
                    "error": "重複 config，已跳過",
                    "trial_name": f"trial_{i:03d}_skipped",
                })
                self.save_history()
                continue

            logger.info(f"建議: {suggestion.mode} | 理由: {suggestion.reasoning}")

            # Step 2: 建立 trial 目錄
            trial_name = _make_trial_name(i, suggestion)
            trial_dir = str(self.exp_dir / trial_name)
            Path(trial_dir).mkdir(parents=True, exist_ok=True)
            # 只要 trial 目錄被建立，就納入清理清單；避免壓縮失敗時遺留空目錄
            trial_dirs.append(trial_dir)
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
                    sparse_out = trial_dir if (not has_asvd and not has_quant) else str(Path(trial_dir) / "sparse")
                    current_model = run_isolated(run_sparse, current_model, suggestion, output_dir=sparse_out)

                if has_asvd:
                    asvd_out = trial_dir if not has_quant else str(Path(trial_dir) / "asvd")
                    current_model = run_isolated(run_asvd, current_model, suggestion, output_dir=asvd_out)

                if has_quant:
                    current_model = run_isolated(run_quantization, current_model, suggestion, output_dir=trial_dir)
            # try:
                # if has_sparse:
                #     # 若後面還有其他步驟，存到子目錄；否則直接存到 trial_dir
                #     sparse_out = trial_dir if (not has_asvd and not has_quant) else str(Path(trial_dir) / "sparse")
                #     current_model = run_sparse(current_model, suggestion, output_dir=sparse_out)

                # if has_asvd:
                #     asvd_out = trial_dir if not has_quant else str(Path(trial_dir) / "asvd")
                #     current_model = run_asvd(current_model, suggestion, output_dir=asvd_out)

                # if has_quant:
                #     current_model = run_quantization(current_model, suggestion, output_dir=trial_dir)

            except Exception as e:
                logger.error(f"壓縮失敗 (iteration {i}): {e}")
                import traceback; traceback.print_exc()
                self.trial_history.append({
                    "iteration": i, "config": suggestion.to_log_dict(),
                    "suggestion": llm_output,
                    "metrics": {"score": 0.0}, "model_path": None, "error": str(e),
                    "trial_name": trial_name,
                })
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.save_history()
                continue

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

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Step 4: 評估
            # output_dir=exp_dir，評估器內部會 append Path(current_model).name = trial_name
            # → 結果存到 exp_dir/trial_name/ = trial_dir/
            # metrics = run_evaluation(
            #     current_model, self.task,
            #     weights=self.weights,
            #     baseline_metrics=self.baseline_metrics,
            #     num_samples=self.num_samples,
            #     output_dir=str(self.exp_dir),
            # )
            metrics = run_isolated(
                run_evaluation,
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
                "details": self.baseline_metrics.get("details", {}),
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

    @staticmethod
    def _config_fingerprint(suggestion) -> str:
        return json.dumps(suggestion.to_log_dict(), sort_keys=True, ensure_ascii=False)

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
    parser.add_argument("--acc_weight", type=float, default=0.6)
    parser.add_argument("--lat_weight", type=float, default=0.1)
    parser.add_argument("--vram_weight", type=float, default=0.1)
    parser.add_argument("--emit_weight", type=float, default=0.2)
    parser.add_argument("--no-cleanup", dest="cleanup", action="store_false",
                        help="跑完後不刪除 trial 模型（預設：刪除）")
    parser.add_argument("--no-keep-best", dest="keep_best", action="store_false",
                        help="刪除時連最佳 trial 也刪（預設：保留最佳）")
    parser.set_defaults(cleanup=True, keep_best=True)

    parser.add_argument("--memory_type", type=str, choices=["full", "window", "summary"], default="full",
                        help="單次執行時使用的 memory 模式")
    parser.add_argument("--benchmark_runs", type=int, default=1,
                        help="大於 1 時，將自動對三種 memory 模式各執行 N 次並輸出 Markdown 比較表")

    args = parser.parse_args()
    weights = {
        "acc": args.acc_weight, "lat": args.lat_weight,
        "vram": args.vram_weight, "emit": args.emit_weight,
    }
    if args.benchmark_runs > 1:
        memory_modes = ["full", "window", "summary"]
        descriptions = {
            "full": "全部實驗結果", 
            "window": "最近 5 個", 
            "summary": "LLM summary"
        }
        results_stats = []
        
        # 1. Prepare a file to save incremental results so data isn't lost if it crashes late
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = _ROOT_DIR / "tuning_results" / f"benchmark_report_{ts}.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Green AI Memory Benchmark ({args.benchmark_runs} runs per type)\n\n")
            f.write("| Memory 方法 | Description | Mean Score | Best Score | Std 標準差 | Valid Runs |\n")
            f.write("|---|---|---|---|---|---|\n")

        logger.info("\n" + "="*60)
        logger.info(f"STARTING MEMORY BENCHMARK ({args.benchmark_runs} runs per type)")
        logger.info(f"Live results will be saved incrementally to: {report_path}")
        logger.info("="*60)
        
        for mode in memory_modes:
            scores = []
            for run in range(args.benchmark_runs):
                logger.info(f"\n>>> Running Benchmark: Mode={mode}, Run={run+1}/{args.benchmark_runs} <<<")
                
                # 2. Add Fault Tolerance (try...except)
                try:
                    orch = OptimizationOrchestrator(
                        args.model_id, args.task, args.max_iterations, weights,
                        num_samples=args.num_samples,
                        cleanup=args.cleanup, keep_best=args.keep_best,
                        memory_type=mode
                    )
                    orch.optimize()
                    
                    # Only append valid scores
                    if orch.best_score > -float('inf'):
                        scores.append(orch.best_score)
                        
                except Exception as e:
                    logger.error(f"❌ Run {run+1} for mode {mode} failed critically: {e}")
                    import traceback
                    traceback.print_exc()
                    # It skips appending to `scores`, moving safely to the next run
            
            # 3. Calculate statistics only for successful runs
            if scores:
                mean_score = statistics.mean(scores)
                best_score = max(scores)
                std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
            else:
                mean_score = best_score = std_score = 0.0
                
            valid_runs = len(scores)
            run_info = f"{valid_runs}/{args.benchmark_runs}"
            
            results_stats.append((mode, descriptions[mode], mean_score, best_score, std_score, run_info))
            
            # 4. Incrementally write to the Markdown file
            with open(report_path, "a", encoding="utf-8") as f:
                f.write(f"| {mode} | {descriptions[mode]} | {mean_score:.4f} | {best_score:.4f} | {std_score:.4f} | {run_info} |\n")
            
            # 5. Print current progress to the console
            print("\n" + "-"*70)
            print(f"🟢 CURRENT BENCHMARK PROGRESS (Saved to {report_path})")
            print("| Memory 方法 | Description | Mean Score | Best Score | Std 標準差 | Valid Runs |")
            print("|---|---|---|---|---|---|")
            for m, d, ms, bs, ss, vr in results_stats:
                print(f"| {m} | {d} | {ms:.4f} | {bs:.4f} | {ss:.4f} | {vr} |")
            print("-" * 70 + "\n")
            
        logger.info(f"✅ Benchmark fully completed. Final report saved to: {report_path}")
            
    else:
        # Standard execution for a single run
        orchestrator = OptimizationOrchestrator(
            args.model_id, args.task, args.max_iterations, weights,
            num_samples=args.num_samples,
            cleanup=args.cleanup, keep_best=args.keep_best,
            memory_type=args.memory_type
        )
        orchestrator.optimize()