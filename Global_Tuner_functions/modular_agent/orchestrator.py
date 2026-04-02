import logging
import gc
import shutil
import torch
import time
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
import signal
import sys
import multiprocessing as mp

# ... (keep the _worker function as it is) ...

def run_isolated(func, *args, **kwargs):
    """
    Runs a function in a completely isolated process using the 'spawn' context.
    This guarantees that the OS will wipe 100% of the allocated VRAM when the function finishes.
    """
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = ctx.Process(target=_worker, args=(queue, func) + args, kwargs=kwargs)
    p.start()

    def cleanup_child(signum, frame):
        logger.warning(f"⚠️ Main process received kill signal ({signum})! Terminating child (PID: {p.pid})...")
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)
            if p.is_alive():
                p.kill()  # Force kill if it refuses to terminate
        sys.exit(1)

    # Temporarily override standard kill signals to protect the child process
    original_sigterm = signal.signal(signal.SIGTERM, cleanup_child)
    original_sigint = signal.signal(signal.SIGINT, cleanup_child)

    try:
        p.join()
    except Exception as e:
        logger.warning(f"⚠️ Exception in main process! Terminating child (PID: {p.pid})...")
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)
            if p.is_alive():
                p.kill()
        raise
    finally:
        # Restore normal signal behavior once the isolated function finishes
        signal.signal(signal.SIGTERM, original_sigterm)
        signal.signal(signal.SIGINT, original_sigint)

    if not queue.empty():
        res = queue.get()
        if res["status"] == "success":
            return res["result"]
        else:
            raise RuntimeError(f"Isolated process failed: {res['error']}\n{res['traceback']}")
    else:
        raise RuntimeError("Process died unexpectedly (likely killed by OS Out-Of-Memory).")
# def run_isolated(func, *args, **kwargs):
#     """
#     Runs a function in a completely isolated process using the 'spawn' context.
#     This guarantees that the OS will wipe 100% of the allocated VRAM when the function finishes.
#     """
#     ctx = mp.get_context('spawn')
#     queue = ctx.Queue()
#     p = ctx.Process(target=_worker, args=(queue, func) + args, kwargs=kwargs)
#     p.start()
#     try:
#         p.join()
#     except (KeyboardInterrupt, SystemExit):
#         # If you cancel the main script, kill the isolated process immediately
#         logger.warning(f"⚠️ Main process interrupted! Terminating isolated child process (PID: {p.pid})...")
#         p.terminate()
#         p.join()
#         raise

#     if not queue.empty():
#         res = queue.get()
#         if res["status"] == "success":
#             return res["result"]
#         else:
#             raise RuntimeError(f"Isolated process failed: {res['error']}\n{res['traceback']}")
#     else:
#         raise RuntimeError("Process died unexpectedly (likely killed by OS Out-Of-Memory).")
# ============================================================================

def _make_trial_name(i: int, suggestion) -> str:
    """生成 trial 目錄名稱，包含編號、方法與關鍵參數"""
    parts = [f"trial_{i:03d}"]
    mode = suggestion.mode
    
    # Append the base mode name first
    parts.append(mode)

    if mode == "sparse_unstructured":
        ratio_pct = int((suggestion.sparsity_ratio or 0.5) * 100)
        parts.append(f"{ratio_pct}pct")
        
    elif mode == "sparse_structured":
        struct = suggestion.sparsity_structure or "2:4"
        parts.append(struct.replace(':', 'x'))

    if mode in ("asvd_only", "hybrid_asvd_bnb") and suggestion.alpha is not None:
        ratio_str = f"{int((suggestion.param_ratio_target or 0.9) * 100):03d}"
        alpha_str = f"{int((suggestion.alpha or 0.5) * 100):02d}"
        parts.append(f"r{ratio_str}_a{alpha_str}")

    # Handle Quantization specific parameters
    if mode == "gptq":
        b = suggestion.quant_bits or 4
        g = suggestion.quant_group_size or 128
        fmt = suggestion.quant_format or "gptq"
        parts.append(f"{b}bit_g{g}_{fmt}")
        
    elif mode == "awq":
        g = suggestion.quant_group_size or 128
        parts.append(f"4bit_g{g}") # AWQ is fixed to 4-bit in your space
        
    elif mode == "qqq":
        g = suggestion.quant_group_size or 128
        parts.append(f"4bit_g{g}") # QQQ is fixed to 4-bit
        
    elif mode in ("bnb", "hybrid_asvd_bnb"):
        b = getattr(suggestion, 'quant_bits', 4)
        parts.append(f"{b}bit")

    return "_".join(parts)


class OptimizationOrchestrator:
    def __init__(self, model_id: str, task: str, max_iterations: int = 10, max_time_hours: float = 0.0,
                 weights: dict = None, num_samples: int = None,
                 cleanup: bool = True, keep_best: bool = True, memory_type: str = "full", base_dir: Path = None, pen_t: float = 0.15, pen_a: float = 10.0,
                 targets: dict = None, patience: int = 2):
        self.model_id = model_id
        self.task = task
        # Handle the "disable" logic
        self.max_iterations = max_iterations if max_iterations > 0 else 100000
        self.max_time_hours = max_time_hours if max_time_hours > 0.0 else float('inf')
        if self.max_iterations == 100000 and self.max_time_hours == float('inf'):
            logger.warning("⚠️ Both max_iterations and max_time_hours are disabled! Setting max_iterations to 10 as a safety fallback.")
            self.max_iterations = 10

        self.base_model_path = model_id
        self.num_samples = num_samples
        self.cleanup = cleanup
        self.keep_best = keep_best
        self.memory_type = memory_type
        self.trial_history = []
        self._seen_configs: set = set()
        self.llm = LLMDecisionMaker(model_id, task, max_iterations, memory_type=self.memory_type, pen_t=pen_t, pen_a=pen_a)
        self.best_score = -float('inf')
        self.best_result = None
        self.weights = weights or {"acc": 3.0, "lat": 1.0, "vram": 1.0, "emit": 1.0}
        self.pen_t = pen_t
        self.pen_a = pen_a
        self.targets = targets or {}
        self.patience_limit = patience
        self.current_patience = patience
        self.target_met = False
        self.best_target_score = -float('inf')
        self.baseline_metrics = None

        # 實驗目錄：tuning_results/exp_{model}_{task}_{timestamp}/
        model_name = Path(model_id).name
        task_str = task.replace(",", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_dir = base_dir if base_dir else _ROOT_DIR / "tuning_results"
        self.exp_dir = parent_dir / f"exp_{model_name}_{task_str}_{memory_type}_{ts}" # Optional: added memory_type to folder name for clarity
        # self.exp_dir = _ROOT_DIR / "tuning_results" / f"exp_{model_name}_{task_str}_{ts}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"實驗目錄: {self.exp_dir}")
        self.llm.exp_dir = self.exp_dir

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

    # def _load_task_detail_from_result_file(self, task_name: str) -> dict:
    #     """嘗試從 baseline 單一 task 結果檔讀取可重用指標。"""
    #     model_name = Path(self.model_id).name
    #     result_path = self._baseline_cache_dir / model_name / f"{task_name}_results.json"
    #     if not result_path.exists():
    #         return None

    #     try:
    #         with open(result_path, "r", encoding="utf-8") as f:
    #             raw = json.load(f)
    #     except Exception as e:
    #         logger.warning(f"讀取 baseline task 檔失敗 {result_path}: {e}")
    #         return None

    #     cached_samples = raw.get("num_samples")
    #     if self.num_samples is None:
    #         default_samples = None
    #         config_path = _ROOT_DIR / "Evals" / "config" / "dataset_config.yaml"
    #         try:
    #             with open(config_path, "r", encoding="utf-8") as f:
    #                 dataset_cfg = yaml.safe_load(f) or {}
    #             task_cfg = dataset_cfg.get(task_name, {})
    #             default_samples = task_cfg.get("num_samples")
    #         except Exception as e:
    #             logger.warning(f"讀取 dataset_config.yaml 失敗 {config_path}: {e}")

    #         if cached_samples != default_samples:
    #             logger.info(
    #                 f"baseline task {task_name} 樣本數不符 dataset 預設，"
    #                 f"cached={cached_samples}, default={default_samples}"
    #             )
    #             return None
    #     elif not self._is_same_num_samples(cached_samples):
    #         return None

    #     return {
    #         "accuracy": raw.get("accuracy", raw.get("pass@1", 0.0)),
    #         "latency": raw.get("total_generation_time_sec", 0.0),
    #         "vram": raw.get("gpu_peak_mb", 0.0) / 1024.0,
    #         "emissions": raw.get("emissions_kg_co2", 0.0),
    #     }

    def _find_task_in_any_cache(self, task_name: str) -> dict:
        """掃描所有 baseline 快取檔，尋找是否有此 task 的評估結果（支援單一與多重任務提取）"""
        model_name = Path(self.model_id).name
        
        # 嚴格定義我們正在尋找的樣本數 (避免 _is_same_num_samples 邏輯出錯)
        expected_samples = self.num_samples if self.num_samples and self.num_samples > 0 else None
        
        for file_path in self._baseline_cache_dir.glob(f"{model_name}_*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                
                # 確保樣本數完全一致
                if raw.get("num_samples") != expected_samples:
                    continue
                    
                # 檢查這個快取檔裡面有沒有包含我們需要的 task
                details = raw.get("details", {})
                if task_name in details:
                    logger.info(f"💡 成功從快取 '{file_path.name}' 提取 '{task_name}' 的基線數據！")
                    return details[task_name]
            except Exception as e:
                logger.error(f"無法讀取快取檔 {file_path}: {e}")
                continue
        return None

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
            detail = self._find_task_in_any_cache(task) #self._load_task_detail_from_result_file(task)
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
                pen_t=self.pen_t,
                pen_a=self.pen_a
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
        self.llm.baseline_metrics = self.baseline_metrics

        # 儲存實驗設定
        self._save_experiment_config()


        trial_dirs = []  # 追蹤所有生成的 trial 目錄
        start_time = time.time()
        max_time_seconds = self.max_time_hours * 3600

        i = 1 # Initialize counter for the while loop
        while i <= self.max_iterations:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds >= max_time_seconds:
                elapsed_hours = elapsed_seconds / 3600
                logger.info(f"\n⏳ 達到時間上限 ({elapsed_hours:.2f} / {self.max_time_hours} hours)。安全結束優化程序。")
                break

            logger.info(f"\n===== Iteration {i} =====")
            
            # Check available memory before each iteration
            import psutil
            mem = psutil.virtual_memory()
            if mem.percent > 90:
                logger.warning(f"⚠️  Memory usage high ({mem.percent}%), cleaning up...")
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(5)

            # Step 1: LLM 決策（含去重重試）
            pareto = get_pareto_frontier(self.trial_history)
            _MAX_DUP_RETRIES = 5
            suggestion, llm_output = None, None

            rejected_this_iter = []
            for _retry in range(_MAX_DUP_RETRIES):
                _s, _raw = self.llm.get_suggestion(i, self.trial_history, pareto=pareto, weights=self.weights, rejected_configs=rejected_this_iter, targets=self.targets)
                _fp = self._config_fingerprint(_s)
                if _fp not in self._seen_configs:
                    suggestion, llm_output = _s, _raw
                    self._seen_configs.add(_fp)
                    break
                logger.warning(f"[去重複] LLM 建議重複 config (retry {_retry+1}/{_MAX_DUP_RETRIES})")
                rejected_this_iter.append(_s.to_log_dict())
            else:
                logger.warning(f"Iteration {i}: LLM 無法產生新 config，跳過")
                self.trial_history.append({
                    "iteration": i, "config": {},
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
            mode = suggestion.mode # New flattened mode string

            # The new explicit boolean flags based on your teammate's ALL_MODES
            has_sparse = mode in ["sparse_unstructured", "sparse_structured"]
            has_asvd = mode in ["asvd_only", "hybrid_asvd_bnb"]
            has_quant = mode in ["gptq", "awq", "qqq", "bnb", "hybrid_asvd_bnb"]
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
                pen_t=self.pen_t,
                pen_a=self.pen_a
            )

            if self.baseline_metrics:
                cur_acc = metrics.get('accuracy', 0.0)
                cur_lat = metrics.get('latency', 0.0)
                cur_vram = metrics.get('vram', 0.0)
                cur_emit = metrics.get('emissions', 0.0)
                
                base_acc = self.baseline_metrics.get('accuracy', 0.0)
                base_lat = self.baseline_metrics.get('latency', 0.0)
                base_vram = self.baseline_metrics.get('vram', 0.0)
                base_emit = self.baseline_metrics.get('emissions', 0.0)

                # Store the reductions inside metrics so history/LLM can use them
                metrics['vram_red'] = (base_vram - cur_vram) / base_vram if base_vram > 0 else 0.0
                metrics['lat_red']  = (base_lat - cur_lat) / base_lat if base_lat > 0 else 0.0
                metrics['emit_red'] = (base_emit - cur_emit) / base_emit if base_emit > 0 else 0.0
                metrics['acc_drop'] = base_acc - cur_acc
                metrics['acc_diff_pct'] = ((cur_acc - base_acc) / base_acc) * 100 if base_acc > 0 else 0.0

            # Check targets if any are set
            targets_active = any(v > 0 for v in self.targets.values())
            if targets_active and self.baseline_metrics:                
                # Verify all active targets
                v_tar = self.targets.get('vram_pct', 0.0)
                l_tar = self.targets.get('lat_pct', 0.0)
                a_tar = self.targets.get('acc_drop', 0.0)
                e_tar = self.targets.get('emit_pct', 0.0)

                meets_vram = (metrics['vram_red'] >= v_tar) if v_tar > 0 else True
                meets_lat = (metrics['lat_red'] >= l_tar) if l_tar > 0 else True
                meets_acc = (metrics['acc_drop'] <= a_tar) if a_tar > 0 else True
                meets_emit = (metrics['emit_red'] >= e_tar) if e_tar > 0 else True
                
                if meets_vram and meets_lat and meets_acc and meets_emit:
                    current_score = metrics['score']
                    if not self.target_met:
                        logger.info(f"🎯 達標！成功滿足所有設定目標。進入耐心 (Patience) 模式 (剩餘 {self.patience_limit} 次)。")
                        self.target_met = True
                        self.best_target_score = current_score
                    else:
                        if current_score > self.best_target_score:
                            logger.info(f"🎯 找到達標且分數更高的配置！重置耐心值為 {self.patience_limit}。")
                            self.best_target_score = current_score
                            self.current_patience = self.patience_limit
                        else:
                            self.current_patience -= 1
                            logger.info(f"⏳ 配置達標，但總分未超越最佳紀錄。剩餘耐心值: {self.current_patience}")
                            
                    if self.current_patience <= 0:
                        logger.info("🛑 耐心值耗盡。已找到符合目標的配置，提前結束優化程序。")
                        
                        # Save history for this final trial before breaking
                        trial_data = {
                            "iteration": i, "trial_name": trial_name, "config": suggestion.to_log_dict(),
                            "suggestion": llm_output, "metrics": metrics, "model_path": str(current_model), "trial_dir": trial_dir
                        }
                        self.trial_history.append(trial_data)
                        self._update_best(trial_data)
                        self.save_history()
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        if self.cleanup:
                            self._eager_cleanup_trials(trial_dirs)
                        break

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

            if self.cleanup:
                self._eager_cleanup_trials(trial_dirs)
            
            i += 1  


        # 最終報告
        self._print_summary()
        self._save_pareto_results()

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

    def _eager_cleanup_trials(self, trial_dirs: list):
        """Deletes trial directories immediately during the run to save disk space, keeping only the best."""
        best_dir = str(self.best_result["trial_dir"]) if self.best_result else None
        
        for d in trial_dirs:
            # Skip the best directory if we are configured to keep it
            if self.keep_best and d == best_dir:
                continue
                
            # If the directory still exists, delete it
            dir_path = Path(d)
            if dir_path.exists():
                try:
                    shutil.rmtree(d, ignore_errors=True)
                    logger.info(f"🗑️ Eagerly deleted trial model to save space: {dir_path.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to delete {d}: {e}")

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

    def _save_pareto_results(self):
        """Extracts the Pareto frontier from history and saves it to a dedicated file."""
        from utils import get_pareto_frontier
        import json
        
        pareto_trials = get_pareto_frontier(self.trial_history)
        if not pareto_trials:
            logger.warning("No valid trials to form a Pareto frontier.")
            return

        pareto_path = self.exp_dir / "pareto_frontier.json"
        with open(pareto_path, "w", encoding="utf-8") as f:
            json.dump(self._make_serializable(pareto_trials), f, indent=2, ensure_ascii=False)
        logger.info(f"Pareto frontier successfully saved to: {pareto_path}")

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
    parser.add_argument("--max_iterations", type=int, default=10,
                        help="Maximum number of trials. Set to 0 to disable and rely only on time.")
    parser.add_argument("--max_time_hours", type=float, default=0.0,
                        help="Maximum time in hours. Set to 0 to disable and rely only on iterations.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="每個 dataset 的評估樣本數（None = 全部）")
    
    # Target arguments
    parser.add_argument("--target_vram_pct", type=float, default=0.0, 
                        help="Target VRAM reduction percentage (e.g., 0.5 for 50%%). 0 to disable.")
    parser.add_argument("--target_lat_pct", type=float, default=0.0, 
                        help="Target latency reduction percentage. 0 to disable.")
    parser.add_argument("--target_acc_drop", type=float, default=0.0, 
                        help="Maximum allowed accuracy drop in absolute points (e.g., 0.05). 0 to disable.")
    parser.add_argument("--target_emit_pct", type=float, default=0.0,
                        help="Target emissions reduction percentage. 0 to disable.")
    parser.add_argument("--patience", type=int, default=2, 
                        help="Iterations to wait after hitting targets to see if it can be improved.")

    parser.add_argument("--acc_weight",  type=float, default=3.0)
    parser.add_argument("--lat_weight",  type=float, default=1.0)
    parser.add_argument("--vram_weight", type=float, default=1.0)
    parser.add_argument("--emit_weight", type=float, default=1.0)
    parser.add_argument("--pen_t",       type=float, default=0.15,
                        help="Accuracy penalty 容忍量（絕對值，掉幅超過此值才扣分）")
    parser.add_argument("--pen_a",       type=float, default=10.0,
                        help="Accuracy penalty 放大倍率")
    parser.add_argument("--no-cleanup", dest="cleanup", action="store_false",
                        help="跑完後不刪除 trial 模型（預設：刪除）")
    parser.add_argument("--no-keep-best", dest="keep_best", action="store_false",
                        help="刪除時連最佳 trial 也刪（預設：保留最佳）")
    parser.set_defaults(cleanup=True, keep_best=True)

    parser.add_argument("--memory_type", type=str, choices=["full", "window", "summary", "tool"], default="full",
                        help="單次執行時使用的 memory 模式")
    parser.add_argument("--benchmark_runs", type=int, default=1,
                        help="大於 1 時，將自動對三種 memory 模式各執行 N 次並輸出 Markdown 比較表")

    args = parser.parse_args()
    weights = {
        "acc": args.acc_weight, "lat": args.lat_weight,
        "vram": args.vram_weight, "emit": args.emit_weight,
    }
    targets = {
        "vram_pct": args.target_vram_pct,
        "lat_pct": args.target_lat_pct,
        "acc_drop": args.target_acc_drop,
        "emit_pct": args.target_emit_pct,
    }
    if args.benchmark_runs > 1:
        memory_modes = [ "window", "summary", "tool"] #"full" 先不用，太久
        descriptions = {
            "full": "全部實驗結果", 
            "window": "最近 5 個", 
            "summary": "LLM summary",
            "tool": "Agentic Tool Retrieval"
        }
        test_iterations = {
            "window": 6,
            "summary": 10,
            "tool": 5
        }
        results_stats = []
        
        # 1. Prepare a file to save incremental results so data isn't lost if it crashes late
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        master_bench_dir = _ROOT_DIR / "tuning_results" / f"benchmark_{ts}"
        master_bench_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = master_bench_dir / "benchmark_report.md" # Save inside master folder
        
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
            current_max_iters = test_iterations.get(mode, args.max_iterations)
            for run in range(args.benchmark_runs):
                logger.info(f"\n>>> Running Benchmark: Mode={mode}, Run={run+1}/{args.benchmark_runs} <<<")
                
                # 2. Add Fault Tolerance (try...except)
                try:  #, args.max_iterations
                    orch = OptimizationOrchestrator(
                        args.model_id, args.task, max_iterations=current_max_iters,max_time_hours=args.max_time_hours, weights=weights,
                        num_samples=args.num_samples,
                        cleanup=args.cleanup, keep_best=args.keep_best,
                        memory_type=mode,
                        base_dir=master_bench_dir,
                        pen_t=args.pen_t,
                        pen_a=args.pen_a,
                        targets=targets,
                        patience=args.patience
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
            model_id=args.model_id, 
            task=args.task, 
            max_iterations=args.max_iterations,
            max_time_hours=args.max_time_hours, 
            weights=weights,
            num_samples=args.num_samples,
            cleanup=args.cleanup, 
            keep_best=args.keep_best,
            memory_type=args.memory_type,
            pen_t=args.pen_t,
            pen_a=args.pen_a,
            targets=targets,
            patience=args.patience
        )
        orchestrator.optimize()