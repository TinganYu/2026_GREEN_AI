import logging
import gc
import torch
import json
from pathlib import Path
import argparse
from llm_client import LLMDecisionMaker
from executors import run_asvd, run_quantization, run_evaluation
from utils import get_pareto_frontier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModularOrchestrator")

CURRENT_DIR = Path(__file__).resolve().parent
GLOBAL_TUNER_DIR = CURRENT_DIR.parent
ROOT_DIR = GLOBAL_TUNER_DIR.parent

class OptimizationOrchestrator:
    def __init__(self, model_id: str, task: str, max_iterations: int = 10, weights: dict = None):
        self.model_id = model_id
        self.task = task
        self.max_iterations = max_iterations
        self.base_model_path = model_id
        self.trial_history = []
        self.llm = LLMDecisionMaker(model_id, task, max_iterations)
        self.best_score = -float('inf')
        self.best_result = None
        self.weights = weights or {"acc": 0.5, "lat": 0.1, "vram": 0.2, "emit": 0.2}
        self.baseline_metrics = None
        
        # Setup output directory
        model_name = Path(model_id).name
        self.output_path = ROOT_DIR / "results" / f"tuning_{model_name}_{task.replace(',', '_')}"
        self.output_path.mkdir(parents=True, exist_ok=True)

    def optimize(self):
        logger.info("--- Extracting Baseline Metrics ---")
        baseline_results = run_evaluation(self.base_model_path, self.task, weights=self.weights, baseline_metrics=None)
        self.baseline_metrics = baseline_results
        logger.info(f"Baseline established: {self.baseline_metrics}")

        for i in range(1, self.max_iterations + 1):
            logger.info(f"\n===== Iteration {i} =====")
            
            # Step 1: LLM 決策
            # Compute pareto frontier for LLM guidance
            pareto = get_pareto_frontier(self.trial_history)
            suggestion = self.llm.get_suggestion(i, self.trial_history, pareto=pareto)
            logger.info(f"💡Suggestion: {suggestion.mode} | Reasoning: {suggestion.reasoning}")
            
            # Step 2: 執行壓縮
            current_model = self.base_model_path
            
            if suggestion.mode in ["asvd_only", "hybrid"]:
                current_model = run_asvd(
                    model_id=current_model,
                    alpha=suggestion.alpha,
                    param_ratio_target=suggestion.param_ratio_target,
                    scaling_method=suggestion.scaling_method
                )
            
            if suggestion.mode in ["quant_only", "hybrid"]:
                current_model = run_quantization(
                    model_path=current_model,
                    method=suggestion.quant_method,
                    bits=suggestion.quant_bits,
                    group_size=suggestion.quant_group_size,
                    quant_type=suggestion.quant_type,
                    desc_act=suggestion.desc_act,
                    use_double_quant=suggestion.use_double_quant
                )
            
            # Step 3: 評估與記錄
            metrics = run_evaluation(
                current_model, 
                self.task, 
                weights=self.weights,
                baseline_metrics=self.baseline_metrics
            )
            # 記錄結果
            trial_data = {
                "iteration": i,
                "config": suggestion.to_log_dict(),
                "suggestion": suggestion.model_dump(),
                "metrics": metrics,
                "model_path": str(current_model)
            }
            self.trial_history.append(trial_data)

            # 檢查並更新最佳模型
            self._update_best(trial_data)

            # 每輪存檔一次，確保安全
            self.save_history()
            
            # Step 4: 清理記憶體
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 最終總結輸出
        if self.best_result:
            logger.info("\n" + "=" * 40)
            logger.info("🏆 OPTIMIZATION COMPLETE - BEST MODEL")
            logger.info("=" * 40)
            logger.info(f"Iteration: {self.best_result['iteration']}")
            logger.info(f"Score: {self.best_score:.4f}")
            logger.info(f"Acc: {self.best_result['metrics']['accuracy']:.4f}")
            logger.info(f"Lat: {self.best_result['metrics']['latency']:.4f}")
            logger.info(f"VRAM: {self.best_result['metrics']['vram']:.4f} GB")
            logger.info(f"Config: {self.best_result['config']}")
            logger.info(f"Path: {self.best_result['model_path']}")
            logger.info("=" * 40 + "\n")

    def save_history(self):
        """將 trial_history 輸出為 JSON 檔案"""
        output_path = self.output_path / "optimization_results.json"
        # 清理資料以確保 JSON 可序列化 (處理可能存在的 Tensor 或 Path 物件)
        serializable_history = self._make_serializable(self.trial_history)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 --- Iteration history saved to {output_path} ---")


    def _make_serializable(self, data):
        """遞迴清理資料，將 non-standard 類型轉為原生 Python 類型"""
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(v) for v in data]
        elif isinstance(data, (Path, torch.device)): # 處理路徑或設備物件
            return str(data)
        return data
    
    def _update_best(self, result):
        """檢查並更新最佳試驗結果"""
        if result['metrics']['score'] > self.best_score:
            self.best_score = result['metrics']['score']
            self.best_result = result
            logger.info(f"🎉 New Best Found! Score: {self.best_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global Tuner Orchestrator")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Huggingface model ID")
    parser.add_argument("--task", type=str, default="gsm8k", help="Evaluation task")
    parser.add_argument("--max_iterations", type=int, default=10, help="Max optimization steps")
    parser.add_argument("--acc_weight", type=float, default=0.7, help="Weight for Accuracy")
    parser.add_argument("--lat_weight", type=float, default=0.1, help="Weight for Latency")
    parser.add_argument("--vram_weight", type=float, default=0.2, help="Weight for VRAM")
    parser.add_argument("--emit_weight", type=float, default=0.0, help="Weight for Emissions")
    
    args = parser.parse_args()
    
    weights = {"acc": args.acc_weight, "lat": args.lat_weight, "vram": args.vram_weight, "emit": args.emit_weight}
    orchestrator = OptimizationOrchestrator(args.model_id, args.task, args.max_iterations, weights)
    orchestrator.optimize()
