import logging
import gc
import torch
from llm_client import LLMDecisionMaker
from executors import run_asvd, run_quantization, run_evaluation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModularOrchestrator")

class OptimizationOrchestrator:
    def __init__(self, model_id: str, task: str, max_iterations: int = 10):
        self.model_id = model_id
        self.task = task
        self.max_iterations = max_iterations
        self.base_model_path = model_id
        self.trial_history = []
        self.llm = LLMDecisionMaker(model_id, task, max_iterations)

    def optimize(self):
        for i in range(1, self.max_iterations + 1):
            logger.info(f"\n===== Iteration {i} =====")
            
            # Step 1: LLM 決策
            suggestion = self.llm.get_suggestion(i, self.trial_history)
            logger.info(f"Suggestion: {suggestion.mode} | Reasoning: {suggestion.reasoning}")
            
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
            metrics = run_evaluation(current_model, self.task)
            
            self.trial_history.append({
                "iteration": i,
                "config": suggestion.to_log_dict(),
                "suggestion": suggestion.model_dump(),
                "metrics": metrics
            })
            
            # Step 4: 清理記憶體
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    orchestrator = OptimizationOrchestrator(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        task="gsm8k",
        max_iterations=5
    )
    orchestrator.optimize()
