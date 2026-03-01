import os
import gc
import sys
import json
import logging
import torch
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.KVpress import KVPressAgent
# from agents.tuner import AdaptiveKVController

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
config_path = BASE_DIR / "config" / "model_config.yaml"


class BaseEvaluator:
    """Base evaluator class with shared functionality for all datasets"""

    def __init__(self, config_path: str = config_path, **override_params):
        self.config_path = config_path
        
        # Load base agent
        base_agent = KVPressAgent(config_path, **override_params)
        self.config = base_agent.config
        
        # Decide which agent to wrap
        if self.config.get("tuning", {}).get("adaptive", False):
            self.agent = base_agent
            # self.agent = AdaptiveKVController(self.config, base_agent=base_agent)
            logger.info("Using AdaptiveKVController")
        else:
            self.agent = base_agent
            logger.info("Using base KVPressAgent")
        
        # Prepare results dir
        self.results_dir = self.config.get('evaluation', {}).get('results_dir', './results')
        os.makedirs(self.results_dir, exist_ok=True)

    def load_dataset(self) -> List[Dict]:
        """Override this method in subclasses to load specific datasets"""
        raise NotImplementedError("Subclasses must implement load_dataset()")

    def create_prompt(self, sample: Dict) -> str:
        """Override this method to create task-specific prompts"""
        raise NotImplementedError("Subclasses must implement create_prompt()")

    def evaluate_prediction(self, sample: Dict, prediction: str) -> Dict:
        """Override this method to implement task-specific evaluation metrics"""
        raise NotImplementedError("Subclasses must implement evaluate_prediction()")

    def evaluate_sample(self, sample: Dict, use_compression: bool = True, **gen_kwargs) -> Dict:
        """Generic sample evaluation - uses abstract methods for task-specific logic"""
        prompt = self.create_prompt(sample)
        start_time = datetime.now()
        
        try:
            if use_compression and hasattr(self.agent, "generate_with_compression"):
                response = self.agent.generate_with_compression(prompt, **gen_kwargs)
                kv_stats = self.agent.get_compression_stats() if hasattr(self.agent, "get_compression_stats") else {}
            else:
                response = self.agent.generate_response(prompt, **gen_kwargs)
                kv_stats = {}
            
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Task-specific evaluation
            eval_result = self.evaluate_prediction(sample, response)
            
            result = {
                "response": response,
                "prompt": prompt,
                "generation_time": generation_time,
                "error": None,
                "kv_stats": kv_stats,
                **eval_result  # Merge task-specific metrics
            }
            
        except Exception as e:
            generation_time = (datetime.now() - start_time).total_seconds()
            result = {
                "response": "",
                "prompt": prompt,
                "generation_time": generation_time,
                "error": str(e),
                "kv_stats": {}
            }
            logger.error(f"Error evaluating sample: {e}")
        
        return result

    def run_evaluation(self) -> Dict:
        """Main evaluation loop - shared across all tasks"""
        eval_cfg = self.config.get('evaluation', {})
        kv_cfg = self.config.get('kv_compression', {})
        use_compression = kv_cfg.get('enabled', True)
        
        logger.info("===================STARTING EVALUATION======================")
        logger.info(f"Model: {self.agent.model_name}, Compression: {use_compression}")
        
        # Reset peak memory
        self._reset_memory_tracking()
        
        # Load model and validate
        if getattr(self.agent, "model", None) is None:
            logger.info("Model not loaded yet — loading now.")
            self.agent.load_model()
        else:
            logger.info("Model already loaded — reusing instance.")
        compatibility = self.agent.validate_model_compatibility()
        
        if not compatibility.get('compatible', False) and use_compression:
            logger.warning("KV compression disabled due to compatibility issues")
            use_compression = False
        
        # Load dataset
        dataset = self.load_dataset()
        results = []
        total_time = 0
        peak_memory_during_gen = 0
        
        # Evaluate each sample
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating samples")):
            r = self.evaluate_sample(sample, use_compression)
            results.append(r)
            total_time += r["generation_time"]
            
            # Track peak memory
            if torch.cuda.is_available():
                device = getattr(self.agent, 'device', None)
                if device and device.type == 'cuda':
                    current_peak = torch.cuda.max_memory_allocated(device)
                    peak_memory_during_gen = max(peak_memory_during_gen, current_peak)
            # del r
            # gc.collect()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            #     torch.cuda.synchronize()
            
            # Clear cache periodically
            if (i+1) % self.config.get('system', {}).get('clear_cache_frequency', 100) == 0:
                self.agent.clear_cache()
        
        # Compute aggregate metrics (task-specific)
        aggregate_metrics = self.compute_aggregate_metrics(results)
        
        # Build summary
        total_samples = len(results)
        avg_time = total_time / total_samples if total_samples else 0
        memory_info = self._get_memory_snapshot()
        memory_info['peak_during_generation_gb'] = peak_memory_during_gen / (1024**3)
        
        summary = {
            "model_info": self.agent.get_model_info(),
            "configuration": {
                "compression_enabled": use_compression,
                "compression_method": kv_cfg.get('compression_method', 'N/A') if use_compression else 'N/A',
                "compression_ratio": kv_cfg.get('compression_ratio', 'N/A') if use_compression else 'N/A',
                "dataset_samples": total_samples
            },
            "results": aggregate_metrics,
            "performance": {
                "total_time": total_time,
                "avg_time": avg_time,
                "samples_per_minute": total_samples / (total_time / 60) if total_time > 0 else 0
            },
            "memory_tracking": memory_info,
            "compatibility": compatibility,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        if eval_cfg.get('save_results', True):
            self.save_results(summary, results)
        
        self._cleanup_after_evaluation()
        return {"summary": summary, "detailed_results": results}

    def compute_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Override this to compute task-specific aggregate metrics"""
        raise NotImplementedError("Subclasses must implement compute_aggregate_metrics()")

    def run_tuning(self):
        """Hyperparameter tuning - shared logic"""
        tuning_cfg = self.config.get('tuning', {})
        if not tuning_cfg.get('enabled', False):
            logger.info("Tuning disabled, running single evaluation")
            return self.run_evaluation()
        
        # if tuning_cfg.get('adaptive', False):
        #     logger.info("Adaptive tuning not fully implemented in base class")
        #     return self.run_evaluation()
        # Check if adaptive (LLM-based) tuning
        if tuning_cfg.get('adaptive', False):
            logger.info("Starting LLM-guided adaptive tuning...")
            from agents.llm_tuner import LLMAdaptiveTuner
            
            tuner = LLMAdaptiveTuner(
                evaluator=self,
                task=self.config.get('dataset', {}),#getattr(self, 'task', 'gsm8k'),
                max_iterations=tuning_cfg.get('max_iterations', 20),
                accuracy_weight=tuning_cfg.get('accuracy_weight', 0.7),
                co2_weight=tuning_cfg.get('co2_weight', 0.3),
                enable_co2_tracking=tuning_cfg.get('enable_co2_tracking', False)  # Default False!
            )
            # tuner.debug_grid_style_test()
            # return
            results = tuner.optimize()
            
            # Save results
            import json
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(self.results_dir, f"llm_tuning_{ts}.json")
            tuner.save_results(results, result_file)
            
            return results
        
        # Otherwise grid search
        ratios = tuning_cfg.get('compression_ratios_to_test', [1.0])
        methods = tuning_cfg.get('compression_methods_to_test', ['snapkv'])
        all_results = []
        
        for method in methods:
            for ratio in ratios:
                logger.info(f"\n{'='*60}")
                logger.info(f"Tuning run: Method={method}, Ratio={ratio}")
                logger.info(f"{'='*60}")
                
                self._reset_memory_tracking()
                
                # Update config
                self.config['kv_compression']['compression_method'] = method
                self.config['kv_compression']['compression_ratio'] = ratio
                # self.agent.compression_method = method
                self.agent.set_method(method)
                # self.agent.compression_ratio = ratio
                self.agent.set_compression_ratio(ratio)
                
                res = self.run_evaluation()
                res['summary']['configuration'].update({
                    'tuning_method': method,
                    'tuning_ratio': ratio
                })
                all_results.append(res['summary'])
        
        # Save tuning results
        all_results.sort(key=self._get_sort_key, reverse=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.results_dir, f"tuning_results_{ts}.json")
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=4)
        logger.info(f"All tuning results saved to: {filename}")
        return all_results

    def _get_sort_key(self, result: Dict) -> float:
        """Override this to change how tuning results are sorted"""
        return result['results'].get('accuracy', 0)

    def save_results(self, summary: Dict, results: List[Dict]):
        """Save evaluation results to files"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.agent.model_name.split('/')[-1]
        suffix = "compressed" if summary['configuration']['compression_enabled'] else "baseline"
        task_name = self._get_task_name()
        
        summary_file = os.path.join(self.results_dir, f"{task_name}_{model_name}_{suffix}_summary_{ts}.json")
        details_file = os.path.join(self.results_dir, f"{task_name}_{model_name}_{suffix}_details_{ts}.json")
        config_file = os.path.join(self.results_dir, f"config_{ts}.yaml")
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(details_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.agent.save_config(config_file)
        
        logger.info(f"Results saved: Summary={summary_file}, Details={details_file}")

    def _get_task_name(self) -> str:
        """Override to specify task name for file naming"""
        return "evaluation"

    def _reset_memory_tracking(self):
        """Reset CUDA memory tracking"""
        if torch.cuda.is_available():
            try:
                device = getattr(self.agent, 'device', None)
                if device and device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("Reset memory tracking")
            except Exception as e:
                logger.warning(f"Failed to reset memory tracking: {e}")

    def _get_memory_snapshot(self) -> Dict:
        """Capture current memory state"""
        memory_info = {}
        if torch.cuda.is_available():
            device = getattr(self.agent, 'device', None)
            if device and device.type == 'cuda':
                memory_info = {
                    'current_allocated_gb': torch.cuda.memory_allocated(device) / (1024**3),
                    'current_reserved_gb': torch.cuda.memory_reserved(device) / (1024**3),
                    'peak_allocated_gb': torch.cuda.max_memory_allocated(device) / (1024**3),
                    'peak_reserved_gb': torch.cuda.max_memory_reserved(device) / (1024**3),
                    'device': str(device)
                }
        return memory_info

    def _cleanup_after_evaluation(self):
        """Cleanup after evaluation"""
        import gc
        import torch
        
        
            
        # 2. 嘗試刪除 KVPress/AdaptiveKVController 實例 (魯棒處理)
        try:
            if hasattr(self.agent, "press") and self.agent.press is not None:

                # 1. 刪除 reference
                del self.agent.press
                self.agent.press = None

                # 2. 強制清理
                import torch, gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                logger.debug("KVPress/Controller instance released successfully.")
        except Exception as e:
            logger.warning(f"Failed to delete self.agent.press: {e}")
        # 1. 刪除模型 (這一步通常是安全的)
        try:
            if hasattr(self.agent, "model") and self.agent.model is not None:
                self.agent.model.to("cpu")
                del self.agent.model
                self.agent.model = None
        except Exception:
            pass
        # 3. 執行最激進的系統級清理
        gc.collect()
        
        if torch.cuda.is_available():
            try:
                # 呼叫 IPC Collect，這是 VRAM 洩漏的最後防線
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect() 

                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            except Exception as e:
                logger.warning(f"Post-eval cleanup warning: {e}")