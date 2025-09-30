
import os
import gc
import sys
import json
import logging
import torch
from datetime import datetime
from typing import Dict, List
import re
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.KVpress import KVPressAgent
from agents.tuner import AdaptiveKVController

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
config_path = BASE_DIR / "config" / "model_config.yaml"

class GSM8KEvaluator:
    """Evaluator for GSM8K dataset with configurable KV compression"""

    def __init__(self, config_path: str = config_path, **override_params):
        self.config_path = config_path

        # First load base config
        base_agent = KVPressAgent(config_path, **override_params)
        self.config = base_agent.config  

        # Decide which agent to wrap
        if self.config.get("tuning", {}).get("adaptive", False):
            self.agent = AdaptiveKVController(self.config, base_agent=base_agent)
            print("Using AdaptiveKVController")
        else:
            self.agent = base_agent
            print("Using base KVPressAgent")

        self.config = self.agent.config  

        # Prepare results dir
        self.results_dir = self.config.get('evaluation', {}).get('results_dir', './results')
        os.makedirs(self.results_dir, exist_ok=True)

    def load_gsm8k_dataset(self) -> List[Dict]:
        cfg = self.config.get('dataset', {})
        split = cfg.get('split', 'test')
        num_samples = cfg.get('num_samples')
        try:
            dataset = load_dataset(cfg.get('name', 'gsm8k'),
                                   cfg.get('config', 'main'),
                                   split=split)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        if num_samples and num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))
            logger.info(f"Limited to {num_samples} samples")

        logger.info(f"Loaded {len(dataset)} samples")
        return dataset

    def extract_numerical_answer(self, text: str) -> float:
        if not text:
            return None

        t = text.lower().replace(",", "").replace("$", "")
        match = re.findall(r"answer[:\s]*(-?\d+\.?\d*)", t)
        if match:
            try:
                return float(match[-1])
            except ValueError:
                pass

        numbers = re.findall(r"-?\d+\.?\d*", t)
        if not numbers:
            return None

        cnt = Counter(numbers)
        mode, freq = cnt.most_common(1)[0]
        final = mode if freq >= 2 else numbers[-1]
        try:
            return float(final)
        except ValueError:
            return None

    def create_prompt(self, question: str, use_cot: bool = True) -> str:
        prompts = self.config.get('dataset', {}).get('prompts', {})
        template = prompts.get('chain_of_thought') if use_cot else prompts.get('direct')
        if not template:
            template = f"Question: {{question}}\nLet's solve this step by step.\nAnswer:" if use_cot else \
                       f"Question: {{question}}\nAnswer:"
        return template.format(question=question)

    def evaluate_sample(self, sample: Dict, use_compression: bool = True, use_cot: bool = True) -> Dict:
        question = sample["question"]
        ground_truth = self.extract_numerical_answer(sample["answer"])
        prompt = self.create_prompt(question, use_cot)
        start_time = datetime.now()
        generation_params = {
            "max_new_tokens": 128,
            "repetition_penalty": 1.2
        }
        try:
            if use_compression and hasattr(self.agent, "generate_with_compression"):
                response = self.agent.generate_with_compression(
                    prompt, mode="math", **generation_params
                )
                # Get compression stats if available
                kv_stats = self.agent.get_compression_stats() if hasattr(self.agent, "get_compression_stats") else {}
            else:
                response = self.agent.generate_response(
                    prompt, mode="math", **generation_params
                )
                kv_stats = {}

            generation_time = (datetime.now() - start_time).total_seconds()
            predicted = self.extract_numerical_answer(response)
            is_correct = predicted is not None and ground_truth is not None and \
                         abs(predicted - ground_truth) <= max(1e-6, abs(ground_truth) * 1e-9)
            error = None
        except Exception as e:
            generation_time = (datetime.now() - start_time).total_seconds()
            response, predicted, is_correct, error = "", None, False, str(e)
            kv_stats = {}

        if isinstance(response, torch.Tensor):
            try:
                response = response.detach().cpu().tolist()
            except Exception:
                response = str(response)

        if isinstance(predicted, torch.Tensor):
            try:
                predicted = float(predicted.detach().cpu().item())
            except Exception:
                predicted = None

        result = {
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "is_correct": is_correct,
            "response": response,
            "prompt": prompt,
            "generation_time": generation_time,
            "error": error,
            "kv_stats": kv_stats  # Add compression stats per sample
        }

        try:
            del response
        except Exception:
            pass

        return result

    def run_evaluation(self) -> Dict:
        eval_cfg = self.config.get('evaluation', {})
        kv_cfg = self.config.get('kv_compression', {})
        use_compression = kv_cfg.get('enabled', True)
        use_cot = eval_cfg.get('use_chain_of_thought', True)

        logger.info("STARTING GSM8K EVALUATION")
        logger.info(f"Model: {self.agent.model_name}, Compression: {use_compression}, CoT: {use_cot}")

        # CRITICAL FIX: Reset peak memory BEFORE starting evaluation
        if torch.cuda.is_available():
            try:
                device = getattr(self.agent, 'device', None)
                if device is not None and device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                else:
                    torch.cuda.reset_peak_memory_stats()
                logger.info("Reset peak memory stats before evaluation")
            except Exception as e:
                logger.warning(f"Failed to reset peak memory: {e}")

        self.agent.load_model()
        compatibility = self.agent.validate_model_compatibility()

        if not compatibility.get('compatible', False) and use_compression:
            logger.warning("KV compression disabled due to compatibility issues")
            use_compression = False
        else:
            logger.info(f"KV compression enabled: {use_compression}")

        dataset = self.load_gsm8k_dataset()
        results, correct_count, total_time = [], 0, 0

        # Track memory during generation
        peak_memory_during_gen = 0
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating samples")):
            r = self.evaluate_sample(sample, use_compression, use_cot)
            results.append(r)
            correct_count += int(r["is_correct"])
            total_time += r["generation_time"]
            
            # Track peak memory after each sample
            if torch.cuda.is_available():
                device = getattr(self.agent, 'device', None)
                if device and device.type == 'cuda':
                    current_peak = torch.cuda.max_memory_allocated(device)
                    peak_memory_during_gen = max(peak_memory_during_gen, current_peak)
            
            if (i+1) % self.config.get('system', {}).get('clear_cache_frequency', 100) == 0:
                self.agent.clear_cache()

        total_samples = len(results)
        accuracy = correct_count / total_samples if total_samples else 0
        avg_time = total_time / total_samples if total_samples else 0

        # Aggregate compression stats across all samples
        total_compressions = sum(r.get('kv_stats', {}).get('total_compressions', 0) for r in results)
        avg_compressed_tokens = sum(r.get('kv_stats', {}).get('compressed_tokens', 0) for r in results) / max(total_samples, 1)
        avg_compression_ratio = sum(r.get('kv_stats', {}).get('compression_ratio', 0) for r in results) / max(total_samples, 1)

        # Get memory info at end of evaluation
        memory_info = self._get_memory_snapshot()
        memory_info['peak_during_generation_gb'] = peak_memory_during_gen / (1024**3)

        summary = {
            "model_info": self.agent.get_model_info(),
            "configuration": {
                "compression_enabled": use_compression,
                "compression_method": kv_cfg.get('compression_method', 'N/A') if use_compression else 'N/A',
                "compression_ratio": kv_cfg.get('compression_ratio', 'N/A') if use_compression else 'N/A',
                "chain_of_thought": use_cot,
                "dataset_samples": total_samples
            },
            "results": {
                "total_samples": total_samples,
                "correct_answers": correct_count,
                "accuracy": accuracy,
                "valid_predictions": sum(1 for r in results if r["predicted"] is not None),
                "errors": sum(1 for r in results if r["error"] is not None),
                "avg_compression_ratio": avg_compression_ratio,
                "total_compressions": total_compressions,
                "avg_compressed_tokens": avg_compressed_tokens
            },
            "performance": {
                "total_time": total_time,
                "avg_time": avg_time,
                "samples_per_minute": total_samples / (total_time / 60) if total_time > 0 else 0
            },
            "memory_tracking": memory_info,  # ADD DEDICATED MEMORY SECTION
            "compatibility": compatibility,
            "timestamp": datetime.now().isoformat()
        }

        if eval_cfg.get('save_results', True):
            self.save_results(summary, results)
            
        self._cleanup_after_evaluation()
        return {"summary": summary, "detailed_results": results}

    def _get_memory_snapshot(self) -> Dict:
        """Capture current memory state with proper tracking"""
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
        """Clean up after evaluation run"""
        try:
            if hasattr(self.agent, "model") and self.agent.model is not None:
                try:
                    self.agent.model.to("cpu")
                except Exception:
                    pass
                del self.agent.model
                self.agent.model = None
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception as e:
                logger.warning(f"Post-eval cuda cleanup warning: {e}")

    def save_results(self, summary: Dict, results: List[Dict]):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.agent.model_name.split('/')[-1]
        suffix = "compressed" if summary['configuration']['compression_enabled'] else "baseline"

        summary_file = os.path.join(self.results_dir, f"gsm8k_{model_name}_{suffix}_summary_{ts}.json")
        details_file = os.path.join(self.results_dir, f"gsm8k_{model_name}_{suffix}_details_{ts}.json")
        config_file = os.path.join(self.results_dir, f"config_{ts}.yaml")

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(details_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.agent.save_config(config_file)

        logger.info(f"Results saved: Summary={summary_file}, Details={details_file}, Config={config_file}")

    def run_tuning(self):
        tuning_cfg = self.config.get('tuning', {})
        if not tuning_cfg.get('enabled', False):
            logger.info("Tuning disabled, running single evaluation")
            return self.run_evaluation()

        if tuning_cfg.get('adaptive', False):
            logger.info("Starting adaptive tuning...")
            detailed_results = self.run_adaptive_tuning()
            correct_count = sum(r['is_correct'] for r in detailed_results)
            total_samples = len(detailed_results)
            accuracy = correct_count / total_samples if total_samples else 0
            summary = {
                "total_samples": total_samples,
                "correct_answers": correct_count,
                "accuracy": accuracy
            }
            return {"summary": summary, "detailed_results": detailed_results}

        else:
            ratios = tuning_cfg.get('compression_ratios_to_test', [1.0])
            methods = tuning_cfg.get('compression_methods_to_test', ['snapkv'])
            all_results = []

            for method in methods:
                for ratio in ratios:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Tuning run: Method={method}, Ratio={ratio}")
                    logger.info(f"{'='*60}")
                    
                    # CRITICAL: Reset peak memory BEFORE each tuning run
                    self._reset_memory_tracking()
                    
                    # Update config
                    self.config['kv_compression']['compression_method'] = method
                    self.config['kv_compression']['compression_ratio'] = ratio
                    self.agent.compression_method = method
                    self.agent.compression_ratio = ratio
                    
                    res = self.run_evaluation()
                    res['summary']['configuration'].update({
                        'tuning_method': method, 
                        'tuning_ratio': ratio
                    })
                    all_results.append(res['summary'])
                    
                    # Log memory for this run
                    if 'memory_tracking' in res['summary']:
                        mem = res['summary']['memory_tracking']
                        logger.info(f"Memory for {method} @ {ratio}: "
                                  f"Peak={mem.get('peak_allocated_gb', 'N/A'):.2f}GB")

            # Save all tuning results
            all_results.sort(key=lambda x: x['results']['accuracy'], reverse=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.results_dir, f"tuning_results_{ts}.json")
            with open(filename, 'w') as f:
                json.dump(all_results, f, indent=4)
            logger.info(f"All tuning results saved to: {filename}")
            return all_results

    def _reset_memory_tracking(self):
        """Reset memory tracking counters before a tuning run"""
        if torch.cuda.is_available():
            try:
                device = getattr(self.agent, 'device', None)
                if device and device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    logger.info("Reset memory tracking for new tuning run")
            except Exception as e:
                logger.warning(f"Failed to reset memory tracking: {e}")

    def get_feedback(self, sample, output):
        return {
            "accuracy": 1.0 if self.extract_numerical_answer(output) == self.extract_numerical_answer(sample["answer"]) else 0.0,
            "memory": self.get_memory_usage(),
            "max_mem": self.config.get("system", {}).get("max_mem", torch.cuda.get_device_properties(0).total_memory),
            "nans": 0
        }
    
    def run_adaptive_tuning(self, samples=None, use_cot=True):
        self.agent.load_model()
        samples = samples or self.load_gsm8k_dataset()
        use_compression = self.config.get('kv_compression', {}).get('enabled', True)

        results = []
        for i, sample in enumerate(samples):
            r = self.evaluate_sample(sample, use_compression=use_compression, use_cot=use_cot)
            results.append(r)

            if isinstance(self.agent, AdaptiveKVController):
                feedback = {
                    "accuracy": 1.0 if r["is_correct"] else 0.0,
                    "memory": self.agent.get_memory_usage(),
                    "max_mem": self.config.get("system", {}).get("max_mem", torch.cuda.get_device_properties(0).total_memory),
                    "nans": 0
                }
                self.agent.adjust(feedback)

            if (i+1) % self.config.get('system', {}).get('clear_cache_frequency', 100) == 0:
                self.agent.clear_cache()

            if isinstance(self.agent, AdaptiveKVController):
                logger.info(f"Step {i+1}: ratio={self.agent.current_ratio:.2f}, method={self.agent.current_method}, accuracy={r['is_correct']}")

        return results