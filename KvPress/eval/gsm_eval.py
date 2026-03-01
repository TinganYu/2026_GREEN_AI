import sys
import os
import logging
import re
from typing import Dict, List
from datasets import load_dataset
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from eval.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


class GSM8KEvaluator(BaseEvaluator):
    """Evaluator for GSM8K math reasoning dataset"""

    def load_dataset(self) -> List[Dict]:
        """Load GSM8K dataset"""
        cfg = self.config.get('dataset', {})
        split = cfg.get('split', 'test')
        num_samples = cfg.get('num_samples')
        
        try:
            dataset = load_dataset(
                cfg.get('name', 'gsm8k'),
                cfg.get('config', 'main'),
                split=split
            )
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        if num_samples and num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))
            logger.info(f"Limited to {num_samples} samples")
        
        logger.info(f"Loaded {len(dataset)} samples from GSM8K")
        return dataset

    def create_prompt(self, sample: Dict) -> str:
        """Create math problem prompt"""
        use_cot = self.config.get('evaluation', {}).get('use_chain_of_thought', True)
        prompts = self.config.get('dataset', {}).get('prompts', {})
        template = prompts.get('chain_of_thought') if use_cot else prompts.get('direct')
        
        if not template:
            template = (
                "Question: {question}\n\nLet's solve this step-by-step:\n\nAnswer:"
                if use_cot else
                "Question: {question}\nAnswer:"
            )
        
        return template.format(question=sample["question"])

    def extract_numerical_answer(self, text: str) -> float:
        """Extract numerical answer from text"""
        if not text:
            return None
        
        if '####' in text:
            try:
                # Split by '####', take the last part (the number) and strip
                final_answer_str = text.split('####')[-1].strip()
                # Clean the final answer string of common noise characters
                return float(final_answer_str.replace(",", "").replace("$", "").strip())
            except ValueError:
                # If conversion fails, proceed to the less strict fallbacks
                pass

        t = text.lower().replace(",", "").replace("$", "")
        
        # Try to find explicit answer markers
        match = re.findall(r"answer[:\s]*(-?\d+\.?\d*)", t)
        if match:
            try:
                return float(match[-1])
            except ValueError:
                pass
        
        # Fallback: extract all numbers and use last
        numbers = re.findall(r"-?\d+\.?\d*", t)
        if not numbers:
            return None
        
        # Rely on the last number, as the final answer is generally the last number generated
        final = numbers[-1]
        
        try:
            return float(final)
        except ValueError:
            return None

    def evaluate_prediction(self, sample: Dict, prediction: str) -> Dict:
        """Evaluate math answer accuracy"""
        prediction = prediction.split("Answer:")[-1].strip() 
        ground_truth = self.extract_numerical_answer(sample["answer"])
        predicted = self.extract_numerical_answer(prediction)
        
        is_correct = (
            predicted is not None and 
            ground_truth is not None and 
            abs(predicted - ground_truth) <= max(1e-6, abs(ground_truth) * 1e-9)
        )
        
        return {
            "question": sample["question"],
            "answer_text": prediction, 
            "ground_truth": ground_truth,
            "predicted": predicted,
            "is_correct": is_correct,
        }

    def compute_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Compute aggregate metrics for GSM8K"""
        total_samples = len(results)
        correct_count = sum(1 for r in results if r.get("is_correct", False))
        valid_predictions = sum(1 for r in results if r.get("predicted") is not None)
        errors = sum(1 for r in results if r.get("error") is not None)
        
        # Aggregate compression stats
        total_compressions = sum(r.get('kv_stats', {}).get('total_compressions', 0) for r in results)
        avg_compressed_tokens = sum(r.get('kv_stats', {}).get('compressed_tokens', 0) for r in results) / max(total_samples, 1)
        avg_compression_ratio = sum(r.get('kv_stats', {}).get('compression_ratio', 0) for r in results) / max(total_samples, 1)
        
        return {
            "total_samples": total_samples,
            "correct_answers": correct_count,
            "accuracy": correct_count / total_samples if total_samples else 0,
            "valid_predictions": valid_predictions,
            "errors": errors,
            "total_compressions": total_compressions,
            "avg_compressed_total_tokens": avg_compressed_tokens,
            "avg_compression_ratio": avg_compression_ratio
        }

    def evaluate_sample(self, sample: Dict, use_compression: bool = True) -> Dict:
        """Override to add math-specific generation parameters"""
        gen_params = {
            "mode": "math",
            "max_new_tokens": self.config.get('generation', {}).get('math', {}).get('max_new_tokens', 1024),
            "repetition_penalty": 1.2
        }
        
        return super().evaluate_sample(sample, use_compression, **gen_params)

    def _get_sort_key(self, result: Dict) -> float:
        """Sort tuning results by accuracy"""
        return result['results'].get('accuracy', 0)

    def _get_task_name(self) -> str:
        """Task name for file naming"""
        return "gsm8k"