import sys
import os
import logging
from typing import Dict, List
from datasets import load_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from eval.base_evaluator import BaseEvaluator

# Optional: for better summarization metrics
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    logging.warning("rouge-score not available. Install with: pip install rouge-score")

logger = logging.getLogger(__name__)


class MultiNewsEvaluator(BaseEvaluator):
    """Evaluator for MultiNews summarization dataset"""

    def __init__(self, config_path: str, **override_params):
        super().__init__(config_path, **override_params)
        
        # Initialize ROUGE scorer if available
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
            logger.warning("ROUGE metrics will not be available")

    def load_dataset(self) -> List[Dict]:
        """Load MultiNews dataset"""
        cfg = self.config.get('dataset', {})
        split = cfg.get('split', 'test')
        num_samples = cfg.get('num_samples')
        
        try:
            dataset = load_dataset(
                'alexfabbri/multi_news',
                split=split,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load MultiNews dataset: {e}")
            raise
        
        # Limit samples if specified
        if num_samples and num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))
            logger.info(f"Limited to {num_samples} samples")
        
        logger.info(f"Loaded {len(dataset)} samples from MultiNews")
        return dataset

    def create_prompt(self, sample: Dict) -> str:
        """Create summarization prompt from sample"""
        prompts = self.config.get('dataset', {}).get('prompts', {})
        template = prompts.get('summarization')
        
        if not template:
            # Default summarization prompt
            template = (
                "Read the following news articles and write a comprehensive summary "
                "that captures the main points:\n\n"
                "{document}\n\n"
                "Summary:"
            )
        
        # MultiNews has 'document' field with source articles
        document = sample.get('document', '')
        
        # Optionally truncate very long documents
        max_input_tokens = self.config.get('dataset', {}).get('max_input_tokens', 4096)
        if self.agent.tokenizer:
            tokens = self.agent.tokenizer.encode(document)
            if len(tokens) > max_input_tokens:
                # Truncate and decode back
                tokens = tokens[:max_input_tokens]
                document = self.agent.tokenizer.decode(tokens, skip_special_tokens=True)
                logger.debug(f"Truncated document to {max_input_tokens} tokens")
        
        return template.format(document=document)

    def evaluate_prediction(self, sample: Dict, prediction: str) -> Dict:
        """Evaluate summarization quality using ROUGE metrics"""
        reference = sample.get('summary', '')
        prediction=prediction.split("Summary:")[-1].strip()
        
        result = {
            "reference_summary": reference,
            "generated_summary": prediction,
        }
        
        # Compute ROUGE scores if available
        if self.rouge_scorer and reference and prediction:
            try:
                scores = self.rouge_scorer.score(reference, prediction)
                result.update({
                    "rouge1_f": scores['rouge1'].fmeasure,
                    "rouge1_p": scores['rouge1'].precision,
                    "rouge1_r": scores['rouge1'].recall,
                    "rouge2_f": scores['rouge2'].fmeasure,
                    "rouge2_p": scores['rouge2'].precision,
                    "rouge2_r": scores['rouge2'].recall,
                    "rougeL_f": scores['rougeL'].fmeasure,
                    "rougeL_p": scores['rougeL'].precision,
                    "rougeL_r": scores['rougeL'].recall,
                })
            except Exception as e:
                logger.error(f"ROUGE computation failed: {e}")
                result["rouge_error"] = str(e)
        else:
            # Fallback: simple length-based metrics
            result.update({
                "reference_length": len(reference.split()),
                "prediction_length": len(prediction.split()),
                "length_ratio": len(prediction.split()) / max(len(reference.split()), 1)
            })
        
        return result

    def compute_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Compute aggregate metrics across all samples"""
        total_samples = len(results)
        successful = [r for r in results if r.get('error') is None]
        
        metrics = {
            "total_samples": total_samples,
            "successful": len(successful),
            "errors": total_samples - len(successful),
        }
        
        # Aggregate ROUGE scores if available
        if ROUGE_AVAILABLE and successful:
            rouge_keys = ['rouge1_f', 'rouge2_f', 'rougeL_f']
            for key in rouge_keys:
                values = [r[key] for r in successful if key in r]
                if values:
                    metrics[f"avg_{key}"] = sum(values) / len(values)
                    metrics[f"max_{key}"] = max(values)
                    metrics[f"min_{key}"] = min(values)
        
        # Aggregate compression stats
        total_tokens_saved = sum(r.get('kv_stats', {}).get('tokens_saved', 0) for r in results)
        avg_compressed_tokens = sum(r.get('kv_stats', {}).get('compressed_tokens', 0) for r in results) / max(total_samples, 1)
        avg_compression_ratio = sum(r.get('kv_stats', {}).get('compression_ratio', 0) for r in results) / max(total_samples, 1)
        
        metrics.update({
            "total_tokens_saved": total_tokens_saved,
            "avg_compressed_total_tokens": avg_compressed_tokens,
            "avg_compression_ratio": avg_compression_ratio
        })
        
        return metrics

    def evaluate_sample(self, sample: Dict, use_compression: bool = True) -> Dict:
        """Override to add summarization-specific generation parameters"""
        gen_params = {
            "max_new_tokens": self.config.get('generation', {}).get('summarization', {}).get('max_new_tokens', 256),
            "temperature": self.config.get('generation', {}).get('summarization', {}).get('temperature', 0.7),
            "do_sample": self.config.get('generation', {}).get('summarization', {}).get('do_sample', True),
            "top_p": 0.9,
            "repetition_penalty": 1.2
        }
        
        return super().evaluate_sample(sample, use_compression, **gen_params)

    def _get_sort_key(self, result: Dict) -> float:
        """Sort tuning results by ROUGE-L F1 score"""
        return result['results'].get('avg_rougeL_f', 0)

    def _get_task_name(self) -> str:
        """Task name for file naming"""
        return "multinews"