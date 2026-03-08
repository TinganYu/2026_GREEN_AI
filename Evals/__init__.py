from .gsm8k_eval import GSM8KEvaluator
from .truthfulqa_eval import TruthfulQAEvaluator
from .commonsenseqa_eval import CommonsenseQAEvaluator
from .humaneval_eval import HumanEvalEvaluator
from .bbh_eval import BBHEvaluator

__all__ = [
    "GSM8KEvaluator",
    "TruthfulQAEvaluator",
    "CommonsenseQAEvaluator",
    "HumanEvalEvaluator",
    "BBHEvaluator",
]

EVALUATOR_MAP = {
    "gsm8k": GSM8KEvaluator,
    "truthfulqa": TruthfulQAEvaluator,
    "commonsenseqa": CommonsenseQAEvaluator,
    "humaneval": HumanEvalEvaluator,
    "bbh": BBHEvaluator,
}
