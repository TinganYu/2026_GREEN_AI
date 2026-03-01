"""
評估模組
========

整合多個基準測試評估器

可用的評估器:
- GSM8KEvaluator: 數學推理（小學數學文字題）
- TruthfulQAEvaluator: 真實性評估
- CommonsenseQAEvaluator: 常識推理
- HumanEvalEvaluator: 程式碼生成
- BBHEvaluator: BIG-Bench Hard 困難推理任務

使用方式:
    from evals import EvaluationManager, GSM8KEvaluator
    from evals.gsm_eval import GSM8KEvalConfig
    
    config = GSM8KEvalConfig(model_path="your-model")
    evaluator = GSM8KEvaluator(config)
    evaluator.load_model()
    results = evaluator.evaluate()
"""

from .base_evaluator import BaseEvaluator, BaseEvalConfig
from .gsm8k_eval import GSM8KEvaluator, GSM8KConfig
from .truthfulqa_eval import TruthfulQAEvaluator, TruthfulQAConfig
from .commonsenseqa_eval import CommonsenseQAEvaluator, CommonsenseQAConfig
from .humaneval_eval import HumanEvalEvaluator, HumanEvalConfig
from .bbh_eval import BBHEvaluator, BBHConfig

__all__ = [
    # 基類
    "BaseEvaluator",
    "BaseEvalConfig",
    
    # 評估器
    "GSM8KEvaluator",
    "TruthfulQAEvaluator",
    "CommonsenseQAEvaluator",
    "HumanEvalEvaluator",
    "BBHEvaluator",
    
    # 配置
    "GSM8KConfig",
    "TruthfulQAConfig",
    "CommonsenseQAConfig",
    "HumanEvalConfig",
    "BBHConfig",
]

__version__ = "1.0.0"

