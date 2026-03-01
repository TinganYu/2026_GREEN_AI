"""
基線評估器

評估未量化模型以建立用於比較的基線指標。
"""

import sys
import os
import json
import logging
import torch
from typing import Dict, Any
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.gsm8k_eval import GSM8KEvaluator
from evals.truthfulqa_eval import TruthfulQAEvaluator
from evals.commonsenseqa_eval import CommonsenseQAEvaluator
from evals.humaneval_eval import HumanEvalEvaluator
from evals.bbh_eval import BBHEvaluator

logger = logging.getLogger("BaselineEvaluator")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class BaselineEvaluator:
    """評估未量化的基線模型"""

    # 將資料集名稱映射到評估器類別
    EVALUATOR_MAP = {
        'gsm8k': GSM8KEvaluator,
        'truthfulqa': TruthfulQAEvaluator,
        'commonsenseqa': CommonsenseQAEvaluator,
        'humaneval': HumanEvalEvaluator,
        'bbh': BBHEvaluator,
    }

    def __init__(self, config: Dict[str, Any]):
        """
        初始化基線評估器

        Args:
            config: 來自 ConfigLoader 的配置字典
        """
        self.config = config
        self.baseline_config = config['baseline']
        self.cache_dir = config.get('cache', {}).get('cache_dir', '.cache/optimization')
        self.enable_cache = config.get('cache', {}).get('enable_baseline_cache', True)

        # 準備快取目錄
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        logger.info("基線評估器已初始化")

    def evaluate(self, datasets: Dict[str, int], force_recompute: bool = False) -> Dict[str, Any]:
        """
        在指定資料集上評估基線模型

        Args:
            datasets: 字典 {dataset_name: num_samples}
            force_recompute: 如果為 True，忽略快取並重新計算

        Returns:
            包含以下內容的字典：
            {
                'per_dataset': {
                    'gsm8k': {'accuracy': float, 'gpu_peak_mb': float, 'avg_latency_ms': float, ...},
                    ...
                },
                'aggregated': {
                    'accuracy': float,
                    'gpu_peak_mb': float,
                    'avg_latency_ms': float
                },
                'model_name': str,
                'dtype': str
            }
        """
        logger.info("="*60)
        logger.info("基線評估")
        logger.info("="*60)
        logger.info(f"模型：{self.baseline_config['model_name']}")
        logger.info(f"資料集：{list(datasets.keys())}")
        logger.info(f"總樣本數：{sum(datasets.values())}")

        # 檢查快取
        cache_key = self._get_cache_key(datasets)
        if not force_recompute and self.enable_cache:
            cached_results = self._load_from_cache(cache_key)
            if cached_results is not None:
                logger.info("已從快取載入基線結果")
                return cached_results

        # 在每個資料集上評估
        per_dataset_results = {}
        for dataset_name, num_samples in datasets.items():
            logger.info(f"\n正在評估 {dataset_name}（{num_samples} 個樣本）...")

            try:
                result = self._evaluate_single_dataset(dataset_name, num_samples)
                per_dataset_results[dataset_name] = result
                logger.info(f"✓ {dataset_name}：準確率={result['accuracy']:.4f}、"
                          f"GPU峰值={result['gpu_peak_mb']:.1f}MB、"
                          f"平均延遲={result['avg_latency_ms']:.1f}ms")
            except Exception as e:
                logger.error(f"✗ 評估 {dataset_name} 失敗：{e}")
                raise

        # 聚合結果
        aggregated = self._aggregate_results(per_dataset_results)

        # 封裝結果
        baseline_results = {
            'per_dataset': per_dataset_results,
            'aggregated': aggregated,
            'model_name': self.baseline_config['model_name'],
            'dtype': self.baseline_config['model_dtype'],
            'device_map': self.baseline_config['device_map'],
        }

        # 保存到快取
        if self.enable_cache:
            self._save_to_cache(cache_key, baseline_results)

        logger.info("\n" + "="*60)
        logger.info("基線評估完成")
        logger.info("="*60)
        logger.info(f"聚合準確率：{aggregated['accuracy']:.4f}")
        logger.info(f"GPU 峰值記憶體：{aggregated['gpu_peak_mb']:.1f} MB")
        logger.info(f"平均延遲：{aggregated['avg_latency_ms']:.1f} ms")
        logger.info("="*60)

        return baseline_results

    def _evaluate_single_dataset(self, dataset_name: str, num_samples: int) -> Dict[str, float]:
        """
        在單一資料集上評估

        Returns:
            包含 accuracy、gpu_peak_mb、avg_latency_ms 等的字典
        """
        # 獲取評估器類別
        evaluator_class = self.EVALUATOR_MAP.get(dataset_name)
        if evaluator_class is None:
            raise ValueError(f"未知的資料集：{dataset_name}。"
                           f"可用的資料集：{list(self.EVALUATOR_MAP.keys())}")

        # 準備評估器配置
        eval_config = {
            'model': {
                'name': self.baseline_config['model_name'],
                'dtype': self.baseline_config['model_dtype'],
                'device_map': self.baseline_config['device_map'],
                'trust_remote_code': self.baseline_config.get('trust_remote_code', True),
            },
            'dataset': {
                'name': dataset_name,
                'num_samples': num_samples,
                'config_file': 'tmp/config/dataset_config.yaml',  # 現有評估器需要此欄位
                'override': {
                    'num_samples': num_samples  # 從 dataset_config 覆蓋 num_samples
                }
            }
        }

        # 建立評估器實例
        evaluator = evaluator_class(eval_config)

        # 載入模型（評估前必需）
        evaluator.load_model()

        # 執行評估
        results = evaluator.evaluate()

        # 提取關鍵指標
        metrics = {
            'accuracy': results.get('accuracy', 0.0),
            'gpu_peak_mb': results.get('gpu_peak_mb', 0.0),
            'num_samples': results.get('total', num_samples),
            'correct': results.get('correct', 0),
        }

        # 計算平均延遲
        if 'total_generation_time_sec' in results and results.get('total', 0) > 0:
            avg_latency_sec = results['total_generation_time_sec'] / results['total']
            metrics['avg_latency_ms'] = avg_latency_sec * 1000  # 轉換為毫秒
        else:
            metrics['avg_latency_ms'] = 0.0

        # 吞吐量
        if 'throughput_tokens_per_sec' in results:
            metrics['throughput_tokens_per_sec'] = results['throughput_tokens_per_sec']

        return metrics

    def _aggregate_results(self, per_dataset_results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        從多個資料集聚合結果

        使用配置中指定的聚合方法（weighted/average/worst_case）
        """
        agg_method = self.config['multiobjective']['aggregation']['method']

        if agg_method == 'weighted':
            weights = self.config['multiobjective']['aggregation']['dataset_weights']
            return self._weighted_aggregation(per_dataset_results, weights)

        elif agg_method == 'average':
            return self._average_aggregation(per_dataset_results)

        elif agg_method == 'worst_case':
            return self._worst_case_aggregation(per_dataset_results)

        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")

    def _weighted_aggregation(self, results: Dict[str, Dict], weights: Dict[str, float]) -> Dict[str, float]:
        """加權平均聚合"""
        accuracy = sum(results[ds]['accuracy'] * weights[ds] for ds in results.keys())
        avg_latency_ms = sum(results[ds]['avg_latency_ms'] * weights[ds] for ds in results.keys())

        # GPU 峰值：取最大值（記憶體的最壞情況）
        gpu_peak_mb = max(results[ds]['gpu_peak_mb'] for ds in results.keys())

        return {
            'accuracy': accuracy,
            'gpu_peak_mb': gpu_peak_mb,
            'avg_latency_ms': avg_latency_ms,
            'aggregation_method': 'weighted'
        }

    def _average_aggregation(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """簡單平均聚合"""
        n = len(results)
        accuracy = sum(results[ds]['accuracy'] for ds in results.keys()) / n
        avg_latency_ms = sum(results[ds]['avg_latency_ms'] for ds in results.keys()) / n
        gpu_peak_mb = max(results[ds]['gpu_peak_mb'] for ds in results.keys())

        return {
            'accuracy': accuracy,
            'gpu_peak_mb': gpu_peak_mb,
            'avg_latency_ms': avg_latency_ms,
            'aggregation_method': 'average'
        }

    def _worst_case_aggregation(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """最壞情況聚合"""
        # 最壞情況：最小準確率、最大 GPU、最大延遲
        accuracy = min(results[ds]['accuracy'] for ds in results.keys())
        avg_latency_ms = max(results[ds]['avg_latency_ms'] for ds in results.keys())
        gpu_peak_mb = max(results[ds]['gpu_peak_mb'] for ds in results.keys())

        return {
            'accuracy': accuracy,
            'gpu_peak_mb': gpu_peak_mb,
            'avg_latency_ms': avg_latency_ms,
            'aggregation_method': 'worst_case'
        }

    def _get_cache_key(self, datasets: Dict[str, int]) -> str:
        """從模型和資料集生成快取鍵值"""
        model_name = self.baseline_config['model_name'].replace('/', '_')
        dataset_str = '_'.join(f"{k}-{v}" for k, v in sorted(datasets.items()))
        return f"{model_name}_{dataset_str}"

    def _load_from_cache(self, cache_key: str) -> Dict[str, Any]:
        """從快取載入基線結果"""
        cache_file = os.path.join(self.cache_dir, f"baseline_{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"載入快取失敗：{e}")
        return None

    def _save_to_cache(self, cache_key: str, results: Dict[str, Any]) -> None:
        """將基線結果保存到快取"""
        cache_file = os.path.join(self.cache_dir, f"baseline_{cache_key}.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"基線結果已快取至：{cache_file}")
        except Exception as e:
            logger.warning(f"保存快取失敗：{e}")
