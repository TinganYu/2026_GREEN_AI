"""
評估代理

執行量化和評估實驗。
"""

import sys
import os
import shutil
import logging
import time
from typing import Dict, Any, List
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import quantization module
from method.quantization import Quantizer, GPTQConfig, AWQConfig, BNBConfig

# Import evaluators
from evals.gsm8k_eval import GSM8KEvaluator
from evals.truthfulqa_eval import TruthfulQAEvaluator
from evals.commonsenseqa_eval import CommonsenseQAEvaluator
from evals.humaneval_eval import HumanEvalEvaluator
from evals.bbh_eval import BBHEvaluator

logger = logging.getLogger("EvaluatorAgent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class EvaluatorAgent:
    """執行量化和評估實驗的代理"""

    EVALUATOR_MAP = {
        'gsm8k': GSM8KEvaluator,
        'truthfulqa': TruthfulQAEvaluator,
        'commonsenseqa': CommonsenseQAEvaluator,
        'humaneval': HumanEvalEvaluator,
        'bbh': BBHEvaluator,
    }

    def __init__(self, config: Dict[str, Any]):
        """
        初始化評估代理

        Args:
            config: 來自 ConfigLoader 的配置字典
        """
        self.config = config
        self.baseline_config = config['baseline']
        self.base_model_name = self.baseline_config['model_name']

        logger.info("評估代理已初始化")

    def run_experiment(self, quant_config: Dict[str, Any], datasets: Dict[str, int],
                       trial_number: int = None, exp_dir: str = None) -> Dict[str, Any]:
        """
        執行完整實驗：量化 + 評估

        Args:
            quant_config: 量化配置，例如：
                {
                    'method': 'gptq',
                    'bits': 4,
                    'group_size': 128,
                    'calib_num': 512,
                    ...
                }
            datasets: 字典 {dataset_name: num_samples}
            trial_number: 試驗編號（用於優化實驗）
            exp_dir: 實驗根目錄（用於優化實驗）

        Returns:
            包含以下內容的字典：
            {
                'per_dataset': {dataset: {accuracy, gpu_peak_mb, avg_latency_ms}},
                'aggregated': {accuracy, gpu_peak_mb, avg_latency_ms},
                'quantized_model_path': str,
                'quantization_time_sec': float,
                'total_evaluation_time_sec': float,
                'config': quant_config
            }
        """
        start_time = time.time()

        logger.info("="*60)
        logger.info(f"實驗：{quant_config['method'].upper()}")
        if trial_number is not None:
            logger.info(f"試驗編號：{trial_number}")
        logger.info("="*60)
        logger.info(f"配置：{quant_config}")

        try:
            # 步驟 1：量化模型
            logger.info("[1/2] 正在量化模型...")
            quant_start = time.time()
            quantized_model_path = self._quantize_model(quant_config, trial_number, exp_dir)
            quant_time = time.time() - quant_start
            logger.info(f"✓ 量化完成，耗時 {quant_time:.1f} 秒")
            logger.info(f"✓ 量化模型已保存至：{quantized_model_path}")

            # 步驟 2：在資料集上評估
            logger.info("[2/2] 正在評估量化模型...")
            eval_start = time.time()
            per_dataset_results = {}

            for dataset_name, num_samples in datasets.items():
                logger.info(f"  正在評估 {dataset_name}（{num_samples} 個樣本）...")
                try:
                    result = self._evaluate_single_dataset(
                        quantized_model_path,
                        dataset_name,
                        num_samples,
                        quant_config['method']
                    )
                    per_dataset_results[dataset_name] = result
                    logger.info(f"  ✓ {dataset_name}：準確率={result['accuracy']:.4f}、"
                              f"GPU峰值={result['gpu_peak_mb']:.1f}MB、"
                              f"延遲={result['avg_latency_ms']:.1f}ms")
                except Exception as e:
                    logger.error(f"  ✗ 評估 {dataset_name} 失敗：{e}")
                    raise

            eval_time = time.time() - eval_start
            logger.info(f"✓ 評估完成，耗時 {eval_time:.1f} 秒")

            # 步驟 3：聚合結果
            aggregated = self._aggregate_results(per_dataset_results)

            # 封裝結果
            total_time = time.time() - start_time
            results = {
                'per_dataset': per_dataset_results,
                'aggregated': aggregated,
                'quantized_model_path': quantized_model_path,
                'quantization_time_sec': quant_time,
                'total_evaluation_time_sec': eval_time,
                'total_time_sec': total_time,
                'config': quant_config,
                'success': True
            }

            logger.info("\n" + "="*60)
            logger.info("實驗完成")
            logger.info("="*60)
            logger.info(f"聚合準確率：{aggregated['accuracy']:.4f}")
            logger.info(f"GPU 峰值記憶體：{aggregated['gpu_peak_mb']:.1f} MB")
            logger.info(f"平均延遲：{aggregated['avg_latency_ms']:.1f} ms")
            logger.info(f"總耗時：{total_time:.1f} 秒")
            logger.info("="*60)

            return results

        except Exception as e:
            logger.error(f"實驗失敗：{e}")
            return {
                'success': False,
                'error': str(e),
                'config': quant_config
            }

    def _quantize_model(self, quant_config: Dict[str, Any],
                        trial_number: int = None, exp_dir: str = None) -> str:
        """
        使用指定配置量化模型

        Args:
            quant_config: 量化配置字典
            trial_number: 試驗編號（用於優化實驗）
            exp_dir: 實驗根目錄（用於優化實驗）

        Returns:
            量化模型的路徑
        """
        method = quant_config['method']

        # 建立量化器
        quantizer = Quantizer(self.base_model_name)

        # 根據方法建立配置物件
        if method == 'gptq':
            config = GPTQConfig(
                bits=quant_config.get('bits', 4),
                group_size=quant_config.get('group_size', 128),
                damp_percent=quant_config.get('damp_percent', 0.01),
                damp_auto_increment=quant_config.get('damp_auto_increment', 0.005),
                desc_act=quant_config.get('desc_act', True),
                act_group_aware=quant_config.get('act_group_aware', False),
                static_groups=quant_config.get('static_groups', False),
                sym=quant_config.get('sym', True),
                true_sequential=quant_config.get('true_sequential', True),
                lm_head=quant_config.get('lm_head', False),
                mse=quant_config.get('mse', 0.0),
                rotation=quant_config.get('rotation', None),
                calib_num=quant_config.get('calib_num', 256),
                output_dir=quant_config.get('output_dir', None)
            )
            output_path = quantizer.quantize(config, trial_number, exp_dir)

        elif method == 'awq':
            config = AWQConfig(
                zero_point=quant_config.get('zero_point', True),
                q_group_size=quant_config.get('q_group_size', 128),
                w_bit=quant_config.get('w_bit', 4),
                version=quant_config.get('version', 'gemm'),
                modules_to_not_convert=quant_config.get('modules_to_not_convert', []),
                output_dir=quant_config.get('output_dir', None)
            )
            output_path = quantizer.quantize(config, trial_number, exp_dir)

        elif method == 'bnb':
            config = BNBConfig(
                bits=quant_config.get('bits', 4),
                bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
                bnb_4bit_compute_dtype=quant_config.get('bnb_4bit_compute_dtype', 'float16'),
                llm_int8_threshold=quant_config.get('llm_int8_threshold', 6.0),
                llm_int8_skip_modules=quant_config.get('llm_int8_skip_modules', ['lm_head']),
                llm_int8_has_fp16_weight=quant_config.get('llm_int8_has_fp16_weight', False),
                llm_int8_enable_fp32_cpu_offload=quant_config.get('llm_int8_enable_fp32_cpu_offload', False),
                output_dir=quant_config.get('output_dir', None)
            )
            output_path = quantizer.quantize(config, trial_number, exp_dir)

        else:
            raise ValueError(f"未知的量化方法：{method}")

        return output_path

    def _evaluate_single_dataset(self, model_path: str, dataset_name: str,
                                 num_samples: int, quant_method: str) -> Dict[str, float]:
        """
        在單一資料集上評估量化模型

        Returns:
            包含 accuracy、gpu_peak_mb、avg_latency_ms 的字典
        """
        # 獲取評估器類別
        evaluator_class = self.EVALUATOR_MAP.get(dataset_name)
        if evaluator_class is None:
            raise ValueError(f"未知的資料集：{dataset_name}")

        # 準備評估器配置
        eval_config = {
            'model': {
                'name': model_path,
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

        # 載入模型並自動偵測量化類型
        evaluator.load_model(quantization_type=quant_method)

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

        使用配置中指定的聚合方法
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
            raise ValueError(f"未知的聚合方法：{agg_method}")

    def _weighted_aggregation(self, results: Dict[str, Dict], weights: Dict[str, float]) -> Dict[str, float]:
        """加權平均聚合"""
        accuracy = sum(results[ds]['accuracy'] * weights[ds] for ds in results.keys())
        avg_latency_ms = sum(results[ds]['avg_latency_ms'] * weights[ds] for ds in results.keys())
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
        accuracy = min(results[ds]['accuracy'] for ds in results.keys())
        avg_latency_ms = max(results[ds]['avg_latency_ms'] for ds in results.keys())
        gpu_peak_mb = max(results[ds]['gpu_peak_mb'] for ds in results.keys())

        return {
            'accuracy': accuracy,
            'gpu_peak_mb': gpu_peak_mb,
            'avg_latency_ms': avg_latency_ms,
            'aggregation_method': 'worst_case'
        }

    def cleanup_quantized_model(self, model_path: str) -> None:
        """刪除量化模型目錄以節省磁碟空間"""
        try:
            if os.path.exists(model_path) and 'quantized' in model_path:
                shutil.rmtree(model_path)
                logger.info(f"已清理量化模型：{model_path}")
        except Exception as e:
            logger.warning(f"清理 {model_path} 失敗：{e}")

    def cleanup_non_pareto_models(self, all_model_paths: List[str],
                                   pareto_model_paths: List[str],
                                   exp_dir: str = None) -> Dict[str, int]:
        """
        清理非 Pareto 前沿的模型，只保留最優解

        Args:
            all_model_paths: 所有試驗生成的模型路徑列表
            pareto_model_paths: Pareto 前沿的模型路徑列表
            exp_dir: 實驗目錄（用於過濾）

        Returns:
            清理統計：{'kept': 保留數量, 'deleted': 刪除數量, 'failed': 清理失敗數量}
        """
        kept = 0
        deleted = 0
        failed = 0

        logger.info("="*60)
        logger.info("開始清理非 Pareto 前沿的模型")
        logger.info("="*60)

        for model_path in all_model_paths:
            # 跳過空路徑
            if not model_path:
                continue

            # 如果指定了實驗目錄，只清理該目錄下的模型
            if exp_dir and not model_path.startswith(exp_dir):
                continue

            # 判斷是否為 Pareto 前沿
            if model_path in pareto_model_paths:
                logger.info(f"✓ 保留 Pareto 模型：{os.path.basename(model_path)}")
                kept += 1
            else:
                # 刪除非 Pareto 模型
                try:
                    if os.path.exists(model_path):
                        shutil.rmtree(model_path)
                        logger.info(f"✗ 已刪除非 Pareto 模型：{os.path.basename(model_path)}")
                        deleted += 1
                    else:
                        logger.warning(f"⚠ 模型路徑不存在：{model_path}")
                except Exception as e:
                    logger.error(f"✗ 清理失敗 {os.path.basename(model_path)}：{e}")
                    failed += 1

        logger.info("="*60)
        logger.info(f"清理完成：保留 {kept} 個，刪除 {deleted} 個，失敗 {failed} 個")
        logger.info("="*60)

        return {'kept': kept, 'deleted': deleted, 'failed': failed}
