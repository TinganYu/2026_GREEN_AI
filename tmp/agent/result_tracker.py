"""
結果追蹤器

追蹤並保存優化結果。
"""

import os
import json
import logging
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("ResultTracker")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class ResultTracker:
    """追蹤並保存優化結果"""

    def __init__(self, config: Dict[str, Any], output_dir: str):
        """
        初始化結果追蹤器

        Args:
            config: 配置字典
            output_dir: 保存結果的目錄
        """
        self.config = config
        self.output_dir = output_dir

        # 建立輸出目錄
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"結果追蹤器已初始化，輸出目錄：{output_dir}")

    def save_results(self, baseline_results: Dict[str, Any],
                    optimization_results: Dict[str, Any]) -> str:
        """
        保存完整的優化結果

        Args:
            baseline_results: 基線評估結果
            optimization_results: 來自優化器的優化結果

        Returns:
            保存的結果檔案路徑
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 統計試驗狀態
        all_trials = optimization_results['all_trials']
        failed_trials = [t for t in all_trials if t.get('status') == 'failed' or 'error' in t]
        pruned_trials = [t for t in all_trials if t.get('status') == 'pruned']
        completed_trials = [t for t in all_trials if t.get('status') == 'completed']

        n_failed = len(failed_trials)
        n_pruned = len(pruned_trials)
        n_completed = len(completed_trials)
        n_total = optimization_results['n_total_trials']

        logger.info(f"試驗統計：總共 {n_total}，完成 {n_completed}，剪枝 {n_pruned}，失敗 {n_failed}")

        # 準備結果字典
        results = {
            'experiment_name': self.config['experiment']['name'],
            'timestamp': timestamp,
            'config_file': 'optimization_config.yaml',

            # 基線
            'baseline': {
                'model': baseline_results['model_name'],
                'dtype': baseline_results['dtype'],
                'per_dataset': baseline_results['per_dataset'],
                'aggregated': baseline_results['aggregated']
            },

            # 優化摘要（增強版）
            'optimization_summary': {
                'optimizer_type': optimization_results['optimizer_type'],
                'n_total_trials': n_total,
                'n_completed_trials': n_completed,
                'n_pruned_trials': n_pruned,
                'n_failed_trials': n_failed,
                'n_pareto_solutions': optimization_results['n_pareto_solutions'],
                'n_satisfying_solutions': optimization_results['n_satisfying_solutions'],
                'success_rate': f"{n_completed / n_total * 100:.1f}%" if n_total > 0 else "0%"
            },

            # 目標和約束
            'objectives': self.config['multiobjective']['objectives'],
            'constraints': self.config['multiobjective']['constraints'],
            'targets': self.config['multiobjective']['targets'],

            # 所有試驗
            'all_trials': self._serialize_trials(all_trials),

            # Pareto 前沿
            'pareto_frontier': self._serialize_trials(optimization_results['pareto_frontier']),

            # 滿足目標的解決方案
            'satisfying_solutions': self._serialize_trials(optimization_results['satisfying_solutions']),

            # 推薦的配置
            'recommended_config': self._serialize_trial(optimization_results['recommended_config'])
                                 if optimization_results['recommended_config'] else None
        }

        # 保存完整結果
        results_file = os.path.join(self.output_dir, 'full_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"完整結果已保存至：{results_file}")

        # 保存所有試驗（包含詳細信息）
        all_trials_file = os.path.join(self.output_dir, 'all_trials.json')
        all_trials_data = {
            'total_trials': n_total,
            'completed': n_completed,
            'pruned': n_pruned,
            'failed': n_failed,
            'trials': self._serialize_trials(all_trials)
        }
        with open(all_trials_file, 'w') as f:
            json.dump(all_trials_data, f, indent=2)

        logger.info(f"所有試驗已保存至：{all_trials_file}")

        # 保存失敗試驗（便於調試）
        if failed_trials:
            failed_file = os.path.join(self.output_dir, 'failed_trials.json')
            failed_data = {
                'n_failed': n_failed,
                'failed_trials': self._serialize_trials(failed_trials)
            }
            with open(failed_file, 'w') as f:
                json.dump(failed_data, f, indent=2)

            logger.info(f"失敗試驗已保存至：{failed_file}")

        # 保存剪枝試驗
        if pruned_trials:
            pruned_file = os.path.join(self.output_dir, 'pruned_trials.json')
            pruned_data = {
                'n_pruned': n_pruned,
                'pruned_trials': self._serialize_trials(pruned_trials)
            }
            with open(pruned_file, 'w') as f:
                json.dump(pruned_data, f, indent=2)

            logger.info(f"剪枝試驗已保存至：{pruned_file}")

        # 保存摘要
        summary_file = os.path.join(self.output_dir, 'summary.json')
        summary = self._create_summary(results)
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"摘要已保存至：{summary_file}")

        # 保存 Pareto 前沿（簡化版）
        pareto_file = os.path.join(self.output_dir, 'pareto_frontier.json')
        pareto_data = {
            'pareto_solutions': len(optimization_results['pareto_frontier']),
            'solutions': [
                {
                    'config': t['config'],
                    'objectives': t['objectives'],
                    'satisfies_targets': t['satisfies_targets']
                }
                for t in optimization_results['pareto_frontier']
            ]
        }
        with open(pareto_file, 'w') as f:
            json.dump(pareto_data, f, indent=2)

        logger.info(f"Pareto 前沿已保存至：{pareto_file}")

        return results_file

    def _serialize_trials(self, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """序列化試驗以供 JSON 儲存"""
        return [self._serialize_trial(t) for t in trials]

    def _sanitize_for_json(self, obj: Any) -> Any:
        """
        遞迴清理物件以供 JSON 序列化
        將 infinity 和 NaN 值替換為 None（在 JSON 中變成 null）

        Args:
            obj: 要清理的物件（dict、list、float 或任何 JSON 相容類型）

        Returns:
            適合 JSON 序列化的清理後物件
        """
        import math

        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._sanitize_for_json(item) for item in obj)
        elif isinstance(obj, float):
            # 將 infinity 和 NaN 替換為 None（在 JSON 中變成 null）
            if math.isinf(obj) or math.isnan(obj):
                return None
            return obj
        else:
            # 所有其他類型按原樣返回（int、str、bool、None 等）
            return obj

    def _serialize_trial(self, trial: Dict[str, Any]) -> Dict[str, Any]:
        """序列化單一試驗，記錄完整的狀態信息"""
        if trial is None:
            return None

        # 提取關鍵資訊
        serialized = {
            'config': trial['config'],
            'objectives': trial['objectives'],
            'satisfies_constraints': trial.get('satisfies_constraints', True),
            'satisfies_targets': trial.get('satisfies_targets', False),
            'violation_score': trial.get('violation_score', 0.0)
        }

        # 添加狀態信息（來自 Optuna trial.user_attrs）
        if 'status' in trial:
            serialized['status'] = trial['status']  # completed / pruned / failed

        # 添加錯誤信息（如果失敗）
        if 'error' in trial:
            serialized['error'] = trial['error']

        # 添加模型路徑
        if 'quantized_model_path' in trial:
            serialized['quantized_model_path'] = trial['quantized_model_path']

        # 添加時間戳（如果可用）
        if 'datetime_start' in trial:
            serialized['datetime_start'] = trial['datetime_start'].isoformat() if hasattr(trial['datetime_start'], 'isoformat') else str(trial['datetime_start'])
        if 'datetime_complete' in trial:
            serialized['datetime_complete'] = trial['datetime_complete'].isoformat() if hasattr(trial['datetime_complete'], 'isoformat') else str(trial['datetime_complete'])

        # 如果可用，新增詳細的評估結果
        if 'results' in trial:
            # 每個數據集的結果（gsm8k, truthfulqa 等）
            if 'per_dataset' in trial['results']:
                serialized['per_dataset'] = trial['results']['per_dataset']

            # 聚合指標
            if 'aggregated' in trial['results']:
                serialized['aggregated'] = trial['results']['aggregated']

            # 量化模型路徑（優先從 results 中獲取）
            if 'quantized_model_path' in trial['results'] and 'quantized_model_path' not in serialized:
                serialized['quantized_model_path'] = trial['results']['quantized_model_path']

        # 清理 infinity 值後再返回
        return self._sanitize_for_json(serialized)

    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """建立簡潔摘要"""
        summary = {
            'experiment_name': results['experiment_name'],
            'timestamp': results['timestamp'],

            'baseline': {
                'model': results['baseline']['model'],
                'accuracy': results['baseline']['aggregated']['accuracy'],
                'gpu_peak_mb': results['baseline']['aggregated']['gpu_peak_mb'],
                'avg_latency_ms': results['baseline']['aggregated']['avg_latency_ms']
            },

            'optimization': {
                'optimizer': results['optimization_summary']['optimizer_type'],
                'total_trials': results['optimization_summary']['n_total_trials'],
                'pareto_solutions': results['optimization_summary']['n_pareto_solutions'],
                'satisfying_solutions': results['optimization_summary']['n_satisfying_solutions']
            },

            'targets': results['targets'],

            'recommended': None
        }

        # 如果可用，新增推薦的配置
        if results['recommended_config'] and results['recommended_config'].get('config'):
            rec = results['recommended_config']
            # 只在 config 非空時創建推薦
            if rec['config']:
                summary['recommended'] = {
                    'method': rec['config'].get('method', 'unknown'),
                    'config': rec['config'],
                    'objectives': rec['objectives'],
                    'satisfies_targets': rec['satisfies_targets'],
                    'per_dataset': rec.get('per_dataset', {}),
                    'aggregated': rec.get('aggregated', {})
                }
            else:
                logger.warning("推薦的配置為空 - 所有試驗可能都失敗了")

        return summary

    def save_config(self, config: Dict[str, Any]) -> None:
        """將配置保存到輸出目錄"""
        config_file = os.path.join(self.output_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"配置已保存至：{config_file}")

    def save_pareto_plot(self, plot_path: str) -> None:
        """將視覺化圖表複製到輸出目錄"""
        import shutil
        if os.path.exists(plot_path):
            dest = os.path.join(self.output_dir, os.path.basename(plot_path))
            shutil.copy(plot_path, dest)
            logger.info(f"圖表已保存至：{dest}")
