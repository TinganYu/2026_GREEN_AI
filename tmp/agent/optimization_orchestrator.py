"""
優化協調器

協調整個多目標優化工作流程的主控制器。
"""

import os
import logging
import time
from typing import Dict, Any
from datetime import datetime

from .config_loader import ConfigLoader
from .baseline_evaluator import BaselineEvaluator
from .evaluator_agent import EvaluatorAgent
from .result_tracker import ResultTracker
from .visualization import ParetoVisualizer
from .optimizers.optuna_mo_optimizer import OptunaMultiObjectiveOptimizer
from .optimizers.random_optimizer import RandomOptimizer
from .optimizers.llm_multiagent_optimizer import LLMMultiAgentOptimizer

logger = logging.getLogger("OptimizationOrchestrator")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class OptimizationOrchestrator:
    """多目標優化工作流程的主控制器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化協調器

        Args:
            config: 來自 ConfigLoader 的配置字典
        """
        self.config = config

        # 建立輸出目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = config['experiment']['name']
        self.output_dir = os.path.join(
            config['experiment']['output_dir'],
            f"{exp_name}_{timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("="*60)
        logger.info("優化協調器已初始化")
        logger.info("="*60)
        logger.info(f"實驗：{exp_name}")
        logger.info(f"輸出目錄：{self.output_dir}")
        logger.info("="*60)

        # 初始化組件
        self.baseline_evaluator = BaselineEvaluator(config)
        self.evaluator_agent = EvaluatorAgent(config)
        self.result_tracker = ResultTracker(config, self.output_dir)
        self.visualizer = ParetoVisualizer(config)

        # 保存配置
        self.result_tracker.save_config(config)

    def run(self) -> Dict[str, Any]:
        """
        執行完整的優化工作流程

        Returns:
            包含結果和路徑的字典
        """
        start_time = time.time()

        logger.info("\n" + "="*60)
        logger.info("開始優化工作流程")
        logger.info("="*60)

        # 階段 1：評估基線
        logger.info("\n" + "-"*60)
        logger.info("階段 1：基線評估")
        logger.info("-"*60)

        baseline_datasets = ConfigLoader.get_enabled_datasets(self.config, mode='baseline')
        baseline_results = self.baseline_evaluator.evaluate(baseline_datasets)

        # 階段 2：執行多目標優化
        logger.info("\n" + "-"*60)
        logger.info("階段 2：多目標優化")
        logger.info("-"*60)

        quant_datasets = ConfigLoader.get_enabled_datasets(self.config, mode='quantization')
        optimization_results = self._run_optimization(baseline_results, quant_datasets)

        # 階段 3：保存結果
        logger.info("\n" + "-"*60)
        logger.info("階段 3：保存結果")
        logger.info("-"*60)

        results_file = self.result_tracker.save_results(baseline_results, optimization_results)

        # 階段 3.5：清理非 Pareto 前沿的模型
        logger.info("\n" + "-"*60)
        logger.info("階段 3.5：清理非 Pareto 前沿的模型")
        logger.info("-"*60)

        self._cleanup_non_pareto_models(optimization_results)

        # 階段 4：建立視覺化
        logger.info("\n" + "-"*60)
        logger.info("階段 4：建立視覺化")
        logger.info("-"*60)

        plot_files = self.visualizer.create_visualizations(
            optimization_results['pareto_frontier'],
            optimization_results['satisfying_solutions'],
            self.output_dir
        )

        # 最終總結
        total_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("優化工作流程完成")
        logger.info("="*60)
        logger.info(f"總耗時：{total_time/60:.1f} 分鐘")
        logger.info(f"輸出目錄：{self.output_dir}")
        logger.info(f"結果檔案：{results_file}")
        if plot_files:
            logger.info(f"視覺化：已建立 {len(plot_files)} 個圖表")
        logger.info("="*60)

        # 返回摘要
        return {
            'success': True,
            'output_dir': self.output_dir,
            'results_file': results_file,
            'plot_files': plot_files,
            'total_time_sec': total_time,
            'baseline_results': baseline_results,
            'optimization_results': optimization_results,
            'n_total_trials': optimization_results['n_total_trials'],
            'n_pareto_solutions': optimization_results['n_pareto_solutions'],
            'n_satisfying_solutions': optimization_results['n_satisfying_solutions'],
            'recommended_config': optimization_results['recommended_config']
        }

    def _run_optimization(self, baseline_results: Dict[str, Any],
                         datasets: Dict[str, int]) -> Dict[str, Any]:
        """
        執行多目標優化

        Args:
            baseline_results: 基線評估結果
            datasets: 用於量化模型評估的資料集

        Returns:
            優化結果
        """
        optimizer_type = self.config['optimizer']['type']

        logger.info(f"優化器類型：{optimizer_type}")
        logger.info(f"資料集：{list(datasets.keys())}")

        # 建立優化器
        if optimizer_type == 'optuna_multiobjective':
            optimizer = OptunaMultiObjectiveOptimizer(
                config=self.config,
                baseline_metrics=baseline_results,
                evaluator_agent=self.evaluator_agent,
                datasets=datasets,
                exp_dir=self.output_dir  # 傳遞實驗目錄
            )

        elif optimizer_type == 'random':
            optimizer = RandomOptimizer(
                config=self.config,
                baseline_metrics=baseline_results,
                evaluator_agent=self.evaluator_agent,
                datasets=datasets,
                exp_dir=self.output_dir  # 傳遞實驗目錄
            )

        elif optimizer_type == 'llm_multiagent':
            optimizer = LLMMultiAgentOptimizer(
                config=self.config,
                baseline_metrics=baseline_results,
                evaluator_agent=self.evaluator_agent,
                datasets=datasets,
                exp_dir=self.output_dir  # 傳遞實驗目錄
            )

        elif optimizer_type == 'qehvi':
            # qEHVI 尚未實現
            raise NotImplementedError("qEHVI 優化器尚未實現")

        else:
            raise ValueError(f"未知的優化器類型：{optimizer_type}")

        # 執行優化
        try:
            results = optimizer.optimize()
            return results

        except Exception as e:
            logger.error(f"優化失敗：{e}")
            raise

    def _cleanup_non_pareto_models(self, optimization_results: Dict[str, Any]) -> None:
        """
        清理非 Pareto 前沿的模型，只保留最優解

        Args:
            optimization_results: 優化結果字典
        """
        # 提取所有模型路徑
        all_model_paths = []
        for trial in optimization_results['all_trials']:
            if 'quantized_model_path' in trial:
                all_model_paths.append(trial['quantized_model_path'])

        # 提取 Pareto 前沿的模型路徑
        pareto_model_paths = []
        for trial in optimization_results['pareto_frontier']:
            if 'quantized_model_path' in trial:
                pareto_model_paths.append(trial['quantized_model_path'])

        logger.info(f"總共 {len(all_model_paths)} 個模型")
        logger.info(f"Pareto 前沿 {len(pareto_model_paths)} 個模型")

        # 如果沒有模型需要清理，直接返回
        if not all_model_paths:
            logger.info("沒有找到任何模型路徑，跳過清理")
            return

        # 調用 evaluator_agent 的清理方法
        cleanup_stats = self.evaluator_agent.cleanup_non_pareto_models(
            all_model_paths=all_model_paths,
            pareto_model_paths=pareto_model_paths,
            exp_dir=self.output_dir
        )

        logger.info(f"清理統計：{cleanup_stats}")

    @classmethod
    def from_config_file(cls, config_path: str) -> 'OptimizationOrchestrator':
        """
        從配置檔案建立協調器

        Args:
            config_path: optimization_config.yaml 的路徑

        Returns:
            OptimizationOrchestrator 實例
        """
        config = ConfigLoader.load(config_path)
        return cls(config)
