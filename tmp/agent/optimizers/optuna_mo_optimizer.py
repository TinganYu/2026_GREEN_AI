"""
Optuna 多目標優化器

使用 Optuna 的 TPE 或 NSGA-II 採樣器進行多目標優化。
"""

import logging
from typing import Dict, List, Any
import optuna
from optuna.samplers import TPESampler, NSGAIISampler

from .base_optimizer import BaseOptimizer

logger = logging.getLogger("OptunaOptimizer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class OptunaMultiObjectiveOptimizer(BaseOptimizer):
    """基於 Optuna 的多目標優化器"""

    def __init__(self, config: Dict[str, Any], baseline_metrics: Dict[str, Any],
                 evaluator_agent, datasets: Dict[str, int], exp_dir: str = None):
        super().__init__(config, baseline_metrics, evaluator_agent, datasets)

        self.optuna_config = config['optimizer']['optuna']
        self.search_space = config['optimizer']['search_space']
        self.exp_dir = exp_dir  # 實驗根目錄

        # 追蹤試驗編號以供記錄
        self.trial_count = 0

    def optimize(self) -> Dict[str, Any]:
        """
        執行 Optuna 多目標優化

        Returns:
            包含優化結果的字典
        """
        logger.info("="*60)
        logger.info("OPTUNA 多目標優化")
        logger.info("="*60)
        logger.info(f"採樣器：{self.optuna_config['sampler']}")
        logger.info(f"總試驗次數：{self.optuna_config['n_trials']}")
        logger.info(f"目標：3 個（accuracy_change、gpu_peak_change、latency_change）")
        logger.info("="*60)

        # 建立採樣器
        sampler = self._create_sampler()

        # 建立 study
        study = optuna.create_study(
            directions=['minimize', 'minimize', 'minimize'],  # 全部轉換為最小化
            sampler=sampler,
            study_name=self.config['experiment']['name']
        )

        # 執行優化
        n_trials = self.optuna_config['n_trials']
        try:
            study.optimize(
                self.objective,
                n_trials=n_trials,
                show_progress_bar=True,
                n_jobs=1  # 順序執行（GPU 約束）
            )
        except KeyboardInterrupt:
            logger.warning("優化已被用戶中斷")

        logger.info("\n" + "="*60)
        logger.info("優化完成")
        logger.info("="*60)

        # 收集所有試驗（成功、剪枝、失敗）
        all_trials = []
        successful_trials = []

        for t in study.trials:
            trial_dict = self._trial_to_dict(t)
            all_trials.append(trial_dict)

            # 只有成功的試驗（無 inf 值）才能用於 Pareto 分析
            if t.state == optuna.trial.TrialState.COMPLETE:
                if t.values and not any(v == float('inf') or v == float('-inf') for v in t.values):
                    successful_trials.append(trial_dict)

        logger.info(f"總試驗：{len(all_trials)} 個")
        logger.info(f"成功的試驗：{len(successful_trials)} 個")
        logger.info(f"被剪枝的試驗：{sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)} 個")
        logger.info(f"失敗的試驗：{len(all_trials) - len(successful_trials) - sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)} 個")

        # 使用成功的試驗計算 Pareto 前沿（而非全部）
        pareto_frontier = self.get_pareto_frontier(successful_trials)

        # 獲取滿足目標的解決方案
        satisfying_solutions = [t for t in pareto_frontier if t['satisfies_targets']]
        logger.info(f"滿足目標的解決方案（符合所有目標）：{len(satisfying_solutions)} 個")

        # 推薦最佳配置
        recommended = self.recommend_config(pareto_frontier, satisfying_solutions)

        if recommended and recommended.get('config'):
            logger.info("\n推薦的配置：")
            logger.info(f"  方法：{recommended['config'].get('method', '未知')}")
            logger.info(f"  準確率變化：{recommended['objectives']['accuracy_change']:+.3%}")
            logger.info(f"  GPU 峰值變化：{recommended['objectives']['gpu_peak_change']:+.3%}")
            logger.info(f"  延遲變化：{recommended['objectives']['latency_change']:+.3%}")
            logger.info(f"  滿足目標：{recommended['satisfies_targets']}")
        else:
            logger.warning("\n未找到有效配置 - 所有試驗均失敗")

        results = {
            'optimizer_type': 'optuna_multiobjective',
            'all_trials': all_trials,  # 所有試驗（完成/剪枝/失敗）
            'pareto_frontier': pareto_frontier,
            'satisfying_solutions': satisfying_solutions,
            'recommended_config': recommended,
            'n_total_trials': len(all_trials),  # 總數（包含所有狀態）
            'n_pareto_solutions': len(pareto_frontier),
            'n_satisfying_solutions': len(satisfying_solutions)
        }

        return results

    def objective(self, trial: optuna.Trial):
        """
        Optuna 目標函數

        返回 3 個目標（根據自然方向）：
            - -accuracy_change（maximize accuracy = minimize -accuracy）
            - gpu_peak_change（minimize gpu）
            - latency_change（minimize latency）
        """
        self.trial_count += 1
        trial_number = trial.number  # Optuna 的試驗編號（從 0 開始）

        logger.info(f"\n{'='*60}")
        logger.info(f"試驗 {self.trial_count} / {self.optuna_config['n_trials']} (Trial #{trial_number})")
        logger.info(f"{'='*60}")

        # 採樣配置
        quant_config = self._sample_config(trial)
        logger.info(f"配置：{quant_config}")

        # 執行實驗
        try:
            trial_result = self.run_trial(quant_config, trial_number, self.exp_dir)
        except Exception as e:
            logger.error(f"試驗失敗，發生異常：{e}")
            # 即使失敗也要記錄配置，以便追蹤
            trial.set_user_attr('config', quant_config)
            trial.set_user_attr('error', str(e))
            trial.set_user_attr('status', 'failed')
            trial.set_user_attr('satisfies_constraints', False)  # 無法評估約束，標記為不滿足
            # 返回最差值以標記為失敗
            return float('inf'), float('inf'), float('inf')

        if not trial_result.get('success', False):
            logger.error(f"試驗失敗：{trial_result.get('error', '未知')}")
            # 即使失敗也要記錄配置，以便追蹤
            trial.set_user_attr('config', quant_config)
            trial.set_user_attr('error', trial_result.get('error', '未知'))
            trial.set_user_attr('status', 'failed')
            trial.set_user_attr('satisfies_constraints', False)  # 無法評估約束，標記為不滿足
            return float('inf'), float('inf'), float('inf')

        # 獲取目標值
        obj = trial_result['objectives']
        accuracy_change = obj['accuracy_change']
        gpu_peak_change = obj['gpu_peak_change']
        latency_change = obj['latency_change']

        # 記錄目標值（使用 :+.3% 顯示正負號）
        logger.info(f"目標值：")
        logger.info(f"  準確率變化：{accuracy_change:+.3%}（目標：≥{self.targets_config['accuracy_min']:+.3%}）")
        logger.info(f"  GPU 峰值變化：{gpu_peak_change:+.3%}（目標：≤{self.targets_config['gpu_peak_max']:+.3%}）")
        logger.info(f"  延遲變化：{latency_change:+.3%}（目標：≤{self.targets_config['latency_max']:+.3%}）")

        # 檢查約束（硬剪枝）
        if not trial_result['satisfies_constraints']:
            logger.warning("✗ 違反約束 - 試驗被剪枝")
            # 記錄配置
            trial.set_user_attr('config', quant_config)
            trial.set_user_attr('status', 'pruned')
            trial.set_user_attr('satisfies_constraints', False)
            trial.set_user_attr('objectives', obj)
            if 'results' in trial_result and 'quantized_model_path' in trial_result['results']:
                trial.set_user_attr('quantized_model_path', trial_result['results']['quantized_model_path'])
            raise optuna.TrialPruned()

        # 設置用戶屬性
        trial.set_user_attr('config', quant_config)
        trial.set_user_attr('results', trial_result['results'])
        trial.set_user_attr('satisfies_targets', trial_result['satisfies_targets'])
        trial.set_user_attr('violation_score', trial_result['violation_score'])
        trial.set_user_attr('status', 'completed')
        trial.set_user_attr('objectives', obj)
        trial.set_user_attr('satisfies_constraints', trial_result['satisfies_constraints'])

        # 記錄模型路徑（如果可用）
        if 'results' in trial_result and 'quantized_model_path' in trial_result['results']:
            trial.set_user_attr('quantized_model_path', trial_result['results']['quantized_model_path'])

        # 記錄狀態
        if trial_result['satisfies_targets']:
            logger.info("✓ 滿足所有目標！")
        else:
            logger.info(f"⚠ 違反分數：{trial_result['violation_score']:.4f}")

        # 返回 3 個目標（全部轉為 minimize）
        # accuracy_change 取負以轉為 minimize（maximize accuracy = minimize -accuracy）
        return -accuracy_change, gpu_peak_change, latency_change

    def _sample_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        從搜索空間採樣量化配置

        Args:
            trial: Optuna trial 物件

        Returns:
            量化配置字典
        """
        # 採樣方法
        methods = self.search_space['methods']
        method = trial.suggest_categorical('method', methods)

        # 採樣方法特定的參數
        if method == 'gptq':
            space = self.search_space['gptq']
            config = {
                'method': 'gptq',
                'bits': trial.suggest_categorical('gptq_bits', space['bits']),
                'group_size': trial.suggest_categorical('gptq_group_size', space['group_size']),
                'calib_num': trial.suggest_categorical('gptq_calib_num', space['calib_num']),
                'desc_act': trial.suggest_categorical('gptq_desc_act', space['desc_act']),
                'sym': trial.suggest_categorical('gptq_sym', space['sym']),
                'damp_percent': trial.suggest_categorical('gptq_damp_percent', space['damp_percent']),
                'damp_auto_increment': trial.suggest_categorical('gptq_damp_auto_increment', space['damp_auto_increment']),
                'act_group_aware': trial.suggest_categorical('gptq_act_group_aware', space['act_group_aware']),
                'static_groups': trial.suggest_categorical('gptq_static_groups', space['static_groups']),
                'true_sequential': trial.suggest_categorical('gptq_true_sequential', space['true_sequential']),
                'lm_head': trial.suggest_categorical('gptq_lm_head', space['lm_head']),
                'mse': trial.suggest_categorical('gptq_mse', space['mse']),
                'rotation': trial.suggest_categorical('gptq_rotation', space['rotation']),
            }

        elif method == 'awq':
            space = self.search_space['awq']
            config = {
                'method': 'awq',
                'w_bit': trial.suggest_categorical('awq_w_bit', space['w_bit']),
                'q_group_size': trial.suggest_categorical('awq_q_group_size', space['q_group_size']),
                'zero_point': trial.suggest_categorical('awq_zero_point', space['zero_point']),
                'version': trial.suggest_categorical('awq_version', space['version']),
                'modules_to_not_convert': trial.suggest_categorical('awq_modules_to_not_convert', space['modules_to_not_convert']),
            }

        elif method == 'bnb':
            space = self.search_space['bnb']
            config = {
                'method': 'bnb',
                'bits': trial.suggest_categorical('bnb_bits', space['bits']),
                'bnb_4bit_quant_type': trial.suggest_categorical('bnb_4bit_quant_type', space['bnb_4bit_quant_type']),
                'bnb_4bit_use_double_quant': trial.suggest_categorical('bnb_4bit_use_double_quant', space['bnb_4bit_use_double_quant']),
                'bnb_4bit_compute_dtype': trial.suggest_categorical('bnb_4bit_compute_dtype', space['bnb_4bit_compute_dtype']),
                'llm_int8_threshold': trial.suggest_categorical('bnb_llm_int8_threshold', space['llm_int8_threshold']),
                'llm_int8_skip_modules': trial.suggest_categorical('bnb_llm_int8_skip_modules', space['llm_int8_skip_modules']),
                'llm_int8_has_fp16_weight': trial.suggest_categorical('bnb_llm_int8_has_fp16_weight', space['llm_int8_has_fp16_weight']),
                'llm_int8_enable_fp32_cpu_offload': trial.suggest_categorical('bnb_llm_int8_enable_fp32_cpu_offload', space['llm_int8_enable_fp32_cpu_offload']),
            }

        else:
            raise ValueError(f"未知的方法：{method}")

        return config

    def _create_sampler(self):
        """建立 Optuna 採樣器"""
        sampler_type = self.optuna_config['sampler']

        if sampler_type == "TPESampler":
            params = self.optuna_config['tpe_params']
            return TPESampler(
                n_startup_trials=self.optuna_config['n_startup_trials'],
                n_ei_candidates=params['n_ei_candidates'],
                multivariate=params['multivariate'],
                constant_liar=params['constant_liar']
            )

        elif sampler_type == "NSGAIISampler":
            params = self.optuna_config['nsgaii_params']
            return NSGAIISampler(
                population_size=params['population_size'],
                mutation_prob=params['mutation_prob'],
                crossover_prob=params['crossover_prob']
            )

        else:
            raise ValueError(f"未知的採樣器：{sampler_type}")

    def _trial_to_dict(self, trial: optuna.Trial) -> Dict[str, Any]:
        """將 Optuna trial 轉換為字典，包含所有狀態的試驗

        Args:
            trial: Optuna Trial 物件

        Returns:
            包含試驗完整信息的字典
        """
        # 基本信息
        trial_dict = {
            'number': trial.number,
            'state': trial.state.name,  # COMPLETE / PRUNED / FAIL
            'datetime_start': trial.datetime_start,
            'datetime_complete': trial.datetime_complete,
        }

        # 從 user_attrs 提取信息（我們在 objective() 中設置的）
        user_attrs = trial.user_attrs

        # 配置
        trial_dict['config'] = user_attrs.get('config', {})

        # 狀態（我們自定義的）
        trial_dict['status'] = user_attrs.get('status', 'unknown')

        # 目標值
        if 'objectives' in user_attrs:
            # 優先使用 user_attrs 中的目標值（包含失敗和剪枝的試驗）
            trial_dict['objectives'] = user_attrs['objectives']
        elif trial.state == optuna.trial.TrialState.COMPLETE and trial.values:
            # 從 trial.values 構建（成功的試驗）
            # 注意：第一個值需要反轉回來（因為我們在 objective() 中取了負）
            trial_dict['objectives'] = {
                'accuracy_change': -trial.values[0],  # 反轉回正常值
                'gpu_peak_change': trial.values[1],
                'latency_change': trial.values[2]
            }
        else:
            # 失敗/剪枝的試驗，使用 inf
            trial_dict['objectives'] = {
                'accuracy_change': float('-inf'),  # 最壞的準確率變化
                'gpu_peak_change': float('inf'),    # 最壞的 GPU 變化
                'latency_change': float('inf')      # 最壞的延遲變化
            }

        # 約束和目標滿足情況
        trial_dict['satisfies_constraints'] = user_attrs.get('satisfies_constraints', True)
        trial_dict['satisfies_targets'] = user_attrs.get('satisfies_targets', False)
        trial_dict['violation_score'] = user_attrs.get('violation_score', float('inf'))

        # 錯誤信息（如果失敗）
        if 'error' in user_attrs:
            trial_dict['error'] = user_attrs['error']

        # 模型路徑
        if 'quantized_model_path' in user_attrs:
            trial_dict['quantized_model_path'] = user_attrs['quantized_model_path']

        # 評估結果（如果有）
        if 'results' in user_attrs:
            trial_dict['results'] = user_attrs['results']

        # 保持向後兼容性
        trial_dict['success'] = trial.state == optuna.trial.TrialState.COMPLETE and not any(
            v == float('inf') or v == float('-inf')
            for v in (trial.values if trial.values else [])
        )

        return trial_dict

