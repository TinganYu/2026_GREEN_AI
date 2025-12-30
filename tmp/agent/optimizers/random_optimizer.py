"""
隨機優化器

用於比較的簡單隨機搜尋基準實作。
"""

import random
import logging
from typing import Dict, List, Any

from .base_optimizer import BaseOptimizer

logger = logging.getLogger("RandomOptimizer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class RandomOptimizer(BaseOptimizer):
    """隨機搜尋優化器（基準實作）"""

    def __init__(self, config: Dict[str, Any], baseline_metrics: Dict[str, Any],
                 evaluator_agent, datasets: Dict[str, int], exp_dir: str = None):
        super().__init__(config, baseline_metrics, evaluator_agent, datasets)
        self.exp_dir = exp_dir  # 實驗根目錄

    def optimize(self) -> Dict[str, Any]:
        """執行隨機搜尋"""
        logger.info("="*60)
        logger.info("RANDOM SEARCH OPTIMIZATION")
        logger.info("="*60)

        n_trials = self.config['optimizer'].get('n_trials', 20)
        all_trials = []

        for i in range(n_trials):
            logger.info(f"\nTrial {i+1}/{n_trials}")

            # 採樣隨機配置
            quant_config = self._sample_random_config()
            logger.info(f"Config: {quant_config}")

            # 執行單次試驗
            try:
                trial_result = self.run_trial(quant_config)
                if trial_result.get('success', False):
                    all_trials.append(trial_result)
            except Exception as e:
                logger.error(f"Trial failed: {e}")

        # 取得 Pareto 前緣（Pareto frontier）
        pareto_frontier = self.get_pareto_frontier(all_trials)

        # 取得滿足目標條件的解（satisfying solutions）
        satisfying_solutions = [t for t in pareto_frontier if t['satisfies_targets']]

        # 推薦配置
        recommended = self.recommend_config(pareto_frontier, satisfying_solutions)

        logger.info("\n" + "="*60)
        logger.info("RANDOM SEARCH COMPLETED")
        logger.info(f"Completed trials: {len(all_trials)}")
        logger.info(f"Pareto solutions: {len(pareto_frontier)}")
        logger.info(f"Satisfying solutions: {len(satisfying_solutions)}")
        logger.info("="*60)

        return {
            'optimizer_type': 'random',
            'all_trials': all_trials,
            'pareto_frontier': pareto_frontier,
            'satisfying_solutions': satisfying_solutions,
            'recommended_config': recommended,
            'hypervolume': 0.0,
            'n_total_trials': len(all_trials),
            'n_pareto_solutions': len(pareto_frontier),
            'n_satisfying_solutions': len(satisfying_solutions)
        }

    def _sample_random_config(self) -> Dict[str, Any]:
        """從搜尋空間中採樣隨機配置"""
        # 隨機選擇量化方法
        method = random.choice(self.search_space['methods'])

        if method == 'gptq':
            space = self.search_space['gptq']
            return {
                'method': 'gptq',
                'bits': random.choice(space['bits']),
                'group_size': random.choice(space['group_size']),
                'calib_num': random.choice(space['calib_num']),
                'desc_act': random.choice(space['desc_act']),
                'sym': random.choice(space['sym']),
            }

        elif method == 'awq':
            space = self.search_space['awq']
            return {
                'method': 'awq',
                'w_bit': random.choice(space['w_bit']),
                'q_group_size': random.choice(space['q_group_size']),
                'zero_point': random.choice(space['zero_point']),
                'version': random.choice(space['version']),
            }

        elif method == 'bnb':
            space = self.search_space['bnb']
            return {
                'method': 'bnb',
                'bits': random.choice(space['bits']),
                'bnb_4bit_quant_type': random.choice(space['bnb_4bit_quant_type']),
                'bnb_4bit_use_double_quant': random.choice(space['bnb_4bit_use_double_quant']),
                'bnb_4bit_compute_dtype': random.choice(space['bnb_4bit_compute_dtype']),
            }

        else:
            raise ValueError(f"Unknown method: {method}")
