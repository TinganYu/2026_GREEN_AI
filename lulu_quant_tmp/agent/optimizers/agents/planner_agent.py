"""
Planner Agent

決定優化策略並提出下一個試驗配置。
"""

import logging
import random
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from ..utils.prompt_templates import PromptTemplates

logger = logging.getLogger("PlannerAgent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class PlannerAgent(BaseAgent):
    """策略規劃Agent（簡化版）"""

    def __init__(self, llm_client, search_space: Dict, temperature: float = 0.7):
        super().__init__(llm_client, "PlannerAgent", temperature)
        self.search_space = search_space

    def process(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        規劃下一個試驗配置

        Args:
            input_data: {
                'analysis': Dict,       # AnalyzerAgent的分析報告
                'trial_num': int,       # 當前試驗編號
                'budget': Dict,         # {'used': int, 'max': int}
                'targets': Dict,        # 優化目標
                'pareto': List[Dict],   # Pareto前沿
                'trials': List[Dict]    # 已嘗試的配置（用於避免重複）
            }
            context: 可選上下文

        Returns:
            決策結果
        """
        analysis = input_data.get('analysis', {})
        trial_num = input_data.get('trial_num', 1)
        budget = input_data.get('budget', {'used': 1, 'max': 100})
        targets = input_data.get('targets', {})
        pareto = input_data.get('pareto', [])
        trials = input_data.get('trials', [])

        logger.info(f"Planning trial {trial_num}/{budget['max']}")

        # 構建prompt（傳入 search_space 和 trials 讓 LLM 知道可用參數和避免重複）
        prompt = PromptTemplates.get_planner_prompt(
            analysis, trial_num, budget, targets, pareto, self.search_space, trials
        )

        # 調用LLM獲取決策
        try:
            decision = self._call_llm_structured(prompt)

            # 驗證輸出
            required_fields = ['strategy', 'next_config', 'rationale']

            if not self._validate_output(decision, required_fields):
                logger.warning("LLM output incomplete, using fallback planning")
                decision = self._fallback_planning(analysis, trial_num, budget)

            # 確保配置有效
            decision['next_config'] = self._ensure_valid_config(decision['next_config'])

            logger.info(f"Strategy: {decision['strategy']}, Method: {decision['next_config'].get('method')}")

            # 創建並保存消息
            msg = self.create_message(
                to_agent="Orchestrator",
                message_type="strategy_decision",
                content={
                    'input_summary': {
                        'trial_num': trial_num,
                        'budget_progress': f"{trial_num}/{budget['max']}",
                        'pareto_size': len(pareto)
                    },
                    'decision': decision
                },
                trial_context=trial_num
            )
            self.history.append(msg)

            return decision

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            decision = self._fallback_planning(analysis, trial_num, budget)

            # 創建並保存消息（fallback模式）
            msg = self.create_message(
                to_agent="Orchestrator",
                message_type="strategy_decision",
                content={
                    'input_summary': {
                        'trial_num': trial_num,
                        'budget_progress': f"{trial_num}/{budget['max']}",
                        'pareto_size': len(pareto)
                    },
                    'decision': decision,
                    'fallback_mode': True
                },
                trial_context=trial_num
            )
            self.history.append(msg)

            return decision

    def _fallback_planning(self, analysis: Dict, trial_num: int, budget: Dict) -> Dict[str, Any]:
        """
        Fallback規劃（基於規則）

        當LLM失敗時使用簡單的規則選擇配置。
        """
        progress = trial_num / budget['max'] if budget['max'] > 0 else 0

        # 根據進度選擇策略
        if progress < 0.2:
            strategy = "exploration"
        elif progress < 0.7:
            strategy = "balanced"
        else:
            strategy = "exploitation"

        # 從未探索區域或隨機選擇
        unexplored = analysis.get('unexplored_regions', [])

        if unexplored and strategy in ['exploration', 'balanced']:
            # 選擇未探索區域
            region = random.choice(unexplored)
            next_config = self._generate_config_for_method(region['method'])
            rationale = f"Exploring {region['method']} method (unexplored region)"
        else:
            # 隨機選擇
            method = random.choice(self.search_space.get('methods', ['gptq']))
            next_config = self._generate_config_for_method(method)
            rationale = f"Random {strategy} configuration"

        return {
            'strategy': strategy,
            'next_config': next_config,
            'rationale': rationale,
            'confidence': 0.5
        }

    def _generate_config_for_method(self, method: str) -> Dict[str, Any]:
        """
        為指定方法生成配置

        Args:
            method: 量化方法 (gptq/awq/bnb)

        Returns:
            配置字典
        """
        config = {'method': method}

        if method not in self.search_space:
            logger.warning(f"Method {method} not in search space, using defaults")
            return config

        method_space = self.search_space[method]

        # 隨機採樣每個參數
        for param, values in method_space.items():
            if isinstance(values, list) and values:
                config[param] = random.choice(values)

        return config

    def _ensure_valid_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        確保配置有效（在搜索空間內）

        Args:
            config: 配置字典

        Returns:
            驗證並修正後的配置
        """
        method = config.get('method')

        if not method or method not in self.search_space:
            # 使用默認方法
            method = self.search_space.get('methods', ['gptq'])[0]
            config['method'] = method
            logger.warning(f"Invalid method, using default: {method}")

        if method in self.search_space:
            method_space = self.search_space[method]

            # 驗證每個參數
            for param, values in method_space.items():
                if param in config:
                    if isinstance(values, list) and config[param] not in values:
                        # 使用最接近的值或默認值
                        config[param] = values[0] if values else None
                        logger.warning(f"Invalid {param}={config[param]}, using {values[0]}")

        return config
