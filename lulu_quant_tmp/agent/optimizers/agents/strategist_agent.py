"""
Strategist Agent

合併 Analyzer 和 Planner 功能的單一 Agent。
在一次 LLM 調用中完成分析和決策，減少 API 成本和延遲。
"""

import logging
import random
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from ..utils.prompt_templates import PromptTemplates

logger = logging.getLogger("StrategistAgent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class StrategistAgent(BaseAgent):
    """
    策略師 Agent（合併版）

    在一次 LLM 調用中完成：
    1. 分析歷史試驗
    2. 決定下一個配置
    """

    def __init__(self, llm_client, search_space: Dict, temperature: float = 0.5):
        super().__init__(llm_client, "StrategistAgent", temperature)
        self.search_space = search_space

    def process(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        分析歷史並決定下一個配置

        Args:
            input_data: {
                'trials': List[Dict],           # 所有試驗
                'pareto_frontier': List[Dict],  # Pareto前沿
                'trial_num': int,               # 當前試驗編號
                'budget': Dict,                 # {'used': int, 'max': int}
                'targets': Dict                 # 優化目標
            }
            context: 可選上下文

        Returns:
            包含分析和決策的結果
        """
        trials = input_data.get('trials', [])
        pareto = input_data.get('pareto_frontier', [])
        trial_num = input_data.get('trial_num', 1)
        budget = input_data.get('budget', {'used': 1, 'max': 100})
        targets = input_data.get('targets', {})

        logger.info(f"Strategist processing: {len(trials)} trials, planning trial {trial_num}/{budget['max']}")

        # 構建合併的 prompt
        prompt = PromptTemplates.get_strategist_prompt(
            trials=trials,
            pareto_frontier=pareto,
            search_space=self.search_space,
            trial_num=trial_num,
            budget=budget,
            targets=targets
        )

        try:
            result = self._call_llm_structured(prompt)

            # 驗證輸出
            required_fields = ['analysis', 'decision']
            if not self._validate_output(result, required_fields):
                logger.warning("LLM output incomplete, using fallback")
                result = self._fallback_strategist(trials, pareto, trial_num, budget)

            # 驗證 decision 內的必要欄位
            decision = result.get('decision', {})
            decision_required = ['strategy', 'next_config', 'rationale']
            if not all(f in decision for f in decision_required):
                logger.warning("Decision incomplete, using fallback")
                result = self._fallback_strategist(trials, pareto, trial_num, budget)

            # 確保配置有效
            if 'decision' in result and 'next_config' in result['decision']:
                result['decision']['next_config'] = self._ensure_valid_config(
                    result['decision']['next_config']
                )

            logger.info(f"Strategy: {result['decision']['strategy']}, "
                       f"Method: {result['decision']['next_config'].get('method')}")

            # 創建並保存消息
            msg = self.create_message(
                to_agent="Orchestrator",
                message_type="strategist_result",
                content={
                    'input_summary': {
                        'trials_count': len(trials),
                        'pareto_count': len(pareto),
                        'trial_num': trial_num,
                        'budget_max': budget['max']
                    },
                    'result': result
                },
                trial_context=trial_num
            )
            self.history.append(msg)

            return result

        except Exception as e:
            logger.error(f"Strategist failed: {e}")
            result = self._fallback_strategist(trials, pareto, trial_num, budget)

            msg = self.create_message(
                to_agent="Orchestrator",
                message_type="strategist_result",
                content={
                    'input_summary': {
                        'trials_count': len(trials),
                        'pareto_count': len(pareto),
                        'trial_num': trial_num,
                        'budget_max': budget['max']
                    },
                    'result': result,
                    'fallback_mode': True
                },
                trial_context=trial_num
            )
            self.history.append(msg)

            return result

    def _fallback_strategist(self, trials: List[Dict], pareto: List[Dict],
                            trial_num: int, budget: Dict) -> Dict[str, Any]:
        """
        Fallback 策略（基於規則）
        """
        methods = self.search_space.get('methods', [])

        # ===== 分析部分 =====
        method_analysis = {}
        for method in methods:
            method_trials = [t for t in trials if t.get('config', {}).get('method') == method]
            successful = [t for t in method_trials if t.get('success', False)]

            if successful:
                avg_acc = sum(t.get('objectives', {}).get('accuracy_change', 0) for t in successful) / len(successful)
                avg_gpu = sum(t.get('objectives', {}).get('gpu_peak_change', 0) for t in successful) / len(successful)
            else:
                avg_acc, avg_gpu = 0, 0

            method_analysis[method] = {
                'trials': len(method_trials),
                'success_rate': len(successful) / max(len(method_trials), 1),
                'avg_accuracy': round(avg_acc, 4),
                'avg_gpu': round(avg_gpu, 4)
            }

        # 未探索區域
        method_counts = {m: 0 for m in methods}
        for t in trials:
            method = t.get('config', {}).get('method', 'unknown')
            if method in method_counts:
                method_counts[method] += 1

        unexplored = [m for m, c in method_counts.items() if c < 3]

        # ===== 決策部分 =====
        progress = trial_num / budget['max'] if budget['max'] > 0 else 0

        if progress < 0.2:
            strategy = "exploration"
        elif progress < 0.7:
            strategy = "balanced"
        else:
            strategy = "exploitation"

        # 選擇方法
        if unexplored and strategy in ['exploration', 'balanced']:
            method = random.choice(unexplored)
            rationale = f"Exploring {method} (under-explored)"
        else:
            method = random.choice(methods)
            rationale = f"Random {strategy} selection"

        next_config = self._generate_config_for_method(method)

        return {
            'analysis': {
                'method_analysis': method_analysis,
                'failure_patterns': [],
                'success_patterns': [],
                'parameter_insights': {
                    'bits': "Lower bits = more compression",
                    'group_size': "Affects quantization granularity"
                },
                'unexplored_regions': [{'method': m, 'params': {}} for m in unexplored],
                'recommendations': [f"Explore {m}" for m in unexplored]
            },
            'decision': {
                'thinking': {
                    'analyzer_insights': "Using fallback rule-based analysis",
                    'target_gap': "Unknown",
                    'avoid_repeating': "N/A",
                    'chosen_strategy_reason': f"Progress {progress:.0%}, using {strategy}"
                },
                'strategy': strategy,
                'next_config': next_config,
                'rationale': rationale,
                'confidence': 0.5
            }
        }

    def _generate_config_for_method(self, method: str) -> Dict[str, Any]:
        """為指定方法生成配置"""
        config = {'method': method}

        if method not in self.search_space:
            return config

        method_space = self.search_space[method]

        for param, values in method_space.items():
            if isinstance(values, list) and values:
                config[param] = random.choice(values)

        return config

    def _ensure_valid_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """確保配置有效"""
        method = config.get('method')

        if not method or method not in self.search_space.get('methods', []):
            method = self.search_space.get('methods', ['gptq'])[0]
            config['method'] = method
            logger.warning(f"Invalid method, using default: {method}")

        if method in self.search_space:
            method_space = self.search_space[method]

            for param, values in method_space.items():
                if param in config:
                    if isinstance(values, list) and config[param] not in values:
                        config[param] = values[0] if values else None
                        logger.warning(f"Invalid {param}, using {values[0]}")

        return config
