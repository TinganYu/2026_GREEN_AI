"""
Monitor Agent

監控優化進度並判斷停止條件。
"""

import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from ..utils.prompt_templates import PromptTemplates

logger = logging.getLogger("MonitorAgent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class MonitorAgent(BaseAgent):
    """進度監控Agent（簡化版）"""

    def __init__(self, llm_client, convergence_window: int = 5,
                 convergence_threshold: float = 0.02, temperature: float = 0.4):
        super().__init__(llm_client, "MonitorAgent", temperature)
        self.convergence_window = convergence_window
        self.convergence_threshold = convergence_threshold

    def process(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        監控優化進度並評估停止條件

        Args:
            input_data: {
                'trials': List[Dict],           # 所有試驗
                'pareto_frontier': List[Dict],  # Pareto前沿
                'budget_status': Dict,          # {'used': int, 'max': int}
                'targets': Dict,                # 優化目標
                'pareto_history': List[Dict]    # Pareto歷史
            }
            context: 可選上下文

        Returns:
            監控評估結果
        """
        trials = input_data.get('trials', [])
        pareto = input_data.get('pareto_frontier', [])
        budget_status = input_data.get('budget_status', {'used': 0, 'max': 100})
        targets = input_data.get('targets', {})
        pareto_history = input_data.get('pareto_history', [])

        logger.info(f"Monitoring progress: {len(trials)} trials, {len(pareto)} Pareto solutions")

        # 首先進行規則檢查
        rule_based_stop = self._rule_based_stopping(
            trials, pareto, budget_status, targets, pareto_history
        )

        if rule_based_stop['should_stop']:
            logger.info(f"Rule-based stop triggered: {rule_based_stop['reason']}")

            # 創建並保存消息（規則停止）
            msg = self.create_message(
                to_agent="Orchestrator",
                message_type="progress_assessment",
                content={
                    'input_summary': {
                        'trials_count': len(trials),
                        'pareto_count': len(pareto),
                        'budget_used': budget_status['used'],
                        'budget_max': budget_status['max']
                    },
                    'assessment': rule_based_stop,
                    'rule_based': True
                },
                trial_context=budget_status['used']
            )
            self.history.append(msg)

            return rule_based_stop

        # 如果規則未觸發停止，使用LLM進行評估
        prompt = PromptTemplates.get_monitor_prompt(
            trials, pareto, budget_status, targets, pareto_history
        )

        try:
            assessment = self._call_llm_structured(prompt)

            # 驗證輸出
            required_fields = ['should_stop', 'progress_metrics']

            if not self._validate_output(assessment, required_fields):
                logger.warning("LLM output incomplete, using rule-based assessment")
                return rule_based_stop

            logger.info(f"LLM assessment: should_stop={assessment['should_stop']}, "
                       f"reason={assessment.get('reason', 'continue')}")

            # 創建並保存消息
            msg = self.create_message(
                to_agent="Orchestrator",
                message_type="progress_assessment",
                content={
                    'input_summary': {
                        'trials_count': len(trials),
                        'pareto_count': len(pareto),
                        'budget_used': budget_status['used'],
                        'budget_max': budget_status['max']
                    },
                    'assessment': assessment
                },
                trial_context=budget_status['used']
            )
            self.history.append(msg)

            return assessment

        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            return rule_based_stop

    def _rule_based_stopping(self, trials: List[Dict], pareto: List[Dict],
                            budget_status: Dict, targets: Dict,
                            pareto_history: List[Dict]) -> Dict[str, Any]:
        """
        基於規則的停止條件檢查

        Args:
            trials: 所有試驗
            pareto: Pareto前沿
            budget_status: 預算狀態
            targets: 優化目標
            pareto_history: Pareto歷史

        Returns:
            評估結果
        """
        should_stop = False
        reason = None

        # 1. 預算耗盡
        budget_used = budget_status['used'] / budget_status['max'] if budget_status['max'] > 0 else 0
        if budget_used >= 0.95:
            should_stop = True
            reason = "budget_exhausted"

        # 2. 收斂檢測（簡化版）
        if not should_stop and len(pareto_history) >= self.convergence_window:
            converged = self._check_convergence(pareto_history)
            if converged:
                should_stop = True
                reason = "convergence"

        # 3. 目標滿足
        satisfying = [t for t in pareto if t.get('satisfies_targets', False)]
        if not should_stop and len(satisfying) >= 3:
            # 如果有3個或更多滿足目標的解，可以考慮停止
            target_satisfaction = len(satisfying) / max(len(pareto), 1)
            if target_satisfaction >= 0.5:
                # 但不立即停止，只增加收斂分數
                pass

        # 計算進度指標
        successful = [t for t in trials if t.get('success', False)]
        pareto_improvement_rate = len(pareto) / len(trials) if trials else 0

        # 計算trials since last improvement
        trials_since_improvement = 0
        if len(pareto_history) >= 2:
            last_size = pareto_history[-2].get('pareto_size', 0)
            current_size = pareto_history[-1].get('pareto_size', 0) if pareto_history else 0
            if current_size <= last_size:
                trials_since_improvement = budget_status['used'] - pareto_history[-2].get('trial_count', 0)

        convergence_score = self._compute_convergence_score(pareto_history)

        progress_metrics = {
            'pareto_improvement_rate': pareto_improvement_rate,
            'target_satisfaction': len(satisfying) / max(len(pareto), 1),
            'budget_used': budget_used,
            'trials_since_improvement': trials_since_improvement
        }

        recommendation = self._generate_recommendation(
            should_stop, reason, progress_metrics
        )

        return {
            'should_stop': should_stop,
            'reason': reason,
            'convergence_score': convergence_score,
            'progress_metrics': progress_metrics,
            'recommendation': recommendation
        }

    def _check_convergence(self, pareto_history: List[Dict]) -> bool:
        """
        檢查是否收斂（基於Pareto大小）

        Args:
            pareto_history: Pareto前沿歷史

        Returns:
            True if converged, False otherwise
        """
        if len(pareto_history) < self.convergence_window:
            return False

        recent = pareto_history[-self.convergence_window:]

        # 檢查Pareto大小是否停止增長
        sizes = [h.get('pareto_size', 0) for h in recent]
        if max(sizes) - min(sizes) <= 1:
            # 連續N輪Pareto大小變化不超過1
            logger.info(f"Convergence detected: Pareto size stable at {sizes[-1]}")
            return True

        return False

    def _compute_convergence_score(self, pareto_history: List[Dict]) -> float:
        """
        計算收斂分數（0=無收斂, 1=完全收斂）

        Args:
            pareto_history: Pareto歷史

        Returns:
            收斂分數
        """
        if len(pareto_history) < 2:
            return 0.0

        recent = pareto_history[-self.convergence_window:] if len(pareto_history) >= self.convergence_window else pareto_history

        # 基於Pareto大小的穩定性
        sizes = [h.get('pareto_size', 0) for h in recent]
        size_variance = max(sizes) - min(sizes)
        size_score = 1.0 - min(size_variance / max(sizes[-1], 1), 1.0)

        return size_score

    def _generate_recommendation(self, should_stop: bool, reason: Optional[str],
                                progress_metrics: Dict) -> str:
        """
        生成推薦建議

        Args:
            should_stop: 是否應該停止
            reason: 停止原因
            progress_metrics: 進度指標

        Returns:
            推薦文本
        """
        if should_stop:
            if reason == "budget_exhausted":
                return "Budget exhausted. Optimization complete."
            elif reason == "convergence":
                return "Pareto frontier has converged. Further trials unlikely to improve."
            elif reason == "target_met":
                return "User targets satisfied. Optimization successful."
            else:
                return "Stopping condition met."
        else:
            trials_since = progress_metrics.get('trials_since_improvement', 0)
            if trials_since > 5:
                return f"No improvement in {trials_since} trials. Consider continuing with exploration strategy."
            else:
                return "Continue optimization. Progress is being made."
