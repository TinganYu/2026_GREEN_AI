"""
Analyzer Agent

分析歷史試驗，識別模式、失敗原因和未探索區域。
"""

import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from ..utils.prompt_templates import PromptTemplates

logger = logging.getLogger("AnalyzerAgent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class AnalyzerAgent(BaseAgent):
    """歷史分析Agent（簡化版）"""

    def __init__(self, llm_client, temperature: float = 0.3):
        super().__init__(llm_client, "AnalyzerAgent", temperature)

    def process(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        分析歷史試驗

        Args:
            input_data: {
                'trials': List[Dict],           # 所有試驗
                'pareto_frontier': List[Dict],  # Pareto前沿
                'search_space': Dict            # 搜索空間
            }
            context: 可選上下文

        Returns:
            分析報告
        """
        trials = input_data.get('trials', [])
        pareto = input_data.get('pareto_frontier', [])
        search_space = input_data.get('search_space', {})

        logger.info(f"Analyzing {len(trials)} trials, {len(pareto)} Pareto solutions")

        # 構建prompt
        prompt = PromptTemplates.get_analyzer_prompt(trials, pareto, search_space)

        # 調用LLM獲取結構化分析
        try:
            analysis = self._call_llm_structured(prompt)

            # 驗證輸出
            required_fields = [
                'failure_patterns',
                'unexplored_regions',
                'parameter_sensitivity',
                'pareto_quality',
                'recommendations'
            ]

            if not self._validate_output(analysis, required_fields):
                # 使用fallback分析
                logger.warning("LLM output incomplete, using fallback analysis")
                analysis = self._fallback_analysis(trials, pareto, search_space)

            logger.info(f"Analysis complete: {len(analysis.get('recommendations', []))} recommendations")

            # 創建並保存消息
            msg = self.create_message(
                to_agent="Orchestrator",
                message_type="analysis_report",
                content={
                    'input_summary': {
                        'trials_count': len(trials),
                        'pareto_count': len(pareto),
                        'methods_available': search_space.get('methods', [])
                    },
                    'analysis': analysis
                },
                trial_context=context.get('trial_num') if context else None
            )
            self.history.append(msg)

            return analysis

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            # 使用fallback
            analysis = self._fallback_analysis(trials, pareto, search_space)

            # 創建並保存消息（fallback模式）
            msg = self.create_message(
                to_agent="Orchestrator",
                message_type="analysis_report",
                content={
                    'input_summary': {
                        'trials_count': len(trials),
                        'pareto_count': len(pareto),
                        'methods_available': search_space.get('methods', [])
                    },
                    'analysis': analysis,
                    'fallback_mode': True
                },
                trial_context=context.get('trial_num') if context else None
            )
            self.history.append(msg)

            return analysis

    def _fallback_analysis(self, trials: List[Dict], pareto: List[Dict],
                          search_space: Dict) -> Dict[str, Any]:
        """
        Fallback分析（基於規則）

        當LLM失敗時使用簡單的統計分析。
        """
        successful = [t for t in trials if t.get('success', False)]
        failed = [t for t in trials if not t.get('success', False)]

        # 簡單的失敗模式識別
        failure_patterns = []
        if len(failed) > 0:
            # 統計失敗配置的共同點
            methods_failed = {}
            for f in failed:
                method = f.get('config', {}).get('method', 'unknown')
                methods_failed[method] = methods_failed.get(method, 0) + 1

            for method, count in methods_failed.items():
                if count >= 2:
                    failure_patterns.append({
                        'pattern': f"{method} method has {count} failures",
                        'confidence': min(count / len(failed), 1.0),
                        'affected_configs': []
                    })

        # 識別未探索區域（簡單方法：檢查哪些方法很少嘗試）
        unexplored_regions = []
        methods = search_space.get('methods', [])
        method_counts = {}
        for t in trials:
            method = t.get('config', {}).get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1

        for method in methods:
            if method_counts.get(method, 0) < 3:  # 少於3次嘗試
                unexplored_regions.append({
                    'method': method,
                    'params': {},
                    'exploration_score': 0.7,
                    'rationale': f"{method} has been tried less than 3 times"
                })

        # 參數敏感度（簡化：基於成功率）
        parameter_sensitivity = {
            'bits': 0.8,  # 假設bits很重要
            'group_size': 0.6,
            'method': 0.9
        }

        # Pareto質量評估
        pareto_quality = {
            'diversity': min(len(pareto) / 5.0, 1.0),  # 至少5個解為滿分
            'coverage': len(pareto) / max(len(successful), 1),
            'improvement_rate': len(pareto) / max(len(trials), 1)
        }

        # 推薦
        recommendations = []
        if len(successful) < 5:
            recommendations.append("Continue exploration to gather more successful trials")
        if len(pareto) < 3:
            recommendations.append("Focus on finding diverse solutions for Pareto frontier")
        for region in unexplored_regions:
            recommendations.append(f"Explore {region['method']} method more")

        return {
            'failure_patterns': failure_patterns,
            'unexplored_regions': unexplored_regions,
            'parameter_sensitivity': parameter_sensitivity,
            'pareto_quality': pareto_quality,
            'recommendations': recommendations
        }
