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

            # 驗證輸出（只檢查實際使用的欄位）
            required_fields = [
                'unexplored_regions',
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

        當LLM失敗時使用簡單的統計分析，輸出 CoT 格式。
        """
        methods = search_space.get('methods', [])

        # Step 1: Method Performance Analysis
        method_analysis = {}
        for method in methods:
            method_trials = [t for t in trials if t.get('config', {}).get('method') == method]
            successful = [t for t in method_trials if t.get('success', False)]

            if method_trials:
                avg_acc = sum(t.get('objectives', {}).get('accuracy_change', 0) for t in successful) / max(len(successful), 1)
                avg_gpu = sum(t.get('objectives', {}).get('gpu_peak_change', 0) for t in successful) / max(len(successful), 1)
            else:
                avg_acc, avg_gpu = 0, 0

            method_analysis[method] = {
                'trials': len(method_trials),
                'success_rate': len(successful) / max(len(method_trials), 1),
                'avg_accuracy': round(avg_acc, 4),
                'avg_gpu': round(avg_gpu, 4),
                'observations': f"Tried {len(method_trials)} times" if method_trials else "Not tried yet"
            }

        # Step 2: Failure Patterns
        failed_trials = [t for t in trials if not t.get('success', False)]
        failure_patterns = []
        if failed_trials:
            failure_patterns.append({
                'pattern': f"{len(failed_trials)} trials failed",
                'affected_params': {},
                'suggestion': "Check error logs for details"
            })

        # Step 3: Success Patterns
        success_patterns = []
        if pareto:
            best = pareto[0]
            success_patterns.append({
                'pattern': "Best Pareto solution",
                'winning_params': best.get('config', {}),
                'trade_off': "Current best balance"
            })

        # Step 4: Parameter Insights
        parameter_insights = {
            'bits': "Lower bits = more compression but less accuracy",
            'group_size': "Smaller group_size = more granular quantization",
            'other': "Need more trials for detailed analysis"
        }

        # Step 5: Unexplored Regions
        unexplored_regions = []
        method_counts = {}
        for t in trials:
            method = t.get('config', {}).get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1

        for method in methods:
            if method_counts.get(method, 0) < 3:
                unexplored_regions.append({
                    'method': method,
                    'params': {},
                    'rationale': f"{method} has been tried less than 3 times"
                })

        # Recommendations
        recommendations = []
        successful_count = sum(1 for t in trials if t.get('success', False))
        if successful_count < 5:
            recommendations.append("Continue exploration to gather more successful trials")
        if len(pareto) < 3:
            recommendations.append("Focus on finding diverse solutions for Pareto frontier")
        for region in unexplored_regions:
            recommendations.append(f"Explore {region['method']} method more")

        return {
            'method_analysis': method_analysis,
            'failure_patterns': failure_patterns,
            'success_patterns': success_patterns,
            'parameter_insights': parameter_insights,
            'unexplored_regions': unexplored_regions,
            'recommendations': recommendations
        }
