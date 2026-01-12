"""
Prompt Templates for LLM Agents

包含所有Agent的prompt templates和few-shot examples。
"""

from typing import Dict, List, Any


class PromptTemplates:
    """Prompt模板集合"""

    @staticmethod
    def get_analyzer_prompt(trials: List[Dict], pareto_frontier: List[Dict],
                           search_space: Dict) -> str:
        """
        AnalyzerAgent的prompt模板

        Args:
            trials: 歷史試驗列表
            pareto_frontier: 當前Pareto前沿
            search_space: 搜索空間定義

        Returns:
            完整的分析prompt
        """
        trials_summary = PromptTemplates._format_trials_summary(trials)
        pareto_summary = PromptTemplates._format_pareto_summary(pareto_frontier)
        space_summary = PromptTemplates._format_search_space(search_space)

        prompt = f"""You are an expert analyzer for quantization optimization trials.

# Your Task
Analyze the historical trials and identify patterns, failures, and unexplored regions.

# Historical Trials Summary
Total trials: {len(trials)}
Successful trials: {sum(1 for t in trials if t.get('success', False))}
Failed/Pruned trials: {len(trials) - sum(1 for t in trials if t.get('success', False))}

{trials_summary}

# Current Pareto Frontier
{len(pareto_frontier)} non-dominated solutions found.
{pareto_summary}

# Search Space
{space_summary}

# Analysis Requirements
Provide a comprehensive analysis in JSON format with these fields:

1. **failure_patterns**: List of failure patterns identified
   - pattern: Description of the pattern
   - confidence: 0.0-1.0 (how confident you are)
   - affected_configs: List of similar configs that might fail

2. **unexplored_regions**: Promising parameter combinations not yet tried
   - method: quantization method
   - params: specific parameter values
   - exploration_score: 0.0-1.0 (how valuable to explore)
   - rationale: why this region is promising

3. **parameter_sensitivity**: Which parameters have the most impact
   - parameter: parameter name
   - impact_score: 0.0-1.0 (higher = more important)
   - observation: what you observed

4. **pareto_quality**: Quality assessment of Pareto frontier
   - diversity: 0.0-1.0 (spread across objectives)
   - coverage: 0.0-1.0 (% of theoretical frontier covered)
   - improvement_rate: trend of new solutions added

5. **recommendations**: List of strategic recommendations
   - recommendation: brief actionable advice

Output ONLY a JSON object (no other text):
{{
  "failure_patterns": [...],
  "unexplored_regions": [...],
  "parameter_sensitivity": {{...}},
  "pareto_quality": {{...}},
  "recommendations": [...]
}}
"""
        return prompt

    @staticmethod
    def get_planner_prompt(analysis: Dict, trial_num: int, budget: Dict,
                          targets: Dict, pareto: List[Dict]) -> str:
        """
        PlannerAgent的prompt模板

        Args:
            analysis: AnalyzerAgent的分析報告
            trial_num: 當前試驗編號
            budget: 預算狀態 (used, max)
            targets: 優化目標
            pareto: Pareto前沿

        Returns:
            完整的規劃prompt
        """
        progress_pct = (trial_num / budget['max']) * 100 if budget['max'] > 0 else 0

        prompt = f"""You are a strategic planner for quantization optimization.

# Current State
Trial: {trial_num} / {budget['max']}
Progress: {progress_pct:.1f}%
Budget remaining: {budget['max'] - trial_num} trials

# Analysis from AnalyzerAgent
{PromptTemplates._format_json(analysis)}

# Optimization Targets
{PromptTemplates._format_json(targets)}

# Current Pareto Frontier
{len(pareto)} solutions on frontier
{PromptTemplates._format_pareto_summary(pareto)}

# Your Task
Decide the next trial configuration based on the analysis.

# Strategy Selection Guide
- **EXPLORATION** (early phase or when stuck):
  - Try unexplored parameter regions
  - Test diverse configurations
  - Focus on coverage

- **EXPLOITATION** (mid-late phase with good patterns):
  - Refine promising configurations
  - Small perturbations around successful trials
  - Focus on improvement

- **BALANCED**:
  - Mix of both strategies
  - Good default choice

# Recommended Strategy Based on Progress
- < 20% progress: Prefer EXPLORATION (70%)
- 20-70% progress: BALANCED (50/50)
- > 70% progress: Prefer EXPLOITATION (80%)

# Output Requirements
Provide your decision in JSON format:

{{
  "strategy": "exploration" / "exploitation" / "balanced",
  "next_config": {{
    "method": "gptq" / "awq" / "bnb",
    "bits": ...,
    "group_size": ...,
    ... (all relevant parameters)
  }},
  "rationale": "Explain why you chose this config (2-3 sentences)",
  "confidence": 0.0-1.0,
  "expected_objectives": {{
    "accuracy_change": expected value,
    "gpu_peak_change": expected value,
    "latency_change": expected value
  }},
  "alternative_configs": [
    {{...}},  # Backup option 1
    {{...}}   # Backup option 2
  ]
}}

Output ONLY the JSON object (no other text).
"""
        return prompt

    @staticmethod
    def get_monitor_prompt(trials: List[Dict], pareto: List[Dict],
                          budget_status: Dict, targets: Dict,
                          pareto_history: List[Dict]) -> str:
        """
        MonitorAgent的prompt模板

        Args:
            trials: 所有試驗
            pareto: Pareto前沿
            budget_status: 預算狀態
            targets: 優化目標
            pareto_history: Pareto前沿歷史

        Returns:
            完整的監控prompt
        """
        prompt = f"""You are monitoring the optimization progress.

# Progress Summary
Total trials: {len(trials)}
Successful trials: {sum(1 for t in trials if t.get('success', False))}
Pareto solutions: {len(pareto)}
Satisfying targets: {sum(1 for t in pareto if t.get('satisfies_targets', False))}

# Budget Status
Trials used: {budget_status['used']} / {budget_status['max']}
Progress: {(budget_status['used'] / budget_status['max'] * 100):.1f}%

# Targets vs Current Best
{PromptTemplates._format_target_comparison(targets, pareto)}

# Your Task
Evaluate whether optimization should continue or stop.

# Stopping Criteria to Consider
1. **CONVERGENCE**: Pareto front hasn't improved in last N trials
2. **TARGET_MET**: Found solutions satisfying all user targets
3. **BUDGET**: Budget almost exhausted (>95%)
4. **DIMINISHING_RETURNS**: Little improvement despite many trials

# Output Requirements
Provide assessment in JSON:

{{
  "should_stop": true / false,
  "reason": "convergence" / "target_met" / "budget_exhausted" / "continue" / null,
  "convergence_score": 0.0-1.0,  # 0=no convergence, 1=fully converged
  "progress_metrics": {{
    "pareto_improvement_rate": recent improvement rate,
    "target_satisfaction": % of targets met (0.0-1.0),
    "budget_used": % of budget consumed (0.0-1.0),
    "trials_since_improvement": count
  }},
  "recommendation": "Brief recommendation (1-2 sentences)"
}}

Output ONLY the JSON object (no other text).
"""
        return prompt

    # ========== 輔助格式化方法 ==========

    @staticmethod
    def _format_trials_summary(trials: List[Dict]) -> str:
        """格式化試驗摘要"""
        if not trials:
            return "No trials yet."

        summary_lines = []

        # 最近5個試驗
        recent = trials[-5:]
        summary_lines.append("Recent trials:")
        for i, t in enumerate(recent, 1):
            status = "✓" if t.get('success', False) else "✗"
            config = t.get('config', {})
            method = config.get('method', 'unknown')
            bits = config.get('bits', '?')
            objectives = t.get('objectives', {})
            acc = objectives.get('accuracy_change', 0)
            gpu = objectives.get('gpu_peak_change', 0)
            lat = objectives.get('latency_change', 0)

            summary_lines.append(
                f"  {status} Trial {len(trials) - 5 + i}: {method}-{bits}bit "
                f"(acc:{acc:+.2%}, gpu:{gpu:+.2%}, lat:{lat:+.2%})"
            )

        return "\n".join(summary_lines)

    @staticmethod
    def _format_pareto_summary(pareto: List[Dict]) -> str:
        """格式化Pareto前沿摘要"""
        if not pareto:
            return "No Pareto solutions yet."

        summary_lines = []
        for i, sol in enumerate(pareto[:5], 1):  # Top 5
            config = sol.get('config', {})
            method = config.get('method', 'unknown')
            bits = config.get('bits', '?')
            objectives = sol.get('objectives', {})
            acc = objectives.get('accuracy_change', 0)
            gpu = objectives.get('gpu_peak_change', 0)
            lat = objectives.get('latency_change', 0)
            satisfies = "✓" if sol.get('satisfies_targets', False) else "✗"

            summary_lines.append(
                f"  {i}. {method}-{bits}bit {satisfies}: "
                f"acc:{acc:+.2%}, gpu:{gpu:+.2%}, lat:{lat:+.2%}"
            )

        if len(pareto) > 5:
            summary_lines.append(f"  ... and {len(pareto) - 5} more")

        return "\n".join(summary_lines)

    @staticmethod
    def _format_search_space(space: Dict) -> str:
        """格式化搜索空間"""
        lines = []
        lines.append(f"Methods: {space.get('methods', [])}")

        for method in space.get('methods', []):
            if method in space:
                lines.append(f"\n{method.upper()}:")
                for param, values in space[method].items():
                    lines.append(f"  {param}: {values}")

        return "\n".join(lines)

    @staticmethod
    def _format_json(obj: Any) -> str:
        """格式化JSON對象為字符串"""
        import json
        return json.dumps(obj, indent=2, ensure_ascii=False)

    @staticmethod
    def _format_target_comparison(targets: Dict, pareto: List[Dict]) -> str:
        """格式化目標與實際對比"""
        if not pareto:
            return "No solutions to compare."

        lines = []
        lines.append("Targets:")
        lines.append(f"  accuracy_min: {targets.get('accuracy_min', 0):+.2%}")
        lines.append(f"  gpu_peak_max: {targets.get('gpu_peak_max', 0):+.2%}")
        lines.append(f"  latency_max: {targets.get('latency_max', 0):+.2%}")

        # 找最佳解
        best = pareto[0] if pareto else None
        if best:
            obj = best.get('objectives', {})
            lines.append("\nBest solution:")
            lines.append(f"  accuracy: {obj.get('accuracy_change', 0):+.2%}")
            lines.append(f"  gpu_peak: {obj.get('gpu_peak_change', 0):+.2%}")
            lines.append(f"  latency: {obj.get('latency_change', 0):+.2%}")
            lines.append(f"  Satisfies targets: {best.get('satisfies_targets', False)}")

        return "\n".join(lines)


if __name__ == "__main__":
    # 測試prompt生成
    templates = PromptTemplates()

    # 測試Analyzer prompt
    test_trials = [
        {
            'success': True,
            'config': {'method': 'gptq', 'bits': 4},
            'objectives': {'accuracy_change': -0.05, 'gpu_peak_change': -0.30,
                          'latency_change': 0.10}
        }
    ]

    analyzer_prompt = templates.get_analyzer_prompt(
        trials=test_trials,
        pareto_frontier=[],
        search_space={'methods': ['gptq', 'awq']}
    )

    print("="*60)
    print("ANALYZER PROMPT")
    print("="*60)
    print(analyzer_prompt[:500] + "...")
