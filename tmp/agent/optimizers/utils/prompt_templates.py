"""
Prompt Templates for LLM Agents

支援從 YAML 配置檔載入多種類型的 prompt 模板。
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger("PromptTemplates")


class PromptConfigLoader:
    """Prompt 配置載入器"""

    _instance: Optional['PromptConfigLoader'] = None
    _config: Optional[Dict] = None
    _current_type: str = "en"
    _config_path: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load(cls, config_path: Optional[str] = None, prompt_type: Optional[str] = None) -> 'PromptConfigLoader':
        """
        載入 prompt 配置

        Args:
            config_path: YAML 配置檔路徑，預設為 tmp/config/prompt_config.yaml
            prompt_type: 使用的 prompt 類型（如 'en', 'zh'），None 則使用預設

        Returns:
            PromptConfigLoader 實例
        """
        instance = cls()

        # 決定配置檔路徑
        if config_path is None:
            # 尋找預設路徑
            base_dir = Path(__file__).parent.parent.parent.parent  # tmp/
            config_path = base_dir / "config" / "prompt_config.yaml"
        else:
            config_path = Path(config_path)

        # 載入配置
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                cls._config = yaml.safe_load(f)
            cls._config_path = str(config_path)
            logger.info(f"Loaded prompt config from: {config_path}")

            # 設定類型
            if prompt_type is not None:
                cls._current_type = prompt_type
            else:
                cls._current_type = cls._config.get('default_type', 'en')

            # 驗證類型是否支援
            supported = cls._config.get('supported_types', ['en'])
            if cls._current_type not in supported:
                logger.warning(f"Prompt type '{cls._current_type}' not supported, using 'en'")
                cls._current_type = 'en'

            logger.info(f"Using prompt type: {cls._current_type}")
        else:
            logger.warning(f"Prompt config not found at {config_path}, using built-in defaults")
            cls._config = None
            cls._current_type = "en"

        return instance

    @classmethod
    def get_prompts(cls, agent: str) -> Dict[str, Any]:
        """
        取得指定 agent 的 prompts

        Args:
            agent: agent 名稱 ('analyzer', 'planner', 'monitor')

        Returns:
            該 agent 的 prompt 配置字典
        """
        if cls._config is None:
            return {}

        prompts = cls._config.get('prompts', {})
        type_prompts = prompts.get(cls._current_type, {})
        return type_prompts.get(agent, {})

    @classmethod
    def get_formatting(cls) -> Dict[str, str]:
        """取得格式化配置"""
        if cls._config is None:
            return {}

        prompts = cls._config.get('prompts', {})
        type_prompts = prompts.get(cls._current_type, {})
        return type_prompts.get('formatting', {})

    @classmethod
    def get_current_type(cls) -> str:
        """取得當前使用的 prompt 類型"""
        return cls._current_type

    @classmethod
    def get_config_path(cls) -> Optional[str]:
        """取得配置檔路徑"""
        return cls._config_path

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """取得當前類型的 metadata"""
        if cls._config is None:
            return {"name": "Built-in English", "version": "1.0"}

        prompts = cls._config.get('prompts', {})
        type_prompts = prompts.get(cls._current_type, {})
        return type_prompts.get('metadata', {})

    @classmethod
    def get_full_prompt_info(cls) -> Dict[str, Any]:
        """
        取得完整的 prompt 資訊，用於輸出到結果

        Returns:
            包含類型、路徑、metadata 和所有 prompts 的字典
        """
        return {
            "prompt_type": cls._current_type,
            "config_path": cls._config_path,
            "metadata": cls.get_metadata(),
            "supported_types": cls._config.get('supported_types', ['en']) if cls._config else ['en'],
            "prompts": {
                "analyzer": cls.get_prompts('analyzer'),
                "planner": cls.get_prompts('planner'),
                "monitor": cls.get_prompts('monitor'),
                "formatting": cls.get_formatting()
            }
        }


class PromptTemplates:
    """Prompt模板集合"""

    @staticmethod
    def _get_config() -> PromptConfigLoader:
        """確保配置已載入"""
        if PromptConfigLoader._config is None:
            PromptConfigLoader.load()
        return PromptConfigLoader()

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
        PromptTemplates._get_config()
        prompts = PromptConfigLoader.get_prompts('analyzer')
        formatting = PromptConfigLoader.get_formatting()

        trials_summary = PromptTemplates._format_trials_summary(trials)
        pareto_summary = PromptTemplates._format_pareto_summary(pareto_frontier)
        space_summary = PromptTemplates._format_search_space(search_space)

        # 如果有配置，使用配置中的 prompts
        if prompts:
            labels = prompts.get('labels', {})
            headers = prompts.get('section_headers', {})

            total = len(trials)
            successful = sum(1 for t in trials if t.get('success', False))
            failed = total - successful

            prompt = f"""{prompts.get('system_role', '')}

{headers.get('your_task', '# Your Task')}
{prompts.get('task_description', '')}

{headers.get('historical_trials', '# Historical Trials Summary')}
{labels.get('total_trials', 'Total trials')}: {total}
{labels.get('successful_trials', 'Successful trials')}: {successful}
{labels.get('failed_trials', 'Failed/Pruned trials')}: {failed}

{trials_summary}

{headers.get('pareto_frontier', '# Current Pareto Frontier')}
{len(pareto_frontier)} {labels.get('non_dominated_solutions', 'non-dominated solutions found')}.
{pareto_summary}

{headers.get('search_space', '# Search Space')}
{space_summary}

{headers.get('analysis_req', '# Analysis Requirements')}
{prompts.get('analysis_requirements', '')}

{prompts.get('output_format', '')}"""
        else:
            # 使用內建預設 (向後相容)
            prompt = PromptTemplates._get_default_analyzer_prompt(
                trials, pareto_frontier, search_space,
                trials_summary, pareto_summary, space_summary
            )

        return prompt

    @staticmethod
    def _get_default_analyzer_prompt(trials, pareto_frontier, search_space,
                                     trials_summary, pareto_summary, space_summary) -> str:
        """內建預設的 Analyzer prompt（向後相容）"""
        return f"""You are an expert analyzer for quantization optimization trials.

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
        PromptTemplates._get_config()
        prompts = PromptConfigLoader.get_prompts('planner')

        progress_pct = (trial_num / budget['max']) * 100 if budget['max'] > 0 else 0

        if prompts:
            labels = prompts.get('labels', {})
            headers = prompts.get('section_headers', {})

            prompt = f"""{prompts.get('system_role', '')}

{headers.get('current_state', '# Current State')}
{labels.get('trial', 'Trial')}: {trial_num} / {budget['max']}
{labels.get('progress', 'Progress')}: {progress_pct:.1f}%
{labels.get('budget_remaining', 'Budget remaining')}: {budget['max'] - trial_num} {labels.get('trials', 'trials')}

{headers.get('analysis_from_analyzer', '# Analysis from AnalyzerAgent')}
{PromptTemplates._format_json(analysis)}

{headers.get('optimization_targets', '# Optimization Targets')}
{PromptTemplates._format_json(targets)}

{headers.get('pareto_frontier', '# Current Pareto Frontier')}
{len(pareto)} {labels.get('solutions_on_frontier', 'solutions on frontier')}
{PromptTemplates._format_pareto_summary(pareto)}

{headers.get('your_task', '# Your Task')}
{labels.get('decide_next_config', 'Decide the next trial configuration based on the analysis.')}

{prompts.get('strategy_guide', '')}

{prompts.get('progress_recommendation', '')}

{prompts.get('output_format', '')}"""
        else:
            prompt = PromptTemplates._get_default_planner_prompt(
                analysis, trial_num, budget, targets, pareto, progress_pct
            )

        return prompt

    @staticmethod
    def _get_default_planner_prompt(analysis, trial_num, budget, targets, pareto, progress_pct) -> str:
        """內建預設的 Planner prompt（向後相容）"""
        return f"""You are a strategic planner for quantization optimization.

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
        PromptTemplates._get_config()
        prompts = PromptConfigLoader.get_prompts('monitor')

        if prompts:
            labels = prompts.get('labels', {})
            headers = prompts.get('section_headers', {})

            total = len(trials)
            successful = sum(1 for t in trials if t.get('success', False))
            pareto_count = len(pareto)
            satisfying = sum(1 for t in pareto if t.get('satisfies_targets', False))
            progress_pct = (budget_status['used'] / budget_status['max'] * 100) if budget_status['max'] > 0 else 0

            prompt = f"""{prompts.get('system_role', '')}

{headers.get('progress_summary', '# Progress Summary')}
{labels.get('total_trials', 'Total trials')}: {total}
{labels.get('successful_trials', 'Successful trials')}: {successful}
{labels.get('pareto_solutions', 'Pareto solutions')}: {pareto_count}
{labels.get('satisfying_targets', 'Satisfying targets')}: {satisfying}

{headers.get('budget_status', '# Budget Status')}
{labels.get('trials_used', 'Trials used')}: {budget_status['used']} / {budget_status['max']}
{labels.get('progress', 'Progress')}: {progress_pct:.1f}%

{headers.get('targets_vs_best', '# Targets vs Current Best')}
{PromptTemplates._format_target_comparison(targets, pareto, prompts)}

{headers.get('your_task', '# Your Task')}
{labels.get('evaluate_task', 'Evaluate whether optimization should continue or stop.')}

{prompts.get('stopping_criteria', '')}

{prompts.get('output_format', '')}"""
        else:
            prompt = PromptTemplates._get_default_monitor_prompt(
                trials, pareto, budget_status, targets, pareto_history
            )

        return prompt

    @staticmethod
    def _get_default_monitor_prompt(trials, pareto, budget_status, targets, pareto_history) -> str:
        """內建預設的 Monitor prompt（向後相容）"""
        return f"""You are monitoring the optimization progress.

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

    # ========== 輔助格式化方法 ==========

    @staticmethod
    def _format_trials_summary(trials: List[Dict]) -> str:
        """格式化試驗摘要"""
        PromptTemplates._get_config()
        prompts = PromptConfigLoader.get_prompts('analyzer')
        formatting = PromptConfigLoader.get_formatting()
        labels = prompts.get('labels', {}) if prompts else {}

        if not trials:
            return labels.get('no_trials', "No trials yet.")

        summary_lines = []

        # 最近5個試驗
        recent = trials[-5:]
        summary_lines.append(f"{labels.get('recent_trials', 'Recent trials')}:")
        for i, t in enumerate(recent, 1):
            status = "✓" if t.get('success', False) else "✗"
            config = t.get('config', {})
            method = config.get('method', 'unknown')
            bits = config.get('bits', '?')
            objectives = t.get('objectives', {})
            acc = objectives.get('accuracy_change', 0)
            gpu = objectives.get('gpu_peak_change', 0)
            lat = objectives.get('latency_change', 0)

            trial_num = len(trials) - 5 + i
            trial_label = formatting.get('trial_format', 'Trial {num}').format(num=trial_num)

            summary_lines.append(
                f"  {status} {trial_label}: {method}-{bits}bit "
                f"(acc:{acc:+.2%}, gpu:{gpu:+.2%}, lat:{lat:+.2%})"
            )

        return "\n".join(summary_lines)

    @staticmethod
    def _format_pareto_summary(pareto: List[Dict]) -> str:
        """格式化Pareto前沿摘要"""
        PromptTemplates._get_config()
        prompts = PromptConfigLoader.get_prompts('analyzer')
        formatting = PromptConfigLoader.get_formatting()
        labels = prompts.get('labels', {}) if prompts else {}

        if not pareto:
            return labels.get('no_pareto', "No Pareto solutions yet.")

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
            more_text = formatting.get('and_more', '... and {count} more').format(count=len(pareto) - 5)
            summary_lines.append(f"  {more_text}")

        return "\n".join(summary_lines)

    @staticmethod
    def _format_search_space(space: Dict) -> str:
        """格式化搜索空間"""
        PromptTemplates._get_config()
        formatting = PromptConfigLoader.get_formatting()

        lines = []
        methods_label = formatting.get('methods', 'Methods') if formatting else 'Methods'
        lines.append(f"{methods_label}: {space.get('methods', [])}")

        for method in space.get('methods', []):
            if method in space:
                lines.append(f"\n{method.upper()}:")
                for param, values in space[method].items():
                    lines.append(f"  {param}: {values}")

        return "\n".join(lines)

    @staticmethod
    def _format_json(obj: Any) -> str:
        """格式化JSON對象為字符串"""
        return json.dumps(obj, indent=2, ensure_ascii=False)

    @staticmethod
    def _format_target_comparison(targets: Dict, pareto: List[Dict],
                                   prompts: Optional[Dict] = None) -> str:
        """格式化目標與實際對比"""
        labels = prompts.get('labels', {}) if prompts else {}

        if not pareto:
            return labels.get('no_solutions', "No solutions to compare.")

        lines = []
        lines.append(f"{labels.get('targets', 'Targets')}:")
        lines.append(f"  {labels.get('accuracy_min', 'accuracy_min')}: {targets.get('accuracy_min', 0):+.2%}")
        lines.append(f"  {labels.get('gpu_peak_max', 'gpu_peak_max')}: {targets.get('gpu_peak_max', 0):+.2%}")
        lines.append(f"  {labels.get('latency_max', 'latency_max')}: {targets.get('latency_max', 0):+.2%}")

        # 找最佳解
        best = pareto[0] if pareto else None
        if best:
            obj = best.get('objectives', {})
            lines.append(f"\n{labels.get('best_solution', 'Best solution')}:")
            lines.append(f"  {labels.get('accuracy', 'accuracy')}: {obj.get('accuracy_change', 0):+.2%}")
            lines.append(f"  {labels.get('gpu_peak', 'gpu_peak')}: {obj.get('gpu_peak_change', 0):+.2%}")
            lines.append(f"  {labels.get('latency', 'latency')}: {obj.get('latency_change', 0):+.2%}")
            lines.append(f"  {labels.get('satisfies_targets', 'Satisfies targets')}: {best.get('satisfies_targets', False)}")

        return "\n".join(lines)


# ========== 便利函數 ==========

def init_prompts(config_path: Optional[str] = None, prompt_type: Optional[str] = None):
    """
    初始化 prompt 配置

    Args:
        config_path: YAML 配置檔路徑
        prompt_type: 使用的 prompt 類型
    """
    PromptConfigLoader.load(config_path, prompt_type)


def get_prompt_info() -> Dict[str, Any]:
    """
    取得完整的 prompt 資訊，用於輸出到結果

    Returns:
        包含類型、路徑、metadata 和所有 prompts 的字典
    """
    return PromptConfigLoader.get_full_prompt_info()


if __name__ == "__main__":
    # 測試prompt生成
    import sys

    # 測試載入配置
    print("=" * 60)
    print("Testing Prompt Config Loading")
    print("=" * 60)

    # 測試英文
    init_prompts(prompt_type="en")
    print(f"Current type: {PromptConfigLoader.get_current_type()}")
    print(f"Metadata: {PromptConfigLoader.get_metadata()}")

    # 測試Analyzer prompt
    test_trials = [
        {
            'success': True,
            'config': {'method': 'gptq', 'bits': 4},
            'objectives': {'accuracy_change': -0.05, 'gpu_peak_change': -0.30,
                          'latency_change': 0.10}
        }
    ]

    analyzer_prompt = PromptTemplates.get_analyzer_prompt(
        trials=test_trials,
        pareto_frontier=[],
        search_space={'methods': ['gptq', 'awq']}
    )

    print("\n" + "=" * 60)
    print("ANALYZER PROMPT (English)")
    print("=" * 60)
    print(analyzer_prompt[:800] + "...")

    # 測試中文
    print("\n" + "=" * 60)
    print("Testing Chinese Prompts")
    print("=" * 60)

    init_prompts(prompt_type="zh")
    print(f"Current type: {PromptConfigLoader.get_current_type()}")
    print(f"Metadata: {PromptConfigLoader.get_metadata()}")

    analyzer_prompt_zh = PromptTemplates.get_analyzer_prompt(
        trials=test_trials,
        pareto_frontier=[],
        search_space={'methods': ['gptq', 'awq']}
    )

    print("\n" + "=" * 60)
    print("ANALYZER PROMPT (中文)")
    print("=" * 60)
    print(analyzer_prompt_zh[:800] + "...")

    # 測試取得完整資訊
    print("\n" + "=" * 60)
    print("Full Prompt Info")
    print("=" * 60)
    info = get_prompt_info()
    print(f"Type: {info['prompt_type']}")
    print(f"Config path: {info['config_path']}")
    print(f"Supported types: {info['supported_types']}")
