"""
Prompt Templates for LLM Agents

支援從 YAML 配置檔載入多種類型的 prompt 模板。
"""

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
            agent: agent 名稱 ('analyzer', 'planner')

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
    def get_full_prompt_info(cls, agent_mode: str = "separate") -> Dict[str, Any]:
        """
        取得完整的 prompt 資訊，用於輸出到結果

        Args:
            agent_mode: "separate" 或 "combined"，決定輸出哪些 agent prompts

        Returns:
            包含類型、路徑、metadata 和實際使用的 prompts 的字典
        """
        if agent_mode == "combined":
            agent_prompts = {
                "strategist": cls.get_prompts('strategist'),
            }
        else:
            agent_prompts = {
                "analyzer": cls.get_prompts('analyzer'),
                "planner": cls.get_prompts('planner'),
            }
        agent_prompts["formatting"] = cls.get_formatting()

        return {
            "agent_mode": agent_mode,
            "prompt_type": cls._current_type,
            "config_path": cls._config_path,
            "metadata": cls.get_metadata(),
            "supported_types": cls._config.get('supported_types', ['en']) if cls._config else ['en'],
            "prompts": agent_prompts
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
                trials, pareto_frontier,
                trials_summary, pareto_summary, space_summary
            )

        return prompt

    @staticmethod
    def _get_default_analyzer_prompt(trials, pareto_frontier,
                                     trials_summary, pareto_summary, space_summary) -> str:
        """內建預設的 Analyzer prompt（向後相容）"""
        return f"""You are an expert analyzer for quantization optimization trials. Think step by step.

# Your Task
Analyze the historical trials systematically to understand what works and what doesn't.

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

# Chain of Thought Analysis Process

## Step 1: Method Performance Analysis
For each quantization method (gptq/awq/bnb), analyze:
- How many trials? How many succeeded/failed?
- Average accuracy_change, gpu_peak_change, latency_change
- Which method performs best for each objective?

## Step 2: Failure Pattern Analysis
Look at failed trials and identify:
- What parameter combinations caused failures?
- Are there common patterns (e.g., bits=2 always fails)?

## Step 3: Success Pattern Analysis
Look at successful trials (especially Pareto solutions):
- What parameter combinations work well?
- What are the trade-offs?

## Step 4: Parameter Sensitivity Analysis
- bits: How does 2/3/4/8 bit affect results?
- group_size: Does smaller/larger help?
- Other notable observations?

## Step 5: Strategic Recommendations
Based on the above, what should be tried next?

# Output Format
Output a JSON object with your complete analysis:
{{
  "method_analysis": {{
    "gptq": {{"trials": N, "success_rate": X, "avg_accuracy": Y, "avg_gpu": Z, "observations": "..."}},
    "awq": {{"trials": N, "success_rate": X, "avg_accuracy": Y, "avg_gpu": Z, "observations": "..."}},
    "bnb": {{"trials": N, "success_rate": X, "avg_accuracy": Y, "avg_gpu": Z, "observations": "..."}}
  }},
  "failure_patterns": [
    {{"pattern": "description", "affected_params": {{}}, "suggestion": "avoid/adjust"}}
  ],
  "success_patterns": [
    {{"pattern": "description", "winning_params": {{}}, "trade_off": "..."}}
  ],
  "parameter_insights": {{
    "bits": "observation",
    "group_size": "observation",
    "other": "observation"
  }},
  "unexplored_regions": [
    {{"method": "...", "params": {{}}, "rationale": "why promising"}}
  ],
  "recommendations": ["advice 1", "advice 2"]
}}

Output ONLY the JSON object (no other text).
"""

    @staticmethod
    def get_planner_prompt(analysis: Dict, trial_num: int, budget: Dict,
                          targets: Dict, pareto: List[Dict], search_space: Dict,
                          trials: List[Dict] = None) -> str:
        """
        PlannerAgent的prompt模板

        Args:
            analysis: AnalyzerAgent的分析報告
            trial_num: 當前試驗編號
            budget: 預算狀態 (used, max)
            targets: 優化目標
            pareto: Pareto前沿
            search_space: 搜索空間定義
            trials: 已嘗試的配置列表（用於避免重複）

        Returns:
            完整的規劃prompt
        """
        PromptTemplates._get_config()
        prompts = PromptConfigLoader.get_prompts('planner')

        progress_pct = (trial_num / budget['max']) * 100 if budget['max'] > 0 else 0
        space_summary = PromptTemplates._format_search_space(search_space)
        tried_configs = PromptTemplates._format_tried_configs(trials or [])
        targets_with_gap = PromptTemplates._format_targets_with_gap(targets, pareto)

        if prompts:
            labels = prompts.get('labels', {})
            headers = prompts.get('section_headers', {})

            prompt = f"""{prompts.get('system_role', '')}

{headers.get('current_state', '# Current State')}
{labels.get('trial', 'Trial')}: {trial_num} / {budget['max']}
{labels.get('progress', 'Progress')}: {progress_pct:.1f}%
{labels.get('budget_remaining', 'Budget remaining')}: {budget['max'] - trial_num} {labels.get('trials', 'trials')}

{headers.get('search_space', '# Search Space (Available Parameters)')}
{space_summary}

{headers.get('already_tried', '# Already Tried Configurations (DO NOT REPEAT)')}
{tried_configs}

{headers.get('optimization_targets', '# Optimization Targets (Your Goals)')}
{targets_with_gap}

{headers.get('analysis_from_analyzer', '# Analysis from AnalyzerAgent')}
{PromptTemplates._format_json(analysis)}

{headers.get('pareto_frontier', '# Current Pareto Frontier')}
{len(pareto)} {labels.get('solutions_on_frontier', 'solutions on frontier')}
{PromptTemplates._format_pareto_summary(pareto)}

{headers.get('your_task', '# Your Task')}
{labels.get('decide_next_config', 'Decide the next trial configuration based on the analysis.')}

{prompts.get('decision_process', '')}

{prompts.get('strategy_guide', '')}

{prompts.get('output_format', '')}"""
        else:
            prompt = PromptTemplates._get_default_planner_prompt(
                analysis, trial_num, budget, targets, pareto, search_space, trials, progress_pct
            )

        return prompt

    @staticmethod
    def _get_default_planner_prompt(analysis, trial_num, budget, targets, pareto, search_space, trials, progress_pct) -> str:
        """內建預設的 Planner prompt（向後相容）"""
        space_summary = PromptTemplates._format_search_space(search_space)
        tried_configs = PromptTemplates._format_tried_configs(trials or [])
        targets_with_gap = PromptTemplates._format_targets_with_gap(targets, pareto)

        return f"""You are a strategic planner for quantization optimization. Think step by step before making decisions.

# Current State
Trial: {trial_num} / {budget['max']}
Progress: {progress_pct:.1f}%
Budget remaining: {budget['max'] - trial_num} trials

# Search Space (Available Parameters)
{space_summary}

# Already Tried Configurations (DO NOT REPEAT)
{tried_configs}

# Optimization Targets (Your Goals)
{targets_with_gap}

# Analysis from AnalyzerAgent
{PromptTemplates._format_json(analysis)}

# Current Pareto Frontier
{len(pareto)} solutions on frontier
{PromptTemplates._format_pareto_summary(pareto)}

# Your Task
Decide the next trial configuration based on the analysis.
IMPORTANT:
- Only use parameter values from the Search Space above!
- DO NOT repeat any configuration from "Already Tried" list!

# Decision Process (Chain of Thought)
Follow these steps:

## Step 1: Review Analyzer's Findings
- What methods performed best/worst?
- What failure patterns should we avoid?
- What unexplored regions are promising?

## Step 2: Consider User Targets
- Check the "Optimization Targets" section above
- Which gaps need to be closed?

## Step 3: Avoid Tried Configurations
- Check the "Already Tried" list above
- Ensure your suggestion is NEW

## Step 4: Choose Strategy
- Early phase (< 20%): Explore diverse methods/params
- Mid phase (20-70%): Balance exploration and exploitation
- Late phase (> 70%): Focus on refining best configs

## Step 5: Design Next Config
- Pick method based on analysis insights
- Choose parameters that are promising but not yet tried

# Strategy Reference
- **EXPLORATION**: Try new methods/params, prioritize unexplored_regions
- **EXPLOITATION**: Refine successful configs, small parameter adjustments
- **BALANCED**: Mix of both

# Output Requirements
Provide your decision in JSON format:

{{
  "thinking": {{
    "analyzer_insights": "What I learned from the analysis (1-2 sentences)",
    "target_gap": "How far are we from meeting targets? (1 sentence)",
    "avoid_repeating": "Which tried configs am I avoiding? (list methods/bits)",
    "chosen_strategy_reason": "Why I chose this strategy (1 sentence)"
  }},
  "strategy": "exploration" / "exploitation" / "balanced",
  "next_config": {{
    "method": "gptq" / "awq" / "bnb",
    "bits": ...,
    "group_size": ...,
    ... (all relevant parameters from Search Space)
  }},
  "rationale": "Final explanation of this config choice (2-3 sentences)",
  "confidence": 0.0-1.0
}}

Output ONLY the JSON object (no other text).
"""

    # ========== Strategist Agent (Combined Mode) ==========

    @staticmethod
    def get_strategist_prompt(trials: List[Dict], pareto_frontier: List[Dict],
                              search_space: Dict, trial_num: int, budget: Dict,
                              targets: Dict) -> str:
        """
        StrategistAgent 的合併 prompt（分析 + 決策在一起）

        Args:
            trials: 歷史試驗列表
            pareto_frontier: 當前 Pareto 前沿
            search_space: 搜索空間定義
            trial_num: 當前試驗編號
            budget: 預算狀態 (used, max)
            targets: 優化目標

        Returns:
            完整的策略師 prompt
        """
        PromptTemplates._get_config()
        prompts = PromptConfigLoader.get_prompts('strategist')

        progress_pct = (trial_num / budget['max']) * 100 if budget['max'] > 0 else 0
        trials_summary = PromptTemplates._format_trials_summary(trials)
        pareto_summary = PromptTemplates._format_pareto_summary(pareto_frontier)
        space_summary = PromptTemplates._format_search_space(search_space)
        tried_configs = PromptTemplates._format_tried_configs(trials)
        targets_with_gap = PromptTemplates._format_targets_with_gap(targets, pareto_frontier)

        if prompts:
            labels = prompts.get('labels', {})
            headers = prompts.get('section_headers', {})

            prompt = f"""{prompts.get('system_role', '')}

{headers.get('current_state', '# Current State')}
{labels.get('trial', 'Trial')}: {trial_num} / {budget['max']}
{labels.get('progress', 'Progress')}: {progress_pct:.1f}%
{labels.get('budget_remaining', 'Budget remaining')}: {budget['max'] - trial_num} {labels.get('trials', 'trials')}

{headers.get('search_space', '# Search Space (Available Parameters)')}
{space_summary}

{headers.get('already_tried', '# Already Tried Configurations (DO NOT REPEAT)')}
{tried_configs}

{headers.get('optimization_targets', '# Optimization Targets (Your Goals)')}
{targets_with_gap}

{headers.get('historical_trials', '# Historical Trials Summary')}
{trials_summary}

{headers.get('pareto_frontier', '# Current Pareto Frontier')}
{len(pareto_frontier)} {labels.get('solutions_on_frontier', 'solutions on frontier')}
{pareto_summary}

{headers.get('your_task', '# Your Task')}
{prompts.get('task_description', '')}

{prompts.get('analysis_steps', '')}

{prompts.get('decision_steps', '')}

{prompts.get('output_format', '')}"""
        else:
            prompt = PromptTemplates._get_default_strategist_prompt(
                trials, pareto_frontier, search_space, trial_num, budget, targets, progress_pct,
                trials_summary, pareto_summary, space_summary, tried_configs, targets_with_gap
            )

        return prompt

    @staticmethod
    def _get_default_strategist_prompt(trials, pareto_frontier, search_space, trial_num, budget, targets,
                                       progress_pct, trials_summary, pareto_summary, space_summary,
                                       tried_configs, targets_with_gap) -> str:
        """內建預設的 Strategist prompt"""
        return f"""You are an expert strategist for quantization optimization.
Your job is to ANALYZE the historical trials AND DECIDE the next configuration in ONE response.
Think step by step.

# Current State
Trial: {trial_num} / {budget['max']}
Progress: {progress_pct:.1f}%
Budget remaining: {budget['max'] - trial_num} trials

# Search Space (Available Parameters)
{space_summary}

# Already Tried Configurations (DO NOT REPEAT)
{tried_configs}

# Optimization Targets (Your Goals)
{targets_with_gap}

# Historical Trials Summary
{trials_summary}

# Current Pareto Frontier
{len(pareto_frontier)} solutions on frontier
{pareto_summary}

# Your Task
Complete BOTH analysis and decision in one response.

## PART 1: ANALYSIS (Chain of Thought)

### Step 1: Method Performance
For each method (gptq/awq/bnb):
- How many trials? Success rate?
- Average accuracy_change and gpu_peak_change?
- Which method is best for each objective?

### Step 2: Failure Patterns
- What parameter combinations caused failures?
- Any common patterns to avoid?

### Step 3: Success Patterns
- What works well? (especially Pareto solutions)
- Trade-offs observed?

### Step 4: Parameter Insights
- How does bits (2/3/4/8) affect results?
- How does group_size affect results?

## PART 2: DECISION (Based on Analysis)

### Step 5: Consider Targets
- Check the gap from targets above
- Which objective needs most improvement?

### Step 6: Avoid Repetition
- Check "Already Tried" list
- Ensure new config is DIFFERENT

### Step 7: Choose Strategy
- Early (< 20%): EXPLORATION
- Mid (20-70%): BALANCED
- Late (> 70%): EXPLOITATION

### Step 8: Design Next Config
- Pick method based on insights
- Choose promising but untried params

# Output Format
Provide your complete analysis and decision in ONE JSON object:

{{
  "analysis": {{
    "method_analysis": {{
      "gptq": {{"trials": N, "success_rate": X, "avg_accuracy": Y, "avg_gpu": Z, "observations": "..."}},
      "awq": {{"trials": N, "success_rate": X, "avg_accuracy": Y, "avg_gpu": Z, "observations": "..."}},
      "bnb": {{"trials": N, "success_rate": X, "avg_accuracy": Y, "avg_gpu": Z, "observations": "..."}}
    }},
    "failure_patterns": [
      {{"pattern": "description", "affected_params": {{}}, "suggestion": "avoid/adjust"}}
    ],
    "success_patterns": [
      {{"pattern": "description", "winning_params": {{}}, "trade_off": "..."}}
    ],
    "parameter_insights": {{
      "bits": "observation",
      "group_size": "observation",
      "other": "observation"
    }},
    "unexplored_regions": [
      {{"method": "...", "params": {{}}, "rationale": "why promising"}}
    ],
    "recommendations": ["advice 1", "advice 2"]
  }},
  "decision": {{
    "thinking": {{
      "analyzer_insights": "Key finding from analysis (1-2 sentences)",
      "target_gap": "Gap from targets (1 sentence)",
      "avoid_repeating": "Configs I'm avoiding (list)",
      "chosen_strategy_reason": "Why this strategy (1 sentence)"
    }},
    "strategy": "exploration" / "exploitation" / "balanced",
    "next_config": {{
      "method": "gptq" / "awq" / "bnb",
      "bits": ...,
      "group_size": ...,
      ... (all params from Search Space)
    }},
    "rationale": "Final explanation (2-3 sentences)",
    "confidence": 0.0-1.0
  }}
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
    def _format_tried_configs(trials: List[Dict]) -> str:
        """
        格式化已嘗試的配置列表（用於避免重複）

        Args:
            trials: 所有試驗列表

        Returns:
            格式化的已嘗試配置摘要
        """
        if not trials:
            return "No configurations tried yet. You have full freedom to explore!"

        # 按方法分組
        by_method = {}
        for t in trials:
            config = t.get('config', {})
            method = config.get('method', 'unknown')
            bits = config.get('bits', '?')
            group_size = config.get('group_size', '?')
            success = t.get('success', False)

            if method not in by_method:
                by_method[method] = []

            status = "✓" if success else "✗"
            by_method[method].append(f"{bits}bit/gs{group_size} {status}")

        lines = []
        for method, configs in by_method.items():
            # 只顯示前 5 個，避免 prompt 過長
            display_configs = configs[:5]
            if len(configs) > 5:
                display_configs.append(f"... and {len(configs) - 5} more")
            lines.append(f"{method.upper()}: {', '.join(display_configs)}")

        lines.append("")
        lines.append("⚠️ DO NOT suggest any of the above configurations!")

        return "\n".join(lines)

    @staticmethod
    def _format_targets_with_gap(targets: Dict, pareto: List[Dict]) -> str:
        """
        格式化目標並顯示與當前最佳解的差距

        Args:
            targets: 用戶目標
            pareto: Pareto 前沿

        Returns:
            目標和差距的格式化字串
        """
        lines = []

        # 顯示目標
        acc_min = targets.get('accuracy_min', 0)
        gpu_max = targets.get('gpu_peak_max', 0)
        lat_max = targets.get('latency_max', 0)

        lines.append("Your targets:")
        lines.append(f"  - accuracy_change ≥ {acc_min:+.1%} (higher is better)")
        lines.append(f"  - gpu_peak_change ≤ {gpu_max:+.1%} (lower is better)")
        lines.append(f"  - latency_change ≤ {lat_max:+.1%} (lower is better)")

        # 如果有 Pareto 解，顯示最佳解與目標的差距
        if pareto:
            best = pareto[0]
            obj = best.get('objectives', {})
            best_acc = obj.get('accuracy_change', 0)
            best_gpu = obj.get('gpu_peak_change', 0)
            best_lat = obj.get('latency_change', 0)

            lines.append("")
            lines.append("Current best vs targets:")

            # 計算差距
            acc_status = "✓ MET" if best_acc >= acc_min else f"✗ GAP: {acc_min - best_acc:+.1%}"
            gpu_status = "✓ MET" if best_gpu <= gpu_max else f"✗ GAP: {best_gpu - gpu_max:+.1%}"
            lat_status = "✓ MET" if best_lat <= lat_max else f"✗ GAP: {best_lat - lat_max:+.1%}"

            lines.append(f"  - accuracy: {best_acc:+.2%} {acc_status}")
            lines.append(f"  - gpu_peak: {best_gpu:+.2%} {gpu_status}")
            lines.append(f"  - latency: {best_lat:+.2%} {lat_status}")

            if best.get('satisfies_targets', False):
                lines.append("")
                lines.append("✓ All targets satisfied! Focus on finding more diverse solutions.")
            else:
                lines.append("")
                lines.append("✗ Targets not yet met. Prioritize configs that can close the gap.")
        else:
            lines.append("")
            lines.append("No solutions yet. Start exploring!")

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


def get_prompt_info(agent_mode: str = "separate") -> Dict[str, Any]:
    """
    取得完整的 prompt 資訊，用於輸出到結果

    Args:
        agent_mode: "separate" 或 "combined"

    Returns:
        包含類型、路徑、metadata 和實際使用的 prompts 的字典
    """
    return PromptConfigLoader.get_full_prompt_info(agent_mode)


if __name__ == "__main__":
    # 測試prompt生成

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
