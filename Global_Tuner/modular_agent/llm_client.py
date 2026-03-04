import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from schemas import StrategySuggestion

# Load environment variables
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

class LLMDecisionMaker:
    def __init__(self, model_id: str, task: str, max_iterations: int):
        self.model_id = model_id
        self.task = task
        self.max_iterations = max_iterations
        self.client = OpenAI(api_key=os.getenv("LLM_API_KEY"))
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o")

    # def _format_history(self, history: list) -> str:
    #     if not history:
    #         return "No trials yet."
        
    #     lines = []
    #     for trial in history[-5:]: # Show last 5 trials
    #         it = trial.get('iteration')
    #         config = trial.get('config', {})
    #         metrics = trial.get('metrics', {})
    #         details = metrics.get('details', {})
            
    #         line = f"Trial {it}: Mode={config.get('mode')}, Score={metrics.get('score', 0):.4f}\n"
    #         line += f"  - Avg Acc: {metrics.get('accuracy', 0):.4f}, Avg Lat: {metrics.get('latency', 0):.4f}\n"
            
    #         if details:
    #             line += "  - Multi-Task Details:\n"
    #             for task_name, task_metrics in details.items():
    #                 line += f"    * {task_name}: Acc={task_metrics.get('accuracy', 0):.4f}\n"
            
    #         lines.append(line)
    #     return "\n".join(lines)

    def _format_history(self, history: list) -> str:
        if not history: return "None"
        
        lines = []
        for trial in history:
            config = trial.get('config', {})
            metrics = trial.get('metrics', {})
            details = metrics.get('details', {})
            
            # 1. 基本分數與表現
            line = f"- [Iter {trial.get('iteration')}]: Score={metrics.get('score', 0):.4f}"
            line += f" (Acc: {metrics.get('accuracy', 0):.4f}, Lat: {metrics.get('latency', 0):.4f}s, VRAM: {metrics.get('vram', 0):.4f}GB)"
            
            # 2. 多任務細節 (壓縮成一行，節省 Token)
            if details:
                task_accs = [f"{k}={v.get('accuracy',0):.4f}" for k, v in details.items()]
                line += f" | Tasks: [{', '.join(task_accs)}]"
            
            line += f"\n    Config: {json.dumps(config)}"
                
            lines.append(line)
        return "\n".join(lines)

    def _create_prompt(self, iteration: int, trial_history: list, pareto: list = None) -> str:
        recent_history = trial_history[-5:] if len(trial_history) > 5 else trial_history
        history_str = self._format_history(recent_history)
        pareto_str = self._format_history(pareto) if pareto else "None"
        
        return f"""You are a multi-objective optimization agent for LLM compression.
Task: {self.task} on {self.model_id}.
Objective: Minimize VRAM and Latency while maximizing Accuracy across all tasks.

### UNDERSTANDING LATENCY:
- **Unit**: All 'latency' values are in seconds (sec).
- **Goal**: MINIMIZE this value. Smaller is better.
- **Score Impact**: Your final 'score' is calculated as `(acc_weight * avg_acc) + (lat_weight * (1.0 / (avg_lat + 1e-6))) + (vram_weight * (1.0 / (max_vram + 1e-6)))`. 
- **Inference Speed**: High latency (e.g., > 10s) indicates a very slow model, which significantly penalizes the score even if Accuracy is high.
- **Strategy**: If Latency is the bottleneck, prioritize 'asvd_only' or 'hybrid' modes to reduce the structural rank of the model.

### Available Strategies:
1. **ASVD (Structural)**: Best for improving Latency (Inference Speed). Note: ASVD significantly improves Latency through low-rank decomposition, but excessive compression (ratio < 0.8) often leads to a sharp decline in Accuracy.
2. **Quantization (GPTQ/AWQ/BNB)**: Best for reducing VRAM (Memory).
   - GPTQ: bits, group_size, desc_act.
   - AWQ: bits, group_size.
   - BNB: bits (4/8), quant_type (nf4/fp4).
3. **Hybrid**: Apply ASVD first, then Quantization bnb. High risk, high reward.

### ⚠️ CRITICAL CONSTRAINTS:
1. **ASVD + Quantization (Hybrid)**: This mode ONLY supports 'bnb'. Using ASVD with 'gptq' or 'awq' will FAIL.
2. **GPTQ/AWQ**: These are standalone quantization methods. Do NOT use them in 'hybrid' mode.
3. **Target**: Keep ASVD ratio >= 0.90 in hybrid mode to prevent logical collapse.
4. **GPTQ rule**: `desc_act` should be True for better accuracy, but some kernels require False.

### ⚠️ PARAMETER LIMITS:
1. **ASVD**:
   - alpha: 0.3 to 0.7 (Higher = more protection for logic).
   - param_ratio_target: 0.70 to 0.99 (Lower = faster but riskier).
   - scaling_method: [abs_mean, abs_max, fisher].

2. **Quantization**:
   - bits: [4, 8].
   - group_size: [32, 64, 128] (Lower = higher accuracy, higher VRAM).
   - quant_type: [nf4, fp4] (nf4 is recommended for GSM8K).
   - use_double_quant: true/false (Set true to save more VRAM).

### Current Status:
Iteration: {iteration}/{self.max_iterations}

### Trial History (Last 5):
{history_str}

### Pareto Frontier (Best Trade-offs found so far):
{pareto_str}

### Your Task:
Suggest the next strategy based on the multi-task results and the Pareto frontier.
- If a specific task's Accuracy dropped significantly, analyze the configuration used.
- Pareto Frontier represents the best balance between Accuracy and Latency. Try to explore configurations that could expand this frontier.
- If Accuracy is stable across all tasks, try to push for higher compression (lower ratio or bits).

[Output ONLY JSON] [no prose]

Return ONLY JSON:
{{
  "reasoning": "4-6 sentences analysis of task-specific performance and pareto status...",
  "mode": "asvd_only" | "quant_only" | "hybrid",
  "alpha": 0.5,
  "param_ratio_target": 0.9,
  "scaling_method": "fisher",
  "quant_method": "gptq" | "awq" | "bnb" | "none",
  "quant_bits": 4,
  "quant_group_size": 128,
  "quant_type": "nf4" | "fp4",
  "desc_act": false,
  "use_double_quant": true
}}
"""

    def get_suggestion(self, iteration: int, trial_history: list, pareto: list = None) -> StrategySuggestion:
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": self._create_prompt(iteration, trial_history, pareto=pareto)}],
            response_format={ "type": "json_object" }
        )
        return StrategySuggestion.model_validate_json(response.choices[0].message.content)
