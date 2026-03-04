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

    def _create_prompt(self, iteration: int, trial_history: list) -> str:
        history_str = json.dumps(trial_history[-3:], indent=2) if trial_history else "No trials yet."
        return f"""You are a multi-objective optimization agent for LLM compression.
Task: {self.task} on {self.model_id}.
Objective: Minimize VRAM and Latency while maximizing Accuracy.

### Available Strategies:
1. **ASVD (Structural)**: Best for improving Latency (Inference Speed).
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
Trial History: {history_str}

### Your Task:
Suggest the next strategy. If Accuracy dropped in the last trial, be more conservative (increase ratio, bits, or group_size). 
If Accuracy is stable, try to push for higher compression.

[Output ONLY JSON] [no prose]

Return ONLY JSON:
{{
  "reasoning": "4-6 sentences analysis...",
  "mode": "asvd_only" | "quant_only" | "hybrid",
  "alpha": 0.5,
  "param_ratio_target": 0.9,
  "scaling_method": "fisher",
  "quant_method": "gptq" | "awq" | "bnb" | "none",
  "quant_bits": 4,
  "quant_group_size": 128,
  "quant_type": "nf4" | "fp4",
  "desc_act": false
}}
"""

    def get_suggestion(self, iteration: int, trial_history: list) -> StrategySuggestion:
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": self._create_prompt(iteration, trial_history)}],
            response_format={ "type": "json_object" }
        )
        return StrategySuggestion.model_validate_json(response.choices[0].message.content)
