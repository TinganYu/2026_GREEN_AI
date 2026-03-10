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

    def _format_history(self, history: list) -> str:
        if not history:
            return "None"

        lines = []
        for trial in history:
            config = trial.get('config', {})
            metrics = trial.get('metrics', {})
            details = metrics.get('details', {})

            line = f"- [Iter {trial.get('iteration')}]: Score={metrics.get('score', 0):.4f}"
            line += (f" (Acc: {metrics.get('accuracy', 0):.4f},"
                     f" Lat: {metrics.get('latency', 0):.4f}s,"
                     f" VRAM: {metrics.get('vram', 0):.4f}GB,"
                     f" Emit: {metrics.get('emissions', 0):.6f}kg CO2)")

            if details:
                task_accs = [f"{k}={v.get('accuracy', 0):.4f}" for k, v in details.items()]
                line += f" | Tasks: [{', '.join(task_accs)}]"

            line += f"\n    Config: {json.dumps(config)}"
            lines.append(line)
        return "\n".join(lines)

    def _create_prompt(self, iteration: int, trial_history: list, pareto: list = None,
                       weights: dict = None) -> str:
        # recent_history = trial_history[-5:] if len(trial_history) > 5 else trial_history
        history_str = self._format_history(trial_history)
        pareto_str = self._format_history(pareto) if pareto else "None"

        w = weights or {}
        w_acc  = w.get("acc",  0.5)
        w_lat  = w.get("lat",  0.1)
        w_vram = w.get("vram", 0.2)
        w_emit = w.get("emit", 0.2)

        return f"""You are an LLM compression optimization agent. Choose the best compression strategy for:
Model: {self.model_id} | Task: {self.task} | Iteration: {iteration}/{self.max_iterations}

GOAL: Maximize score = {w_acc}*(Acc/Base_acc) + {w_lat}*(Base_lat/Lat) + {w_vram}*(Base_vram/VRAM) + {w_emit}*(Base_emit/Emit)
Score > 1.0 means improvement over uncompressed baseline.

=== AVAILABLE MODES ===

[MODE: asvd_only] Low-rank decomposition → reduces Latency. Often hurts Accuracy; keep param_ratio_target high.
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "asvd_only",
  "alpha": 0.5,              // weight of activation error [0.3, 0.4, 0.5, 0.6, 0.7]
  "param_ratio_target": 0.90, // fraction of params to keep [0.70, 0.80, 0.85, 0.90, 0.95, 0.99]; higher = less compression but more accurate
  "scaling_method": "fisher"  // [abs_mean, abs_max, fisher]; fisher usually best
}}

[MODE: quant_only / GPTQ] Hessian-based weight quantization → reduces VRAM. Most compatible (2/3/4/8-bit).
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "quant_only", "quant_method": "gptq",
  "quant_bits": 4,        // [2,3,4,8]; lower = smaller model but more accuracy loss
  "quant_group_size": 128, // [32,64,128,256]; smaller = more accurate but larger file; -1 = per-channel (fastest, lowest accuracy)
  "quant_format": "gptq", // [gptq, gptq_v2]; gptq_v2 slightly better accuracy; marlin/bitblas FAIL — do NOT use
  "desc_act": false,      // Reorder weight columns by activation magnitude before quantizing. Helps preserve key weights. Set true ONLY when bits<=3 (otherwise <0.5% effect)
  "damp_percent": 0.05,   // Hessian dampening [0.005, 0.01, 0.05, 0.1]; 0.05 is safe default, 0.01 more aggressive
  "mse": 0.0              // MSE regularization in loss [0.0, 0.01, 0.05, 0.1]; 0.0 = standard GPTQ; >0 may slightly improve accuracy
}}

[MODE: quant_only / AWQ] Activation-aware quantization → reduces VRAM. FIXED: bits=4, format=gemm (auto).
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "quant_only", "quant_method": "awq",
  "quant_group_size": 128  // kernel supports ONLY [16, 32, 64, 128]; smaller = more accurate but larger file
}}

[MODE: quant_only / QQQ] W4A8 quantization (4-bit weights + 8-bit activations) → fastest inference via INT8 GEMM. FIXED: bits=4, format=qqq.
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "quant_only", "quant_method": "qqq",
  "quant_group_size": 128, // kernel supports ONLY [-1, 128]; -1=per-channel (no grouping, fastest, lowest accuracy), 128=standard
  "desc_act": true,        // Recommended true for W4A8 (more important than in GPTQ)
  "damp_percent": 0.005    // [0.001, 0.005, 0.01]; NOTE: QQQ Hessian scale is ~2.0 vs GPTQ ~32000, so default 0.005 here ≈ 0.05 in GPTQ
}}

[MODE: quant_only / BNB] On-the-fly quantization (no saved model file) → best for hybrid with ASVD.
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "quant_only", "quant_method": "bnb",
  "quant_bits": 4,         // [4, 8]
  "use_double_quant": false // Only effective when quant_bits=4; second quantization of scale factors for slightly better accuracy
}}

[MODE: sparse_only / unstructured] SparseGPT weight pruning → reduces VRAM and Latency.
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "sparse_only", "sparsity_structure": "unstructured",
  "sparsity_ratio": 0.5  // Fraction of weights set to zero [0.3, 0.4, 0.5, 0.6, 0.7]; start conservative (0.4-0.5), increase if accuracy holds; smaller models are more sensitive
}}

[MODE: sparse_only / structured] Structured sparsity (N:M pattern). FIXED: 50% sparsity, no sparsity_ratio needed.
  2:4 — fine-grained, requires Ampere+ GPU, slightly lower accuracy
  4:8 — coarser, more hardware-compatible, slightly better accuracy than 2:4
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "sparse_only",
  "sparsity_structure": "2:4"  // or "4:8"
}}

[MODE: hybrid / ASVD+BNB] Combine low-rank + quantization. ONLY bnb works after ASVD (gptq/awq/qqq incompatible).
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "hybrid",
  "alpha": 0.5, "param_ratio_target": 0.92, "scaling_method": "fisher",
  "quant_method": "bnb", "quant_bits": 4, "use_double_quant": false
}}

[MODE: hybrid / Sparse+Quant] Apply SparseGPT first, then quantization.
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "hybrid", "sparsity_structure": "unstructured", "sparsity_ratio": 0.4,
  "quant_method": "gptq", "quant_bits": 4, "quant_group_size": 128, "quant_format": "gptq"
}}

=== CURRENT STATUS ===
Trial History :
{history_str}

Pareto Frontier (best trade-offs found):
{pareto_str}

=== STRATEGY ===
- Early iterations: try each mode independently to understand isolated impact.
- For quant_only: start with GPTQ 4-bit, then explore bits/group_size/format variations, then try AWQ/QQQ.
- Later iterations: combine methods in hybrid mode.
- NEVER repeat identical configs. Use Pareto frontier to find unexplored trade-off regions.


Output ONLY the JSON for your chosen mode (copy and fill one template above). No extra fields, no prose.
"""

    @staticmethod
    def _strip_json_comments(text: str) -> str:
        """移除 JSON 中的 // 行尾注釋（LLM 有時會帶入 prompt 的 template 格式）"""
        import re
        # 移除 // 後面到行尾的內容（避免誤刪字串內的 //）
        return re.sub(r'(?<!:)//[^\n"]*', '', text)

    def get_suggestion(self, iteration: int, trial_history: list, pareto: list = None,
                       weights: dict = None):
        """
        Returns (suggestion: StrategySuggestion, raw_llm_output: dict)
        raw_llm_output is the raw LLM output (without pydantic defaults), used for logging.
        """
        import json
        from pydantic import ValidationError
        import logging
        
        logger = logging.getLogger("LLMClient")
        max_retries = 3
        
        # FIXED: Added weights parameter back in
        prompt = self._create_prompt(iteration, trial_history, pareto=pareto, weights=weights)
        messages = [{"role": "user", "content": prompt}]
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                
                # FIXED: Strip comments to prevent json.loads from crashing
                raw_content = self._strip_json_comments(response.choices[0].message.content)
                
                # Attempt to validate the JSON against our strict Literal rules
                suggestion = StrategySuggestion.model_validate_json(raw_content)
                
                # FIXED: Return the exact tuple specified in the docstring
                return suggestion, json.loads(raw_content)
                
            except ValidationError as e:
                logger.warning(f"⚠️ LLM Validation failed on attempt {attempt + 1}/{max_retries}. Retrying...\nError: {e}")
                
                if attempt == max_retries - 1:
                    logger.error("Max retries reached. LLM failed to produce valid JSON.")
                    raise # Crash loudly if it fails 3 times so you know something is broken
                
                # Feed the error back to the LLM so it can learn and correct itself
                messages.append({"role": "assistant", "content": raw_content})
                messages.append({
                    "role": "user", 
                    "content": f"Your JSON failed Pydantic validation. Please fix the following errors and strictly follow the schema:\n{e}"
                })