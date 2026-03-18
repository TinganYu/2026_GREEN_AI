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
    def __init__(self, model_id: str, task: str, max_iterations: int, memory_type: str = "full"):
        self.model_id = model_id
        self.task = task
        self.max_iterations = max_iterations
        self.client = OpenAI(api_key=os.getenv("LLM_API_KEY"))
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o")

        self.memory_type = memory_type  # 'full', 'window', or 'summary'
        self.knowledge_summary = "No previous summary available."
        self.last_summarized_idx = 0

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

    def _execute_retrieve_trials(self, query: str, trial_history: list, top_n: int = 5) -> str:
        """Filters the trial history based on the requested sub-method and returns top N by score."""
        if not trial_history:
            return "No past trials available."

        filtered = []
        for trial in trial_history:
            cfg = trial.get("config", {})
            if not cfg:
                continue
                
            mode = cfg.get("mode", "")
            q_method = cfg.get("quant_method", "")

            # Match logic based on the query
            is_match = False
            if query == "asvd" and mode in ["asvd_only", "hybrid"]:
                is_match = True
            elif query in ["gptq", "awq", "qqq", "bnb"] and q_method == query:
                is_match = True
            elif query == "sparse" and mode in ["sparse_only", "hybrid"]:
                is_match = True
            elif query == "hybrid" and mode == "hybrid":
                is_match = True

            if is_match:
                filtered.append(trial)

        if not filtered:
            return f"No trials found matching query: '{query}'."

        # Sort by score descending and take top N
        filtered.sort(key=lambda x: x.get("metrics", {}).get("score", 0), reverse=True)
        filtered = filtered[:top_n]

        return f"--- RETRIEVAL RESULTS FOR '{query}' (Top {len(filtered)} by Score) ---\n" + self._format_history(filtered)

    def _update_knowledge_summary(self, trial_history: list):
        """Updates the LLM summary every 5 trials, correcting past assumptions."""
        if len(trial_history) - self.last_summarized_idx >= 5:
            # recent_batch = trial_history[self.last_summarized_idx : self.last_summarized_idx + 5]
            batch_str = self._format_history(trial_history)
            
            prompt = f"""You are an AI assistant maintaining an evolving knowledge base for a model compression agent. 

=== CURRENT KNOWLEDGE SUMMARY (May contain outdated or incorrect early assumptions) ===
{self.knowledge_summary}

=== FULL EXPERIMENTAL HISTORY (Trials 1 through Current) ===
{batch_str}

=== INSTRUCTIONS ===
Your task is to rewrite and update the current knowledge summary based on the new trial results.
1. FIND CORRELATIONS: Identify how specific parameter movements affect the metrics. (e.g., "When parameter X goes down, accuracy drops sharply").
2. DEFINE BOUNDARIES: Identify safe and unsafe zones for hyperparameters based on past failures (e.g., "Ratio values below 0.85 cause failure").
3. STRATEGIC SUGGESTION: Suggest the next logical phase of exploration based on the trade-offs observed (e.g., "We have hit a wall with VRAM reduction using Method A; explore Method B to push VRAM lower").
4. 🛑 CRITICAL CONSTRAINT 🛑: DO NOT specify exact parameter combinations to run next. Define the "rules of the game" (what works and what fails), and let the execution agent decide the exact numbers.

Output ONLY the newly updated summary text. Do not include conversational filler.
"""

            try:
                response = self.client.chat.completions.create(
                    model=os.getenv("SUMMARY_MODEL", "gpt-4o-mini"), # Cheaper/faster model
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                # Overwrite the old summary with the newly evolved one
                self.knowledge_summary = response.choices[0].message.content.strip()
                self.last_summarized_idx += 5
                
                # Log the update so you can watch the agent "change its mind" in the terminal
                import logging
                logger = logging.getLogger("LLMClient")
                logger.info(f"\n🧠 [Knowledge Base Updated]\n{self.knowledge_summary}\n")
                
            except Exception as e:
                import logging
                logger = logging.getLogger("LLMClient")
                logger.error(f"Failed to update knowledge summary: {e}")

    def _create_prompt(self, iteration: int, trial_history: list, pareto: list = None,
                       weights: dict = None) -> str:
        
        # Handle the different memory modes
        if self.memory_type == "window":
            history_str = self._format_history(trial_history[-5:])
        elif self.memory_type == "tool":
            # Keep baseline context tight; the agent will fetch what else it needs
            history_str = self._format_history(trial_history[-3:])
        elif self.memory_type == "summary":
            self._update_knowledge_summary(trial_history)
            unsummarized = trial_history[self.last_summarized_idx:]
            history_str = f"--- LLM KNOWLEDGE SUMMARY ---\n{self.knowledge_summary}\n\n"
            history_str += f"--- RECENT UNSUMMARIZED TRIALS ---\n{self._format_history(unsummarized) if unsummarized else 'None'}"
        else: # default to "full"
            history_str = self._format_history(trial_history)

        pareto_str = self._format_history(pareto) if pareto else "None"

        w = weights or {}
        w_acc  = w.get("acc",  0.5)
        w_lat  = w.get("lat",  0.1)
        w_vram = w.get("vram", 0.2)
        w_emit = w.get("emit", 0.2)

        return f"""You are an LLM compression optimization agent. Choose the best compression strategy for:
Model: {self.model_id} | Task: {self.task} | Iteration: {iteration}/{self.max_iterations}

GOAL: Maximize score = 1.0 + {w_acc}*ln(Acc/Base_acc) + {w_lat}*ln(Base_lat/Lat) + {w_vram}*ln(Base_vram/VRAM) + {w_emit}*ln(Base_emit/Emit)
Score > 1.0 means improvement over uncompressed baseline. Logarithmic scaling dampens extreme outliers.

=== AVAILABLE MODES ===
[MODE: asvd_only] Low-rank decomposition → reduces Latency. Often hurts Accuracy; keep param_ratio_target high.
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "asvd_only",
  "alpha": 0.5,              
  "param_ratio_target": 0.90, 
  "scaling_method": "fisher"  
}}

[MODE: quant_only / GPTQ] Hessian-based weight quantization → reduces VRAM. Most compatible (2/3/4/8-bit).
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "quant_only", "quant_method": "gptq",
  "quant_bits": 4,        
  "quant_group_size": 128, 
  "quant_format": "gptq", 
  "damp_percent": 0.05,   
  "mse": 0.0              
}}

[MODE: quant_only / AWQ] Activation-aware quantization → reduces VRAM. FIXED: bits=4, format=gemm (auto).
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "quant_only", "quant_method": "awq",
  "quant_group_size": 128  
}}

[MODE: quant_only / QQQ] W4A8 quantization. FIXED: bits=4, format=qqq.
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "quant_only", "quant_method": "qqq",
  "quant_group_size": 128, 
  "damp_percent": 0.005    
}}

[MODE: quant_only / BNB] On-the-fly quantization.
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "quant_only", "quant_method": "bnb",
  "quant_bits": 4,         
  "use_double_quant": false 
}}

[MODE: sparse_only / unstructured] SparseGPT weight pruning.
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "sparse_only", "sparsity_structure": "unstructured",
  "sparsity_ratio": 0.5  
}}

[MODE: sparse_only / structured] Structured sparsity (N:M pattern). 
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "sparse_only",
  "sparsity_structure": "2:4"  
}}

[MODE: hybrid / ASVD+BNB] Combine low-rank + quantization. 
OUTPUT FORMAT:
{{"reasoning": "...", "mode": "hybrid",
  "alpha": 0.5, "param_ratio_target": 0.92, "scaling_method": "fisher",
  "quant_method": "bnb", "quant_bits": 4, "use_double_quant": false
}}

=== CURRENT STATUS ===
Trial Context ({self.memory_type} mode):
{history_str}

Pareto Frontier (best trade-offs found):
{pareto_str}

=== STRATEGY ===
- Early iterations: try each mode independently to understand isolated impact.
- For quant_only: start with GPTQ 4-bit, then explore variations.
- Later iterations: combine methods in hybrid mode.
- NEVER repeat identical configs. Use Pareto frontier to find unexplored regions.

Output ONLY the JSON for your chosen mode. No extra fields, no prose.
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
        Routes the request based on memory type.
        """
        if self.memory_type == "tool":
            return self._get_suggestion_with_tools(iteration, trial_history, pareto, weights)

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

    def _get_suggestion_with_tools(self, iteration: int, trial_history: list, pareto: list = None, weights: dict = None):
        """Bounded multi-round tool-calling loop."""
        import json
        from pydantic import ValidationError
        import logging
        logger = logging.getLogger("LLMClient")
        MAX_TURNS = 3 # Turn 1: Tool Call, Turn 2: Tool Call or Answer, Turn 3: Forced Answer

        # The prompt is simpler because the agent will fetch what it needs
        system_prompt = self._create_prompt(
            iteration, 
            trial_history, # Only show the absolute most recent 3 trials by default
            pareto=pareto, 
            weights=weights
        )
        system_prompt += (
            "\n\n=== TOOL USAGE RULES ===\n"
            "1. You have a 'retrieve_trials' tool to search past experiments by method.\n"
            f"2. You have a STRICT LIMIT of {MAX_TURNS - 1} search queries per iteration. Plan your queries carefully!\n"
            "3. Once you have enough information, or if you run out of turns, you MUST output ONLY the final StrategySuggestion JSON."
        )
        messages = [{"role": "system", "content": system_prompt}]
        
        tools = [{
            "type": "function",
            "function": {
                "name": "retrieve_trials",
                "description": "Fetch past trial results by method to see what parameters succeeded or failed. (Pareto best configs are already in your prompt).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "enum": ["asvd", "gptq", "awq", "qqq", "bnb", "sparse", "hybrid"],
                            "description": "The specific compression method to search for."
                        }
                    },
                    "required": ["query"]
                }
            }
        }]
        
        for turn in range(MAX_TURNS):
            force_answer = (turn == MAX_TURNS - 1)
            
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                tools=tools if not force_answer else None,
                tool_choice="auto" if not force_answer else "none",
            )
            
            msg = response.choices[0].message
            messages.append(msg) # Append assistant message to history
            
            # If the model decided to use a tool
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query")
                    
                    if not query:
                        # 防呆機制：如果 LLM 漏給參數，強制它重新思考
                        retrieval_results = "System Error: Missing required parameter 'query'. Please specify a method like 'gptq' or 'asvd'."
                        logger.warning("Agent called tool without a query.")
                    else:
                        logger.info(f"🔍 Agent requested retrieval for: {query}")
                        retrieval_results = self._execute_retrieve_trials(query, trial_history)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": retrieval_results
                    })
                continue # Go to the next turn to let the LLM analyze the results
                
            # If no tool calls, it means the model outputted the final JSON
            else:
                raw_content = self._strip_json_comments(msg.content)
                try:
                    from schemas import StrategySuggestion
                    suggestion = StrategySuggestion.model_validate_json(raw_content)
                    return suggestion, json.loads(raw_content)
                except ValidationError as e:
                    logger.error(f"Validation failed in tool mode: {e}. Falling back to default.")
                    # Fallback if the agent messes up the JSON after tool calling
                    return self.get_suggestion(iteration, trial_history, pareto, weights) # Fallback to standard flow