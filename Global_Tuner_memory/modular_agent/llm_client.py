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
    def __init__(self, model_id: str, task: str, max_iterations: int, memory_type: str = "full", pen_t: float = 0.15, pen_a: float = 10.0, baseline_metrics: dict = None):
        self.model_id = model_id
        self.task = task
        self.max_iterations = max_iterations
        self.client = OpenAI(api_key=os.getenv("LLM_API_KEY"))
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o")
        self.exp_dir = None  # Will be set by the orchestrator when the experiment starts

        self.memory_type = memory_type  # 'full', 'window', or 'summary'
        self.knowledge_summary = "No previous summary available."
        self.last_summarized_idx = 0

        self.pen_t = pen_t
        self.pen_a = pen_a
        self.baseline_metrics = baseline_metrics

    def _format_history(self, history: list) -> str:
        if not history:
            return "None"

        lines = []
        for trial in history:
            config = trial.get('config', {})
            metrics = trial.get('metrics', {})
            details = metrics.get('details', {})

            # line = f"- [Iter {trial.get('iteration')}]: Score={metrics.get('score', 0):.4f}"
            # line += (f" (Acc: {metrics.get('accuracy', 0):.4f},"
            #          f" Lat: {metrics.get('latency', 0):.4f}s,"
            #          f" VRAM: {metrics.get('vram', 0):.4f}GB,"
            #          f" Emit: {metrics.get('emissions', 0):.6f}kg CO2)")
            score = metrics.get('score', 0)
            acc = metrics.get('accuracy', 0)
            lat = metrics.get('latency', 0)
            vram = metrics.get('vram', 0)
            emit = metrics.get('emissions', 0)

            acc_str = f"{acc:.4f}"
            lat_str = f"{lat:.4f}s"
            vram_str = f"{vram:.4f}GB"
            emit_str = f"{emit:.6f}kg"

            # Change 5: Provide change percentage at the end of metrics
            if self.baseline_metrics:
                b_acc = self.baseline_metrics.get('accuracy', acc)
                b_lat = self.baseline_metrics.get('latency', lat)
                b_vram = self.baseline_metrics.get('vram', vram)
                b_emit = self.baseline_metrics.get('emissions', emit)

                # Accuracy: Calculate BOTH absolute diff (pp) and relative change (%)
                acc_diff = acc - b_acc if b_acc else 0
                acc_pct = ((acc - b_acc) / b_acc) * 100 if b_acc else 0
                
                # Others: Just relative change (%)
                lat_pct = ((lat - b_lat) / b_lat) * 100 if b_lat else 0
                vram_pct = ((vram - b_vram) / b_vram) * 100 if b_vram else 0
                emit_pct = ((emit - b_emit) / b_emit) * 100 if b_emit else 0

                # Append to strings
                acc_str += f" ({acc_diff:+.4f} pp, {acc_pct:+.2f}%)"
                lat_str += f" ({lat_pct:+.2f}%)"
                vram_str += f" ({vram_pct:+.2f}%)"
                emit_str += f" ({emit_pct:+.2f}%)"

            line = f"- [Iter {trial.get('iteration')}]: Score={score:.4f} (Acc: {acc_str}, Lat: {lat_str}, VRAM: {vram_str}, Emit: {emit_str})"

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

            # Match logic based on the query
            is_match = False
            if "asvd" in query and mode in ["asvd_only", "hybrid_asvd_bnb"]:
                is_match = True
            elif query in ["gptq", "awq", "qqq", "bnb"] and mode == query:
                is_match = True
            elif "sparse" in query and mode in ["sparse_unstructured", "sparse_structured"]:
                is_match = True
            elif "hybrid" in query and mode == "hybrid_asvd_bnb":
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
            new_batch = trial_history[self.last_summarized_idx : self.last_summarized_idx + 5]
            batch_str = self._format_history(new_batch)
            
            prompt = f"""You are an AI assistant maintaining an evolving knowledge base for a model compression agent. 

=== CURRENT KNOWLEDGE SUMMARY (May contain outdated or incorrect early assumptions) ===
{self.knowledge_summary}

=== FULL EXPERIMENTAL HISTORY (Trials 1 through Current) ===
{batch_str}

=== INSTRUCTIONS ===
Your task is to rewrite and update the current knowledge summary based on the new trial results.
🛑 CRITICAL CONSTRAINT: DO NOT specify exact parameter combinations to run next. Define the "rules of the game".

Output the summary in EXACTLY this structure:
## Correlations
(parameter → metric relationships)

## Safe Zones
(known safe parameter ranges)

## Danger Zones
(configurations that caused failures or heavy penalties)

## Recommended Direction
(next exploration focus, NO specific numbers)

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

        # Tracking for tried_modes and untried_modes directly in the original agent
        tried_modes_counts = {}
        for trial in trial_history:
            mode = trial.get('config', {}).get('mode')
            if mode:
                tried_modes_counts[mode] = tried_modes_counts.get(mode, 0) + 1
                
        all_modes = {"asvd_only", "gptq", "awq", "qqq", "bnb", "sparse_unstructured", "sparse_structured", "hybrid_asvd_bnb"}
        untried_modes = sorted(list(all_modes - set(tried_modes_counts.keys())))
        
        tried_str = ", ".join([f"{k} ({v}x)" for k, v in tried_modes_counts.items()]) if tried_modes_counts else "None"
        untried_str = ", ".join(untried_modes) if untried_modes else "None (All modes explored)"



        return f"""You are an LLM compression optimization agent. Choose the best compression strategy for:
Model: {self.model_id} | Task: {self.task} | Iteration: {iteration}/{self.max_iterations}

GOAL: Maximize Final Score.
1. Base Score = 1.0 + {w_acc}*ln(Acc/Base_acc) + {w_lat}*ln(Base_lat/Lat) + {w_vram}*ln(Base_vram/VRAM) + {w_emit}*ln(Base_emit/Emit)  
(explain:Prioritize positive relative % changes in Lat, VRAM, and Emit, while minimizing negative absolute drops (pp) in Acc.)
2. PENALTY: A massive penalty (multiplier {self.pen_a}) is applied ONLY if the Accuracy drop exceeds {self.pen_t} (i.e., the "pp" diff is more negative than -{self.pen_t}).
3. Final Score = Base Score - Penalty

=== AVAILABLE MODES & OUTPUT FORMATS ===
Strictly output ONLY valid JSON matching one of these structures. Do not wrap in markdown formatting.
"reasoning" MUST follow this structure: "A detailed explanation of why this mode fills a gap in current coverage, followed by a comprehensive analysis of the expected trade-offs and potential risks."
[MODE: asvd_only]
{{"reasoning": "...", "mode": "asvd_only", 
  "alpha": 0.5,               // categorical [0.3, 0.4, 0.5, 0.6, 0.7] Higher = preserves activation distribution more
  "param_ratio_target": 0.90, // linear [0.70 - 0.99] Lower = heavier compression
  "scaling_method": "fisher"  // categorical ["abs_mean", "abs_max", "fisher"] fisher typically best
}}

[MODE: gptq]
{{"reasoning": "...", "mode": "gptq",
  "quant_bits": 4,         // categorical [3, 4, 8] 4-bit is sweet spot
  "quant_group_size": 128, // categorical [16, 32, 64, 128, 256] Smaller = higher precision, larger size
  "quant_format": "gptq",  // categorical ["gptq", "gptq_v2"] v2 fixes overflow
  "damp_percent": 0.05     // log [0.001 - 0.1] Try 0.01~0.05
}}

[MODE: awq]
{{"reasoning": "...", "mode": "awq",
  "quant_group_size": 128  // categorical [16, 32, 64, 128] Fixed at 4-bit
}}

[MODE: qqq]
{{"reasoning": "...", "mode": "qqq",
  "quant_group_size": 128, // categorical [-1, 128] -1 is full matrix
  "damp_percent": 0.005    // log [0.0005 - 0.05] Hessian dampening
}}

[MODE: bnb]
{{"reasoning": "...", "mode": "bnb",
  "quant_bits": 4,         // categorical [4, 8] 4 saves VRAM aggressively
  "use_double_quant": false // categorical [true, false] True saves ~0.4 bit/param (if bits=4)
}}

[MODE: sparse_unstructured]
{{"reasoning": "...", "mode": "sparse_unstructured",
  "sparsity_ratio": 0.5    // linear [0.3 - 0.7] e.g., 0.5 = 50% weights pruned
}}

[MODE: sparse_structured]
{{"reasoning": "...", "mode": "sparse_structured",
  "sparsity_structure": "2:4" // categorical ["2:4", "4:8"] Hardware-acceleration friendly
}}

[MODE: hybrid_asvd_bnb]
{{"reasoning": "...", "mode": "hybrid_asvd_bnb",
  "alpha": 0.5, "param_ratio_target": 0.92, "scaling_method": "fisher",
  "quant_bits": 4, "use_double_quant": false
}}

=== CURRENT STATUS ===
Modes tried so far: {tried_str}
Modes NOT yet tried: {untried_str}

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
                raw_content = self._strip_json_comments(response.choices[0].message.content or "{}") # 防呆：如果 content 是 None，給一個空 JSON 讓它至少能通過 json.loads 而不是崩潰
                
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
        from datetime import datetime
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
            "2. STRICT RULE: DO NOT use the tool for any method listed in 'Modes NOT yet tried'. The database will be empty.\n"
            f"3. You have a STRICT LIMIT of {MAX_TURNS - 1} search queries per iteration. Plan your queries carefully!\n"
            "4. Once you have enough information, or if you run out of turns, you MUST output ONLY the final StrategySuggestion JSON.\n"
            "5. Do not make more than 2 parallel search queries at the exact same time."
        )
        messages = [{"role": "system", "content": system_prompt}]
        
        tools = [{
            "type": "function",
            "function": {
                "name": "retrieve_trials",
                "description": "Fetch full history for a specific method. STRICT LIMITATION: Do NOT query methods you haven't tried yet (check the 'Modes NOT yet tried' list). Only use this to investigate variations or failures of methods you HAVE already tried.",
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

            if force_answer and turn > 0:
                messages.append({
                    "role": "user",
                    "content": "You have run out of query chances. Based on all information gathered, output ONLY the final JSON strategy. No tool calls, no prose."
                })

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
                        debug_log = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "iteration": iteration,
                            "turn": turn + 1,
                            "query": query,
                            "results": retrieval_results
                        }
                        
                        # Use the specific experiment directory instead of the root!
                        if hasattr(self, 'exp_dir'):
                            log_path = self.exp_dir / "tool_debug_log.jsonl"
                            with open(log_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(debug_log, ensure_ascii=False) + "\n")
                        else:
                            logger.warning("No exp_dir set for LLMDecisionMaker; skipping tool log.")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": retrieval_results
                    })
                continue # Go to the next turn to let the LLM analyze the results
                
            # If no tool calls, it means the model outputted the final JSON
            else:
                raw_content = self._strip_json_comments(msg.content or "{}") # 防呆：如果 content 是 None，給一個空 JSON 讓它至少能通過 json.loads 而不是崩潰
                
                # ---  Mini-retry loop to preserve tool context ---
                for attempt in range(2): 
                    try:
                        from schemas import StrategySuggestion
                        suggestion = StrategySuggestion.model_validate_json(raw_content)
                        return suggestion, json.loads(raw_content)
                    except ValidationError as e:
                        logger.warning(f"⚠️ Tool-mode JSON validation failed (attempt {attempt+1}/2): {e}")
                        
                        if attempt == 1: # Last attempt failed
                            logger.error("Agent failed to fix JSON. Falling back to window mode.")
                            original_memory = self.memory_type
                            self.memory_type = "window"
                            try:  # just use fallback memory mode to get a valid suggestion without crashing the whole system, even if it's not tool-optimized
                                return self.get_suggestion(iteration, trial_history, pareto, weights)
                            finally:
                                self.memory_type = original_memory
                                
                        # Feed the error back into the SAME message array (Preserves tool context!)
                        messages.append({"role": "assistant", "content": raw_content})
                        messages.append({
                            "role": "user", 
                            "content": f"Your JSON failed Pydantic validation. Please fix these errors and output valid JSON:\n{e}"
                        })
                        
                        # Ask the LLM one more time to fix it
                        retry_response = self.client.chat.completions.create(
                            model=self.llm_model,
                            messages=messages,
                            response_format={"type": "json_object"}
                        )
                        raw_content = self._strip_json_comments(retry_response.choices[0].message.content)