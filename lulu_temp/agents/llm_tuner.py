"""
LLM-based Adaptive Tuner for KV Compression
Uses LLM to suggest hyperparameters based on previous trial feedback
Inspired by OPRO (Optimization by PROmpting) and similar work
"""
import logging
import json
from eval.base_evaluator import BaseEvaluator
import importlib
import os
import gc
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
import torch
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError 
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    logger.warning("codecarbon not available. Install: pip install codecarbon")

import os

# Check if using API
USE_API = os.getenv("USE_LLM_API", "false").lower() == "true"
API_KEY = os.getenv("LLM_API_KEY", "")  # or ANTHROPIC_API_KEY
API_MODEL = os.getenv("LLM_MODEL", "gpt-4o")


from typing import Dict, List, Optional, Any
# ... (其他 imports)

def _deep_clean_dict(data: Any) -> Any:
    """Recursively converts PyTorch Tensors to native Python types (float/list)"""
    import torch
    
    if isinstance(data, dict):
        # 遞迴處理字典
        return {k: _deep_clean_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        # 遞迴處理列表
        return [_deep_clean_dict(v) for v in data]
    # 檢查並轉換 PyTorch Tensor
    elif isinstance(data, torch.Tensor):
        # 處理標量張量 (Scalar Tensor)
        if data.numel() == 1:
            # 安全地移動到 CPU 並取得數值
            return data.cpu().item() 
        # 處理非標量張量 (Non-scalar Tensor)
        else:
            # 轉換為 NumPy 列表
            return data.cpu().numpy().tolist()
    # 處理 NumPy 數值類型 (例如 np.float32)
    elif hasattr(data, 'item') and callable(data.item):
        return data.item()
    else:
        # 返回原生類型
        return data

# ... (LLMAdaptiveTuner 類別定義開始)
# ----------------------------------------------------
# Pydantic 模型定義 (取代手動 JSON 驗證)
# ----------------------------------------------------
class CompressionConfig(BaseModel):
    """定義 LLM 期望輸出的超參數配置結構"""
    
    # 這是 LLM 的分析部分 (會被解析但最終從參數中移除)
    reasoning: str = Field(..., description="Brief explanation of why this config should be tried (2-3 sentences).")
    
    # 核心超參數
    compression_method: str = Field(..., description="One of: snapkv, observed_attention, knorm, expected_attention.")
    compression_ratio: float = Field(..., ge=0.2, le=0.9, description="Fraction of tokens to KEEP (0.2 to 0.9).")
    
    # 選填參數 (LLM 應根據任務和方法填充)
    # window_size: Optional[int] = Field(None, description="Recent tokens to always keep (e.g., 32-128).")
    # kernel_size: Optional[int] = Field(None, description="Pooling kernel for important token selection (e.g., 3, 5, 7, 9, 11).")
    # max_input_tokens: Optional[int] = Field(None, description="For MultiNews: Truncate long documents (e.g., 2048-6144).")

if USE_API:
    try:
        OpenAI.api_key = API_KEY
        API_AVAILABLE = True
    except ImportError:
        logger.warning("openai package not available")
        API_AVAILABLE = False
else:
    API_AVAILABLE = False

class LLMAdaptiveTuner:
    """
    LLM-based adaptive tuner that learns from trial feedback.
    The LLM acts as the optimization algorithm!
    """
    
    def __init__(self,
                 evaluator,
                 task: str = "gsm8k",
                 max_iterations: int = 20,
                 accuracy_weight: float = 0.7,
                 memory_weight: float = 0.4,
                 latency_weight: float = 0.1,
                 co2_weight: float = 0.1,
                 enable_co2_tracking: bool = False,
                 llm_model: str = None):
        """
        Args:
            evaluator: GSM8KEvaluator or MultiNewsEvaluator
            task: 'gsm8k' or 'multinews'
            max_iterations: Maximum optimization iterations
            accuracy_weight: Weight for accuracy objective
            memory_weight: Weight for memory objective
            latency_weight: Weight for latency objective

            co2_weight: Weight for CO2 objective
            enable_co2_tracking: Track CO2 emissions
            llm_model: LLM model name (uses evaluator's model if None)
        """
        self.evaluator = evaluator
        self.task = task
        self.max_iterations = max_iterations
        self.memory_weight = memory_weight
        self.latency_weight = latency_weight
        self.accuracy_weight = accuracy_weight
        self.co2_weight = co2_weight
        self.enable_co2_tracking = enable_co2_tracking and CODECARBON_AVAILABLE
        self.baseline_score = 0.0
        self.baseline_latency=0.0
        self.baseline_memory=0.0
        self.baseline_co2=0.001
        self.full_num_samples = evaluator.config['dataset'].get('num_samples', None)
        
        # Use the evaluator's model for suggestions
        self.llm_model = llm_model or evaluator.agent
        
        # Track trial history
        self.trial_history = []
        self.best_config = None
        self.best_score = -float('inf')

        self.api_client = None
        if USE_API and API_KEY:
            try:
                # 實例化客戶端，使用 API_KEY
                self.api_client = OpenAI(api_key=API_KEY)
                logger.info(f"OpenAI API client initialized for model: {API_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                global API_AVAILABLE
                API_AVAILABLE = False

        # CO2 tracker
        self.emissions_tracker = None
        if self.enable_co2_tracking:
            self.emissions_tracker = EmissionsTracker(
                project_name=f"llm_tuning_{task}",
                output_dir="./emissions",
                log_level="ERROR",
                save_to_file=True
            )
        
        # logger.info(f"LLM Adaptive Tuner initialized for {task}")
    
#     def _create_optimization_prompt(self, iteration: int) -> str:
#         """
#         Create prompt for LLM to suggest next hyperparameters.
#         Based on OPRO-style prompting.
#         """
#         # Task description
#         if self.task == "gsm8k":
#             task_desc = "mathematical reasoning (GSM8K dataset)"
#             metric_name = "accuracy"
#         else:
#             task_desc = "text summarization (MultiNews dataset)"
#             metric_name = "ROUGE-L F1"
        
#         # Build prompt with history
#         prompt = f"""You are an expert hyperparameter optimization assistant. Your goal is to find the best KV cache compression settings for {task_desc}.
# ## Baseline Performance (No Compression)
# {metric_name}: {self.baseline_score:.4f}
        
# ## Objective
# Maximize: {self.accuracy_weight:.0%} {metric_name} + {self.co2_weight:.0%} CO2 efficiency

# ## Available Hyperparameters

# 1. **compression_method** (categorical):
#    - snapkv: Attention-based, keeps recent + important tokens
#    - observed_attention: More accurate attention-based (slower)
#    - knorm: Magnitude-based, fastest
#    - expected_attention: Expected attention patterns

# 2. **compression_ratio** (float, 0.2-0.9):
#    - Fraction of tokens to KEEP (higher = more tokens, better quality, higher cost)
#    - Typically 0.3-0.7 works best



# """
# #        3. **window_size** (int, method-specific):
# #    - For snapkv: {"16-64 (GSM8K)" if self.task == "gsm8k" else "32-128 (MultiNews)"}
# #    - For streaming_llm: {"256-1024" if self.task == "gsm8k" else "512-2048"}
# #    - Recent tokens to always keep

# # 4. **kernel_size** (int, snapkv only):
# #    - {3, 5, 7, 9, 11}
# #    - Pooling kernel for important token selection 
#         if self.task == "multinews":
#             prompt += """5. **max_input_tokens** (int, 2048-6144):
#    - Truncate long documents
#    - Lower = faster but may lose info

# """
        
#         # Add trial history
#         if self.trial_history:
#             prompt += f"""## Previous Trials (Iteration {iteration}/{self.max_iterations})

# """
#             # Show last 5 trials
#             recent_trials = self.trial_history[-5:]
#             for i, trial in enumerate(recent_trials, 1):
#                 params = trial['params']
#                 score = trial['score']
#                 acc = trial['accuracy']
#                 co2 = trial['co2_kg'] * 1000  # to grams
                
#                 prompt += f"""Trial {trial['iteration']}:
#   Config: method={params['compression_method']}, ratio={params['compression_ratio']:.2f}"""
                
#                 # if 'window_size' in params:
#                 #     prompt += f", window={params['window_size']}"
#                 # if 'kernel_size' in params:
#                 #     prompt += f", kernel={params['kernel_size']}"
                
#                 prompt += f"""
#   Results: {metric_name}={acc:.4f}, CO2={co2:.2f}g, Score={score:.4f}
#   Analysis: """
                
#                 # # Add analysis
#                 # if score > 0.7:
#                 #     prompt += "✓ Good configuration"
#                 # elif score > 0.5:
#                 #     prompt += "○ Moderate performance"
#                 # else:
#                 #     prompt += "✗ Poor performance"
                
#                 prompt += "\n\n"
            
#             # Show best so far
#             best = max(self.trial_history, key=lambda x: x['score'])
#             prompt += f"""🏆 Best so far (Trial {best['iteration']}): Score={best['score']:.4f}
#   Config: {best['params']}
  
# """
        
#         else:
#             prompt += """## Previous Trials
# No trials yet. This is the first iteration.

# ## Suggestions for First Trial
# - Start with snapkv (good balance)
# - Try compression_ratio around 0.5-0.6


# """
        
#         # Request for next configuration
#         prompt += """## Your Task
#         Based on the trials above, suggest the NEXT configuration to try.
#         ## Exploration Encouragement
#         - Try to explore methods that have not been used in previous trials.
#         - Avoid always picking the same method unless you have strong evidence it works best.
#         - You should balance exploration and exploitation.

#         Return a JSON object with these fields:
#         - reasoning: Why this config (2-3 sentences)
#         - compression_method: One of [snapkv, observed_attention, knorm, expected_attention]
#         - compression_ratio: Float between 0.2 and 0.9
#         """
#         # - window_size: Integer (optional, method-specific)
#         # - kernel_size: Integer (optional, for snapkv)

#         if self.task == "multinews":
#             prompt += """
#         - max_input_tokens: Integer between 2048-6144 (optional)"""

#         prompt += """

# [Output ONLY JSON]
# [no prose]
# Do not include any prefix like "(Valid)" or "Output:".
# Your output will be parsed directly with `json.loads()`. 
# Here is the expected format:

# {
#   "reasoning": "brief reasoning",
#   "compression_method": "one of [snapkv, observed_attention, knorm, expected_attention]",
#   "compression_ratio": 0.2-0.9
# }

# Example:
# {"reasoning": "snapkv gives balanced accuracy and cost", "compression_method": "snapkv", "compression_ratio": 0.6}
# """
# # ,
# #   "window_size": optional int,
# #   "kernel_size": optional int,
# #   "max_input_tokens": optional int


#         return prompt

    def _create_optimization_prompt(self, iteration: int) -> str:
        """
        Create prompt for LLM to suggest next hyperparameters.
        使用標準化效益 (Benefit) R 公式，並維持 memory_gb 鍵值。
        """
        # Task description
        if self.task == "gsm8k":
            task_desc = "mathematical reasoning (GSM8K dataset)"
            metric_name = "accuracy"
        else:
            task_desc = "text summarization (MultiNews dataset)"
            metric_name = "ROUGE-L F1"
            
        # 確保分母不為零，並給 LLM 參考
        base_lat = max(1e-6, self.baseline_latency)
        base_mem = max(1e-6, self.baseline_memory)
        
        # ---------------------- I. 核心結構和 Baseline ----------------------

        # **標準化公式的核心：將成本項轉換為效益 Benefit**
        
        score_formula = (
            f"Maximize: **R = (Model_Quality x {self.accuracy_weight}) "
            f"+ (Latency_Benefit x {self.latency_weight}) "
            f"+ (Memory_Benefit x {self.memory_weight})"
        )
        
        co2_benefit_note = ""
        if self.enable_co2_tracking and self.co2_weight > 0:
            # 假設 self.baseline_co2_kg 已存在
            base_co2 = max(1e-6, self.baseline_co2)
            score_formula += f" + (CO2_Benefit x {self.co2_weight})**"
            co2_benefit_note = f"\n- **CO2_Benefit** = Max(0, 1 - (CO2_kg / Baseline_CO2={base_co2:.4f} kg))"
        else:
            score_formula += "**"

        benefit_notes = (
            f"\n\n--- BENEFIT CALCULATION (All cost metrics are converted to 0.0-1.0 Benefit) ---\n"
            f"**Latency_Benefit** = Max(0, 1 - (Latency_s / Baseline_Latency={base_lat:.2f} s))"
            f"\n**Memory_Benefit** = Max(0, 1 - (Peek Memory_GB / Baseline_Memory={base_mem:.2f} GB))"
            f"{co2_benefit_note}"
        )

        # ---------------------- II. Prompt 內容 ----------------------

        prompt = f"""You are an expert hyperparameter optimization assistant. Your goal is to find the best KV cache compression settings for {task_desc}.

    ## Baseline Performance (No Compression)
    {metric_name}: {self.baseline_score:.4f}
    Latency: {self.baseline_latency:.2f} s
    Peek Memory: {self.baseline_memory:.2f} GB

    ## Objective (Standardized)
    The primary goal is to optimize the trade-off between model quality and performance efficiency by **maximizing the total Benefit (R)**.
    1. **Model Quality (Primary):** Maximize {metric_name} ({self.accuracy_weight:.0%} weight).
    2. **Inference Speed (Crucial):** Maximize Latency_Benefit.
    3. **Hardware Cost (Crucial):** Maximize Memory_Benefit.
    4. **Environmental Cost (Important):** Maximize CO2_Benefit.

    {score_formula}
    {benefit_notes}

    ## Available Hyperparameters

    1. **compression_method** (categorical):
        - snapkv: Attention-based, keeps recent + important tokens
        - observed_attention: More accurate attention-based (slower)
        - knorm: Magnitude-based, fastest
        - expected_attention: Expected attention patterns

    2. **compression_ratio** (float, 0.2-0.9):
        - Fraction of tokens to KEEP (higher = more tokens, better quality, higher cost)
        - Typically 0.3-0.7 works best
    """
        # ---------------------- III. MultiNews 條件 ----------------------

        if self.task == "multinews":
            prompt += """
    3. **max_input_tokens** (int, 2048-6144):
        - Truncate long documents
        - Lower = faster but may lose info

    """
            
        # ---------------------- IV. 歷史紀錄修正 ----------------------
        
        # Add trial history
        if self.trial_history:
            prompt += f"""## Previous Trials (Iteration {iteration}/{self.max_iterations if hasattr(self, 'max_iterations') else 'N/A'})

    """
            # Show last 5 trials
            recent_trials = self.trial_history[-5:]
            for i, trial in enumerate(recent_trials, 1):
                params = trial['params']
                score = trial['score']
                acc = trial['accuracy']
                # CO2 轉換為克 (g)
                co2 = trial.get('co2_kg', 0.0) * 1000  
                latency = trial['latency']
                # *** 關鍵修正: 保持使用 memory_gb 鍵值 ***
                memory = trial.get('memory_gb', trial.get('memory', 0.0)) # 優先使用 memory_gb，否則嘗試 memory
                
                # 假設 params['compression_ratio'] 總是存在
                ratio_display = params.get('compression_ratio', 0.0) 
                
                prompt += f"""Trial {trial['iteration']}:
      Config: method={params['compression_method']}, ratio={ratio_display:.2f}"""
                
                # 處理可選參數 max_input_tokens
                if self.task == "multinews":
                    prompt += f", max_tokens={params.get('max_input_tokens', 'N/A')}"

                prompt += f"""
      Results: {metric_name}={acc:.4f} | Latency={latency:.2f}s | Memory={memory:.2f}GB | CO2={co2:.2f}g | Score={score:.4f}
      Analysis: """ # 這裡應該是讓 LLM 進行分析
                
                prompt += "\n\n"
            
            # Show best so far
            best = max(self.trial_history, key=lambda x: x['score'])
            prompt += f"""🏆 Best so far (Trial {best['iteration']}): Score={best['score']:.4f}
      Config: {best['params']}
    """
            
        else:
            prompt += """## Previous Trials
    No trials yet. This is the first iteration.

    ## Suggestions for First Trial
    - Start with snapkv (good balance)
    - Try compression_ratio around 0.5-0.6


    """
        # ---------------------- V. 任務請求 ----------------------
        
        # Request for next configuration
        prompt += """## Your Task
            Based on the trials above, suggest the NEXT configuration to try.
            ## Exploration Encouragement
            - Try to explore methods that have not been used in previous trials.
            - Avoid always picking the same method unless you have strong evidence it works best.
            - You should balance exploration and exploitation.

            Return a JSON object with these fields:
            - reasoning: Why this config (2-3 sentences)
            - compression_method: One of [snapkv, observed_attention, knorm, expected_attention]
            - compression_ratio: Float between 0.2 and 0.9
            """

        if self.task == "multinews":
            prompt += """
            - max_input_tokens: Integer between 2048-6144 (optional)"""

        prompt += """

    [Output ONLY JSON]
    [no prose]
    Do not include any prefix like "(Valid)" or "Output:".
    Your output will be parsed directly with `json.loads()`. 
    Here is the expected format:

    {
      "reasoning": "brief reasoning",
      "compression_method": "one of [snapkv, observed_attention, knorm, expected_attention]",
      "compression_ratio": 0.2-0.9
    }

    Example:
    {"reasoning": "snapkv gives balanced accuracy and cost", "compression_method": "snapkv", "compression_ratio": 0.6}
    """
        return prompt
    
    def _parse_llm_suggestion(self, llm_response: str) -> Optional[Dict]:
        """Parse LLM's JSON suggestion with Pydantic validation"""
        import re, json

        # --- unified cleaning ---
        json_str = llm_response.strip()
        # remove common noise like (Valid), Output:, or markdown
        json_str = re.sub(r'^\(?[Vv]alid\)?', '', json_str).strip()
        json_str = re.sub(r'^(Output|Result|Response)\s*[:：-]?', '', json_str, flags=re.I).strip()
        json_str = re.sub(r'```(?:json)?|```', '', json_str).strip()
        json_str = re.sub(r'^[^({\[]*', '', json_str, count=1).strip()  # remove non-JSON prefix
        json_str = re.sub(r'[^)}\]]*$', '', json_str, count=1).strip()  # remove trailing junk

        
        # For local models, extract JSON from markdown
        if not (USE_API and API_AVAILABLE):
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0).strip()
                else:
                    logger.error("No JSON found in local model response")
                    return None
        
        try:
            # Parse with Pydantic
            config_obj = CompressionConfig.model_validate_json(json_str)
            suggestion = config_obj.model_dump(exclude_none=True)
            
            # Validate method
            valid_methods = ['snapkv', 'observed_attention', 'knorm', 'expected_attention', 'streaming_llm']
            if suggestion['compression_method'] not in valid_methods:
                logger.error(f"Invalid method: {suggestion['compression_method']}")
                return None
            
            logger.info(f"LLM reasoning: {suggestion.get('reasoning', 'N/A')}")
            
            # Remove reasoning, keep only params
            suggestion.pop('reasoning', None)
            
            return suggestion
            
        except (ValidationError, json.JSONDecodeError) as e:
            logger.error(f"Validation error: {e}")
            logger.error(f"Attempted to parse: {json_str[:300]}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_llm_suggestion(self, iteration: int) -> Optional[Dict]:
        """Get hyperparameter suggestion from LLM"""

        prompt = self._create_optimization_prompt(iteration)
        
        logger.info("Asking LLM for next configuration...")
        
        try:
            if USE_API and API_AVAILABLE and self.api_client:
                # Use system message to force JSON output
                response = self.api_client.chat.completions.create(
                    model=API_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that outputs ONLY valid JSON. Do not include any text before or after the JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=512,
                    response_format={"type": "json_object"}
                )
                llm_response = response.choices[0].message.content
                
            else:
                # Local model
                llm_response = self.llm_model.generate_response(
                    prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True
                )
            # print("=== LLM RAW RESPONSE START ===")
            # print(llm_response)
            # print("=== LLM RAW RESPONSE END ===")

            if not llm_response:
                logger.error("LLM returned empty response")
                return None
            
            logger.debug(f"LLM response: {llm_response[:200]}...")
            suggestion = self._parse_llm_suggestion(llm_response)
            return suggestion
            
        except Exception as e:
            logger.error(f"Error getting LLM suggestion: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _apply_config(self, config: Dict):
        """Apply configuration safely without allocating new KVPress"""
        kv_conf = self.evaluator.config.setdefault('kv_compression', {})
        kv_conf['compression_method'] = config['compression_method']
        kv_conf['compression_ratio'] = config['compression_ratio']

        # Update dataset config
        if 'max_input_tokens' in config and config['max_input_tokens'] is not None:
            self.evaluator.config.setdefault('dataset', {})['max_input_tokens'] = config['max_input_tokens']

        # MEMORY-SAFE: only update agent params, no new allocations
        agent = self.evaluator.agent
        agent.set_method(config['compression_method'])
        agent.set_compression_ratio(config['compression_ratio'])


    def _evaluate_config(self, config: Dict, iteration: int) -> Dict:
        """Evaluate a configuration and return metrics"""
        import torch, gc

        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration}/{self.max_iterations}")
        logger.info(f"Testing config: {config}")
        logger.info(f"{'='*60}")

        # MEMORY SAFE RESET
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Decide dataset size (partial/full)
        evaluator_cfg = self.evaluator.config['dataset']
        if iteration <= self.max_iterations - 2:
            evaluator_cfg['num_samples'] = 1  # partial
        else:
            evaluator_cfg['num_samples'] = self.full_num_samples  # full

        # Apply configuration
        self._apply_config(config)

        # Track CO2 if needed
        if self.emissions_tracker:
            self.emissions_tracker.start()

        try:
            results = self.evaluator.run_evaluation()
            summary = results['summary']

            # Extract accuracy metric
            if self.task == 'gsm8k':
                accuracy = summary['results']['accuracy']
            else:
                accuracy = summary['results'].get('avg_rougeL_f', 0)
                accuracy = min(accuracy * 2, 1.0)

            # CO2 metric
            if self.emissions_tracker:
                self.emissions_tracker.stop()
                co2_kg = self.emissions_tracker.final_emissions
            else:
                co2_kg = 0

            # Extract memory & latency metric
            memory=summary['memory_tracking'].get('peak_allocated_gb', 0)
            latency=summary['performance'].get('avg_time', 0)
            memory_normalized = max(0, 1 - (memory / self.baseline_memory))
            latency_normalized = max(0, 1 - (latency / self.baseline_latency))

            # Compute score (don't normalize CO2 if not tracking)
            if self.enable_co2_tracking and co2_kg > 0:
                co2_normalized = max(0, 1 - (co2_kg / self.baseline_co2))
                score = self.accuracy_weight * accuracy + self.co2_weight * co2_normalized + self.memory_weight * memory_normalized + self.latency_weight * latency_normalized
            else:
                # Just use accuracy if no CO2 tracking
                score = self.accuracy_weight *accuracy + self.memory_weight * memory_normalized + self.latency_weight * latency_normalized

            trial_result = {
                'iteration': iteration,
                'params': config,
                'accuracy': accuracy,
                'memory_gb':memory,
                'latency':latency,
                'co2_kg': co2_kg,
                'score': score,
                'timestamp': datetime.now().isoformat()
            }
            return trial_result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            if self.emissions_tracker:
                self.emissions_tracker.stop()
            return {
                'iteration': iteration,
                'params': config,
                'accuracy': 0,
                'co2_kg': 0,
                'score': 0,
                'error': str(e)
            }


    def optimize(self) -> Dict:
        """LLM-guided tuning, memory-safe like grid search"""
        import torch, gc

        # BASELINE (no compression)
        logger.info("Getting baseline performance (no compression)...")
        self.evaluator.config['kv_compression']['enabled'] = False
        # Track CO2 if needed
        if self.emissions_tracker:
            self.emissions_tracker.start()
        baseline_result = self.evaluator.run_evaluation()
        if self.emissions_tracker:
            self.emissions_tracker.stop()
            self.baseline_co2 = self.emissions_tracker.final_emissions
        else:
            self.baseline_co2 = 0
        self.evaluator.config['kv_compression']['enabled'] = True

        self.baseline_score = (
            baseline_result['summary']['results'].get('accuracy', 0)
            if self.task == 'gsm8k'
            else baseline_result['summary']['results'].get('avg_rougeL_f', 0)
        )

        self.baseline_latency=baseline_result['summary']['performance'].get('avg_time', 0)
        self.baseline_memory=baseline_result['summary']['memory_tracking'].get('peak_allocated_gb', 0)


        logger.info(f"Baseline accuracy: {self.baseline_score:.4f}")

        # ITERATIVE LLM-GUIDED TUNING
        for iteration in range(1, self.max_iterations + 1):
            # MEMORY SAFE RESET
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # LLM suggestion
            suggestion = self._get_llm_suggestion(iteration)
            if suggestion is None:
                logger.warning(f"LLM suggestion failed for iteration {iteration}, using fallback")
                suggestion = self._get_fallback_config()

            # Evaluate configuration
            trial_result = self._evaluate_config(suggestion, iteration)

            self.trial_history.append(trial_result)

            # Update best
            if trial_result['score'] > self.best_score:
                self.best_score = trial_result['score']
                self.best_config = trial_result
                logger.info(f"🎯 New best! Score: {self.best_score:.4f}")
            else:
                logger.info(f"Score: {trial_result['score']:.4f}")

        logger.info(f"\n{'='*60}")
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best config: {self.best_config['params']}")
        logger.info(f"{'='*60}\n")

        # FINAL FULL EVAL
        self._final_full_eval()
        return {
            'best_config': self.best_config,
            'trial_history': self.trial_history,
            'task': self.task
        }

    
    def _get_fallback_config(self) -> Dict:
        """Fallback configuration if LLM fails"""
        if self.task == "gsm8k":
            return {
                'compression_method': 'snapkv',
                'compression_ratio': 0.5,
                'window_size': 32,
                'kernel_size': 5
            }
        else:
            return {
                'compression_method': 'snapkv',
                'compression_ratio': 0.6,
                'window_size': 64,
                'kernel_size': 7,
                'max_input_tokens': 4096
            }
    
    def save_results(self, results: Dict, output_path: str):
        """Save optimization results"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_path}")

    def _final_full_eval(self):
        if not self.trial_history:
            return
        
        # Find best config from all trials
        best_trial = max(self.trial_history, key=lambda t: t['score'])
        best_config = best_trial['params']

        logger.info(f"Re-evaluating best config on full dataset: {best_config}")

        # Force full dataset
        evaluator_cfg = self.evaluator.config['dataset']
        evaluator_cfg['num_samples'] = self.full_num_samples   # full dataset

        # Apply and evaluate
        self._apply_config(best_config)
        # Track CO2
        if self.emissions_tracker:
            self.emissions_tracker.start()
        
        try:
            # Run evaluation
            results = self.evaluator.run_evaluation()
            summary = results['summary']
            
            # Extract metrics
            if self.task == 'gsm8k':
                accuracy = summary['results']['accuracy']
                # Normalize to 0-1 (already normalized)
            else:
                accuracy = summary['results'].get('avg_rougeL_f', 0)
                # ROUGE is typically 0-0.5, normalize to 0-1
                accuracy = min(accuracy * 2, 1.0)  # Scale up ROUGE
            
            # Get CO2
            if self.emissions_tracker:
                self.emissions_tracker.stop()
                co2_kg = self.emissions_tracker.final_emissions
            else:
                co2_kg = 0

            # get memory & latency
            memory=summary['memory_tracking'].get('peak_allocated_gb', 0)
            latency=summary['performance'].get('avg_time', 0)
            memory_normalized = max(0, 1 - (memory / self.baseline_memory))
            latency_normalized = max(0, 1 - (latency / self.baseline_latency))

            # Compute score (don't normalize CO2 if not tracking)
            if self.enable_co2_tracking and co2_kg > 0:
                co2_normalized = max(0, 1 - (co2_kg / self.baseline_co2))
                score = self.accuracy_weight * accuracy + self.co2_weight * co2_normalized + self.memory_weight * memory_normalized + self.latency_weight * latency_normalized
            else:
                # Just use accuracy if no CO2 tracking
                score = self.accuracy_weight *accuracy + self.memory_weight * memory_normalized + self.latency_weight * latency_normalized

            
            trial_result = {
                'iteration': 99,
                'params': best_config,
                'accuracy': accuracy,
                'memory_gb':memory,
                'latency':latency,
                'co2_kg': co2_kg,
                'score': score,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Final evaluation: Accuracy={accuracy:.4f}, CO2={co2_kg*1000:.2f}g, Score={score:.4f}")
            
            return trial_result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            if self.emissions_tracker:
                self.emissions_tracker.stop()
            
            return {
                'iteration': 99,
                'params': best_config,
                'accuracy': 0,
                'co2_kg': 0,
                'score': 0,
                'error': str(e)
            }

