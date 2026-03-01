##待修正，改成只需要build_asvd_repo了

"""
ASVD-based Adaptive Tuner
Uses LLM to suggest hyperparameters based on previous trial feedback
Inspired by OPRO (Optimization by PROmpting) and similar work

ASVD 優化的超參數（asvd.py 和 build_asvd_repo.py 的重疊參數）:
- alpha: 0.3-0.7 (ASVD的主要超參數)
- param_ratio_target: 0.5-0.95 (目標參數壓縮比)
- scaling_method: abs_mean, abs_max, fisher (注意：build_repo支持fisher_abs_mean但asvd.py不支持)
- n_calib_samples: 32-128 (校準樣本數)
- calib_dataset: wikitext2, c4, ptb (校準數據集)  ##c4太大暫時刪除
- ppl_target: 困惑度目標（可選）

Note: ASVD 和 KVPress 是獨立的系統，tuner 只優化 ASVD 參數
"""

import logging
import os
import sys
import torch
import json
import subprocess
import traceback
import random
import gc
import time
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, ValidationError

# 移除環境變數避免衝突
os.environ.pop("PYTORCH_ALLOC_CONF", None)

from dotenv import load_dotenv
# Load .env from project root (same as llm_tuner.py)
# This allows sharing one .env file for both ASVD and KVPress systems
load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    logger.warning("codecarbon not available. Install: pip install codecarbon")

# 檢查是否使用 API
USE_API = os.getenv("USE_LLM_API", "false").lower() == "true"
API_KEY = os.getenv("LLM_API_KEY", "")
API_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# 本地 LLM 模型設定（如果不使用 OpenAI API）
# 可通過以下環境變數或 .env 文件設定本地 LLM 模型
LOCAL_LLM_MODEL_PATH = os.getenv("LOCAL_LLM_MODEL_PATH", None)


def _deep_clean_dict(data: Any) -> Any:
    """遞迴轉換 PyTorch Tensors 為原生 Python 類型"""
    if isinstance(data, dict):
        return {k: _deep_clean_dict(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_deep_clean_dict(v) for v in data]
    elif isinstance(data, torch.Tensor):
        if data.numel() == 1:
            return data.item()
        else:
            return data.cpu().numpy().tolist()
    elif hasattr(data, 'item') and callable(data.item):
        return data.item()
    else:
        return data


# ============================================================
# Pydantic 模型定義
# ============================================================
class ASVDConfig(BaseModel):
    """ASVD 超參數配置結構 - 僅包含 asvd.py 和 build_asvd_repo.py 的重疊參數"""
    
    # LLM 的分析部分 (會被解析但從參數中移除)
    reasoning: str = Field(..., description="Deep logical analysis (5-8 sentences) explaining the trade-off between alpha and compression ratio for GSM8K.")
    
    # 核心超參數（必填）
    alpha: float = Field(..., ge=0.3, le=0.7, description="ASVD 主要超參數 (0.3-0.7).")
    param_ratio_target: float = Field(..., ge=0.5, le=0.95, description="Target parameter ratio (0.5-0.95).")
    
    # 可選參數（有默認值）
    scaling_method: str = Field(default="abs_mean", 
                                description="One of: abs_mean, abs_max, fisher (注意：兩個腳本都支持)")
    n_calib_samples: int = Field(default=32,
                                description="Number of calibration samples (typically 32-128).")
    calib_dataset: str = Field(default="wikitext2",
                              description="One of: wikitext2,  ptb.")


if USE_API:
    try:
        from openai import OpenAI
        API_AVAILABLE = True
    except ImportError:
        logger.warning("openai package not available. Will use local LLM fallback.")
        API_AVAILABLE = False
else:
    API_AVAILABLE = False
    logger.info("USE_LLM_API is false. Will use local LLM fallback.")


class ASVDAdaptiveTuner:
    """
    LLM 基礎的 ASVD 適應性調整器
    LLM 作為優化算法！
    
    支持動態樣本數：
    - 前幾次迭代用少量樣本快速評估
    - 最後幾次迭代用全部樣本做完整評估
    """
    
    def __init__(self,
                 evaluator,
                 model_id: str = "facebook/opt-1.3b",
                 task: str = "gsm8k",
                 max_iterations: int = 20,
                 accuracy_weight: float = 0.7,
                 compression_weight: float = 0.2,
                 ppl_weight: float = 0.1,
                 enable_co2_tracking: bool = False,
                 llm_model: str = None,
                 num_samples_early: Optional[List[int]] = None,
                 num_samples_final: Optional[int] = None,
                 baseline_acc: float = 0.058,
                 baseline_ppl: float = 15.0):
        """
        Args:
            evaluator: GSM8KEvaluator or other evaluator instance
            model_id: HuggingFace model ID to compress
            task: 'gsm8k', 'multinews', etc.
            max_iterations: Maximum optimization iterations
            accuracy_weight: Weight for accuracy/task performance
            compression_weight: Weight for compression ratio
            ppl_weight: Weight for PPL/language modeling performance
            enable_co2_tracking: Track CO2 emissions
            llm_model: LLM model name (uses GPT-4o if None)
            num_samples_early: List of sample counts for early iterations
                              e.g., [10, 20, 50] means:
                              - Iteration 1: 10 samples
                              - Iteration 2: 20 samples
                              - Iteration 3: 50 samples
                              - Rest: use evaluator's default
            num_samples_final: Number of samples for final iteration (None = all samples)
            baseline_acc: Baseline accuracy of original model
            baseline_ppl: Baseline PPL of original model
        """
        self.evaluator = evaluator
        self.model_id = model_id
        self.task = task
        self.max_iterations = max_iterations
        self.accuracy_weight = accuracy_weight
        self.compression_weight = compression_weight
        self.ppl_weight = ppl_weight
        self.enable_co2_tracking = enable_co2_tracking
        self.baseline_acc = baseline_acc
        self.baseline_ppl = baseline_ppl
        
        # 設定本地 LLM：優先用傳入參數，否則使用 evaluator 的 agent
        self.llm_model = llm_model or getattr(evaluator, 'agent', None)
        if self.llm_model is None:
            logger.warning("No local LLM model available (evaluator.agent not found). Will use fallback config if API fails.")
        
        # 動態樣本數配置
        self.num_samples_early = num_samples_early or []
        self.num_samples_final = num_samples_final
        self.original_num_samples = None  # 保存原始設置
        
        # 試驗歷史
        self.trial_history: List[Dict] = []
        self.baseline_score = None
        
        # API 客戶端初始化
        self.api_client = None
        if USE_API and API_AVAILABLE and API_KEY:
            try:
                self.api_client = OpenAI(api_key=API_KEY)
                logger.info(f"OpenAI API client initialized for model: {API_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                logger.info("Will use local LLM fallback instead.")
                self.api_client = None
        
        # CO2 追蹤
        self.co2_tracker = None
        if CODECARBON_AVAILABLE and enable_co2_tracking:
            self.co2_tracker = EmissionsTracker(country_iso_code="US")
        
        logger.info(f"ASVDAdaptiveTuner initialized for model: {model_id}")
        logger.info(f"Task: {task}, Max iterations: {max_iterations}")
        logger.info(f"LLM mode: {'OpenAI API' if self.api_client else 'Local LLM fallback'}")
        if self.num_samples_early:
            logger.info(f"Early iterations sample counts: {self.num_samples_early}")
        if self.num_samples_final is not None:
            logger.info(f"Final iteration sample count: {self.num_samples_final}")
        elif self.num_samples_final is None and self.num_samples_early:
            logger.info("Final iteration will use all available samples")
    
    def _set_evaluator_samples(self, iteration: int):
        """
        根据迭代次数動態設置評估器的樣本數
        
        Args:
            iteration: 當前迭代次數（1-based）
        """
        # 保存原始值（首次調用時）
        if self.original_num_samples is None:
            if hasattr(self.evaluator.config, 'num_samples'):
                self.original_num_samples = self.evaluator.config.num_samples
        
        # 確定本次迭代應使用的樣本數
        if iteration <= len(self.num_samples_early) and self.num_samples_early:
            # 在 early iterations 列表中
            num_samples = self.num_samples_early[iteration - 1]
            logger.info(f"[Iteration {iteration}] Setting num_samples={num_samples} (early iteration)")
            
        elif iteration == self.max_iterations and self.num_samples_final is not None:
            # 最後一次迭代：使用指定的最終樣本數
            num_samples = self.num_samples_final
            logger.info(f"[Iteration {iteration}] Setting num_samples={num_samples} (final iteration)")
            
        elif iteration == self.max_iterations and self.num_samples_early:
            # 最後一次迭代且沒指定 num_samples_final：使用全部
            num_samples = None  # None 表示全部
            logger.info(f"[Iteration {iteration}] Using all available samples (final iteration)")
            
        else:
            # 其他情況：使用原始設置
            num_samples = self.original_num_samples
        
        # 應用設置
        if num_samples is not None:
            if hasattr(self.evaluator.config, 'num_samples'):
                self.evaluator.config.num_samples = num_samples
            elif hasattr(self.evaluator, 'num_samples'):
                self.evaluator.num_samples = num_samples
    
#     def _create_optimization_prompt(self, iteration: int) -> str:
#         """
#         為 LLM 創建超參數建議提示
#         基於 OPRO 風格提示
#         """
#         prompt = f"""You are an expert hyperparameter optimization assistant for ASVD (Activation-aware Singular Value Decomposition).
# Your goal is to find the best ASVD compression settings for the {self.task.upper()} task on {self.model_id}.

# ## ASVD Compression Objectives
# - Maximize: {self.accuracy_weight:.0%} Task Performance + {self.compression_weight:.0%} Compression Ratio + {self.ppl_weight:.0%} Language Model Quality
# - Minimize: Model parameters while maintaining performance

# ## Available Hyperparameters

# These are the parameters that work with both asvd.py and build_asvd_repo.py:

# 1. **alpha** (float, 0.3-0.7):
#    - Main ASVD hyperparameter
#    - Lower alpha: More aggressive compression
#    - Higher alpha: Better quality preservation
#    - Suggested range: 0.4-0.6

# 2. **param_ratio_target** (float, 0.5-0.95):
#    - Target parameter ratio after compression
#    - 0.5 = 50% compression (keep 50% of parameters)
#    - 0.8 = 20% compression (keep 80% of parameters)
#    - Suggested range: 0.6-0.9 for good quality

# 3. **scaling_method** (categorical):
#    - abs_mean: Activation-aware mean (default, recommended)
#    - abs_max: Activation-aware max
#    - fisher: Fisher information based
#    - If accuracy drops too much on logical tasks, try 'fisher' to better identify important weights.
#    - Note: build_asvd_repo also supports fisher_abs_mean, but asvd.py does not

# 4. **n_calib_samples** (int, typically 16-32):
#    - Number of samples used for calibration
#    - More samples = better calibration but **MUCH SLOWER**
#    - Keep it ≤ 32 for quick iterations, use 32-64 only for final evaluation
#    - Default: 32 (good balance)

# 5. **calib_dataset** (categorical):
#    - wikitext2: Common benchmark dataset (default)
#    - ptb: Smaller dataset
# """
# #         prompt += f"""
# # ## 🏆 Oracle Reference (Baseline)
# # Before compression, the original model performs as follows:
# # - Baseline Task Accuracy: {self.baseline_acc} (Approx {self.baseline_acc*100:.1f}%)
# # - Baseline Language Quality (PPL): {self.baseline_ppl} (Lower is better)

# # ## 💡 Optimization Strategy
# # - Your goal is to keep Accuracy close to or potentially higher than {self.baseline_acc}, while minimizing 'param_ratio_target'.
# # - A significant drop in Accuracy (e.g., more than 30% relative to baseline) indicates that 'alpha' is too low or 'param_ratio_target' is too aggressive, changing Scaling Method could also help.
# # - If PPL increases significantly compared to {self.baseline_ppl}, the model is losing its linguistic coherence.
# # """
#         if self.trial_history:
#             prompt += f"\n## Previous Trials (Iteration {iteration}/{self.max_iterations})\n\n"
            
#             # Show last 5 trials
#             recent_trials = self.trial_history[-5:]
#             for i, trial in enumerate(recent_trials, 1):
#                 params = trial['params']
#                 score = trial['score']
#                 accuracy = trial['accuracy']
#                 compression_ratio = trial.get('compression_ratio', 'N/A')
#                 ppl = trial.get('ppl', 'N/A')
                
#                 prompt += f"""Trial {trial['iteration']}:
#   Config:
#     alpha={params['alpha']:.2f}
#     param_ratio={params['param_ratio_target']:.2f}
#     scaling={params['scaling_method']}
#     n_calib_samples={params.get('n_calib_samples', 32)}
#     calib_dataset={params.get('calib_dataset', 'wikitext2')}
#   Results:
#     Score={score:.4f}
#     Task Accuracy={accuracy:.4f}

# """
            
#             # Show best so far
#             best = max(self.trial_history, key=lambda x: x['score'])
#             prompt += f"""🏆 Best Configuration so far (Trial {best['iteration']}):
#   Score: {best['score']:.4f}
#   Config: alpha={best['params']['alpha']:.2f}, param_ratio={best['params']['param_ratio_target']:.2f}

# """
#         else:
#             prompt += """
# ## Previous Trials
# No trials yet. This is the first iteration.

# ## Suggestions for First Trial
# - Start with alpha=0.5 (moderate compression)
# - Try param_ratio_target=0.8 (20% compression)
# - Use default method: abs_mean for scaling
# - Use n_calib_samples=32 for quick calibration

# """
        
#         prompt += """## Your Task
# Based on the trials above, suggest the NEXT configuration to try.

# ## Guidelines
# - Balance exploration and exploitation
# - If all trials use same method, try different approaches
# - High compression (low param_ratio) + low alpha may degrade performance
# - For first trials, keep configurations conservative

# Return a JSON object with these fields:
# - reasoning: Why this config (2-3 sentences)
# - alpha: Float between 0.3 and 0.7
# - param_ratio_target: Float between 0.5 and 0.95
# - scaling_method: One of [abs_mean, abs_max, fisher]
# - n_calib_samples: Integer between 32 and 128
# - calib_dataset: One of [wikitext2, ptb]

# [Output ONLY JSON]
# [no prose]
# Do not include any prefix like "(Valid)" or "Output:".
# Your output will be parsed directly with `json.loads()`.

# Example:
# {
#   "reasoning": "Try moderate compression with conservative alpha to preserve quality",
#   "alpha": 0.5,
#   "param_ratio_target": 0.8,
#   "scaling_method": "abs_mean",
#   "n_calib_samples": 32,
#   "calib_dataset": "wikitext2"
# }
# """
        
#         return prompt
    
    def _create_optimization_prompt(self, iteration: int) -> str:
        """
        為 LLM 創建超參數建議提示 - 深度詳解參數版
        """
        prompt = f"""You are an expert hyperparameter optimization assistant for ASVD (Activation-aware Singular Value Decomposition).
    Your goal is to find the best ASVD compression settings for the **GSM8K** (Logical Reasoning) task on {self.model_id}.

    ### 🧩 Task Sensitivity: GSM8K
    GSM8K requires multi-step logical consistency. High compression ratios often lead to "logical collapse," where the model maintains grammar but loses mathematical precision. Your priority is to maintain accuracy while exploring compression boundaries.

    ### 🛠 Deep Parameter Insights (Maintain these constraints)

    1. **alpha** (float, 0.3-0.7): 
    - **Role:** Controls the sensitivity to activation magnitudes. 
    - **Mechanism:** A higher alpha (e.g., 0.3-0.7) puts more weight on preserving high-activation features, which are often critical for "knowledge" and "logic". 
    - **Tuning Intuition:** If GSM8K accuracy drops significantly but PPL remains stable, alpha is likely too low to protect logic-critical singular values.

    2. **param_ratio_target** (float, 0.5-0.95): 
    - **Role:** Defines the budget of remaining parameters.
    - **Mechanism:** Directly determines the rank truncation in SVD. 0.8 means keeping 80% of parameters.
    - **Tuning Intuition:** This is your primary "aggression" dial. Start conservatively (0.90+) to find a "safe zone" before pushing for higher compression.

    3. **scaling_method** (categorical):
    - **abs_mean:** (Default) Standard activation-aware scaling. Good for general linguistic stability.
    - **abs_max:** More sensitive to outliers in activations. Can be risky if activations are noisy.
    - **fisher:** Uses Fisher Information to weight the importance of parameters based on the loss gradient. 
    - **Tuning Intuition:** For complex tasks like GSM8K, 'fisher' is often the most robust as it prioritizes weights that actually impact the model's output quality.

    4. **n_calib_samples** (int, 32-128):
    - **Role:** Sample size for calculating activation statistics.
    - **Trade-off:** More samples lead to more stable SVD decomposition but significantly increase computation time. 
    - **Constraint:** Use 32 for exploration; only increase if you suspect the calibration is "noisy".

    5. **calib_dataset** [wikitext2, ptb]:
    - **Role:** The distribution used to "warm up" the activations. 
    - **Strategy:** wikitext2 is generally more diverse and recommended for keeping the model's general intelligence intact.

    ### 📈 Objectives & Trial History (Iteration {iteration}/{self.max_iterations})
    - **Objective:** Maximize {self.accuracy_weight:.0%} Accuracy + {self.compression_weight:.0%} Ratio + {self.ppl_weight:.0%} PPL.
    """

        if self.trial_history:
            prompt += "\n### 📜 Previous Trials & Reasoning Evolution\n"
            recent_trials = self.trial_history[-5:]
            for trial in recent_trials:
                params = trial['params']
                prev_reasoning = trial.get('reasoning', 'No previous reasoning recorded.')
                prompt += f"""
    Trial {trial['iteration']}:
    - Reasoning: {prev_reasoning}
    - Config: alpha={params['alpha']:.3f}, ratio={params['param_ratio_target']:.3f}, method={params['scaling_method']}
    - Results: Accuracy={trial['accuracy']:.4f}, Score={trial['score']:.4f}, Avg_PPL={trial.get('ppl', {}).get('avg', 'N/A')}
    """
            best = max(self.trial_history, key=lambda x: x['score'])
            prompt += f"\n🏆 Current Best: Trial {best['iteration']} (Score: {best['score']:.4f})\n"

            last_trial = self.trial_history[-1]
            if last_trial['accuracy'] < best['accuracy'] * 0.8:
                # 情況 A：上次搞砸了 -> 強制恢復
                prompt += f"\n⚠️ ALERT: Last trial FAILED (Acc dropped significantly). DO NOT decrease ratio. Fix it by increasing Ratio(preferred), Alpha or changing Method.\n"
            elif last_trial['accuracy'] >= best['accuracy'] * 0.98:
                # 情況 B：效能很穩 -> 鼓勵突破，不要停留在原位
                prompt += f"\n💡 STRATEGY: Performance is solid. You are ENCOURAGED to push the ratio 2-3% lower, but you must justify your choice.\n"

        else:
            prompt += "\n### 🚀 Initial Strategy: Start with 0.90-0.95 ratio to establish a performance ceiling.\n"

        prompt += """
    ### 🎯 Your Task
    Suggest the NEXT configuration using a **4-6 sentence reasoning** process that MUST address: 
    1. Comparison between Trial {iteration-1} and the Best Trial.
    2. Why the previous choice succeeded or failed.
    3. Your justification for increasing or decreasing the aggression (ratio).
    Analyze the trade-off between the 'alpha' protection layer and the 'param_ratio' budget.

    ### 🔎 Trend Analysis & Guardrails
    - **Performance Floor:** If Accuracy drops below 0.1 (10%) or significant logical collapse occurs (compared to your best trial), you MUST immediately increase `param_ratio_target`(preferred) or `alpha`. 
    - **Learning from Failure:** Analyze why previous low-ratio trials failed. If a ratio of 0.7 resulted in 0% accuracy, do NOT suggest 0.65 or 0.7. Instead, pivot back to a known "safe zone" (e.g., 0.90) and adjust `alpha` or `scaling_method` first.
    - **Strict Logic:** If the current trial's accuracy is worse than the best trial, your next suggestion should prioritize "Exploitation" (recovering performance) over "Exploration" (pushing compression), such as increase `alpha` (to protect remaining weights), increase ratio (the most significant parameter) or switch to a more robust `scaling_method` (like `fisher`).

    ### 🧭 Exploration Protocol
    - **Phased Approach:** 
        - **Phase 1 (Early Trials):** Map the "Safe Zone" by testing different `scaling_method` and `alpha` at midium ratios.
        - **Phase 2 (Discovery):** Only when a stable method is found, incrementally push `param_ratio_target` down (e.g., steps of 0.05).
    - **The Recovery Pivot:** If a trial results in a significant Accuracy drop (>20% relative loss), the next trial MUST NOT decrease the ratio further. Instead, it must either:
        1. Revert to the last "Safe Ratio" and try a different `scaling_method`.
        2. Increase `alpha` significantly to see if the current ratio can be "saved" by better protection.
    - **Diversity over Repetition:** Avoid testing the same `param_ratio_target` with the same `scaling_method` if it has already failed.

    Return a JSON object with these fields:
    - reasoning: Why this config (2-3 sentences)
    - alpha: Float between 0.3 and 0.7
    - param_ratio_target: Float between 0.5 and 0.99
    - scaling_method: One of [abs_mean, abs_max, fisher]
    - n_calib_samples: Integer between 32 and 128
    - calib_dataset: One of [wikitext2, ptb]  
    
    [Output ONLY JSON]
    [no prose]
    Do not include any prefix like "(Valid)" or "Output:".
    Your output will be parsed directly with `json.loads()`.
    
    Example:
    {
    "reasoning": "Detailed 5-8 sentence analysis goes here...",
    "alpha": 0.6,
    "param_ratio_target": 0.90,
    "scaling_method": "fisher",
    "n_calib_samples": 32,
    "calib_dataset": "wikitext2"
    }
    """
        return prompt
    def _parse_llm_suggestion(self, llm_response: str) -> Optional[Dict]:
        """
        解析 LLM 的 JSON 回應
        """
        try:
            # 嘗試找到 JSON 物件
            start = llm_response.find('{')
            end = llm_response.rfind('}') + 1
            
            if start == -1 or end == 0:
                logger.error(f"No JSON found in LLM response: {llm_response}")
                return None
            
            json_str = llm_response[start:end]
            config_dict = json.loads(json_str)
            
            # 驗證配置
            config = ASVDConfig(**config_dict)
            
            # 轉換為字典
            result = config.model_dump()
            reasoning = result.get('reasoning', 'No reasoning provided')
            logger.info(f"LLM Reasoning: {reasoning}")
            
            return result
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Response: {llm_response}")
            return None
    
    def _get_llm_suggestion(self, iteration: int) -> Optional[Dict]:
        """
        從 LLM 獲得超參數建議
        支持 OpenAI API 或本地 LLM fallback（類似 llm_tuner.py 的做法）
        """
        prompt = self._create_optimization_prompt(iteration)
        logger.info("Asking LLM for next configuration...")
        
        try:
            if self.api_client and USE_API and API_AVAILABLE:
                # 使用 OpenAI API
                logger.debug(f"Using OpenAI API (model: {API_MODEL})")
                response = self.api_client.chat.completions.create(
                    model=API_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a hyperparameter optimization expert. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                llm_response = response.choices[0].message.content
                
            elif self.llm_model:
                # 使用本地 LLM（fallback）
                logger.debug("Using local LLM fallback")
                if hasattr(self.llm_model, 'generate_response'):
                    # 假設本地模型有 generate_response 方法
                    llm_response = self.llm_model.generate_response(
                        prompt,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True
                    )
                else:
                    logger.warning("Local LLM model doesn't have generate_response method, using fallback config")
                    return self._get_fallback_config()
            else:
                logger.warning("No LLM available (neither API nor local model), using fallback config")
                return self._get_fallback_config()
            
            if not llm_response:
                logger.error("LLM returned empty response")
                return self._get_fallback_config()
            
            logger.debug(f"LLM response: {llm_response[:200]}...")
            suggestion = self._parse_llm_suggestion(llm_response)
            
            if suggestion:
                logger.info(f"LLM suggested config: {suggestion}")
                return suggestion
            else:
                logger.warning("LLM suggestion parsing failed, using fallback config")
                return self._get_fallback_config()
            
        except Exception as e:
            logger.error(f"Error getting LLM suggestion: {e}")
            import traceback
            traceback.print_exc()
            return self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict:
        """
        當 LLM 不可用時的備用配置
        """
        # 簡單的探索策略
        if not self.trial_history:
            # 第一次嘗試：默認值
            return {
                "alpha": 0.5,
                "param_ratio_target": 0.8,
                "scaling_method": "abs_mean",
                "n_calib_samples": 32,
                "calib_dataset": "wikitext2"
            }
        else:
            # 後續嘗試：隨機變化
            best = max(self.trial_history, key=lambda x: x['score'])
            
            # 輕微變化最好的配置
            config = best['params'].copy()
            
            # 50% 機率調整 alpha
            if random.random() < 0.5:
                config['alpha'] = min(0.7, max(0.3, config['alpha'] + random.uniform(-0.1, 0.1)))
            
            # 50% 機率調整 param_ratio
            if random.random() < 0.5:
                config['param_ratio_target'] = min(0.95, max(0.5, config['param_ratio_target'] + random.uniform(-0.05, 0.05)))
            
            return config
        
    # def get_baseline_metrics(self):
    #         """獲取原始模型的基準數據"""
    #         logger.info("=" * 40)
    #         logger.info("Step 0: Evaluating Baseline (Raw Model)")
    #         logger.info("=" * 40)

    #         # --- Part A: 獲取 Baseline PPL ---
    #         # 執行 asvd.py --raw_model
    #         raw_asvd_cmd = [
    #             sys.executable, "asvd.py",
    #             "--model_id", self.model_id,
    #             "--raw_model",
    #             "--eval_ppl", "wikitext2,ptb"
    #         ]
    #         logger.info("Running asvd.py --raw_model for baseline PPL...")
    #         subprocess.run(raw_asvd_cmd, cwd=str(Path(__file__).parent))
            
    #         # 讀取 PPL
    #         metrics_path = Path.cwd() / "ASVD4LLM-main" / "output" / "temp_asvd_metrics.json"
    #         if metrics_path.exists():
    #             with open(metrics_path, "r") as f:
    #                 data = json.load(f)
    #                 self.baseline_ppl = (data.get('wikitext2', 1000.0) + data.get('ptb', 1000.0)) / 2
    #         else:
    #             self.baseline_ppl = 15.0 # 預設參考值

    #         # --- Part B: 獲取 Baseline Accuracy ---
    #         # 1. 更新 config 為原始模型
    #         config_path = Path.cwd() / "tmp/config/model_config.yaml"
    #         with open(config_path, "r") as f:
    #             config_yaml = yaml.safe_load(f)
            
    #         original_model_id = self.model_id
    #         config_yaml['model']['name'] = original_model_id # 使用原始 ID
            
    #         with open(config_path, "w") as f:
    #             yaml.dump(config_yaml, f)

    #         # 2. 呼叫 test_eval.py (模仿你的步驟 3)
    #         try:
    #             test_eval_path = Path.cwd() / "tmp/test/test_eval.py"
    #             subprocess.run([sys.executable, str(test_eval_path)], cwd=str(Path.cwd()))
                
    #             # 從 results 讀取 (原始模型名稱處理)
    #             raw_model_name = original_model_id.split("/")[-1]
    #             json_path = Path.cwd() / "results" / raw_model_name / f"{self.task}_results.json"
                
    #             if json_path.exists():
    #                 with open(json_path, 'r') as f:
    #                     self.baseline_acc = json.load(f).get('accuracy', 0.058)
    #             else:
    #                 self.baseline_acc = 0.058
    #         except Exception as e:
    #             logger.error(f"Baseline Acc evaluation failed: {e}")
    #             self.baseline_acc = 0.058

    #         logger.info(f"✅ Baseline established: Acc={self.baseline_acc:.4f}, PPL={self.baseline_ppl:.2f}")

    def _build_asvd_model(self, config: Dict) -> Dict:
        """
        使用 build_asvd_repo.py 構建壓縮後的模型
        """
        logger.info(f"Building ASVD model with config: {config}")
        
        # 獲得當前目錄 (ASVD4LLM-main)
        asvd_dir = Path(__file__).parent
        #build_repo_dir = asvd_dir / "huggingface_repos"
        
        # 構建模型保存路徑
        model_name = self.model_id.split("/")[-1]
        output_path = asvd_dir / "output" / f"{model_name}-asvd{int(config['param_ratio_target']*100)}-alpha{int(config['alpha']*100)}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 構建 build_asvd_repo.py 命令
        build_cmd = [
            "python", "huggingface_repos/build_asvd_repo.py",
            "--model_id", self.model_id,
            "--alpha", str(config['alpha']),
            "--param_ratio_target", str(config['param_ratio_target']),
            "--scaling_method", config['scaling_method'],
            "--n_calib_samples", str(config.get('n_calib_samples', 32)),
            "--calib_dataset", config.get('calib_dataset', 'wikitext2'),
            "--act_aware",
            "--use_cache"
        ]
        
        try:
            logger.info(f"Running: {' '.join(build_cmd)}")
            
            # 設置環境變數以避免 GPU 記憶體衝突
            env = os.environ.copy()
            env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
            env['CUDA_VISIBLE_DEVICES'] = '0'  # 強制只用 CUDA:0 (RTX 4090)
            
            logger.info("Forcing single GPU (CUDA:0) for build subprocess")
            
            # 不使用 capture_output，讓輸出實時顯示
            result = subprocess.run(build_cmd, cwd=str(asvd_dir), timeout=None, env=env)
            
            if result.returncode != 0:
                logger.error(f"Build failed with return code {result.returncode}")
                return {"error": f"Build failed with return code {result.returncode}"}
            
            logger.info("Model building completed successfully")
            logger.info(f"Compressed model saved to: {output_path}")
            
            return {"status": "success", "model_path": str(output_path)}
            
        except subprocess.TimeoutExpired:
            logger.error("Model building timed out")
            return {"error": "Timeout"}
        except Exception as e:
            logger.error(f"Error building model: {e}")
            return {"error": str(e)}
    
    # def _run_asvd_experiment(self, config: Dict) -> Dict:
    #     """
    #     執行 ASVD 實驗
        
    #     這只是啟動 asvd.py 進程。
    #     """
    #     logger.info(f"Running ASVD experiment with config: {config}")
        
    #     # 獲得當前目錄 (ASVD4LLM-main)
    #     asvd_dir = Path(__file__).parent
        
    #     # 構建 asvd.py 命令 - 只傳遞 asvd.py 支持的參數
    #     asvd_cmd = [
    #         "python", "asvd.py",
    #         "--model_id", self.model_id,
    #         "--alpha", str(config['alpha']),
    #         "--param_ratio_target", str(config['param_ratio_target']),
    #         "--scaling_method", config['scaling_method'],
    #         "--n_calib_samples", str(config.get('n_calib_samples', 32)),
    #         "--calib_dataset", config.get('calib_dataset', 'wikitext2'),
    #         "--act_aware",
    #         "--use_cache"
    #     ]
        
    #     try:
    #         logger.info(f"Running: {' '.join(asvd_cmd)}")
            
    #         # 設置環境變數以避免 GPU 記憶體碎片化和多GPU衝突
    #         env = os.environ.copy()
    #         env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
            
    #         # 之前的 CUDA:1 (RTX 3060 12GB) 會 OOM
    #         env['CUDA_VISIBLE_DEVICES'] = '0'
            
    #         logger.info("Forcing single GPU (CUDA:0) for ASVD subprocess (RTX 4090 has sufficient memory)")
            
    #         # 捕捉輸出以便記錄詳細的錯誤信息
    #         result = subprocess.run(asvd_cmd, cwd=str(asvd_dir), 
    #                               timeout=None, env=env)
            
    #         # 記錄所有輸出
    #         if result.stdout:
    #             logger.info(f"ASVD stdout:\n{result.stdout}")
    #         if result.stderr:
    #             logger.error(f"ASVD stderr:\n{result.stderr}")
            
    #         if result.returncode != 0:
    #             logger.error(f"ASVD failed with return code {result.returncode}")
    #             error_msg = f"ASVD failed with return code {result.returncode}"
    #             if result.stderr:
    #                 error_msg += f"\nError details:\n{result.stderr}"
    #             return {"error": error_msg}
            
    #         logger.info("ASVD completed successfully")
            
    #     except subprocess.TimeoutExpired:
    #         logger.error("ASVD experiment timed out")
    #         return {"error": "Timeout"}
    #     except Exception as e:
    #         logger.error(f"Error running ASVD: {e}")
    #         return {"error": str(e)}
        
    #     return {"status": "success"}
    
    def _evaluate_config(self, config: Dict, iteration: int) -> Dict:
        """
        評估配置
        
        1. 執行 ASVD 實驗
        2. 構建模型 (build_asvd_repo.py)
        3. 評估模型
        4. 計算綜合分數
        """
        start_time = datetime.now()
        
        # 根據迭代次數動態設置樣本數
        self._set_evaluator_samples(iteration)
        
        # 啟動 CO2 追蹤
        if self.co2_tracker:
            self.co2_tracker.start()
        
        try:
            # # 步驟 1: 執行 ASVD  -->不需要了，因為 build_asvd_repo.py 已經包含了 ASVD 的步驟
            # logger.info(f"[Iteration {iteration}] Starting ASVD experiment")
            # 提取推理內容並從傳給腳本的配置中移除
            current_reasoning = config.pop('reasoning', 'No reasoning provided')
            # asvd_result = self._run_asvd_experiment(config)
            
            # if "error" in asvd_result:
            #     logger.warning(f"ASVD experiment failed: {asvd_result['error']}")
            #     return {
            #         "iteration": iteration,
            #         "params": config,
            #         "reasoning": current_reasoning, # 保存 LLM 當時的思考邏輯
            #         "score": 0.0,
            #         "accuracy": 0.0,
            #         "status": "failed",
            #         "error": asvd_result.get("error")
            #     }
            
            # 步驟 2: 構建壓縮後的模型 (build_asvd_repo.py)
            logger.info(f"[Iteration {iteration}] Building ASVD compressed model")
            build_result = self._build_asvd_model(config)
            
            if "error" in build_result:
                logger.warning(f"Model building failed: {build_result['error']}")
                return {
                    "iteration": iteration,
                    "params": config,
                    "reasoning": current_reasoning, # 保存 LLM 當時的思考邏輯
                    "score": 0.0,
                    "accuracy": 0.0,
                    "status": "failed",
                    "error": f"Build failed: {build_result.get('error')}"
                }
            
            # 步驟 3: 評估模型 (使用 test_eval.py 的評估框架)
            logger.info(f"[Iteration {iteration}] Evaluating model on {self.task}")
            
            # 更新 model_config.yaml 中的模型路徑為壓縮後的模型
            model_path = build_result.get('model_path')
            if model_path:
                logger.info(f"Updating model_config.yaml with model path: {model_path}")
                # 讀取現有配置
                config_path = Path.cwd() / "tmp/config/model_config.yaml"
                with open(config_path, "r") as f:
                    config_yaml = yaml.safe_load(f)
                
                # 更新模型路徑
                config_yaml['model']['name'] = model_path
                
                # 更新 num_samples（如果有的話）
                if self.evaluator.config.num_samples:
                    if 'dataset' not in config_yaml:
                        config_yaml['dataset'] = {}
                    if 'override' not in config_yaml['dataset']:
                        config_yaml['dataset']['override'] = {}
                    config_yaml['dataset']['override']['num_samples'] = self.evaluator.config.num_samples
                
                # 寫回配置
                with open(config_path, "w") as f:
                    yaml.dump(config_yaml, f)
            
            # 呼叫 test_eval.py 進行評估
            try:
                logger.info(f"Running test_eval.py for evaluation with task: {self.task}...")
                
                test_eval_path = Path.cwd() / "tmp/test/test_eval.py"
                eval_cmd = [
                    sys.executable,
                    str(test_eval_path)
                ]
                
                result = subprocess.run(eval_cmd, cwd=str(Path.cwd()))
                
                if result.returncode != 0:
                    logger.warning(f"test_eval.py failed: {result.stderr}")
                    accuracy = 0.0
                else:
                    if model_path:
                        # 2. 取得模型資料夾名稱 (例如: Llama-3.1-Minitron-4B-Width-Base-asvd70)
                        model_name = Path(model_path).name
                        
                        # 3. 定位 JSON 檔案
                        # test_eval.py 在 root 執行，save_results 會在 root/output/{model_name}/ 下產生檔案
                        # 我們使用 root_path (目前 CWD) 來定位
                        json_path = Path.cwd() / "results" / model_name / f"{self.task}_results.json"

                        if json_path.exists():
                            try:
                                with open(json_path, 'r', encoding='utf-8') as f:
                                    eval_data = json.load(f)
                                    # 從 evaluator 的回傳字典中提取 accuracy
                                    accuracy = eval_data.get('accuracy', 0.0)
                                logger.info(f"成功從 JSON 讀取分數: {accuracy:.4f}")
                            except Exception as e:
                                logger.error(f"讀取結果 JSON 失敗: {e}")
                                accuracy = 0.0
                        else:
                            logger.warning(f"找不到結果檔案，請檢查路徑: {json_path}")
                            accuracy = 0.0
                    else:
                        accuracy = 0.0
                
            except subprocess.TimeoutExpired:
                logger.warning("test_eval.py evaluation timed out")
                accuracy = 0.0
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                traceback.print_exc()
                accuracy = 0.0
            
            # 步驟 4: 計算綜合分數
            # 1. 讀取真實壓縮比 (從模型 config.json)
            actual_ratio = config['param_ratio_target']  # 備用值
            model_config_path = Path(model_path) / "config.json"
            if model_config_path.exists():
                with open(model_config_path, "r") as f:
                    m_cfg = json.load(f)
                    actual_ratio = m_cfg.get("actual_param_ratio", actual_ratio)
            
            # 2. 讀取真實 PPL 分數 (從 asvd.py 輸出的 JSON)
            # 假設 asvd_tuner.py 執行路徑下的 output/temp_asvd_metrics.json
            metrics_path = Path(__file__).parent/ "output" / "temp_asvd_metrics.json"
            ppl_wiki = 1000.0
            ppl_ptb = 1000.0
            avg_ppl = 1000.0
            
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics_data = json.load(f)
                    ppl_wiki = metrics_data.get('wikitext2', 1000.0)
                    ppl_ptb = metrics_data.get('ptb', 1000.0)
                    avg_ppl = (ppl_wiki + ppl_ptb) / 2
            
            # --- 核心分數轉換邏輯 ---
            
            # 壓縮比得分：目標是最小化 actual_ratio
            # 我們定義 1.0 - actual_ratio，這樣壓越多(ratio越小)，分數越高
            compression_score = 1.0 - actual_ratio 
            
            # PPL 得分：目標是最小化 PPL (PPL 越低越流暢)
            # 使用指數衰減公式：如果 PPL=15(優秀) 分數接近 1.0；如果 PPL=500(崩潰) 分數趨近 0
            # 公式：e^(- (PPL - 基準) / 縮放係數)
            ppl_score = np.exp(-(avg_ppl - 15) / 100) if avg_ppl > 15 else 1.0
            ppl_score = max(0.0, ppl_score) # 確保不為負
            
            # 3. 綜合分數 (根據權重)
            score = (
                self.accuracy_weight * accuracy +
                self.compression_weight * compression_score +
                self.ppl_weight * ppl_score
            )
            # # 如果 Accuracy 比 Baseline 掉超過 60%，強制將分數大幅扣減
            # if accuracy < (self.baseline_acc * 0.6):
            #     score *= 0.1  # 懲罰崩潰模型

            elapsed = (datetime.now() - start_time).total_seconds()
            
            result = {
                "iteration": iteration,
                "params": config,
                "reasoning": current_reasoning, # 保存 LLM 當時的思考邏輯
                "score": float(score),
                "accuracy": float(accuracy),
                "compression_ratio": float(actual_ratio),
                "ppl": {
                    "avg": float(avg_ppl),
                    "wikitext2": float(ppl_wiki),
                    "ptb": float(ppl_ptb)
                },
                "elapsed_seconds": elapsed,
                "status": "success"
            }
            
            logger.info(f"[Iteration {iteration}] Score: {score:.4f}, Acc: {accuracy:.4f}, Real_Ratio: {actual_ratio:.4f}, Avg_PPL: {avg_ppl:.2f}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error evaluating config: {e}")
            return {
                "iteration": iteration,
                "params": config,
                "score": 0.0,
                "accuracy": 0.0,
                "status": "failed",
                "error": str(e)
            }
        
        finally:
            # 停止 CO2 追蹤
            if self.co2_tracker:
                self.co2_tracker.stop()
    
    def optimize(self) -> Dict:
        """
        執行優化迴圈
        """
        logger.info("=" * 80)
        logger.info("Starting ASVD Adaptive Tuning")
        logger.info("=" * 80)
        
        best_result = None
        best_score = -float('inf')
        # self.get_baseline_metrics()

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            logger.info(f"{'='*80}")
            
            # 獲得 LLM 建議
            config = self._get_llm_suggestion(iteration)
            if not config:
                logger.error("Failed to get LLM suggestion")
                break
            
            logger.info(f"Suggested config: {config}")
            
            # 評估配置
            result = self._evaluate_config(config, iteration)
            self.trial_history.append(result)
            
            # 更新最佳結果
            if result['score'] > best_score:
                best_score = result['score']
                best_result = result
                logger.info(f"🎉 New best score: {best_score:.4f}")
            
            # 顯示進度
            logger.info(f"\nIteration {iteration} Summary:")
            logger.info(f"  Config: {config}")
            logger.info(f"  Score: {result['score']:.4f}")
            logger.info(f"  Accuracy: {result.get('accuracy', 'N/A')}")
            logger.info(f"  Best so far: {best_score:.4f}")
            
            # 清理 GPU 記憶體以準備下一個疊代
            if iteration < self.max_iterations:
                logger.info("Cleaning up GPU memory for next iteration...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    logger.info(f"GPU memory cleared. Free memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GiB")
                time.sleep(2)  # 等待記憶體完全釋放
        
        # 總結
        logger.info("\n" + "=" * 80)
        logger.info("ASVD Tuning Complete")
        logger.info("=" * 80)
        
        if best_result:
            logger.info(f"\n🏆 Best Configuration:")
            logger.info(f"  Iteration: {best_result['iteration']}")
            logger.info(f"  Score: {best_result['score']:.4f}")
            logger.info(f"  Params: {best_result['params']}")
            
            # Provide execution command for final evaluation with all samples
            params = best_result['params']
            exec_cmd = (
                f"PYTORCH_ALLOC_CONF='expandable_segments:True' python asvd.py "
                f"--model_id {self.model_id} "
                f"--alpha {params['alpha']:.2f} "
                f"--param_ratio_target {params['param_ratio_target']:.2f} "
                f"--scaling_method {params['scaling_method']} "
                f"--n_calib_samples {params.get('n_calib_samples', 32)} "
                f"--calib_dataset {params.get('calib_dataset', 'wikitext2')} "
                f"--act_aware --use_cache"
            )
            logger.info(f"\nTo evaluate best model with all samples, run:")
            logger.info(f"   {exec_cmd}")
        
        # 在 optimize 結束前加入還原邏輯
        logger.info("Cleaning up model_config.yaml (setting num_samples back to null)...")
        try:
            config_path = Path.cwd() / "tmp/config/model_config.yaml"
            if config_path.exists():
                with open(config_path, "r") as f:
                    final_cfg = yaml.safe_load(f)
                
                # 將 num_samples 設回 None (寫入 YAML 會顯示為 null)
                if 'dataset' in final_cfg and 'override' in final_cfg['dataset']:
                    final_cfg['dataset']['override']['num_samples'] = None
                    
                with open(config_path, "w") as f:
                    yaml.dump(final_cfg, f)
                logger.info("✅ model_config.yaml has been reset to null.")
        except Exception as e:
            logger.error(f"Failed to reset model_config.yaml: {e}")

        return {
            "best_result": best_result,
            "trial_history": self.trial_history,
            "total_iterations": len(self.trial_history)
        }
    
    def save_results(self, results: Dict, output_path: str = "asvd_tuning_results.json"):
        """
        保存優化結果
        """
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 清理結果以確保可序列化
        clean_results = _deep_clean_dict(results)
        
        with open(output_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
        return output_path


if __name__ == "__main__":
    """
    直接運行此腳本來執行 ASVD 自動調優
    
    用法：
        python asvd_tuner.py
    """
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('asvd_tuning.log'),
            logging.StreamHandler()
        ]
    )
    
    parser = argparse.ArgumentParser(description='ASVD Adaptive Tuner')
    parser.add_argument('--model_id', type=str, default='facebook/opt-1.3b',
                       help='Model ID to compress')
    parser.add_argument('--task', type=str, default='gsm8k',
                       help='Task name (gsm8k, multinews, etc.)')
    parser.add_argument('--max_iterations', type=int, default=10,
                       help='Maximum number of iterations')
    parser.add_argument('--accuracy_weight', type=float, default=0.7,
                       help='Weight for accuracy')
    parser.add_argument('--compression_weight', type=float, default=0.2,
                       help='Weight for compression ratio')
    parser.add_argument('--ppl_weight', type=float, default=0.1,
                       help='Weight for PPL')
    parser.add_argument('--num_samples_early', type=int, nargs='*', default=[],
                       help='Sample counts for early iterations')
    parser.add_argument('--num_samples_final', type=int, default=None,
                       help='Sample count for final iteration (None = all)')
    parser.add_argument('--output', type=str, default='asvd_tuning_results.json',
                       help='Output file path')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("ASVD Adaptive Tuner - Starting")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Max iterations: {args.max_iterations}")
    
    # 步驟 1: 初始化評估器
    logger.info("\n" + "=" * 80)
    logger.info("Step 1: Initializing Evaluator")
    logger.info("=" * 80)
    
    # 添加 tmp 路徑
    tmp_path = Path(__file__).parent.parent / "tmp"
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    
    # 重要：改變工作目錄到根目錄，以便相對路徑正確
    root_path = Path(__file__).parent.parent
    original_cwd = Path.cwd()
    os.chdir(root_path)
    logger.info(f"Changed working directory to: {root_path}")
    
    try:
        # 讀取配置（使用相對於根目錄的路徑）
        import yaml
        config_path = "tmp/config/model_config.yaml"
        with open(config_path, "r") as f:
            eval_config = yaml.safe_load(f)
        
        logger.info("✅ Config loaded successfully")
        logger.info(f"   Model: {eval_config.get('model', {}).get('name', 'N/A')}")
        logger.info(f"   Task: {args.task}")
        
        # 為簡單起見，創建一個簡單的容器類來保存配置
        # （避免直接依賴 GSM8KEvaluator）
        class EvalConfig:
            def __init__(self, yaml_config):
                self.num_samples = yaml_config.get('dataset', {}).get('override', {}).get('num_samples')
        
        config_obj = EvalConfig(eval_config)
        
        # 創建一個簡單的 evaluator-like 對象用於 tuner
        class SimpleEvaluator:
            def __init__(self, yaml_config):
                self.config = EvalConfig(yaml_config)
        
        evaluator = SimpleEvaluator(eval_config)
        
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        logger.error(f"Make sure {root_path}/tmp/config/model_config.yaml exists")
        os.chdir(original_cwd)  # 恢復原工作目錄
        sys.exit(1)
    
    # 步驟 2: 創建調整器
    logger.info("\n" + "=" * 80)
    logger.info("Step 2: Creating ASVDAdaptiveTuner")
    logger.info("=" * 80)
    
    tuner = ASVDAdaptiveTuner(
        evaluator=evaluator,
        model_id=args.model_id,
        task=args.task,
        max_iterations=args.max_iterations,
        accuracy_weight=args.accuracy_weight,
        compression_weight=args.compression_weight,
        ppl_weight=args.ppl_weight,
        num_samples_early=args.num_samples_early if args.num_samples_early else [],
        num_samples_final=args.num_samples_final
    )
    logger.info("✅ Tuner created successfully")
    
    # 步驟 3: 執行優化
    logger.info("\n" + "=" * 80)
    logger.info("Step 3: Running Optimization")
    logger.info("=" * 80)
    
    results = tuner.optimize()
    
    # 步驟 4: 保存結果
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: Saving Results")
    logger.info("=" * 80)
    
    output_path = tuner.save_results(results, args.output)
    
    # 步驟 5: 顯示總結
    logger.info("\n" + "=" * 80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 80)
    
    if results["best_result"]:
        best = results["best_result"]
        logger.info(f"\n🏆 Best Configuration Found at Iteration {best['iteration']}:")
        logger.info(f"   Score: {best['score']:.4f}")
        logger.info(f"   Accuracy: {best.get('accuracy', 'N/A'):.4f}")
        logger.info(f"\n   Best Parameters:")
        for k, v in best['params'].items():
            logger.info(f"     - {k}: {v}")
    
    logger.info(f"\n📊 Total Iterations: {results['total_iterations']}")
    logger.info(f"💾 Results saved to: {output_path}")
    logger.info("\n✅ Done!")
    
    # 恢復原工作目錄
    os.chdir(original_cwd)
