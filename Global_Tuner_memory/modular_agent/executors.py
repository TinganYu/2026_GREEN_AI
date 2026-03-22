# """
# 執行器模組（Global_Tuner_v2 版本）
# =====================================

# 主要改進（相對於 Global_Tuner/modular_agent/executors.py）：
# 1. 使用根目錄 Method/ 介面（QuantConfig/ASVDConfig/SparseConfig）
# 2. 直接呼叫 Evals/ 評估器，不再透過 subprocess + test_eval.py
# 3. 模型一次載入，多個 dataset 重複使用
# """

# import logging
# import sys
# import math
# import traceback
# from pathlib import Path
# from typing import Optional

# logger = logging.getLogger("Executors")
# logger.setLevel(logging.INFO)
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     handler.setFormatter(logging.Formatter(
#         '%(asctime)s | %(levelname)s | %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     ))
#     logger.addHandler(handler)
# logger.propagate = False

# # 確保能 import 根目錄的 Method/ 和 Evals/
# _ROOT_DIR = Path(__file__).resolve().parent.parent.parent
# sys.path.insert(0, str(_ROOT_DIR))

# from Method.asvd import ASVDConfig, run_asvd as _run_asvd
# from Method.sparse import SparseConfig, run_sparse as _run_sparse
# from Method.quantize import QuantConfig, run_quantization as _run_quantization
# from Evals import EVALUATOR_MAP


# # ============================================================================
# # 壓縮執行函數（橋接 StrategySuggestion → Method/ 介面）
# # ============================================================================

# def run_asvd(model_id: str, suggestion, output_dir: Optional[str] = None) -> str:
#     """執行 ASVD 壓縮，完成後將模型搬至 output_dir（若指定）"""
#     import shutil
#     config = ASVDConfig(
#         alpha=suggestion.alpha,
#         param_ratio_target=suggestion.param_ratio_target,
#         scaling_method=suggestion.scaling_method,
#     )
#     asvd_path = _run_asvd(model_id, config)

#     if output_dir and Path(asvd_path).resolve() != Path(output_dir).resolve():
#         dest = Path(output_dir)
#         if dest.exists():
#             shutil.rmtree(dest)
#         shutil.move(asvd_path, str(dest))
#         logger.info(f"ASVD 模型已移至: {dest}")
#         return str(dest)

#     return asvd_path


# def run_sparse(model_path: str, suggestion, output_dir: Optional[str] = None) -> str:
#     """執行 SparseGPT 稀疏化"""
#     structure = suggestion.sparsity_structure or "unstructured"
#     # N:M structured sparsity 有固定 ratio（由結構本身決定），不使用 sparsity_ratio
#     is_structured = structure != "unstructured"
#     ratio = 0.5 if is_structured else (suggestion.sparsity_ratio or 0.5)
#     config = SparseConfig(
#         sparsity_ratio=ratio,
#         structure=structure,
#         output_dir=output_dir,
#     )
#     return _run_sparse(model_path, config)


# # def run_quantization(model_path: str, suggestion, output_dir: Optional[str] = None) -> str:
# #     """執行量化（gptq/awq/qqq/bnb）"""
# #     method = suggestion.quant_method.lower()

# #     if method == "awq":
# #         fmt = "gemm"
# #     elif method == "qqq":
# #         fmt = "qqq"
# #     else:
# #         fmt = suggestion.quant_format or "gptq"

# #     config = QuantConfig(
# #         method=method,
# #         bits=suggestion.quant_bits,
# #         group_size=suggestion.quant_group_size or 128,
# #         format=fmt,
# #         damp_percent=suggestion.damp_percent or 0.05,
# #         mse=suggestion.mse or 0.0,
# #         quant_type=suggestion.quant_type or "nf4",
# #         use_double_quant=suggestion.use_double_quant or False,
# #         output_dir=output_dir,
# #     )
# #     return _run_quantization(model_path, config)
# def run_quantization(model_path: str, suggestion, output_dir: Optional[str] = None) -> str:
#     """執行量化（gptq/awq/qqq/bnb）"""
#     # Map hybrid mode to BNB, otherwise use the mode name directly
#     method = "bnb" if suggestion.mode == "hybrid_asvd_bnb" else suggestion.mode

#     if method == "awq":
#         fmt = "gemm"
#     elif method == "qqq":
#         fmt = "qqq"
#     else:
#         fmt = suggestion.quant_format or "gptq"

#     config = QuantConfig(
#         method=method,
#         bits=4 if method in ("awq", "qqq") else (suggestion.quant_bits or 4),
#         group_size=suggestion.quant_group_size or 128,
#         format=fmt,
#         damp_percent=suggestion.damp_percent or 0.05,
#         mse=suggestion.mse or 0.0,
#         quant_type=getattr(suggestion, 'quant_type', 'nf4'),
#         use_double_quant=getattr(suggestion, 'use_double_quant', False),
#         output_dir=output_dir,
#     )
#     return _run_quantization(model_path, config)

# # ============================================================================
# # 評估函數（直接呼叫 Evals/ 評估器）
# # ============================================================================

# def _load_model_for_eval(model_path: str):
#     """
#     載入模型供評估使用。
#     自動偵測量化類型（GPTQModel 統一處理 gptq/awq/qqq，其餘使用 transformers）。
#     """
#     from transformers import AutoTokenizer
#     import torch

#     quant_type = _detect_quantization_type(model_path)
#     logger.info(f"偵測到量化類型: {quant_type}")

#     if quant_type in ("gptq", "awq", "qqq"):
#         from gptqmodel import GPTQModel
#         logger.info(f"使用 GPTQModel.from_quantized() 載入: {model_path}")
#         model = GPTQModel.from_quantized(
#             model_path,
#             device_map={"": "cuda:0"},
#         )
#     else:
#         from transformers import AutoModelForCausalLM
#         import json
#         config_path = Path(model_path) / "config.json"
#         if config_path.exists():
#             try:
#                 with open(config_path, "r", encoding="utf-8") as f:
#                     config_data = json.load(f)
                
#                 # Check if it is a pure sparse model with the buggy 'compressed-tensors' config
#                 q_config = config_data.get("quantization_config", {})
#                 if q_config.get("quant_method") == "compressed-tensors" and "sparsity_config" in q_config:
#                     # If it has sparsity but no actual weight quantization (config_groups), delete the block
#                     if "config_groups" not in q_config:
#                         print(f"🔧 Fixing config.json: Removing buggy quantization_config to prevent transformers crash.")
#                         del config_data["quantization_config"]
                        
#                         # Save the cleaned config back to the file
#                         with open(config_path, "w", encoding="utf-8") as f:
#                             json.dump(config_data, f, indent=2)
#             except Exception as e:
#                 print(f"⚠️ Failed to clean config.json: {e}")
#             logger.info(f"使用 AutoModelForCausalLM 載入: {model_path}")
            
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             device_map={"": "cuda:0"},
#             torch_dtype=torch.float16,
#             trust_remote_code=True,
#         )

#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

#     return model, tokenizer


# def _detect_quantization_type(model_path: str) -> Optional[str]:
#     """從 config.json 或路徑名稱偵測量化類型"""
#     import json
#     config_path = Path(model_path) / "config.json"
#     if config_path.exists():
#         try:
#             with open(config_path, "r") as f:
#                 cfg = json.load(f)
#             quant_cfg = cfg.get("quantization_config", {})
#             quant_type = quant_cfg.get("quant_type", "")
#             if quant_type:
#                 return quant_type.lower()
#         except Exception:
#             pass
#     # 從路徑名稱推斷
#     path_lower = model_path.lower()
#     for method in ("gptq", "awq", "qqq", "bnb"):
#         if method in path_lower:
#             return method
#     return None


# def _build_eval_config(model_path: str, output_dir: str = "results", num_samples: Optional[int] = None) -> dict:
#     """建立評估器所需的 config dict"""
#     override = {"output_dir": output_dir}
#     if num_samples is not None:
#         override["num_samples"] = num_samples
#     return {
#         "model": {"name": model_path},
#         "dataset": {
#             "config_file": str(_ROOT_DIR / "Evals" / "config" / "dataset_config.yaml"),
#             "override": override,
#         },
#     }


# def run_evaluation(model_path: str, tasks, weights: dict, baseline_metrics: dict = None,
#                    num_samples: Optional[int] = None, output_dir: Optional[str] = None) -> dict:
#     """
#     直接呼叫 Evals/ 評估器，無需 subprocess。

#     Args:
#         model_path: 模型路徑
#         tasks: str (逗號分隔) 或 list
#         weights: 評分權重 {"acc": ..., "lat": ..., "vram": ..., "emit": ...}
#         baseline_metrics: 基線指標（用於計算歸一化分數）
#         num_samples: 每個 dataset 的評估樣本數（None = 全部）

#     Returns:
#         {"accuracy": ..., "latency": ..., "vram": ..., "emissions": ..., "score": ..., "details": {...}}
#     """
#     import gc
#     import torch

#     if isinstance(tasks, str):
#         tasks_list = [t.strip() for t in tasks.split(',')]
#     else:
#         tasks_list = list(tasks)

#     # 載入模型一次，所有 dataset 重複使用
#     logger.info(f"載入模型: {model_path}")
#     model, tokenizer = _load_model_for_eval(model_path)

#     eval_out = output_dir or str(_ROOT_DIR / "tuning_results")
#     config = _build_eval_config(model_path, output_dir=eval_out, num_samples=num_samples)
#     all_task_results = {}
#     total_acc = 0.0
#     total_lat = 0.0
#     total_emit = 0.0
#     vram_list = []

#     for task in tasks_list:
#         logger.info(f"{'=' * 50}")
#         logger.info(f"評估: {task.upper()}")

#         if task not in EVALUATOR_MAP:
#             logger.error(f"未知資料集: {task}。可用: {list(EVALUATOR_MAP.keys())}")
#             all_task_results[task] = {"accuracy": 0.0, "latency": 0.0, "vram": 0.0, "emissions": 0.0}
#             continue

#         try:
#             evaluator_cls = EVALUATOR_MAP[task]
#             evaluator = evaluator_cls(config)

#             # 注入已載入的 model 和 tokenizer，避免重複載入
#             evaluator.model = model
#             evaluator.tokenizer = tokenizer
#             evaluator._use_vllm = False
#             evaluator._setup_generation_pipeline()

#             results = evaluator.evaluate()
#             evaluator.save_results(results)

#             acc = results.get("accuracy", results.get('pass@1', 0.0))
#             lat = results.get("total_generation_time_sec", 0.0)
#             vram = results.get("gpu_peak_mb", 0.0) / 1024.0
#             emit = results.get("emissions_kg_co2", 0.0)

#             all_task_results[task] = {"accuracy": acc, "latency": lat, "vram": vram, "emissions": emit}
#             total_acc += acc
#             total_lat += lat
#             total_emit += emit
#             vram_list.append(vram)

#             logger.info(f"{task.upper()} 完成: accuracy={acc:.4f}, total={results.get('total', 0)}")

#         except Exception as e:
#             logger.error(f"{task} 評估失敗: {e}")
#             traceback.print_exc()
#             all_task_results[task] = {"accuracy": 0.0, "latency": 0.0, "vram": 0.0, "emissions": 0.0}

#     # 釋放模型記憶體
#     del model
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     n = len(tasks_list)
#     avg_acc = total_acc / n if n else 0.0
#     avg_lat = total_lat / n if n else 0.0
#     max_vram = max(vram_list) if vram_list else 0.0
#     avg_emit = total_emit / n if n else 0.0

#     if not baseline_metrics:
#         evaluator.unload_model()
#         return {
#             "accuracy": avg_acc, "latency": avg_lat, "vram": max_vram, "emissions": avg_emit,
#             "score": 1.0,
#             "details": all_task_results,
#         }

#     # 歸一化計分
#     base_acc = baseline_metrics.get("accuracy", 1e-6)
#     base_lat = baseline_metrics.get("latency", 1e-6)
#     base_vram = baseline_metrics.get("vram", 1e-6)
#     base_emit = baseline_metrics.get("emissions", 1e-6)

#     norm_acc = avg_acc / (base_acc + 1e-6)
#     norm_lat = base_lat / (avg_lat + 1e-6)
#     norm_vram = base_vram / (max_vram + 1e-6)
#     norm_emit = base_emit / (avg_emit + 1e-6)

#     final_score = 1.0 + (
#         weights.get("acc", 0.0) * math.log(norm_acc + 1e-9) +
#         weights.get("lat", 0.0) * math.log(norm_lat + 1e-9) +
#         weights.get("vram", 0.0) * math.log(norm_vram + 1e-9) +
#         weights.get("emit", 0.0) * math.log(norm_emit + 1e-9)
#     )
#     evaluator.unload_model()
#     return {
#         "accuracy": avg_acc, "latency": avg_lat, "vram": max_vram, "emissions": avg_emit,
#         "score": final_score,
#         "details": all_task_results,
#     }

"""
執行器模組（Systematic_Tuner 版本）
=====================================
基於 Global_Tuner_memory/modular_agent/executors.py，調整如下：
1. _ROOT_DIR 修正為 Systematic_Tuner/ 往上兩層（Green_AI/）
2. cleanup 順序修正：evaluator.unload_model() 優先，移除冗餘的 del model
3. 以 run_isolated 執行時，子進程結束即 100% 釋放 VRAM（cleanup 為保險備用）
"""

import logging
import math
import sys
import traceback
from pathlib import Path
from typing import Optional

logger = logging.getLogger("Executors")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
logger.propagate = False

# Systematic_Tuner/executors.py → 往上三層到 Green_AI/
_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT_DIR))

from Method.asvd import ASVDConfig, run_asvd as _run_asvd
from Method.sparse import SparseConfig, run_sparse as _run_sparse
from Method.quantize import QuantConfig, run_quantization as _run_quantization
from Evals import EVALUATOR_MAP


# ============================================================================
# 壓縮執行函數（橋接 StrategySuggestion → Method/ 介面）
# ============================================================================

def run_asvd(model_id: str, suggestion, output_dir: Optional[str] = None) -> str:
    """執行 ASVD 壓縮，完成後將模型搬至 output_dir（若指定）"""
    import shutil
    config = ASVDConfig(
        alpha=suggestion.alpha,
        param_ratio_target=suggestion.param_ratio_target,
        scaling_method=suggestion.scaling_method,
    )
    asvd_path = _run_asvd(model_id, config)

    if output_dir and Path(asvd_path).resolve() != Path(output_dir).resolve():
        dest = Path(output_dir)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(asvd_path, str(dest))
        logger.info(f"ASVD 模型已移至: {dest}")
        return str(dest)

    return asvd_path


def run_sparse(model_path: str, suggestion, output_dir: Optional[str] = None) -> str:
    """執行 SparseGPT 稀疏化"""
    structure = suggestion.sparsity_structure or "unstructured"
    # N:M structured sparsity 有固定 ratio（由結構本身決定），不使用 sparsity_ratio
    is_structured = structure != "unstructured"
    ratio = 0.5 if is_structured else (suggestion.sparsity_ratio or 0.5)
    config = SparseConfig(
        sparsity_ratio=ratio,
        structure=structure,
        output_dir=output_dir,
    )
    return _run_sparse(model_path, config)


def run_quantization(model_path: str, suggestion, output_dir: Optional[str] = None) -> str:
    """執行量化（gptq/awq/qqq/bnb/hybrid）"""
    # Use mode to determine method, mapping hybrid to bnb
    method = "bnb" if suggestion.mode == "hybrid_asvd_bnb" else suggestion.mode
    method = method.lower()

    if method == "awq":
        fmt = "gemm"
    elif method == "qqq":
        fmt = "qqq"
    else:
        fmt = suggestion.quant_format or "gptq"

    config = QuantConfig(
        method=method,
        bits=suggestion.quant_bits,
        group_size=suggestion.quant_group_size or 128,
        format=fmt,
        damp_percent=suggestion.damp_percent or 0.05,
        mse=suggestion.mse or 0.0,
        quant_type=suggestion.quant_type or "nf4",
        use_double_quant=suggestion.use_double_quant or False,
        output_dir=output_dir,
    )
    return _run_quantization(model_path, config, output_dir=output_dir)

def _sanitize_sparse_config(model_path: str):
    """Removes buggy compressed-tensors config left by sparse methods."""
    import json
    from pathlib import Path
    
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        
        q_config = config_data.get("quantization_config") or {}
        if q_config.get("quant_method") == "compressed-tensors":
            if "config_groups" not in q_config:
                logger.info("🔧 Fixing config.json: Removing buggy quantization_config to prevent transformers crash.")
                del config_data["quantization_config"]
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_data, f, indent=2)
    except Exception as e:
        logger.warning(f"⚠️ Failed to clean config.json: {e}")

# ============================================================================
# 評估函數（直接呼叫 Evals/ 評估器）
# ============================================================================


def _detect_quantization_type(model_path: str) -> Optional[str]:
    """從 bnb_config.json / config.json / 路徑名稱偵測量化類型"""
    import json
    # BNB metadata（_run_bnb 寫入，不存模型權重）
    bnb_meta_path = Path(model_path) / "bnb_config.json"
    if bnb_meta_path.exists():
        return "bnb"

    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            q = cfg.get("quantization_config") or {}
            # GPTQ / AWQ / QQQ
            quant_type = q.get("quant_type", "").lower()
            if quant_type in ("gptq", "awq", "qqq"):
                return quant_type
            # 舊版 BNB（有存模型時）
            if (q.get("quant_method", "").lower() == "bitsandbytes"
                    or q.get("load_in_4bit") or q.get("load_in_8bit")):
                return "bnb"
        except Exception:
            pass
    _sanitize_sparse_config(model_path)
    # 從路徑名稱推斷
    path_lower = model_path.lower()
    for method in ("gptq", "awq", "qqq", "bnb"):
        if method in path_lower:
            return method
    return None


def _build_eval_config(model_path: str, output_dir: str = "results", num_samples: Optional[int] = None) -> dict:
    """建立評估器所需的 config dict"""
    override = {"output_dir": output_dir}
    if num_samples is not None:
        override["num_samples"] = num_samples
    return {
        "model": {"name": model_path},
        "dataset": {
            "config_file": str(_ROOT_DIR / "Evals" / "config" / "dataset_config.yaml"),
            "override": override,
        },
    }


def run_evaluation(model_path: str, tasks, weights: dict, baseline_metrics: dict = None,
                   num_samples: Optional[int] = None, output_dir: Optional[str] = None,
                   pen_t: float = 0.15, pen_a: float = 10.0) -> dict:
    """
    直接呼叫 Evals/ 評估器，無需 subprocess。
    通常透過 run_isolated() 在子進程執行，子進程結束後 OS 保證 VRAM 完整釋放。

    Args:
        model_path      : 模型路徑（本地或 HuggingFace ID）
        tasks           : str（逗號分隔）或 list
        weights         : 評分權重 {"acc": ..., "lat": ..., "vram": ..., "emit": ...}
        baseline_metrics: 基線指標（用於計算歸一化分數）；None 時 score 固定為 1.0
        num_samples     : 每個 dataset 的評估樣本數（None = 全部）
        output_dir      : 結果輸出目錄

    Returns:
        {"accuracy": ..., "latency": ..., "vram": ..., "emissions": ..., "score": ..., "details": {...}}

    Score 公式（log scale，對齊 Global_Tuner_memory）：
        score = 1.0 + Σ weight_i * log(norm_i + 1e-9)
        norm_acc  = quantized_acc  / baseline_acc   （越大越好）
        norm_lat  = baseline_lat   / quantized_lat  （越大越好 = 越快越好）
        norm_vram = baseline_vram  / quantized_vram （越大越好 = 越省越好）
        norm_emit = baseline_emit  / quantized_emit （越大越好 = 越少越好）
    """
    import gc
    import torch

    import json as _json

    if isinstance(tasks, str):
        tasks_list = [t.strip() for t in tasks.split(",")]
    else:
        tasks_list = list(tasks)

    # 偵測量化類型
    quant_type = _detect_quantization_type(model_path)
    logger.info(f"偵測到量化類型: {quant_type}，準備載入模型: {model_path}")

    # BNB：從 bnb_config.json 讀取參數，用原始模型路徑評估
    bnb_meta = None
    eval_model_path = model_path
    if quant_type == "bnb":
        bnb_meta_path = Path(model_path) / "bnb_config.json"
        with open(bnb_meta_path, "r", encoding="utf-8") as f:
            bnb_meta = _json.load(f)
        eval_model_path = bnb_meta["original_model"]
        logger.info(f"BNB 模式：從原始模型載入 {eval_model_path}")

    eval_out = output_dir or str(_ROOT_DIR / "tuning_results")
    config = _build_eval_config(eval_model_path, output_dir=eval_out, num_samples=num_samples)
    all_task_results = {}
    total_acc = 0.0
    total_lat = 0.0
    total_emit = 0.0
    vram_list = []
    evaluator = None
    shared_model = None
    shared_tokenizer = None

    for task in tasks_list:
        logger.info("=" * 50)
        logger.info(f"評估: {task.upper()}")

        if task not in EVALUATOR_MAP:
            logger.error(f"未知資料集: {task}。可用: {list(EVALUATOR_MAP.keys())}")
            all_task_results[task] = {"accuracy": 0.0, "latency": 0.0, "vram": 0.0, "emissions": 0.0}
            continue

        try:
            evaluator_cls = EVALUATOR_MAP[task]
            evaluator = evaluator_cls(config)

            if shared_model is None:
                if bnb_meta is not None:
                    # BNB：直接用 metadata 參數載入，不走 base_evaluator._load_bnb_model()
                    # 因為原始模型的 config.json 沒有 BNB quantization_config
                    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
                    bnb_cfg = BitsAndBytesConfig(
                        load_in_4bit=(bnb_meta["bits"] == 4),
                        load_in_8bit=(bnb_meta["bits"] == 8),
                        bnb_4bit_quant_type=bnb_meta.get("quant_type", "nf4"),
                        bnb_4bit_use_double_quant=bnb_meta.get("use_double_quant", False),
                        bnb_4bit_compute_dtype=dtype_map.get(
                            bnb_meta.get("compute_dtype", "bfloat16"), torch.bfloat16
                        ),
                    )
                    shared_model = AutoModelForCausalLM.from_pretrained(
                        eval_model_path,
                        quantization_config=bnb_cfg,
                        device_map={"": "cuda:0"},
                        trust_remote_code=True,
                    )
                    shared_tokenizer = AutoTokenizer.from_pretrained(eval_model_path)
                    if shared_tokenizer.pad_token is None:
                        shared_tokenizer.pad_token = (
                            shared_tokenizer.unk_token or shared_tokenizer.eos_token
                        )
                    evaluator.model = shared_model
                    evaluator.tokenizer = shared_tokenizer
                    evaluator._use_vllm = False
                    evaluator._setup_generation_pipeline()
                else:
                    # 其他量化類型：讓 base_evaluator.load_model() 處理
                    evaluator.load_model(quant_type)
                    shared_model = evaluator.model
                    shared_tokenizer = evaluator.tokenizer
            else:
                # 後續 task：注入已載入的 model / tokenizer
                evaluator.model = shared_model
                evaluator.tokenizer = shared_tokenizer
                evaluator._use_vllm = False
                evaluator._setup_generation_pipeline()

            results = evaluator.evaluate()
            evaluator.save_results(results)

            acc  = results.get("accuracy", results.get("pass@1", 0.0))
            lat  = results.get("total_generation_time_sec", 0.0)
            vram = results.get("gpu_peak_mb", 0.0) / 1024.0
            emit = results.get("emissions_kg_co2", 0.0)

            all_task_results[task] = {"accuracy": acc, "latency": lat, "vram": vram, "emissions": emit}
            total_acc  += acc
            total_lat  += lat
            total_emit += emit
            vram_list.append(vram)

            logger.info(f"{task.upper()} 完成: accuracy={acc:.4f}, total={results.get('total', 0)}")

        except Exception as e:
            logger.error(f"{task} 評估失敗: {e}")
            traceback.print_exc()
            all_task_results[task] = {"accuracy": 0.0, "latency": 0.0, "vram": 0.0, "emissions": 0.0}

    # 釋放模型記憶體：先 unload_model（del self.model + gc + empty_cache + reset_peak）
    # 再做一次 gc 確保子進程乾淨退出
    if evaluator is not None:
        evaluator.unload_model()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n = len(tasks_list)
    avg_acc  = total_acc  / n if n else 0.0
    avg_lat  = total_lat  / n if n else 0.0
    max_vram = max(vram_list) if vram_list else 0.0
    avg_emit = total_emit / n if n else 0.0

    if not baseline_metrics:
        return {
            "accuracy": avg_acc, "latency": avg_lat, "vram": max_vram, "emissions": avg_emit,
            "score": 1.0,
            "details": all_task_results,
        }

    # 歸一化計分（log scale）
    base_acc  = baseline_metrics.get("accuracy",  1e-6)
    base_lat  = baseline_metrics.get("latency",   1e-6)
    base_vram = baseline_metrics.get("vram",      1e-6)
    base_emit = baseline_metrics.get("emissions", 1e-6)

    norm_acc  = avg_acc  / (base_acc  + 1e-6)
    norm_lat  = base_lat / (avg_lat   + 1e-6)
    norm_vram = base_vram / (max_vram + 1e-6)
    norm_emit = base_emit / (avg_emit  + 1e-6)

    weight_score = 1.0 + (
        weights.get("acc",  0.0) * math.log(norm_acc  + 1e-9) +
        weights.get("lat",  0.0) * math.log(norm_lat  + 1e-9) +
        weights.get("vram", 0.0) * math.log(norm_vram + 1e-9) +
        weights.get("emit", 0.0) * math.log(norm_emit + 1e-9)
    )

    # Accuracy penalty: a * max(0, (base_acc - t) - acc)
    penalty = pen_a * max(0.0, (base_acc - pen_t) - avg_acc)
    final_score = weight_score - penalty
    if penalty > 0:
        logger.info(f"Accuracy penalty 觸發: {penalty:.4f} "
                    f"(threshold={base_acc - pen_t:.4f}, acc={avg_acc:.4f})")

    return {
        "accuracy": avg_acc, "latency": avg_lat, "vram": max_vram, "emissions": avg_emit,
        "score": final_score,
        "details": all_task_results,
    }
