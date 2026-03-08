"""
量化模組
=========

統一介面，完全自給自足，不依賴 experiments/ 或 tmp/：
- gptq / awq / qqq → gptqmodel 套件（GPTQModel.load + quantize + save_quantized）
- bnb              → transformers BitsAndBytesConfig（on-the-fly，不儲存）

用法:
    from Method.quantize import QuantConfig, run_quantization

    config = QuantConfig(method="gptq", bits=4, group_size=128)
    output_path = run_quantization("meta-llama/Llama-3.2-1B-Instruct", config)
"""

import gc
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("Method.Quantize")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
logger.propagate = False


@dataclass
class QuantConfig:
    """統一量化配置"""
    method: str = "gptq"       # "gptq" | "awq" | "qqq" | "bnb"
    bits: int = 4
    group_size: int = 128
    format: str = "gptq"       # GPTQModel format（gptq / gemm / marlin / qqq …）
    desc_act: bool = False
    damp_percent: float = 0.05
    mse: float = 0.0
    sym: bool = True

    # BNB 專用
    quant_type: str = "nf4"    # "nf4" | "fp4"
    use_double_quant: bool = False

    # 校準設定（gptq / awq / qqq 使用）
    calib_dataset: str = "openai/gsm8k"
    calib_split: str = "train"
    calib_text_column: str = "question"
    n_calib_samples: int = 256
    batch_size: int = 1

    # 輸出設定
    output_dir: Optional[str] = None
    output_base: str = "quantized"


# ============================================================================
# 公開介面
# ============================================================================

def run_quantization(model_path: str, config: QuantConfig) -> str:
    """
    執行量化。

    Returns:
        量化後模型的本地路徑（BNB 模式回傳原路徑，on-the-fly 不儲存）
    """
    method = config.method.lower()
    logger.info(f"--- 執行量化: {method} ({config.bits} bits) ---")
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

    if method in ("gptq", "awq", "qqq"):
        return _run_gptqmodel(model_path, config)
    elif method == "bnb":
        return _run_bnb(model_path, config)
    else:
        raise ValueError(f"不支援的量化方法: {method}。支援: gptq, awq, qqq, bnb")


def get_bnb_load_kwargs(config: QuantConfig) -> dict:
    """取得 BNB 載入所需的 kwargs，供 executor 注入到 AutoModelForCausalLM。"""
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(config.bits == 4),
        load_in_8bit=(config.bits == 8),
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_use_double_quant=config.use_double_quant,
    )
    return {"quantization_config": bnb_config}


# ============================================================================
# 內部實作
# ============================================================================

def _run_gptqmodel(model_path: str, config: QuantConfig) -> str:
    """直接使用 gptqmodel 套件執行 gptq/awq/qqq 量化。"""
    try:
        from gptqmodel import GPTQModel
        from gptqmodel.quantization.config import QuantizeConfig, METHOD, FORMAT
    except ImportError as e:
        raise ImportError(f"請安裝 gptqmodel: pip install gptqmodel\n錯誤: {e}")

    METHOD_MAP = {
        "gptq": METHOD.GPTQ,
        "awq":  METHOD.AWQ,
        "qqq":  METHOD.QQQ,
    }
    FORMAT_MAP = {
        "gptq":     FORMAT.GPTQ,
        "gptq_v2":  FORMAT.GPTQ_V2,
        "marlin":   FORMAT.MARLIN,
        "gemm":     FORMAT.GEMM,
        "gemv":     FORMAT.GEMV,
        "gemv_fast":FORMAT.GEMV_FAST,
        "llm_awq":  FORMAT.LLM_AWQ,
        "qqq":      FORMAT.QQQ,
    }

    method_str = config.method.lower()
    # AWQ 預設用 GEMM 格式（最穩定）
    default_fmt = "gemm" if method_str == "awq" else "gptq"
    format_str = (config.format or default_fmt).lower()

    if method_str not in METHOD_MAP:
        raise ValueError(f"未知的 method: {method_str}")
    if format_str not in FORMAT_MAP:
        raise ValueError(f"未知的 format: {format_str}。可用: {list(FORMAT_MAP.keys())}")

    quant_config = QuantizeConfig(
        quant_method=METHOD_MAP[method_str],
        bits=config.bits,
        group_size=config.group_size,
        format=FORMAT_MAP[format_str],
        desc_act=config.desc_act,
        damp_percent=config.damp_percent,
        mse=config.mse,
        sym=config.sym,
    )

    # 輸出目錄
    model_id = model_path.split("/")[-1]
    output_dir = config.output_dir or _generate_output_dir(model_id, config)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"模型: {model_path}")
    logger.info(f"Method: {method_str}, Bits: {config.bits}, Group: {config.group_size}")
    logger.info(f"輸出: {output_dir}")
    logger.info("=" * 70)

    # 載入 tokenizer
    tokenizer = _load_tokenizer(model_path)

    # 載入模型（量化前）
    _cleanup_memory()
    model = GPTQModel.load(model_path, quant_config)

    # 校準資料
    calib_data = _prepare_calib_data(config)
    logger.info(f"校準樣本數: {len(calib_data)}")

    # 執行量化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model.quantize(calib_data, batch_size=config.batch_size)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"GPU 峰值記憶體: {peak_mb:.2f} MB")

    # 儲存
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model
    _cleanup_memory()

    # 修復 AWQ/QQQ offload 路徑：將相對 offload 目錄移至 output_dir 並改為絕對路徑
    _fix_offload_path(output_dir)

    logger.info(f"量化完成，輸出: {output_dir}")
    return output_dir


def _run_bnb(model_path: str, config: QuantConfig) -> str:
    """BNB on-the-fly 量化（不儲存），回傳原路徑供 executor 以 BNB 載入。"""
    logger.info("BNB 為 on-the-fly 量化，不預先儲存")
    logger.info(f"路徑: {model_path} | {config.bits}bit / {config.quant_type}")
    return model_path


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _prepare_calib_data(config: QuantConfig) -> list:
    """載入校準資料集，返回 list[str]。"""
    from datasets import load_dataset

    # 需要 config name 的已知 dataset
    _DATASET_CONFIGS = {
        "openai/gsm8k": "main",
        "wikitext":     "wikitext-2-raw-v1",
    }
    ds_name = config.calib_dataset
    if ds_name in _DATASET_CONFIGS:
        ds = load_dataset(ds_name, _DATASET_CONFIGS[ds_name], split=config.calib_split)
    else:
        ds = load_dataset(ds_name, split=config.calib_split)
    n = min(config.n_calib_samples, len(ds))
    ds = ds.select(range(n))

    columns = [c.strip() for c in config.calib_text_column.split("+")]
    data = []
    for example in ds:
        parts = [str(example.get(c, "")) for c in columns if example.get(c)]
        data.append(" ".join(parts))
    return data


def _generate_output_dir(model_id: str, config: QuantConfig) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(config.output_base, f"{model_id}-{config.method}-{config.bits}bit-{ts}")


def _fix_offload_path(output_dir: str):
    """
    AWQ/QQQ 量化時 gptqmodel 會把 scales 等暫存在相對路徑
    ./gptqmodel_offload/uuid/，若不修正則推理時因 CWD 改變而找不到。
    此函數：讀取 config.json → 若 offload 目錄存在 → 移入 output_dir/gptq_offload → 更新為絕對路徑。
    """
    import json
    import shutil

    config_path = Path(output_dir) / "config.json"
    if not config_path.exists():
        return
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        quant_cfg = cfg.get("quantization_config", {})
        meta = quant_cfg.get("meta", {})
        offload_rel = meta.get("offload_to_disk_path", "")
        if not offload_rel:
            return
        offload_src = Path(offload_rel)
        if not offload_src.exists():
            logger.warning(f"offload 目錄不存在，跳過修復: {offload_src}")
            return
        new_offload_dir = Path(output_dir) / "gptq_offload"
        if new_offload_dir.exists():
            shutil.rmtree(new_offload_dir)
        shutil.copytree(str(offload_src), str(new_offload_dir))
        meta["offload_to_disk_path"] = str(new_offload_dir.resolve())
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        logger.info(f"offload 路徑已更新為絕對路徑: {new_offload_dir.resolve()}")
    except Exception as e:
        logger.warning(f"修復 offload 路徑失敗（非致命）: {e}")


def _cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
