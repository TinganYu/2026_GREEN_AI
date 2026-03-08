"""
SparseGPT 壓縮模組
===================

使用 llm-compressor 套件執行 SparseGPT 稀疏化壓縮。

安裝需求:
    pip install llmcompressor transformers datasets
"""

import gc
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch

logger = logging.getLogger("Method.Sparse")
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
class SparseConfig:
    """SparseGPT 稀疏化配置"""
    sparsity_ratio: float = 0.5         # 非結構化稀疏比例 (0.0-1.0)
    structure: str = "unstructured"     # "unstructured" | "2:4" | "N:M"
    targets: str = "Linear"
    ignore: List[str] = field(default_factory=lambda: ["lm_head"])

    # 校準設定
    calib_dataset: str = "openai/gsm8k"
    calib_split: str = "train"
    calib_text_column: str = "question"
    n_calib_samples: int = 512
    max_seq_length: int = 2048

    # 輸出設定
    output_dir: Optional[str] = None
    save_compressed: bool = True


def run_sparse(model_path: str, config: SparseConfig) -> str:
    """
    執行 SparseGPT 稀疏化壓縮。

    Args:
        model_path: 模型路徑或 HuggingFace model ID
        config: SparseConfig 配置物件

    Returns:
        壓縮後模型的本地路徑
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from llmcompressor import oneshot
        from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            f"請先安裝必要的依賴:\n"
            f"   pip install llmcompressor transformers datasets\n"
            f"錯誤: {e}"
        )

    model_id = model_path.split("/")[-1]

    # 解析結構模式
    prunen, prunem = _parse_structure(config.structure)

    # 生成輸出目錄
    output_dir = config.output_dir or _generate_output_dir(model_id, config, prunen, prunem)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"SparseGPT 稀疏化開始")
    logger.info(f"模型: {model_path}")
    if prunen > 0:
        logger.info(f"稀疏模式: {prunen}:{prunem} 結構化稀疏")
    else:
        logger.info(f"稀疏比例: {config.sparsity_ratio * 100:.0f}%（非結構化）")
    logger.info(f"輸出: {output_dir}")
    logger.info("=" * 70)

    # 載入 tokenizer
    logger.info("載入 tokenizer...")
    tokenizer = _load_tokenizer(model_path)

    # 載入模型
    logger.info("載入模型...")
    _cleanup_memory()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    # 準備校準資料
    logger.info("準備校準資料...")
    calib_data = _load_calib_data(config)

    # 建立 SparseGPT recipe
    modifier_kwargs = {
        "targets": config.targets,
        "sparsity": config.sparsity_ratio,
    }
    if prunen > 0:
        modifier_kwargs["mask_structure"] = f"{prunen}:{prunem}"
        actual_sparsity = (prunem - prunen) / prunem
        logger.info(f"N:M 稀疏比例: {actual_sparsity:.0%}")

    recipe = [SparseGPTModifier(**modifier_kwargs)]

    # 執行稀疏化
    logger.info("開始 SparseGPT 稀疏化...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=calib_data,
        recipe=recipe,
        max_seq_length=config.max_seq_length,
        num_calibration_samples=config.n_calib_samples,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info(f"GPU 峰值記憶體: {peak_mb:.2f} MB")

    # 儲存模型
    logger.info("儲存稀疏化模型...")
    if config.save_compressed:
        model.save_pretrained(output_dir, save_compressed=True)
    else:
        model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model
    _cleanup_memory()

    logger.info("=" * 70)
    logger.info(f"SparseGPT 完成，輸出: {output_dir}")
    logger.info("=" * 70)

    return output_dir


def _parse_structure(structure: str):
    """解析結構模式，返回 (prunen, prunem)"""
    if structure == "unstructured":
        return 0, 0
    if ":" in structure:
        parts = structure.split(":")
        return int(parts[0]), int(parts[1])
    return 0, 0


def _generate_output_dir(model_id: str, config: SparseConfig, prunen: int, prunem: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prunen > 0:
        suffix = f"{prunen}x{prunem}"
    else:
        suffix = f"sparse{int(config.sparsity_ratio * 100)}"
    return f"quantized/{model_id}-sparsegpt-{suffix}-{ts}"


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_calib_data(config: SparseConfig):
    from datasets import load_dataset, Dataset

    if config.calib_dataset == "openai/gsm8k":
        ds = load_dataset(config.calib_dataset, "main", split=config.calib_split)
    else:
        ds = load_dataset(config.calib_dataset, split=config.calib_split)

    n = min(config.n_calib_samples, len(ds))
    ds = ds.select(range(n))

    col = config.calib_text_column

    def extract_text(example):
        return {"text": example.get(col, "")}

    ds = ds.map(extract_text, remove_columns=ds.column_names)
    return ds


def _cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
