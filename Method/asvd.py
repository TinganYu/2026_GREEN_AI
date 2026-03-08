"""
ASVD 壓縮模組
=============

封裝 ASVD4LLM 低秩分解壓縮，提供統一的 dataclass 介面。
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("Method.ASVD")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
logger.propagate = False

# Resolve paths
_ROOT_DIR = Path(__file__).resolve().parent.parent  # Green_AI/
_ASVD_ROOT = _ROOT_DIR / "ASVD4LLM"
_ASVD_REPO_DIR = _ASVD_ROOT / "huggingface_repos"


@dataclass
class ASVDConfig:
    """ASVD 低秩分解配置"""
    alpha: float = 0.5
    param_ratio_target: float = 0.9
    scaling_method: str = "fisher"
    act_aware: bool = True
    use_cache: bool = True
    sensitivity_metric: str = "ppl"
    calib_dataset: str = "wikitext2"
    n_calib_samples: int = 32
    seed: int = 233
    rank_align: int = 128
    sigma_fuse: str = "UV"


def run_asvd(model_id: str, config: ASVDConfig) -> str:
    """
    執行 ASVD 低秩分解壓縮。

    Args:
        model_id: HuggingFace 模型 ID 或本地路徑
        config: ASVDConfig 配置物件

    Returns:
        壓縮後模型的本地路徑
    """
    # 動態加入 ASVD 路徑
    sys.path.insert(0, str(_ASVD_ROOT))
    sys.path.insert(0, str(_ASVD_REPO_DIR))

    try:
        from build_asvd_repo import main as asvd_main
    except ImportError as e:
        logger.error(f"無法匯入 build_asvd_repo: {e}")
        logger.error(f"請確認 ASVD4LLM 已存在於: {_ASVD_ROOT}")
        raise

    logger.info(f"--- 執行 ASVD: ratio={config.param_ratio_target}, alpha={config.alpha} ---")
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

    args = argparse.Namespace(
        model_id=model_id,
        alpha=config.alpha,
        param_ratio_target=config.param_ratio_target,
        scaling_method=config.scaling_method,
        act_aware=config.act_aware,
        use_cache=config.use_cache,
        sensitivity_metric=config.sensitivity_metric,
        weight_quant="none",
        eval_mmlu=False,
        calib_dataset=config.calib_dataset,
        n_calib_samples=config.n_calib_samples,
        seed=config.seed,
        compress_kv_cache=False,
        rank_align=config.rank_align,
        sigma_fuse=config.sigma_fuse,
        push=False,
        eval_tasks="",
        eval_ppl="",  # orchestrator 有獨立評估流程，不需重複跑 PPL
        use_bos=False,
        ppl_target=-1.0,
    )

    asvd_main(args)

    # 重建輸出路徑（與 build_asvd_repo.py 邏輯一致）
    model_name = model_id.split("/")[-1]
    output_dir = _ASVD_REPO_DIR.parent / "output" / (
        f"{model_name}-asvd{int(config.param_ratio_target * 100)}"
        f"-alpha{int(config.alpha * 100)}"
    )

    if not output_dir.exists():
        raise RuntimeError(f"ASVD 輸出目錄不存在: {output_dir}")

    logger.info(f"ASVD 完成，輸出: {output_dir}")
    return str(output_dir)
