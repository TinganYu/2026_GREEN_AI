"""
使用 YAML 配置執行量化
======================

從 YAML 配置文件讀取參數並執行模型量化

Usage:
    python run_quant_yaml.py
"""

import sys
import os
from pathlib import Path
import yaml

# 添加父目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantization import (
    QuantizationManager,
    AWQConfig,
    BNBConfig,
    GPTQConfig,
    logger
)


def load_config(config_path: Path) -> dict:
    """載入 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_quant_config(quant_section: dict, backend: str):
    """根據後端類型創建量化配置"""
    if backend == "awq":
        return AWQConfig(**quant_section["awq"])
    elif backend == "bnb":
        return BNBConfig(**quant_section["bnb"])
    elif backend == "gptq":
        return GPTQConfig(**quant_section["gptq"])
    else:
        raise ValueError(f"不支援的量化後端: {backend}")


def main():
    """主函數"""
    # 配置文件路徑
    BASE_DIR = Path(__file__).resolve().parent
    config_path = BASE_DIR / "quant_config.yaml"
    
    if not config_path.exists():
        logger.error(f"❌ 配置文件不存在: {config_path}")
        return
    
    logger.info("="*80)
    logger.info("🚀 從 YAML 配置執行量化")
    logger.info("="*80)
    
    # 讀取配置
    logger.info(f"📖 讀取配置: {config_path}")
    cfg = load_config(config_path)
    
    # 提取參數
    model_path = cfg["model"]["name"]
    hf_token = cfg.get("hf_token")
    
    # 量化配置
    quant_section = cfg.get("quantization", {})
    backend = quant_section.get("framework", "").lower()
    
    logger.info(f"📦 模型: {model_path}")
    logger.info(f"🔧 量化後端: {backend.upper()}")
    
    # 創建量化配置
    try:
        quant_cfg = create_quant_config(quant_section, backend)
        logger.info(f"✓ 量化配置創建完成")
    except Exception as e:
        logger.error(f"❌ 量化配置創建失敗: {e}")
        return
    
    # 執行量化
    try:
        output_path = QuantizationManager.quantize(
            model_path=model_path,
            backend=backend,
            config=quant_cfg,
            hf_token=hf_token,
        )
        
        logger.info("="*80)
        logger.info("🎉 量化流程完成!")
        logger.info(f"📁 輸出路徑: {output_path}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error("="*80)
        logger.error(f"❌ 量化失敗: {e}")
        logger.error("="*80)
        raise


if __name__ == "__main__":
    main()
