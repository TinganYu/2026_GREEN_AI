"""
Quantization Package
====================

提供三種主流量化方法的統一接口：
- GPTQ: 使用 auto-gptq 進行 4-bit/8-bit 量化
- AWQ: 使用 autoawq 進行 4-bit 量化
- BNB: 使用 bitsandbytes 進行 4-bit/8-bit 量化

Usage:
    from quantization import QuantizationManager, AWQConfig, BNBConfig, GPTQConfig
    
    # 創建配置
    config = AWQConfig(w_bit=4, q_group_size=128)
    
    # 執行量化
    output_path = QuantizationManager.quantize(
        model_path="meta-llama/Llama-3.2-1B-Instruct",
        backend="awq",
        config=config
    )
"""

from .configs import AWQConfig, BNBConfig, GPTQConfig
from .manager import QuantizationManager
from .base_quantizer import BaseQuantizer
from .awq_quantizer import AWQQuantizer
from .bnb_quantizer import BNBQuantizer
from .gptq_quantizer import GPTQQuantizer
from .utils import logger, safe_load_tokenizer

__version__ = "1.0.0"

__all__ = [
    # 配置類
    "AWQConfig",
    "BNBConfig",
    "GPTQConfig",
    
    # 管理器
    "QuantizationManager",
    
    # 量化器
    "BaseQuantizer",
    "AWQQuantizer",
    "BNBQuantizer",
    "GPTQQuantizer",
    
    # 工具
    "logger",
    "safe_load_tokenizer",
]
