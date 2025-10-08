"""
量化配置類
===========

定義三種量化方法的配置參數：
- AWQConfig: AWQ 量化配置
- BNBConfig: BitsAndBytes 量化配置  
- GPTQConfig: GPTQ 量化配置
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AWQConfig:
    """
    AWQ (Activation-aware Weight Quantization) 配置
    
    Args:
        zero_point: 是否使用零點量化
        q_group_size: 量化分組大小
        w_bit: 權重位元數 (通常為 4)
        version: 量化版本 (gemm, gemv, marlin, gemv_fast)
        modules_to_not_convert: 不進行量化的模組列表
        output_dir: 輸出目錄 (None 則自動生成)
    """
    zero_point: bool = True
    q_group_size: int = 128
    w_bit: int = 4
    version: str = field(default="gemm", metadata={"choices": ["gemm", "gemv", "marlin", "gemv_fast"]})
    modules_to_not_convert: Optional[List[str]] = None
    output_dir: Optional[str] = None


@dataclass
class BNBConfig:
    """
    BitsAndBytes 量化配置
    
    Args:
        bits: 量化位元數 (4 或 8)
        bnb_4bit_quant_type: 4-bit 量化類型 (nf4, fp4)
        bnb_4bit_use_double_quant: 是否使用雙重量化
        bnb_4bit_compute_dtype: 計算資料型別
        llm_int8_threshold: INT8 量化閾值
        llm_int8_skip_modules: 跳過 INT8 量化的模組
        llm_int8_has_fp16_weight: 是否有 FP16 權重
        llm_int8_enable_fp32_cpu_offload: 是否啟用 FP32 CPU offload
        output_dir: 輸出目錄 (None 則自動生成)
    """
    bits: int = 4
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: Optional[List[str]] = None
    llm_int8_has_fp16_weight: bool = False
    llm_int8_enable_fp32_cpu_offload: bool = False
    output_dir: Optional[str] = None


@dataclass
class GPTQConfig:
    """
    GPTQ (Generalized Post-training Quantization) 配置
    
    Args:
        bits: 量化位元數 (通常為 4)
        group_size: 分組大小 (-1 表示不分組)
        damp_percent: 阻尼百分比
        desc_act: 是否使用降序激活順序
        sym: 是否使用對稱量化
        true_sequential: 是否使用真實順序量化
        calib_num: 校準樣本數量
        output_dir: 輸出目錄 (None 則自動生成)
    """
    bits: int = 4
    group_size: int = -1
    damp_percent: float = 0.01
    desc_act: bool = True
    sym: bool = True
    true_sequential: bool = True
    calib_num: int = 32
    output_dir: Optional[str] = None
