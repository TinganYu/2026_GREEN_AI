from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class AWQConfig:
    zero_point: bool = True
    q_group_size: int = 128
    w_bit: int = 4
    version: str = field(default="gemm", metadata={"choices": ["gemm", "gemv", "marlin", "gemv_fast"]})
    modules_to_not_convert: Optional[List[str]] = None
    output_dir: Optional[str] = None

@dataclass
class BNBConfig:
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
    bits: int = 4
    group_size: int = -1
    damp_percent: float = 0.01
    desc_act: bool = True
    sym: bool = True
    true_sequential: bool = True
    calib_num: int = 32
    output_dir: Optional[str] = None
