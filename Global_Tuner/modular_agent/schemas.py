from pydantic import BaseModel, Field
from typing import Optional

class StrategySuggestion(BaseModel):
    reasoning: str = Field(..., description="分析為何選擇此策略組合")
    mode: str = Field(..., description="[asvd_only, quant_only, hybrid]")
    
    # ASVD 參數
    alpha: Optional[float] = Field(None, ge=0.3, le=0.7)
    param_ratio_target: Optional[float] = Field(None, ge=0.7, le=0.99)
    scaling_method: str = Field(default="fisher")
    
    # 量化參數
    quant_method: str = Field(default="none", description="[gptq, awq, bnb, none]")
    quant_bits: int = Field(default=4)
    quant_group_size: Optional[int] = Field(128, description="For GPTQ/AWQ group size. Options: [-1, 32, 64, 128]")
    quant_type: Optional[str] = Field("nf4", description="For BNB: nf4, fp4")
    desc_act: Optional[bool] = Field(False, description="For GPTQ: Whether to use desc_act. Note: act_group_aware requires desc_act=False")
    use_double_quant: Optional[bool] = Field(False, description="For BNB: Whether to use double quantization to save more VRAM")

    def to_log_dict(self):
        """將配置轉為簡潔的字典，方便餵回給 LLM"""
        return {
            "mode": self.mode,
            "asvd": {"alpha": self.alpha, "ratio": self.param_ratio_target} if self.alpha else None,
            "quant": {
                "method": self.quant_method, 
                "bits": self.quant_bits,
                "group_size": self.quant_group_size,
                "type": self.quant_type
            } if self.quant_method != "none" else None
        }
