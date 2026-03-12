from pydantic import BaseModel, Field
from typing import Optional


class StrategySuggestion(BaseModel):
    reasoning: str = Field(..., description="分析為何選擇此策略組合")
    mode: str = Field(..., description="[asvd_only, quant_only, sparse_only, hybrid]")

    # ── ASVD 參數 ────────────────────────────────────────────────────────────
    alpha: Optional[float] = Field(None, ge=0.3, le=0.7)
    param_ratio_target: Optional[float] = Field(None, ge=0.7, le=0.99)
    scaling_method: str = Field(default="fisher",
                                description="[abs_mean, abs_max, fisher]")

    # ── SparseGPT 參數 ───────────────────────────────────────────────────────
    sparsity_ratio: Optional[float] = Field(None, ge=0.0, le=0.9,
                                            description="稀疏比例 0.1-0.9，0.0=不稀疏")
    sparsity_structure: str = Field(default="unstructured",
                                    description="[unstructured, 2:4, 4:8]")

    # ── 量化方法選擇 ─────────────────────────────────────────────────────────
    quant_method: str = Field(default="none",
                              description="[gptq, awq, qqq, bnb, none]")

    # ── GPTQ 專用（AWQ/QQQ 忽略這些）───────────────────────────────────────
    quant_bits: int = Field(default=4,
                            description="GPTQ:[2,3,4,8] | AWQ:固定4 | QQQ:固定4 | BNB:[4,8]")
    quant_group_size: int = Field(default=128,
                                  description="GPTQ:[-1,16,32,64,128,256] | AWQ:[16,32,64,128] | QQQ:[-1,128]")
    quant_format: str = Field(default="gptq",
                              description="GPTQ:[gptq,gptq_v2] | AWQ:自動gemm | QQQ:自動qqq")
    damp_percent: float = Field(default=0.05,
                                description="GPTQ:[0.005,0.01,0.05,0.1] | QQQ:[0.001,0.005,0.01]")
    mse: float = Field(default=0.0,
                       description="GPTQ 專用 MSE 正則化：[0.0,0.01,0.05,0.1]，0.0=停用")

    # ── BNB 專用 ─────────────────────────────────────────────────────────────
    quant_type: str = Field(default="nf4",
                            description="BNB 專用：[nf4, fp4]")
    use_double_quant: bool = Field(default=False,
                                   description="BNB 專用：雙重量化節省 VRAM")

    def to_log_dict(self):
        d = {"mode": self.mode}

        if self.alpha is not None:
            d["asvd"] = {
                "alpha": self.alpha,
                "ratio": self.param_ratio_target,
                "scaling": self.scaling_method,
            }

        _structured = self.sparsity_structure not in ("unstructured", None)
        if _structured or (self.sparsity_ratio and self.sparsity_ratio > 0):
            d["sparse"] = {"structure": self.sparsity_structure}
            if not _structured:
                d["sparse"]["ratio"] = self.sparsity_ratio

        if self.quant_method != "none":
            d["quant"] = {
                "method": self.quant_method,
                "bits": self.quant_bits,
                "group_size": self.quant_group_size,
            }
            if self.quant_method == "gptq":
                d["quant"].update({
                    "format": self.quant_format,
                    "damp": self.damp_percent,
                    "mse": self.mse,
                })
            elif self.quant_method == "qqq":
                d["quant"].update({
                    "damp": self.damp_percent,
                })
            elif self.quant_method == "bnb":
                d["quant"].update({
                    "type": self.quant_type,
                    "double_quant": self.use_double_quant,
                })

        return d
