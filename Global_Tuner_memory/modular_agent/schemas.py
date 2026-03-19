# from pydantic import BaseModel, Field, model_validator
# from typing import Optional, Literal


# class StrategySuggestion(BaseModel):
#     reasoning: str = Field(..., description="分析為何選擇此策略組合")
#     mode: Literal[
#         "asvd_only", 
#         "gptq", 
#         "awq", 
#         "qqq", 
#         "bnb", 
#         "sparse_unstructured", 
#         "sparse_structured", 
#         "hybrid_asvd_bnb"
#     ] = Field(description="壓縮模式")

#     # ── ASVD 參數 ────────────────────────────────────────────────────────────
#     alpha: Optional[float] = Field(None, ge=0.3, le=0.7)
#     param_ratio_target: Optional[float] = Field(None, ge=0.7, le=0.99)
#     scaling_method: str = Field(default="fisher",
#                                 description="[abs_mean, abs_max, fisher]")

#     # ── SparseGPT 參數 ───────────────────────────────────────────────────────
#     sparsity_ratio: Optional[float] = Field(None, ge=0.0, le=0.9,
#                                             description="稀疏比例 0.1-0.9，0.0=不稀疏")
#     sparsity_structure: str = Field(default="unstructured",
#                                     description="[unstructured, 2:4, 4:8]")

#     # ── 量化方法選擇 ─────────────────────────────────────────────────────────
#     quant_method: str = Field(default="none",
#                               description="[gptq, awq, qqq, bnb, none]")

#     # ── GPTQ 專用（AWQ/QQQ 忽略這些）───────────────────────────────────────
#     quant_bits: int = Field(default=4,
#                             description="GPTQ:[2,3,4,8] | AWQ:固定4 | QQQ:固定4 | BNB:[4,8]")
#     quant_group_size: int = Field(default=128,
#                                   description="GPTQ:[-1,16,32,64,128,256] | AWQ:[16,32,64,128] | QQQ:[-1,128]")
#     quant_format: str = Field(default="gptq",
#                               description="GPTQ:[gptq,gptq_v2] | AWQ:自動gemm | QQQ:自動qqq")
#     damp_percent: float = Field(default=0.05,
#                                 description="GPTQ:[0.005,0.01,0.05,0.1] | QQQ:[0.001,0.005,0.01]")
#     mse: float = Field(default=0.0,
#                        description="GPTQ 專用 MSE 正則化：[0.0,0.01,0.05,0.1]，0.0=停用")

#     # ── BNB 專用 ─────────────────────────────────────────────────────────────
#     quant_type: str = Field(default="nf4",
#                             description="BNB 專用：[nf4, fp4]")
#     use_double_quant: bool = Field(default=False,
#                                    description="BNB 專用：雙重量化節省 VRAM")

#     @model_validator(mode="after")
#     def _fix_bnb_double_quant(self):
#         if self.quant_method == "bnb" and self.quant_bits != 4:
#             self.use_double_quant = False
#         return self

#     def to_log_dict(self):
#         d = {"mode": self.mode}

#         if self.alpha is not None:
#             d["asvd"] = {
#                 "alpha": self.alpha,
#                 "ratio": self.param_ratio_target,
#                 "scaling": self.scaling_method,
#             }

#         _structured = self.sparsity_structure not in ("unstructured", None)
#         if _structured or (self.sparsity_ratio and self.sparsity_ratio > 0):
#             d["sparse"] = {"structure": self.sparsity_structure}
#             if not _structured:
#                 d["sparse"]["ratio"] = self.sparsity_ratio

#         if self.quant_method != "none":
#             d["quant"] = {
#                 "method": self.quant_method,
#                 "bits": self.quant_bits,
#                 "group_size": self.quant_group_size,
#             }
#             if self.quant_method == "gptq":
#                 d["quant"].update({
#                     "format": self.quant_format,
#                     "damp": self.damp_percent,
#                     "mse": self.mse,
#                 })
#             elif self.quant_method == "qqq":
#                 d["quant"].update({
#                     "damp": self.damp_percent,
#                 })
#             elif self.quant_method == "bnb":
#                 d["quant"].update({
#                     "type": self.quant_type,
#                     "double_quant": self.use_double_quant,
#                 })

#         return d

from pydantic import BaseModel, Field, model_validator
from typing import Optional

class StrategySuggestion(BaseModel):
    reasoning: str = Field(..., description="分析為何選擇此策略組合")
    mode: str = Field(..., description="[asvd_only, gptq, awq, qqq, bnb, sparse_unstructured, sparse_structured, hybrid_asvd_bnb]")

    # ── ASVD 參數 ────────────────────────────────────────────────────────────
    alpha: Optional[float] = Field(None, ge=0.3, le=0.7)
    param_ratio_target: Optional[float] = Field(None, ge=0.70, le=0.99)
    scaling_method: str = Field(default="fisher", description="[abs_mean, abs_max, fisher]")

    # ── SparseGPT 參數 ───────────────────────────────────────────────────────
    # 改為連續區間
    sparsity_ratio: Optional[float] = Field(None, ge=0.3, le=0.7, description="稀疏比例: 0.3 到 0.7 的浮點數")
    sparsity_structure: str = Field(default="2:4", description="結構化稀疏: [2:4, 4:8]")

    # ── 量化參數 (由 mode 直接決定方法) ──────────────────────────────────────────
    quant_bits: int = Field(default=4, description="GPTQ:[2,3,4,8] | BNB:[4,8] | AWQ/QQQ:固定4")
    quant_group_size: int = Field(default=128, description="GPTQ:[16,32,64,128,256] | AWQ:[16,32,64,128] | QQQ:[-1,128]")
    quant_format: str = Field(default="gptq", description="GPTQ:[gptq,gptq_v2]")
    
    # damp_percent 範圍放寬，交由 LLM 在對應 mode 下生成合理數值
    damp_percent: float = Field(default=0.05, description="GPTQ:(0.001~0.1) | QQQ:(0.0005~0.05)")
    
    # 以下兩個參數固定，不需要 LLM 去猜，但保留欄位供 executors.py 讀取
    mse: float = Field(default=0.0, description="固定值")
    quant_type: str = Field(default="nf4", description="固定值")
    use_double_quant: bool = Field(default=False, description="BNB 專用：雙重量化 (僅 bits=4 有效)")

    @model_validator(mode="after")
    def _validate_mode_constraints(self):
        # 1. BNB / Hybrid constraints
        if self.mode in ("bnb", "hybrid_asvd_bnb"):
            if self.quant_bits not in [4, 8]:
                raise ValueError(f"BNB requires quant_bits to be 4 or 8, got {self.quant_bits}")
            if self.quant_bits != 4:
                self.use_double_quant = False

        # 2. GPTQ constraints
        elif self.mode == "gptq":
            if self.quant_bits not in [2, 3, 4, 8]:
                raise ValueError(f"GPTQ requires quant_bits in [2, 3, 4, 8], got {self.quant_bits}")
            if self.quant_group_size not in [16, 32, 64, 128, 256]:
                raise ValueError(f"GPTQ invalid quant_group_size: {self.quant_group_size}")
            if not (0.001 <= self.damp_percent <= 0.1):
                raise ValueError(f"GPTQ damp_percent must be between 0.001 and 0.1, got {self.damp_percent}")

        # 3. AWQ constraints
        elif self.mode == "awq":
            self.quant_bits = 4  # AWQ is fixed at 4-bit
            if self.quant_group_size not in [16, 32, 64, 128]:
                raise ValueError(f"AWQ invalid quant_group_size: {self.quant_group_size}")

        # 4. QQQ constraints
        elif self.mode == "qqq":
            self.quant_bits = 4  # QQQ is fixed at 4-bit
            if self.quant_group_size not in [-1, 128]:
                raise ValueError(f"QQQ requires quant_group_size to be -1 or 128, got {self.quant_group_size}")
            if not (0.0005 <= self.damp_percent <= 0.05):
                raise ValueError(f"QQQ damp_percent must be between 0.0005 and 0.05, got {self.damp_percent}")

        return self

    def to_log_dict(self):
        d = {"mode": self.mode}

        if self.mode in ("asvd_only", "hybrid_asvd_bnb") and self.alpha is not None:
            d["asvd"] = {
                "alpha": self.alpha,
                "ratio": self.param_ratio_target,
                "scaling": self.scaling_method,
            }

        if self.mode in ("sparse_unstructured", "sparse_structured"):
            d["sparse"] = {"structure": self.sparsity_structure if self.mode == "sparse_structured" else "unstructured"}
            if self.mode == "sparse_unstructured":
                d["sparse"]["ratio"] = self.sparsity_ratio

        if self.mode in ("gptq", "awq", "qqq", "bnb", "hybrid_asvd_bnb"):
            q_method = "bnb" if self.mode == "hybrid_asvd_bnb" else self.mode
            d["quant"] = {
                "method": q_method,
                "bits": 4 if self.mode in ("awq", "qqq") else self.quant_bits,
                "group_size": self.quant_group_size,
            }
            if q_method == "gptq":
                d["quant"].update({"format": self.quant_format, "damp": self.damp_percent, "mse": self.mse})
            elif q_method == "qqq":
                d["quant"].update({"damp": self.damp_percent})
            elif q_method == "bnb":
                d["quant"].update({"type": self.quant_type, "double_quant": self.use_double_quant})

        return d