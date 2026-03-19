"""
搜尋空間定義

取樣慣例：
  tuple (low, high)         → suggest_float，linear scale
  tuple (low, high, "log")  → suggest_float，log scale（適合跨數量級的參數，如 damp）
  list  [...]               → suggest_categorical（只有這幾個合法值）
"""

# ── ASVD ──────────────────────────────────────────────────────────────────────
ASVD_SPACE = {
    "alpha":              (0.3, 0.7),                      # linear：範圍均勻，linear OK
    "param_ratio_target": (0.70, 0.99),                    # linear：範圍均勻，linear OK
    "scaling_method":     ["abs_mean", "abs_max", "fisher"],
}

# ── GPTQ ──────────────────────────────────────────────────────────────────────
GPTQ_SPACE = {
    "quant_bits":       [2, 3, 4, 8],                      # categorical（只有這些合法值）
    "quant_group_size": [-1, 16, 32, 64, 128, 256],        # categorical（-1=全矩陣，其餘須為 2 的冪次）
    "quant_format":     ["gptq", "gptq_v2"],               # categorical
    "damp_percent":     (0.001, 0.1, "log"),               # log：0.001↔0.01 和 0.01↔0.1 各佔一半探索量
    # mse 移除：幾乎都用 0.0，放進 search space 只是浪費 trial
}

# ── AWQ ───────────────────────────────────────────────────────────────────────
AWQ_SPACE = {
    "quant_group_size": [16, 32, 64, 128],                 # categorical（必須是 2 的冪次）
}

# ── QQQ ───────────────────────────────────────────────────────────────────────
QQQ_SPACE = {
    "quant_group_size": [-1, 128],                         # categorical（-1 = 無 group）
    "damp_percent":     (0.0005, 0.05, "log"),             # log：同 GPTQ 理由
}

# ── BNB ───────────────────────────────────────────────────────────────────────
BNB_SPACE = {
    "quant_bits":       [4, 8],                            # categorical
    "use_double_quant": [False, True],                     # categorical（只在 bits=4 有效）
    # quant_type 固定 nf4：nf4 幾乎必贏 fp4，放進 space 只是浪費 trial
}

# ── SparseGPT Unstructured ────────────────────────────────────────────────────
SPARSE_UNSTRUCTURED_SPACE = {
    "sparsity_ratio": (0.3, 0.7),                          # linear：範圍均勻，linear OK
}

# ── SparseGPT Structured ─────────────────────────────────────────────────────
SPARSE_STRUCTURED_SPACE = {
    "sparsity_structure": ["2:4", "4:8"],                  # categorical
}

# ── 模式清單 ──────────────────────────────────────────────────────────────────
ALL_MODES = [
    "asvd_only",
    "gptq",
    "awq",
    "qqq",
    "bnb",
    "sparse_unstructured",
    "sparse_structured",
    "hybrid_asvd_bnb",
]
