"""
搜尋空間定義
根據 Global_Tuner_v2/modular_agent/llm_client.py 的 prompt 中所列的合法參數範圍定義
"""

# ── ASVD ──────────────────────────────────────────────────────────────────────
ASVD_SPACE = {
    "alpha":              [0.3, 0.4, 0.5, 0.6, 0.7],
    "param_ratio_target": [0.70, 0.80, 0.85, 0.90, 0.95, 0.99],
    "scaling_method":     ["abs_mean", "abs_max", "fisher"],
}

# ── GPTQ ──────────────────────────────────────────────────────────────────────
GPTQ_SPACE = {
    "quant_bits":       [2, 3, 4, 8],
    "quant_group_size": [16, 32, 64, 128, 256],
    "quant_format":     ["gptq", "gptq_v2"],
    "damp_percent":     [0.005, 0.01, 0.05, 0.1],
    "mse":              [0.0, 0.01, 0.05, 0.1],
}

# ── AWQ ───────────────────────────────────────────────────────────────────────
AWQ_SPACE = {
    "quant_group_size": [16, 32, 64, 128],
}

# ── QQQ ───────────────────────────────────────────────────────────────────────
QQQ_SPACE = {
    "quant_group_size": [-1, 128],
    "damp_percent":     [0.001, 0.005, 0.01],
}

# ── BNB ───────────────────────────────────────────────────────────────────────
BNB_SPACE = {
    "quant_bits":       [4, 8],
    "use_double_quant": [False, True],  # 只在 quant_bits=4 時有效，bits=8 固定 False
}

# ── SparseGPT Unstructured ────────────────────────────────────────────────────
SPARSE_UNSTRUCTURED_SPACE = {
    "sparsity_ratio": [0.3, 0.4, 0.5, 0.6, 0.7],
}

# ── SparseGPT Structured ─────────────────────────────────────────────────────
SPARSE_STRUCTURED_SPACE = {
    "sparsity_structure": ["2:4", "4:8"],
}

# ── 模式清單（讓 orchestrator 顯示 & 使用者過濾）───────────────────────────────
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
