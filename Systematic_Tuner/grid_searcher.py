"""
Grid Search 實作
枚舉 search_space.py 中所有合法的 StrategySuggestion 組合
"""

import itertools
import logging
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Global_Tuner_v2.modular_agent.schemas import StrategySuggestion
from .search_space import (
    ASVD_SPACE, GPTQ_SPACE, AWQ_SPACE, QQQ_SPACE, BNB_SPACE,
    SPARSE_UNSTRUCTURED_SPACE, SPARSE_STRUCTURED_SPACE, ALL_MODES,
)

logger = logging.getLogger("GridSearcher")


def _cartesian(space: dict) -> List[dict]:
    """將 {key: [values]} dict 轉成所有組合的 list[dict]"""
    keys = list(space.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*space.values())]


class GridSearcher:
    """
    以固定順序逐一提供所有合法量化配置。

    modes 可過濾要搜尋的模式（預設搜尋全部）：
        ['asvd_only', 'gptq', 'awq', 'qqq', 'bnb',
         'sparse_unstructured', 'sparse_structured',
         'hybrid_asvd_bnb', 'hybrid_sparse_gptq']
    """

    def __init__(self, modes: Optional[List[str]] = None):
        self.modes = modes or ALL_MODES
        self._configs: List[StrategySuggestion] = []
        self._idx = 0
        self._build()
        logger.info(f"Grid search 共產生 {len(self._configs)} 個配置 (模式: {self.modes})")

    # ──────────────────────────────────────────────────────────────────────────
    def _build(self):
        cfgs = []

        if "asvd_only" in self.modes:
            for p in _cartesian(ASVD_SPACE):
                cfgs.append(StrategySuggestion(
                    reasoning="grid:asvd_only",
                    mode="asvd_only", **p,
                ))

        if "gptq" in self.modes:
            for p in _cartesian(GPTQ_SPACE):
                cfgs.append(StrategySuggestion(
                    reasoning="grid:quant_only/gptq",
                    mode="quant_only", quant_method="gptq", **p,
                ))

        if "awq" in self.modes:
            for p in _cartesian(AWQ_SPACE):
                cfgs.append(StrategySuggestion(
                    reasoning="grid:quant_only/awq",
                    mode="quant_only", quant_method="awq", quant_bits=4, **p,
                ))

        if "qqq" in self.modes:
            for p in _cartesian(QQQ_SPACE):
                cfgs.append(StrategySuggestion(
                    reasoning="grid:quant_only/qqq",
                    mode="quant_only", quant_method="qqq",
                    quant_bits=4, quant_format="qqq", **p,
                ))

        if "bnb" in self.modes:
            for p in _cartesian(BNB_SPACE):
                # use_double_quant 只在 bits=4 有效
                if p["use_double_quant"] and p["quant_bits"] != 4:
                    continue
                cfgs.append(StrategySuggestion(
                    reasoning="grid:quant_only/bnb",
                    mode="quant_only", quant_method="bnb", **p,
                ))

        if "sparse_unstructured" in self.modes:
            for p in _cartesian(SPARSE_UNSTRUCTURED_SPACE):
                cfgs.append(StrategySuggestion(
                    reasoning="grid:sparse_only/unstructured",
                    mode="sparse_only", sparsity_structure="unstructured", **p,
                ))

        if "sparse_structured" in self.modes:
            for p in _cartesian(SPARSE_STRUCTURED_SPACE):
                cfgs.append(StrategySuggestion(
                    reasoning="grid:sparse_only/structured",
                    mode="sparse_only", **p,
                ))

        if "hybrid_asvd_bnb" in self.modes:
            for a in _cartesian(ASVD_SPACE):
                for b in _cartesian(BNB_SPACE):
                    if b["use_double_quant"] and b["quant_bits"] != 4:
                        continue
                    cfgs.append(StrategySuggestion(
                        reasoning="grid:hybrid/asvd+bnb",
                        mode="hybrid", quant_method="bnb", **a, **b,
                    ))


        self._configs = cfgs

    # ──────────────────────────────────────────────────────────────────────────
    @property
    def total(self) -> int:
        return len(self._configs)

    def has_next(self) -> bool:
        return self._idx < len(self._configs)

    def get_suggestion(self, iteration: int, trial_history: list, **kwargs):
        """與 LLMDecisionMaker 相容的介面，回傳 (StrategySuggestion, raw_dict)"""
        if not self.has_next():
            raise StopIteration("Grid search 已完成——所有配置皆已嘗試")
        suggestion = self._configs[self._idx]
        self._idx += 1
        logger.info(f"Grid [{self._idx}/{self.total}] {suggestion.mode}")
        return suggestion, suggestion.to_log_dict()
