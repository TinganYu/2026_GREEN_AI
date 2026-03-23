"""
Optuna 演算法搜尋實作
使用 TPE / NSGA-II / Random 搜尋最佳量化配置

搜尋空間慣例（對應 search_space.py）：
  tuple (low, high)   → suggest_float（連續，TPE 可充分探索）
  list  [...]         → suggest_categorical（只有這幾個合法值）
"""

import logging
from typing import Optional, List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from Global_Tuner_v2.modular_agent.schemas import StrategySuggestion
from .search_space import (
    ASVD_SPACE, GPTQ_SPACE, AWQ_SPACE, QQQ_SPACE, BNB_SPACE,
    SPARSE_UNSTRUCTURED_SPACE, SPARSE_STRUCTURED_SPACE, ALL_MODES,
)

logger = logging.getLogger("OptunaSearcher")


def _suggest(trial: optuna.Trial, name: str, space):
    """
    根據 space 型別自動選擇 suggest 方式：
      tuple (low, high)         → suggest_float，linear scale
      tuple (low, high, "log")  → suggest_float，log scale
      list  [...]               → suggest_categorical
    """
    if isinstance(space, tuple):
        if len(space) == 3 and space[2] == "log":
            return trial.suggest_float(name, space[0], space[1], log=True)
        return trial.suggest_float(name, space[0], space[1])
    return trial.suggest_categorical(name, space)


class OptunaSearcher:
    """
    以 Optuna 的 ask-and-tell 模式逐步建議量化配置。

    sampler:  "tpe" (預設) | "nsga2" | "random"
    modes:    要包含的模式清單（None = 全部）
    seed:     亂數種子（可重現性）
    """

    def __init__(
        self,
        sampler: str = "tpe",
        modes: Optional[List[str]] = None,
        seed: Optional[int] = None,
        n_startup_trials: int = 10,
        population_size: int = 50,
    ):
        self.modes = modes or ALL_MODES
        self.sampler = sampler

        if sampler == "tpe":
            _sampler = optuna.samplers.TPESampler(
                seed=seed,
                multivariate=True,
                n_startup_trials=n_startup_trials,
            )
        elif sampler == "nsga2":
            _sampler = optuna.samplers.NSGAIISampler(
                seed=seed,
                population_size=population_size,
            )
        elif sampler == "random":
            _sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            raise ValueError(f"未知的 sampler: {sampler}，可選 tpe/nsga2/random")

        self.study = optuna.create_study(direction="maximize", sampler=_sampler)
        self._pending_trial: Optional[optuna.Trial] = None
        logger.info(f"Optuna study 建立完成 (sampler={sampler}, modes={self.modes})")

    # ──────────────────────────────────────────────────────────────────────────
    def _trial_to_suggestion(self, trial: optuna.Trial) -> StrategySuggestion:
        """將 Optuna trial 的建議值對應至 StrategySuggestion"""
        mode_key = trial.suggest_categorical("mode", self.modes)
        kwargs: dict = {"reasoning": f"optuna:{mode_key}", "mode": mode_key}

        # ── ASVD only ────────────────────────────────────────────────────────
        if mode_key == "asvd_only":
            kwargs["alpha"]              = _suggest(trial, "alpha",              ASVD_SPACE["alpha"])
            kwargs["param_ratio_target"] = _suggest(trial, "param_ratio_target", ASVD_SPACE["param_ratio_target"])
            kwargs["scaling_method"]     = _suggest(trial, "scaling_method",     ASVD_SPACE["scaling_method"])

        # ── GPTQ ─────────────────────────────────────────────────────────────
        elif mode_key == "gptq":
            kwargs["mode"]           = "quant_only"
            kwargs["quant_method"]   = "gptq"
            kwargs["quant_bits"]     = _suggest(trial, "gptq_bits",       GPTQ_SPACE["quant_bits"])
            kwargs["quant_group_size"] = _suggest(trial, "gptq_group_size", GPTQ_SPACE["quant_group_size"])
            kwargs["quant_format"]   = _suggest(trial, "gptq_format",     GPTQ_SPACE["quant_format"])
            kwargs["damp_percent"]   = _suggest(trial, "gptq_damp",       GPTQ_SPACE["damp_percent"])
            kwargs["mse"]            = 0.0

        # ── AWQ ──────────────────────────────────────────────────────────────
        elif mode_key == "awq":
            kwargs["mode"]             = "quant_only"
            kwargs["quant_method"]     = "awq"
            kwargs["quant_bits"]       = 4
            kwargs["quant_group_size"] = _suggest(trial, "awq_group_size", AWQ_SPACE["quant_group_size"])

        # ── QQQ ──────────────────────────────────────────────────────────────
        elif mode_key == "qqq":
            kwargs["mode"]             = "quant_only"
            kwargs["quant_method"]     = "qqq"
            kwargs["quant_bits"]       = 4
            kwargs["quant_format"]     = "qqq"
            kwargs["quant_group_size"] = _suggest(trial, "qqq_group_size", QQQ_SPACE["quant_group_size"])
            kwargs["damp_percent"]     = _suggest(trial, "qqq_damp",       QQQ_SPACE["damp_percent"])

        # ── BNB ──────────────────────────────────────────────────────────────
        elif mode_key == "bnb":
            kwargs["mode"]         = "quant_only"
            kwargs["quant_method"] = "bnb"
            bits = _suggest(trial, "bnb_bits", BNB_SPACE["quant_bits"])
            kwargs["quant_bits"]   = bits
            kwargs["use_double_quant"] = (
                _suggest(trial, "bnb_double_quant", BNB_SPACE["use_double_quant"])
                if bits == 4 else False
            )

        # ── Sparse Unstructured ───────────────────────────────────────────────
        elif mode_key == "sparse_unstructured":
            kwargs["mode"]              = "sparse_only"
            kwargs["sparsity_structure"] = "unstructured"
            kwargs["sparsity_ratio"]    = _suggest(trial, "sparse_ratio", SPARSE_UNSTRUCTURED_SPACE["sparsity_ratio"])

        # ── Sparse Structured ─────────────────────────────────────────────────
        elif mode_key == "sparse_structured":
            kwargs["mode"]               = "sparse_only"
            kwargs["sparsity_structure"] = _suggest(trial, "sparse_structure", SPARSE_STRUCTURED_SPACE["sparsity_structure"])

        # ── Hybrid: ASVD + BNB ───────────────────────────────────────────────
        elif mode_key == "hybrid_asvd_bnb":
            kwargs["mode"]               = "hybrid"
            kwargs["alpha"]              = _suggest(trial, "h_asvd_alpha",   ASVD_SPACE["alpha"])
            kwargs["param_ratio_target"] = _suggest(trial, "h_asvd_ratio",   ASVD_SPACE["param_ratio_target"])
            kwargs["scaling_method"]     = _suggest(trial, "h_asvd_scaling", ASVD_SPACE["scaling_method"])
            kwargs["quant_method"]       = "bnb"
            h_bnb_bits = _suggest(trial, "h_bnb_bits", BNB_SPACE["quant_bits"])
            kwargs["quant_bits"]         = h_bnb_bits
            kwargs["use_double_quant"]   = (
                _suggest(trial, "h_bnb_double_quant", BNB_SPACE["use_double_quant"])
                if h_bnb_bits == 4 else False
            )

        return StrategySuggestion(**kwargs)

    # ──────────────────────────────────────────────────────────────────────────
    def get_suggestion(self, iteration: int, trial_history: list, **kwargs):
        """回傳 (StrategySuggestion, raw_dict)"""
        trial = self.study.ask()
        self._pending_trial = trial
        suggestion = self._trial_to_suggestion(trial)
        logger.info(f"Optuna 建議 [{iteration}]: mode={suggestion.mode} / "
                    f"quant={suggestion.quant_method}")
        return suggestion, suggestion.to_log_dict()

    def report_score(self, score: float):
        """回報本次 trial 的得分，讓 Optuna 更新模型"""
        if self._pending_trial is not None:
            self.study.tell(self._pending_trial, score)
            self._pending_trial = None

    def report_failure(self):
        """回報本次 trial 失敗"""
        if self._pending_trial is not None:
            self.study.tell(self._pending_trial, optuna.trial.TrialState.FAIL)
            self._pending_trial = None

    def best_params(self) -> Optional[dict]:
        """回傳到目前為止最佳的參數（若有）"""
        try:
            return self.study.best_params
        except ValueError:
            return None
