"""
跨資料夾共用的資料載入邏輯。與 final_results/app.py、final_results_30/app.py 內部
的版本邏輯相同，差別是這裡的每個函式都吃一個 base_dir 參數，而不是依賴檔案內的
全域 BASE_DIR——因為「綜合比較頁」跟「no-retry 報告頁」需要同時讀好幾個不同的
資料夾，不能像原本兩支 app.py 那樣只服務單一固定路徑。

新增第 4 個實驗資料夾時，這裡的函式不用改，直接把新的 base_dir 傳進來即可。
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st


def is_valid_trial(trial: dict) -> bool:
    m = trial.get("metrics", {})
    if "accuracy" not in m:
        return False
    if m.get("accuracy") == 0 and m.get("latency") == 0 and m.get("vram") == 0 and m.get("emissions") == 0:
        return False
    return True


def is_skipped_trial(trial: dict) -> bool:
    """去重複重試機制放棄後，記成空 trial 的那種（只有 0-retry 資料夾會出現）。"""
    return trial.get("error") == "重複 config，已跳過"


def compute_pareto(trials: list) -> list:
    valid = [t for t in trials if is_valid_trial(t)]
    pareto = []
    for t in valid:
        m = t["metrics"]
        dominated = False
        for other in valid:
            if other is t:
                continue
            o = other["metrics"]
            if (o["accuracy"] >= m["accuracy"] and o["latency"] <= m["latency"] and
                    o["vram"] <= m["vram"] and o["emissions"] <= m["emissions"] and
                    (o["accuracy"] > m["accuracy"] or o["latency"] < m["latency"] or
                     o["vram"] < m["vram"] or o["emissions"] < m["emissions"])):
                dominated = True
                break
        if not dominated:
            pareto.append(t)
    return pareto


def detect_experiment_info(dir_name: str, config: dict) -> dict:
    if "sampler" in config:
        sampler = config["sampler"]
    elif dir_name.startswith("tpe_"):
        sampler = "tpe"
    elif dir_name.startswith("nsga2_"):
        sampler = "nsga2"
    elif dir_name.startswith("random_"):
        sampler = "random"
    elif "_full_" in dir_name:
        sampler = "full"
    elif "_window_" in dir_name:
        sampler = "window"
    elif "_summary_" in dir_name:
        sampler = "summary"
    elif "_tool_" in dir_name:
        sampler = "tool"
    else:
        sampler = "unknown"
    tuner = "Systematic_Tuner" if sampler in ("tpe", "nsga2", "random") else "Global_Tuner_memory"
    return {"sampler": sampler, "tuner": tuner}


def config_summary(trial: dict) -> str:
    cfg = trial.get("config", {})
    mode = cfg.get("mode", "?")
    q = cfg.get("quant", {})
    a = cfg.get("asvd", {})
    s = cfg.get("sparse", {})
    if q:
        method = q.get("method", "?")
        bits = q.get("bits", "")
        gs = q.get("group_size", "")
        gs_str = f"-g{gs}" if gs and gs != -1 else ""
        return f"{method}-{bits}bit{gs_str}"
    if a:
        return f"asvd-r{a.get('ratio', 0):.2f}-α{a.get('alpha', '')}"
    if s:
        st_val = s.get("structure", "unstr")
        r = s.get("ratio", "")
        r_str = f"-{r:.0%}" if r else ""
        return f"sparse-{st_val}{r_str}"
    return mode


@st.cache_data
def load_all_experiments(base_dir: str) -> list:
    """base_dir 傳字串而不是 Path，是為了讓 st.cache_data 能正確 hash 當快取 key。"""
    base = Path(base_dir)
    experiments = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or d.name == "baselines":
            continue
        cfg_path = d / "experiment_config.json"
        res_path = d / "optimization_results.json"
        if not cfg_path.exists() or not res_path.exists():
            continue
        config = json.load(open(cfg_path, encoding="utf-8"))
        trials = json.load(open(res_path, encoding="utf-8"))

        pareto_path = d / "pareto_frontier.json"
        if pareto_path.exists():
            pareto = json.load(open(pareto_path, encoding="utf-8"))
        else:
            pareto = compute_pareto(trials)

        info = detect_experiment_info(d.name, config)
        experiments.append({
            "dir_name": d.name,
            "config": config,
            "trials": trials,
            "pareto": pareto,
            **info,
        })
    return experiments


@st.cache_data
def load_baseline(base_dir: str) -> dict:
    """優先讀 baselines/ 底下的檔案；若沒有（例如 no-retry 資料夾未附帶），
    退而求其次從任一個 experiment_config.json 裡的 baseline 欄位取得，
    因為每個 run 存檔時都會把當時讀到的 baseline 一併寫入。"""
    base = Path(base_dir)
    baseline_files = list((base / "baselines").glob("*.json")) if (base / "baselines").exists() else []
    if baseline_files:
        return json.load(open(baseline_files[0], encoding="utf-8"))

    for d in sorted(base.iterdir()):
        cfg_path = d / "experiment_config.json"
        if cfg_path.exists():
            cfg = json.load(open(cfg_path, encoding="utf-8"))
            if "baseline" in cfg:
                return cfg["baseline"]
    return {}


def build_headline_rows(experiments: list) -> pd.DataFrame:
    """每個 sampler 的 headline 指標（跨 run 池化），供比較頁使用的輕量彙總。"""
    import statistics as st_

    from collections import defaultdict
    by_sampler = defaultdict(list)
    for exp in experiments:
        by_sampler[exp["sampler"]].append(exp)

    rows = []
    for sampler, exps in by_sampler.items():
        all_scores, all_bests = [], []
        n_skipped_total, n_total = 0, 0
        for exp in exps:
            trials = exp["trials"]
            n_total += len(trials)
            n_skipped_total += sum(1 for t in trials if is_skipped_trial(t))
            scores = [t["metrics"]["score"] for t in trials if is_valid_trial(t)]
            all_scores.extend(scores)
            if scores:
                all_bests.append(max(scores))
        if not all_scores:
            continue
        n_valid = len(all_scores)
        rows.append({
            "sampler": sampler,
            "runs": len(exps),
            "best_mean": st_.mean(all_bests) if all_bests else 0,
            "best_std": st_.pstdev(all_bests) if len(all_bests) > 1 else 0.0,
            "trial_mean": st_.mean(all_scores),
            "pos_rate": sum(1 for s in all_scores if s > 0) / n_valid,
            "cat_rate": sum(1 for s in all_scores if s < -10) / n_valid,
            "skip_rate": n_skipped_total / n_total if n_total else 0.0,
            "n_valid_trials": n_valid,
            "n_total_trials": n_total,
        })
    return pd.DataFrame(rows)
