"""
Green AI Tuner Analysis Dashboard
Streamlit app for analyzing quantization experiment results in final_results/
Run: streamlit run app.py
"""

import json
import math
import statistics
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
# 資料載入
# ─────────────────────────────────────────────────────────────────────────────

SAMPLER_LABELS = {
    "tpe":     "TPE（貝葉斯）",
    "nsga2":   "NSGA-II（遺傳演算法）",
    "random":  "Random Search（基線）",
    "full":    "Full Memory（全部歷史）",
    "window":  "Window Memory（最近 N 筆）",
    "summary": "Summary Memory（LLM 摘要）",
    "tool":    "Tool Agent（LLM 工具）",
}

SAMPLER_LABELS_EN = {
    "tpe":     "TPE",
    "nsga2":   "NSGA-II",
    "random":  "Random",
    "full":    "Full Memory",
    "window":  "Window Memory",
    "summary": "Summary Memory",
    "tool":    "Tool Agent",
}

SAMPLER_TUNER = {
    "tpe": "Systematic_Tuner", "nsga2": "Systematic_Tuner", "random": "Systematic_Tuner",
    "full": "Global_Tuner_memory", "window": "Global_Tuner_memory", "summary": "Global_Tuner_memory", "tool": "Global_Tuner_memory",
}

SAMPLER_SORT_ORDER = {
    "tpe": 0,
    "nsga2": 1,
    "random": 2,
    "window": 3,
    "summary": 4,
    "tool": 5,
    "full": 6,
}

DEFAULT_SAMPLER_ORDER = ["tpe", "nsga2", "random", "window", "summary", "tool", "full"]

SAMPLER_COLORS = {
    "tpe":     "#2471a3",
    "nsga2":   "#1e8449",
    "random":  "#b7950b",
    "window":  "#e377c2",
    "summary": "#a04000",
    "tool":    "#9467bd",
    "full":    "#17becf",
}


def sampler_sort_key(value) -> int:
    if value is None:
        return len(DEFAULT_SAMPLER_ORDER)
    if isinstance(value, str):
        return SAMPLER_SORT_ORDER.get(value, len(DEFAULT_SAMPLER_ORDER))
    return SAMPLER_SORT_ORDER.get(str(value), len(DEFAULT_SAMPLER_ORDER))


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
    return {
        "sampler":  sampler,
        "label":    SAMPLER_LABELS.get(sampler, sampler),
        "tuner":    SAMPLER_TUNER.get(sampler, "Unknown"),
    }


def compute_pareto(trials: list) -> list:
    def is_valid_trial(trial: dict) -> bool:
        m = trial.get("metrics", {})
        if "accuracy" not in m:
            return False
        if m.get("accuracy") == 0 and m.get("latency") == 0 and m.get("vram") == 0 and m.get("emissions") == 0:
            return False
        return True

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


def detailed_method(trial: dict) -> str:
    """回傳具體壓縮方法名稱（gptq / awq / bnb / qqq / asvd / sparse_unstr / sparse_2:4 / sparse_4:8 / hybrid）"""
    cfg = trial.get("config", {})
    mode = cfg.get("mode", "")
    q = cfg.get("quant", {})
    s = cfg.get("sparse", {})
    if mode == "quant_only" and q:
        return q.get("method", "quant")
    if mode == "asvd_only":
        return "asvd"
    if mode == "sparse_only" and s:
        struct = s.get("structure", "unstructured")
        if struct == "unstructured":
            return "sparse_unstructured"
        return "sparse_structured"
    if mode == "hybrid":
        return "hybrid_asvd_bnb"
    return mode or "unknown"


def config_summary(trial: dict) -> str:
    cfg = trial.get("config", {})
    mode = cfg.get("mode", "?")
    q = cfg.get("quant", {})
    a = cfg.get("asvd", {})
    s = cfg.get("sparse", {})
    if q:
        method = q.get("method", "?")
        bits   = q.get("bits", "")
        gs     = q.get("group_size", "")
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
def load_all_experiments() -> list:
    experiments = []
    for d in sorted(BASE_DIR.iterdir()):
        if not d.is_dir() or d.name == "baselines":
            continue
        cfg_path = d / "experiment_config.json"
        res_path = d / "optimization_results.json"
        if not cfg_path.exists() or not res_path.exists():
            continue
        with open(cfg_path) as f:
            config = json.load(f)
        with open(res_path) as f:
            trials = json.load(f)

        pareto_path = d / "pareto_frontier.json"
        if pareto_path.exists():
            with open(pareto_path) as f:
                pareto = json.load(f)
        else:
            pareto = compute_pareto(trials)
            with open(pareto_path, "w") as f:
                json.dump(pareto, f, indent=2, ensure_ascii=False)

        info = detect_experiment_info(d.name, config)
        experiments.append({
            "dir_name": d.name,
            "config":   config,
            "trials":   trials,
            "pareto":   pareto,
            **info,
        })
    return experiments


@st.cache_data
def load_baseline() -> dict:
    p = BASE_DIR / "baselines" / "Llama-3.2-3B-Instruct_gsm8k.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame 建立與樣式
# ─────────────────────────────────────────────────────────────────────────────

def build_trials_df(trials: list, baseline: dict) -> pd.DataFrame:
    rows = []
    b_acc  = baseline.get("accuracy",  1e-6)
    b_lat  = baseline.get("latency",   1e-6)
    b_vram = baseline.get("vram",      1e-6)
    b_emit = baseline.get("emissions", 1e-6)
    for t in trials:
        m = t.get("metrics", {})
        if "accuracy" not in m:
            continue
        # 排除所有指標都是 0 的無效 trial
        if m.get("accuracy") == 0 and m.get("latency") == 0 and m.get("vram") == 0 and m.get("emissions") == 0:
            continue
        acc  = m["accuracy"]
        lat  = m["latency"]
        vram = m["vram"]
        emit = m["emissions"]
        rows.append({
            "Trial":        t.get("trial_name", f"iter {t.get('iteration')}"),
            "Mode":         t.get("config", {}).get("mode", "-"),
            "設定":          config_summary(t),
            "Score":        round(m.get("score", 0.0), 4),
            "Accuracy":     acc,
            "Latency (s)":  round(lat, 1),
            "VRAM (GB)":    round(vram, 3),
            "Emissions (g)": round(emit * 1000, 5),
            "Δ Accuracy %": round((acc - b_acc) / b_acc * 100, 1),
            "Δ Latency %":  round((lat - b_lat) / b_lat * 100, 1),
            "Δ VRAM %":     round((vram - b_vram) / b_vram * 100, 1),
            "Δ Emissions %": round((emit - b_emit) / b_emit * 100, 1),
        })
    return pd.DataFrame(rows)


def extract_param_cols(trial: dict) -> dict:
    """將 trial config 展開成各參數欄位（NaN 代表該 mode 不使用此參數）。"""
    cfg = trial.get("config", {})
    q = cfg.get("quant", {})
    a = cfg.get("asvd", {})
    s = cfg.get("sparse", {})
    return {
        # 量化參數
        "bits":         q.get("bits")       if q else None,
        "group_size":   q.get("group_size") if q else None,
        "q_format":     q.get("format") or q.get("type") if q else None,
        "damp":         round(q["damp"], 4) if q and "damp" in q else None,
        "double_quant": q.get("double_quant") if q and "double_quant" in q else None,
        # ASVD 參數
        "asvd_ratio":   round(a["ratio"], 4)   if a and "ratio" in a else None,
        "asvd_α":       a.get("alpha")         if a else None,
        "asvd_scaling": a.get("scaling")       if a else None,
        # 稀疏參數
        "sparse_struct": s.get("structure")    if s else None,
        "sparse_ratio":  round(s["ratio"], 4) if s and "ratio" in s else None,
    }


def style_df(df: pd.DataFrame):
    styled = df.style
    for col, cmap in [
        ("Score",         "RdYlGn"),
        ("Δ Accuracy %",  "RdYlGn"),
        ("Δ Latency %",   "RdYlGn_r"),
        ("Δ VRAM %",      "RdYlGn_r"),
        ("Δ Emissions %", "RdYlGn_r"),
    ]:
        if col in df.columns:
            styled = styled.background_gradient(subset=[col], cmap=cmap)
    styled = styled.format({
        "Score":          "{:.4f}",
        "Accuracy":       "{:.4f}",
        "Latency (s)":    "{:.1f}",
        "VRAM (GB)":      "{:.3f}",
        "Emissions (g)":  "{:.5f}",
        "Δ Accuracy %":   "{:+.1f}%",
        "Δ Latency %":    "{:+.1f}%",
        "Δ VRAM %":       "{:+.1f}%",
        "Δ Emissions %":  "{:+.1f}%",
    }, na_rep="-")
    styled = styled.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#2c3e50"), ("color", "white"),
            ("font-size", "12px"), ("text-align", "center"), ("padding", "6px 10px"),
        ]},
        {"selector": "td", "props": [
            ("font-size", "11px"), ("padding", "4px 8px"), ("text-align", "center"),
        ]},
        {"selector": "tr:hover td", "props": [("filter", "brightness(0.92)")]},
    ])
    return styled


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 統計
# ─────────────────────────────────────────────────────────────────────────────

def compute_trial_health(experiments: list) -> tuple[pd.DataFrame, dict]:
    """
    回傳 (detail_df, summary_dict)
    detail_df: 每個有問題的 trial 一列（skipped / missing）
    summary_dict: 全局統計數字
    """
    def is_valid_trial(trial: dict) -> bool:
        """判斷 trial 是否有效：有 accuracy 且至少有一個非零指標"""
        m = trial.get("metrics", {})
        if "accuracy" not in m:
            return False
        # 檢查是否所有指標都是 0（無效評估）
        if m.get("accuracy") == 0 and m.get("latency") == 0 and m.get("vram") == 0 and m.get("emissions") == 0:
            return False
        return True

    rows = []
    total_expected = total_valid = total_skipped = total_missing = 0

    for exp in experiments:
        cfg = exp["config"]
        max_iter = cfg.get("max_iterations", 20)
        trials = exp["trials"]
        total_expected += max_iter

        recorded_iters = set(t.get("iteration") for t in trials)
        valid_iters   = set(t.get("iteration") for t in trials if is_valid_trial(t))
        skipped_iters = set(t.get("iteration") for t in trials if not is_valid_trial(t))
        missing_iters = set(range(1, max_iter + 1)) - recorded_iters

        total_valid   += len(valid_iters)
        total_skipped += len(skipped_iters)
        total_missing += len(missing_iters)

        for t in trials:
            if not is_valid_trial(t):
                err = t.get("error", "未知錯誤")
                m = t.get("metrics", {})
                # 新增檢測：所有指標都是 0
                if "accuracy" in m and m.get("accuracy") == 0 and m.get("latency") == 0 and m.get("vram") == 0 and m.get("emissions") == 0:
                    etype = "評估失敗（所有指標為 0，可能 crash 或無輸出）"
                    color = "🔴"
                # 歸類錯誤類型
                elif "重複" in str(err):
                    etype = "重複 config（LLM 建議已嘗試過的配置）"
                    color = "🟡"
                elif "OOM" in str(err) or "memory" in str(err).lower():
                    etype = "OOM（顯存不足）"
                    color = "🔴"
                elif "timeout" in str(err).lower():
                    etype = "Timeout"
                    color = "🔴"
                else:
                    etype = f"執行失敗：{str(err)[:60]}"
                    color = "🔴"
                rows.append({
                    "實驗":     exp["dir_name"].replace("Llama-3.2-3B-Instruct_gsm8k_", ""),
                    "Tuner":   exp["tuner"],
                    "Sampler": exp["sampler"],
                    "Iter":    t.get("iteration"),
                    "Trial":   t.get("trial_name", "?"),
                    "狀態":    f"{color} skipped",
                    "原因":    etype,
                })

        for it in sorted(missing_iters):
            rows.append({
                "實驗":     exp["dir_name"].replace("Llama-3.2-3B-Instruct_gsm8k_", ""),
                "Tuner":   exp["tuner"],
                "Sampler": exp["sampler"],
                "Iter":    it,
                "Trial":   "(未記錄)",
                "狀態":    "🔴 missing",
                "原因":    "實驗中斷（進程在此 iteration 前被終止）",
            })

    summary = {
        "total_expected": total_expected,
        "total_valid":    total_valid,
        "total_skipped":  total_skipped,
        "total_missing":  total_missing,
    }
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["實驗", "Tuner", "Sampler", "Iter", "Trial", "狀態", "原因"]
    )
    return df, summary


def compute_benchmark_stats_combined(experiments: list) -> pd.DataFrame:
    """所有 tuner 合併成一張表，依 Tuner 分組排序。"""
    def is_valid_trial(trial: dict) -> bool:
        m = trial.get("metrics", {})
        if "accuracy" not in m:
            return False
        if m.get("accuracy") == 0 and m.get("latency") == 0 and m.get("vram") == 0 and m.get("emissions") == 0:
            return False
        return True

    group: dict[tuple, list] = {}
    for exp in experiments:
        key = (exp["tuner"], exp["sampler"])
        valid_scores = [
            t["metrics"]["score"]
            for t in exp["trials"]
            if is_valid_trial(t)
        ]
        if valid_scores:
            group.setdefault(key, []).append(max(valid_scores))

    tuner_order = {"Systematic_Tuner": 0, "Global_Tuner_memory": 1}
    rows = []
    for (tuner, sampler), best_scores in sorted(
        group.items(),
        key=lambda x: (
            tuner_order.get(x[0][0], 9),
            sampler_sort_key(x[0][1]),
        ),
    ):
        n = len(best_scores)
        rows.append({
            "Tuner":      tuner,
            "Sampler":    sampler,
            "說明":        SAMPLER_LABELS.get(sampler, sampler),
            "Runs":       n,
            "Mean Score": round(statistics.mean(best_scores), 4),
            "Best Score": round(max(best_scores), 4),
            "Std 標準差":  round(statistics.stdev(best_scores) if n > 1 else 0.0, 4),
        })
    return pd.DataFrame(rows)


def compute_benchmark_stats(experiments: list, tuner: str) -> pd.DataFrame:
    def is_valid_trial(trial: dict) -> bool:
        m = trial.get("metrics", {})
        if "accuracy" not in m:
            return False
        if m.get("accuracy") == 0 and m.get("latency") == 0 and m.get("vram") == 0 and m.get("emissions") == 0:
            return False
        return True

    group: dict[str, list] = {}
    for exp in experiments:
        if exp["tuner"] != tuner:
            continue
        sampler = exp["sampler"]
        valid_scores = [
            t["metrics"]["score"]
            for t in exp["trials"]
            if is_valid_trial(t)
        ]
        if valid_scores:
            best = max(valid_scores)
            group.setdefault(sampler, []).append(best)

    rows = []
    for sampler, best_scores in sorted(group.items(), key=lambda kv: sampler_sort_key(kv[0])):
        n = len(best_scores)
        rows.append({
            "Sampler":    sampler,
            "說明":        SAMPLER_LABELS.get(sampler, sampler),
            "Mean Score": round(statistics.mean(best_scores), 4),
            "Best Score": round(max(best_scores), 4),
            "Std 標準差":  round(statistics.stdev(best_scores) if n > 1 else 0.0, 4),
            "Valid Runs": f"{n}/{n}",
        })
    return pd.DataFrame(rows)


def style_benchmark_df(df: pd.DataFrame):
    if df.empty:
        return df.style
    styled = df.style
    for col in ["Mean Score", "Best Score"]:
        if col in df.columns:
            styled = styled.background_gradient(subset=[col], cmap="RdYlGn",
                                                vmin=df[col].min(), vmax=df[col].max())
    styled = styled.format({"Mean Score": "{:.4f}", "Best Score": "{:.4f}", "Std 標準差": "{:.4f}"})
    styled = styled.set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#2c3e50"), ("color", "white"),
            ("font-size", "13px"), ("text-align", "center"), ("padding", "8px 14px"),
        ]},
        {"selector": "td", "props": [
            ("font-size", "12px"), ("padding", "6px 12px"), ("text-align", "center"),
        ]},
    ])
    return styled


# ─────────────────────────────────────────────────────────────────────────────
# Weight 敏感度分析
# ─────────────────────────────────────────────────────────────────────────────

def rescore(trials: list, baseline: dict, weights: dict, pen_t: float, pen_a: float) -> list:
    b_acc  = baseline.get("accuracy",  1e-6)
    b_lat  = baseline.get("latency",   1e-6)
    b_vram = baseline.get("vram",      1e-6)
    b_emit = baseline.get("emissions", 1e-6)
    result = []
    for t in trials:
        m = t.get("metrics", {})
        if "accuracy" not in m:
            continue
        acc, lat, vram, emit = m["accuracy"], m["latency"], m["vram"], m["emissions"]
        ws = 1.0 + (
            weights["acc"]  * math.log(acc  / (b_acc  + 1e-9) + 1e-9) +
            weights["lat"]  * math.log(b_lat / (lat   + 1e-9) + 1e-9) +
            weights["vram"] * math.log(b_vram / (vram + 1e-9) + 1e-9) +
            weights["emit"] * math.log(b_emit / (emit + 1e-9) + 1e-9)
        )
        penalty = pen_a * max(0.0, (b_acc - pen_t) - acc)
        result.append({
            "Trial":    t.get("trial_name", f"iter {t.get('iteration')}"),
            "設定":      config_summary(t),
            "Accuracy": round(acc, 4),
            "Score":    round(ws - penalty, 4),
        })
    result.sort(key=lambda x: x["Score"], reverse=True)
    for i, r in enumerate(result, 1):
        r["Rank"] = i
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Green AI Tuner Dashboard",
    page_icon="🌿",
    layout="wide",
)

experiments = load_all_experiments()
baseline    = load_baseline()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌿 Green AI Tuner")
    st.markdown("---")

    st.subheader("📌 Baseline（未壓縮）")
    if baseline:
        st.metric("Accuracy",    f"{baseline.get('accuracy', '-'):.3f}")
        st.metric("Latency",     f"{baseline.get('latency', '-'):.1f} s")
        st.metric("VRAM",        f"{baseline.get('vram', '-'):.3f} GB")
        st.metric("Emissions",   f"{baseline.get('emissions', 0)*1000:.4f} g CO₂")
    st.markdown("---")

    tuner_filter = st.radio(
        "Tuner 類型篩選",
        ["全部", "Systematic_Tuner", "Global_Tuner_memory"],
        index=0,
    )
    st.markdown(f"共載入 **{len(experiments)}** 個實驗")

filtered = experiments if tuner_filter == "全部" else [
    e for e in experiments if e["tuner"] == tuner_filter
]

# ── 預先計算所有實驗統計（多個 tab 共用）────────────────────────────────────
_all_exp_rows = []
for _exp in experiments:
    _is_valid = lambda t: ("accuracy" in t.get("metrics", {}) and
                           not (t["metrics"].get("accuracy") == 0 and t["metrics"].get("latency") == 0 and
                                t["metrics"].get("vram") == 0 and t["metrics"].get("emissions") == 0))
    _valid = [t for t in _exp["trials"] if _is_valid(t)]
    _scores = [t["metrics"]["score"] for t in _valid]
    if not _scores:
        continue
    _bsf, _conv = [], 0.0
    for t in _exp["trials"]:
        if _is_valid(t):
            _conv = max(_conv, t["metrics"]["score"])
        _bsf.append(_conv)
    _all_exp_rows.append({
        "exp":         _exp,
        "valid":       _valid,
        "scores":      _scores,
        "best":        max(_scores),
        "best_so_far": _bsf,
    })

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab_cross, tab2, tab3, tab4, tab_report = st.tabs([
    "🏆 Benchmark 比較", "🔬 跨實驗比較", "📊 Trial 彩色分析", "🎯 Pareto Frontier", "ℹ️ 演算法說明", "📋 LLM 優勢分析報告"
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1：Benchmark 比較
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 搜尋演算法 Benchmark 比較")
    st.caption(
        "比較 Systematic Tuner（Optuna：TPE / NSGA-II / Random）與 "
        "Global Tuner（LLM Agent：Window / Summary / Tool）在 GSM8K 任務上的最佳壓縮效果。"
        "每個 sampler 可能有多次獨立執行（runs），表中統計其 Best Score 的 Mean / Best / Std。"
    )

    df_combined = compute_benchmark_stats_combined(experiments)
    if not df_combined.empty:
        # 樣式：Tuner 欄位靠左，分數欄漸層
        def style_combined(df):
            styled = df.style
            for col in ["Mean Score", "Best Score"]:
                if col in df.columns:
                    styled = styled.background_gradient(
                        subset=[col], cmap="RdYlGn",
                        vmin=df[col].min(), vmax=df[col].max()
                    )
            styled = styled.background_gradient(
                subset=["Std 標準差"], cmap="RdYlGn_r",
                vmin=0, vmax=df["Std 標準差"].max() or 0.01
            )
            styled = styled.format({
                "Mean Score": "{:.4f}", "Best Score": "{:.4f}", "Std 標準差": "{:.4f}"
            })
            styled = styled.set_table_styles([
                {"selector": "th", "props": [
                    ("background-color", "#2c3e50"), ("color", "white"),
                    ("font-size", "13px"), ("text-align", "center"), ("padding", "8px 14px"),
                ]},
                {"selector": "td", "props": [
                    ("font-size", "12px"), ("padding", "7px 14px"), ("text-align", "center"),
                ]},
                {"selector": "tr:hover td", "props": [("filter", "brightness(0.93)")]},
            ])
            return styled

        st.write(style_combined(df_combined).to_html(), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Box plot：每個 sampler 所有 runs 的 score 分布
        st.markdown("#### 各 Sampler Score 分布（Box Plot）")
        st.caption("每個點代表單次 run 的 best score；箱型顯示 runs 間的穩定性")
        box_rows = []
        for exp in experiments:
            valid = [t["metrics"]["score"] for t in exp["trials"]
                     if "accuracy" in t.get("metrics", {}) and
                     not (t["metrics"].get("accuracy") == 0 and t["metrics"].get("latency") == 0 and
                          t["metrics"].get("vram") == 0 and t["metrics"].get("emissions") == 0)]
            if valid:
                box_rows.append({
                    "Sampler": exp["sampler"],
                    "Tuner":   exp["tuner"],
                    "Best Score": max(valid),
                    "說明": SAMPLER_LABELS.get(exp["sampler"], exp["sampler"]),
                })
        if box_rows:
            df_box = pd.DataFrame(box_rows)
            df_box["Sampler"] = df_box["Sampler"].astype(str)
            df_box["_sort"] = df_box["Sampler"].map(sampler_sort_key)
            df_box = df_box.sort_values(["_sort", "Sampler"]).drop(columns=["_sort"])
            fig_box = px.box(
                df_box, x="說明", y="Best Score", color="說明",
                points="all", hover_data=["Sampler"],
                color_discrete_map={
                    SAMPLER_LABELS.get(s, s): SAMPLER_COLORS.get(s, "#888")
                    for s in DEFAULT_SAMPLER_ORDER
                },
            )
            fig_box.update_traces(marker=dict(size=9, opacity=0.8))
            fig_box.update_layout(
                height=400, xaxis_title="Sampler",
                yaxis_title="Best Score（該 run 最高分）",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend_title="Sampler",
            )
            fig_box.update_yaxes(gridcolor="#eee")
            st.plotly_chart(fig_box, width="stretch")

    # ── ⑥ Variance 分解：Between-run vs Within-run ─────────────────────────
    st.markdown("#### 📐 Variance 分解：Between-run vs Within-run")
    st.caption(
        "**Within-run Var**：同一 sampler 單次 run 內所有 trial score 的方差（探索隨機性）。"
        "**Between-run Var**：不同 run 之間最終 best score 的方差（結果穩定性）。"
        "理想 sampler 應同時降低兩者。"
    )
    _var_rows = []
    _sampler_run_scores: dict[str, list] = {}
    for row in _all_exp_rows:
        s = row["exp"]["sampler"]
        _sampler_run_scores.setdefault(s, []).append(row["scores"])
    for _s, _runs_scores in _sampler_run_scores.items():
        _within = statistics.mean(
            [statistics.variance(sc) if len(sc) > 1 else 0.0 for sc in _runs_scores]
        )
        _between = statistics.variance(
            [statistics.mean(sc) for sc in _runs_scores]
        ) if len(_runs_scores) > 1 else 0.0
        _total = _within + _between
        _var_rows.append({
            "Sampler":         SAMPLER_LABELS.get(_s, _s),
            "Within-run Var":  round(_within, 4),
            "Between-run Var": round(_between, 4),
            "Total Var":       round(_total, 4),
            "穩定性（Between/Total）": f"{(_between / _total * 100):.1f}%" if _total > 0 else "—",
        })
    if _var_rows:
        _df_var = pd.DataFrame(_var_rows)
        _df_var["_s_key"] = _df_var["Sampler"].map(
            {SAMPLER_LABELS.get(s, s): s for s in SAMPLER_SORT_ORDER}
        )
        _df_var["_sort"] = _df_var["_s_key"].map(lambda s: SAMPLER_SORT_ORDER.get(s, 99))
        _df_var = _df_var.sort_values("_sort").drop(columns=["_s_key", "_sort"])
        styled_var = (
            _df_var.style
            .background_gradient(subset=["Within-run Var"],  cmap="Reds", vmin=0)
            .background_gradient(subset=["Between-run Var"], cmap="Oranges", vmin=0)
            .background_gradient(subset=["Total Var"],       cmap="RdYlGn_r", vmin=0)
            .set_table_styles([
                {"selector": "th", "props": [("background-color", "#2c3e50"), ("color", "white"),
                                              ("font-size", "12px"), ("padding", "6px 10px"), ("text-align", "center")]},
                {"selector": "td", "props": [("font-size", "11px"), ("padding", "5px 10px"), ("text-align", "center")]},
            ])
        )
        st.write(styled_var.to_html(), unsafe_allow_html=True)
        st.markdown("""
**計算方式**：
- **Within-run Var** = 單次 run 內所有 trial score 的 `statistics.variance`，再取多次 run 的平均。反映搜尋過程的隨機性，值越大代表每次試驗的分數波動越大（多元探索）。
- **Between-run Var** = 不同 run 的平均 score 之間的 `statistics.variance`。反映結果的可重現性，值越大代表同一方法每次執行的最終水準差異越大（不穩定）。
- **穩定性（Between/Total）** = Between-run Var 佔總方差的比例。比例越高，代表不穩定主要來自跑次間的差異（初始化敏感）而非探索本身。
""")

    st.markdown("---")

    # ── 搜尋效率量化比較 ──────────────────────────────────────────────────────
    st.markdown("#### 📊 搜尋效率量化比較（跨 Run 明細）")
    st.caption(
        "**Final Score**：最終最高分｜"
        "**AUC**：學習曲線面積 ÷ 迭代數（整體搜尋品質）｜"
        "**Iter@90%**：首次達到最終分數 90% 所需的迭代數（越少收斂越快）"
    )
    _conv_stats = []
    for row in _all_exp_rows:
        bsf   = row["best_so_far"]
        final = bsf[-1] if bsf else 0
        auc   = sum(bsf) / len(bsf) if bsf else 0
        target = final * 0.9
        iter90 = next((i + 1 for i, v in enumerate(bsf) if v >= target), len(bsf))
        best_t = max(row["valid"], key=lambda t: t["metrics"].get("score", -999), default=None)
        _conv_stats.append({
            "實驗":        row["exp"]["dir_name"].replace("Llama-3.2-3B-Instruct_gsm8k_", ""),
            "Sampler":    row["exp"]["sampler"],
            "Final Score": round(final, 4),
            "AUC":         round(auc, 4),
            "Iter@90%":   iter90,
            "最佳配置":    config_summary(best_t) if best_t else "?",
            "最佳 Acc":   round(best_t["metrics"]["accuracy"], 4) if best_t else 0,
            "最佳 Score": round(best_t["metrics"].get("score", 0), 4) if best_t else 0,
        })
    if _conv_stats:
        df_cs = pd.DataFrame(_conv_stats)
        df_cs["Sampler"] = df_cs["Sampler"].astype(str)
        df_cs["_sort"] = df_cs["Sampler"].map(sampler_sort_key)
        df_cs = df_cs.sort_values(["_sort", "Sampler", "Final Score"], ascending=[True, True, False]).drop(columns=["_sort"])
        styled_cs = df_cs.style
        for col, cmap in [("Final Score", "RdYlGn"), ("AUC", "RdYlGn"), ("最佳 Score", "RdYlGn")]:
            styled_cs = styled_cs.background_gradient(subset=[col], cmap=cmap)
        styled_cs = styled_cs.background_gradient(
            subset=["Iter@90%"], cmap="RdYlGn_r",
            vmin=1, vmax=df_cs["Iter@90%"].max()
        )
        styled_cs = styled_cs.format({
            "Final Score": "{:.4f}", "AUC": "{:.4f}", "Iter@90%": "{:.0f}", "最佳 Acc": "{:.4f}", "最佳 Score": "{:.4f}",
        })
        styled_cs = styled_cs.set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#2c3e50"), ("color", "white"),
                ("font-size", "12px"), ("padding", "6px 12px"), ("text-align", "center"),
            ]},
            {"selector": "td", "props": [
                ("font-size", "11px"), ("padding", "5px 10px"), ("text-align", "center"),
            ]},
        ])
        st.write(styled_cs.to_html(), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # grouped bar：Mean Final Score vs AUC per sampler
        _sagg = df_cs.groupby("Sampler").agg(
            mean_final=("Final Score", "mean"),
            mean_auc=("AUC", "mean"),
        ).reset_index()
        _sagg["Sampler"] = _sagg["Sampler"].astype(str)
        _sagg["_sort"] = _sagg["Sampler"].map(sampler_sort_key)
        _sagg = _sagg.sort_values(["_sort", "Sampler"]).drop(columns=["_sort"])
        _sagg["說明"] = _sagg["Sampler"].map(SAMPLER_LABELS)
        fig_eff = go.Figure()
        fig_eff.add_trace(go.Bar(
            x=_sagg["說明"], y=_sagg["mean_final"], name="Mean Final Score",
            marker_color="#3498db",
            text=_sagg["mean_final"].apply(lambda v: f"{v:.4f}"), textposition="outside",
        ))
        fig_eff.add_trace(go.Bar(
            x=_sagg["說明"], y=_sagg["mean_auc"], name="Mean AUC",
            marker_color="#2ecc71",
            text=_sagg["mean_auc"].apply(lambda v: f"{v:.4f}"), textposition="outside",
        ))
        fig_eff.update_layout(
            barmode="group", height=360,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend_title="指標", yaxis_title="Score",
            title="各 Sampler：平均 Final Score vs. AUC（品質 vs. 搜尋效率）",
        )
        fig_eff.update_yaxes(gridcolor="#eee")
        st.plotly_chart(fig_eff, width="stretch")

    st.markdown("---")

    # ── 準確率保留率分布（Violin）──────────────────────────────────────────────
    st.markdown("#### 🎻 準確率保留率分布（所有 Trial）")
    _violin_rows = []
    _violin_rows_en = []
    for row in _all_exp_rows:
        for t in row["valid"]:
            _violin_rows.append({
                "Sampler": SAMPLER_LABELS.get(row["exp"]["sampler"], row["exp"]["sampler"]),
                "Tuner":   row["exp"]["tuner"],
                "Accuracy": t["metrics"]["accuracy"],
                "Score":    t["metrics"]["score"],
            })
            _violin_rows_en.append({
                "Sampler": SAMPLER_LABELS_EN.get(row["exp"]["sampler"], row["exp"]["sampler"]),
                "Tuner":   row["exp"]["tuner"],
                "Accuracy": t["metrics"]["accuracy"],
                "Score":    t["metrics"]["score"],
            })
    if _violin_rows:
        _vln_tab_zh, _vln_tab_en = st.tabs(["中文", "English"])

        # 每個 sampler 對應的顏色（使用全域 SAMPLER_COLORS）
        _vln_color_zh = {
            SAMPLER_LABELS.get(s, s): SAMPLER_COLORS.get(s, "#888")
            for s in DEFAULT_SAMPLER_ORDER
        }
        _vln_color_en = {
            SAMPLER_LABELS_EN.get(s, s): SAMPLER_COLORS.get(s, "#888")
            for s in DEFAULT_SAMPLER_ORDER
        }

        with _vln_tab_zh:
            st.caption(
                "小提琴圖顯示每個 sampler 在所有 trial（非僅最佳）中的準確率分布。"
                "分布越集中在高準確率 → 穩定產出高品質解；尾部向下延伸過長 → 偶爾產出極差解。"
            )
            df_vln = pd.DataFrame(_violin_rows)
            sampler_order_vln = [SAMPLER_LABELS[s] for s in DEFAULT_SAMPLER_ORDER if s in SAMPLER_LABELS]
            df_vln["Sampler"] = df_vln["Sampler"].astype(str)
            df_vln["_sort"] = df_vln["Sampler"].map(lambda s: SAMPLER_SORT_ORDER.get(s, len(DEFAULT_SAMPLER_ORDER)))
            df_vln = df_vln.sort_values(["_sort", "Sampler"]).drop(columns=["_sort"])
            fig_vln = px.violin(
                df_vln, x="Sampler", y="Accuracy",
                color="Sampler", box=True, points="all",
                color_discrete_map=_vln_color_zh,
                category_orders={"Sampler": sampler_order_vln},
                title="各 Sampler 所有 Trial 的準確率分布",
                labels={"Accuracy": "Accuracy", "Sampler": ""},
            )
            if baseline.get("accuracy"):
                fig_vln.add_hline(
                    y=baseline["accuracy"], line_dash="dash", line_color="#e74c3c",
                    annotation_text=f"Baseline {baseline['accuracy']:.3f}", annotation_position="top right",
                    annotation_font=dict(size=11),
                )
            fig_vln.update_layout(
                height=550, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(tickfont=dict(size=17)),
                title_font=dict(size=18),
            )
            fig_vln.update_yaxes(gridcolor="#eee")
            st.plotly_chart(fig_vln, width="stretch")

        with _vln_tab_en:
            st.caption(
                "Violin plot showing each sampler's accuracy distribution across all trials (not just the best). "
                "A distribution concentrated at high accuracy → consistently good solutions; "
                "a long lower tail → occasional catastrophic outputs."
            )
            df_vln_en = pd.DataFrame(_violin_rows_en)
            sampler_order_vln_en = [SAMPLER_LABELS_EN[s] for s in DEFAULT_SAMPLER_ORDER if s in SAMPLER_LABELS_EN]
            df_vln_en["Sampler"] = df_vln_en["Sampler"].astype(str)
            df_vln_en["_sort"] = df_vln_en["Sampler"].map(lambda s: SAMPLER_SORT_ORDER.get(s, len(DEFAULT_SAMPLER_ORDER)))
            df_vln_en = df_vln_en.sort_values(["_sort", "Sampler"]).drop(columns=["_sort"])
            fig_vln_en = px.violin(
                df_vln_en, x="Sampler", y="Accuracy",
                color="Sampler", box=True, points="all",
                color_discrete_map=_vln_color_en,
                category_orders={"Sampler": sampler_order_vln_en},
                title="Accuracy Distribution Across All Trials by Sampler",
                labels={"Accuracy": "Accuracy", "Sampler": ""},
            )
            if baseline.get("accuracy"):
                fig_vln_en.add_hline(
                    y=baseline["accuracy"], line_dash="dash", line_color="#e74c3c",
                    annotation_text=f"Baseline {baseline['accuracy']:.3f}", annotation_position="top right",
                    annotation_font=dict(size=11),
                )
            fig_vln_en.update_layout(
                height=550, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(tickfont=dict(size=30)),
                title_font=dict(size=30),
            )
            fig_vln_en.update_yaxes(gridcolor="#eee")
            st.plotly_chart(fig_vln_en, width="stretch")

    st.markdown("---")

    # ── Score 分布（Violin）──────────────────────────────────────────────────
    st.markdown("#### 🎻 Score 分布（所有 Trial）")
    if _violin_rows:
        _score_tab_zh, _score_tab_en = st.tabs(["中文", "English"])

        with _score_tab_zh:
            st.caption(
                "小提琴圖顯示每個 sampler 在所有 trial 中的 score 分布。"
                "分布越集中在高 score → 穩定產出好方案；尾部向下延伸 → 偶爾產出差方案。"
            )
            df_score = pd.DataFrame(_violin_rows)
            sampler_order_score = [SAMPLER_LABELS[s] for s in DEFAULT_SAMPLER_ORDER if s in SAMPLER_LABELS]
            df_score["Sampler"] = df_score["Sampler"].astype(str)
            df_score["_sort"] = df_score["Sampler"].map(lambda s: SAMPLER_SORT_ORDER.get(s, len(DEFAULT_SAMPLER_ORDER)))
            df_score = df_score.sort_values(["_sort", "Sampler"]).drop(columns=["_sort"])
            fig_score = px.violin(
                df_score, x="Sampler", y="Score",
                color="Sampler", box=True, points="all",
                color_discrete_map=_vln_color_zh,
                category_orders={"Sampler": sampler_order_score},
                title="各 Sampler 所有 Trial 的 Score 分布",
                labels={"Score": "Score", "Sampler": ""},
            )
            fig_score.update_layout(
                height=550, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(tickfont=dict(size=17)),
                title_font=dict(size=18),
            )
            fig_score.update_yaxes(gridcolor="#eee")
            st.plotly_chart(fig_score, width="stretch")

        with _score_tab_en:
            st.caption(
                "Violin plot showing each sampler's score distribution across all trials. "
                "A distribution concentrated at high score → consistently good configurations; "
                "a long lower tail → occasional poor configurations."
            )
            df_score_en = pd.DataFrame(_violin_rows_en)
            sampler_order_score_en = [SAMPLER_LABELS_EN[s] for s in DEFAULT_SAMPLER_ORDER if s in SAMPLER_LABELS_EN]
            df_score_en["Sampler"] = df_score_en["Sampler"].astype(str)
            df_score_en["_sort"] = df_score_en["Sampler"].map(lambda s: SAMPLER_SORT_ORDER.get(s, len(DEFAULT_SAMPLER_ORDER)))
            df_score_en = df_score_en.sort_values(["_sort", "Sampler"]).drop(columns=["_sort"])
            fig_score_en = px.violin(
                df_score_en, x="Sampler", y="Score",
                color="Sampler", box=True, points="all",
                color_discrete_map=_vln_color_en,
                category_orders={"Sampler": sampler_order_score_en},
                title="Score Distribution Across All Trials by Sampler",
                labels={"Score": "Score", "Sampler": ""},
            )
            fig_score_en.update_layout(
                height=550, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(tickfont=dict(size=30)),
                title_font=dict(size=30),
            )
            fig_score_en.update_yaxes(gridcolor="#eee")
            st.plotly_chart(fig_score_en, width="stretch")

    st.markdown("---")
    st.markdown("#### 📊 Score 品質層級分布（各 Sampler 的 Trial 組成）")
    st.caption(
        "將所有 trial 的 score 分為五個層級，以 100% 堆疊長條圖呈現各 sampler 的分布結構。"
        "LLM 方法傾向雙峰型（高 Excellent + 高 Bad）；統計方法則多 Catastrophic。"
    )
    _tiers = [
        ("Excellent (>4)",       lambda s: s > 4,           "#2ecc71"),
        ("Good (1~4)",           lambda s: 1 <= s <= 4,     "#3498db"),
        ("Poor (0~1)",           lambda s: 0 <= s < 1,      "#f39c12"),
        ("Bad (-10~0)",          lambda s: -10 <= s < 0,    "#e67e22"),
        ("Catastrophic (<-10)",  lambda s: s < -10,         "#e74c3c"),
    ]
    _tier_rows = []
    _sampler_tier_data: dict[str, list] = {}
    for row in _all_exp_rows:
        _s = row["exp"]["sampler"]
        _sampler_tier_data.setdefault(_s, []).extend(row["scores"])
    for _s in sorted(_sampler_tier_data.keys(), key=sampler_sort_key):
        _scores = _sampler_tier_data[_s]
        _n = len(_scores)
        for _tier_label, _fn, _color in _tiers:
            _tier_rows.append({
                "Sampler": SAMPLER_LABELS.get(_s, _s),
                "Tier": _tier_label,
                "Pct": round(sum(1 for sc in _scores if _fn(sc)) / _n * 100, 1),
                "Color": _color,
            })
    if _tier_rows:
        _df_tier = pd.DataFrame(_tier_rows)
        _sampler_order_tier = [SAMPLER_LABELS.get(s, s) for s in sorted(_sampler_tier_data.keys(), key=sampler_sort_key)]
        _tier_order = [t[0] for t in _tiers]
        _tier_color_map = {t[0]: t[2] for t in _tiers}
        fig_tier = px.bar(
            _df_tier, x="Sampler", y="Pct", color="Tier",
            barmode="stack",
            color_discrete_map=_tier_color_map,
            category_orders={"Sampler": _sampler_order_tier, "Tier": _tier_order},
            labels={"Pct": "Trial 比例 (%)", "Sampler": ""},
            title="各 Sampler 的 Score 品質層級分布（100% 堆疊）",
        )
        fig_tier.update_layout(
            height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 100], ticksuffix="%"),
            legend_title="層級",
        )
        fig_tier.update_xaxes(tickangle=0)
        st.plotly_chart(fig_tier, width="stretch")

    st.markdown("---")
    st.markdown("#### ⚡ 首次達到 Score > 4 所需 Trial 數")
    st.caption(
        "每條 run 中，best-so-far 首次達到 score > 4 的 iteration 編號。"
        "點代表個別 run，長條代表平均值。越少 → 越早找到高品質解。"
    )
    _first4_rows = []
    for row in _all_exp_rows:
        _s = row["exp"]["sampler"]
        _bsf = row["best_so_far"]
        _hit = next((i + 1 for i, v in enumerate(_bsf) if v >= 4.0), None)
        _first4_rows.append({
            "Sampler": SAMPLER_LABELS.get(_s, _s),
            "Trial": _hit if _hit else len(_bsf) + 1,
            "實驗": row["exp"]["dir_name"].replace("Llama-3.2-3B-Instruct_gsm8k_", ""),
            "達到": _hit is not None,
        })
    if _first4_rows:
        _df_f4 = pd.DataFrame(_first4_rows)
        _sampler_order_f4 = [SAMPLER_LABELS.get(s, s) for s in sorted(
            _df_f4["Sampler"].unique(), key=lambda x: sampler_sort_key(next((k for k,v in SAMPLER_LABELS.items() if v==x), x))
        )]
        # 平均值長條
        _df_f4_avg = _df_f4.groupby("Sampler")["Trial"].mean().reset_index()
        _df_f4_avg.columns = ["Sampler", "Avg Trial"]
        fig_f4 = go.Figure()
        # bars
        for _, _row in _df_f4_avg.iterrows():
            _sk = next((k for k, v in SAMPLER_LABELS.items() if v == _row["Sampler"]), _row["Sampler"])
            _col = SAMPLER_COLORS.get(_sk, "#888")
            fig_f4.add_trace(go.Bar(
                x=[_row["Sampler"]], y=[_row["Avg Trial"]],
                name=_row["Sampler"],
                marker_color=_col,
                text=[f"{_row['Avg Trial']:.1f}"],
                textposition="outside",
                showlegend=False,
            ))
        # individual dots
        for _, _row in _df_f4.iterrows():
            _sk = next((k for k, v in SAMPLER_LABELS.items() if v == _row["Sampler"]), _row["Sampler"])
            _col = SAMPLER_COLORS.get(_sk, "#888")
            fig_f4.add_trace(go.Scatter(
                x=[_row["Sampler"]], y=[_row["Trial"]],
                mode="markers",
                marker=dict(color=_col, size=10, symbol="circle",
                            line=dict(color="white", width=1.5)),
                showlegend=False,
                hovertemplate=f"<b>{_row['實驗']}</b><br>Trial: {_row['Trial']}<extra></extra>",
            ))
        fig_f4.update_layout(
            height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="", yaxis_title="Trial #（首次 score > 4）",
            title="各 Sampler：首次達到 Score > 4 所需的 Trial 數（平均 + 個別 Run）",
            barmode="overlay",
        )
        fig_f4.update_yaxes(gridcolor="#eee", rangemode="tozero")
        st.plotly_chart(fig_f4, width="stretch")

    st.markdown("---")
    st.markdown("#### 🎯 Score 門檻命中率（Hit Rate）")
    st.caption(
        "各 sampler 在所有 trial 中，達到特定 score 門檻的比例。"
        "門檻越高越嚴苛；命中率越高 → 該 sampler 更可靠、更少浪費 trial。"
    )
    _thresholds = [0.0, 1.0, 2.0, 3.0, 4.0]
    _hr_rows = []
    _sampler_trials: dict[str, list] = {}
    for row in _all_exp_rows:
        s = row["exp"]["sampler"]
        _sampler_trials.setdefault(s, []).extend(
            t["metrics"]["score"] for t in row["valid"]
        )
    for s, scores in _sampler_trials.items():
        for thr in _thresholds:
            hit = sum(1 for sc in scores if sc >= thr)
            _hr_rows.append({
                "Sampler": SAMPLER_LABELS.get(s, s),
                "門檻":    f"≥ {thr:.0f}",
                "命中率 %": round(hit / len(scores) * 100, 1) if scores else 0,
            })
    if _hr_rows:
        df_hr = pd.DataFrame(_hr_rows)
        fig_hr = px.bar(
            df_hr, x="門檻", y="命中率 %", color="Sampler", barmode="group",
            color_discrete_map={SAMPLER_LABELS.get(s,s): c for s, c in {
                "tpe": "#3498db", "nsga2": "#e74c3c", "random": "#95a5a6",
                "window": "#e67e22", "summary": "#a8632e", "tool": "#8e44ad", "full": "#6e4816",
            }.items()},
            text=df_hr["命中率 %"].apply(lambda v: f"{v:.0f}%"),
            title="各 Sampler 在不同 Score 門檻的命中率",
            labels={"命中率 %": "命中率 %", "門檻": "Score 門檻"},
        )
        fig_hr.update_traces(textposition="outside")
        fig_hr.update_layout(
            height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend_title="Sampler", yaxis_range=[0, 115],
        )
        fig_hr.update_yaxes(gridcolor="#eee")
        st.plotly_chart(fig_hr, width="stretch")

    st.markdown("---")

    # ── 實驗健康度診斷 ─────────────────────────────────────────────────────────
    st.markdown("#### 🩺 實驗健康度診斷")
    health_df, health_sum = compute_trial_health(experiments)
    total_e = health_sum["total_expected"]
    total_v = health_sum["total_valid"]
    total_s = health_sum["total_skipped"]
    total_m = health_sum["total_missing"]

    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("預期 Trial 總數", total_e)
    hc2.metric("✅ 有效完成", total_v, f"{total_v/total_e*100:.1f}%")
    hc3.metric("🟡 Skipped", total_s,
               f"−{total_s/total_e*100:.1f}%",
               delta_color="inverse")
    hc4.metric("🔴 Missing（中斷）", total_m,
               f"−{total_m/total_e*100:.1f}%",
               delta_color="inverse")

    if health_df.empty:
        st.success("所有實驗的 Trial 均正常完成，無異常記錄。")
    else:
        with st.expander(
            f"📋 查看詳細異常記錄（共 {len(health_df)} 筆）",
            expanded=True,
        ):
            # 異常類型說明
            st.markdown("""
**異常類型說明：**

| 狀態 | 原因 | 影響 |
|---|---|---|
| 🟡 **Skipped（重複 config）** | LLM Agent（Global_Tuner_memory）建議了已嘗試過的相同壓縮配置，系統自動偵測到重複並跳過，避免浪費 GPU 資源 | Trial 記錄為 skipped，不計入有效分數；探索多樣性降低 |
| 🔴 **Skipped（評估失敗）** | 所有指標（accuracy/latency/vram/emissions）都為 0，可能模型推論 crash、無輸出或評估異常 | Trial 記錄為無效，不計入結果；需要檢查相應日誌 |
| 🔴 **Missing（實驗中斷）** | 進程在最後一個 trial 完成後、下一個 iteration 啟動前被外部終止（可能是 GPU 搶佔、手動停止或系統 timeout）。JSON 已正常儲存前 N 筆結果，第 20 筆未被寫入 | 缺少 1 筆資料點；由於前 19 筆已完整，對統計影響極小 |
""")

            # 彩色表格
            def style_health(df):
                def row_color(row):
                    if "missing" in str(row["狀態"]):
                        return ["background-color: #fde8e8; color: #2c3e50"] * len(row)
                    elif "skipped" in str(row["狀態"]):
                        return ["background-color: #fef9e7; color: #2c3e50"] * len(row)
                    return ["background-color: #f0f9f0; color: #2c3e50"] * len(row)
                styled = df.style.apply(row_color, axis=1)
                styled = styled.set_table_styles([
                    {"selector": "th", "props": [
                        ("background-color", "#2c3e50"), ("color", "white"),
                        ("font-size", "12px"), ("padding", "6px 12px"), ("text-align", "center"),
                    ]},
                    {"selector": "td", "props": [
                        ("font-size", "11px"), ("padding", "5px 10px"), ("color", "#2c3e50"),
                    ]},
                ])
                return styled

            st.write(
                style_health(health_df.reset_index(drop=True)).to_html(),
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # 各實驗 skipped 次數統計
            if total_s > 0:
                st.markdown("**各實驗 Skipped 次數（重複 config 被去重跳過）**")
                skip_sum = (
                    health_df[health_df["狀態"].str.contains("skipped")]
                    .groupby(["實驗", "Sampler"])
                    .size()
                    .reset_index(name="Skipped 次數")
                    .sort_values("Skipped 次數", ascending=False)
                )
                fig_skip = px.bar(
                    skip_sum, x="實驗", y="Skipped 次數", color="Sampler",
                    color_discrete_map={
                        "window": "#e67e22", "summary": "#9b59b6", "tool": "#8e44ad", "full": "#7d3c98",
                    },
                    text="Skipped 次數",
                    title="各實驗 Skipped Trial 次數（LLM 重複建議）",
                )
                fig_skip.update_xaxes(tickangle=25)
                fig_skip.update_traces(textposition="outside")
                fig_skip.update_layout(
                    height=340, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=True,
                )
                fig_skip.update_yaxes(gridcolor="#eee", dtick=1)
                st.plotly_chart(fig_skip, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2：Trial 彩色分析
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Trial 彩色分析")

    exp_names = [f"[{e['sampler']}] {e['dir_name']}" for e in filtered]
    if not exp_names:
        st.warning("沒有符合篩選條件的實驗")
    else:
        sel = st.selectbox("選擇實驗", exp_names, key="tab2_sel")
        exp = filtered[exp_names.index(sel)]
        cfg = exp["config"]

        # 實驗設定摘要（緊湊小字版）
        w = cfg.get("weights", {})
        info_html = f"""
        <div style="font-size:12px; color:#2c3e50; background:#f0f4f8; border-radius:6px;
                    padding:10px 16px; margin-bottom:12px; line-height:2;
                    border-left: 4px solid #3498db;">
          <b>Model</b>: {cfg.get("model_id", "-")} &nbsp;|&nbsp;
          <b>Task</b>: {cfg.get("task", "-").upper()} &nbsp;|&nbsp;
          <b>Sampler</b>: {exp.get("sampler", "-")} &nbsp;|&nbsp;
          <b>Iterations</b>: {cfg.get("max_iterations", "-")} &nbsp;|&nbsp;
          <b>Samples</b>: {cfg.get("num_samples", "-")}<br>
          <b>Weights</b>: acc={w.get("acc","?")} &nbsp; lat={w.get("lat","?")} &nbsp;
          vram={w.get("vram","?")} &nbsp; emit={w.get("emit","?")} &nbsp;|&nbsp;
          <b>pen_t</b>={cfg.get("pen_t", 0.15)} &nbsp;
          <b>pen_a</b>={cfg.get("pen_a", 10.0)}
        </div>
        """
        st.write(info_html, unsafe_allow_html=True)

        b = cfg.get("baseline", baseline)
        df = build_trials_df(exp["trials"], b)

        if df.empty:
            st.warning("此實驗無有效 trial 資料")
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(
                    df, x="Score", nbins=15,
                    title="Score 分佈", color_discrete_sequence=["#3498db"],
                )
                fig_hist.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=320
                )
                st.plotly_chart(fig_hist, width="stretch")
            with c2:
                mode_counts = df["Mode"].value_counts().reset_index()
                mode_counts.columns = ["Mode", "Count"]
                fig_pie = px.pie(
                    mode_counts, names="Mode", values="Count",
                    title="壓縮 Mode 分布",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig_pie.update_layout(height=320)
                st.plotly_chart(fig_pie, width="stretch")

            st.subheader(f"全部 {len(df)} 個 Trial 結果")
            st.write(style_df(df).to_html(), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # ── 詳細參數表 ──────────────────────────────────────────────────
            with st.expander("⚙️ 詳細參數一覽", expanded=False):
                st.caption(
                    "各 trial 的完整壓縮參數，NaN 代表該模式不使用此參數。"
                    "量化：bits / group_size / q_format / damp / double_quant｜"
                    "ASVD：asvd_ratio / asvd_α / asvd_scaling｜"
                    "稀疏：sparse_struct / sparse_ratio"
                )
                param_rows = []
                for t in exp["trials"]:
                    m = t.get("metrics", {})
                    if "accuracy" not in m:
                        continue
                    # 排除所有指標都是 0 的無效 trial
                    if m.get("accuracy") == 0 and m.get("latency") == 0 and m.get("vram") == 0 and m.get("emissions") == 0:
                        continue
                    tname = t.get("trial_name", f"iter {t.get('iteration')}")
                    tnum  = int(tname.split("_")[1]) if "_" in tname and tname.split("_")[1].isdigit() else tname
                    row = {"Trial #": tnum, "設定": config_summary(t), "Score": round(t["metrics"].get("score", 0), 4)}
                    row.update(extract_param_cols(t))
                    param_rows.append(row)
                if param_rows:
                    df_params = pd.DataFrame(param_rows).sort_values("Trial #")
                    # 只顯示至少有一個非 NaN 的參數欄
                    param_cols = ["bits", "group_size", "q_format", "damp", "double_quant",
                                  "asvd_ratio", "asvd_α", "asvd_scaling",
                                  "sparse_struct", "sparse_ratio"]
                    visible_params = [c for c in param_cols if df_params[c].notna().any()]
                    show_cols = ["Trial #", "設定", "Score"] + visible_params
                    df_show = df_params[show_cols].copy()
                    styled_p = df_show.style.background_gradient(subset=["Score"], cmap="RdYlGn")
                    styled_p = styled_p.format({"Score": "{:.4f}"}, na_rep="-")
                    styled_p = styled_p.set_table_styles([
                        {"selector": "th", "props": [
                            ("background-color", "#2c3e50"), ("color", "white"),
                            ("font-size", "12px"), ("text-align", "center"), ("padding", "6px 10px"),
                        ]},
                        {"selector": "td", "props": [
                            ("font-size", "11px"), ("padding", "4px 8px"), ("text-align", "center"),
                        ]},
                    ])
                    st.write(styled_p.to_html(), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

            # ── LLM Suggestion 列表（僅 Global_Tuner_memory 實驗）────────────
            if exp.get("tuner") == "Global_Tuner_memory":
                st.markdown("---")
                st.subheader("🤖 LLM Suggestion 記錄")
                st.caption("每個 iteration 中 LLM 給出的壓縮建議與推理說明。")

                sugg_trials = [t for t in exp["trials"] if t.get("suggestion")]
                if not sugg_trials:
                    st.info("此實驗無 suggestion 記錄。")
                else:
                    for t in sugg_trials:
                        sg = t["suggestion"]
                        tname = t.get("trial_name", f"iter {t.get('iteration')}")
                        tnum  = int(tname.split("_")[1]) if "_" in tname and tname.split("_")[1].isdigit() else t.get("iteration")
                        m = t.get("metrics", {})
                        score_str = f"Score: {m['score']:.4f}" if "accuracy" in m and not (m.get("accuracy")==0 and m.get("latency")==0) else "（無效）"
                        is_valid  = "accuracy" in m and not (m.get("accuracy")==0 and m.get("latency")==0 and m.get("vram")==0 and m.get("emissions")==0)

                        # 頭部：trial 號 + 配置 + score（展開在裡面放 reasoning）
                        header = f"Iter {tnum:02d} ｜ {config_summary(t)} ｜ {score_str}"
                        with st.expander(header, expanded=False):
                            reasoning = sg.get("reasoning", "")
                            if reasoning:
                                st.markdown(f"> {reasoning}")

                            # 建議參數（去掉 reasoning 欄位後其餘 key）
                            params = {k: v for k, v in sg.items() if k != "reasoning"}
                            if params:
                                st.markdown("**建議參數：**")
                                cols = st.columns(min(len(params), 4))
                                for i, (k, v) in enumerate(params.items()):
                                    cols[i % len(cols)].metric(k, v)

                            if not is_valid:
                                st.warning("此 trial 未產生有效結果（被 skip 或評估失敗）")
                st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3：Pareto Frontier
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🎯 多目標壓縮最佳化：Pareto Frontier 分析")
    st.caption(
        "Pareto 非支配解集合（Pareto Frontier）代表在準確率、推理延遲、顯存使用、碳排放四個目標上，"
        "無法在不犧牲任一目標的情況下進一步改善的配置集合。"
        "這些解共同構成壓縮效果與模型品質之間的最佳權衡邊界。"
    )

    exp_names3 = [f"[{e['sampler']}] {e['dir_name']}" for e in filtered]
    if not exp_names3:
        st.warning("沒有符合篩選條件的實驗")
    else:
        sel3 = st.selectbox("選擇實驗", exp_names3, key="tab3_sel")
        exp3 = filtered[exp_names3.index(sel3)]
        b3   = exp3["config"].get("baseline", baseline)
        pareto = exp3["pareto"]
        all_valid = [t for t in exp3["trials"]
                     if "accuracy" in t.get("metrics", {}) and
                     not (t["metrics"].get("accuracy") == 0 and t["metrics"].get("latency") == 0 and
                          t["metrics"].get("vram") == 0 and t["metrics"].get("emissions") == 0)]

        if not pareto:
            st.warning("此實驗無 Pareto 解")
        else:
            df_p = build_trials_df(pareto, b3)
            df_all_p = build_trials_df(all_valid, b3)

            # ── 摘要指標卡 ────────────────────────────────────────────────────
            st.markdown("### 📌 Pareto Frontier 摘要")
            best_acc_row  = df_p.loc[df_p["Accuracy"].idxmax()]
            best_vram_row = df_p.loc[df_p["VRAM (GB)"].idxmin()]
            best_lat_row  = df_p.loc[df_p["Latency (s)"].idxmin()]
            best_score_row = df_p.loc[df_p["Score"].idxmax()]

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Pareto 解數量",
                      f"{len(df_p)} / {len(all_valid)}",
                      help="Pareto 非支配解 / 全部有效 trial 數")
            m2.metric("最高準確率（Pareto 內）",
                      f"{best_acc_row['Accuracy']:.4f}",
                      f"{best_acc_row['Δ Accuracy %']:+.1f}%")
            m3.metric("最低 VRAM（Pareto 內）",
                      f"{best_vram_row['VRAM (GB)']:.3f} GB",
                      f"{best_vram_row['Δ VRAM %']:+.1f}%")
            m4.metric("最低延遲（Pareto 內）",
                      f"{best_lat_row['Latency (s)']:.1f} s",
                      f"{best_lat_row['Δ Latency %']:+.1f}%")
            m5.metric("最高 Score（Pareto 內）",
                      f"{best_score_row['Score']:.4f}",
                      best_score_row["設定"])

            st.markdown("---")

            # ── Pareto 解彩色表格 ─────────────────────────────────────────────
            st.markdown("### 📋 Pareto Frontier 非支配解一覽")
            st.caption("以下配置在四個壓縮目標上互不支配，各自代表不同的效能-效率權衡點。")
            df_p_sorted = df_p.sort_values("Score", ascending=False).reset_index(drop=True)
            df_p_sorted.index += 1
            st.write(style_df(df_p_sorted).to_html(), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # ── 散佈圖：Accuracy vs VRAM / Latency ───────────────────────────
            st.markdown("### 📉 壓縮效果與準確率的權衡空間")
            st.caption(
                "灰色圓點為所有搜尋過的配置；紅色星形為 Pareto 非支配解。"
                "黑色菱形為未壓縮基線（Baseline）。"
            )

            df_all_p["Type"] = "所有 Trial"
            df_p2 = df_p.copy(); df_p2["Type"] = "Pareto Frontier"
            df_base_pt = pd.DataFrame([{
                "Accuracy": b3.get("accuracy"), "VRAM (GB)": b3.get("vram"),
                "Latency (s)": b3.get("latency"),
                "Emissions (g)": b3.get("emissions", 0) * 1000,
                "Score": 0, "Trial": "Baseline", "設定": "未壓縮基線", "Type": "Baseline",
                "Δ Accuracy %": 0, "Δ Latency %": 0, "Δ VRAM %": 0, "Δ Emissions %": 0,
                "Mode": "-",
            }])
            df_vis = pd.concat([df_all_p, df_p2, df_base_pt], ignore_index=True)

            c1, c2 = st.columns(2)
            for ax, ay, title_s, xlab, ylab in [
                ("VRAM (GB)", "Accuracy",
                 "準確率 vs. 顯存用量",
                 "VRAM 用量（GB）↓ 越小越好", "準確率（Accuracy）↑ 越高越好"),
                ("Latency (s)", "Accuracy",
                 "準確率 vs. 推理延遲",
                 "推理總時間（秒）↓ 越小越好", "準確率（Accuracy）↑ 越高越好"),
            ]:
                fig_s = px.scatter(
                    df_vis, x=ax, y=ay, color="Type",
                    hover_data=["Trial", "設定", "Score"],
                    title=title_s,
                    color_discrete_map={
                        "所有 Trial":      "#cccccc",
                        "Pareto Frontier": "#e74c3c",
                        "Baseline":        "#2c3e50",
                    },
                    symbol="Type",
                    symbol_map={
                        "所有 Trial":      "circle",
                        "Pareto Frontier": "star",
                        "Baseline":        "diamond",
                    },
                    size_max=14,
                )
                fig_s.update_traces(selector=dict(name="Pareto Frontier"), marker=dict(size=13))
                fig_s.update_traces(selector=dict(name="Baseline"),        marker=dict(size=14))
                fig_s.update_traces(selector=dict(name="所有 Trial"),       marker=dict(size=7, opacity=0.5))
                fig_s.update_layout(
                    xaxis_title=xlab, yaxis_title=ylab,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=420,
                    legend_title="類型",
                )
                fig_s.update_xaxes(gridcolor="#eee")
                fig_s.update_yaxes(gridcolor="#eee")
                if ax == "VRAM (GB)":
                    c1.plotly_chart(fig_s, width="stretch")
                else:
                    c2.plotly_chart(fig_s, width="stretch")

            # ── Parallel Coordinates ──────────────────────────────────────────
            st.markdown("### 🔀 多目標權衡平行座標圖")
            st.caption(
                "每條線代表一個 Pareto 非支配解，顏色對應 Score。"
                "可觀察哪些配置在多個目標上同時表現優異。"
            )
            df_para = df_p_sorted[["設定", "Score", "Accuracy", "Latency (s)", "VRAM (GB)", "Emissions (g)"]].copy()
            df_para["Emissions (g)"] = df_para["Emissions (g)"].round(4)
            fig_para = go.Figure(go.Parcoords(
                line=dict(
                    color=df_para["Score"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Score"),
                ),
                dimensions=[
                    dict(label="Accuracy ↑",     values=df_para["Accuracy"],      range=[0, 1]),
                    dict(label="VRAM (GB) ↓",    values=df_para["VRAM (GB)"],     range=[df_para["VRAM (GB)"].max()*1.1, 0]),
                    dict(label="Latency (s) ↓",  values=df_para["Latency (s)"],   range=[df_para["Latency (s)"].max()*1.1, 0]),
                    dict(label="Emissions (g) ↓",values=df_para["Emissions (g)"], range=[df_para["Emissions (g)"].max()*1.1, 0]),
                    dict(label="Score ↑",        values=df_para["Score"]),
                ],
            ))
            fig_para.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_para, width="stretch")

            # ── Radar Chart：Top 5 ────────────────────────────────────────────
            st.markdown("### 🕸️ Top-5 Pareto 解雷達圖")
            st.caption("從 Pareto Frontier 中取分數最高的前 5 個配置，以雷達圖比較其歸一化指標。")
            top5 = df_p_sorted.head(5)
            radar_metrics = ["Accuracy", "Latency (s)", "VRAM (GB)", "Emissions (g)"]
            b_vals = {
                "Accuracy":      b3.get("accuracy", 1),
                "Latency (s)":   b3.get("latency",  1),
                "VRAM (GB)":     b3.get("vram",     1),
                "Emissions (g)": b3.get("emissions", 1) * 1000,
            }
            fig_radar = go.Figure()
            colors_r = px.colors.qualitative.Set1
            all_radar_vals = []
            traces_r = []
            labels = ["Accuracy", "Speed\n(1/Latency)", "Memory\n(1/VRAM)", "Green\n(1/Emit)"]
            labels_closed = labels + [labels[0]]
            for idx, (_, row) in enumerate(top5.iterrows()):
                # 歸一化：越大越好的方向統一（latency/vram/emit 取倒數）
                vals = [
                    row["Accuracy"]      / b_vals["Accuracy"],
                    b_vals["Latency (s)"]   / max(row["Latency (s)"],   0.01),
                    b_vals["VRAM (GB)"]     / max(row["VRAM (GB)"],     0.01),
                    b_vals["Emissions (g)"] / max(row["Emissions (g)"], 0.0001),
                ]
                all_radar_vals.extend(vals)
                vals_closed = vals + [vals[0]]
                traces_r.append((vals_closed, row["設定"], colors_r[idx % len(colors_r)]))
            radar_max = max(all_radar_vals) * 1.15 if all_radar_vals else 2.0
            radar_max = max(radar_max, 1.5)  # 至少顯示到 1.5
            for vals_closed, name_r, color_r in traces_r:
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_closed, theta=labels_closed,
                    fill="toself", name=name_r,
                    line=dict(color=color_r),
                    opacity=0.75,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, radar_max])),
                height=460, showlegend=True,
                legend_title="配置（Score 排序）",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_radar, width="stretch")

            # ── 自動分析文字 ──────────────────────────────────────────────────
            st.markdown("### 📝 自動分析摘要")
            pareto_ratio = len(df_p) / len(all_valid) * 100
            best_cfg  = best_score_row["設定"]
            vram_save = abs(best_vram_row["Δ VRAM %"])
            lat_save  = abs(best_lat_row["Δ Latency %"])
            st.info(
                f"**實驗 `{exp3['dir_name']}`** 共執行 {len(all_valid)} 個有效 trial，"
                f"其中 **{len(df_p)} 個（{pareto_ratio:.0f}%）** 構成 Pareto Frontier。\n\n"
                f"- 綜合評分最佳的配置為 **{best_cfg}**，Score = **{best_score_row['Score']:.4f}**\n"
                f"- Pareto 內準確率最高達 **{best_acc_row['Accuracy']:.4f}**"
                f"（vs. baseline {b3.get('accuracy', 0):.4f}，"
                f"變化 {best_acc_row['Δ Accuracy %']:+.1f}%）\n"
                f"- 最大顯存節省達 **{vram_save:.1f}%**（{best_vram_row['設定']}）\n"
                f"- 最大延遲縮短達 **{lat_save:.1f}%**（{best_lat_row['設定']}）\n"
                f"- 非支配解橫跨多種壓縮方法，反映不同部署場景各有最適配置。"
            )

            # ── ⑧ Green AI Frontier（Accuracy vs Emissions Pareto） ────────
            st.markdown("### 🌿 Green AI Frontier（Accuracy vs Emissions Reduction）")
            st.caption(
                "X 軸：準確率保留率（壓縮後 accuracy / baseline accuracy）。"
                "Y 軸：碳排放削減率（(baseline emissions - emissions) / baseline emissions）。"
                "越往右上角越好。綠色星號標記 Accuracy-Emissions 二維 Pareto 解。"
            )
            b_acc_ga = baseline.get("accuracy", 1e-6)
            b_emit_ga = baseline.get("emissions", 1e-6)
            _ga_rows = []
            for _row in _all_exp_rows:
                _sl = SAMPLER_LABELS.get(_row["exp"]["sampler"], _row["exp"]["sampler"])
                for _t in _row["valid"]:
                    _m = _t["metrics"]
                    _ga_rows.append({
                        "Sampler": _sl,
                        "設定": config_summary(_t),
                        "Accuracy Retention": round(_m.get("accuracy", 0) / max(b_acc_ga, 1e-9), 4),
                        "Emissions Reduction": round(
                            (b_emit_ga - _m.get("emissions", 0)) / max(b_emit_ga, 1e-9), 4
                        ),
                    })
            if _ga_rows:
                _df_ga = pd.DataFrame(_ga_rows)
                # 計算 Pareto frontier（最大化兩者）
                _ga_pts = list(zip(_df_ga["Accuracy Retention"], _df_ga["Emissions Reduction"]))
                _ga_pareto_mask = []
                for i, (ax, ay) in enumerate(_ga_pts):
                    dominated = any(
                        (bx >= ax and by >= ay and (bx > ax or by > ay))
                        for j, (bx, by) in enumerate(_ga_pts) if j != i
                    )
                    _ga_pareto_mask.append(not dominated)
                _df_ga["Pareto"] = _ga_pareto_mask
                fig_ga = px.scatter(
                    _df_ga, x="Accuracy Retention", y="Emissions Reduction",
                    color="Sampler", hover_data=["設定"],
                    color_discrete_map={
                        SAMPLER_LABELS.get("tpe", "tpe"): "#3498db",
                        SAMPLER_LABELS.get("nsga2", "nsga2"): "#e74c3c",
                        SAMPLER_LABELS.get("random", "random"): "#95a5a6",
                        SAMPLER_LABELS.get("window", "window"): "#e67e22",
                        SAMPLER_LABELS.get("summary", "summary"): "#a8632e",
                        SAMPLER_LABELS.get("tool", "tool"): "#8e44ad",
                        SAMPLER_LABELS.get("full", "full"): "#6e4816",
                    },
                    title="Green AI Frontier：準確率保留率 vs 碳排放削減率",
                    opacity=0.6,
                )
                # 標記 Pareto 解
                _df_pareto_ga = _df_ga[_df_ga["Pareto"]]
                if not _df_pareto_ga.empty:
                    fig_ga.add_trace(go.Scatter(
                        x=_df_pareto_ga["Accuracy Retention"],
                        y=_df_pareto_ga["Emissions Reduction"],
                        mode="markers",
                        marker=dict(symbol="star", size=14, color="lime",
                                    line=dict(color="green", width=1)),
                        name="Pareto Frontier",
                        text=_df_pareto_ga["設定"],
                        hovertemplate="<b>%{text}</b><br>Acc Ret: %{x:.3f}<br>Emit Red: %{y:.3f}",
                    ))
                fig_ga.add_vline(x=1.0, line_dash="dash", line_color="#e74c3c",
                                 annotation_text="Baseline Acc", annotation_position="top right")
                fig_ga.add_hline(y=0.0, line_dash="dash", line_color="#3498db",
                                 annotation_text="Baseline Emit", annotation_position="bottom right")
                fig_ga.update_layout(
                    height=480, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Accuracy Retention（壓縮後 / Baseline）",
                    yaxis_title="Emissions Reduction（削減比例）",
                    legend_title="Sampler",
                )
                fig_ga.update_yaxes(gridcolor="#eee")
                fig_ga.update_xaxes(gridcolor="#eee")
                st.plotly_chart(fig_ga, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# Tab Cross：跨實驗比較
# ══════════════════════════════════════════════════════════════════════════════
with tab_cross:
    st.header("🔬 跨實驗多維度比較分析")
    st.caption(
        "從不同角度比較 TPE / NSGA-II / Random / Window / Summary 五種搜尋策略，"
        "包含品質穩定性、Pareto 解豐富度、優化學習曲線與壓縮方法偏好分布。"
    )

    all_exp_rows = _all_exp_rows  # 使用預先計算的資料

    # ── 0️⃣ 各實驗最高分 Trial 摘要 ────────────────────────────────────────────
    st.markdown("### 🏅 各實驗最高分 Trial 一覽")
    st.caption("每個實驗中，綜合評分最高的那筆 trial 的詳細配置與指標。")
    best_rows = []
    b0 = baseline
    for row in all_exp_rows:
        best_t = max(row["valid"], key=lambda t: t["metrics"].get("score", -999))
        m = best_t["metrics"]
        b_acc  = row["exp"]["config"].get("baseline", b0).get("accuracy",  b0.get("accuracy",  1e-6))
        b_vram = row["exp"]["config"].get("baseline", b0).get("vram",      b0.get("vram",      1e-6))
        _tname = best_t.get("trial_name", "")
        _tnum  = int(_tname.split("_")[1]) if "_" in _tname and _tname.split("_")[1].isdigit() else 0
        b_lat  = row["exp"]["config"].get("baseline", b0).get("latency",   b0.get("latency",   1e-6))
        b_emit = row["exp"]["config"].get("baseline", b0).get("emissions", b0.get("emissions", 1e-6))
        best_rows.append({
            "實驗":         row["exp"]["dir_name"].replace("Llama-3.2-3B-Instruct_gsm8k_", ""),
            "_tuner":       row["exp"]["tuner"],
            "_sampler":     row["exp"]["sampler"],
            "Trial #":      _tnum,
            "最佳配置":      config_summary(best_t),
            "Score":        round(m.get("score", 0), 4),
            "Accuracy":     round(m.get("accuracy", 0), 4),
            "Δ Acc %":      round((m.get("accuracy", 0) - b_acc)  / b_acc  * 100, 1),
            "Δ VRAM %":     round((m.get("vram", 0)      - b_vram) / b_vram * 100, 1),
            "Δ Latency %":  round((m.get("latency", 0)   - b_lat)  / b_lat  * 100, 1),
            "Δ Emit %":     round((m.get("emissions", 0) - b_emit) / b_emit * 100, 1),
        })
    if best_rows:
        # 預設排序：Tuner 分組，組內依 Sampler 排，同 Sampler 再依 Score 降序
        _TUNER_ORDER   = {"Systematic_Tuner": 0, "Global_Tuner_memory": 1}

        _sort_col = st.radio(
            "排序依據",
            ["策略分組", "Score", "Trial #", "Accuracy", "Δ VRAM %", "Δ Latency %", "Δ Emit %"],
            horizontal=True,
            key="best_trial_sort",
        )

        df_best = pd.DataFrame(best_rows)
        df_best["_sampler"] = df_best["_sampler"].astype(str)
        if _sort_col == "策略分組":
            df_best["_to"] = df_best["_tuner"].map(_TUNER_ORDER).fillna(9)
            df_best["_so"] = df_best["_sampler"].map(sampler_sort_key)
            df_best = df_best.sort_values(["_to", "_so", "Score"], ascending=[True, True, False])
            df_best = df_best.drop(columns=["_to", "_so"])
        elif _sort_col == "Trial #":
            df_best = df_best.sort_values("Trial #", ascending=True)
        elif _sort_col in ["Δ VRAM %", "Δ Latency %", "Δ Emit %"]:
            # 越小越好的指標，升序排（最小=最省在上）
            df_best = df_best.sort_values(_sort_col, ascending=True)
        else:
            # 越大越好（Score / Accuracy），降序排
            df_best = df_best.sort_values(_sort_col, ascending=False)
        df_best = df_best.reset_index(drop=True)

        # 深色 badge：白字在任何主題都清晰可見
        _SAMPLER_BADGE = {
            "tpe":     ("TPE",     "#2471a3", "white"),
            "nsga2":   ("NSGA-II", "#1e8449", "white"),
            "random":  ("Random",  "#b7950b", "white"),
            "window":  ("Window",  "#e377c2", "white"),
            "summary": ("Summary", "#a04000", "white"),
            "tool":    ("Tool",    "#9467bd", "white"),
            "full":    ("Full",    "#17becf", "white"),
        }

        # 用 map 對「策略」欄單獨上色，不影響其他欄的 gradient
        def _badge_style(val):
            for _, (label, bg, fg) in _SAMPLER_BADGE.items():
                if val == label:
                    return (f"background-color:{bg}; color:{fg}; font-weight:700; "
                            f"border-radius:4px; padding:2px 6px;")
            return ""

        display_cols = ["實驗", "Trial #", "最佳配置", "Score",
                        "Accuracy", "Δ Acc %", "Δ VRAM %", "Δ Latency %", "Δ Emit %"]
        df_disp = df_best[["_sampler"] + display_cols].copy()
        df_disp.insert(0, "策略", df_disp["_sampler"].map(
            {k: v[0] for k, v in _SAMPLER_BADGE.items()}
        ))

        styled_best = df_disp.drop(columns=["_sampler"]).style
        styled_best = styled_best.map(_badge_style, subset=["策略"])
        styled_best = styled_best.background_gradient(subset=["Score"],       cmap="RdYlGn")
        styled_best = styled_best.background_gradient(subset=["Δ Acc %"],     cmap="RdYlGn")
        styled_best = styled_best.background_gradient(subset=["Δ VRAM %"],    cmap="RdYlGn_r")
        styled_best = styled_best.background_gradient(subset=["Δ Latency %"], cmap="RdYlGn_r")
        styled_best = styled_best.background_gradient(subset=["Δ Emit %"],    cmap="RdYlGn_r")
        styled_best = styled_best.format({
            "Score": "{:.4f}", "Accuracy": "{:.4f}",
            "Δ Acc %": "{:+.1f}%", "Δ VRAM %": "{:+.1f}%",
            "Δ Latency %": "{:+.1f}%", "Δ Emit %": "{:+.1f}%",
        })
        styled_best = styled_best.set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#2c3e50"), ("color", "white"),
                ("font-size", "12px"), ("padding", "6px 12px"), ("text-align", "center"),
            ]},
            {"selector": "td", "props": [
                ("font-size", "11px"), ("padding", "5px 10px"), ("text-align", "center"),
            ]},
        ])

        st.write(styled_best.to_html(), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    # ── ① 品質-穩定性散佈（Best vs Std）─────────────────────────────────────
    st.markdown("### 1️⃣ 搜尋品質 vs. 穩定性（Best Score vs. Std）")
    st.caption(
        "X 軸為同一 sampler 多次 run 的 best score 標準差（越小=越穩定），"
        "Y 軸為最高 best score（越高=越好）。"
        "越往左上越理想（高品質、高穩定）。"
    )
    _qual_view_mode = st.radio(
        "呈現模式",
        ["強化散點", "穩健排行榜表格", "相對基線四象限", "全部"],
        horizontal=True,
        key="quality_view_mode",
    )
    _robust_lambda = st.slider(
        "穩健係數 λ（Robust Score = Mean Best - λ × Std）",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        key="quality_robust_lambda",
    )

    qual_rows = []
    grp = {}
    for row in all_exp_rows:
        exp = row["exp"]
        tuner = exp["tuner"]
        sampler = exp["sampler"]
        gk = (tuner, sampler)
        grp.setdefault(gk, {
            "bests": [],
            "vram_save": [],
            "lat_save": [],
            "emit_save": [],
            "acc_delta": [],
            "pareto_ratio": [],
        })
        grp[gk]["bests"].append(row["best"])

        best_trial = max(row["valid"], key=lambda t: t.get("metrics", {}).get("score", -999))
        m = best_trial.get("metrics", {})
        b_local = exp.get("config", {}).get("baseline", baseline)
        b_acc = b_local.get("accuracy", baseline.get("accuracy", 1e-6)) or 1e-6
        b_vram = b_local.get("vram", baseline.get("vram", 1e-6)) or 1e-6
        b_lat = b_local.get("latency", baseline.get("latency", 1e-6)) or 1e-6
        b_emit = b_local.get("emissions", baseline.get("emissions", 1e-6)) or 1e-6

        grp[gk]["acc_delta"].append((m.get("accuracy", 0) - b_acc) / b_acc * 100)
        grp[gk]["vram_save"].append((b_vram - m.get("vram", b_vram)) / b_vram * 100)
        grp[gk]["lat_save"].append((b_lat - m.get("latency", b_lat)) / b_lat * 100)
        grp[gk]["emit_save"].append((b_emit - m.get("emissions", b_emit)) / b_emit * 100)
        grp[gk]["pareto_ratio"].append(len(exp.get("pareto", [])) / len(row["valid"]) * 100 if row["valid"] else 0.0)

    for (tuner, sampler), vals in grp.items():
        bests = vals["bests"]
        mean_best = statistics.mean(bests) if bests else 0.0
        std_best = statistics.stdev(bests) if len(bests) > 1 else 0.0
        cv_pct = std_best / max(abs(mean_best), 1e-6) * 100
        qual_rows.append({
            "Sampler": sampler,
            "Tuner": tuner,
            "說明": SAMPLER_LABELS.get(sampler, sampler),
            "Best Score": round(max(bests), 4),
            "Mean Best": round(mean_best, 4),
            "Std": round(std_best, 4),
            "CV %": round(cv_pct, 1),
            "Runs": len(bests),
            "Avg Δ Acc %": round(statistics.mean(vals["acc_delta"]) if vals["acc_delta"] else 0.0, 1),
            "Avg VRAM Save %": round(statistics.mean(vals["vram_save"]) if vals["vram_save"] else 0.0, 1),
            "Avg Latency Save %": round(statistics.mean(vals["lat_save"]) if vals["lat_save"] else 0.0, 1),
            "Avg Emit Save %": round(statistics.mean(vals["emit_save"]) if vals["emit_save"] else 0.0, 1),
            "Avg Pareto 佔比 %": round(statistics.mean(vals["pareto_ratio"]) if vals["pareto_ratio"] else 0.0, 1),
        })

    if qual_rows:
        df_qual = pd.DataFrame(qual_rows)
        df_qual["Robust Score"] = (df_qual["Mean Best"] - _robust_lambda * df_qual["Std"]).round(4)
        df_qual["Bubble Size"] = df_qual["Avg VRAM Save %"].clip(lower=1.0)

        def _style_qual_rank(df: pd.DataFrame):
            styled = df.style
            styled = styled.background_gradient(subset=["Best Score", "Mean Best", "Robust Score"], cmap="RdYlGn")
            styled = styled.background_gradient(subset=["Std", "CV %"], cmap="RdYlGn_r")
            styled = styled.background_gradient(
                subset=["Avg Δ Acc %", "Avg VRAM Save %", "Avg Latency Save %", "Avg Emit Save %", "Avg Pareto 佔比 %"],
                cmap="RdYlGn",
            )
            styled = styled.format({
                "Best Score": "{:.4f}",
                "Mean Best": "{:.4f}",
                "Std": "{:.4f}",
                "CV %": "{:.1f}%",
                "Robust Score": "{:.4f}",
                "Avg Δ Acc %": "{:+.1f}%",
                "Avg VRAM Save %": "{:+.1f}%",
                "Avg Latency Save %": "{:+.1f}%",
                "Avg Emit Save %": "{:+.1f}%",
                "Avg Pareto 佔比 %": "{:.1f}%",
            })
            styled = styled.set_table_styles([
                {"selector": "th", "props": [
                    ("background-color", "#2c3e50"), ("color", "white"),
                    ("font-size", "12px"), ("padding", "6px 10px"), ("text-align", "center"),
                ]},
                {"selector": "td", "props": [
                    ("font-size", "11px"), ("padding", "5px 9px"), ("text-align", "center"),
                ]},
            ])
            return styled

        if _qual_view_mode in ["強化散點", "全部"]:
            fig_qual = px.scatter(
                df_qual,
                x="Std",
                y="Best Score",
                color="Avg Latency Save %",
                text="說明",
                size="Bubble Size",
                size_max=34,
                color_continuous_scale="RdYlGn",
                hover_data=[
                    "Tuner", "Runs", "Mean Best", "Robust Score",
                    "Avg Δ Acc %", "Avg VRAM Save %", "Avg Latency Save %", "Avg Emit Save %", "Avg Pareto 佔比 %",
                ],
                title="品質-穩定-壓縮收益空間（點大小=VRAM 節省，顏色=Latency 節省）",
                labels={
                    "Std": "Std of Best Score across Runs ↓ 越小越穩定",
                    "Best Score": "Best Score ↑ 越高越好",
                    "Avg Latency Save %": "平均延遲節省 %",
                },
            )
            fig_qual.update_traces(
                textposition="top center",
                marker=dict(opacity=0.88, line=dict(width=0.8, color="white")),
            )
            fig_qual.add_vline(x=df_qual["Std"].median(), line_dash="dot", line_color="#999")
            fig_qual.add_hline(y=df_qual["Best Score"].median(), line_dash="dot", line_color="#999")
            fig_qual.add_annotation(
                x=df_qual["Std"].min(),
                y=df_qual["Best Score"].max(),
                text="⭐ 理想區域（左上）",
                showarrow=False,
                font=dict(size=12, color="#27ae60"),
                xanchor="left",
            )
            fig_qual.update_layout(
                height=460,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_colorbar=dict(title="延遲節省 %"),
            )
            fig_qual.update_xaxes(gridcolor="#eee")
            fig_qual.update_yaxes(gridcolor="#eee")
            st.plotly_chart(fig_qual, width="stretch")

        if _qual_view_mode in ["相對基線四象限", "全部"]:
            base_candidates = df_qual[df_qual["Sampler"] == "random"]
            if not base_candidates.empty:
                base_row = base_candidates.iloc[0]
                base_name = base_row["說明"]
                base_note = ""
            else:
                base_row = df_qual.sort_values("Runs", ascending=False).iloc[0]
                base_name = base_row["說明"]
                base_note = "未偵測到 Random Search，已使用 runs 最多的策略作為暫代基線。"

            df_quad = df_qual.copy()
            df_quad["穩定性提升"] = (base_row["Std"] - df_quad["Std"]).round(4)
            df_quad["品質提升"] = (df_quad["Best Score"] - base_row["Best Score"]).round(4)

            fig_quad = px.scatter(
                df_quad,
                x="穩定性提升",
                y="品質提升",
                color="Tuner",
                text="說明",
                size="Runs",
                size_max=28,
                hover_data=["Best Score", "Std", "Robust Score", "Avg VRAM Save %", "Avg Latency Save %"],
                color_discrete_map={
                    "Systematic_Tuner": "#3498db",
                    "Global_Tuner_memory": "#e67e22",
                },
                title=f"相對基線四象限（基線 = {base_name}）",
                labels={
                    "穩定性提升": "穩定性提升（基線 Std - 本策略 Std）↑ 越大越穩",
                    "品質提升": "品質提升（本策略 Best - 基線 Best）↑ 越大越好",
                },
            )
            fig_quad.add_vline(x=0, line_dash="dash", line_color="#666")
            fig_quad.add_hline(y=0, line_dash="dash", line_color="#666")
            fig_quad.add_annotation(
                x=df_quad["穩定性提升"].max(),
                y=df_quad["品質提升"].max(),
                text="優勢象限（右上）",
                showarrow=False,
                font=dict(size=12, color="#27ae60"),
                xanchor="right",
            )
            fig_quad.update_traces(textposition="top center", marker=dict(opacity=0.88))
            fig_quad.update_layout(
                height=460,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig_quad.update_xaxes(gridcolor="#eee")
            fig_quad.update_yaxes(gridcolor="#eee")
            st.plotly_chart(fig_quad, width="stretch")
            if base_note:
                st.caption(base_note)

        if _qual_view_mode in ["穩健排行榜表格", "全部"]:
            st.markdown("#### 📋 穩健優勢排行榜")
            st.caption("整合品質、穩定性與壓縮收益，方便快速比較各搜尋策略。")
            df_rank = df_qual[[
                "說明", "Tuner", "Runs", "Best Score", "Mean Best", "Std", "CV %", "Robust Score",
                "Avg Δ Acc %", "Avg VRAM Save %", "Avg Latency Save %", "Avg Emit Save %", "Avg Pareto 佔比 %",
            ]].copy()
            df_rank = df_rank.sort_values(
                ["Robust Score", "Avg VRAM Save %", "Avg Latency Save %"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
            df_rank.index += 1
            st.write(_style_qual_rank(df_rank).to_html(), unsafe_allow_html=True)

    st.markdown("---")

    # ── ② Pareto 解豐富度 ─────────────────────────────────────────────────────
    st.markdown("### 2️⃣ Pareto Frontier 解豐富度比較")
    st.caption(
        "各實驗的 Pareto 非支配解數量（絕對值）與佔比（%），"
        "反映不同搜尋策略探索解空間的多樣性。解越多 → 給決策者更多選擇。"
    )
    pareto_rows = []
    for row in all_exp_rows:
        exp = row["exp"]
        n_pareto = len(exp["pareto"])
        n_valid  = len(row["valid"])
        pareto_rows.append({
            "實驗":       exp["dir_name"].replace("Llama-3.2-3B-Instruct_gsm8k_", ""),
            "Sampler":   exp["sampler"],
            "Tuner":     exp["tuner"],
            "說明":       SAMPLER_LABELS.get(exp["sampler"], exp["sampler"]),
            "Pareto 解數": n_pareto,
            "有效 Trial 數": n_valid,
            "Pareto 佔比 %": round(n_pareto / n_valid * 100, 1) if n_valid else 0,
        })
    if pareto_rows:
        df_par = pd.DataFrame(pareto_rows)
        df_par["Sampler"] = df_par["Sampler"].astype(str)
        df_par["_sort"] = df_par["Sampler"].map(sampler_sort_key)
        df_par = df_par.sort_values(["_sort", "Sampler"]).drop(columns=["_sort"])
        c1, c2 = st.columns(2)
        with c1:
            fig_pn = px.bar(
                df_par, x="實驗", y="Pareto 解數",
                color="Tuner", text="Pareto 解數",
                color_discrete_map={
                    "Systematic_Tuner":    "#3498db",
                    "Global_Tuner_memory": "#e67e22",
                },
                title="各實驗 Pareto 解數量",
            )
            fig_pn.update_xaxes(tickangle=45)
            fig_pn.update_traces(textposition="outside")
            fig_pn.update_layout(
                height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            fig_pn.update_yaxes(gridcolor="#eee")
            st.plotly_chart(fig_pn, width="stretch")
        with c2:
            # 依 sampler 聚合平均 Pareto 佔比
            grp_par = (
                df_par.assign(
                    _sort=df_par["Sampler"].map(sampler_sort_key)
                )
                .sort_values(["_sort", "Sampler"])
                .groupby("說明", as_index=False, sort=False)
                .agg(avg_pareto=("Pareto 佔比 %", "mean"), avg_count=("Pareto 解數", "mean"))
            )
            fig_ratio = px.bar(
                grp_par, x="avg_pareto", y="說明", orientation="h",
                text=grp_par["avg_pareto"].apply(lambda v: f"{v:.1f}%"),
                title="各 Sampler 平均 Pareto 佔比（%）",
                color="avg_pareto",
                color_continuous_scale="RdYlGn",
            )
            fig_ratio.update_traces(textposition="outside")
            fig_ratio.update_layout(
                height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )
            fig_ratio.update_xaxes(gridcolor="#eee", title="平均 Pareto 佔比 %")
            st.plotly_chart(fig_ratio, width="stretch")

    # ── 門檻版 Pareto：僅計算 score > 4 的解 ─────────────────────────────────
    st.markdown("#### 📌 門檻版（僅計算 Score > 4 的 Pareto 解）")
    st.caption(
        "過濾出 Pareto 解中 score > 4 的子集，排除低品質無意義的非支配解，"
        "更清楚呈現各策略在高品質解空間中的覆蓋能力。"
    )
    _PARETO_SCORE_THR = 4.0
    pareto_thr_rows = []
    for row in all_exp_rows:
        exp = row["exp"]
        n_valid = len(row["valid"])
        pareto_above = [
            t for t in exp["pareto"]
            if t.get("metrics", {}).get("score", -999) > _PARETO_SCORE_THR
        ]
        n_above = len(pareto_above)
        pareto_thr_rows.append({
            "實驗":            exp["dir_name"].replace("Llama-3.2-3B-Instruct_gsm8k_", ""),
            "Sampler":        exp["sampler"],
            "Tuner":          exp["tuner"],
            "說明":            SAMPLER_LABELS.get(exp["sampler"], exp["sampler"]),
            f"Pareto(>{int(_PARETO_SCORE_THR)}) 解數": n_above,
            "有效 Trial 數":   n_valid,
            f"佔比 %":         round(n_above / n_valid * 100, 1) if n_valid else 0,
        })
    if pareto_thr_rows:
        df_thr = pd.DataFrame(pareto_thr_rows)
        df_thr["Sampler"] = df_thr["Sampler"].astype(str)
        df_thr["_sort"] = df_thr["Sampler"].map(sampler_sort_key)
        df_thr = df_thr.sort_values(["_sort", "Sampler"]).drop(columns=["_sort"])
        _col_count = f"Pareto(>{int(_PARETO_SCORE_THR)}) 解數"
        _col_ratio = "佔比 %"
        c1t, c2t = st.columns(2)
        with c1t:
            fig_thr_n = px.bar(
                df_thr, x="實驗", y=_col_count,
                color="Tuner", text=_col_count,
                color_discrete_map={
                    "Systematic_Tuner":    "#3498db",
                    "Global_Tuner_memory": "#e67e22",
                },
                title=f"各實驗 Pareto 解數（Score > {int(_PARETO_SCORE_THR)}）",
            )
            fig_thr_n.update_xaxes(tickangle=45)
            fig_thr_n.update_traces(textposition="outside")
            fig_thr_n.update_layout(
                height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            fig_thr_n.update_yaxes(gridcolor="#eee")
            st.plotly_chart(fig_thr_n, width="stretch")
        with c2t:
            grp_thr = (
                df_thr.assign(_sort=df_thr["Sampler"].map(sampler_sort_key))
                .sort_values(["_sort", "Sampler"])
                .groupby("說明", as_index=False, sort=False)
                .agg(avg_ratio=(_col_ratio, "mean"), avg_count=(_col_count, "mean"))
            )
            fig_thr_ratio = px.bar(
                grp_thr, x="avg_ratio", y="說明", orientation="h",
                text=grp_thr["avg_ratio"].apply(lambda v: f"{v:.1f}%"),
                title=f"各 Sampler 平均 Pareto 佔比（Score > {int(_PARETO_SCORE_THR)}）",
                color="avg_ratio",
                color_continuous_scale="RdYlGn",
            )
            fig_thr_ratio.update_traces(textposition="outside")
            fig_thr_ratio.update_layout(
                height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )
            fig_thr_ratio.update_xaxes(gridcolor="#eee", title="平均佔比 %")
            st.plotly_chart(fig_thr_ratio, width="stretch")

    st.markdown("---")

    # ── ③ 壓縮方法偏好分布 ──────────────────────────────────────────────────
    st.markdown("### 3️⃣ 各 Sampler 壓縮方法偏好分布")
    st.caption(
        "統計各搜尋策略在所有 run 中嘗試過的壓縮方法分布（含 Pareto 解的高亮）。"
        "顯示不同策略是否對特定方法有偏好。"
    )

    # 整體 mode 分布（用具體方法名稱）
    mode_data: dict[str, dict[str, int]] = {}
    pareto_mode_data: dict[str, dict[str, int]] = {}
    for row in all_exp_rows:
        s = row["exp"]["sampler"]
        for t in row["valid"]:
            m_key = detailed_method(t)
            mode_data.setdefault(s, {}).setdefault(m_key, 0)
            mode_data[s][m_key] += 1
        for t in row["exp"]["pareto"]:
            if "accuracy" not in t.get("metrics", {}):
                continue
            m_key = detailed_method(t)
            pareto_mode_data.setdefault(s, {}).setdefault(m_key, 0)
            pareto_mode_data[s][m_key] += 1

    mode_rows = []
    for s, mc in mode_data.items():
        for mode, cnt in mc.items():
            mode_rows.append({
                "Sampler": SAMPLER_LABELS.get(s, s),
                "Mode":    mode,
                "Count":   cnt,
                "Type":    "全部 Trial",
            })
    for s, mc in pareto_mode_data.items():
        for mode, cnt in mc.items():
            mode_rows.append({
                "Sampler": SAMPLER_LABELS.get(s, s),
                "Mode":    mode,
                "Count":   cnt,
                "Type":    "Pareto 解",
            })
    if mode_rows:
        df_mode = pd.DataFrame(mode_rows)
        c1, c2 = st.columns(2)
        # 統一 legend 顏色 + 順序：量化 → 結構 → 稀疏 → 混合
        # 量化：暖色系；ASVD：藍；稀疏：綠色系；混合：紫（與 ASVD 明顯區別）
        _METHOD_ORDER = [
            "gptq", "awq", "bnb", "qqq",
            "asvd",
            "sparse_unstructured", "sparse_structured",
            "hybrid_asvd_bnb",
            "unknown",
        ]
        _METHOD_COLORS = {
            "gptq":               "#c0392b",  # 深紅（量化）
            "awq":                "#e74c3c",  # 紅（量化）
            "bnb":                "#e67e22",  # 橘（量化）
            "qqq":                "#f39c12",  # 琥珀（量化）
            "asvd":               "#3498db",  # 藍（結構壓縮）
            "sparse_unstructured":"#27ae60",  # 綠（稀疏）
            "sparse_structured":  "#1abc9c",  # 青綠（稀疏）
            "hybrid_asvd_bnb":    "#9b59b6",  # 紫（混合，與 ASVD 藍明顯區別）
            "unknown":            "#95a5a6",
        }
        for col_filter, title_m, container in [
            ("全部 Trial", "所有 Trial 的方法分布（按 Sampler）", c1),
            ("Pareto 解",  "Pareto 解的方法分布（按 Sampler）",   c2),
        ]:
            fig_m = px.bar(
                df_mode[df_mode["Type"] == col_filter],
                x="Sampler", y="Count", color="Mode", barmode="stack",
                title=title_m,
                color_discrete_map=_METHOD_COLORS,
                category_orders={"Mode": _METHOD_ORDER},
            )
            fig_m.update_xaxes(tickangle=20)
            fig_m.update_layout(
                height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend_title="壓縮方法",
            )
            fig_m.update_yaxes(gridcolor="#eee")
            container.plotly_chart(fig_m, width="stretch")

    st.markdown("---")

    # ── ④ Best-so-far 收斂曲線 ──────────────────────────────────────────────
    _SAMPLER_CONV_COLORS = SAMPLER_COLORS

    def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    st.markdown("### 4️⃣ Best-so-far 收斂曲線（各 Sampler 平均 ± Std）")
    st.caption(
        "每條線為同一 Sampler 多次 Run 的 best-so-far 平均值，陰影帶為 ±1 標準差。"
        "實線越早趨平 → 收斂越快；陰影帶越窄 → 結果越穩定。"
    )

    # 同一 Sampler 的 best_so_far 對齊取平均/std
    _bsf_by_sampler: dict[str, list] = {}
    for row in all_exp_rows:
        _s = row["exp"]["sampler"]
        _bsf_by_sampler.setdefault(_s, []).append(row["best_so_far"])

    _conv_sampler_opts = sorted(_bsf_by_sampler.keys(), key=sampler_sort_key)
    _conv_sel = st.multiselect(
        "篩選 Sampler",
        options=_conv_sampler_opts,
        default=_conv_sampler_opts,
        format_func=lambda s: SAMPLER_LABELS.get(s, s),
        key="conv_curve_sel",
    )

    fig_conv = go.Figure()
    for _s in _conv_sampler_opts:
        if _s not in _conv_sel:
            continue
        _runs = _bsf_by_sampler[_s]
        _max_t = max(len(r) for r in _runs)
        _avg, _std_up, _std_dn = [], [], []
        for _i in range(_max_t):
            _vals = [r[_i] for r in _runs if _i < len(r)]
            _m = statistics.mean(_vals)
            _sd = statistics.stdev(_vals) if len(_vals) > 1 else 0
            _avg.append(_m)
            _std_up.append(_m + _sd)
            _std_dn.append(_m - _sd)
        _x = list(range(1, _max_t + 1))
        _col = SAMPLER_COLORS.get(_s, "#888")
        _label = SAMPLER_LABELS.get(_s, _s)
        # upper bound (invisible line)
        fig_conv.add_trace(go.Scatter(
            x=_x, y=_std_up, mode="lines",
            line=dict(width=0), showlegend=False,
            hoverinfo="skip", legendgroup=_s,
        ))
        # filled band
        fig_conv.add_trace(go.Scatter(
            x=_x, y=_std_dn, mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=_hex_to_rgba(_col, 0.15),
            showlegend=False, hoverinfo="skip", legendgroup=_s,
        ))
        # mean line
        fig_conv.add_trace(go.Scatter(
            x=_x, y=_avg, mode="lines+markers",
            name=_label,
            line=dict(color=_col, width=2.5),
            marker=dict(size=5),
            legendgroup=_s,
            hovertemplate=f"<b>{_label}</b><br>Trial: %{{x}}<br>Avg Best: %{{y:.3f}}<extra></extra>",
        ))
    fig_conv.update_layout(
        xaxis_title="Trial", yaxis_title="Avg Best Score So Far",
        height=500, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend_title="Sampler", hovermode="x unified",
    )
    fig_conv.update_xaxes(gridcolor="#eee", dtick=2)
    fig_conv.update_yaxes(gridcolor="#eee")
    st.plotly_chart(fig_conv, width="stretch")

    # ── ④-b 聚焦圖：Score > 4 區段 ───────────────────────────────────────────
    st.markdown("#### 📌 聚焦：Score > 4 區段")
    st.caption(
        "僅顯示 best-so-far 平均曲線中 score ≥ 4 的部分，便於觀察高品質解之間的細部差異。"
        "低於 4 分的 iteration 設為 None 不顯示。"
    )
    _SCORE_THRESHOLD = 4.0
    fig_conv_zoom = go.Figure()
    for _s in _conv_sampler_opts:
        if _s not in _conv_sel:
            continue
        _runs = _bsf_by_sampler[_s]
        _max_t = max(len(r) for r in _runs)
        _avg_zoom = []
        for _i in range(_max_t):
            _vals = [r[_i] for r in _runs if _i < len(r)]
            _m = statistics.mean(_vals)
            _avg_zoom.append(_m if _m >= _SCORE_THRESHOLD else None)
        if all(v is None for v in _avg_zoom):
            continue
        _col = SAMPLER_COLORS.get(_s, "#888")
        _label = SAMPLER_LABELS.get(_s, _s)
        fig_conv_zoom.add_trace(go.Scatter(
            x=list(range(1, _max_t + 1)),
            y=_avg_zoom,
            mode="lines+markers",
            name=_label,
            line=dict(color=_col, width=2.5),
            marker=dict(size=5),
            connectgaps=False,
            legendgroup=_s,
            hovertemplate=f"<b>{_label}</b><br>Trial: %{{x}}<br>Avg Best: %{{y:.3f}}<extra></extra>",
        ))
    _all_vals_above = [
        v for _s in _conv_sampler_opts if _s in _conv_sel
        for _runs in [_bsf_by_sampler[_s]]
        for _i in range(max(len(r) for r in _runs))
        for _vals in [[r[_i] for r in _runs if _i < len(r)]]
        for v in [statistics.mean(_vals)]
        if v >= _SCORE_THRESHOLD
    ]
    _y_max = max(_all_vals_above) * 1.02 if _all_vals_above else _SCORE_THRESHOLD + 1
    fig_conv_zoom.update_layout(
        xaxis_title="Trial",
        yaxis_title="Avg Best Score So Far",
        yaxis=dict(range=[_SCORE_THRESHOLD, _y_max], gridcolor="#eee"),
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title="Sampler",
        hovermode="x unified",
    )
    fig_conv_zoom.update_xaxes(gridcolor="#eee", dtick=2)
    fig_conv_zoom.add_hline(
        y=_SCORE_THRESHOLD,
        line_dash="dot", line_color="gray",
        annotation_text="Score = 4",
        annotation_position="right",
    )
    st.plotly_chart(fig_conv_zoom, width="stretch")

    # ── ① Cumulative Regret（5️⃣） ────────────────────────────────────────────
    st.markdown("### 5️⃣ Cumulative Regret 分析")
    st.caption(
        "對每個實驗，以最終 best_so_far 為代理最優值，計算每個 iteration 的 regret = optimal - best_so_far[t]，"
        "再累加得 Cumulative Regret。Regret 越低、越快趨近零 → 收斂越快。"
    )
    _SAMPLER_CONV_COLORS = SAMPLER_COLORS
    fig_regret = go.Figure()
    _regret_legend_shown: set = set()
    for _row in all_exp_rows:
        _bsf = _row["best_so_far"]
        if not _bsf:
            continue
        _optimal = _bsf[-1]
        _regret = [_optimal - v for v in _bsf]
        _cum_regret = []
        _cr = 0.0
        for _r in _regret:
            _cr += _r
            _cum_regret.append(_cr)
        _samp = _row["exp"]["sampler"]
        _color = _SAMPLER_CONV_COLORS.get(_samp, "#888")
        _label = SAMPLER_LABELS.get(_samp, _samp)
        _show_legend = _samp not in _regret_legend_shown
        _regret_legend_shown.add(_samp)
        fig_regret.add_trace(go.Scatter(
            x=list(range(1, len(_cum_regret) + 1)),
            y=_cum_regret,
            mode="lines",
            name=_label,
            line=dict(color=_color, width=1.5),
            opacity=0.7,
            legendgroup=_samp,
            showlegend=_show_legend,
            hovertemplate=f"<b>{_label}</b><br>Iter: %{{x}}<br>Cum Regret: %{{y:.3f}}",
        ))
    fig_regret.update_layout(
        height=420, title="Cumulative Regret by Experiment",
        xaxis_title="Iteration", yaxis_title="Cumulative Regret",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend_title="Sampler", hovermode="x unified",
    )
    fig_regret.update_yaxes(gridcolor="#eee")
    fig_regret.update_xaxes(gridcolor="#eee")
    st.plotly_chart(fig_regret, width="stretch")

    st.markdown("---")
    st.markdown("### 💥 災難性失敗 Timeline（Score < −10）")
    st.caption(
        "每個點代表一次 score < −10 的 trial，X 軸為 iteration，Y 軸為 sampler。"
        "LLM 方法的失敗集中在前期廣探階段（trial 4-8），後期幾乎消失；"
        "統計方法的失敗則散佈整個搜尋過程。"
    )
    _cat_rows = []
    for row in all_exp_rows:
        _s = row["exp"]["sampler"]
        for _i, _sc in enumerate(row["scores"]):
            if _sc < -10:
                _cat_rows.append({
                    "Sampler":  SAMPLER_LABELS.get(_s, _s),
                    "Iteration": _i + 1,
                    "Score":    round(_sc, 2),
                    "實驗":     row["exp"]["dir_name"].replace("Llama-3.2-3B-Instruct_gsm8k_", ""),
                })
    if _cat_rows:
        _df_cat = pd.DataFrame(_cat_rows)
        _sampler_order_cat = [SAMPLER_LABELS.get(s, s) for s in sorted(
            _df_cat["Sampler"].unique(), key=lambda x: sampler_sort_key(
                next((k for k,v in SAMPLER_LABELS.items() if v==x), x))
        )]
        fig_cat_tl = go.Figure()
        for _s_key in sorted({next((k for k,v in SAMPLER_LABELS.items() if v==r["Sampler"]), "") for _, r in _df_cat.iterrows()}, key=sampler_sort_key):
            _s_label = SAMPLER_LABELS.get(_s_key, _s_key)
            _df_s = _df_cat[_df_cat["Sampler"] == _s_label]
            _col = SAMPLER_COLORS.get(_s_key, "#888")
            fig_cat_tl.add_trace(go.Scatter(
                x=_df_s["Iteration"], y=_df_s["Sampler"],
                mode="markers",
                marker=dict(color=_col, size=12, symbol="x", line=dict(color="white", width=1)),
                name=_s_label,
                hovertemplate="<b>%{y}</b><br>Trial: %{x}<br>Score: %{customdata:.2f}<extra></extra>",
                customdata=_df_s["Score"],
            ))
        fig_cat_tl.update_layout(
            height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Trial #", yaxis_title="",
            title="災難性失敗（score < −10）的 Iteration 分布",
            showlegend=False,
            yaxis=dict(categoryorder="array", categoryarray=list(reversed(_sampler_order_cat))),
        )
        fig_cat_tl.update_xaxes(gridcolor="#eee", dtick=2)
        # add shading for early phase (trial 1-10)
        fig_cat_tl.add_vrect(x0=0.5, x1=10.5, fillcolor="rgba(52,152,219,0.07)",
                              layer="below", line_width=0,
                              annotation_text="廣探期(1-10)", annotation_position="top left",
                              annotation_font=dict(size=11))
        st.plotly_chart(fig_cat_tl, width="stretch")
    else:
        st.caption("（目前資料中無 score < −10 的 trial）")

    st.markdown("---")
    st.markdown("### 🔄 災難失敗後的恢復率")
    st.caption(
        "災難性失敗（score < −10）後，下一個 trial 是否為正向（score > 0）。"
        "恢復率越高 → 該方法越能快速從失敗中調整策略。"
    )
    _rec_rows = []
    for row in all_exp_rows:
        _s = row["exp"]["sampler"]
        _scores = row["scores"]
        _rec, _fail = 0, 0
        for _i in range(1, len(_scores)):
            if _scores[_i - 1] < -10:
                if _scores[_i] > 0:
                    _rec += 1
                else:
                    _fail += 1
        _total_cat = _rec + _fail
        if _total_cat > 0:
            _rec_rows.append({
                "Sampler": SAMPLER_LABELS.get(_s, _s),
                "_sk":     _s,
                "恢復次數": _rec,
                "未恢復":  _fail,
                "總失敗後": _total_cat,
                "恢復率 %": round(_rec / _total_cat * 100, 1),
            })
    if _rec_rows:
        _df_rec = pd.DataFrame(_rec_rows)
        _df_rec_agg = (
            _df_rec.groupby(["Sampler", "_sk"])
            .agg(總恢復=("恢復次數", "sum"), 總未恢復=("未恢復", "sum"))
            .reset_index()
        )
        _df_rec_agg["恢復率 %"] = (_df_rec_agg["總恢復"] / (_df_rec_agg["總恢復"] + _df_rec_agg["總未恢復"]) * 100).round(1)
        _df_rec_agg["_sort"] = _df_rec_agg["_sk"].map(sampler_sort_key)
        _df_rec_agg = _df_rec_agg.sort_values("_sort")
        fig_rec = go.Figure()
        for _, _r in _df_rec_agg.iterrows():
            _col = SAMPLER_COLORS.get(_r["_sk"], "#888")
            fig_rec.add_trace(go.Bar(
                x=[_r["Sampler"]], y=[_r["恢復率 %"]],
                marker_color=_col,
                text=[f"{_r['恢復率 %']:.0f}%<br>({int(_r['總恢復'])}/{int(_r['總恢復']+_r['總未恢復'])})"],
                textposition="outside",
                showlegend=False,
                hovertemplate=f"<b>{_r['Sampler']}</b><br>恢復率: {_r['恢復率 %']}%<br>恢復: {int(_r['總恢復'])} / {int(_r['總恢復']+_r['總未恢復'])}<extra></extra>",
            ))
        fig_rec.update_layout(
            height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0, 115], ticksuffix="%", gridcolor="#eee"),
            title="各 Sampler 在災難失敗後的即時恢復率（下一 trial score > 0）",
        )
        st.plotly_chart(fig_rec, width="stretch")
        if not _df_rec_agg.empty:
            _best_rec = _df_rec_agg.loc[_df_rec_agg["恢復率 %"].idxmax()]
            st.caption(
                f"恢復率最高：**{_best_rec['Sampler']}**（{_best_rec['恢復率 %']:.0f}%）"
                "—— 失敗後能立刻切回安全配置，顯示良好的即時調適能力。"
            )
    else:
        st.caption("（目前資料中無發生 catastrophic failure 的 run）")

    # ── ② Sample Complexity CDF（6️⃣） ────────────────────────────────────────
    st.markdown("### 6️⃣ Sample Complexity CDF")
    st.caption(
        "各 sampler 在 N 個 trial 內達到特定 score 門檻的累積比例（CDF）。"
        "曲線越陡越左 → 越少 trial 即可達到目標品質。"
        "每條線代表一個 sampler，顏色統一；每個子圖對應一個門檻。"
    )
    _cdf_thresholds = [2.0, 3.0, 4.0, 4.5]
    _max_t = max((len(r["best_so_far"]) for r in all_exp_rows), default=1)
    _sampler_bsf_map: dict[str, list] = {}
    for _row in all_exp_rows:
        _s = _row["exp"]["sampler"]
        _sampler_bsf_map.setdefault(_s, []).append(_row["best_so_far"])

    fig_cdf = go.Figure()
    for _ti, _thr in enumerate(_cdf_thresholds):
        for _s, _bsf_list in sorted(_sampler_bsf_map.items(), key=lambda kv: sampler_sort_key(kv[0])):
            _n_exp = len(_bsf_list)
            _cdf_y = []
            for _x in range(1, _max_t + 1):
                _hit = sum(
                    1 for _bsf in _bsf_list
                    if len(_bsf) >= _x and _bsf[_x - 1] >= _thr
                )
                _cdf_y.append(_hit / _n_exp * 100)
            _color = _SAMPLER_CONV_COLORS.get(_s, "#888")
            _label = SAMPLER_LABELS.get(_s, _s)
            fig_cdf.add_trace(go.Scatter(
                x=list(range(1, _max_t + 1)),
                y=_cdf_y,
                mode="lines",
                name=f"{_label}",
                line=dict(color=_color, width=1.8),
                opacity=0.85,
                legendgroup=_s,
                showlegend=(_ti == 0),
                visible=(_ti == 0),
                hovertemplate=f"<b>{_label}</b> (≥{_thr})<br>Trial: %{{x}}<br>Hit: %{{y:.0f}}%",
            ))
    # 用 updatemenus 切換門檻
    _n_samplers = len(_sampler_bsf_map)
    _cdf_buttons = []
    for _ti, _thr in enumerate(_cdf_thresholds):
        _vis = [False] * (len(_cdf_thresholds) * _n_samplers)
        for _si in range(_n_samplers):
            _vis[_ti * _n_samplers + _si] = True
        _cdf_buttons.append(dict(
            label=f"Score ≥ {_thr}",
            method="update",
            args=[{"visible": _vis}, {"title": f"Sample Complexity CDF（Score ≥ {_thr}）"}],
        ))
    fig_cdf.update_layout(
        height=400, title=f"Sample Complexity CDF（Score ≥ {_cdf_thresholds[0]}）",
        xaxis_title="Trial #", yaxis_title="% 實驗達到門檻",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend_title="Sampler",
        updatemenus=[dict(
            type="buttons", direction="right", showactive=True,
            x=0.0, y=1.15, buttons=_cdf_buttons,
        )],
    )
    fig_cdf.update_yaxes(gridcolor="#eee", range=[0, 105])
    fig_cdf.update_xaxes(gridcolor="#eee")
    st.plotly_chart(fig_cdf, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4：演算法說明
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("ℹ️ 系統設計說明")

    sec = st.radio(
        "選擇章節",
        ["📐 評分公式", "🗂️ 搜尋空間", "🤖 Optuna Sampler", "🧠 LLM Agent 與記憶機制", "📝 LLM Prompt 設計", "⚖️ Weight 敏感度分析"],
        horizontal=True,
    )

    # ── 評分公式 ──────────────────────────────────────────────────────────────
    if sec == "📐 評分公式":
        st.subheader("📐 評分公式（Log-Scale Weighted Score）")
        st.markdown("""
兩套 Tuner 共用相同的評分公式，確保跨方法比較的一致性。

#### 基礎分數
```
score = 1.0
      + w_acc  × log( accuracy    / baseline_accuracy    )
      + w_lat  × log( baseline_latency   / latency   )
      + w_vram × log( baseline_vram      / vram       )
      + w_emit × log( baseline_emissions / emissions  )
```

| 項目 | 方向 | 說明 |
|---|---|---|
| `accuracy` | ↑ 越高越好 | 壓縮後 / 基線，log > 0 代表超越基線 |
| `latency` | ↓ 越低越好 | 基線 / 壓縮後，延遲越短分數越高 |
| `vram` | ↓ 越低越好 | 基線 / 壓縮後，顯存越省分數越高 |
| `emissions` | ↓ 越低越好 | 基線 / 壓縮後，碳排越少分數越高 |

#### 準確率懲罰（Accuracy Penalty）
```
threshold = baseline_accuracy − pen_t        # 預設 0.718 − 0.15 = 0.568
penalty   = pen_a × max(0, threshold − accuracy)

Final Score = score − penalty
```

| 參數 | 預設值 | 意義 |
|---|---|---|
| `pen_t` | `0.15` | 容忍準確率最多掉 15 個百分點（絕對值） |
| `pen_a` | `10.0` | 每超出 1% 門檻，扣 10 分（放大倍率） |

#### 設計直覺
- **log 邊際遞減**：從 6 GB 壓到 3 GB（節省 50%）的增益，遠大於從 3 GB 壓到 1.5 GB（同樣 50%）
- **acc 權重 3×**：本實驗以 acc=3.0, lat=vram=emit=1.0，確保準確率為最優先目標
- **penalty 斷崖**：accuracy 一旦跌破門檻，penalty 快速放大，防止模型退化到不可用
""")

    # ── 搜尋空間 ──────────────────────────────────────────────────────────────
    elif sec == "🗂️ 搜尋空間":
        st.subheader("🗂️ 搜尋空間定義")
        st.caption(
            "兩套 Tuner（Systematic_Tuner Optuna 與 Global_Tuner_memory LLM）共享相同的搜尋空間定義，"
            "保證比較公平。以下為每種壓縮模式的完整參數範圍。"
        )

        st.markdown("#### 🔀 壓縮模式清單（8 種）")
        modes_df = pd.DataFrame([
            {"Mode Key":          "asvd_only",           "類型": "結構壓縮",   "說明": "低秩奇異值分解，不改變精度"},
            {"Mode Key":          "gptq",                "類型": "量化",       "說明": "GPTQ 後訓練量化，Hessian 最小化誤差"},
            {"Mode Key":          "awq",                 "類型": "量化",       "說明": "Activation-aware 量化，固定 4-bit"},
            {"Mode Key":          "qqq",                 "類型": "量化",       "說明": "W4A8 混合精度量化"},
            {"Mode Key":          "bnb",                 "類型": "量化",       "說明": "BitsAndBytes NF4/FP4，無需校準資料"},
            {"Mode Key":          "sparse_unstructured", "類型": "稀疏化",     "說明": "任意非結構剪枝"},
            {"Mode Key":          "sparse_structured",   "類型": "稀疏化",     "說明": "2:4 / 4:8 半結構，支援 NVIDIA 硬體加速"},
            {"Mode Key":          "hybrid_asvd_bnb",     "類型": "混合",       "說明": "ASVD 低秩分解 + BNB 量化疊加"},
        ])
        st.dataframe(modes_df, width="stretch", hide_index=True)

        st.markdown("---")

        def mode_card(title, body, color="#3498db"):
            st.markdown(
                f'<div style="border-left:4px solid {color}; padding:4px 14px; '
                f'margin-bottom:4px; background:{color}11; border-radius:0 6px 6px 0;">'
                f'<b style="color:{color}">{title}</b></div>',
                unsafe_allow_html=True,
            )
            st.markdown(body)

        mode_card("① ASVD（Activation-aware SVD）", """
對每個線性層做低秩分解 $W \\approx A \\times B$，根據 activation 重要性加權奇異值保留比例。

| 參數 | 取樣方式 | 合法值 / 範圍 | 說明 |
|---|---|---|---|
| `alpha` | categorical | `[0.3, 0.4, 0.5, 0.6, 0.7]` | Activation sensitivity 平衡係數；設為 categorical 以命中預先計算的 sensitivity cache |
| `param_ratio_target` | float (linear) | `[0.70, 0.99]` | 目標保留參數比例（0.9 = 保留 90% 參數量） |
| `scaling_method` | categorical | `["abs_mean", "abs_max", "fisher"]` | 奇異值重要性評估；`fisher` 使用二階資訊，通常最佳 |
""", "#3498db")

        mode_card("② GPTQ（Post-Training Quantization）", """
利用 Hessian 二階資訊逐層最小化量化誤差，是目前最主流的 PTQ 方法。

| 參數 | 取樣方式 | 合法值 / 範圍 | 說明 |
|---|---|---|---|
| `quant_bits` | categorical | `[3, 4, 8]` | 4-bit 為最佳平衡；3-bit 極致壓縮；8-bit 幾乎無損 |
| `quant_group_size` | categorical | `[16, 32, 64, 128, 256]` | 越小精度越高但模型越大；須為 2 的冪次 |
| `quant_format` | categorical | `["gptq", "gptq_v2"]` | `gptq_v2` 修正數值溢出，低 bit 更穩定 |
| `damp_percent` | float (log scale) | `[0.001, 0.1]` | Hessian 阻尼，使用 log scale 均勻探索 |

> `mse` 參數已移除：實驗觀察幾乎都使用 0.0，加入搜尋空間只會浪費 trial。
""", "#e74c3c")

        mode_card("③ AWQ（Activation-aware Weight Quantization）", """
根據 activation 重要性對權重做縮放後再量化，保護重要通道不被量化誤差破壞。精度固定 4-bit。

| 參數 | 取樣方式 | 合法值 / 範圍 | 說明 |
|---|---|---|---|
| `quant_group_size` | categorical | `[16, 32, 64, 128]` | 分組大小，AWQ 通常在 128 附近最佳 |

> `quant_bits` 固定為 4，因為 AWQ 的 activation-aware scaling 專為 4-bit 設計。
""", "#9b59b6")

        mode_card("④ QQQ（Quattuor-bit Quantization）", """
GPTQ 的變體，針對 W4A8（4-bit 權重 + 8-bit activation）推理路徑最佳化。適合支援 INT8 矩陣乘法的硬體。

| 參數 | 取樣方式 | 合法值 / 範圍 | 說明 |
|---|---|---|---|
| `quant_group_size` | categorical | `[-1, 128]` | `-1` 整個矩陣共用一組 scale；`128` 為分組量化 |
| `damp_percent` | float (log scale) | `[0.0005, 0.05]` | Hessian 阻尼，範圍比 GPTQ 小（QQQ 對阻尼更敏感）|

> `quant_bits` 固定 4，`quant_format` 固定 `qqq`。
""", "#1abc9c")

        mode_card("⑤ BNB（BitsAndBytes）", """
bitsandbytes 函式庫提供的動態量化，無需校準資料集，是使用門檻最低的量化方法。

| 參數 | 取樣方式 | 合法值 / 範圍 | 說明 |
|---|---|---|---|
| `quant_bits` | categorical | `[4, 8]` | 4-bit（NF4）激進壓縮；8-bit 幾乎無損 |
| `use_double_quant` | categorical | `[True, False]` | 對 scale factor 再量化，額外節省約 0.4 bit/param；僅 bits=4 有效 |

> `quant_type` 固定 `nf4`：nf4 幾乎必勝 fp4，加入搜尋空間只是浪費 trial。
""", "#f39c12")

        mode_card("⑥ Sparse Unstructured（非結構稀疏化）", """
使用 SparseGPT 演算法，在 Hessian 引導下移除重要性最低的個別權重，不受位置限制。

| 參數 | 取樣方式 | 合法值 / 範圍 | 說明 |
|---|---|---|---|
| `sparsity_ratio` | float (linear) | `[0.3, 0.7]` | 剪枝比例：0.5 代表移除 50% 的權重。低於 0.3 效果有限；高於 0.7 通常 accuracy 崩潰 |
""", "#27ae60")

        mode_card("⑦ Sparse Structured（半結構稀疏化）", """
NVIDIA 支援的 N:M 稀疏格式，可在 Ampere（A100+）GPU 上獲得硬體加速推理。

| 參數 | 取樣方式 | 合法值 / 範圍 | 說明 |
|---|---|---|---|
| `sparsity_structure` | categorical | `["2:4", "4:8"]` | `2:4` = 每 4 個保留 2 個（50% 稀疏）；`4:8` = 每 8 個保留 4 個（粒度更細）|
""", "#16a085")

        mode_card("⑧ Hybrid ASVD + BNB（混合壓縮）", """
先做 ASVD 低秩分解降低參數量，再用 BNB 量化降低 bit-width，兩步疊加最大化壓縮比。

| 參數 | 來源 | 取樣方式 | 合法值 / 範圍 |
|---|---|---|---|
| `alpha` | ASVD | categorical | `[0.3, 0.4, 0.5, 0.6, 0.7]` |
| `param_ratio_target` | ASVD | float (linear) | `[0.70, 0.99]` |
| `scaling_method` | ASVD | categorical | `["abs_mean", "abs_max", "fisher"]` |
| `quant_bits` | BNB | categorical | `[4, 8]` |
| `use_double_quant` | BNB | categorical | `[True, False]`（僅 bits=4 有效）|
""", "#8e44ad")

    # ── Optuna Sampler ─────────────────────────────────────────────────────────
    elif sec == "🤖 Optuna Sampler":
        st.subheader("🤖 Optuna Sampler 說明（Systematic Tuner）")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
**TPE（Tree-structured Parzen Estimator）**

貝葉斯最佳化的變體，是 Optuna 的預設 sampler。

**運作機制：**
1. 前 `n_startup_trials`（預設 10）輪純隨機探索
2. 之後將過去結果分為「好的」（top 25%）與「差的」（bottom 75%）
3. 分別對兩組建立 Parzen 密度估計（kernel density）
4. 選擇最大化 `l(x) / g(x)` 的超參數（Expected Improvement）

**設定重點：**
- `multivariate=True`：考慮參數間的聯合相關性（更準確）
- `n_startup_trials=10`：前 10 輪隨機，確保足夠先驗

**優缺點：**
- ✅ 收斂快、對高維空間有效
- ✅ 適合搜尋空間有明顯局部最優
- ❌ 可能陷入局部最優，多樣性較低
""")
        with col2:
            st.markdown("""
**NSGA-II（Non-dominated Sorting GA）**

多目標遺傳演算法，Optuna 的 `NSGAIISampler`。

**運作機制：**
1. 維護大小為 `population_size` 的族群
2. 用 Pareto 支配關係對解做非支配排序
3. 同一 rank 內以 crowding distance 保持多樣性
4. 交叉（crossover）+突變（mutation）產生下一代

**設定重點：**
- `population_size=10`（本實驗預設）
- 建議 `max_iterations ≥ 2 × population_size`

**優缺點：**
- ✅ 多樣性好，能探索更廣的解空間
- ✅ 天然適合多目標優化
- ❌ 收斂較慢，需要較多 trial 才能看出優勢
""")
        with col3:
            st.markdown("""
**Random Search（基線）**

Optuna 的 `RandomSampler`，完全隨機取樣。

**運作機制：**
- 從每個參數的分布中獨立均勻取樣
- 無任何學習或記憶機制

**用途：**
- 作為比較基準（lower bound）
- 驗證 TPE / NSGA-II 的改善是否顯著

**優缺點：**
- ✅ 無偏探索，不會陷入局部最優
- ✅ 在低維離散空間有時與 TPE 相當
- ❌ 高維或連續空間效率低
""")

    # ── LLM Agent ─────────────────────────────────────────────────────────────
    elif sec == "🧠 LLM Agent 與記憶機制":
        st.subheader("🧠 Global_Tuner_memory：LLM Agent 與記憶機制")
        st.markdown("""
**Global_Tuner_memory** 使用 GPT-4o 作為決策核心，每輪 iteration 依據歷史結果與 Pareto Frontier
分析並輸出下一個壓縮配置。與 Optuna 的差異在於：**LLM 能理解語意、做類比推理、並主動解釋決策**。

#### 整體流程
```
Iteration i:
  1. 準備 history context（依 memory_type 決定格式）
  2. 準備 Pareto Frontier context
  3. 呼叫 GPT-4o → 輸出 JSON 壓縮配置
  4. 執行壓縮（ASVD / Quantization / Sparse）
  5. 評估 → 計算 score → 更新 Pareto
  6. 回到步驟 1
```
""")

        st.markdown("---")
        st.markdown("#### 四種記憶模式（Memory Types）")

        mem_cols = st.columns(4)
        mem_info = [
            ("full", "完整歷史", "#3498db",
             "將所有 trial 歷史完整傳入 prompt。\n\n"
             "**優點**：資訊最完整\n\n"
             "**缺點**：隨 iteration 增加，token 暴增（「lost in the middle」問題），"
             "LLM 可能忽略早期重要發現"),
            ("window", "滑動視窗", "#2ecc71",
             "只傳入**最近 5 筆** trial 的結果。\n\n"
             "**優點**：token 固定、穩定\n\n"
             "**缺點**：災難性遺忘（catastrophic forgetting）——"
             "第 1 輪的最佳配置到第 15 輪時已被遺忘"),
            ("summary", "LLM 摘要", "#e67e22",
             "每 5 個 trial 呼叫 **GPT-4o-mini** 更新一份「知識摘要」，\n"
             "摘要結構：\n"
             "- **Correlations**：參數 → 指標關係\n"
             "- **Safe Zones**：安全參數範圍\n"
             "- **Danger Zones**：導致失敗的配置\n"
             "- **Recommended Direction**：下一步探索方向\n\n"
             "**優點**：壓縮記憶、保留洞見\n\n"
             "**缺點**：摘要可能遺失細節；需額外 API 呼叫"),
            ("tool", "Agentic 工具", "#9b59b6",
             "主 prompt 只顯示最近 3 筆，但 LLM 可呼叫 **`retrieve_trials` 工具**"
             "按方法查詢特定歷史。\n\n"
             "工具參數：`query`（方法名稱）、`reason`（查詢動機）\n\n"
             "**優點**：最像人類分析師，按需取用資訊\n\n"
             "**缺點**：實作複雜；最多 3 輪工具呼叫，超過強制輸出"),
        ]
        for col, (key, name, color, desc) in zip(mem_cols, mem_info):
            with col:
                st.markdown(
                    f'<div style="background:{color}22; border-left:4px solid {color}; '
                    f'padding:10px; border-radius:4px; margin-bottom:8px;">'
                    f'<b style="color:{color}">{key}</b> — {name}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(desc)

        st.markdown("---")
        st.markdown("#### History 格式（LLM 看到的每筆 trial 描述）")
        st.code(
            "- [Iter 3]: Score=4.6288 "
            "(Acc: 0.6760 (-0.0420 pp, -5.85%), "
            "Lat: 484.7954s (-65.93%), "
            "VRAM: 2.3205GB (-62.53%), "
            "Emit: 0.016257kg (-82.65%))\n"
            "    Config: {\"mode\": \"gptq\", \"quant\": {\"bits\": 4, \"group_size\": 128, ...}}",
            language="text"
        )
        st.caption("每筆包含：分數、四個指標的絕對值與相對 baseline 的變化量，幫助 LLM 推理「壓縮了多少、代價多少」")

    # ── LLM Prompt ────────────────────────────────────────────────────────────
    elif sec == "📝 LLM Prompt 設計":
        st.subheader("📝 LLM Prompt 完整設計")
        st.caption(
            "以下為 Global_Tuner_memory 每輪呼叫 GPT-4o 時的完整 prompt 結構。"
            "括號內為實際執行時填入的動態內容。"
        )

        st.markdown("#### Prompt 結構總覽")
        st.markdown("""
| 區塊 | 內容 | 目的 |
|---|---|---|
| **角色定義** | `You are an LLM compression optimization agent.` | 設定 LLM 角色 |
| **任務目標** | 模型、任務、當前 iteration | 確立搜尋目標 |
| **評分公式** | log-scale 公式、penalty 設定 | 讓 LLM 理解優化目標 |
| **搜尋空間** | 8 種 mode 的完整 JSON schema + 參數說明 | 限制輸出格式，避免幻覺 |
| **搜尋狀態** | 已嘗試 / 未嘗試的 mode 統計 | 引導 LLM 填補探索空白 |
| **拒絕清單** | 最近重複嘗試的配置 | 強制多樣性 |
| **歷史記憶** | 依 memory_type 決定格式 | 提供學習依據 |
| **Pareto Frontier** | 當前非支配解集合 | 引導 LLM 找 Pareto 空白 |
| **策略指引** | 早期廣泛探索、晚期 hybrid | 防止過早收斂 |
""")

        with st.expander("📄 查看完整 Prompt 原文（window mode 範例）", expanded=False):
            st.code("""You are an LLM compression optimization agent. Choose the best compression strategy for:
Model: meta-llama/Llama-3.2-3B-Instruct | Task: gsm8k | Iteration: 7/20

GOAL: Maximize Final Score.
1. Base Score = 1.0 + 3.0*ln(Acc/Base_acc) + 1.0*ln(Base_lat/Lat) + 1.0*ln(Base_vram/VRAM) + 1.0*ln(Base_emit/Emit)
   (explain: Prioritize positive relative % changes in Lat, VRAM, and Emit,
    while minimizing negative absolute drops (pp) in Acc.)
2. PENALTY: A massive penalty (multiplier 10.0) is applied ONLY if the Accuracy drop exceeds 0.15
   (i.e., the "pp" diff is more negative than -0.15).
3. Final Score = Base Score - Penalty

=== AVAILABLE MODES & OUTPUT FORMATS ===
Strictly output ONLY valid JSON matching one of these structures.
"reasoning" MUST follow this structure: "A detailed explanation of why this mode fills a gap
in current coverage, followed by a comprehensive analysis of the expected trade-offs and risks."

[MODE: asvd_only]
{"reasoning": "...", "mode": "asvd_only",
  "alpha": 0.5,               // categorical [0.3, 0.4, 0.5, 0.6, 0.7]
  "param_ratio_target": 0.90, // linear [0.70 - 0.99]
  "scaling_method": "fisher"  // categorical ["abs_mean", "abs_max", "fisher"]
}

[MODE: gptq]
{"reasoning": "...", "mode": "gptq",
  "quant_bits": 4,             // categorical [3, 4, 8]
  "quant_group_size": 128,     // categorical [16, 32, 64, 128, 256]
  "quant_format": "gptq",      // categorical ["gptq", "gptq_v2"]
  "damp_percent": 0.05         // log [0.001 - 0.1]
}
... (awq / qqq / bnb / sparse_unstructured / sparse_structured / hybrid_asvd_bnb)

=== CURRENT STATUS ===
Modes tried so far: gptq (2x), bnb (1x), asvd_only (1x), sparse_unstructured (1x), awq (1x)
Modes NOT yet tried: qqq, sparse_structured, hybrid_asvd_bnb

Trial Context (window mode):
- [Iter 6]: Score=4.5213 (Acc: 0.6740 (-0.0440 pp, -6.13%), ...)
    Config: {"mode": "bnb", "quant": {"bits": 4, "double_quant": true}}
... (最近 5 筆)

Pareto Frontier (best trade-offs found):
- [Iter 1]: Score=4.6288 (Acc: 0.6760 ...) Config: {...}
... (當前非支配解)

=== STRATEGY ===
- Early iterations: try each mode independently to understand isolated impact.
- For quant_only: start with GPTQ 4-bit, then explore variations.
- Later iterations: combine methods in hybrid mode.
- NEVER repeat identical configs. Use Pareto frontier to find unexplored regions.

Output ONLY the JSON for your chosen mode. No extra fields, no prose.
""", language="text")

        st.markdown("---")
        st.markdown("#### Summary 模式的二次 Prompt（每 5 輪更新知識摘要）")
        st.caption("使用較便宜的 GPT-4o-mini 呼叫，將歷史壓縮成結構化知識。")
        with st.expander("📄 查看 Summary Prompt", expanded=False):
            st.code("""You are an AI assistant maintaining an evolving knowledge base
for a model compression agent.

=== CURRENT KNOWLEDGE SUMMARY (May contain outdated or incorrect early assumptions) ===
{current_summary}

=== FULL EXPERIMENTAL HISTORY (Trials 1 through Current) ===
{new_batch_of_5_trials}

=== INSTRUCTIONS ===
Rewrite and update the current knowledge summary based on the new trial results.
🛑 CRITICAL CONSTRAINT: DO NOT specify exact parameter combinations to run next.
                        Define the "rules of the game".

Output the summary in EXACTLY this structure:
## Correlations
(parameter → metric relationships)

## Safe Zones
(known safe parameter ranges)

## Danger Zones
(configurations that caused failures or heavy penalties)

## Recommended Direction
(next exploration focus, NO specific numbers)

Output ONLY the newly updated summary text. Do not include conversational filler.
""", language="text")

        st.markdown("---")
        st.markdown("#### Tool Mode 的 `retrieve_trials` 工具定義")
        with st.expander("📄 查看 Tool Schema", expanded=False):
            st.code("""{
  "type": "function",
  "function": {
    "name": "retrieve_trials",
    "description": "Fetch full history for a specific method.
      STRICT LIMITATION: Do NOT query methods you haven't tried yet.
      Only use this to investigate variations or failures of methods you HAVE already tried.",
    "parameters": {
      "query": {
        "type": "string",
        "enum": ["asvd", "gptq", "awq", "qqq", "bnb", "sparse", "hybrid"],
        "description": "The specific compression method to search for."
      },
      "reason": {
        "type": "string",
        "description": "Why are you querying this method? What hypothesis are you testing?"
      }
    }
  }
}

Rules:
- Max 2 tool calls per iteration
- Max 3 turns total (Turn 3 forces final JSON answer)
- Tool results are prepended with [Your query goal: {reason}] for context continuity
""", language="json")

    # ── Weight 敏感度分析 ──────────────────────────────────────────────────────
    elif sec == "⚖️ Weight 敏感度分析":
        st.subheader("⚖️ Weight 敏感度分析")
        st.caption("調整權重與 penalty 參數，即時看各 trial 排名如何改變")

        exp_names4 = [f"[{e['sampler']}] {e['dir_name']}" for e in filtered]
        if exp_names4:
            sel4 = st.selectbox("選擇實驗", exp_names4, key="tab4_sel")
            exp4 = filtered[exp_names4.index(sel4)]
            b4   = exp4["config"].get("baseline", baseline)

            c1, c2, c3, c4 = st.columns(4)
            w_acc  = c1.slider("w_acc",  0.0, 5.0, 3.0, 0.5)
            w_lat  = c2.slider("w_lat",  0.0, 5.0, 1.0, 0.5)
            w_vram = c3.slider("w_vram", 0.0, 5.0, 1.0, 0.5)
            w_emit = c4.slider("w_emit", 0.0, 5.0, 1.0, 0.5)
            pen_t_s = st.slider("pen_t（準確率容忍門檻）", 0.0, 0.5, 0.15, 0.05)
            pen_a_s = st.slider("pen_a（懲罰放大倍率）",   0.0, 20.0, 10.0, 1.0)

            new_weights = {"acc": w_acc, "lat": w_lat, "vram": w_vram, "emit": w_emit}
            rescored = rescore(exp4["trials"], b4, new_weights, pen_t_s, pen_a_s)
            if rescored:
                df_rs = pd.DataFrame(rescored)[["Rank", "Trial", "設定", "Accuracy", "Score"]]
                styled_rs = df_rs.style.background_gradient(subset=["Score"], cmap="RdYlGn")
                styled_rs = styled_rs.format({"Score": "{:.4f}", "Accuracy": "{:.4f}"})
                styled_rs = styled_rs.set_table_styles([
                    {"selector": "th", "props": [
                        ("background-color", "#2c3e50"), ("color", "white"),
                        ("font-size", "12px"), ("text-align", "center"),
                    ]},
                    {"selector": "td", "props": [
                        ("font-size", "11px"), ("padding", "4px 8px"), ("text-align", "center"),
                    ]},
                ])
                st.write(styled_rs.to_html(), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 報告：LLM 優勢分析報告
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.header("📋 LLM-Guided 量化搜尋優勢分析報告")
    st.caption("基於 final_results/ 中 21 組實驗（7 種方法 × 3 次執行 × 20 trial = 420 次實際測量）的完整數據分析，模型：Llama-3.2-3B-Instruct，任務：GSM8K 數學推理。")

    # ── 0. Baseline ────────────────────────────────────────────────────────────
    st.subheader("0. Baseline（未壓縮原始模型）")
    baseline_report = load_baseline()
    if baseline_report:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{baseline_report.get('accuracy', 0):.1%}")
        c2.metric("Latency", f"{baseline_report.get('latency', 0):.1f} s")
        c3.metric("VRAM", f"{baseline_report.get('vram', 0):.2f} GB")
        c4.metric("Emissions", f"{baseline_report.get('emissions', 0):.4f}")

    st.markdown("""
所有壓縮方法的效益均以此 baseline 為參照基準計算相對變化。
分析涵蓋 7 種搜尋策略：**LLM-Full**、**LLM-Summary**、**LLM-Window**、**Tool Agent**、**TPE**、**NSGA-II**、**Random Search**，每種各執行 3 次，共 21 組實驗、420 次量化-評估測量。
""")

    st.divider()

    # ── 1. 整體數值比較 ─────────────────────────────────────────────────────────
    st.subheader("1. 整體數值比較")

    report_groups = {
        "LLM-Full":    sorted(
                    p.name for p in BASE_DIR.glob("exp_Llama-3.2-3B-Instruct_gsm8k_full_*")
                    if (p / "optimization_results.json").exists()
                ),
        "LLM-Summary": sorted(
                    p.name for p in BASE_DIR.glob("exp_Llama-3.2-3B-Instruct_gsm8k_summary_*")
                    if (p / "optimization_results.json").exists()
                ),
        "LLM-Window":  sorted(
                    p.name for p in BASE_DIR.glob("exp_Llama-3.2-3B-Instruct_gsm8k_window_*")
                    if (p / "optimization_results.json").exists()
                ),
        "Tool Agent":  sorted(
                    p.name for p in BASE_DIR.glob("exp_Llama-3.2-3B-Instruct_gsm8k_tool_*")
                    if (p / "optimization_results.json").exists()
                ),
        "TPE":         sorted(
                    p.name for p in BASE_DIR.glob("tpe_Llama-3.2-3B-Instruct_gsm8k_*")
                    if (p / "optimization_results.json").exists()
                ),
        "NSGA-II":     sorted(
                    p.name for p in BASE_DIR.glob("nsga2_Llama-3.2-3B-Instruct_gsm8k_*")
                    if (p / "optimization_results.json").exists()
                ),
        "Random":      sorted(
                    p.name for p in BASE_DIR.glob("random_Llama-3.2-3B-Instruct_gsm8k_*")
                    if (p / "optimization_results.json").exists()
                ),
    }

    @st.cache_data
    def compute_report_stats(groups: dict) -> pd.DataFrame:
        rows = []
        for group_name, folders in groups.items():
            all_scores, all_accs, best_scores = [], [], []
            pos_count, total = 0, 0
            catastrophic = 0
            convergence_iters = []  # 收集每個 folder 的 convergence 迭代數
            for folder in folders:
                path = BASE_DIR / folder / "optimization_results.json"
                if not path.exists():
                    continue
                with open(path) as f:
                    data = json.load(f)
                scores = [d["metrics"].get("score", 0) for d in data if d.get("metrics")]
                accs   = [d["metrics"].get("accuracy", 0) for d in data if d.get("metrics")]
                all_scores.extend(scores)
                all_accs.extend(accs)
                best_scores.append(max(scores) if scores else 0)
                pos_count   += sum(1 for s in scores if s > 0)
                catastrophic += sum(1 for s in scores if s < -10)
                total += len(scores)

                # 計算收斂：達到最終 best score 所需的迭代數
                if scores:
                    final_best = max(scores)
                    for i, s in enumerate(scores, 1):
                        if s >= final_best:
                            convergence_iters.append(i)
                            break
                    else:
                        # 如果沒達到（應該不會），記錄總迭代數
                        convergence_iters.append(len(scores))

            avg_best = statistics.mean(best_scores) if best_scores else 0
            avg_convergence = statistics.mean(convergence_iters) if convergence_iters else 0
            avg_all  = statistics.mean(all_scores) if all_scores else 0
            rows.append({
                "方法": group_name,
                "平均最佳 Score": round(avg_best, 3),
                "平均收斂迭代": round(avg_convergence, 1),
                "平均所有 Score": round(avg_all, 3),
                "正向 Trial 率": f"{pos_count/total*100:.1f}% ({pos_count}/{total})",
                "正向率_num": pos_count / total * 100,
                "災難性失敗率": f"{catastrophic/total*100:.1f}% ({catastrophic}/{total})",
                "災難率_num": catastrophic / total * 100,
                "最高 Accuracy": round(max(all_accs), 3) if all_accs else 0,
            })
        return pd.DataFrame(rows)

    df_report = compute_report_stats(report_groups)

    col_display = ["方法", "平均最佳 Score", "平均收斂迭代", "平均所有 Score",
                   "正向 Trial 率", "災難性失敗率", "最高 Accuracy"]

    col_display_en = ["Method", "Avg Best Score", "Avg Convergence Iter", "Avg All Scores",
                      "Positive Trial Rate", "Catastrophic Failure Rate", "Peak Accuracy"]

    df_report_en = df_report[col_display].rename(columns=dict(zip(col_display, col_display_en)))

    table_style = [
        {"selector": "th", "props": [("background-color", "#1a252f"), ("color", "white"),
                                      ("font-size", "13px"), ("text-align", "center"), ("padding", "6px 10px")]},
        {"selector": "td", "props": [("font-size", "12px"), ("padding", "5px 10px"), ("text-align", "center")]},
    ]

    tab_zh, tab_en = st.tabs(["中文", "English"])

    with tab_zh:
        styled_report = (
            df_report[col_display]
            .style
            .background_gradient(subset=["平均最佳 Score"], cmap="RdYlGn")
            .background_gradient(subset=["平均所有 Score"], cmap="RdYlGn")
            .format({
                "平均最佳 Score": "{:.3f}",
                "平均收斂迭代": "{:.1f}",
                "平均所有 Score": "{:.3f}",
                "最高 Accuracy": "{:.3f}",
            })
            .set_table_styles(table_style)
        )
        st.write(styled_report.to_html(), unsafe_allow_html=True)
        st.markdown("""
> **解讀重點**：LLM 方法在「正向 Trial 率」與「災難性失敗率」上優勢最明顯。
> 平均最佳 Score 的差距雖然不大（約 0.2），但當計算預算有限（≤20 trials）時，「每一個 trial 是否有效」才是關鍵。
""")

    with tab_en:
        styled_report_en = (
            df_report_en
            .style
            .background_gradient(subset=["Avg Best Score"], cmap="RdYlGn")
            .background_gradient(subset=["Avg All Scores"], cmap="RdYlGn")
            .format({
                "Avg Best Score": "{:.3f}",
                "Avg Convergence Iter": "{:.1f}",
                "Avg All Scores": "{:.3f}",
                "Peak Accuracy": "{:.3f}",
            })
            .set_table_styles(table_style)
        )
        st.write(styled_report_en.to_html(), unsafe_allow_html=True)
        st.markdown("""
> **Key Takeaway**: LLM methods show the clearest advantage in "Positive Trial Rate" and "Catastrophic Failure Rate".
> The gap in Avg Best Score is small (~0.2), but under limited compute budgets (≤20 trials), **whether each trial is useful** is what matters most.
""")

    st.divider()

    # ── 2. 核心優勢一：災難性失敗 ──────────────────────────────────────────────
    st.subheader("2. 核心優勢一：大幅減少災難性失敗")

    st.markdown("""
**災難性失敗**定義為 score < −10，通常意味著壓縮後模型 accuracy 崩潰到 **2% 以下**（等同模型完全失效）。
這些失敗不只浪費一次 trial 的 GPU 時間，還可能讓統計優化器的下一步決策被這個極端值拉偏。
""")

    fig_catastrophic = go.Figure()
    colors_cat = {
        "LLM-Full":    "#17becf",
        "LLM-Summary": "#2ecc71",
        "LLM-Window":  "#27ae60",
        "Tool Agent":  "#e67e22",
        "TPE":         "#f39c12",
        "NSGA-II":     "#e74c3c",
        "Random":      "#c0392b",
    }
    for _, row in df_report.iterrows():
        fig_catastrophic.add_trace(go.Bar(
            x=[row["方法"]],
            y=[row["災難率_num"]],
            name=row["方法"],
            marker_color=colors_cat.get(row["方法"], "#95a5a6"),
            text=[f"{row['災難率_num']:.1f}%"],
            textposition="outside",
        ))
    fig_catastrophic.update_layout(
        title="各方法災難性失敗率（score < −10）",
        yaxis_title="失敗率 (%)",
        xaxis_title="搜尋方法",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=350,
    )
    st.plotly_chart(fig_catastrophic, width="stretch")

    st.markdown("""
### 為什麼 LLM 能避免災難性失敗？

LLM 帶有**領域先驗知識**，在第一個 trial 選擇 GPTQ 4-bit 之前，它就已經「知道」哪些配置危險：

| 危險配置 | 實際 Accuracy | 為什麼 LLM 知道要避開 |
|---------|--------------|-------------------|
| `qqq 4-bit g128` | 0.8%～1.6% | LLM 描述 QQQ 尚未驗證，謹慎試驗 |
| `sparse 2:4 結構化` | 5%～8% | LLM 知道 2:4 稀疏對推理模型較激進 |
| `asvd ratio<0.8` | 1%～3% | LLM 說明激進 SVD 壓縮必然造成大量精度損失 |
| `hybrid ASVD+BNB` | 2%～17% | LLM 推斷兩種技術存在相容性問題 |

以下是四種 LLM 策略的實測災難性失敗率：
- **LLM-Full**：5.0%（擁有完整歷史記憶，從不遺忘危險配置）
- **LLM-Summary**：6.7%（摘要歷史偶有細節遺失，但整體仍安全）
- **LLM-Window**：8.3%（僅看最近 N 筆，遠期失敗記憶可能消退）
- **Tool Agent**：5.0%（工具呼叫架構仍依賴 LLM 推理，表現與 LLM-Full 相當）

對比之下，**統計方法**（TPE：25.0%，NSGA-II：31.7%，Random：30.0%）
沒有任何先驗知識，必須在實際踩雷後才能透過統計模型逐漸規避危險配置。
在 20 個 trial 的預算內，每多一次災難性失敗就少一次有效探索。
""")

    st.divider()

    # ── 3. 核心優勢二：更高正向 Trial 率 ──────────────────────────────────────
    st.subheader("3. 核心優勢二：更高的正向 Trial 率")

    fig_pos = go.Figure()
    for _, row in df_report.iterrows():
        fig_pos.add_trace(go.Bar(
            x=[row["方法"]],
            y=[row["正向率_num"]],
            name=row["方法"],
            marker_color=colors_cat.get(row["方法"], "#95a5a6"),
            text=[f"{row['正向率_num']:.1f}%"],
            textposition="outside",
        ))
    fig_pos.update_layout(
        title="各方法正向 Trial 率（score > 0）",
        yaxis_title="正向率 (%)",
        xaxis_title="搜尋方法",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=350,
    )
    st.plotly_chart(fig_pos, width="stretch")

    # 動態計算最高與最低正向率方法
    _pos_sorted = df_report.sort_values("正向率_num", ascending=False)
    _pos_best_name  = _pos_sorted.iloc[0]["方法"]
    _pos_best_val   = _pos_sorted.iloc[0]["正向率_num"]
    _pos_worst_name = _pos_sorted.iloc[-1]["方法"]
    _pos_worst_val  = _pos_sorted.iloc[-1]["正向率_num"]
    _pos_diff       = _pos_best_val - _pos_worst_val
    _pos_best_cnt   = round(_pos_best_val / 100 * 20, 1)
    _pos_worst_cnt  = round(_pos_worst_val / 100 * 20, 1)

    st.markdown(f"""
正向 Trial 率最高的方法為 **{_pos_best_name}**（**{_pos_best_val:.1f}%**），
最低為 **{_pos_worst_name}**（**{_pos_worst_val:.1f}%**），差距 **{_pos_diff:.1f} 個百分點**。

換句話說：在同樣 20 次試驗的預算下——
- {_pos_best_name} 平均有 **{_pos_best_cnt} 個**有效 trial（score > 0）
- {_pos_worst_name} 平均只有 **{_pos_worst_cnt} 個**

每一次有效 trial 都在推進對最佳壓縮配置的理解，無效 trial 純粹是資源浪費（GPU 時間 + 電力排放）。
LLM 記憶方法整體（Full / Summary / Window / Tool Agent）正向率均 ≥ 41.7%，
其中 LLM-Summary 與 LLM-Window 並列最高（55.0%）。
""")

    st.divider()

    # ── 4. 核心優勢三：快速收斂 ────────────────────────────────────────────────
    st.subheader("4. 核心優勢三：更快的收斂速度")

    @st.cache_data
    def compute_convergence(groups: dict) -> dict:
        result = {}
        for group_name, folders in groups.items():
            all_trajectories = []
            first_bests = []
            for folder in folders:
                path = BASE_DIR / folder / "optimization_results.json"
                if not path.exists():
                    continue
                with open(path) as f:
                    data = json.load(f)
                scores = [d["metrics"].get("score", 0) for d in data if d.get("metrics")]
                if not scores:
                    continue
                best = max(scores)
                first_bests.append(next(i + 1 for i, s in enumerate(scores) if s == best))
                running_best = []
                cur_best = float("-inf")
                for s in scores:
                    if s > cur_best:
                        cur_best = s
                    running_best.append(cur_best)
                all_trajectories.append(running_best)
            if all_trajectories:
                max_len = max(len(t) for t in all_trajectories)
                avg_traj = []
                for i in range(max_len):
                    vals = [t[i] for t in all_trajectories if i < len(t)]
                    avg_traj.append(statistics.mean(vals))
                result[group_name] = {
                    "trajectory": avg_traj,
                    "avg_first_best": statistics.mean(first_bests),
                    "first_bests": first_bests,
                }
        return result

    conv_data = compute_convergence(report_groups)

    fig_conv = go.Figure()
    colors_conv = {
        "LLM-Full":    "#17becf",
        "LLM-Summary": "#2ecc71",
        "LLM-Window":  "#1abc9c",
        "Tool Agent":  "#e67e22",
        "TPE":         "#f39c12",
        "NSGA-II":     "#e74c3c",
        "Random":      "#9b59b6",
    }
    for name, d in conv_data.items():
        traj = d["trajectory"]
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(traj) + 1)),
            y=traj,
            mode="lines+markers",
            name=name,
            line=dict(color=colors_conv.get(name, "#95a5a6"), width=2),
            marker=dict(size=5),
        ))
    fig_conv.update_layout(
        title="Best-so-far Score 收斂曲線（3 次平均）",
        xaxis_title="Trial number",
        yaxis_title="Best Score",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=420,
    )
    st.plotly_chart(fig_conv, width="stretch")

    first_best_rows = []
    for name, d in conv_data.items():
        first_best_rows.append({
            "方法": name,
            "平均首次達到最佳的 Iteration": round(d["avg_first_best"], 1),
            "各次執行": str(d["first_bests"]),
        })
    st.dataframe(pd.DataFrame(first_best_rows), width="stretch", hide_index=True)

    # 動態計算最快和最慢收斂的方法
    _conv_rows = [(name, d["avg_first_best"]) for name, d in conv_data.items()]
    _conv_rows_sorted = sorted(_conv_rows, key=lambda x: x[1])
    _conv_fastest_name, _conv_fastest_val = _conv_rows_sorted[0]
    _conv_slowest_name, _conv_slowest_val = _conv_rows_sorted[-1]

    st.markdown(f"""
**{_conv_fastest_name} 平均在第 {_conv_fastest_val:.1f} 個 trial 就找到最終最佳配置**，
{_conv_slowest_name} 則要等到第 {_conv_slowest_val:.0f} 個。

這不是偶然——LLM 在 Iteration 1 就選 GPTQ 4-bit（一個高品質的初始猜測），
而 TPE Iteration 1 嘗試 BNB 8-bit，score 只有 1.17，之後花了許多 trial 才走回正軌。

> LLM 的先驗知識讓它**跳過了大部分「初期摸索」階段**，直接從較好的起點出發。
""")

    st.divider()

    # ── 5. LLM Suggestion 內容深度分析 ─────────────────────────────────────────
    st.subheader("5. LLM Suggestion 內容的深度分析")

    st.markdown("### 5.1 系統性覆蓋式探索策略（Iter 1～10）")
    st.markdown("""
LLM 在前 8 個 trial 中，對每一種壓縮類型各試一次，且每次都有明確的選擇理由。
這不是隨機的廣度探索，而是**有組織的類型覆蓋**。以下動態載入 LLM-Summary Run1 的前 10 次探索：
""")

    @st.cache_data
    def load_exploration_table() -> pd.DataFrame:
        """Load first 10 iterations from LLM-Summary Run1 for exploration display."""
        summary_folders = sorted(
            p.name for p in BASE_DIR.glob("exp_Llama-3.2-3B-Instruct_gsm8k_summary_*")
            if (p / "optimization_results.json").exists()
        )
        if not summary_folders:
            return pd.DataFrame()
        path = BASE_DIR / summary_folders[0] / "optimization_results.json"
        with open(path) as f:
            data = json.load(f)
        rows = []
        for d in data[:10]:
            sug = d.get("suggestion") or {}
            m = d.get("metrics", {})
            mode = sug.get("mode", "unknown")
            reasoning_full = sug.get("reasoning", "")
            reasoning_short = reasoning_full[:120] + "…" if len(reasoning_full) > 120 else reasoning_full
            score = m.get("score", 0)
            rows.append({
                "Iteration": d["iteration"],
                "模式（mode）": mode,
                "LLM 選擇理由摘要（前 120 字）": reasoning_short,
                "實際 Score": round(score, 3),
            })
        return pd.DataFrame(rows)

    df_exploration = load_exploration_table()
    if not df_exploration.empty:
        styled_expl = df_exploration.style.background_gradient(subset=["實際 Score"], cmap="RdYlGn").set_table_styles([
            {"selector": "th", "props": [("background-color", "#1a252f"), ("color", "white"),
                                          ("font-size", "12px"), ("padding", "6px")]},
            {"selector": "td", "props": [("font-size", "11px"), ("padding", "5px 8px")]},
        ])
        st.write(styled_expl.to_html(), unsafe_allow_html=True)
    else:
        st.warning("找不到 LLM-Summary Run1 資料，請確認 final_results/ 目錄。")

    st.markdown("### 5.2 學習失敗、建立機制性解釋（全部 12 個 LLM 實驗）")
    st.markdown("""
此節動態載入所有 12 個 LLM 實驗（Full × 3 + Summary × 3 + Window × 3 + Tool Agent × 3）的原文推理，
依「決策模式」分 4 個主題整理，讓你看到**跨次執行的一致性**，而不只是單次結果。
""")

    @st.cache_data
    def load_all_llm_reasoning() -> dict:
        """Load reasoning from all 12 LLM experiments, keyed by (exp_type, run_id, iter)."""
        def _glob_runs(pattern: str, exp_type: str) -> dict:
            folders = sorted(
                p.name for p in BASE_DIR.glob(pattern)
                if (p / "optimization_results.json").exists()
            )
            return {(exp_type, f"Run{i+1}"): folder for i, folder in enumerate(folders)}

        llm_exps: dict = {}
        llm_exps.update(_glob_runs("exp_Llama-3.2-3B-Instruct_gsm8k_full_*",    "Full"))
        llm_exps.update(_glob_runs("exp_Llama-3.2-3B-Instruct_gsm8k_summary_*", "Summary"))
        llm_exps.update(_glob_runs("exp_Llama-3.2-3B-Instruct_gsm8k_window_*",  "Window"))
        llm_exps.update(_glob_runs("exp_Llama-3.2-3B-Instruct_gsm8k_tool_*",    "Tool"))

        result = {}
        for (exp_type, run_id), folder in llm_exps.items():
            path = BASE_DIR / folder / "optimization_results.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            for d in data:
                sug = d.get("suggestion") or {}
                reasoning = sug.get("reasoning", "")
                if not reasoning:
                    continue
                m = d.get("metrics", {})
                result[(exp_type, run_id, d["iteration"])] = {
                    "trial_name": d.get("trial_name", ""),
                    "reasoning": reasoning,
                    "score": round(m.get("score", 0), 3),
                    "accuracy": m.get("accuracy", 0),
                }
        return result

    all_reasoning = load_all_llm_reasoning()

    def show_reasoning_block(exp_type: str, run_id: str, iters: list, _category: str = ""):  # noqa: ARG001
        """Render one expander block for a given (exp_type, run_id) across multiple iterations."""
        entries = [(it, all_reasoning.get((exp_type, run_id, it))) for it in iters]
        entries = [(it, e) for it, e in entries if e]
        if not entries:
            st.caption("（此次執行無對應 iteration）")
            return
        for it, e in entries:
            score_color = "🟢" if e["score"] > 0 else "🔴"
            st.markdown(f"**Iter {it} — `{e['trial_name']}`** &nbsp; {score_color} score={e['score']}  acc={e['accuracy']:.3f}")
            st.markdown(f"> {e['reasoning']}")
            st.markdown("---")

    # ── 主題一：失敗後立即反應（Iter 3～5）────────────────────────────────────
    st.markdown("#### 🔴 主題一：遭遇失敗後的即時反應與策略調整（Iter 3～5）")
    st.markdown("""
ASVD 或 sparse 在 Iter 2～3 導致 accuracy 崩潰後，LLM 在**下一個 iteration 立刻做出清晰的診斷**，
並直接回到安全的量化路線——這發生在全部 12 次執行中，且理由措辭高度一致。
""")

    f_tab1, s_tab1, w_tab1, t_tab1 = st.tabs([
        "🔵 Full 實驗（Run1 / 2 / 3）",
        "📄 Summary 實驗（Run1 / 2 / 3）",
        "🪟 Window 實驗（Run1 / 2 / 3）",
        "🔧 Tool Agent（Run1 / 2 / 3）",
    ])
    with f_tab1:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.markdown("**Full-Run1**")
            show_reasoning_block("Full", "Run1", [3, 4], "recovery")
        with fc2:
            st.markdown("**Full-Run2**")
            show_reasoning_block("Full", "Run2", [3, 4], "recovery")
        with fc3:
            st.markdown("**Full-Run3**")
            show_reasoning_block("Full", "Run3", [3, 4], "recovery")
    with s_tab1:
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**Summary-Run1**")
            show_reasoning_block("Summary", "Run1", [3, 4], "recovery")
        with sc2:
            st.markdown("**Summary-Run2**")
            show_reasoning_block("Summary", "Run2", [3, 4], "recovery")
        with sc3:
            st.markdown("**Summary-Run3**")
            show_reasoning_block("Summary", "Run3", [3, 4], "recovery")
    with w_tab1:
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            st.markdown("**Window-Run1**")
            show_reasoning_block("Window", "Run1", [3, 4], "recovery")
        with wc2:
            st.markdown("**Window-Run2**")
            show_reasoning_block("Window", "Run2", [3, 4], "recovery")
        with wc3:
            st.markdown("**Window-Run3**")
            show_reasoning_block("Window", "Run3", [3, 4], "recovery")
    with t_tab1:
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.markdown("**Tool-Run1**")
            show_reasoning_block("Tool", "Run1", [3, 4], "recovery")
        with tc2:
            st.markdown("**Tool-Run2**")
            show_reasoning_block("Tool", "Run2", [3, 4], "recovery")
        with tc3:
            st.markdown("**Tool-Run3**")
            show_reasoning_block("Tool", "Run3", [3, 4], "recovery")

    st.info("""
**跨 12 次執行的共同模式**：無論 Full、Summary、Window 還是 Tool Agent，當 ASVD 或 sparse 造成 accuracy 崩潰時，
LLM 都會在下一次明確說「accuracy drop beyond threshold」或「triggering the penalty」，
並立刻切回 AWQ / BNB 等量化方法。這種即時反應在統計方法中完全不存在。
""")

    # ── 主題二：所有方法探索完後的收斂認識（Iter 9）──────────────────────────
    st.markdown("#### 🟡 主題二：全模式探索完畢後的收斂認識（Iter 9～12）")
    st.markdown("""
在前 8 個 trial 涵蓋所有壓縮類型後，LLM 在第 9 個 iteration 時幾乎**全部都得出相同結論**：
「GPTQ 4-bit 分數最高，下一步應聚焦精調 GPTQ 的超參數。」
但 Window-Run2 在這個階段有趣地先嘗試 Hybrid 再回頭，顯示不同記憶格式如何影響決策。
""")

    f_tab2, s_tab2, w_tab2, t_tab2 = st.tabs([
        "🔵 Full 實驗（Run1 / 2 / 3）",
        "📄 Summary 實驗（Run1 / 2 / 3）",
        "🪟 Window 實驗（Run1 / 2 / 3）",
        "🔧 Tool Agent（Run1 / 2 / 3）",
    ])
    with f_tab2:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.markdown("**Full-Run1**（Iter 9 收斂）")
            show_reasoning_block("Full", "Run1", [9, 10], "convergence")
        with fc2:
            st.markdown("**Full-Run2**（Iter 9 收斂）")
            show_reasoning_block("Full", "Run2", [9, 10], "convergence")
        with fc3:
            st.markdown("**Full-Run3**（Iter 9 收斂）")
            show_reasoning_block("Full", "Run3", [9, 10], "convergence")
    with s_tab2:
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**Summary-Run1**")
            show_reasoning_block("Summary", "Run1", [9, 10], "convergence")
        with sc2:
            st.markdown("**Summary-Run2**")
            show_reasoning_block("Summary", "Run2", [9, 10], "convergence")
        with sc3:
            st.markdown("**Summary-Run3**")
            show_reasoning_block("Summary", "Run3", [9, 10], "convergence")
    with w_tab2:
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            st.markdown("**Window-Run1**（Iter 11 才正式收斂）")
            show_reasoning_block("Window", "Run1", [11, 12], "convergence")
        with wc2:
            st.markdown("**Window-Run2**（Iter 9 先試 Hybrid，Iter 12 才收斂）")
            show_reasoning_block("Window", "Run2", [9, 12], "convergence")
        with wc3:
            st.markdown("**Window-Run3**（Iter 10 收斂）")
            show_reasoning_block("Window", "Run3", [10, 16], "convergence")
    with t_tab2:
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.markdown("**Tool-Run1**（Iter 9 收斂）")
            show_reasoning_block("Tool", "Run1", [9, 10], "convergence")
        with tc2:
            st.markdown("**Tool-Run2**（Iter 9 收斂）")
            show_reasoning_block("Tool", "Run2", [9, 10], "convergence")
        with tc3:
            st.markdown("**Tool-Run3**（Iter 9 收斂）")
            show_reasoning_block("Tool", "Run3", [9, 10], "convergence")

    st.warning("""
**Window-Run2 的特殊路徑**：這次執行在 Iter 9～11 仍在嘗試 Hybrid ASVD+BNB，比其他執行晚 2～3 個 iter 才收斂。
原因可能是 Window 模式的近期記憶中，Hybrid 的某些指標看起來「有一點點進步」，
誤導 LLM 認為值得再試一次。這說明 Window 模式也有侷限——近期偏誤（recency bias）。
""")

    # ── 主題三：機制性解釋（失敗原因推論）────────────────────────────────────
    st.markdown("#### 🔵 主題三：機制性失敗原因推論（不只記住分數）")
    st.markdown("""
LLM 最令人印象深刻的能力之一：它不只記住「哪個方法分數低」，
還會**推斷原因**，例如「ASVD+BNB 有相容性問題」、「gptq_v2 修正 overflow」。
以下是 12 次執行中出現的機制性推理摘錄：
""")

    f_tab3, s_tab3, w_tab3, t_tab3 = st.tabs([
        "🔵 Full 實驗",
        "📄 Summary 實驗",
        "🪟 Window 實驗",
        "🔧 Tool Agent",
    ])
    with f_tab3:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.markdown("**Full-Run1**（Iter 10–11 機制推理）")
            show_reasoning_block("Full", "Run1", [10, 11], "mechanism")
        with fc2:
            st.markdown("**Full-Run2**（Iter 10–11 機制推理）")
            show_reasoning_block("Full", "Run2", [10, 11], "mechanism")
        with fc3:
            st.markdown("**Full-Run3**（Iter 10–11 機制推理）")
            show_reasoning_block("Full", "Run3", [10, 11], "mechanism")
    with s_tab3:
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**Summary-Run1**（Iter 10–11 格式 + 相容性）")
            show_reasoning_block("Summary", "Run1", [10, 11], "mechanism")
        with sc2:
            st.markdown("**Summary-Run2**（Iter 10–11 群組大小推理）")
            show_reasoning_block("Summary", "Run2", [10, 11], "mechanism")
        with sc3:
            st.markdown("**Summary-Run3**（Iter 11 ASVD 保守重試）")
            show_reasoning_block("Summary", "Run3", [11, 12], "mechanism")
    with w_tab3:
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            st.markdown("**Window-Run1**（Iter 11 ASVD 認識修正）")
            show_reasoning_block("Window", "Run1", [11], "mechanism")
        with wc2:
            st.markdown("**Window-Run2**（Iter 11 ASVD 保守嘗試）")
            show_reasoning_block("Window", "Run2", [11], "mechanism")
        with wc3:
            st.markdown("**Window-Run3**（Iter 12 Hybrid 再測試）")
            show_reasoning_block("Window", "Run3", [12], "mechanism")
    with t_tab3:
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.markdown("**Tool-Run1**（Iter 10–11 機制推理）")
            show_reasoning_block("Tool", "Run1", [10, 11], "mechanism")
        with tc2:
            st.markdown("**Tool-Run2**（Iter 10–11 機制推理）")
            show_reasoning_block("Tool", "Run2", [10, 11], "mechanism")
        with tc3:
            st.markdown("**Tool-Run3**（Iter 10–11 機制推理）")
            show_reasoning_block("Tool", "Run3", [10, 11], "mechanism")

    st.info("""
**跨 12 次執行的機制推理一致性**：
- Full/Tool Agent：完整歷史或工具呼叫式記憶，推理直接排除早期失敗配置，解釋更有層次
- Summary-Run1 推斷 Hybrid 失敗是「ASVD+BNB 相容性問題」
- Summary-Run2 觀察到 gptq 群組大小（g64→g32）改善了分數
- Window-Run1 注意到 ASVD 高 ratio (0.95) 比低 ratio 效果更好，嘗試更保守的壓縮率
- 所有 12 次執行都最終放棄 ASVD ratio < 0.88 的配置，但理由精細度因記憶策略而不同
""")

    # ── 主題四：超參數空間縮窄（Iter 15+）────────────────────────────────────
    st.markdown("#### 🟢 主題四：超參數安全區間的自動歸納（Iter 15+）")
    st.markdown("""
在後期 iteration，LLM 展現出**即時建立替代模型**的能力：
它從多次實驗中歸納出「damp_percent 0.03～0.05 是安全有效區間」，
並在這個範圍內做精密的網格搜尋，而不是漫無目的地重複試驗。
""")

    f_tab4, s_tab4, w_tab4, t_tab4 = st.tabs([
        "🔵 Full 實驗",
        "📄 Summary 實驗",
        "🪟 Window 實驗",
        "🔧 Tool Agent",
    ])
    with f_tab4:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            st.markdown("**Full-Run1**（Iter 15–17 damp 精調）")
            show_reasoning_block("Full", "Run1", [15, 16, 17], "hyperparam")
        with fc2:
            st.markdown("**Full-Run2**（Iter 15–16 精調）")
            show_reasoning_block("Full", "Run2", [15, 16], "hyperparam")
        with fc3:
            st.markdown("**Full-Run3**（Iter 15–16 精調）")
            show_reasoning_block("Full", "Run3", [15, 16], "hyperparam")
    with s_tab4:
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**Summary-Run1**（Iter 17–19 damp 精調）")
            show_reasoning_block("Summary", "Run1", [17, 18, 19], "hyperparam")
        with sc2:
            st.markdown("**Summary-Run2**（Iter 16–17 group_size 搜尋）")
            show_reasoning_block("Summary", "Run2", [16, 17], "hyperparam")
        with sc3:
            st.markdown("**Summary-Run3**（Iter 15–16 格式 + damp）")
            show_reasoning_block("Summary", "Run3", [15, 16], "hyperparam")
    with w_tab4:
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            st.markdown("**Window-Run1**（Iter 18–19 精確 damp 區間）")
            show_reasoning_block("Window", "Run1", [18, 19], "hyperparam")
        with wc2:
            st.markdown("**Window-Run2**（Iter 18–19 GPTQ 精調）")
            show_reasoning_block("Window", "Run2", [18, 19], "hyperparam")
        with wc3:
            st.markdown("**Window-Run3**（Iter 16–19 最密集精調）")
            show_reasoning_block("Window", "Run3", [16, 17, 18, 19], "hyperparam")
    with t_tab4:
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            st.markdown("**Tool-Run1**（Iter 15–17 damp 精調）")
            show_reasoning_block("Tool", "Run1", [15, 16, 17], "hyperparam")
        with tc2:
            st.markdown("**Tool-Run2**（Iter 15–16 精調）")
            show_reasoning_block("Tool", "Run2", [15, 16], "hyperparam")
        with tc3:
            st.markdown("**Tool-Run3**（Iter 15–17 精調）")
            show_reasoning_block("Tool", "Run3", [15, 16, 17], "hyperparam")

    st.info("""
**Window-Run3 在 Iter 16–19 是最密集的精調範例**：
連續 4 個 iteration 都在 GPTQ 4-bit g128 上微調 damp_percent，
Iter 17 推理中甚至說「Iter 16 達到平衡，現在探索 damp 從 0.04 到某個值...」，
展現了類似人工研究員逐步縮小搜尋範圍的行為。

**Full 與 Tool Agent 的後期精調**：因為具備完整/工具呼叫式的歷史記憶，
在精調階段能更快識別哪個 damp_percent 曾造成效能下降，精調效率優於 Summary 的摘要壓縮。
""")

    # ── 跨 12 次執行一致性總結 ─────────────────────────────────────────────────
    st.markdown("#### 📊 跨 12 次執行：決策模式一致性熱力圖")
    st.markdown("以下統計各次執行中，哪個 iteration 首次做出關鍵決策轉折點（含 Full 與 Tool Agent）：")

    @st.cache_data
    def compute_decision_patterns() -> pd.DataFrame:
        def _glob_dp(pattern: str, exp_type: str) -> dict:
            folders = sorted(
                p.name for p in BASE_DIR.glob(pattern)
                if (p / "optimization_results.json").exists()
            )
            return {(exp_type, f"Run{i+1}"): folder for i, folder in enumerate(folders)}

        llm_exps_dp: dict = {}
        llm_exps_dp.update(_glob_dp("exp_Llama-3.2-3B-Instruct_gsm8k_full_*",    "Full"))
        llm_exps_dp.update(_glob_dp("exp_Llama-3.2-3B-Instruct_gsm8k_summary_*", "Summary"))
        llm_exps_dp.update(_glob_dp("exp_Llama-3.2-3B-Instruct_gsm8k_window_*",  "Window"))
        llm_exps_dp.update(_glob_dp("exp_Llama-3.2-3B-Instruct_gsm8k_tool_*",    "Tool"))

        rows = []
        for (exp_type, run_id), folder in llm_exps_dp.items():
            path = BASE_DIR / folder / "optimization_results.json"
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)

            scores    = [d["metrics"].get("score", 0) for d in data if d.get("metrics")]
            methods   = [detailed_method(d) for d in data]
            first_gptq = next((i + 1 for i, m in enumerate(methods) if m == "gptq"), None)
            first_recovery = None
            for i in range(1, len(scores)):
                if scores[i - 1] < -3 and methods[i] in ("gptq", "awq", "bnb"):
                    first_recovery = i + 1
                    break
            first_gptq_streak = None
            for i in range(len(methods) - 2):
                if all(m == "gptq" for m in methods[i:i + 3]):
                    first_gptq_streak = i + 1
                    break
            best_score = max(scores) if scores else 0
            best_iter  = next((i + 1 for i, s in enumerate(scores) if s == best_score), None)

            rows.append({
                "實驗": f"{exp_type}-{run_id}",
                "類型": exp_type,
                "首次 GPTQ (iter)": first_gptq or "—",
                "失敗後回量化 (iter)": first_recovery or "—",
                "連續 GPTQ 開始 (iter)": first_gptq_streak or "—",
                "達到最佳 (iter)": best_iter or "—",
                "最終最佳 Score": round(best_score, 3),
                "正向 Trial 數": sum(1 for s in scores if s > 0),
            })
        return pd.DataFrame(rows)

    df_patterns = compute_decision_patterns()

    style_pat = df_patterns.style.background_gradient(
        subset=["最終最佳 Score"], cmap="RdYlGn"
    ).background_gradient(
        subset=["正向 Trial 數"], cmap="Blues"
    ).set_table_styles([
        {"selector": "th", "props": [("background-color", "#1a252f"), ("color", "white"),
                                      ("font-size", "12px"), ("padding", "6px 8px"), ("text-align", "center")]},
        {"selector": "td", "props": [("font-size", "11px"), ("padding", "5px 10px"), ("text-align", "center")]},
    ])
    st.write(style_pat.to_html(), unsafe_allow_html=True)

    st.markdown("""
**觀察重點**：
- **首次 GPTQ** 全部 12 次都在 Iter 1，說明 LLM 對 GPTQ 作為起點的共識非常一致（含 Full 和 Tool Agent）
- **失敗後回量化**：Full/Tool Agent 最快（Iter 3）；Window-Run2 因近期偏誤較慢
- **連續 GPTQ 開始**：Full 組約在 Iter 9-11；Summary 組約在 Iter 9-11；Window 組因多出探索迴路，約 Iter 11-16
- **最終最佳 iter 的分散性**：Tool Agent 最穩定（std=0.038）；Full 次之（std=0.044）；Summary 變異最大（std=0.164）
""")

    st.markdown("### 5.3 LLM 準確判斷壓縮方法品質排序")
    st.markdown("""
根據 LLM 的 suggestion 與 **7×3×20=420 次實際測量**的對照，LLM 的判斷排序幾乎完全正確。
各方法最終均收斂至 **GPTQ 4-bit g128** 為最優（LLM 方法）或接近最優的配置（統計方法）：
""")

    method_quality_data = {
        "壓縮方法": ["GPTQ 4-bit g128", "AWQ 4-bit", "BNB 4-bit (double_quant)", "BNB 4-bit", "BNB 8-bit",
                  "GPTQ 3-bit", "QQQ 4-bit g-1", "ASVD ratio≥0.95", "ASVD ratio<0.90",
                  "Sparse unstructured 50%", "Sparse 2:4", "QQQ 4-bit g128", "Hybrid ASVD+BNB"],
        "平均 Accuracy": ["66%～69%", "62%～67%", "67%～68%", "66%～68%", "69%～70%",
                       "39%～56%", "42%～52%", "38%～48%", "2%～20%",
                       "5%～38%", "5%～15%", "1%～2%", "2%～25%"],
        "風險等級": ["✅ 低風險", "✅ 低風險", "✅ 低風險", "✅ 低風險", "✅ 低風險",
                  "⚠️ 中風險", "⚠️ 中風險", "⚠️ 中風險", "❌ 高風險",
                  "❌ 高風險", "❌ 高風險", "❌ 高風險", "❌ 高風險"],
        "LLM 方法判斷": ["正確偏好（全 12 次 Iter 1 起點）", "正確偏好", "正確偏好", "正確偏好", "正確偏好",
                       "謹慎嘗試", "謹慎嘗試", "謹慎嘗試", "後期主動規避",
                       "後期主動規避", "後期主動規避", "後期主動規避", "後期主動規避"],
        "統計方法最終選擇": ["TPE 最佳（g32）", "NSGA-II/Random 最佳", "偶爾選中", "偶爾選中", "TPE 初期常選",
                          "少量出現", "少量出現", "少量出現", "統計踩雷後漸少",
                          "統計踩雷後漸少", "統計踩雷後漸少", "統計踩雷後漸少", "統計踩雷後漸少"],
    }
    st.dataframe(pd.DataFrame(method_quality_data), width="stretch", hide_index=True)

    st.divider()

    # ── 6. 趨勢分析 ────────────────────────────────────────────────────────────
    st.subheader("6. 趨勢分析")

    st.markdown("### 6.1 四種 LLM 策略差異，以及與統計方法的根本區別")

    _llm_std_data = {}
    for _method_name, _pattern in [
        ("LLM-Full",    "exp_Llama-3.2-3B-Instruct_gsm8k_full_*"),
        ("LLM-Summary", "exp_Llama-3.2-3B-Instruct_gsm8k_summary_*"),
        ("LLM-Window",  "exp_Llama-3.2-3B-Instruct_gsm8k_window_*"),
        ("Tool Agent",  "exp_Llama-3.2-3B-Instruct_gsm8k_tool_*"),
    ]:
        _folders = sorted(
            p.name for p in BASE_DIR.glob(_pattern)
            if (p / "optimization_results.json").exists()
        )
        _bests = []
        for _f in _folders:
            _p = BASE_DIR / _f / "optimization_results.json"
            with open(_p) as _fh:
                _d = json.load(_fh)
            _scores = [x["metrics"].get("score", 0) for x in _d if x.get("metrics")]
            if _scores:
                _bests.append(max(_scores))
        _llm_std_data[_method_name] = statistics.stdev(_bests) if len(_bests) > 1 else 0.0

    col_f, col_s, col_w, col_t = st.columns(4)
    col_f.metric("LLM-Full std", f"±{_llm_std_data['LLM-Full']:.3f}", help="3 次執行的變異；完整歷史記憶")
    col_s.metric("LLM-Summary std", f"±{_llm_std_data['LLM-Summary']:.3f}", help="3 次執行的變異；摘要式記憶可能遺失細節")
    col_w.metric("LLM-Window std", f"±{_llm_std_data['LLM-Window']:.3f}", help="3 次執行的變異；近期視窗記憶")
    col_t.metric("Tool Agent std", f"±{_llm_std_data['Tool Agent']:.3f}", help="3 次執行的變異；工具呼叫式架構")

    st.markdown("""
**四種策略差異解讀**：
- **LLM-Full**（std≈0.044）：完整歷史不遺失任何資訊，後期推理可能受長文本影響
- **LLM-Summary**（std≈0.164）：摘要壓縮歷史，最容易遺失微調細節，執行間差異最大
- **LLM-Window**（std≈0.056）：只看最近 N 筆，近期偏誤可能讓 Window-Run2 多繞路，但整體穩定
- **Tool Agent**（std≈0.038）：工具呼叫架構允許精確查詢特定歷史，是四種策略中最穩定的

**與統計方法的根本區別**：TPE（std≈0.219）變異最大，Random（std≈0.068），NSGA-II（std≈0.026）——
統計方法的穩定性來源不同：NSGA-II 穩定是因為所有 run 都落在較低的分數平台；
TPE 變異最大是因為有時運氣好早早找到 GPTQ g128，有時一直在低分區打轉。
""")

    st.markdown("### 6.2 最佳配置的效益分析（Green AI 角度）")
    st.markdown("""
LLM-Full 搜尋出的最佳配置（**GPTQ 4-bit g128，damp_percent ≈ 0.03～0.05**，score=4.729）
實現了以下的資源節省效益（以 LLM-Full best trial 為代表）：
""")

    b_lat  = baseline_report.get("latency",   1422.6073)
    b_vram = baseline_report.get("vram",      6.1952)
    b_emit = baseline_report.get("emissions", 0.09367)
    b_acc  = baseline_report.get("accuracy",  0.718)

    best_lat  = 445.9
    best_vram = 2.32
    best_emit = 0.01502
    best_acc  = 0.662

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy 損失", f"{(best_acc - b_acc) / b_acc * 100:.1f}%",
              delta_color="inverse", help="代價")
    c2.metric("Latency 降低", f"{(b_lat - best_lat) / b_lat * 100:.1f}%",
              delta=f"−{b_lat - best_lat:.0f}s", help="效益")
    c3.metric("VRAM 節省", f"{(b_vram - best_vram) / b_vram * 100:.1f}%",
              delta=f"−{b_vram - best_vram:.2f} GB", help="效益")
    c4.metric("碳排放降低", f"{(b_emit - best_emit) / b_emit * 100:.1f}%",
              delta=f"−{b_emit - best_emit:.5f}", help="效益")

    st.markdown(f"""
> 以 **{(best_acc - b_acc) / b_acc * 100:.1f}% 準確率損失** 換取 **{(b_lat - best_lat) / b_lat * 100:.1f}% Latency 降低**、
> **{(b_vram - best_vram) / b_vram * 100:.1f}% VRAM 節省**、**{(b_emit - best_emit) / b_emit * 100:.1f}% 碳排放降低**。
>
> 這是一個極為有利的 Green AI 交換比——模型仍保留原始準確率的 **{best_acc / b_acc * 100:.1f}%**，
> 同時資源消耗縮減至不到原來的 **35%**。

四種 LLM 策略的最佳 trial 均收斂至 GPTQ 4-bit g128，差異僅在精確的 damp_percent 值：
- LLM-Full best：acc=66.2%（-7.8%），VRAM=2.32GB（-62.5%），lat=445.9s（-68.7%），emit=0.01502（-84.0%）
- LLM-Summary best：acc=67.8%（-5.6%），VRAM=2.32GB（-62.5%），lat=476.1s（-66.5%），emit=0.01608（-82.8%）
- LLM-Window best：acc=67.0%（-6.7%），VRAM=2.32GB（-62.5%），lat=467.4s（-67.1%），emit=0.01596（-83.0%）
- Tool Agent best：acc=67.2%（-6.4%），VRAM=2.32GB（-62.5%），lat=459.7s（-67.7%），emit=0.01544（-83.5%）
""")

    st.markdown("### 6.3 LLM 記憶方法 vs 統計方法：根本差異對比")
    comparison_data = {
        "比較維度": [
            "先驗知識", "初始配置品質", "失敗處理",
            "探索策略", "收斂速度", "正向 Trial 率",
            "災難性失敗率", "可解釋性", "絕對最優搜尋能力",
        ],
        "LLM 記憶方法（Full/Summary/Window/Tool）": [
            "✅ 有（量化方法領域知識）",
            "✅ 高（Iter 1 即 GPTQ 4-bit，接近最優）",
            "✅ 立即推理失敗原因，下一步主動規避",
            "✅ 有組織的系統性壓縮類型覆蓋（Iter 1–8）",
            "✅ 快（LLM-Summary 平均 9.7 iter 達最佳）",
            "✅ 41.7%～55.0%（Tool/Full 稍低；Summary/Window 最高）",
            "✅ 低：5%～8.3%（3～6× 優於統計方法）",
            "✅ 完整文字推理，每次決策可審計",
            "⚠️ 受限於先驗偏見，偶爾錯過非主流配置（如 AWQ g32）",
        ],
        "統計方法（TPE / NSGA-II / Random）": [
            "❌ 無（從零開始統計學習）",
            "❌ 低（依賴隨機採樣，初期配置品質不穩定）",
            "❌ 需多次踩雷後統計模型才逐漸規避",
            "❌ 統計驅動（TPE/NSGA-II）或純隨機（Random）",
            "❌ 慢（TPE 平均 18 iter；NSGA-II/Random 亦長）",
            "❌ 41.7%～51.7%（平均低於 LLM 記憶方法）",
            "❌ 高：25%～31.7%（踩雷成本高）",
            "❌ 黑盒決策，無法解釋配置選擇原因",
            "✅ 理論上無偏，預算充足時可找到全局最優（TPE peak=4.693）",
        ],
    }
    st.dataframe(pd.DataFrame(comparison_data), width="stretch", hide_index=True)

    st.divider()

    # ── 7. 結論 ─────────────────────────────────────────────────────────────────
    st.subheader("7. 結論與建議")

    st.success("""
**LLM-Guided 搜尋的核心競爭力（有限預算場景下）**

1. **安全性最強**：LLM 記憶方法（Full/Summary/Window/Tool Agent）災難性失敗率僅 5%～8.3%，
   比統計方法（25%～31.7%）低 **3～6 倍**。
   對需要大量 GPU 時間的量化實驗，這直接等於節省計算成本。

2. **搜尋效率最高**：LLM-Summary 平均在第 9.7 個 trial 就達到最終最佳配置，
   TPE 需要 18 個 trial，差距接近 2 倍。

3. **決策品質有保障**：全 12 次 LLM 執行，Iter 1 均選 GPTQ 4-bit——接近最優的初始配置，
   因為 LLM 具備量化技術的領域先驗知識。

4. **自我修正能力**：LLM 能從失敗中提煉機制性解釋（如「ASVD+BNB 相容性問題」），
   後續 trial 的決策更有方向感，而非靠統計碰運氣。

5. **可解釋性**：每個決策都有文字化推理，研究人員能理解 AI 為何做出特定選擇，
   這在 Green AI 研究中有重要的學術與實務價值。

6. **所有方法均收斂 GPTQ 4-bit g128**：7 種搜尋策略的最佳 trial 均落在 GPTQ 4-bit g128（或類似配置），
   確認此為 Llama-3.2-3B-Instruct 在 GSM8K 任務的最佳 Green AI 壓縮配置。
""")

    st.warning("""
**限制與注意事項**

- **TPE 的優點**：雖然 catastrophic rate 高（25%），但 peak accuracy 最高（0.690）。
  在無預算限制、可接受更多失敗 trial、且 accuracy 優先的場景，TPE 仍是有價值的選項。

- LLM 的優勢在 **≤20 trials** 的場景最明顯；trial 數量大幅增加後，統計方法的優劣可能重新洗牌。

- **LLM-Summary 的風險**：3 次執行間標準差最大（std=0.164），摘要式歷史有時遺失細節，
  導致某次執行（Run2）最佳分數僅 4.343，遠低於平均。

- **Tool Agent 的穩定性**：雖然正向 trial 率（41.7%）略低於 LLM-Summary/Window（55%），
  但執行間最穩定（std=0.038），適合要求結果可重現的場景。
""")

    st.info("""
**場景化建議**

- **30 trial 有限預算、重視可靠性** → **LLM-Summary 或 LLM-Window**
  （最快收斂，9.7～10.7 iter；Summary 最快但較不穩定，Window 稍慢但更一致）

- **要求跨次執行結果高度一致** → **Tool Agent**
  （std=0.038，最穩定；工具呼叫架構確保歷史查詢精確）

- **無預算限制、最大化 accuracy 上界** → **TPE**
  （peak accuracy 0.690，理論上無偏的全局搜尋能力）

- **最推薦壓縮配置**（所有方法共識）：**GPTQ 4-bit g128（damp_percent ≈ 0.03～0.05，format=gptq）**
  能以約 6%～8% 的準確率損失換取超過 62% 的 VRAM 節省、68% 的 Latency 降低、83% 的碳排放降低。
""")
