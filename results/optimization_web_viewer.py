"""
優化實驗報告網頁檢視器

使用 Streamlit 建立的網頁介面，用於查看和分析量化優化實驗報告。
支援標準優化實驗和 LLM 多代理優化實驗。

執行方式：streamlit run optimization_web_viewer.py
"""

import streamlit as st
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# 頁面配置
st.set_page_config(
    page_title="優化實驗報告檢視器",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 結果目錄路徑
RESULTS_DIR = Path(__file__).parent
OPTIMIZATION_DIR = RESULTS_DIR / "optimization"
LLM_OPTIMIZATION_DIR = RESULTS_DIR / "llm_optimization"


# ==========================================
# 資料載入函數
# ==========================================

def get_pareto_solutions(pareto_data) -> List[Dict]:
    """
    從 pareto_data 提取 solutions 列表
    相容兩種格式：list（新）或 dict with "solutions"（舊）
    """
    if isinstance(pareto_data, list):
        return pareto_data
    elif isinstance(pareto_data, dict):
        return pareto_data.get("solutions", [])
    return []


def scan_experiments(exp_type: str) -> List[str]:
    """掃描指定類型的實驗目錄"""
    if exp_type == "標準優化":
        base_dir = OPTIMIZATION_DIR
    else:
        base_dir = LLM_OPTIMIZATION_DIR

    if not base_dir.exists():
        return []

    experiments = []
    for item in base_dir.iterdir():
        if item.is_dir() and (item / "summary.json").exists():
            experiments.append(item.name)

    return sorted(experiments, reverse=True)  # 最新的在前面


def load_experiment_data(exp_type: str, exp_name: str) -> Dict[str, Any]:
    """載入實驗的所有資料"""
    if exp_type == "標準優化":
        exp_dir = OPTIMIZATION_DIR / exp_name
    else:
        exp_dir = LLM_OPTIMIZATION_DIR / exp_name

    data = {"exp_type": exp_type, "exp_name": exp_name, "exp_dir": str(exp_dir)}

    # 載入各個 JSON 檔案
    json_files = [
        "summary.json",
        "all_trials.json",
        "pareto_frontier.json",
        "pareto_deep_analysis.json",
        "config.json",
        "satisfying_trials_scored.json",
        "failed_trials.json",
        "pruned_trials.json",
        "prompt_config_used.json"
    ]

    for filename in json_files:
        filepath = exp_dir / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                key = filename.replace(".json", "")
                data[key] = json.load(f)

    # LLM 版本：載入對話記錄
    conversations_file = exp_dir / "agent_conversations.jsonl"
    if conversations_file.exists():
        data["conversations"] = load_conversations(conversations_file)

    return data


def load_conversations(filepath: Path) -> List[Dict]:
    """載入 JSONL 格式的對話記錄"""
    conversations = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    conversations.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return conversations


# ==========================================
# 輔助函數
# ==========================================

def get_target_status_color(value: float, target: float, direction: str) -> str:
    """根據目標差距返回顏色"""
    if direction == "maximize":
        # 準確率: 越大越好，大於等於 target 為滿足
        if value >= target:
            return "green"
        elif value >= target * 0.9:  # 接近 (90% 以內)
            return "yellow"
        else:
            return "red"
    else:
        # GPU/延遲: 越小越好，小於等於 target 為滿足
        if value <= target:
            return "green"
        elif value <= target * 1.1:  # 接近 (110% 以內)
            return "yellow"
        else:
            return "red"


def get_status_emoji(satisfies: bool) -> str:
    """根據是否滿足目標返回 emoji"""
    return "✅" if satisfies else "❌"


def format_config_summary(config: Dict) -> str:
    """格式化配置摘要字串"""
    method = config.get("method", "unknown")
    if method == "gptq":
        bits = config.get("bits", "?")
        group_size = config.get("group_size", "?")
        return f"GPTQ {bits}bit g{group_size}"
    elif method == "awq":
        w_bit = config.get("w_bit", "?")
        q_group_size = config.get("q_group_size", "?")
        return f"AWQ {w_bit}bit g{q_group_size}"
    elif method == "bnb":
        bits = config.get("bits", "?")
        quant_type = config.get("bnb_4bit_quant_type", "nf4")
        return f"BNB {bits}bit {quant_type}"
    return method


def display_config_details(config: Dict, method: str):
    """通用的配置參數顯示函數"""
    if method == "gptq":
        st.write(f"- **bits**: `{config.get('bits', 'N/A')}`")
        st.write(f"- **group_size**: `{config.get('group_size', 'N/A')}`")
        st.write(f"- **calib_num**: `{config.get('calib_num', 'N/A')}`")
        st.write(f"- **desc_act**: `{config.get('desc_act', 'N/A')}`")
        st.write(f"- **sym**: `{config.get('sym', 'N/A')}`")
        st.write(f"- **damp_percent**: `{config.get('damp_percent', 'N/A')}`")
        st.write(f"- **damp_auto_increment**: `{config.get('damp_auto_increment', 'N/A')}`")
        st.write(f"- **static_groups**: `{config.get('static_groups', 'N/A')}`")
        st.write(f"- **true_sequential**: `{config.get('true_sequential', 'N/A')}`")
        st.write(f"- **mse**: `{config.get('mse', 'N/A')}`")
        st.write(f"- **lm_head**: `{config.get('lm_head', 'N/A')}`")
    elif method == "awq":
        st.write(f"- **w_bit**: `{config.get('w_bit', 'N/A')}`")
        st.write(f"- **q_group_size**: `{config.get('q_group_size', 'N/A')}`")
        st.write(f"- **zero_point**: `{config.get('zero_point', 'N/A')}`")
        st.write(f"- **version**: `{config.get('version', 'N/A')}`")
        modules = config.get('modules_to_not_convert', [])
        st.write(f"- **modules_to_not_convert**: `{modules if modules else '(無)'}`")
    elif method == "bnb":
        bits = config.get('bits', 4)
        st.write(f"- **bits**: `{bits}`")
        if bits == 4:
            st.write(f"- **bnb_4bit_quant_type**: `{config.get('bnb_4bit_quant_type', 'N/A')}`")
            st.write(f"- **bnb_4bit_use_double_quant**: `{config.get('bnb_4bit_use_double_quant', 'N/A')}`")
            st.write(f"- **bnb_4bit_compute_dtype**: `{config.get('bnb_4bit_compute_dtype', 'N/A')}`")
        else:
            st.write(f"- **llm_int8_threshold**: `{config.get('llm_int8_threshold', 'N/A')}`")
            st.write(f"- **llm_int8_skip_modules**: `{config.get('llm_int8_skip_modules', 'N/A')}`")
    else:
        for key, value in config.items():
            if key != "modules_to_not_convert":
                st.write(f"- **{key}**: `{value}`")


def styled_metric_with_color(label: str, value: float, target: float, direction: str, format_str: str = "{:+.2f}%"):
    """顯示帶顏色的指標"""
    color = get_target_status_color(value, target, direction)
    color_map = {"green": "#28a745", "yellow": "#ffc107", "red": "#dc3545"}
    formatted = format_str.format(value * 100) if abs(value) < 100 else format_str.format(value)
    st.markdown(f"**{label}**: <span style='color: {color_map[color]}; font-weight: bold;'>{formatted}</span>", unsafe_allow_html=True)


def calculate_violation_score(obj: Dict, targets: Dict) -> float:
    """計算 violation_score（向下相容用）"""
    # 若已有 violation_score，直接返回
    vio = obj.get("violation_score")
    if vio is not None:
        return vio

    # 否則計算
    acc = obj.get("accuracy_change")
    gpu = obj.get("gpu_peak_change")
    lat = obj.get("latency_change")

    if acc is None or gpu is None or lat is None:
        return None

    target_acc = targets.get('accuracy_min', -0.1)
    target_gpu = targets.get('gpu_peak_max', -0.5)
    target_lat = targets.get('latency_max', 1.5)

    vio = 0.0
    if acc < target_acc:
        vio += abs(acc - target_acc)
    if gpu > target_gpu:
        vio += abs(gpu - target_gpu)
    if lat > target_lat:
        vio += abs(lat - target_lat)

    return vio


# ==========================================
# 視覺化函數
# ==========================================

def get_trial_status(trial: Dict) -> str:
    """
    獲取試驗狀態，兼容新舊數據格式
    新格式：有 status 欄位
    舊格式：用 success 欄位推斷
    """
    if "status" in trial:
        return trial["status"]
    # 舊格式兼容
    if trial.get("success", False):
        return "completed"
    elif "error" in trial:
        return "failed"
    return "unknown"


def is_trial_completed(trial: Dict) -> bool:
    """檢查試驗是否完成（成功），兼容新舊格式"""
    status = get_trial_status(trial)
    if status == "completed":
        return True
    # 舊格式：沒有 status 但有有效的 objectives
    if status == "unknown" and trial.get("objectives", {}).get("accuracy_change") is not None:
        obj = trial.get("objectives", {})
        # 確保 objectives 不是 inf（失敗時的預設值）
        if obj.get("accuracy_change") != float('inf'):
            return True
    return False


def create_3d_pareto_plot(pareto_data: Dict, all_trials: Dict, targets: Dict) -> go.Figure:
    """建立 3D Pareto 前沿圖"""
    fig = go.Figure()

    # 獲取 Pareto 解
    pareto_solutions = get_pareto_solutions(pareto_data)

    # 獲取所有完成的試驗（兼容新舊格式）
    all_completed = [t for t in all_trials.get("trials", [])
                     if is_trial_completed(t) and
                     t.get("objectives", {}).get("accuracy_change") is not None]

    # 建立 Pareto 解的識別集合
    def trial_key(t):
        config = t["config"]
        return (config.get("method"), str(config))

    pareto_keys = {trial_key(t) for t in pareto_solutions}

    # 分類試驗
    pareto_trials = []
    non_pareto_trials = []

    for t in all_completed:
        obj = t["objectives"]
        trial_info = {
            "acc": obj["accuracy_change"] * 100,
            "gpu": obj["gpu_peak_change"] * 100,
            "lat": obj["latency_change"] * 100,
            "method": t["config"]["method"],
            "satisfies": t.get("satisfies_targets", False),
            "config": t["config"]
        }

        if trial_key(t) in pareto_keys:
            pareto_trials.append(trial_info)
        else:
            non_pareto_trials.append(trial_info)

    # 繪製非 Pareto 解
    if non_pareto_trials:
        fig.add_trace(go.Scatter3d(
            x=[t["acc"] for t in non_pareto_trials],
            y=[t["gpu"] for t in non_pareto_trials],
            z=[t["lat"] for t in non_pareto_trials],
            mode="markers",
            marker=dict(size=6, color="lightgray", opacity=0.5),
            name="非 Pareto 解",
            hovertemplate="準確率: %{x:+.2f}%<br>GPU: %{y:+.2f}%<br>延遲: %{z:+.2f}%<extra></extra>"
        ))

    # 繪製 Pareto 解
    if pareto_trials:
        colors = ["green" if t["satisfies"] else "blue" for t in pareto_trials]
        fig.add_trace(go.Scatter3d(
            x=[t["acc"] for t in pareto_trials],
            y=[t["gpu"] for t in pareto_trials],
            z=[t["lat"] for t in pareto_trials],
            mode="markers+text",
            marker=dict(size=10, color=colors, opacity=0.9),
            text=[t["method"] for t in pareto_trials],
            textposition="top center",
            name="Pareto 前沿",
            hovertemplate="<b>%{text}</b><br>準確率: %{x:+.2f}%<br>GPU: %{y:+.2f}%<br>延遲: %{z:+.2f}%<extra></extra>"
        ))

    # 繪製目標區域方框（12條邊）
    if targets:
        acc_min = targets.get('accuracy_min', -0.1) * 100  # 例如 -10%
        gpu_max = targets.get('gpu_peak_max', -0.5) * 100  # 例如 -50%
        lat_max = targets.get('latency_max', 1.5) * 100    # 例如 +150%

        # 從所有試驗中獲取數據範圍以確定邊界
        all_acc = [t["acc"] for t in pareto_trials + non_pareto_trials]
        all_gpu = [t["gpu"] for t in pareto_trials + non_pareto_trials]
        all_lat = [t["lat"] for t in pareto_trials + non_pareto_trials]
        
        # 計算數據範圍，並稍微擴展以確保完整顯示
        data_acc_max = max(all_acc) if all_acc else 5
        data_gpu_min = min(all_gpu) if all_gpu else -100
        data_lat_min = min(all_lat) if all_lat else -50
        
        # 目標區域範圍：從目標值延伸到數據邊界（擴展10%以確保完整顯示）
        acc_extend = (data_acc_max - acc_min) * 0.1
        gpu_extend = abs(data_gpu_min - gpu_max) * 0.1
        lat_extend = abs(data_lat_min - lat_max) * 0.1
        
        x_range = [acc_min, data_acc_max + acc_extend]      # 準確率: 從目標最小值延伸到數據最大值
        y_range = [data_gpu_min - gpu_extend, gpu_max]      # GPU: 從數據最小值延伸到目標最大值
        z_range = [data_lat_min - lat_extend, lat_max]      # 延遲: 從數據最小值延伸到目標最大值

        # 定義立方體的 12 條邊
        edges = [
            # 底面 4 條邊 (z = z_min)
            ([x_range[0], x_range[1]], [y_range[0], y_range[0]], [z_range[0], z_range[0]]),
            ([x_range[1], x_range[1]], [y_range[0], y_range[1]], [z_range[0], z_range[0]]),
            ([x_range[1], x_range[0]], [y_range[1], y_range[1]], [z_range[0], z_range[0]]),
            ([x_range[0], x_range[0]], [y_range[1], y_range[0]], [z_range[0], z_range[0]]),
            # 頂面 4 條邊 (z = z_max)
            ([x_range[0], x_range[1]], [y_range[0], y_range[0]], [z_range[1], z_range[1]]),
            ([x_range[1], x_range[1]], [y_range[0], y_range[1]], [z_range[1], z_range[1]]),
            ([x_range[1], x_range[0]], [y_range[1], y_range[1]], [z_range[1], z_range[1]]),
            ([x_range[0], x_range[0]], [y_range[1], y_range[0]], [z_range[1], z_range[1]]),
            # 垂直 4 條邊
            ([x_range[0], x_range[0]], [y_range[0], y_range[0]], [z_range[0], z_range[1]]),
            ([x_range[1], x_range[1]], [y_range[0], y_range[0]], [z_range[0], z_range[1]]),
            ([x_range[1], x_range[1]], [y_range[1], y_range[1]], [z_range[0], z_range[1]]),
            ([x_range[0], x_range[0]], [y_range[1], y_range[1]], [z_range[0], z_range[1]]),
        ]

        # 繪製每條邊
        for i, (x, y, z) in enumerate(edges):
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(color="lightgreen", width=3),
                name="目標區域" if i == 0 else None,
                showlegend=(i == 0),
                hoverinfo="skip"
            ))

    fig.update_layout(
        title="3D Pareto 前沿<br><sub>灰色 = 非 Pareto | 藍色 = Pareto (未滿足) | 綠色 = Pareto (滿足) | 綠色框 = 目標區域</sub>",
        scene=dict(
            xaxis_title="準確率變化 (%)",
            yaxis_title="GPU 峰值變化 (%)",
            zaxis_title="延遲變化 (%)"
        ),
        height=700,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def create_trials_table(trials_data: Dict, targets: Dict = None, pareto_trial_ids: set = None) -> pd.DataFrame:
    """建立試驗表格

    Args:
        trials_data: 試驗資料
        targets: 目標閾值
        pareto_trial_ids: Pareto 前沿解的 trial_id 集合
    """
    trials = trials_data.get("trials", [])
    pareto_ids = pareto_trial_ids or set()

    # 目標值（用於漸層顏色）
    target_acc = targets.get('accuracy_min', -0.1) if targets else -0.1
    target_gpu = targets.get('gpu_peak_max', -0.5) if targets else -0.5
    target_lat = targets.get('latency_max', 1.5) if targets else 1.5

    table_data = []
    for i, t in enumerate(trials, 1):
        config = t.get("config", {})
        obj = t.get("objectives", {})
        trial_id = t.get("trial_id", i)

        acc = obj.get("accuracy_change")
        gpu = obj.get("gpu_peak_change")
        lat = obj.get("latency_change")

        # 使用輔助函數計算 violation_score（向下相容）
        vio = calculate_violation_score(obj, targets if targets else {})

        # 確保所有值都是字串以避免 Arrow 序列化問題
        bits = config.get("bits", config.get("w_bit", "-"))
        group_size = config.get("group_size", config.get("q_group_size", "-"))

        # 滿足目標欄位：如果是 Pareto 前沿解則加星號
        satisfies = t.get("satisfies_targets", False)
        is_pareto = trial_id in pareto_ids
        if satisfies and is_pareto:
            satisfy_str = "✅⭐"
        elif satisfies:
            satisfy_str = "✅"
        elif is_pareto:
            satisfy_str = "❌⭐"
        else:
            satisfy_str = "❌"

        table_data.append({
            "試驗": trial_id,
            "狀態": get_trial_status(t),
            "方法": config.get("method", "-"),
            "位元數": str(bits) if bits is not None else "-",
            "群組大小": str(group_size) if group_size is not None else "-",
            "準確率變化": f"{acc*100:+.2f}%" if acc is not None else "N/A",
            "GPU 變化": f"{gpu*100:+.2f}%" if gpu is not None else "N/A",
            "延遲變化": f"{lat*100:+.2f}%" if lat is not None else "N/A",
            "Violation Score": f"{vio:.4f}" if vio is not None else "N/A",
            "滿足目標": satisfy_str,
            # 原始值用於顏色判斷
            "_acc": acc,
            "_gpu": gpu,
            "_lat": lat,
            "_vio": vio,
            "_target_acc": target_acc,
            "_target_gpu": target_gpu,
            "_target_lat": target_lat
        })

    return pd.DataFrame(table_data)


def style_trials_table(df: pd.DataFrame):
    """為試驗表格添加漸層顏色樣式"""

    # 先重設索引確保連續
    df = df.reset_index(drop=True)

    # 取得顯示用欄位（不含 _ 開頭的內部欄位）
    display_cols = [c for c in df.columns if not c.startswith('_')]
    df_display = df[display_cols].copy()

    # 建立 styler
    styler = df_display.style

    # 準確率漸層：高 → 綠 (Greens)
    if '準確率變化' in df_display.columns and '_acc' in df.columns:
        styler = styler.background_gradient(
            subset=['準確率變化'],
            cmap='Greens',
            gmap=df['_acc']
        )

    # GPU 漸層：低 → 綠 (RdYlGn_r，反轉讓低值為綠)
    if 'GPU 變化' in df_display.columns and '_gpu' in df.columns:
        styler = styler.background_gradient(
            subset=['GPU 變化'],
            cmap='RdYlGn_r',
            gmap=df['_gpu']
        )

    # 延遲漸層：低 → 綠 (RdYlGn_r)
    if '延遲變化' in df_display.columns and '_lat' in df.columns:
        styler = styler.background_gradient(
            subset=['延遲變化'],
            cmap='RdYlGn_r',
            gmap=df['_lat']
        )

    # Violation Score 漸層：低 → 綠 (RdYlGn_r)
    if 'Violation Score' in df_display.columns and '_vio' in df.columns:
        styler = styler.background_gradient(
            subset=['Violation Score'],
            cmap='RdYlGn_r',
            gmap=df['_vio']
        )

    return styler


def create_correlation_heatmap(tradeoff_data: Dict) -> go.Figure:
    """建立相關性熱力圖"""
    correlations = tradeoff_data.get("correlations", {})

    objectives = ["準確率", "GPU 峰值", "延遲"]
    n = len(objectives)
    corr_matrix = np.eye(n)

    corr_map = {
        ("準確率", "GPU 峰值"): correlations.get("accuracy_vs_gpu_peak", {}).get("pearson_r", 0),
        ("準確率", "延遲"): correlations.get("accuracy_vs_latency", {}).get("pearson_r", 0),
        ("GPU 峰值", "延遲"): correlations.get("gpu_peak_vs_latency", {}).get("pearson_r", 0)
    }

    for i, obj1 in enumerate(objectives):
        for j, obj2 in enumerate(objectives):
            if i != j:
                key = (obj1, obj2) if (obj1, obj2) in corr_map else (obj2, obj1)
                corr_matrix[i, j] = corr_map.get(key, 0)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=objectives,
        y=objectives,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        text=[[f"{v:.3f}" for v in row] for row in corr_matrix],
        texttemplate="%{text}",
        textfont={"size": 14}
    ))

    fig.update_layout(
        title="目標間相關性熱力圖",
        height=400,
        width=500
    )

    return fig


def create_method_comparison_chart(all_trials: Dict) -> go.Figure:
    """建立量化方法比較圖表 - 2x2 子圖布局，含誤差條"""
    trials = all_trials.get("trials", [])
    completed = [t for t in trials if is_trial_completed(t) and t.get("objectives")]

    # 按方法分組統計
    method_stats = {}
    for t in completed:
        method = t["config"].get("method", "unknown")
        obj = t["objectives"]
        if method not in method_stats:
            method_stats[method] = {"acc": [], "gpu": [], "lat": [], "count": 0}
        method_stats[method]["acc"].append(obj.get("accuracy_change", 0) * 100)
        method_stats[method]["gpu"].append(obj.get("gpu_peak_change", 0) * 100)
        method_stats[method]["lat"].append(obj.get("latency_change", 0) * 100)
        method_stats[method]["count"] += 1

    if not method_stats:
        fig = go.Figure()
        fig.add_annotation(text="無已完成的試驗數據", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    # 計算平均值和標準差
    methods = list(method_stats.keys())
    avg_acc = [np.mean(method_stats[m]["acc"]) for m in methods]
    std_acc = [np.std(method_stats[m]["acc"]) for m in methods]
    avg_gpu = [np.mean(method_stats[m]["gpu"]) for m in methods]
    std_gpu = [np.std(method_stats[m]["gpu"]) for m in methods]
    avg_lat = [np.mean(method_stats[m]["lat"]) for m in methods]
    std_lat = [np.std(method_stats[m]["lat"]) for m in methods]
    counts = [method_stats[m]["count"] for m in methods]

    # 計算 y 軸範圍（考慮誤差條和文字標籤的空間）
    def calc_y_range(avg_vals, std_vals):
        """計算適當的 y 軸範圍，確保誤差條和文字不被截斷"""
        max_with_err = max([avg_vals[i] + std_vals[i] for i in range(len(avg_vals))])
        min_with_err = min([avg_vals[i] - std_vals[i] for i in range(len(avg_vals))])
        range_span = max_with_err - min_with_err
        # 上方預留 25% 空間給文字，下方預留 15% 空間
        y_max = max_with_err + range_span * 0.25
        y_min = min_with_err - range_span * 0.15
        return [y_min, y_max]
    
    y_range_acc = calc_y_range(avg_acc, std_acc)
    y_range_gpu = calc_y_range(avg_gpu, std_gpu)
    y_range_lat = calc_y_range(avg_lat, std_lat)
    
    # 試驗次數的 y 軸範圍
    max_count = max(counts)
    y_range_count = [0, max_count * 1.2]  # 上方預留 20% 空間

    # 方法顏色映射
    method_colors = {
        "gptq": "#636EFA",
        "awq": "#EF553B",
        "bnb": "#00CC96",
        "unknown": "#AB63FA"
    }
    colors = [method_colors.get(m, "#AB63FA") for m in methods]

    # 建立 2x2 子圖
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("平均準確率變化", "平均 GPU 峰值變化", "平均延遲變化", "試驗次數分佈"),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    # 子圖 1：準確率變化
    fig.add_trace(
        go.Bar(
            name='',  # 移除預設 trace 名稱
            x=methods, y=avg_acc,
            error_y=dict(type='data', array=std_acc, visible=True, color='rgba(245, 255, 170, 0.8)', thickness=2),
            marker_color=colors,
            text=[f"{v:.1f}%" for v in avg_acc],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=1)

    # 子圖 2：GPU 變化
    fig.add_trace(
        go.Bar(
            name='',  # 移除預設 trace 名稱
            x=methods, y=avg_gpu,
            error_y=dict(type='data', array=std_gpu, visible=True, color='rgba(245, 255, 170, 0.8)', thickness=2),
            marker_color=colors,
            text=[f"{v:.1f}%" for v in avg_gpu],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=2)

    # 子圖 3：延遲變化
    fig.add_trace(
        go.Bar(
            name='',  # 移除預設 trace 名稱
            x=methods, y=avg_lat,
            error_y=dict(type='data', array=std_lat, visible=True, color='rgba(245, 255, 170, 0.8)', thickness=2),
            marker_color=colors,
            text=[f"{v:.1f}%" for v in avg_lat],
            textposition='outside',
            showlegend=False
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=1)

    # 子圖 4：試驗次數分佈
    fig.add_trace(
        go.Bar(
            name='',  # 移除預設 trace 名稱
            x=methods, y=counts,
            marker_color=colors,
            text=[str(c) for c in counts],
            textposition='outside',
            showlegend=False
        ),
        row=2, col=2
    )

    # 更新軸標籤和範圍
    fig.update_yaxes(title_text="變化 (%)", row=1, col=1, range=y_range_acc)
    fig.update_yaxes(title_text="變化 (%)", row=1, col=2, range=y_range_gpu)
    fig.update_yaxes(title_text="變化 (%)", row=2, col=1, range=y_range_lat)
    fig.update_yaxes(title_text="試驗數", row=2, col=2, dtick=1, range=y_range_count)

    fig.update_layout(
        title="量化方法比較（平均值 ± 標準差）",
        height=600,
        showlegend=False
    )

    return fig


def create_parallel_coordinates(all_trials: Dict, targets: Dict = None, pareto_trial_ids: set = None) -> go.Figure:
    """建立平行座標圖 - 使用所有試驗資料"""
    trials = all_trials.get("trials", [])
    completed = [t for t in trials if is_trial_completed(t) and
                 t.get("objectives", {}).get("accuracy_change") is not None]

    if not completed:
        return None

    pareto_ids = pareto_trial_ids or set()

    # 準備資料
    acc_values = []
    gpu_saving = []
    lat_saving = []
    is_pareto = []
    violation_scores = []

    # 計算目標線位置（也需要取反）
    target_acc = (targets.get('accuracy_min', -0.1) * 100) if targets else -10
    target_gpu_saving = -(targets.get('gpu_peak_max', -0.5) * 100) if targets else 50  # 取反
    target_lat_saving = -(targets.get('latency_max', 1.5) * 100) if targets else -150  # 取反

    for t in completed:
        obj = t.get("objectives", {})
        trial_id = t.get("trial_id", 0)

        acc = obj.get("accuracy_change", 0) * 100
        gpu = -obj.get("gpu_peak_change", 0) * 100  # 取反：節省越多越好
        lat = -obj.get("latency_change", 0) * 100   # 取反：加速越多越好

        acc_values.append(acc)
        gpu_saving.append(gpu)
        lat_saving.append(lat)
        is_pareto.append(1 if trial_id in pareto_ids else 0)

        # 計算違反分數
        vio = t.get("violation_score")
        if vio is not None:
            violation_scores.append(vio)
        else:
            score = 0
            if acc < target_acc:
                score += abs(acc - target_acc) / 10
            if gpu < target_gpu_saving:
                score += abs(gpu - target_gpu_saving) / 10
            if lat < target_lat_saving:
                score += abs(lat - target_lat_saving) / 10
            violation_scores.append(score)

    max_violation = max(violation_scores) if violation_scores else 1

    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=violation_scores,
            colorscale=[
                [0.0, 'rgb(0, 128, 0)'],      # 深綠（低違反，好）
                [0.5, 'rgb(255, 140, 0)'],    # 深橙（中等）
                [1.0, 'rgb(220, 20, 60)']     # 深紅（高違反，差）
            ],
            showscale=True,
            cmin=0,
            cmax=max_violation if max_violation > 0 else 1,
            colorbar=dict(
                title="違反程度<br>(越低越好)",
                len=0.7
            )
        ),
        dimensions=[
            dict(
                range=[min(acc_values) - 5, max(acc_values) + 5],
                constraintrange=[target_acc, max(acc_values) + 5] if target_acc <= max(acc_values) else None,
                label='準確率變化 (%)',
                values=acc_values
            ),
            dict(
                range=[min(gpu_saving) - 5, max(gpu_saving) + 5],
                constraintrange=[target_gpu_saving, max(gpu_saving) + 5] if target_gpu_saving <= max(gpu_saving) else None,
                label='GPU 節省 (%)',
                values=gpu_saving
            ),
            dict(
                range=[min(lat_saving) - 5, max(lat_saving) + 5],
                constraintrange=[target_lat_saving, max(lat_saving) + 5] if target_lat_saving <= max(lat_saving) else None,
                label='延遲節省 (%)',
                values=lat_saving
            )
        ]
    ))

    # 計算目標區域說明文字
    target_info = (
        f"目標區域: 準確率 ≥ {target_acc:.1f}% | "
        f"GPU 節省 ≥ {target_gpu_saving:.1f}% | "
        f"延遲節省 ≥ {target_lat_saving:.1f}%"
    )

    fig.update_layout(
        title=f'所有試驗 - 平行座標圖<br><sub>{target_info}</sub>',
        height=550,
        margin=dict(l=50, r=5, t=150)
    )

    return fig


def create_tradeoff_matrix(all_trials: Dict, pareto_trial_ids: set = None) -> go.Figure:
    """建立權衡散點圖矩陣 - 使用所有試驗資料"""
    trials = all_trials.get("trials", [])
    completed = [t for t in trials if is_trial_completed(t) and
                 t.get("objectives", {}).get("accuracy_change") is not None]

    if not completed:
        return None

    pareto_ids = pareto_trial_ids or set()

    # 分類資料
    pareto_acc, pareto_gpu, pareto_lat, pareto_methods, pareto_colors = [], [], [], [], []
    other_acc, other_gpu, other_lat, other_methods = [], [], [], []

    for t in completed:
        obj = t.get("objectives", {})
        config = t.get("config", {})
        trial_id = t.get("trial_id", 0)

        acc = obj.get("accuracy_change", 0) * 100
        gpu = obj.get("gpu_peak_change", 0) * 100
        lat = obj.get("latency_change", 0) * 100
        method = config.get("method", "?")
        satisfies = t.get("satisfies_targets", False)

        if trial_id in pareto_ids:
            pareto_acc.append(acc)
            pareto_gpu.append(gpu)
            pareto_lat.append(lat)
            pareto_methods.append(method)
            pareto_colors.append("green" if satisfies else "blue")
        else:
            other_acc.append(acc)
            other_gpu.append(gpu)
            other_lat.append(lat)
            other_methods.append(method)

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "準確率 vs GPU", "準確率 vs 延遲", "GPU vs 延遲", "統計摘要"
    ))

    # 繪製非 Pareto 解（灰色背景點）
    if other_acc:
        fig.add_trace(go.Scatter(
            x=other_acc, y=other_gpu, mode="markers",
            marker=dict(color="lightgray", size=8, opacity=0.5),
            name="非 Pareto", showlegend=True
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=other_acc, y=other_lat, mode="markers",
            marker=dict(color="lightgray", size=8, opacity=0.5),
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=other_gpu, y=other_lat, mode="markers",
            marker=dict(color="lightgray", size=8, opacity=0.5),
            showlegend=False
        ), row=2, col=1)

    # 繪製 Pareto 解（彩色前景點）
    if pareto_acc:
        fig.add_trace(go.Scatter(
            x=pareto_acc, y=pareto_gpu, mode="markers+text", text=pareto_methods,
            marker=dict(color=pareto_colors, size=10), textposition="top center",
            name="Pareto ⭐", showlegend=True
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=pareto_acc, y=pareto_lat, mode="markers+text", text=pareto_methods,
            marker=dict(color=pareto_colors, size=10), textposition="top center",
            showlegend=False
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=pareto_gpu, y=pareto_lat, mode="markers+text", text=pareto_methods,
            marker=dict(color=pareto_colors, size=10), textposition="top center",
            showlegend=False
        ), row=2, col=1)

    # 統計摘要
    all_acc = pareto_acc + other_acc
    all_gpu = pareto_gpu + other_gpu
    all_lat = pareto_lat + other_lat
    n_satisfies = sum(1 for t in completed if t.get("satisfies_targets"))

    stats_text = f"總試驗數: {len(completed)}<br>"
    stats_text += f"Pareto 解數: {len(pareto_acc)}<br>"
    stats_text += f"滿足目標: {n_satisfies}<br>"
    stats_text += f"準確率範圍: [{min(all_acc):.1f}%, {max(all_acc):.1f}%]<br>"
    stats_text += f"GPU 範圍: [{min(all_gpu):.1f}%, {max(all_gpu):.1f}%]<br>"
    stats_text += f"延遲範圍: [{min(all_lat):.1f}%, {max(all_lat):.1f}%]"

    fig.add_trace(go.Scatter(
        x=[0.5], y=[0.5], mode="text", text=[stats_text],
        textfont=dict(size=12), showlegend=False
    ), row=2, col=2)

    fig.update_xaxes(title_text="準確率 (%)", row=1, col=1)
    fig.update_yaxes(title_text="GPU (%)", row=1, col=1)
    fig.update_xaxes(title_text="準確率 (%)", row=1, col=2)
    fig.update_yaxes(title_text="延遲 (%)", row=1, col=2)
    fig.update_xaxes(title_text="GPU (%)", row=2, col=1)
    fig.update_yaxes(title_text="延遲 (%)", row=2, col=1)
    fig.update_xaxes(visible=False, row=2, col=2)
    fig.update_yaxes(visible=False, row=2, col=2)

    fig.update_layout(title="權衡散點圖矩陣", height=600, showlegend=False)
    return fig


def create_radar_chart(scored_data: Dict) -> go.Figure:
    """建立維度評分雷達圖（以 0 為圓心，顯示超越目標的幅度）"""
    if not scored_data:
        return None

    ranked_trials = scored_data.get("ranked_trials", [])
    if not ranked_trials:
        return None

    fig = go.Figure()
    categories = ["準確率", "GPU 峰值", "延遲"]
    colors = px.colors.qualitative.Set2

    # 收集所有 surplus 值以計算動態範圍
    all_surpluses = []
    for trial in ranked_trials[:5]:
        dim_scores = trial.get("dimension_scores", {})
        for dim in ["accuracy", "gpu_peak", "latency"]:
            surplus = dim_scores.get(dim, {}).get("surplus", 0)
            all_surpluses.append(surplus)

    # 動態範圍（基於實際數據，確保 0 在範圍內）
    if all_surpluses:
        max_surplus = max(all_surpluses) if all_surpluses else 0.5
        min_surplus = min(all_surpluses) if all_surpluses else 0
        # 確保範圍包含 0 且有適當的邊距
        range_max = max(0.1, max_surplus * 1.2)
        range_min = min(0, min_surplus * 1.2) if min_surplus < 0 else 0
    else:
        range_max = 0.5
        range_min = 0

    # 顯示前 5 個試驗
    for i, trial in enumerate(ranked_trials[:5]):
        dim_scores = trial.get("dimension_scores", {})
        # 使用 surplus 值（超越目標的幅度）
        values = [
            dim_scores.get("accuracy", {}).get("surplus", 0),
            dim_scores.get("gpu_peak", {}).get("surplus", 0),
            dim_scores.get("latency", {}).get("surplus", 0)
        ]
        # 閉合雷達圖
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]

        config = trial.get("config", {})
        label = format_config_summary(config)
        is_pareto = trial.get("is_pareto", False)
        pareto_mark = " ⭐" if is_pareto else ""
        color = colors[i % len(colors)]
        rgba_color = color.replace("rgb", "rgba").replace(")", ", 0.4)")

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            # fill="toself",
            fillcolor=rgba_color,
            line=dict(color=color, width=2.5),
            name=f"#{i+1} {label}{pareto_mark}"
        ))

    # 計算刻度間隔
    tick_interval = (range_max - range_min) / 5

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[range_min, range_max],
                tickformat=".2f",
                tickfont=dict(size=11, color="#333"),
                dtick=tick_interval
            ),
            angularaxis=dict(
                tickfont=dict(size=13),
            )
        ),
        title="維度評分雷達圖<br><sub>0 = 恰好滿足目標 | >0 = 超越目標 | ⭐ = Pareto 前沿</sub>",
        height=650,
        showlegend=True
    )

    return fig


def create_cluster_3d_plot(analysis_data: Dict) -> go.Figure:
    """建立 Pareto 聚類 3D 視覺化"""
    clusters = analysis_data.get("pareto_clusters", {})
    if not clusters:
        return None

    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for i, (cluster_name, cluster_info) in enumerate(clusters.items()):
        centroid = cluster_info.get("centroid", {})
        label = cluster_info.get("label", cluster_name)
        size = cluster_info.get("size", 0)
        color = colors[i % len(colors)]

        # 繪製聚類中心（菱形）
        fig.add_trace(go.Scatter3d(
            x=[centroid.get("accuracy_change", 0) * 100],
            y=[centroid.get("gpu_peak_change", 0) * 100],
            z=[centroid.get("latency_change", 0) * 100],
            mode="markers+text",
            marker=dict(size=15, color=color, symbol="diamond"),
            text=[f"{label} (n={size})"],
            textposition="top center",
            name=f"{cluster_name}: {label}",
            hovertemplate=f"<b>{label}</b><br>準確率: %{{x:+.2f}}%<br>GPU: %{{y:+.2f}}%<br>延遲: %{{z:+.2f}}%<extra></extra>"
        ))

    fig.update_layout(
        title="Pareto 聚類視覺化<br><sub>菱形 = 聚類中心</sub>",
        scene=dict(
            xaxis_title="準確率變化 (%)",
            yaxis_title="GPU 峰值變化 (%)",
            zaxis_title="延遲變化 (%)"
        ),
        height=500
    )

    return fig


def create_strategy_comparison_chart(strategies: Dict, targets: Dict = None) -> go.Figure:
    """建立策略對比條形圖 - 分開三個子圖"""
    if not strategies:
        return None

    strategy_names = {
        "closest_to_target": "最接近目標",
        "best_accuracy": "最佳準確率",
        "best_compression": "最佳壓縮",
        "balanced": "平衡策略"
    }

    names = []
    acc_values = []
    gpu_values = []
    lat_values = []

    for key, info in strategies.items():
        if info and info.get("objectives"):
            obj = info["objectives"]
            names.append(strategy_names.get(key, key))
            acc_values.append(obj.get("accuracy_change", 0) * 100)
            gpu_values.append(obj.get("gpu_peak_change", 0) * 100)
            lat_values.append(obj.get("latency_change", 0) * 100)

    if not names:
        return None

    # 使用 make_subplots 分開畫三個子圖
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('準確率變化', 'GPU 峰值變化', '延遲變化')
    )

    # 準確率（目標：≥ -10%）
    target_acc = (targets.get('accuracy_min', -0.1) * 100) if targets else -10
    colors_acc = ['green' if v >= target_acc else 'red' for v in acc_values]
    fig.add_trace(go.Bar(
        x=names, y=acc_values,
        marker_color=colors_acc,
        text=[f"{v:+.1f}%" for v in acc_values],
        textposition='outside',
        name='準確率',
        showlegend=False
    ), row=1, col=1)

    # GPU（目標：≤ -50%）
    target_gpu = (targets.get('gpu_peak_max', -0.5) * 100) if targets else -50
    colors_gpu = ['green' if v <= target_gpu else 'orange' for v in gpu_values]
    fig.add_trace(go.Bar(
        x=names, y=gpu_values,
        marker_color=colors_gpu,
        text=[f"{v:+.1f}%" for v in gpu_values],
        textposition='outside',
        name='GPU',
        showlegend=False
    ), row=1, col=2)

    # 延遲（目標：≤ +150%）
    target_lat = (targets.get('latency_max', 1.5) * 100) if targets else 150
    colors_lat = ['green' if v <= target_lat else 'red' for v in lat_values]
    fig.add_trace(go.Bar(
        x=names, y=lat_values,
        marker_color=colors_lat,
        text=[f"{v:+.1f}%" for v in lat_values],
        textposition='outside',
        name='延遲',
        showlegend=False
    ), row=1, col=3)

    # 添加目標線
    fig.add_hline(y=target_acc, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1,
                  annotation_text=f"目標 ≥{target_acc:.0f}%", annotation_position="bottom right")
    fig.add_hline(y=target_gpu, line_dash="dash", line_color="red", opacity=0.5, row=1, col=2,
                  annotation_text=f"目標 ≤{target_gpu:.0f}%", annotation_position="bottom right")
    fig.add_hline(y=target_lat, line_dash="dash", line_color="red", opacity=0.5, row=1, col=3,
                  annotation_text=f"目標 ≤{target_lat:.0f}%", annotation_position="top right")

    fig.update_layout(
        title=dict(
            text="四種推薦策略的目標值對比<br><sub>綠色 = 滿足目標 | 紅色/橙色 = 未滿足目標</sub>",
            x=0.5,
            xanchor='center'
        ),
        height=450,
        showlegend=False
    )

    fig.update_yaxes(title_text="變化 (%)", row=1, col=1)
    fig.update_yaxes(title_text="變化 (%)", row=1, col=2)
    fig.update_yaxes(title_text="變化 (%)", row=1, col=3)

    return fig


def create_scenario_comparison_chart(recommendations: Dict, targets: Dict = None) -> go.Figure:
    """建立場景推薦效果對比圖（三張並排圖表）

    Args:
        recommendations: 推薦資料字典
        targets: 目標設定（用於繪製目標線）
    """
    if not recommendations:
        return None

    scenario_names = {
        "for_production": "生產環境",
        "for_memory_critical": "記憶體關鍵",
        "for_accuracy_critical": "準確率關鍵"
    }

    # 場景顏色
    scenario_colors = {
        "for_production": "#636EFA",       # 藍色
        "for_memory_critical": "#EF553B",  # 紅色
        "for_accuracy_critical": "#00CC96" # 綠色
    }

    # 收集資料
    scenarios_data = []
    for key, info in recommendations.items():
        if info and info.get("objectives"):
            obj = info["objectives"]
            trial_id = info.get("recommended_trial_id", "?")
            scenarios_data.append({
                "key": key,
                "name": scenario_names.get(key, key),
                "trial_id": trial_id,
                "acc": obj.get("accuracy_change", 0) * 100,
                "gpu": obj.get("gpu_peak_change", 0) * 100,
                "lat": obj.get("latency_change", 0) * 100,
                "color": scenario_colors.get(key, "#888888")
            })

    if not scenarios_data:
        return None

    # 建立三張並排圖表
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("準確率變化 (%)", "GPU 變化 (%)", "延遲變化 (%)")
    )

    # 目標值
    target_acc = (targets.get('accuracy_min', -0.3) * 100) if targets else -30
    target_gpu = (targets.get('gpu_peak_max', -0.4) * 100) if targets else -40
    target_lat = (targets.get('latency_max', 1.0) * 100) if targets else 100

    # 準備資料
    names = [f"{s['name']}<br>(Trial {s['trial_id']})" for s in scenarios_data]
    acc_values = [s["acc"] for s in scenarios_data]
    gpu_values = [s["gpu"] for s in scenarios_data]
    lat_values = [s["lat"] for s in scenarios_data]
    colors = [s["color"] for s in scenarios_data]

    # 計算 y 軸範圍以避免文字被截斷
    def calc_y_range_with_target(values, target):
        """計算包含目標線的 y 軸範圍"""
        all_vals = values + [target]
        y_max = max(all_vals)
        y_min = min(all_vals)
        range_span = y_max - y_min
        if range_span == 0:
            range_span = abs(y_max) if y_max != 0 else 1
        return [y_min - range_span * 0.2, y_max + range_span * 0.25]
    
    y_range_acc = calc_y_range_with_target(acc_values, target_acc)
    y_range_gpu = calc_y_range_with_target(gpu_values, target_gpu)
    y_range_lat = calc_y_range_with_target(lat_values, target_lat)

    # 第一張圖：準確率變化
    fig.add_trace(
        go.Bar(
            name='',
            x=names, y=acc_values,
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in acc_values],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )
    # 準確率目標線
    fig.add_hline(
        y=target_acc, row=1, col=1,
        line=dict(color="red", dash="dash", width=2),
        annotation_text=f"目標: {target_acc:+.0f}% ↑",
        annotation_position="bottom right"
    )

    # 第二張圖：GPU 變化
    fig.add_trace(
        go.Bar(
            name='',
            x=names, y=gpu_values,
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in gpu_values],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    # GPU 目標線
    fig.add_hline(
        y=target_gpu, row=1, col=2,
        line=dict(color="red", dash="dash", width=2),
        annotation_text=f"目標: {target_gpu:+.0f}% ↓",
        annotation_position="bottom right"
    )

    # 第三張圖：延遲變化
    fig.add_trace(
        go.Bar(
            name='',
            x=names, y=lat_values,
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in lat_values],
            textposition='outside',
            showlegend=False
        ),
        row=1, col=3
    )
    # 延遲目標線
    fig.add_hline(
        y=target_lat, row=1, col=3,
        line=dict(color="red", dash="dash", width=2),
        annotation_text=f"目標: {target_lat:+.0f}% ↓",
        annotation_position="bottom right"
    )

    # 更新佈局
    fig.update_layout(
        title="三種場景推薦配置效果對比<br><sub>負值 = 減少/節省 | 正值 = 增加 | 紅色虛線 = 目標界線</sub>",
        height=450,
        showlegend=False,
        margin=dict(t=120)
    )

    fig.update_yaxes(title_text="變化 (%)", row=1, col=1, range=y_range_acc)
    fig.update_yaxes(title_text="變化 (%)", row=1, col=2, range=y_range_gpu)
    fig.update_yaxes(title_text="變化 (%)", row=1, col=3, range=y_range_lat)

    return fig


# ==========================================
# 頁面組件
# ==========================================

def render_summary_tab(data: Dict):
    """渲染實驗摘要標籤頁"""
    summary = data.get("summary", {})

    # 基本資訊
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("實驗名稱", summary.get("experiment_name", "N/A"))

    with col2:
        opt_info = summary.get("optimization", {})
        st.metric("優化器", opt_info.get("optimizer", "N/A"))

    with col3:
        st.metric("時間戳記", summary.get("timestamp", "N/A"))

    st.divider()

    # 統計資訊
    col1, col2, col3, col4 = st.columns(4)

    opt_info = summary.get("optimization", {})
    with col1:
        st.metric("總試驗數", opt_info.get("total_trials", 0))
    with col2:
        st.metric("Pareto 解", opt_info.get("pareto_solutions", 0))
    with col3:
        st.metric("滿足目標", opt_info.get("satisfying_solutions", 0))
    with col4:
        total = opt_info.get("total_trials", 1)
        satisfying = opt_info.get("satisfying_solutions", 0)
        rate = satisfying / total * 100 if total > 0 else 0
        st.metric("成功率", f"{rate:.1f}%")

    st.divider()

    # 基準模型
    st.subheader("基準模型")
    baseline = summary.get("baseline", {})
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write(f"**模型**: {baseline.get('model', 'N/A')}")
    with col2:
        st.write(f"**準確率**: {baseline.get('accuracy', 0)*100:.2f}%")
    with col3:
        st.write(f"**GPU 峰值**: {baseline.get('gpu_peak_mb', 0):.0f} MB")
    with col4:
        st.write(f"**平均延遲**: {baseline.get('avg_latency_ms', 0):.2f} ms")

    # 優化目標
    st.subheader("優化目標")
    targets = summary.get("targets", {})
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"**準確率**: ≥ {targets.get('accuracy_min', 0)*100:+.1f}%")
    with col2:
        st.write(f"**GPU 峰值**: ≤ {targets.get('gpu_peak_max', 0)*100:+.1f}%")
    with col3:
        st.write(f"**延遲**: ≤ {targets.get('latency_max', 0)*100:+.1f}%")

    # 推薦配置
    recommended = summary.get("recommended", {})
    if recommended and recommended.get("config"):
        st.subheader("推薦配置")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**配置參數**")
            config = recommended.get("config", {})
            for key, value in config.items():
                st.write(f"- {key}: `{value}`")

        with col2:
            st.write("**預期效果**")
            obj = recommended.get("objectives", {})
            st.write(f"- 準確率變化: `{obj.get('accuracy_change', 0)*100:+.2f}%`")
            st.write(f"- GPU 峰值變化: `{obj.get('gpu_peak_change', 0)*100:+.2f}%`")
            st.write(f"- 延遲變化: `{obj.get('latency_change', 0)*100:+.2f}%`")

            if recommended.get("satisfies_targets"):
                st.success("✓ 滿足所有目標")
            else:
                st.warning("✗ 未滿足所有目標")


def render_trials_tab(data: Dict):
    """渲染所有試驗標籤頁"""
    all_trials = data.get("all_trials", {})
    summary = data.get("summary", {})
    targets = summary.get("targets", {})
    pareto = data.get("pareto_frontier", {})

    if not all_trials.get("trials"):
        st.warning("沒有試驗資料")
        return

    # 獲取 Pareto 前沿解的 trial_id 集合
    # 由於 pareto_frontier.json 可能沒有 trial_id，需要透過比較配置來判斷
    pareto_solutions = get_pareto_solutions(pareto)

    def config_key(config: Dict) -> str:
        """建立配置的唯一識別 key"""
        method = config.get("method", "")
        # 取得 params 或直接從 config 取得參數
        params = config.get("params", config)
        bits = params.get("bits", params.get("w_bit", ""))
        group_size = params.get("group_size", params.get("q_group_size", ""))
        # 組合主要參數作為 key
        return f"{method}_{bits}_{group_size}_{str(sorted(params.items()))}"

    # 建立 Pareto 解的配置 key 集合
    pareto_config_keys = set()
    for sol in pareto_solutions:
        sol_config = sol.get("config", {})
        pareto_config_keys.add(config_key(sol_config))

    # 找出 all_trials 中對應的 trial_id
    trials = all_trials["trials"]
    pareto_trial_ids = set()
    for i, t in enumerate(trials, 1):
        trial_config = t.get("config", {})
        trial_id = t.get("trial_id", i)
        if config_key(trial_config) in pareto_config_keys:
            pareto_trial_ids.add(trial_id)

    # 統計（使用兼容函數）
    completed = len([t for t in trials if get_trial_status(t) == "completed"])
    failed = len([t for t in trials if get_trial_status(t) == "failed"])
    pruned = len([t for t in trials if get_trial_status(t) == "pruned"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("總試驗數", len(trials))
    with col2:
        st.metric("完成", completed)
    with col3:
        st.metric("失敗", failed)
    with col4:
        st.metric("剪枝", pruned)

    

    # 篩選器
    st.subheader("📋 試驗列表")

    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.multiselect(
            "狀態篩選",
            ["completed", "failed", "pruned"],
            default=["completed"]
        )
    with col2:
        # 方法篩選
        methods = list(set(t.get("config", {}).get("method", "unknown") for t in trials))
        method_filter = st.multiselect(
            "量化方法篩選",
            methods,
            default=methods
        )

    # 表格（帶顏色），並標記 Pareto 前沿解
    df = create_trials_table(all_trials, targets, pareto_trial_ids)

    # 顯示圖例
    st.caption("⭐ = Pareto 前沿解")

    if status_filter:
        df = df[df["狀態"].isin(status_filter)]
    if method_filter:
        df = df[df["方法"].isin(method_filter)]

    # 使用帶顏色的 styled dataframe
    styled_df = style_trials_table(df)
    st.dataframe(styled_df, width="stretch", hide_index=True)

    st.divider()

    # ==========================================
    # 試驗詳細參數 - 直接列出所有試驗
    # ==========================================
    st.subheader("📋 試驗詳細參數")

    completed_trials = [t for t in trials if is_trial_completed(t)]
    if not completed_trials:
        st.info("沒有完成的試驗可以檢視")
        return

    st.write(f"共 {len(completed_trials)} 個完成的試驗，點擊展開查看詳細參數：")

    # 直接列出所有試驗（使用 expander）
    for i, trial in enumerate(trials):
        if not is_trial_completed(trial):
            continue
        config = trial.get("config", {})
        obj = trial.get("objectives", {})
        method = config.get("method", "unknown")
        satisfies = trial.get("satisfies_targets", False)
        trial_id = trial.get("trial_id", i + 1)

        # 標題包含關鍵資訊
        acc = obj.get("accuracy_change", 0) * 100
        gpu = obj.get("gpu_peak_change", 0) * 100
        lan = obj.get("latency_change", 0) * 100
        emoji = "✅" if satisfies else "❌"

        title = f"{emoji} Trial {trial_id}: {format_config_summary(config)} | 準確率: {acc:+.1f}% | GPU: {gpu:+.1f}% | 延遲: {lan:+.1f}%"

        with st.expander(title):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**📊 評估結果**")
                st.write(f"- 準確率變化: `{obj.get('accuracy_change', 0)*100:+.2f}%`")
                st.write(f"- GPU 峰值變化: `{obj.get('gpu_peak_change', 0)*100:+.2f}%`")
                st.write(f"- 延遲變化: `{obj.get('latency_change', 0)*100:+.2f}%`")
                st.write(f"- 滿足目標: `{'是 ✅' if satisfies else '否 ❌'}`")

            with col2:
                st.write(f"**⚙️ {method.upper()} 量化參數**")
                display_config_details(config, method)

    st.divider()

    # 量化方法比較圖
    st.subheader("📊 量化方法比較")
    fig_method = create_method_comparison_chart(all_trials)
    if fig_method:
        st.plotly_chart(fig_method, width="stretch")

    st.divider()

    # ==========================================
    # 視覺化分析（3D 圖、平行座標圖、權衡散點圖）- 使用所有試驗
    # ==========================================
    st.subheader("📊 視覺化分析")
    st.caption("所有圖表使用全部試驗繪製。灰色 = 非 Pareto | 藍色 = Pareto (未滿足目標) | 綠色 = Pareto (滿足目標)")

    # 3D Pareto 前沿圖（使用所有試驗）
    st.write("**3D Pareto 前沿圖**")
    if all_trials.get("trials"):
        fig_3d = create_3d_pareto_plot(pareto, all_trials, targets)
        st.plotly_chart(fig_3d, width="stretch")
    else:
        st.info("沒有足夠的資料繪製 3D 圖")

    # 平行座標圖（使用所有試驗）
    st.write("**平行座標圖**")
    if all_trials.get("trials"):
        fig_parallel = create_parallel_coordinates(all_trials, targets, pareto_trial_ids)
        if fig_parallel:
            st.plotly_chart(fig_parallel, width="stretch")
    else:
        st.info("沒有試驗資料")

    # 權衡散點圖矩陣（使用所有試驗）
    st.write("**權衡散點圖矩陣**")
    if all_trials.get("trials"):
        fig_tradeoff = create_tradeoff_matrix(all_trials, pareto_trial_ids)
        if fig_tradeoff:
            st.plotly_chart(fig_tradeoff, width="stretch")
    else:
        st.info("沒有試驗資料")

    st.divider()

    # ==========================================
    # 多試驗比較
    # ==========================================
    st.subheader("🔄 多試驗比較")

    # 建立選項
    trial_options = []
    trial_ids = []  # 保存真正的 trial_id
    for i, t in enumerate(completed_trials):
        config = t.get("config", {})
        obj = t.get("objectives", {})
        trial_id = t.get("trial_id", i + 1)
        trial_ids.append(trial_id)
        acc = obj.get("accuracy_change", 0) * 100 if obj.get("accuracy_change") is not None else 0
        trial_options.append(f"Trial {trial_id}: {format_config_summary(config)} ({acc:+.1f}%)")

    selected_indices = st.multiselect(
        "選擇要比較的試驗（可多選）",
        range(len(trial_options)),
        format_func=lambda x: trial_options[x],
        default=list(range(min(3, len(trial_options))))  # 預設選前3個
    )

    if len(selected_indices) >= 2:
        # 建立比較表格
        compare_data = {"項目": ["方法", "位元數", "群組大小", "準確率變化", "GPU 變化", "延遲變化","Violation Score", "滿足目標"]}

        for idx in selected_indices:
            trial = completed_trials[idx]
            config = trial.get("config", {})
            obj = trial.get("objectives", {})

            bits = config.get("bits", config.get("w_bit", "-"))
            group_size = config.get("group_size", config.get("q_group_size", "-"))

            # 計算 violation_score（向下相容）
            vio = calculate_violation_score(obj, targets)
            trial_id = trial_ids[idx]

            compare_data[f"Trial {trial_id}"] = [
                config.get("method", "-"),
                str(bits),
                str(group_size),
                f"{obj.get('accuracy_change', 0)*100:+.2f}%",
                f"{obj.get('gpu_peak_change', 0)*100:+.2f}%",
                f"{obj.get('latency_change', 0)*100:+.2f}%",
                f"{vio:.4f}" if vio is not None else "N/A",
                "✅" if trial.get("satisfies_targets") else "❌"
            ]

        df_compare = pd.DataFrame(compare_data)
        st.dataframe(df_compare, width="stretch", hide_index=True)

        # 視覺化比較
        st.write("**視覺化比較：**")
        fig_compare = go.Figure()

        all_values = []  # 收集所有數值以計算 y 軸範圍
        
        for idx in selected_indices:
            trial = completed_trials[idx]
            obj = trial.get("objectives", {})
            config = trial.get("config", {})

            # 計算 violation_score（向下相容）
            vio = calculate_violation_score(obj, targets)

            values = [
                obj.get("accuracy_change", 0),
                obj.get("gpu_peak_change", 0),
                obj.get("latency_change", 0),
                vio if vio is not None else 0
            ]
            all_values.extend(values)

            trial_id = trial_ids[idx]
            fig_compare.add_trace(go.Bar(
                name=f"Trial {trial_id}: {format_config_summary(config)}",
                x=["準確率變化", "GPU 變化", "延遲變化", "Violation Score"],
                y=values,
                text=[f"{v:.2f}" for v in values],
                textposition='outside'
            ))

        # 計算 y 軸範圍，避免文字被截斷
        y_max = max(all_values) if all_values else 1
        y_min = min(all_values) if all_values else 0
        y_range_span = y_max - y_min
        y_axis_max = y_max + y_range_span * 0.2  # 上方預留 20%
        y_axis_min = y_min - y_range_span * 0.15  # 下方預留 15%
        
        fig_compare.update_layout(
            title="選中試驗的指標比較",
            barmode="group",
            yaxis_title="變化百分比",
            yaxis=dict(range=[y_axis_min, y_axis_max]),
            height=400
        )
        st.plotly_chart(fig_compare, width="stretch")

    elif len(selected_indices) == 1:
        st.info("請至少選擇 2 個試驗進行比較")
    else:
        st.info("請選擇要比較的試驗")


def render_satisfying_trials_tab(data: Dict):
    """渲染滿足目標的試驗評分標籤頁（來自 satisfying_trials_scored.json）"""
    scored = data.get("satisfying_trials_scored", {})

    if not scored:
        st.warning("沒有滿足目標的試驗評分資料")
        return

    ranked_trials = scored.get("ranked_trials", [])
    if not ranked_trials:
        st.info("沒有滿足目標的試驗")
        return

    # 統計資訊
    st.subheader("📊 評分統計")
    statistics = scored.get("statistics", {})
    pareto_overlap = scored.get("pareto_overlap", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("滿足目標試驗數", scored.get("total_satisfying_trials", len(ranked_trials)))
    with col2:
        st.metric("Pareto 前沿重疊", pareto_overlap.get("count", 0))
    with col3:
        score_range = statistics.get("score_range", {})
        st.metric("最高分", f"{score_range.get('max', 0):.4f}")
    with col4:
        st.metric("平均分", f"{statistics.get('avg_overall_score', 0):.4f}")

    st.divider()

    # 評分表格
    st.subheader("📋 滿足目標的試驗評分")

    # 顯示 weights 資訊
    scoring_details = scored.get("scoring_details", {})
    weights = scoring_details.get("weights", {"accuracy": 2.0, "gpu_peak": 1.0, "latency": 1.0})
    st.markdown(
        f"**評分權重:** 準確率=`{weights.get('accuracy', 2.0)}`, "
        f"GPU=`{weights.get('gpu_peak', 1.0)}`, "
        f"延遲=`{weights.get('latency', 1.0)}`  |  "
        f"**公式:** `總分 = Σ(weight × surplus)`"
        f"<br/>**目標:** 準確率≥`{scoring_details.get('accuracy_target', -0.1)*100:+.1f}%`, "
        f"GPU≤`{scoring_details.get('gpu_peak_target', -0.5)*100:+.1f}%`, "
        f"延遲≤`{scoring_details.get('latency_target', 1.5)*100:+.1f}%`",
        unsafe_allow_html=True
    )

    # 顯示 Pareto 重疊統計
    st.caption(f"⭐ {pareto_overlap.get('count', 0)}/{len(ranked_trials)} 個試驗同時為 Pareto 前沿解")

    score_data = []
    for i, trial in enumerate(ranked_trials):
        config = trial.get("config", {})
        dim_scores = trial.get("dimension_scores", {})
        is_pareto = trial.get("is_pareto", False)
        trial_id = trial.get("trial_id", i)

        acc_score = dim_scores.get("accuracy", {}).get("score", 0)
        gpu_score = dim_scores.get("gpu_peak", {}).get("score", 0)
        lat_score = dim_scores.get("latency", {}).get("score", 0)
        overall = trial.get('overall_score', 0)

        score_data.append({
            "排名": i + 1,
            "Trial ID": trial_id,
            "Pareto": "⭐" if is_pareto else "",
            "配置": format_config_summary(config),
            "總分": overall,
            "準確率": acc_score,
            "GPU": gpu_score,
            "延遲": lat_score
        })
    df_scores = pd.DataFrame(score_data)

    # 漸層上色函數
    def score_gradient(val, col_name):
        """根據分數值生成漸層背景色"""
        if col_name in ["總分", "準確率", "GPU", "延遲"]:
            if isinstance(val, (int, float)):
                if val >= 0:
                    intensity = min(val * 100, 100)
                    return f"background-color: rgba(76, 175, 80, {intensity/100 * 0.5})"
                else:
                    intensity = min(abs(val) * 100, 100)
                    return f"background-color: rgba(244, 67, 54, {intensity/100 * 0.5})"
        return ""

    # 應用樣式
    styled_df = df_scores.style.map(
        lambda x: score_gradient(x, "總分") if df_scores.columns.get_loc("總分") else "",
        subset=["總分"]
    ).map(
        lambda x: score_gradient(x, "準確率") if df_scores.columns.get_loc("準確率") else "",
        subset=["準確率"]
    ).map(
        lambda x: score_gradient(x, "GPU") if df_scores.columns.get_loc("GPU") else "",
        subset=["GPU"]
    ).map(
        lambda x: score_gradient(x, "延遲") if df_scores.columns.get_loc("延遲") else "",
        subset=["延遲"]
    ).format({
        "總分": "{:.4f}",
        "準確率": "{:.4f}",
        "GPU": "{:.4f}",
        "延遲": "{:.4f}"
    })

    st.dataframe(styled_df, width="stretch", hide_index=True)

    st.divider()

    # 所有滿足目標試驗的詳細參數
    st.subheader("🎯 滿足目標試驗的詳細參數")
    st.write(f"共 {len(ranked_trials)} 個滿足目標的試驗：")

    for i, trial in enumerate(ranked_trials):
        config = trial.get("config", {})
        obj = trial.get("objectives", {})
        dim_scores = trial.get("dimension_scores", {})
        method = config.get("method", "unknown")
        is_pareto = trial.get("is_pareto", False)
        trial_id = trial.get("trial_id", i)

        acc = obj.get("accuracy_change", 0) * 100
        gpu = obj.get("gpu_peak_change", 0) * 100
        pareto_mark = " ⭐" if is_pareto else ""

        title = f"✅ 排名 {i+1} (Trial {trial_id}): {format_config_summary(config)} | 總分: {trial.get('overall_score', 0):.4f}{pareto_mark}"

        with st.expander(title, expanded=(i == 0)):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**📊 準確率**")
                acc_info = dim_scores.get("accuracy", {})
                st.write(f"變化: `{acc:+.2f}%`")
                st.write(f"- 目標: ≥ `{acc_info.get('target', -0.1)*100:+.1f}%`")
                st.write(f"- 超越幅度: `{acc_info.get('surplus', 0)*100:+.2f}%`")
                st.write(f"- 加權分數: `{acc_info.get('score', 0):.4f}`")

            with col2:
                gpu_info = dim_scores.get("gpu_peak", {})
                st.write("**📊 GPU 峰值**")
                st.write(f"變化: `{gpu:+.2f}%`")
                st.write(f"- 目標: ≤ `{gpu_info.get('target', -0.5)*100:+.1f}%`")
                st.write(f"- 超越幅度: `{gpu_info.get('surplus', 0)*100:+.2f}%`")
                st.write(f"- 加權分數: `{gpu_info.get('score', 0):.4f}`")

            with col3:
                lat_info = dim_scores.get("latency", {})
                lat = obj.get("latency_change", 0) * 100
                st.write("**📊 延遲**")
                st.write(f"變化: `{lat:+.2f}%`")
                st.write(f"- 目標: ≤ `{lat_info.get('target', 1.5)*100:+.1f}%`")
                st.write(f"- 超越幅度: `{lat_info.get('surplus', 0)*100:+.2f}%`")
                st.write(f"- 加權分數: `{lat_info.get('score', 0):.4f}`")

            st.divider()
            st.write(f"**⚙️ {method.upper()} 完整量化參數**")
            display_config_details(config, method)
        
    # 維度評分雷達圖
    st.subheader("🎯 維度評分雷達圖")
    fig_radar = create_radar_chart(scored)
    if fig_radar:
        st.plotly_chart(fig_radar, width="stretch")

    st.divider()


def render_analysis_tab(data: Dict):
    """渲染深度分析標籤頁（來自 pareto_deep_analysis.json）"""
    analysis = data.get("pareto_deep_analysis", {})

    if not analysis:
        st.warning("沒有深度分析資料")
        return

    # Pareto 統計
    st.subheader("📊 Pareto 前沿統計")
    stats = analysis.get("pareto_statistics", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("總 Pareto 解", stats.get("total_pareto_solutions", 0))
    with col2:
        st.metric("滿足目標", stats.get("satisfying_targets", 0))
    with col3:
        st.metric("接近目標", stats.get("close_to_targets", 0))
    with col4:
        st.metric("違反目標", stats.get("violating_targets", 0))

    st.divider()

    # 相關性熱力圖和權衡分析
    tradeoff = analysis.get("tradeoff_analysis", {})
    if tradeoff:
        st.subheader("🔥 權衡分析")

        col1, col2 = st.columns([1, 1])

        with col1:
            fig = create_correlation_heatmap(tradeoff)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.write("**關鍵洞察**")
            for insight in tradeoff.get("key_insights", []):
                st.write(f"- {insight}")

    st.divider()

    # 場景推薦
    recommendations = analysis.get("recommendations", {})
    if recommendations:
        st.subheader("🎯 場景推薦")

        # 從 config 獲取目標設定
        config = data.get("config", {})
        targets = config.get("multiobjective", {}).get("targets", {})

        # 場景對比圖
        fig_scenario = create_scenario_comparison_chart(recommendations, targets)
        if fig_scenario:
            st.plotly_chart(fig_scenario, width="stretch")

        scenario_names = {
            "for_production": "🏭 生產環境",
            "for_memory_critical": "💾 記憶體關鍵",
            "for_accuracy_critical": "🎯 準確率關鍵"
        }

        cols = st.columns(3)
        for i, (key, info) in enumerate(recommendations.items()):
            if info:
                with cols[i % 3]:
                    name = scenario_names.get(key, key)
                    trial_id = info.get("recommended_trial_id", "N/A")
                    st.write(f"**{name}** (Trial {trial_id})")

                    config = info.get("config", {})
                    obj = info.get("objectives", {})
                    method = config.get("method", "unknown")

                    st.write(f"配置: `{format_config_summary(config)}`")
                    st.write(f"準確率: `{obj.get('accuracy_change', 0)*100:+.1f}%`")
                    st.write(f"GPU: `{obj.get('gpu_peak_change', 0)*100:+.1f}%`")
                    st.write(f"延遲: `{obj.get('latency_change', 0)*100:+.1f}%`")
                    st.write(f"總分: `{info.get('overall_score', 0):.4f}`")
                    st.write(f"原因: {info.get('reason', 'N/A')}")

                    if info.get("warning"):
                        st.warning(info.get("warning"))

                    # 可展開的詳細參數
                    with st.expander(f"查看 {name} 詳細參數"):
                        display_config_details(config, method)


def render_pruned_failed_tab(data: Dict):
    """渲染剪枝和失敗試驗分析標籤頁"""
    pruned_data = data.get("pruned_trials", {})
    failed_data = data.get("failed_trials", {})
    config = data.get("config", {})

    # 獲取約束條件（剪枝界線）
    constraints = config.get("multiobjective", {}).get("constraints", {})
    constraints_enabled = constraints.get("enabled", False)

    # 統計資訊
    n_pruned = pruned_data.get("n_pruned", 0) if pruned_data else 0
    n_failed = failed_data.get("n_failed", 0) if failed_data else 0

    st.subheader("📊 剪枝與失敗試驗統計")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("剪枝試驗數", n_pruned)
    with col2:
        st.metric("失敗試驗數", n_failed)
    with col3:
        pruning_status = "啟用" if constraints_enabled else "停用"
        st.metric("剪枝功能", pruning_status)

    st.divider()

    # ==========================================
    # 剪枝界線和剪枝統計圖表（一左一右）
    # ==========================================
    st.subheader("✂️ 剪枝試驗分析")

    if n_pruned > 0 and pruned_data.get("pruned_trials"):
        pruned_trials = pruned_data["pruned_trials"]

        # 量化方法統計
        method_counts = {}
        for trial in pruned_trials:
            method = trial.get("config", {}).get("method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

        # 一左一右排列：剪枝界線 + 統計圖表
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.write("**🔧 剪枝界線 (Constraints)**")
            if constraints_enabled:
                st.markdown(
                    f"當試驗結果違反以下任一界線時，將被剪枝：\n\n"
                    f"- **準確率下降上限**: ≥ `{constraints.get('accuracy_min', -0.5)*100:+.1f}%`\n"
                    f"- **GPU 節省下限**: ≤ `{constraints.get('gpu_peak_max', -0.2)*100:+.1f}%`\n"
                    f"- **延遲增加上限**: ≤ `{constraints.get('latency_max', 10.0)*100:+.1f}%`"
                )
            else:
                st.info("剪枝功能已停用 (constraints.enabled = false)")

        with col_right:
            st.write("**📊 量化方法剪枝統計**")
            fig_pruned_methods = go.Figure(data=[
                go.Bar(
                    x=list(method_counts.keys()),
                    y=list(method_counts.values()),
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(method_counts)],
                    text=list(method_counts.values()),
                    textposition='outside'
                )
            ])
            fig_pruned_methods.update_layout(
                title="各量化方法被剪枝的試驗數",
                xaxis_title="量化方法",
                yaxis_title="剪枝數量",
                height=300,
                yaxis=dict(dtick=1, tickformat='d')  # y 軸使用整數單位
            )
            st.plotly_chart(fig_pruned_methods, width="stretch")

        # 剪枝試驗詳細資訊（可展開的下拉選項）
        st.write("**剪枝試驗詳細資訊**")

        for i, trial in enumerate(pruned_trials):
            config_info = trial.get("config", {})
            obj = trial.get("objectives", {})
            method = config_info.get("method", "unknown")

            acc_change = obj.get("accuracy_change")
            gpu_change = obj.get("gpu_peak_change")
            lat_change = obj.get("latency_change")

            # 判斷違反了哪些約束
            violations = []
            if acc_change is not None and constraints_enabled:
                acc_limit = constraints.get("accuracy_min", -0.5)
                if acc_change < acc_limit:
                    violations.append("準確率")

            if gpu_change is not None and constraints_enabled:
                gpu_limit = constraints.get("gpu_peak_max", -0.2)
                if gpu_change > gpu_limit:
                    violations.append("GPU")

            if lat_change is not None and constraints_enabled:
                lat_limit = constraints.get("latency_max", 10.0)
                if lat_change > lat_limit:
                    violations.append("延遲")

            violation_str = ", ".join(violations) if violations else "未知"
            title = f"⚠️ 剪枝 {i+1}: {format_config_summary(config_info)} | 違反: {violation_str}"

            with st.expander(title, expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("準確率變化", f"{acc_change*100:+.1f}%" if acc_change is not None else "N/A")
                with col2:
                    st.metric("GPU 變化", f"{gpu_change*100:+.1f}%" if gpu_change is not None else "N/A")
                with col3:
                    st.metric("延遲變化", f"{lat_change*100:+.1f}%" if lat_change is not None else "N/A")

                st.write("**詳細配置參數**")
                display_config_details(config_info, method)

    else:
        # 沒有剪枝試驗，但仍顯示剪枝界線說明
        if constraints_enabled:
            st.write("**🔧 剪枝界線 (Constraints)**")
            st.markdown(
                f"當試驗結果違反以下任一界線時，將被剪枝：\n\n"
                f"- **準確率下降上限**: ≥ `{constraints.get('accuracy_min', -0.5)*100:+.1f}%`\n"
                f"- **GPU 節省下限**: ≤ `{constraints.get('gpu_peak_max', -0.2)*100:+.1f}%`\n"
                f"- **延遲增加上限**: ≤ `{constraints.get('latency_max', 10.0)*100:+.1f}%`"
            )
            st.success("沒有被剪枝的試驗")
        else:
            st.info("剪枝功能已停用 (constraints.enabled = false)，無剪枝試驗")

    st.divider()

    # ==========================================
    # 失敗試驗分析
    # ==========================================
    st.subheader("❌ 失敗試驗分析")

    if n_failed > 0 and failed_data.get("failed_trials"):
        failed_trials = failed_data["failed_trials"]

        # 量化方法統計
        method_counts = {}
        error_counts = {}
        for trial in failed_trials:
            method = trial.get("config", {}).get("method", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

            error = trial.get("error", "Unknown error")
            # 簡化錯誤訊息（取前 50 字）
            error_short = error[:50] + "..." if len(error) > 50 else error
            error_counts[error_short] = error_counts.get(error_short, 0) + 1

        # 長條圖
        st.write("**量化方法失敗統計**")
        fig_failed_methods = go.Figure(data=[
            go.Bar(
                x=list(method_counts.keys()),
                y=list(method_counts.values()),
                marker_color=['#E74C3C', '#9B59B6', '#3498DB'][:len(method_counts)]
            )
        ])
        fig_failed_methods.update_layout(
            title="各量化方法失敗的試驗數",
            xaxis_title="量化方法",
            yaxis_title="失敗數量",
            height=300
        )
        st.plotly_chart(fig_failed_methods, width="stretch")

        # 失敗試驗詳細資訊（可展開的下拉選項）
        st.write("**失敗試驗詳細資訊**")
        for i, trial in enumerate(failed_trials):
            config_info = trial.get("config", {})
            method = config_info.get("method", "unknown")
            error = trial.get("error", "Unknown error")
            error_short = error[:50] + "..." if len(error) > 50 else error

            title = f"❌ 失敗 {i+1}: {format_config_summary(config_info)} | {error_short}"

            with st.expander(title, expanded=False):
                st.error(f"**錯誤訊息**: {error}")
                st.write("**詳細配置參數**")
                display_config_details(config_info, method)

        # 錯誤類型統計
        if len(error_counts) > 1:
            st.write("**錯誤類型統計**")
            for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                st.write(f"- `{error}`: {count} 次")
    else:
        st.success("沒有失敗的試驗")


def render_config_tab(data: Dict):
    """渲染實驗配置標籤頁 - 顯示詳細的量化參數"""
    config = data.get("config", {})

    if not config:
        st.warning("沒有配置資料")
        return

    # ==========================================
    # 1. 實驗基本資訊
    # ==========================================
    st.subheader("📋 實驗基本資訊")
    exp_info = config.get("experiment", {})
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**實驗名稱**: `{exp_info.get('name', 'N/A')}`")
    with col2:
        st.write(f"**輸出目錄**: `{exp_info.get('output_dir', 'N/A')}`")

    st.divider()

    # ==========================================
    # 2. 基準模型配置
    # ==========================================
    st.subheader("🤖 基準模型配置")
    baseline = config.get("baseline", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**模型名稱**: `{baseline.get('model_name', 'N/A')}`")
    with col2:
        st.write(f"**資料類型**: `{baseline.get('model_dtype', 'N/A')}`")
    with col3:
        st.write(f"**裝置映射**: `{baseline.get('device_map', 'N/A')}`")

    # 資料集設定
    datasets = baseline.get("datasets", {})
    if datasets:
        st.write("**評估資料集**:")
        dataset_data = []
        for name, ds_config in datasets.items():
            dataset_data.append({
                "資料集": name,
                "啟用": "✓" if ds_config.get("enabled") else "✗",
                "樣本數": ds_config.get("num_samples", 0)
            })
        st.dataframe(pd.DataFrame(dataset_data), width="stretch", hide_index=True)

    st.divider()

    # ==========================================
    # 3. 多目標優化設定
    # ==========================================
    st.subheader("🎯 多目標優化設定")
    multiobjective = config.get("multiobjective", {})

    # 優化目標
    objectives = multiobjective.get("objectives", [])
    if objectives:
        st.write("**優化目標**:")
        obj_data = []
        for obj in objectives:
            direction_text = "最大化 ↑" if obj.get("direction") == "maximize" else "最小化 ↓"
            obj_data.append({
                "目標名稱": obj.get("name", "N/A"),
                "方向": direction_text,
                "權重": obj.get("weight", 1.0)
            })
        st.dataframe(pd.DataFrame(obj_data), width="stretch", hide_index=True)

    # 目標閾值
    col1, col2 = st.columns(2)

    with col1:
        st.write("**目標閾值 (Targets)**:")
        targets = multiobjective.get("targets", {})
        st.write(f"- 準確率最小變化: `{targets.get('accuracy_min', 0)*100:+.1f}%`")
        st.write(f"- GPU 峰值最大變化: `{targets.get('gpu_peak_max', 0)*100:+.1f}%`")
        st.write(f"- 延遲最大變化: `{targets.get('latency_max', 0)*100:+.1f}%`")

    with col2:
        st.write("**硬約束 (Constraints)**:")
        constraints = multiobjective.get("constraints", {})
        enabled = constraints.get("enabled", False)
        st.write(f"- 啟用: `{enabled}`")
        if enabled:
            st.write(f"- 準確率最小: `{constraints.get('accuracy_min', 0)*100:+.1f}%`")
            st.write(f"- GPU 峰值最大: `{constraints.get('gpu_peak_max', 0)*100:+.1f}%`")
            st.write(f"- 延遲最大: `{constraints.get('latency_max', 0)*100:+.1f}%`")

    # 聚合方法
    aggregation = multiobjective.get("aggregation", {})
    if aggregation:
        st.write(f"**聚合方法**: `{aggregation.get('method', 'N/A')}`")
        weights = aggregation.get("dataset_weights", {})
        if weights:
            st.write("**資料集權重**:")
            for ds, w in weights.items():
                st.write(f"- {ds}: `{w}`")

    st.divider()

    # ==========================================
    # 4. 優化器設定
    # ==========================================
    st.subheader("⚙️ 優化器設定")
    optimizer = config.get("optimizer", {})

    opt_type = optimizer.get("type", "N/A")
    st.write(f"**優化器類型**: `{opt_type}`")

    if "optuna" in opt_type:
        optuna_config = optimizer.get("optuna", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**試驗數**: `{optuna_config.get('n_trials', 0)}`")
        with col2:
            st.write(f"**啟動試驗數**: `{optuna_config.get('n_startup_trials', 0)}`")
        with col3:
            st.write(f"**採樣器**: `{optuna_config.get('sampler', 'N/A')}`")

        # TPE 參數
        tpe_params = optuna_config.get("tpe_params", {})
        if tpe_params:
            with st.expander("TPE 採樣器參數"):
                for key, value in tpe_params.items():
                    st.write(f"- {key}: `{value}`")

    # LLM 多代理優化器配置
    if "llm" in opt_type.lower():
        llm_config = optimizer.get("llm_agent", {})
        if llm_config:
            st.write("**🤖 LLM 代理配置**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**提供者**: `{llm_config.get('provider', 'N/A')}`")
                st.write(f"**模型**: `{llm_config.get('model', 'N/A')}`")
            with col2:
                prompt_config = llm_config.get("prompt", {})
                st.write(f"**Prompt 類型**: `{prompt_config.get('type', 'N/A')}`")
                st.write(f"**Prompt 配置檔**: `{prompt_config.get('config_file', 'N/A')}`")
            with col3:
                st.write(f"**溫度**: `{llm_config.get('temperature', 'N/A')}`")
                st.write(f"**最大 Token**: `{llm_config.get('max_tokens', 'N/A')}`")

            # Agent 溫度設定
            agent_temps = llm_config.get("agent_temperatures", {})
            if agent_temps:
                st.write("**Agent 溫度設定**:")
                temp_cols = st.columns(len(agent_temps))
                for i, (agent, temp) in enumerate(agent_temps.items()):
                    with temp_cols[i]:
                        st.metric(agent.capitalize(), f"{temp}")

            # 試驗數限制
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**最小試驗數**: `{llm_config.get('min_trials', 'N/A')}`")
            with col2:
                st.write(f"**最大試驗數**: `{llm_config.get('max_trials', 'N/A')}`")

        # 停止條件
        stopping = optimizer.get("stopping", {})
        if stopping:
            with st.expander("⏹️ 停止條件"):
                budget = stopping.get("budget", {})
                convergence = stopping.get("convergence", {})
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**預算限制**:")
                    st.write(f"- 最大試驗數: `{budget.get('max_trials', 'N/A')}`")
                    st.write(f"- 最大時間 (小時): `{budget.get('max_time_hours', 'N/A')}`")
                with col2:
                    st.write("**收斂條件**:")
                    st.write(f"- 啟用: `{convergence.get('enabled', False)}`")
                    if convergence.get("enabled"):
                        st.write(f"- 視窗大小: `{convergence.get('window', 'N/A')}`")
                        st.write(f"- 閾值: `{convergence.get('threshold', 'N/A')}`")

        # 驗證規則
        validation = optimizer.get("validation", {})
        if validation:
            with st.expander("✅ 驗證規則"):
                rules = validation.get("rules", [])
                if rules:
                    st.write("**規則**:")
                    for rule in rules:
                        st.write(f"- `{rule}`")
                st.write(f"**檢查 GPU 記憶體**: `{validation.get('check_gpu_memory', False)}`")
                st.write(f"**預期最大 GPU (GB)**: `{validation.get('max_expected_gpu_gb', 'N/A')}`")

        # 失敗處理
        fallback = optimizer.get("fallback", {})
        if fallback:
            with st.expander("🔄 失敗處理"):
                st.write(f"**LLM 失敗時**: `{fallback.get('on_llm_failure', 'N/A')}`")
                st.write(f"**最大 LLM 失敗次數**: `{fallback.get('max_llm_failures', 'N/A')}`")

    st.divider()

    # ==========================================
    # 5. 量化方法搜尋空間（重點！）
    # ==========================================
    st.subheader("🔍 量化方法搜尋空間")
    search_space = optimizer.get("search_space", {})

    methods = search_space.get("methods", [])
    st.write(f"**啟用的量化方法**: `{', '.join(methods)}`")

    # GPTQ 參數
    gptq_space = search_space.get("gptq", {})
    if gptq_space:
        with st.expander("📦 GPTQ 量化參數", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**基本參數**:")
                st.write(f"- bits (位元數): `{gptq_space.get('bits', [])}`")
                st.write(f"- group_size (群組大小): `{gptq_space.get('group_size', [])}`")
                st.write(f"- calib_num (校正樣本數): `{gptq_space.get('calib_num', [])}`")
                st.write(f"- desc_act (降序激活): `{gptq_space.get('desc_act', [])}`")
                st.write(f"- sym (對稱量化): `{gptq_space.get('sym', [])}`")

            with col2:
                st.write("**進階參數**:")
                st.write(f"- damp_percent (阻尼百分比): `{gptq_space.get('damp_percent', [])}`")
                st.write(f"- damp_auto_increment: `{gptq_space.get('damp_auto_increment', [])}`")
                st.write(f"- static_groups (靜態群組): `{gptq_space.get('static_groups', [])}`")
                st.write(f"- true_sequential (真正循序): `{gptq_space.get('true_sequential', [])}`")
                st.write(f"- mse (MSE 損失權重): `{gptq_space.get('mse', [])}`")
                st.write(f"- lm_head (量化 LM Head): `{gptq_space.get('lm_head', [])}`")

    # AWQ 參數
    awq_space = search_space.get("awq", {})
    if awq_space:
        with st.expander("📦 AWQ 量化參數", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**基本參數**:")
                st.write(f"- w_bit (權重位元數): `{awq_space.get('w_bit', [])}`")
                st.write(f"- q_group_size (群組大小): `{awq_space.get('q_group_size', [])}`")
                st.write(f"- zero_point (零點量化): `{awq_space.get('zero_point', [])}`")
                st.write(f"- version (版本): `{awq_space.get('version', [])}`")

            with col2:
                st.write("**排除模組**:")
                modules_not_convert = awq_space.get("modules_to_not_convert", [])
                for i, modules in enumerate(modules_not_convert):
                    if modules:
                        st.write(f"- 選項 {i+1}: `{modules}`")
                    else:
                        st.write(f"- 選項 {i+1}: `(無排除)`")

    # BNB 參數
    bnb_space = search_space.get("bnb", {})
    if bnb_space:
        with st.expander("📦 BitsAndBytes 量化參數", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**4-bit 量化參數**:")
                st.write(f"- bits (位元數): `{bnb_space.get('bits', [])}`")
                st.write(f"- quant_type (量化類型): `{bnb_space.get('bnb_4bit_quant_type', [])}`")
                st.write(f"- use_double_quant (雙重量化): `{bnb_space.get('bnb_4bit_use_double_quant', [])}`")
                st.write(f"- compute_dtype (計算類型): `{bnb_space.get('bnb_4bit_compute_dtype', [])}`")

            with col2:
                st.write("**8-bit 量化參數**:")
                st.write(f"- threshold (閾值): `{bnb_space.get('llm_int8_threshold', [])}`")
                st.write(f"- skip_modules (跳過模組): `{bnb_space.get('llm_int8_skip_modules', [])}`")
                st.write(f"- has_fp16_weight: `{bnb_space.get('llm_int8_has_fp16_weight', [])}`")

    st.divider()

    # ==========================================
    # 6. 完整配置 JSON
    # ==========================================
    with st.expander("📄 完整配置 JSON"):
        st.json(config)


def render_conversations_tab(data: Dict):
    """渲染 Agent 對話標籤頁（僅 LLM 版本）- 簡化版，直接顯示 JSON"""
    conversations = data.get("conversations", [])

    if not conversations:
        st.warning("沒有對話記錄")
        return

    st.write(f"共 {len(conversations)} 條訊息")

    # Agent 圖示映射
    agent_icons = {
        "AnalyzerAgent": "🔍",
        "PlannerAgent": "📋",
        "MonitorAgent": "📊",
        "Orchestrator": "🎯"
    }

    # 過濾選項
    event_types = set()
    agents = set()

    for msg in conversations:
        if "event" in msg:
            event_types.add(msg["event"])
        if "from_agent" in msg:
            agents.add(msg["from_agent"])

    col1, col2 = st.columns(2)
    with col1:
        selected_events = st.multiselect("事件類型", list(event_types), default=list(event_types))
    with col2:
        selected_agents = st.multiselect("Agent", list(agents), default=list(agents))

    st.divider()

    # 顯示對話（簡化版：直接用 JSON）
    for msg in conversations:
        # 檢查是否顯示
        if "event" in msg and msg["event"] not in selected_events:
            continue
        if "from_agent" in msg and msg["from_agent"] not in selected_agents:
            continue

        # 試驗開始事件
        if msg.get("event") == "trial_start":
            st.markdown(f"---\n### 🚀 試驗 {msg.get('trial_num')} 開始")
            st.caption(f"時間: {msg.get('timestamp', 'N/A')}")
            continue

        # 試驗結束事件
        if msg.get("event") == "trial_end":
            result = msg.get("result", {})
            success = result.get("success", False)
            satisfies = result.get("satisfies_targets", False)

            # 標題
            if success:
                target_str = "🎯" if satisfies else ""
                st.success(f"✅ 試驗 {msg.get('trial_num')} 完成 {target_str}")
            else:
                st.error(f"❌ 試驗 {msg.get('trial_num')} 失敗")

            # 顯示結果指標
            if result.get("objectives"):
                obj = result["objectives"]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    acc = obj.get('accuracy_change', 0)
                    st.metric("準確率", f"{acc*100:+.2f}%" if acc is not None else "N/A")
                with col2:
                    gpu = obj.get('gpu_peak_change', 0)
                    st.metric("GPU", f"{gpu*100:+.2f}%" if gpu is not None else "N/A")
                with col3:
                    lat = obj.get('latency_change', 0)
                    st.metric("延遲", f"{lat*100:+.2f}%" if lat is not None else "N/A")
                with col4:
                    st.metric("達標", "✅" if satisfies else "❌")
            continue

        # Agent 訊息
        if "from_agent" in msg:
            agent = msg["from_agent"]
            icon = agent_icons.get(agent, "💬")
            msg_type = msg.get("message_type", "message")
            timestamp = msg.get("timestamp", "N/A")

            with st.expander(f"{icon} {agent} → {msg_type} ({timestamp})", expanded=False):
                st.json(msg)


def _render_prompt_section(agent_data: Dict, section_key: str, label: str, expanded: bool = False):
    """渲染一個 prompt 區塊（若存在）"""
    value = agent_data.get(section_key)
    if value is not None:
        if isinstance(value, dict):
            with st.expander(label, expanded=expanded):
                st.json(value)
        else:
            with st.expander(label, expanded=expanded):
                st.code(value, language=None)


def render_prompt_tab(data: Dict):
    """渲染 Prompt 配置標籤頁"""
    prompt_data = data.get("prompt_config_used")

    if not prompt_data:
        st.info("此實驗沒有 Prompt 配置記錄（可能是較舊的實驗或非 LLM 優化實驗）")
        return

    st.subheader("📝 Prompt 配置資訊")

    # 基本資訊
    agent_mode = prompt_data.get("agent_mode", "separate")
    mode_label = {"separate": "Separate (Analyzer + Planner)", "combined": "Combined (Strategist)"}.get(agent_mode, agent_mode)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Prompt 類型", prompt_data.get("prompt_type", "N/A"))
    with col2:
        metadata = prompt_data.get("metadata", {})
        st.metric("版本", metadata.get("version", "N/A"))
    with col3:
        supported = prompt_data.get("supported_types", [])
        st.metric("支援類型數", len(supported))
    with col4:
        st.metric("Agent 模式", agent_mode)

    # Metadata
    st.write("**Metadata**")
    metadata = prompt_data.get("metadata", {})
    st.write(f"- 名稱: `{metadata.get('name', 'N/A')}`")
    st.write(f"- 描述: `{metadata.get('description', 'N/A')}`")
    st.write(f"- 配置檔路徑: `{prompt_data.get('config_path', 'N/A')}`")
    st.write(f"- 支援的類型: `{', '.join(prompt_data.get('supported_types', []))}`")
    st.write(f"- Agent 模式: `{mode_label}`")

    st.divider()

    # Prompts 詳細內容
    prompts = prompt_data.get("prompts", {})

    # 根據 agent_mode 動態決定 tabs
    if agent_mode == "combined":
        # Combined 模式：顯示 Strategist
        agent_tabs = st.tabs(["🧠 Strategist", "🔧 Formatting"])

        with agent_tabs[0]:
            st.subheader("Strategist Agent Prompt (Combined Mode)")
            st.caption("合併模式：在一次 LLM 調用中同時完成分析和決策")
            strategist = prompts.get("strategist", {})
            if strategist:
                st.write("**System Role**")
                st.code(strategist.get("system_role", "N/A"), language=None)

                if strategist.get("task_description"):
                    st.write("**Task Description**")
                    st.code(strategist.get("task_description", "N/A"), language=None)

                _render_prompt_section(strategist, "analysis_steps", "Analysis Steps (Part 1)")
                _render_prompt_section(strategist, "decision_steps", "Decision Steps (Part 2)")
                _render_prompt_section(strategist, "output_format", "Output Format")
                _render_prompt_section(strategist, "section_headers", "Section Headers")
                _render_prompt_section(strategist, "labels", "Labels")
            else:
                st.info("沒有 Strategist prompt 資料")

        with agent_tabs[1]:
            st.subheader("Formatting 設定")
            formatting = prompts.get("formatting", {})
            if formatting:
                st.json(formatting)
            else:
                st.info("沒有 Formatting 資料")

    else:
        # Separate 模式：顯示 Analyzer + Planner
        agent_tabs = st.tabs(["🔍 Analyzer", "📋 Planner", "🔧 Formatting"])

        with agent_tabs[0]:
            st.subheader("Analyzer Agent Prompt")
            st.caption("分析歷史試驗，識別模式和建議")
            analyzer = prompts.get("analyzer", {})
            if analyzer:
                st.write("**System Role**")
                st.code(analyzer.get("system_role", "N/A"), language=None)

                if analyzer.get("task_description"):
                    st.write("**Task Description**")
                    st.code(analyzer.get("task_description", "N/A"), language=None)

                _render_prompt_section(analyzer, "analysis_requirements", "Analysis Requirements")
                _render_prompt_section(analyzer, "output_format", "Output Format")
                _render_prompt_section(analyzer, "section_headers", "Section Headers")
                _render_prompt_section(analyzer, "labels", "Labels")
            else:
                st.info("沒有 Analyzer prompt 資料")

        with agent_tabs[1]:
            st.subheader("Planner Agent Prompt")
            st.caption("根據分析結果決定下一個試驗配置")
            planner = prompts.get("planner", {})
            if planner:
                st.write("**System Role**")
                st.code(planner.get("system_role", "N/A"), language=None)

                _render_prompt_section(planner, "decision_process", "Decision Process (Chain of Thought)")
                _render_prompt_section(planner, "strategy_guide", "Strategy Guide")
                _render_prompt_section(planner, "progress_recommendation", "Progress Recommendation")
                _render_prompt_section(planner, "output_format", "Output Format")
                _render_prompt_section(planner, "section_headers", "Section Headers")
                _render_prompt_section(planner, "labels", "Labels")
            else:
                st.info("沒有 Planner prompt 資料")

        with agent_tabs[2]:
            st.subheader("Formatting 設定")
            formatting = prompts.get("formatting", {})
            if formatting:
                st.json(formatting)
            else:
                st.info("沒有 Formatting 資料")

    st.divider()

    # 完整 JSON
    st.subheader("📥 完整 Prompt 配置")
    with st.expander("查看完整 JSON", expanded=False):
        st.json(prompt_data)


# ==========================================
# 主應用
# ==========================================

def main():
    st.title("📊 優化實驗報告檢視器")

    # 側邊欄 - 實驗選擇
    with st.sidebar:
        st.header("實驗選擇")

        # 實驗類型
        exp_type = st.radio(
            "實驗類型",
            ["標準優化", "LLM 多代理優化"],
            index=0
        )

        # 掃描實驗
        experiments = scan_experiments(exp_type)

        if not experiments:
            st.warning(f"沒有找到 {exp_type} 實驗")
            st.stop()

        # 選擇實驗
        selected_exp = st.selectbox(
            "選擇實驗",
            experiments,
            index=0
        )

        st.divider()

        # 顯示實驗基本資訊
        if selected_exp:
            data = load_experiment_data(exp_type, selected_exp)
            summary = data.get("summary", {})
            opt_info = summary.get("optimization", {})

            st.write("**實驗資訊**")
            st.write(f"- 優化器: {opt_info.get('optimizer', 'N/A')}")
            st.write(f"- 試驗數: {opt_info.get('total_trials', 0)}")
            st.write(f"- Pareto 解: {opt_info.get('pareto_solutions', 0)}")
            st.write(f"- 滿足目標: {opt_info.get('satisfying_solutions', 0)}")
            st.write(f"- 滿足目標且為 Pareto 解: {summary.get('satisfying_and_pareto', 0)}")

    # 主頁面 - 分頁標籤
    if selected_exp:
        data = load_experiment_data(exp_type, selected_exp)

        # 根據實驗類型決定標籤頁
        if exp_type == "LLM 多代理優化":
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📋 實驗摘要",
                "⚙️ 實驗配置",
                "📊 所有試驗",
                "✅ 滿足目標",
                "🔬 深度分析",
                "⚠️ 剪枝/失敗",
                "💬 Agent 對話",
                "📝 Prompt 配置"
            ])
        else:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📋 實驗摘要",
                "⚙️ 實驗配置",
                "📊 所有試驗",
                "✅ 滿足目標",
                "🔬 深度分析",
                "⚠️ 剪枝/失敗"
            ])
            tab7 = None
            tab8 = None

        with tab1:
            render_summary_tab(data)

        with tab2:
            render_config_tab(data)

        with tab3:
            render_trials_tab(data)

        with tab4:
            render_satisfying_trials_tab(data)

        with tab5:
            render_analysis_tab(data)

        with tab6:
            render_pruned_failed_tab(data)

        if tab7 is not None:
            with tab7:
                render_conversations_tab(data)

        if tab8 is not None:
            with tab8:
                render_prompt_tab(data)


if __name__ == "__main__":
    main()
