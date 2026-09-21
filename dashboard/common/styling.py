"""
跨資料夾共用的樣式常數與小工具：sampler 標籤、顏色、排序、表格美化。
和 final_results/app.py、final_results_30/app.py 內部定義的版本保持同一套視覺語言，
好讓「綜合比較頁」畫出來的圖表跟各自的詳細報告用同一套配色，不會看起來像不同專案。

新增第 5 種以上的實驗配置（例如未來的新模型/新任務/新 retry 設計）時，
若引入了新的 sampler 名稱，記得在這裡的字典也加一筆，否則會退回灰色/預設排序。
"""
import pandas as pd

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

SAMPLER_SORT_ORDER = {
    "tpe": 0, "nsga2": 1, "random": 2,
    "window": 3, "summary": 4, "tool": 5, "full": 6,
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

# 三個資料集配置各自的識別色，用在「綜合比較頁」區分不同資料來源（跟 sampler 顏色是兩套獨立系統）
CONFIG_COLORS = {
    "final_results":          "#3498db",   # 20 trial，5-retry
    "final_results_30":       "#2ecc71",   # 30 trial，5-retry
    "final_results_30_noretry": "#e74c3c", # 30 trial，0-retry（去重複安全網關閉）
}

CONFIG_LABELS = {
    "final_results":            "20-Trial（5-retry 安全網）",
    "final_results_30":         "30-Trial（5-retry 安全網）",
    "final_results_30_noretry": "30-Trial（0-retry，無安全網）",
}

TABLE_STYLE = [
    {"selector": "th", "props": [
        ("background-color", "#1a252f"), ("color", "white"),
        ("font-size", "13px"), ("text-align", "center"), ("padding", "6px 10px"),
    ]},
    {"selector": "td", "props": [
        ("font-size", "12px"), ("padding", "5px 10px"), ("text-align", "center"),
    ]},
]


def sampler_sort_key(value) -> int:
    if value is None:
        return len(DEFAULT_SAMPLER_ORDER)
    return SAMPLER_SORT_ORDER.get(str(value), len(DEFAULT_SAMPLER_ORDER))


def style_ranked_df(df: pd.DataFrame, gradient_cols: dict) -> "pd.io.formats.style.Styler":
    """gradient_cols: {欄位名: cmap 名稱}"""
    styled = df.style
    for col, cmap in gradient_cols.items():
        if col in df.columns:
            styled = styled.background_gradient(subset=[col], cmap=cmap)
    return styled.set_table_styles(TABLE_STYLE)
