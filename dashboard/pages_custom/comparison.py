"""
綜合比較頁：跨三個資料集配置（final_results / final_results_30 / final_results_30_noretry）
的輕量比較，不重複各自 dashboard 裡已經有的詳細分析。

目的是回答「這三組實驗設定之間差在哪？值得再加哪些實驗？」，
所以只放 headline 指標的並排比較，細節請切到左側對應的頁面看。
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DASHBOARD_ROOT = Path(__file__).resolve().parent.parent
if str(DASHBOARD_ROOT) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_ROOT))

from common.loaders import load_all_experiments, build_headline_rows  # noqa: E402
from common.styling import (  # noqa: E402
    SAMPLER_LABELS, SAMPLER_COLORS, sampler_sort_key,
    CONFIG_COLORS, CONFIG_LABELS, TABLE_STYLE,
)

REPO_ROOT = DASHBOARD_ROOT.parent

CONFIGS = {
    "final_results":            REPO_ROOT / "final_results",
    "final_results_30":         REPO_ROOT / "final_results_30",
    "final_results_30_noretry": REPO_ROOT / "final_results_30_noretry",
}
CONFIG_ORDER = list(CONFIGS.keys())

st.set_page_config(page_title="綜合比較", page_icon="🆚", layout="wide")
st.title("🆚 綜合比較：三組實驗配置")
st.caption(
    "模型與任務皆為 Llama-3.2-3B-Instruct / GSM8K。三組差異在於 trial 預算與去重複重試機制："
)

_desc_rows = [
    {"配置": CONFIG_LABELS[k], "資料夾": k, "說明": v}
    for k, v in {
        "final_results": "20 trial/run，_MAX_DUP_RETRIES=5（LLM 建議重複配置時最多重打 5 次）",
        "final_results_30": "30 trial/run，_MAX_DUP_RETRIES=5",
        "final_results_30_noretry": "30 trial/run，_MAX_DUP_RETRIES=1（無安全網，重複即記為浪費 trial）",
    }.items()
]
st.dataframe(pd.DataFrame(_desc_rows), width="stretch", hide_index=True)

st.divider()

# ── 載入三組資料 ──────────────────────────────────────────────────────────
all_data = {}
for key, path in CONFIGS.items():
    if path.exists():
        exps = load_all_experiments(str(path))
        if exps:
            all_data[key] = build_headline_rows(exps)

if not all_data:
    st.error("三個資料夾都讀不到資料，請確認路徑。")
    st.stop()

LLM_MODES = ["full", "summary", "window", "tool"]
STAT_METHODS = ["tpe", "nsga2", "random"]

CONFIG_COLOR_BY_LABEL = {CONFIG_LABELS[k]: CONFIG_COLORS[k] for k in CONFIGS}


def _config_badge_style(val):
    color = CONFIG_COLOR_BY_LABEL.get(val, "#888888")
    return f"background-color: {color}26; color: {color}; font-weight: 700; border-radius: 4px;"


# ══════════════════════════════════════════════════════════════════════════
# 1. 彙總表（headline 指標一次看）—— 放最上面，涵蓋全部 7 種方法 × 3 組配置
# ══════════════════════════════════════════════════════════════════════════
st.header("1. 彙總表（headline 指標一次看）")
st.caption(
    "涵蓋全部 7 種方法（4 種 LLM 記憶策略 + TPE / NSGA-II / Random）× 3 組配置。"
    "「配置」欄位用顏色標籤區分，同一個模式的三組配置會相鄰排列，方便直向比較。"
    "TPE / NSGA-II / Random 沒有 no-retry 版本，所以只會出現在前兩組配置。"
)

combo_rows = []
for key, df in all_data.items():
    for _, r in df.iterrows():
        combo_rows.append({
            "配置": CONFIG_LABELS.get(key, key), "_config": key,
            "模式": SAMPLER_LABELS.get(r["sampler"], r["sampler"]), "_sampler": r["sampler"],
            "Runs": int(r["runs"]),
            "總 Trial": int(r["n_total_trials"]),
            "跳過率": r.get("skip_rate", 0.0),
            "有效 Trial 平均分": round(r["trial_mean"], 3),
            "Best Score 平均": round(r["best_mean"], 4),
            "Best Score Std": round(r["best_std"], 4),
            "正向率": r["pos_rate"],
            "災難率": r["cat_rate"],
        })

df_combo = pd.DataFrame(combo_rows)
if not df_combo.empty:
    df_combo["_sampler_sort"] = df_combo["_sampler"].map(sampler_sort_key)
    df_combo["_config_sort"] = df_combo["_config"].map(lambda k: CONFIG_ORDER.index(k))
    df_combo = (
        df_combo.sort_values(["_sampler_sort", "_config_sort"])
        .drop(columns=["_sampler", "_config", "_sampler_sort", "_config_sort"])
    )
    styled_combo = (
        df_combo.style
        .map(_config_badge_style, subset=["配置"])
        .background_gradient(subset=["Best Score 平均"], cmap="RdYlGn")
        .background_gradient(subset=["有效 Trial 平均分"], cmap="RdYlGn")
        .background_gradient(subset=["正向率"], cmap="RdYlGn")
        .background_gradient(subset=["災難率"], cmap="RdYlGn_r")
        .background_gradient(subset=["跳過率"], cmap="RdYlGn_r")
        .format({
            "正向率": "{:.1%}", "災難率": "{:.1%}", "跳過率": "{:.1%}",
            "Best Score 平均": "{:.4f}", "Best Score Std": "{:.4f}",
            "有效 Trial 平均分": "{:.3f}",
        })
        .set_table_styles(TABLE_STYLE)
    )
    st.write(styled_combo.to_html(), unsafe_allow_html=True)

    st.markdown("""
**欄位說明**：
- **有效 Trial 平均分**：排除跳過的空 trial 後，該方法所有 trial 分數的平均——反映「每一步決策的平均品質」。
- **Best Score 平均 / Std**：3~6 個 run 的最終最佳分數平均與標準差——反映「搜尋結果的天花板與穩定性」。
- **跳過率**：被記成「重複 config，已跳過」的 trial 比例。**不是只有 no-retry 配置才有**——
  5-retry 配置下，summary/window/tool 偶爾也會把 5 次重試都用完仍給重複建議（見下方第 4 節），
  只是機率遠低於關掉安全網之後。TPE/NSGA-II/Random 在三組配置下都是 0%。
""")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# 2. 最終最佳分數
# ══════════════════════════════════════════════════════════════════════════
st.header("2. 最終最佳分數（Best Score）—— 三組配置對照")
st.caption("這個指標在三組配置下差異都很小——trial 預算與 retry 機制主要影響的是『搜尋過程』而非『天花板』。")

rows = []
for key, df in all_data.items():
    sub = df[df["sampler"].isin(LLM_MODES)]
    for _, r in sub.iterrows():
        rows.append({
            "配置": CONFIG_LABELS.get(key, key), "_config": key,
            "模式": SAMPLER_LABELS.get(r["sampler"], r["sampler"]), "_sampler": r["sampler"],
            "Best Score 平均": r["best_mean"],
        })
df_best = pd.DataFrame(rows)
if not df_best.empty:
    fig1 = go.Figure()
    for key in all_data:
        sub = df_best[df_best["_config"] == key].sort_values("_sampler", key=lambda s: s.map(sampler_sort_key))
        if sub.empty:
            continue
        fig1.add_trace(go.Bar(
            x=sub["模式"], y=sub["Best Score 平均"], name=CONFIG_LABELS.get(key, key),
            marker_color=CONFIG_COLORS.get(key, "#888"),
            text=sub["Best Score 平均"].apply(lambda v: f"{v:.3f}"), textposition="outside",
        ))
    fig1.update_layout(
        barmode="group", height=420, yaxis_title="Best Score（3 run 平均）",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend_title="",
    )
    fig1.update_yaxes(gridcolor="#eee")
    st.plotly_chart(fig1, width="stretch")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# 3. 決策品質
# ══════════════════════════════════════════════════════════════════════════
st.header("3. 決策品質：正向率 / 災難率")
st.caption("這兩個指標比最終分數更能反映『記憶設計本身的優劣』，且跨配置的差異更清楚可辨。")

metric_choice = st.radio("選擇指標", ["正向率（score > 0）", "災難率（score < -10）"], horizontal=True)
metric_col = "pos_rate" if metric_choice.startswith("正向") else "cat_rate"

rows2 = []
for key, df in all_data.items():
    sub = df[df["sampler"].isin(LLM_MODES)]
    for _, r in sub.iterrows():
        rows2.append({
            "配置": CONFIG_LABELS.get(key, key), "_config": key,
            "模式": SAMPLER_LABELS.get(r["sampler"], r["sampler"]), "_sampler": r["sampler"],
            "值": r[metric_col],
        })
df_m = pd.DataFrame(rows2)
if not df_m.empty:
    fig2 = go.Figure()
    for key in all_data:
        sub = df_m[df_m["_config"] == key].sort_values("_sampler", key=lambda s: s.map(sampler_sort_key))
        if sub.empty:
            continue
        fig2.add_trace(go.Bar(
            x=sub["模式"], y=sub["值"] * 100, name=CONFIG_LABELS.get(key, key),
            marker_color=CONFIG_COLORS.get(key, "#888"),
            text=sub["值"].apply(lambda v: f"{v:.1%}"), textposition="outside",
        ))
    fig2.update_layout(
        barmode="group", height=420, yaxis_title=f"{metric_choice} (%)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend_title="",
    )
    fig2.update_yaxes(gridcolor="#eee")
    st.plotly_chart(fig2, width="stretch")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# 4. Trial 浪費率（跳過率）—— 三組配置都會出現，不是 no-retry 特有
# ══════════════════════════════════════════════════════════════════════════
st.header("4. Trial 浪費率（跳過率）—— 三組配置都有機會出現")
st.caption(
    "「重複 config，已跳過」這個實驗健康度狀態，**在所有配置下都可能發生**，"
    "只是機率差很多：5-retry 配置下，LLM 最多重打 5 次才會放棄，所以大多數情況下"
    "跳過率接近 0%；無安全網（no-retry）則是一次重複就直接放棄，跳過率大幅升高。"
    "TPE / NSGA-II / Random 在三組配置下都是 0%（連續超參數空間本來就幾乎不會重複取樣）。"
)

rows3 = []
for key, df in all_data.items():
    sub = df[df["sampler"].isin(LLM_MODES)]
    for _, r in sub.iterrows():
        rows3.append({
            "配置": CONFIG_LABELS.get(key, key), "_config": key,
            "模式": SAMPLER_LABELS.get(r["sampler"], r["sampler"]), "_sampler": r["sampler"],
            "跳過率": r.get("skip_rate", 0.0),
        })
df_sk = pd.DataFrame(rows3)
if not df_sk.empty:
    fig3 = go.Figure()
    for key in all_data:
        sub = df_sk[df_sk["_config"] == key].sort_values("_sampler", key=lambda s: s.map(sampler_sort_key))
        if sub.empty:
            continue
        fig3.add_trace(go.Bar(
            x=sub["模式"], y=sub["跳過率"] * 100, name=CONFIG_LABELS.get(key, key),
            marker_color=CONFIG_COLORS.get(key, "#888"),
            text=sub["跳過率"].apply(lambda v: f"{v:.1%}"), textposition="outside",
        ))
    fig3.update_layout(
        barmode="group", height=420, title="各模式跳過率：三組配置對照",
        yaxis_title="跳過率 (%)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend_title="",
    )
    fig3.update_yaxes(gridcolor="#eee")
    st.plotly_chart(fig3, width="stretch")
    st.caption("詳細對照與統計檢定請見左側「🧪 No-Retry 消融 (30 Trial)」頁。")
else:
    st.info("找不到可比較的資料。")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# 5. Token 成本比較：no-retry（真實記錄）vs 5-retry（離線估計）
# ══════════════════════════════════════════════════════════════════════════
st.header("5. Token 成本比較：No-Retry（真實）vs 5-Retry（估計）")
st.caption(
    "**final_results_30_noretry 是真實記錄**（跑在 `llm_client.py` 加上 `_log_usage()` 之後，"
    "每次 API 呼叫的 token 數都被即時存下來）。"
    "**final_results_30 是離線估計值**（跑在加上 token 記錄之前，只能事後用同樣的 prompt 樣板"
    "重建 prompt 文字＋離線計算 token 數；隱藏的去重複重試/JSON 驗證重試次數是真實觀測到的，"
    "但那些重試呼叫本身的 token 量是用同一個 run 內其他呼叫的平均值去估算，不是精確值）。"
    "兩者是不同性質的數字，放在一起比較時這個差異要保留在心裡，但仍然可以看出量級與排序上的參考價值。"
)

_token_path_est = REPO_ROOT / "token_usage_estimated_final_results_30.json"
_token_path_real = REPO_ROOT / "token_usage_real_final_results_30_noretry.json"

if _token_path_est.exists() and _token_path_real.exists():
    tok_est = json.load(open(_token_path_est, encoding="utf-8"))
    tok_real = json.load(open(_token_path_real, encoding="utf-8"))
    df_tok = pd.DataFrame(tok_est + tok_real)
    df_tok["說明"] = df_tok["mode"].map(SAMPLER_LABELS)
    df_tok["_sort"] = df_tok["mode"].map(sampler_sort_key)
    df_tok["資料性質"] = df_tok["data_type"].map({"real": "✅ 真實記錄", "estimated": "🧮 離線估計"})

    # 5-retry（估計）拆成 EXACT（可信部分）＋ 隱藏重試（估計值）兩段疊加；
    # 0-retry（真實）沒有隱藏事件（hidden_events=0），畫成單一實色長條。
    # 兩組用不同 offsetgroup，讓每個模式底下並排顯示「5-retry 疊加柱」與「0-retry 單柱」。
    HIDDEN_COLOR = "rgba(231,76,60,0.55)"

    fig_tok = go.Figure()

    sub_est = df_tok[df_tok["config"] == "final_results_30"].sort_values("_sort")
    if not sub_est.empty:
        hidden_est = sub_est["avg_total_tokens"] - sub_est["avg_exact_tokens"]
        fig_tok.add_trace(go.Bar(
            x=sub_est["說明"], y=sub_est["avg_exact_tokens"],
            name="5-Retry · EXACT（可信部分）",
            marker_color=CONFIG_COLORS.get("final_results_30", "#888"),
            offsetgroup="est", legendgroup="est",
            text=sub_est["avg_exact_tokens"].apply(lambda v: f"{v:,.0f}"), textposition="inside",
        ))
        fig_tok.add_trace(go.Bar(
            x=sub_est["說明"], y=hidden_est,
            name="5-Retry · 隱藏重試（估計值，非精確）",
            marker_color=HIDDEN_COLOR,
            offsetgroup="est", legendgroup="est",
            text=hidden_est.apply(lambda v: f"+{v:,.0f}"), textposition="outside",
        ))

    sub_real = df_tok[df_tok["config"] == "final_results_30_noretry"].sort_values("_sort")
    if not sub_real.empty:
        fig_tok.add_trace(go.Bar(
            x=sub_real["說明"], y=sub_real["avg_total_tokens"],
            name="0-Retry · 真實總量（無隱藏事件）",
            marker_color=CONFIG_COLORS.get("final_results_30_noretry", "#888"),
            offsetgroup="real",
            text=sub_real["avg_total_tokens"].apply(lambda v: f"{v:,.0f}"), textposition="outside",
        ))

    fig_tok.update_layout(
        barmode="stack", height=460,
        title="平均總 Token 用量：5-Retry（EXACT + 隱藏重試估計，疊加）vs 0-Retry（真實總量）",
        yaxis_title="Token 數", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend_title="",
    )
    fig_tok.update_yaxes(gridcolor="#eee")
    st.plotly_chart(fig_tok, width="stretch")

    _tbl_cols = {"說明": "模式", "資料性質": "資料性質", "avg_calls": "平均呼叫數",
                 "avg_total_tokens": "平均總 Token", "avg_hidden_events": "平均隱藏事件數"}
    df_tok_show = df_tok.sort_values(["_sort", "config"])[list(_tbl_cols.keys())].rename(columns=_tbl_cols)
    styled_tok = (
        df_tok_show.style
        .background_gradient(subset=["平均總 Token"], cmap="RdYlGn_r")
        .format({"平均呼叫數": "{:.1f}", "平均總 Token": "{:,.0f}", "平均隱藏事件數": "{:.1f}"})
        .set_table_styles(TABLE_STYLE)
    )
    st.write(styled_tok.to_html(), unsafe_allow_html=True)

    st.markdown("""
**兩種設計把「記憶不足」的代價轉嫁到不同地方**：5-retry 把代價變成看不見的額外 API 呼叫
（估計值裡的「平均隱藏事件數」），0-retry 把代價變成看得見的浪費 trial 名額（見上方第 4 節的
跳過率）——但 0-retry 因為每個 iteration 固定只呼叫 1 次（tool 模式除外，仍會有多輪工具呼叫），
沒有隱藏重試的額外開銷，所以四個模式在 0-retry 下的 token 總量普遍比 5-retry 估計值低，
尤其是 summary 和 window（記憶容量小、重複率高，5-retry 版本要付出的隱藏重試代價也最大）。
**Tool 模式是唯一的例外**——不管有沒有安全網，它的多輪工具檢索設計本身就很貴，
是四者中 token 成本最高的。

⚠️ **Tool 模式左側「5-Retry・估計」那根柱子的 EXACT/隱藏拆分比其他三個模式不可靠**：
它重新呼叫時會觸發自己的多輪工具檢索，但 debug log 只記錄 iteration 編號、不記錄
「這是第幾次重試」，導致部分本該算進隱藏重試的成本被誤標成 EXACT（實測某個 run
理論上限應該 ≤30 次嘗試，實際卻偵測到 52 次）。full/summary/window 的 EXACT 是直接從
最終存檔的建議反推，沒有這個問題，可信度較高。

`final_results`（20-trial）目前還沒有真實或估計的 token 數據
（它橫跨兩個不同的舊 log 檔案，需要另外處理才能重建），暫不列入本比較。
""")
else:
    st.info(
        f"缺少 token 用量檔案（{_token_path_est.name} 或 {_token_path_real.name}），"
        "無法顯示這個比較。"
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# 6. 下一步
# ══════════════════════════════════════════════════════════════════════════
st.header("6. 下一步該加哪些實驗？")
st.markdown("""
根據三組配置的比較，目前還沒有被驗證、值得優先考慮的擴充方向：

1. **換模型 / 換任務**：三組資料全部只測過 Llama-3.2-3B-Instruct + GSM8K，
   目前所有結論（例如「Full 記憶最可靠」）都可能是這個模型/任務的特例。
2. **更多 run 數**：目前 LLM 記憶策略之間的「最終分數」比較，因為 run 數少（3-6 run）
   而效果量太小，難以達到統計顯著；若想再驗證最終分數上的差異，需要大幅增加 run 數，
   但這個投入報酬可能不如追加下面兩項。
3. **guiding LLM 消融**：四種記憶策略目前都固定用同一顆 LLM（gpt-4o）做決策，
   還沒驗證換一顆較弱的模型（如 gpt-4o-mini）是否會讓記憶設計的差異更明顯或消失。
4. **長 trial 數驗證**：no-retry 消融只跑了 30 trial，若要驗證「summary/window 的記憶
   缺陷在更長的搜尋中會不會更嚴重」，值得挑 1-2 個模式跑 50-80 trial 觀察跳過率的成長曲線。
""")
