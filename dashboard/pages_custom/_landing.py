"""首頁：導覽說明，沒有任何資料載入邏輯，純粹是入口說明頁。"""
import streamlit as st

st.set_page_config(page_title="Green AI 量化搜尋分析", page_icon="🌿", layout="wide")

st.title("🌿 Green AI 量化搜尋分析總覽")
st.markdown("""
本站整合三組實驗配置的分析，模型與任務皆為 **Llama-3.2-3B-Instruct / GSM8K**：

| 配置 | Trial 數 | 去重複重試 | 內容 |
|------|---------|-----------|------|
| **final_results（20-Trial）** | 20/run | 5 次安全網 | 7 種搜尋策略（LLM ×4 + TPE/NSGA-II/Random）完整比較，含 LLM 優勢分析報告 |
| **final_results_30（30-Trial）** | 30/run | 5 次安全網 | 同上，trial 數擴大到 30，含 Late-stage 改善分析 |
| **final_results_30_noretry（No-Retry 消融，30 Trial）** | 30/run | **關閉**（重複即浪費該 trial） | 只涵蓋 4 種 LLM 記憶策略，驗證「安全網被拿掉後，各模式的原始記憶可靠度」 |

請從左側導覽選擇要查看的頁面：

- **📁 Final Results (20 Trial)** / **📁 Final Results 30 (30 Trial)**：各自完整的詳細分析
  dashboard（Benchmark 比較、跨實驗比較、Trial 彩色分析、Pareto Frontier、LLM 優勢分析報告等），
  內容包含針對各自資料集客製化撰寫的文字說明，兩者結構相同但文字與部分圖表因 trial 數不同而有調整。
- **🧪 No-Retry 消融 (30 Trial)**：新增的第三組配置，聚焦在去重複重試機制的消融實驗。
  標題特意標出 trial 數，因為之後若出現其他 trial 數的 no-retry 對照組
  （例如 20-Trial No-Retry），才不會跟這個混淆。
- **🆚 綜合比較**：跨三組配置的輕量 headline 比較，不重複前面三頁的詳細內容，
  適合用來快速看「這三組設定之間差在哪、下一步該補哪個實驗」。

---

擴充建議：未來若新增第 4 組實驗配置（例如換模型、換任務、或另一種消融設計），
只需要：
1. 把新資料夾的必要檔案（`experiment_config.json` / `optimization_results.json` /
   `pareto_frontier.json` / 視情況加 `tool_debug_log.jsonl`、`llm_usage_log.jsonl`）
   整理好放進 repo 根目錄的新資料夾。
2. 在 `dashboard/app.py` 的 `st.navigation(...)` 清單裡加一行 `st.Page(...)` 指向新頁面。
3. 若需要客製化文字報告，複製一份 `pages_custom/report_30trial_noretry.py` 當模板調整，
   並依 trial 數/實驗設計取一個看得出差異的檔名（例如 `report_50trial_noretry.py`）；
   若只是想併進既有比較邏輯，改 `pages_custom/comparison.py` 的 `CONFIGS` 字典即可。
""")
