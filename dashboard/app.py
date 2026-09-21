"""
Green AI 量化搜尋分析——多頁 Streamlit 入口。

dashboard/pages_custom/report_20trial.py、report_30trial.py 是
final_results/app.py、final_results_30/app.py 的完整複本（自訂文字說明全部保留），
只改了 BASE_DIR 的計算方式使其指回原本的資料夾。這樣 final_results/app.py、
final_results_30/app.py 兩個原始檔案就不再被任何東西依賴，之後確認沒問題就可以刪除。

執行方式（在 repo 根目錄下）：
  streamlit run dashboard/app.py

新增第 4 組實驗配置時，只需要在下面的 pages 清單多加一個 st.Page(...)。
本檔案本身除了 st.navigation()/pg.run() 之外不應該呼叫其他 Streamlit 指令
（例如 st.set_page_config），因為每個子頁面自己會呼叫一次。
"""
from pathlib import Path

import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent

pages = [
    st.Page(str(DASHBOARD_DIR / "pages_custom" / "_landing.py"), title="首頁", icon="🌿", default=True),
    st.Page(str(DASHBOARD_DIR / "pages_custom" / "report_20trial.py"), title="Final Results (20 Trial)", icon="📁"),
    st.Page(str(DASHBOARD_DIR / "pages_custom" / "report_30trial.py"), title="Final Results 30 (30 Trial)", icon="📁"),
    st.Page(str(DASHBOARD_DIR / "pages_custom" / "report_30trial_noretry.py"), title="No-Retry 消融 (30 Trial)", icon="🧪"),
    st.Page(str(DASHBOARD_DIR / "pages_custom" / "comparison.py"), title="綜合比較", icon="🆚"),
]

pg = st.navigation(pages)
pg.run()
