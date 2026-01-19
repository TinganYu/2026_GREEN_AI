# 評分與分析機制說明文件 (Scoring Mechanism Documentation)

本文檔詳細說明了 `scoring.py` 中實現的多目標優化評分與分析機制。該系統旨在評估 LLM 量化與優化後的模型表現，並從 Pareto 前沿中篩選出最佳配置。

系統主要由兩個核心組件構成：
1. **LayeredScorer (分層評分系統)**：針對滿足目標的試驗進行精細評分與排名。
2. **ParetoAnalyzer (Pareto 前沿分析器)**：分析所有非支配解（Non-dominated solutions）的權衡關係與聚類特徵。

---

## 1. 分層評分系統 (LayeredScorer)

此組件負責將複雜的多維度指標轉換為直觀的分數（0-100 分）和等級（A-F），以便於比較和選擇。

### 1.1 總體評分 (Overall Score)

總分是排名的主要依據，計算邏輯基於**加權距離**。

*   **公式概念**：
    $$ Score = 100 - \min(100, \text{Weighted Distance} \times \text{Scale Factor}) $$
    
*   **計算步驟**：
    1. **計算偏差**：計算每個指標（準確率、GPU 峰值、延遲）與設定目標值（Targets）的絕對偏差。
    2. **加權求和**：根據使用者配置的權重（Weights），計算總加權距離。
       $$ D_{weighted} = W_{acc} \cdot | \Delta Acc | + W_{gpu} \cdot | \Delta GPU | + W_{lat} \cdot | \Delta Lat | $$
    3. **歸一化與轉換**：將距離轉換為 0-100 的分數。距離越小（越接近或超越目標），分數越高。

### 1.2 單維度評分 (Dimension Scores)

除了總分，系統還會為每個維度單獨評分，幫助理解模型在特定方面的表現。

*   **評分範圍**：0 - 150 分
    *   **基準分**：100 分（恰好滿足目標）
    *   **獎勵分**：> 100 分（優於目標，最高 150）
    *   **懲罰分**：< 100 分（未達目標）
*   **評估指標**：
    *   **Accuracy (Maximize)**：數值越大越好。
    *   **GPU Peak & Latency (Minimize)**：數值越小越好。

### 1.3 等級制 (Grading)

根據總體評分分配等級：

| 等級 | 分數範圍 | 描述 |
| :--- | :--- | :--- |
| **A** | 90 - 100 | 優異 (Excellent) |
| **B** | 80 - 89 | 良好 (Good) |
| **C** | 70 - 79 | 普通 (Fair) |
| **D** | 60 - 69 | 及格邊緣 (Poor) |
| **F** | < 60 | 不及格 (Fail) |

### 1.4 策略排名 (Strategy Rankings)

系統會根據不同的優先級策略對試驗進行排名：

1.  **Closest to Target**：在滿足目標的試驗中，尋找加權距離最近的解。
2.  **Best Accuracy**：在 Pareto 前沿中，準確率最高的解。
3.  **Best Compression**：在 Pareto 前沿中，GPU 記憶體節省最多的解。
4.  **Balanced**：綜合考慮所有維度權重的平衡解。

---

## 2. Pareto 前沿分析 (ParetoAnalyzer)

此組件深入分析優化過程產生的 Pareto 前沿（即無法在不犧牲某一目標的情況下改善另一目標的解集合）。

### 2.1 深度分析功能

*   **統計分析**：
    *   計算滿足目標的解數量。
    *   識別「接近目標」（只有一個維度微幅未達標）的解。
    *   識別嚴重違反目標的解。

*   **聚類分析 (K-Means)**：
    *   將 Pareto 前沿上的解分組（最多 3 組）。
    *   **常見類型**：
        *   `high_accuracy_high_memory`：高準確度但記憶體佔用高。
        *   `high_compression_low_latency`：高壓縮率且低延遲。
        *   `balanced`：各方面表現均衡。

*   **權衡分析 (Trade-off Analysis)**：
    *   計算指標間的 Pearson 相關係數。
    *   **洞察範例**：「保持高準確率需要顯著犧牲 GPU 節省（強負相關）」。

### 2.2 場景化推薦 (Recommendations)

根據分析結果，系統會自動生成針對不同應用場景的推薦配置：

1.  **For Production (生產環境)**：
    *   優先選擇滿足所有目標且綜合評分最高的配置。
2.  **For Memory Critical (記憶體受限場景)**：
    *   推薦 GPU 節省最多的配置（即使可能稍微犧牲準確率）。
3.  **For Accuracy Critical (準確率優先場景)**：
    *   推薦準確率損失最小的配置。

---

## 3. 總結

此評分機制結合了**規則導向（Targets）**與**探索導向（Pareto）**的優點。既能確保模型滿足硬性指標（如最大允許延遲），又能挖掘出在特定維度表現極致的潛力配置，為開發者提供全面的決策支持。
