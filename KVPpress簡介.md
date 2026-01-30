# KVPress: LLM KV Cache 壓縮與自適應調整方案簡介

KVPress 是專為 LLM 推理優化設計的組件，位於 `lulu_temp/agents/` 目錄下。它通過在 Attention 層中識別並剔除「不重要」的 Key-Value 對，顯著降低長文本生成的顯存壓力。

## 1. 核心壓縮方法 (Base Logic)

KVPress 整合了多種先進的 KV 緩存壓縮算法。這些算法決定了在推理過程中哪些 Token 應該被保留在快取中：

### KV Cache 壓縮技術對照表

| 方法名稱 | 技術原理 | 特點 |
| :--- | :--- | :--- |
| **SnapKV** | 基於 Attention 權重聚合，保留關鍵指標 Token 與近期 Token。 | 效果穩定，平衡性佳。 |
| **Observed Attention** | 實時觀察注意力分佈，進行更精確的篩選。 | 準確度高，但計算開銷略大。 |
| **K-Norm** | 基於 Key 向量的範數 (Norm) 大小篩選 Token。 | 計算極快，無需 Attention 矩陣。 |
| **Expected Attention** | 基於期望注意力模式進行靜態或動態預測。 | 適合特定分佈的任務。 |
| **StreamingLLM** | 保留「Attention Sink」（前幾個 Token）與滑動窗口。 | 解決長文本流式推論，防止模型崩潰。 |

> **📌 補充筆記：**
> 在實際應用中，**SnapKV** 目前是平衡效能與準確率的首選；而 **StreamingLLM** 則是處理超長對話流（甚至無限長度）的適合方案。
---
### **核心參數**

`compression_ratio`: 保留 Token 的比例（例如 0.5 表示刪除一半）。

`compression_method`: 選擇採用的壓縮演算法（如上述表格所示）。

## 2. 三種優化與部署模式

根據不同的自動化需求，KVPress 提供以下三種運行模式：

**A. 基礎模式 (Base / Manual)**

直接在 `model_config.yaml` 中手動設定參數。適合已經知道最佳參數，直接進行生產部署或評估的場景。

**B. 網格搜索 (Grid Search)**

通過 `run.py` 觸發，系統會遍歷配置中定義的所有組合。

* 觸發方式: `tuning.adaptive: false`。

* 測試列表: 讀取 `compression_methods_to_test` 與 `compression_ratios_to_test。`

**C. LLM 自適應調整 (LLM Adaptive Tuning)**

這是最先進的模式，由 `llm_tuner.py` 驅動。利用 LLM（如 GPT-4o）作為「優化器」，根據前幾輪實驗的結果建議下一輪的超參數。
* 觸發方式: `tuning.adaptive: true`。
## 3. LLM Adaptive Tuner 工作流程

以下為 `LLMAdaptiveTuner` 類別在執行優化時的核心邏輯樹：
```
LLMAdaptiveTuner (lulu_temp/agents/llm_tuner.py)
   └─ 執行優化流程:
      ├─ 1. 基準測試 (Baseline Phase)
      │    ├─ 暫時關閉 KV 壓縮功能
      │    └─ 運行完整評估以獲取基準指標 (Acc, Latency, Memory, CO2)
      │
      ├─ 2. 疊代優化循環 (Optimization Loop)
      │    └─ 迴圈執行 (重複 Max Iterations 次):
      │       ├─ A. 生成 Prompt (構建包含基準與歷史嘗試紀錄的 Context)
      │       ├─ B. LLM 建議建議 (呼叫 API 或本地模型獲取 JSON 超參數)
      │       ├─ C. 執行評估 (於數據集子集快速運行)
      │       │    └─ 套用建議參數 (Method, Ratio)
      │       └─ D. 計算綜合得分 R
      │            ├─ 公式: R = (Acc * wa) + (Lat_B * wl) + (Mem_B * wm) + (CO2_B * wc)
      │            └─ 更新並紀錄當前最佳配置 (best_config)
      │
      └─ 3. 最終驗證 (Final Full Eval)
           ├─ 套用疊代中獲得的 best_config
           └─ 在完整數據集上執行最後評估並保存結果
```

## 4. 效益評估指標

KVPress 不僅關注準確率，還引入了多維度的評估體系：

* Model Quality: GSM8K 準確率或 MultiNews 的 ROUGE-L。

* Latency Benefit: 與基準相比節省的時間比例。

* Memory Benefit: 峰值顯存（Peak Allocated GB）的優化程度。

* Environmental Impact: 使用 `codecarbon` 追蹤推理產生的二氧化碳排放量，實現「Green AI」。

## 5. 快速啟動 KV 壓縮優化

要開始 KVPress 的 LLM 自適應優化，請確保環境變量已設定，並運行：
```
# 設定 API 密鑰
export USE_LLM_API=true
export LLM_API_KEY="your-api-key"

# 執行優化腳本 (以 GSM8K 為例)
python run.py --task gsm8k
```

在 `model_config.yaml` 中，確保以下設定開啟：
```
tuning:
  enabled: true
  adaptive: true # 開啟 LLM 調優
```


## 6. 致謝與參考 

本專案的部分核心實作參考或使用了NVIDIA 的 [KVPress](https://github.com/NVIDIA/kvpress?tab=readme-ov-file) 工具庫。

* **技術論文**: [Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution](https://arxiv.org/abs/2510.00636)
* **授權協定**: Apache-2.0 License
```
@article{devoto2025expectedattention,
  title={Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution},
  author={Devoto, Alessio and Jeblick, Maximilian and J{\'e}gou, Simon},
  journal={arXiv preprint arXiv:2510.00636},
  year={2025},
  url={[https://arxiv.org/abs/2510.00636](https://arxiv.org/abs/2510.00636)}
}
```