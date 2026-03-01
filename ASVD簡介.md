# ASVD (Activation-aware SVD) 權重壓縮系統簡介

## 1. 系統簡介

ASVD 是一種靜態模型壓縮技術，旨在透過「低秩近似（Low-Rank Approximation）」減少大型語言模型（LLM）的參數數量。與傳統的 SVD 不同，ASVD 引入了**激活感知（Activation-aware）** 機制，考慮了模型在推論時神經元的真實活化分佈，從而能在更高的壓縮比下維持邏輯推理能力（如 GSM8K）。

## 2. 核心技術原理

### 2.1 權重分解公式

在標準 Transformer 中，線性層執行的運算為 $Y = XW + b$。ASVD 將權重矩陣 $W$（維度 $M \times N$）分解為兩個較小的矩陣：

$$W \approx W_{B} \times W_{A}$$

其中：

$W_{B}$ 的維度為 $M \times R$

$W_{A}$ 的維度為 $R \times N$

$R$（秩，Rank）遠小於 $M$ 或 $N$。

### 2.2 激活感知縮放 (Activation-aware Scaling)

並非所有權重維度都同等重要。ASVD 透過校準數據集（如 Alpaca 或 Wikitext）獲取激活值 $X$ 的特徵分佈，並計算縮放矩陣 $S$。其分解邏輯轉化為：

$$W \cdot S \approx U \Sigma V^T$$

這確保了那些對輸出影響較大的「敏感維度」在分解過程中被優先保留，減少對模型核心能力的損傷。

## 3. 系統架構組件

### 3.1 核心腳本功能

`asvd.py`：核心計算引擎。執行校準、計算奇異值、並透過二進位搜尋（Binary Search）決定每一層最優的截斷秩（Truncation Rank）。

`build_asvd_repo.py`：物理構建器。讀取 `asvd.py` 產出的 Rank 配置，將模型權重正式拆解並儲存為新的 HuggingFace Repo。

`modeling_asvd_llama.py`：定義 ASVDLinear 類別，將原本的 nn.Linear 替換為 nn.Sequential(BLinear, ALinear)，實現模型結構的動態轉換。

### 3.2 數據流結構
```
ASVDAdaptiveTuner (ASVD4LLM-main/asvd_tuner.py)
   └─ 迴圈執行:
      ├─ 1. 使用 LLM 建議超參數 (構建包含基準與歷史嘗試紀錄的 Context)
      │
      ├─ 2. asvd.py (ASVD4LLM-main/)
      │    └─ 執行 ASVD 實驗
      │       ├─ 加載模型
      │       ├─ 校準數據集
      │       ├─ 敏感度分析
      │       └─ 二進制搜索截斷秩
      │
      ├─ 3. build_asvd_repo.py (ASVD4LLM-main/huggingface_repos/)
      │    └─ 構建 HuggingFace 模型
      │       ├─ 讀取 SVD 緩存
      │       └─ 保存到 ASVD4LLM-main/output/
      │
      ├─ 4. test_eval.py (tmp/test/)
      │    └─ 評估壓縮模型
      │       ├─ 加載評估器
      │       └─ 返回性能指標
      │
      └─ 5. 計算綜合分數
         score = 0.7*accuracy + 0.2*compression + 0.1*ppl
```
## 4. 超參數策略
## 模型壓縮與優化參數指南 (針對 GSM8K 任務)

| 參數 | 推薦範圍 | 對 GSM8K 的影響 |
| :--- | :--- | :--- |
| **Alpha** | 0.5 - 0.7 | **核心保護層**。較高的 Alpha (如 0.6) 會保護高激活神經元，防止模型在壓縮後發生「邏輯崩潰」。 |
| **Param Ratio** | 0.7 - 0.9 | **壓縮強度**。0.8 代表減少 20% 參數。建議從 0.85 開始測試，若 Acc 掉超過一半則需調高比例。 |
| **Scaling Method** | fisher | **敏感度指標**。對於數學任務，`fisher` 信息矩陣能比 `abs_mean` 更準確地定位哪些權重對 Loss 影響最大。 |
| **Calib Dataset** | wikitext2 | **基礎語言特徵保留**。使用通用語料校準，旨在維持模型的語言流暢度與基礎分布，避免壓縮後 PPL (困惑度) 劇烈上升。 |

> **💡 提示：**
> 調整 GSM8K 相關模型時，建議優先固定 `Scaling Method` 為 `fisher`，再微調 `Param Ratio`。如果數學推理邏輯出現明顯胡言亂語，通常是 `Alpha` 值過低導致關鍵神經元被誤刪。
## 5. 評分機制 (Score Calculation)

`asvd_tuner.py` 使用多目標優化評分來決定該次試驗的優劣，計算公式如下：

$$Score = w_{acc} \cdot Accuracy + w_{comp} \cdot (1 - ActualRatio) + w_{ppl} \cdot PPL_{score}$$

### 5.1 權重分配 (預設值)

* $w_{acc}$ (0.7)：任務準確度權重。性能維持是最重要的指標。

* $w_{comp}$ (0.2)：壓縮效率權重。鼓勵模型在維持性能的前提下儘可能變小。

* $w_{ppl}$ (0.1)：語言流暢度權重。確保模型不會因為過度壓縮而開始胡言亂語。

### 5.2 核心分數轉換邏輯

PPL Score：
使用指數衰減公式：$e^{-(AvgPPL - 15) / 100}$（當 $AvgPPL > 15$ 時）。這意味著當 PPL 異常飆升（例如高於 100）時，該分數會迅速趨近於 0。


## 6. 如何執行 (Execution Guide)

### 6.1 自動調優模式 (ASVD Tuner)

使用 LLM 代理自動尋找最優超參數：
```bash
python asvd_tuner.py \
    --model_id [你的模型路徑] \
    --task gsm8k \
    --max_iterations 20 \
    --accuracy_weight 0.7 \
    --compression_weight 0.2
```

### 6.2 手動執行模式

如果你已有理想的參數，可以直接執行壓縮流程：
```
* 步驟 1: 執行 ASVD 實驗 (生成緩存)
PYTORCH_ALLOC_CONF='expandable_segments:True' python asvd.py \
    --model_id [你的模型路徑] \
    --alpha 0.55 \
    --param_ratio_target 0.8 \
    --scaling_method fisher \
    --n_calib_samples 32 \
    --calib_dataset wikitext2 \
    --act_aware --use_cache

* 步驟 2: 構建壓縮模型 Repo
python huggingface_repos/build_asvd_repo.py \
    --model_id [你的模型路徑] \
    --alpha 0.55 \
    --param_ratio_target 0.8 \
    --scaling_method fisher \
    --act_aware --use_cache
```
## 7. 致謝與參考 (Acknowledgements & Citations)

本專案的部分核心實作參考或使用了 [ASVD4LLM](https://github.com/hahnyuan/ASVD4LLM) 的開源程式碼。

* **技術論文**: [ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models](https://arxiv.org/abs/2312.05821)
* **授權協定**: MIT License

如果您在研究中使用了本工具，請引用原始論文：

```bibtex
@misc{yuan2023asvd,
      title={ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models}, 
      author={Zhihang Yuan and Yuzhang Shang and Yue Song and Qiang Wu and Yan Yan and Guangyu Sun},
      year={2023},
      eprint={2312.05821},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}