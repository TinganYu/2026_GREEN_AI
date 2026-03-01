# ASVD Adaptive Tuner 文檔

## 概述

`ASVDAdaptiveTuner` 是一個 LLM 基礎的超參數優化工具，用於自動調整 ASVD（Activation-aware Singular Value Decomposition）的壓縮參數。它受到 OPRO（Optimization by PROmpting）的啟發，使用 LLM 作為優化算法。

位置：`ASVD4LLM-main/asvd_tuner.py`

**更新說明：** 本工具已簡化流程，不再需要執行 `asvd.py` 進行預實驗，改為直接調用 `build_asvd_repo.py` 一鍵完成壓縮與 PPL 評估。

**重要：此調整器支持兩種 LLM 模式：**
1. **OpenAI API 模式**（可選，通過環境變數啟用）
2. **本地 LLM Fallback 模式**（默認，無需 API 密鑰）

## 核心概念

### ASVD 超參數

ASVD 調整器管理以下主要超參數：

#### 1. **alpha** (0.3-0.7)
- ASVD 的主要超參數，用於權衡 權重矩陣（Weight）與激發值分布（Activation）的重要性。
- **低值 (0.3-0.4)**：更激進的壓縮，但可能降低品質
- **中等值 (0.5)**：平衡壓縮和品質
- **高值 (0.6-0.7)**：保守壓縮，更好的品質保留

#### 2. **param_ratio_target** (0.5-0.95)
- 壓縮後保留的參數比例
- **0.5**：壓縮 50%（激進）
- **0.8**：壓縮 20%（推薦）
- **0.95**：壓縮 5%（保守）

#### 3. **scaling_method** (選擇)
- `abs_mean`：激活感知平均值（推薦）
- `abs_max`：激活感知最大值
- `fisher`：Fisher 信息
- `fisher_abs_mean`：Fisher + 激活平均混合

#### 4. **sensitivity_metric** (選擇)
- `ppl`：困惑度（推薦用於語言模型）
- `stable_rank`：穩定秩指標


## LLM 配置方式

### 重要說明

本 ASVD Tuner 與 KVPress 完全分離：
- **ASVD 參數**：由 asvd.py 直接控制，包括 alpha、param_ratio_target、scaling_method 等
- **KVPress 參數**：由 kvpress 內部管理，不在 ASVD tuner 中設定
- Tuner 的作用：優化 ASVD 超參數，評估不同配置的性能

### 模式 1：使用本地 LLM

**最簡單的使用方式**：直接運行，無需任何配置

```bash
python asvd_tuner.py --max_iterations 10
```

### 模式 2：使用 OpenAI API（可選 - 更好的建議）

**步驟 1**：在項目根目錄創建或編輯 `.env` 文件

```env
USE_LLM_API=true
LLM_API_KEY=sk-your-api-key-here
LLM_MODEL=gpt-4o
```

**步驟 2**：運行調整器

```bash
python asvd_tuner.py --max_iterations 10
```

## 使用流程

### 1. 完整工作流程

```
ASVDAdaptiveTuner (ASVD4LLM-main/asvd_tuner.py)
   └─ 迴圈執行:
      ├─ 1. 使用 LLM 建議超參數 (構建包含基準與歷史嘗試紀錄的 Context)
      |
      ├─ 2. build_asvd_repo.py (ASVD4LLM-main/huggingface_repos/)
      |    └─ 執行 ASVD 實驗
      │       ├─ 加載模型
      │       ├─ 校準數據集
      │       ├─ 敏感度分析
      │       └─ 二進制搜索截斷秩
      │    └─ 構建 HuggingFace 模型
      │       ├─ 讀取 SVD 緩存
      │       └─ 保存到 ASVD4LLM-main/output/
      │
      ├─ 3. test_eval.py (tmp/test/)
      │    └─ 評估壓縮模型
      │       ├─ 加載評估器
      │       └─ 返回性能指標
      │
      └─ 4. 計算綜合分數
         score = 0.7*accuracy + 0.2*compression + 0.1*ppl
```

### 3. 檔案結構

整個優化流程涉及多個檔案：

```
ASVD4LLM-main/
├─ asvd_tuner.py          ← 優化主邏輯
├─ asvd.py                ← 執行 ASVD 實驗
├─ huggingface_repos/
│  └─ build_asvd_repo.py  ← 構建 HuggingFace 模型
└─ asvd_tuning_results.json ← 輸出結果

tmp/
├─ eval/
│  └─ base_evaluator.py   ← 評估器基類
├─ config/
│  └─ model_config.yaml   ← 評估配置
└─ test/
   └─ test_eval.py        ← 評估腳本
   └─ test_quantization.py   ← 執行量化
```

## 配置參數解釋

### 權重參數

調整器通過加權組合多個目標來計算綜合分數：

```python
score = (
    accuracy_weight * task_accuracy +
    compression_weight * (1 - compression_ratio) +
    ppl_weight * ppl_score
)
```

**默認配置**（推薦）：
```python
accuracy_weight = 0.7      # 70% 重視任務準確性
compression_weight = 0.2   # 20% 重視壓縮效率
ppl_weight = 0.1          # 10% 重視語言模型質量
```

### 迭代數

- **少量迭代 (5-10)**：快速探索，用於測試
- **中等迭代 (15-20)**：平衡探索，適合大多數情況
- **多次迭代 (30+)**：深度探索，用於生產環境

## LLM 提示策略

調整器使用 OPRO 風格的提示，包含：

1. **任務描述**：模型 ID、任務名稱
2. **可用超參數**：詳細解釋每個參數
3. **試驗歷史**：最近 5 次試驗的配置和結果
4. **最佳配置**：迄今為止的最佳結果
5. **探索建議**：鼓勵平衡探索和利用

### 備用策略

當 LLM 都不可用時，調整器使用簡單的備用策略：

- **第一次迭代**：使用保守的默認配置
- **後續迭代**：在最佳配置周圍隨機變化

## 輸出結果

### 結果結構範例

```json
{
  "best_result": {
    "iteration": 5,
    "params": {
      "alpha": 0.5,
      "param_ratio_target": 0.8,
      "scaling_method": "abs_mean",
      "sensitivity_metric": "ppl",
      "weight_quant": "none"
    },
    "score": 0.6234,
    "accuracy": 0.75,
    "compression_ratio": 0.8,
    "elapsed_seconds": 1250,
    "status": "success"
  },
  "trial_history": [
    { "iteration": 1, "params": {...}, "score": 0.58, ... },
    { "iteration": 2, "params": {...}, "score": 0.61, ... },
    ...
  ],
  "total_iterations": 5
}
```

## 集成 test_eval.py

優化後，可以將最佳配置的模型集成到評估流程：

```bash
# 1. 從 asvd_tuning_results.json 獲取最佳參數

# 2. 更新 model_config.yaml
model:
  path: ASVD4LLM-main/output/opt-125m-asvd80

# 3. 運行評估
python tmp/test/test_eval.py
```

## 環境變數

如果使用 LLM API，設置以下環境變數：

```bash
export USE_LLM_API=true
export LLM_API_KEY="your-api-key"
export LLM_MODEL="gpt-4o"  # 或其他模型
```
