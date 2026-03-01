# ASVD Adaptive Tuner 文檔

## 概述

`ASVDAdaptiveTuner` 是一個 LLM 基礎的超參數優化工具，用於自動調整 ASVD（Activation-aware Singular Value Decomposition）的壓縮參數。它受到 OPRO（Optimization by PROmpting）的啟發，使用 LLM 作為優化算法。

位置：`ASVD4LLM-main/asvd_tuner.py`

**重要：此調整器支持兩種 LLM 模式：**
1. **OpenAI API 模式**（可選，通過環境變數啟用）
2. **本地 LLM Fallback 模式**（默認，無需 API 密鑰）

## 核心概念

### ASVD 超參數

ASVD 調整器管理以下主要超參數：

#### 1. **alpha** (0.3-0.7)
- ASVD 的主要超參數
- **低值 (0.3-0.4)**：更激進的壓縮，但可能降低質量
- **中等值 (0.5)**：平衡壓縮和質量
- **高值 (0.6-0.7)**：保守壓縮，更好的質量保留

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

#### 5. **weight_quant** (選擇)
- `none`：無量化（默認）
- `rtn_int8`/`rtn_int6`：Round-to-nearest 量化
- `awq_int8`/`awq_int4`：激活感知權重量化

#### 6. **compress_kv_cache** (布爾值)
- 是否壓縮注意力層的 KV 緩存
- 推薦用於長序列任務

#### 7. **kv_cache_ratio_target** (0.2-0.9，可選)
- KV 緩存的壓縮比
- 低於 0.5：激進
- 0.5-0.8：平衡（推薦）

## 文件結構

```
ASVD4LLM-main/
├─ asvd_tuner.py              ← 核心調整器類
├─ asvd_tuner_example.py      ← 使用示例
├─ ASVD_TUNER_README.md       ← 本文檔
├─ asvd.py                    ← ASVD 實驗腳本
├─ build_asvd_repo.py         ← 模型構建
└─ ...
```

## LLM 配置方式

### 重要說明

本 ASVD Tuner 與 KVPress 完全分離：
- **ASVD 參數**：由 asvd.py 直接控制，包括 alpha、param_ratio_target、scaling_method 等
- **KVPress 參數**：由 kvpress_evaluator 內部管理，不在 ASVD tuner 中設定
- Tuner 的作用：優化 ASVD 超參數，評估不同配置的性能

### 模式 1：使用本地 LLM（推薦 - 無需付費）

**最簡單的使用方式**：直接運行，無需任何配置

```bash
python asvd_tuner.py --max_iterations 10
```

**工作流程**：
1. ✅ 自動使用 evaluator 內的本地 LLM 模型
2. ✅ 無需 OpenAI API 密鑰
3. ✅ LLM 針對 ASVD 超參數提供建議
4. ✅ 完全免費

**日誌輸出**：
```
INFO - LLM mode: Local LLM fallback
INFO - Asking LLM for next configuration...
DEBUG - Using local LLM fallback
INFO - LLM suggested config: {'alpha': 0.5, 'param_ratio_target': 0.8, ...}
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

**特點**：
- 需要 OpenAI API 密鑰
- 需要網路連接
- 建議質量更高（使用 GPT-4o）
- 成本：約 ¢ 每次調用

## 使用流程

### 1. 基本使用

```python
from asvd_tuner import ASVDAdaptiveTuner
from sys import path
from pathlib import Path

# 添加路徑以導入評估器
path.insert(0, str(Path(__file__).parent.parent))

# 初始化評估器
import yaml
with open("../tmp/config/model_config.yaml") as f:
    config = yaml.safe_load(f)
from eval.base_evaluator import GSM8KEvaluator
evaluator = GSM8KEvaluator(config)

# 創建調整器
tuner = ASVDAdaptiveTuner(
    evaluator=evaluator,
    model_id="facebook/opt-1.3b",
    task="gsm8k",
    max_iterations=10,
    accuracy_weight=0.7,      # 任務準確性權重
    compression_weight=0.2,   # 壓縮比權重
    ppl_weight=0.1           # PPL 權重
)

# 執行優化
results = tuner.optimize()

# 保存結果
tuner.save_results(results, "asvd_tuning_results.json")
```

### 2. 完整工作流程

```
ASVDAdaptiveTuner.optimize()
    ↓
[迴圈 1 to max_iterations]
    ↓
LLM 建議超參數配置
    ↓
執行 asvd.py (生成 SVD 緩存)
    ↓
執行 build_asvd_repo.py (構建模型)
    ↓
evaluator.evaluate() (評估模型)
    ↓
計算綜合分數
    ↓
更新試驗歷史和最佳配置
    ↓
[返回最佳配置]
```

### 3. 分布式流程

整個優化流程涉及多個檔案：

```
ASVD4LLM-main/
├─ asvd_tuner.py          ← 優化主邏輯
├─ asvd.py                ← 執行 ASVD 實驗
├─ huggingface_repos/
│  └─ build_asvd_repo.py  ← 構建 HuggingFace 模型
└─ asvd_tuning_results.json ← 輸出結果

lulu_temp/
├─ eval/
│  └─ base_evaluator.py   ← 評估器基類
└─ ...

tmp/
├─ config/
│  └─ model_config.yaml   ← 評估配置
└─ test/
   └─ test_eval.py        ← 評估腳本
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

## LLM 模式比較

| 特性 | OpenAI API | 本地 LLM Fallback |
|------|-----------|------------------|
| **成本** | 按次數計費 (~¢ 每次) | 免費 |
| **建議質量** | 更好（GPT-4o） | 基本（本地模型） |
| **依賴** | 需要網路 + API 密鑰 | 無需外部依賴 |
| **推薦場景** | 生產環境、關鍵優化 | 開發/測試、成本敏感 |
| **自動切換** | 無需手動設置 - 自動 fallback |

**推薦**：默認使用本地 LLM（無需配置），只在需要更好質量時啟用 OpenAI API。

## 輸出結果

### 結果結構

```json
{
  "best_result": {
    "iteration": 5,
    "params": {
      "alpha": 0.5,
      "param_ratio_target": 0.8,
      "scaling_method": "abs_mean",
      "sensitivity_metric": "ppl",
      "weight_quant": "none",
      "compress_kv_cache": false,
      "kv_cache_ratio_target": null
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
  path: ASVD4LLM-main/huggingface_repos/opt-125m-asvd80

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
