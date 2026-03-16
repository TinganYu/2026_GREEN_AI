# Global Tuner v2

LLM 壓縮自動化優化框架，透過 LLM Agent（GPT-4o）循環決策，對目標模型依序嘗試量化、稀疏化、低秩分解等壓縮策略，最大化壓縮後的評估分數。

## 架構

```
OptimizationOrchestrator
    ├── LLMDecisionMaker      ← GPT-4o 根據歷史決定下一組策略
    ├── Executors             ← 執行壓縮（ASVD / SparseGPT / 量化）
    ├── run_evaluation        ← 評估壓縮後模型（accuracy / latency / VRAM / CO2）
    └── utils.get_pareto_frontier  ← 追蹤 Pareto 最佳解
```

**主要流程：**

```
載入 baseline（快取）→ 儲存 experiment_config.json
    for i in max_iterations:
        LLM 分析歷史 + Pareto → 建議策略
        執行壓縮（sparse → asvd → quant，視 mode 而定）
        評估 → 計算 score
        更新 trial history + optimization_results.json
cleanup（保留最佳，刪除其餘 trial 模型）
```

---

## 執行

```bash
python -m Global_Tuner_v2.modular_agent.orchestrator \
  --model_id meta-llama/Llama-3.2-1B-Instruct \
  --task gsm8k \
  --max_iterations 10 \
  --num_samples 100 \
  --acc_weight 0.5 \
  --lat_weight 0.1 \
  --vram_weight 0.2 \
  --emit_weight 0.2 \
  --no-cleanup \
  > tuner_v2.log 2>&1
```

### 參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--model_id` | `meta-llama/Llama-3.2-1B-Instruct` | HuggingFace 模型 ID 或本地路徑 |
| `--task` | `gsm8k` | 評估資料集，可逗號分隔多個（e.g. `gsm8k,truthfulqa`） |
| `--max_iterations` | `10` | 優化輪數 |
| `--num_samples` | `None`（全量） | 每個資料集的評估樣本數 |
| `--acc_weight` | `0.7` | 分數公式中 accuracy 的權重 |
| `--lat_weight` | `0.1` | latency 權重（越低越好） |
| `--vram_weight` | `0.2` | VRAM 權重（越低越好） |
| `--emit_weight` | `0.0` | CO2 排放量權重 |
| `--no-cleanup` | — | 保留所有 trial 模型（預設跑完後刪除非最佳） |
| `--no-keep-best` | — | cleanup 時連最佳 trial 模型也刪除 |

### 環境變數（`.env`）

```
LLM_API_KEY=sk-...          # OpenAI API key（必填）
LLM_MODEL=gpt-4o            # 使用的 LLM（選填，預設 gpt-4o）
HF_TOKEN=hf_...             # HuggingFace token（下載受限模型需要）
```

---

## 分數公式

```
score = acc_w  × (Acc / Base_acc)
      + lat_w  × (Base_lat / Lat)
      + vram_w × (Base_vram / VRAM)
      + emit_w × (Base_emit / Emit)
```

`score > 1.0` 代表整體優於未壓縮基線。

---

## 支援的壓縮模式（LLM 可選）

### `quant_only`

| 方法 | bits | group_size | 說明 |
|------|------|------------|------|
| **GPTQ** | 2/3/4/8 | -1,32,64,128,256 | Hessian-based，最通用 |
| **AWQ** | 固定 4 | 16,32,64,128 | Activation-aware，需 Ampere+ GPU |
| **QQQ** | 固定 4 | -1,128 | W4A8，INT8 GEMM 推理最快 |
| **BNB** | 4/8 | — | On-the-fly，不儲存檔案，適合 hybrid |

### `sparse_only`

| 模式 | 說明 |
|------|------|
| `unstructured` | 非結構化剪枝，ratio 可設 0.3–0.7 |
| `2:4` | 固定 50%，需 Ampere+ GPU sparse tensor core |
| `4:8` | 固定 50%，硬體相容性較 2:4 廣 |

### `asvd_only`

低秩分解（SVD），降低 latency。`param_ratio_target` 設越高保留越多參數（精度損失越小）。

### `hybrid`

組合模式：
- `sparse → quant`：先稀疏化再量化（GPTQ/AWQ/QQQ）
- `asvd → bnb`：低秩分解後 BNB 量化（唯一相容的 hybrid）

---

## 輸出結構

```
tuning_results/
├── baselines/
│   └── Llama-3.2-1B-Instruct_gsm8k.json   ← baseline 快取（跨實驗共用）
│
└── exp_Llama-3.2-1B-Instruct_gsm8k_20260308_131806/
    ├── experiment_config.json               ← 實驗設定（weights、baseline 分數）
    ├── optimization_results.json            ← 所有 trial 結果
    ├── trial_001_gptq_4bit_g128_gptq/
    │   ├── config.json, *.safetensors ...  ← 壓縮後模型
    │   └── gsm8k_results.json              ← 評估詳細結果
    ├── trial_002_awq_4bit_g128/
    └── trial_003_sparse_40pct_gptq_4bit_g128_gptq/
```

### `experiment_config.json`

```json
{
  "model_id": "meta-llama/Llama-3.2-1B-Instruct",
  "task": "gsm8k",
  "max_iterations": 10,
  "num_samples": 100,
  "weights": { "acc": 0.7, "lat": 0.1, "vram": 0.2, "emit": 0.0 },
  "cleanup": true,
  "keep_best": true,
  "baseline": {
    "accuracy": 0.367,
    "latency": 18.13,
    "vram": 2.42,
    "emissions": 0.00079
  }
}
```

### `optimization_results.json`

每個 trial 的記錄格式：

```json
{
  "iteration": 1,
  "trial_name": "trial_001_gptq_4bit_g128_gptq",
  "config": { "mode": "quant_only", "quant": { ... } },
  "suggestion": { "reasoning": "...", "mode": "quant_only", "quant_method": "gptq", ... },
  "metrics": {
    "accuracy": 0.26, "latency": 25.88, "vram": 1.16,
    "emissions": 0.00082, "score": 1.24,
    "details": { "gsm8k": { ... } }
  },
  "model_path": "tuning_results/exp_.../trial_001_...",
  "trial_dir":  "tuning_results/exp_.../trial_001_..."
}
```

失敗的 trial 會有 `"error": "..."` 欄位，`model_path` 為 `null`。

### Trial 命名規則

```
trial_{編號}_{sparse_METHOD|sparse_NxM}_{asvd_rRATIO}_{方法_bits_g群組}
```

範例：
- `trial_001_gptq_4bit_g128_gptq`
- `trial_002_sparse_40pct_gptq_4bit_g128_gptq`
- `trial_003_sparse_2x4`
- `trial_004_asvd_r090_a50`
- `trial_005_asvd_r092_a50_bnb_4bit`

---

## 模組說明

| 檔案 | 職責 |
|------|------|
| `orchestrator.py` | 主控制器，管理優化迴圈、baseline 快取、cleanup |
| `llm_client.py` | GPT-4o prompt 構建、回應解析 |
| `schemas.py` | `StrategySuggestion` pydantic model，定義所有可調參數 |
| `executors.py` | 橋接 `StrategySuggestion` → `Method/` 介面 |
| `utils.py` | Pareto frontier 計算 |
