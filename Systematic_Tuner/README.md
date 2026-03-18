# Systematic Tuner

以 **Grid Search** 或 **Optuna 演算法** 自動搜尋 LLM 壓縮配置，取代 Global_Tuner_v2 的 LLM 決策。

結果統一存放於 `Green_AI/systematic_results/`。

---

## 搜尋空間大小

| 模式 | 組合數 | 說明 |
|------|-------:|------|
| `asvd_only` | 90 | 5×6×3 (alpha × ratio × scaling) |
| `gptq` | 640 | 4×5×2×4×4 (bits × group × format × damp × mse) |
| `awq` | 4 | group_size 4 種 |
| `qqq` | 6 | 2×3 (group × damp) |
| `bnb` | 3 | bits=4×2 + bits=8×1 |
| `sparse_unstructured` | 5 | sparsity_ratio 5 種 |
| `sparse_structured` | 2 | 2:4 / 4:8 |
| `hybrid_asvd_bnb` | 270 | 90×3 |
| **Total** | **1,020** | Grid Search 全跑的上限 |

---

## 執行方式

```bash
# Optuna TPE（推薦起點）
python -m Systematic_Tuner.orchestrator \
  --model_id meta-llama/Llama-3.2-1B-Instruct \
  --task gsm8k \
  --search_method optuna

# Grid Search（僅搜部分模式，避免全跑 4220 次）
python -m Systematic_Tuner.orchestrator \
  --search_method grid \
  --modes gptq awq bnb
```

---

## 參數說明

### 基本參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--model_id` | Llama-3.2-1B-Instruct | HuggingFace model ID 或本地路徑 |
| `--task` | gsm8k | 評估資料集，逗號分隔多個（如 `gsm8k,truthfulqa`）|
| `--search_method` | optuna | `grid`（枚舉）或 `optuna`（演算法）|
| `--max_iterations` | 20 | Optuna 試驗次數（grid 模式忽略此值，自動跑完所有配置）|
| `--num_samples` | None（全部）| 每個 dataset 評估幾筆樣本，設小可加速 |
| `--modes` | 全部 9 種 | 過濾要搜尋的模式，空格分隔 |
| `--seed` | 42 | 亂數種子，固定可重現結果 |
| `--no-cleanup` | — | 跑完後保留所有 trial 模型（預設只保留最佳）|

### 評分權重

分數 = `acc_weight × (acc/base_acc)` + `lat_weight × (base_lat/lat)` + `vram_weight × (base_vram/vram)` + `emit_weight × (base_emit/emit)`

> 分數 > 1.0 表示優於未壓縮基線。

| 參數 | 預設 | 說明 |
|------|------|------|
| `--acc_weight` | 0.6 | 準確率權重 |
| `--lat_weight` | 0.1 | 推理延遲權重 |
| `--vram_weight` | 0.1 | GPU 記憶體權重 |
| `--emit_weight` | 0.2 | CO₂ 排放權重 |

---

## Optuna Sampler 參數

### `--optuna_sampler`

| Sampler | 適用情境 |
|---------|----------|
| `tpe`（預設）| 大多數情況首選，從過去結果學習，效率高 |
| `nsga2` | 想看多目標 Pareto 分布時使用 |
| `random` | 作為基準比較，或 debug 用 |

### `--n_startup_trials`（TPE 專用）

TPE 在開始學習之前先做純隨機探索的次數。

| `max_iterations` | 建議值 | 說明 |
|-----------------|--------|------|
| ≤ 15 | 3–5 | 避免大半都是隨機，讓 TPE 有足夠學習 |
| 20–50 | 5–10 | 預設 10 適合此範圍 |
| > 50 | 10–15 | 可稍微拉高，讓初始探索更多樣 |

```bash
# 只跑 20 次時建議降低 n_startup_trials
python -m Systematic_Tuner.orchestrator \
  --search_method optuna --max_iterations 20 --n_startup_trials 5
```

### `--population_size`（NSGA-II 專用）

遺傳演算法每代的族群大小。需滿足 `max_iterations >= population_size`，否則連第一代都跑不完。

| `max_iterations` | 建議 `population_size` |
|-----------------|----------------------|
| 20 | 8–10 |
| 50 | 15–20 |
| 100 | 30–50（預設值適合）|

```bash
python -m Systematic_Tuner.orchestrator \
  --search_method optuna --optuna_sampler nsga2 \
  --max_iterations 30 --population_size 10
```

---

## 建議搜尋策略

### 快速探索（< 1 小時）
目的：了解哪個模式值得深入。

```bash
python -m Systematic_Tuner.orchestrator \
  --search_method optuna \
  --max_iterations 20 \
  --n_startup_trials 5 \
  --num_samples 50 \
  --modes gptq awq bnb sparse_unstructured
```

### 重點精搜（已知最佳模式）
目的：在單一模式內找最佳參數組合。

```bash
# 只搜 GPTQ（640 種組合）
python -m Systematic_Tuner.orchestrator \
  --search_method optuna \
  --max_iterations 60 \
  --n_startup_trials 10 \
  --modes gptq
```

### 全面 Grid Search（有充裕時間）
建議縮小 modes 範圍，否則全跑 4,220 次費時極長。

```bash
# 只跑量化方法（共 653 種）
python -m Systematic_Tuner.orchestrator \
  --search_method grid \
  --modes gptq awq qqq bnb \
  --num_samples 100
```

### 省 VRAM 優先
調高 `vram_weight`，降低其他權重。

```bash
python -m Systematic_Tuner.orchestrator \
  --search_method optuna \
  --acc_weight 0.4 --vram_weight 0.4 --emit_weight 0.1 --lat_weight 0.1 \
  --modes gptq awq qqq bnb
```

### 省電優先
調高 `emit_weight`。

```bash
python -m Systematic_Tuner.orchestrator \
  --search_method optuna \
  --acc_weight 0.4 --emit_weight 0.4 --vram_weight 0.1 --lat_weight 0.1
```

---

## 輸出結構

```
systematic_results/
└── systematic_{method}_{model}_{task}_{timestamp}/
    ├── experiment_config.json   # 本次實驗設定 + baseline
    ├── optimization_results.json  # 所有 trial 結果（每輪更新）
    └── trial_NNN_*/             # 最佳 trial 的模型（其餘被清理）

systematic_results/baselines/
└── {model}_{task}.json          # baseline 快取，下次重用
```
