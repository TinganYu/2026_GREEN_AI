# Systematic Tuner

以 **Optuna 演算法**（TPE / NSGA-II / Random）自動搜尋 LLM 壓縮配置，取代 Global_Tuner_v2 的 LLM 決策。每個 trial 在獨立子進程（`spawn`）執行，保證 VRAM 完整釋放。

結果統一存放於 `Green_AI/systematic_results/`。

---

## 搜尋空間

搜尋空間採混合設計：連續參數用範圍取樣，離散合法值用 categorical。

### ASVD（低秩分解）

| 參數 | 類型 | 範圍 / 選項 | 預設 | 說明 |
|------|------|------------|------|------|
| `alpha` | float linear | 0.3 ~ 0.7 | 0.5 | 激活感知縮放強度，越大越保留激活分佈 |
| `param_ratio_target` | float linear | 0.70 ~ 0.99 | 0.9 | 保留參數比例，越低壓縮越多 |
| `scaling_method` | categorical | `abs_mean`, `abs_max`, `fisher` | `fisher` | 奇異值縮放方式，`fisher` 通常最好 |

### GPTQ

| 參數 | 類型 | 範圍 / 選項 | 預設 | 說明 |
|------|------|------------|------|------|
| `quant_bits` | categorical | `2, 3, 4, 8` | 4 | 量化位元數，4bit 為主流平衡點 |
| `quant_group_size` | categorical | `16, 32, 64, 128, 256` | 128 | 分組大小；越小精度越高但檔案越大 |
| `quant_format` | categorical | `gptq`, `gptq_v2` | `gptq` | gptq_v2 修正了 v1 的溢位問題，通常略優 |
| `damp_percent` | float **log** | 0.001 ~ 0.1 | 0.05 | Hessian 穩定項，實務常用 0.01~0.05；log scale 避免低值探索不足 |
| `mse` | — | 固定 0.0 | 0.0 | 已移除：幾乎都用 0.0，放進 search space 浪費 trial |

### AWQ

| 參數 | 類型 | 範圍 / 選項 | 預設 | 說明 |
|------|------|------------|------|------|
| `quant_bits` | — | 固定 4 | 4 | AWQ 只支援 4bit |
| `quant_group_size` | categorical | `16, 32, 64, 128` | 128 | 同 GPTQ，越小精度越高 |

### QQQ

| 參數 | 類型 | 範圍 / 選項 | 預設 | 說明 |
|------|------|------------|------|------|
| `quant_bits` | — | 固定 4 | 4 | QQQ 只支援 4bit |
| `quant_group_size` | categorical | `-1, 128` | 128 | -1=全矩陣（精度高但慢），128=常規 |
| `damp_percent` | float **log** | 0.0005 ~ 0.05 | 0.005 | 同 GPTQ dampening；log scale |

### BNB（BitsAndBytes）

| 參數 | 類型 | 範圍 / 選項 | 預設 | 說明 |
|------|------|------------|------|------|
| `quant_bits` | categorical | `4, 8` | 4 | 4bit 更省 VRAM；8bit 精度更高 |
| `quant_type` | — | 固定 `nf4` | `nf4` | 已移除：nf4 針對常態分佈設計幾乎必贏 fp4 |
| `use_double_quant` | categorical | `False, True` | `False` | 只在 bits=4 有效；對量化常數再量化，額外省 ~0.4 bit/param |

### SparseGPT Unstructured

| 參數 | 類型 | 範圍 / 選項 | 預設 | 說明 |
|------|------|------------|------|------|
| `sparsity_ratio` | float linear | 0.3 ~ 0.7 | 0.5 | 稀疏比例，0.5 = 50% 權重歸零 |

### SparseGPT Structured

| 參數 | 類型 | 範圍 / 選項 | 預設 | 說明 |
|------|------|------------|------|------|
| `sparsity_structure` | categorical | `2:4`, `4:8` | — | 2:4 = 每 4 個保留 2 個，硬體加速友好 |

### Hybrid（ASVD + BNB）

ASVD 參數同上 + BNB 參數同上。

---

## 取樣方式說明

| 標記 | Optuna API | 說明 |
|------|-----------|------|
| float **linear** | `suggest_float(low, high)` | 均勻取樣，適合範圍內無數量級差距的參數 |
| float **log** | `suggest_float(low, high, log=True)` | 對數取樣，0.001~0.01 和 0.01~0.1 各佔一半探索量，適合 damp 類參數 |
| categorical | `suggest_categorical(choices)` | 只從合法值中選，適合只能是特定值的參數（bits、group_size）|

---

## 執行方式

```bash
# Optuna TPE（推薦起點）
python -m Systematic_Tuner.orchestrator \
  --model_id meta-llama/Llama-3.2-1B-Instruct \
  --task gsm8k \
  --search_method optuna
```

---

## 參數說明

### 基本參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--model_id` | `meta-llama/Llama-3.2-1B-Instruct` | HuggingFace model ID 或本地路徑 |
| `--task` | `gsm8k` | 評估資料集，逗號分隔多個（如 `gsm8k,truthfulqa`）|
| `--search_method` | `optuna` | `optuna`（TPE/NSGA-II/Random 演算法）|
| `--max_iterations` | `20` | 試驗次數 |
| `--num_samples` | `None`（全部）| 每個 dataset 評估幾筆樣本，設小可加速 |
| `--modes` | 全部 8 種 | 過濾要搜尋的模式，空格分隔 |
| `--seed` | `42` | 亂數種子，固定可重現結果 |
| `--no-cleanup` | — | 跑完後保留所有 trial 模型（預設只保留最佳）|
| `--no-keep-best` | — | 跑完後連最佳 trial 模型也清除 |

### 評分權重

Score 公式（log scale）：

```
score = 1.0 + acc_weight  × log(acc / base_acc)
             + lat_weight  × log(base_lat / lat)
             + vram_weight × log(base_vram / vram)
             + emit_weight × log(base_emit / emit)
```

> `score > 1.0` 表示優於未壓縮基線；`score < 1.0` 表示退步。

| 參數 | 預設 | 說明 |
|------|------|------|
| `--acc_weight` | `0.6` | 準確率權重（最重要） |
| `--lat_weight` | `0.1` | 推理延遲權重 |
| `--vram_weight` | `0.1` | GPU 記憶體權重 |
| `--emit_weight` | `0.2` | CO₂ 排放權重 |

---

## Optuna Sampler 參數

### `--optuna_sampler`

| Sampler | 適用情境 |
|---------|----------|
| `tpe`（預設）| 大多數情況首選，從過去結果學習，效率高 |
| `nsga2` | 想看多目標 Pareto 分布時使用 |
| `random` | 作為基準比較，或 debug 用 |

### `--n_startup_trials`（TPE 專用，預設 10）

TPE 在開始學習之前先做純隨機探索的次數。

| `max_iterations` | 建議值 | 說明 |
|-----------------|--------|------|
| ≤ 15 | 3–5 | 避免大半都是隨機，讓 TPE 有足夠學習 |
| 20–50 | 5–10 | 預設 10 適合此範圍 |
| > 50 | 10–15 | 可稍微拉高，讓初始探索更多樣 |

### `--population_size`（NSGA-II 專用，預設 50）

遺傳演算法每代的族群大小。需滿足 `max_iterations >= population_size`，否則連第一代都跑不完。

| `max_iterations` | 建議 `population_size` |
|-----------------|----------------------|
| 20 | 8–10 |
| 50 | 15–20 |
| 100 | 30–50 |

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
# 只搜 GPTQ
python -m Systematic_Tuner.orchestrator \
  --search_method optuna \
  --max_iterations 60 \
  --n_startup_trials 10 \
  --modes gptq
```

### 省 VRAM 優先

```bash
python -m Systematic_Tuner.orchestrator \
  --search_method optuna \
  --acc_weight 0.4 --vram_weight 0.4 --emit_weight 0.1 --lat_weight 0.1 \
  --modes gptq awq qqq bnb
```

### 省電優先

```bash
python -m Systematic_Tuner.orchestrator \
  --search_method optuna \
  --acc_weight 0.4 --emit_weight 0.4 --vram_weight 0.1 --lat_weight 0.1
```

---

## 輸出結構

```
systematic_results/
└── optuna_{model}_{task}_{timestamp}/
    ├── experiment_config.json     # 本次實驗設定 + baseline 數值
    ├── optimization_results.json  # 所有 trial 結果（每輪更新）
    └── trial_NNN_*/               # 最佳 trial 的模型（其餘被清理）

systematic_results/baselines/
└── {model}_{task}.json            # baseline 快取，下次自動重用
```
