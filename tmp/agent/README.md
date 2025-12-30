# 多目標量化優化框架

一個全面的框架，用於自動優化跨多個衝突目標的量化參數。

## 🎯 概述

本框架使用**多目標優化**（Optuna TPE/NSGA-II）來找到以下目標之間的 Pareto 最優權衡：

1. **準確率下降**（最小化）
2. **GPU 峰值減少**（最大化）
3. **延遲增加**（最小化）

不是尋找單一的「最佳」配置，而是發現 **Pareto 前沿**：一組代表不同權衡的配置集合。

## 📦 安裝

### 依賴套件

```bash
# 核心依賴
pip install optuna pyyaml

# 視覺化（可選）
pip install plotly kaleido

# 量化方法（已安裝）
# - gptqmodel
# - autoawq
# - bitsandbytes
```

## 🚀 快速開始

### 1. 快速測試（15-20 分鐘）

```bash
python tmp/test/test_optimization.py \
  --config tmp/config/optimization_config_quick.yaml
```

### 2. 完整優化（1-2 小時）

```bash
python tmp/test/test_optimization.py \
  --config tmp/config/optimization_config.yaml
```

### 3. 查看結果

```bash
# 查看摘要
cat results/optimization/experiment_name_timestamp/summary.json

# 查看視覺化
open results/optimization/experiment_name_timestamp/pareto_3d.html
```

## 📁 架構

```
tmp/agent/
├── config_loader.py              # 配置載入
├── baseline_evaluator.py         # 基線模型評估
├── evaluator_agent.py            # 量化 + 評估
├── optimization_orchestrator.py  # 主控制器
├── result_tracker.py             # 結果保存
├── visualization.py              # Pareto 前沿圖表
└── optimizers/
    ├── base_optimizer.py         # 抽象基類
    ├── optuna_mo_optimizer.py    # Optuna 多目標（TPE/NSGA-II）
    └── random_optimizer.py       # 隨機搜索基線
```

## ⚙️ 配置

編輯 `tmp/config/optimization_config.yaml`：

### 關鍵參數

```yaml
# 基線評估
baseline:
  model_name: "meta-llama/Llama-3.2-1B"
  datasets:
    gsm8k:
      enabled: true
      num_samples: 100    # 自定義樣本數量

# 目標和約束
multiobjective:
  targets:                # 期望目標
    accuracy_drop_max: 0.10
    gpu_peak_reduction_min: 0.20
    latency_increase_max: 0.30

  constraints:            # 硬約束（剪枝）
    enabled: true
    accuracy_drop_max: 0.15
    gpu_peak_reduction_min: 0.10

  aggregation:
    method: "weighted"    # weighted / average / worst_case
    dataset_weights:
      gsm8k: 0.40
      truthfulqa: 0.30

# 優化器
optimizer:
  type: "optuna_multiobjective"  # 或 "random"
  optuna:
    n_trials: 50          # 總試驗次數
    sampler: "TPESampler" # 或 "NSGAIISampler"

  search_space:
    methods: ["gptq", "awq", "bnb"]
    gptq:
      bits: [4]
      group_size: [32, 64, 128, 256]
      calib_num: [256, 512, 1024]
```

## 📊 輸出

### 結果結構

```
results/optimization/experiment_name_timestamp/
├── summary.json              # 簡潔摘要
├── full_results.json         # 完整結果
├── pareto_frontier.json      # Pareto 最優解
├── pareto_3d.html            # 3D 互動圖表
├── parallel_coordinates.html # 平行座標圖
└── trade_off_matrix.html     # 2x2 權衡矩陣
```

### 範例摘要

```json
{
  "baseline": {
    "accuracy": 0.8092,
    "gpu_peak_mb": 3300,
    "avg_latency_ms": 1340
  },
  "optimization": {
    "pareto_solutions": 12,
    "satisfying_solutions": 5
  },
  "recommended": {
    "method": "gptq",
    "config": {"bits": 4, "group_size": 64, "calib_num": 1024},
    "objectives": {
      "accuracy_drop": 0.07,
      "gpu_peak_reduction": 0.28,
      "latency_increase": 0.15
    },
    "satisfies_targets": true
  }
}
```

## 🔧 進階使用

### 自定義聚合方式

```yaml
aggregation:
  method: "worst_case"  # 所有資料集都必須表現良好
```

### 不同的採樣器

```yaml
optuna:
  sampler: "NSGAIISampler"  # 遺傳演算法（對於 3+ 個目標更好）
  nsgaii_params:
    population_size: 20
    mutation_prob: 0.1
```

### 停用視覺化

```yaml
output:
  visualization:
    enabled: false
```

## 📈 理解結果

### Pareto 前沿

框架會返回位於 **Pareto 前沿**上的多個解決方案：

- **解決方案 A**：accuracy_drop=5%、gpu=24%、latency=22%
- **解決方案 B**：accuracy_drop=7%、gpu=28%、latency=15%
- **解決方案 C**：accuracy_drop=3%、gpu=22%、latency=28%

每個代表不同的權衡。根據您的優先順序選擇！

### 推薦策略

```yaml
recommendation:
  strategy: "closest_to_target"   # 最接近用戶目標
  # 或者："best_accuracy" / "best_compression" / "balanced"
```

## 🐛 故障排除

### 記憶體不足

在配置中減少樣本數量：

```yaml
baseline:
  datasets:
    gsm8k:
      num_samples: 50  # 從 100 減少
```

### 找不到 Optuna

```bash
pip install optuna
```

### 沒有有效的解決方案

放寬約束：

```yaml
constraints:
  accuracy_drop_max: 0.20  # 從 0.15 增加
```

## 📚 引用

如果您使用本框架，請引用：

```bibtex
@software{multi_objective_quant,
  title = {Multi-objective Quantization Optimization Framework},
  year = {2025},
  author = {Your Name}
}
```

## 📝 授權

MIT License

## 🤝 貢獻

歡迎貢獻！可改進的領域：

- [ ] qEHVI 優化器實現（BoTorch）
- [ ] 多 GPU 並行評估
- [ ] 超體積指標計算
- [ ] 更多視覺化
- [ ] 支援更多量化方法
