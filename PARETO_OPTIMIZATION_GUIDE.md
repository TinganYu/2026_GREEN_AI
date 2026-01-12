# Pareto 多目標優化與參數選擇演算法指南

## 📋 目錄

1. [什麼是多目標優化](#1-什麼是多目標優化)
2. [Pareto 前沿的定義](#2-pareto-前沿的定義)
3. [支配關係的判斷邏輯](#3-支配關係的判斷邏輯)
4. [我們的實作：三目標量化優化](#4-我們的實作三目標量化優化)
5. [優化演算法：Optuna](#5-優化演算法optuna)
6. [實際案例分析](#6-實際案例分析)
7. [配置推薦策略](#7-配置推薦策略)

---

## 1. 什麼是多目標優化？

### 單目標 vs 多目標

**單目標優化：**
- 目標：找到使 `f(x)` 最大或最小的 `x`
- 例子：找到準確率最高的模型配置
- 結果：唯一最優解

**多目標優化：**
- 目標：同時優化多個相互衝突的目標
- 例子：同時優化準確率、記憶體使用、推理速度
- 結果：一組 Pareto 最優解（trade-off）

### 為什麼需要多目標優化？

在模型量化中，我們面臨三個相互衝突的目標：

| 目標 | 期望 | 衝突 |
|------|------|------|
| **準確率** | 保持高準確率 | ⚔️ 低位元數量化會降低準確率 |
| **記憶體** | 減少 GPU 記憶體使用 | ⚔️ 高位元數需要更多記憶體 |
| **速度** | 加快推理速度 | ⚔️ 某些量化方法會增加計算開銷 |

**範例衝突：**
- GPTQ 2-bit: GPU 記憶體 ↓↓↓, 但準確率 ↓↓↓
- GPTQ 8-bit: 準確率 ↑, 但 GPU 記憶體 ↓
- BnB 4-bit: 速度慢 ↑↑, GPU 記憶體節省少 ↓

→ **沒有單一最優解**，只有一組權衡方案（Pareto 前沿）

---

## 2. Pareto 前沿的定義

### 2.1 支配關係（Dominance）

**定義：** 解 A **支配** 解 B（記作 `A ≻ B`），當且僅當：

1. **條件 1（弱支配）：** A 在所有目標上都 **不比** B 差
2. **條件 2（嚴格更優）：** A 在至少一個目標上 **嚴格優於** B

**數學表示：**

對於最小化問題，A 支配 B 如果：

```
∀i: f_i(A) ≤ f_i(B)  （在所有目標上不比 B 差）
∃j: f_j(A) < f_j(B)  （在至少一個目標上嚴格更優）
```

**重要：** 如果目標是最大化（如準確率），需要反轉比較方向：
- 最大化 `f` → 比較時看 `f(A) ≥ f(B)`

### 2.2 Pareto 最優解

**定義：** 一個解是 **Pareto 最優的**，如果：
- 沒有其他解支配它

**Pareto 前沿（Pareto Frontier）：**
- 所有 Pareto 最優解的集合
- 也稱為 **非支配解集合（Non-dominated Set）**

### 2.3 視覺化理解

```
準確率變化 (↑ maximize)
    │
 +1 │     ★ A
    │
  0 │     ★ B         ● E
    │
 -5 │     ★ C
    │
-10 │                 ● F
    │            ★ D
-20 │
    └───────────────────────────→ GPU 記憶體變化 (↓ minimize)
       -60  -50  -40  -30  -20

★ = Pareto 最優解（不被任何其他解支配）
● = 非 Pareto 解（被某些解支配）

範例：
- 解 E 被 B 支配：B 準確率更好，GPU 也更好
- 解 F 被 D 支配：D 準確率更好，GPU 也更好
- A, B, C, D 都在 Pareto 前沿上（相互之間沒有支配關係）
```

**Pareto 前沿的特性：**
1. 沿著前沿移動 = 改善一個目標必然犧牲另一個目標
2. 前沿上的點代表了不同的 trade-off 偏好
3. 選擇哪個點取決於用戶的需求和權重

---

## 3. 支配關係的判斷邏輯

### 3.1 通用判斷演算法

```python
def is_dominated(trial_A, trial_B, objectives):
    """
    檢查 trial_A 是否被 trial_B 支配

    Args:
        trial_A: 被檢查的解
        trial_B: 候選支配解
        objectives: 目標列表 [{'name': str, 'direction': 'maximize'/'minimize'}, ...]

    Returns:
        True if B 支配 A, False otherwise
    """
    better_or_equal_count = 0
    strictly_better_count = 0

    for obj in objectives:
        value_A = trial_A[obj['name']]
        value_B = trial_B[obj['name']]

        if obj['direction'] == 'maximize':
            # 對於最大化目標，B 的值越大越好
            if value_B > value_A:
                strictly_better_count += 1
                better_or_equal_count += 1
            elif abs(value_B - value_A) < 1e-9:  # 相等（浮點數誤差）
                better_or_equal_count += 1
        else:  # minimize
            # 對於最小化目標，B 的值越小越好
            if value_B < value_A:
                strictly_better_count += 1
                better_or_equal_count += 1
            elif abs(value_B - value_A) < 1e-9:  # 相等
                better_or_equal_count += 1

    # B 支配 A 需要：
    # 1. 在所有目標上不比 A 差（better_or_equal_count == len(objectives)）
    # 2. 在至少一個目標上嚴格更優（strictly_better_count > 0）
    return better_or_equal_count == len(objectives) and strictly_better_count > 0
```

### 3.2 計算 Pareto 前沿

```python
def get_pareto_frontier(trials):
    """
    從所有試驗中提取 Pareto 前沿

    時間複雜度: O(n²)，其中 n = 試驗數量
    """
    pareto_frontier = []

    for trial in trials:
        is_dominated = False

        # 檢查是否被任何其他試驗支配
        for other_trial in trials:
            if trial == other_trial:
                continue

            if is_dominated(trial, other_trial, objectives):
                is_dominated = True
                break  # 找到一個支配者就足夠了

        if not is_dominated:
            pareto_frontier.append(trial)

    return pareto_frontier
```

**效率優化：**
- 原始算法：O(n²) - 每個試驗與所有其他試驗比較
- 快速非支配排序（Fast Non-dominated Sorting）：O(n² log n)
- 適用於大規模問題（1000+ 試驗）

---

## 4. 我們的實作：三目標量化優化

### 4.1 目標定義

我們的量化優化有三個目標（**全部相對於 baseline**）：

| 目標 | 變數名 | 方向 | 計算公式 | 期望 |
|------|--------|------|---------|------|
| **準確率變化** | `accuracy_change` | **Maximize** | `(量化 - 基準) / 基準` | **≥ -10%** (損失 ≤10%) |
| **GPU 峰值變化** | `gpu_peak_change` | **Minimize** | `(量化 - 基準) / 基準` | **≤ -50%** (節省 ≥50%) |
| **延遲變化** | `latency_change` | **Minimize** | `(量化 - 基準) / 基準` | **≤ +150%** (變慢 ≤1.5倍) |

**注意方向：**
- `accuracy_change = -0.10` → 準確率下降 10% → 希望最大化（接近 0）
- `gpu_peak_change = -0.50` → GPU 減少 50% → 希望最小化（越負越好）
- `latency_change = +0.20` → 延遲增加 20% → 希望最小化（越小越好）

### 4.2 Optuna 統一表示

為了使用 Optuna 的 `minimize` 方向，我們將準確率轉換：

```python
# Optuna Study 設置（全部最小化）
study = optuna.create_study(
    directions=['minimize', 'minimize', 'minimize'],  # 三個目標都是 minimize
    sampler=sampler
)

# 目標函數返回值
def objective(trial):
    # ... 評估量化配置 ...

    return (
        -accuracy_change,     # 轉換：maximize → minimize（取負號）
        gpu_peak_change,      # 不變：minimize
        latency_change        # 不變：minimize
    )
```

**但在判斷支配時，我們使用原始邏輯：**
- `accuracy_change`: 比較時用 `>` (maximize)
- `gpu_peak_change`: 比較時用 `<` (minimize)
- `latency_change`: 比較時用 `<` (minimize)

### 4.3 實際判斷代碼

**位置：** `tmp/agent/optimizers/base_optimizer.py:246-308`

```python
def get_pareto_frontier(self, trials):
    """從試驗中提取 Pareto 前沿"""
    valid_trials = [t for t in trials if t.get('success', False)]
    pareto_frontier = []

    for trial in valid_trials:
        is_dominated = False
        obj = trial['objectives']

        for other_trial in valid_trials:
            if trial == other_trial:
                continue

            other_obj = other_trial['objectives']

            # 比較每個目標（注意方向！）
            better_accuracy = other_obj['accuracy_change'] > obj['accuracy_change']  # maximize
            better_gpu = other_obj['gpu_peak_change'] < obj['gpu_peak_change']      # minimize
            better_latency = other_obj['latency_change'] < obj['latency_change']    # minimize

            equal_accuracy = abs(other_obj['accuracy_change'] - obj['accuracy_change']) < 1e-9
            equal_gpu = abs(other_obj['gpu_peak_change'] - obj['gpu_peak_change']) < 1e-9
            equal_latency = abs(other_obj['latency_change'] - obj['latency_change']) < 1e-9

            # 檢查支配性
            dominates_accuracy = better_accuracy or equal_accuracy
            dominates_gpu = better_gpu or equal_gpu
            dominates_latency = better_latency or equal_latency

            strictly_better = better_accuracy or better_gpu or better_latency

            # other_trial 支配 trial 如果：
            # 1. 在所有三個目標上都不比 trial 差
            # 2. 在至少一個目標上嚴格更優
            if dominates_accuracy and dominates_gpu and dominates_latency and strictly_better:
                is_dominated = True
                break

        if not is_dominated:
            pareto_frontier.append(trial)

    return pareto_frontier
```

---

## 5. 優化演算法：Optuna

### 5.1 Optuna 簡介

**Optuna** 是一個先進的超參數優化框架，支援：
- 單目標和多目標優化
- 自動剪枝（pruning）機制
- 多種採樣演算法
- 分佈式優化

**我們的配置：**
```yaml
optimizer:
  type: "optuna_multiobjective"
  optuna:
    n_trials: 20                    # 總試驗次數
    n_startup_trials: 5             # 隨機探索階段
    sampler: "TPESampler"           # 採樣器類型（或 NSGAIISampler）
```

### 5.2 採樣器 1：TPESampler（Tree-structured Parzen Estimator）

#### 工作原理

**TPE 是一種貝葉斯優化方法：**

1. **建模目標函數：**
   - 不直接建模 `P(y|x)`（給定參數 x，預測目標 y）
   - 而是建模 `P(x|y)`（給定目標值 y，預測參數 x）
   - 使用兩個分佈：
     - `l(x)` = 參數在「好」結果中的分佈
     - `g(x)` = 參數在「壞」結果中的分佈

2. **採樣新參數：**
   - 計算 Expected Improvement (EI)：`EI(x) ∝ l(x) / g(x)`
   - 選擇 EI 最大的參數作為下一個試驗
   - 平衡 exploration（探索新區域）vs exploitation（利用已知好區域）

3. **多目標擴展：**
   - 分別為每個目標建立 TPE 模型
   - 使用 **Pareto 支配計數** 作為好壞判斷標準
   - 被支配次數少的試驗被認為是「好」的

#### 參數說明

```yaml
tpe_params:
  n_ei_candidates: 24               # Expected Improvement 候選數
  multivariate: true                # 多變量建模（考慮參數間相關性）
  constant_liar: true               # 並行優化加速（模擬平行試驗結果）
```

**關鍵參數：**
- `n_startup_trials`: 初始隨機探索次數（建議 ≥ 參數數量）
- `n_ei_candidates`: 每次採樣時評估的候選參數數（越多越精確但越慢）
- `multivariate`: 是否考慮參數之間的相關性（建議開啟）

#### 適用場景

- ✅ **連續參數空間**（如 `damp_percent`）
- ✅ **小到中等規模問題**（< 100 試驗）
- ✅ **快速收斂**（較少試驗即可找到好解）
- ❌ 大規模離散空間（不如 NSGA-II）

### 5.3 採樣器 2：NSGAIISampler（Non-dominated Sorting Genetic Algorithm II）

#### 工作原理

**NSGA-II 是一種進化演算法：**

1. **初始化種群：**
   - 隨機生成 `population_size` 個解

2. **非支配排序（Non-dominated Sorting）：**
   - 將種群分層：
     - Front 0: 非支配解（Pareto 前沿）
     - Front 1: 被 Front 0 支配的解
     - Front 2: 被 Front 1 支配的解
     - ...

3. **擁擠度距離（Crowding Distance）：**
   - 在同一 front 中，計算每個解周圍的「稀疏程度」
   - 距離越大 = 周圍解越少 = 更具多樣性
   - 用於保持 Pareto 前沿的分佈均勻性

4. **選擇、交叉、突變：**
   - **選擇（Selection）：** 優先選擇 front 較低的解；同一 front 內選擇擁擠度距離大的
   - **交叉（Crossover）：** 兩個父代生成子代（`crossover_prob`）
   - **突變（Mutation）：** 隨機改變參數值（`mutation_prob`）

5. **精英策略：**
   - 父代和子代合併，重新排序
   - 保留最優的 `population_size` 個解
   - 確保最優解不會丟失

#### 參數說明

```yaml
nsgaii_params:
  population_size: 20               # 種群大小（建議 ≥ 目標數量 × 5）
  mutation_prob: 0.1                # 突變機率（10% 的基因突變）
  crossover_prob: 0.9               # 交叉機率（90% 的交叉操作）
```

**關鍵參數：**
- `population_size`: 種群大小（越大越能探索空間，但需要更多試驗）
- `mutation_prob`: 探索新區域（太低易陷入局部最優；太高收斂慢）
- `crossover_prob`: 組合現有好解（通常設為 0.8-0.9）

#### 適用場景

- ✅ **多個目標（≥ 3）**
- ✅ **大規模問題**（1000+ 試驗）
- ✅ **離散參數空間**（如 `bits = [2, 3, 4, 8]`）
- ✅ **保持 Pareto 前沿多樣性**
- ❌ 小規模問題（不如 TPE 快速收斂）

### 5.4 兩者對比

| 特性 | TPESampler | NSGAIISampler |
|------|-----------|---------------|
| **類型** | 貝葉斯優化 | 進化演算法 |
| **收斂速度** | ⚡ 快（10-30 trials） | 🐢 慢（50-100 trials） |
| **參數空間** | 連續、混合 | 離散、組合 |
| **多樣性** | 中等 | ⭐ 優秀（擁擠度機制） |
| **並行化** | 有限（constant_liar） | ⭐ 原生支持（種群） |
| **記憶體** | 低 | 中（需存儲種群） |
| **適用目標數** | 2-3 個 | ⭐ 3+ 個 |
| **推薦場景** | 快速原型、小規模 | 生產環境、大規模 |

### 5.5 參數搜索空間定義

**位置：** `tmp/config/optimization_config.yaml`

```yaml
search_space:
  methods: ["gptq", "awq", "bnb"]   # 類別參數（Optuna: suggest_categorical）

  gptq:
    bits: [2, 3, 4, 8]              # 離散選擇
    group_size: [-1, 16, 32, 64, 128]
    desc_act: [true, false]         # 布林參數
    sym: [true, false]
    damp_percent: [0.001, 0.03]     # 連續範圍（Optuna: suggest_float）
    damp_auto_increment: [0.001, 0.05]
    mse: [0.0, 0.2]
```

**Optuna 轉換邏輯：**

```python
def objective(trial: optuna.Trial):
    # 選擇量化方法
    method = trial.suggest_categorical('method', self.search_space['methods'])

    if method == 'gptq':
        # 離散選擇
        bits = trial.suggest_categorical('bits', self.search_space['gptq']['bits'])
        group_size = trial.suggest_categorical('group_size', self.search_space['gptq']['group_size'])

        # 布林參數
        desc_act = trial.suggest_categorical('desc_act', [True, False])

        # 連續參數
        damp_percent = trial.suggest_float('damp_percent', 0.001, 0.03)
        mse = trial.suggest_float('mse', 0.0, 0.2)
```

---

## 6. 實際案例分析

### 6.1 案例：為什麼 BnB 4bit 不在 Pareto 前沿？

**實驗數據：** `gemma-2-2b-it-multiobjective-opt_20260108_224919`

#### 完成的試驗

| Trial | Method | Acc Change | GPU Change | Latency Change | 在 Pareto？ |
|-------|--------|------------|------------|----------------|------------|
| **#8** | **BnB 4bit** | **-18.64%** | **-19.05%** | **+41.08%** | ❌ **否** |
| #7 | GPTQ 4bit gs=128 | -10.17% | -53.63% | +8.57% | ✅ 是 |
| #6 | GPTQ 4bit gs=128 | -1.69% | -53.69% | +47.34% | ✅ 是 |
| #1 | GPTQ 8bit gs=-1 | +1.69% | -36.49% | +41.35% | ✅ 是 |

#### 詳細支配分析

**檢查：GPTQ 4bit #7 是否支配 BnB 4bit #8？**

| 目標 | BnB 4bit (#8) | GPTQ 4bit (#7) | 比較結果 |
|------|--------------|----------------|---------|
| Acc Change (max) | -18.64% | -10.17% | **-10.17 > -18.64** ✓ (GPTQ 更好) |
| GPU Change (min) | -19.05% | -53.63% | **-53.63 < -19.05** ✓ (GPTQ 更好) |
| Latency Change (min) | +41.08% | +8.57% | **+8.57 < +41.08** ✓ (GPTQ 更好) |

**結論：**
- GPTQ #7 在 **所有三個目標** 上都優於 BnB #8
- → GPTQ #7 **支配** BnB #8
- → BnB #8 **不是** Pareto 最優解
- → BnB #8 **不在** Pareto 前沿中

#### 支配關係視覺化

```
準確率變化 (↑ maximize)
    │
+2% │         GPTQ 8bit (#1) ★
    │
  0 │
    │         GPTQ 4bit (#6) ★
 -2 │
    │
-10 │         GPTQ 4bit (#7) ★
    │                    ╲
    │                     ╲ 支配
-18 │                      ╲
    │                       ↘
-20 │                         BnB 4bit (#8) ●
    │
    └──────────────────────────────────────→ GPU 峰值變化 (↓ minimize)
       -60    -54    -40     -30     -20    -10

★ = Pareto 最優解（在前沿上）
● = 被支配解（不在前沿上）

可以看到 GPTQ 4bit (#7) 在準確率和 GPU 兩個維度上都比 BnB 4bit (#8) 更接近理想點。
```

### 6.2 案例：為什麼 GPTQ 2bit 在 Pareto 前沿？

雖然 GPTQ 2bit 的準確率損失很大（-95%），但它在 GPU 記憶體節省上是最優的（-59%），因此它提供了一個極端的 trade-off 選項：

| Trial | Method | Acc Change | GPU Change | Latency Change |
|-------|--------|------------|------------|----------------|
| #5 | GPTQ 2bit gs=16 | -94.92% | **-59.02%** | +114.12% |

**為什麼在前沿？**
- 沒有其他解能在保持相似準確率的同時提供更好的 GPU 節省
- 對於某些極端記憶體受限的場景，這可能是唯一可行的選項

---

## 7. 配置推薦策略

計算出 Pareto 前沿後，我們需要從中選擇一個配置推薦給用戶。

### 7.1 推薦策略類型

**位置：** `tmp/config/optimization_config.yaml`

```yaml
output:
  recommendation:
    strategy: "closest_to_target"  # 推薦策略

multiobjective:
  targets:  # 用於 closest_to_target 策略
    accuracy_min: -0.10   # 準確率損失 ≤ 10%
    gpu_peak_max: -0.50   # GPU 節省 ≥ 50%
    latency_max: 1.50     # 延遲增加 ≤ 150%

  objectives:  # 用於 balanced 策略的權重
    - name: "accuracy_change"
      weight: 2.0         # 準確率權重（最重要）
    - name: "gpu_peak_change"
      weight: 1.0         # GPU 權重
    - name: "latency_change"
      weight: 1.0         # 延遲權重
```

### 7.2 策略 1：`closest_to_target` ⭐ 推薦

**目標：** 找到最接近用戶指定目標的配置

**計算方式：**

```python
# 1. 篩選滿足目標的試驗
satisfying_trials = [t for t in pareto_trials if t['satisfies_targets']]

if satisfying_trials:
    # 2. 計算到目標點的加權距離
    weights = [2.0, 1.0, 1.0]  # [accuracy, gpu, latency]

    def distance_to_target(trial):
        return (
            weights[0] * abs(-trial['accuracy_change'] - (-targets['accuracy_min'])) +
            weights[1] * abs(trial['gpu_peak_change'] - targets['gpu_peak_max']) +
            weights[2] * abs(trial['latency_change'] - targets['latency_max'])
        )

    # 3. 選擇距離最小的
    recommended = min(satisfying_trials, key=distance_to_target)
else:
    # 沒有滿足目標的配置，降級到 balanced 策略
    recommended = balanced_strategy(pareto_trials)
```

**適用場景：**
- 用戶對目標有明確要求（如必須達到 50% GPU 節省）
- 需要在滿足約束的前提下找到最優解

### 7.3 策略 2：`balanced`

**目標：** 使用目標權重平衡所有目標

**計算方式：**

```python
weights = [2.0, 1.0, 1.0]  # [accuracy, gpu, latency]

# 計算加權總分（全部轉為 minimize）
def weighted_score(trial):
    return (
        weights[0] * (-trial['accuracy_change']) +  # maximize accuracy → minimize -accuracy
        weights[1] * trial['gpu_peak_change'] +     # minimize gpu
        weights[2] * trial['latency_change']        # minimize latency
    )

# 選擇總分最小的（即綜合最優）
recommended = min(pareto_trials, key=weighted_score)
```

**適用場景：**
- 沒有特定目標約束
- 需要在所有目標之間取得平衡
- 權重反映用戶對不同目標的重視程度

### 7.4 策略 3：`best_accuracy`

**目標：** 優先保證準確率

```python
recommended = max(pareto_trials, key=lambda t: t['objectives']['accuracy_change'])
```

**選擇：** 準確率損失最小（或提升最大）的配置

### 7.5 策略 4：`best_compression`

**目標：** 最大化模型壓縮

```python
recommended = min(pareto_trials, key=lambda t: t['objectives']['gpu_peak_change'])
```

**選擇：** GPU 記憶體節省最多的配置

### 7.6 策略選擇指南

| 場景 | 推薦策略 | 原因 |
|------|----------|------|
| 生產環境，有性能要求 | `closest_to_target` ⭐ | 確保滿足最低性能標準 |
| 探索性研究 | `balanced` | 平衡所有目標，避免極端解 |
| 準確率為王 | `best_accuracy` | 優先保證模型質量 |
| 記憶體受限環境 | `best_compression` | 最大化資源節省 |

---

## 8. 總結與最佳實踐

### 8.1 關鍵要點

1. **Pareto 前沿的本質：**
   - 不存在單一「最優解」
   - Pareto 前沿提供一組 trade-off 選項
   - 選擇哪個取決於用戶需求和權重

2. **支配關係判斷：**
   - 在所有目標上都不比對方差 + 在至少一個目標上嚴格更優
   - 方向很重要：maximize vs minimize
   - 浮點數比較需要容忍度（1e-9）

3. **優化演算法選擇：**
   - 小規模、快速原型 → **TPESampler**
   - 大規模、生產環境 → **NSGAIISampler**
   - 連續參數主導 → **TPESampler**
   - 離散參數、多目標（≥3） → **NSGAIISampler**

### 8.2 常見問題

**Q1: 為什麼我的試驗成功了但不在 Pareto 前沿？**
- A: 被其他更優的配置支配了。這是正常的，說明優化過程找到了更好的解。

**Q2: Pareto 前沿上的解數量多少合適？**
- A: 沒有固定標準，通常 5-20 個。太少（< 3）說明搜索空間不夠多樣；太多（> 50）說明目標之間沒有足夠的衝突。

**Q3: 如何選擇採樣器參數？**
- A:
  - `n_trials`: 至少 10 × 參數數量
  - `n_startup_trials`: ≥ 參數數量（TPE）
  - `population_size`: ≥ 5 × 目標數量（NSGA-II）

**Q4: 可以動態調整目標權重嗎？**
- A: 可以。在 `optimization_config.yaml` 中修改 `objectives[].weight`，然後重新運行推薦邏輯（不需要重新優化）。

### 8.3 參考資料

**論文：**
1. Deb et al. (2002). "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
2. Bergstra et al. (2011). "Algorithms for Hyper-Parameter Optimization" (TPE)
3. Zitzler et al. (2003). "Performance Assessment of Multiobjective Optimizers"

**文檔：**
- Optuna 官方文檔: https://optuna.readthedocs.io/
- Multi-objective Optimization Tutorial: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_multi_objective.html

**我們的實作：**
- 核心邏輯：`tmp/agent/optimizers/base_optimizer.py` (get_pareto_frontier)
- Optuna 整合：`tmp/agent/optimizers/optuna_mo_optimizer.py`
- 配置範例：`tmp/config/optimization_config.yaml`

---

## 附錄：完整工作流程圖

```
1. 配置階段
   ├─ 定義目標（accuracy, gpu, latency）
   ├─ 設置目標約束（targets）
   ├─ 選擇採樣器（TPE / NSGA-II）
   └─ 定義搜索空間（methods, bits, group_size, ...）

2. 優化階段（Optuna）
   ├─ 初始化 Study
   ├─ For each trial (N trials):
   │   ├─ 採樣器建議新配置
   │   ├─ 量化模型
   │   ├─ 評估 benchmarks
   │   ├─ 計算目標值（相對於 baseline）
   │   └─ 保存結果到 trial
   └─ 完成優化

3. Pareto 分析階段
   ├─ 收集所有成功的試驗
   ├─ For each trial:
   │   ├─ 與其他所有試驗比較
   │   ├─ 檢查是否被支配
   │   └─ 如果未被支配 → 加入 Pareto 前沿
   └─ 輸出 Pareto 前沿

4. 推薦階段
   ├─ 根據策略（closest_to_target / balanced / ...）
   ├─ 從 Pareto 前沿中選擇最佳配置
   └─ 輸出推薦結果

5. 輸出
   ├─ all_trials.json（所有試驗）
   ├─ pareto_frontier.json（Pareto 前沿）
   ├─ recommendation.json（推薦配置）
   └─ 視覺化圖表（3D Pareto, parallel coordinates, ...）
```

---

**文檔版本：** v1.0
**最後更新：** 2026-01-08
**維護者：** Green_AI 項目團隊
