# GPTQModel 量化方法參數分析文檔

## 概覽

本文檔分析 GPTQModel 支援的量化方法（GPTQ / AWQ / QQQ / GPTAQ）各參數對三個核心指標的影響：
- **Accuracy**：評估基準上的正確率
- **GPU Peak Memory**：推理時的 GPU 峰值記憶體使用量（MB）
- **Latency**：推理延遲（tokens/sec）

**配套配置文件**：
- `gptq_optimization_space.yaml`：GPTQ / GPTQ v2 搜索空間
- `awq_optimization_space.yaml`：AWQ 搜索空間
- `qqq_optimization_space.yaml`：QQQ 搜索空間
- `gptaq_optimization_space.yaml`：GPTAQ 搜索空間

---

## 方法選擇快速參考

| 方法 | 核心原理 | 主要優勢 | 限制 |
|------|---------|---------|------|
| **GPTQ** | Hessian-based 逐層權重量化 | 最廣泛相容，2/3/4/8-bit | 推理速度一般 |
| **GPTQ v2** | GPTQ 改進格式（有號零點） | 精度略優於 v1 | 舊版 vllm 不支援 |
| **AWQ** | 激活感知縮放，保護關鍵 1% 權重 | 精度通常略優於 GPTQ，量化速度快 | 主要 4-bit，相容性要求多 |
| **QQQ** | W4A8（權重 4-bit + 激活值 8-bit） | 推理利用 INT8 GEMM 加速 | damp 敏感，format 固定 |
| **GPTAQ** | GPTQ + 激活值誤差感知（alpha 控制） | 比 GPTQ 精度更高（+0.5-2%） | 量化時間略長 |

---

## GPTQ 參數影響總覽表

| 參數 | Accuracy | GPU Memory | Latency | 建議策略 | 搜索空間 |
|------|:--------:|:----------:|:-------:|---------|---------|
| `bits` | ⬆⬆⬆ | ⬆⬆⬆ | ⬆⬆⬆ | 必須調整 | `[2, 3, 4, 8]` |
| `group_size` | ⬆⬆ | ⬆⬆ | ⬆ | 強烈建議調整 | `[-1, 16, 32, 64, 128, 256]` |
| `format` | ✗ | ⬆ | ⬆⬆ | 建議調整 | `["gptq", "gptq_v2"]` |
| `desc_act` | ⬆⬆ | ✗ | ✗ | 低位數時重要 | `[true, false]` |
| `damp_percent` | ⬆ | ✗ | ✗ | 選擇性調整 | `[0.005, 0.01, 0.05, 0.1]` |
| `rotation` | ⬆ | ✗ | ✗ | **固定** `null`（hadamard 實測失敗） | — |
| `mse` | ⬆ | ✗ | ✗ | 選擇性調整 | `[0.0, 0.01, 0.05, 0.1]` |
| `sym` | ✗ | ✗ | ✗ | **固定** `true` | — |
| `true_sequential` | ✗ | ✗ | ✗ | **固定** `true` | — |
| `lm_head` | ✗ | ✗ | ✗ | **固定** `false` | — |
| `pack_dtype` | ✗ | ✗ | ✗ | **固定** `"int32"` | — |
| `offload_to_disk` | ✗ | ✗ | ✗ | **固定** `true`（量化過程） | — |

> ⬆⬆⬆ 高影響 | ⬆⬆ 中影響 | ⬆ 低影響 | ✗ 無顯著影響

---

## 各目標的關鍵參數分析

### 1. Accuracy（精度保留）

**最關鍵參數（按重要性排序）：**

#### `bits`（最重要）
量化精度是影響 accuracy 最大的單一因素：

| bits | 典型 Accuracy 損失 | 說明 |
|------|-----------------|------|
| 8 | < 1% | 幾乎無損，與 FP16 相近 |
| 4 | 1-3% | 輕微損失，多數任務可接受 |
| 3 | 3-8% | 中等損失，需謹慎評估 |
| 2 | > 10% | 嚴重損失，僅限特定用途 |

#### `group_size`（第二重要）
決定量化粒度（縮放因子的細緻程度）：
- `16-32`：最細粒度，精度最高，但模型檔案較大
- `128`：業界標準，精度與檔案大小的最佳平衡
- `-1`（per-channel）：粒度最粗，精度最低，但推理最快

#### `desc_act`（低位數時關鍵）
對 2-3 bit 量化有顯著影響：
- `true`：激活值按降序排列再量化，重要特徵的誤差更小
- `false`：標準順序，4-8 bit 時差異通常 < 0.5%
- **建議**：2-3 bit 使用 `true`；4-8 bit 保持 `false` 即可

#### `damp_percent`（穩定性影響精度）
- 過低（< 0.005）：Hessian 矩陣奇異，可能導致 NaN 或收斂失敗
- 推薦範圍：0.01 - 0.05
- 預設 0.05 是保守安全值；若想更激進可試 0.01

#### `rotation`
- 理論上 `"hadamard"` 可改善 outlier 分佈，提升 0.5-2%
- **實測結果：rotation="hadamard" 量化失敗，固定為 null 不使用**

#### `mse`
- `0.0`：標準 GPTQ，最小化二階誤差
- `> 0.0`：在目標函數中加入 MSE 項，通常 0.01-0.05 有微幅提升
- 效果因模型和任務而異，量化時間略增

---

### 2. GPU Peak Memory（推理記憶體使用）

**關鍵公式：**
```
GPU Peak Memory ≈ 模型參數（量化後）+ KV Cache
量化模型大小 ≈ FP16 大小 × (bits / 16)
```

**關鍵參數：**

#### `bits`（最重要）
| bits | 相對 FP16 大小 | 典型 GPU 節省 |
|------|-------------|-------------|
| 8 | ~50% | ~40% |
| 4 | ~25% | ~70% |
| 3 | ~19% | ~80% |
| 2 | ~13% | ~87% |

#### `group_size`（次要影響）
較小的 group_size 導致更多 scales/zeros 張量：
- `group_size=16`：比 `group_size=128` 多 8x 的 scales 記憶體
- 對大模型而言影響相對較小（scales 通常佔 <5% 總記憶體）

#### `format`（間接影響）
- `marlin`/`bitblas`：優化的 kernel 可能略微增加工作記憶體，但整體 throughput 更高
- `gptq`/`gptq_v2`：標準記憶體使用

**注意**：不同資料集的 GPU Peak 會有差異（這是正常的，不是 bug）：
- BBH > GSM8K > CommonsenseQA > TruthfulQA（因 few-shot 提示長度不同）
- 較長的提示 + 較長的生成 = 較高的 KV Cache 需求

---

### 3. Latency（推理速度）

**最關鍵參數：**

#### `format`（最重要）
不同格式對推理速度的影響巨大：

> 實測有效格式（RTX 4090）：只用 `gptq` 和 `gptq_v2`。

| format | 速度 vs gptq | 狀態 | 備註 |
|--------|------------|------|------|
| `gptq` (v1) | 1x（基準） | ✅ 穩定 | 最廣泛相容（vllm/transformers） |
| `gptq_v2` | ~1x | ✅ 穩定 | 精度略優於 v1，vllm ≥ 0.4.0 支援 |
| `marlin` | 2-4x 更快 | ❌ 量化失敗 | 實測失敗，暫不使用 |
| `bitblas` | 2-4x 更快 | ⚠️ 套件問題 | 需 torch 先載入，風險高 |
| `gemm` / `gemv` / `gemv_fast` / `llm_awq` | — | ❌ Config ERR | AWQ 專用，GPTQ 不支援 |

#### `bits`（次要影響）
- 較低 bits 的矩陣乘法：4-bit 比 8-bit 快約 1.3-1.7x（取決於 kernel 優化）
- 影響 memory bandwidth，低 bits 通常有更高的 throughput

#### `group_size`（輕微影響）
- 小 group_size：dequantization 時需要更多縮放操作，略慢
- `-1`（per-channel）：最快，因縮放因子最少

---

## gptq vs gptq_v2 詳細比較

| 面向 | `gptq` (v1) | `gptq_v2` |
|------|------------|----------|
| **標準化** | 原始論文格式 | GPTQModel 改進格式 |
| **零點表示** | 無號整數 | 有號整數，精度更準確 |
| **典型精度差異** | 基準 | 通常 < 0.5%（低位數時更明顯）|
| **vllm 相容性** | 所有版本 ✓ | vllm >= 0.4.0 ✓ |
| **transformers 相容性** | ✓ | ✓ |
| **config_loader.py 支援** | ✓ | ✓ (FORMAT_MAP 已有映射) |
| **建議場景** | 最廣泛部署 | 本地實驗、最新 vllm |

**實驗建議**：將 `"gptq"` 和 `"gptq_v2"` 都納入搜索空間，直接比較兩者在相同 bits/group_size 下的精度差異。

---

## 推薦的優化實驗序列

### 階段一：快速篩選（3-5 次實驗）
先確定 bits 和 format 的基本方向：

```yaml
# 實驗 1: 基準（4-bit, gptq, group_size=128）
bits: 4, group_size: 128, format: "gptq"

# 實驗 2: 更高壓縮率
bits: 3, group_size: 128, format: "gptq"

# 實驗 3: 測試 gptq_v2 格式
bits: 4, group_size: 128, format: "gptq_v2"

# 實驗 4: 測試 marlin（如果 GPU 支援）
bits: 4, group_size: 128, format: "marlin"

# 實驗 5: 更細粒度
bits: 4, group_size: 64, format: "gptq"
```

### 階段二：精細調整（針對精度）
基於階段一選定最佳 bits/format，調整其他參數：

```yaml
# 測試 desc_act（若 bits <= 3）
desc_act: true, bits: 3, group_size: 128

# 測試 rotation
rotation: "hadamard", bits: 4, group_size: 128

# 測試 mse
mse: 0.05, bits: 4, group_size: 128

# 測試 damp_percent
damp_percent: 0.01, bits: 4, group_size: 128
```

### 階段三：最優配置驗證
在所有評估資料集上（gsm8k, truthfulqa, commonsenseqa）驗證最終配置。

---

## 與已量化模型的參數對照

以下是實際量化後 `config.json` 中的參數（`Llama-3.2-1B-Instruct-gptq-4bit-20260302_024452`）：

```json
{
  "quantization_config": {
    "bits": 4,
    "group_size": 128,
    "sym": true,
    "desc_act": false,
    "format": "gptq",
    "lm_head": false,
    "pack_dtype": "int32",
    "meta": {
      "damp_percent": 0.05,
      "damp_auto_increment": 0.01,
      "mse": 0.0,
      "true_sequential": true,
      "static_groups": false,
      "offload_to_disk": true,
      "pack_impl": "cpu",
      "gc_mode": "interval",
      "wait_for_submodule_finalizers": false
    }
  }
}
```

這確認了以下參數由 GPTQModel 自動設置，無需手動指定：
- `damp_auto_increment: 0.01`（量化失敗時自動遞增阻尼）
- `pack_impl: "cpu"`（CPU 打包，最穩定）
- `gc_mode: "interval"`（間隔式 GC）
- `static_groups: false`（動態 groups）
- `wait_for_submodule_finalizers: false`

---

---

## AWQ 參數分析

### AWQ 核心差異

AWQ 使用**激活感知縮放**而非 Hessian 矩陣：找出對激活值影響最大的 1% 權重並給予縮放保護，剩餘 99% 按標準量化。這使得 AWQ 量化速度更快（無需逐層 Hessian 計算）。

### AWQ 可調參數

| 參數 | Accuracy | GPU Memory | Latency | 建議策略 | 搜索空間 |
|------|:--------:|:----------:|:-------:|---------|---------|
| `bits` | — | — | — | **固定** `4`（kernel 限制，8-bit 會報錯） | — |
| `group_size` | ⬆⬆ | ⬆ | ⬆ | 建議調整 | `[32, 64, 128]` |
| `format` | ✗ | ⬆ | ⬆⬆ | **固定** `gemm` | — |
| `sym` | ✗ | ✗ | ✗ | **固定** `false` | — |

### AWQ format 選擇（已實測）

| format | autoawq 相容 | 推理速度 | 適用場景 |
|--------|:-----------:|---------|---------|
| `"llm_awq"` | ✓ | 基準 | 需跨工具部署（autoawq 標準格式） |
| `"gemv_fast"` | ✗ | 2x+ | GPTQModel 純推理，batch=1 最快 |
| `"gemv"` | ✗ | 1.5-2x | GPTQModel，batch=1 優化 |
| `"gemm"` | ✗ | 1-2x | GPTQModel，batch > 1 優化 |
| `"marlin"` | ✗ | 2-4x | RTX 30xx/40xx (SM≥8.0)，group_size 需 128 或 -1 |
| `"gptq"` / `"gptq_v2"` | — | — | ⚠️ **會被自動 fix 成 gemm**（不報錯但設定失效） |
| `"bitblas"` | — | — | ❌ **AWQ 不支援**（Config ERR） |

### AWQ 不適用的參數

AWQ 有自己的激活感知機制，以下 GPTQ 專用參數**設置無效**：
- `desc_act`、`mse`、`rotation`、`damp_percent`、`gptaq`

### AWQ dynamic 支援

**❌ 不支援（已實測）**

`AwqTorchQuantLinear` kernel 固定只接受 4-bit，任何 `dynamic` 覆寫（包含排除層的 `-:` 前綴）都會拋出：
```
ValueError: AwqTorchQuantLinear not supported dynamic_bits, only support [4] bits
```

---

## QQQ 參數分析

### QQQ 核心特性

QQQ 實現 **W4A8 量化**（4-bit 權重 + 8-bit 激活值），在推理時利用 INT8 GEMM 硬體指令加速，通常比純 W4 GPTQ 推理更快。

### QQQ ⚠️ 已知注意事項

- `format` 固定為 `"qqq"`，無法更換
- `bits` 固定為 `4`（W4A8 設計）
- 建議 `num_samples >= 512`（需更準確的激活值統計）
- **`dynamic` ❌ 不支援（已實測）**：`QQQQuantLinear` kernel 固定只接受 4-bit，任何 dynamic 覆寫皆拋出 `ValueError: not supported dynamic_bits`

### QQQ damp_percent 設計原理

QQQ 的 Hessian 矩陣在累積時做了**動態加權歸一化**（qqq.py）：

```python
self.H *= self.nsamples / (self.nsamples + tmp)
inp = math.sqrt(2 / self.nsamples) * inp   # 歸一化
self.H += inp.matmul(inp.t())
```

這使得 QQQ 的 Hessian 對角線規模恆定在 O(1)（約 2.0），而 GPTQ 直接累積所以規模是 O(n_samples)（約 32,000+）。

因此兩者**實際施加的阻尼量是相近的**：

| 方法 | damp_percent | mean(diag(H)) | 實際阻尼 |
|------|:-----------:|:------------:|:-------:|
| GPTQ | 0.05 | ~32,764 | ~1,638 |
| QQQ  | 0.005 | ~2.0 | ~0.01 |

套件 `damp_percent=0.005` / `damp_auto_increment=0.001` 是對 QQQ 算法刻意設計的預設值，和 GPTQ 的 0.05 / 0.01 各自在各自的 Hessian 尺度上是等效的。如果 QQQ 出現收斂失敗，建議先確認 `num_samples` 是否足夠，而非直接調高 damp_percent。

### QQQ 可調參數

| 參數 | Accuracy | GPU Memory | Latency | 建議策略 | 搜索空間 |
|------|:--------:|:----------:|:-------:|---------|---------|
| `group_size` | ⬆⬆ | ⬆ | ⬆ | 建議調整 | `[64, 128]` |
| `damp_percent` | ⬆⬆（穩定性） | ✗ | ✗ | 套件預設 0.005，在此尺度調整 | `[0.001, 0.005, 0.01]` |
| `desc_act` | ⬆⬆ | ✗ | ✗ | 建議調整（W4A8 較重要） | `[true, false]` |
| `num_samples`（校準） | ⬆ | ✗ | ✗ | 建議增加 | `[256, 512, 1024]` |
| `bits` | — | — | — | **固定** `4` | — |
| `format` | — | — | — | **固定** `"qqq"` | — |

---

## GPTAQ 參數分析

### GPTAQ 核心特性

GPTAQ 是 GPTQ 的擴展，在量化目標函數中同時考慮權重誤差和激活值誤差：

```
量化目標 = (1 - alpha) × 權重誤差 + alpha × 激活值誤差
```

`alpha` 是 GPTAQ 最關鍵的參數，控制兩種誤差的平衡。

### GPTAQ 可調參數

| 參數 | Accuracy | GPU Memory | Latency | 建議策略 | 搜索空間 |
|------|:--------:|:----------:|:-------:|---------|---------|
| `gptaq.alpha` | ⬆⬆⬆ | ✗ | ✗ | **GPTAQ 核心參數** | `[0.1, 0.25, 0.5, 0.75]` |
| `bits` | ⬆⬆⬆ | ⬆⬆⬆ | ⬆⬆⬆ | 必須調整（3-bit 優勢明顯） | `[3, 4]` |
| `group_size` | ⬆⬆ | ⬆⬆ | ⬆ | 建議調整 | `[64, 128]` |
| `format` | ✗ | ⬆ | ⬆⬆ | 同 GPTQ | `["gptq", "gptq_v2"]` |
| `desc_act` | ⬆⬆ | ✗ | ✗ | 低 bits 時重要 | `[true, false]` |
| `rotation` | ⬆ | ✗ | ✗ | **固定** `null`（hadamard 實測失敗） | — |
| `gptaq.device` | — | — | — | **固定** `"auto"` | — |

### GPTAQ alpha 選擇指南

| alpha | 含義 | 建議場景 |
|-------|------|---------|
| `0.0` | 退化為標準 GPTQ | 對照基準實驗 |
| `0.1` | 輕微激活感知 | 保守策略 |
| `0.25` | 論文推薦預設 | **起點，適合多數模型** |
| `0.5` | 均等平衡 | 若 0.25 效果不明顯 |
| `0.75` | 激活感知主導 | 模型有大量 outlier 激活值 |
| `1.0` | 純激活值優化 | 通常效果不佳，不建議 |

**GPTAQ 在低 bits (3-bit) 時比 GPTQ 優勢更明顯**，因為激活感知修正能更好補償低精度誤差。

---

## 附：Format × Method 相容性矩陣（已實測，RTX 4090 / CUDA 12.8）

| format | GPTQ | AWQ | QQQ | 備註 |
|--------|:----:|:---:|:---:|------|
| `gptq` | ✓ | ⚠️→gemm | ✗ | AWQ 使用時被自動 fix 為 gemm |
| `gptq_v2` | ✓ | ⚠️→gemm | ✗ | AWQ 使用時被自動 fix 為 gemm |
| `marlin` | ✓ | ✓ | ✗ | 需 SM≥8.0，group_size=128 或 -1 |
| `bitblas` | ✓ | ✗ | ✗ | 需透過 gptqmodel 使用（torch 需先載入）|
| `gemm` | ✗ | ✓ | ✗ | GPTQ 使用時 Config ERR |
| `gemv` | ✗ | ✓ | ✗ | GPTQ 使用時 Config ERR |
| `gemv_fast` | ✗ | ✓ | ✗ | GPTQ 使用時 Config ERR |
| `llm_awq` | ✗ | ✓ | ✗ | GPTQ 使用時 Config ERR；autoawq 相容格式 |
| `qqq` | ✗ | ✗ | ✓ | QQQ 唯一可用格式 |

> ✓ 可用 | ✗ Config ERR | ⚠️→xxx 被自動修正為 xxx

### config_loader.py FORMAT_MAP

```python
FORMAT_MAP = {
    "gptq":      FORMAT.GPTQ,      # v1，最廣泛相容（GPTQ/GPTAQ 用）
    "gptq_v2":   FORMAT.GPTQ_V2,   # v2，精度略優（GPTQ/GPTAQ 用）
    "marlin":    FORMAT.MARLIN,    # 快 2-4x，需 SM>=8.0（GPTQ/AWQ/GPTAQ 用）
    "bitblas":   FORMAT.BITBLAS,   # 快 2-4x，需透過 gptqmodel 使用（GPTQ/GPTAQ 用）
    "qqq":       FORMAT.QQQ,       # QQQ 方法專用
    "gemm":      FORMAT.GEMM,      # batch 推理優化（AWQ 專用）
    "gemv":      FORMAT.GEMV,      # batch=1 優化（AWQ 專用）
    "gemv_fast": FORMAT.GEMV_FAST, # GEMV 加速版（AWQ 專用）
    "llm_awq":   FORMAT.LLM_AWQ,   # autoawq 標準格式（AWQ 專用）
}
```

YAML 的 `quantization.format` 使用字串形式設定；方法不相容的格式會在 `QuantizeConfig` 建立時拋出錯誤（bitblas for AWQ 除外，是 Config ERR）。
