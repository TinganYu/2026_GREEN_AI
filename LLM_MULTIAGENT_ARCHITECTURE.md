# LLM Multi-Agent 量化優化系統架構文檔

> **版本**: Phase 1.1
> **更新日期**: 2026-01-27
> **用途**: 使用 LLM 驅動的多 Agent 系統自動優化量化超參數

---

## 目錄

1. [系統概述](#系統概述)
2. [整體架構](#整體架構)
3. [Agent 模式](#agent-模式)
4. [AnalyzerAgent 分析代理](#analyzeragent-分析代理)
5. [PlannerAgent 規劃代理](#planneragent-規劃代理)
6. [StrategistAgent 策略師代理](#strategistagent-策略師代理)
7. [MonitorAgent 監控代理](#monitoragent-監控代理)
8. [Prompt 配置系統](#prompt-配置系統)
9. [Agent 對話協議](#agent-對話協議)
10. [對話記錄系統](#對話記錄系統)
11. [配置說明](#配置說明)
12. [使用方式](#使用方式)

---

## 系統概述

### 核心理念

傳統的超參數優化（如 Optuna）使用統計算法（TPE、NSGA-II）來探索搜索空間。本系統使用 **LLM 驅動的多 Agent 系統**，讓 AI 扮演"優化專家"的角色，基於歷史試驗數據做出智能決策。

### 核心優勢

1. **語義理解**: LLM 能理解量化參數的含義（如 `bits`、`group_size`）以及它們對性能的影響
2. **自適應策略**: 根據優化階段（早期/中期/後期）自動調整探索/利用比例
3. **可解釋性**: 每個決策都有自然語言的 `rationale`（理由），便於理解和調試
4. **容錯機制**: LLM 失敗時可降級到規則引擎（Rule-based Fallback）
5. **多語言 Prompt**: 支援 English / 繁體中文等多種 Prompt 模板，透過 YAML 配置檔管理

### Agent 總覽

| Agent | 職責 | 使用 LLM | 模式 |
|-------|------|----------|------|
| **AnalyzerAgent** | 分析歷史試驗，識別失敗模式、未探索區域、參數敏感度 | 是 | Separate |
| **PlannerAgent** | 根據分析報告決定下一個試驗的配置 | 是 | Separate |
| **StrategistAgent** | 合併分析與規劃，一次 LLM 調用完成 | 是 | Combined |
| **MonitorAgent** | 監控優化進度，判斷停止條件 | **否（純規則）** | 兩種模式共用 |

---

## 整體架構

### 系統組件圖

```
┌─────────────────────────────────────────────────────────────┐
│                  LLMMultiAgentOptimizer                     │
│                     (主控制器)                               │
│                                                             │
│  agent_mode: "separate" | "combined"                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────────────┐
        │              │                      │
   (separate)     (combined)             (共用)
        │              │                      │
        ▼              ▼                      ▼
┌──────────────┐ ┌──────────────┐   ┌──────────────┐
│ AnalyzerAgent│ │ Strategist   │   │ MonitorAgent │
│   (分析器)    │ │   Agent      │   │  (監控器)     │
│    [LLM]     │ │ (策略師)      │   │ [純規則]      │
└──────┬───────┘ │  [LLM]       │   └──────────────┘
       │         │ 合併分析+決策  │
       ▼         └──────────────┘
┌──────────────┐
│ PlannerAgent │
│   (規劃器)    │
│    [LLM]     │
└──────────────┘

所有 LLM Agent 共用:
┌──────────────┐   ┌────────────────────┐
│  LLMClient   │   │ PromptConfigLoader │
│ (OpenAI API) │   │ (prompt_config.yaml│
└──────────────┘   └────────────────────┘
```

### 工作流程 (每個 Trial)

```
Trial N 開始
    │
    ▼
┌──────────────────────────────────────────────┐
│ [1] 分析與規劃                                │
│                                              │
│  Separate 模式:                               │
│    AnalyzerAgent 分析 → PlannerAgent 決策     │
│    (2 次 LLM 調用)                            │
│                                              │
│  Combined 模式:                               │
│    StrategistAgent 一次完成分析 + 決策          │
│    (1 次 LLM 調用)                            │
└──────────────┬───────────────────────────────┘
               ▼
┌──────────────────────────────────────────────┐
│ [2] 配置驗證 (簡單規則檢查)                    │
│   檢查: GPTQ 參數兼容性、搜索空間合法性         │
└──────────────┬───────────────────────────────┘
               ▼
┌──────────────────────────────────────────────┐
│ [3] 執行試驗 (量化 + 評估)                     │
│   - 量化模型                                  │
│   - 在數據集上評估                            │
│   - 計算目標變化 (accuracy, GPU, latency)     │
└──────────────┬───────────────────────────────┘
               ▼
┌──────────────────────────────────────────────┐
│ [4] MonitorAgent 檢查停止條件（純規則）         │
│   - 預算耗盡？                                │
│   - Pareto 前沿收斂？                         │
│   - 目標達成？                                │
└──────────────┬───────────────────────────────┘
               ▼
       是否停止? ──No──► 下一個 Trial
               │
              Yes
               ▼
       返回最佳配置
```

---

## Agent 模式

系統支援兩種 Agent 運作模式，透過 `agent_mode` 配置切換：

### Separate 模式（`agent_mode: "separate"`）

```
AnalyzerAgent (分析) → PlannerAgent (決策) → MonitorAgent (監控)
```

- **2 次 LLM 調用**：分析和決策分開進行
- **優點**：分析更詳細，各 Agent 職責清晰
- **缺點**：API 成本較高、延遲較長
- **適用場景**：需要深入分析的複雜優化

### Combined 模式（`agent_mode: "combined"`）

```
StrategistAgent (分析 + 決策) → MonitorAgent (監控)
```

- **1 次 LLM 調用**：分析和決策合併完成
- **優點**：省錢省時間，回應更一致
- **缺點**：分析可能不如分開模式詳細
- **適用場景**：日常優化，推薦使用

### 配置方式

```yaml
optimizer:
  llm_agent:
    agent_mode: "combined"  # 或 "separate"
    agent_temperatures:
      analyzer: 0.3    # separate 模式用
      planner: 0.7     # separate 模式用
      strategist: 0.5  # combined 模式用
```

---

## AnalyzerAgent 分析代理

> **模式**: Separate 模式專用

### 職責

分析歷史試驗數據，識別：
1. **方法表現** (Method Analysis): 各量化方法的成功率和平均指標
2. **失敗模式** (Failure Patterns): 哪些配置容易失敗？
3. **成功模式** (Success Patterns): 哪些配置效果好？
4. **參數洞察** (Parameter Insights): 哪些參數對性能影響最大？
5. **未探索區域** (Unexplored Regions): 哪些參數組合值得嘗試？

### 輸入/輸出

**輸入** (`input_data`):
```python
{
    'trials': List[Dict],           # 所有歷史試驗（包含成功和失敗）
    'pareto_frontier': List[Dict],  # 當前 Pareto 前沿解
    'search_space': Dict            # 搜索空間定義
}
```

**輸出** (`analysis`):
```python
{
    "method_analysis": {
        "gptq": {"trials": N, "success_rate": X, "avg_accuracy": Y, "avg_gpu": Z, "observations": "..."},
        "awq": {...},
        "bnb": {...}
    },
    "failure_patterns": [
        {"pattern": "description", "affected_params": {...}, "suggestion": "avoid/adjust"}
    ],
    "success_patterns": [
        {"pattern": "description", "winning_params": {...}, "trade_off": "..."}
    ],
    "parameter_insights": {
        "bits": "observation",
        "group_size": "observation",
        "other": "observation"
    },
    "unexplored_regions": [
        {"method": "...", "params": {...}, "rationale": "why promising"}
    ],
    "recommendations": ["advice 1", "advice 2"]
}
```

### Fallback 機制

當 LLM 失敗時，使用簡單的統計規則：
- 統計每個方法的失敗次數識別失敗模式
- 檢查哪些方法嘗試次數 < 3 作為未探索區域
- 使用固定的參數敏感度估計值

**代碼位置**: `tmp/agent/optimizers/agents/analyzer_agent.py:_fallback_analysis()`

---

## PlannerAgent 規劃代理

> **模式**: Separate 模式專用

### 職責

根據 AnalyzerAgent 的分析報告，決定：
1. **優化策略** (Strategy): 探索 (exploration) / 利用 (exploitation) / 平衡 (balanced)
2. **下一個配置** (Next Config): 具體的量化參數設置
3. **理由** (Rationale): 為什麼選擇這個配置（可解釋性）

### 輸入/輸出

**輸入** (`input_data`):
```python
{
    'analysis': Dict,       # AnalyzerAgent 的分析報告
    'trial_num': int,       # 當前試驗編號
    'budget': {             # 預算狀態
        'used': 2,
        'max': 10
    },
    'targets': Dict,        # 優化目標
    'pareto': List[Dict],   # 當前 Pareto 前沿
    'trials': List[Dict]    # 已嘗試的配置（用於避免重複）
}
```

**輸出** (`decision`):
```python
{
    "thinking": {
        "analyzer_insights": "...",
        "target_gap": "...",
        "avoid_repeating": "...",
        "chosen_strategy_reason": "..."
    },
    "strategy": "balanced",
    "next_config": {
        "method": "gptq",
        "bits": 4,
        "group_size": 32,
        ...
    },
    "rationale": "解釋（2-3 句話）",
    "confidence": 0.75
}
```

**重要**: `next_config` 中的參數必須是**扁平結構**，直接放在物件中，不可嵌套在 `"parameters"` 中。

### Fallback 機制

當 LLM 失敗時，使用基於規則的策略選擇：

```python
if progress < 0.2:
    strategy = "exploration"   # 早期: 探索
elif progress < 0.7:
    strategy = "balanced"      # 中期: 平衡
else:
    strategy = "exploitation"  # 後期: 利用
```

然後從未探索區域或搜索空間中隨機採樣配置。

**代碼位置**: `tmp/agent/optimizers/agents/planner_agent.py:_fallback_planning()`

---

## StrategistAgent 策略師代理

> **模式**: Combined 模式專用

### 職責

合併 AnalyzerAgent 和 PlannerAgent 的功能，在**一次 LLM 調用**中完成：
1. **分析歷史試驗**（等同 AnalyzerAgent 的工作）
2. **決定下一個配置**（等同 PlannerAgent 的工作）

### 輸入/輸出

**輸入** (`input_data`):
```python
{
    'trials': List[Dict],           # 所有試驗
    'pareto_frontier': List[Dict],  # Pareto前沿
    'trial_num': int,               # 當前試驗編號
    'budget': {'used': int, 'max': int},
    'targets': Dict                 # 優化目標
}
```

**輸出** (合併結果):
```python
{
    "analysis": {
        "method_analysis": {...},
        "failure_patterns": [...],
        "success_patterns": [...],
        "parameter_insights": {...},
        "unexplored_regions": [...],
        "recommendations": [...]
    },
    "decision": {
        "thinking": {
            "analyzer_insights": "...",
            "target_gap": "...",
            "avoid_repeating": "...",
            "chosen_strategy_reason": "..."
        },
        "strategy": "exploration" / "exploitation" / "balanced",
        "next_config": {
            "method": "gptq" / "awq" / "bnb",
            "bits": ...,
            "group_size": ...,
            ...
        },
        "rationale": "...",
        "confidence": 0.0-1.0
    }
}
```

**重要**: `next_config` 中的參數必須是**扁平結構**，不可嵌套在 `"parameters"` 中。

### Fallback 機制

與 Separate 模式的 Fallback 邏輯相同：
- 分析部分：統計各方法表現、識別未探索區域
- 決策部分：基於進度選擇策略，從搜索空間隨機採樣

**代碼位置**: `tmp/agent/optimizers/agents/strategist_agent.py:_fallback_strategist()`

---

## MonitorAgent 監控代理

> **模式**: 兩種模式共用
> **特性**: **純規則版本，不使用 LLM**

### 職責

監控優化進度並判斷是否應該停止優化，檢查：
1. **預算耗盡** (Budget Exhausted): 試驗次數是否用完？（≥95%）
2. **收斂** (Convergence): Pareto 前沿是否停止改善？
3. **目標達成** (Target Met): 是否已找到足夠多滿足目標的解？

### 輸入/輸出

**輸入** (`input_data`):
```python
{
    'trials': List[Dict],           # 所有試驗
    'pareto_frontier': List[Dict],  # Pareto 前沿
    'budget_status': {              # 預算狀態
        'used': 3,
        'max': 10
    },
    'targets': Dict,                # 優化目標
    'pareto_history': [             # Pareto 前沿歷史
        {'trial_count': 1, 'pareto_size': 1},
        {'trial_count': 2, 'pareto_size': 2},
        ...
    ]
}
```

**輸出** (`assessment`):
```python
{
    'should_stop': False,
    'reason': 'continue',    # convergence / target_met / budget_exhausted / continue
    'convergence_score': 0.15,
    'progress_metrics': {
        'pareto_improvement_rate': 0.67,
        'target_satisfaction': 0.50,
        'budget_used': 0.30,
        'trials_since_improvement': 0
    },
    'recommendation': "Continue optimization. Progress is being made."
}
```

### 停止規則

MonitorAgent 使用純規則判斷，不調用 LLM：

1. **預算耗盡**: `budget_used >= 0.95`（95%）
2. **收斂檢測**: 連續 N 輪（`convergence_window`，預設 5）Pareto 大小變化不超過 1
3. **目標滿足**: 有 3 個以上滿足目標的解，且佔 Pareto 前沿 ≥50%（目前僅增加收斂分數，不直接觸發停止）

**代碼位置**: `tmp/agent/optimizers/agents/monitor_agent.py:_rule_based_stopping()`

---

## Prompt 配置系統

### 概述

系統使用外部 YAML 配置檔管理所有 Agent 的 Prompt 模板，支援多語言和自定義 Prompt。

### 配置檔結構

**檔案**: `tmp/config/prompt_config.yaml`

```yaml
default_type: "en"
supported_types:
  - "en"      # English
  - "zh"      # 繁體中文

prompts:
  en:
    metadata:
      name: "English"
      version: "1.0"
    analyzer:
      system_role: "..."
      task_description: "..."
      analysis_requirements: "..."
      output_format: "..."
      section_headers: {...}
      labels: {...}
    planner:
      system_role: "..."
      decision_process: "..."
      strategy_guide: "..."
      output_format: "..."
      section_headers: {...}
      labels: {...}
    strategist:
      system_role: "..."
      task_description: "..."
      analysis_steps: "..."
      decision_steps: "..."
      output_format: "..."
      section_headers: {...}
      labels: {...}
    formatting:
      methods: "Methods"
      trial_format: "Trial {num}"
      ...

  zh:
    # 繁體中文版本，結構相同
    ...
```

### 載入機制

```
optimization_config_llm.yaml
  └── optimizer.llm_agent.prompt:
        config_file: "tmp/config/prompt_config.yaml"
        type: "zh"
            │
            ▼
    PromptConfigLoader (Singleton)
      └── init_prompts(config_file, type)
            │
            ▼
    PromptTemplates.get_analyzer_prompt()
    PromptTemplates.get_planner_prompt()
    PromptTemplates.get_strategist_prompt()
```

**代碼位置**: `tmp/agent/optimizers/utils/prompt_templates.py`

- `PromptConfigLoader`: Singleton，負責載入和快取 YAML 配置
- `PromptTemplates`: 靜態方法集合，根據配置生成各 Agent 的 Prompt
- `init_prompts()`: 便利函數，初始化 Prompt 配置
- `get_prompt_info()`: 取得完整 Prompt 資訊（用於結果輸出）

### 向後相容

若找不到 `prompt_config.yaml`，系統會使用 `PromptTemplates` 中的內建預設 Prompt（英文版），確保向後相容。

---

## Agent 對話協議

### AgentMessage 消息結構

所有 Agent 間的通信使用統一的 `AgentMessage` 格式：

```python
{
    "from_agent": "AnalyzerAgent",      # 發送者
    "to_agent": "Orchestrator",         # 接收者
    "message_type": "analysis_report",  # 消息類型
    "content": {                        # 消息內容（各 Agent 不同）
        "input_summary": {...},         # 輸入摘要
        "analysis": {...}               # 實際結果
    },
    "trial_context": 1,                 # 關聯的 Trial 編號
    "timestamp": "2025-12-30T18:39:10"  # 時間戳
}
```

### 消息類型 (message_type)

| 消息類型 | 發送者 | 模式 | 用途 |
|---------|-------|------|------|
| `analysis_report` | AnalyzerAgent | Separate | 歷史試驗分析報告 |
| `strategy_decision` | PlannerAgent | Separate | 下一個試驗的配置決策 |
| `strategist_result` | StrategistAgent | Combined | 合併的分析 + 決策結果 |
| `trial_start` | Orchestrator | 共用 | Trial 開始事件 |
| `trial_end` | Orchestrator | 共用 | Trial 結束事件 |
| `error` | Any | 共用 | 錯誤事件 |

### 對話流程示例

**Separate 模式:**
```
Trial 1:
  [18:39:06] Orchestrator: trial_start
  [18:39:10] AnalyzerAgent → Orchestrator: analysis_report
  [18:39:15] PlannerAgent → Orchestrator: strategy_decision
  [18:41:16] Orchestrator: trial_end (SUCCESS)
```

**Combined 模式:**
```
Trial 1:
  [18:39:06] Orchestrator: trial_start
  [18:39:12] StrategistAgent → Orchestrator: strategist_result
  [18:41:16] Orchestrator: trial_end (SUCCESS)
```

> **注意**: MonitorAgent 使用純規則，不產生 AgentMessage，其結果直接由 Optimizer 處理。

---

## 對話記錄系統

### 雙格式記錄

系統會生成**兩個**檔案來記錄 Agent 對話：

1. **JSONL 格式** (機器可讀)
   檔案: `results/llm_optimization/*/agent_conversations.jsonl`
   用途: 完整的結構化數據，可用於分析、重放、調試

2. **TXT 格式** (人類可讀)
   檔案: `results/llm_optimization/*/agent_conversations.txt`
   用途: 格式化的對話流程，方便查看和理解
   生成: 優化完成後由 `ConversationFormatter` 自動從 JSONL 轉換

### JSONL 格式示例

每行是一個 JSON 對象：

```jsonl
{"event": "trial_start", "trial_num": 1, "timestamp": "2025-12-30T18:39:06"}
{"from_agent": "StrategistAgent", "to_agent": "Orchestrator", "message_type": "strategist_result", "content": {...}, "trial_context": 1, "timestamp": "2025-12-30T18:39:12"}
{"event": "trial_end", "trial_num": 1, "success": true, "result": {...}, "timestamp": "2025-12-30T18:41:16"}
```

### 查看對話記錄

**查看 TXT 檔案**（推薦）:
```bash
cat results/llm_optimization/*/agent_conversations.txt
```

**解析 JSONL 檔案**:
```python
import json

with open('agent_conversations.jsonl', 'r') as f:
    for line in f:
        msg = json.loads(line)
        print(f"[{msg.get('from_agent', 'SYSTEM')}] {msg.get('message_type', msg.get('event'))}")
```

### ConversationFormatter

**功能**: 自動將 JSONL 轉換為 TXT

**代碼位置**: `tmp/agent/optimizers/utils/conversation_formatter.py`

**使用方式**:
```python
from tmp.agent.optimizers.utils import ConversationFormatter

ConversationFormatter.format_to_txt(
    'agent_conversations.jsonl',
    'agent_conversations.txt'
)
```

優化完成後會自動生成，無需手動調用。

---

## 配置說明

### 主配置文件

**檔案**: `tmp/config/optimization_config_llm.yaml`

### 關鍵配置區塊

#### 1. LLM Agent 配置

```yaml
optimizer:
  type: "llm_multiagent"

  llm_agent:
    # LLM Provider
    provider: "openai"
    model: "gpt-4o"
    api_key_env: "OPENAI_API_KEY"

    # 生成參數
    temperature: 0.7
    max_tokens: 2048

    # Prompt 配置
    prompt:
      config_file: "tmp/config/prompt_config.yaml"
      type: "zh"    # "en" 或 "zh"

    # Agent 模式
    agent_mode: "combined"    # "separate" 或 "combined"

    # Agent 特定溫度
    agent_temperatures:
      analyzer: 0.3    # 低溫（separate 模式）
      planner: 0.7     # 中高溫（separate 模式）
      strategist: 0.5  # 中溫（combined 模式）

    # Trial 限制
    max_trials: 25
    min_trials: 20
```

#### 2. 停止條件

```yaml
  stopping:
    budget:
      max_trials: 5
      max_time_hours: 24
    convergence:
      enabled: false
      window: 3
      threshold: 0.05
```

#### 3. Fallback 機制

```yaml
  fallback:
    on_llm_failure: "stop"    # "random" 或 "stop"
    max_llm_failures: 3
```

**推薦設置**:
- 測試階段: `"stop"` - 發現問題立即停止
- 生產階段: `"random"` - 容錯繼續運行

#### 4. 對話日誌

```yaml
  logging:
    save_conversations: true
    conversation_file: "agent_conversations.jsonl"
    save_prompts: true    # 調試用：保存完整 prompts
```

#### 5. 搜索空間

```yaml
  search_space:
    methods: ["gptq", "awq", "bnb"]

    gptq:
      bits: [2, 3, 4, 8]
      group_size: [-1, 16, 32, 64, 128]
      desc_act: [true, false]
      sym: [true, false]
      damp_percent: [0.001, 0.005, 0.01, 0.03, 0.05]
      ...

    awq:
      w_bit: [4]
      q_group_size: [16, 32, 64, 128]
      zero_point: [true, false]
      version: ["gemm"]
      ...

    bnb:
      bits: [4, 8]
      bnb_4bit_quant_type: ["nf4", "fp4"]
      bnb_4bit_use_double_quant: [true, false]
      bnb_4bit_compute_dtype: ["float16", "bfloat16"]
      ...
```

---

## 使用方式

### 1. 執行優化

**標準執行**:
```bash
python tmp/test/test_optimization.py --config tmp/config/optimization_config_llm.yaml
```

**後台執行**:
```bash
./test.sh  # 輸出到 run_optimization.log
```

### 2. 查看結果

**優化完成後**，檢查輸出目錄:
```
results/llm_optimization/{experiment}_{timestamp}/
├── all_trials.json              # 所有試驗結果
├── pareto_frontier.json         # Pareto 前沿解
├── recommendation.json          # 推薦配置
├── agent_conversations.jsonl    # Agent 對話（機器可讀）
└── agent_conversations.txt      # Agent 對話（人類可讀）
```

### 3. Web 檢視器

使用 Streamlit 檢視實驗結果：
```bash
streamlit run results/optimization_web_viewer.py
```

### 4. 調試和監控

**實時監控日誌**:
```bash
tail -f results/llm_optimization/*/optimization.log
```

**查看 Agent 決策過程**:
1. 打開 `agent_conversations.txt`
2. 按 Trial 分組查看每個 Agent 的輸入、輸出和理由
3. 檢查 `rationale` 字段理解為什麼做出某個決策

**啟用 Prompt 保存**（調試用）:
```yaml
logging:
  save_prompts: true
```

---

## 附錄: 文件結構

```
Green_AI/
├── tmp/
│   ├── agent/
│   │   ├── optimizers/
│   │   │   ├── llm_multiagent_optimizer.py    # 主控制器
│   │   │   ├── base_optimizer.py              # 優化器基類
│   │   │   ├── optuna_mo_optimizer.py         # Optuna 優化器
│   │   │   ├── random_optimizer.py            # 隨機優化器
│   │   │   ├── agents/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base_agent.py              # Agent 基類
│   │   │   │   ├── analyzer_agent.py          # 分析代理（Separate）
│   │   │   │   ├── planner_agent.py           # 規劃代理（Separate）
│   │   │   │   ├── strategist_agent.py        # 策略師代理（Combined）
│   │   │   │   └── monitor_agent.py           # 監控代理（純規則）
│   │   │   └── utils/
│   │   │       ├── __init__.py
│   │   │       ├── llm_client.py              # LLM 客戶端封裝
│   │   │       ├── prompt_templates.py        # Prompt 模板 + 配置載入器
│   │   │       ├── conversation_logger.py     # 對話記錄器
│   │   │       └── conversation_formatter.py  # JSONL → TXT 轉換器
│   │   ├── optimization_orchestrator.py
│   │   ├── config_loader.py
│   │   ├── baseline_evaluator.py
│   │   ├── evaluator_agent.py
│   │   ├── result_tracker.py
│   │   └── visualization.py
│   ├── config/
│   │   ├── optimization_config_llm.yaml       # LLM 優化配置
│   │   ├── prompt_config.yaml                 # Prompt 模板配置（多語言）
│   │   ├── model_config.yaml
│   │   ├── dataset_config.yaml
│   │   └── optimization_config.yaml
│   └── test/
│       └── test_optimization.py
├── results/
│   ├── llm_optimization/                      # LLM 優化結果輸出目錄
│   └── optimization_web_viewer.py             # Streamlit Web 檢視器
└── LLM_MULTIAGENT_ARCHITECTURE.md             # 本文檔
```

---

## 常見問題 (FAQ)

### Q1: Combined 模式和 Separate 模式哪個好？

**A**: 推薦使用 **Combined 模式**（`agent_mode: "combined"`）。它只需 1 次 LLM 調用，API 成本減半，且回應更一致。除非需要特別深入的分析，否則 Combined 模式足夠使用。

### Q2: LLM 失敗會導致優化中斷嗎？

**A**: 取決於 `fallback.on_llm_failure` 設置：
- `"stop"`: 會中斷優化並拋出異常
- `"random"`: 降級到規則引擎，繼續優化

推薦測試時用 `"stop"` 發現問題，生產時用 `"random"` 容錯。

### Q3: MonitorAgent 為什麼不使用 LLM？

**A**: MonitorAgent 的停止條件判斷是確定性的（預算、收斂、目標達成），使用規則更穩定可靠，且不需要額外的 API 成本。

### Q4: 如何切換 Prompt 語言？

**A**: 修改 `optimization_config_llm.yaml`：
```yaml
optimizer:
  llm_agent:
    prompt:
      type: "zh"    # 改為 "en" 即切換為英文
```

或在 `prompt_config.yaml` 中添加新的語言類型。

### Q5: 如何理解 Agent 的決策？

**A**: 查看 `agent_conversations.txt` 文件：
- 每個決策都有 `rationale`（理由）字段
- Combined 模式下，`strategist_result` 同時包含 `analysis` 和 `decision`
- 可以在 Web 檢視器的「Agent 對話」頁面查看原始 JSON

### Q6: 如何調整 Agent 的"創造力"？

**A**: 修改 `agent_temperatures`：
- **Analyzer**: 0.1-0.3（低溫）- 保證分析穩定性
- **Planner**: 0.5-0.9（中高溫）- 增加策略多樣性
- **Strategist**: 0.3-0.7（中溫）- 平衡分析和創造
- **Monitor**: 不需要，純規則

### Q7: `next_config` 格式需要注意什麼？

**A**: LLM 輸出的 `next_config` 必須是**扁平結構**：
```json
// 正確
"next_config": {"method": "awq", "w_bit": 4, "q_group_size": 128}

// 錯誤（嵌套在 parameters 中）
"next_config": {"method": "awq", "parameters": {"w_bit": 4, ...}}
```

Prompt 中已包含提醒，但若 LLM 仍輸出錯誤格式，`evaluator_agent.py` 將無法正確讀取參數。

---

**文檔版本**: v1.1
**最後更新**: 2026-01-27
**相關文件**: `CLAUDE.md`, `tmp/config/optimization_config_llm.yaml`, `tmp/config/prompt_config.yaml`
