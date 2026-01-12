# LLM Multi-Agent 量化優化系統架構文檔

> **版本**: Phase 1 MVP
> **更新日期**: 2026-01-02
> **用途**: 使用 LLM 驅動的多 Agent 系統自動優化量化超參數

---

## 目錄

1. [系統概述](#系統概述)
2. [整體架構](#整體架構)
3. [Agent 對話協議](#agent-對話協議)
4. [AnalyzerAgent 分析代理](#analyzeragent-分析代理)
5. [PlannerAgent 規劃代理](#planneragent-規劃代理)
6. [MonitorAgent 監控代理](#monitoragent-監控代理)
7. [對話記錄系統](#對話記錄系統)
8. [配置說明](#配置說明)
9. [使用方式](#使用方式)

---

## 系統概述

### 核心理念

傳統的超參數優化（如 Optuna）使用統計算法（TPE、NSGA-II）來探索搜索空間。本系統使用 **LLM 驅動的多 Agent 系統**，讓 AI 扮演"優化專家"的角色，基於歷史試驗數據做出智能決策。

### 核心優勢

1. **語義理解**: LLM 能理解量化參數的含義（如 `bits`、`group_size`）以及它們對性能的影響
2. **自適應策略**: 根據優化階段（早期/中期/後期）自動調整探索/利用比例
3. **可解釋性**: 每個決策都有自然語言的 `rationale`（理由），便於理解和調試
4. **容錯機制**: LLM 失敗時可降級到規則引擎（Rule-based Fallback）

### 三個核心 Agent

| Agent | 職責 | 輸出 |
|-------|------|------|
| **AnalyzerAgent** | 分析歷史試驗，識別失敗模式、未探索區域、參數敏感度 | 分析報告 (JSON) |
| **PlannerAgent** | 根據分析報告決定下一個試驗的配置 | 配置決策 (JSON) |
| **MonitorAgent** | 監控優化進度，判斷是否收斂或達到目標 | 停止評估 (JSON) |

---

## 整體架構

### 系統組件圖

```
┌─────────────────────────────────────────────────────────────┐
│                  LLMMultiAgentOptimizer                     │
│                     (主控制器)                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ AnalyzerAgent│ │ PlannerAgent │ │ MonitorAgent │
│   (分析器)    │ │   (規劃器)    │ │   (監控器)    │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
                 ┌──────────────┐
                 │  LLMClient   │
                 │ (OpenAI GPT-4o)│
                 └──────────────┘
```

### 工作流程 (每個 Trial)

```
Trial N 開始
    │
    ▼
┌──────────────────────────────────────────────┐
│ [1] AnalyzerAgent 分析歷史試驗                │
│   輸入: trials, pareto_frontier, search_space │
│   輸出: analysis (失敗模式、未探索區域等)       │
└──────────────┬───────────────────────────────┘
               ▼
┌──────────────────────────────────────────────┐
│ [2] PlannerAgent 規劃下一個配置               │
│   輸入: analysis, trial_num, targets         │
│   輸出: decision (strategy, next_config)     │
└──────────────┬───────────────────────────────┘
               ▼
┌──────────────────────────────────────────────┐
│ [3] 配置驗證 (簡單規則檢查)                   │
│   檢查: GPTQ 參數兼容性、合理性               │
└──────────────┬───────────────────────────────┘
               ▼
┌──────────────────────────────────────────────┐
│ [4] 執行試驗 (量化 + 評估)                    │
│   - 量化模型                                  │
│   - 在數據集上評估                            │
│   - 計算目標變化 (accuracy, GPU, latency)    │
└──────────────┬───────────────────────────────┘
               ▼
┌──────────────────────────────────────────────┐
│ [5] MonitorAgent 檢查停止條件                 │
│   輸入: trials, pareto, budget_status        │
│   輸出: should_stop, reason                  │
└──────────────┬───────────────────────────────┘
               ▼
       是否停止? ──No──► 下一個 Trial
               │
              Yes
               ▼
       返回最佳配置
```

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
        "analysis": {...}               # 實際分析結果
    },
    "trial_context": 1,                 # 關聯的 Trial 編號
    "timestamp": "2025-12-30T18:39:10"  # 時間戳
}
```

### 消息類型 (message_type)

| 消息類型 | 發送者 | 用途 |
|---------|-------|------|
| `analysis_report` | AnalyzerAgent | 歷史試驗分析報告 |
| `strategy_decision` | PlannerAgent | 下一個試驗的配置決策 |
| `progress_assessment` | MonitorAgent | 優化進度評估和停止建議 |
| `trial_start` | Orchestrator | Trial 開始事件 |
| `trial_end` | Orchestrator | Trial 結束事件 |
| `error` | Any | 錯誤事件 |

### 對話流程示例

```
Trial 1:
  [18:39:10] AnalyzerAgent → Orchestrator: analysis_report
      "推薦探索未嘗試的參數組合..."

  [18:39:15] PlannerAgent → Orchestrator: strategy_decision
      "策略: balanced, 配置: GPTQ 8-bit, 理由: ..."

  [18:41:16] Orchestrator: trial_end (SUCCESS)
      準確率 +33%, GPU -36%, 延遲 +92%

  [18:41:18] MonitorAgent → Orchestrator: progress_assessment
      "建議: CONTINUE, 預算尚未用完..."
```

---

## AnalyzerAgent 分析代理

### 職責

分析歷史試驗數據，識別：
1. **失敗模式** (Failure Patterns): 哪些配置容易失敗？
2. **未探索區域** (Unexplored Regions): 哪些參數組合值得嘗試？
3. **參數敏感度** (Parameter Sensitivity): 哪些參數對性能影響最大？
4. **Pareto 質量** (Pareto Quality): 當前解的多樣性和覆蓋率如何？

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
    'failure_patterns': [           # 失敗模式
        {
            'pattern': "GPTQ 2-bit 配置失敗率高",
            'confidence': 0.85,     # 信心度 (0-1)
            'affected_configs': []  # 可能受影響的配置
        }
    ],
    'unexplored_regions': [         # 未探索區域
        {
            'method': 'awq',
            'params': {'w_bit': 4, 'q_group_size': 64},
            'exploration_score': 0.75,  # 探索價值分數
            'rationale': "AWQ 方法嘗試次數少，值得探索"
        }
    ],
    'parameter_sensitivity': {      # 參數敏感度
        'bits': 0.9,                # 影響分數 (0-1)
        'group_size': 0.6,
        'method': 0.85
    },
    'pareto_quality': {             # Pareto 質量評估
        'diversity': 0.7,           # 多樣性 (0-1)
        'coverage': 0.5,            # 覆蓋率 (0-1)
        'improvement_rate': 0.3     # 改善率
    },
    'recommendations': [            # 建議
        "嘗試較小的 group_size 以改善延遲",
        "探索 AWQ 方法與不同 zero_point 設置"
    ]
}
```

### Prompt 設計（中文版）

<details>
<summary><b>點擊展開 Analyzer Prompt 中文翻譯</b></summary>

```
你是量化優化試驗的專家分析師。

# 你的任務
分析歷史試驗數據，識別模式、失敗原因和未探索區域。

# 歷史試驗摘要
總試驗數: {N}
成功試驗: {M}
失敗/剪枝試驗: {K}

最近的試驗:
  ✓ Trial 1: gptq-8bit (準確率:+33%, GPU:-36%, 延遲:+92%)
  ✓ Trial 2: gptq-4bit (準確率:+8%, GPU:-52%, 延遲:+83%)
  ...

# 當前 Pareto 前沿
找到 {P} 個非支配解。
  1. gptq-4bit ✓: 準確率:+8%, GPU:-52%, 延遲:+83%
  2. gptq-8bit ✗: 準確率:+33%, GPU:-36%, 延遲:+92%
  ...

# 搜索空間
方法: ['gptq', 'awq']

GPTQ:
  bits: [2, 3, 4, 8]
  group_size: [32, 64, 128]
  desc_act: [true, false]
  ...

AWQ:
  w_bit: [4]
  q_group_size: [64, 128]
  zero_point: [true, false]
  ...

# 分析要求
請以 JSON 格式提供全面的分析，包含以下字段:

1. **failure_patterns** (失敗模式列表):
   - pattern: 模式描述
   - confidence: 0.0-1.0 (你有多確信)
   - affected_configs: 可能失敗的相似配置列表

2. **unexplored_regions** (未探索的有前景參數組合):
   - method: 量化方法
   - params: 具體參數值
   - exploration_score: 0.0-1.0 (探索價值有多高)
   - rationale: 為什麼這個區域有前景

3. **parameter_sensitivity** (哪些參數影響最大):
   - parameter: 參數名稱
   - impact_score: 0.0-1.0 (越高 = 越重要)
   - observation: 你觀察到什麼

4. **pareto_quality** (Pareto 前沿質量評估):
   - diversity: 0.0-1.0 (在目標空間的分布)
   - coverage: 0.0-1.0 (理論前沿覆蓋率 %)
   - improvement_rate: 新解加入的趨勢

5. **recommendations** (戰略建議列表):
   - recommendation: 簡短可執行的建議

只輸出 JSON 對象（不要其他文字）:
{
  "failure_patterns": [...],
  "unexplored_regions": [...],
  "parameter_sensitivity": {...},
  "pareto_quality": {...},
  "recommendations": [...]
}
```

</details>

### Fallback 機制

當 LLM 失敗時，使用簡單的統計規則：
- 統計每個方法的失敗次數識別失敗模式
- 檢查哪些方法嘗試次數 < 3 作為未探索區域
- 使用固定的參數敏感度估計值
- 基於 Pareto 大小計算質量指標

**代碼位置**: `tmp/agent/optimizers/agents/analyzer_agent.py:_fallback_analysis()`

---

## PlannerAgent 規劃代理

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
    'pareto': List[Dict]    # 當前 Pareto 前沿
}
```

**輸出** (`decision`):
```python
{
    'strategy': 'balanced',         # 策略: exploration/exploitation/balanced
    'next_config': {                # 下一個試驗配置
        'method': 'gptq',
        'bits': 4,
        'group_size': 32,
        'desc_act': False,
        ...
    },
    'rationale': "探索 GPTQ 4-bit + 小 group_size 可能顯著改善 GPU 使用並保持可接受的準確率。符合分析建議。",
    'confidence': 0.75,             # 信心度 (0-1)
    'expected_objectives': {        # 預期目標變化
        'accuracy_change': -0.05,
        'gpu_peak_change': -0.45,
        'latency_change': 0.15
    },
    'alternative_configs': [...]    # 備選配置
}
```

### Prompt 設計（中文版）

<details>
<summary><b>點擊展開 Planner Prompt 中文翻譯</b></summary>

```
你是量化優化的戰略規劃師。

# 當前狀態
試驗: 2 / 10
進度: 20.0%
剩餘預算: 8 次試驗

# 來自 AnalyzerAgent 的分析
{
  "failure_patterns": [...],
  "unexplored_regions": [
    {
      "method": "awq",
      "params": {"w_bit": 4, "q_group_size": 64},
      "exploration_score": 0.75,
      "rationale": "AWQ 方法嘗試次數少，值得探索"
    }
  ],
  "parameter_sensitivity": {...},
  "pareto_quality": {...},
  "recommendations": [
    "嘗試較小的 group_size 以改善延遲",
    "探索 AWQ 方法與不同 zero_point 設置"
  ]
}

# 優化目標
{
  "accuracy_min": -0.20,    # 準確率下降不超過 20%
  "gpu_peak_max": -0.40,    # GPU 記憶體減少至少 40%
  "latency_max": 1.0        # 延遲增加不超過 100%
}

# 當前 Pareto 前沿
2 個解在前沿上
  1. gptq-4bit ✓: 準確率:+8%, GPU:-52%, 延遲:+83%
  2. gptq-8bit ✗: 準確率:+33%, GPU:-36%, 延遲:+92%

# 你的任務
基於分析決定下一次試驗的配置。

# 策略選擇指南
- **EXPLORATION** (探索) - 早期階段或卡住時:
  - 嘗試未探索的參數區域
  - 測試多樣化的配置
  - 專注於覆蓋率

- **EXPLOITATION** (利用) - 中後期有良好模式時:
  - 優化有前景的配置
  - 在成功試驗周圍做小擾動
  - 專注於改善

- **BALANCED** (平衡):
  - 混合兩種策略
  - 良好的默認選擇

# 基於進度的推薦策略
- < 20% 進度: 偏好 EXPLORATION (70%)
- 20-70% 進度: BALANCED (50/50)
- > 70% 進度: 偏好 EXPLOITATION (80%)

# 輸出要求
以 JSON 格式提供你的決策:

{
  "strategy": "exploration" / "exploitation" / "balanced",
  "next_config": {
    "method": "gptq" / "awq" / "bnb",
    "bits": ...,
    "group_size": ...,
    ... (所有相關參數)
  },
  "rationale": "解釋為什麼選擇這個配置 (2-3 句話)",
  "confidence": 0.0-1.0,
  "expected_objectives": {
    "accuracy_change": 預期值,
    "gpu_peak_change": 預期值,
    "latency_change": 預期值
  },
  "alternative_configs": [
    {...},  # 備選方案 1
    {...}   # 備選方案 2
  ]
}

只輸出 JSON 對象（不要其他文字）。
```

</details>

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

## MonitorAgent 監控代理

### 職責

監控優化進度並判斷是否應該停止優化，檢查：
1. **收斂** (Convergence): Pareto 前沿是否停止改善？
2. **目標達成** (Target Met): 是否已找到足夠多滿足目標的解？
3. **預算耗盡** (Budget Exhausted): 試驗次數是否用完？

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
        {'trial_count': 1, 'pareto_size': 1, 'hypervolume': 0.0},
        {'trial_count': 2, 'pareto_size': 2, 'hypervolume': 0.0},
        ...
    ]
}
```

**輸出** (`assessment`):
```python
{
    'should_stop': False,           # 是否應該停止
    'reason': 'continue',           # 原因: convergence/target_met/budget_exhausted/continue
    'convergence_score': 0.15,      # 收斂分數 (0=無收斂, 1=完全收斂)
    'progress_metrics': {           # 進度指標
        'pareto_improvement_rate': 0.67,  # Pareto 改善率
        'target_satisfaction': 0.50,      # 目標滿足率
        'budget_used': 0.30,              # 預算使用率
        'trials_since_improvement': 0     # 自上次改善以來的試驗數
    },
    'recommendation': "繼續優化。正在取得進展。"
}
```

### Prompt 設計（中文版）

<details>
<summary><b>點擊展開 Monitor Prompt 中文翻譯</b></summary>

```
你正在監控優化進度。

# 進度摘要
總試驗數: 3
成功試驗: 3
Pareto 解: 2
滿足目標: 1

# 預算狀態
已使用試驗: 3 / 10
進度: 30.0%

# Pareto 前沿演化
最近 3 個檢查點:
  1. Trial 1: 1 個解, HV=0.0000
  2. Trial 2: 2 個解, HV=0.0000
  3. Trial 3: 2 個解, HV=0.0000
  改善: +0.00%

# 目標 vs 當前最佳
目標:
  accuracy_min: -20.00%
  gpu_peak_max: -40.00%
  latency_max: +100.00%

最佳解:
  accuracy: +8.33%
  gpu_peak: -52.09%
  latency: +83.31%
  滿足目標: True

# 你的任務
評估優化是否應該繼續或停止。

# 停止標準考慮
1. **CONVERGENCE** (收斂): Pareto 前沿在最近 N 次試驗中沒有改善
2. **TARGET_MET** (目標達成): 找到滿足所有用戶目標的解
3. **BUDGET** (預算): 預算幾乎耗盡 (>95%)
4. **DIMINISHING_RETURNS** (收益遞減): 儘管多次試驗，改善很小

# 輸出要求
以 JSON 格式提供評估:

{
  "should_stop": true / false,
  "reason": "convergence" / "target_met" / "budget_exhausted" / "continue" / null,
  "convergence_score": 0.0-1.0,  # 0=無收斂, 1=完全收斂
  "progress_metrics": {
    "pareto_improvement_rate": 最近的改善率,
    "target_satisfaction": 滿足目標的百分比 (0.0-1.0),
    "budget_used": 預算消耗百分比 (0.0-1.0),
    "trials_since_improvement": 計數
  },
  "recommendation": "簡短建議 (1-2 句話)"
}

只輸出 JSON 對象（不要其他文字）。
```

</details>

### 規則引擎 (Rule-based Stopping)

MonitorAgent 總是**先**執行規則檢查，只有規則未觸發停止時才調用 LLM：

**停止規則**:
1. **預算耗盡**: `budget_used >= 0.95` (95%)
2. **收斂檢測**: 連續 N 輪 (window=5) Pareto 大小不變
3. **目標滿足**: 有 3 個以上滿足目標的解，且佔 Pareto 前沿 ≥50%

**代碼位置**: `tmp/agent/optimizers/agents/monitor_agent.py:_rule_based_stopping()`

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

### JSONL 格式示例

每行是一個 JSON 對象：

```jsonl
{"from_agent": "AnalyzerAgent", "to_agent": "Orchestrator", "message_type": "analysis_report", "content": {...}, "trial_context": 1, "timestamp": "2025-12-30T18:39:10"}
{"event": "trial_start", "trial_num": 1, "timestamp": "2025-12-30T18:39:06"}
{"from_agent": "PlannerAgent", "to_agent": "Orchestrator", "message_type": "strategy_decision", "content": {...}, "trial_context": 1, "timestamp": "2025-12-30T18:39:15"}
{"event": "trial_end", "trial_num": 1, "success": true, "result": {...}, "timestamp": "2025-12-30T18:41:16"}
{"from_agent": "MonitorAgent", "to_agent": "Orchestrator", "message_type": "progress_assessment", "content": {...}, "trial_context": 1, "timestamp": "2025-12-30T18:41:18"}
```

### TXT 格式示例

```
======================================================================
LLM MULTI-AGENT OPTIMIZATION - CONVERSATION LOG
======================================================================

======================================================================
Trial 1
======================================================================

[18:39:06] TRIAL START

----------------------------------------------------------------------

[18:39:10] AnalyzerAgent → Orchestrator
Type: analysis_report

Input: 0 trials analyzed, 0 Pareto solutions

Recommendations:
  1. {'recommendation': 'Initiate trials with unexplored parameter combinations...'}
  2. {'recommendation': 'Focus on varying group sizes and bit precision...'}

Unexplored regions: 2
----------------------------------------------------------------------

[18:39:15] PlannerAgent → Orchestrator
Type: strategy_decision

Progress: 1/5
Strategy: balanced

Next Configuration:
  Method: gptq
  Bits: 8
  Group size: 128

Rationale:
  This configuration explores a high exploration score option...

Confidence: 0.70
----------------------------------------------------------------------

[18:41:16] TRIAL END - ✓ SUCCESS

Objectives achieved:
  Accuracy change: +33.33%
  GPU peak change: -35.56%
  Latency change: +92.63%

----------------------------------------------------------------------

[18:41:18] MonitorAgent → Orchestrator
Type: progress_assessment

Trials: 1, Pareto: 1
Budget: 1/5

Decision: CONTINUE
Reason: continue

Recommendation:
  Continue optimization as the budget is largely unused...

Convergence: 0.00
----------------------------------------------------------------------
```

### 查看對話記錄

**查看 TXT 檔案**（推薦）:
```bash
cat results/llm_optimization/llm-multiagent-mvp_20251230_183905/agent_conversations.txt
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
  type: "llm_multiagent"  # 使用 LLM 多 Agent 優化器

  llm_agent:
    # LLM Provider
    provider: "openai"
    model: "gpt-4o"          # OpenAI: gpt-4o, gpt-4, gpt-3.5-turbo
                             # Anthropic: claude-3-opus, claude-3-sonnet
    api_key_env: "OPENAI_API_KEY"  # 環境變量名稱

    # 生成參數
    temperature: 0.7         # 全局默認溫度
    max_tokens: 2048         # 最大回應長度

    # Agent 特定溫度（覆蓋全局）
    agent_temperatures:
      analyzer: 0.3          # 低溫保證分析一致性
      planner: 0.7           # 中高溫增加策略創造力
      monitor: 0.4           # 低中溫保證穩定評估

    # Trial 限制
    max_trials: 5            # 最大試驗次數
    min_trials: 3            # 最少試驗次數（防止過早停止）
```

#### 2. Fallback 機制

```yaml
  fallback:
    on_llm_failure: "stop"   # LLM 失敗時的行為
                             # "random": 降級到隨機採樣（容錯）
                             # "stop": 直接停止優化（嚴格）
    max_llm_failures: 3      # 超過 N 次失敗後觸發 fallback
```

**推薦設置**:
- 測試階段: `"stop"` - 發現問題立即停止
- 生產階段: `"random"` - 容錯繼續運行

#### 3. 對話日誌

```yaml
  logging:
    save_conversations: true                    # 啟用對話記錄
    conversation_file: "agent_conversations.jsonl"  # JSONL 檔名
    save_prompts: true                          # 保存完整 prompts（調試用）
```

#### 4. 搜索空間

```yaml
  search_space:
    methods: ["gptq", "awq"]  # Phase 1: 只測試 2 個方法

    gptq:
      bits: [4, 8]            # 量化位元數
      group_size: [32, 64, 128]  # 分組大小
      desc_act: [true, false]    # 是否使用 activation 重排序
      ...

    awq:
      w_bit: [4]
      q_group_size: [64, 128]
      zero_point: [true, false]
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

### 3. 查看結果

**優化完成後**，檢查輸出目錄:
```
results/llm_optimization/llm-multiagent-mvp_20251230_183905/
├── all_trials.json              # 所有試驗結果
├── pareto_frontier.json         # Pareto 前沿解
├── recommendation.json          # 推薦配置
├── agent_conversations.jsonl    # Agent 對話（機器可讀）
└── agent_conversations.txt      # Agent 對話（人類可讀）✨
```

**查看對話流程**:
```bash
cat results/llm_optimization/llm-multiagent-mvp_*/agent_conversations.txt
```

**查看推薦配置**:
```bash
cat results/llm_optimization/llm-multiagent-mvp_*/recommendation.json
```

### 4. 調試和監控

**實時監控日誌**:
```bash
tail -f results/llm_optimization/llm-multiagent-mvp_*/optimization.log
```

**查看 Agent 決策過程**:
1. 打開 `agent_conversations.txt`
2. 按 Trial 分組查看每個 Agent 的輸入、輸出和理由
3. 檢查 `rationale` 字段理解為什麼做出某個決策

**啟用 Prompt 保存**（調試用）:
```yaml
logging:
  save_prompts: true  # 會在 JSONL 中保存完整的 prompt 和 response
```

---

## 附錄: 文件結構

```
Green_AI/
├── tmp/
│   ├── agent/
│   │   ├── optimizers/
│   │   │   ├── llm_multiagent_optimizer.py    # 主控制器
│   │   │   ├── agents/
│   │   │   │   ├── base_agent.py              # Agent 基類
│   │   │   │   ├── analyzer_agent.py          # 分析代理
│   │   │   │   ├── planner_agent.py           # 規劃代理
│   │   │   │   └── monitor_agent.py           # 監控代理
│   │   │   └── utils/
│   │   │       ├── llm_client.py              # LLM 客戶端封裝
│   │   │       ├── prompt_templates.py        # Prompt 模板
│   │   │       ├── conversation_logger.py     # 對話記錄器
│   │   │       └── conversation_formatter.py  # JSONL → TXT 轉換器
│   │   └── config_loader.py
│   ├── config/
│   │   └── optimization_config_llm.yaml       # LLM 優化配置
│   └── test/
│       └── test_optimization.py               # 優化測試腳本
└── results/
    └── llm_optimization/                      # 優化結果輸出目錄
```

---

## 常見問題 (FAQ)

### Q1: LLM 失敗會導致優化中斷嗎？

**A**: 取決於 `fallback.on_llm_failure` 設置：
- `"stop"`: 會中斷優化並拋出異常
- `"random"`: 降級到規則引擎，繼續優化

推薦測試時用 `"stop"` 發現問題，生產時用 `"random"` 容錯。

### Q2: 如何理解 Agent 的決策？

**A**: 查看 `agent_conversations.txt` 文件：
- 每個決策都有 `rationale`（理由）字段
- 可以看到 Agent 基於什麼分析做出決策
- 檢查 `recommendations` 和 `strategy` 的對應關係

### Q3: 可以只用規則引擎不用 LLM 嗎？

**A**: 可以，但需要修改代碼。每個 Agent 都有 `_fallback_*` 方法實現了規則版本。可以直接調用這些方法而不是 `_call_llm_structured()`。

### Q4: 為什麼 Analyzer 的建議沒有被 Planner 採納？

**A**: Planner 會綜合考慮：
- Analyzer 的建議（`recommendations`）
- 當前優化階段（early/mid/late）
- 預算剩餘情況
- Pareto 前沿質量

Planner 可能因為更高優先級的因素（如預算快用完需要利用）而忽略某些建議。

### Q5: 如何調整 Agent 的"創造力"？

**A**: 修改 `agent_temperatures`:
- **Analyzer**: 0.1-0.3（低溫）- 保證分析穩定性
- **Planner**: 0.5-0.9（中高溫）- 增加策略多樣性
- **Monitor**: 0.1-0.4（低溫）- 保證停止判斷一致性

---

## 總結

LLM Multi-Agent 優化系統通過三個專門的 Agent（Analyzer、Planner、Monitor）協同工作，實現智能的量化超參數優化：

✅ **可解釋**: 每個決策都有自然語言理由
✅ **自適應**: 根據優化階段自動調整策略
✅ **容錯**: LLM 失敗時降級到規則引擎
✅ **可追溯**: 完整記錄 Agent 對話過程（JSONL + TXT）

配合詳細的對話記錄系統，您可以清楚地看到 AI 如何一步步做出優化決策！

---

**文檔版本**: v1.0
**最後更新**: 2026-01-02
**相關文件**: `CLAUDE.md`, `tmp/config/optimization_config_llm.yaml`
