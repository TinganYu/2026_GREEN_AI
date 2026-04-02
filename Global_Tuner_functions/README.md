# 🌱 Green AI Compression Orchestrator

An automated, LLM-driven orchestrator designed to compress large language models (via Sparsity, ASVD, and Quantization) while autonomously finding the optimal Pareto frontier between **Accuracy**, **VRAM**, **Latency**, and **Carbon Emissions**.

## ✨ Key Features

* **🖥️ Interactive Web Dashboard:** A robust Streamlit UI for real-time experiment tracking, complete with a live terminal, dynamic Pareto charts, and crash-resilient state management.
* **🛑 Flexible Stopping Conditions:** Tailor the run to your exact constraints. Choose to stop the optimization based on a maximum number of iterations, a strict time limit (e.g., max 6 hours), or the moment specific performance targets are achieved.
* **🎯 Target-Driven Optimization:** Set strict performance goals (e.g., "50% less VRAM with < 5% accuracy drop"). The orchestrator uses a "Patience" mechanism to early-stop once optimal targets are met, saving time and API costs.
* **🧠 Agentic Memory Architectures:** Four distinct LLM context-management strategies (`full`, `window`, `summary`, `tool`) to navigate complex optimization spaces without context window exhaustion.
* **⚡ Smart Baseline Caching:** Automatically caches and intelligently reuses baseline evaluations across different runs and task combinations to avoid redundant compute.
* **🛡️ Process Isolation:** Executes all model evaluations in isolated, safely-killed subprocesses to guarantee 100% VRAM recovery and prevent out-of-memory (OOM) crashes.

---

## 🖥️ The Streamlit Dashboard

The easiest and most visual way to run the orchestrator is via the web dashboard. It provides real-time insights into what the LLM is testing and how the metrics are changing.

```bash
streamlit run Global_Tuner_functions/modular_agent/app.py
```

### Dashboard Features:
* **Live Progress & Smart Status:** Watch the exact state of the orchestrator (Baseline Evaluation → Iteration Generation → Plotting).
* **Scrollable Live Terminal:** View real-time logs, PyTorch outputs, and tracebacks directly in the browser. If an API call fails, the dashboard freezes the timer and preserves the error log for easy debugging.
* **Dynamic Trade-off Explorer:** An interactive scatter chart where you can map any metric against another (e.g., *Emissions vs. Accuracy* or *Latency vs. VRAM*) to visually identify the Pareto frontier.
* **Detailed Results Table:** A color-coded, sortable dataframe showing exactly what configurations the LLM agent generated and their resulting score deltas.

---

## ⚙️ CLI Usage & Advanced Targets

You can also run the orchestrator completely headless via the CLI. This is particularly useful for setting up automated, target-driven runs on remote servers.

```bash
python Global_Tuner_functions/modular_agent/orchestrator.py \
  --model_id "meta-llama/Llama-3.2-1B-Instruct" \
  --task "gsm8k,math" \
  --max_iterations 20 \
  --max_time_hours 6.5 \
  --target_vram_pct 0.5 \
  --target_acc_drop 0.05 \
  --patience 3
```
*In this example, the orchestrator will attempt to reduce VRAM by 50% while dropping no more than 5 points in accuracy. Once it hits this target, it enters "Patience" mode for 3 iterations to see if it can optimize it even further before early-stopping. It will also forcefully terminate if it hits 20 iterations or runs for 6.5 hours.*

---

## 🧠 Agentic Memory Architectures

As compression optimization requires tracking complex parameter interactions over many iterations, passing the entire history to the LLM quickly exhausts the context window. You can control how the agent remembers past experiments using the `--memory_type` argument.

### 1. `full` (Baseline)
* **How it works:** Passes the entire, unfiltered trial history into the prompt every single iteration.
* **Pros:** Maximum context; the LLM sees everything.
* **Cons:** Extremely high token cost; prone to "lost in the middle" hallucination as the context grows.

### 2. `window` (Short-term Memory)
* **How it works:** Only passes the most recent $N$ trials into the prompt.
* **Pros:** Highly token-efficient; keeps the LLM focused on recent local gradients.
* **Cons:** The agent suffers from "catastrophic forgetting" and might repeat failed configurations.

### 3. `summary` (Knowledge Graph / Evolving State)
* **How it works:** Uses a cheaper secondary LLM to periodically rewrite and update a textual "Knowledge Summary" of the experiment. The main agent only sees this summary plus the newest trials.
* **Pros:** Preserves long-term insights (e.g., *"ASVD alpha < 0.3 always crashes VRAM"*) without token bloat.
* **Cons:** Requires extra API calls for the summarization step.

### 4. `tool` (ReAct / Retrieval Agent) 
* **How it works:** The agent is given a strict, small context but is equipped with a `retrieve_trials` tool. It actively queries the database (e.g., `query="asvd"`) to fetch relevant past configurations before making a final decision.
* **Pros:** The most "human-like" debugging approach. Keeps the baseline prompt tiny while allowing deep-dives into specific sub-methods on demand.

---

## 📊 Automated Benchmarking 
```
Note: this feature is not added in the dashboard.
```
You can automatically evaluate and compare all four memory modes using the built-in benchmark flag. The orchestrator will sequentially test `full`, `window`, `summary`, and `tool` modes, utilizing process isolation to flush VRAM between runs.

```bash
# Run a comparison benchmark (e.g., 3 full optimization runs per memory mode)
python Global_Tuner_functions/modular_agent/orchestrator.py --benchmark_runs 3 --max_iterations 15
```

### Automated Reporting
The benchmark automatically generates a live-updating Markdown report (`tuning_results/benchmark_report_YYYYMMDD.md`) comparing the strategies:

| Memory 方法 | Description | Mean Score | Best Score | Std 標準差 | Valid Runs |
|---|---|---|---|---|---|
| full | 全部實驗結果 | 1.1042 | 1.1250 | 0.0150 | 3/3 |
| window | 最近 5 個 | 1.0520 | 1.0800 | 0.0210 | 3/3 |
| summary | LLM summary | 1.0950 | 1.1100 | 0.0120 | 3/3 |
| tool | Agentic Tool Retrieval | 1.1180 | 1.1320 | 0.0080 | 3/3 |