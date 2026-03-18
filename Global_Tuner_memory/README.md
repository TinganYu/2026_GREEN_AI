# Agentic Memory Modes Benchmark (Branch)

This branch extends the core LLM Compression Orchestrator by introducing and benchmarking different **Agentic Memory Architectures**. 

As compression optimization requires tracking complex parameter interactions over many iterations, passing the entire history to the LLM quickly exhausts the context window. This branch implements four distinct memory strategies to evaluate how the agent performs under different context constraints.

## 🧠 The 4 Memory Modes

You can control how the agent remembers past experiments using the `--memory_type` argument.

### 1. `full` (Baseline)
* **How it works:** Passes the entire, unfiltered trial history into the prompt every single iteration.
* **Pros:** Maximum context; the LLM sees everything.
* **Cons:** Extremely high token cost; prone to "lost in the middle" hallucination as the context grows too large.

### 2. `window` (Short-term Memory)
* **How it works:** Only passes the most recent $N$ trials (default: last 5) into the prompt.
* **Pros:** Highly token-efficient; keeps the LLM focused on recent local gradients.
* **Cons:** The agent suffers from "catastrophic forgetting" and might unknowingly repeat failed configurations from early iterations.

### 3. `summary` (Knowledge Graph / Evolving State)
* **How it works:** Uses a cheaper secondary LLM (e.g., `gpt-4o-mini`) to periodically rewrite and update a textual "Knowledge Summary" of the experiment so far. The main agent only sees this summary plus the few most recent unsummarized trials.
* **Pros:** Preserves long-term insights (e.g., "ASVD alpha < 0.3 always crashes VRAM") without the raw token bloat.
* **Cons:** Requires extra API calls for the summarization step; relies heavily on the summarizer model's accuracy.

### 4. `tool` (ReAct / Retrieval Agent) 
* **How it works:** An interactive, bounded multi-round loop. The agent is given a strict, small context (last 3 trials + Pareto best), but is equipped with a `retrieve_trials` tool. If it wants to explore a specific method, it actively queries the database (e.g., `query="asvd"` or `query="gptq"`) to fetch relevant past configurations before making a final decision.
* **Pros:** The most "human-like" debugging approach. Keeps the baseline prompt tiny while allowing deep-dives into specific sub-methods on demand. Prevents infinite loops via a strict `MAX_TURNS` limit.
* **Cons:** Slightly more complex execution flow; multiple API calls per iteration if the tool is utilized.

---

## 📊 Running the Benchmark

You can automatically evaluate and compare all four memory modes using the built-in benchmark flag.

```bash
# Run a comparison benchmark (e.g., 3 full optimization runs per memory mode)
python orchestrator.py --benchmark_runs 3 --max_iterations 15 --model_id "meta-llama/Llama-3.2-1B-Instruct"
```

## Automated Reporting
When --benchmark_runs > 1 is triggered, the orchestrator will:

1. Sequentially test full, window, summary, and tool modes.

2. Utilize process isolation to ensure VRAM is completely flushed between runs.

3. Automatically generate a live-updating Markdown report (tuning_results/benchmark_report_YYYYMMDD_HHMMSS.md) comparing the strategies.

## Example Output:
| Memory 方法 | Description | Mean Score | Best Score | Std 標準差 | Valid Runs |
|---|---|---|---|---|---|
| full | 全部實驗結果 | 1.1042 | 1.1250 | 0.0150 | 3/3 |
| window | 最近 5 個 | 1.0520 | 1.0800 | 0.0210 | 3/3 |
| summary | LLM summary | 1.0950 | 1.1100 | 0.0120 | 3/3 |
| tool | Agentic Tool Retrieval | 1.1180 | 1.1320 | 0.0080 | 3/3 |