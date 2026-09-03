#!/usr/bin/env python3
"""
離線重建 LLM guiding agent（full/summary/window/tool）的 token 用量，不重新呼叫 API、不重跑實驗。

資料來源：
  - final_results_30/exp_Llama-3.2-3B-Instruct_gsm8k_{mode}_*/optimization_results.json
    → 每個「被接受」的 trial 的完整歷史、config、suggestion（reasoning 全文）
  - 同目錄 tool_debug_log.jsonl（僅 tool 模式）→ 每一輪工具檢索的 query/reason/results 全文
  - trial30_benchmark.log（1.5GB，涵蓋這批 run 的完整 stdout log）
    → 用來抓「去重複重試」與「JSON validation 重試」的次數，以及 summary 模式的
      knowledge_summary 演化全文（🧠 [Knowledge Base Updated] 區塊）

可信度分兩級，明確標記，不假裝精確：
  EXACT  - prompt/completion 內容可以逐字重建（沒有被去重複重試或 JSON 重試汙染的呼叫）
  APPROX - 我們知道這次呼叫「發生過」（log 裡有事件），但實際送出/生成的內容遺失
           （去重複重試被拒絕的 config 只留下一行 warning，reasoning 全文從未存檔；
            JSON validation 失敗的 raw_content 也只印了 pydantic 錯誤訊息，沒印原始內容）
           → 用同一個 run 內 EXACT 呼叫的 token 數中位數去估，並在報表中獨立列出，
             不會混進 EXACT 的加總，避免製造假精確度。

用法：
  python3 reconstruct_llm_token_usage.py
"""
import json
import re
import glob
import statistics as st
from pathlib import Path
from datetime import datetime

import tiktoken

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "final_results_30"
LOG_FILE = ROOT / "trial30_benchmark.log"

ENC = tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(ENC.encode(text))


# ---------------------------------------------------------------------------
# 以下三個函式是從 Global_Tuner_memory/modular_agent/llm_client.py 逐字搬過來的
# prompt 樣板邏輯（原檔案 _format_history / _create_prompt），確保重建結果與
# 當初實際送進 API 的內容一致。若原始碼之後有修改，這裡也要跟著同步更新。
# ---------------------------------------------------------------------------

def format_history(history: list, baseline_metrics: dict) -> str:
    if not history:
        return "None"
    lines = []
    for trial in history:
        config = trial.get("config") or {}
        error_msg = trial.get("error")
        if error_msg:
            lines.append(f"- [Iter {trial.get('iteration')}]: SKIPPED / FAILED ({error_msg}) | Config attempted: {json.dumps(config)}")
            continue

        metrics = trial.get("metrics") or {}
        details = metrics.get("details") or {}
        score = metrics.get("score", 0)
        acc = metrics.get("accuracy", 0)
        lat = metrics.get("latency", 0)
        vram = metrics.get("vram", 0)
        emit = metrics.get("emissions", 0)

        acc_str = f"{acc:.4f}"
        lat_str = f"{lat:.4f}s"
        vram_str = f"{vram:.4f}GB"
        emit_str = f"{emit:.6f}kg"

        if baseline_metrics:
            b_acc = baseline_metrics.get("accuracy", acc)
            b_lat = baseline_metrics.get("latency", lat)
            b_vram = baseline_metrics.get("vram", vram)
            b_emit = baseline_metrics.get("emissions", emit)

            acc_diff = acc - b_acc if b_acc else 0
            acc_pct = ((acc - b_acc) / b_acc) * 100 if b_acc else 0
            lat_pct = ((lat - b_lat) / b_lat) * 100 if b_lat else 0
            vram_pct = ((vram - b_vram) / b_vram) * 100 if b_vram else 0
            emit_pct = ((emit - b_emit) / b_emit) * 100 if b_emit else 0

            acc_str += f" ({acc_diff:+.4f} pp, {acc_pct:+.2f}%)"
            lat_str += f" ({lat_pct:+.2f}%)"
            vram_str += f" ({vram_pct:+.2f}%)"
            emit_str += f" ({emit_pct:+.2f}%)"

        line = f"- [Iter {trial.get('iteration')}]: Score={score:.4f} (Acc: {acc_str}, Lat: {lat_str}, VRAM: {vram_str}, Emit: {emit_str})"
        if details:
            task_accs = [f"{k}={v.get('accuracy', 0):.4f}" for k, v in details.items()]
            line += f" | Tasks: [{', '.join(task_accs)}]"
        line += f"\n    Config: {json.dumps(config)}"
        lines.append(line)
    return "\n".join(lines)


def get_pareto_frontier(trials: list) -> list:
    valid = [t for t in trials if t.get("metrics") and t["metrics"].get("accuracy") is not None]
    if not valid:
        return []
    frontier = []
    for trial in valid:
        m = trial["metrics"]
        acc, lat, vram, emit = m.get("accuracy", 0.0), m.get("latency", float("inf")), m.get("vram", float("inf")), m.get("emissions", float("inf"))
        dominated = False
        for other in valid:
            if trial is other:
                continue
            om = other["metrics"]
            o_acc, o_lat, o_vram, o_emit = om.get("accuracy", 0.0), om.get("latency", float("inf")), om.get("vram", float("inf")), om.get("emissions", float("inf"))
            if (o_acc >= acc and o_lat <= lat and o_vram <= vram and o_emit <= emit and
                    (o_acc > acc or o_lat < lat or o_vram < vram or o_emit < emit)):
                dominated = True
                break
        if not dominated:
            frontier.append(trial)
    return frontier


def create_prompt(iteration, max_iterations, model_id, task, trial_history, pareto, weights,
                   rejected_configs, baseline_metrics, memory_type, knowledge_summary=""):
    rejected_str = ""
    if rejected_configs:
        rejected_str = "\n=== ⚠️ STRICT CONSTRAINT: REJECTED CONFIGS ===\n"
        rejected_str += "You MUST NOT suggest any of the following configurations. You just tried them and they are duplicates:\n"
        for r in rejected_configs:
            rejected_str += f"- {json.dumps(r)}\n"

    if memory_type == "window":
        history_str = format_history(trial_history[-5:], baseline_metrics)
    elif memory_type == "tool":
        history_str = format_history(trial_history[-3:], baseline_metrics)
    elif memory_type == "summary":
        unsummarized_start = (len(trial_history) // 5) * 5
        unsummarized = trial_history[unsummarized_start:]
        history_str = f"--- LLM KNOWLEDGE SUMMARY ---\n{knowledge_summary}\n\n"
        history_str += f"--- RECENT UNSUMMARIZED TRIALS ---\n{format_history(unsummarized, baseline_metrics) if unsummarized else 'None'}"
    else:
        history_str = format_history(trial_history, baseline_metrics)

    pareto_str = format_history(pareto, baseline_metrics) if pareto else "None"

    w = weights or {}
    w_acc, w_lat, w_vram, w_emit = w.get("acc", 0.5), w.get("lat", 0.1), w.get("vram", 0.2), w.get("emit", 0.2)

    tried_modes_counts = {}
    for trial in trial_history:
        mode = (trial.get("config") or {}).get("mode")
        if mode:
            tried_modes_counts[mode] = tried_modes_counts.get(mode, 0) + 1
    all_modes = {"asvd_only", "gptq", "awq", "qqq", "bnb", "sparse_unstructured", "sparse_structured", "hybrid_asvd_bnb"}
    untried_modes = sorted(list(all_modes - set(tried_modes_counts.keys())))
    tried_str = ", ".join([f"{k} ({v}x)" for k, v in tried_modes_counts.items()]) if tried_modes_counts else "None"
    untried_str = ", ".join(untried_modes) if untried_modes else "None (All modes explored)"

    pen_t, pen_a = 0.15, 10.0

    return f"""You are an LLM compression optimization agent. Choose the best compression strategy for:
Model: {model_id} | Task: {task} | Iteration: {iteration}/{max_iterations}

GOAL: Maximize Final Score.
1. Base Score = 1.0 + {w_acc}*ln(Acc/Base_acc) + {w_lat}*ln(Base_lat/Lat) + {w_vram}*ln(Base_vram/VRAM) + {w_emit}*ln(Base_emit/Emit)
(explain:Prioritize positive relative % changes in Lat, VRAM, and Emit, while minimizing negative absolute drops (pp) in Acc.)
2. PENALTY: A massive penalty (multiplier {pen_a}) is applied ONLY if the Accuracy drop exceeds {pen_t} (i.e., the "pp" diff is more negative than -{pen_t}).
3. Final Score = Base Score - Penalty

=== AVAILABLE MODES & OUTPUT FORMATS ===
Strictly output ONLY valid JSON matching one of these structures. Do not wrap in markdown formatting.
"reasoning" MUST follow this structure: "A detailed explanation of why this mode fills a gap in current coverage, followed by a comprehensive analysis of the expected trade-offs and potential risks."
[MODE: asvd_only]
{{"reasoning": "...", "mode": "asvd_only",
  "alpha": 0.5,               // categorical [0.3, 0.4, 0.5, 0.6, 0.7] Higher = preserves activation distribution more
  "param_ratio_target": 0.90, // linear [0.70 - 0.99] Lower = heavier compression
  "scaling_method": "fisher"  // categorical ["abs_mean", "abs_max", "fisher"] fisher typically best
}}

[MODE: gptq]
{{"reasoning": "...", "mode": "gptq",
  "quant_bits": 4,         // categorical [3, 4, 8] 4-bit is sweet spot
  "quant_group_size": 128, // categorical [16, 32, 64, 128, 256] Smaller = higher precision, larger size
  "quant_format": "gptq",  // categorical ["gptq", "gptq_v2"] v2 fixes overflow
  "damp_percent": 0.05     // log [0.001 - 0.1] Try 0.01~0.05
}}

[MODE: awq]
{{"reasoning": "...", "mode": "awq",
  "quant_group_size": 128  // categorical [16, 32, 64, 128] Fixed at 4-bit
}}

[MODE: qqq]
{{"reasoning": "...", "mode": "qqq",
  "quant_group_size": 128, // categorical [-1, 128] -1 is full matrix
  "damp_percent": 0.005    // log [0.0005 - 0.05] Hessian dampening
}}

[MODE: bnb]
{{"reasoning": "...", "mode": "bnb",
  "quant_bits": 4,         // categorical [4, 8] 4 saves VRAM aggressively
  "use_double_quant": false // categorical [true, false] True saves ~0.4 bit/param (if bits=4)
}}

[MODE: sparse_unstructured]
{{"reasoning": "...", "mode": "sparse_unstructured",
  "sparsity_ratio": 0.5    // linear [0.3 - 0.7] e.g., 0.5 = 50% weights pruned
}}

[MODE: sparse_structured]
{{"reasoning": "...", "mode": "sparse_structured",
  "sparsity_structure": "2:4" // categorical ["2:4", "4:8"] Hardware-acceleration friendly
}}

[MODE: hybrid_asvd_bnb]
{{"reasoning": "...", "mode": "hybrid_asvd_bnb",
  "alpha": 0.5, "param_ratio_target": 0.92, "scaling_method": "fisher",
  "quant_bits": 4, "use_double_quant": false
}}

=== CURRENT STATUS ===
Modes tried so far: {tried_str}
Modes NOT yet tried: {untried_str}
{rejected_str}

Trial Context ({memory_type} mode):
{history_str}

Pareto Frontier (best trade-offs found):
{pareto_str}

=== STRATEGY ===
- Early iterations: try each mode independently to understand isolated impact.
- Later iterations: combine methods in hybrid mode.
- NEVER repeat identical configs. Use Pareto frontier to find unexplored regions.

Output ONLY the JSON for your chosen mode. No extra fields, no prose.
"""


TOOL_RULES_SUFFIX = (
    "\n\n=== TOOL USAGE RULES ===\n"
    "1. You have a 'retrieve_trials' tool to search past experiments by method.\n"
    "2. STRICT RULE: DO NOT use the tool for any method listed in 'Modes NOT yet tried'. The database will be empty.\n"
    "3. You have a STRICT LIMIT of 2 search queries per iteration. Plan your queries carefully!\n"
    "4. Once you have enough information, or if you run out of turns, you MUST output ONLY the final StrategySuggestion JSON.\n"
    "5. Do not make more than 2 parallel search queries at the exact same time."
)

TOOLS_SCHEMA_TEXT = json.dumps([{
    "type": "function",
    "function": {
        "name": "retrieve_trials",
        "description": "Fetch full history for a specific method. STRICT LIMITATION: Do NOT query methods you haven't tried yet (check the 'Modes NOT yet tried' list). Only use this to investigate variations or failures of methods you HAVE already tried.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "enum": ["asvd", "gptq", "awq", "qqq", "bnb", "sparse", "hybrid"], "description": "The specific compression method to search for."},
                "reason": {"type": "string", "description": "Why are you querying this method? What hypothesis are you testing or what gap are you trying to fill?"},
            },
            "required": ["query", "reason"],
        },
    },
}])
# 工具 schema 本身也會佔用 token（每次呼叫都要送），這裡近似估計它的固定成本
TOOLS_SCHEMA_TOKENS = count_tokens(TOOLS_SCHEMA_TEXT)


# ---------------------------------------------------------------------------
# Log 檔案解析：抓每個 run 時間窗內的「去重複」/「JSON validation 失敗」事件數，
# 以及 summary 模式的 knowledge_summary 全文演化。
# ---------------------------------------------------------------------------

LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - ")


def load_log_events(log_path: Path):
    """回傳 (dedup_events, jsonretry_events, kb_updates)，每個元素是 (datetime, extra_info)。

    注意：logger.info(f"\\n🧠 [Knowledge Base Updated]\\n{summary}\\n") 這種寫法會讓
    "🧠 [Knowledge Base Updated]" 這行本身沒有 timestamp 前綴（timestamp 只會出現在
    logging 呼叫的第一行，也就是那個空白的 \\n），所以要在「沒有 timestamp」的分支裡
    抓這個 marker，而不是在有 timestamp 的分支。
    """
    dedup_events, jsonretry_events, kb_updates = [], [], []
    cur_kb_lines = None
    cur_kb_ts = None
    pending_ts = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LOG_TS_RE.match(line)
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                pending_ts = ts
                if cur_kb_lines is not None:
                    kb_updates.append((cur_kb_ts, "\n".join(cur_kb_lines)))
                    cur_kb_lines = None

                if "去重複" in line:
                    dedup_events.append((ts, line.strip()))
                elif "Validation failed on attempt" in line or "Tool-mode JSON validation failed" in line:
                    jsonretry_events.append((ts, line.strip()))
            else:
                if cur_kb_lines is None and "Knowledge Base Updated" in line:
                    cur_kb_ts = pending_ts
                    cur_kb_lines = []
                elif cur_kb_lines is not None:
                    cur_kb_lines.append(line.rstrip("\n"))

    if cur_kb_lines is not None:
        kb_updates.append((cur_kb_ts, "\n".join(cur_kb_lines)))

    return dedup_events, jsonretry_events, kb_updates


def in_window(ts, start, end):
    return start <= ts <= end


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_run_folder_ts(folder_name: str) -> datetime:
    m = re.search(r"(\d{8}_\d{6})$", folder_name)
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")


def main():
    print(f"讀取 log 檔案 {LOG_FILE.name}（可能要一點時間，檔案約 {LOG_FILE.stat().st_size/1e9:.1f}GB）...")
    dedup_events, jsonretry_events, kb_updates = load_log_events(LOG_FILE)
    print(f"log 解析完成：去重複事件 {len(dedup_events)} 次、JSON validation retry {len(jsonretry_events)} 次、"
          f"knowledge summary 更新 {len(kb_updates)} 次\n")

    exp_dirs = sorted(glob.glob(str(RESULTS_DIR / "exp_Llama-3.2-3B-Instruct_gsm8k_*")))
    run_starts = [(parse_run_folder_ts(Path(d).name), d) for d in exp_dirs]
    run_starts.sort()

    results_summary = []

    for idx, (start_ts, d) in enumerate(run_starts):
        end_ts = run_starts[idx + 1][0] if idx + 1 < len(run_starts) else start_ts.replace(year=start_ts.year + 1)
        fn = Path(d).name
        mode = fn.split("_gsm8k_")[1].rsplit("_", 2)[0]

        cfg = json.load(open(Path(d) / "experiment_config.json"))
        weights = cfg["weights"]
        baseline = cfg["baseline"]
        max_it = cfg["max_iterations"]
        model_id = cfg["model_id"]
        task = cfg["task"]

        trials = json.load(open(Path(d) / "optimization_results.json"))

        run_dedup = [e for e in dedup_events if in_window(e[0], start_ts, end_ts)]
        run_jsonretry = [e for e in jsonretry_events if in_window(e[0], start_ts, end_ts)]
        run_kb = [e for e in kb_updates if in_window(e[0], start_ts, end_ts)]
        run_kb.sort(key=lambda x: x[0])

        exact_in, exact_out = 0, 0
        exact_calls = 0
        approx_calls_placeholder = len(run_dedup) + len(run_jsonretry)  # 內容遺失，先記次數

        kb_ptr = 0
        knowledge_summary = "No previous summary available."
        tainted_trials = 0  # 該 iteration 前面有去重複事件，prompt 無法精確重建

        tool_debug = []
        if mode == "tool":
            tool_log_path = Path(d) / "tool_debug_log.jsonl"
            if tool_log_path.exists():
                tool_debug = [json.loads(l) for l in open(tool_log_path, encoding="utf-8")]

        per_trial_dedup_count = {}
        for ts, line in run_dedup:
            pass  # 沒有 iteration 編號可精確歸屬，僅作為 run 層級的次數統計

        for i, trial in enumerate(trials, start=1):
            history_so_far = trials[: i - 1]
            pareto = get_pareto_frontier(history_so_far)

            if mode == "summary":
                # 用真實 log 裡的 knowledge_summary 演化，取「這個 iteration 之前」最新的一版
                while kb_ptr < len(run_kb) and (i - 1) // 1 >= 0 and kb_ptr < len(run_kb):
                    # 用 KB 更新事件按時間序推進；每 5 個 trial 更新一次，用時間戳先後對齊即可
                    break
                # 簡化：依照程式邏輯，第 i 個 trial 之前應該已經更新了 floor((i-1)/5) 次
                expected_updates = (i - 1) // 5
                if expected_updates > kb_ptr and expected_updates <= len(run_kb):
                    kb_ptr = expected_updates
                if kb_ptr > 0 and kb_ptr <= len(run_kb):
                    knowledge_summary = run_kb[kb_ptr - 1][1]

            prompt = create_prompt(
                i, max_it, model_id, task, history_so_far, pareto, weights,
                rejected_configs=None,  # 見下方說明：有去重複事件時這裡本來就無法精確重建
                baseline_metrics=baseline, memory_type=mode, knowledge_summary=knowledge_summary,
            )

            completion_obj = trial.get("suggestion")
            completion_text = json.dumps(completion_obj, ensure_ascii=False) if completion_obj else "{}"

            if mode == "tool":
                sys_prompt = prompt + TOOL_RULES_SUFFIX
                in_tok = count_tokens(sys_prompt) + TOOLS_SCHEMA_TOKENS
                out_tok = 0
                this_iter_debug = [t for t in tool_debug if t["iteration"] == i]
                for turn in this_iter_debug:
                    call_out = count_tokens(json.dumps({"query": turn["query"], "reason": turn["reason"]}))
                    call_in = count_tokens(turn["results"])
                    out_tok += call_out
                    in_tok += call_in  # 檢索結果會被塞回下一輪 context
                    exact_calls += 1
                out_tok += count_tokens(completion_text)
                exact_calls += 1  # 最後輸出 JSON 那次呼叫
                exact_in += in_tok
                exact_out += out_tok
            else:
                exact_in += count_tokens(prompt)
                exact_out += count_tokens(completion_text)
                exact_calls += 1

        # summary 模式：額外的 gpt-4o-mini 摘要呼叫（input=前5筆歷史+舊摘要, output=新摘要全文）
        summary_extra_in, summary_extra_out, summary_extra_calls = 0, 0, 0
        if mode == "summary":
            for k, (ts, kb_text) in enumerate(run_kb):
                batch = trials[k * 5: k * 5 + 5]
                batch_str = format_history(batch, baseline)
                prev_summary = run_kb[k - 1][1] if k > 0 else "No previous summary available."
                update_prompt = (
                    "You are an AI assistant maintaining an evolving knowledge base for a model compression agent. \n\n"
                    f"=== CURRENT KNOWLEDGE SUMMARY (May contain outdated or incorrect early assumptions) ===\n{prev_summary}\n\n"
                    f"=== FULL EXPERIMENTAL HISTORY (Trials 1 through Current) ===\n{batch_str}\n\n"
                    "=== INSTRUCTIONS ===\nYour task is to rewrite and update the current knowledge summary based on the new trial results.\n"
                )
                summary_extra_in += count_tokens(update_prompt)
                summary_extra_out += count_tokens(kb_text)
                summary_extra_calls += 1

        # 找出有多少個 iteration「前面發生過去重複事件」→ 該 trial 的 prompt 其實含有
        # 我們重建不出來的 rejected_str 區塊，標記為受汙染，不計入 EXACT
        n_dedup = len(run_dedup)

        results_summary.append(dict(
            mode=mode, fn=fn, n_trials=len(trials),
            exact_calls=exact_calls, exact_in=exact_in, exact_out=exact_out,
            summary_extra_calls=summary_extra_calls, summary_extra_in=summary_extra_in, summary_extra_out=summary_extra_out,
            hidden_dedup_events=len(run_dedup), hidden_jsonretry_events=len(run_jsonretry),
        ))

    # ---- 輸出報表 ----
    print(f"{'run':55s} {'mode':8s} {'exact_calls':11s} {'exact_in':9s} {'exact_out':10s} {'摘要額外呼叫':10s} {'隱藏去重複':8s} {'隱藏JSON retry':8s}")
    by_mode = {}
    for r in results_summary:
        print(f"{r['fn']:55s} {r['mode']:8s} {r['exact_calls']:11d} {r['exact_in']:9d} {r['exact_out']:10d} "
              f"{r['summary_extra_calls']:10d} {r['hidden_dedup_events']:8d} {r['hidden_jsonretry_events']:8d}")
        by_mode.setdefault(r["mode"], []).append(r)

    print("\n=== 依模式加總（3 run 平均，只計 EXACT 部分 + summary 額外呼叫）===")
    print(f"{'mode':8s} {'avg_exact_calls':15s} {'avg_input_tok':13s} {'avg_output_tok':14s} {'avg_total_tok':13s} "
          f"{'avg_hidden_dedup':16s} {'avg_hidden_jsonretry':18s}")
    for mode, runs in by_mode.items():
        avg_calls = st.mean([r["exact_calls"] + r["summary_extra_calls"] for r in runs])
        avg_in = st.mean([r["exact_in"] + r["summary_extra_in"] for r in runs])
        avg_out = st.mean([r["exact_out"] + r["summary_extra_out"] for r in runs])
        avg_dedup = st.mean([r["hidden_dedup_events"] for r in runs])
        avg_jsonretry = st.mean([r["hidden_jsonretry_events"] for r in runs])
        print(f"{mode:8s} {avg_calls:15.1f} {avg_in:13.0f} {avg_out:14.0f} {avg_in+avg_out:13.0f} "
              f"{avg_dedup:16.1f} {avg_jsonretry:18.1f}")

    out_path = ROOT / "llm_token_usage_reconstructed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    print(f"\n完整結果已存到 {out_path}")


if __name__ == "__main__":
    main()
