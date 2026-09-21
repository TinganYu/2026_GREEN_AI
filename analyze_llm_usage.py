#!/usr/bin/env python3
"""
讀取 llm_usage_log.jsonl（由 Global_Tuner_memory/modular_agent/llm_client.py 即時記錄，
每一次 API 呼叫一行，包含 input/output/total token、呼叫階段、去重複重試輪數、
JSON 驗證重試次數、tool 呼叫輪數），對照 optimization_results.json 產生：

  1. 每個 trial（iteration）的明細：正式呼叫 vs 隱藏重試次數、tool 輪數、token 分布
  2. 整個 run 的加總統計

只適用於「已經套用新版 llm_client.py 之後」跑的實驗——沒有 llm_usage_log.jsonl 的舊資料
（例如 final_results_30/）沒有這個檔案，仍要用 reconstruct_llm_token_usage.py 的離線估計法。

用法：
  python3 analyze_llm_usage.py <exp_dir>            # 單一實驗的詳細報表
  python3 analyze_llm_usage.py <parent_dir> --all    # 掃描 parent_dir 底下所有含
                                                       # llm_usage_log.jsonl 的實驗，各印一份摘要
"""
import argparse
import json
import statistics as st
from pathlib import Path
from collections import defaultdict


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_trial_names(exp_dir: Path) -> dict:
    """iteration -> trial_name（若有 optimization_results.json）"""
    p = exp_dir / "optimization_results.json"
    if not p.exists():
        return {}
    trials = json.load(open(p, encoding="utf-8"))
    return {t.get("iteration"): t.get("trial_name", f"iter_{t.get('iteration')}") for t in trials}


def per_iteration_breakdown(entries: list) -> dict:
    """把 usage log 依 iteration 分組，算出每個 iteration 的呼叫組成。"""
    by_iter = defaultdict(list)
    for e in entries:
        by_iter[e["iteration"]].append(e)

    result = {}
    for it, calls in by_iter.items():
        calls_sorted = sorted(calls, key=lambda c: (c.get("dedup_attempt", 0), c.get("tool_turn") or 0, c.get("json_attempt", 1)))

        n_total = len(calls_sorted)
        n_dedup_retry = len({c["dedup_attempt"] for c in calls_sorted if c.get("dedup_attempt", 0) > 0})
        n_json_retry = sum(1 for c in calls_sorted if c.get("json_attempt", 1) > 1)
        n_tool_turns = sum(1 for c in calls_sorted if c.get("phase") == "tool_call_turn")
        n_summary_updates = sum(1 for c in calls_sorted if c.get("phase") == "summary_update")

        in_tok = sum(c.get("input_tokens") or 0 for c in calls_sorted)
        out_tok = sum(c.get("output_tokens") or 0 for c in calls_sorted)

        by_phase = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        for c in calls_sorted:
            ph = c.get("phase", "?")
            by_phase[ph]["calls"] += 1
            by_phase[ph]["input_tokens"] += c.get("input_tokens") or 0
            by_phase[ph]["output_tokens"] += c.get("output_tokens") or 0

        result[it] = {
            "total_calls": n_total,
            "dedup_retry_rounds": n_dedup_retry,
            "json_retry_calls": n_json_retry,
            "tool_call_turns": n_tool_turns,
            "summary_update_calls": n_summary_updates,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "by_phase": dict(by_phase),
            "calls": calls_sorted,
        }
    return result


def print_run_report(exp_dir: Path, verbose: bool = True):
    entries = load_jsonl(exp_dir / "llm_usage_log.jsonl")
    if not entries:
        print(f"[{exp_dir.name}] 沒有 llm_usage_log.jsonl（舊資料，或這個 run 還沒套用新版 logging）")
        return None

    trial_names = load_trial_names(exp_dir)
    breakdown = per_iteration_breakdown(entries)

    print(f"\n{'=' * 100}")
    print(f"實驗：{exp_dir.name}")
    print(f"{'=' * 100}")

    if verbose:
        header = (f"{'Iter':5s} {'Trial':30s} {'總呼叫':6s} {'去重複輪':8s} {'JSON重試':8s} "
                   f"{'Tool輪':6s} {'摘要呼叫':8s} {'Input Tok':10s} {'Output Tok':11s} {'Total Tok':10s}")
        print(header)
        print("-" * len(header))
        for it in sorted(breakdown.keys()):
            b = breakdown[it]
            name = trial_names.get(it, f"iter_{it}")[:30]
            print(f"{it:<5d} {name:30s} {b['total_calls']:<6d} {b['dedup_retry_rounds']:<8d} "
                  f"{b['json_retry_calls']:<8d} {b['tool_call_turns']:<6d} {b['summary_update_calls']:<8d} "
                  f"{b['input_tokens']:<10d} {b['output_tokens']:<11d} {b['total_tokens']:<10d}")

            # 逐次呼叫細節（每筆 token 用在哪個 phase）
            for c in b["calls"]:
                tag = c["phase"]
                extra = []
                if c.get("dedup_attempt"):
                    extra.append(f"去重複第{c['dedup_attempt']}輪")
                if c.get("tool_turn"):
                    extra.append(f"tool turn {c['tool_turn']}")
                if c.get("json_attempt", 1) > 1:
                    extra.append(f"JSON重試第{c['json_attempt']}次")
                if c.get("validation_passed") is False:
                    extra.append("驗證失敗")
                extra_str = f"（{', '.join(extra)}）" if extra else ""
                print(f"      └─ {tag}{extra_str}: input={c.get('input_tokens')}, "
                      f"output={c.get('output_tokens')}, model={c.get('model')}")

    # ── Run 層級加總 ──────────────────────────────────────────────
    total_calls = sum(b["total_calls"] for b in breakdown.values())
    total_dedup = sum(b["dedup_retry_rounds"] for b in breakdown.values())
    total_jsonretry = sum(b["json_retry_calls"] for b in breakdown.values())
    total_tool_turns = sum(b["tool_call_turns"] for b in breakdown.values())
    total_summary = sum(b["summary_update_calls"] for b in breakdown.values())
    total_in = sum(b["input_tokens"] for b in breakdown.values())
    total_out = sum(b["output_tokens"] for b in breakdown.values())

    phase_totals = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})
    for b in breakdown.values():
        for ph, v in b["by_phase"].items():
            phase_totals[ph]["calls"] += v["calls"]
            phase_totals[ph]["input_tokens"] += v["input_tokens"]
            phase_totals[ph]["output_tokens"] += v["output_tokens"]

    print(f"\n--- Run 層級加總（{len(breakdown)} 個 iteration）---")
    print(f"總 API 呼叫數：{total_calls}")
    print(f"  其中去重複重試輪數：{total_dedup}　JSON 驗證重試次數：{total_jsonretry}　"
          f"tool 檢索輪數：{total_tool_turns}　摘要更新呼叫：{total_summary}")
    print(f"總 Input Token：{total_in:,}　總 Output Token：{total_out:,}　總 Token：{total_in + total_out:,}")
    print("\nToken 用在哪裡（依 phase 分解）：")
    for ph, v in sorted(phase_totals.items(), key=lambda kv: -kv[1]["input_tokens"] - kv[1]["output_tokens"]):
        print(f"  {ph:20s} 呼叫 {v['calls']:4d} 次｜Input {v['input_tokens']:>8,}｜Output {v['output_tokens']:>7,}")

    return {
        "exp_dir": exp_dir.name,
        "n_iterations": len(breakdown),
        "total_calls": total_calls,
        "total_dedup_retry_rounds": total_dedup,
        "total_json_retry_calls": total_jsonretry,
        "total_tool_call_turns": total_tool_turns,
        "total_summary_update_calls": total_summary,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "by_phase": {k: dict(v) for k, v in phase_totals.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="單一實驗資料夾，或搭配 --all 時的上層資料夾")
    ap.add_argument("--all", action="store_true", help="掃描 path 底下所有含 llm_usage_log.jsonl 的子資料夾")
    ap.add_argument("--quiet", action="store_true", help="只印 run 層級加總，不列逐 trial 明細")
    ap.add_argument("--out", default=None, help="把所有 run 的加總結果存成 JSON")
    args = ap.parse_args()

    root = Path(args.path)
    summaries = []

    if args.all:
        exp_dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "llm_usage_log.jsonl").exists())
        if not exp_dirs:
            print(f"{root} 底下沒有找到任何含 llm_usage_log.jsonl 的實驗資料夾。")
            return
        for d in exp_dirs:
            s = print_run_report(d, verbose=not args.quiet)
            if s:
                summaries.append(s)
    else:
        s = print_run_report(root, verbose=not args.quiet)
        if s:
            summaries.append(s)

    if args.out and summaries:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"\n加總結果已存到 {args.out}")


if __name__ == "__main__":
    main()
