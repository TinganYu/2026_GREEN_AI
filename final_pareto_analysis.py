#!/usr/bin/env python3
"""
Final Pareto frontier analysis report with visualizations.
"""
import json
from pathlib import Path
from typing import List, Dict, Any
import math

def print_header(title):
    print("\n" + "="*100)
    print(title.center(100))
    print("="*100)

def print_divider(char="-"):
    print(char * 100)

def main():
    input_dir = Path("/home/claire/Documents/Green_AI/final_results/random_Llama-3.2-3B-Instruct_gsm8k_20260320_012524")

    with open(input_dir / "optimization_results_corrected.json", "r") as f:
        all_trials = json.load(f)

    with open(input_dir / "pareto_frontier_corrected.json", "r") as f:
        pareto_trials = json.load(f)

    # Separate dominated trials
    pareto_names = {t.get("trial_name") for t in pareto_trials}
    dominated_trials = [t for t in all_trials if t.get("trial_name") not in pareto_names]

    # Configuration
    baseline = {
        "accuracy": 0.718,
        "latency": 1422.6073,
        "vram": 6.1951861328125,
        "emissions": 0.09366875,
    }

    print_header("PARETO FRONTIER ANALYSIS REPORT")
    print(f"\nDataset: GSM8K")
    print(f"Model: Llama-3.2-3B-Instruct")
    print(f"Baseline (Unquantized) Metrics:")
    print(f"  - Accuracy : {baseline['accuracy']:.3f}")
    print(f"  - Latency  : {baseline['latency']:.2f}s")
    print(f"  - VRAM     : {baseline['vram']:.2f}GB")
    print(f"  - Emissions: {baseline['emissions']:.4f}")

    print_header("SUMMARY")
    print(f"Total trials evaluated: {len(all_trials)}")
    print(f"Pareto frontier size: {len(pareto_trials)}")
    print(f"Dominated solutions: {len(dominated_trials)}")
    print(f"Pareto frontier coverage: {len(pareto_trials)/len(all_trials)*100:.1f}%")

    print_header("PARETO FRONTIER TRIALS (Ranked by Score)")
    print_divider()

    sorted_pareto = sorted(pareto_trials, key=lambda t: t["metrics"]["score"], reverse=True)

    print(f"\n{'Rank':<4} {'Trial Name':<35} {'Acc':<6} {'Lat(s)':<8} {'VRAM(GB)':<10} {'Emit':<8} {'Score':<8}")
    print_divider()

    for rank, trial in enumerate(sorted_pareto, 1):
        m = trial["metrics"]
        print(f"{rank:<4} {trial.get('trial_name', 'unknown'):<35} "
              f"{m['accuracy']:<6.3f} {m['latency']:<8.1f} {m['vram']:<10.2f} "
              f"{m['emissions']:<8.4f} {m['score']:<8.4f}")

    print_header("DOMINATED TRIALS (Ranked by Score)")
    print_divider()

    sorted_dominated = sorted(dominated_trials, key=lambda t: t["metrics"]["score"], reverse=True)

    print(f"\n{'Rank':<4} {'Trial Name':<35} {'Acc':<6} {'Lat(s)':<8} {'VRAM(GB)':<10} {'Emit':<8} {'Score':<8} {'Dominated By':<35}")
    print_divider()

    for rank, trial_d in enumerate(sorted_dominated, 1):
        m_d = trial_d["metrics"]

        # Find what dominates this trial
        dominated_by = None
        for trial_p in pareto_trials:
            m_p = trial_p["metrics"]
            if (m_p["accuracy"] >= m_d["accuracy"] and
                m_p["latency"] <= m_d["latency"] and
                m_p["vram"] <= m_d["vram"] and
                m_p["emissions"] <= m_d["emissions"]):
                if (m_p["accuracy"] > m_d["accuracy"] or
                    m_p["latency"] < m_d["latency"] or
                    m_p["vram"] < m_d["vram"] or
                    m_p["emissions"] < m_d["emissions"]):
                    dominated_by = trial_p.get("trial_name", "unknown")
                    break

        print(f"{rank:<4} {trial_d.get('trial_name', 'unknown'):<35} "
              f"{m_d['accuracy']:<6.3f} {m_d['latency']:<8.1f} {m_d['vram']:<10.2f} "
              f"{m_d['emissions']:<8.4f} {m_d['score']:<8.4f} {dominated_by if dominated_by else 'Multiple':<35}")

    print_header("PARETO FRONTIER STATISTICS")
    print_divider()

    pareto_metrics = {
        "Accuracy": [t["metrics"]["accuracy"] for t in pareto_trials],
        "Latency": [t["metrics"]["latency"] for t in pareto_trials],
        "VRAM": [t["metrics"]["vram"] for t in pareto_trials],
        "Emissions": [t["metrics"]["emissions"] for t in pareto_trials],
        "Score": [t["metrics"]["score"] for t in pareto_trials],
    }

    print(f"\n{'Metric':<15} {'Min':<12} {'Max':<12} {'Mean':<12} {'Baseline':<12} {'Change':<12}")
    print_divider()

    baseline_map = {
        "Accuracy": baseline["accuracy"],
        "Latency": baseline["latency"],
        "VRAM": baseline["vram"],
        "Emissions": baseline["emissions"],
        "Score": None,
    }

    for metric, values in pareto_metrics.items():
        min_val = min(values)
        max_val = max(values)
        mean_val = sum(values) / len(values)

        if baseline_map[metric] is None:
            print(f"{metric:<15} {min_val:<12.4f} {max_val:<12.4f} {mean_val:<12.4f} {'N/A':<12} {'N/A':<12}")
        else:
            baseline_val = baseline_map[metric]
            # For accuracy: higher is better
            if metric == "Accuracy":
                change = (mean_val - baseline_val) / baseline_val * 100
                print(f"{metric:<15} {min_val:<12.4f} {max_val:<12.4f} {mean_val:<12.4f} "
                      f"{baseline_val:<12.4f} {change:+.2f}%")
            # For others: lower is better
            else:
                change = (baseline_val - mean_val) / baseline_val * 100
                print(f"{metric:<15} {min_val:<12.4f} {max_val:<12.4f} {mean_val:<12.4f} "
                      f"{baseline_val:<12.4f} {change:+.2f}%")

    print_header("TRADEOFF INSIGHTS")
    print_divider()

    best_acc = max(pareto_trials, key=lambda t: t["metrics"]["accuracy"])
    best_lat = min(pareto_trials, key=lambda t: t["metrics"]["latency"])
    best_vram = min(pareto_trials, key=lambda t: t["metrics"]["vram"])
    best_emit = min(pareto_trials, key=lambda t: t["metrics"]["emissions"])
    best_score = max(pareto_trials, key=lambda t: t["metrics"]["score"])

    print(f"\nOptimal on Individual Metrics:")
    print(f"  - Best Accuracy : {best_acc.get('trial_name'):<35s} ({best_acc['metrics']['accuracy']:.3f})")
    print(f"  - Best Latency  : {best_lat.get('trial_name'):<35s} ({best_lat['metrics']['latency']:.1f}s)")
    print(f"  - Best VRAM     : {best_vram.get('trial_name'):<35s} ({best_vram['metrics']['vram']:.2f}GB)")
    print(f"  - Best Emissions: {best_emit.get('trial_name'):<35s} ({best_emit['metrics']['emissions']:.4f})")
    print(f"  - Best Overall  : {best_score.get('trial_name'):<35s} (Score: {best_score['metrics']['score']:.4f})")

    print_header("RECOMMENDED SOLUTIONS")
    print_divider()

    print(f"\n1. OVERALL BEST (Highest Score)")
    print(f"   Trial: {best_score.get('trial_name')}")
    m = best_score["metrics"]
    print(f"   Metrics: Acc={m['accuracy']:.3f}, Lat={m['latency']:.1f}s, VRAM={m['vram']:.2f}GB, Emit={m['emissions']:.4f}")
    print(f"   Improvements over baseline:")
    print(f"     - Accuracy: {(m['accuracy']/baseline['accuracy']-1)*100:.2f}%")
    print(f"     - Latency: {(baseline['latency']/m['latency']-1)*100:.2f}% faster")
    print(f"     - VRAM: {(baseline['vram']/m['vram']-1)*100:.2f}% less")
    print(f"     - Emissions: {(baseline['emissions']/m['emissions']-1)*100:.2f}% less")

    print(f"\n2. BEST FOR ACCURACY")
    print(f"   Trial: {best_acc.get('trial_name')}")
    m = best_acc["metrics"]
    print(f"   Metrics: Acc={m['accuracy']:.3f}, Lat={m['latency']:.1f}s, VRAM={m['vram']:.2f}GB, Emit={m['emissions']:.4f}")

    print(f"\n3. BEST FOR LATENCY (Fastest)")
    print(f"   Trial: {best_lat.get('trial_name')}")
    m = best_lat["metrics"]
    print(f"   Metrics: Acc={m['accuracy']:.3f}, Lat={m['latency']:.1f}s, VRAM={m['vram']:.2f}GB, Emit={m['emissions']:.4f}")
    print(f"   Speed improvement: {(baseline['latency']/m['latency']-1)*100:.2f}% faster")

    print(f"\n4. BEST FOR COMPRESSION (Minimal VRAM)")
    print(f"   Trial: {best_vram.get('trial_name')}")
    m = best_vram["metrics"]
    print(f"   Metrics: Acc={m['accuracy']:.3f}, Lat={m['latency']:.1f}s, VRAM={m['vram']:.2f}GB, Emit={m['emissions']:.4f}")
    print(f"   Memory reduction: {(baseline['vram']/m['vram']-1)*100:.2f}%")

    print("\n")


if __name__ == "__main__":
    main()
