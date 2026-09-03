#!/usr/bin/env python3
"""
Detailed Pareto frontier analysis and visualization.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import math

def analyze_pareto_frontier():
    """Analyze and visualize the Pareto frontier."""

    input_dir = Path("/home/claire/Documents/Green_AI/final_results/random_Llama-3.2-3B-Instruct_gsm8k_20260320_012524")

    # Load data
    with open(input_dir / "optimization_results_recalculated.json", "r") as f:
        all_trials = json.load(f)

    with open(input_dir / "pareto_frontier_recalculated.json", "r") as f:
        pareto_trials = json.load(f)

    print("="*100)
    print("PARETO FRONTIER ANALYSIS")
    print("="*100)
    print()

    # Identify Pareto indices
    pareto_names = {t.get("trial_name") for t in pareto_trials}

    # Separate dominated from non-dominated
    dominated_trials = [t for t in all_trials if t.get("trial_name") not in pareto_names]

    print(f"Total Trials: {len(all_trials)}")
    print(f"Pareto Frontier: {len(pareto_trials)} trials")
    print(f"Dominated: {len(dominated_trials)} trials")
    print()

    # Print Pareto frontier sorted by score
    print("PARETO FRONTIER (sorted by score):")
    print("-" * 100)
    sorted_pareto = sorted(pareto_trials, key=lambda t: t["metrics"]["score"], reverse=True)

    for rank, trial in enumerate(sorted_pareto, 1):
        m = trial["metrics"]
        print(f"{rank:2d}. {trial.get('trial_name', 'unknown'):35s} | "
              f"acc={m['accuracy']:.3f} lat={m['latency']:7.1f}s vram={m['vram']:4.2f}GB "
              f"emit={m['emissions']:.4f} score={m['score']:7.4f}")

    print()
    print("DOMINATED TRIALS (sorted by score):")
    print("-" * 100)
    sorted_dominated = sorted(dominated_trials, key=lambda t: t["metrics"]["score"], reverse=True)

    for rank, trial in enumerate(sorted_dominated, 1):
        m = trial["metrics"]
        print(f"{rank:2d}. {trial.get('trial_name', 'unknown'):35s} | "
              f"acc={m['accuracy']:.3f} lat={m['latency']:7.1f}s vram={m['vram']:4.2f}GB "
              f"emit={m['emissions']:.4f} score={m['score']:7.4f}")

    print()

    # Verify no domination within Pareto frontier
    print("VERIFICATION: Checking for dominance within Pareto frontier...")
    print("-" * 100)

    errors = []
    for i, trial_a in enumerate(pareto_trials):
        for j, trial_b in enumerate(pareto_trials):
            if i >= j:
                continue

            m_a = trial_a["metrics"]
            m_b = trial_b["metrics"]

            # Check if a dominates b
            acc_ok = m_a["accuracy"] >= m_b["accuracy"]
            lat_ok = m_a["latency"] <= m_b["latency"]
            vram_ok = m_a["vram"] <= m_b["vram"]
            emit_ok = m_a["emissions"] <= m_b["emissions"]

            if acc_ok and lat_ok and vram_ok and emit_ok:
                if (m_a["accuracy"] > m_b["accuracy"] or
                    m_a["latency"] < m_b["latency"] or
                    m_a["vram"] < m_b["vram"] or
                    m_a["emissions"] < m_b["emissions"]):
                    errors.append(f"ERROR: {trial_a.get('trial_name')} dominates {trial_b.get('trial_name')}")

    if errors:
        for error in errors:
            print(error)
    else:
        print("PASS: No dominance detected within Pareto frontier")

    print()

    # Statistics
    print("PARETO FRONTIER STATISTICS:")
    print("-" * 100)

    pareto_metrics = {
        "accuracy": [t["metrics"]["accuracy"] for t in pareto_trials],
        "latency": [t["metrics"]["latency"] for t in pareto_trials],
        "vram": [t["metrics"]["vram"] for t in pareto_trials],
        "emissions": [t["metrics"]["emissions"] for t in pareto_trials],
        "score": [t["metrics"]["score"] for t in pareto_trials],
    }

    baseline = {
        "accuracy": 0.718,
        "latency": 1422.6073,
        "vram": 6.1951861328125,
        "emissions": 0.09366875,
    }

    for metric, values in pareto_metrics.items():
        if metric == "score":
            print(f"  {metric:12s} | Min: {min(values):8.4f} | Max: {max(values):8.4f} | "
                  f"Mean: {sum(values)/len(values):8.4f}")
        else:
            mean_val = sum(values) / len(values)
            improvement = ((baseline[metric] - mean_val) / baseline[metric] * 100) if metric != "accuracy" else ((mean_val - baseline[metric]) / baseline[metric] * 100)

            print(f"  {metric:12s} | Min: {min(values):8.4f} | Max: {max(values):8.4f} | "
                  f"Mean: {mean_val:8.4f} (baseline: {baseline[metric]:8.4f}, "
                  f"change: {improvement:+6.2f}%)")

    print()

    # Tradeoff analysis
    print("TRADEOFF ANALYSIS:")
    print("-" * 100)

    # Best in each dimension
    best_acc = max(pareto_trials, key=lambda t: t["metrics"]["accuracy"])
    best_lat = min(pareto_trials, key=lambda t: t["metrics"]["latency"])
    best_vram = min(pareto_trials, key=lambda t: t["metrics"]["vram"])
    best_emit = min(pareto_trials, key=lambda t: t["metrics"]["emissions"])
    best_score = max(pareto_trials, key=lambda t: t["metrics"]["score"])

    print(f"Best Accuracy : {best_acc.get('trial_name'):35s} ({best_acc['metrics']['accuracy']:.3f})")
    print(f"Best Latency  : {best_lat.get('trial_name'):35s} ({best_lat['metrics']['latency']:7.1f}s)")
    print(f"Best VRAM     : {best_vram.get('trial_name'):35s} ({best_vram['metrics']['vram']:4.2f}GB)")
    print(f"Best Emissions: {best_emit.get('trial_name'):35s} ({best_emit['metrics']['emissions']:.4f})")
    print(f"Best Score    : {best_score.get('trial_name'):35s} ({best_score['metrics']['score']:7.4f})")

    print()


if __name__ == "__main__":
    analyze_pareto_frontier()
