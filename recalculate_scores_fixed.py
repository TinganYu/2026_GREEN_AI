#!/usr/bin/env python3
"""
Recalculate scores and Pareto frontier according to Systematic_Tuner rules.
FIXED VERSION - Correct Pareto dominance logic.
"""
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Any

# Configuration parameters
BASELINE = {
    "accuracy": 0.718,
    "latency": 1422.6073,
    "vram": 6.1951861328125,
    "emissions": 0.09366875,
}

WEIGHTS = {
    "acc": 3.0,
    "lat": 1.0,
    "vram": 1.0,
    "emit": 1.0,
}

PENALTY_CONFIG = {
    "pen_t": 0.15,  # threshold
    "pen_a": 10.0,  # amplitude
}


def calculate_score(trial: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate score for a single trial using Systematic_Tuner rules."""
    metrics = trial["metrics"]

    avg_acc = metrics["accuracy"]
    avg_lat = metrics["latency"]
    max_vram = metrics["vram"]
    avg_emit = metrics["emissions"]

    # Normalized metrics (log scale)
    norm_acc = avg_acc / BASELINE["accuracy"]
    norm_lat = BASELINE["latency"] / avg_lat
    norm_vram = BASELINE["vram"] / max_vram
    norm_emit = BASELINE["emissions"] / avg_emit

    # Weighted score
    weight_score = 1.0 + (
        WEIGHTS["acc"] * math.log(norm_acc + 1e-9) +
        WEIGHTS["lat"] * math.log(norm_lat + 1e-9) +
        WEIGHTS["vram"] * math.log(norm_vram + 1e-9) +
        WEIGHTS["emit"] * math.log(norm_emit + 1e-9)
    )

    # Accuracy penalty
    penalty = PENALTY_CONFIG["pen_a"] * max(
        0, (BASELINE["accuracy"] - PENALTY_CONFIG["pen_t"]) - avg_acc
    )

    # Final score
    final_score = weight_score - penalty

    return {
        "normalized_metrics": {
            "norm_acc": norm_acc,
            "norm_lat": norm_lat,
            "norm_vram": norm_vram,
            "norm_emit": norm_emit,
        },
        "weight_score": weight_score,
        "penalty": penalty,
        "score": final_score,
    }


def identify_pareto_frontier(trials: List[Dict[str, Any]]) -> List[int]:
    """
    Identify Pareto frontier with CORRECT dominance logic.

    A solution A DOMINATES B if and only if:
    1. A >= B on all objectives (not strictly better - just >= or better)
    2. A > B on at least one objective (strictly better on at least one)

    Objectives: accuracy (maximize), latency (minimize), vram (minimize), emissions (minimize)

    Returns list of indices of non-dominated solutions.
    """
    pareto_indices = []
    n = len(trials)

    for i in range(n):
        metrics_i = trials[i]["metrics"]
        is_dominated = False

        # Check if any other trial dominates this one
        for j in range(n):
            if i == j:
                continue

            metrics_j = trials[j]["metrics"]

            # Check all objectives
            # For accuracy: higher is better (j >= i)
            acc_better_or_equal = metrics_j["accuracy"] >= metrics_i["accuracy"]
            # For latency: lower is better (j <= i)
            lat_better_or_equal = metrics_j["latency"] <= metrics_i["latency"]
            # For vram: lower is better (j <= i)
            vram_better_or_equal = metrics_j["vram"] <= metrics_i["vram"]
            # For emissions: lower is better (j <= i)
            emit_better_or_equal = metrics_j["emissions"] <= metrics_i["emissions"]

            # Check if j is strictly better on at least one
            acc_strictly_better = metrics_j["accuracy"] > metrics_i["accuracy"]
            lat_strictly_better = metrics_j["latency"] < metrics_i["latency"]
            vram_strictly_better = metrics_j["vram"] < metrics_i["vram"]
            emit_strictly_better = metrics_j["emissions"] < metrics_i["emissions"]

            # j dominates i if j is >= on all AND > on at least one
            if (acc_better_or_equal and lat_better_or_equal and
                vram_better_or_equal and emit_better_or_equal):
                if (acc_strictly_better or lat_strictly_better or
                    vram_strictly_better or emit_strictly_better):
                    is_dominated = True
                    break

        if not is_dominated:
            pareto_indices.append(i)

    return sorted(pareto_indices)


def main():
    input_path = Path("/home/claire/Documents/Green_AI/final_results/random_Llama-3.2-3B-Instruct_gsm8k_20260320_012524/optimization_results.json")
    output_dir = input_path.parent

    # Load data
    with open(input_path, "r") as f:
        trials = json.load(f)

    print(f"Loaded {len(trials)} trials")
    print(f"Baseline: acc={BASELINE['accuracy']}, lat={BASELINE['latency']:.2f}s, "
          f"vram={BASELINE['vram']:.2f}GB, emit={BASELINE['emissions']:.4f}")
    print(f"Weights: acc={WEIGHTS['acc']}, lat={WEIGHTS['lat']}, "
          f"vram={WEIGHTS['vram']}, emit={WEIGHTS['emit']}")
    print(f"Penalty: pen_t={PENALTY_CONFIG['pen_t']}, pen_a={PENALTY_CONFIG['pen_a']}")
    print()

    # Recalculate scores
    score_results = []
    for i, trial in enumerate(trials):
        score_info = calculate_score(trial)
        trial["metrics"]["score"] = score_info["score"]
        trial["metrics"]["calculation"] = {
            "normalized_metrics": score_info["normalized_metrics"],
            "weight_score": score_info["weight_score"],
            "penalty": score_info["penalty"],
        }
        score_results.append({
            "iteration": trial.get("iteration", i + 1),
            "trial_name": trial.get("trial_name", f"trial_{i+1:03d}"),
            "accuracy": trial["metrics"]["accuracy"],
            "latency": trial["metrics"]["latency"],
            "vram": trial["metrics"]["vram"],
            "emissions": trial["metrics"]["emissions"],
            "old_score": trial["metrics"].get("score"),  # Will be overwritten
            "new_score": score_info["score"],
            "calculation": score_info,
        })

    # Identify Pareto frontier
    pareto_indices = identify_pareto_frontier(trials)
    pareto_trials = [trials[i] for i in pareto_indices]

    print(f"Pareto frontier size: {len(pareto_indices)} / {len(trials)}")
    print("\nPareto frontier trials (sorted by score):")
    pareto_sorted = sorted(pareto_trials, key=lambda t: t["metrics"]["score"], reverse=True)
    for rank, trial in enumerate(pareto_sorted, 1):
        metrics = trial["metrics"]
        print(f"  {rank:2d}. {trial.get('trial_name', 'unknown'):30s} | "
              f"acc={metrics['accuracy']:.3f} lat={metrics['latency']:.1f}s "
              f"vram={metrics['vram']:.2f}GB emit={metrics['emissions']:.4f} "
              f"score={metrics['score']:.4f}")

    # Save updated optimization_results.json
    output_path = output_dir / "optimization_results_corrected.json"
    with open(output_path, "w") as f:
        json.dump(trials, f, indent=2)
    print(f"\nSaved corrected results to: {output_path}")

    # Save pareto_frontier.json
    pareto_path = output_dir / "pareto_frontier_corrected.json"
    with open(pareto_path, "w") as f:
        json.dump(pareto_trials, f, indent=2)
    print(f"Saved Pareto frontier to: {pareto_path}")

    # Save detailed score calculation report
    report_path = output_dir / "score_calculation_report_detailed.json"
    with open(report_path, "w") as f:
        json.dump(score_results, f, indent=2)
    print(f"Saved score calculation report to: {report_path}")


if __name__ == "__main__":
    main()
