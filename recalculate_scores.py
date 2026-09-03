#!/usr/bin/env python3
"""
Recalculate scores and Pareto frontier according to Systematic_Tuner rules.
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
    Identify Pareto frontier.
    A solution A dominates B if:
    - A is better or equal on all objectives, AND
    - A is strictly better on at least one objective

    Returns list of indices of non-dominated solutions.
    """
    # Objectives: accuracy (maximize), latency (minimize), vram (minimize), emissions (minimize)
    pareto_indices = []

    for i, trial_i in enumerate(trials):
        metrics_i = trial_i["metrics"]
        is_dominated = False

        for j, trial_j in enumerate(trials):
            if i == j:
                continue

            metrics_j = trial_j["metrics"]

            # Check if j dominates i
            acc_ok = metrics_j["accuracy"] >= metrics_i["accuracy"]
            lat_ok = metrics_j["latency"] <= metrics_i["latency"]
            vram_ok = metrics_j["vram"] <= metrics_i["vram"]
            emit_ok = metrics_j["emissions"] <= metrics_i["emissions"]

            # j dominates i if j is >= on all and > on at least one
            if acc_ok and lat_ok and vram_ok and emit_ok:
                if (metrics_j["accuracy"] > metrics_i["accuracy"] or
                    metrics_j["latency"] < metrics_i["latency"] or
                    metrics_j["vram"] < metrics_i["vram"] or
                    metrics_j["emissions"] < metrics_i["emissions"]):
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
    print("\nPareto frontier trials:")
    for idx in pareto_indices:
        trial = trials[idx]
        metrics = trial["metrics"]
        print(f"  {trial.get('trial_name', f'trial_{idx+1:03d}'):30s} | "
              f"acc={metrics['accuracy']:.3f} lat={metrics['latency']:.1f}s "
              f"vram={metrics['vram']:.2f}GB emit={metrics['emissions']:.4f} "
              f"score={metrics['score']:.4f}")

    # Save updated optimization_results.json
    output_path = output_dir / "optimization_results_recalculated.json"
    with open(output_path, "w") as f:
        json.dump(trials, f, indent=2)
    print(f"\nSaved recalculated results to: {output_path}")

    # Save pareto_frontier.json
    pareto_path = output_dir / "pareto_frontier_recalculated.json"
    with open(pareto_path, "w") as f:
        json.dump(pareto_trials, f, indent=2)
    print(f"Saved Pareto frontier to: {pareto_path}")

    # Save detailed score calculation report
    report_path = output_dir / "score_calculation_report.json"
    with open(report_path, "w") as f:
        json.dump(score_results, f, indent=2)
    print(f"Saved score calculation report to: {report_path}")

    # Print comparison of old vs new scores
    print("\n" + "="*80)
    print("Score Comparison (Old vs New):")
    print("="*80)
    for result in score_results:
        old = result["old_score"]
        new = result["new_score"]
        diff = new - old if old is not None else 0
        print(f"{result['trial_name']:30s} | Old: {old:8.4f} | New: {new:8.4f} | Diff: {diff:+8.4f}")


if __name__ == "__main__":
    main()
