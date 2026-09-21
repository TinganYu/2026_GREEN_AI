#!/usr/bin/env python3
"""
比較「去重複重試安全網開啟 vs 關閉」對四種 LLM 記憶策略（full/summary/window/tool）的影響。

背景：
  final_results_30/          orchestrator.py 的 _MAX_DUP_RETRIES = 5（有安全網）
                              LLM 建議重複配置時會重打 API 最多 5 次，直到給出新配置為止，
                              所以呈現出來的 30 個 trial 幾乎都是「有效」的。

  final_results_30_noretry/  _MAX_DUP_RETRIES = 1（無安全網，range(1) 只跑一次）
                              LLM 建議重複配置時直接把該 iteration 記成
                              "重複 config，已跳過"，不重打 API，30 個 trial 裡有一部分
                              是浪費掉的空 trial。

這支腳本量化「安全網被拿掉後，各模式實際浪費了多少 trial」，以及浪費之後
「真正產出新資訊的 trial」的分數品質有沒有跟著變化。

用法：
  python3 analyze_retry_ablation.py
"""
import json
import glob
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy import stats as sstats

ROOT = Path(__file__).resolve().parent

CONDITIONS = {
    "5-retry（有安全網）": ROOT / "final_results_30",
    "0-retry（無安全網）": ROOT / "final_results_30_noretry",
}

MODES = ["full", "summary", "window", "tool"]


def load_runs(base_dir: Path, mode: str) -> list:
    """回傳這個資料夾底下，某個模式的所有 run 的 trial 列表。"""
    runs = []
    for d in sorted(base_dir.glob(f"exp_Llama-3.2-3B-Instruct_gsm8k_{mode}_*")):
        p = d / "optimization_results.json"
        if p.exists():
            runs.append((d.name, json.load(open(p, encoding="utf-8"))))
    return runs


def is_skipped(trial: dict) -> bool:
    return trial.get("error") == "重複 config，已跳過"


def is_valid_scored(trial: dict) -> bool:
    m = trial.get("metrics") or {}
    return "score" in m and not is_skipped(trial)


def main():
    print("=" * 100)
    print("去重複重試機制消融分析：5-retry（有安全網）vs 0-retry（無安全網）")
    print("=" * 100)

    summary_rows = []
    trial_scores = defaultdict(dict)  # {mode: {condition: [scores]}}

    for mode in MODES:
        print(f"\n### 模式：{mode}")
        print(f"{'條件':22s} {'runs':4s} {'總trial':7s} {'跳過數':6s} {'跳過率':7s} "
              f"{'有效trial':9s} {'有效trial平均分':14s} {'正向率':7s} {'災難率':7s}")

        for cond_name, base_dir in CONDITIONS.items():
            runs = load_runs(base_dir, mode)
            if not runs:
                print(f"{cond_name:22s} 找不到資料")
                continue

            total_trials, total_skipped = 0, 0
            all_scores = []
            for _, trials in runs:
                total_trials += len(trials)
                total_skipped += sum(1 for t in trials if is_skipped(t))
                all_scores.extend(t["metrics"]["score"] for t in trials if is_valid_scored(t))

            n_valid = len(all_scores)
            skip_rate = total_skipped / total_trials if total_trials else 0
            mean_score = st.mean(all_scores) if all_scores else 0
            pos_rate = sum(1 for s in all_scores if s > 0) / n_valid if n_valid else 0
            cat_rate = sum(1 for s in all_scores if s < -10) / n_valid if n_valid else 0

            print(f"{cond_name:22s} {len(runs):4d} {total_trials:7d} {total_skipped:6d} "
                  f"{skip_rate:6.1%} {n_valid:9d} {mean_score:14.3f} {pos_rate:6.1%} {cat_rate:6.1%}")

            trial_scores[mode][cond_name] = all_scores
            summary_rows.append({
                "mode": mode, "condition": cond_name, "runs": len(runs),
                "total_trials": total_trials, "skipped": total_skipped,
                "skip_rate": skip_rate, "n_valid": n_valid,
                "mean_score": mean_score, "pos_rate": pos_rate, "cat_rate": cat_rate,
            })

        # 統計檢定：同一模式底下，兩種條件的「有效 trial 分數」分布是否不同
        conds = list(trial_scores[mode].keys())
        if len(conds) == 2 and trial_scores[mode][conds[0]] and trial_scores[mode][conds[1]]:
            u, p = sstats.mannwhitneyu(
                trial_scores[mode][conds[0]], trial_scores[mode][conds[1]], alternative="two-sided"
            )
            print(f"  → 有效 trial 分數分布比較（{conds[0]} vs {conds[1]}）："
                  f"Mann-Whitney U={u:.1f}, p={p:.4f}")

    # ── 總結：各模式的跳過率排名（核心結論）──────────────────────────
    print("\n" + "=" * 100)
    print("核心結論：各模式在「無安全網」條件下的 trial 浪費率排名")
    print("=" * 100)
    noretry_rows = [r for r in summary_rows if r["condition"] == "0-retry（無安全網）"]
    noretry_rows.sort(key=lambda r: r["skip_rate"])
    for rank, r in enumerate(noretry_rows, 1):
        print(f"  {rank}. {r['mode']:10s} 跳過率 {r['skip_rate']:.1%}"
              f"（{r['skipped']}/{r['total_trials']} trial 浪費在重複建議上）")

    out_path = ROOT / "retry_ablation_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    print(f"\n完整結果已存到 {out_path}")


if __name__ == "__main__":
    main()
