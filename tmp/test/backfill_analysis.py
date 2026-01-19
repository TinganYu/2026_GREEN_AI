"""
為歷史實驗結果補全分析檔案

該腳本讀取現有優化實驗結果，使用 LayeredScorer 和 ParetoAnalyzer
為每個實驗生成 satisfying_trials_scored.json 和 pareto_deep_analysis.json。
"""

import sys
import os
import json
import argparse
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tmp.agent.scoring import LayeredScorer, ParetoAnalyzer


def compute_pareto_frontier(trials: list) -> list:
    """
    計算 Pareto 前沿（從 BaseOptimizer.get_pareto_frontier 提取的邏輯）

    Args:
        trials: 試驗字典列表

    Returns:
        Pareto 最優試驗列表（每個試驗包含 trial_id）
    """
    # 過濾成功的試驗，並記錄原始索引作為 trial_id
    valid_trials = []
    for i, t in enumerate(trials):
        # 檢查是否有有效的 objectives
        obj = t.get('objectives')
        if not obj:
            continue
        # 確保所有目標值都不是 None
        if obj.get('accuracy_change') is None or \
           obj.get('gpu_peak_change') is None or \
           obj.get('latency_change') is None:
            continue
        t['trial_id'] = i
        valid_trials.append(t)

    if not valid_trials:
        return []

    pareto_frontier = []

    for trial in valid_trials:
        is_dominated = False
        obj = trial['objectives']

        for other_trial in valid_trials:
            if trial == other_trial:
                continue

            other_obj = other_trial['objectives']

            # 檢查 other_trial 是否支配 trial
            better_accuracy = other_obj['accuracy_change'] > obj['accuracy_change']
            better_gpu = other_obj['gpu_peak_change'] < obj['gpu_peak_change']
            better_latency = other_obj['latency_change'] < obj['latency_change']

            equal_accuracy = abs(other_obj['accuracy_change'] - obj['accuracy_change']) < 1e-9
            equal_gpu = abs(other_obj['gpu_peak_change'] - obj['gpu_peak_change']) < 1e-9
            equal_latency = abs(other_obj['latency_change'] - obj['latency_change']) < 1e-9

            dominates_accuracy = better_accuracy or equal_accuracy
            dominates_gpu = better_gpu or equal_gpu
            dominates_latency = better_latency or equal_latency

            strictly_better = better_accuracy or better_gpu or better_latency

            if dominates_accuracy and dominates_gpu and dominates_latency and strictly_better:
                is_dominated = True
                break

        if not is_dominated:
            pareto_frontier.append(trial)

    return pareto_frontier


def load_experiment_data(exp_dir: Path) -> dict:
    """
    從實驗目錄載入資料

    Args:
        exp_dir: 實驗目錄路徑

    Returns:
        包含 config, baseline, satisfying_trials, pareto_frontier 的字典
    """
    print(f"  讀取配置和結果檔案...")

    # 1. 讀取配置
    config_file = exp_dir / 'config.json'
    if not config_file.exists():
        raise FileNotFoundError(f"找不到 config.json")

    with open(config_file, 'r') as f:
        config = json.load(f)

    # 2. 讀取完整結果
    full_results_file = exp_dir / 'full_results.json'
    if not full_results_file.exists():
        raise FileNotFoundError(f"找不到 full_results.json")

    with open(full_results_file, 'r') as f:
        full_results = json.load(f)

    # 3. 提取資料
    baseline = full_results.get('baseline', {})
    all_trials = full_results.get('all_trials', [])
    pareto_frontier = full_results.get('pareto_frontier', [])

    # 4. 建立 config -> trial_id 的映射
    config_to_trial_id = {}
    for i, t in enumerate(all_trials):
        config_key = str(t.get('config', {}))
        config_to_trial_id[config_key] = i

    # 5. 過濾滿足目標的試驗
    # 注意：試驗可能沒有 'status' 欄位（舊版本），需要相容處理
    satisfying_trials = []
    for i, t in enumerate(all_trials):
        # 檢查是否滿足目標
        if not t.get('satisfies_targets', False):
            continue

        # 檢查狀態（如果有的話）
        status = t.get('status')
        if status and status != 'completed':
            continue

        # 確保有 objectives（跳過失敗的試驗）
        if not t.get('objectives'):
            continue

        # 保留原始試驗索引作為 trial_id
        t['trial_id'] = i
        satisfying_trials.append(t)

    # 6. 為 Pareto 前沿資料添加 trial_id
    for p in pareto_frontier:
        config_key = str(p.get('config', {}))
        p['trial_id'] = config_to_trial_id.get(config_key, None)

    print(f"  找到 {len(all_trials)} 個試驗，其中 {len(satisfying_trials)} 個滿足目標")
    print(f"  Pareto 前沿包含 {len(pareto_frontier)} 個解")

    return {
        'config': config,
        'baseline': baseline,
        'satisfying_trials': satisfying_trials,
        'pareto_frontier': pareto_frontier,
        'all_trials': all_trials
    }


def generate_analysis_files(exp_dir: Path, data: dict, regen_pareto: bool = False) -> None:
    """
    生成分析檔案

    Args:
        exp_dir: 實驗目錄路徑
        data: 從 load_experiment_data 返回的資料
        regen_pareto: 是否重新生成 pareto_frontier.json
    """
    config = data['config']
    all_trials = data['all_trials']
    satisfying_trials = data['satisfying_trials']
    pareto_frontier = data['pareto_frontier']

    # 0. 重新生成 Pareto 前沿（如果需要）
    if regen_pareto:
        try:
            print(f"  正在重新計算 Pareto 前沿...")
            pareto_frontier = compute_pareto_frontier(all_trials)

            # 更新 satisfying_trials 的 trial_id（因為 all_trials 已經被更新）
            for t in satisfying_trials:
                if 'trial_id' not in t:
                    # 從 all_trials 中找到對應的 trial_id
                    config_key = str(t.get('config', {}))
                    for at in all_trials:
                        if str(at.get('config', {})) == config_key:
                            t['trial_id'] = at.get('trial_id')
                            break

            output_file = exp_dir / 'pareto_frontier.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(pareto_frontier, f, indent=2, ensure_ascii=False)

            print(f"  ✓ 重新生成：pareto_frontier.json（{len(pareto_frontier)} 個解）")
        except Exception as e:
            print(f"  ❌ Pareto 前沿重新計算失敗：{e}")
            import traceback
            traceback.print_exc()

    # 1. 生成滿足目標的配置評分
    satisfying_scored = None
    if satisfying_trials:
        try:
            print(f"  正在評分滿足目標的配置...")
            scorer = LayeredScorer(config)
            satisfying_scored = scorer.score_satisfying_trials(
                satisfying_trials,
                pareto_frontier
            )

            output_file = exp_dir / 'satisfying_trials_scored.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(satisfying_scored, f, indent=2, ensure_ascii=False)

            print(f"  ✓ 生成：satisfying_trials_scored.json")
        except Exception as e:
            print(f"  ❌ 評分失敗：{e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ⚠ 跳過評分（沒有滿足目標的試驗）")

    # 2. 生成 Pareto 前沿深度分析
    if pareto_frontier:
        try:
            print(f"  正在進行 Pareto 前沿深度分析...")
            analyzer = ParetoAnalyzer(config)
            pareto_analysis = analyzer.analyze_pareto_frontier(
                pareto_frontier,
                satisfying_trials,
                satisfying_scored
            )

            output_file = exp_dir / 'pareto_deep_analysis.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(pareto_analysis, f, indent=2, ensure_ascii=False)

            print(f"  ✓ 生成：pareto_deep_analysis.json")
        except Exception as e:
            print(f"  ❌ Pareto 分析失敗：{e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ⚠ 跳過分析（Pareto 前沿為空）")


def backfill_experiments(base_dirs: list, force: bool = False, single_dir: str = None,
                         regen_pareto: bool = False) -> None:
    """
    補全所有實驗的分析檔案

    Args:
        base_dirs: 實驗結果根目錄列表
        force: 是否強制重新生成（覆蓋已有檔案）
        single_dir: 如果指定，只處理這一個目錄
        regen_pareto: 是否重新生成 pareto_frontier.json
    """
    total_experiments = 0
    processed_experiments = 0
    skipped_experiments = 0
    failed_experiments = 0

    # 如果指定了單個目錄，直接處理
    if single_dir:
        exp_dir = Path(single_dir)
        if not exp_dir.is_dir():
            print(f"❌ 錯誤：{single_dir} 不是有效目錄")
            return

        base_dirs = [exp_dir.parent]
        exp_dirs_to_process = [exp_dir]
    else:
        # 掃描所有實驗目錄
        exp_dirs_to_process = []
        for base_dir in base_dirs:
            base_path = Path(base_dir)
            if not base_path.exists():
                print(f"⚠ 警告：目錄不存在：{base_dir}")
                continue

            # 獲取所有子目錄
            subdirs = [d for d in base_path.iterdir() if d.is_dir()]
            exp_dirs_to_process.extend(subdirs)

    total_experiments = len(exp_dirs_to_process)
    print(f"\n找到 {total_experiments} 個實驗目錄")
    print("=" * 80)

    for exp_dir in sorted(exp_dirs_to_process):
        print(f"\n處理實驗：{exp_dir.name}")
        print("-" * 80)

        # 檢查是否需要跳過
        scored_file = exp_dir / 'satisfying_trials_scored.json'
        analysis_file = exp_dir / 'pareto_deep_analysis.json'

        if not force and scored_file.exists() and analysis_file.exists():
            print(f"  ✓ 分析檔案已存在，跳過")
            skipped_experiments += 1
            continue

        # 載入資料
        try:
            data = load_experiment_data(exp_dir)
        except Exception as e:
            print(f"  ❌ 載入資料失敗：{e}")
            failed_experiments += 1
            continue

        # 生成分析檔案
        try:
            generate_analysis_files(exp_dir, data, regen_pareto=regen_pareto)
            processed_experiments += 1
            print(f"  ✅ 完成")
        except Exception as e:
            print(f"  ❌ 生成分析失敗：{e}")
            import traceback
            traceback.print_exc()
            failed_experiments += 1
            continue

    # 總結
    print("\n" + "=" * 80)
    print("補全完成！")
    print("=" * 80)
    print(f"總實驗數：{total_experiments}")
    print(f"已處理：{processed_experiments}")
    print(f"跳過：{skipped_experiments}")
    print(f"失敗：{failed_experiments}")
    print("=" * 80)


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='為歷史優化實驗結果補全分析檔案',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 處理所有實驗（預設）
  python tmp/test/backfill_analysis.py

  # 指定特定目錄
  python tmp/test/backfill_analysis.py --dirs results/optimization

  # 處理單個實驗目錄
  python tmp/test/backfill_analysis.py --single results/optimization/quick-mo-test_20251127_165523

  # 強制重新生成（覆蓋已有檔案）
  python tmp/test/backfill_analysis.py --force

  # 重新生成 pareto_frontier.json（添加 trial_id）
  python tmp/test/backfill_analysis.py --regen-pareto --force

  # 處理 LLM 優化實驗
  python tmp/test/backfill_analysis.py --dirs results/llm_optimization
        """
    )

    parser.add_argument(
        '--dirs',
        nargs='+',
        default=['results/optimization', 'results/llm_optimization'],
        help='實驗結果根目錄（預設：results/optimization 和 results/llm_optimization）'
    )

    parser.add_argument(
        '--single',
        type=str,
        help='只處理指定的單個實驗目錄（例如：results/optimization/quick-mo-test_20251127_165523）'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='強制重新生成，覆蓋已有的分析檔案'
    )

    parser.add_argument(
        '--regen-pareto',
        action='store_true',
        help='重新計算並輸出 pareto_frontier.json（添加 trial_id）'
    )

    args = parser.parse_args()

    # 列印配置
    print("\n" + "=" * 80)
    print("為歷史實驗結果補全分析檔案")
    print("=" * 80)

    if args.single:
        print(f"單個目錄模式：{args.single}")
    else:
        print(f"目標目錄：{', '.join(args.dirs)}")

    print(f"強制重新生成：{'是' if args.force else '否'}")
    print(f"重新生成 Pareto 前沿：{'是' if args.regen_pareto else '否'}")

    # 執行補全
    backfill_experiments(args.dirs, force=args.force, single_dir=args.single,
                         regen_pareto=args.regen_pareto)


if __name__ == '__main__':
    main()
