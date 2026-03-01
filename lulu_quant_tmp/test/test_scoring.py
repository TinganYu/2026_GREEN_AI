"""
測試評分和分析模組
"""

import sys
import os

# 添加專案根目錄到路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from lulu_quant_tmp.agent.scoring import LayeredScorer, ParetoAnalyzer


def create_mock_config():
    """建立模擬配置"""
    return {
        'multiobjective': {
            'objectives': [
                {'name': 'accuracy_change', 'direction': 'maximize', 'weight': 2.0},
                {'name': 'gpu_peak_change', 'direction': 'minimize', 'weight': 1.0},
                {'name': 'latency_change', 'direction': 'minimize', 'weight': 1.0}
            ],
            'targets': {
                'accuracy_min': -0.10,
                'gpu_peak_max': -0.50,
                'latency_max': 1.5
            }
        }
    }


def create_mock_trials():
    """建立模擬試驗資料"""
    # 建立滿足目標的試驗
    satisfying_trials = [
        {
            'trial_id': 1,
            'config': {'method': 'gptq', 'bits': 4},
            'objectives': {
                'accuracy_change': -0.08,  # 滿足 >= -0.10
                'gpu_peak_change': -0.55,  # 滿足 <= -0.50
                'latency_change': 0.5      # 滿足 <= 1.5
            },
            'satisfies_targets': True,
            'results': {
                'per_dataset': {'gsm8k': {'accuracy': 0.23}},
                'aggregated': {'accuracy': 0.23}
            }
        },
        {
            'trial_id': 2,
            'config': {'method': 'awq', 'bits': 4},
            'objectives': {
                'accuracy_change': -0.09,
                'gpu_peak_change': -0.52,
                'latency_change': 0.8
            },
            'satisfies_targets': True,
            'results': {
                'per_dataset': {'gsm8k': {'accuracy': 0.22}},
                'aggregated': {'accuracy': 0.22}
            }
        },
        {
            'trial_id': 3,
            'config': {'method': 'bnb', 'bits': 4},
            'objectives': {
                'accuracy_change': -0.05,
                'gpu_peak_change': -0.48,
                'latency_change': 1.2
            },
            'satisfies_targets': True,
            'results': {
                'per_dataset': {'gsm8k': {'accuracy': 0.24}},
                'aggregated': {'accuracy': 0.24}
            }
        }
    ]

    # 建立 Pareto 前沿（包括滿足和不滿足目標的）
    pareto_trials = satisfying_trials + [
        {
            'trial_id': 4,
            'config': {'method': 'gptq', 'bits': 2},
            'objectives': {
                'accuracy_change': -0.15,  # 不滿足
                'gpu_peak_change': -0.65,  # 滿足
                'latency_change': 0.3      # 滿足
            },
            'satisfies_targets': False,
            'results': {
                'per_dataset': {'gsm8k': {'accuracy': 0.21}},
                'aggregated': {'accuracy': 0.21}
            }
        },
        {
            'trial_id': 5,
            'config': {'method': 'awq', 'bits': 3},
            'objectives': {
                'accuracy_change': -0.12,  # 不滿足
                'gpu_peak_change': -0.60,
                'latency_change': 0.7
            },
            'satisfies_targets': False,
            'results': {
                'per_dataset': {'gsm8k': {'accuracy': 0.22}},
                'aggregated': {'accuracy': 0.22}
            }
        }
    ]

    return satisfying_trials, pareto_trials


def test_layered_scoring():
    """測試 LayeredScorer"""
    print("=" * 60)
    print("測試 LayeredScorer（分層評分系統）")
    print("=" * 60)

    config = create_mock_config()
    satisfying_trials, pareto_trials = create_mock_trials()

    # 建立評分器
    scorer = LayeredScorer(config)

    # 執行評分
    results = scorer.score_satisfying_trials(satisfying_trials, pareto_trials)

    # 驗證輸出格式
    assert 'scoring_method' in results
    assert results['scoring_method'] == 'layered_scoring'
    assert 'ranked_trials' in results
    assert 'statistics' in results
    assert 'total_satisfying_trials' in results
    assert results['total_satisfying_trials'] == 3

    print(f"✓ 評分方法：{results['scoring_method']}")
    print(f"✓ 滿足目標的試驗數量：{results['total_satisfying_trials']}")

    # 驗證排名
    ranked = results['ranked_trials']
    assert len(ranked) == 3
    assert all('rank' in t for t in ranked)
    assert all('overall_score' in t for t in ranked)
    assert all('grade' in t for t in ranked)
    assert all('dimension_scores' in t for t in ranked)
    assert all('strategy_rankings' in t for t in ranked)

    print(f"✓ 排名試驗數量：{len(ranked)}")
    print(f"✓ 最高分：{ranked[0]['overall_score']} (評級：{ranked[0]['grade']})")
    print(f"✓ 最低分：{ranked[-1]['overall_score']} (評級：{ranked[-1]['grade']})")

    # 驗證單維度評分
    dim_scores = ranked[0]['dimension_scores']
    assert 'accuracy' in dim_scores
    assert 'gpu_peak' in dim_scores
    assert 'latency' in dim_scores

    for dim in ['accuracy', 'gpu_peak', 'latency']:
        assert 'value' in dim_scores[dim]
        assert 'target' in dim_scores[dim]
        assert 'score' in dim_scores[dim]
        assert 'grade' in dim_scores[dim]
        assert 'status' in dim_scores[dim]
        assert 'interpretation' in dim_scores[dim]

    print(f"✓ 單維度評分：")
    print(f"  - 準確率：{dim_scores['accuracy']['score']} ({dim_scores['accuracy']['grade']})")
    print(f"  - GPU 峰值：{dim_scores['gpu_peak']['score']} ({dim_scores['gpu_peak']['grade']})")
    print(f"  - 延遲：{dim_scores['latency']['score']} ({dim_scores['latency']['grade']})")

    # 驗證策略排名
    strat_rankings = ranked[0]['strategy_rankings']
    assert 'closest_to_target' in strat_rankings
    assert 'best_accuracy' in strat_rankings
    assert 'best_compression' in strat_rankings
    assert 'balanced' in strat_rankings

    print(f"✓ 策略排名：")
    for strategy, info in strat_rankings.items():
        print(f"  - {strategy}: 排名 {info['rank']}/{info['total']}")

    # 驗證統計資訊
    stats = results['statistics']
    assert 'score_distribution' in stats
    assert 'avg_overall_score' in stats
    assert 'median_overall_score' in stats

    print(f"✓ 統計資訊：")
    print(f"  - 平均分：{stats['avg_overall_score']}")
    print(f"  - 中位數：{stats['median_overall_score']}")
    print(f"  - 分數分佈：{stats['score_distribution']}")

    print("\n✅ LayeredScorer 測試通過！\n")


def test_pareto_analysis():
    """測試 ParetoAnalyzer"""
    print("=" * 60)
    print("測試 ParetoAnalyzer（Pareto 前沿深度分析）")
    print("=" * 60)

    config = create_mock_config()
    satisfying_trials, pareto_trials = create_mock_trials()

    # 先執行評分（為了傳入分析器）
    scorer = LayeredScorer(config)
    layered_scores = scorer.score_satisfying_trials(satisfying_trials, pareto_trials)

    # 建立分析器
    analyzer = ParetoAnalyzer(config)

    # 執行分析
    results = analyzer.analyze_pareto_frontier(
        pareto_trials,
        satisfying_trials,
        layered_scores
    )

    # 驗證輸出格式
    assert 'analysis_method' in results
    assert results['analysis_method'] == 'pareto_frontier_deep_analysis'
    assert 'pareto_statistics' in results
    assert 'satisfying_trials_summary' in results
    assert 'pareto_clusters' in results
    assert 'strategy_comparison' in results
    assert 'tradeoff_analysis' in results
    assert 'recommendations' in results

    print(f"✓ 分析方法：{results['analysis_method']}")

    # 驗證統計資訊
    stats = results['pareto_statistics']
    assert 'total_pareto_solutions' in stats
    assert stats['total_pareto_solutions'] == 5
    assert stats['satisfying_targets'] == 3

    print(f"✓ Pareto 統計：")
    print(f"  - 總解決方案：{stats['total_pareto_solutions']}")
    print(f"  - 滿足目標：{stats['satisfying_targets']}")
    print(f"  - 接近目標：{stats['close_to_targets']}")
    print(f"  - 違反目標：{stats['violating_targets']}")

    # 驗證聚類（可能會跳過，如果點太少或沒有 sklearn）
    if results['pareto_clusters']:
        print(f"✓ 聚類分析：{len(results['pareto_clusters'])} 個聚類")
        for cluster_id, cluster_info in results['pareto_clusters'].items():
            print(f"  - {cluster_id}: {cluster_info['label']} (大小：{cluster_info['size']})")
            print(f"    特徵：{cluster_info['characteristics']}")
    else:
        print(f"⚠ 聚類分析已跳過（點太少或 scikit-learn 未安裝）")

    # 驗證策略對比
    strat_comp = results['strategy_comparison']
    assert 'closest_to_target' in strat_comp or 'best_accuracy' in strat_comp

    print(f"✓ 策略對比：")
    for strategy, info in strat_comp.items():
        print(f"  - {strategy}:")
        print(f"    推薦試驗 ID：{info['recommended_trial_id']}")
        print(f"    理由：{info['reason']}")
        if 'warning' in info:
            print(f"    警告：{info['warning']}")

    # 驗證權衡分析
    tradeoff = results['tradeoff_analysis']
    assert 'correlations' in tradeoff
    assert 'key_insights' in tradeoff

    print(f"✓ 權衡分析：")
    if tradeoff['correlations']:
        for corr_name, corr_info in tradeoff['correlations'].items():
            print(f"  - {corr_name}: r={corr_info['pearson_r']} ({corr_info['interpretation']})")
    print(f"✓ 關鍵洞察：")
    for insight in tradeoff['key_insights']:
        print(f"  - {insight}")

    # 驗證推薦
    recs = results['recommendations']
    assert len(recs) > 0

    print(f"✓ 推薦建議：")
    for rec_name, rec_info in recs.items():
        print(f"  - {rec_name}:")
        print(f"    推薦試驗：{rec_info['recommended_trial']}")
        print(f"    理由：{rec_info['reason']}")
        if 'warning' in rec_info:
            print(f"    警告：{rec_info['warning']}")

    print("\n✅ ParetoAnalyzer 測試通過！\n")


def test_edge_cases():
    """測試邊界情況"""
    print("=" * 60)
    print("測試邊界情況")
    print("=" * 60)

    config = create_mock_config()

    # 測試 1：空的滿足目標試驗
    print("\n1. 測試空的滿足目標試驗...")
    scorer = LayeredScorer(config)
    results = scorer.score_satisfying_trials([], [])
    assert results['total_satisfying_trials'] == 0
    assert results['ranked_trials'] == []
    print("✓ 空試驗處理正確")

    # 測試 2：空的 Pareto 前沿
    print("\n2. 測試空的 Pareto 前沿...")
    analyzer = ParetoAnalyzer(config)
    results = analyzer.analyze_pareto_frontier([], [], None)
    assert results['pareto_statistics']['total_pareto_solutions'] == 0
    print("✓ 空 Pareto 前沿處理正確")

    # 測試 3：單個試驗
    print("\n3. 測試單個試驗...")
    single_trial = [{
        'trial_id': 1,
        'config': {'method': 'gptq', 'bits': 4},
        'objectives': {
            'accuracy_change': -0.08,
            'gpu_peak_change': -0.55,
            'latency_change': 0.5
        },
        'satisfies_targets': True,
        'results': {
            'per_dataset': {},
            'aggregated': {}
        }
    }]
    results = scorer.score_satisfying_trials(single_trial, single_trial)
    assert results['total_satisfying_trials'] == 1
    assert results['ranked_trials'][0]['rank'] == 1
    print("✓ 單個試驗處理正確")

    print("\n✅ 邊界情況測試通過！\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("開始測試評分和分析模組")
    print("=" * 60 + "\n")

    try:
        # 執行測試
        test_layered_scoring()
        test_pareto_analysis()
        test_edge_cases()

        print("=" * 60)
        print("🎉 所有測試通過！")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ 測試失敗：{e}")
        import traceback
        traceback.print_exc()
        exit(1)

    except Exception as e:
        print(f"\n❌ 測試出錯：{e}")
        import traceback
        traceback.print_exc()
        exit(1)
