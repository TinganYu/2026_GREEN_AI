"""
測試多目標優化

用於執行多目標量化優化的入口程式。

用法：
    # 完整優化
    python tmp/test/test_optimization.py --config tmp/config/optimization_config.yaml

    # 快速測試
    python tmp/test/test_optimization.py --config tmp/config/optimization_config_quick.yaml
"""

import sys
import os
import argparse
import logging

# 載入 .env 環境變量
from dotenv import load_dotenv
load_dotenv()

# 將專案根目錄加入模組搜尋路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent import OptimizationOrchestrator, ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("Main")


def main():
    """主要進入點"""
    parser = argparse.ArgumentParser(
        description='Multi-objective Quantization Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full optimization
  python tmp/test/test_optimization.py --config tmp/config/optimization_config.yaml

  # Run quick test (15-20 min)
  python tmp/test/test_optimization.py --config tmp/config/optimization_config_quick.yaml

  # Use default config
  python tmp/test/test_optimization.py
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='tmp/config/optimization_config.yaml',
        help='Path to optimization configuration file (default: optimization_config.yaml)'
    )

    args = parser.parse_args()

    # 檢查配置檔是否存在
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return 1

    logger.info("="*70)
    logger.info("MULTI-OBJECTIVE QUANTIZATION OPTIMIZATION")
    logger.info("="*70)
    logger.info(f"Configuration: {args.config}")
    logger.info("="*70)

    try:
        # 創建協調器 (orchestrator)
        logger.info("Loading configuration...")
        orchestrator = OptimizationOrchestrator.from_config_file(args.config)

        # 開始執行優化流程
        logger.info("Starting optimization workflow...\n")
        results = orchestrator.run()

        # Print summary
        print_summary(results)

        logger.info("\n✓ Optimization completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("\n\nOptimization interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\n\n✗ Optimization failed: {e}", exc_info=True)
        return 1


def print_summary(results: dict):
    """列印結果摘要"""
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*70)

    # 基線 (Baseline)
    baseline = results['baseline_results']['aggregated']
    print("\nBaseline Model:")
    print(f"  Model: {results['baseline_results']['model_name']}")
    print(f"  Accuracy: {baseline['accuracy']:.4f}")
    print(f"  GPU Peak: {baseline['gpu_peak_mb']:.1f} MB")
    print(f"  Avg Latency: {baseline['avg_latency_ms']:.1f} ms")

    # 優化摘要
    print("\nOptimization Results:")
    print(f"  Total Trials: {results['n_total_trials']}")
    print(f"  Pareto Solutions: {results['n_pareto_solutions']}")
    print(f"  Solutions Satisfying Targets: {results['n_satisfying_solutions']}")
    print(f"  Total Time: {results['total_time_sec']/60:.1f} minutes")

    # 推薦配置
    if results['recommended_config'] and results['recommended_config'].get('config'):
        rec = results['recommended_config']
        obj = rec.get('objectives', {})

        print("\nRecommended Configuration:")
        print(f"  Method: {rec['config'].get('method', 'unknown').upper()}")
        print(f"  Config: {rec['config']}")

        # 檢查 objectives 是否有效（不是 null 或空）
        if obj and all(v is not None for v in obj.values()):
            print(f"\n  Objectives:")
            print(f"    Accuracy Change: {obj['accuracy_change']:+.2%} (target: ≥-10%)")
            print(f"    GPU Peak Change: {obj['gpu_peak_change']:+.2%} (target: ≤-20%)")
            print(f"    Latency Change: {obj['latency_change']:+.2%} (target: ≤+30%)")
        else:
            print(f"\n  Objectives: (unavailable - trial may have failed)")

        print(f"\n  Satisfies All Targets: {'✓ Yes' if rec.get('satisfies_targets') else '✗ No'}")

        if not rec.get('satisfies_targets'):
            violation = rec.get('violation_score')
            if violation is not None and violation != float('inf'):
                print(f"  Violation Score: {violation:.4f}")
    else:
        print("\n✗ No valid configurations found")

    # 輸出檔案
    print("\nOutput Files:")
    print(f"  Directory: {results['output_dir']}")
    print(f"  Full Results: {results['results_file']}")
    if results['plot_files']:
        print(f"  Visualizations: {len(results['plot_files'])} plots")
        for plot_type, path in results['plot_files'].items():
            print(f"    - {plot_type}: {os.path.basename(path)}")

    print("="*70)


if __name__ == "__main__":
    sys.exit(main())
