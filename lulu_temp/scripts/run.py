import os
import sys
import argparse
import logging
import yaml
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from eval.gsm_eval import GSM8KEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent 
CONFIG_DIR = BASE_DIR / "config"

def create_default_config_if_missing(config_dir: Path = CONFIG_DIR):
    config_path = config_dir / "model_config.yaml"
    if config_path.exists():
        return str(config_path)

    config_dir.mkdir(parents=True, exist_ok=True)
    default_config = {
        'model': {'name': 'google/gemma-3-270m-it'},
        'dataset': {'name': 'gsm8k', 'split': 'test', 'num_samples': 100},
        'kv_compression': {'enabled': True, 'compression_ratio': 0.5, 'compression_method': 'attention'},
        'evaluation': {'use_chain_of_thought': True, 'save_results': True, 'results_dir': str(BASE_DIR / "results")},
        'system': {'device': 'auto'}
    }

    with open(config_path, 'w') as f:
        yaml.dump(default_config, f)
    logger.info(f"Created default config: {config_path}")
    return str(config_path)

def main():
    parser = argparse.ArgumentParser(description="Run GSM8K evaluation and KV tuning")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--create-config", action="store_true", help="Create default config and exit")
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--no-cot", action="store_true")
    args = parser.parse_args()

    if args.create_config:
        create_default_config_if_missing()
        return

    config_path = args.config or create_default_config_if_missing()

    evaluator = GSM8KEvaluator(config_path)
    if args.no_compression:
        evaluator.config['kv_compression']['enabled'] = False
    if args.no_cot:
        evaluator.config['evaluation']['use_chain_of_thought'] = False

    results = evaluator.run_tuning()
    # results = evaluator.run_evaluation() # only have base model

    logger.info("Tuning/Evaluation finished!")

if __name__ == "__main__":
    main()
