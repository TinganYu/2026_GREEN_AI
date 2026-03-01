# import os
# import sys
# import argparse
# import logging
# import yaml
# from pathlib import Path

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from eval.gsm_eval import GSM8KEvaluator
# from eval.multinews_eval import MultiNewsEvaluator

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# BASE_DIR = Path(__file__).resolve().parent.parent 
# CONFIG_DIR = BASE_DIR / "config"


# def create_default_config(task: str = "gsm8k", config_dir: Path = CONFIG_DIR):
#     """Create default config for specified task"""
#     config_dir.mkdir(parents=True, exist_ok=True)
#     config_path = config_dir / f"model_config_{task}.yaml"
    
#     if config_path.exists():
#         logger.info(f"Config already exists: {config_path}")
#         return str(config_path)
    
#     if task == "gsm8k":
#         default_config = {
#             'model': {'name': 'google/gemma-3-270m-it'},
#             'dataset': {
#                 'name': 'gsm8k', 
#                 'split': 'test', 
#                 'num_samples': 100,
#                 'prompts': {
#                     'chain_of_thought': 'Question: {question}\n\nLet\'s solve this step-by-step:\n\nAnswer:',
#                     'direct': 'Question: {question}\nAnswer:'
#                 }
#             },
#             'generation': {
#                 'max_new_tokens': 512,
#                 'math': {
#                     'temperature': 0.1,
#                     'max_new_tokens': 1024,
#                     'do_sample': False
#                 }
#             },
#             'kv_compression': {
#                 'enabled': True, 
#                 'compression_ratio': 0.5, 
#                 'compression_method': 'snapkv'
#             },
#             'evaluation': {
#                 'use_chain_of_thought': True, 
#                 'save_results': True, 
#                 'results_dir': str(BASE_DIR / "results")
#             },
#             'system': {'device': 'auto'},
#             'tuning': {
#                 'enabled': True,
#                 'adaptive': False,
#                 'compression_methods_to_test': ['snapkv', 'observed_attention', 'knorm'],
#                 'compression_ratios_to_test': [0.75, 0.5, 0.3]
#             }
#         }
    
#     elif task == "multinews":
#         default_config = {
#             'model': {'name': 'google/gemma-3-270m-it'},
#             'dataset': {
#                 'name': 'alexfabbri/multi_news',
#                 'split': 'test',
#                 'num_samples': 500,
#                 'max_input_tokens': 4096,
#                 'prompts': {
#                     'summarization': (
#                         'Read the following news articles and write a concise summary:\n\n'
#                         '{document}\n\nSummary:'
#                     )
#                 }
#             },
#             'generation': {
#                 'max_new_tokens': 512,
#                 'summarization': {
#                     'max_new_tokens': 256,
#                     'temperature': 0.7,
#                     'do_sample': True,
#                     'repetition_penalty': 1.3
#                 }
#             },
#             'kv_compression': {
#                 'enabled': True,
#                 'compression_ratio': 0.5,
#                 'compression_method': 'snapkv',
#                 'methods': {
#                     'snapkv': {
#                         'window_size': 64,
#                         'kernel_size': 7
#                     },
#                     'streaming_llm': {
#                         'window_size': 1024
#                     }
#                 }
#             },
#             'evaluation': {
#                 'save_results': True,
#                 'results_dir': str(BASE_DIR / "results")
#             },
#             'system': {'device': 'auto'},
#             'tuning': {
#                 'enabled': True,
#                 'adaptive': False,
#                 'compression_methods_to_test': ['snapkv', 'observed_attention', 'knorm'],
#                 'compression_ratios_to_test': [0.75, 0.5, 0.3]
#             }
#         }
    
#     else:
#         raise ValueError(f"Unknown task: {task}")
    
#     with open(config_path, 'w') as f:
#         yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
#     logger.info(f"Created default config: {config_path}")
#     return str(config_path)


# def main():
#     parser = argparse.ArgumentParser(
#         description="Run evaluation with KV compression tuning",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Run GSM8K evaluation with default config
#   python run_unified.py --task gsm8k
  
#   # Run MultiNews evaluation with custom config
#   python run_unified.py --task multinews --config path/to/config.yaml
  
#   # Create default config and exit
#   python run_unified.py --task multinews --create-config
  
#   # Run without compression or tuning
#   python run_unified.py --task gsm8k --no-compression --no-tuning
#         """
#     )
    
#     parser.add_argument(
#         "--task", 
#         type=str, 
#         required=True,
#         choices=['gsm8k', 'multinews'],
#         help="Task to evaluate: gsm8k (math) or multinews (summarization)"
#     )
#     parser.add_argument(
#         "--config", 
#         type=str, 
#         help="Path to config file (default: auto-generate)"
#     )
#     parser.add_argument(
#         "--create-config", 
#         action="store_true",
#         help="Create default config file and exit"
#     )
#     parser.add_argument(
#         "--no-compression", 
#         action="store_true",
#         help="Disable KV compression"
#     )
#     parser.add_argument(
#         "--no-tuning", 
#         action="store_true",
#         help="Disable hyperparameter tuning (run single evaluation)"
#     )
#     parser.add_argument(
#         "--no-cot", 
#         action="store_true",
#         help="Disable chain-of-thought prompting (GSM8K only)"
#     )
#     parser.add_argument(
#         "--num-samples",
#         type=int,
#         help="Override number of samples to evaluate"
#     )
    
#     args = parser.parse_args()
    
#     # Handle config creation
#     if args.create_config:
#         create_default_config(args.task)
#         return
    
#     # Get or create config path
#     config_path = args.config or create_default_config(args.task)
    
#     # Initialize appropriate evaluator
#     logger.info(f"{'='*60}")
#     logger.info(f"Starting {args.task.upper()} evaluation")
#     logger.info(f"Config: {config_path}")
#     logger.info(f"{'='*60}\n")
    
#     if args.task == "gsm8k":
#         evaluator = GSM8KEvaluator(config_path)
#     elif args.task == "multinews":
#         evaluator = MultiNewsEvaluator(config_path)
#     else:
#         raise ValueError(f"Unknown task: {args.task}")
    
#     # Apply command-line overrides
#     if args.no_compression:
#         evaluator.config['kv_compression']['enabled'] = False
#         logger.info("KV compression disabled via command line")
    
#     if args.no_tuning:
#         evaluator.config['tuning']['enabled'] = False
#         logger.info("Tuning disabled via command line")
    
#     if args.no_cot and args.task == "gsm8k":
#         evaluator.config['evaluation']['use_chain_of_thought'] = False
#         logger.info("Chain-of-thought disabled via command line")
    
#     if args.num_samples:
#         evaluator.config['dataset']['num_samples'] = args.num_samples
#         logger.info(f"Number of samples overridden to: {args.num_samples}")
    
#     # Run evaluation or tuning
#     try:
#         if evaluator.config.get('tuning', {}).get('enabled', False):
#             logger.info("Running hyperparameter tuning...")
#             results = evaluator.run_tuning()
            
#             # Print best result
#             if isinstance(results, list) and len(results) > 0:
#                 best = results[0]  # Already sorted
#                 logger.info(f"\n{'='*60}")
#                 logger.info("BEST CONFIGURATION:")
#                 logger.info(f"Method: {best['configuration']['tuning_method']}")
#                 logger.info(f"Ratio: {best['configuration']['tuning_ratio']}")
                
#                 if args.task == "gsm8k":
#                     logger.info(f"Accuracy: {best['results']['accuracy']:.4f}")
#                 elif args.task == "multinews":
#                     logger.info(f"ROUGE-L F1: {best['results'].get('avg_rougeL_f', 0):.4f}")
                
#                 logger.info(f"{'='*60}\n")
#         else:
#             logger.info("Running single evaluation...")
#             results = evaluator.run_evaluation()
            
#             # Print summary
#             summary = results['summary']
#             logger.info(f"\n{'='*60}")
#             logger.info("EVALUATION SUMMARY:")
#             logger.info(f"Total samples: {summary['results']['total_samples']}")
            
#             if args.task == "gsm8k":
#                 logger.info(f"Accuracy: {summary['results']['accuracy']:.4f}")
#                 logger.info(f"Correct: {summary['results']['correct_answers']}")
#             elif args.task == "multinews":
#                 if 'avg_rougeL_f' in summary['results']:
#                     logger.info(f"ROUGE-1 F1: {summary['results'].get('avg_rouge1_f', 0):.4f}")
#                     logger.info(f"ROUGE-2 F1: {summary['results'].get('avg_rouge2_f', 0):.4f}")
#                     logger.info(f"ROUGE-L F1: {summary['results'].get('avg_rougeL_f', 0):.4f}")
            
#             logger.info(f"Avg time: {summary['performance']['avg_time']:.2f}s")
            
#             if 'memory_tracking' in summary:
#                 mem = summary['memory_tracking']
#                 logger.info(f"Peak memory: {mem.get('peak_allocated_gb', 'N/A'):.2f} GB")
            
#             logger.info(f"{'='*60}\n")
        
#         logger.info("✓ Evaluation completed successfully!")
        
#     except Exception as e:
#         logger.error(f"✗ Evaluation failed: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import yaml
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from eval.gsm_eval import GSM8KEvaluator
from eval.multinews_eval import MultiNewsEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent 
DEFAULT_CONFIG = BASE_DIR / "config" / "model_config.yaml"


def deep_merge(base: dict, update: dict) -> dict:
    """Merge two dicts - update overrides base"""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--task", type=str, required=True, choices=['gsm8k', 'multinews'])
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--no-compression", action="store_true")
    parser.add_argument("--no-tuning", action="store_true")
    parser.add_argument("--num-samples", type=int)
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)
    
    logger.info(f"Task: {args.task.upper()} | Config: {args.config}\n")
    
    # Load config
    with open(args.config, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Merge: shared + task-specific
    shared = {k: v for k, v in full_config.items() if k != 'tasks'}
    tasks = full_config.get('tasks', {})
    
    if args.task not in tasks:
        logger.error(f"Task '{args.task}' not in config. Available: {list(tasks.keys())}")
        sys.exit(1)
    
    config = deep_merge(shared, tasks[args.task])
    
    # Initialize evaluator
    if args.task == "gsm8k":
        evaluator = GSM8KEvaluator(args.config)
    elif args.task == "multinews":
        evaluator = MultiNewsEvaluator(args.config)
    
    # Override with merged config
    evaluator.config = config
    evaluator.agent.config = config
    
    # Command-line overrides
    if args.no_compression:
        config['kv_compression']['enabled'] = False
        logger.info("⚠️  Compression disabled")
    if args.no_tuning:
        config['tuning']['enabled'] = False
        logger.info("⚠️  Tuning disabled")
    if args.num_samples:
        config['dataset']['num_samples'] = args.num_samples
        logger.info(f"⚠️  Samples: {args.num_samples}")
    
    # Run
    try:
        if config.get('tuning', {}).get('enabled'):
            results = evaluator.run_tuning()
            if isinstance(results, list) and results:
                print(f"\n🏆 Best: {results[0]['configuration']['tuning_method']} @ {results[0]['configuration']['tuning_ratio']}")
                if args.task == "gsm8k":
                    print(f"   Accuracy: {results[0]['results']['accuracy']:.4f}")
                else:
                    print(f"   ROUGE-L: {results[0]['results'].get('avg_rougeL_f', 0):.4f}")
        else:
            results = evaluator.run_evaluation()
            summary = results['summary']
            print(f"\n📊 Results: {summary['results']['total_samples']} samples")
            if args.task == "gsm8k":
                print(f"   Accuracy: {summary['results']['accuracy']:.4f}")
            else:
                print(f"   ROUGE-L: {summary['results'].get('avg_rougeL_f', 0):.4f}")
        
        logger.info("\n✅ Done!")
    except Exception as e:
        logger.error(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()