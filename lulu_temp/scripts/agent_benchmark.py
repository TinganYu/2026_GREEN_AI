# scripts/agent_benchmark.py

'''
Quick experiments before doing full evaluation (to catch obvious problems).
Debugging KV compression logic.
Benchmarking speed and memory trade-offs.
'''
import json
import logging
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lulu_temp.agents.KVpress_n import KVPressAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent 
config_path = BASE_DIR / "config" / "model_config.yaml"

def benchmark_agent_compression(config_path=config_path):
    """
    Quick benchmark for KVPressAgent across multiple compression ratios/methods.
    Does not depend on a dataset.
    """
    agent = KVPressAgent(config_path=config_path)
    agent.load_model()
    prompt = "This is a test input for benchmarking." 
    input_ids = agent.encode_text(prompt)
    config = agent.config.get('tuning', {})
    ratios = config.get('compression_ratios_to_test', [0.5])
    methods = config.get('compression_methods_to_test', ['attention'])

    results = []

    for method in methods:
        for ratio in ratios:
            logger.info(f"Benchmarking agent: method={method}, ratio={ratio}")
            agent.compression_ratio = ratio
            agent.compression_method = method

            # Use built-in agent benchmarking
            benchmark_stats = agent.benchmark_compression_methods(input_ids=input_ids, method=method)
            compression_stats = agent.get_compression_stats()

            results.append({
                "method": method,
                "ratio": ratio,
                "compression_stats": compression_stats,
                "benchmark_stats": benchmark_stats
            })

    # Save results
    out_file = Path("agent_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Agent benchmark results saved to {out_file.resolve()}")
    return results

if __name__ == "__main__":
    benchmark_agent_compression()
