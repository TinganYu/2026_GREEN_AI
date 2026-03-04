import sys
import os
import json
import yaml
import logging
import argparse
import subprocess
from pathlib import Path

logger = logging.getLogger("Executors")

CURRENT_DIR = Path(__file__).resolve().parent
GLOBAL_TUNER_DIR = CURRENT_DIR.parent
ROOT_DIR = GLOBAL_TUNER_DIR.parent

# Set up paths for direct imports
sys.path.insert(0, str(ROOT_DIR / "tmp"))
sys.path.insert(0, str(GLOBAL_TUNER_DIR))
ASVD_ROOT = ROOT_DIR / "ASVD4LLM"
ASVD_REPO_DIR = ASVD_ROOT / "huggingface_repos"
sys.path.insert(0, str(ASVD_ROOT))
sys.path.insert(0, str(ASVD_REPO_DIR))
# Config path for evaluation (test_eval.py)
CONFIG_PATH = ROOT_DIR / "tmp/config/model_config.yaml"
EVAL_SCRIPT = ROOT_DIR / "tmp/test/test_eval.py"

def run_asvd(model_id: str, alpha: float, param_ratio_target: float, scaling_method: str) -> str:
    """Directly calls the ASVD build process as a Python module."""
    try:
        from build_asvd_repo import main as asvd_main
    except ImportError as e:
        logger.error(f"Failed to import build_asvd_repo: {e}. sys.path: {sys.path}")
        raise
    
    logger.info(f"--- Executing ASVD Stage: Ratio={param_ratio_target} ---")
    
    # Ensure device is set
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    
    args = argparse.Namespace(
        model_id=model_id,
        alpha=alpha,
        param_ratio_target=param_ratio_target,
        scaling_method=scaling_method,
        act_aware=True,
        use_cache=True,
        sensitivity_metric="ppl",
        weight_quant="none",
        eval_mmlu=False,
        calib_dataset="wikitext2",
        n_calib_samples=32,
        seed=233,
        compress_kv_cache=False,
        rank_align=128,
        sigma_fuse="UV",
        push=False,
        eval_tasks="",
        eval_ppl="wikitext2,ptb",
        use_bos=False,
        ppl_target=-1.0
    )
    
    # Run the ASVD main function
    asvd_main(args)
    
    # Reconstruct the output path based on build_asvd_repo.py logic
    model_name = model_id.split("/")[-1]
    output_dir = ASVD_REPO_DIR.parent / "output" / f"{model_name}-asvd{int(param_ratio_target*100)}-alpha{int(alpha*100)}"
    
    if not output_dir.exists():
        raise RuntimeError(f"ASVD output directory not found at {output_dir}")
        
    return str(output_dir)

def run_quantization(model_path: str, method: str, bits: int, group_size: int = 128, quant_type: str = "nf4", desc_act: bool = False, use_double_quant:bool = False) -> str:
    """Directly calls the quantization methods."""
    from method.quantization import Quantizer, GPTQConfig, AWQConfig, BNBConfig
    
    logger.info(f"--- Executing Quantization Stage: {method} ({bits} bits) ---")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    
    quantizer = Quantizer(model_path)
    
    if method == "gptq":
        config = GPTQConfig(bits=bits, group_size=group_size, desc_act=desc_act)
    elif method == "awq":
        config = AWQConfig(w_bit=bits, q_group_size=group_size, zero_point=True)
    elif method == "bnb":
        config = BNBConfig(bits=bits, bnb_4bit_quant_type=quant_type, bnb_4bit_use_double_quant=use_double_quant)
    else:
        logger.warning(f"Unknown quantization method: {method}. Skipping.")
        return model_path
        
    output_path = quantizer.quantize(config)
    return output_path

def _update_yaml_config_for_eval(model_path: str):
    """Helper purely for test_eval.py since we haven't refactored it yet."""
    if not CONFIG_PATH.exists():
        logger.warning(f"Config file not found: {CONFIG_PATH}. Skipping YAML update.")
        return
        
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if 'model' not in config: config['model'] = {}
    config['model']['name'] = model_path
    
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

def run_evaluation(model_path: str, task: str) -> dict:
    """Runs evaluation. Kept as subprocess to avoid touching test_eval.py for now."""
    logger.info(f"--- Evaluating Model at: {model_path} ---")
    
    _update_yaml_config_for_eval(model_path)
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0')
    
    subprocess.run([sys.executable, str(EVAL_SCRIPT)], env=env, cwd=str(ROOT_DIR))
    
    model_name = Path(model_path).name
    json_path = ROOT_DIR / "results" / model_name / f"{task}_results.json"

    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
            accuracy = eval_data.get('accuracy', 0.0)
            latency = eval_data.get('total_generation_time_sec', 0.0)
            score = (0.7 * accuracy) + (0.3 * (1.0 / (latency + 1e-6)))
            return {"accuracy": accuracy, "latency": latency, "score": score}
    
    logger.warning(f"Result file not found: {json_path}")
    return {"accuracy": 0.0, "latency": 0.0, "score": 0.0}
