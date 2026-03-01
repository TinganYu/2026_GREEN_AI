import argparse
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_utils import evaluate_model
from datautils import get_calib_data
from act_aware_utils import calib_input_distribution, calib_fisher_info
from sensitivity import calib_sensitivity_ppl, calib_sensitivity_stable_rank
from quantization import rtn_quant_sequential, awq_quant_sequential
from binary_search import binary_search_truncation_rank
import numpy as np


def main(args):
    # setting random seed of numpy and torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # 檢查是否有明確的 CUDA_VISIBLE_DEVICES 設置 (避免多GPU衝突)
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not cuda_visible:
        # 如果沒有設置，預設使用 device 0
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device_id = '0'
    else:
        # 如果設置了，取第一個可見的 GPU
        device_id = cuda_visible.split(',')[0]
    
    # Load model
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # 改用明確的 device_map 而不是 "auto"，避免在多 GPU 間分散
    # 使用 cuda:X 而不是 "auto" 以避免層級分散
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    
    try:
        # 先嘗試用單 GPU 的 device_map
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", dtype=torch.float16, trust_remote_code=True
        )
    except Exception as e:
        print(f"Warning: device_map='auto' 失敗: {e}")
        print(f"嘗試用 device_map='cuda' 代替...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="cuda", dtype=torch.float16, trust_remote_code=True
        )

    # if "llama" in model_id or "opt" in model_id:
    #     model = model.to_bettertransformer()
    if not args.raw_model:
        # sensitivity calibration
        calib_loader = get_calib_data(
            args.calib_dataset, tokenizer, model_id, args.n_calib_samples, seed=args.seed, use_bos=args.use_bos
        )
        if "fisher" in args.scaling_method:
            calib_fisher_info(model, calib_loader, args.use_cache)
        if "abs" in args.scaling_method:
            calib_input_distribution(model, calib_loader, args.scaling_method, args.use_cache)
        if args.sensitivity_metric == "ppl":
            sensitivity = calib_sensitivity_ppl(model, calib_loader, args, args.use_cache)
        elif args.sensitivity_metric == "stable_rank":
            sensitivity = calib_sensitivity_stable_rank(model, calib_loader, args, args.use_cache)

        # search best truncation rank for each layer

        binary_search_truncation_rank(model, sensitivity, calib_loader, args)

        # quantization
        if args.weight_quant != "none":
            if args.weight_quant == "rtn_int8":
                rtn_quant_sequential(model, 8)
            elif args.weight_quant == "rtn_int6":
                rtn_quant_sequential(model, 6)
            elif args.weight_quant == "awq_int8":
                model = awq_quant_sequential(model, tokenizer, 8)
            elif args.weight_quant == "awq_int4":
                model = awq_quant_sequential(model, tokenizer, 4)

    # evaluate
    result = evaluate_model(
        model,
        tokenizer,
        args.model_id,
        "mmlu" if args.eval_mmlu else args.eval_tasks,
        eval_ppl=args.eval_ppl,
        limit=-1,
        use_bos=args.use_bos,
    )
    print(result)
    if not os.path.exists("output"):
        os.makedirs("output")
    with open("output/result.txt", "a+") as f:
        f.write(f"{args}\n")
        f.write(f"{result}\n")

    import json
    with open("output/temp_asvd_metrics.json", "w") as f:
        json.dump(result, f)

    # finished
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--ppl_target",
        type=float,
        default=-1,
        help="target ppl",
    )
    parser.add_argument(
        "--param_ratio_target",
        type=float,
        default=-1,
        help="target param ratio",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd (ASVD)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hyper-parameter alpha for ASVD",
    )
    parser.add_argument(
        "--n_calib_samples",
        type=int,
        default=32,
        help="number of samples used for calibration",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb", "alpaca", "selfgen"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max", "fisher", "fisher_abs_mean"],
        help="scaling method",
    )
    parser.add_argument(
        "--sensitivity_metric",
        type=str,
        default="ppl",
        choices=["ppl", "stable_rank"],
        help="search metric",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="use cached calibration results",
    )
    parser.add_argument(
        "--weight_quant",
        type=str,
        default="none",
        choices=["none", "rtn_int8", "rtn_int6", "awq_int8", "awq_int4"],
        help="weight quantization method",
    )
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="evaluate mmlu",
    )
    parser.add_argument(
        "--eval_ppl",
        default="wikitext2,ptb",
        type=str,
    )
    parser.add_argument("--eval_tasks", type=str, default="")
    parser.add_argument(
        "--sigma_fuse",
        type=str,
        default="UV",
        help="sigma fuse method",
        choices=["U", "V", "UV"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=233,
        help="random seed, which can significantly affect the calibration results",
    )
    parser.add_argument(
        "--compress_kv_cache",
        action="store_true",
        help="compress kv cache by asvd for k_proj and v_proj",
    )
    parser.add_argument(
        "--kv_cache_ratio_target",
        type=float,
        default=-1,
        help="kv cache ratio",
    )
    parser.add_argument(
        "--rank_align",
        type=int,
        default=1,
        help="align rank in SVD",
    )
    parser.add_argument(
        "--raw_model",
        action="store_true",
        help="use the raw model without ASVD",
    )
    parser.add_argument(
        "--use_bos",
        action="store_true",
        help="use bos token in calibration",
    )
    args = parser.parse_args()

    main(args)


# 在 asvd.py 最後面加入
def run_asvd_exp(config_dict):
    """
    提供給 Agent 呼叫的 API。
    config_dict: 例如 {"alpha": 0.6, "param_ratio_target": 0.8}
    """
    import argparse
    # 建立一個不讀取 CLI 的預設 args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/opt-1.3b",
        help="Pretrained model ID",
    )
    parser.add_argument(
        "--ppl_target",
        type=float,
        default=-1,
        help="target ppl",
    )
    parser.add_argument(
        "--param_ratio_target",
        type=float,
        default=-1,
        help="target param ratio",
    )
    parser.add_argument(
        "--act_aware",
        action="store_true",
        help="use act aware svd (ASVD)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hyper-parameter alpha for ASVD",
    )
    parser.add_argument(
        "--n_calib_samples",
        type=int,
        default=32,
        help="number of samples used for calibration",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb", "alpaca", "selfgen"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="abs_mean",
        choices=["abs_mean", "abs_max", "fisher", "fisher_abs_mean"],
        help="scaling method",
    )
    parser.add_argument(
        "--sensitivity_metric",
        type=str,
        default="ppl",
        choices=["ppl", "stable_rank"],
        help="search metric",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="use cached calibration results",
    )
    parser.add_argument(
        "--weight_quant",
        type=str,
        default="none",
        choices=["none", "rtn_int8", "rtn_int6", "awq_int8", "awq_int4"],
        help="weight quantization method",
    )
    parser.add_argument(
        "--eval_mmlu",
        action="store_true",
        help="evaluate mmlu",
    )
    parser.add_argument(
        "--eval_ppl",
        default="wikitext2,ptb",
        type=str,
    )
    parser.add_argument("--eval_tasks", type=str, default="")
    parser.add_argument(
        "--sigma_fuse",
        type=str,
        default="UV",
        help="sigma fuse method",
        choices=["U", "V", "UV"],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=233,
        help="random seed, which can significantly affect the calibration results",
    )
    parser.add_argument(
        "--compress_kv_cache",
        action="store_true",
        help="compress kv cache by asvd for k_proj and v_proj",
    )
    parser.add_argument(
        "--kv_cache_ratio_target",
        type=float,
        default=-1,
        help="kv cache ratio",
    )
    parser.add_argument(
        "--rank_align",
        type=int,
        default=1,
        help="align rank in SVD",
    )
    parser.add_argument(
        "--raw_model",
        action="store_true",
        help="use the raw model without ASVD",
    )
    parser.add_argument(
        "--use_bos",
        action="store_true",
        help="use bos token in calibration",
    )
    args = parser.parse_args([]) 
    
    # 覆蓋參數
    for k, v in config_dict.items():
        setattr(args, k, v)
        
    # 執行 main 並回傳結果
    # 建議修改 main 讓它回傳 result 字典
    return main(args)