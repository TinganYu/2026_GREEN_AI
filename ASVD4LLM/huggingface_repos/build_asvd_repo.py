import sys
import os

# 1. 取得當前腳本的目錄
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 取得父目錄 (上一個層級)
parent_dir = os.path.dirname(current_dir)

# 3. 將父目錄添加到 Python 搜索路徑中
sys.path.append(parent_dir)

import argparse
import torch
from pathlib import Path
from transformers import LlamaConfig, AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.utils import cached_file
from evaluate_utils import evaluate_model, evaluate_perplexity
from datautils import get_calib_data
from act_aware_utils import calib_input_distribution, calib_fisher_info
from sensitivity import calib_sensitivity_ppl, calib_sensitivity_stable_rank
from quantization import rtn_quant_sequential, find_layers, awq_quant_sequential
from binary_search import binary_search_truncation_rank
from modules.svd_linear import SVDLinear
import os
import json

def main(args):
    model_id = args.model_id

    # 檢查是否有明確的 CUDA_VISIBLE_DEVICES 設置 (避免多GPU衝突)
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not cuda_visible:
        # 如果沒有設置，預設使用 device 0
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device_id = '0'
    else:
        # 如果設置了，取第一個可見的 GPU
        device_id = cuda_visible.split(',')[0]
    
    print(f"🚀 準備載入模型: {model_id}")

    try:
        # --- A. 載入 Tokenizer ---
        print(f"🚀 正在載入 Tokenizer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # --- B. 獲取並修正 Config (支援本地與 Hugging Face) ---
        print(f"🔍 正在獲取設定檔...")
        # 判斷是本地路徑還是 HF Hub ID
        if Path(model_id).exists() and (Path(model_id) / "config.json").exists():
            c_path = str(Path(model_id) / "config.json")
        else:
            # 從 Hugging Face 快取中抓取 config.json 的路徑
            c_path = cached_file(model_id, "config.json")

        # 使用讀取字典的方式來避開 4.56.0 的檢查員
        with open(c_path, "r", encoding="utf-8") as f_in:
            d_dict = json.load(f_in)

        # 暫時移除 rope_scaling 欄位
        r_info = d_dict.pop("rope_scaling", None)
        
        # 建立 Config 物件 (此時不含 rope_scaling，保證通過)
        my_config = LlamaConfig.from_dict(d_dict)

        # 手動注入屬性，繞過 LlamaConfig.__init__ 的驗證
        if r_info:
            # 補齊 4.56.0 需要的 type 欄位
            if "rope_type" in r_info and "type" not in r_info:
                r_info["type"] = r_info["rope_type"]
            my_config.rope_scaling = r_info
            print("✅ 已成功從遠端/本地獲取並修正 RoPE 設定")

        # --- C. 載入模型權重 ---
        print(f"📦 正在載入模型權重...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=my_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("🎉 模型與 Tokenizer 載入成功！")

    except Exception as e:
        # 使用 e 避免與 json 等模組名衝突
        print(f"❌ 載入失敗: {e}")
        # 如果失敗了，嘗試最後的掙扎 (不帶 config 載入)
        print("嘗試直接載入...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    orig_params = sum(p.numel() for p in model.parameters())

    # sensitivity calibration
    calib_loader = get_calib_data(args.calib_dataset, tokenizer, model_id, args.n_calib_samples,seed=args.seed,use_bos=args.use_bos)
    if "fisher" in args.scaling_method:
        calib_fisher_info(model, calib_loader, args.use_cache)
    if "abs" in args.scaling_method:
        calib_input_distribution(
            model, calib_loader, args.scaling_method, args.use_cache
        )
    if args.sensitivity_metric == "ppl":
        sensitivity = calib_sensitivity_ppl(model, calib_loader, args, args.use_cache)
    elif args.sensitivity_metric == "stable_rank":
        sensitivity = calib_sensitivity_stable_rank(
            model, calib_loader, args, args.use_cache
        )

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
# Optional verification check
    sample_layer = model.model.layers[0].self_attn.q_proj
    if isinstance(sample_layer, SVDLinear):
        # We must check the weights INSIDE the SVD components
        a_unique = len(torch.unique(sample_layer.ALinear.weight))
        b_unique = len(torch.unique(sample_layer.BLinear.weight))
        print(f"✅ ALinear unique values: {a_unique}") # Expect <= 256
        print(f"✅ BLinear unique values: {b_unique}") # Expect <= 256
    else:
        u_count = len(torch.unique(sample_layer.weight))
        print(f"Standard Linear unique values: {u_count}")
    print(sample_layer.ALinear.weight.dtype)

    new_params = sum(p.numel() for p in model.parameters())
    actual_ratio = new_params / orig_params

    # build huggingface model
    assert args.param_ratio_target > 0
    assert args.act_aware
    assert args.alpha >= 0 and args.alpha <= 1
    #assert args.calib_dataset == "wikitext2"
    #assert args.scaling_method == "abs_mean"
    assert args.sensitivity_metric == "ppl"
    assert args.use_cache
    #assert args.weight_quant == "none"
    assert not args.eval_mmlu

    # 將模型保存到父目錄的 output 資料夾而不是嵌套的 huggingface_repos
    model_name = model_id.split("/")[-1]
    parent_dir = os.path.dirname(current_dir)  # 返回 ASVD4LLM-main 目錄
    save_path = os.path.join(
        parent_dir,
        "output",
        f"{model_name}-asvd{int(args.param_ratio_target*100)}-alpha{int(args.alpha*100)}",
    )
    
        # f"{model_name}-asvd{int(args.param_ratio_target*100)}quant{args.weight_quant}-alpha{int(args.alpha*100)}",
    os.makedirs(save_path, exist_ok=True)
    
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    
    config = model.config.to_dict()
    config["actual_param_ratio"] = actual_ratio  # 注入真實壓縮比例

    config["truncation_ranks"] = {}
    for name, module in model.named_modules():
        if isinstance(module, SVDLinear):
            config["truncation_ranks"][name] = module.truncation_rank
    if "opt" in model_id.lower():
        config["auto_map"] = {
            "AutoConfig": "configuration_asvd_opt.ASVDOPTConfig",
            "AutoModelForCausalLM": "modeling_asvd_opt.ASVDOPTForCausalLM",
        }
        config["architectures"] = ["ASVDOPTForCausalLM"]
        os.system(
            f"cp {current_dir}/configuration_asvd_opt.py {current_dir}/modeling_asvd_opt.py {save_path}/"
        )
    elif "llama" in model_id.lower():
        print("Detected LLaMA model for ASVD conversion.!!!!!!!!!!!!!!!!!!!!!!!")
        config["auto_map"] = {
            "AutoConfig": "configuration_asvd_llama.ASVDLlamaConfig",
            "AutoModelForCausalLM": "modeling_asvd_llama.ASVDLlamaForCausalLM",
        }
        config["architectures"] = ["ASVDLlamaForCausalLM"]
        os.system(
            f"cp {current_dir}/configuration_asvd_llama.py {current_dir}/modeling_asvd_llama.py {save_path}/"
        )

    # json.dump(config, open(save_path + "/config.json", "w"), indent=2)
    final_config_path = os.path.join(save_path, "config.json")

    with open(final_config_path, "w") as f:
        # 這裡的寫入會取代 model.save_pretrained 產生的檔案
        json.dump(config, f, indent=2)

    print("Done building huggingface model")
    
    # 評估模型
    print("\n=== 開始評估模型 ===")
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
    
    # 保存評估結果
    # 1. 取得 build_asvd_repo.py 的絕對路徑並推算根目錄 (ASVD4LLM-main)
    current_file_path = Path(__file__).resolve()
    root_dir = current_file_path.parent.parent 
    
    # 2. 定義統一的輸出目錄 (ASVD4LLM-main/output)
    output_dir = root_dir / "output"
    # 3. 確保目錄存在 (使用 Path.mkdir 更簡潔且防錯)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 4. 使用絕對路徑寫入 result.txt
    result_txt_path = output_dir / "result.txt"
    with open(result_txt_path, "a+", encoding="utf-8") as f:
        f.write(f"{args}\n")
        f.write(f"{result}\n")
    
    # 5. 使用絕對路徑寫入 temp_asvd_metrics.json (供 asvd_tuner 讀取)
    metrics_json_path = output_dir / "temp_asvd_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"✅ 評估結果已儲存至: {output_dir}")
    
    del model
    del tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.push:
        # load
        hub_name = model_id.split("/")[-1] + f"-asvd{int(args.param_ratio_target*100)}"
        tokenizer = AutoTokenizer.from_pretrained(save_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            save_path,
            device_map="cpu",
            dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer.push_to_hub(hub_name)
        model.push_to_hub(hub_name)


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
        "--seed",
        type=int,
        default=233,
        help="random seed, which can significantly affect the calibration results",
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb", "gsm8k"],
        help="calibration dataset",
    )
    parser.add_argument(
        "--compress_kv_cache",
        action="store_true",
        help="compress kv cache by asvd for k_proj and v_proj",
    )
    parser.add_argument(
        "--rank_align",
        type=int,
        default=128,
        help="align rank in SVD",
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
        "--sigma_fuse",
        type=str,
        default="UV",
        help="sigma fuse method",
        choices=["U", "V", "UV"],
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="push to hub",
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default="",
        help="evaluation tasks",
    )
    parser.add_argument(
        "--eval_ppl",
        type=str,
        default="wikitext2,ptb",
        help="evaluation datasets for ppl",
    )
    parser.add_argument(
        "--use_bos",
        action="store_true",
        help="use bos token",
    )
    args = parser.parse_args()

    main(args)
