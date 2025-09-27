from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
import gc
import os
from dotenv import load_dotenv

def safe_load_tokenizer(model_path):
    """
    保險載入 tokenizer（處理 fast tokenizer 出錯的情況）
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except Exception as e:
        print(f"⚠️ fast tokenizer 載入失敗，改用 slow 版本: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    return tokenizer

def awq_quantization(model_path, model_id, bits=4, group_size=128, zero_point=True, output_dir=None, hf_token=None):
    if output_dir is None:
        output_dir = f"../quant_models/{model_id}-awq-{bits}bit"

    print("🔹 檢查 GPU 是否可用...")
    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU 不可用，請確認是否正確安裝 CUDA 和驅動程式。")

    device = torch.device("cuda")
    print(f"✅ 使用 GPU: {torch.cuda.get_device_name(device)}")

    print("🔹 載入 tokenizer...")
    tokenizer = safe_load_tokenizer(model_path)

    print("🔹 建立 AWQ 配置...")
    quant_config = { 
        "zero_point": zero_point, 
        "q_group_size": group_size, 
        "w_bit": bits, 
        "version": "GEMM" 
    }

    print("🔹 載入原始模型...")
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True, 
        device_map="auto", 
        use_auth_token=hf_token
    )
    
    print("🔹 開始 AWQ 量化...")
    model.quantize(tokenizer, quant_config=quant_config)

    print("🔹 儲存 AWQ 量化模型...")
    model.save_quantized(output_dir, safetensors=True)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ AWQ 量化完成，已存到 {output_dir}")
    return output_dir

if __name__ == "__main__":

    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
    MODEL_ID = "llama-3.2-1b"
    OUTDIR = "../quant_models/llama-3.2-1b-awq-4bit"

    awq_quantization(MODEL_PATH, MODEL_ID, bits=4, group_size=128, output_dir=OUTDIR, hf_token=HF_TOKEN)