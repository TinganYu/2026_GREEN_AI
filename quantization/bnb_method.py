from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
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

def bnb_quantization(model_path, model_id, bits=4, output_dir=None, hf_token=None):
    if output_dir is None:
        output_dir = f"../quant_models/{model_id}-bnb-{bits}bit"

    print("🔹 檢查 GPU 是否可用...")
    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU 不可用，請確認是否正確安裝 CUDA 和驅動程式。")

    device = torch.device("cuda")
    print(f"✅ 使用 GPU: {torch.cuda.get_device_name(device)}")

    print("🔹 載入 tokenizer...")
    tokenizer = safe_load_tokenizer(model_path)

    print("🔹 載入原始模型...")
    gc.collect()
    torch.cuda.empty_cache()
    quant_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        low_cpu_mem_usage=True, 
        device_map="auto",
        use_auth_token=hf_token
    ).to(device)

    print("🔹 儲存 bitsandbytes 量化模型...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ bitsandbytes 量化完成，已存到 {output_dir}")
    return output_dir

if __name__ == "__main__":

    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
    MODEL_ID = "llama-3.2-1b"
    OUTDIR = "../quant_models/llama-3.2-1b-bnb-4bit"

    bnb_quantization(MODEL_PATH, bits=4, output_dir=OUTDIR, hf_token=HF_TOKEN)