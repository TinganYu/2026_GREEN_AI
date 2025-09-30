from dataclasses import field
from typing import Optional, List
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
import torch
import gc

def safe_load_tokenizer(model_path: str) -> AutoTokenizer:
    """
    保險載入 tokenizer（處理 fast tokenizer 出錯的情況）
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except Exception as e:
        print(f"⚠️ fast tokenizer 載入失敗，改用 slow 版本: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    return tokenizer

def awq_quantization(
    model_path: str,
    zero_point: bool = field(default=True),
    group_size: int = field(default=128),
    bits: int = field(default=4),
    version: str = field(default="gemm", metadata={"choices": ["gemm", "gemv", "marlin", "gemv_fast"]}),
    modules_to_not_convert: Optional[List] = None,
    output_dir: Optional[str] = None,
    hf_token: Optional[str] = None
) -> str:
    """
    使用 AWQ 進行模型量化。

    Args:
        model_path (str): 模型的路徑或名稱。
        quant_method (str): 量化方法，預設為 "awq"。
        zero_point (bool): 是否使用 zero-point 量化 (偏移補償)。
        group_size (int): group size，多少權重為一組做量化。
        bits (int): 權重量化位元數，常見為 4-bit。
        version (str): 推理時的 backend 版本 (矩陣乘法最佳化)。
        modules_to_not_convert (Optional[List]): 指定不量化的模組 (如 lm_head)。
        output_dir (Optional[str]): 儲存量化模型的目錄。
        hf_token (Optional[str]): Hugging Face 的存取權杖。

    Returns:
        str: 儲存量化模型的目錄。
    """
    model_id = model_path.split("/")[-1].lower().replace("instruct", "it")
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
        "version": version,
        "modules_to_not_convert": modules_to_not_convert
    }

    print("🔹 載入原始模型...")
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        device_map="auto",
        token=hf_token
    )
    
    print("🔹 開始 AWQ 量化...")
    model.quantize(tokenizer, quant_config=quant_config)

    print("🔹 儲存 AWQ 量化模型...")
    model.save_quantized(output_dir, safetensors=True)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ AWQ 量化完成，已存到 {output_dir}")
    return output_dir

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
    MODEL_ID = MODEL_PATH.split("/")[-1].lower().replace("instruct", "it")
    OUTDIR = f"quant_models/{MODEL_ID}-awq-test"

    awq_quantization(
        model_path=MODEL_PATH,
        zero_point=True,
        group_size=128,
        bits=4,
        version="gemm",
        modules_to_not_convert=None,
        output_dir=OUTDIR,
        hf_token=HF_TOKEN
    )