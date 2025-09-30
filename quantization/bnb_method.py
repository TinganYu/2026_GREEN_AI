from dataclasses import field
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
import gc
from dotenv import load_dotenv

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

def bnb_quantization(
    model_path: str,
    bits: int = field(default=4, metadata={"choices": [4, 8]}),
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"choices": ["fp4", "nf4"]}),
    bnb_4bit_use_double_quant: bool = field(default=True),
    bnb_4bit_compute_dtype: str = field(default="bfloat16", metadata={"choices": ["fp16", "bfloat16", "float32"]}),
    llm_int8_threshold: float = field(default=6.0),
    llm_int8_skip_modules: Optional[list] = field(default=None),
    llm_int8_has_fp16_weight: bool = field(default=False),
    llm_int8_enable_fp32_cpu_offload: bool = field(default=False),
    output_dir: Optional[str] = None,
    hf_token: Optional[str] = None
) -> str:
    """
    使用 BitsAndBytes 進行模型量化。

    Args:
        model_path (str): 模型的路徑或名稱。
        bits (int): 量化位元數，預設為 4，可選 [4, 8]。
        bnb_4bit_quant_type (str): 4-bit 量化類型，預設為 "nf4"，可選 ["fp4", "nf4"]。
        bnb_4bit_use_double_quant (bool): 是否啟用 二次量化，預設為 True。
        bnb_4bit_compute_dtype (str): 計算時使用的精度，預設為 "bfloat16"，可選 ["fp16", "bfloat16", "float32"]。
        llm_int8_threshold (float): LLM.int8 的 outlier 閾值，預設為 6.0。
        llm_int8_skip_modules (Optional[list]): 跳過 int8 量化的模組，預設為 None。
        llm_int8_has_fp16_weight (bool): 是否保留原始 fp16 權重副本，預設為 False。
        llm_int8_enable_fp32_cpu_offload (bool): 是否將部分 fp32 權重丟到 CPU，預設為 False。
        output_dir (Optional[str]): 儲存量化模型的目錄，若為 None 則自動生成。
        hf_token (Optional[str]): Hugging Face 的存取權杖。

    Returns:
        str: 儲存量化模型的目錄。
    """
    model_id = model_path.split("/")[-1].lower().replace("instruct", "it")
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
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        llm_int8_threshold=llm_int8_threshold,
        llm_int8_skip_modules=llm_int8_skip_modules,
        llm_int8_has_fp16_weight=llm_int8_has_fp16_weight,
        llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        low_cpu_mem_usage=True, 
        device_map="auto",
        token=hf_token
    ).to(device)

    print("🔹 儲存 bitsandbytes 量化模型...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ bitsandbytes 量化完成，已存到 {output_dir}")
    return output_dir

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
    MODEL_ID = MODEL_PATH.split("/")[-1].lower().replace("instruct", "it")
    OUTDIR = f"quant_models/{MODEL_ID}-bnb-test"

    bnb_quantization(
        model_path=MODEL_PATH,
        bits=4,
        output_dir=OUTDIR,
        hf_token=HF_TOKEN
    )