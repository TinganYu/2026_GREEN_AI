from dataclasses import field
from typing import Optional
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
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

def gptq_quantization(
    model_path: str,
    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]}),
    group_size: int = field(default=-1),
    damp_percent: float = field(default=0.01),
    desc_act: bool = field(default=True),
    # static_groups: bool = field(default=False),
    sym: bool = field(default=True),
    true_sequential: bool = field(default=True),
    output_dir: Optional[str] = None,
    calib_num: int = 32,
    hf_token: Optional[str] = None
) -> str:
    """
    使用 GPTQ 進行模型量化。

    Args:
        model_path (str): 模型的路徑或名稱。
        bits (int): 量化位元數，預設為 4。
        group_size (int): 分組大小，預設為 -1（不分組）。
        damp_percent (float): 抑制百分比，預設為 0.01。
        desc_act (bool): 是否啟用描述性激活函數。
        static_groups (bool): 是否使用靜態分組。
        sym (bool): 是否啟用對稱量化。
        true_sequential (bool): 是否啟用順序處理。
        output_dir (Optional[str]): 儲存量化模型的目錄。
        calib_num (int): 校準資料數量，預設為 32。
        hf_token (Optional[str]): Hugging Face 的存取權杖。

    Returns:
        str: 儲存量化模型的目錄。
    """
    model_id = model_path.split("/")[-1].lower().replace("instruct", "it")
    if output_dir is None:
        output_dir = f"../quant_models/{model_id}-gptq-{bits}bit"

    print("🔹 檢查 GPU 是否可用...")
    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU 不可用，請確認是否正確安裝 CUDA 和驅動程式。")

    device = torch.device("cuda")
    print(f"✅ 使用 GPU: {torch.cuda.get_device_name(device)}")

    print("🔹 載入 tokenizer...")
    tokenizer = safe_load_tokenizer(model_path)

    print("🔹 建立 GPTQ 配置...")
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        damp_percent=damp_percent,
        desc_act=desc_act,
        # static_groups=static_groups, (會出錯)
        sym=sym,
        true_sequential=true_sequential
    )

    print("🔹 載入原始模型...")
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
        low_cpu_mem_usage=True, 
        device_map="auto", 
        token=hf_token
    )

    print("🔹 準備校準資料...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_path)
    max_len = config.max_position_embeddings

    samples = dataset.select(range(calib_num))  # 避免太多造成 OOM

    examples = []
    for ex in samples:
        tok = tokenizer(
            ex["question"] + " " + ex["answer"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
        examples.append({
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0)
        })
    
    print("🔹 開始 GPTQ 量化...")
    model.quantize(examples=examples, batch_size=1)

    print("🔹 儲存 GPTQ 量化模型...")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ GPTQ 量化完成，已存到 {output_dir}")
    return output_dir

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"
    MODEL_ID = MODEL_PATH.split("/")[-1].lower().replace("instruct", "it")
    OUTDIR = f"quant_models/{MODEL_ID}-gptq-test"

    gptq_quantization(
        model_path=MODEL_PATH,
        bits=4,
        group_size=128,
        damp_percent=0.01,
        output_dir=OUTDIR,
        calib_num=32,
        hf_token=HF_TOKEN
    )