from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
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

def gptq_quantization(model_path, model_id, bits=4, group_size=128, damp_percent=0.01, desc_act=False, output_dir=None, calib_num=32, hf_token=None):
    if output_dir is None:
        output_dir = f"../quant_models/{model_id}-bnb-{bits}bit"

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
        true_sequential=True
    )

    print("🔹 載入原始模型...")
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        quantize_config=quantize_config,
        low_cpu_mem_usage=True, 
        device_map="auto", 
        use_auth_token=hf_token
    )

    print("🔹 準備校準資料...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_path)
    max_len = config.max_position_embeddings

    dataset = load_dataset("openai/gsm8k", "main", split="train")
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

    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    MODEL_PATH = "facebook/opt-350m"
    MODEL_ID = "opt-350m"
    OUTDIR = f"../quant_models/{MODEL_ID}-gptq-8bit"

    gptq_quantization(MODEL_PATH, MODEL_ID, bits=8, group_size=128, output_dir=OUTDIR, hf_token=HF_TOKEN)