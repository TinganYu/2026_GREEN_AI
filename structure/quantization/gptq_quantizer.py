import gc
import torch
from .base_quantizer import BaseQuantizer
from .utils import logger
from dataclasses import asdict

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoConfig
    from datasets import load_dataset
except ImportError:
    AutoGPTQForCausalLM = BaseQuantizeConfig = AutoConfig = load_dataset = None

class GPTQQuantizer(BaseQuantizer):
    def quantize(self, config):
        if AutoGPTQForCausalLM is None:
            raise ImportError("請先安裝 auto_gptq 套件")

        output_dir = config.output_dir or f"quant_models/{self.model_id}-gptq-{config.bits}bit"

        logger.info("🔹 建立 GPTQ 配置...")
        quantize_config = BaseQuantizeConfig(
            bits=config.bits,
            group_size=config.group_size,
            damp_percent=config.damp_percent,
            desc_act=config.desc_act,
            sym=config.sym,
            true_sequential=config.true_sequential
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("🔹 載入原始模型...")
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoGPTQForCausalLM.from_pretrained(
            self.model_path,
            quantize_config=quantize_config,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=self.hf_token
        )

        logger.info("🔹 準備校準資料...")
        dataset = load_dataset("openai/gsm8k", "main", split="train").select(range(config.calib_num))
        max_len = AutoConfig.from_pretrained(self.model_path).max_position_embeddings

        examples = []
        for ex in dataset:
            tok = self.tokenizer(ex["question"] + " " + ex["answer"],
                                 truncation=True,
                                 padding="max_length",
                                 max_length=max_len,
                                 return_tensors="pt")
            examples.append({"input_ids": tok["input_ids"].squeeze(0),
                             "attention_mask": tok["attention_mask"].squeeze(0)})

        logger.info("🔹 開始 GPTQ 量化...")
        model.quantize(examples=examples, batch_size=1)

        logger.info("🔹 儲存模型與 tokenizer...")
        model.save_quantized(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"✅ GPTQ 量化完成: {output_dir}")
        return output_dir
