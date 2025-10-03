import gc

import torch
from .base_quantizer import BaseQuantizer
from .utils import logger

try:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
except ImportError:
    AutoModelForCausalLM = BitsAndBytesConfig = None

class BNBQuantizer(BaseQuantizer):
    def quantize(self, config):
        if BitsAndBytesConfig is None:
            raise ImportError("請先安裝 transformers bitsandbytes 套件")

        output_dir = config.output_dir or f"quant_models/{self.model_id}-bnb-{config.bits}bit"
        logger.info("🔹 建立 BitsAndBytes 配置...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=(config.bits==4),
            load_in_8bit=(config.bits==8),
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
            llm_int8_threshold=config.llm_int8_threshold,
            llm_int8_skip_modules=config.llm_int8_skip_modules,
            llm_int8_has_fp16_weight=config.llm_int8_has_fp16_weight,
            llm_int8_enable_fp32_cpu_offload=config.llm_int8_enable_fp32_cpu_offload
        )

        logger.info("🔹 載入原始模型...")
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=self.hf_token
        ).to(self.device)

        logger.info("🔹 儲存模型與 tokenizer...")
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"✅ BNB 量化完成: {output_dir}")
        return output_dir
