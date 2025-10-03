import gc
import torch
from .base_quantizer import BaseQuantizer
from .utils import logger
from dataclasses import asdict

try:
    from awq import AutoAWQForCausalLM
except ImportError:
    AutoAWQForCausalLM = None

class AWQQuantizer(BaseQuantizer):
    def quantize(self, config):
        if AutoAWQForCausalLM is None:
            raise ImportError("請先安裝 awq 套件")

        output_dir = config.output_dir or f"quant_models/{self.model_id}-awq-{config.bits}bit"
        logger.info("🔹 建立 AWQ 配置...")
        quant_config = asdict(config)
        quant_config.pop("output_dir")

        logger.info("🔹 載入原始模型...")
        gc.collect()
        torch.cuda.empty_cache()
        model = AutoAWQForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=self.hf_token
        )

        logger.info("🔹 開始 AWQ 量化...")
        model.quantize(self.tokenizer, quant_config=quant_config)

        logger.info("🔹 儲存模型與 tokenizer...")
        model.save_quantized(output_dir, safetensors=True)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"✅ AWQ 量化完成: {output_dir}")
        return output_dir
