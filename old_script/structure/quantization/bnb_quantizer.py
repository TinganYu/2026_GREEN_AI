"""
BitsAndBytes 量化器
==================

使用 BitsAndBytes 進行 4-bit/8-bit 量化
"""

import gc
import torch
from .base_quantizer import BaseQuantizer
from .utils import logger

try:
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
except ImportError:
    AutoModelForCausalLM = BitsAndBytesConfig = None


class BNBQuantizer(BaseQuantizer):
    """
    BitsAndBytes 量化器
    
    特點：
    - 支援 4-bit 和 8-bit 量化
    - 整合於 Transformers 庫
    - 支援混合精度訓練
    - NF4 (NormalFloat4) 量化類型
    
    依賴：
        pip install transformers bitsandbytes
    """
    
    def quantize(self, config):
        """
        執行 BitsAndBytes 量化
        
        Args:
            config: BNBConfig 配置物件
            
        Returns:
            str: 量化後模型的輸出路徑
            
        Raises:
            ImportError: 如果未安裝 transformers 或 bitsandbytes
        """
        if BitsAndBytesConfig is None:
            raise ImportError(
                "❌ 請先安裝 transformers 和 bitsandbytes 套件\n"
                "   安裝命令: pip install transformers bitsandbytes"
            )

        # 設定輸出目錄
        output_dir = config.output_dir or f"quant_models/{self.model_id}-bnb-{config.bits}bit"
        
        logger.info("="*80)
        logger.info("� 開始 BitsAndBytes 量化")
        logger.info("="*80)
        
        # 建立量化配置
        logger.info("�🔹 建立 BitsAndBytes 配置...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=(config.bits == 4),
            load_in_8bit=(config.bits == 8),
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            llm_int8_threshold=config.llm_int8_threshold,
            llm_int8_skip_modules=config.llm_int8_skip_modules,
            llm_int8_has_fp16_weight=config.llm_int8_has_fp16_weight,
            llm_int8_enable_fp32_cpu_offload=config.llm_int8_enable_fp32_cpu_offload
        )
        logger.info(f"   量化位元數: {config.bits}-bit")
        logger.info(f"   量化類型: {config.bnb_4bit_quant_type}")

        # 載入並量化模型
        logger.info("🔹 載入並量化模型...")
        self.cleanup()
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=self.hf_token
        )
        logger.info("   ✓ 模型載入與量化完成")

        # 儲存模型
        logger.info("🔹 儲存量化模型...")
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("="*80)
        logger.info(f"✅ BitsAndBytes 量化成功完成!")
        logger.info(f"📁 輸出路徑: {output_dir}")
        logger.info("="*80)
        
        return output_dir
