"""
AWQ 量化器
==========

使用 AutoAWQ 進行激活感知權重量化
"""

import gc
import torch
from dataclasses import asdict
from .base_quantizer import BaseQuantizer
from .utils import logger

try:
    from awq import AutoAWQForCausalLM
except ImportError:
    AutoAWQForCausalLM = None


class AWQQuantizer(BaseQuantizer):
    """
    AWQ (Activation-aware Weight Quantization) 量化器
    
    特點：
    - 支援 4-bit 權重量化
    - 保留激活分布以提高精度
    - 快速推理速度
    
    依賴：
        pip install autoawq
    """
    
    def quantize(self, config):
        """
        執行 AWQ 量化
        
        Args:
            config: AWQConfig 配置物件
            
        Returns:
            str: 量化後模型的輸出路徑
            
        Raises:
            ImportError: 如果未安裝 autoawq
        """
        if AutoAWQForCausalLM is None:
            raise ImportError(
                "❌ 請先安裝 autoawq 套件\n"
                "   安裝命令: pip install autoawq"
            )

        # 設定輸出目錄
        output_dir = config.output_dir or f"quant_models/{self.model_id}-awq-{config.w_bit}bit"
        
        logger.info("="*80)
        logger.info("� 開始 AWQ 量化")
        logger.info("="*80)
        
        # 建立量化配置
        logger.info("�🔹 建立 AWQ 配置...")
        quant_config = asdict(config)
        quant_config.pop("output_dir", None)  # 移除 output_dir
        logger.info(f"   配置: {quant_config}")

        # 載入原始模型
        logger.info("🔹 載入原始模型...")
        self.cleanup()
        
        model = AutoAWQForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=self.hf_token
        )
        logger.info("   ✓ 模型載入完成")

        # 執行量化
        logger.info("🔹 開始 AWQ 量化...")
        logger.info("   這可能需要幾分鐘時間...")
        model.quantize(self.tokenizer, quant_config=quant_config)
        logger.info("   ✓ 量化完成")

        # 儲存模型
        logger.info("🔹 儲存量化模型...")
        model.save_quantized(output_dir, safetensors=True)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("="*80)
        logger.info(f"✅ AWQ 量化成功完成!")
        logger.info(f"📁 輸出路徑: {output_dir}")
        logger.info("="*80)
        
        return output_dir
