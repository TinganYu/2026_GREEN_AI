"""
GPTQ 量化器
===========

使用 AutoGPTQ 進行後訓練量化
"""

import gc
import torch
from dataclasses import asdict
from .base_quantizer import BaseQuantizer
from .utils import logger

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoConfig
    from datasets import load_dataset
except ImportError:
    AutoGPTQForCausalLM = BaseQuantizeConfig = AutoConfig = load_dataset = None


class GPTQQuantizer(BaseQuantizer):
    """
    GPTQ (Generalized Post-training Quantization) 量化器
    
    特點：
    - 支援 4-bit/8-bit 權重量化
    - 使用校準資料集進行量化
    - 高精度保留
    - 快速推理
    
    依賴：
        pip install auto-gptq datasets
    """
    
    def quantize(self, config):
        """
        執行 GPTQ 量化
        
        Args:
            config: GPTQConfig 配置物件
            
        Returns:
            str: 量化後模型的輸出路徑
            
        Raises:
            ImportError: 如果未安裝 auto-gptq 或 datasets
        """
        if AutoGPTQForCausalLM is None:
            raise ImportError(
                "❌ 請先安裝 auto-gptq 和 datasets 套件\n"
                "   安裝命令: pip install auto-gptq datasets"
            )

        # 設定輸出目錄
        output_dir = config.output_dir or f"quant_models/{self.model_id}-gptq-{config.bits}bit"
        
        logger.info("="*80)
        logger.info("🚀 開始 GPTQ 量化")
        logger.info("="*80)

        # 建立量化配置
        logger.info("🔹 建立 GPTQ 配置...")
        quantize_config = BaseQuantizeConfig(
            bits=config.bits,
            group_size=config.group_size,
            damp_percent=config.damp_percent,
            desc_act=config.desc_act,
            sym=config.sym,
            true_sequential=config.true_sequential
        )
        logger.info(f"   量化位元數: {config.bits}-bit")
        logger.info(f"   分組大小: {config.group_size}")
        
        # 確保 tokenizer 有 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("   ✓ 設定 pad_token = eos_token")

        # 載入原始模型
        logger.info("🔹 載入原始模型...")
        self.cleanup()
        
        model = AutoGPTQForCausalLM.from_pretrained(
            self.model_path,
            quantize_config=quantize_config,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=self.hf_token
        )
        logger.info("   ✓ 模型載入完成")

        # 準備校準資料
        logger.info("🔹 準備校準資料...")
        logger.info(f"   使用 GSM8K 資料集，樣本數: {config.calib_num}")
        
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        dataset = dataset.select(range(config.calib_num))
        
        # 獲取模型最大序列長度
        model_config = AutoConfig.from_pretrained(self.model_path)
        max_len = model_config.max_position_embeddings
        logger.info(f"   最大序列長度: {max_len}")

        # 準備校準樣本
        examples = []
        for ex in dataset:
            text = ex["question"] + " " + ex["answer"]
            tok = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )
            examples.append({
                "input_ids": tok["input_ids"].squeeze(0),
                "attention_mask": tok["attention_mask"].squeeze(0)
            })
        logger.info(f"   ✓ 準備了 {len(examples)} 個校準樣本")

        # 執行量化
        logger.info("🔹 開始 GPTQ 量化...")
        logger.info("   這可能需要較長時間（取決於模型大小和校準樣本數）...")
        model.quantize(examples=examples, batch_size=1)
        logger.info("   ✓ 量化完成")

        # 儲存模型
        logger.info("🔹 儲存量化模型...")
        model.save_quantized(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("="*80)
        logger.info(f"✅ GPTQ 量化成功完成!")
        logger.info(f"📁 輸出路徑: {output_dir}")
        logger.info("="*80)
        
        return output_dir
