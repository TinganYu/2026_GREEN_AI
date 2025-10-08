"""
基礎量化器
==========

所有量化器的抽象基類
"""

import torch
import gc
from abc import ABC, abstractmethod
from .utils import safe_load_tokenizer, logger


class BaseQuantizer(ABC):
    """
    量化器基類
    
    所有具體的量化器（AWQ, BNB, GPTQ）都繼承自此類
    
    Attributes:
        model_path: 模型路徑或 HuggingFace model ID
        model_id: 從 model_path 提取的模型標識符
        hf_token: HuggingFace API token（可選）
        device: 使用的設備（CUDA 或 CPU）
        tokenizer: 載入的 tokenizer
    """
    
    def __init__(self, model_path: str, hf_token: str = None):
        """
        初始化量化器
        
        Args:
            model_path: 模型路徑或 HuggingFace model ID
            hf_token: HuggingFace API token（用於訪問私有模型）
            
        Raises:
            RuntimeError: 如果 GPU 不可用
        """
        self.model_path = model_path
        self.model_id = model_path.split("/")[-1].lower().replace("instruct", "it")
        self.hf_token = hf_token
        
        # 檢查 GPU 可用性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("❌ GPU 不可用，量化需要 GPU 支援")
        
        logger.info(f"✅ 使用 GPU: {torch.cuda.get_device_name(self.device)}")
        
        # 載入 tokenizer
        self.tokenizer = safe_load_tokenizer(model_path, token=hf_token)
        logger.info(f"✅ Tokenizer 載入完成")
    
    @abstractmethod
    def quantize(self, config):
        """
        執行量化（抽象方法）
        
        子類必須實現此方法以執行具體的量化操作
        
        Args:
            config: 量化配置（AWQConfig, BNBConfig, 或 GPTQConfig）
            
        Returns:
            str: 量化後模型的輸出路徑
        """
        raise NotImplementedError("子類必須實作 quantize 方法")
    
    def cleanup(self):
        """清理 GPU 記憶體"""
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("🧹 記憶體清理完成")
