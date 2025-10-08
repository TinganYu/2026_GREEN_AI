"""
量化管理器
==========

提供統一的量化接口，支援多種量化後端
"""

from .awq_quantizer import AWQQuantizer
from .bnb_quantizer import BNBQuantizer
from .gptq_quantizer import GPTQQuantizer
from .utils import logger


class QuantizationManager:
    """
    量化管理器
    
    提供統一的接口來使用不同的量化方法
    
    支援的後端：
    - awq: AWQ 量化
    - bnb: BitsAndBytes 量化
    - gptq: GPTQ 量化
    
    Examples:
        >>> from quantization import QuantizationManager, AWQConfig
        >>> config = AWQConfig(w_bit=4, q_group_size=128)
        >>> output_path = QuantizationManager.quantize(
        ...     model_path="meta-llama/Llama-3.2-1B-Instruct",
        ...     backend="awq",
        ...     config=config
        ... )
    """
    
    # 支援的量化後端
    BACKENDS = {
        "awq": AWQQuantizer,
        "bnb": BNBQuantizer,
        "gptq": GPTQQuantizer
    }

    @staticmethod
    def quantize(model_path: str, backend: str, config, hf_token: str = None) -> str:
        """
        執行量化
        
        Args:
            model_path: 模型路徑或 HuggingFace model ID
            backend: 量化後端 ("awq", "bnb", "gptq")
            config: 量化配置 (AWQConfig, BNBConfig, 或 GPTQConfig)
            hf_token: HuggingFace API token（可選）
            
        Returns:
            str: 量化後模型的輸出路徑
            
        Raises:
            ValueError: 如果指定的後端不支援
            
        Examples:
            >>> from quantization import QuantizationManager, GPTQConfig
            >>> config = GPTQConfig(bits=4, group_size=128)
            >>> output = QuantizationManager.quantize(
            ...     model_path="facebook/opt-350m",
            ...     backend="gptq",
            ...     config=config
            ... )
        """
        backend = backend.lower()
        
        # 檢查後端是否支援
        if backend not in QuantizationManager.BACKENDS:
            raise ValueError(
                f"❌ 不支援的量化後端: {backend}\n"
                f"   支援的後端: {list(QuantizationManager.BACKENDS.keys())}"
            )
        
        # 獲取對應的量化器類別
        quantizer_class = QuantizationManager.BACKENDS[backend]
        
        # 創建量化器實例
        logger.info(f"🔧 使用 {backend.upper()} 量化器")
        quantizer = quantizer_class(model_path, hf_token)
        
        # 執行量化
        output_path = quantizer.quantize(config)
        
        return output_path
    
    @staticmethod
    def list_backends():
        """
        列出所有支援的量化後端
        
        Returns:
            list: 支援的後端名稱列表
        """
        return list(QuantizationManager.BACKENDS.keys())
    
    @staticmethod
    def get_backend_info(backend: str) -> dict:
        """
        獲取指定後端的資訊
        
        Args:
            backend: 後端名稱
            
        Returns:
            dict: 包含後端資訊的字典
        """
        backend = backend.lower()
        if backend not in QuantizationManager.BACKENDS:
            raise ValueError(f"未知的後端: {backend}")
        
        quantizer_class = QuantizationManager.BACKENDS[backend]
        
        return {
            "name": backend.upper(),
            "class": quantizer_class.__name__,
            "doc": quantizer_class.__doc__
        }
