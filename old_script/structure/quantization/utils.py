"""
工具函數
========

提供通用的工具函數和日誌配置
"""

import logging
from transformers import AutoTokenizer
from typing import Any


# ============================================================================
# Logger 配置
# ============================================================================

logger = logging.getLogger("Quantization")
logger.setLevel(logging.INFO)

# 避免重複添加 handler
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# ============================================================================
# Tokenizer 載入
# ============================================================================

def safe_load_tokenizer(model_path: str, **kwargs) -> AutoTokenizer:
    """
    安全載入 tokenizer，自動處理 fast/slow tokenizer 載入失敗的問題
    
    Args:
        model_path: 模型路徑或 HuggingFace model ID
        **kwargs: 傳遞給 AutoTokenizer.from_pretrained 的額外參數
        
    Returns:
        AutoTokenizer: 載入的 tokenizer
        
    Examples:
        >>> tokenizer = safe_load_tokenizer("meta-llama/Llama-3.2-1B-Instruct")
        >>> tokenizer = safe_load_tokenizer("./local_model", trust_remote_code=True)
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **kwargs)
        logger.debug(f"✅ 成功載入 fast tokenizer: {model_path}")
    except Exception as e:
        logger.warning(f"⚠️ Fast tokenizer 載入失敗，改用 slow 版本: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, **kwargs)
            logger.debug(f"✅ 成功載入 slow tokenizer: {model_path}")
        except Exception as e2:
            logger.error(f"❌ Tokenizer 載入失敗: {e2}")
            raise
    
    return tokenizer