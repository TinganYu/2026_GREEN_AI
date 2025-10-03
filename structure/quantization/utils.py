import logging
import json
from transformers import AutoTokenizer
from typing import Any

# ----------------------------
# Logger
# ----------------------------
logger = logging.getLogger("Quantization")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# ----------------------------
# Safe load tokenizer
# ----------------------------
def safe_load_tokenizer(model_path: str) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    except Exception as e:
        logger.warning(f"fast tokenizer 載入失敗，改用 slow 版本: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    return tokenizer