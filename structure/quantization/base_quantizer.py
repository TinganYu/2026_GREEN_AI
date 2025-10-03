import torch
import gc
from .utils import safe_load_tokenizer, logger

class BaseQuantizer:
    def __init__(self, model_path: str, hf_token: str = None):
        self.model_path = model_path
        self.model_id = model_path.split("/")[-1].lower().replace("instruct", "it")
        self.hf_token = hf_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise RuntimeError("❌ GPU 不可用")
        logger.info(f"✅ 使用 GPU: {torch.cuda.get_device_name(self.device)}")
        self.tokenizer = safe_load_tokenizer(model_path)

    def quantize(self, config):
        raise NotImplementedError("子類必須實作 quantize 方法")
