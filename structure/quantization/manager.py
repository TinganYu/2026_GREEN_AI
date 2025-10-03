from .awq_quantizer import AWQQuantizer
from .bnb_quantizer import BNBQuantizer
from .gptq_quantizer import GPTQQuantizer

class QuantizationManager:
    BACKENDS = {
        "awq": AWQQuantizer,
        "bnb": BNBQuantizer,
        "gptq": GPTQQuantizer
    }

    @staticmethod
    def quantize(model_path: str, backend: str, config, hf_token=None):
        backend = backend.lower()
        if backend not in QuantizationManager.BACKENDS:
            raise ValueError(f"Unsupported backend {backend}. Choose from {list(QuantizationManager.BACKENDS.keys())}")
        quantizer_class = QuantizationManager.BACKENDS[backend]
        quantizer = quantizer_class(model_path, hf_token)
        return quantizer.quantize(config)
