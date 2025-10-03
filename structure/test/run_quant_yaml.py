from pathlib import Path
import yaml
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantization.configs import AWQConfig, BNBConfig, GPTQConfig
from quantization.manager import QuantizationManager

# ----------------------------
# 讀取 YAML 配置
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
config_path = BASE_DIR / "quant_config.yaml"

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# 模型參數
model_path = cfg["model"]["name"]
hf_token = cfg.get("hf_token")

# 量化配置
quant_section = cfg.get("quantization", {})
backend = quant_section.get("framework", "").lower()

if backend == "awq":
    quant_cfg = AWQConfig(**quant_section["awq"])
elif backend == "bnb":
    quant_cfg = BNBConfig(**quant_section["bnb"])
elif backend == "gptq":
    quant_cfg = GPTQConfig(**quant_section["gptq"])
else:
    raise ValueError(f"Unsupported quantization backend: {backend}")

# ----------------------------
# 執行量化
# ----------------------------
QuantizationManager.quantize(
    model_path=model_path,
    backend=backend,
    config=quant_cfg,
    hf_token=hf_token,
)
