import logging
from pathlib import Path
from typing import Any

import torch
import yaml
from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig

logger = logging.getLogger("ExperimentConfigLoader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
logger.propagate = False

METHOD_MAP = {
    "gptq": METHOD.GPTQ,
    "awq": METHOD.AWQ,
    "qqq": METHOD.QQQ,
}

FORMAT_MAP = {
    "gptq": FORMAT.GPTQ,
    "gptq_v2": FORMAT.GPTQ_V2,
    "marlin": FORMAT.MARLIN,
    "bitblas": FORMAT.BITBLAS,
    "qqq": FORMAT.QQQ,
    "gemm": FORMAT.GEMM,
    "gemv": FORMAT.GEMV,
    "gemv_fast": FORMAT.GEMV_FAST,
    "llm_awq": FORMAT.LLM_AWQ,
}

PACK_DTYPE_MAP = {
    "int32": torch.int32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    """Load and parse experiment YAML config file.

    Supports two formats:
    1. Flat format (original): all quantization params at top level
    2. Methods format: params nested under quantization.methods.<method_name>

    Returns dict with three keys: 'model', 'calibration', 'quantization'.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    required_sections = ["model", "calibration", "quantization"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config")

    # Support methods: structure — flatten selected method's params into quantization
    quant = config["quantization"]
    if "methods" in quant:
        methods = quant.pop("methods")
        method_name = quant.get("quant_method")
        if method_name is None:
            raise ValueError("quantization.quant_method must be set when using quantization.methods")
        if method_name not in methods:
            raise ValueError(
                f"quant_method '{method_name}' not found in quantization.methods. "
                f"Available: {list(methods.keys())}"
            )
        # Method-specific params override top-level params
        method_params = methods[method_name] or {}
        quant.update(method_params)
        logger.info(f"Using method '{method_name}' params from quantization.methods")

    logger.info(f"Loaded config from {config_path}")
    return config


def build_quantize_config(quant_section: dict[str, Any]) -> QuantizeConfig:
    """Convert the 'quantization' section of YAML to a GPTQModel QuantizeConfig.

    String values are mapped to GPTQModel enums. None values are removed
    so that QuantizeConfig uses its own defaults.
    """
    kwargs = dict(quant_section)

    # Map string → METHOD enum
    if "quant_method" in kwargs and kwargs["quant_method"] is not None:
        method_str = kwargs["quant_method"].lower()
        if method_str not in METHOD_MAP:
            raise ValueError(
                f"Unknown quant_method '{method_str}'. "
                f"Available: {list(METHOD_MAP.keys())}"
            )
        kwargs["quant_method"] = METHOD_MAP[method_str]

    # Map string → FORMAT enum
    if "format" in kwargs and kwargs["format"] is not None:
        format_str = kwargs["format"].lower()
        if format_str not in FORMAT_MAP:
            raise ValueError(
                f"Unknown format '{format_str}'. "
                f"Available: {list(FORMAT_MAP.keys())}"
            )
        kwargs["format"] = FORMAT_MAP[format_str]

    # Map string → torch dtype for pack_dtype
    if "pack_dtype" in kwargs and kwargs["pack_dtype"] is not None:
        dtype_str = kwargs["pack_dtype"].lower()
        if dtype_str not in PACK_DTYPE_MAP:
            raise ValueError(
                f"Unknown pack_dtype '{dtype_str}'. "
                f"Available: {list(PACK_DTYPE_MAP.keys())}"
            )
        kwargs["pack_dtype"] = PACK_DTYPE_MAP[dtype_str]

    # Remove None values so QuantizeConfig uses its defaults
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    logger.info(f"Building QuantizeConfig with: {_summarize_kwargs(kwargs)}")

    return QuantizeConfig(**kwargs)


def _summarize_kwargs(kwargs: dict) -> dict:
    """Create a summary of kwargs for logging, truncating large values."""
    summary = {}
    for k, v in kwargs.items():
        if isinstance(v, dict) and len(str(v)) > 100:
            summary[k] = f"<dict with {len(v)} keys>"
        else:
            summary[k] = v
    return summary
