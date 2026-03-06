import argparse
import logging
import os
import sys

from dotenv import load_dotenv

# Allow running from project root or from this directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.gptqmodel_methods.config_loader import (
    build_quantize_config,
    load_experiment_config,
)
from experiments.gptqmodel_methods.quantize import GPTQModelQuantizer

logger = logging.getLogger("RunExperiment")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
logger.propagate = False


def main():
    parser = argparse.ArgumentParser(
        description="Run GPTQModel quantization experiments from YAML config"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment YAML config file"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Enable mock quantization (skip actual computation for quick validation)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level)
    logging.getLogger("ExperimentConfigLoader").setLevel(log_level)
    logging.getLogger("GPTQModelQuantizer").setLevel(log_level)
    logger.setLevel(log_level)

    # Load .env for HF_TOKEN
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not found in environment. Private models may fail to load.")

    # Load config
    config = load_experiment_config(args.config)
    model_config = config["model"]
    calib_config = config["calibration"]
    quant_section = config["quantization"]

    # Build QuantizeConfig
    quant_config = build_quantize_config(quant_section)

    # Resolve output directory
    output_dir = args.output_dir or model_config.get("output_dir")
    output_base = model_config.get("output_base", "quantized")

    # Run quantization
    quantizer = GPTQModelQuantizer(
        model_path=model_config["name"],
        hf_token=hf_token,
    )

    result_dir = quantizer.quantize(
        quant_config=quant_config,
        calib_config=calib_config,
        output_dir=output_dir,
        output_base=output_base,
        mock=args.mock,
    )

    logger.info(f"Experiment complete. Model saved to: {result_dir}")


if __name__ == "__main__":
    main()
