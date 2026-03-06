import gc
import logging
import os
from datetime import datetime
from typing import Any

import torch
from datasets import load_dataset
from gptqmodel import GPTQModel
from gptqmodel.quantization.config import QuantizeConfig
from transformers import AutoTokenizer

logger = logging.getLogger("GPTQModelQuantizer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
logger.propagate = False


class GPTQModelQuantizer:
    """Quantizer that directly uses GPTQModel with QuantizeConfig from YAML."""

    def __init__(self, model_path: str, hf_token: str = None):
        self.model_path = model_path
        self.model_id = model_path.split("/")[-1]
        self.hf_token = hf_token

        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available, quantization requires CUDA")
        self.device = torch.device("cuda:0")
        logger.info(f"GPU: {torch.cuda.get_device_name(self.device)}")

    def quantize(
        self,
        quant_config: QuantizeConfig,
        calib_config: dict[str, Any],
        output_dir: str | None = None,
        output_base: str = "quantized",
        mock: bool = False,
    ) -> str:
        """Run quantization with the given QuantizeConfig and calibration settings.

        Args:
            quant_config: GPTQModel QuantizeConfig instance.
            calib_config: Dict with keys: dataset, split, num_samples, text_column,
                          and optionally dataset_config, batch_size.
            output_dir: Explicit output directory. If None, auto-generates.
            output_base: Base directory for auto-generated output paths.
            mock: If True, enable mock quantization (skip actual computation).

        Returns:
            Path to the saved quantized model directory.
        """
        if mock:
            quant_config.mock_quantization = True
            logger.info("Mock mode enabled - skipping actual quantization computation")

        # Resolve output directory
        if output_dir is None:
            output_dir = self._generate_output_dir(quant_config, output_base)
        os.makedirs(output_dir, exist_ok=True)

        logger.info("=" * 70)
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"QuantizeConfig: {quant_config}")
        logger.info("=" * 70)

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = self._load_tokenizer()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        logger.info("Loading model...")
        self._cleanup_memory()
        model = GPTQModel.load(self.model_path, quant_config)
        logger.info("Model loaded")

        # Prepare calibration data
        logger.info("Preparing calibration data...")
        calib_data = self._prepare_calibration_data(calib_config)
        logger.info(f"Prepared {len(calib_data)} calibration samples")

        # Reset GPU stats for peak tracking
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Quantize
        batch_size = calib_config.get("batch_size", 1)
        logger.info(f"Starting quantization (batch_size={batch_size})...")
        model.quantize(calib_data, batch_size=batch_size)

        # Track GPU peak
        peak_memory_mb = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logger.info(f"GPU peak memory during quantization: {peak_memory_mb:.2f} MB")

        logger.info("Quantization complete")

        # Save
        logger.info("Saving quantized model...")
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved to {output_dir}")

        # Cleanup
        del model
        self._cleanup_memory()

        logger.info("=" * 70)
        logger.info(f"Done. Output: {output_dir}")
        if peak_memory_mb is not None:
            logger.info(f"GPU peak: {peak_memory_mb:.2f} MB")
        logger.info("=" * 70)

        return output_dir

    def _load_tokenizer(self) -> AutoTokenizer:
        try:
            return AutoTokenizer.from_pretrained(
                self.model_path, use_fast=True, token=self.hf_token
            )
        except Exception:
            logger.warning("Fast tokenizer failed, falling back to slow tokenizer")
            return AutoTokenizer.from_pretrained(
                self.model_path, use_fast=False, token=self.hf_token
            )

    def _prepare_calibration_data(self, calib_config: dict[str, Any]) -> list[str]:
        """Load calibration dataset and convert to list of strings.

        Supports 'text_column: "question+answer"' syntax to concatenate
        multiple columns with a space separator.
        """
        dataset_name = calib_config["dataset"]
        split = calib_config.get("split", "train")
        num_samples = calib_config.get("num_samples", 256)
        text_column = calib_config.get("text_column", "text")
        dataset_config_name = calib_config.get("dataset_config", None)

        ds = load_dataset(dataset_name, dataset_config_name, split=split)
        ds = ds.select(range(min(num_samples, len(ds))))

        # Parse text_column: "col1+col2" → concatenate columns
        columns = [c.strip() for c in text_column.split("+")]

        data = []
        for example in ds:
            parts = []
            for col in columns:
                val = example.get(col, "")
                if val:
                    parts.append(str(val))
            data.append(" ".join(parts))

        return data

    def _generate_output_dir(self, quant_config: QuantizeConfig, output_base: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        method = quant_config.quant_method.value if hasattr(quant_config.quant_method, 'value') else str(quant_config.quant_method)
        bits = quant_config.bits
        return os.path.join(output_base, f"{self.model_id}-{method}-{bits}bit-{ts}")

    @staticmethod
    def _cleanup_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
