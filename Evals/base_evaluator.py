"""
基礎評估器（Global_Tuner_v2 版本）
=====================================

相對於 tmp/evals/base_evaluator.py 的主要修改：
1. gptq / awq / qqq 統一使用 GPTQModel.from_quantized() 載入
2. 移除 autoawq 依賴（_load_awq_model）
3. config_file 預設指向 Evals/config/dataset_config.yaml
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging
import os
import sys
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, GenerationConfig

# 確保能找到 Evals/config/
sys.path.insert(0, str(Path(__file__).parent))

from config.dataset_config_manager import DatasetConfigManager


logger = logging.getLogger("BaseEvaluator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
logger.propagate = False

# 預設 dataset_config.yaml 位置（Global_Tuner_v2/Evals/config/）
_DEFAULT_DATASET_CONFIG = str(
    Path(__file__).parent / "config" / "dataset_config.yaml"
)


@dataclass
class BaseEvalConfig:
    """基礎評估配置"""
    model_path: str
    output_dir: str = "output"

    max_new_tokens: int = 1024
    do_sample: bool = False
    batch_size: int = 1
    temperature: float = 0.0
    top_p: float = 1.0

    num_samples: Optional[int] = None
    dataset_name: Optional[str] = None
    dataset_split: str = "test"
    prompt_type: str = "direct"
    prompts: Dict[str, str] = None
    fewshot_examples: Optional[List[Any]] = None

    hf_token: Optional[str] = None
    device_map: str = "cuda:0"
    dtype: str = "float16"
    trust_remote_code: bool = True
    use_safetensors: bool = True


class BaseEvaluator(ABC):
    """評估器基類（Global_Tuner_v2 版本）"""

    def __init__(self, config: Union[Dict, BaseEvalConfig], dataset_name: Optional[str]):
        if isinstance(config, dict):
            self.config = self._load_from_dict(config, dataset_name)
        elif isinstance(config, BaseEvalConfig):
            self.config = config
        else:
            raise ValueError(f"不支援的配置類型: {type(config)}")

        self.model = None
        self.tokenizer = None
        self.generator = None
        self.results = []
        self.dataset_name = dataset_name
        self._energy_tracker = None
        self._use_vllm = False

        logger.info(f"評估配置已載入: {self.config.model_path}")

    def _load_from_dict(self, cfg: Dict, dataset_name: str) -> BaseEvalConfig:
        try:
            model_cfg = cfg["model"]
            dataset_cfg = cfg["dataset"]

            config_file = dataset_cfg.get("config_file", _DEFAULT_DATASET_CONFIG)
            logger.info(f"從 {config_file} 載入 {dataset_name} 配置")

            self.dataset_manager = DatasetConfigManager(config_file)
            model_path = model_cfg["name"]
            dataset_dict = self.dataset_manager.get_dataset_config(dataset_name)
            dataset_dict["model_path"] = model_path

            override = dataset_cfg.get("override", {})
            if override:
                logger.info(f"套用覆寫參數: {list(override.keys())}")
                dataset_dict.update(override)

            config_class = getattr(self, "ConfigClass", BaseEvalConfig)
            return config_class(**dataset_dict)
        except Exception as e:
            raise ValueError(f"從資料集配置載入配置失敗: {e}")

    @abstractmethod
    def build_prompt(self, sample: Dict[str, Any]) -> str:
        pass

    @abstractmethod
    def extract_answer(self, output: str) -> Any:
        pass

    @abstractmethod
    def check_answer(self, true_answer: Any, predicted_answer: Any) -> bool:
        pass

    def load_model(self, quantization_type: Optional[str] = None):
        """
        載入模型和 tokenizer。

        quantization_type:
            "gptq" | "awq" | "qqq"  → GPTQModel.from_quantized()
            "bnb"                    → BitsAndBytesConfig
            "llmcompressor"          → llm-compressor / sparse
            None                     → 普通模型
        """
        logger.info(f"載入 tokenizer: {self.config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            token=self.config.hf_token
        )
        if self.tokenizer.pad_token is None:
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                logger.info("設定 pad_token = unk_token")
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.warning("pad_token 設為 eos_token")

        self._use_vllm = False

        if quantization_type in ("gptq", "awq", "qqq"):
            self._load_gptqmodel_model()
        elif quantization_type == "bnb":
            self._load_bnb_model()
        elif quantization_type == "llmcompressor":
            self._load_llmcompressor_model()
        elif quantization_type == "sparse_bitmask":
            self._load_sparse_only_model()
        else:
            self._load_normal_model()

        if not self._use_vllm:
            self._setup_generation_pipeline()

    def _load_normal_model(self):
        """載入普通（未量化）模型"""
        logger.info("載入普通模型...")
        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(self.config.dtype, torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            device_map=self.config.device_map,
            dtype=dtype,
            trust_remote_code=self.config.trust_remote_code,
            token=self.config.hf_token
        )

    def _load_gptqmodel_model(self):
        """
        統一使用 GPTQModel.from_quantized() 載入 gptq / awq / qqq 模型。
        GPTQModel 會自動從 config.json 偵測量化類型。
        """
        logger.info("載入量化模型（GPTQModel）...")
        try:
            from gptqmodel import GPTQModel
        except ImportError:
            raise ImportError("請安裝 gptqmodel: pip install gptqmodel")

        try:
            self.model = GPTQModel.from_quantized(
                self.config.model_path,
                device_map={"": self.config.device_map},
                trust_remote_code=self.config.trust_remote_code,
                use_safetensors=self.config.use_safetensors,
            )
        except Exception as e:
            logger.error(f"GPTQModel 載入失敗: {e}")
            raise

        logger.info(f"GPTQModel 載入成功: {type(self.model).__name__}, device={self.model.device}")

    def _load_bnb_model(self):
        """載入 BitsAndBytes (BNB) 量化模型"""
        logger.info("載入 BitsAndBytes (BNB) 量化模型...")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError("請安裝 transformers 支援 BitsAndBytes")

        bnb_config = BitsAndBytesConfig(**self.get_quantization_config())
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            device_map={"": self.config.device_map},
            quantization_config=bnb_config,
            trust_remote_code=self.config.trust_remote_code,
            token=self.config.hf_token
        )
        logger.info(f"BNB 模型載入成功: device={self.model.device}")

    def _is_sparse_only_model(self) -> bool:
        """檢查是否為純稀疏模型（沒有量化）"""
        model_path = Path(self.config.model_path)
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                quant_config = config.get("quantization_config", {})
                has_quantization = "config_groups" in quant_config
                has_sparsity = "sparsity_config" in quant_config
                if has_sparsity and not has_quantization:
                    return True
            except Exception:
                pass

        recipe_path = model_path / "recipe.yaml"
        if recipe_path.exists():
            try:
                with open(recipe_path, "r") as f:
                    recipe = yaml.safe_load(f)
                has_sparse = False
                has_quant = False
                for stage_content in recipe.values():
                    if not isinstance(stage_content, dict):
                        continue
                    modifiers = stage_content.get("default_modifiers", {})
                    if not isinstance(modifiers, dict):
                        continue
                    for name in modifiers.keys():
                        lower = name.lower()
                        if "sparse" in lower or "prune" in lower:
                            has_sparse = True
                        if "quant" in lower or "gptq" in lower or "w8a8" in lower:
                            has_quant = True
                if has_sparse and not has_quant:
                    logger.info("從 recipe.yaml 偵測到純稀疏模型")
                    return True
            except Exception:
                pass
        return False

    def _load_llmcompressor_model(self):
        """載入 llm-compressor 模型（compressed-tensors 格式）"""
        if self._is_sparse_only_model():
            logger.info("偵測到純稀疏模型，使用 transformers 載入...")
            self._load_sparse_only_model()
            return

        try:
            import vllm
            logger.info("偵測到 vLLM，使用 vLLM 載入...")
            self._load_llmcompressor_model_vllm()
            return
        except ImportError:
            logger.warning("vLLM 未安裝，嘗試使用 transformers 載入")

        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(self.config.dtype, torch.float16)
        self._fix_llmcompressor_config()

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=dtype,
                device_map={"": self.config.device_map},
                trust_remote_code=self.config.trust_remote_code,
                low_cpu_mem_usage=True,
                token=self.config.hf_token,
                _fast_init=False,
                ignore_mismatched_sizes=True
            )
            logger.info("llm-compressor 模型載入成功")
        except Exception as e:
            raise RuntimeError(
                f"compressed-tensors 格式需要 vLLM 進行推理。\n"
                f"請安裝 vLLM 或設置 save_compressed: false。\n"
                f"原始錯誤: {e}"
            )

    def _fix_llmcompressor_config(self):
        """修復 llm-compressor 產生的非標準 config.json"""
        config_path = Path(self.config.model_path) / "config.json"
        if not config_path.exists():
            return
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            if "dtype" in config_dict and "quantization_config" in config_dict:
                logger.info("修復 config.json: 移除非標準 'dtype' 欄位")
                del config_dict["dtype"]
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=2)
        except Exception as e:
            logger.warning(f"無法修復 config.json: {e}")

    def _load_sparse_only_model(self):
        """載入純稀疏模型（transformers）。
        sparse-24-bitmask 模型須加 tie_word_embeddings=False：
        compressed-tensors 在解壓縮時會暫時替換 embed_tokens 的 weight，
        導致 tie_weights() 找不到 .weight 而 crash。"""
        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        dtype = dtype_map.get(self.config.dtype, torch.float16)
        self._fix_llmcompressor_config()
        logger.info(f"載入純稀疏模型 (dtype={dtype})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=dtype,
            device_map={"": self.config.device_map},
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=True,
            token=self.config.hf_token,
            tie_word_embeddings=False,
        )
        logger.info(f"純稀疏模型載入成功: {type(self.model).__name__}")

    def _load_llmcompressor_model_vllm(self):
        """使用 vLLM 載入 llm-compressor 模型"""
        from vllm import LLM, SamplingParams
        logger.info("使用 vLLM 載入 llm-compressor 模型...")
        self.model = LLM(
            model=self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
            gpu_memory_utilization=0.6,
            max_model_len=4096,
            disable_log_stats=True,
        )
        self._use_vllm = True
        self._vllm_sampling_params_cls = SamplingParams
        logger.info("vLLM 模型載入成功")

    def _setup_generation_pipeline(self):
        """創建 TextGenerationPipeline"""
        logger.info("創建生成 pipeline...")
        if not hasattr(self.model, "can_generate"):
            self.model.can_generate = lambda: True

        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is None:
            gen_cfg = GenerationConfig()
            self.model.generation_config = gen_cfg
        if getattr(gen_cfg, "max_length", None) is not None:
            gen_cfg.max_length = None

        self.generator = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)
        logger.info("模型生成管線初始化完成")

    def generate(self, prompt: str, **kwargs) -> str:
        """統一文字生成介面（支援 transformers pipeline 和 vLLM）"""
        if self._use_vllm:
            return self._generate_vllm(prompt, **kwargs)
        return self._generate_transformers(prompt, **kwargs)

    def _generate_transformers(self, prompt: str, **kwargs) -> str:
        if self.generator is None:
            raise RuntimeError("Generator 未初始化，請先呼叫 load_model()")
        gen_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        gen_kwargs["return_full_text"] = False
        output = self.generator(prompt, **gen_kwargs)
        return output[0]["generated_text"]

    def _generate_vllm(self, prompt: str, **kwargs) -> str:
        sampling_params = self._vllm_sampling_params_cls(
            max_tokens=kwargs.get("max_new_tokens", 256),
            temperature=kwargs.get("temperature", 0.0) if kwargs.get("do_sample", False) else 0.0,
            top_p=kwargs.get("top_p", 1.0) if kwargs.get("do_sample", False) else 1.0,
        )
        outputs = self.model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def is_model_loaded(self) -> bool:
        if self._use_vllm:
            return self.model is not None
        return self.generator is not None

    def get_quantization_config(self) -> Optional[Dict[str, Any]]:
        config_path = Path(self.config.model_path) / "config.json"
        if not config_path.exists():
            return None
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as exc:
            logger.warning(f"讀取量化配置失敗: {exc}")
            return None
        return config_data.get("quantization_config")

    def _start_energy_tracking(self):
        try:
            from codecarbon import EmissionsTracker
            # 讓 CodeCarbon log 寫到 trial 目錄（exp_dir/model_name/）
            model_name = Path(self.config.model_path).name
            tracker_output_dir = os.path.join(self.config.output_dir, model_name)
            os.makedirs(tracker_output_dir, exist_ok=True)
            self._energy_tracker = EmissionsTracker(
                log_level="warning", tracking_mode="process",
                output_dir=tracker_output_dir,
            )
            self._energy_tracker.start()
            logger.info("CodeCarbon 能耗追蹤已啟動")
        except ImportError:
            self._energy_tracker = None
            logger.warning("codecarbon 未安裝，跳過能耗追蹤")
        except Exception as e:
            self._energy_tracker = None
            logger.warning(f"CodeCarbon 啟動失敗: {e}")

    def _stop_energy_tracking(self) -> Dict[str, Any]:
        if self._energy_tracker is None:
            return {}
        try:
            emissions_kg = self._energy_tracker.stop()
            energy_kwh = self._energy_tracker._total_energy.kWh
            return {
                "energy_consumed_kwh": round(energy_kwh, 8),
                "emissions_kg_co2": round(emissions_kg, 8),
            }
        except Exception as e:
            logger.warning(f"CodeCarbon 停止失敗: {e}")
            return {}

    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        os.makedirs(self.config.output_dir, exist_ok=True)
        model_name = Path(self.config.model_path).name
        dataset_name = getattr(self, "dataset_name", "unknown_dataset")
        model_output_dir = os.path.join(self.config.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        if filename is None:
            filename = f"{dataset_name}_results.json"
        output_path = os.path.join(model_output_dir, filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"結果已儲存: {output_path}")

    def unload_model(self):
        """徹底卸載模型並清除 GPU 記憶體 (Unload model and clear GPU memory)"""
        logger.info("開始卸載模型並清除 VRAM...")
        
        # 1. Delete the pipeline and model objects
        if self.generator is not None:
            del self.generator
            self.generator = None
            
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # 2. Force Python's Garbage Collector
        import gc
        gc.collect()

        # 3. Force PyTorch to empty the CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # 4. Reset peak memory stats so the next iteration starts fresh
            torch.cuda.reset_peak_memory_stats()
            
        self._use_vllm = False
        logger.info("模型已成功卸載，VRAM 已清除。")
