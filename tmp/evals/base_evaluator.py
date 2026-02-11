"""
基礎評估器
==========

所有評估器的抽象基類，整合 gsm_eval.py 的完整實作
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import os
import sys
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, GenerationConfig

# 導入 DatasetConfigManager
sys.path.insert(0, str(Path(__file__).parent.parent))

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


@dataclass
class BaseEvalConfig:
    """基礎評估配置（完整版本）"""
    model_path: str
    output_dir: str = "output"

    max_new_tokens: int = 1024
    do_sample: bool = False
    batch_size: int = 1
    temperature: float = 0.0
    top_p: float = 1.0

    num_samples: Optional[int] = None  # None 表示評估全部
    dataset_name: Optional[str] = None  # 資料集名稱
    dataset_split: str = "test"  # 可選: "train", "validation", "test"
    prompt_type: str = "direct"  # 可選: "direct", "fewshot", "cot"
    prompts: Dict[str, str] = None  # 可選的 prompt 模板
    fewshot_examples: Optional[List[Any]] = None  # 可選的 few-shot 範例

    hf_token: Optional[str] = None
    device_map: str = "cuda:0"
    dtype: str = "float16"
    trust_remote_code: bool = True
    use_safetensors: bool = True

class BaseEvaluator(ABC):
    """評估器基類（完整版本）"""
    
    def __init__(self, config: Union[Dict, BaseEvalConfig], dataset_name: Optional[str]):
        """
        初始化評估器
        
        Args:
            config: 配置，可以是：
                - 配置字典 (Dict)
                - BaseEvalConfig 物件
            dataset_name: 資料集名稱（用於從 dataset_config.yaml 載入）
        """
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
        
        logger.info(f"📋 評估配置已載入: {self.config.model_path}")

    def _load_from_dict(self, cfg: Dict, dataset_name: str) -> BaseEvalConfig:
        """
        從字典載入配置
        """
        try:
            # model_config.yaml 格式
            model_cfg = cfg["model"]
            dataset_cfg = cfg["dataset"]
            
            # 如果有指定 dataset.config_file，載入並合併
            config_file = dataset_cfg["config_file"]
            
            logger.info(f"🔗 從 {config_file} 載入 {dataset_name} 配置")
            
            # 載入 dataset config
            self.dataset_manager = DatasetConfigManager(config_file)
            model_path = model_cfg["name"]
            dataset_dict = self.dataset_manager.get_dataset_config(dataset_name)
            dataset_dict["model_path"] = model_path

            # 套用 override
            override = dataset_cfg.get("override", {})
            if override:
                logger.info(f"⚙️  套用覆寫參數: {list(override.keys())}")
                dataset_dict.update(override)
            
            config_class = getattr(self, "ConfigClass", BaseEvalConfig)
            return config_class(**dataset_dict)
        except Exception as e:
            raise ValueError(f"從資料集配置載入配置失敗: {e}")
    
    @abstractmethod
    def build_prompt(self, sample: Dict[str, Any]) -> str:
        """構建提示詞（子類必須實現）"""
        pass
    
    @abstractmethod
    def extract_answer(self, output: str) -> Any:
        """從輸出提取答案（子類必須實現）"""
        pass
    
    @abstractmethod
    def check_answer(self, true_answer: Any, predicted_answer: Any) -> bool:
        """檢查答案是否正確（子類必須實現）"""
        pass
    
    def load_model(self, quantization_type: Optional[str] = None):
        """
        載入模型和 tokenizer（完整版本）
        
        Args:
            quantization_type: 量化類型，可選 "gptq", "awq", "bnb", None (普通模型)
        """
        logger.info(f"🔹 載入 tokenizer: {self.config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            token=self.config.hf_token
        )
        if self.tokenizer.pad_token is None:
            # 避免將 pad_token 設為 eos_token（會導致 Gemma 等模型提早停止）
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                logger.info("⚙️  設定 pad_token = unk_token")
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.warning("⚠️  pad_token 設為 eos_token（可能導致某些模型提早停止生成）")

        # 根據量化類型載入模型
        # 追蹤是否使用 vLLM（用於生成邏輯）
        self._use_vllm = False

        if quantization_type == "gptq":
            self._load_gptq_model()
        elif quantization_type == "awq":
            self._load_awq_model()
        elif quantization_type == "bnb":
            self._load_bnb_model()
        elif quantization_type == "llmcompressor":
            self._load_llmcompressor_model()
        else:
            self._load_normal_model()

        # vLLM 有自己的生成機制，不需要 transformers pipeline
        if not self._use_vllm:
            self._setup_generation_pipeline()
    
    def _load_normal_model(self):
        """載入普通（未量化）模型"""
        logger.info("🔹 載入普通模型...")
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self.config.dtype, torch.float16)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            device_map=self.config.device_map,
            dtype=dtype,
            trust_remote_code=self.config.trust_remote_code,
            token=self.config.hf_token
        )
    
    def _load_gptq_model(self):
        """載入 GPTQ 量化模型"""
        logger.info("🔹 載入 GPTQ 量化模型...")

        try:
            from gptqmodel import GPTQModel
        except ImportError:
            raise ImportError("請安裝 gptqmodel: pip install gptqmodel")

        # 嘗試載入 GPTQ 模型
        try:
            self.model = GPTQModel.from_quantized(
                self.config.model_path,
                device_map={"": self.config.device_map},  # 強制指定裝置
                trust_remote_code=self.config.trust_remote_code,
                use_safetensors=self.config.use_safetensors,
            )
        except Exception as e:
            logger.error(f"❌ 載入 GPTQ 模型失敗: {e}")
            raise
        
        # 最終驗證
        logger.info("🔍 驗證模型配置...")
        logger.info(f"   Tokenizer vocab size: {len(self.tokenizer)}")
        logger.info(f"   Model vocab size: {self.model.config.vocab_size}")
        logger.info(f"   Pad token ID: {self.model.config.pad_token_id}")
        logger.info(f"   EOS token ID: {self.model.config.eos_token_id}")
        logger.info(f"   Device: {self.model.device}")
    
    def _load_awq_model(self):
        """載入 AWQ 量化模型（優化版本）"""
        logger.info("🔹 載入 AWQ 量化模型...")
        
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError("請安裝 autoawq: pip install autoawq")
        
        # 檢查量化配置
        config_path = Path(self.config.model_path) / "config.json"
        if config_path.exists():
            logger.info(f"📋 找到模型配置: {config_path}")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                if "quantization_config" in config_dict:
                    logger.info(f"   量化參數: {config_dict['quantization_config']}")
        
        load_successful = False
        last_error = None
        
        # 策略 1: 標準載入
        logger.info("📦 策略 1: 標準 AWQ 載入...")
        try:
            self.model = AutoAWQForCausalLM.from_quantized(
                self.config.model_path,
                fuse_layers=True,
                device_map={"": self.config.device_map},  # 強制指定裝置
                trust_remote_code=self.config.trust_remote_code,
                safetensors=self.config.use_safetensors,
            )
            self.model.device = self.config.device_map
            
            load_successful = True
            logger.info(f"✅ 策略 1 成功! vocab_size={self.model.config.vocab_size}")
            
        except Exception as e:
            last_error = e
            logger.warning(f"❌ 策略 1 失敗: {e}")
        
        # 策略 2: 禁用 fuse_layers
        if not load_successful:
            logger.info("📦 策略 2: 禁用 layer fusion...")
            try:
                self.model = AutoAWQForCausalLM.from_quantized(
                    self.config.model_path,
                    fuse_layers=False,  # 禁用可能有問題的優化
                    device_map={"": self.config.device_map},  # 強制指定裝置
                    trust_remote_code=self.config.trust_remote_code,
                    safetensors=self.config.use_safetensors,
                )
                
                load_successful = True
                logger.info(f"✅ 策略 2 成功!")
                
            except Exception as e:
                last_error = e
                logger.warning(f"❌ 策略 2 失敗: {e}")
        
        # 如果所有策略都失敗
        if not load_successful:
            error_msg = f"AWQ 模型載入失敗。最後錯誤: {last_error}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # 最終驗證
        logger.info("🔍 驗證模型配置...")
        logger.info(f"   Tokenizer vocab size: {len(self.tokenizer)}")
        logger.info(f"   Model vocab size: {self.model.config.vocab_size}")
        logger.info(f"   Pad token ID: {self.model.config.pad_token_id}")
        logger.info(f"   EOS token ID: {self.model.config.eos_token_id}")
        logger.info(f"   Device: {self.model.device}")

    def _load_bnb_model(self):
        """載入 BitsAndBytes (BNB) 量化模型"""
        logger.info("🔹 載入 BitsAndBytes (BNB) 量化模型...")

        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError("請安裝 transformers 支援 BitsAndBytes: pip install transformers")

        bnb_config = BitsAndBytesConfig(
            **self.get_quantization_config()
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            device_map={"": self.config.device_map},  # 強制指定裝置
            quantization_config=bnb_config,
            trust_remote_code=self.config.trust_remote_code,
            token=self.config.hf_token
        )

        # 最終驗證
        logger.info("🔍 驗證模型配置...")
        logger.info(f"   Tokenizer vocab size: {len(self.tokenizer)}")
        logger.info(f"   Model vocab size: {self.model.config.vocab_size}")
        logger.info(f"   Pad token ID: {self.model.config.pad_token_id}")
        logger.info(f"   EOS token ID: {self.model.config.eos_token_id}")
        logger.info(f"   Device: {self.model.device}")

    def _is_sparse_only_model(self) -> bool:
        """檢查是否為純稀疏模型（沒有量化）"""
        model_path = Path(self.config.model_path)

        # 方法 1: 檢查 config.json 中的 quantization_config
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                quant_config = config.get("quantization_config", {})

                # 檢查是否有 config_groups（量化配置）
                has_quantization = "config_groups" in quant_config
                # 檢查是否有稀疏配置
                has_sparsity = "sparsity_config" in quant_config

                # 純稀疏 = 有稀疏配置但沒有量化配置
                if has_sparsity and not has_quantization:
                    return True
            except Exception:
                pass

        # 方法 2: 檢查 recipe.yaml（llm-compressor 純稀疏模型可能沒有 quantization_config）
        recipe_path = model_path / "recipe.yaml"
        if recipe_path.exists():
            try:
                import yaml
                with open(recipe_path, "r") as f:
                    recipe = yaml.safe_load(f)

                # 遍歷所有 stage 和 modifier 檢查是否只有稀疏化
                has_sparse_modifier = False
                has_quant_modifier = False

                for stage_name, stage_content in recipe.items():
                    if not isinstance(stage_content, dict):
                        continue
                    modifiers = stage_content.get("default_modifiers", {})
                    if not isinstance(modifiers, dict):
                        continue
                    for modifier_name in modifiers.keys():
                        modifier_lower = modifier_name.lower()
                        if "sparse" in modifier_lower or "prune" in modifier_lower:
                            has_sparse_modifier = True
                        if "quant" in modifier_lower or "gptq" in modifier_lower or "w8a8" in modifier_lower:
                            has_quant_modifier = True

                # 純稀疏 = 有稀疏 modifier 但沒有量化 modifier
                if has_sparse_modifier and not has_quant_modifier:
                    logger.info(f"🔍 從 recipe.yaml 偵測到純稀疏模型")
                    return True
            except Exception as e:
                logger.debug(f"讀取 recipe.yaml 失敗: {e}")

        return False

    def _load_llmcompressor_model(self):
        """
        載入 llm-compressor 量化模型 (compressed-tensors 格式)

        重要限制：
        - llm-compressor 輸出的 compressed-tensors 格式需要 vLLM 進行推理
        - 純稀疏模型（沒有量化）應使用 transformers 載入
        - 標準 transformers 可能無法正確載入量化模型
        """
        # 檢查是否為純稀疏模型
        if self._is_sparse_only_model():
            logger.info("🔍 偵測到純稀疏模型（無量化），使用 transformers 載入...")
            self._load_sparse_only_model()
            return

        # 嘗試檢查是否有 vLLM
        try:
            import vllm
            logger.info("✅ 偵測到 vLLM，嘗試使用 vLLM 載入...")
            self._load_llmcompressor_model_vllm()
            return
        except ImportError:
            logger.warning("❌ vLLM 未安裝，嘗試使用 transformers 載入（可能失敗）")

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self.config.dtype, torch.float16)

        # 修復 llm-compressor 產生的非標準 config.json
        # 移除 "dtype" 欄位（會被 transformers 誤解析為 torch.dtype 導致 JSON 序列化失敗）
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

            logger.info("✅ llm-compressor 模型載入成功")

        except Exception as e:
            logger.error(f"❌ 載入 llm-compressor 模型失敗: {e}")
            raise RuntimeError(
                f"compressed-tensors 格式需要 vLLM 進行推理。\n"
                f"請安裝 vLLM 或重新量化模型（設置 save_compressed: false）。\n"
                f"原始錯誤: {e}"
            )

        # 最終驗證
        logger.info("🔍 驗證模型配置...")
        logger.info(f"   Tokenizer vocab size: {len(self.tokenizer)}")
        logger.info(f"   Model vocab size: {self.model.config.vocab_size}")
        logger.info(f"   Pad token ID: {self.model.config.pad_token_id}")
        logger.info(f"   EOS token ID: {self.model.config.eos_token_id}")
        logger.info(f"   Device: {self.model.device}")

    def _fix_llmcompressor_config(self):
        """
        修復 llm-compressor 產生的非標準 config.json

        llm-compressor 會在 config.json 中加入 "dtype" 欄位，
        但 transformers 會將其誤解析為 torch.dtype 物件，
        導致後續 JSON 序列化失敗。
        """
        config_path = Path(self.config.model_path) / "config.json"
        if not config_path.exists():
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            # 檢查是否有非標準的 "dtype" 欄位（不在 quantization_config 內）
            if "dtype" in config_dict and "quantization_config" in config_dict:
                logger.info(f"🔧 修復 config.json: 移除非標準 'dtype' 欄位")
                del config_dict["dtype"]

                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=2)
                logger.info("✅ config.json 已修復")
        except Exception as e:
            logger.warning(f"⚠️ 無法修復 config.json: {e}")

    def _load_sparse_only_model(self):
        """載入純稀疏模型（沒有量化，使用 transformers）"""
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self.config.dtype, torch.float16)

        # 修復可能的非標準 config.json
        self._fix_llmcompressor_config()

        logger.info(f"🔹 載入純稀疏模型 (dtype={dtype})...")

        # 使用單一設備載入，避免多 GPU 設備不匹配問題
        # device_map="auto" 會分散到多 GPU，導致 tensor 設備不一致錯誤
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=dtype,
            device_map={"": self.config.device_map},
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=True,
            token=self.config.hf_token,
        )

        logger.info(f"✅ 純稀疏模型載入成功: {type(self.model).__name__}")

    def _load_llmcompressor_model_vllm(self):
        """使用 vLLM 載入 llm-compressor 模型"""
        from vllm import LLM, SamplingParams

        logger.info("🔹 使用 vLLM 載入 llm-compressor 模型...")

        # 降低 GPU 記憶體使用率，避免 OOM
        # - gpu_memory_utilization: 預設 0.9 太高，改為 0.6
        # - max_model_len: 限制上下文長度以減少 KV cache 記憶體佔用
        self.model = LLM(
            model=self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
            gpu_memory_utilization=0.6,
            max_model_len=4096,  # 限制上下文長度，減少 KV cache
            disable_log_stats=True,
        )

        # 標記使用 vLLM
        self._use_vllm = True
        # 儲存 SamplingParams 類別供生成時使用
        self._vllm_sampling_params_cls = SamplingParams

        logger.info("✅ vLLM 模型載入成功")

    def _setup_generation_pipeline(self):
        """安全創建 TextGenerationPipeline（支援 AWQ / GPTQ / BNB）"""
        logger.info("🔹 創建生成 pipeline...")

        if not hasattr(self.model, "can_generate"):
            self.model.can_generate = lambda: True

        gen_cfg = getattr(self.model, "generation_config", None)

        if gen_cfg is None:
            # 若模型沒這個屬性，自己建一個
            gen_cfg = GenerationConfig()
            self.model.generation_config = gen_cfg

        self.generator = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
        )

        logger.info("✅ 模型生成管線初始化完成")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        統一的文字生成介面（支援 transformers pipeline 和 vLLM）

        Args:
            prompt: 輸入提示詞
            **kwargs: 生成參數
                - max_new_tokens: 最大生成 token 數
                - do_sample: 是否取樣
                - temperature: 溫度
                - top_p: nucleus sampling

        Returns:
            生成的文字（不含輸入 prompt）
        """
        if self._use_vllm:
            return self._generate_vllm(prompt, **kwargs)
        else:
            return self._generate_transformers(prompt, **kwargs)

    def _generate_transformers(self, prompt: str, **kwargs) -> str:
        """使用 transformers pipeline 生成"""
        if self.generator is None:
            raise RuntimeError("Generator 未初始化，請先呼叫 load_model()")

        # 過濾掉 vLLM 特有的參數
        gen_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        gen_kwargs["return_full_text"] = False

        output = self.generator(prompt, **gen_kwargs)
        return output[0]["generated_text"]

    def _generate_vllm(self, prompt: str, **kwargs) -> str:
        """使用 vLLM 生成"""
        # 建立 SamplingParams
        sampling_params = self._vllm_sampling_params_cls(
            max_tokens=kwargs.get("max_new_tokens", 256),
            temperature=kwargs.get("temperature", 0.0) if kwargs.get("do_sample", False) else 0.0,
            top_p=kwargs.get("top_p", 1.0) if kwargs.get("do_sample", False) else 1.0,
        )

        # vLLM 生成
        outputs = self.model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def is_model_loaded(self) -> bool:
        """檢查模型是否已載入"""
        if self._use_vllm:
            return self.model is not None
        else:
            return self.generator is not None

    def get_quantization_config(self) -> Optional[Dict[str, Any]]:
        """取得量化配置內容"""
        config_path = Path(self.config.model_path) / "config.json"
        if not config_path.exists():
            logger.info(f"📁 未找到量化配置檔案: {config_path}")
            return None

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as exc:
            logger.warning(f"⚠️ 讀取量化配置失敗: {exc}")
            return None

        quant_config = config_data.get("quantization_config")
        if quant_config is None:
            logger.info("ℹ️ config.json 未包含 quantization_config")
        else:
            logger.info(f"✅ 取得 quantization_config: {list(quant_config.keys())}")
        return quant_config
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """
        儲存評估結果為 JSON
        
        Args:
            results: 評估結果
            filename: 輸出文件名，若為 None 則使用預設名稱
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        # 取得模型名稱（取最後一層資料夾名）
        model_name = Path(self.config.model_path).name
        # 取得資料集名稱
        dataset_name = getattr(self, "dataset_name", "unknown_dataset")

        # 組成完整輸出資料夾路徑： output_dir/model_name/
        model_output_dir = os.path.join(self.config.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # 檔名邏輯
        if filename is None:
            filename = f"{dataset_name}_results.json"
        
        output_path = os.path.join(model_output_dir, filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 結果已儲存: {output_path}")