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
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 根據量化類型載入模型
        if quantization_type == "gptq":
            self._load_gptq_model()
        elif quantization_type == "awq":
            self._load_awq_model()
        else:
            self._load_normal_model()

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