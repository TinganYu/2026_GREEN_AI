"""
統一量化工具
=====================

整合 GPTQ、AWQ、BNB 三種量化方法的完整實現

使用方式:
    from quantization_unified import Quantizer, GPTQConfig, AWQConfig, BNBConfig
    
    # 方法 1: 使用專門的配置類
    config = AWQConfig(w_bit=4, q_group_size=128)
    quantizer = Quantizer(model_path="meta-llama/Llama-3.2-1B-Instruct")
    output = quantizer.quantize(config)
    
    # 方法 2: 快捷方法
    quantizer = Quantizer("meta-llama/Llama-3.2-1B-Instruct")
    output = quantizer.quantize_awq(w_bit=4)
"""

from datetime import datetime
import gc
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Union

import torch
from transformers import AutoTokenizer


logger = logging.getLogger("Quantization")
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
class BaseQuantConfig(ABC):
    """
    量化配置基類
    
    通用參數:
        output_dir: 輸出目錄 (None 則自動生成)
        hf_token: HuggingFace token
    """
    save_model: bool = True
    output_dir: Optional[str] = None
    hf_token: Optional[str] = None
    
    @property
    @abstractmethod
    def method_name(self) -> str:
        """返回量化方法名稱"""
        pass
    
    @abstractmethod
    def validate(self):
        """驗證配置參數"""
        pass


@dataclass
class GPTQConfig(BaseQuantConfig):
    """
    GPTQ 量化配置
    
    專用參數:
        bits: 量化位元數 (2, 3, 4, 8)
        group_size: 分組大小 (-1 表示不分組，建議 128，預設 128)
        damp_percent: 阻尼百分比，用於穩定量化過程 (None 表示自動計算)
        damp_auto_increment: 阻尼自動遞增幅度，協助在校準期間逐步調整阻尼 (None 表示停用)
        desc_act: 是否使用降序激活順序 (預設 True)
        act_group_aware: 是否啟用 activation group aware 的排序策略 (預設 False)
        static_groups: 是否使用靜態分組權重 (預設 False)
        sym: 是否使用對稱量化 (預設 True)
        true_sequential: 是否使用真實順序量化 (預設 True)
        lm_head: 是否同時量化 LM head 層 (預設 False)
        mse: 量化過程的 MSE 正則化權重 (預設 0.0)
        rotation: 權重旋轉策略 (可選 "hadamard" 或 "random"，預設 None)
        calib_num: 校準樣本數量，更多樣本 = 更高精度但更慢 (預設 256)
    
    範例:
        >>> config = GPTQConfig(
        ...     bits=4,
        ...     group_size=128,
        ...     calib_num=64,
        ...     output_dir="models/gptq"
        ... )
    """
    bits: int = field(default=4, metadata={"choices": [2, 3, 4, 8]})
    group_size: int = field(default=128, metadata={"choices": [-1, 16, 32, 64, 128, 256, 512, 1024]})
    damp_percent: Optional[float] = field(default=0.05)
    damp_auto_increment: Optional[float] = field(default=0.01)
    desc_act: bool = field(default=True)
    act_group_aware: bool = field(default=False)
    static_groups: bool = field(default=False)
    sym: bool = field(default=True)
    true_sequential: bool = field(default=True)
    lm_head: bool = field(default=False)
    mse: float = field(default=0.0)
    rotation: Optional[str] = field(default=None, metadata={"choices": ["hadamard", "random"]})
    calib_num: int = 512

    @property
    def method_name(self) -> str:
        return "gptq"
    
    def validate(self):
        """驗證配置參數"""
        # 驗證 bits
        valid_bits = self.__dataclass_fields__['bits'].metadata.get('choices', [])
        if self.bits not in valid_bits:
            raise ValueError(f"bits 必須是 {valid_bits} 之一，得到: {self.bits}")
        
        # 驗證 calib_num
        if self.calib_num < 1:
            raise ValueError(f"calib_num 必須 >= 1，得到: {self.calib_num}")
        
        # 驗證 group_size
        valid_group_sizes = self.__dataclass_fields__['group_size'].metadata.get('choices', [])
        if self.group_size not in valid_group_sizes:
            raise ValueError(f"group_size 必須是 {valid_group_sizes} 之一，得到: {self.group_size}")

        # 驗證 damp_percent
        if not 0 < self.damp_percent < 1:
            raise ValueError(f"damp_percent 必須在 0-1 之間，得到: {self.damp_percent}")

        # 驗證 damp_auto_increment
        if self.damp_auto_increment < 0:
            raise ValueError(f"damp_auto_increment 必須 >= 0，得到: {self.damp_auto_increment}")

        if self.act_group_aware and self.desc_act:
            raise ValueError("`act_group_aware` == `True`時，需要 `desc_act` == `False`")


@dataclass
class AWQConfig(BaseQuantConfig):
    """
    AWQ 量化配置
    
    專用參數:
        w_bit: 量化位元數 (預設 4)
        zero_point: 是否使用零點量化 (預設 True)
        q_group_size: 量化分組大小，建議 128 (預設 128)
        version: 量化版本 (預設 "gemm")
            - "gemm": 通用矩陣乘法（相容性最好）
            - "gemv": 矩陣向量乘法（推理優化）
            - "marlin": Marlin kernel（最快，需要特定硬體）
            - "gemv_fast": 快速矩陣向量乘法
        modules_to_not_convert: 不進行量化的模組列表 (預設 None)
    
    範例:
        >>> config = AWQConfig(
        ...     w_bit=4,
        ...     q_group_size=128,
        ...     version="gemm",
        ...     output_dir="models/awq"
        ... )
    """
    w_bit: int = field(default=4)
    zero_point: bool = field(default=True)
    q_group_size: int = field(default=128)
    version: str = field(default="gemm", metadata={"choices": ["gemm", "gemv", "marlin", "gemv_fast"]})
    modules_to_not_convert: Optional[List[str]] = field(default=None)
    calib_num: int = 512
    
    @property
    def method_name(self) -> str:
        return "awq"
    
    def validate(self):
        """驗證配置參數"""
        # 驗證 w_bit
        if self.w_bit != 4:
            raise ValueError(f"AWQ 目前只支援 4-bit 量化，得到: {self.w_bit}")
        
        # 驗證 version
        valid_versions = self.__dataclass_fields__['version'].metadata.get('choices', [])
        if self.version not in valid_versions:
            raise ValueError(f"version 必須是 {valid_versions} 之一，得到: {self.version}")
        
        # 驗證 q_group_size
        if self.q_group_size <= 0:
            raise ValueError(f"q_group_size 必須 > 0，得到: {self.q_group_size}")


@dataclass
class BNBConfig(BaseQuantConfig):
    """
    BitsAndBytes 量化配置
    
    專用參數:
        bits: 量化位元數 (4 或 8，預設 4)
        bnb_4bit_quant_type: 4-bit 量化類型 (預設 "nf4")
            - "nf4": NormalFloat4（精度較高）
            - "fp4": Float4（速度較快）
        bnb_4bit_use_double_quant: 是否使用雙重量化，進一步壓縮 (預設 True)
        bnb_4bit_compute_dtype: 計算時使用的資料型別 (預設 "bfloat16")
            - "bfloat16": BFloat16（平衡性能）
            - "float16": Float16（相容性好）
            - "float32": Float32（精度最高但較慢）
        llm_int8_threshold: INT8 量化閾值，僅 8-bit 時使用 (預設 6.0)
        llm_int8_skip_modules: 跳過量化的模組列表 (預設 None)
        llm_int8_has_fp16_weight: 是否有 FP16 權重 (預設 False)
        llm_int8_enable_fp32_cpu_offload: 是否啟用 FP32 CPU 卸載 (預設 False)
    
    範例:
        >>> config = BNBConfig(
        ...     bits=4,
        ...     bnb_4bit_quant_type="nf4",
        ...     bnb_4bit_use_double_quant=True,
        ...     output_dir="models/bnb"
        ... )
    """
    bits: int = field(default=4, metadata={"choices": [4, 8]})
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"choices": ["fp4", "nf4"]})
    bnb_4bit_use_double_quant: bool = field(default=True)
    bnb_4bit_compute_dtype: str = field(default="bfloat16", metadata={"choices": ["float16", "bfloat16", "float32"]})
    llm_int8_threshold: float = field(default=6.0)
    llm_int8_skip_modules: Optional[List[str]] = field(default=None)
    llm_int8_has_fp16_weight: bool = field(default=False)
    llm_int8_enable_fp32_cpu_offload: bool = field(default=False)
    
    @property
    def method_name(self) -> str:
        return "bnb"
    
    def validate(self):
        """驗證配置參數"""
        # 驗證 bits
        valid_bits = self.__dataclass_fields__['bits'].metadata.get('choices', [])
        if self.bits not in valid_bits:
            raise ValueError(f"bits 必須是 {valid_bits} 之一，得到: {self.bits}")
        
        # 驗證 bnb_4bit_quant_type
        valid_quant_types = self.__dataclass_fields__['bnb_4bit_quant_type'].metadata.get('choices', [])
        if self.bnb_4bit_quant_type not in valid_quant_types:
            raise ValueError(f"bnb_4bit_quant_type 必須是 {valid_quant_types} 之一，得到: {self.bnb_4bit_quant_type}")
        
        # 驗證 bnb_4bit_compute_dtype
        valid_dtypes = self.__dataclass_fields__['bnb_4bit_compute_dtype'].metadata.get('choices', [])
        if self.bnb_4bit_compute_dtype not in valid_dtypes:
            raise ValueError(f"bnb_4bit_compute_dtype 必須是 {valid_dtypes} 之一，得到: {self.bnb_4bit_compute_dtype}")
        
        # 驗證 llm_int8_threshold
        if self.llm_int8_threshold <= 0:
            raise ValueError(f"llm_int8_threshold 必須 > 0，得到: {self.llm_int8_threshold}")


# ============================================================================
# 統一的 Quantizer 類
# ============================================================================

class Quantizer:
    """
    統一的量化器類
    
    整合 GPTQ、AWQ、BNB 三種量化方法，提供一致的接口和錯誤處理。
    
    屬性:
        model_path: 模型路徑或 HuggingFace model ID
        model_id: 從 model_path 提取的簡短標識符
        device: 使用的設備
        tokenizer: 載入的 tokenizer（延遲載入）
    
    使用範例:
        >>> # 基本使用
        >>> quantizer = Quantizer("meta-llama/Llama-3.2-1B-Instruct")
        >>> config = AWQConfig(w_bit=4)
        >>> output = quantizer.quantize(config)
        
        >>> # 快捷方法
        >>> output = quantizer.quantize_awq(w_bit=4, q_group_size=128)
        >>> output = quantizer.quantize_gptq(bits=4, group_size=128)
        >>> output = quantizer.quantize_bnb(bits=4)
    """
    
    def __init__(self, model_path: str):
        """
        初始化量化器
        
        Args:
            model_path: 模型路徑或 HuggingFace model ID
        """
        self.model_path = model_path
        self.model_id = self._extract_model_id(model_path)
        self._tokenizer = None
        
        # 檢查 GPU 可用性
        if not torch.cuda.is_available():
            raise RuntimeError("❌ GPU 不可用，量化需要 GPU 支援")
        
        self.device = torch.device("cuda:0")
        logger.info(f"✅ 使用 GPU: {torch.cuda.get_device_name(self.device)}")
    
    @staticmethod
    def _extract_model_id(model_path: str) -> str:
        """從路徑提取模型 ID"""
        return model_path.split("/")[-1]
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        """延遲載入 tokenizer"""
        if self._tokenizer is None:
            self._tokenizer = self._safe_load_tokenizer()
        return self._tokenizer
    
    def _safe_load_tokenizer(self, hf_token: str = None) -> AutoTokenizer:
        """安全載入 tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=True, token=hf_token
            )
            logger.debug(f"✅ 載入 fast tokenizer")
        except Exception as e:
            logger.warning(f"⚠️ Fast tokenizer 失敗，使用 slow 版本: {e}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=False, token=hf_token
            )
        return tokenizer
    
    @staticmethod
    def _cleanup_memory():
        """清理 GPU 記憶體"""
        gc.collect()
        torch.cuda.empty_cache()
    
    def _generate_output_dir(self, config: BaseQuantConfig, trial_number: int = None, exp_dir: str = None) -> str:
        """生成輸出目錄

        Args:
            config: 量化配置
            trial_number: 試驗編號（用於優化實驗）
            exp_dir: 實驗根目錄（用於優化實驗）

        Returns:
            輸出目錄路徑
        """
        if config.output_dir:
            return config.output_dir

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 確定量化方法和位元數
        if isinstance(config, AWQConfig):
            method = config.method_name
            bits = config.w_bit
        elif isinstance(config, GPTQConfig) or isinstance(config, BNBConfig):
            method = config.method_name
            bits = config.bits
        else:
            method = "unknown"
            bits = 4

        # 如果是優化實驗（提供了 trial_number 和 exp_dir）
        if trial_number is not None and exp_dir is not None:
            model_dir_name = f"trial{trial_number:02d}-{ts}-{method}-{bits}bit"
            return os.path.join(exp_dir, "models", model_dir_name)

        # 否則使用原有命名格式
        return f"quantized/{self.model_id}-{method}-{bits}bit-{ts}"

    def _log_start(self, method: str):
        """記錄開始訊息"""
        logger.info("=" * 80)
        logger.info(f"🚀 開始 {method.upper()} 量化")
        logger.info(f"📦 模型: {self.model_path}")
        logger.info("=" * 80)
    
    def _log_complete(self, output_dir: str, method: str):
        """記錄完成訊息"""
        logger.info("=" * 80)
        logger.info(f"✅ {method.upper()} 量化成功完成!")
        logger.info(f"📁 輸出路徑: {output_dir}")
        logger.info("=" * 80)
    
    # ========================================================================
    # GPTQ 量化
    # ========================================================================
    
    def _quantize_gptq(self, config: GPTQConfig, trial_number: int = None, exp_dir: str = None) -> str:
        """執行 GPTQ 量化"""
        try:
            from gptqmodel import GPTQModel, QuantizeConfig
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "❌ 請先安裝 GPTQ 依賴:\n"
                "   pip install gptqmodel datasets"
            )

        self._log_start("GPTQ")
        output_dir = self._generate_output_dir(config, trial_number, exp_dir)
        
        # 載入 tokenizer
        logger.info("🔹 載入 tokenizer...")
        tokenizer = self._safe_load_tokenizer(config.hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("   ✓ 設定 pad_token = eos_token")
        
        # 建立量化配置
        logger.info("🔹 建立 GPTQ 配置...")
        quant_config = QuantizeConfig(
            bits=config.bits,
            group_size=config.group_size,
            damp_percent=config.damp_percent,
            damp_auto_increment=config.damp_auto_increment,
            desc_act=config.desc_act,
            act_group_aware=config.act_group_aware,
            static_groups=config.static_groups,
            sym=config.sym,
            true_sequential=config.true_sequential,
            lm_head=config.lm_head,
            mse=config.mse,
            rotation=config.rotation,
        )
        logger.info(f"   • 配置: {quant_config}")
        
        # 載入模型
        logger.info("🔹 載入原始模型...")
        self._cleanup_memory()

        model = GPTQModel.load(self.model_path, quant_config)

        logger.info("   ✓ 模型載入完成")
        
        # 準備校準資料
        logger.info("🔹 準備校準資料...")
        calibration_dataset = [
            ex["question"] + " " + ex["answer"]
            for ex in load_dataset("openai/gsm8k", "main", split="train").select(range(config.calib_num))
        ]
        logger.info(f"   ✓ 準備了 {len(calibration_dataset)} 個校準樣本")

        # 執行量化
        logger.info("🔹 開始量化（這可能需要較長時間）...")
        # 同步 GPU，準備正式計時
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model.quantize(calibration_dataset, batch_size=1)

        # 統計 GPU 記憶體與 throughput
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logger.info(f"   • 量化過程中 GPU 峰值記憶體使用: {peak_memory:.2f} MB")
            
        logger.info("   ✓ 量化完成")
        
        # 儲存
        logger.info("🔹 儲存模型...")
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("   ✓ 儲存完成")
        
        self._log_complete(output_dir, "GPTQ")
        return output_dir
    
    # ========================================================================
    # AWQ 量化
    # ========================================================================
    
    def _quantize_awq(self, config: AWQConfig, trial_number: int = None, exp_dir: str = None) -> str:
        """執行 AWQ 量化"""
        try:
            from awq import AutoAWQForCausalLM
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "❌ 請先安裝 AWQ 依賴:\n"
                "   pip install autoawq datasets"
            )

        self._log_start("AWQ")
        output_dir = self._generate_output_dir(config, trial_number, exp_dir)
        
        # 載入 tokenizer
        logger.info("🔹 載入 tokenizer...")
        tokenizer = self._safe_load_tokenizer(config.hf_token)
        
        # 建立量化配置
        logger.info("🔹 建立 AWQ 配置...")
        quant_config = {
            "zero_point": config.zero_point,
            "q_group_size": config.q_group_size,
            "w_bit": config.w_bit,
            "version": config.version
        }
        logger.info(f"   • 配置: {quant_config}")
        
        # 載入模型
        logger.info("🔹 載入原始模型...")
        self._cleanup_memory()
        
        model = AutoAWQForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            device_map="cuda:0",
            token=config.hf_token
        )
        logger.info("   ✓ 模型載入完成")

        # 準備校準資料
        logger.info("🔹 準備校準資料...")
        calibration_dataset = [
            ex["question"] + " " + ex["answer"]
            for ex in load_dataset("openai/gsm8k", "main", split="train").select(range(config.calib_num))
        ]
        logger.info(f"   ✓ 準備了 {len(calibration_dataset)} 個校準樣本")
        
        # 執行量化
        logger.info("🔹 開始量化（這可能需要幾分鐘）...")
        # 同步 GPU，準備正式計時
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_dataset)

        # 統計 GPU 記憶體與 throughput
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logger.info(f"   • 量化過程中 GPU 峰值記憶體使用: {peak_memory:.2f} MB")

        logger.info("   ✓ 量化完成")
        
        # 儲存
        logger.info("🔹 儲存模型...")
        model.save_quantized(output_dir, safetensors=True)
        tokenizer.save_pretrained(output_dir)
        logger.info("   ✓ 儲存完成")
        
        self._log_complete(output_dir, "AWQ")
        return output_dir
    
    # ========================================================================
    # BNB 量化
    # ========================================================================
    
    def _quantize_bnb(self, config: BNBConfig, trial_number: int = None, exp_dir: str = None) -> str:
        """執行 BitsAndBytes 量化"""
        try:
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "❌ 請先安裝 BNB 依賴:\n"
                "   pip install transformers bitsandbytes"
            )

        self._log_start("BitsAndBytes")
        output_dir = self._generate_output_dir(config, trial_number, exp_dir)
        
        # 載入 tokenizer
        logger.info("🔹 載入 tokenizer...")
        tokenizer = self._safe_load_tokenizer(config.hf_token)
        
        # 建立量化配置
        logger.info("🔹 建立 BNB 配置...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=(config.bits == 4),
            load_in_8bit=(config.bits == 8),
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            llm_int8_threshold=config.llm_int8_threshold,
            llm_int8_skip_modules=config.llm_int8_skip_modules,
            llm_int8_has_fp16_weight=config.llm_int8_has_fp16_weight,
            llm_int8_enable_fp32_cpu_offload=config.llm_int8_enable_fp32_cpu_offload,
        )
        logger.info(f"   • 配置: {quant_config}")
        
        # 載入並量化模型
        logger.info("🔹 載入並量化模型...")
        self._cleanup_memory()
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
            device_map="cuda:0",
            token=config.hf_token
        )
        logger.info("   ✓ 模型載入與量化完成")
        
        # 儲存
        logger.info("🔹 儲存模型...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("   ✓ 儲存完成")
        
        self._log_complete(output_dir, "BitsAndBytes")
        return output_dir
    
    # ========================================================================
    # 公開接口
    # ========================================================================
    
    def quantize(self, config: Union[GPTQConfig, AWQConfig, BNBConfig],
                 trial_number: int = None, exp_dir: str = None) -> str:
        """
        使用指定配置執行量化

        Args:
            config: 量化配置物件（GPTQConfig, AWQConfig, 或 BNBConfig）
            trial_number: 試驗編號（用於優化實驗）
            exp_dir: 實驗根目錄（用於優化實驗）

        Returns:
            str: 量化後模型的輸出路徑

        Raises:
            ValueError: 如果配置驗證失敗
            ImportError: 如果缺少必要的依賴

        Examples:
            >>> quantizer = Quantizer("meta-llama/Llama-3.2-1B-Instruct")
            >>> config = AWQConfig(w_bit=4, q_group_size=128)
            >>> output = quantizer.quantize(config)
        """
        # 驗證配置
        config.validate()

        # 根據配置類型執行對應的量化
        if isinstance(config, GPTQConfig):
            return self._quantize_gptq(config, trial_number, exp_dir)
        elif isinstance(config, AWQConfig):
            return self._quantize_awq(config, trial_number, exp_dir)
        elif isinstance(config, BNBConfig):
            return self._quantize_bnb(config, trial_number, exp_dir)
        else:
            raise ValueError(f"不支援的配置類型: {type(config)}")
    
    def quantize_gptq(self, **kwargs) -> str:
        """
        快捷方法: GPTQ 量化
        
        Args:
            bits: 量化位元數
            **kwargs: 其他 GPTQConfig 參數
        
        Returns:
            str: 量化後模型的輸出路徑
        
        Examples:
            >>> quantizer = Quantizer("facebook/opt-350m")
            >>> output = quantizer.quantize_gptq(
            ...     bits=4,
            ...     group_size=128,
            ...     calib_num=64
            ... )
        """
        config = GPTQConfig(**kwargs)
        return self.quantize(config)
    
    def quantize_awq(self, **kwargs) -> str:
        """
        快捷方法: AWQ 量化
        
        Args:
            bits: 量化位元數
            **kwargs: 其他 AWQConfig 參數
        
        Returns:
            str: 量化後模型的輸出路徑
        
        Examples:
            >>> quantizer = Quantizer("meta-llama/Llama-3.2-1B-Instruct")
            >>> output = quantizer.quantize_awq(
            ...     w_bit=4,
            ...     q_group_size=128,
            ...     version="gemm"
            ... )
        """
        config = AWQConfig(**kwargs)
        return self.quantize(config)
    
    def quantize_bnb(self, **kwargs) -> str:
        """
        快捷方法: BitsAndBytes 量化
        
        Args:
            bits: 量化位元數
            **kwargs: 其他 BNBConfig 參數
        
        Returns:
            str: 量化後模型的輸出路徑
        
        Examples:
            >>> quantizer = Quantizer("meta-llama/Llama-3.2-1B-Instruct")
            >>> output = quantizer.quantize_bnb(
            ...     bits=4,
            ...     bnb_4bit_quant_type="nf4"
            ... )
        """
        config = BNBConfig(**kwargs)
        return self.quantize(config)
    
    @staticmethod
    def list_methods() -> List[str]:
        """列出所有支援的量化方法"""
        return ["gptq", "awq", "bnb"]