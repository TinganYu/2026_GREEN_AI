"""
llm-compressor 統一量化工具
=============================

整合 llm-compressor 套件支援的多種量化演算法：
- GPTQ: 基於 Hessian 的權重量化
- AWQ: 激活感知權重量化
- SmoothQuant: 平滑激活異常值（輔助前處理）
- SparseGPT: 稀疏化剪枝
- AutoRound: 基於 SignSGD 的量化調整

使用方式:
    from llmcompressor_quantization import LLMCompressorQuantizer, LLMCompressorConfig

    # 方法 1: 使用配置類
    config = LLMCompressorConfig(method="gptq", scheme="W4A16")
    quantizer = LLMCompressorQuantizer(model_path="meta-llama/Llama-3.2-1B-Instruct")
    output = quantizer.quantize(config)

    # 方法 2: 快捷方法
    quantizer = LLMCompressorQuantizer("meta-llama/Llama-3.2-1B-Instruct")
    output = quantizer.quantize_gptq(scheme="W4A16")
"""

from datetime import datetime
import gc
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, List, Any

import torch


logger = logging.getLogger("LLMCompressor")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)

logger.propagate = False


# ============================================================================
# 配置類
# ============================================================================

@dataclass
class LLMCompressorConfig:
    """
    llm-compressor 量化配置

    通用參數:
        method: 主量化方法 (gptq, awq, autoround, sparsegpt)
        scheme: 量化方案 (W4A16, W8A16, W8A8, FP8)
        targets: 目標層類型 (預設 "Linear")
        ignore: 忽略的層列表 (預設 ["lm_head"])

    SmoothQuant 前處理:
        smoothquant_enabled: 是否啟用 SmoothQuant 前處理
        smoothquant_strength: 平滑強度 (0.0-1.0)

    GPTQ 專用:
        gptq_block_size: 分組大小，只有 GPTQ 支援 (-1 表示 per-channel)
        gptq_actorder: 激活排序 ("weight", "group", "dynamic", "static")
        gptq_dampening: 阻尼係數

    AWQ 專用:
        awq_duo_scaling: 雙重縮放
        awq_n_grid: 網格搜索數量

    AutoRound 專用:
        autoround_iters: 迭代次數
        autoround_batch_size: 批次大小
        autoround_lr: 學習率 (None 表示自動)
        autoround_enable_torch_compile: 啟用 torch.compile 加速

    SparseGPT 專用 (兩種模式擇一):
        模式 1 - 非結構化稀疏: 設定 sparsegpt_sparsity，保持 prunen=0, prunem=0
        模式 2 - N:M 結構化稀疏: 設定 prunen 和 prunem，sparsity 被忽略
        sparsegpt_sparsity: 非結構化稀疏比例 (0.0-1.0)
        sparsegpt_prunen: N:M 稀疏中的 N (0 表示使用非結構化稀疏)
        sparsegpt_prunem: N:M 稀疏中的 M (稀疏比例 = (M-N)/M)

    校準設定:
        calibration_dataset: 校準資料集名稱
        calibration_split: 資料集分割
        calibration_text_column: 文本欄位名稱
        num_calibration_samples: 校準樣本數量
        max_seq_length: 最大序列長度

    輸出設定:
        output_dir: 輸出目錄 (None 則自動生成)
        save_compressed: 是否以壓縮格式儲存
        hf_token: HuggingFace token

    範例:
        >>> config = LLMCompressorConfig(
        ...     method="gptq",
        ...     scheme="W4A16",
        ...     gptq_block_size=128,
        ...     num_calibration_samples=512
        ... )
    """
    # ===== 主量化方法 =====
    method: str = field(default="gptq", metadata={"choices": ["gptq", "awq", "autoround", "sparsegpt"]})

    # ===== 通用量化設定 =====
    scheme: str = field(default="W4A16", metadata={"choices": ["W4A16", "W8A16", "W8A8", "FP8"]})
    targets: str = "Linear"
    ignore: List[str] = field(default_factory=lambda: ["lm_head"])

    # ===== SmoothQuant 設定（可選前處理）=====
    smoothquant_enabled: bool = False
    smoothquant_strength: float = 0.8

    # ===== GPTQ 專用參數 =====
    # block_size (group_size): 只有 GPTQ 支援，AWQ/AutoRound 由 scheme 內部控制
    gptq_block_size: int = field(default=128, metadata={"choices": [-1, 16, 32, 64, 128, 256, 512, 1024]})
    # actorder: 支援字串 ("weight", "group", "dynamic", "static") 或布林值 (向後相容)
    gptq_actorder: Any = "weight"  # 使用 Any 以支援 str 或 bool
    gptq_dampening: float = 0.01

    # ===== AWQ 專用參數 =====
    awq_duo_scaling: bool = True
    awq_n_grid: int = 20

    # ===== AutoRound 專用參數 =====
    autoround_iters: int = 200
    autoround_batch_size: int = 8
    autoround_lr: Optional[float] = None
    autoround_enable_torch_compile: bool = True

    # ===== SparseGPT 專用參數 =====
    sparsegpt_sparsity: float = 0.5
    sparsegpt_prunen: int = 0
    sparsegpt_prunem: int = 0

    # ===== Calibration 設定 =====
    calibration_dataset: str = "openai/gsm8k"
    calibration_split: str = "train"
    calibration_text_column: str = "question"
    num_calibration_samples: int = 512
    max_seq_length: int = 2048

    # ===== 輸出設定 =====
    output_dir: Optional[str] = None
    save_compressed: bool = True
    hf_token: Optional[str] = None

    def validate(self):
        """驗證配置參數"""
        # 驗證 method
        valid_methods = self.__dataclass_fields__['method'].metadata.get('choices', [])
        if self.method not in valid_methods:
            raise ValueError(f"method 必須是 {valid_methods} 之一，得到: {self.method}")

        # 驗證 scheme
        valid_schemes = self.__dataclass_fields__['scheme'].metadata.get('choices', [])
        if self.scheme not in valid_schemes:
            raise ValueError(f"scheme 必須是 {valid_schemes} 之一，得到: {self.scheme}")

        # 驗證 gptq_block_size (只有 GPTQ 需要)
        if self.method == "gptq":
            valid_block_sizes = self.__dataclass_fields__['gptq_block_size'].metadata.get('choices', [])
            if self.gptq_block_size not in valid_block_sizes:
                raise ValueError(f"gptq_block_size 必須是 {valid_block_sizes} 之一，得到: {self.gptq_block_size}")

        # 驗證 smoothquant_strength
        if not 0.0 <= self.smoothquant_strength <= 1.0:
            raise ValueError(f"smoothquant_strength 必須在 0.0-1.0 之間，得到: {self.smoothquant_strength}")

        # 驗證 sparsegpt_sparsity
        if not 0.0 <= self.sparsegpt_sparsity <= 1.0:
            raise ValueError(f"sparsegpt_sparsity 必須在 0.0-1.0 之間，得到: {self.sparsegpt_sparsity}")

        # 驗證 calibration 參數
        if self.num_calibration_samples < 1:
            raise ValueError(f"num_calibration_samples 必須 >= 1，得到: {self.num_calibration_samples}")

        if self.max_seq_length < 1:
            raise ValueError(f"max_seq_length 必須 >= 1，得到: {self.max_seq_length}")

        # SparseGPT N:M 稀疏驗證
        if self.method == "sparsegpt":
            if self.sparsegpt_prunen > 0 and self.sparsegpt_prunem <= 0:
                raise ValueError("N:M 稀疏需要同時設定 prunen 和 prunem")
            if self.sparsegpt_prunem > 0 and self.sparsegpt_prunen <= 0:
                raise ValueError("N:M 稀疏需要同時設定 prunen 和 prunem")
            if self.sparsegpt_prunen > 0 and self.sparsegpt_prunen >= self.sparsegpt_prunem:
                raise ValueError(f"prunen ({self.sparsegpt_prunen}) 必須小於 prunem ({self.sparsegpt_prunem})")


# ============================================================================
# 量化器類
# ============================================================================

class LLMCompressorQuantizer:
    """
    llm-compressor 統一量化器

    整合 GPTQ、AWQ、AutoRound、SparseGPT 等量化方法，提供一致的接口。

    屬性:
        model_path: 模型路徑或 HuggingFace model ID
        model_id: 從 model_path 提取的簡短標識符
        device: 使用的設備
        tokenizer: 載入的 tokenizer（延遲載入）

    使用範例:
        >>> quantizer = LLMCompressorQuantizer("meta-llama/Llama-3.2-1B-Instruct")
        >>> config = LLMCompressorConfig(method="gptq", scheme="W4A16")
        >>> output = quantizer.quantize(config)

        >>> # 快捷方法
        >>> output = quantizer.quantize_gptq(scheme="W4A16")
        >>> output = quantizer.quantize_awq(scheme="W4A16")
        >>> output = quantizer.quantize_autoround(scheme="W4A16")
        >>> output = quantizer.quantize_sparsegpt(sparsity=0.5)
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
        self._model = None

        # 檢查 GPU 可用性
        if not torch.cuda.is_available():
            raise RuntimeError("GPU 不可用，量化需要 GPU 支援")

        self.device = torch.device("cuda:0")
        logger.info(f"使用 GPU: {torch.cuda.get_device_name(self.device)}")

    @staticmethod
    def _extract_model_id(model_path: str) -> str:
        """從路徑提取模型 ID"""
        return model_path.split("/")[-1]

    @staticmethod
    def _cleanup_memory():
        """清理 GPU 記憶體"""
        gc.collect()
        torch.cuda.empty_cache()

    def _safe_load_tokenizer(self, hf_token: str = None):
        """安全載入 tokenizer"""
        from transformers import AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=True, token=hf_token
            )
            logger.debug("載入 fast tokenizer")
        except Exception as e:
            logger.warning(f"Fast tokenizer 失敗，使用 slow 版本: {e}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, use_fast=False, token=hf_token
            )

        # 確保有 pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("設定 pad_token = eos_token")

        return tokenizer

    def _generate_output_dir(self, config: LLMCompressorConfig) -> str:
        """生成輸出目錄"""
        if config.output_dir:
            return config.output_dir

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # SparseGPT 是稀疏化方法，使用 sparsity 而非 scheme
        if config.method == "sparsegpt":
            if config.sparsegpt_prunen > 0 and config.sparsegpt_prunem > 0:
                # N:M 結構化稀疏
                suffix = f"{config.sparsegpt_prunen}x{config.sparsegpt_prunem}"
            else:
                # 非結構化稀疏，顯示稀疏比例
                suffix = f"sparse{int(config.sparsegpt_sparsity * 100)}"
            return f"quantized/{self.model_id}-llmc-{config.method}-{suffix}-{ts}"

        return f"quantized/{self.model_id}-llmc-{config.method}-{config.scheme}-{ts}"

    def _log_start(self, method: str, config: LLMCompressorConfig):
        """記錄開始訊息"""
        logger.info("=" * 80)

        # SparseGPT 是稀疏化方法，不是量化方法
        if method == "sparsegpt":
            logger.info(f"開始 llm-compressor {method.upper()} 稀疏化")
            logger.info(f"模型: {self.model_path}")
            if config.sparsegpt_prunen > 0 and config.sparsegpt_prunem > 0:
                logger.info(f"稀疏模式: {config.sparsegpt_prunen}:{config.sparsegpt_prunem} 結構化稀疏")
            else:
                logger.info(f"稀疏比例: {config.sparsegpt_sparsity * 100:.0f}%")
        else:
            logger.info(f"開始 llm-compressor {method.upper()} 量化")
            logger.info(f"模型: {self.model_path}")
            logger.info(f"方案: {config.scheme}")

        if config.smoothquant_enabled:
            logger.info(f"SmoothQuant: 啟用 (strength={config.smoothquant_strength})")
        logger.info("=" * 80)

    def _log_complete(self, output_dir: str, method: str):
        """記錄完成訊息"""
        logger.info("=" * 80)
        if method == "sparsegpt":
            logger.info(f"llm-compressor {method.upper()} 稀疏化成功完成!")
        else:
            logger.info(f"llm-compressor {method.upper()} 量化成功完成!")
        logger.info(f"輸出路徑: {output_dir}")
        logger.info("=" * 80)

    def _prepare_calibration_data(self, config: LLMCompressorConfig):
        """
        準備校準資料集

        Returns:
            Dataset: HuggingFace Dataset 對象，包含 'text' 欄位
        """
        from datasets import load_dataset, Dataset

        logger.info("準備校準資料...")
        logger.info(f"  資料集: {config.calibration_dataset}")
        logger.info(f"  分割: {config.calibration_split}")
        logger.info(f"  文本欄位: {config.calibration_text_column}")

        # 載入資料集
        # 處理特殊資料集格式 (如 gsm8k 需要 config name)
        if config.calibration_dataset == "openai/gsm8k":
            ds = load_dataset(config.calibration_dataset, "main", split=config.calibration_split)
        else:
            ds = load_dataset(config.calibration_dataset, split=config.calibration_split)

        # 限制樣本數量
        num_samples = min(config.num_calibration_samples, len(ds))
        ds = ds.select(range(num_samples))
        logger.info(f"  樣本數: {num_samples}")

        # 提取文本並建立新的 Dataset
        # llmcompressor 期望 'text' 欄位
        text_column = config.calibration_text_column

        def extract_text(example):
            return {"text": example.get(text_column, "")}

        # 只保留 text 欄位
        ds = ds.map(extract_text, remove_columns=ds.column_names)
        logger.info(f"  資料集欄位: {ds.column_names}")

        return ds

    def _build_recipe(self, config: LLMCompressorConfig) -> List[Any]:
        """根據配置建立 recipe"""
        recipe = []

        # 1. SmoothQuant 前處理（可選）
        if config.smoothquant_enabled:
            try:
                from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
                recipe.append(SmoothQuantModifier(
                    smoothing_strength=config.smoothquant_strength
                ))
                logger.info(f"添加 SmoothQuant 前處理 (strength={config.smoothquant_strength})")
            except ImportError:
                logger.warning("SmoothQuantModifier 不可用，跳過")

        # 2. 主量化方法
        if config.method == "gptq":
            from llmcompressor.modifiers.quantization import GPTQModifier
            modifier = GPTQModifier(
                scheme=config.scheme,
                targets=config.targets,
                ignore=config.ignore,
                dampening_frac=config.gptq_dampening,
                actorder=config.gptq_actorder,
            )
            recipe.append(modifier)
            logger.info(f"添加 GPTQModifier (scheme={config.scheme})")

        elif config.method == "awq":
            try:
                from llmcompressor.modifiers.awq import AWQModifier
                modifier = AWQModifier(
                    scheme=config.scheme,
                    targets=config.targets,
                    ignore=config.ignore,
                    duo_scaling=config.awq_duo_scaling,
                    n_grid=config.awq_n_grid,
                )
                recipe.append(modifier)
                logger.info(f"添加 AWQModifier (scheme={config.scheme})")
            except ImportError:
                # 如果 AWQModifier 不存在，回退到 GPTQModifier
                logger.warning("AWQModifier 不可用，使用 GPTQModifier 替代")
                from llmcompressor.modifiers.quantization import GPTQModifier
                modifier = GPTQModifier(
                    scheme=config.scheme,
                    targets=config.targets,
                    ignore=config.ignore,
                )
                recipe.append(modifier)

        elif config.method == "autoround":
            try:
                from llmcompressor.modifiers.autoround import AutoRoundModifier
                modifier_kwargs = {
                    "scheme": config.scheme,
                    "targets": config.targets,
                    "ignore": config.ignore,
                    "iters": config.autoround_iters,
                    "batch_size": config.autoround_batch_size,
                    "enable_torch_compile": config.autoround_enable_torch_compile,
                }
                if config.autoround_lr is not None:
                    modifier_kwargs["lr"] = config.autoround_lr
                modifier = AutoRoundModifier(**modifier_kwargs)
                recipe.append(modifier)
                logger.info(f"添加 AutoRoundModifier (scheme={config.scheme}, iters={config.autoround_iters})")
            except ImportError as e:
                raise ImportError(
                    f"AutoRoundModifier 不可用，請確認 llmcompressor 版本支援 AutoRound: {e}"
                )

        elif config.method == "sparsegpt":
            from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
            modifier_kwargs = {
                "targets": config.targets,
                "sparsity": config.sparsegpt_sparsity,
            }
            # N:M 結構化稀疏使用 mask_structure 參數（如 "2:4"）
            if config.sparsegpt_prunen > 0 and config.sparsegpt_prunem > 0:
                modifier_kwargs["mask_structure"] = f"{config.sparsegpt_prunen}:{config.sparsegpt_prunem}"
                # N:M 稀疏比例 = (M-N)/M
                nm_sparsity = (config.sparsegpt_prunem - config.sparsegpt_prunen) / config.sparsegpt_prunem
                logger.info(f"添加 SparseGPTModifier ({config.sparsegpt_prunen}:{config.sparsegpt_prunem} 結構化稀疏, 實際稀疏比例={nm_sparsity:.0%})")
            else:
                # 非結構化稀疏（mask_structure 預設 "0:0"）
                logger.info(f"添加 SparseGPTModifier (非結構化稀疏, sparsity={config.sparsegpt_sparsity})")

            modifier = SparseGPTModifier(**modifier_kwargs)
            recipe.append(modifier)

        return recipe

    def quantize(self, config: LLMCompressorConfig) -> str:
        """
        執行量化

        Args:
            config: LLMCompressorConfig 配置物件

        Returns:
            str: 量化後模型的輸出路徑

        Raises:
            ValueError: 如果配置驗證失敗
            ImportError: 如果缺少必要的依賴
        """
        try:
            from transformers import AutoModelForCausalLM
            from llmcompressor import oneshot
        except ImportError as e:
            raise ImportError(
                f"請先安裝必要的依賴:\n"
                f"   pip install llmcompressor transformers datasets\n"
                f"錯誤: {e}"
            )

        # 驗證配置
        config.validate()

        self._log_start(config.method, config)
        output_dir = self._generate_output_dir(config)

        # 1. 載入 tokenizer
        logger.info("載入 tokenizer...")
        self._tokenizer = self._safe_load_tokenizer(config.hf_token)

        # 2. 載入模型
        logger.info("載入模型...")
        self._cleanup_memory()

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype="auto",
            device_map="auto",
            token=config.hf_token,
            trust_remote_code=True,
        )
        logger.info(f"  模型載入完成: {type(self._model).__name__}")

        # 3. 準備校準資料
        calibration_data = self._prepare_calibration_data(config)

        # 4. 建立 recipe
        logger.info("建立量化 recipe...")
        recipe = self._build_recipe(config)

        # 5. 執行量化
        logger.info("開始量化（這可能需要較長時間）...")

        # 重置 GPU 記憶體統計
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        oneshot(
            model=self._model,
            tokenizer=self._tokenizer,
            dataset=calibration_data,
            recipe=recipe,
            max_seq_length=config.max_seq_length,
            num_calibration_samples=config.num_calibration_samples,
        )

        # 統計 GPU 記憶體
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logger.info(f"  量化過程中 GPU 峰值記憶體使用: {peak_memory:.2f} MB")

        logger.info("  量化完成")

        # 6. 儲存
        logger.info("儲存模型...")
        os.makedirs(output_dir, exist_ok=True)

        if config.save_compressed:
            self._model.save_pretrained(output_dir, save_compressed=True)
        else:
            self._model.save_pretrained(output_dir)

        self._tokenizer.save_pretrained(output_dir)
        logger.info(f"  儲存完成")

        self._log_complete(output_dir, config.method)

        # 清理
        self._cleanup_memory()

        return output_dir

    # ========================================================================
    # 快捷方法
    # ========================================================================

    def quantize_gptq(self, **kwargs) -> str:
        """
        快捷方法: GPTQ 量化

        Args:
            scheme: 量化方案 (預設 "W4A16")
            **kwargs: 其他 LLMCompressorConfig 參數

        Returns:
            str: 量化後模型的輸出路徑

        Examples:
            >>> quantizer = LLMCompressorQuantizer("meta-llama/Llama-3.2-1B-Instruct")
            >>> output = quantizer.quantize_gptq(scheme="W4A16", group_size=128)
        """
        kwargs['method'] = 'gptq'
        config = LLMCompressorConfig(**kwargs)
        return self.quantize(config)

    def quantize_awq(self, **kwargs) -> str:
        """
        快捷方法: AWQ 量化

        Args:
            scheme: 量化方案 (預設 "W4A16")
            **kwargs: 其他 LLMCompressorConfig 參數

        Returns:
            str: 量化後模型的輸出路徑

        Examples:
            >>> quantizer = LLMCompressorQuantizer("meta-llama/Llama-3.2-1B-Instruct")
            >>> output = quantizer.quantize_awq(scheme="W4A16")
        """
        kwargs['method'] = 'awq'
        config = LLMCompressorConfig(**kwargs)
        return self.quantize(config)

    def quantize_autoround(self, **kwargs) -> str:
        """
        快捷方法: AutoRound 量化

        Args:
            scheme: 量化方案 (預設 "W4A16")
            iters: 迭代次數 (預設 200)
            **kwargs: 其他 LLMCompressorConfig 參數

        Returns:
            str: 量化後模型的輸出路徑

        Examples:
            >>> quantizer = LLMCompressorQuantizer("meta-llama/Llama-3.2-1B-Instruct")
            >>> output = quantizer.quantize_autoround(scheme="W4A16", autoround_iters=200)
        """
        kwargs['method'] = 'autoround'
        config = LLMCompressorConfig(**kwargs)
        return self.quantize(config)

    def quantize_sparsegpt(self, **kwargs) -> str:
        """
        快捷方法: SparseGPT 稀疏化

        Args:
            sparsegpt_sparsity: 稀疏比例 (預設 0.5)
            **kwargs: 其他 LLMCompressorConfig 參數

        Returns:
            str: 量化後模型的輸出路徑

        Examples:
            >>> quantizer = LLMCompressorQuantizer("meta-llama/Llama-3.2-1B-Instruct")
            >>> output = quantizer.quantize_sparsegpt(sparsegpt_sparsity=0.5)

            # N:M 稀疏
            >>> output = quantizer.quantize_sparsegpt(sparsegpt_prunen=2, sparsegpt_prunem=4)
        """
        kwargs['method'] = 'sparsegpt'
        config = LLMCompressorConfig(**kwargs)
        return self.quantize(config)

    @staticmethod
    def list_methods() -> List[str]:
        """列出所有支援的量化方法"""
        return ["gptq", "awq", "autoround", "sparsegpt"]

    @staticmethod
    def list_schemes() -> List[str]:
        """列出所有支援的量化方案"""
        return ["W4A16", "W8A16", "W8A8", "FP8"]

    # ========================================================================
    # Recipe 組合方法（支援同時量化+剪枝）
    # ========================================================================

    def _build_modifier(self, modifier_config: dict, defaults: dict) -> Any:
        """
        根據配置建立單個 modifier

        Args:
            modifier_config: modifier 配置字典，必須包含 'type' 欄位
            defaults: 預設參數

        Returns:
            Modifier 物件
        """
        modifier_type = modifier_config.get("type")
        common = defaults.get("common", {})

        if modifier_type == "smoothquant":
            from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
            strength = modifier_config.get("smoothing_strength",
                       defaults.get("smoothquant", {}).get("smoothing_strength", 0.8))
            logger.info(f"  - SmoothQuantModifier (strength={strength})")
            return SmoothQuantModifier(smoothing_strength=strength)

        elif modifier_type == "sparsegpt":
            from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
            sparse_defaults = defaults.get("sparsegpt", {})
            sparsity = modifier_config.get("sparsity", sparse_defaults.get("sparsity", 0.5))
            kwargs = {
                "targets": modifier_config.get("targets", common.get("targets", "Linear")),
                "sparsity": sparsity,
            }
            prunen = modifier_config.get("prunen", sparse_defaults.get("prunen", 0))
            prunem = modifier_config.get("prunem", sparse_defaults.get("prunem", 0))
            # N:M 結構化稀疏使用 mask_structure 參數（如 "2:4"）
            if prunen > 0 and prunem > 0:
                kwargs["mask_structure"] = f"{prunen}:{prunem}"
                nm_sparsity = (prunem - prunen) / prunem
                logger.info(f"  - SparseGPTModifier ({prunen}:{prunem} 結構化稀疏, 實際稀疏比例={nm_sparsity:.0%})")
            else:
                # 非結構化稀疏（mask_structure 預設 "0:0"）
                logger.info(f"  - SparseGPTModifier (非結構化稀疏, sparsity={sparsity})")
            return SparseGPTModifier(**kwargs)

        elif modifier_type == "gptq":
            from llmcompressor.modifiers.quantization import GPTQModifier
            gptq_defaults = defaults.get("gptq", {})

            # 處理 actorder：新版需要字串 ('group', 'weight', 'dynamic', 'static')
            actorder_val = modifier_config.get("actorder", gptq_defaults.get("actorder", True))
            if isinstance(actorder_val, bool):
                actorder_val = "weight" if actorder_val else None

            kwargs = {
                "scheme": modifier_config.get("scheme", gptq_defaults.get("scheme", "W4A16")),
                "targets": modifier_config.get("targets", common.get("targets", "Linear")),
                "ignore": modifier_config.get("ignore", common.get("ignore", ["lm_head"])),
                "dampening_frac": modifier_config.get("dampening", gptq_defaults.get("dampening", 0.01)),
            }
            if actorder_val:
                kwargs["actorder"] = actorder_val

            # block_size (group_size) - 只有明確指定時才傳入
            block_size = modifier_config.get("block_size", gptq_defaults.get("block_size"))
            if block_size is not None:
                kwargs["block_size"] = block_size

            logger.info(f"  - GPTQModifier (scheme={kwargs['scheme']}, block_size={block_size or 'default'})")
            return GPTQModifier(**kwargs)

        elif modifier_type == "awq":
            try:
                from llmcompressor.modifiers.awq import AWQModifier
                awq_defaults = defaults.get("awq", {})
                kwargs = {
                    "scheme": modifier_config.get("scheme", awq_defaults.get("scheme", "W4A16")),
                    "targets": modifier_config.get("targets", common.get("targets", "Linear")),
                    "ignore": modifier_config.get("ignore", common.get("ignore", ["lm_head"])),
                    "duo_scaling": modifier_config.get("duo_scaling", awq_defaults.get("duo_scaling", True)),
                    "n_grid": modifier_config.get("n_grid", awq_defaults.get("n_grid", 20)),
                }
                logger.info(f"  - AWQModifier (scheme={kwargs['scheme']})")
                return AWQModifier(**kwargs)
            except ImportError:
                logger.warning("AWQModifier 不可用，使用 GPTQModifier 替代")
                return self._build_modifier({**modifier_config, "type": "gptq"}, defaults)

        elif modifier_type == "autoround":
            try:
                from llmcompressor.modifiers.autoround import AutoRoundModifier
                autoround_defaults = defaults.get("autoround", {})
                kwargs = {
                    "scheme": modifier_config.get("scheme", autoround_defaults.get("scheme", "W4A16")),
                    "targets": modifier_config.get("targets", common.get("targets", "Linear")),
                    "ignore": modifier_config.get("ignore", common.get("ignore", ["lm_head"])),
                    "iters": modifier_config.get("iters", autoround_defaults.get("iters", 200)),
                    "batch_size": modifier_config.get("batch_size", autoround_defaults.get("batch_size", 8)),
                    "enable_torch_compile": modifier_config.get("enable_torch_compile",
                                           autoround_defaults.get("enable_torch_compile", True)),
                }
                lr = modifier_config.get("lr", autoround_defaults.get("lr"))
                if lr is not None:
                    kwargs["lr"] = lr
                logger.info(f"  - AutoRoundModifier (scheme={kwargs['scheme']}, iters={kwargs['iters']})")
                return AutoRoundModifier(**kwargs)
            except ImportError as e:
                raise ImportError(f"AutoRoundModifier 不可用，請確認 llmcompressor 版本: {e}")

        else:
            raise ValueError(f"不支援的 modifier 類型: {modifier_type}")

    def quantize_with_recipe(
        self,
        recipe: List[dict],
        calibration: dict,
        output: dict,
        modifier_defaults: Optional[dict] = None,
        execution_mode: str = "single"
    ) -> str:
        """
        使用 recipe 列表執行量化+剪枝組合

        Args:
            recipe: modifier 配置列表，每個元素包含 'type' 和參數
            calibration: 校準資料集配置
            output: 輸出配置
            modifier_defaults: modifier 預設參數
            execution_mode: 執行模式 ("single" 或 "staged")

        Returns:
            str: 輸出模型路徑

        Example:
            >>> recipe = [
            ...     {"type": "sparsegpt", "sparsity": 0.5, "prunen": 2, "prunem": 4},
            ...     {"type": "gptq", "scheme": "W4A16"}
            ... ]
            >>> output_path = quantizer.quantize_with_recipe(recipe, calibration, output)
        """
        try:
            from transformers import AutoModelForCausalLM
            from llmcompressor import oneshot
        except ImportError as e:
            raise ImportError(f"請安裝必要依賴: pip install llmcompressor transformers datasets\n{e}")

        if not recipe:
            raise ValueError("recipe 不能為空")

        defaults = modifier_defaults or {}

        # 解析配置
        hf_token = output.get("hf_token") or os.environ.get("HF_TOKEN")
        save_compressed = output.get("save_compressed", True)

        # 生成輸出目錄名稱
        method_names = [m.get("type", "unknown") for m in recipe]
        method_str = "+".join(method_names)
        scheme = next((m.get("scheme") for m in recipe if m.get("scheme")), "mixed")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_path.replace("/", "-").replace("\\", "-")
        output_dir = output.get("dir") or f"quantized/{model_name}-llmc-{method_str}-{scheme}-{timestamp}"

        logger.info("=" * 80)
        logger.info(f"開始 llm-compressor Recipe 組合壓縮")
        logger.info(f"模型: {self.model_path}")
        logger.info(f"Recipe: {method_str}")
        logger.info("=" * 80)

        # 載入 tokenizer
        logger.info("載入 tokenizer...")
        self._tokenizer = self._safe_load_tokenizer(hf_token)

        # 載入模型
        logger.info("載入模型...")
        self._cleanup_memory()
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            token=hf_token,
        )
        logger.info(f"  模型載入完成: {type(self._model).__name__}")

        # 準備校準資料集
        logger.info("準備校準資料...")
        calib_config = LLMCompressorConfig(
            calibration_dataset=calibration.get("dataset", "openai/gsm8k"),
            calibration_split=calibration.get("split", "train"),
            calibration_text_column=calibration.get("text_column", "question"),
            num_calibration_samples=calibration.get("num_samples", 256),
            max_seq_length=calibration.get("max_seq_length", 512),
        )
        dataset = self._prepare_calibration_data(calib_config)

        # 根據執行模式處理
        if execution_mode == "staged":
            # 分階段執行：先剪枝，再量化
            output_path = self._execute_staged(recipe, defaults, dataset, output_dir, save_compressed, calib_config)
        else:
            # 單次執行所有 modifier
            output_path = self._execute_single(recipe, defaults, dataset, output_dir, save_compressed, calib_config)

        self._cleanup_memory()
        return output_path

    def _execute_single(self, recipe, defaults, dataset, output_dir, save_compressed, calib_config):
        """單次執行所有 modifier"""
        from llmcompressor import oneshot

        logger.info("建立 Recipe（單次執行模式）...")
        modifiers = []
        for mod_config in recipe:
            modifier = self._build_modifier(mod_config, defaults)
            modifiers.append(modifier)

        logger.info("開始壓縮...")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        oneshot(
            model=self._model,
            tokenizer=self._tokenizer,
            dataset=dataset,
            recipe=modifiers,
            max_seq_length=calib_config.max_seq_length,
            num_calibration_samples=calib_config.num_calibration_samples,
        )

        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            logger.info(f"  GPU 峰值記憶體: {peak_memory:.2f} MB")

        # 儲存
        logger.info("儲存模型...")
        os.makedirs(output_dir, exist_ok=True)
        if save_compressed:
            self._model.save_pretrained(output_dir, save_compressed=True)
        else:
            self._model.save_pretrained(output_dir)
        self._tokenizer.save_pretrained(output_dir)

        logger.info("=" * 80)
        logger.info(f"壓縮完成！輸出: {output_dir}")
        logger.info("=" * 80)

        return output_dir

    def _execute_staged(self, recipe, defaults, dataset, output_dir, save_compressed, calib_config):
        """分階段執行（先剪枝，再量化）"""
        from llmcompressor import oneshot

        # 分類 modifier
        sparsity_mods = [m for m in recipe if m.get("type") in ["sparsegpt", "smoothquant"]]
        quant_mods = [m for m in recipe if m.get("type") in ["gptq", "awq", "autoround"]]

        current_model = self._model
        intermediate_dir = None

        # 階段 1: 剪枝
        if sparsity_mods:
            logger.info("=" * 40)
            logger.info("階段 1: 稀疏化")
            logger.info("=" * 40)

            modifiers = [self._build_modifier(m, defaults) for m in sparsity_mods]

            oneshot(
                model=current_model,
                tokenizer=self._tokenizer,
                dataset=dataset,
                recipe=modifiers,
                max_seq_length=calib_config.max_seq_length,
                num_calibration_samples=calib_config.num_calibration_samples,
            )

            # 如果還有量化階段，先存到中間目錄
            if quant_mods:
                intermediate_dir = output_dir + "_sparse_intermediate"
                os.makedirs(intermediate_dir, exist_ok=True)
                current_model.save_pretrained(intermediate_dir)
                self._tokenizer.save_pretrained(intermediate_dir)
                logger.info(f"  中間模型儲存至: {intermediate_dir}")

                # 重新載入模型
                from transformers import AutoModelForCausalLM
                self._cleanup_memory()
                current_model = AutoModelForCausalLM.from_pretrained(
                    intermediate_dir,
                    torch_dtype="auto",
                    device_map="auto",
                )

        # 階段 2: 量化
        if quant_mods:
            logger.info("=" * 40)
            logger.info("階段 2: 量化")
            logger.info("=" * 40)

            modifiers = [self._build_modifier(m, defaults) for m in quant_mods]

            oneshot(
                model=current_model,
                tokenizer=self._tokenizer,
                dataset=dataset,
                recipe=modifiers,
                max_seq_length=calib_config.max_seq_length,
                num_calibration_samples=calib_config.num_calibration_samples,
            )

        # 儲存最終模型
        logger.info("儲存最終模型...")
        os.makedirs(output_dir, exist_ok=True)
        if save_compressed:
            current_model.save_pretrained(output_dir, save_compressed=True)
        else:
            current_model.save_pretrained(output_dir)
        self._tokenizer.save_pretrained(output_dir)

        # 清理中間文件
        if intermediate_dir and os.path.exists(intermediate_dir):
            import shutil
            shutil.rmtree(intermediate_dir)
            logger.info("  已清理中間文件")

        logger.info("=" * 80)
        logger.info(f"分階段壓縮完成！輸出: {output_dir}")
        logger.info("=" * 80)

        return output_dir
