"""
llm-compressor 量化測試腳本
============================

使用 llmcompressor_config.yaml 配置進行量化測試

使用方式:
    # 使用預設配置
    python tmp/test/test_llmcompressor.py

    # 指定配置檔案
    python tmp/test/test_llmcompressor.py --config tmp/config/llmcompressor_config.yaml

    # 覆寫量化方法
    python tmp/test/test_llmcompressor.py --method autoround

    # 覆寫量化方案
    python tmp/test/test_llmcompressor.py --scheme W8A16

    # 啟用 SmoothQuant 前處理
    python tmp/test/test_llmcompressor.py --smoothquant
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# 確保可以導入專案模組
sys.path.insert(0, str(Path(__file__).parent.parent))

from method.llmcompressor_quantization import LLMCompressorQuantizer, LLMCompressorConfig


# 設定 logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class LLMCompressorConfigLoader:
    """從 YAML 載入 llm-compressor 配置"""

    def __init__(self, config_path: str):
        """
        初始化配置載入器

        Args:
            config_path: 配置檔案路徑
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置檔案不存在: {config_path}")
        self.config = self._load_yaml()

    def _load_yaml(self) -> dict:
        """載入 YAML 配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def get_model_path(self) -> str:
        """取得模型路徑"""
        return self.config['model']['name']

    def is_enabled(self) -> bool:
        """檢查量化是否啟用"""
        return self.config.get('quantization', {}).get('enabled', True)

    def get_config(self, hf_token: str = None) -> LLMCompressorConfig:
        """
        建立 LLMCompressorConfig

        Args:
            hf_token: HuggingFace token

        Returns:
            LLMCompressorConfig 物件
        """
        q = self.config.get('quantization', {})
        c = self.config.get('calibration', {})
        o = self.config.get('output', {})

        # 取得當前方法
        method = q.get('method', 'gptq')

        # 取得方法專用參數
        method_params = q.get('methods', {}).get(method, {})

        # 取得 GPTQ 專用參數（block_size 只有 GPTQ 使用）
        gptq_params = q.get('methods', {}).get('gptq', {})

        # 建立配置
        return LLMCompressorConfig(
            # 主量化方法
            method=method,
            scheme=q.get('scheme', 'W4A16'),
            targets=q.get('targets', 'Linear'),
            ignore=q.get('ignore', ['lm_head']),

            # SmoothQuant
            smoothquant_enabled=q.get('smoothquant', {}).get('enabled', False),
            smoothquant_strength=q.get('smoothquant', {}).get('smoothing_strength', 0.8),

            # GPTQ 專用（block_size 只從 GPTQ 區段讀取）
            gptq_block_size=gptq_params.get('block_size', 128),
            gptq_actorder=gptq_params.get('actorder', 'weight'),
            gptq_dampening=gptq_params.get('dampening', 0.01),

            # AWQ 專用
            awq_duo_scaling=method_params.get('duo_scaling', True),
            awq_n_grid=method_params.get('n_grid', 20),

            # AutoRound 專用
            autoround_iters=method_params.get('iters', 200),
            autoround_batch_size=method_params.get('batch_size', 8),
            autoround_lr=method_params.get('lr'),
            autoround_enable_torch_compile=method_params.get('enable_torch_compile', True),

            # SparseGPT 專用
            sparsegpt_sparsity=method_params.get('sparsity', 0.5),
            sparsegpt_prunen=method_params.get('prunen', 0),
            sparsegpt_prunem=method_params.get('prunem', 0),

            # 校準設定
            calibration_dataset=c.get('dataset', 'openai/gsm8k'),
            calibration_split=c.get('split', 'train'),
            calibration_text_column=c.get('text_column', 'question'),
            num_calibration_samples=c.get('num_samples', 512),
            max_seq_length=c.get('max_seq_length', 2048),

            # 輸出設定
            output_dir=o.get('dir'),
            save_compressed=o.get('save_compressed', True),
            hf_token=hf_token,
        )

    def is_recipe_mode(self) -> bool:
        """檢查是否為 recipe 列表模式（v2 格式）"""
        return 'recipe' in self.config and isinstance(self.config['recipe'], list)

    def get_recipe_config(self) -> dict:
        """
        取得 recipe 模式配置（v2 格式）

        Returns:
            dict: 包含 recipe, calibration, output, modifier_defaults, execution_mode
        """
        return {
            'recipe': self.config.get('recipe', []),
            'calibration': self.config.get('calibration', {}),
            'output': self.config.get('output', {}),
            'modifier_defaults': self.config.get('modifier_defaults', {}),
            'execution_mode': self.config.get('execution', {}).get('mode', 'single'),
        }

    def print_summary(self):
        """打印配置摘要"""
        q = self.config.get('quantization', {})
        c = self.config.get('calibration', {})
        method = q.get('method', 'gptq')
        method_params = q.get('methods', {}).get(method, {})

        print("\n" + "=" * 60)
        print("llm-compressor 量化配置摘要")
        print("=" * 60)
        print(f"模型: {self.get_model_path()}")
        print(f"方法: {method.upper()}")
        print(f"方案: {q.get('scheme', 'W4A16')}")

        # 根據方法顯示專用參數
        if method == 'gptq':
            print(f"分組大小 (block_size): {method_params.get('block_size', 128)}")
            print(f"激活排序 (actorder): {method_params.get('actorder', 'weight')}")
        elif method == 'awq':
            print(f"雙重縮放: {method_params.get('duo_scaling', True)}")
            print(f"網格搜索: {method_params.get('n_grid', 20)}")
        elif method == 'autoround':
            print(f"迭代次數: {method_params.get('iters', 200)}")
        elif method == 'sparsegpt':
            print(f"稀疏比例: {method_params.get('sparsity', 0.5)}")
            prunen = method_params.get('prunen', 0)
            prunem = method_params.get('prunem', 0)
            if prunen > 0 and prunem > 0:
                print(f"結構化稀疏: {prunen}:{prunem}")
            else:
                print("非結構化稀疏")

        print(f"SmoothQuant: {'啟用' if q.get('smoothquant', {}).get('enabled', False) else '停用'}")
        print("-" * 60)
        print(f"校準資料集: {c.get('dataset', 'openai/gsm8k')}")
        print(f"校準樣本數: {c.get('num_samples', 512)}")
        print(f"最大序列長度: {c.get('max_seq_length', 2048)}")
        print("=" * 60 + "\n")


class LLMCompressorTester:
    """llm-compressor 量化測試器"""

    def __init__(self, config_loader: LLMCompressorConfigLoader):
        """
        初始化測試器

        Args:
            config_loader: 配置載入器
        """
        self.config_loader = config_loader
        self.output_path = None

    def run_test(
        self,
        hf_token: str = None,
        override_method: str = None,
        override_scheme: str = None,
        enable_smoothquant: bool = None,
    ) -> str:
        """
        執行量化測試

        Args:
            hf_token: HuggingFace token
            override_method: 覆寫量化方法
            override_scheme: 覆寫量化方案
            enable_smoothquant: 覆寫 SmoothQuant 設定

        Returns:
            量化後模型的輸出路徑
        """
        # 檢查是否為 recipe 模式（v2 格式）
        if self.config_loader.is_recipe_mode():
            return self._run_recipe_test(hf_token)

        if not self.config_loader.is_enabled():
            logger.warning("量化未啟用 (quantization.enabled = false)")
            return None

        # 打印配置摘要
        self.config_loader.print_summary()

        # 取得模型路徑
        model_path = self.config_loader.get_model_path()

        # 建立量化器
        quantizer = LLMCompressorQuantizer(model_path)

        # 取得配置
        config = self.config_loader.get_config(hf_token)

        # 應用覆寫
        if override_method:
            logger.info(f"覆寫量化方法: {config.method} -> {override_method}")
            config.method = override_method

        if override_scheme:
            logger.info(f"覆寫量化方案: {config.scheme} -> {override_scheme}")
            config.scheme = override_scheme

        if enable_smoothquant is not None:
            logger.info(f"覆寫 SmoothQuant: {config.smoothquant_enabled} -> {enable_smoothquant}")
            config.smoothquant_enabled = enable_smoothquant

        # 執行量化
        try:
            self.output_path = quantizer.quantize(config)
            print(f"\n已保存至: {self.output_path}\n")
            return self.output_path
        except Exception as e:
            logger.error(f"量化失敗: {e}")
            raise

    def _run_recipe_test(self, hf_token: str = None) -> str:
        """
        執行 recipe 模式測試（v2 格式，支援組合）

        Args:
            hf_token: HuggingFace token

        Returns:
            量化後模型的輸出路徑
        """
        recipe_config = self.config_loader.get_recipe_config()

        # 打印 Recipe 摘要
        print("\n" + "=" * 60)
        print("llm-compressor Recipe 組合模式")
        print("=" * 60)
        print(f"模型: {self.config_loader.get_model_path()}")
        print(f"執行模式: {recipe_config['execution_mode']}")
        print("Recipe 步驟:")
        for i, mod in enumerate(recipe_config['recipe'], 1):
            mod_type = mod.get('type', 'unknown')
            params = {k: v for k, v in mod.items() if k != 'type'}
            print(f"  {i}. {mod_type}: {params}")
        print("=" * 60 + "\n")

        # 取得模型路徑
        model_path = self.config_loader.get_model_path()

        # 建立量化器
        quantizer = LLMCompressorQuantizer(model_path)

        # 添加 hf_token 到 output 配置
        output_config = recipe_config['output'].copy()
        output_config['hf_token'] = hf_token or os.getenv('HF_TOKEN')

        # 執行組合壓縮
        try:
            self.output_path = quantizer.quantize_with_recipe(
                recipe=recipe_config['recipe'],
                calibration=recipe_config['calibration'],
                output=output_config,
                modifier_defaults=recipe_config['modifier_defaults'],
                execution_mode=recipe_config['execution_mode'],
            )
            print(f"\n已保存至: {self.output_path}\n")
            return self.output_path
        except Exception as e:
            logger.error(f"Recipe 組合壓縮失敗: {e}")
            raise


def main():
    """主程式"""
    # 載入環境變數
    load_dotenv()

    # 解析命令列參數
    parser = argparse.ArgumentParser(
        description="llm-compressor 量化測試",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    # 使用預設配置
    python test_llmcompressor.py

    # 使用 AutoRound 方法
    python test_llmcompressor.py --method autoround

    # 使用 W8A8 方案並啟用 SmoothQuant
    python test_llmcompressor.py --scheme W8A8 --smoothquant

    # 使用 SparseGPT 稀疏化
    python test_llmcompressor.py --method sparsegpt
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="tmp/config/llmcompressor_config.yaml",
        help="配置檔案路徑 (預設: tmp/config/llmcompressor_config.yaml)"
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["gptq", "awq", "autoround", "sparsegpt"],
        help="覆寫量化方法"
    )

    parser.add_argument(
        "--scheme",
        type=str,
        choices=["W4A16", "W8A16", "W8A8", "FP8"],
        help="覆寫量化方案"
    )

    parser.add_argument(
        "--smoothquant",
        action="store_true",
        help="啟用 SmoothQuant 前處理"
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace token (也可透過 HF_TOKEN 環境變數設定)"
    )

    args = parser.parse_args()

    # 取得 HuggingFace token
    hf_token = args.hf_token or os.getenv('HF_TOKEN')

    # 載入配置
    try:
        config_loader = LLMCompressorConfigLoader(args.config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # 執行測試
    tester = LLMCompressorTester(config_loader)

    try:
        output_path = tester.run_test(
            hf_token=hf_token,
            override_method=args.method,
            override_scheme=args.scheme,
            enable_smoothquant=args.smoothquant if args.smoothquant else None,
        )

        if output_path:
            print("量化完成!")
            print(f"模型已保存至: {output_path}")
            print("\n下一步:")
            print(f"  1. 使用 test_eval.py 評估量化後的模型:")
            print(f"     # 編輯 tmp/config/model_config.yaml")
            print(f"     # 將 model.name 設為: {output_path}")
            print(f"     python tmp/test/test_eval.py")

    except Exception as e:
        logger.error(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
