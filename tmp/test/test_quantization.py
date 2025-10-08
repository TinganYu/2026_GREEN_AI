"""
量化測試腳本 - 簡化版
=====================

使用 model_config.yaml 配置文件進行模型量化測試
"""

import sys
import yaml
import argparse
from pathlib import Path

# 添加上層目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from method.quantization import Quantizer, GPTQConfig, AWQConfig, BNBConfig


class QuantizationConfigLoader:
    """從 YAML 文件加載量化配置"""
    
    def __init__(self, config_path: str = "../config/model_config.yaml"):
        self.config_path = Path(__file__).parent / config_path
        self.config = self._load_yaml()
    
    def _load_yaml(self):
        """加載 YAML 配置"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_model_path(self) -> str:
        return self.config['model']['name']
    
    def is_quantization_enabled(self) -> bool:
        return self.config['quantization']['enabled']
    
    def should_save_model(self) -> bool:
        return self.config['quantization'].get('save_model', True)
    
    def get_framework(self) -> str:
        return self.config['quantization']['framework'].lower()
    
    def get_gptq_config(self, hf_token: str = None) -> GPTQConfig:
        """創建 GPTQ 配置"""
        params = self.config['quantization']['methods']['gptq']
        return GPTQConfig(
            bits=params['bits'],
            group_size=params['group_size'],
            damp_percent=params['damp_percent'],
            desc_act=params['desc_act'],
            sym=params['sym'],
            true_sequential=params['true_sequential'],
            calib_num=params['calib_num'],
            output_dir=params.get('output_dir'),
            hf_token=hf_token
        )
    
    def get_awq_config(self, hf_token: str = None) -> AWQConfig:
        """創建 AWQ 配置"""
        params = self.config['quantization']['methods']['awq']
        modules = params.get('modules_to_not_convert')
        if modules is not None and len(modules) == 0:
            modules = None
        
        return AWQConfig(
            w_bit=params['w_bit'],
            zero_point=params['zero_point'],
            q_group_size=params['q_group_size'],
            version=params['version'],
            modules_to_not_convert=modules,
            output_dir=params.get('output_dir'),
            hf_token=hf_token
        )
    
    def get_bnb_config(self, hf_token: str = None) -> BNBConfig:
        """創建 BNB 配置"""
        params = self.config['quantization']['methods']['bnb']
        skip_modules = params.get('llm_int8_skip_modules')
        if skip_modules is not None and len(skip_modules) == 0:
            skip_modules = None
        
        return BNBConfig(
            bits=params['bits'],
            bnb_4bit_quant_type=params['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=params['bnb_4bit_use_double_quant'],
            bnb_4bit_compute_dtype=params['bnb_4bit_compute_dtype'],
            llm_int8_threshold=params['llm_int8_threshold'],
            llm_int8_skip_modules=skip_modules,
            llm_int8_has_fp16_weight=params.get('llm_int8_has_fp16_weight', False),
            llm_int8_enable_fp32_cpu_offload=params.get('llm_int8_enable_fp32_cpu_offload', False),
            output_dir=params.get('output_dir'),
            hf_token=hf_token
        )
    
    def print_summary(self):
        """打印配置摘要"""
        print("\n" + "=" * 60)
        print(f"模型: {self.get_model_path()}")
        print(f"方法: {self.get_framework().upper()}")
        print(f"量化: {'啟用' if self.is_quantization_enabled() else '停用'}")
        print(f"保存: {'是' if self.should_save_model() else '否'}")
        print("=" * 60 + "\n")


class QuantizationTester:
    """量化測試器"""
    
    def __init__(self, config_loader: QuantizationConfigLoader):
        self.config_loader = config_loader
        self.quantizer = None
        self.output_path = None
    
    def run_test(self, hf_token: str = None, override_method: str = None):
        """運行量化測試"""
        if not self.config_loader.is_quantization_enabled():
            print("量化未啟用")
            return None
        
        # 打印配置
        self.config_loader.print_summary()
        
        # 創建量化器
        model_path = self.config_loader.get_model_path()
        self.quantizer = Quantizer(model_path)
        
        # 選擇方法
        method = override_method or self.config_loader.get_framework()
        
        # 創建配置
        if method == "gptq":
            config = self.config_loader.get_gptq_config(hf_token)
        elif method == "awq":
            config = self.config_loader.get_awq_config(hf_token)
        elif method == "bnb":
            config = self.config_loader.get_bnb_config(hf_token)
        else:
            raise ValueError(f"不支援的量化方法: {method}")
        
        # 處理 save_model 參數
        should_save = self.config_loader.should_save_model()
        if not should_save:
            config.output_dir = None
        
        # 執行量化
        self.output_path = self.quantizer.quantize(config)
        
        if should_save and self.output_path:
            print(f"\n✓ 已保存至: {self.output_path}\n")
        else:
            print("\n✓ 量化完成（未保存模型）\n")
        
        return self.output_path
    
    def verify_output(self) -> bool:
        """驗證輸出模型"""
        if not self.output_path:
            print("未保存模型，跳過驗證")
            return False
        
        output_dir = Path(self.output_path)
        if not output_dir.exists():
            print(f"輸出目錄不存在: {output_dir}")
            return False
        
        print("\n驗證輸出:")
        
        # 檢查文件
        files = ["config.json", "model.safetensors", "tokenizer.json"]
        all_good = True
        
        for file in files:
            exists = (output_dir / file).exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {file}")
            if file == "config.json" and not exists:
                all_good = False
        
        print()
        return all_good


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description="模型量化測試")
    parser.add_argument("--config", default="../config/model_config.yaml", help="配置文件")
    parser.add_argument("--method", choices=["gptq", "awq", "bnb"], help="量化方法")
    parser.add_argument("--hf-token", help="HuggingFace token")
    parser.add_argument("--verify", action="store_true", help="驗證輸出")
    
    args = parser.parse_args()
    
    try:
        print("\n量化測試開始\n")
        
        # 載入配置並運行測試
        config_loader = QuantizationConfigLoader(args.config)
        tester = QuantizationTester(config_loader)
        output = tester.run_test(hf_token=args.hf_token, override_method=args.method)
        
        # 驗證
        if args.verify and output:
            tester.verify_output()
        
        print("測試完成\n")
        
    except KeyboardInterrupt:
        print("\n測試被中斷\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n測試失敗: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
