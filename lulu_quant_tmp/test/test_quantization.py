"""
量化測試腳本 - 簡化版
=====================

使用 model_config.yaml 配置文件進行模型量化測試
"""

import sys
import yaml
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
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
        # 1. 取得原始參數字典的副本，避免修改原始配置
        params = self.config['quantization']['methods']['gptq'].copy()
        
        # 2. 定義動態量化規則
        dynamic_rules = {
            r"-:.*\.ALinear$": {}, # 跳過 ALinear
            r"+:.*\.BLinear$": {"bits": 4, "group_size": 32}, # 量化 BLinear
        }
        
        # 3. 使用 update 統一合併參數，這會自動覆蓋掉 YAML 裡的舊值，且不重複
        params.update({
            "hf_token": hf_token,
            "dynamic": dynamic_rules,
            # 如果 YAML 沒寫，這裡會補上預設值；如果 YAML 有寫，這裡會確保值正確
            "use_eora": params.get('use_eora', False),
            "eora_rank": params.get('eora_rank', 32)
        })
        
        # 4. 只傳遞解包後的 params
        return GPTQConfig(**params)
    # def get_gptq_config(self, hf_token: str = None) -> GPTQConfig:
    #     """創建 GPTQ 配置"""
    #     params = self.config['quantization']['methods']['gptq']
    #     # --- 新增動態量化規則 ---
    #     # 這裡會根據層的名稱路徑進行正則匹配
    #     dynamic_rules = {
    #         # 負匹配 (-:)：只要名稱包含 ALinear 就跳過（保持 FP16）
    #         r"-:.*\.ALinear$": {},
            
    #         # 正匹配 (+:)：只要名稱包含 BLinear 就執行 4-bit 量化
    #         # 這裡的參數會覆蓋掉 YAML 裡的預設 bits
    #         r"+:.*\.BLinear$": {"bits": 4, "group_size": 128},
    #     }
    #     use_eora = params.get('use_eora', False)
    #     eora_rank = params.get('eora_rank', 32)
    #     return GPTQConfig(
    #         **params,
    #         hf_token=hf_token,
    #         dynamic=dynamic_rules,
    #         use_eora=use_eora,
    #         eora_rank=eora_rank
    #     )
    
    def get_awq_config(self, hf_token: str = None) -> AWQConfig:
        """創建 AWQ 配置"""
        params = self.config['quantization']['methods']['awq']
        
        return AWQConfig(
            **params,
            hf_token=hf_token
        )
    
    def get_bnb_config(self, hf_token: str = None) -> BNBConfig:
        """創建 BNB 配置"""
        params = self.config['quantization']['methods']['bnb']
        
        return BNBConfig(
            **params,
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
    # 載入環境變數
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="模型量化測試")
    parser.add_argument("--config", default="../config/model_config.yaml", help="配置文件")
    parser.add_argument("--method", choices=["gptq", "awq", "bnb"], help="量化方法")
    parser.add_argument("--hf-token", help="HuggingFace token (可從 .env 文件讀取)")
    parser.add_argument("--verify", action="store_true", help="驗證輸出")
    
    args = parser.parse_args()
    
    # 優先從環境變數讀取 hf_token，如果沒有再從命令行參數讀取
    hf_token = os.getenv('HF_TOKEN') or args.hf_token
    
    try:
        print("\n量化測試開始\n")
        
        # 載入配置並運行測試
        config_loader = QuantizationConfigLoader(args.config)
        tester = QuantizationTester(config_loader)
        output = tester.run_test(hf_token=hf_token, override_method=args.method)
        
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
