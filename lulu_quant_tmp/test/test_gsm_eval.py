"""
GSM8K Evaluator 測試腳本
展示如何使用 GSM8KEvaluator 類別進行評估
"""
import sys
import os
import yaml
from pathlib import Path


# 添加父目錄到路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.gsm8k_eval import GSM8KEvaluator

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    print("=" * 80)
    print("GSM8K Evaluator - 測試腳本")
    print("=" * 80)
    
    # 方法 1: 從 YAML 配置文件創建評估器
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    print(f"\n📋 從 YAML 載入配置: {config_path}")
    config = load_yaml(config_path)
    
    evaluator = GSM8KEvaluator(config)
    
    # 載入模型（根據模型路徑自動判斷是否為量化模型）
    model_path = evaluator.config.model_path
    
    # 判斷是否為 GPTQ 量化模型
    quantization_type = None
    if "gptq" in model_path.lower():
        quantization_type = "gptq"
        print(f"\n🔧 偵測到 GPTQ 量化模型")
    elif "awq" in model_path.lower():
        quantization_type = "awq"
        print(f"\n🔧 偵測到 AWQ 量化模型")
    else:
        print(f"\n📦 普通模型")
    
    # 載入模型
    evaluator.load_model(quantization_type=quantization_type)
    
    # 執行評估
    print(f"\n🚀 開始評估...")
    print(f"   - Prompt 類型: {evaluator.config.prompt_type}")
    print(f"   - 樣本數量: {evaluator.config.num_samples or '全部'}")
    
    results = evaluator.evaluate()
    
    # 儲存結果
    evaluator.save_results(results)
    
    # 顯示摘要
    print("\n" + "=" * 80)
    print("評估結果摘要")
    print("=" * 80)
    print(f"模型: {results['model_path']}")
    print(f"Prompt 類型: {results['prompt_type']}")
    print(f"準確率: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"輸出文件: {results['output_file']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
