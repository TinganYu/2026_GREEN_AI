"""
GSM8K 評估器 - 簡化測試腳本
=============================

直接從 model_config.yaml 讀取配置並執行評估
"""
import sys
from pathlib import Path

# 添加父目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tmp.evals.gsm8k_eval import GSM8KEvaluator


def main():
    print("=" * 80)
    print("GSM8K 評估器 - 簡化測試")
    print("=" * 80)
    
    # 從 YAML 配置文件創建評估器
    config_path = "tmp/config/model_config.yaml"
    print(f"\n📋 從 YAML 載入配置: {config_path}\n")
    
    # 創建評估器（會自動處理 model_config 和 dataset_config 的整合）
    evaluator = GSM8KEvaluator(config_path)
    
    # 顯示配置資訊
    print(f"⚙️  評估配置:")
    print(f"   模型路徑: {evaluator.config.model_path}")
    print(f"   Prompt 類型: {evaluator.config.prompt_type}")
    print(f"   樣本數: {evaluator.config.num_samples or '全部'}")
    print(f"   輸出目錄: {evaluator.config.output_dir}")
    print(f"   Max tokens: {evaluator.config.max_new_tokens}")
    
    if evaluator.dataset_manager:
        print(f"\n✅ Dataset manager 已連接")

    # 判斷量化類型
    model_path = evaluator.config.model_path.lower()
    quantization_type = None
    if "gptq" in model_path:
        quantization_type = "gptq"
        print(f"\n🔧 偵測到 GPTQ 量化模型")
    elif "awq" in model_path:
        quantization_type = "awq"
        print(f"\n🔧 偵測到 AWQ 量化模型")
    elif "bnb" in model_path or "4bit" in model_path:
        quantization_type = "bnb"
        print(f"\n🔧 偵測到 BNB 量化模型")
    else:
        print(f"\n📦 普通模型")
    
    # 載入模型
    print(f"\n🔹 開始載入模型...")
    evaluator.load_model(quantization_type=quantization_type)
    
    # 執行評估
    print(f"\n🚀 開始評估...\n")
    results = evaluator.evaluate()
    
    # 顯示結果
    print("\n" + "=" * 80)
    print("評估結果")
    print("=" * 80)
    print(f"📊 準確率: {results['accuracy']:.4f}")
    print(f"✓ 正確: {results['correct']}/{results['total']}")
    print(f"📁 輸出檔案: {results['output_file']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
