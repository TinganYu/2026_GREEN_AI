"""
GSM8K Evaluator 測試腳本
展示如何使用 GSM8KEvaluator 類別進行評估
"""
import sys
import os
from pathlib import Path

# 添加父目錄到路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.gsm_eval import GSM8KEvaluator

def main():
    print("=" * 80)
    print("GSM8K Evaluator - 測試腳本")
    print("=" * 80)
    
    # 方法 1: 從 YAML 配置文件創建評估器
    config_path = Path(__file__).parent.parent / "config" / "model_config.yaml"
    print(f"\n📋 從 YAML 載入配置: {config_path}")
    
    evaluator = GSM8KEvaluator.from_yaml(str(config_path))
    
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


def example_custom_config():
    """展示如何使用自訂配置"""
    from evals.gsm_eval import EvalConfig
    
    print("\n" + "=" * 80)
    print("範例：使用自訂配置")
    print("=" * 80)
    
    # 創建自訂配置
    config = EvalConfig(
        model_path="meta-llama/Llama-3.2-1B-Instruct",
        output_dir="./custom_output",
        max_new_tokens=512,
        num_samples=10,  # 只評估 10 個樣本
        prompt_type="fewshot",
        hf_token=os.getenv("HUGGINGFACE_TOKEN"),
    )
    
    # 創建評估器
    evaluator = GSM8KEvaluator(config)
    
    print(f"✅ 評估器已創建")
    print(f"   - 模型: {config.model_path}")
    print(f"   - Prompt: {config.prompt_type}")
    print(f"   - 樣本數: {config.num_samples}")
    
    # 後續可以呼叫 evaluator.load_model() 和 evaluator.evaluate()


def example_prompt_comparison():
    """展示如何比較不同 prompt 類型的效果"""
    from evals.gsm_eval import EvalConfig
    
    print("\n" + "=" * 80)
    print("範例：比較不同 Prompt 類型")
    print("=" * 80)
    
    model_path = "../quantized/gptq"  # 替換為您的模型路徑
    prompt_types = ["direct", "fewshot", "cot"]
    
    results_summary = []
    
    for prompt_type in prompt_types:
        print(f"\n📊 評估 Prompt 類型: {prompt_type}")
        
        config = EvalConfig(
            model_path=model_path,
            output_dir=f"./results_{prompt_type}",
            num_samples=20,  # 快速測試用少量樣本
            prompt_type=prompt_type,
        )
        
        evaluator = GSM8KEvaluator(config)
        
        # 這裡應該載入模型並評估，但為了示範我們跳過
        # evaluator.load_model(quantization_type="gptq")
        # results = evaluator.evaluate()
        # results_summary.append({
        #     "prompt_type": prompt_type,
        #     "accuracy": results["accuracy"]
        # })
        
        print(f"   ✓ 配置已創建: {prompt_type}")
    
    # 比較結果
    # print("\n" + "=" * 80)
    # print("Prompt 類型比較")
    # print("=" * 80)
    # for r in results_summary:
    #     print(f"{r['prompt_type']:15s}: {r['accuracy']:.4f}")


if __name__ == "__main__":
    # 執行主要測試

    main()

    # print(f"\n❌ 錯誤: {e}")
    # print("\n提示：")
    # print("1. 請確保 model_config.yaml 中的模型路徑正確")
    # print("2. 如果是量化模型，請確保已安裝對應的套件 (auto-gptq, autoawq)")
    # print("3. 如果需要 HuggingFace token，請設定環境變數 HUGGINGFACE_TOKEN")
    
    # # 展示其他範例（不實際執行）
    # print("\n\n" + "=" * 80)
    # print("其他使用範例（程式碼示範）")
    # print("=" * 80)
    
    # example_custom_config()
    # example_prompt_comparison()
