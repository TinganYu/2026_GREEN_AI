"""
評估測試腳本
===========

從 model_config.yaml 讀取配置，自動整合 dataset_config.yaml 並執行評估
根據 dataset.selected 自動選擇對應的評估器
"""

import sys
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["PYTORCH_ENABLE_SDPA"] = "0"

# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# 添加路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals import (
    GSM8KEvaluator,
    TruthfulQAEvaluator,
    CommonsenseQAEvaluator,
    HumanEvalEvaluator,
    BBHEvaluator,
)

# 資料集到評估器的映射
EVALUATOR_MAP = {
    "gsm8k": GSM8KEvaluator,
    "truthfulqa": TruthfulQAEvaluator,
    "commonsenseqa": CommonsenseQAEvaluator,
    "humaneval": HumanEvalEvaluator,
    "bbh": BBHEvaluator,
}


def load_model_config(config_path: str) -> Dict[str, Any]:
    """載入 model_config.yaml"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_selected_datasets(config: Dict[str, Any]) -> list:
    """從配置中獲取要評估的資料集列表"""
    selected = config.get("dataset", {}).get("selected", "gsm8k")
    
    if isinstance(selected, str):
        return [selected]
    elif isinstance(selected, list):
        return selected
    else:
        raise ValueError(f"Invalid dataset.selected format: {selected}")

def create_evaluator(dataset_name: str, config: Dict[str, Any]):
    """根據資料集名稱創建對應的評估器"""
    if dataset_name not in EVALUATOR_MAP:
        raise ValueError(
            f"不支援的資料集: {dataset_name}\n"
            f"可用的資料集: {list(EVALUATOR_MAP.keys())}"
        )
    
    evaluator_class = EVALUATOR_MAP[dataset_name]
    return evaluator_class(config)


def detect_quantization(model_path: str) -> str:
    """從模型路徑偵測量化類型"""
    model_lower = model_path.lower()
    
    if "gptq" in model_lower:
        return "gptq"
    elif "awq" in model_lower:
        return "awq"
    elif "bnb" in model_lower or "4bit" in model_lower or "8bit" in model_lower:
        return "bnb"
    else:
        return None


def main():
    """主程式"""
    # 載入配置
    config = load_model_config("tmp/config/model_config.yaml")
    
    datasets = get_selected_datasets(config)
    
    print("=" * 80)
    print("評估測試腳本")
    print("=" * 80)
    print(f"📊 要評估的資料集: {datasets}")
    
    # 對每個資料集執行評估
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*80}")
        print(f"開始評估: {dataset_name.upper()}")
        print(f"{'='*80}")
        
        try:
            # 步驟 1: 創建評估器
            print(f"\n🔹 創建 {dataset_name} 評估器...")
            evaluator = create_evaluator(dataset_name, config)
            
            # 步驟 2: 顯示配置
            print(f"\n⚙️  評估配置:")
            print(f"   模型: {evaluator.config.model_path}")
            print(f"   Prompt: {evaluator.config.prompt_type}")
            print(f"   樣本數: {evaluator.config.num_samples or '全部'}")
            print(f"   Max tokens: {evaluator.config.max_new_tokens}")
            print(f"   輸出: {evaluator.config.output_dir}")
            
            # 檢查 dataset manager
            if evaluator.dataset_manager:
                print(f"\n✅ Dataset manager 已連接")
            
            # 步驟 3: 偵測量化
            quant_type = detect_quantization(evaluator.config.model_path)
            if quant_type:
                print(f"\n🔧 使用 {quant_type.upper()} 量化")
            else:
                print(f"\n📦 使用普通模型")
            
            # 步驟 4: 載入模型
            print(f"\n🔹 載入模型...")
            evaluator.load_model(quantization_type=quant_type)
            
            # 步驟 5: 評估
            print(f"\n🚀 開始評估...")
            results = evaluator.evaluate()
            evaluator.save_results(results)
            
            # 記錄結果
            all_results[dataset_name] = results
            
            print(f"\n✅ {dataset_name.upper()} 評估完成")
            print(f"共 {results.get('total', 0)} 個樣本")
            # print(f"   準確率: {results.get('accuracy', 0):.4f}")
            # print(f"   正確: {results.get('correct', 0)}/{results.get('total', 0)}")
            
        except Exception as e:
            print(f"\n❌ {dataset_name} 評估失敗: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {"error": str(e)}
    
    # 顯示總結
    print("\n" + "=" * 80)
    print("評估總結")
    print("=" * 80)
    
    if len(all_results) == 1:
        # 單一資料集，顯示詳細結果
        dataset_name = list(all_results.keys())[0]
        result = all_results[dataset_name]
        
        if "error" in result:
            print(f"\n❌ 評估失敗")
        else:
            print(f"\n資料集: {dataset_name.upper()}")
            print(f"📁 輸出: {result.get('output_file', 'N/A')}")
    else:
        # 多個資料集，顯示表格
        print(f"\n{'資料集':<15} {'總數':<15} {'狀態'}")
        print("-" * 60)
        
        for dataset_name, result in all_results.items():
            if "error" in result:
                print(f"{dataset_name:<15} {'N/A':<10} {'N/A':<15} ❌ 失敗")
            else:
                total = result.get("total", 0)
                print(f"{dataset_name:<15} {total:<12} ✅ 完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
