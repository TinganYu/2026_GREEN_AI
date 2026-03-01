import asvd
import argparse

def simple_agent():
    # 定義你想測試的 alpha 範圍
    alphas = [0.4, 0.5, 0.6]
    results = []

    for a in alphas:
        print(f"Agent 嘗試測試 Alpha: {a}")   #act aware很重要，應該要開著(ASVD重點)
        config = {
            "model_id": "facebook/opt-125m", # 先用小模型跑
            "alpha": a,
            "param_ratio_target": 0.8,
            "calib_dataset": "wikitext2",
            "eval_ppl": "wikitext2",
            "use_cache": True
        }
        # 執行 ASVD 實驗
        res = asvd.run_asvd_exp(config)
        results.append((a, res))
    
    # 找出 PPL 最好的 alpha (假設 res 回傳 PPL)
    best_alpha = min(results, key=lambda x: x[1]['wikitext2'])[0]
    print(f"Agent 決策結果：最佳 Alpha 為 {best_alpha}")

if __name__ == "__main__":
    simple_agent()