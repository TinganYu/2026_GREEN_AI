from typing import List, Dict, Any
import logging

logger = logging.getLogger("TunerUtils")

def get_pareto_frontier(trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    從試驗中提取 Pareto 前沿 (Adapted from tmp)
    
    支配關係：
    - accuracy: maximize
    - latency: minimize (1/latency maximize)
    - vram: minimize (1/vram maximize)
    """
    valid_trials = [t for t in trials if t.get('metrics') and t['metrics'].get('accuracy') is not None]

    if not valid_trials:
        return []

    pareto_frontier = []

    for trial in valid_trials:
        is_dominated = False
        metrics = trial['metrics']
        
        # 我們主要看平均 accuracy, latency 和 vram
        acc = metrics.get('accuracy', 0.0)
        lat = metrics.get('latency', float('inf'))
        vram = metrics.get('vram', float('inf'))

        for other_trial in valid_trials:
            if trial == other_trial:
                continue

            other_metrics = other_trial['metrics']
            other_acc = other_metrics.get('accuracy', 0.0)
            other_lat = other_metrics.get('latency', float('inf'))
            other_vram = other_metrics.get('vram', float('inf'))

            # 檢查 other_trial 是否支配 trial
            # 支配：在所有目標上更好或相等，在至少一個目標上嚴格更好
            better_acc = other_acc > acc
            better_lat = other_lat < lat
            better_vram = other_vram < vram
            
            equal_acc = abs(other_acc - acc) < 1e-9
            equal_lat = abs(other_lat - lat) < 1e-9
            equal_vram = abs(other_vram - vram) < 1e-9

            dominates_acc = better_acc or equal_acc
            dominates_lat = better_lat or equal_lat
            dominates_vram = better_vram or equal_vram

            strictly_better = better_acc or better_lat or better_vram

            if dominates_acc and dominates_lat and dominates_vram and strictly_better:
                is_dominated = True
                break

        if not is_dominated:
            pareto_frontier.append(trial)

    return pareto_frontier
