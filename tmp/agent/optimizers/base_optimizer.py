"""
基礎優化器

所有多目標優化器的抽象基類。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger("BaseOptimizer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class BaseOptimizer(ABC):
    """多目標優化器的抽象基類"""

    def __init__(self, config: Dict[str, Any], baseline_metrics: Dict[str, Any],
                 evaluator_agent, datasets: Dict[str, int]):
        """
        初始化優化器

        Args:
            config: 配置字典
            baseline_metrics: 基線評估結果
            evaluator_agent: EvaluatorAgent 實例
            datasets: 用於量化評估的字典 {dataset_name: num_samples}
        """
        self.config = config
        self.baseline = baseline_metrics['aggregated']
        self.baseline_full = baseline_metrics  # 保留完整基線以供參考
        self.evaluator = evaluator_agent
        self.datasets = datasets

        # 提取目標配置
        self.objectives_config = config['multiobjective']['objectives']
        self.constraints_config = config['multiobjective']['constraints']
        self.targets_config = config['multiobjective']['targets']

        # 試驗儲存
        self.trials = []

        logger.info(f"{self.__class__.__name__} 已初始化")

    @abstractmethod
    def optimize(self) -> Dict[str, Any]:
        """
        執行優化過程

        Returns:
            包含以下內容的字典：
            {
                'optimizer_type': str,
                'all_trials': List[Dict],
                'pareto_frontier': List[Dict],
                'satisfying_solutions': List[Dict],
                'recommended_config': Dict,
                'n_total_trials': int,
                'n_pareto_solutions': int,
                'n_satisfying_solutions': int
            }
        """
        pass

    def calculate_objectives(self, metrics: Dict[str, float]) -> Tuple[float, float, float]:
        """
        計算三個目標值（使用直接的變化量）

        Args:
            metrics: 評估指標（聚合）

        Returns:
            元組 (accuracy_change, gpu_peak_change, latency_change)
            - accuracy_change: 準確率變化量（正=上升，負=下降）
            - gpu_peak_change: GPU 峰值變化量（正=增加，負=減少）
            - latency_change: 延遲變化量（正=增加，負=減少）
        """
        # 準確率變化量：(量化後 - 基線) / 基線
        # 正值 = 準確率上升（好），負值 = 準確率下降（壞）
        # 例如：基線 0.25 → 量化 0.30，change = +0.20 = +20% 上升
        # 例如：基線 0.25 → 量化 0.20，change = -0.20 = -20% 下降
        accuracy_change = (metrics['accuracy'] - self.baseline['accuracy']) / self.baseline['accuracy']

        # GPU 峰值變化量：(量化後 - 基線) / 基線
        # 正值 = GPU 增加（壞），負值 = GPU 減少（好）
        # 例如：基線 2475MB → 量化 1185MB，change = -0.52 = -52% 減少
        # 例如：基線 2475MB → 量化 3000MB，change = +0.21 = +21% 增加
        gpu_peak_change = (metrics['gpu_peak_mb'] - self.baseline['gpu_peak_mb']) / self.baseline['gpu_peak_mb']

        # 延遲變化量：(量化後 - 基線) / 基線
        # 正值 = 延遲增加（壞），負值 = 延遲減少（好）
        # 例如：基線 404ms → 量化 2192ms，change = +4.43 = +443% 增加
        # 例如：基線 404ms → 量化 301ms，change = -0.25 = -25% 減少
        latency_change = (metrics['avg_latency_ms'] - self.baseline['avg_latency_ms']) / self.baseline['avg_latency_ms']

        return accuracy_change, gpu_peak_change, latency_change

    def check_constraints(self, objectives: Tuple[float, float, float]) -> bool:
        """
        檢查目標是否滿足硬約束

        Args:
            objectives: 元組 (accuracy_change, gpu_peak_change, latency_change)

        Returns:
            如果滿足所有約束則返回 True，否則返回 False（會被剪枝）
        """
        if not self.constraints_config['enabled']:
            return True

        accuracy_change, gpu_peak_change, latency_change = objectives

        # 準確率約束：accuracy_change >= accuracy_min（允許最多下降 X%）
        # 例如：accuracy_min = -0.15 表示允許最多下降 15%
        satisfies_accuracy = accuracy_change >= self.constraints_config['accuracy_min']

        # GPU 峰值約束：gpu_peak_change <= gpu_peak_max（要求至少減少 X%）
        # 例如：gpu_peak_max = -0.10 表示必須至少減少 10%（不允許超過 -10%）
        satisfies_gpu = gpu_peak_change <= self.constraints_config['gpu_peak_max']

        # 延遲約束：latency_change <= latency_max（允許最多增加 X%）
        # 例如：latency_max = 0.50 表示允許最多增加 50%
        satisfies_latency = latency_change <= self.constraints_config['latency_max']

        return satisfies_accuracy and satisfies_gpu and satisfies_latency

    def check_targets(self, objectives: Tuple[float, float, float]) -> bool:
        """
        檢查目標是否滿足用戶目標（比約束更寬鬆）

        Args:
            objectives: 元組 (accuracy_change, gpu_peak_change, latency_change)

        Returns:
            如果滿足所有目標則返回 True，否則返回 False
        """
        accuracy_change, gpu_peak_change, latency_change = objectives

        # 準確率目標：期望準確率下降不超過 X%
        # 理想情況：accuracy_change >= accuracy_min
        satisfies_accuracy = accuracy_change >= self.targets_config['accuracy_min']

        # GPU 峰值目標：期望至少減少 X%
        # 理想情況：gpu_peak_change <= gpu_peak_max
        satisfies_gpu = gpu_peak_change <= self.targets_config['gpu_peak_max']

        # 延遲目標：期望延遲增加不超過 X%
        # 理想情況：latency_change <= latency_max
        satisfies_latency = latency_change <= self.targets_config['latency_max']

        return satisfies_accuracy and satisfies_gpu and satisfies_latency

    def calculate_violation_score(self, objectives: Tuple[float, float, float]) -> float:
        """
        計算目標違反程度（0 = 完美滿足，值越高 = 違反越嚴重）

        使用 multiobjective.objectives 中配置的權重，確保與推薦邏輯一致。

        Args:
            objectives: 元組 (accuracy_change, gpu_peak_change, latency_change)

        Returns:
            加權違反分數
        """
        accuracy_change, gpu_peak_change, latency_change = objectives

        # 使用目標配置中的權重（與推薦邏輯保持一致）
        weights = [obj['weight'] for obj in self.objectives_config]
        # weights[0] = accuracy, weights[1] = gpu_peak, weights[2] = latency

        # 計算每個目標的違反程度（正值 = 違反，0 = 滿足）
        # 準確率：期望 >= accuracy_min，違反時為負數差距
        acc_violation = max(0, self.targets_config['accuracy_min'] - accuracy_change)

        # GPU 峰值：期望 <= gpu_peak_max，違反時為正數超出
        gpu_violation = max(0, gpu_peak_change - self.targets_config['gpu_peak_max'])

        # 延遲：期望 <= latency_max，違反時為正數超出
        latency_violation = max(0, latency_change - self.targets_config['latency_max'])

        # 加權總和
        violation_score = (
            weights[0] * acc_violation +
            weights[1] * gpu_violation +
            weights[2] * latency_violation
        )

        return violation_score

    def run_trial(self, quant_config: Dict[str, Any],
                  trial_number: int = None, exp_dir: str = None) -> Dict[str, Any]:
        """
        執行單次試驗：量化 + 評估

        Args:
            quant_config: 量化配置
            trial_number: 試驗編號（用於優化實驗）
            exp_dir: 實驗根目錄（用於優化實驗）

        Returns:
            試驗結果字典
        """
        # 執行實驗
        results = self.evaluator.run_experiment(quant_config, self.datasets, trial_number, exp_dir)

        if not results.get('success', False):
            # 實驗失敗
            return {
                'success': False,
                'config': quant_config,
                'error': results.get('error', '未知錯誤')
            }

        # 計算目標
        objectives = self.calculate_objectives(results['aggregated'])

        # 檢查約束和目標
        satisfies_constraints = self.check_constraints(objectives)
        satisfies_targets = self.check_targets(objectives)
        violation_score = self.calculate_violation_score(objectives)

        # 封裝試驗結果
        trial = {
            'success': True,
            'config': quant_config,
            'results': results,
            'objectives': {
                'accuracy_change': objectives[0],
                'gpu_peak_change': objectives[1],
                'latency_change': objectives[2]
            },
            'satisfies_constraints': satisfies_constraints,
            'satisfies_targets': satisfies_targets,
            'violation_score': violation_score
        }

        return trial

    def get_pareto_frontier(self, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        從試驗中提取 Pareto 前沿

        如果沒有其他試驗在所有目標上支配它，則該試驗是 Pareto 最優的。

        支配關係（新命名）：
        - accuracy_change: maximize（越大越好，即準確率上升）
        - gpu_peak_change: minimize（越小越好，即 GPU 峰值減少）
        - latency_change: minimize（越小越好，即延遲減少）

        Args:
            trials: 試驗字典列表

        Returns:
            Pareto 最優試驗列表
        """
        # 過濾成功的試驗
        valid_trials = [t for t in trials if t.get('success', False)]

        if not valid_trials:
            return []

        pareto_frontier = []

        for trial in valid_trials:
            is_dominated = False
            obj = trial['objectives']

            for other_trial in valid_trials:
                if trial == other_trial:
                    continue

                other_obj = other_trial['objectives']

                # 檢查 other_trial 是否支配 trial
                # 支配：在所有目標上更好或相等，在至少一個目標上嚴格更好

                # 比較邏輯（注意方向！）
                better_accuracy = other_obj['accuracy_change'] > obj['accuracy_change']  # maximize
                better_gpu = other_obj['gpu_peak_change'] < obj['gpu_peak_change']      # minimize
                better_latency = other_obj['latency_change'] < obj['latency_change']    # minimize

                equal_accuracy = abs(other_obj['accuracy_change'] - obj['accuracy_change']) < 1e-9
                equal_gpu = abs(other_obj['gpu_peak_change'] - obj['gpu_peak_change']) < 1e-9
                equal_latency = abs(other_obj['latency_change'] - obj['latency_change']) < 1e-9

                # 檢查支配性
                dominates_accuracy = better_accuracy or equal_accuracy
                dominates_gpu = better_gpu or equal_gpu
                dominates_latency = better_latency or equal_latency

                strictly_better = better_accuracy or better_gpu or better_latency

                if dominates_accuracy and dominates_gpu and dominates_latency and strictly_better:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_frontier.append(trial)

        logger.info(f"Pareto 前沿：{len(valid_trials)} 個試驗中有 {len(pareto_frontier)} 個")
        return pareto_frontier

    def recommend_config(self, pareto_trials: List[Dict[str, Any]],
                        satisfying_trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        根據策略推薦最佳配置

        Args:
            pareto_trials: Pareto 最優試驗
            satisfying_trials: 滿足目標的試驗

        Returns:
            推薦的配置字典
        """
        strategy = self.config['output']['recommendation']['strategy']

        if not pareto_trials:
            return None

        if strategy == "closest_to_target" and satisfying_trials:
            # 在滿足目標的試驗中推薦最接近目標的配置
            # 使用加權距離（全部轉為 minimize 形式）
            weights = [obj['weight'] for obj in self.objectives_config]
            best_trial = min(satisfying_trials, key=lambda t: (
                weights[0] * (-t['objectives']['accuracy_change']) +  # maximize accuracy（取負轉為minimize）
                weights[1] * t['objectives']['gpu_peak_change'] +     # minimize gpu
                weights[2] * t['objectives']['latency_change']        # minimize latency
            ))

        elif strategy == "best_accuracy":
            # 推薦準確率變化最大的配置（即準確率提升最多或下降最少）
            best_trial = max(pareto_trials, key=lambda t: t['objectives']['accuracy_change'])

        elif strategy == "best_compression":
            # 推薦 GPU 峰值變化最小的配置（即 GPU 減少最多）
            best_trial = min(pareto_trials, key=lambda t: t['objectives']['gpu_peak_change'])

        else:  # balanced
            # 使用目標權重的平衡推薦（全部轉為 minimize 形式）
            weights = [obj['weight'] for obj in self.objectives_config]
            best_trial = min(pareto_trials, key=lambda t: (
                weights[0] * (-t['objectives']['accuracy_change']) +
                weights[1] * t['objectives']['gpu_peak_change'] +
                weights[2] * t['objectives']['latency_change']
            ))

        return {
            'config': best_trial['config'],
            'results': best_trial['results'],
            'objectives': best_trial['objectives'],
            'satisfies_targets': best_trial['satisfies_targets'],
            'violation_score': best_trial['violation_score'],
            'strategy': strategy
        }
