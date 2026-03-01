"""
評分和分析模組

為多目標優化結果提供詳細的評分和分析功能。
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger("Scoring")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class LayeredScorer:
    """加權評分系統

    為滿足 target 的配置提供詳細評分和排名。
    評分公式：weight * change（統一為越高越好）
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化評分器

        Args:
            config: 配置字典
        """
        self.config = config
        self.objectives_config = config['multiobjective']['objectives']
        self.targets_config = config['multiobjective']['targets']

        # 提取權重
        self.weights = {
            'accuracy': self.objectives_config[0]['weight'],
            'gpu_peak': self.objectives_config[1]['weight'],
            'latency': self.objectives_config[2]['weight']
        }

        logger.info("LayeredScorer 已初始化")

    def score_satisfying_trials(self, satisfying_trials: List[Dict],
                               pareto_trials: List[Dict]) -> Dict[str, Any]:
        """
        為所有滿足 target 的配置評分

        Args:
            satisfying_trials: 滿足目標的試驗列表
            pareto_trials: Pareto 前沿試驗列表（用於標記和策略排名）

        Returns:
            包含排名和評分的字典
        """
        if not satisfying_trials:
            logger.warning("沒有滿足目標的試驗，返回空結果")
            return {
                'scoring_method': 'weighted_surplus',
                'scoring_details': self._get_scoring_details(),
                'targets': self.targets_config,
                'total_satisfying_trials': 0,
                'pareto_overlap': {'count': 0, 'trial_ids': []},
                'ranked_trials': [],
                'statistics': {}
            }

        logger.info(f"正在為 {len(satisfying_trials)} 個滿足目標的試驗評分...")

        # 建立 Pareto 前沿的 trial_id 集合（用於判斷是否為 Pareto 解）
        pareto_trial_ids = {t['trial_id'] for t in pareto_trials if 'trial_id' in t} if pareto_trials else set()

        # 為每個試驗評分
        ranked = []
        pareto_overlap_ids = []
        for i, trial in enumerate(satisfying_trials):
            trial_id = trial.get('trial_id', i)
            is_pareto = trial_id in pareto_trial_ids
            score_info = self._score_single_trial(trial, is_pareto)
            # 確保有 trial_id
            if score_info.get('trial_id') is None:
                score_info['trial_id'] = trial_id
            ranked.append(score_info)
            if is_pareto:
                pareto_overlap_ids.append(score_info['trial_id'])

        # 按總分排序（降序）
        ranked.sort(key=lambda x: x['overall_score'], reverse=True)

        # 添加排名
        for i, trial_score in enumerate(ranked):
            trial_score['rank'] = i + 1

        logger.info(f"評分完成，最高分：{ranked[0]['overall_score']:.4f}")

        return {
            'scoring_method': 'weighted_surplus',
            'scoring_details': self._get_scoring_details(),
            'targets': self.targets_config,
            'total_satisfying_trials': len(satisfying_trials),
            'pareto_overlap': {
                'count': len(pareto_overlap_ids),
                'trial_ids': pareto_overlap_ids
            },
            'ranked_trials': ranked,
            'statistics': self._calculate_statistics(ranked)
        }

    def _score_single_trial(self, trial: Dict, is_pareto: bool = False) -> Dict:
        """
        為單個試驗評分

        Args:
            trial: 當前試驗
            is_pareto: 是否為 Pareto 前沿解

        Returns:
            評分資訊字典
        """
        objectives = trial['objectives']

        # 總體評分
        overall_score = self._calculate_overall_score(objectives)

        # 單維度評分
        dimension_scores = self._calculate_dimension_scores(objectives)

        return {
            'trial_id': trial.get('trial_id'),
            'overall_score': round(overall_score, 4),
            'is_pareto': is_pareto,
            'dimension_scores': dimension_scores,
            'config': trial['config'],
            'objectives': objectives
        }

    def _calculate_overall_score(self, objectives: Dict[str, float]) -> float:
        """
        計算加權總分

        公式：weight * surplus（超越目標的幅度）
        - 滿足目標的試驗 surplus >= 0，所以總分 >= 0
        - 越超越目標，分數越高

        Args:
            objectives: 目標值字典

        Returns:
            加權總分（滿足目標時 >= 0）
        """
        accuracy_change = objectives['accuracy_change']
        gpu_peak_change = objectives['gpu_peak_change']
        latency_change = objectives['latency_change']

        acc_target = self.targets_config['accuracy_min']
        gpu_target = self.targets_config['gpu_peak_max']
        lat_target = self.targets_config['latency_max']

        # 計算超越目標的幅度（surplus）
        acc_surplus = accuracy_change - acc_target      # accuracy 越高越好，超過 target 為正
        gpu_surplus = gpu_target - gpu_peak_change      # gpu 越低越好，低於 target 為正
        lat_surplus = lat_target - latency_change       # latency 越低越好，低於 target 為正

        # 加權評分（使用 surplus，滿足目標時都是正數）
        score = (
            self.weights['accuracy'] * acc_surplus +
            self.weights['gpu_peak'] * gpu_surplus +
            self.weights['latency'] * lat_surplus
        )

        return score

    def _calculate_dimension_scores(self, objectives: Dict[str, float]) -> Dict[str, Dict]:
        """
        計算單維度評分

        Args:
            objectives: 目標值字典

        Returns:
            單維度評分字典，每個維度包含 value, score, target, surplus
        """
        accuracy_change = objectives['accuracy_change']
        gpu_peak_change = objectives['gpu_peak_change']
        latency_change = objectives['latency_change']

        acc_target = self.targets_config['accuracy_min']
        gpu_target = self.targets_config['gpu_peak_max']
        lat_target = self.targets_config['latency_max']

        # 計算超越目標的幅度（surplus）
        acc_surplus = accuracy_change - acc_target
        gpu_surplus = gpu_target - gpu_peak_change
        lat_surplus = lat_target - latency_change

        return {
            'accuracy': {
                'value': round(accuracy_change, 4),
                'target': round(acc_target, 4),
                'surplus': round(acc_surplus, 4),
                'score': round(self.weights['accuracy'] * acc_surplus, 4)  # weight * surplus
            },
            'gpu_peak': {
                'value': round(gpu_peak_change, 4),
                'target': round(gpu_target, 4),
                'surplus': round(gpu_surplus, 4),
                'score': round(self.weights['gpu_peak'] * gpu_surplus, 4)  # weight * surplus
            },
            'latency': {
                'value': round(latency_change, 4),
                'target': round(lat_target, 4),
                'surplus': round(lat_surplus, 4),
                'score': round(self.weights['latency'] * lat_surplus, 4)  # weight * surplus
            }
        }

    def _calculate_statistics(self, ranked_trials: List[Dict]) -> Dict[str, Any]:
        """
        計算統計資訊

        Args:
            ranked_trials: 已排名的試驗列表

        Returns:
            統計資訊字典
        """
        if not ranked_trials:
            return {}

        # 平均分和中位數
        scores = [t['overall_score'] for t in ranked_trials]
        avg_score = sum(scores) / len(scores)
        sorted_scores = sorted(scores)
        median_score = sorted_scores[len(sorted_scores) // 2]

        # 分數範圍
        score_range = {
            'min': round(min(scores), 4),
            'max': round(max(scores), 4)
        }

        # 最佳單維度表現者
        best_accuracy = max(ranked_trials, key=lambda t: t['dimension_scores']['accuracy']['score'])
        best_gpu = max(ranked_trials, key=lambda t: t['dimension_scores']['gpu_peak']['score'])
        best_latency = max(ranked_trials, key=lambda t: t['dimension_scores']['latency']['score'])

        # Pareto 前沿統計
        pareto_count = sum(1 for t in ranked_trials if t.get('is_pareto', False))

        return {
            'avg_overall_score': round(avg_score, 4),
            'median_overall_score': round(median_score, 4),
            'score_range': score_range,
            'pareto_in_satisfying': pareto_count,
            'best_dimension_performers': {
                'accuracy': best_accuracy.get('trial_id'),
                'gpu_peak': best_gpu.get('trial_id'),
                'latency': best_latency.get('trial_id')
            }
        }

    def _get_scoring_details(self) -> Dict[str, Any]:
        """獲取評分方法詳細資訊"""
        return {
            'formula': 'w_acc*(acc - acc_min) + w_gpu*(gpu_max - gpu) + w_lat*(lat_max - lat)',
            'description': '基於超越目標幅度(surplus)的加權評分，滿足目標時分數 >= 0',
            'weights': self.weights
        }


class ParetoAnalyzer:
    """Pareto 前沿深度分析（方案 4）

    對整個 Pareto 前沿進行全面分析。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化分析器

        Args:
            config: 配置字典
        """
        self.config = config
        self.objectives_config = config['multiobjective']['objectives']
        self.targets_config = config['multiobjective']['targets']

        # 提取權重
        self.weights = {
            'accuracy': self.objectives_config[0]['weight'],
            'gpu_peak': self.objectives_config[1]['weight'],
            'latency': self.objectives_config[2]['weight']
        }

        logger.info("ParetoAnalyzer 已初始化")

    def analyze_pareto_frontier(self, pareto_trials: List[Dict],
                               satisfying_trials: List[Dict],
                               layered_scores: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        分析 Pareto 前沿

        Args:
            pareto_trials: Pareto 前沿試驗列表
            satisfying_trials: 滿足目標的試驗列表
            layered_scores: LayeredScorer 的評分結果（可選）

        Returns:
            分析結果字典
        """
        if not pareto_trials:
            logger.warning("Pareto 前沿為空，返回空結果")
            return {
                'analysis_method': 'pareto_frontier_deep_analysis',
                'pareto_statistics': {'total_pareto_solutions': 0},
                'tradeoff_analysis': {},
                'recommendations': {}
            }

        logger.info(f"正在分析 Pareto 前沿（{len(pareto_trials)} 個解決方案）...")

        # 1. 統計資訊
        statistics = self._calculate_pareto_statistics(pareto_trials, satisfying_trials)

        # 2. 權衡分析
        tradeoff_analysis = self._analyze_tradeoffs(pareto_trials)

        # 3. 推薦建議
        recommendations = self._generate_recommendations(
            pareto_trials, satisfying_trials, layered_scores
        )

        logger.info("Pareto 前沿分析完成")

        return {
            'analysis_method': 'pareto_frontier_deep_analysis',
            'pareto_statistics': statistics,
            'tradeoff_analysis': tradeoff_analysis,
            'recommendations': recommendations
        }

    def _calculate_pareto_statistics(self, pareto_trials: List[Dict],
                                    satisfying_trials: List[Dict]) -> Dict[str, Any]:
        """
        計算 Pareto 前沿統計資訊

        Args:
            pareto_trials: Pareto 前沿試驗列表
            satisfying_trials: 滿足目標的試驗列表

        Returns:
            統計資訊字典
        """
        # 分類試驗
        satisfying_ids = {id(t) for t in satisfying_trials}
        close_to_targets = []
        violating_targets = []

        for trial in pareto_trials:
            if id(trial) in satisfying_ids:
                continue  # 已經在滿足列表中

            # 檢查是否接近目標（單個維度略微違反）
            objectives = trial['objectives']
            violations = []

            if objectives['accuracy_change'] < self.targets_config['accuracy_min']:
                violations.append('accuracy')
            if objectives['gpu_peak_change'] > self.targets_config['gpu_peak_max']:
                violations.append('gpu_peak')
            if objectives['latency_change'] > self.targets_config['latency_max']:
                violations.append('latency')

            if len(violations) == 1:
                # 只有一個維度違反，算作「接近」
                close_to_targets.append(trial)
            elif len(violations) > 1:
                # 多個維度違反
                violating_targets.append(trial)

        return {
            'total_pareto_solutions': len(pareto_trials),
            'satisfying_targets': len(satisfying_trials),
            'close_to_targets': len(close_to_targets),
            'violating_targets': len(violating_targets)
        }

    def _analyze_tradeoffs(self, pareto_trials: List[Dict]) -> Dict[str, Any]:
        """
        分析目標之間的權衡關係（相關性）

        Args:
            pareto_trials: Pareto 前沿試驗列表

        Returns:
            權衡分析字典
        """
        if len(pareto_trials) < 3:
            logger.warning("Pareto 前沿點太少，無法進行相關性分析")
            return {
                'correlations': {},
                'key_insights': ["資料點太少，無法進行相關性分析"]
            }

        # 提取目標值
        accuracy_values = [t['objectives']['accuracy_change'] for t in pareto_trials]
        gpu_values = [t['objectives']['gpu_peak_change'] for t in pareto_trials]
        latency_values = [t['objectives']['latency_change'] for t in pareto_trials]

        # 計算 Pearson 相關系数
        correlations = {}

        # accuracy vs gpu_peak
        corr_acc_gpu = np.corrcoef(accuracy_values, gpu_values)[0, 1]
        correlations['accuracy_vs_gpu_peak'] = {
            'pearson_r': round(corr_acc_gpu, 3),
            'interpretation': self._interpret_correlation(corr_acc_gpu, "準確率", "GPU 節省")
        }

        # accuracy vs latency
        corr_acc_lat = np.corrcoef(accuracy_values, latency_values)[0, 1]
        correlations['accuracy_vs_latency'] = {
            'pearson_r': round(corr_acc_lat, 3),
            'interpretation': self._interpret_correlation(corr_acc_lat, "準確率", "延遲")
        }

        # gpu_peak vs latency
        corr_gpu_lat = np.corrcoef(gpu_values, latency_values)[0, 1]
        correlations['gpu_peak_vs_latency'] = {
            'pearson_r': round(corr_gpu_lat, 3),
            'interpretation': self._interpret_correlation(corr_gpu_lat, "GPU 峰值", "延遲")
        }

        # 生成關鍵洞察
        key_insights = self._generate_insights(correlations, pareto_trials)

        return {
            'correlations': correlations,
            'key_insights': key_insights
        }

    def _interpret_correlation(self, r: float, obj1: str, obj2: str) -> str:
        """
        解釋相關係數

        Args:
            r: 相關係數
            obj1: 目標 1 名稱
            obj2: 目標 2 名稱

        Returns:
            解釋字串
        """
        if np.isnan(r):
            return "無法計算相關性"

        abs_r = abs(r)
        if abs_r > 0.7:
            strength = "強"
        elif abs_r > 0.4:
            strength = "中等"
        elif abs_r > 0.2:
            strength = "弱"
        else:
            strength = "幾乎沒有"

        direction = "負" if r < 0 else "正"

        # 根據具體目標給出更具體的解釋
        if "準確率" in obj1 and "GPU" in obj2 and r < -0.5:
            return f"{strength}{direction}相關：保持高準確率需要犧牲 GPU 節省"
        elif "準確率" in obj1 and "延遲" in obj2 and r < -0.3:
            return f"{strength}{direction}相關：準確率越高，延遲可能越長"
        elif "GPU" in obj1 and "延遲" in obj2:
            return f"{strength}{direction}相關：GPU 節省和延遲變化之間的關係"
        else:
            return f"{strength}{direction}相關"

    def _generate_insights(self, correlations: Dict, pareto_trials: List[Dict]) -> List[str]:
        """
        生成關鍵洞察

        Args:
            correlations: 相關性字典
            pareto_trials: Pareto 前沿試驗列表

        Returns:
            洞察列表
        """
        insights = []

        # 準確率 vs GPU 洞察
        corr_acc_gpu = correlations['accuracy_vs_gpu_peak']['pearson_r']
        if corr_acc_gpu < -0.6:
            insights.append(f"要保持高準確率必須犧牲 GPU 節省（{corr_acc_gpu:.2f} 相關性）")
        elif corr_acc_gpu > 0.6:
            insights.append(f"準確率和 GPU 節省可以同時優化（{corr_acc_gpu:.2f} 相關性）")

        # 最佳壓縮策略洞察
        best_compression_trial = min(pareto_trials, key=lambda t: t['objectives']['gpu_peak_change'])
        gpu_save_pct = -best_compression_trial['objectives']['gpu_peak_change'] * 100
        acc_drop_pct = -best_compression_trial['objectives']['accuracy_change'] * 100
        insights.append(
            f"最佳壓縮配置節省 {gpu_save_pct:.1f}% GPU，但準確率下降 {acc_drop_pct:.1f}%"
        )

        # 滿足目標的配置洞察
        satisfying_count = sum(1 for t in pareto_trials if t.get('satisfies_targets', False))
        if satisfying_count > 0:
            insights.append(
                f"{satisfying_count}/{len(pareto_trials)} 個 Pareto 解滿足所有目標"
            )

        return insights

    def _generate_recommendations(self, pareto_trials: List[Dict],
                                 satisfying_trials: List[Dict],
                                 layered_scores: Optional[Dict[str, Any]] = None) -> Dict[str, Dict]:
        """
        生成針對不同場景的推薦

        Args:
            pareto_trials: Pareto 前沿試驗列表
            satisfying_trials: 滿足目標的試驗列表
            layered_scores: LayeredScorer 的評分結果（可選）

        Returns:
            推薦字典
        """
        recommendations = {}

        # 建立 config -> trial_id 的映射（從 layered_scores 中獲取）
        config_to_trial_id = {}
        if layered_scores and 'ranked_trials' in layered_scores:
            for ranked in layered_scores['ranked_trials']:
                config_key = str(ranked.get('config', {}))
                config_to_trial_id[config_key] = ranked.get('trial_id', 0)

        # 輔助函數：獲取 trial_id
        def get_trial_id(trial: Dict, index: int) -> int:
            # 優先使用試驗本身的 trial_id
            if trial.get('trial_id') is not None:
                return trial['trial_id']
            # 否則從映射中查找
            config_key = str(trial.get('config', {}))
            return config_to_trial_id.get(config_key, index)

        # 獲取最高評分（如果有）
        best_score_trial = None
        best_score_trial_id = 0
        if layered_scores and 'ranked_trials' in layered_scores and layered_scores['ranked_trials']:
            best_score_info = layered_scores['ranked_trials'][0]
            best_score_trial_id = best_score_info.get('trial_id', 0)
            # 在 satisfying_trials 中找到對應的 trial
            for trial in satisfying_trials:
                if trial['config'] == best_score_info['config']:
                    best_score_trial = trial
                    break

        # 推薦 1: 生產環境
        if best_score_trial:
            recommendations['for_production'] = {
                'recommended_trial_id': best_score_trial_id,
                'config': best_score_trial['config'],
                'objectives': best_score_trial['objectives'],
                'overall_score': round(layered_scores['ranked_trials'][0]['overall_score'], 4),
                'reason': f"滿足所有目標且綜合評分最高（{layered_scores['ranked_trials'][0]['overall_score']:.4f} 分）"
            }
        elif satisfying_trials:
            trial = satisfying_trials[0]
            obj = trial['objectives']
            score = (
                self.weights['accuracy'] * obj['accuracy_change'] +
                self.weights['gpu_peak'] * (-obj['gpu_peak_change']) +
                self.weights['latency'] * (-obj['latency_change'])
            )
            recommendations['for_production'] = {
                'recommended_trial_id': get_trial_id(trial, 0),
                'config': trial['config'],
                'objectives': trial['objectives'],
                'overall_score': round(score, 4),
                'reason': "滿足所有目標"
            }

        # 推薦 2: 記憶體關鍵場景
        min_gpu_idx = 0
        min_gpu_trial = pareto_trials[0]
        for i, t in enumerate(pareto_trials):
            if t['objectives']['gpu_peak_change'] < min_gpu_trial['objectives']['gpu_peak_change']:
                min_gpu_trial = t
                min_gpu_idx = i

        gpu_save_pct = -min_gpu_trial['objectives']['gpu_peak_change'] * 100
        obj = min_gpu_trial['objectives']
        score = (
            self.weights['accuracy'] * obj['accuracy_change'] +
            self.weights['gpu_peak'] * (-obj['gpu_peak_change']) +
            self.weights['latency'] * (-obj['latency_change'])
        )
        warning = None if min_gpu_trial.get('satisfies_targets', False) else "不滿足所有目標"
        recommendations['for_memory_critical'] = {
            'recommended_trial_id': get_trial_id(min_gpu_trial, min_gpu_idx),
            'config': min_gpu_trial['config'],
            'objectives': min_gpu_trial['objectives'],
            'overall_score': round(score, 4),
            'reason': f"GPU 節省最多（{gpu_save_pct:.1f}%）",
            'warning': warning
        }

        # 推薦 3: 準確率關鍵場景
        max_acc_idx = 0
        max_acc_trial = pareto_trials[0]
        for i, t in enumerate(pareto_trials):
            if t['objectives']['accuracy_change'] > max_acc_trial['objectives']['accuracy_change']:
                max_acc_trial = t
                max_acc_idx = i

        acc_drop_pct = -max_acc_trial['objectives']['accuracy_change'] * 100
        obj = max_acc_trial['objectives']
        score = (
            self.weights['accuracy'] * obj['accuracy_change'] +
            self.weights['gpu_peak'] * (-obj['gpu_peak_change']) +
            self.weights['latency'] * (-obj['latency_change'])
        )
        recommendations['for_accuracy_critical'] = {
            'recommended_trial_id': get_trial_id(max_acc_trial, max_acc_idx),
            'config': max_acc_trial['config'],
            'objectives': max_acc_trial['objectives'],
            'overall_score': round(score, 4),
            'reason': f"準確率下降最少（{acc_drop_pct:.1f}%）"
        }

        return recommendations
