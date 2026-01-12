"""
多目標優化配置載入器

載入並驗證 optimization_config.yaml 配置檔案。
"""

import yaml
import os
import logging
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger("ConfigLoader")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class ConfigLoader:
    """載入並驗證優化配置檔案"""

    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """
        從 YAML 檔案載入配置

        Args:
            config_path: optimization_config.yaml 的路徑

        Returns:
            配置字典
        """
        logger.info(f"正在從以下路徑載入配置：{config_path}")

        # 檢查檔案是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到配置檔案：{config_path}")

        # 載入 YAML
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 驗證配置
        ConfigLoader._validate_config(config)

        # 如需要，載入資料集配置
        config = ConfigLoader._load_dataset_config(config)

        logger.info("配置載入並驗證成功")
        return config

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """驗證配置結構"""

        # 檢查必需的頂層鍵值
        required_keys = ['experiment', 'baseline', 'quantization_eval',
                        'multiobjective', 'optimizer', 'output']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"缺少必需的配置區段：{key}")

        # 驗證多目標區段
        mo_config = config['multiobjective']
        if 'objectives' not in mo_config:
            raise ValueError("多目標配置中缺少 'objectives'")

        if len(mo_config['objectives']) != 3:
            raise ValueError("必須定義恰好 3 個目標")

        # 驗證優化器類型
        optimizer_type = config['optimizer']['type']
        valid_types = ['optuna_multiobjective', 'qehvi', 'random', 'llm_multiagent']
        if optimizer_type not in valid_types:
            raise ValueError(f"無效的優化器類型：{optimizer_type}。"
                           f"必須是以下之一：{valid_types}")

        # 驗證聚合方法
        agg_method = mo_config['aggregation']['method']
        valid_methods = ['weighted', 'average', 'worst_case']
        if agg_method not in valid_methods:
            raise ValueError(f"無效的聚合方法：{agg_method}。"
                           f"必須是以下之一：{valid_methods}")

        # 如果是加權方法，檢查是否提供權重
        if agg_method == 'weighted':
            if 'dataset_weights' not in mo_config['aggregation']:
                raise ValueError("使用加權聚合時必須提供 dataset_weights")

        logger.info("配置驗證通過")

    @staticmethod
    def _load_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        如需要，從 dataset_config.yaml 載入資料集配置

        這允許重用現有的資料集配置。
        """
        # 目前我們直接使用 optimization_config 中的資料集配置
        # 未來可以從 dataset_config.yaml 載入
        return config

    @staticmethod
    def get_enabled_datasets(config: Dict[str, Any], mode: str = 'baseline') -> Dict[str, int]:
        """
        獲取已啟用的資料集及其樣本數

        Args:
            config: 配置字典
            mode: 'baseline' 或 'quantization'

        Returns:
            字典 {dataset_name: num_samples}
        """
        if mode == 'baseline':
            datasets = config['baseline']['datasets']
        else:
            # 檢查是使用與基線相同的資料集還是自訂資料集
            if config['quantization_eval']['use_same_datasets_as_baseline']:
                datasets = config['baseline']['datasets']
            else:
                datasets = config['quantization_eval']['custom_datasets']

        # 過濾已啟用的資料集
        enabled = {}
        for name, cfg in datasets.items():
            if cfg.get('enabled', True):
                enabled[name] = cfg['num_samples']

        logger.info(f"已啟用的資料集（{mode}）：{list(enabled.keys())}")
        return enabled

    @staticmethod
    def get_search_space(config: Dict[str, Any]) -> Dict[str, Any]:
        """獲取所有方法的參數搜索空間"""
        return config['optimizer']['search_space']

    @staticmethod
    def get_objectives(config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """獲取目標列表及其配置"""
        return config['multiobjective']['objectives']

    @staticmethod
    def get_aggregation_weights(config: Dict[str, Any]) -> Dict[str, float]:
        """
        獲取聚合用的資料集權重

        Returns:
            字典 {dataset_name: weight}
            如果方法是 'average'，返回相等權重
        """
        agg_config = config['multiobjective']['aggregation']
        method = agg_config['method']

        if method == 'weighted':
            return agg_config['dataset_weights']
        elif method == 'average':
            # 獲取所有已啟用的資料集
            datasets = ConfigLoader.get_enabled_datasets(config, mode='baseline')
            n = len(datasets)
            return {name: 1.0/n for name in datasets.keys()}
        else:  # worst_case
            # worst_case 不使用權重，返回空字典
            return {}

    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str) -> None:
        """將配置保存到 YAML 檔案"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"配置已保存至：{output_path}")
