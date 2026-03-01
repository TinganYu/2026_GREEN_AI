"""
Dataset Configuration Manager
==============================

管理資料集配置的載入和處理
"""

import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger("DatasetConfigManager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
logger.propagate = False


class DatasetConfigManager:
    """
    資料集配置管理器
    
    負責載入和管理 dataset_config.yaml
    """
    
    def __init__(self, config_path: str = "tmp/config/dataset_config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路徑
        """
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
        self.common_config = self.config_data.get("common", {})
        
        logger.info(f"📋 資料集配置已載入: {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """載入 YAML 配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """
        獲取指定資料集的配置
        
        Args:
            dataset_name: 資料集名稱（如 "gsm8k", "truthfulqa"）
        
        Returns:
            dict: 資料集的完整配置
        """
        if dataset_name not in self.config_data:
            raise ValueError(
                f"資料集 '{dataset_name}' 不存在於配置中。"
            )
        
        dataset_cfg = self.config_data[dataset_name].copy()
        
        # 合併共用配置（資料集特定配置優先）
        merged_config = {**self.common_config, **dataset_cfg}
        
        return merged_config