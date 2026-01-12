"""
Base Agent

所有Agent的抽象基類，定義通用接口和消息協議。
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

from ..utils.llm_client import LLMClient

logger = logging.getLogger("BaseAgent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class AgentMessage:
    """Agent間通信的消息"""

    def __init__(self,
                 from_agent: str,
                 to_agent: str,
                 message_type: str,
                 content: Any,
                 trial_context: Optional[int] = None):
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.message_type = message_type
        self.content = content
        self.trial_context = trial_context
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            'from_agent': self.from_agent,
            'to_agent': self.to_agent,
            'message_type': self.message_type,
            'content': self.content,
            'trial_context': self.trial_context,
            'timestamp': self.timestamp
        }

    def __str__(self) -> str:
        return f"[{self.from_agent} → {self.to_agent}] {self.message_type}"


class BaseAgent(ABC):
    """Agent基類"""

    def __init__(self, llm_client: LLMClient, agent_name: str, temperature: float):
        """
        初始化Agent

        Args:
            llm_client: LLM客戶端
            agent_name: Agent名稱
            temperature: LLM溫度
        """
        self.llm_client = llm_client
        self.agent_name = agent_name
        self.temperature = temperature
        self.history = []  # 消息歷史

        logger.info(f"{self.agent_name} initialized (temperature={temperature})")

    @abstractmethod
    def process(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        處理輸入並返回結果

        Args:
            input_data: 輸入數據
            context: 可選的上下文信息

        Returns:
            處理結果（字典）
        """
        pass

    def create_message(self,
                      to_agent: str,
                      message_type: str,
                      content: Any,
                      trial_context: Optional[int] = None) -> AgentMessage:
        """
        創建消息

        Args:
            to_agent: 目標Agent
            message_type: 消息類型
            content: 消息內容
            trial_context: Trial上下文（編號）

        Returns:
            AgentMessage對象
        """
        msg = AgentMessage(
            from_agent=self.agent_name,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            trial_context=trial_context
        )

        # 記錄到歷史
        self.history.append(msg)

        return msg

    def _call_llm(self, prompt: str) -> str:
        """
        調用LLM（使用指定溫度）

        Args:
            prompt: Prompt文本

        Returns:
            LLM回應
        """
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_retries=3
            )
            return response

        except Exception as e:
            logger.error(f"{self.agent_name} LLM call failed: {e}")
            raise

    def _call_llm_structured(self, prompt: str) -> Dict[str, Any]:
        """
        調用LLM並獲取結構化JSON

        Args:
            prompt: Prompt文本

        Returns:
            解析後的JSON字典
        """
        try:
            response = self.llm_client.generate_structured(
                prompt=prompt,
                temperature=self.temperature,
                max_retries=3
            )
            return response

        except Exception as e:
            logger.error(f"{self.agent_name} structured LLM call failed: {e}")
            raise

    def _validate_output(self, output: Dict[str, Any], required_fields: list) -> bool:
        """
        驗證輸出包含必需字段

        Args:
            output: 輸出字典
            required_fields: 必需字段列表

        Returns:
            True if valid, False otherwise
        """
        missing = [f for f in required_fields if f not in output]

        if missing:
            logger.warning(
                f"{self.agent_name} output missing fields: {missing}"
            )
            return False

        return True

    def get_history(self) -> list:
        """獲取消息歷史"""
        return [msg.to_dict() for msg in self.history]

    def clear_history(self):
        """清空消息歷史"""
        self.history = []
