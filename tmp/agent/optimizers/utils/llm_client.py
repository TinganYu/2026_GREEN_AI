"""
LLM客戶端

統一的LLM API封裝，支持多個provider，包含錯誤處理和重試機制。
"""

import os
import json
import re
import time
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger("LLMClient")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class BaseLLMProvider(ABC):
    """LLM Provider基類"""

    @abstractmethod
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """生成文本回應"""
        pass


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude Provider"""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """調用Claude API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class OpenAIProvider(BaseLLMProvider):
    """OpenAI Provider"""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """調用OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class LLMClient:
    """統一的LLM客戶端"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM客戶端

        Args:
            config: LLM配置字典，包含:
                - provider: 'anthropic' or 'openai'
                - model: 模型名稱
                - api_key_env: API key的環境變量名
                - temperature: 默認溫度
                - max_tokens: 默認最大token數
        """
        self.config = config
        self.provider_name = config.get('provider', 'anthropic')
        self.model = config.get('model', 'claude-sonnet-4')
        self.default_temperature = config.get('temperature', 0.7)
        self.default_max_tokens = config.get('max_tokens', 2048)

        # 獲取API key
        api_key_env = config.get('api_key_env', 'ANTHROPIC_API_KEY')
        api_key = os.getenv(api_key_env)

        if not api_key:
            raise ValueError(
                f"API key not found in environment variable: {api_key_env}"
            )

        # 初始化provider
        if self.provider_name == 'anthropic':
            self.provider = AnthropicProvider(api_key, self.model)
        elif self.provider_name == 'openai':
            self.provider = OpenAIProvider(api_key, self.model)
        else:
            raise ValueError(f"Unsupported provider: {self.provider_name}")

        logger.info(f"LLMClient initialized: {self.provider_name}/{self.model}")

    def generate(self,
                prompt: str,
                temperature: Optional[float] = None,
                max_tokens: Optional[int] = None,
                max_retries: int = 3,
                retry_delay: float = 2.0) -> str:
        """
        生成文本回應（帶重試機制）

        Args:
            prompt: 提示文本
            temperature: 溫度（None則使用默認值）
            max_tokens: 最大token數（None則使用默認值）
            max_retries: 最大重試次數
            retry_delay: 重試延遲（秒）

        Returns:
            LLM生成的文本

        Raises:
            RuntimeError: 所有重試都失敗
        """
        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        for attempt in range(max_retries):
            try:
                response = self.provider.generate(prompt, temp, tokens)
                return response

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"All {max_retries} retry attempts failed")
                    raise RuntimeError(
                        f"LLM generation failed after {max_retries} attempts: {e}"
                    )

    def generate_structured(self,
                           prompt: str,
                           temperature: Optional[float] = None,
                           max_tokens: Optional[int] = None,
                           max_retries: int = 3) -> Dict[str, Any]:
        """
        生成結構化JSON回應

        Args:
            prompt: 提示文本（應該要求JSON格式輸出）
            temperature: 溫度
            max_tokens: 最大token數
            max_retries: 最大重試次數

        Returns:
            解析後的JSON字典

        Raises:
            RuntimeError: 無法解析為有效JSON
        """
        response = self.generate(prompt, temperature, max_tokens, max_retries)

        # 嘗試提取JSON
        json_obj = self._extract_json(response)

        if json_obj is None:
            raise RuntimeError(
                f"Failed to extract valid JSON from LLM response:\n{response}"
            )

        return json_obj

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        從文本中提取JSON對象

        嘗試多種策略：
        1. 直接解析整個文本
        2. 尋找```json...```代碼塊
        3. 尋找{ ... }對象

        Args:
            text: 包含JSON的文本

        Returns:
            解析後的JSON字典，或None（失敗）
        """
        # 策略1: 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 策略2: 尋找```json...```代碼塊
        json_code_block = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_code_block:
            try:
                return json.loads(json_code_block.group(1))
            except json.JSONDecodeError:
                pass

        # 策略3: 尋找{ ... }對象
        json_object = re.search(r'\{.*\}', text, re.DOTALL)
        if json_object:
            try:
                return json.loads(json_object.group(0))
            except json.JSONDecodeError:
                pass

        # 所有策略都失敗
        logger.warning(f"Could not extract JSON from text:\n{text[:200]}...")
        return None

    def test_connection(self) -> bool:
        """
        測試LLM連接

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.generate(
                "Say 'Hello' in JSON format: {\"message\": \"Hello\"}",
                temperature=0.0,
                max_tokens=50,
                max_retries=1
            )
            logger.info(f"Connection test successful: {response[:100]}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# 輔助函數：從配置創建LLMClient
def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """
    從配置創建LLMClient

    Args:
        config: 配置字典

    Returns:
        LLMClient實例
    """
    return LLMClient(config)


if __name__ == "__main__":
    # 測試代碼
    test_config = {
        'provider': 'anthropic',
        'model': 'claude-sonnet-4',
        'api_key_env': 'ANTHROPIC_API_KEY',
        'temperature': 0.7,
        'max_tokens': 1024
    }

    try:
        client = LLMClient(test_config)

        # 測試連接
        if client.test_connection():
            print("✓ Connection test passed")

        # 測試結構化輸出
        prompt = """
Generate a JSON object with the following fields:
- status: "success"
- message: "Test completed"
- data: {"value": 42}

Return only the JSON object.
"""
        result = client.generate_structured(prompt, temperature=0.3)
        print(f"✓ Structured generation test passed: {result}")

    except Exception as e:
        print(f"✗ Test failed: {e}")
