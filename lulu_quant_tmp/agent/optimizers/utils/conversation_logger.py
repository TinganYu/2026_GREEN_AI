"""
Conversation Logger

記錄Agent間的對話到JSONL文件。
"""

import json
import logging
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("ConversationLogger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class ConversationLogger:
    """Agent對話記錄器"""

    def __init__(self, log_file: str, save_prompts: bool = False):
        """
        初始化對話記錄器

        Args:
            log_file: 日誌文件路徑（JSONL格式）
            save_prompts: 是否保存完整的prompts（調試用）
        """
        self.log_file = Path(log_file)
        self.save_prompts = save_prompts
        self.conversation_history = []

        # 確保目錄存在
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # 如果文件已存在，清空或備份
        if self.log_file.exists():
            backup = self.log_file.with_suffix('.jsonl.bak')
            self.log_file.rename(backup)
            logger.info(f"Backed up existing log to: {backup}")

        logger.info(f"ConversationLogger initialized: {self.log_file}")

    def log_message(self, message: Dict[str, Any]):
        """
        記錄一條消息

        Args:
            message: 消息字典（來自AgentMessage.to_dict()）
        """
        # 添加到內存歷史
        self.conversation_history.append(message)

        # 寫入文件（JSONL格式：每行一個JSON對象）
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(message, ensure_ascii=False) + '\n')

    def log_trial_start(self, trial_num: int):
        """記錄Trial開始"""
        message = {
            'event': 'trial_start',
            'trial_num': trial_num,
            'timestamp': datetime.now().isoformat()
        }
        self.log_message(message)

    def log_trial_end(self, trial_num: int, success: bool, result: Dict[str, Any]):
        """記錄Trial結束"""
        message = {
            'event': 'trial_end',
            'trial_num': trial_num,
            'success': success,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        self.log_message(message)

    def log_llm_call(self, agent_name: str, prompt: str, response: str,
                    trial_context: int = None):
        """
        記錄LLM調用（如果啟用save_prompts）

        Args:
            agent_name: Agent名稱
            prompt: 發送的prompt
            response: LLM回應
            trial_context: Trial編號
        """
        if not self.save_prompts:
            return

        message = {
            'event': 'llm_call',
            'agent': agent_name,
            'prompt': prompt,
            'response': response,
            'trial_context': trial_context,
            'timestamp': datetime.now().isoformat()
        }
        self.log_message(message)

    def log_error(self, agent_name: str, error: str, trial_context: int = None):
        """記錄錯誤"""
        message = {
            'event': 'error',
            'agent': agent_name,
            'error': str(error),
            'trial_context': trial_context,
            'timestamp': datetime.now().isoformat()
        }
        self.log_message(message)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """獲取完整對話歷史"""
        return self.conversation_history

    def get_trial_conversations(self, trial_num: int) -> List[Dict[str, Any]]:
        """獲取特定trial的對話"""
        return [
            msg for msg in self.conversation_history
            if msg.get('trial_context') == trial_num or
               msg.get('trial_num') == trial_num
        ]

    def save_summary(self, output_file: str):
        """
        保存對話摘要

        Args:
            output_file: 輸出文件路徑（JSON格式）
        """
        summary = {
            'total_messages': len(self.conversation_history),
            'trials': self._summarize_by_trial(),
            'agents': self._summarize_by_agent(),
            'errors': [msg for msg in self.conversation_history if msg.get('event') == 'error']
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dumps(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Conversation summary saved to: {output_file}")

    def _summarize_by_trial(self) -> Dict[int, int]:
        """按trial統計消息數量"""
        trial_counts = {}
        for msg in self.conversation_history:
            trial = msg.get('trial_context') or msg.get('trial_num')
            if trial is not None:
                trial_counts[trial] = trial_counts.get(trial, 0) + 1
        return trial_counts

    def _summarize_by_agent(self) -> Dict[str, int]:
        """按agent統計消息數量"""
        agent_counts = {}
        for msg in self.conversation_history:
            agent = msg.get('from_agent') or msg.get('agent')
            if agent:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
        return agent_counts


if __name__ == "__main__":
    # 測試代碼
    logger_test = ConversationLogger("test_conversation.jsonl", save_prompts=True)

    logger_test.log_trial_start(1)
    logger_test.log_message({
        'from_agent': 'AnalyzerAgent',
        'to_agent': 'PlannerAgent',
        'message_type': 'analysis_report',
        'content': {'test': 'data'},
        'trial_context': 1,
        'timestamp': datetime.now().isoformat()
    })
    logger_test.log_trial_end(1, True, {'accuracy': 0.85})

    print(f"✓ Logged {len(logger_test.get_conversation_history())} messages")
    print(f"✓ Log file: {logger_test.log_file}")
