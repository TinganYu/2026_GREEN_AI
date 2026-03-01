"""
Conversation Formatter

將 JSONL 格式的 agent 對話記錄轉換為人類可讀的 txt 格式。
"""

import json
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger("ConversationFormatter")


class ConversationFormatter:
    """格式化 agent 對話為人類可讀的文本"""

    @staticmethod
    def format_to_txt(jsonl_file: str, output_file: str) -> None:
        """
        將 JSONL 對話記錄轉換為可讀的 txt 文件

        Args:
            jsonl_file: agent_conversations.jsonl 的路徑
            output_file: 輸出 txt 文件的路徑
        """
        try:
            # 讀取 JSONL
            messages = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        messages.append(json.loads(line))

            # 按 trial 分組
            trials = ConversationFormatter._group_by_trial(messages)

            # 格式化並寫入 txt
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("LLM MULTI-AGENT OPTIMIZATION - CONVERSATION LOG\n")
                f.write("=" * 70 + "\n\n")

                for trial_num, trial_messages in sorted(trials.items()):
                    trial_text = ConversationFormatter._format_trial(trial_num, trial_messages)
                    f.write(trial_text)
                    f.write("\n\n")

            logger.info(f"Human-readable conversation saved to: {output_file}")

        except Exception as e:
            logger.error(f"Failed to format conversation: {e}")
            raise

    @staticmethod
    def _group_by_trial(messages: List[Dict]) -> Dict[int, List[Dict]]:
        """按 trial 分組消息（基於時間順序和 trial_start 事件）"""
        # 先按時間戳排序所有消息
        sorted_messages = sorted(messages, key=lambda m: m.get('timestamp', ''))

        trials = {}
        current_trial = 0  # 當前 trial 編號

        for msg in sorted_messages:
            # 檢測 trial_start 事件，更新當前 trial
            if msg.get('event') == 'trial_start':
                current_trial = msg.get('trial_num', current_trial)

            # 所有消息歸入當前 trial（包括 trial_start 本身）
            if current_trial not in trials:
                trials[current_trial] = []
            trials[current_trial].append(msg)

        return trials

    @staticmethod
    def _format_trial(trial_num: int, messages: List[Dict]) -> str:
        """格式化單個 trial 的消息"""
        lines = []

        # Trial 標題
        lines.append("=" * 70)
        lines.append(f"Trial {trial_num}")
        lines.append("=" * 70)

        for msg in messages:
            formatted = ConversationFormatter._format_message(msg)
            if formatted:
                lines.append(formatted)
                lines.append("-" * 70)

        return "\n".join(lines)

    @staticmethod
    def _format_message(msg: Dict) -> str:
        """格式化單個消息"""
        # 處理 trial 事件
        if msg.get('event') == 'trial_start':
            timestamp = msg.get('timestamp', '')
            return f"\n[{ConversationFormatter._format_timestamp(timestamp)}] TRIAL START\n"

        elif msg.get('event') == 'trial_end':
            timestamp = msg.get('timestamp', '')
            success = msg.get('success', False)
            status = "✓ SUCCESS" if success else "✗ FAILED"
            result_summary = ConversationFormatter._format_trial_result(msg.get('result', {}))
            return f"\n[{ConversationFormatter._format_timestamp(timestamp)}] TRIAL END - {status}\n{result_summary}\n"

        # 處理 agent 消息
        elif msg.get('from_agent'):
            return ConversationFormatter._format_agent_message(msg)

        return ""

    @staticmethod
    def _format_agent_message(msg: Dict) -> str:
        """格式化 agent 消息"""
        from_agent = msg.get('from_agent', 'Unknown')
        to_agent = msg.get('to_agent', 'Unknown')
        message_type = msg.get('message_type', 'unknown')
        timestamp = msg.get('timestamp', '')
        content = msg.get('content', {})

        lines = []
        lines.append(f"\n[{ConversationFormatter._format_timestamp(timestamp)}] {from_agent} → {to_agent}")
        lines.append(f"Type: {message_type}")
        lines.append("")

        # 根據消息類型格式化內容
        if message_type == "analysis_report":
            lines.append(ConversationFormatter._format_analysis(content))
        elif message_type == "strategy_decision":
            lines.append(ConversationFormatter._format_decision(content))
        elif message_type == "progress_assessment":
            lines.append(ConversationFormatter._format_assessment(content))
        else:
            lines.append(f"Content: {json.dumps(content, indent=2, ensure_ascii=False)}")

        return "\n".join(lines)

    @staticmethod
    def _format_analysis(content: Dict) -> str:
        """格式化分析報告"""
        lines = []

        input_summary = content.get('input_summary', {})
        analysis = content.get('analysis', {})

        lines.append(f"Input: {input_summary.get('trials_count', 0)} trials analyzed, "
                    f"{input_summary.get('pareto_count', 0)} Pareto solutions")

        if analysis.get('recommendations'):
            lines.append("\nRecommendations:")
            for i, rec in enumerate(analysis['recommendations'][:3], 1):
                lines.append(f"  {i}. {rec}")

        if analysis.get('unexplored_regions'):
            lines.append(f"\nUnexplored regions: {len(analysis['unexplored_regions'])}")

        if content.get('fallback_mode'):
            lines.append("\n⚠️  Fallback mode (LLM unavailable)")

        return "\n".join(lines)

    @staticmethod
    def _format_decision(content: Dict) -> str:
        """格式化策略決策"""
        lines = []

        input_summary = content.get('input_summary', {})
        decision = content.get('decision', {})

        lines.append(f"Progress: {input_summary.get('budget_progress', 'N/A')}")
        lines.append(f"Strategy: {decision.get('strategy', 'unknown')}")

        next_config = decision.get('next_config', {})
        lines.append(f"\nNext Configuration:")
        lines.append(f"  Method: {next_config.get('method', 'unknown')}")

        # 顯示關鍵參數
        if 'bits' in next_config:
            lines.append(f"  Bits: {next_config['bits']}")
        if 'w_bit' in next_config:
            lines.append(f"  W-bit: {next_config['w_bit']}")
        if 'group_size' in next_config:
            lines.append(f"  Group size: {next_config['group_size']}")
        if 'q_group_size' in next_config:
            lines.append(f"  Q-group size: {next_config['q_group_size']}")

        if decision.get('rationale'):
            lines.append(f"\nRationale:")
            lines.append(f"  {decision['rationale']}")

        if decision.get('confidence'):
            lines.append(f"\nConfidence: {decision['confidence']:.2f}")

        if content.get('fallback_mode'):
            lines.append("\n⚠️  Fallback mode (LLM unavailable)")

        return "\n".join(lines)

    @staticmethod
    def _format_assessment(content: Dict) -> str:
        """格式化進度評估"""
        lines = []

        input_summary = content.get('input_summary', {})
        assessment = content.get('assessment', {})

        lines.append(f"Trials: {input_summary.get('trials_count', 0)}, "
                    f"Pareto: {input_summary.get('pareto_count', 0)}")
        lines.append(f"Budget: {input_summary.get('budget_used', 0)}/{input_summary.get('budget_max', 0)}")

        should_stop = assessment.get('should_stop', False)
        reason = assessment.get('reason', 'N/A')

        lines.append(f"\nDecision: {'STOP' if should_stop else 'CONTINUE'}")
        lines.append(f"Reason: {reason}")

        if assessment.get('recommendation'):
            lines.append(f"\nRecommendation:")
            lines.append(f"  {assessment['recommendation']}")

        if assessment.get('convergence_score') is not None:
            lines.append(f"\nConvergence: {assessment['convergence_score']:.2f}")

        if content.get('rule_based'):
            lines.append("\n📏 Rule-based assessment")

        return "\n".join(lines)

    @staticmethod
    def _format_trial_result(result: Dict) -> str:
        """格式化 trial 結果"""
        if not result or not result.get('success'):
            return ""

        lines = []
        objectives = result.get('objectives', {})

        if objectives:
            lines.append("\nObjectives achieved:")
            lines.append(f"  Accuracy change: {objectives.get('accuracy_change', 0):+.2%}")
            lines.append(f"  GPU peak change: {objectives.get('gpu_peak_change', 0):+.2%}")
            lines.append(f"  Latency change: {objectives.get('latency_change', 0):+.2%}")

        if result.get('satisfies_targets'):
            lines.append("\n✓ Satisfies all targets!")

        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(timestamp_str: str) -> str:
        """格式化時間戳為簡短格式"""
        try:
            if not timestamp_str:
                return "??:??:??"
            dt = datetime.fromisoformat(timestamp_str)
            return dt.strftime("%H:%M:%S")
        except:
            return timestamp_str[:8] if len(timestamp_str) >= 8 else timestamp_str
