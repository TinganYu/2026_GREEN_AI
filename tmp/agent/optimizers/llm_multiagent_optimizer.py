"""
LLM Multi-Agent Optimizer

基於LLM的多agent優化器，替代Optuna。
"""

import logging
import os
import random
from typing import Dict, List, Any
from datetime import datetime

from .base_optimizer import BaseOptimizer
from .agents import AnalyzerAgent, PlannerAgent, MonitorAgent
from .utils import LLMClient, ConversationLogger

logger = logging.getLogger("LLMMultiAgentOptimizer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))
logger.addHandler(handler)
logger.propagate = False


class LLMMultiAgentOptimizer(BaseOptimizer):
    """LLM-based multi-agent optimizer"""

    def __init__(self, config: Dict[str, Any], baseline_metrics: Dict[str, Any],
                 evaluator_agent, datasets: Dict[str, int], exp_dir: str = None):
        super().__init__(config, baseline_metrics, evaluator_agent, datasets)

        self.llm_config = config['optimizer']['llm_agent']
        self.search_space = config['optimizer']['search_space']
        self.stopping_config = config['optimizer']['stopping']
        self.validation_rules = config['optimizer'].get('validation', {})
        self.fallback_config = config['optimizer'].get('fallback', {})
        self.logging_config = config['optimizer'].get('logging', {})

        self.exp_dir = exp_dir

        # 初始化LLM客戶端
        self.llm_client = LLMClient(self.llm_config)

        # 初始化Agents
        agent_temps = self.llm_config.get('agent_temperatures', {})

        self.analyzer = AnalyzerAgent(
            self.llm_client,
            temperature=agent_temps.get('analyzer', 0.3)
        )

        self.planner = PlannerAgent(
            self.llm_client,
            search_space=self.search_space,
            temperature=agent_temps.get('planner', 0.7)
        )

        self.monitor = MonitorAgent(
            self.llm_client,
            convergence_window=self.stopping_config['convergence'].get('window', 5),
            convergence_threshold=self.stopping_config['convergence'].get('threshold', 0.02),
            temperature=agent_temps.get('monitor', 0.4)
        )

        # 初始化對話記錄器
        if self.logging_config.get('save_conversations', True):
            log_file = os.path.join(
                exp_dir or ".",
                self.logging_config.get('conversation_file', 'agent_conversations.jsonl')
            )
            self.conversation_logger = ConversationLogger(
                log_file,
                save_prompts=self.logging_config.get('save_prompts', False)
            )
        else:
            self.conversation_logger = None

        # Pareto歷史（用於收斂檢測）
        self.pareto_history = []

        # LLM失敗計數（用於fallback）
        self.llm_failure_count = 0
        self.max_llm_failures = self.fallback_config.get('max_llm_failures', 3)
        self.fallback_mode = False
        self.fallback_strategy = self.fallback_config.get('on_llm_failure', 'random')  # 'random' or 'stop'

        logger.info("LLMMultiAgentOptimizer initialized")
        logger.info(f"  Max trials: {self.llm_config.get('max_trials', 30)}")
        logger.info(f"  Min trials: {self.llm_config.get('min_trials', 10)}")
        logger.info(f"  Fallback strategy: {self.fallback_strategy}")

    def optimize(self) -> Dict[str, Any]:
        """
        執行multi-agent優化循環

        Returns:
            優化結果字典
        """
        logger.info("="*60)
        logger.info("LLM MULTI-AGENT OPTIMIZATION")
        logger.info("="*60)

        max_trials = self.llm_config.get('max_trials', 30)
        min_trials = self.llm_config.get('min_trials', 10)
        trial_count = 0

        # 測試LLM連接
        if not self.fallback_mode:
            if not self.llm_client.test_connection():
                if self.fallback_strategy == 'stop':
                    raise RuntimeError(
                        "LLM connection test failed. Fallback strategy is 'stop', terminating optimization."
                    )
                else:
                    logger.warning("LLM connection test failed, entering fallback mode")
                    self.fallback_mode = True

        while trial_count < max_trials:
            trial_count += 1

            logger.info(f"\n{'='*60}")
            logger.info(f"Trial {trial_count} / {max_trials}")
            logger.info(f"{'='*60}")

            if self.conversation_logger:
                self.conversation_logger.log_trial_start(trial_count)

            try:
                # ===== STEP 1: ANALYZE =====
                logger.info("[1/4] Analyzer: Analyzing trial history...")

                analysis = self._run_analyzer(trial_count)

                # ===== STEP 2: PLAN =====
                logger.info("[2/4] Planner: Deciding next configuration...")

                decision = self._run_planner(analysis, trial_count, max_trials)

                logger.info(f"  Strategy: {decision['strategy']}")
                logger.info(f"  Method: {decision['next_config'].get('method')}")
                logger.info(f"  Rationale: {decision['rationale']}")

                # ===== STEP 3: VALIDATE =====
                logger.info("[3/4] Validating configuration...")

                validated_config, is_valid = self._validate_config(decision['next_config'])

                if not is_valid:
                    logger.warning("  Config validation failed, skipping trial")
                    continue

                logger.info(f"  Config validated: {validated_config}")

                # ===== STEP 4: EXECUTE =====
                logger.info("[4/4] Executing trial...")

                trial_result = self.run_trial(validated_config, trial_count, self.exp_dir)

                if trial_result.get('success', False):
                    logger.info("  ✓ Trial succeeded")
                    self.trials.append(trial_result)

                    if self.conversation_logger:
                        self.conversation_logger.log_trial_end(
                            trial_count, True, trial_result
                        )
                else:
                    logger.error(f"  ✗ Trial failed: {trial_result.get('error', 'Unknown')}")

                    if self.conversation_logger:
                        self.conversation_logger.log_trial_end(
                            trial_count, False, trial_result
                        )

            except Exception as e:
                logger.error(f"Trial {trial_count} crashed: {e}")

                if self.conversation_logger:
                    self.conversation_logger.log_error("Optimizer", str(e), trial_count)

                # 增加LLM失敗計數
                self.llm_failure_count += 1
                if self.llm_failure_count >= self.max_llm_failures:
                    if self.fallback_strategy == 'stop':
                        logger.error(f"LLM failures ({self.llm_failure_count}) exceeded threshold. Fallback strategy is 'stop', terminating optimization.")
                        raise RuntimeError(
                            f"LLM failed {self.llm_failure_count} times (threshold: {self.max_llm_failures}). "
                            "Fallback strategy is 'stop', terminating optimization."
                        )
                    else:
                        logger.warning(f"LLM failures ({self.llm_failure_count}) exceeded threshold, entering fallback mode")
                        self.fallback_mode = True

                continue

            # ===== STEP 5: MONITOR =====
            logger.info("[5/5] Monitor: Checking stopping conditions...")

            # 更新Pareto歷史
            current_pareto = self.get_pareto_frontier(self.trials)
            self.pareto_history.append({
                'trial_count': trial_count,
                'pareto_size': len(current_pareto)
            })

            assessment = self._run_monitor(trial_count, max_trials)

            logger.info(f"  Convergence: {assessment['convergence_score']:.2f}")
            logger.info(f"  Recommendation: {assessment['recommendation']}")

            if assessment['should_stop'] and trial_count >= min_trials:
                logger.info(f"  ✓ Stopping: {assessment['reason']}")
                break
            else:
                logger.info("  → Continue")

        # ===== FINALIZATION =====
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("="*60)

        pareto_frontier = self.get_pareto_frontier(self.trials)
        satisfying_solutions = [t for t in pareto_frontier if t.get('satisfies_targets', False)]
        recommended = self.recommend_config(pareto_frontier, satisfying_solutions)

        logger.info(f"Total trials: {len(self.trials)}")
        logger.info(f"Pareto solutions: {len(pareto_frontier)}")
        logger.info(f"Satisfying solutions: {len(satisfying_solutions)}")

        if recommended:
            logger.info("\nRecommended configuration:")
            logger.info(f"  Method: {recommended['config'].get('method', 'unknown')}")
            logger.info(f"  Accuracy change: {recommended['objectives']['accuracy_change']:+.3%}")
            logger.info(f"  GPU peak change: {recommended['objectives']['gpu_peak_change']:+.3%}")
            logger.info(f"  Latency change: {recommended['objectives']['latency_change']:+.3%}")

        # 生成人類可讀的對話記錄
        if self.conversation_logger:
            try:
                from .utils import ConversationFormatter
                jsonl_file = str(self.conversation_logger.log_file)
                txt_file = jsonl_file.replace('.jsonl', '.txt')

                ConversationFormatter.format_to_txt(jsonl_file, txt_file)
                logger.info(f"\n✓ Human-readable conversation saved to: {txt_file}")
            except Exception as e:
                logger.warning(f"Failed to generate txt conversation: {e}")

        return {
            'optimizer_type': 'llm_multiagent',
            'all_trials': self.trials,
            'pareto_frontier': pareto_frontier,
            'satisfying_solutions': satisfying_solutions,
            'recommended_config': recommended,
            'n_total_trials': len(self.trials),
            'n_pareto_solutions': len(pareto_frontier),
            'n_satisfying_solutions': len(satisfying_solutions),
            'conversation_log': self.conversation_logger.get_conversation_history() if self.conversation_logger else [],
            'llm_failures': self.llm_failure_count,
            'fallback_mode': self.fallback_mode
        }

    def _run_analyzer(self, trial_num: int) -> Dict[str, Any]:
        """運行AnalyzerAgent"""
        if self.fallback_mode:
            # Fallback: 使用簡單分析
            return self.analyzer._fallback_analysis(
                self.trials,
                self.get_pareto_frontier(self.trials),
                self.search_space
            )

        try:
            analysis = self.analyzer.process(
                input_data={
                    'trials': self.trials,
                    'pareto_frontier': self.get_pareto_frontier(self.trials),
                    'search_space': self.search_space
                },
                context={'trial_num': trial_num}
            )

            # 記錄消息
            if self.conversation_logger and self.analyzer.history:
                msg = self.analyzer.history[-1]
                self.conversation_logger.log_message(msg.to_dict())

            return analysis

        except Exception as e:
            logger.error(f"Analyzer failed: {e}")
            self.llm_failure_count += 1

            # 如果策略是 'stop'，直接拋出異常
            if self.fallback_strategy == 'stop':
                raise RuntimeError(f"AnalyzerAgent failed and fallback strategy is 'stop': {e}")

            # 否則使用 fallback
            return self.analyzer._fallback_analysis(
                self.trials,
                self.get_pareto_frontier(self.trials),
                self.search_space
            )

    def _run_planner(self, analysis: Dict, trial_num: int, max_trials: int) -> Dict[str, Any]:
        """運行PlannerAgent"""
        if self.fallback_mode:
            # Fallback: 使用簡單規劃
            return self.planner._fallback_planning(
                analysis,
                trial_num,
                {'used': trial_num, 'max': max_trials}
            )

        try:
            decision = self.planner.process({
                'analysis': analysis,
                'trial_num': trial_num,
                'budget': {'used': trial_num, 'max': max_trials},
                'targets': self.targets_config,
                'pareto': self.get_pareto_frontier(self.trials)
            })

            # 記錄消息
            if self.conversation_logger and self.planner.history:
                msg = self.planner.history[-1]
                self.conversation_logger.log_message(msg.to_dict())

            return decision

        except Exception as e:
            logger.error(f"Planner failed: {e}")
            self.llm_failure_count += 1

            # 如果策略是 'stop'，直接拋出異常
            if self.fallback_strategy == 'stop':
                raise RuntimeError(f"PlannerAgent failed and fallback strategy is 'stop': {e}")

            # 否則使用 fallback
            return self.planner._fallback_planning(
                analysis,
                trial_num,
                {'used': trial_num, 'max': max_trials}
            )

    def _run_monitor(self, trial_num: int, max_trials: int) -> Dict[str, Any]:
        """運行MonitorAgent"""
        if self.fallback_mode:
            # Fallback: 使用規則評估
            return self.monitor._rule_based_stopping(
                self.trials,
                self.get_pareto_frontier(self.trials),
                {'used': trial_num, 'max': max_trials},
                self.targets_config,
                self.pareto_history
            )

        try:
            assessment = self.monitor.process({
                'trials': self.trials,
                'pareto_frontier': self.get_pareto_frontier(self.trials),
                'budget_status': {'used': trial_num, 'max': max_trials},
                'targets': self.targets_config,
                'pareto_history': self.pareto_history
            })

            # 記錄消息
            if self.conversation_logger and self.monitor.history:
                msg = self.monitor.history[-1]
                self.conversation_logger.log_message(msg.to_dict())

            return assessment

        except Exception as e:
            logger.error(f"Monitor failed: {e}")
            self.llm_failure_count += 1

            # 如果策略是 'stop'，直接拋出異常
            if self.fallback_strategy == 'stop':
                raise RuntimeError(f"MonitorAgent failed and fallback strategy is 'stop': {e}")

            # 否則使用 fallback
            return self.monitor._rule_based_stopping(
                self.trials,
                self.get_pareto_frontier(self.trials),
                {'used': trial_num, 'max': max_trials},
                self.targets_config,
                self.pareto_history
            )

    def _validate_config(self, config: Dict[str, Any]) -> tuple:
        """
        驗證配置（簡單規則驗證，Phase 1不使用ExpertAgent）

        Args:
            config: 配置字典

        Returns:
            (validated_config, is_valid)
        """
        # 基本驗證
        method = config.get('method')
        if not method or method not in self.search_space.get('methods', []):
            logger.warning(f"Invalid method: {method}")
            return config, False

        # 檢查GPTQ特定規則
        if method == 'gptq':
            # act_group_aware requires desc_act=False
            if config.get('act_group_aware') and config.get('desc_act', True):
                logger.warning("act_group_aware=True requires desc_act=False, adjusting...")
                config['desc_act'] = False

        # 檢查bits是否合理
        bits = config.get('bits')
        if bits and bits == 2:
            logger.warning("2-bit quantization may be unstable")

        return config, True
