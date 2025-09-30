import logging
import torch
import psutil
import os
from agents.KVpress import KVPressAgent

logger = logging.getLogger(__name__)

class AdaptiveKVController:
    """Adaptive controller that tunes KVPressAgent per sample"""

    def __init__(self, config, base_agent=None):
        self.config = config
        self.kv_agent = base_agent or KVPressAgent(config)

        # Mimic KVPressAgent attributes
        self.model_name = self.kv_agent.model_name
        self.tokenizer = self.kv_agent.tokenizer
        self.model = self.kv_agent.model
        self.device = self.kv_agent.device

    # ------------- Passthroughs for properties ----------------
    @property
    def current_method(self):
        return self.kv_agent.compression_method

    # ------------- Passthrough for methods ------------------
    def __getattr__(self, name):
        return getattr(self.kv_agent, name)

    # ------------- Main methods ------------------
    def generate_with_compression(self, prompt, **kwargs):
        return self.kv_agent.generate_with_compression(prompt, **kwargs)

    def run(self, sample, evaluator=None):
        """Forward pass with adaptive tuning"""
        output = self.kv_agent.generate_with_compression(sample.get("prompt", sample.get("question", "")))

        if evaluator:
            feedback = evaluator.get_feedback(sample, output)
            self.adjust(feedback)

        return output

    def adjust(self, feedback: dict):
        """
        Adjust KV parameters conservatively based on per-sample feedback.
        Uses rollback logic to avoid oscillation.
        """
        old_ratio = self.kv_agent.compression_ratio
        old_accuracy = getattr(self, "prev_accuracy", 0)

        candidate_ratio = old_ratio

        # Detect NaNs and switch to safer method
        if feedback.get("nans", 0) > 0:
            logger.warning("Detected NaNs! Switching to 'knorm' compression.")
            self.kv_agent.set_method("knorm")

        # Adjust compression based on accuracy
        if "accuracy" in feedback:
            acc = feedback["accuracy"]
            if acc < 0.4:  # Low accuracy threshold
                candidate_ratio = min(1.0, candidate_ratio + 0.05)  # Keep more tokens
                logger.info(f"Accuracy low ({acc:.2f}), increasing ratio to: {candidate_ratio:.2f}")
            elif acc > 0.8:  # High accuracy threshold
                candidate_ratio = max(0.2, candidate_ratio - 0.05)  # Can compress more
                logger.info(f"Accuracy high ({acc:.2f}), decreasing ratio to: {candidate_ratio:.2f}")

        # Adjust compression based on memory
        if "memory" in feedback and "max_mem" in feedback:
            mem_usage = feedback["memory"] / feedback["max_mem"]
            if mem_usage > 0.9:  # Using >90% memory
                candidate_ratio = max(0.2, candidate_ratio - 0.1)  # Compress more aggressively
                logger.info(f"Memory critical ({mem_usage:.1%}), decreasing ratio to: {candidate_ratio:.2f}")

        # Commit if score improves, else rollback
        score_old = old_accuracy - 0.01 * max(0, feedback.get("memory", 0) - feedback.get("max_mem", 0))
        score_new = feedback.get("accuracy", 0) - 0.01 * max(0, feedback.get("memory", 0) - feedback.get("max_mem", 0))

        if score_new >= score_old or abs(score_new - score_old) < 0.01:  # Allow small degradation
            if candidate_ratio != old_ratio:
                logger.info(f"Committing new compression ratio: {old_ratio:.2f} → {candidate_ratio:.2f} (score {score_new:.3f} vs {score_old:.3f})")
                self.kv_agent.set_compression_ratio(candidate_ratio)
            self.prev_accuracy = feedback.get("accuracy", old_accuracy)
        else:
            logger.info(f"Rollback: keeping ratio {old_ratio:.2f} (score {score_old:.3f} > {score_new:.3f})")

    def get_memory_usage(self):
        """Return current memory usage in bytes."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device)
        # Fallback to process memory
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    def compression_ratio(self):
        return self.kv_agent.compression_ratio

    @property
    def compression_method(self):
        return self.kv_agent.compression_method

    @property
    def current_ratio(self):
        return self.kv_agent.compression_ratio

    @property
    def current_method(self):
        return self.kv_agent.compression_method

    # ------------- passthrough for methods ------------------
    def __getattr__(self, name):
        return getattr(self.kv_agent, name)

    # ------------- main methods ------------------
    def generate_with_compression(self, prompt, **kwargs):
        return self.kv_agent.generate_with_compression(prompt, **kwargs)

    def run(self, sample, evaluator=None):
        """Forward pass with adaptive tuning"""
        output = self.kv_agent.run(sample)

        if evaluator:
            feedback = evaluator.get_feedback(sample, output)
            self.adjust(feedback)

        return output

    def adjust(self, feedback: dict):
        """
        Adjust KV parameters conservatively based on per-sample feedback.
        Uses rollback logic to avoid oscillation.
        """
        old_ratio = self.kv_agent.compression_ratio
        old_accuracy = getattr(self, "prev_accuracy", 0)

        candidate_ratio = old_ratio

        # Detect NaNs and switch to safer method
        if feedback.get("nans", 0) > 0:
            logger.warning("Detected NaNs! Switching to 'topk' compression.")
            self.kv_agent.set_method("topk")

        # Adjust compression based on accuracy
        if "accuracy" in feedback:
            acc = feedback["accuracy"]
            if acc < 0.4:  # adjust threshold as needed
                candidate_ratio = min(1.0, candidate_ratio - 0.05)
                logger.info(f"Accuracy low ({acc:.2f}), candidate ratio: {candidate_ratio:.2f}")

        # Adjust compression based on memory
        if "memory" in feedback and "max_mem" in feedback:
            if feedback["memory"] > feedback["max_mem"]:
                candidate_ratio = max(0.1, candidate_ratio + 0.05)
                logger.info(f"Memory high ({feedback['memory']} > {feedback['max_mem']}), candidate ratio: {candidate_ratio:.2f}")

        # Commit if score improves, else rollback
        score_old = old_accuracy - 0.01 * max(0, feedback.get("memory", 0) - feedback.get("max_mem", 0))
        score_new = feedback.get("accuracy", 0) - 0.01 * max(0, feedback.get("memory", 0) - feedback.get("max_mem", 0))

        if score_new >= score_old:
            if candidate_ratio != old_ratio:
                logger.info(f"Committing new compression ratio: {old_ratio:.2f} → {candidate_ratio:.2f} (score {score_new:.3f} >= {score_old:.3f})")
                self.kv_agent.set_compression_ratio(candidate_ratio)
            self.prev_accuracy = feedback.get("accuracy", old_accuracy)
        else:
            logger.info(f"Rollback: keeping ratio {old_ratio:.2f} (score {score_old:.3f} > {score_new:.3f})")

