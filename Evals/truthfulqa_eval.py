"""
TruthfulQA 評估器
================

評估模型生成答案的真實性和準確性
"""

from datetime import datetime
import logging
import os
from pathlib import Path
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datasets import load_dataset
import torch

from .base_evaluator import BaseEvaluator, BaseEvalConfig

logger = logging.getLogger("TruthfulQA")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
logger.propagate = False


@dataclass
class TruthfulQAConfig(BaseEvalConfig):
    """TruthfulQA 配置（繼承完整的 BaseEvalConfig）"""
    dataset_name: str = "truthful_qa"
    dataset_config: str = "generation"  # 資料集配置
    dataset_split: str = "validation"  # 使用驗證集
    task_type: str = "multiple_choice"  # "generation" 或 "multiple_choice"
    max_new_tokens: int = 128
    prompt_type: str = "multiple_choice"

class TruthfulQAEvaluator(BaseEvaluator):
    """
    TruthfulQA 評估器
    
    評估模型是否會生成誤導性或不真實的資訊
    """
    ConfigClass = TruthfulQAConfig
    
    def __init__(self, config: Union[Dict, TruthfulQAConfig]):
        """初始化評估器，支援 YAML、Dict 或 Config 物件"""
        super().__init__(config, dataset_name="truthfulqa")
    
    def build_prompt(self, sample: Dict[str, Any]) -> str:
        """構建提示詞"""
        question = sample["question"]
        
        if self.config.prompt_type == "multiple_choice":
            return self._build_mc_prompt(question, sample["mc1_targets"]["choices"])
        else:
            raise ValueError(f"暫不支援的任務類型: {self.config.prompt_type}")
        
    def _build_mc_prompt(self, question: str, choices: List[str]) -> str:
        """構建多選題提示詞"""
        choices = " ".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        try:
            messages = [
                {"role": "user", "content": self.config.prompts["multiple_choice"].format(question=question, choices=choices)}
            ]
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except:
            return self.config.prompts["multiple_choice"].format(question=question, choices=choices)
    
    def extract_answer(self, output: str) -> str:
        """
        提取模型輸出的答案（單一大寫字母 A-E）
        
        支援多種輸出格式，例如：
            - "Answer: C"
            - "The correct answer is D"
            - "I think it's (b)"
            - "Final answer → E."
            - "So answer is a"
            - "C"
        """
        import re

        # 清理輸入
        output = output.strip().upper()

        # === 1️⃣ 第一優先：尋找明確標記 ===
        patterns = [
            r'\b(?:ANSWER|ANS|OPTION|CHOICE)\s*[:\-–=]*\s*\(?\s*([A-E])\s*\)?',  # e.g. Answer: C / Choice - B
            r'\b(?:IS|BE|→|=)\s*\(?\s*([A-E])\s*\)?',                           # e.g. Answer is D / -> A
            r'\bFINAL\s+ANSWER\s*[:\-–=]*\s*\(?\s*([A-E])\s*\)?'                # e.g. Final answer: E
        ]

        for p in patterns:
            match = re.search(p, output, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # === 2️⃣ 第二優先：尋找單獨出現的選項 ===
        # 例如只回答 "A" 或 "B." 等
        match = re.match(r'^[\(\[]?\s*([A-E])[\)\].\s]*$', output)
        if match:
            return match.group(1).upper()

        # === 3️⃣ 第三優先：找到句中第一個 A-E ===
        for ch in output:
            if ch in "ABCDE":
                return ch

        return ""
    
    def check_answer(self, true_answer: Any, predicted_answer: Any) -> bool:
        """
        檢查答案（TruthfulQA 需要人工評估或使用 GPT-judge）
        這裡提供選擇題匹配。
        """        
        return true_answer == predicted_answer
    
    def evaluate(self) -> Dict[str, Any]:
        """執行評估"""
        if self.generator is None:
            raise RuntimeError("請先呼叫 load_model()")
        
        if self.config.dataset_split:
            split = self.config.dataset_split
        if self.config.dataset_config:
            dataset_config = self.config.dataset_config

        logger.info(f"🔹 載入資料集: {self.config.dataset_name} ({split})")
        dataset = load_dataset(self.config.dataset_name, dataset_config, split=split)

        if self.config.num_samples:
            dataset = dataset.select(range(min(self.config.num_samples, len(dataset))))
        
        logger.info(f"📊 評估樣本數: {len(dataset)}")

        os.makedirs(self.config.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.config.output_dir, f"{ts}.txt")
        
        # 準備 generation 參數
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature if self.config.do_sample else None,
            "top_p": self.config.top_p if self.config.do_sample else None,
            "use_cache": True,
            "return_full_text": False,
        }

        # 只有當 pad_token_id != eos_token_id 時才傳遞（避免提早停止）
        if self.tokenizer.pad_token_id != self.tokenizer.eos_token_id:
            generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        # 明確設定 eos_token_id
        if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        model_info_lines = [
            "=" * 80,
            f"Model path: {self.config.model_path}",
            f"Dataset: {self.config.dataset_name} ({self.config.dataset_split})",
            f"Prompt type: {self.config.prompt_type}",
            f"Num samples: {self.config.num_samples or 'ALL'}",
            f"Max new tokens: {self.config.max_new_tokens}",
            "=" * 80,
        ]
        header = "\n".join(model_info_lines) + "\n\n"

        logger.info(f"🔹 開始評估...")

        total, correct = 0, 0
        total_prompt_tokens = 0
        total_generated_tokens = 0
        total_generation_time = 0.0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        self._start_energy_tracking()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(header)
            for idx, sample in enumerate(dataset):
                question = sample["question"]
                choices = " ".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(sample["mc1_targets"]["choices"])])
                true_answer = chr(sample["mc1_targets"]["labels"].index(1) + 65)
                
                prompt = self.build_prompt(sample)

                # 同步 GPU，準備正式計時
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gen_start = time.time()

                # 模型生成
                output = self.generator(prompt, **generation_kwargs)[0]["generated_text"]

                # 同步 GPU，確保生成結束
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gen_end = time.time()

                # 計算生成時間
                gen_time = gen_end - gen_start
                total_generation_time += gen_time

                # 計算 token 數量
                if hasattr(self, "tokenizer"):
                    prompt_token_count = len(self.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
                    output_token_count = len(self.tokenizer(output, return_tensors="pt")["input_ids"][0])
                else:
                    prompt_token_count = len(prompt.split())
                    output_token_count = len(output.split())

                total_prompt_tokens += prompt_token_count
                total_generated_tokens += output_token_count
                
                pred_answer = self.extract_answer(output)
                is_correct = self.check_answer(true_answer, pred_answer)

                total += 1
                if is_correct:
                    correct += 1
                
                result = {
                    "index": idx,
                    "question": f"{question} {choices}",
                    "true_answer": true_answer,
                    "predicted_answer": pred_answer,
                    "is_correct": is_correct,
                    "output": output,
                    "prompt_tokens": prompt_token_count,
                    "output_tokens": output_token_count,
                    "gen_time": gen_time,
                }
                self.results.append(result)

                # 寫入檔案
                f.write(f"{'='*80}\n")
                f.write(f"問題 #{idx + 1}\n")
                f.write(f"{'='*80}\n")
                f.write(f"問題: {question} {choices}\n")
                f.write(f"標準答案: {true_answer}\n")
                f.write(f"模型輸出:\n{output}\n")
                f.write(f"提取答案: {pred_answer}\n")
                f.write(f"正確性: {'✓' if is_correct else '✗'}\n")
                f.write(f"生成時間: {gen_time:.4f} 秒\n")
                f.write(f"Prompt tokens: {prompt_token_count} | Output tokens: {output_token_count}\n")
                f.write(f"當前準確率: {correct}/{total} = {correct/total:.4f}\n\n")
                
                if (idx + 1) % 10 == 0:
                    acc = correct / total
                    logger.info(f"{self.dataset_name} 進度: {idx+1}/{len(dataset)} | 準確率: {acc:.4f}")
        
        # 統計 GPU 記憶體與 throughput
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_memory = None

        final_accuracy = correct / total if total > 0 else 0.0

        # throughput: (prompt + output) tokens / generation_time
        total_tokens = total_prompt_tokens + total_generated_tokens
        throughput = total_tokens / total_generation_time if total_generation_time > 0 else 0.0

        logger.info(f"🔹 平均 Throughput: {throughput:.2f} tokens/sec")
        if peak_memory:
            logger.info(f"🔹 峰值 GPU 記憶體使用量: {peak_memory:.2f} MB")
        logger.info(f"✅ 評估完成！最終準確率: {final_accuracy:.4f} ({correct}/{total})")
        logger.info(f"📄 結果已儲存至: {output_file}")

        energy_metrics = self._stop_energy_tracking()

        return {
            "accuracy": final_accuracy,
            "correct": correct,
            "total": total,
            "model_path": self.config.model_path,
            "dataset":  f"{self.config.dataset_name} ({self.config.dataset_split})",
            "num_samples": self.config.num_samples or "ALL",
            "prompt_type": self.config.prompt_type,
            "temperature": self.config.temperature,
            "max_new_tokens": self.config.max_new_tokens,
            "output_file": output_file,
            "gpu_peak_mb": round(peak_memory, 4) if peak_memory else None,
            "total_prompt_tokens": total_prompt_tokens,
            "total_output_tokens": total_generated_tokens,
            "total_generation_time_sec": round(total_generation_time, 4),
            "throughput_tokens_per_sec": round(throughput, 4),
            **energy_metrics,
            "quantization_config": self.get_quantization_config(),
            # "results": self.results
        }
