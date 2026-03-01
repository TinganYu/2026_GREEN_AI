"""
BIG-Bench Hard 評估器
====================

評估模型在困難推理任務上的表現
"""

from datetime import datetime
from io import TextIOWrapper
import logging
import os
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datasets import load_dataset
import re

import numpy as np
import torch

from .base_evaluator import BaseEvaluator, BaseEvalConfig

logger = logging.getLogger("BBH")
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
class BBHConfig(BaseEvalConfig):
    """BIG-Bench Hard 配置（繼承完整的 BaseEvalConfig）"""
    dataset_name: str = "lukaemon/bbh"
    dataset_split: str = "test"  # 使用測試集
    max_new_tokens: int = 1024
    prompt_type: str = "fewshot"  # direct, fewshot, cot
    task_name: Optional[str] = None  # 指定任務，None 表示全部
    available_tasks: List[str] = field(default_factory=list)


class BBHEvaluator(BaseEvaluator):
    """
    BIG-Bench Hard 評估器
    
    包含 27 個困難的推理任務
    """
    ConfigClass = BBHConfig

    def __init__(self, config: Union[Dict, BBHConfig]):
        """初始化評估器，支援 YAML、Dict 或 Config 物件"""
        super().__init__(config, dataset_name="bbh")

    def build_prompt(self, sample: Dict[str, Any], task_name: str) -> str:
        """構建提示詞"""
        input_text = sample["input"]
        
        if self.config.prompt_type == "direct":
            return self._build_direct_prompt(input_text)
        elif self.config.prompt_type == "fewshot":
            return self._build_fewshot_prompt(input_text, task_name)
        elif self.config.prompt_type == "cot":
            return self._build_cot_prompt(input_text)
    
    def _build_direct_prompt(self, input_text: str) -> str:
        """直接提示詞"""
        try:
            messages = [
                {"role": "user", "content": self.config.prompts["direct"].format(input=input_text)},
            ]
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except:
            return self.config.prompts["direct"].format(input=input_text)

    def _build_fewshot_prompt(self, input_text: str, task_name: str) -> str:
        """Few-shot 提示"""
        try:
            messages = []
            for ex in self.config.fewshot_examples[task_name]:
                messages.append({
                    "role": "user", 
                    "content": self.config.prompts["fewshot"].format(input=ex['input'])
                })
                messages.append({
                    "role": "assistant", 
                    "content": ex["target"]
                })
            
            messages.append({
                "role": "user", 
                "content": self.config.prompts["fewshot"].format(input=input_text)
            })
            
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            prompt = ""
            for ex in self.config.fewshot_examples[task_name]:
                prompt += self.config.prompts["fewshot"].format(input=ex['input'])
                prompt += f"\n{ex['target']}\n\n"
            prompt += self.config.prompts["fewshot"].format(input=input_text)
            return prompt
    
    def _build_cot_prompt(self, input_text: str) -> str:
        """Chain-of-thought 提示詞"""
        try:
            messages = [
                {"role": "user", "content": self.config.prompts["cot"].format(input=input_text)},
            ]
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except:
            return self.config.prompts["cot"].format(input=input_text)
    
    def extract_answer(self, output: str) -> str:
        """提取答案（支援多種格式與 Markdown 樣式）"""
        text = output.strip()
        lines = text.split('\n')

        # 由下往上掃描（通常答案在結尾）
        for line in reversed(lines):
            line = line.strip()
            lower_line = line.lower()

            if "answer" in lower_line:
                # 擷取 "answer" 之後的內容
                match = re.search(r"(?:answer\s*(?:is|:)?)\s*(.+)", line, re.IGNORECASE)
                if match:
                    ans = match.group(1).strip()
                    # 去掉尾端句號（但保留中間的句號，例如小數點）
                    ans = re.sub(r"\.$", "", ans)
                    return ans

        # 若沒有明確 "answer" 行，取最後一行
        if lines:
            ans = lines[-1].strip()
            # 移除 Markdown、標點
            ans = re.sub(r"[\*\`]", "", ans)
            return ans.strip().strip('.').strip('"').strip("'")

        return text
    
    def check_answer(self, true_answer: str, predicted_answer: str) -> bool:
        """檢查答案"""
        # 標準化答案
        true_ans = true_answer.strip().lower()
        pred_ans = predicted_answer.strip().lower()
        
        # 精確匹配
        if true_ans == pred_ans:
            return True
        
        # 檢查包含關係
        if true_ans in pred_ans:
            return True
        
        # 處理選項答案（A, B, C, D等）
        if len(true_ans) == 1 and true_ans.isalpha():
            if true_ans in pred_ans:
                return True
        
        # 處理 Yes/No 答案
        if true_ans in ["yes", "no"]:
            return true_ans in pred_ans
        
        return False

    def evaluate_task(self, f: TextIOWrapper, task_name: str) -> Dict[str, Any]:
        """評估單一任務"""
        logger.info(f"🔹 評估任務: {task_name}")

        if self.config.dataset_split:
            split = self.config.dataset_split

        logger.info(f"🔹 載入資料集: {self.config.dataset_name} ({split})")
        
        try:
            dataset = load_dataset(self.config.dataset_name, task_name, split=split)
        except:
            logger.error(f"❌ 無法載入任務: {task_name}")
            return None
        
        if self.config.num_samples:
            dataset = dataset.select(range(min(self.config.num_samples, len(dataset))))

        logger.info(f"📊 評估 {task_name} 樣本數: {len(dataset)}")
        
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

        f.write(f"\n{'='*80}\n")
        f.write(f"任務: {task_name}\n")
        f.write(f"{'='*80}\n\n")

        logger.info(f"🔹 開始評估...")

        total, correct = 0, 0
        task_results = []
        task_prompt_tokens = 0
        task_generated_tokens = 0
        task_generation_time = 0.0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        for idx, sample in enumerate(dataset):
            input_text = sample["input"]
            true_answer = sample["target"]

            prompt = self.build_prompt(sample, task_name)

            # 同步 GPU，準備正式計時
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_start = time.time()

            output = self.generator(prompt, **generation_kwargs)[0]["generated_text"]

            # 同步 GPU，確保生成結束
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_end = time.time()

            # 計算生成時間
            gen_time = gen_end - gen_start
            task_generation_time += gen_time

            # 計算 token 數量
            if hasattr(self, "tokenizer"):
                prompt_token_count = len(self.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
                output_token_count = len(self.tokenizer(output, return_tensors="pt")["input_ids"][0])
            else:
                prompt_token_count = len(prompt.split())
                output_token_count = len(output.split())

            task_prompt_tokens += prompt_token_count
            task_generated_tokens += output_token_count
            
            pred_answer = self.extract_answer(output)
            is_correct = self.check_answer(true_answer, pred_answer)
            
            total += 1
            if is_correct:
                correct += 1
            
            result = {
                "task_name": task_name,
                "index": idx,
                "input": input_text,
                "true_answer": true_answer,
                "predicted_answer": pred_answer,
                "is_correct": is_correct,
                "output": output,
                "prompt_tokens": prompt_token_count,
                "output_tokens": output_token_count,
                "gen_time": gen_time,
            }
            task_results.append(result)

            # 寫入文件
            f.write(f"{'='*80}\n")
            f.write(f"問題 #{idx + 1}\n")
            f.write(f"{'='*80}\n")
            f.write(f"問題: {input_text}\n")
            f.write(f"標準答案: {true_answer}\n")
            f.write(f"模型輸出:\n{output}\n")
            f.write(f"提取答案: {pred_answer}\n")
            f.write(f"正確性: {'✓' if is_correct else '✗'}\n")
            f.write(f"生成時間: {gen_time:.4f} 秒\n")
            f.write(f"Prompt tokens: {prompt_token_count} | Output tokens: {output_token_count}\n")
            f.write(f"當前準確率: {correct}/{total} = {correct/total:.4f}\n\n")
            
            # 定期顯示進度
            if (idx + 1) % 10 == 0:
                acc = correct / total
                logger.info(f"{self.dataset_name}/{task_name} 任務進度: {idx+1}/{len(dataset)} | 準確率: {acc:.4f}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        accuracy = correct / total if total > 0 else 0
        task_tokens = task_prompt_tokens + task_generated_tokens
        throughput = task_tokens / task_generation_time if task_generation_time > 0 else 0.0

        logger.info(f"🔹 平均 Throughput: {throughput:.2f} tokens/sec")
        logger.info(f"   任務 {task_name} 完成 | 準確率: {accuracy:.4f}")
        
        return {
            "task_name": task_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "gen_time": task_generation_time,
            "prompt_tokens": task_prompt_tokens,
            "output_tokens": task_generated_tokens,
            "results": task_results
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """執行評估"""
        if self.generator is None:
            raise RuntimeError("請先呼叫 load_model()")
        
        # 確定要評估的任務
        if self.config.task_name:
            tasks = [self.config.task_name]
        else:
            tasks = self.config.available_tasks
        
        logger.info(f"📊 評估 {len(tasks)} 個任務")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.config.output_dir, f"{ts}.txt")

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

        task_results = []
        total_correct, total_samples = 0, 0
        total_prompt_tokens = 0
        total_generated_tokens = 0
        total_generation_time = 0.0

        self._start_energy_tracking()

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(header)
            for task_name in tasks:
                task_result = self.evaluate_task(f, task_name)
                self.results.extend(task_result["results"])
                task_results.append({
                    "task_name": task_result["task_name"],
                    "accuracy": task_result["accuracy"],
                    "correct": task_result["correct"],
                    "total": task_result["total"],
                    "gen_time": task_result["gen_time"],
                    "prompt_tokens": task_result["prompt_tokens"],
                    "output_tokens": task_result["output_tokens"],
                })
                total_generation_time += task_result["gen_time"]
                total_prompt_tokens += task_result["prompt_tokens"]
                total_generated_tokens += task_result["output_tokens"]
                total_correct += task_result["correct"]
                total_samples += task_result["total"]

        # 統計 GPU 記憶體與 throughput
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_memory = None

        final_accuracy = total_correct / total_samples

        # throughput: (prompt + output) tokens / generation_time
        total_tokens = total_prompt_tokens + total_generated_tokens
        throughput = total_tokens / total_generation_time

        logger.info(f"🔹 平均 Throughput: {throughput:.2f} tokens/sec")
        if peak_memory:
            logger.info(f"🔹 峰值 GPU 記憶體使用量: {peak_memory:.2f} MB")
        logger.info(f"✅ 全部評估完成！整體準確率: {final_accuracy:.4f}")
        logger.info(f"📄 結果已儲存至: {output_file}")

        energy_metrics = self._stop_energy_tracking()

        return {
            "accuracy": final_accuracy,
            "correct": total_correct,
            "total": total_samples,
            "model_path": self.config.model_path,
            "dataset": f"{self.config.dataset_name} ({self.config.dataset_split})",
            "num_samples": self.config.num_samples or "ALL",
            "num_tasks": len(tasks),
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
            "task_results": task_results,
            "results": self.results
        }
