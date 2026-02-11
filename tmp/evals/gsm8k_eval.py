"""
GSM8K Evaluator
==============

GSM8K 數學推理基準評估器
"""

import re
import os
import logging
from pathlib import Path
from fractions import Fraction
import time
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass
from datasets import load_dataset
from datetime import datetime

import torch

from .base_evaluator import BaseEvaluator, BaseEvalConfig

# 數字類型定義
Number = Union[int, float]

logger = logging.getLogger("GSM8K")
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
class GSM8KConfig(BaseEvalConfig):
    """GSM8K 配置（繼承完整的 BaseEvalConfig）"""
    dataset_name: str = "openai/gsm8k"
    dataset_config: str = "main"  # 資料集配置
    dataset_split: str = "test"  # 使用測試集
    max_new_tokens: int = 1024
    prompt_type: str = "fewshot"    # direct, fewshot, cot

class GSM8KEvaluator(BaseEvaluator):
    """
    GSM8K 數學推理基準評估器
    
    支援：
    - 從 YAML 配置載入
    - Few-shot prompting
    - Chain-of-thought prompting
    - 量化模型評估 (GPTQ, AWQ, BNB)
    """
    ConfigClass = GSM8KConfig
    
    def __init__(self, config: Union[Dict, GSM8KConfig]):
        """
        初始化評估器，支援 YAML、Dict 或 Config 物件
        
        Args:
            config: 配置來源
                - str: YAML 檔案路徑（會自動處理 model_config.yaml 或 dataset_config.yaml）
                - Dict: 配置字典
                - GSM8KConfig: 配置物件
        """
        super().__init__(config, dataset_name="gsm8k")
    
    def build_prompt(self, sample: Dict[str, Any]) -> str:
        """構建提示詞"""
        question = sample.get("question", "")
        
        try:
            if self.config.prompt_type == "fewshot":
                return self._build_fewshot_prompt(question)
            elif self.config.prompt_type == "cot":
                return self._build_cot_prompt(question)
            else:
                return self._build_direct_prompt(question)
        except Exception as e:
            logger.error(f"構建提示詞 {self.config.prompt_type} 失敗: {e}")
            raise
    
    def _build_direct_prompt(self, question: str) -> str:
        """直接提問"""
        try:
            messages = [
                {"role": "user", "content": self.config.prompts["direct"].format(question=question)}
            ]
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            return self.config.prompts["direct"].format(question=question)
    
    def _build_fewshot_prompt(self, question: str) -> str:
        """Few-shot 提示"""
        try:
            messages = []
            for ex in self.config.fewshot_examples:
                messages.append({
                    "role": "user", 
                    "content": self.config.prompts["fewshot"].format(question=ex['question'])
                })
                messages.append({
                    "role": "assistant", 
                    "content": ex["answer"]
                })
            
            messages.append({
                "role": "user", 
                "content": self.config.prompts["fewshot"].format(question=question)
            })
            
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            prompt = ""
            for ex in self.config.fewshot_examples:
                prompt += self.config.prompts["fewshot"].format(question=ex['question'])
                prompt += f"\n{ex['answer']}\n\n"
            prompt += self.config.prompts["fewshot"].format(question=question)
            return prompt
    
    def _build_cot_prompt(self, question: str) -> str:
        """Chain-of-thought 提示"""
        try:
            messages = [
                {"role": "user", "content": self.config.prompts["cot"].format(question=question)}
            ]
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            return self.config.prompts["cot"].format(question=question)
    
    def extract_answer(self, output: str, last_n_lines: int = 3) -> Optional[Number]:
        """
        從模型輸出中提取數字答案
        
        優先順序：
        1. '####' 後的數字 (最後一個)
        2. 含有 "final answer/答案" 等關鍵詞的行 (最後一個)
        3. 輸出最後幾行的數字 (最後一個)
        4. 最後一個混合分數/分數/數字
        """
        if not output:
            return None

        text = output.strip()
        candidates = []

        # Case 1: '#### number'
        if "####" in text:
            after_hashes_all = re.findall(r'####\s*([^\n]+)', text)
            for seg in after_hashes_all:
                parsed = self._parse_number_str(seg)
                if parsed is not None:
                    candidates.append(("case1", parsed))

        # Case 2: 關鍵詞標記的答案
        answer_labels = [
            r'final answer', r'final', r'answer', r'ans', r'solution',
        ]
        lines = text.splitlines()
        for i, raw_line in enumerate(lines):
            line = raw_line.strip()
            for label in answer_labels:
                if re.search(rf'(?i)\b{re.escape(label)}\b', line):
                    parsed = self._parse_number_str(line)
                    if parsed is not None:
                        candidates.append(("case2", parsed))

        # Case 3: 看最後幾行
        non_empty_lines = [ln for ln in lines if ln.strip()]
        for line in non_empty_lines[-last_n_lines:]:
            parsed = self._parse_number_str(line)
            if parsed is not None:
                candidates.append(("case3", parsed))

        # Case 4: Fallback
        mixed_all = re.findall(r'[-+]?\d+\s+\d+\/\d+', text)
        for val in mixed_all:
            candidates.append(("case4", self._parse_number_str(val)))

        frac_all = re.findall(r'[-+]?\d+\/\d+', text)
        for val in frac_all:
            candidates.append(("case4", self._parse_number_str(val)))

        num_all = re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', text)
        for val in num_all:
            candidates.append(("case4", self._parse_number_str(val)))

        # 依照優先順序選最後一個
        for case in ["case1", "case2", "case3", "case4"]:
            case_candidates = [val for tag, val in candidates if tag == case]
            if case_candidates:
                return case_candidates[-1]

        return None
    
    def check_answer(self, true_answer: Any, predicted_answer: Any, tol: float = 1e-6) -> bool:
        """檢查答案是否正確（實現 BaseEvaluator 的抽象方法）"""
        if true_answer is None or predicted_answer is None:
            return False

        # Case: true_answer 來自 '####'
        if isinstance(true_answer, str) and "####" in true_answer:
            true_answer = self._parse_number_str(true_answer.split("####")[-1]) or true_answer

        parsed_true = self._parse_number_str(str(true_answer)) if not isinstance(true_answer, (int, float)) else true_answer
        parsed_pred = self._parse_number_str(str(predicted_answer)) if not isinstance(predicted_answer, (int, float)) else predicted_answer

        # 數字比較
        if isinstance(parsed_true, (int, float)) and isinstance(parsed_pred, (int, float)):
            if isinstance(parsed_true, int) and isinstance(parsed_pred, int):
                return parsed_true == parsed_pred
            try:
                return abs(float(parsed_true) - float(parsed_pred)) <= tol * max(1.0, abs(float(parsed_true)))
            except:
                return False

        # 字串比較
        try:
            s_true = str(true_answer).strip().rstrip('.').lower()
            s_pred = str(predicted_answer).strip().rstrip('.').lower()
            return s_true == s_pred
        except:
            return False
        
    def extract_true_answer(self, answer_str: str) -> Optional[Number]:
        """從 GSM8K 的答案格式取出 '#### number'"""
        if "####" in answer_str:
            parts = answer_str.split("####")
            final = parts[-1].strip().replace(",", "")
            return self._parse_number_str(final)
        return None
    
    def _normalize_text(self, s: str) -> str:
        """去除多餘空格、逗號、特殊空白符號"""
        if s is None:
            return ""
        s = s.replace("\u00A0", " ")  # 替換不換行空格
        s = s.replace(",", "")        # 去除千分位逗號
        return s.strip()

    def _parse_number_str(self, s: str) -> Optional[Union[Number, str]]:
        """將字串解析成 int 或 float，支援百分比、小數、分數、科學記號等"""
        if s is None:
            return None
        s = self._normalize_text(s)
        s = s.strip(" \t\n\r.()[]")
        if s == "":
            return None

        # 百分比 (e.g. "50%")
        m = re.fullmatch(r'([-+]?\d+(?:\.\d+)?)\s*%$', s)
        if m:
            try:
                val = float(m.group(1))
                return int(val) if val.is_integer() else val
            except:
                return None

        # 帶整數的分數 (e.g. "2 1/3")
        m = re.fullmatch(r'([-+]?\d+)\s+(\d+)\/(\d+)$', s)
        if m:
            try:
                whole = int(m.group(1))
                num = int(m.group(2))
                den = int(m.group(3))
                frac = Fraction(num, den)
                value = whole + (frac if whole >= 0 else -frac)
                return int(value) if value.denominator == 1 else float(value)
            except:
                return None

        # 單純分數 (e.g. "3/4")
        m = re.fullmatch(r'([-+]?\d+)\/(\d+)$', s)
        if m:
            try:
                num = int(m.group(1))
                den = int(m.group(2))
                value = Fraction(num, den)
                return int(value) if value.denominator == 1 else float(value)
            except:
                return None

        # 科學記號 or 一般數字 (e.g. "1e-3", "42.5")
        sci_float_re = r'^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$'
        if re.fullmatch(sci_float_re, s):
            try:
                val = float(s)
                return int(val) if val.is_integer() else val
            except:
                return None

        # 嘗試找混合分數 (只取最後一個)
        mixed_matches = re.findall(r'[-+]?\d+\s+\d+\/\d+', s)
        if mixed_matches:
            return self._parse_number_str(mixed_matches[-1])

        # 嘗試找分數
        frac_matches = re.findall(r'[-+]?\d+\/\d+', s)
        if frac_matches:
            return self._parse_number_str(frac_matches[-1])

        # 嘗試找數字 (最後一個)
        num_matches = re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', s)
        if num_matches:
            return self._parse_number_str(num_matches[-1])

        return None
    
    def evaluate(self) -> Dict[str, Any]:
        """執行評估"""
        if not self.is_model_loaded():
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

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(header)
            for idx, sample in enumerate(dataset):
                question = sample["question"]
                true_answer_str = sample["answer"]
                true_answer = self.extract_true_answer(true_answer_str)

                # 準備 prompt
                prompt = self.build_prompt(sample)

                # 同步 GPU，準備正式計時
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gen_start = time.time()

                # 模型生成（使用統一介面，支援 transformers 和 vLLM）
                output = self.generate(prompt, **generation_kwargs)

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

                # 驗證答案
                pred_answer = self.extract_answer(output)
                is_correct = self.check_answer(true_answer, pred_answer)

                total += 1
                if is_correct:
                    correct += 1

                # 記錄結果
                result = {
                    "index": idx,
                    "question": question,
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
                f.write(f"問題: {question}\n")
                f.write(f"標準答案: {true_answer}\n")
                f.write(f"模型輸出:\n{output}\n")
                f.write(f"提取答案: {pred_answer}\n")
                f.write(f"正確性: {'✓' if is_correct else '✗'}\n")
                f.write(f"生成時間: {gen_time:.4f} 秒\n")
                f.write(f"Prompt tokens: {prompt_token_count} | Output tokens: {output_token_count}\n")
                f.write(f"當前準確率: {correct}/{total} = {correct/total:.4f}\n\n")

                # 進度顯示
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

        return {
            "accuracy": final_accuracy,
            "correct": correct,
            "total": total,
            "model_path": self.config.model_path,
            "dataset": f"{self.config.dataset_name} ({self.config.dataset_split})",
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
            "quantization_config": self.get_quantization_config(),
            "results": self.results
        }