"""
HumanEval 評估器
===============

評估模型的程式碼生成能力
"""

from datetime import datetime
import logging
import os
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datasets import load_dataset
import evaluate as hf_evaluate
import re

import torch

from .base_evaluator import BaseEvaluator, BaseEvalConfig

logger = logging.getLogger("HumanEval")
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
class HumanEvalConfig(BaseEvalConfig):
    """HumanEval 配置（繼承完整的 BaseEvalConfig）"""
    dataset_name: str = "openai_humaneval"
    dataset_split: str = "test"  # 使用測試集
    max_new_tokens: int = 1024
    prompt_type: str = "instruct"  # direct, instruct

    num_samples_per_task: int = 1  # pass@k 評估時使用
    pass_k: List[int] = field(default_factory=lambda: [1])  # pass@k 列表


class HumanEvalEvaluator(BaseEvaluator):
    """
    HumanEval 評估器
    
    評估 Python 程式碼生成能力
    需要執行生成的程式碼來驗證正確性
    """
    ConfigClass = HumanEvalConfig
    
    def __init__(self, config: Union[Dict, HumanEvalConfig]):
        """初始化評估器，支援 YAML、Dict 或 Config 物件"""
        super().__init__(config, dataset_name="humaneval")
    
    def build_prompt(self, sample: Dict[str, Any]) -> str:
        """構建提示詞"""
        prompt_text = sample["prompt"]
        
        if self.config.prompt_type == "direct":
            return self._build_direct_prompt(prompt_text)
        elif self.config.prompt_type == "instruct":
            return self._build_instruct_prompt(prompt_text)
        else:
            raise ValueError(f"暫不支援的任務類型: {self.config.prompt_type}")
        
    def _build_direct_prompt(self, prompt: str) -> str:
        """直接提示詞"""
        try:
            messages = [
                {"role": "user", "content": self.config.prompts["direct"].format(prompt=prompt)},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            return self.config.prompts["direct"].format(prompt=prompt)

    def _build_instruct_prompt(self, prompt: str) -> str:
        """Instruct 風格提示詞"""
        try:
            messages = [
                {"role": "user", "content": self.config.prompts["instruct"].format(prompt=prompt)},
                {"role": "assistant", "content": self.config.prompts["gen_prefix"].format(prompt=prompt)}
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            return self.config.prompts["instruct"].format(prompt=prompt)

    def extract_answer(self, output: str) -> str:
        """提取程式碼"""
        # 嘗試提取 ```python 代碼塊
        code_block_pattern = r'```(?:python)?\s*(.*?)```'
        matches = re.findall(code_block_pattern, output, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        return output.strip()
    
    def _build_predictions_instruct(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        """
        讓每個候選程式碼都附上 prompt（以符合 code_eval 格式）
        """
        return [
            [
                doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
                for r in resp
            ]
            for resp, doc in zip(resps, docs)
        ]
    
    def check_answer(self, true_answer, predicted_answer):
        pass
    
    def evaluate(self) -> Dict[str, Any]:
        """執行評估"""
        if self.generator is None:
            raise RuntimeError("請先呼叫 load_model()")

        logger.info(f"🔹 載入資料集: {self.config.dataset_name}")
        dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split)

        if self.config.num_samples:
            dataset = dataset.select(range(min(self.config.num_samples, len(dataset))))
        
        logger.info(f"📊 評估樣本數: {len(dataset)}")
        logger.info("⚠️ HumanEval 需要執行生成的程式碼，請確保在安全環境中運行")

        os.makedirs(self.config.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.config.output_dir, f"{ts}.txt")

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature if self.config.do_sample else None,
            "top_p": self.config.top_p if self.config.do_sample else None,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
            "return_full_text": False,
        }

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

        total = 0
        references, predictions, docs, resps = [], [], [], []
        total_prompt_tokens = 0
        total_generated_tokens = 0
        total_generation_time = 0.0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.time()
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(header)
            for idx, sample in enumerate(dataset):
                task_id = sample["task_id"]
                prompt_text = sample["prompt"]
                canonical_solution = sample["canonical_solution"]
                test = sample["test"]
                
                prompt = self.build_prompt(sample)

                if self.config.num_samples_per_task > 1:
                    generation_kwargs["num_return_sequences"] = self.config.num_samples_per_task

                # 同步 GPU，準備正式計時
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gen_start = time.time()

                # 生成模型輸出
                outputs = self.generator(prompt, **generation_kwargs)

                # 同步 GPU，確保生成結束
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gen_end = time.time()

                # 計算生成時間
                gen_time = gen_end - gen_start
                total_generation_time += gen_time

                # 提取文字
                outputs_text = [out["generated_text"] for out in outputs]
                pred_codes = [self.extract_answer(o) for o in outputs_text]

                # 統計 prompt + output tokens
                if hasattr(self, "tokenizer"):
                    prompt_token_count = len(self.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
                    output_token_count = 0
                    for out in outputs:
                        text = out["generated_text"]
                        output_token_count += len(self.tokenizer(text, return_tensors="pt")["input_ids"][0])
                else:
                    prompt_token_count = len(prompt.split())
                    output_token_count = sum(len(out["generated_text"].split()) for out in outputs)
                
                total_prompt_tokens += prompt_token_count
                total_generated_tokens += output_token_count

                total += 1
                references.append(test)
                predictions.append(pred_codes)
                docs.append(sample)
                resps.append(pred_codes)

                result = {
                    "index": idx,
                    "prompt": prompt_text,
                    "canonical_solution": canonical_solution,
                    "predicted_code": pred_codes,
                    "test": test,
                    "output": outputs,
                    "prompt_tokens": prompt_token_count,
                    "output_tokens": output_token_count,
                    "gen_time": gen_time,
                }
                self.results.append(result)

                # 寫入文件
                f.write(f"{'='*80}\n")
                f.write(f"問題 #{task_id}\n")
                f.write(f"{'='*80}\n")
                f.write(f"問題: {prompt_text}\n")
                f.write(f"參考解答:\n{canonical_solution}\n")
                f.write("生成程式碼:\n" + "\n---\n".join(pred_codes) + "\n\n")
                f.write(f"測試程式碼:\n{test}\n")
                f.write("程式碼輸出:\n" + "\n---\n".join(d['generated_text'] for d in outputs) + "\n\n")
                f.write(f"生成時間: {gen_time:.4f} 秒\n")
                f.write(f"Prompt tokens: {prompt_token_count} | Output tokens: {output_token_count}\n")

                if (idx + 1) % 10 == 0:
                    logger.info(f"{self.dataset_name} 進度: {idx+1}/{len(dataset)}")

        # 統計 GPU 記憶體與 throughput
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_memory = None

        # throughput: (prompt + output) tokens / generation_time
        total_tokens = total_prompt_tokens + total_generated_tokens
        throughput = total_tokens / total_generation_time if total_generation_time > 0 else 0.0

        logger.info(f"🔹 平均 Throughput: {throughput:.2f} tokens/sec")
        if peak_memory:
            logger.info(f"🔹 峰值 GPU 記憶體使用量: {peak_memory:.2f} MB")

        # 使用 HuggingFace code_eval 驗證
        k_list = self.config.pass_k or [1]
        formatted_preds = self._build_predictions_instruct(resps, docs)
        logger.info(f"🔹 開始 code_eval (k={k_list})")
        
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        hf_code_eval = hf_evaluate.load("code_eval")
        metrics_dict, details_dict = hf_code_eval.compute(
            references=references,
            predictions=formatted_preds,
            k=k_list,
        )

        pass_at_1 = metrics_dict.get("pass@1", 0.0)
        logger.info(f"✅ 評估完成！Pass@1: {pass_at_1:.4f}")
  
        for item in self.results:
            idx = item["index"]
            run_info = details_dict.get(idx, [])
            if run_info:
                _, info = run_info[0]
                item.update({
                    "passed": info["passed"],
                    "completion_id": info["completion_id"],
                })
            else:
                raise ValueError(f"找不到 idx={idx} 的詳細結果")

        return {
            **metrics_dict,
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
