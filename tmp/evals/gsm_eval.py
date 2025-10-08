"""
GSM8K Evaluator
基於 evals/eval.ipynb 的評估邏輯封裝成類別,並支援 YAML 配置
"""
import re
import os
import json
import logging
from pathlib import Path
from fractions import Fraction
from typing import Optional, Union, Dict, List, Any
from dataclasses import dataclass, asdict

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("⚠️ 警告: yaml 未安裝，無法使用 YAML 配置。安裝: pip install pyyaml")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, GenerationConfig

# 數字類型定義
Number = Union[int, float]

# ----------------------------
# Logger
# ----------------------------
logger = logging.getLogger("GSM8KEvaluator")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(ch)


# ----------------------------
# 答案解析工具函數
# ----------------------------
def _normalize_text(s: str) -> str:
    """去除多餘空格、逗號、特殊空白符號"""
    if s is None:
        return ""
    s = s.replace("\u00A0", " ")  # 替換不換行空格
    s = s.replace(",", "")        # 去除千分位逗號
    return s.strip()


def _parse_number_str(s: str) -> Optional[Union[Number, str]]:
    """將字串解析成 int 或 float，支援百分比、小數、分數、科學記號等"""
    if s is None:
        return None
    s = _normalize_text(s)
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
        return _parse_number_str(mixed_matches[-1])

    # 嘗試找分數
    frac_matches = re.findall(r'[-+]?\d+\/\d+', s)
    if frac_matches:
        return _parse_number_str(frac_matches[-1])

    # 嘗試找數字 (最後一個)
    num_matches = re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', s)
    if num_matches:
        return _parse_number_str(num_matches[-1])

    return None


def extract_true_answer(answer_str: str) -> Optional[Number]:
    """從 GSM8K 的答案格式取出 '#### number'"""
    if "####" in answer_str:
        parts = answer_str.split("####")
        final = parts[-1].strip().replace(",", "")
        return _parse_number_str(final)
    return None


def extract_predicted_answer(predicted_str: str, last_n_lines: int = 3) -> Optional[Number]:
    """
    從模型輸出中提取數字答案
    優先順序：
      1. '####' 後的數字 (最後一個)
      2. 含有 "final answer/答案" 等關鍵詞的行 (最後一個)
      3. 輸出最後幾行的數字 (最後一個)
      4. 最後一個混合分數/分數/數字
    """
    if not predicted_str:
        return None

    text = predicted_str.strip()
    candidates = []

    # case 1: '#### number'
    if "####" in text:
        after_hashes_all = re.findall(r'####\s*([^\n]+)', text)
        for seg in after_hashes_all:
            parsed = _parse_number_str(seg)
            if parsed is not None:
                candidates.append(("case1", parsed))

    # case 2: 關鍵詞標記的答案
    answer_labels = [
        r'final answer', r'final', r'answer', r'ans', r'solution',
    ]
    lines = text.splitlines()
    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        for label in answer_labels:
            if re.search(rf'(?i)\b{re.escape(label)}\b', line):
                # 嘗試抓 label 後的字
                parts = re.split(rf'(?i)\b{re.escape(label)}\b', line, maxsplit=1)
                candidate_after = parts[1].strip() if len(parts) > 1 else ""
                if candidate_after:
                    parsed = _parse_number_str(candidate_after)
                    if parsed is not None:
                        candidates.append(("case2", parsed))
                # 否則往下找數字
                for j in range(i+1, min(i+4, len(lines))):
                    if lines[j].strip():
                        parsed = _parse_number_str(lines[j])
                        if parsed is not None:
                            candidates.append(("case2", parsed))
                        break

    # case 3: 看最後幾行
    non_empty_lines = [ln for ln in lines if ln.strip()]
    for line in non_empty_lines[-last_n_lines:]:
        parsed = _parse_number_str(line)
        if parsed is not None:
            candidates.append(("case3", parsed))

    # case 4: fallback
    mixed_all = re.findall(r'[-+]?\d+\s+\d+\/\d+', text)
    for val in mixed_all:
        candidates.append(("case4", _parse_number_str(val)))

    frac_all = re.findall(r'[-+]?\d+\/\d+', text)
    for val in frac_all:
        candidates.append(("case4", _parse_number_str(val)))

    num_all = re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', text)
    for val in num_all:
        candidates.append(("case4", _parse_number_str(val)))

    # 依照優先順序選最後一個
    for case in ["case1", "case2", "case3", "case4"]:
        case_candidates = [val for tag, val in candidates if tag == case]
        if case_candidates:
            return case_candidates[-1]

    return None


def answers_match(true_ans, pred_ans, tol: float = 1e-6) -> bool:
    """比較標準答案與模型輸出是否相符"""
    if true_ans is None or pred_ans is None:
        return False

    # case: true_ans 來自 '####'
    if isinstance(true_ans, str) and "####" in true_ans:
        true_ans = _parse_number_str(true_ans.split("####")[-1]) or true_ans

    parsed_true = _parse_number_str(str(true_ans)) if not isinstance(true_ans, (int, float)) else true_ans
    parsed_pred = _parse_number_str(str(pred_ans)) if not isinstance(pred_ans, (int, float)) else pred_ans

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
        s_true = str(true_ans).strip().rstrip('.').lower()
        s_pred = str(pred_ans).strip().rstrip('.').lower()
        return s_true == s_pred
    except:
        return False


# ----------------------------
# 評估配置數據類
# ----------------------------
@dataclass
class EvalConfig:
    """評估配置"""
    model_path: str
    output_dir: str = "output"
    max_new_tokens: int = 1024
    do_sample: bool = False
    batch_size: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    num_samples: Optional[int] = None  # None 表示評估全部
    prompt_type: str = "fewshot"  # 可選: "direct", "fewshot", "cot"
    hf_token: Optional[str] = None
    device_map: str = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    use_safetensors: bool = True


# ----------------------------
# GSM8K Evaluator 類別
# ----------------------------
class GSM8KEvaluator:
    """
    GSM8K 數學推理基準評估器
    
    支援：
    - 從 YAML 配置載入
    - Few-shot prompting
    - Chain-of-thought prompting
    - 量化模型評估 (GPTQ, AWQ, BNB)
    - 結果輸出與統計
    """
    
    # Few-shot 範例（8-shot）
    FEWSHOT_EXAMPLES = [
        {
            "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "answer": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6"
        },
        {
            "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5"
        },
        {
            "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "answer": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39"
        },
        {
            "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
            "answer": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8"
        },
        {
            "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
            "answer": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9"
        },
        {
            "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            "answer": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29"
        },
        {
            "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            "answer": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33"
        },
        {
            "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
            "answer": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8"
        }
    ]
    
    def __init__(self, config: Union[str, Dict, EvalConfig]):
        """
        初始化評估器
        
        Args:
            config: 配置，可以是：
                - YAML 文件路徑 (str)
                - 配置字典 (Dict)
                - EvalConfig 物件
        """
        if isinstance(config, str):
            # 從 YAML 文件載入
            self.config = self._load_from_yaml(config)
        elif isinstance(config, dict):
            # 從字典載入
            self.config = self._load_from_dict(config)
        elif isinstance(config, EvalConfig):
            self.config = config
        else:
            raise ValueError(f"不支援的配置類型: {type(config)}")
        
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.results = []
        
        logger.info(f"📋 評估配置已載入: {self.config.model_path}")
    
    def _load_from_yaml(self, yaml_path: str) -> EvalConfig:
        """從 YAML 文件載入配置"""
        if not YAML_AVAILABLE:
            raise ImportError("請安裝 pyyaml: pip install pyyaml")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return self._load_from_dict(cfg)
    
    def _load_from_dict(self, cfg: Dict) -> EvalConfig:
        """從字典載入配置"""
        eval_cfg = cfg.get("evaluation", {})
        model_cfg = cfg.get("model", {})
        dataset_cfg = cfg.get("dataset", {})
        
        return EvalConfig(
            model_path=model_cfg.get("name", ""),
            output_dir=eval_cfg.get("results_dir", "output"),
            max_new_tokens=dataset_cfg.get("max_new_tokens", 1024),
            do_sample=dataset_cfg.get("do_sample", False),
            batch_size=dataset_cfg.get("batch_size", 1),
            temperature=dataset_cfg.get("temperature", 0.0),
            top_p=dataset_cfg.get("top_p", 1.0),
            num_samples=dataset_cfg.get("num_samples"),
            prompt_type=eval_cfg.get("prompt_type", "fewshot"),
            hf_token=cfg.get("hf_token"),
            device_map=model_cfg.get("device_map", "cuda:0"),
            torch_dtype=model_cfg.get("torch_dtype", "float16"),
            trust_remote_code=model_cfg.get("trust_remote_code", True),
            use_safetensors=model_cfg.get("use_safetensors", True),
        )
    
    def load_model(self, quantization_type: Optional[str] = None):
        """
        載入模型和 tokenizer
        
        Args:
            quantization_type: 量化類型，可選 "gptq", "awq", "bnb", None (普通模型)
        """
        logger.info(f"🔹 載入 tokenizer: {self.config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            token=self.config.hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 根據量化類型載入模型
        if quantization_type == "gptq":
            self._load_gptq_model()
        elif quantization_type == "awq":
            self._load_awq_model()
        else:
            self._load_normal_model()

        self._setup_generation_pipeline()
    
    def _load_normal_model(self):
        """載入普通（未量化）模型"""
        logger.info("🔹 載入普通模型...")
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            device_map=self.config.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
            token=self.config.hf_token
        )
    
    def _load_gptq_model(self):
        """載入 GPTQ 量化模型（優化版本，處理 vocab 不匹配和 CUDA 錯誤）"""
        logger.info("🔹 載入 GPTQ 量化模型...")
        
        try:
            from auto_gptq import AutoGPTQForCausalLM
        except ImportError:
            raise ImportError("請安裝 auto-gptq: pip install auto-gptq")
        
        # 載入策略：嘗試多種方法
        load_successful = False
        last_error = None
        
        # 策略 1: 使用 from_quantized (推薦方式)
        logger.info("📦 策略 1: 使用 from_quantized 載入...")
        try:
            self.model = AutoGPTQForCausalLM.from_quantized(
                self.config.model_path,
                device=self.config.device_map if torch.cuda.is_available() else "cpu",
                use_safetensors=self.config.use_safetensors,
                trust_remote_code=self.config.trust_remote_code,
                use_triton=False,
                warmup_triton=False,
                disable_exllama=True,  # 禁用可能有問題的加速
                disable_exllamav2=True,
            )
            
            load_successful = True
            logger.info(f"✅ 策略 1 成功! vocab_size={self.model.config.vocab_size}")
            
        except Exception as e:
            last_error = e
            logger.warning(f"❌ 策略 1 失敗: {e}")
        
        # 策略 2: 禁用所有加速選項
        if not load_successful:
            logger.info("📦 策略 2: 禁用所有加速選項...")
            try:
                self.model = AutoGPTQForCausalLM.from_quantized(
                    self.config.model_path,
                    device_map=self.config.device_map if torch.cuda.is_available() else "cpu",
                    use_safetensors=self.config.use_safetensors,
                    trust_remote_code=self.config.trust_remote_code,
                    use_triton=False,
                    inject_fused_attention=False,
                    inject_fused_mlp=False,
                    disable_exllama=True,
                    disable_exllamav2=True,
                )
                
                load_successful = True
                logger.info(f"✅ 策略 2 成功!")
                
            except Exception as e:
                last_error = e
                logger.warning(f"❌ 策略 2 失敗: {e}")
        
        # 如果所有策略都失敗
        if not load_successful:
            error_msg = f"所有載入策略都失敗。最後錯誤: {last_error}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # 最終驗證
        logger.info("🔍 驗證模型配置...")
        logger.info(f"   Tokenizer vocab size: {len(self.tokenizer)}")
        logger.info(f"   Model vocab size: {self.model.config.vocab_size}")
        logger.info(f"   Pad token ID: {self.model.config.pad_token_id}")
        logger.info(f"   EOS token ID: {self.model.config.eos_token_id}")
        logger.info(f"   Device: {self.model.device}")

    
    def _load_awq_model(self):
        """載入 AWQ 量化模型（優化版本）"""
        logger.info("🔹 載入 AWQ 量化模型...")
        
        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            raise ImportError("請安裝 autoawq: pip install autoawq")
        
        # 檢查量化配置
        config_path = Path(self.config.model_path) / "config.json"
        if config_path.exists():
            logger.info(f"📋 找到模型配置: {config_path}")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                if "quantization_config" in config_dict:
                    logger.info(f"   量化參數: {config_dict['quantization_config']}")
        
        load_successful = False
        last_error = None
        
        # 策略 1: 標準載入
        logger.info("📦 策略 1: 標準 AWQ 載入...")
        try:
            self.model = AutoAWQForCausalLM.from_quantized(
                self.config.model_path,
                fuse_layers=True,
                trust_remote_code=self.config.trust_remote_code,
                safetensors=self.config.use_safetensors,
            )
            self.model.device = self.config.device_map if torch.cuda.is_available() else "cpu"
            
            load_successful = True
            logger.info(f"✅ 策略 1 成功! vocab_size={self.model.config.vocab_size}")
            
        except Exception as e:
            last_error = e
            logger.warning(f"❌ 策略 1 失敗: {e}")
        
        # 策略 2: 禁用 fuse_layers
        if not load_successful:
            logger.info("📦 策略 2: 禁用 layer fusion...")
            try:
                self.model = AutoAWQForCausalLM.from_quantized(
                    self.config.model_path,
                    fuse_layers=False,  # 禁用可能有問題的優化
                    trust_remote_code=self.config.trust_remote_code,
                    safetensors=self.config.use_safetensors,
                )
                
                load_successful = True
                logger.info(f"✅ 策略 2 成功!")
                
            except Exception as e:
                last_error = e
                logger.warning(f"❌ 策略 2 失敗: {e}")
        
        # 如果所有策略都失敗
        if not load_successful:
            error_msg = f"AWQ 模型載入失敗。最後錯誤: {last_error}"
            logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        # 最終驗證
        logger.info("🔍 驗證模型配置...")
        logger.info(f"   Tokenizer vocab size: {len(self.tokenizer)}")
        logger.info(f"   Model vocab size: {self.model.config.vocab_size}")
        logger.info(f"   Pad token ID: {self.model.config.pad_token_id}")
        logger.info(f"   EOS token ID: {self.model.config.eos_token_id}")
        logger.info(f"   Device: {self.model.device}")

    def _setup_generation_pipeline(self):
        """安全創建 TextGenerationPipeline（支援 AWQ / GPTQ / BNB）"""
        logger.info("🔹 創建生成 pipeline...")

        if not hasattr(self.model, "can_generate"):
            self.model.can_generate = lambda: True

        gen_cfg = getattr(self.model, "generation_config", None)

        if gen_cfg is None:
            # 若模型沒這個屬性，自己建一個
            gen_cfg = GenerationConfig()
            self.model.generation_config = gen_cfg

        self.generator = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
        )

        logger.info("✅ 模型生成管線初始化完成")
    
    def build_prompt(self, question: str) -> str:
        """
        根據配置的 prompt_type 構建提示詞
        
        Args:
            question: 問題文本
            
        Returns:
            構建好的提示詞
        """
        if self.config.prompt_type == "direct":
            return self._build_direct_prompt(question)
        elif self.config.prompt_type == "fewshot":
            return self._build_fewshot_prompt(question)
        elif self.config.prompt_type == "cot":
            return self._build_cot_prompt(question)
        else:
            raise ValueError(f"不支援的 prompt_type: {self.config.prompt_type}")
    
    def _build_direct_prompt(self, question: str) -> str:
        try:
            messages = [
                {"role": "user", "content": f"Given the following problem, reason and give a final answer.\nProblem: {question}\nYour response should end with 'The final answer is [answer]'."}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            logger.error(f"❌ 構建 apply_chat_template 失敗: {e}")
            logger.info(" 🔹 回退到簡單的 direct prompt")
            return f"Given the following problem, reason and give a final answer.\nProblem: {question}\nYour response should end with 'The final answer is [answer]'."

    
    def _build_fewshot_prompt(self, question: str) -> str:
        try:
            messages = []
            for ex in self.FEWSHOT_EXAMPLES:
                messages.append({"role": "user", "content": f"Given the following problem, reason and give a final answer.\nProblem: {ex['question']}\nYour response should end with 'The final answer is [answer]'."})
                messages.append({"role": "assistant", "content": ex["answer"]})

            # 最後加入要解的題目
            messages.append({"role": "user", "content": f"Given the following problem, reason and give a final answer.\nProblem: {question}\nYour response should end with 'The final answer is [answer]'."})

            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            logger.error(f"❌ 構建 apply_chat_template 失敗: {e}")
            logger.info(" 🔹 回退到簡單的 few-shot prompt")
            # 回退到簡單的 few-shot prompt
            prompt = ""
            for ex in self.FEWSHOT_EXAMPLES:
                prompt += f"Given the following problem, reason and give a final answer.\nProblem: {ex['question']}\nYour response should end with 'The final answer is [answer]'.\n{ex['answer']}\n\n"
            prompt += f"Given the following problem, reason and give a final answer.\nProblem: {question}\nYour response should end with 'The final answer is [answer]'.\n"
            return prompt
    
    def _build_cot_prompt(self, question: str) -> str:
        try:
            messages = [
                {"role": "user", "content": f"Question: {question}\n\nLet's solve this step-by-step:"}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            logger.error(f"❌ 構建 apply_chat_template 失敗: {e}")
            logger.info("🔹 回退到簡單的 chain-of-thought prompt")
            # 回退到簡單的 chain-of-thought prompt
            return f"Question: {question}\n\nLet's solve this step-by-step:\nAnswer:"

    
    def evaluate(self, dataset_name: str = "openai/gsm8k", split: str = "test") -> Dict[str, Any]:
        """
        執行評估
        
        Args:
            dataset_name: 資料集名稱
            split: 資料集分割（train, test）
            
        Returns:
            評估結果字典
        """
        if self.generator is None:
            raise RuntimeError("請先呼叫 load_model() 載入模型")
        
        logger.info(f"🔹 載入資料集: {dataset_name} ({split})")
        dataset = load_dataset(dataset_name, "main", split=split)
        
        if self.config.num_samples:
            dataset = dataset.select(range(min(self.config.num_samples, len(dataset))))
            logger.info(f"📊 評估樣本數: {len(dataset)}")
        else:
            logger.info(f"📊 評估全部樣本: {len(dataset)}")
        
        total, correct = 0, 0
        self.results = []
        
        # 確保輸出目錄存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        output_file = os.path.join(
            self.config.output_dir,
            f"{Path(self.config.model_path).name}_{self.config.prompt_type}.txt"
        )
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature if self.config.do_sample else None,
            "top_p": self.config.top_p if self.config.do_sample else None,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "use_cache": True,
            "return_full_text": False,
        }
        logger.info(f"🔹 生成參數: {generation_kwargs}")
        prompt_format = self.build_prompt("{{question}}")  # 預留位置
        prompts = [prompt_format.replace("{{question}}", s["question"]) for s in dataset]
        true_answers = [extract_true_answer(s["answer"]) for s in dataset]
        batch_size = self.config.batch_size
        logger.info(f"🔹 開始評估...")
        logger.info(f"   使用 batch size: {batch_size}")

        with open(output_file, "w", encoding="utf-8") as f:
            if batch_size == 1:
                for idx, sample in enumerate(dataset):
                    total += 1
                    question = sample["question"]
                    true_ans = extract_true_answer(sample["answer"])
                    
                    # 構建提示詞
                    prompt = prompt_format.replace("{{question}}", question)
                    
                    # 生成回答
                    output = self.generator(
                        prompt,
                        **generation_kwargs,
                    )[0]["generated_text"]
                    
                    # 提取預測答案
                    pred_ans = extract_predicted_answer(output)
                    is_correct = answers_match(true_ans, pred_ans)
                    
                    if is_correct:
                        correct += 1
                    
                    acc = correct / total
                    
                    # 記錄結果
                    result = {
                        "index": idx,
                        "question": question,
                        "true_answer": true_ans,
                        "predicted_answer": pred_ans,
                        "is_correct": is_correct,
                        "current_accuracy": acc,
                        "output": output
                    }
                    self.results.append(result)
                    
                    # 寫入日誌
                    log = (
                        f"[{idx+1}/{len(dataset)}] Question: {question}\n"
                        f"Output: {output}\n"
                        f"True Answer: {true_ans}\n"
                        f"Predicted Answer: {pred_ans}\n"
                        f"Correct: {is_correct}\n"
                        f"Current Accuracy: {acc:.4f}\n"
                        f"{'-'*80}\n"
                    )
                    f.write(log)
                    
                    # 每 10 題報告一次進度
                    if (idx + 1) % 10 == 0:
                        logger.info(f"進度: {idx+1}/{len(dataset)} | 當前準確率: {acc:.4f}")
            else:
                for start in range(0, len(prompts), batch_size):
                    end = start + batch_size
                    batch_prompts = prompts[start:end]
                    batch_true_answers = true_answers[start:end]

                    outputs = self.generator(
                        batch_prompts,
                        **generation_kwargs,
                    )

                    for i, (output, true_ans) in enumerate(zip(outputs, batch_true_answers)):
                        idx = start + i
                        question = dataset[idx]["question"]
                        output_text = output[0]["generated_text"]
                        pred_ans = extract_predicted_answer(output_text)
                        is_correct = answers_match(true_ans, pred_ans)
                        
                        total += 1
                        correct += int(is_correct)
                        acc = correct / total

                        # 記錄結果
                        result = {
                            "index": idx,
                            "question": question,
                            "true_answer": true_ans,
                            "predicted_answer": pred_ans,
                            "is_correct": is_correct,
                            "current_accuracy": acc,
                            "output": output_text
                        }
                        self.results.append(result)

                        log = (
                            f"[{idx+1}/{len(dataset)}] Question: {question}\n"
                            f"Output: {output_text}\n"
                            f"True Answer: {true_ans}\n"
                            f"Predicted Answer: {pred_ans}\n"
                            f"Correct: {is_correct}\n"
                            f"Current Accuracy: {acc:.4f}\n"
                            f"{'-'*80}\n"
                        )
                        f.write(log)

                    logger.info(f"進度: {end}/{len(dataset)} | 當前準確率: {acc:.4f}")

        final_accuracy = correct / total
        logger.info(f"✅ 評估完成！最終準確率: {final_accuracy:.4f} ({correct}/{total})")
        logger.info(f"📄 結果已儲存至: {output_file}")
        
        return {
            "accuracy": final_accuracy,
            "correct": correct,
            "total": total,
            "model_path": self.config.model_path,
            "prompt_type": self.config.prompt_type,
            "output_file": output_file,
            "results": self.results
        }
    
    def save_results(self, results: Dict[str, Any], output_path: Optional[str] = None):
        """
        儲存評估結果為 JSON
        
        Args:
            results: 評估結果
            output_path: 輸出路徑，若為 None 則使用預設路徑
        """
        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir,
                f"{Path(self.config.model_path).name}_{self.config.prompt_type}_results.json"
            )
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 結果已儲存至: {output_path}")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "GSM8KEvaluator":
        """從 YAML 文件創建評估器（類別方法）"""
        return cls(yaml_path)
