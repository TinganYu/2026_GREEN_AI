import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import yaml
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent 
config_path = BASE_DIR / "config" / "model_config.yaml"

class ConfigurableBaseAgent:
    """Enhanced base agent class with configuration file support"""
    
    def __init__(self, config_path: str = config_path, model_name: str = None, device: str = None):
        self.config = self._load_config(config_path)
        
        # Override config with provided parameters
        if model_name:
            self.config['model']['name'] = model_name
        if device:
            self.config['system']['device'] = device
            
        self.model_name = self.config['model']['name']
        self.device = self._setup_device(self.config['system']['device'])
        
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        
        # Setup logging
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}. Using default config.")
            return self._get_default_config()
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using default config.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'model': {
                'name': 'google/gemma-3-270m', 
                'torch_dtype': 'float16',
                'device_map': 'auto',
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
                'max_position_embeddings': 8192  
            },
            'generation': {
                'max_new_tokens': 512,
                'temperature': 0.7,
                'do_sample': True,
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': 1.1,
                'math': {
                    'temperature': 0.1,
                    'max_new_tokens': 1024,
                    'do_sample': False
                }
            },
            'kv_compression': {
                'enabled': True,
                'compression_ratio': 0.5,
                'compression_method': 'snapkv',
                'compression_frequency': 64
            },
            'system': {
                'device': 'auto',
                'mixed_precision': True
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        if log_config.get('save_logs', False):
            log_dir = log_config.get('log_dir', './logs')
            os.makedirs(log_dir, exist_ok=True)
            
    def _setup_device(self, device: str) -> torch.device:
        """Setup computing device with enhanced logic"""
        if device == "auto":
            if torch.cuda.is_available():
                device_obj = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device_obj = torch.device("mps")
                logger.info("Using MPS (Apple Silicon) device")
            else:
                device_obj = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device_obj = torch.device(device)
            
        return device_obj
    
    def load_model(self):
        """Load tokenizer and model with specific configurations"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            logger.info("Reset CUDA peak memory stats before model load")
        logger.info(f"Loading model: {self.model_name}")
        
        model_config = self.config['model']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=model_config.get('trust_remote_code', True),
            padding_side="left"
        )

        # Handle pad token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 
                # logger.info("Set pad_token to eos_token")
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                logger.info("Added new pad_token: [PAD]")
            
        # Determine torch dtype
        torch_dtype = model_config.get('torch_dtype', 'float16')
        if torch_dtype == 'float16':
            dtype = torch.float16
        elif torch_dtype == 'bfloat16':
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
            
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map=model_config.get('device_map', 'auto'),
                trust_remote_code=model_config.get('trust_remote_code', True),
                low_cpu_mem_usage=model_config.get('low_cpu_mem_usage', True),
                attn_implementation=model_config.get('attn_implementation', 'eager'),
            )
        except Exception as e:
            logger.warning(f"Failed to load with specific config: {e}")
            logger.info("Trying with minimal config...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=dtype,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Resize model embeddings if we added tokens
        if hasattr(self.tokenizer, 'added_tokens_encoder') and len(self.tokenizer.added_tokens_encoder) > 0:
            original_vocab_size = self.model.config.vocab_size
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized embeddings from {original_vocab_size} to {len(self.tokenizer)}")
        
        # Setup generation config
        self._setup_generation_config()
        
        # Enable mixed precision if configured
        if self.config['system'].get('mixed_precision', False):
            self.model = self.model.half()
            
        logger.info(f"Model loaded successfully on device: {self.device}")
        
        # Log model info
        info = self.get_model_info()
        logger.info(f"Model parameters: {info['num_parameters']:,}")
        logger.info(f"Vocab size: {info['vocab_size']:,}")
        
    def _setup_generation_config(self):
        """Setup generation configuration"""
        # gen_config = self.config['generation']
        gen_config = self.config.get('generation', {})
        
        self.generation_config = GenerationConfig(
            max_new_tokens=gen_config.get('max_new_tokens', 512),
            temperature=gen_config.get('temperature', 0.7),
            do_sample=gen_config.get('do_sample', True),
            top_p=gen_config.get('top_p', 0.9),
            top_k=gen_config.get('top_k', 50),
            repetition_penalty=gen_config.get('repetition_penalty', 1.1),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to token ids"""
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)
    
    def decode_tokens(self, token_ids: torch.Tensor) -> str:
        """Decode token ids to text"""
        if token_ids.dim() > 1:
            return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def generate_response(self, 
                         prompt: str, 
                         mode: str = "default",
                         **kwargs) -> str:
        """
        Generate response without compression (baseline)
        
        Args:
            prompt: Input prompt
            mode: Generation mode ("default", "math", "creative")
            **kwargs: Override parameters
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        input_ids = self.encode_text(prompt)
        
        # Get mode-specific config
        # if mode == "math":
        #     gen_params = self.config['generation']['math'].copy()
        # else:
        #     gen_params = {
        #         'max_new_tokens': self.generation_config.max_new_tokens,
        #         'temperature': self.generation_config.temperature,
        #         'do_sample': self.generation_config.do_sample,
        #         'top_p': self.generation_config.top_p,
        #         'top_k': self.generation_config.top_k,
        #         'repetition_penalty': self.generation_config.repetition_penalty
        #     }

        gen_params = {
                'max_new_tokens': self.generation_config.max_new_tokens,
                'temperature': self.generation_config.temperature,
                'do_sample': self.generation_config.do_sample,
                'top_p': self.generation_config.top_p,
                'top_k': self.generation_config.top_k,
                'repetition_penalty': self.generation_config.repetition_penalty
            }
        
        # Override with provided kwargs
        gen_params.update(kwargs)
        
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **gen_params
                )
                
                # Extract only the new tokens
                new_tokens = outputs[0][input_ids.shape[1]:]
                return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise
    
    def clear_cache(self):
        """Clear GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
            
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.debug("MPS cache cleared")
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        if self.model is None:
            return {"status": "Model not loaded"}
        
        # Calculate memory usage if on CUDA
        memory_info = {}
        if torch.cuda.is_available() and self.device.type == 'cuda':
            memory_info = {
                'gpu_memory_allocated': torch.cuda.memory_allocated(self.device) / 1024**3,
                'gpu_memory_cached': torch.cuda.memory_reserved(self.device) / 1024**3,
                'gpu_max_memory': torch.cuda.max_memory_allocated(self.device) / 1024**3,
                'gpu_max_memory_cached': torch.cuda.max_memory_reserved(self.device) / 1024**3
            }
        
        # Get model config info
        config_info = {}
        if hasattr(self.model, 'config'):
            config = self.model.config
            config_info = {
                'hidden_size': getattr(config, 'hidden_size', 'N/A'),
                'num_hidden_layers': getattr(config, 'num_hidden_layers', 'N/A'),
                'num_attention_heads': getattr(config, 'num_attention_heads', 'N/A'),
                'max_position_embeddings': getattr(config, 'max_position_embeddings', 'N/A'),
                'vocab_size': getattr(config, 'vocab_size', 'N/A'),
                'attn_implementation': getattr(config, 'attn_implementation', 'default')
            }
        
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "tokenizer_vocab_size": self.tokenizer.vocab_size if self.tokenizer else 'N/A',
            **config_info,
            **memory_info
        }
    
    def save_config(self, path: str):
        """Save current configuration to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {path}")
    
    def validate_model_compatibility(self) -> Dict:
        """Validate if the model is compatible with compression"""
        if self.model is None:
            return {"compatible": False, "reason": "Model not loaded"}

        try:
            test_input = self.encode_text("Hello")

            with torch.no_grad():
                outputs = self.model.generate(
                    test_input,
                    max_new_tokens=5,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            return {
                "compatible": True,
                "model_type": type(self.model).__name__,
                "architecture": getattr(self.model.config, 'model_type', 'unknown') if hasattr(self.model, 'config') else 'unknown'
            }

        except Exception as e:
            return {
                "compatible": False,
                "reason": f"Validation failed: {str(e)}",
                "error_type": type(e).__name__
            }