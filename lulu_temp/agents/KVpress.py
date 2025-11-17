import torch
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from .base import ConfigurableBaseAgent

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent 
config_path = BASE_DIR / "config" / "model_config.yaml"
# Import kvpress - correct imports based on actual API
try:
    from kvpress import (
        ObservedAttentionPress,
        SnapKVPress,
        StreamingLLMPress,
        KnormPress,
        ExpectedAttentionPress
    )
    KVPRESS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"kvpress not available: {e}")
    KVPRESS_AVAILABLE = False



class KVPressAgent(ConfigurableBaseAgent):
    """KV Cache compression agent using NVIDIA kvpress library"""
    
    def __init__(self, 
                 config_path: str = config_path,
                 **override_params):
        super().__init__(config_path, **override_params)
        
        # KV compression parameters from config
        kv_config = self.config.get('kv_compression', {})
        self.compression_ratio = kv_config.get('compression_ratio', 0.5)
        self.compression_method = kv_config.get('compression_method', 'snapkv')
        self.compression_frequency = kv_config.get('compression_frequency', 4)
        
        # kvpress instance (initialized after model load)
        self.press = None
        
        # Compression statistics
        self.compression_stats = {
            'total_compressions': 0,
            'original_tokens': 0,
            'compressed_tokens': 0,
            'compression_time': 0.0,
            'memory_saved': 0
        }
        
    def load_model(self):
        """Load model and setup kvpress compression"""
        super().load_model()
        
        if not KVPRESS_AVAILABLE:
            raise ImportError("kvpress not installed. Install with: pip install git+https://github.com/NVIDIA/kvpress.git")
        
        # Initialize compression method
        self._setup_kvpress()
        
        # logger.info(f"KVPress compression initialized: {self.compression_method}")
       
    def _setup_kvpress(self, compression_method=None, compression_ratio=None,
                   window_size=None, kernel_size=None):
        """Setup kvpress compression method based on config"""
        kv_config = self.config.get('kv_compression', {})

        method = compression_method or self.compression_method
        compression_ratio = compression_ratio or self.compression_ratio

        if method == 'observed_attention' or method == 'attention':
            self.press = ObservedAttentionPress(compression_ratio=compression_ratio)

        elif method == 'snapkv':
            w = window_size or kv_config.get('methods', {}).get('snapkv', {}).get('window_size', 16)
            k = kernel_size or kv_config.get('methods', {}).get('snapkv', {}).get('kernel_size', 5)
            self.press = SnapKVPress(compression_ratio=compression_ratio, window_size=w, kernel_size=k)

        elif method == 'streaming_llm' or method == 'streaming':
            self.press = StreamingLLMPress(compression_ratio=compression_ratio)

        elif method == 'knorm':
            self.press = KnormPress(compression_ratio=compression_ratio)

        elif method == 'expected_attention':
            self.press = ExpectedAttentionPress(compression_ratio=compression_ratio)

        else:
            logger.warning(f"Unknown method '{method}', defaulting to SnapKV")
            self.press = SnapKVPress(compression_ratio=compression_ratio)

    def generate_with_compression(self, prompt: str, mode: str = "default", max_new_tokens: int = None, **kwargs) -> str:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Generation configuration
        gen_config = self.config.get('generation', {})
        if max_new_tokens is None:
            # if mode == "math":
            #     max_new_tokens = gen_config.get('math', {}).get('max_new_tokens', 1024)
            # else:
            max_new_tokens = gen_config.get('max_new_tokens', 512)

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Generation parameters
        gen_params = {
            'max_new_tokens': max_new_tokens,
            'do_sample': gen_config.get('do_sample', True),
            'temperature': gen_config.get('temperature', 0.7),
            'top_p': gen_config.get('top_p', 0.9),
            'top_k': gen_config.get('top_k', 50),
            'use_cache': True,
            'return_dict_in_generate': True,
            'output_attentions': False,
            'output_hidden_states': False,
        }

        # if mode == "math":
        #     gen_params.update({'do_sample': False, 'num_beams': 1})

        # Update with any extra kwargs
        gen_params.update(kwargs)

        # Initialize compression stats
        self.compression_stats = {
            'total_compressions': 0,
            'original_tokens': input_ids.shape[1],
            'compressed_tokens': 0,
            'compression_time': 0.0,
            'memory_saved': 0
        }

        try:
            # If compression is enabled and press object exists
            # If compression is enabled and press object exists
            if self.press is not None:
                # Check sequence length before applying compression
                estimated_tokens = input_ids.shape[1]  # or include past_key_values if available
                min_tokens_for_compression = getattr(self.press, 'window_size', 128)  # use press window size or default

                if estimated_tokens >= min_tokens_for_compression:
                    # Apply compression normally
                    use_cuda = ((self.device=="cuda:0" or self.device=="cuda:1") and torch.cuda.is_available())
                    start_time = torch.cuda.Event(enable_timing=True) if use_cuda else None
                    end_time   = torch.cuda.Event(enable_timing=True) if use_cuda else None
                    if start_time: start_time.record()

                    with self.press(self.model):
                        outputs = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **gen_params
                        )

                    if end_time: end_time.record()
                    if start_time and end_time:
                        torch.cuda.synchronize()
                        self.compression_stats['compression_time'] = start_time.elapsed_time(end_time) / 1000.0  # seconds

                    # Extract compressed past_key_values for stats
                    past_kvs = getattr(outputs, "past_key_values", None)   
                    if past_kvs is not None:
                        # num_layers is length of the top-level tuple
                        num_layers = len(past_kvs)

                        # helper: find a representative tensor in the nested structure
                        def _find_example_tensor(past):
                            for layer in past:
                                # layer can be a tuple/list of tensors (k, v) or more
                                for part in layer:
                                    if isinstance(part, torch.Tensor) and part.dim() >= 3:
                                        return part
                            return None

                        example = _find_example_tensor(past_kvs)
                        if example is None:
                            # unexpected structure — fall back to a safe default
                            seq_len = None
                            batch_size = input_ids.shape[0]
                            num_heads = None
                            head_dim = None
                        else:
                            # common HF layout: (batch, num_heads, seq_len, head_dim)
                            # but be defensive: check dims and fallback to -2
                            batch_size = example.size(0)
                            if example.dim() >= 4:
                                # most likely layout
                                num_heads = example.size(1)
                                seq_len = example.size(2)
                                head_dim = example.size(3)
                            elif example.dim() == 3:
                                # shape could be (batch, seq_len, head_dim) in some cases
                                # assume seq_len is middle dim
                                num_heads = None
                                seq_len = example.size(1)
                                head_dim = example.size(2)
                            else:
                                seq_len = example.size(-2)
                                num_heads = None
                                head_dim = example.size(-1)

                        # Update stats properly
                        # - compressed_tokens is seq_len (per sequence)
                        # - num_layers returns the number of layers (useful)
                        # - total_compression_calls increments each time compression actually ran
                        self.compression_stats['num_layers'] = num_layers
                        self.compression_stats['batch_size'] = batch_size
                        self.compression_stats['num_heads'] = num_heads
                        self.compression_stats['head_dim'] = head_dim
                        if seq_len is not None:
                            self.compression_stats['compressed_tokens'] = seq_len
                            self.compression_stats['tokens_saved'] = max(0, self.compression_stats['original_tokens'] - seq_len)
                            # keep a separate counter of how many times compression was applied
                            self.compression_stats['total_compressions'] = self.compression_stats.get('total_compressions', 0) + 1
                            if self.compression_stats['original_tokens'] > 0:
                                self.compression_stats['compression_ratio'] = 1-(seq_len / float(self.compression_stats['original_tokens']))
                        else:
                            # couldn't infer seq_len — leave values as None / 0
                            self.compression_stats['compressed_tokens'] = None
                else:
                    # Sequence too short → skip compression
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **gen_params
                    )


            return self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Compressed generation failed: {e}")


    def get_compression_stats(self) -> Dict:
        """Get compression statistics"""
        stats = self.compression_stats.copy()

        # Compute compression metrics if possible
        orig = stats.get('original_tokens', 0)
        comp = stats.get('compressed_tokens', 0)
        if orig > 0:
            stats['compression_ratio'] = 1- (comp / orig)
            stats['tokens_saved'] = orig - comp
        else:
            stats['compression_ratio'] = None
            stats['tokens_saved'] = None

        return stats

    
    def set_compression_ratio(self, ratio: float):
        """Update compression ratio"""
        logger.info(f"Setting compression ratio to {ratio}")
        self.compression_ratio = float(ratio)
        if self.press is not None:
            self.press.compression_ratio = ratio
    
    def set_method(self, method: str):
        """Switch compression method"""
        logger.info(f"Switching compression method to {method}")
        self.compression_method = str(method)
        if self.model is not None:
            self._setup_kvpress()
    
    def validate_model_compatibility(self) -> Dict:
        """Validate kvpress compatibility"""
        if self.model is None:
            return {"compatible": False, "reason": "Model not loaded"}
        
        try:
            # Test basic generation with compression
            test_input = self.encode_text("Test")
            if self.press is None:
                self._setup_kvpress()
            
            with torch.no_grad():
                # Try a simple forward pass
                outputs = self.model(
                    test_input,
                    use_cache=True,
                    return_dict=True
                )
                
                # Try applying press to past_key_values
                if outputs.past_key_values is not None:
                    compressed = self.press(outputs.past_key_values)
                    compression_worked = compressed is not None

                else:
                    compression_worked = False
            
            return {
                "compatible": True,
                "press_method": type(self.press).__name__,
                "model_type": type(self.model).__name__,
                "compression_test": compression_worked
            }
            
        except Exception as e:
            return {
                "compatible": False,
                "reason": f"Validation failed: {str(e)}",
                "error_type": type(e).__name__
            }
        
    def log_kv_size(self):
        """Print current KV size to debug compression."""
        if self.press is None or not hasattr(self.press, 'kv'):
            logger.info("No KV cache present.")
            return

        # Assuming self.press.kv is a tensor of shape [num_tokens, hidden_size]
        kv_shape = self.press.kv.shape
        kv_dtype = self.press.kv.dtype
        num_elements = self.press.kv.numel()
        kv_bytes = self.press.kv.element_size() * num_elements
        logger.info(f"[KVPress] KV tensor shape: {kv_shape}, dtype={kv_dtype}, elements={num_elements}, size={kv_bytes / 1024**2:.2f} MB")
