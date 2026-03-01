#!/usr/bin/env python3
"""
Quick test script to verify kvpress integration works correctly.
Run this before doing full evaluation to catch issues early.
"""

import sys
import os
from pathlib import Path
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.KVpress import KVPressAgent

def test_basic_import():
    """Test 1: Can we import everything?"""
    print("Test 1: Testing imports...")
    try:
        from kvpress import (
            SnapKVPress,
            ObservedAttentionPress,
            KnormPress,
            StreamingLLMPress,
            ExpectedAttentionPress
        )
        print("✓ All kvpress imports successful")
        print("  Available methods:")
        print("    - SnapKVPress")
        print("    - ObservedAttentionPress") 
        print("    - KnormPress")
        print("    - StreamingLLMPress")
        print("    - ExpectedAttentionPress")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("  Install kvpress: pip install git+https://github.com/NVIDIA/kvpress.git")
        return False

def test_agent_creation():
    """Test 2: Can we create the agent?"""
    print("\nTest 2: Creating KVPressAgent...")
    try:
        config_path = Path("lulu_temp/config/model_config.yaml")
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False, None
        
        agent = KVPressAgent(str(config_path))
        print(f"✓ Agent created successfully")
        print(f"  Model: {agent.model_name}")
        print(f"  Method: {agent.compression_method}")
        print(f"  Ratio: {agent.compression_ratio}")
        return True, agent
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        return False, None

def test_model_loading(agent):
    """Test 3: Can we load the model?"""
    print("\nTest 3: Loading model...")
    try:
        agent.load_model()
        print("✓ Model loaded successfully")
        
        # Check if kvpress mixin was applied
        has_mixin = hasattr(agent.model, 'generate')
        print(f"  Has generate method: {has_mixin}")
        
        # Check press object
        has_press = agent.press is not None
        print(f"  Press object initialized: {has_press}")
        if has_press:
            print(f"  Press type: {type(agent.press).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compatibility(agent):
    """Test 4: Is the model compatible?"""
    print("\nTest 4: Checking compatibility...")
    try:
        compat = agent.validate_model_compatibility()
        print(f"  Compatible: {compat['compatible']}")
        print(f"  Model type: {compat.get('model_type', 'N/A')}")
        print(f"  Press method: {compat.get('press_method', 'N/A')}")
        
        if compat['compatible']:
            print("✓ Model is compatible with kvpress")
            return True
        else:
            print(f"✗ Model incompatible: {compat.get('reason', 'Unknown')}")
            return False
    except Exception as e:
        print(f"✗ Compatibility check failed: {e}")
        return False

def test_generation_baseline(agent):
    """Test 5: Can we generate without compression?"""
    print("\nTest 5: Testing baseline generation (no compression)...")
    try:
        prompt = "Question: What is 5 + 3?\nAnswer:"
        response = agent.generate_response(prompt, mode="math", max_new_tokens=20)
        print(f"✓ Baseline generation successful")
        print(f"  Prompt: {prompt}")
        print(f"  Response: {response}")
        return True
    except Exception as e:
        print(f"✗ Baseline generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_compressed(agent):
    """Test 6: Can we generate with compression?"""
    print("\nTest 6: Testing compressed generation...")
    try:
        prompt = "Question: What is 7 + 2?\nAnswer:"
        response = agent.generate_with_compression(
            prompt, 
            mode="math", 
            max_new_tokens=20
        )
        print(f"✓ Compressed generation successful")
        print(f"  Prompt: {prompt}")
        print(f"  Response: {response}")
        
        # Check stats
        stats = agent.get_compression_stats()
        print(f"  Compression stats: {stats}")
        return True
    except Exception as e:
        print(f"✗ Compressed generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def test_synthetic_long_input(agent):
    """Test 6b: Generate with a long synthetic input to trigger compression"""
    print("\nTest 6b: Synthetic long input for compression stats")
    try:
        # Make a long prompt (~2000–3000 tokens)
        agent.set_method("KnormPress")
        long_prompt = "This is a test sentence. " * 300 + " What is 2+2?\nAnswer:" # ~2500 tokens
        response = agent.generate_with_compression(
            long_prompt,
            mode="default",
            max_new_tokens=50
        )
        print(f"✓ Compressed generation successful on synthetic long input")
        print(f"  Response (first 200 chars): {response[:200]}...")
        
        # Print compression stats
        stats = agent.get_compression_stats()
        print(f"  Compression stats: {stats}")
        return True
    except Exception as e:
        print(f"✗ Synthetic long input generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_method_switching(agent):
    """Test 7: Can we switch compression methods?"""
    print("\nTest 7: Testing method switching...")
    methods_to_test = ["snapkv", "knorm", "observed_attention"]
    
    for method in methods_to_test:
        try:
            print(f"  Testing {method}...")
            agent.set_method(method)
            prompt = "Question: What is 10 - 3?\nAnswer:"
            response = agent.generate_with_compression(
                prompt, 
                mode="math", 
                max_new_tokens=20
            )
            print(f"  ✓ {method} works: {response[:50]}...")
        except Exception as e:
            print(f"  ✗ {method} failed: {e}")
            return False
    
    print("✓ All methods work")
    return True

def test_memory_management(agent):
    """Test 8: Check memory management"""
    print("\nTest 8: Testing memory management...")
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
            # Generate a few times
            for i in range(3):
                prompt = f"Question: What is {i} + 1?\nAnswer:"
                _ = agent.generate_with_compression(prompt, mode="math", max_new_tokens=10)
            
            # Clear cache
            agent.clear_cache()
            
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved  = torch.cuda.memory_reserved() / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)

            print(f"[GPU MEM] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Peak Reserved: {peak_reserved:.2f} GB")

            print("✓ Memory management test completed")
            return True
        else:
            print("  ⚠ No CUDA available, skipping GPU memory test")
            return True
    except Exception as e:
        print(f"✗ Memory management test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("KVPress Integration Test Suite")
    print("="*60)
    
    # Track results
    results = []
    
    # Test 1: Imports
    success = test_basic_import()
    results.append(("Import kvpress", success))
    if not success:
        print("\n⚠ Cannot proceed without kvpress installed")
        return
    
    # Test 2: Agent creation
    success, agent = test_agent_creation()
    results.append(("Create agent", success))
    if not success:
        return
    
    # Test 3: Model loading
    success = test_model_loading(agent)
    results.append(("Load model", success))
    if not success:
        return
    
    # Test 4: Compatibility
    success = test_compatibility(agent)
    results.append(("Compatibility check", success))
    if not success:
        print("\n⚠ Model may not be compatible, but continuing tests...")
    
    # Test 5: Baseline generation
    success = test_generation_baseline(agent)
    results.append(("Baseline generation", success))
    
    # Test 6: Compressed generation
    success = test_generation_compressed(agent)
    results.append(("Compressed generation", success))

    # Test 6b: Synthetic long input
    success = test_synthetic_long_input(agent)
    results.append(("Synthetic long input compression", success))
    
    # Test 7: Method switching
    success = test_method_switching(agent)
    results.append(("Method switching", success))
    
    # Test 8: Memory management
    success = test_memory_management(agent)
    results.append(("Memory management", success))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {test_name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your kvpress integration is ready.")
        print("\nNext steps:")
        print("1. Run small evaluation: python eval/run.py")
        print("2. Check results in ./results/")
        print("3. Run full tuning if satisfied")
    else:
        print("\n⚠ Some tests failed. Please fix issues before proceeding.")

if __name__ == "__main__":
    main()