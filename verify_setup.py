"""
Quick test to verify setup works correctly.
Run this after installing dependencies to check if everything is configured properly.
"""

import torch
import sys


def check_pytorch():
    """Check PyTorch installation and GPU"""
    print("✓ PyTorch version:", torch.__version__)
    print("✓ CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("✓ GPU:", torch.cuda.get_device_name(0))
        print("✓ GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
    return True


def check_imports():
    """Check all required imports"""
    try:
        import transformers
        print("✓ Transformers:", transformers.__version__)
    except ImportError:
        print("✗ Transformers not found - run: pip install transformers")
        return False
    
    try:
        import datasets
        print("✓ Datasets:", datasets.__version__)
    except ImportError:
        print("✗ Datasets not found - run: pip install datasets")
        return False
    
    try:
        import accelerate
        print("✓ Accelerate:", accelerate.__version__)
    except ImportError:
        print("✗ Accelerate not found - run: pip install accelerate")
        return False
    
    return True


def check_model():
    """Test model creation"""
    from config import MODEL_CONFIG
    from model import TransformerLM
    
    print("\n✓ Creating model...")
    model = TransformerLM(MODEL_CONFIG)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    
    # Test forward pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    batch_size, seq_len = 2, 256
    input_ids = torch.randint(0, MODEL_CONFIG["vocab_size"], (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"✓ Forward pass successful - output shape: {outputs.shape}")
    return True


def main():
    print("=" * 60)
    print("LLM Setup Verification")
    print("=" * 60 + "\n")
    
    try:
        print("Checking PyTorch...\n")
        check_pytorch()
        
        print("\nChecking imports...\n")
        if not check_imports():
            print("\n✗ Setup incomplete!")
            return False
        
        print("\nChecking model...\n")
        check_model()
        
        print("\n" + "=" * 60)
        print("✓ All checks passed! Ready to train.")
        print("=" * 60)
        print("\nTo start training, run:")
        print("  python train.py")
        print("\nTo test inference, run:")
        print("  python inference.py")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
