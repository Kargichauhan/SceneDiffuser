#!/usr/bin/env python3
"""Check GPU availability and setup"""

import torch
import subprocess
import platform

print("=== GPU Environment Check ===\n")

# Check PyTorch GPU support
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
else:
    print("\n⚠️  No GPU detected!")
    print("\nOptions:")
    print("1. Use Google Colab (free GPU)")
    print("2. Use cloud services (AWS, GCP, Azure)")
    print("3. Install CUDA locally if you have NVIDIA GPU")
    
    if platform.system() == "Darwin":  # macOS
        print("\n📱 On Mac: Use MPS (Metal Performance Shaders)")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print("Use device = torch.device('mps')")

# Test computation
print("\n🧪 Testing computation...")
device = torch.device('cuda' if torch.cuda.is_available() else 
                     'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Simple benchmark
x = torch.randn(1000, 1000, device=device)
import time
start = time.time()
for _ in range(100):
    x = torch.matmul(x, x)
torch.cuda.synchronize() if torch.cuda.is_available() else None
elapsed = time.time() - start
print(f"100 matrix multiplications: {elapsed:.2f}s")

print("\n✅ GPU check complete!")
