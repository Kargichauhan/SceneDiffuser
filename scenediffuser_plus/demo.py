#!/usr/bin/env python3
"""Simple test to verify installation"""

import torch
import numpy as np
import matplotlib.pyplot as plt

print("✓ PyTorch version:", torch.__version__)
print("✓ NumPy version:", np.__version__)
print("✓ CUDA available:", torch.cuda.is_available())

# Simple test
x = torch.randn(10, 10)
print("✓ Created random tensor with shape:", x.shape)

print("\n✅ All imports successful! Environment is ready.")
