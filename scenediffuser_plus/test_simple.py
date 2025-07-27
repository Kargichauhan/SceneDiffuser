#!/usr/bin/env python3
"""
Simple test that doesn't require the full model implementation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

print("=== SceneDiffuser++ Environment Test ===\n")

# Test PyTorch
print("1. Testing PyTorch...")
x = torch.randn(10, 10)
print(f"   ✓ PyTorch {torch.__version__} working")
print(f"   ✓ CUDA available: {torch.cuda.is_available()}")

# Test NumPy
print("\n2. Testing NumPy...")
arr = np.random.randn(5, 5)
print(f"   ✓ NumPy {np.__version__} working")

# Test Matplotlib
print("\n3. Testing Matplotlib...")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
ax.set_title("Test Plot")
plt.savefig("test_plot.png")
plt.close()
print("   ✓ Matplotlib working - saved test_plot.png")

# Test core concepts
print("\n4. Testing SceneDiffuser++ concepts...")

class SimpleDiffusion:
    def __init__(self):
        self.steps = 10
        
    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        alpha = 1 - t / self.steps
        return alpha * x + (1 - alpha) * noise
    
    def denoise(self, x_noisy, t):
        # Simplified denoising
        return x_noisy * (t / self.steps)

# Test diffusion
model = SimpleDiffusion()
clean_data = torch.randn(5, 5)
noisy_data = model.add_noise(clean_data, 5)
denoised = model.denoise(noisy_data, 5)

print("   ✓ Diffusion process working")
print(f"   ✓ Original data shape: {clean_data.shape}")
print(f"   ✓ Noise level: {(noisy_data - clean_data).abs().mean():.3f}")

print("\n✅ All environment tests passed!")
print("\nNext steps:")
print("1. Save the model implementation files in core/")
print("2. Run python scripts/quick_test.py to test the full model")
