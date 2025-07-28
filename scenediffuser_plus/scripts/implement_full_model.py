#!/usr/bin/env python3
"""
Script to guide through full model implementation
"""

print("""
=== Full SceneDiffuser++ Implementation Checklist ===

□ 1. Core Components:
   □ Multi-head axial attention transformer
   □ Proper noise scheduling (cosine schedule)
   □ V-parameterization for diffusion
   □ Soft clipping for sparse tensors

□ 2. Advanced Features:
   □ Amortized diffusion for faster sampling
   □ Map encoding with polylines
   □ Agent-roadgraph interaction attention
   □ Traffic light lane association

□ 3. Training Improvements:
   □ Mixed precision training (fp16)
   □ Gradient accumulation
   □ EMA (Exponential Moving Average)
   □ Learning rate scheduling

□ 4. Evaluation Metrics:
   □ Jensen-Shannon divergence
   □ Collision rate checking
   □ Off-road detection
   □ Traffic light violation detection

Which component would you like to implement first?
1. Axial attention transformer
2. Proper diffusion training
3. Map encoding
4. Evaluation pipeline
""")

choice = input("\nEnter choice (1-4): ")
print(f"\nGreat! Let's implement option {choice}")
