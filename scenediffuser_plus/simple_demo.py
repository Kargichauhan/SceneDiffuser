#!/usr/bin/env python3
"""Simple demo of SceneDiffuser++ concepts"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from core.model import SceneDiffuserPlusPlus, SceneConfig

# Initialize model
print("Initializing SceneDiffuser++...")
config = SceneConfig()
config.num_agents = 16  # Smaller for demo
model = SceneDiffuserPlusPlus(config)

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Generate a scene
print("\nGenerating traffic scene...")
with torch.no_grad():
    scene = model.generate({}, num_rollout_steps=100)

print(f"Generated scene with:")
print(f"  - Agents shape: {scene['agents'].shape}")
print(f"  - Lights shape: {scene['lights'].shape}")

# Visualize
agents = scene['agents'][0]  # Remove batch dimension
validity = agents[:, :, 0] > 0.5
num_valid = validity.sum(dim=0)

plt.figure(figsize=(10, 4))
plt.plot(num_valid.numpy())
plt.xlabel('Timestep')
plt.ylabel('Number of Valid Agents')
plt.title('Agent Count Over Time')
plt.grid(True)
plt.savefig('agent_count.png')
print("\nSaved visualization to agent_count.png")

print("\n✅ Demo complete!")
