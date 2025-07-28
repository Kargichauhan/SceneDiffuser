#!/usr/bin/env python3
"""Test synthetic data pipeline"""

from core.synthetic_data_loader import SyntheticWOMDDataset
from core.model import SceneDiffuserPlusPlus, SceneConfig
import torch

print("Testing synthetic data pipeline...")

# Load data
dataset = SyntheticWOMDDataset()
print(f"Dataset size: {len(dataset)}")

# Get one sample
sample = dataset[0]
print(f"\nSample shapes:")
print(f"  Agents: {sample['agents'].shape}")
print(f"  Lights: {sample['lights'].shape}")
print(f"  Roadgraph: {sample['context']['roadgraph'].shape}")

# Test with model
config = SceneConfig()
model = SceneDiffuserPlusPlus(config)

# Test forward pass
with torch.no_grad():
    batch = {
        'agents': sample['agents'].unsqueeze(0),
        'lights': sample['lights'].unsqueeze(0),
        'context': sample['context']
    }
    loss = model.training_step(batch)
    print(f"\nModel forward pass successful! Loss: {loss.item():.4f}")

print("\n✅ Synthetic data pipeline working!")
