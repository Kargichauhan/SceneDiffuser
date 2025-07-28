#!/usr/bin/env python3
"""Quick diffusion training test"""

import torch
from torch.utils.data import DataLoader
from core.model import SceneConfig
from core.diffusion_model import ImprovedSceneDiffuser, DiffusionConfig
from core.synthetic_data_loader import SyntheticWOMDDataset

# Smaller config for quick test
scene_config = SceneConfig()
scene_config.batch_size = 5  # Bigger batch
scene_config.num_agents = 16  # Fewer agents
scene_config.timesteps = 30  # Shorter sequences
scene_config.num_layers = 2  # Fewer layers
scene_config.hidden_dim = 128  # Smaller model

diffusion_config = DiffusionConfig(num_diffusion_steps=100)  # Fewer steps

# Create model and data
model = ImprovedSceneDiffuser(scene_config, diffusion_config)
dataset = SyntheticWOMDDataset()
dataloader = DataLoader(dataset, batch_size=scene_config.batch_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f"Quick model: {sum(p.numel() for p in model.parameters()):,} parameters")
print("Training for 2 epochs only...")

# Just 2 epochs
for epoch in range(2):
    losses = []
    for batch in dataloader:
        loss = model.training_step(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch}: Loss = {sum(losses)/len(losses):.4f}")

print("✅ Quick training done!")
