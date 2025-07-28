#!/usr/bin/env python3
"""Train SceneDiffuser++ with proper diffusion"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from core.model import SceneConfig
from core.diffusion_model import ImprovedSceneDiffuser, DiffusionConfig
from core.synthetic_data_loader import SyntheticWOMDDataset

# Configure
scene_config = SceneConfig()
scene_config.batch_size = 2  # Small for testing
scene_config.num_agents = 32  # Reduced for faster training
scene_config.timesteps = 50  # Reduced for testing

diffusion_config = DiffusionConfig(
    num_diffusion_steps=1000,
    beta_schedule="cosine",
    prediction_type="v_prediction"
)

# Create model
model = ImprovedSceneDiffuser(scene_config, diffusion_config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create dataset
dataset = SyntheticWOMDDataset()
dataloader = DataLoader(dataset, batch_size=scene_config.batch_size, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training
losses = []
print("\nStarting training...")

for epoch in range(5):  # Just 5 epochs for testing
    epoch_losses = []
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        # Forward pass
        loss = model.training_step(batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(avg_loss)
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

# Plot losses
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Diffusion Training Loss')
plt.grid(True)
plt.savefig('diffusion_training_loss.png')
print("\nSaved loss plot to diffusion_training_loss.png")

# Generate a sample
print("\nGenerating sample...")
device = next(model.parameters()).device
with torch.no_grad():
    sample = model.sample(1, device)
    print(f"Generated agents shape: {sample['agents'].shape}")
    print(f"Generated lights shape: {sample['lights'].shape}")
    
    # Check validity
    agent_validity = torch.sigmoid(sample['agents'][0, :, :, 0])
    valid_agents = (agent_validity > 0.5).sum()
    print(f"Valid agents in generated sample: {valid_agents}")

print("\n✅ Diffusion training complete!")
