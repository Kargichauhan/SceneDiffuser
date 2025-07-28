#!/usr/bin/env python3
"""Train SceneDiffuser++ using Apple Silicon GPU (MPS)"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from core.model import SceneConfig
from core.diffusion_model import ImprovedSceneDiffuser, DiffusionConfig
from core.synthetic_data_loader import SyntheticWOMDDataset

# Check MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🎉 Using Apple Silicon GPU (MPS)!")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Configure for MPS - moderate size
scene_config = SceneConfig()
scene_config.batch_size = 2
scene_config.num_agents = 16      # Reduced from 128
scene_config.timesteps = 40       # Reduced from 91
scene_config.num_layers = 4       # Reduced from 8
scene_config.hidden_dim = 256     # Reduced from 512
scene_config.num_heads = 4        # Reduced from 8

diffusion_config = DiffusionConfig(
    num_diffusion_steps=500,      # Reduced from 1000
    beta_schedule="cosine",
    prediction_type="v_prediction"
)

# Create model and move to MPS
print("\nCreating model...")
model = ImprovedSceneDiffuser(scene_config, diffusion_config).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Model on device: {next(model.parameters()).device}")

# Create dataset
dataset = SyntheticWOMDDataset()
dataloader = DataLoader(dataset, batch_size=scene_config.batch_size, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training
print("\nStarting MPS-accelerated training...")
losses = []

for epoch in range(5):
    epoch_losses = []
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        # Move batch to MPS
        batch_mps = {
            'agents': batch['agents'].to(device),
            'lights': batch['lights'].to(device),
            'context': {
                'roadgraph': batch['context']['roadgraph'].to(device),
                'scenario_id': batch['context']['scenario_id']
            }
        }
        
        # Forward pass
        loss = model.training_step(batch_mps)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(avg_loss)
    print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

# Plot losses
plt.figure(figsize=(8, 4))
plt.plot(losses, 'b-o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Diffusion Training Loss (MPS Accelerated)')
plt.grid(True)
plt.savefig('mps_training_loss.png')
print("\nSaved loss plot to mps_training_loss.png")

# Generate a sample
print("\nGenerating sample on MPS...")
with torch.no_grad():
    sample = model.sample(1, device)
    
    # Move to CPU for analysis
    agents_cpu = sample['agents'].cpu()
    lights_cpu = sample['lights'].cpu()
    
    # Analyze
    agent_validity = torch.sigmoid(agents_cpu[0, :, :, 0])
    valid_count = (agent_validity > 0.5).sum()
    print(f"Generated {valid_count} valid agent observations")
    
    # Simple visualization
    plt.figure(figsize=(10, 4))
    
    # Plot validity over time
    plt.subplot(1, 2, 1)
    plt.imshow(agent_validity[:10].numpy(), aspect='auto', cmap='RdYlGn')
    plt.xlabel('Time')
    plt.ylabel('Agent ID')
    plt.title('Agent Validity (First 10 agents)')
    plt.colorbar()
    
    # Plot trajectories
    plt.subplot(1, 2, 2)
    for i in range(min(5, scene_config.num_agents)):
        valid_mask = agent_validity[i] > 0.5
        if valid_mask.any():
            x_pos = agents_cpu[0, i, valid_mask, 1]
            y_pos = agents_cpu[0, i, valid_mask, 2]
            plt.plot(x_pos, y_pos, '-', label=f'Agent {i}')
    
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Generated Agent Trajectories')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('mps_generated_sample.png')
    print("Saved sample visualization to mps_generated_sample.png")

# Save model
torch.save(model.state_dict(), 'checkpoints/diffusion_model_mps.pt')
print("\n✅ MPS training complete! Model saved.")
