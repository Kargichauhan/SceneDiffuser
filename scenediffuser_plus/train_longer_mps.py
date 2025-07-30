#!/usr/bin/env python3
"""Extended MPS training for better quality"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('.')

from fix_mps_training import SimpleMPSDiffusion, FixedDiffusionScheduler
from core.model import SceneConfig
from core.diffusion_model import DiffusionConfig
from core.synthetic_data_loader import SyntheticWOMDDataset

device = torch.device("mps")
print(f"Training on {device}")

# Config
scene_config = SceneConfig()
scene_config.batch_size = 4  # Larger batch
scene_config.num_agents = 8
scene_config.timesteps = 30

diffusion_config = DiffusionConfig(
    num_diffusion_steps=100,
    beta_schedule="cosine",
    prediction_type="v_prediction"
)

# Load previous model or create new
model = SimpleMPSDiffusion(scene_config, diffusion_config).to(device)
try:
    model.load_state_dict(torch.load('checkpoints/mps_model.pt'))
    print("✓ Loaded previous checkpoint")
except:
    print("Starting fresh")

# Data
dataset = SyntheticWOMDDataset()
dataloader = DataLoader(dataset, batch_size=scene_config.batch_size, shuffle=True)

# Train with better settings
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

losses = []
print("\nExtended training...")

for epoch in range(20):  # More epochs
    epoch_losses = []
    
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/20"):
        # Prepare batch
        batch_mps = {
            'agents': batch['agents'][:, :scene_config.num_agents, :scene_config.timesteps].to(device),
            'lights': batch['lights'][:, :8, :scene_config.timesteps].to(device),
        }
        
        # Train
        loss = model.training_step(batch_mps)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(avg_loss)
    scheduler.step()
    
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
    
    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'checkpoints/mps_model_epoch{epoch+1}.pt')

# Plot training curve
plt.figure(figsize=(10, 5))
plt.plot(losses, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Extended MPS Training')
plt.grid(True, alpha=0.3)
plt.savefig('extended_training_curve.png')
print("\n✓ Saved training curve")

# Generate better samples
print("\nGenerating improved samples...")

def generate_better_sample(model, steps=100):
    """Improved sampling with more steps"""
    B = 1
    agents = torch.randn(B, scene_config.num_agents, scene_config.timesteps, 10).to(device)
    lights = torch.randn(B, 8, scene_config.timesteps, 5).to(device)
    
    with torch.no_grad():
        for t in tqdm(reversed(range(0, steps)), desc="Sampling"):
            timesteps = torch.tensor([t]).to(device)
            pred = model.forward(agents, lights, timesteps)
            
            # Better denoising
            alpha = 1 - (t / steps)
            agents = agents - (1 - alpha) * pred['agents'] * 0.01
            lights = lights - (1 - alpha) * pred['lights'] * 0.01
            
            if t > 0:
                noise_scale = (t / steps) * 0.5
                agents += torch.randn_like(agents) * noise_scale
                lights += torch.randn_like(lights) * noise_scale
    
    return agents.cpu(), lights.cpu()

# Generate and visualize
agents, lights = generate_better_sample(model)

# Create a better visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Validity heatmap
validity = torch.sigmoid(agents[0, :, :, 0])
im = ax1.imshow(validity, aspect='auto', cmap='RdYlGn')
ax1.set_xlabel('Time')
ax1.set_ylabel('Agent ID')
ax1.set_title('Improved Agent Validity Pattern')
plt.colorbar(im, ax=ax1)

# Cleaner trajectories
ax2.set_title('Improved Trajectories')
colors = plt.cm.tab10(range(10))
for i in range(scene_config.num_agents):
    valid = validity[i] > 0.7  # Higher threshold
    if valid.sum() > 5:
        x = agents[0, i, valid, 1].numpy()
        y = agents[0, i, valid, 2].numpy()
        ax2.plot(x, y, 'o-', color=colors[i], label=f'Agent {i}', markersize=4)

ax2.set_xlabel('X position')
ax2.set_ylabel('Y position')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.tight_layout()
plt.savefig('improved_generation.png')
print("✓ Saved improved generation")

print("\n✅ Extended training complete!")
