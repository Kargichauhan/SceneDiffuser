#!/usr/bin/env python3
"""Visualize the trained MPS model results"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import sys
sys.path.append('.')

from fix_mps_training import SimpleMPSDiffusion, FixedDiffusionScheduler
from core.model import SceneConfig
from core.diffusion_model import DiffusionConfig

# Load the trained model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Same config as training
scene_config = SceneConfig()
scene_config.batch_size = 2
scene_config.num_agents = 8
scene_config.timesteps = 30
scene_config.agent_features = 10
scene_config.light_features = 5

diffusion_config = DiffusionConfig(
    num_diffusion_steps=100,
    beta_schedule="cosine",
    prediction_type="v_prediction"
)

# Load model
model = SimpleMPSDiffusion(scene_config, diffusion_config).to(device)
checkpoint = torch.load('checkpoints/mps_model.pt', map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print("✓ Model loaded from checkpoint")

# Generate samples using simplified sampling
print("\nGenerating samples...")

def generate_sample(model, steps=50):
    """Generate a sample using simple DDPM sampling"""
    # Start from noise
    agents = torch.randn(1, scene_config.num_agents, scene_config.timesteps, 
                        scene_config.agent_features).to(device)
    lights = torch.randn(1, 8, scene_config.timesteps, 
                        scene_config.light_features).to(device)
    
    # Simple denoising loop
    with torch.no_grad():
        for t in reversed(range(0, diffusion_config.num_diffusion_steps, 2)):  # Skip steps for speed
            timesteps = torch.tensor([t]).to(device)
            
            # Predict
            pred = model.forward(agents, lights, timesteps)
            
            # Simple denoising step (simplified DDPM)
            noise_level = (t / diffusion_config.num_diffusion_steps)
            agents = agents - 0.02 * pred['agents'] * noise_level
            lights = lights - 0.02 * pred['lights'] * noise_level
            
            # Add small noise except at last step
            if t > 0:
                agents += 0.01 * torch.randn_like(agents) * noise_level
                lights += 0.01 * torch.randn_like(lights) * noise_level
    
    return agents.cpu(), lights.cpu()

# Generate multiple samples
print("Generating 3 different samples...")
samples = []
for i in range(3):
    agents, lights = generate_sample(model)
    samples.append((agents, lights))
    print(f"  Sample {i+1} generated")

# Visualize the samples
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for row, (agents, lights) in enumerate(samples):
    agents = agents[0]  # Remove batch dimension
    lights = lights[0]
    
    # 1. Validity pattern
    ax = axes[row, 0]
    validity = torch.sigmoid(agents[:, :, 0])
    im = ax.imshow(validity.numpy(), aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Agent ID')
    ax.set_title(f'Sample {row+1}: Agent Validity')
    plt.colorbar(im, ax=ax)
    
    # 2. Trajectories
    ax = axes[row, 1]
    for i in range(min(5, scene_config.num_agents)):
        valid_mask = validity[i] > 0.5
        if valid_mask.sum() > 2:  # Need at least 3 points
            x_pos = agents[i, valid_mask, 1].numpy()
            y_pos = agents[i, valid_mask, 2].numpy()
            ax.plot(x_pos, y_pos, '-o', label=f'Agent {i}', markersize=3)
    
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title(f'Sample {row+1}: Agent Trajectories')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    
    # 3. Scene at specific timestep
    ax = axes[row, 2]
    t = 15  # Middle timestep
    
    # Draw simple road
    ax.axhline(y=0, color='gray', linewidth=20, alpha=0.3)
    ax.axvline(x=0, color='gray', linewidth=20, alpha=0.3)
    
    # Draw agents
    for i in range(scene_config.num_agents):
        if validity[i, t] > 0.5:
            x = agents[i, t, 1].item()
            y = agents[i, t, 2].item()
            heading = agents[i, t, 4].item() if agents.shape[2] > 4 else 0
            
            # Simple rectangle for agent
            rect = patches.Rectangle((x-2, y-1), 4, 2, 
                                   angle=np.degrees(heading),
                                   color='blue', alpha=0.6)
            ax.add_patch(rect)
    
    # Draw traffic lights
    light_validity = torch.sigmoid(lights[:, :, 0])
    for i in range(min(4, lights.shape[0])):
        if light_validity[i, t] > 0.5:
            x = lights[i, t, 1].item()
            y = lights[i, t, 2].item()
            state = int(torch.sigmoid(lights[i, t, 4]).item() * 3)  # 0,1,2 for R,Y,G
            colors = ['red', 'yellow', 'green']
            
            circle = patches.Circle((x, y), 0.5, color=colors[state % 3])
            ax.add_patch(circle)
    
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.set_title(f'Sample {row+1}: Scene at t={t}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mps_generated_samples.png', dpi=150)
print("\n✓ Saved visualization to mps_generated_samples.png")

# Analyze generation quality
print("\n=== Generation Statistics ===")
for i, (agents, lights) in enumerate(samples):
    validity = torch.sigmoid(agents[0, :, :, 0])
    valid_count = (validity > 0.5).sum()
    
    # Check if agents have reasonable positions
    valid_positions = agents[0, validity > 0.5, 1:3]
    if len(valid_positions) > 0:
        pos_std = valid_positions.std(dim=0)
        print(f"\nSample {i+1}:")
        print(f"  Valid observations: {valid_count}/{scene_config.num_agents * scene_config.timesteps}")
        print(f"  Position spread (std): X={pos_std[0]:.2f}, Y={pos_std[1]:.2f}")

# Create a simple animation of one sample
print("\nCreating animation...")
agents, lights = samples[0]
agents = agents[0]

fig, ax = plt.subplots(figsize=(8, 8))

def animate(frame):
    ax.clear()
    
    # Draw road
    ax.axhline(y=0, color='gray', linewidth=30, alpha=0.3)
    ax.axvline(x=0, color='gray', linewidth=30, alpha=0.3)
    
    # Draw agents
    validity = torch.sigmoid(agents[:, frame, 0])
    for i in range(scene_config.num_agents):
        if validity[i] > 0.5:
            x = agents[i, frame, 1].item()
            y = agents[i, frame, 2].item()
            
            circle = patches.Circle((x, y), 1, color='blue', alpha=0.7)
            ax.add_patch(circle)
            ax.text(x, y, str(i), ha='center', va='center', color='white', fontsize=8)
    
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.set_title(f'Generated Traffic Scene - Frame {frame}/{scene_config.timesteps}')
    ax.grid(True, alpha=0.3)

# Create animation
anim = FuncAnimation(fig, animate, frames=scene_config.timesteps, interval=200)
anim.save('mps_traffic_animation.gif', writer='pillow')
print("✓ Saved animation to mps_traffic_animation.gif")

print("\n✅ Visualization complete!")
print("\nNext steps:")
print("1. Train longer for better quality")
print("2. Implement better sampling (DDIM)")
print("3. Add conditional generation")
print("4. Scale up to full model")
