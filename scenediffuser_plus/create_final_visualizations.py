#!/usr/bin/env python3
"""Create final visualizations and animations with the trained model"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
sys.path.append('.')

from fix_mps_training import SimpleMPSDiffusion
from core.model import SceneConfig
from core.diffusion_model import DiffusionConfig

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the best model
scene_config = SceneConfig()
scene_config.num_agents = 8
scene_config.timesteps = 30
diffusion_config = DiffusionConfig(num_diffusion_steps=100)

model = SimpleMPSDiffusion(scene_config, diffusion_config).to(device)
model.load_state_dict(torch.load('checkpoints/mps_model_epoch20.pt', map_location=device))
model.eval()
print("✓ Loaded trained model")

# Better sampling function
@torch.no_grad()
def sample_traffic_scene(model, num_steps=100):
    """Generate a traffic scene with proper denoising"""
    agents = torch.randn(1, 8, 30, 10).to(device)
    lights = torch.randn(1, 8, 30, 5).to(device)
    
    # Denoising loop
    for t in reversed(range(0, num_steps)):
        timesteps = torch.tensor([t]).to(device)
        
        # Model prediction
        pred = model.forward(agents, lights, timesteps)
        
        # DDPM denoising step
        alpha = 1 - (t / num_steps)
        beta = 1 - alpha
        
        # Remove predicted noise
        agents = agents - beta * pred['agents'] * 0.02
        lights = lights - beta * pred['lights'] * 0.02
        
        # Add noise for non-final steps
        if t > 0:
            noise_level = np.sqrt(beta) * 0.5
            agents += torch.randn_like(agents) * noise_level
            lights += torch.randn_like(lights) * noise_level
    
    return agents[0].cpu(), lights[0].cpu()

print("\nGenerating high-quality samples...")

# Generate multiple scenes
scenes = []
for i in range(3):
    print(f"  Generating scene {i+1}/3...")
    agents, lights = sample_traffic_scene(model)
    scenes.append((agents, lights))

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))

for scene_idx, (agents, lights) in enumerate(scenes):
    # Process validity
    agent_validity = torch.sigmoid(agents[:, :, 0])
    light_validity = torch.sigmoid(lights[:, :, 0])
    
    # 1. Validity heatmap
    ax = plt.subplot(3, 4, scene_idx * 4 + 1)
    im = ax.imshow(agent_validity.numpy(), aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title(f'Scene {scene_idx+1}: Agent Validity')
    ax.set_xlabel('Time')
    ax.set_ylabel('Agent ID')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 2. Trajectories with spawning/despawning
    ax = plt.subplot(3, 4, scene_idx * 4 + 2)
    colors = plt.cm.tab10(np.arange(8))
    
    for i in range(8):
        valid_mask = agent_validity[i] > 0.5
        if valid_mask.sum() > 2:
            valid_indices = torch.where(valid_mask)[0]
            x = agents[i, valid_mask, 1].numpy()
            y = agents[i, valid_mask, 2].numpy()
            
            # Plot with spawn/despawn markers
            ax.plot(x, y, '-', color=colors[i], alpha=0.7, linewidth=2)
            ax.plot(x[0], y[0], 'o', color=colors[i], markersize=8, label=f'Agent {i}')  # Spawn
            ax.plot(x[-1], y[-1], 's', color=colors[i], markersize=8)  # Despawn
    
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title(f'Scene {scene_idx+1}: Trajectories')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    
    # 3. Traffic light states over time
    ax = plt.subplot(3, 4, scene_idx * 4 + 3)
    light_states = torch.sigmoid(lights[:4, :, 4]).numpy()
    im = ax.imshow(light_states, aspect='auto', cmap='RdYlGn')
    ax.set_title(f'Scene {scene_idx+1}: Traffic Light States')
    ax.set_xlabel('Time')
    ax.set_ylabel('Light ID')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 4. Scene snapshot at multiple times
    ax = plt.subplot(3, 4, scene_idx * 4 + 4)
    
    # Draw intersection
    ax.axhline(y=0, color='gray', linewidth=40, alpha=0.3)
    ax.axvline(x=0, color='gray', linewidth=40, alpha=0.3)
    
    # Draw lane markings
    for lane in [-5, 0, 5]:
        ax.axhline(y=lane, color='white', linewidth=1, linestyle='--', alpha=0.5)
        ax.axvline(x=lane, color='white', linewidth=1, linestyle='--', alpha=0.5)
    
    # Draw agents at different times with transparency
    times_to_show = [5, 15, 25]
    alphas = [0.3, 0.6, 1.0]
    
    for t_idx, (t, alpha) in enumerate(zip(times_to_show, alphas)):
        for i in range(8):
            if agent_validity[i, t] > 0.5:
                x = agents[i, t, 1].item()
                y = agents[i, t, 2].item()
                
                rect = patches.Rectangle(
                    (x-2, y-1), 4, 2,
                    color=colors[i],
                    alpha=alpha * 0.7,
                    edgecolor='black',
                    linewidth=1
                )
                ax.add_patch(rect)
                
                if t_idx == len(times_to_show) - 1:  # Label only the last time
                    ax.text(x, y, str(i), ha='center', va='center', fontsize=8)
    
    # Draw traffic lights
    light_positions = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    for i in range(min(4, lights.shape[0])):
        if light_validity[i, 15] > 0.5:
            x, y = light_positions[i]
            state = torch.sigmoid(lights[i, 15, 4]).item()
            color = ['red', 'yellow', 'green'][int(state * 2.99)]
            
            circle = patches.Circle((x, y), 1.5, color=color, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
    
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')
    ax.set_title(f'Scene {scene_idx+1}: Multi-time Overlay')
    ax.text(0.02, 0.98, 'Time: light→dark', transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('final_generation_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved final results visualization")

# Create animation for the best scene
print("\nCreating animation...")
best_scene_idx = 0
agents, lights = scenes[best_scene_idx]
agent_validity = torch.sigmoid(agents[:, :, 0])

fig, ax = plt.subplots(figsize=(10, 10))

def animate(frame):
    ax.clear()
    
    # Draw intersection
    ax.axhline(y=0, color='gray', linewidth=40, alpha=0.3)
    ax.axvline(x=0, color='gray', linewidth=40, alpha=0.3)
    
    # Lane markings
    for lane in [-5, 0, 5]:
        ax.axhline(y=lane, color='white', linewidth=1, linestyle='--', alpha=0.5)
        ax.axvline(x=lane, color='white', linewidth=1, linestyle='--', alpha=0.5)
    
    # Draw agents
    colors = plt.cm.tab10(np.arange(8))
    active_agents = 0
    
    for i in range(8):
        if agent_validity[i, frame] > 0.5:
            x = agents[i, frame, 1].item()
            y = agents[i, frame, 2].item()
            
            # Car rectangle
            rect = patches.Rectangle(
                (x-2, y-1), 4, 2,
                color=colors[i],
                edgecolor='black',
                linewidth=2,
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Direction indicator
            heading = agents[i, frame, 4].item() if agents.shape[2] > 4 else 0
            dx = 2 * np.cos(heading)
            dy = 2 * np.sin(heading)
            ax.arrow(x, y, dx, dy, head_width=0.5, head_length=0.3, fc='white', ec='white')
            
            # Agent ID
            ax.text(x, y, str(i), ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
            
            active_agents += 1
    
    # Draw traffic lights
    light_positions = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    light_validity = torch.sigmoid(lights[:, frame, 0])
    
    for i in range(min(4, lights.shape[0])):
        if light_validity[i] > 0.5:
            x, y = light_positions[i]
            state = torch.sigmoid(lights[i, frame, 4]).item()
            color = ['red', 'yellow', 'green'][int(state * 2.99)]
            
            # Traffic light circle
            circle = patches.Circle((x, y), 1.5, color=color, 
                                  edgecolor='black', linewidth=3)
            ax.add_patch(circle)
    
    # Set view
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Title and info
    ax.set_title(f'SceneDiffuser++ Generated Traffic - Frame {frame+1}/30', fontsize=16)
    ax.text(0.02, 0.98, f'Active Agents: {active_agents}', 
           transform=ax.transAxes, va='top', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Add timestamp
    ax.text(0.98, 0.98, f't = {frame/10:.1f}s', 
           transform=ax.transAxes, va='top', ha='right', fontsize=12,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Create and save animation
anim = FuncAnimation(fig, animate, frames=30, interval=200, repeat=True)
writer = PillowWriter(fps=5)
anim.save('scenediffuser_traffic.gif', writer=writer)
print("✓ Saved animation as scenediffuser_traffic.gif")

# Summary statistics
print("\n=== Generation Quality Metrics ===")
for i, (agents, lights) in enumerate(scenes):
    validity = torch.sigmoid(agents[:, :, 0])
    total_valid = (validity > 0.5).sum().item()
    
    # Count spawning/despawning events
    spawns = 0
    despawns = 0
    for agent_id in range(8):
        valid_mask = validity[agent_id] > 0.5
        if valid_mask.any():
            # Find transitions
            diff = valid_mask[1:].float() - valid_mask[:-1].float()
            spawns += (diff > 0).sum().item()
            despawns += (diff < 0).sum().item()
    
    print(f"\nScene {i+1}:")
    print(f"  Total valid observations: {total_valid}/240")
    print(f"  Agent spawning events: {spawns}")
    print(f"  Agent despawning events: {despawns}")
    print(f"  Validity rate: {total_valid/240:.1%}")

print("\n✅ All visualizations complete!")
print("\nGenerated files:")
print("  - final_generation_results.png (comprehensive results)")
print("  - scenediffuser_traffic.gif (animated traffic scene)")
print("  - extended_training_curve.png (training progress)")
print("  - improved_generation.png (sample trajectories)")
