#!/usr/bin/env python3
"""Fixed MPS training with proper device handling"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# First, let's fix the scheduler in the diffusion model
import sys
sys.path.append('.')

from core.model import SceneConfig
from core.synthetic_data_loader import SyntheticWOMDDataset

# Import and patch the diffusion model
from core.diffusion_model import DiffusionConfig, DiffusionScheduler, SinusoidalPositionEmbeddings, TransformerBlock

class FixedDiffusionScheduler(DiffusionScheduler):
    """Fixed scheduler that handles MPS properly"""
    
    def __init__(self, config: DiffusionConfig):
        super().__init__(config)
        # Move all tensors to a consistent device when needed
        
    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to data for training - fixed for MPS"""
        device = x_start.device
        
        # Ensure timesteps are on CPU for indexing
        timesteps_cpu = timesteps.cpu()
        
        # Get values and move to correct device
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps_cpu].to(device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps_cpu].to(device)
        
        # Handle shape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(x_start.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy = sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
        return noisy
    
    def get_velocity(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Get velocity for v-prediction parameterization - fixed for MPS"""
        device = x_start.device
        timesteps_cpu = timesteps.cpu()
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps_cpu].to(device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps_cpu].to(device)
        
        while len(sqrt_alpha_prod.shape) < len(x_start.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * x_start
        return velocity

# Simple diffusion model for MPS
class SimpleMPSDiffusion(nn.Module):
    def __init__(self, scene_config, diffusion_config):
        super().__init__()
        self.scene_config = scene_config
        self.diffusion_config = diffusion_config
        self.scheduler = FixedDiffusionScheduler(diffusion_config)
        
        # Simple architecture that works well on MPS
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(128),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        
        self.agent_net = nn.Sequential(
            nn.Linear(scene_config.agent_features, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, scene_config.agent_features),
        )
        
        self.light_net = nn.Sequential(
            nn.Linear(scene_config.light_features, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, scene_config.light_features),
        )
    
    def forward(self, agents, lights, timesteps):
        # Get time embedding
        t_emb = self.time_embed(timesteps.float())
        
        # Process agents
        B, N, T, F = agents.shape
        agents_flat = agents.reshape(B * N * T, F)
        agents_out = self.agent_net(agents_flat)
        agents_out = agents_out.reshape(B, N, T, F)
        
        # Add time embedding influence (simplified)
        time_scale = t_emb.mean(dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        agents_out = agents_out + 0.1 * time_scale
        
        # Process lights similarly
        B, N, T, F = lights.shape
        lights_flat = lights.reshape(B * N * T, F)
        lights_out = self.light_net(lights_flat)
        lights_out = lights_out.reshape(B, N, T, F)
        lights_out = lights_out + 0.1 * time_scale
        
        return {'agents': agents_out, 'lights': lights_out}
    
    def training_step(self, batch):
        agents = batch['agents']
        lights = batch['lights']
        B = agents.shape[0]
        device = agents.device
        
        # Sample timesteps
        timesteps = torch.randint(0, self.scheduler.num_steps, (B,), device=device)
        
        # Add noise
        noise_agents = torch.randn_like(agents)
        noise_lights = torch.randn_like(lights)
        
        noisy_agents = self.scheduler.add_noise(agents, noise_agents, timesteps)
        noisy_lights = self.scheduler.add_noise(lights, noise_lights, timesteps)
        
        # Predict
        pred = self.forward(noisy_agents, noisy_lights, timesteps)
        
        # Compute target based on parameterization
        if self.diffusion_config.prediction_type == "v_prediction":
            target_agents = self.scheduler.get_velocity(agents, noise_agents, timesteps)
            target_lights = self.scheduler.get_velocity(lights, noise_lights, timesteps)
        else:
            target_agents = noise_agents
            target_lights = noise_lights
        
        # Simple MSE loss
        loss = nn.functional.mse_loss(pred['agents'], target_agents)
        loss += nn.functional.mse_loss(pred['lights'], target_lights)
        
        return loss

# Main training
print("🎉 Using MPS-optimized training")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Configure
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

# Create model
model = SimpleMPSDiffusion(scene_config, diffusion_config).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Dataset
dataset = SyntheticWOMDDataset()
dataloader = DataLoader(dataset, batch_size=scene_config.batch_size, shuffle=True)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []

print("\nTraining on MPS...")
for epoch in range(5):
    epoch_losses = []
    
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Just 3 batches per epoch for quick test
            break
            
        # Move to MPS
        batch_mps = {
            'agents': batch['agents'][:, :scene_config.num_agents, :scene_config.timesteps].to(device),
            'lights': batch['lights'][:, :8, :scene_config.timesteps].to(device),
        }
        
        # Train
        loss = model.training_step(batch_mps)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
        print(f"  Epoch {epoch}, Batch {i}: Loss = {loss.item():.4f}")
    
    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
    losses.append(avg_loss)
    print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}\n")

# Plot
plt.figure(figsize=(8, 4))
plt.plot(losses, 'b-o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MPS Diffusion Training')
plt.grid(True)
plt.savefig('mps_fixed_training.png')
print("Saved plot to mps_fixed_training.png")

print("\n✅ MPS training successful!")

# Save model
torch.save(model.state_dict(), 'checkpoints/mps_model.pt')
print("Model saved to checkpoints/mps_model.pt")
