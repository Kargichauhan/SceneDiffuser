#!/usr/bin/env python3
"""
Improve SceneDiffuser++ quality based on evaluation results
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('.')

from train_with_realistic_synthetic import EnhancedMPSDiffusion
from core.model import SceneConfig
from core.diffusion_model import DiffusionConfig

class ImprovedSceneDiffuser(EnhancedMPSDiffusion):
    """Enhanced model addressing evaluation weaknesses"""
    
    def __init__(self, scene_config, diffusion_config):
        super().__init__(scene_config, diffusion_config)
        
        # Add spatial attention for collision avoidance
        self.spatial_attention = SpatialAttention(512)
        
        # Add trajectory length predictor
        self.trajectory_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probability of continuing
        )
        
        # Enhanced agent encoder with positional encoding
        self.positional_encoder = PositionalEncoding(512)
        
    def forward(self, agents, lights, timesteps, context=None):
        # Get time embedding
        t_emb = self.time_embed(timesteps.float())
        
        # Enhanced agent encoding
        B, N, T, F = agents.shape
        agent_emb = self.agent_encoder(agents)  # [B, N, T, 512]
        
        # Add positional encoding
        agent_emb = self.positional_encoder(agent_emb)
        
        # Apply spatial attention for collision avoidance
        agent_emb = self.spatial_attention(agent_emb, agents[:, :, :, 1:3])  # Pass positions
        
        # Add time embedding
        agent_emb = agent_emb + t_emb.unsqueeze(1).unsqueeze(1)
        
        # Process lights
        light_emb = self.light_encoder(lights)
        light_emb = light_emb + t_emb.unsqueeze(1).unsqueeze(1)
        
        # Predict outputs
        agent_pred = self.agent_head(agent_emb)
        light_pred = self.light_head(light_emb)
        
        # Predict trajectory continuation probability
        traj_prob = self.trajectory_predictor(agent_emb.mean(dim=2))  # [B, N, 1]
        
        return {
            'agents': agent_pred,
            'lights': light_pred,
            'trajectory_prob': traj_prob
        }
    
    def training_step(self, batch):
        """Enhanced training with collision penalty and trajectory loss"""
        agents = batch['agents']
        lights = batch['lights']
        B = agents.shape[0]
        device = agents.device
        
        timesteps = torch.randint(0, self.scheduler.num_steps, (B,), device=device)
        
        noise_agents = torch.randn_like(agents)
        noise_lights = torch.randn_like(lights)
        
        noisy_agents = self.scheduler.add_noise(agents, noise_agents, timesteps)
        noisy_lights = self.scheduler.add_noise(lights, noise_lights, timesteps)
        
        pred = self.forward(noisy_agents, noisy_lights, timesteps)
        
        # Standard diffusion targets
        if self.diffusion_config.prediction_type == "v_prediction":
            target_agents = self.scheduler.get_velocity(agents, noise_agents, timesteps)
            target_lights = self.scheduler.get_velocity(lights, noise_lights, timesteps)
        else:
            target_agents = noise_agents
            target_lights = noise_lights
        
        # Base diffusion loss
        agent_validity = agents[:, :, :, 0]
        light_validity = lights[:, :, :, 0]
        
        validity_loss = nn.functional.mse_loss(
            pred['agents'][:, :, :, 0], target_agents[:, :, :, 0]
        )
        
        agent_mask = agent_validity > 0.5
        if agent_mask.sum() > 0:
            agent_feature_loss = nn.functional.mse_loss(
                pred['agents'][:, :, :, 1:][agent_mask],
                target_agents[:, :, :, 1:][agent_mask]
            )
        else:
            agent_feature_loss = torch.tensor(0.0, device=device)
        
        light_mask = light_validity > 0.5
        if light_mask.sum() > 0:
            light_feature_loss = nn.functional.mse_loss(
                pred['lights'][:, :, :, 1:][light_mask],
                target_lights[:, :, :, 1:][light_mask]
            )
        else:
            light_feature_loss = torch.tensor(0.0, device=device)
        
        # Collision penalty
        collision_loss = self.compute_collision_penalty(pred['agents'], agent_validity)
        
        # Trajectory length loss
        trajectory_loss = self.compute_trajectory_loss(pred['trajectory_prob'], agents)
        
        total_loss = (validity_loss + agent_feature_loss + light_feature_loss + 
                     0.1 * collision_loss + 0.1 * trajectory_loss)
        
        return total_loss
    
    def compute_collision_penalty(self, pred_agents, validity):
        """Penalize predicted positions that would cause collisions"""
        positions = pred_agents[:, :, :, 1:3]  # x, y positions
        penalty = 0.0
        count = 0
        
        for b in range(positions.shape[0]):
            for t in range(positions.shape[2]):
                valid_mask = validity[b, :, t] > 0.5
                if valid_mask.sum() > 1:
                    valid_positions = positions[b, valid_mask, t]
                    distances = torch.cdist(valid_positions, valid_positions)
                    
                    # Penalty for being too close (< 4m)
                    close_mask = (distances < 4.0) & (distances > 0)
                    if close_mask.any():
                        penalty += (4.0 - distances[close_mask]).mean()
                        count += 1
        
        return penalty / max(count, 1)
    
    def compute_trajectory_loss(self, traj_prob, agents):
        """Encourage longer, more realistic trajectories"""
        validity = agents[:, :, :, 0] > 0.5
        
        # Target: agents should continue if they were valid in previous timestep
        target_prob = torch.zeros_like(traj_prob[:, :, 0])
        
        for b in range(validity.shape[0]):
            for n in range(validity.shape[1]):
                # Count how long this agent should continue
                valid_length = validity[b, n].sum().float()
                target_prob[b, n] = min(valid_length / 91.0, 0.8)  # Cap at 80%
        
        return nn.functional.mse_loss(traj_prob[:, :, 0], target_prob)

class SpatialAttention(nn.Module):
    """Spatial attention for collision avoidance"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, features, positions):
        """
        features: [B, N, T, D]
        positions: [B, N, T, 2] - x, y coordinates
        """
        B, N, T, D = features.shape
        
        # Flatten for attention
        feat_flat = features.reshape(B * T, N, D)
        pos_flat = positions.reshape(B * T, N, 2)
        
        attended = []
        for bt in range(B * T):
            q = self.query(feat_flat[bt])  # [N, D]
            k = self.key(feat_flat[bt])
            v = self.value(feat_flat[bt])
            
            # Compute attention with spatial bias
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Add spatial bias (closer agents get more attention)
            spatial_dist = torch.cdist(pos_flat[bt], pos_flat[bt])
            spatial_bias = -spatial_dist * 0.01  # Negative distance = higher attention for closer
            attn = attn + spatial_bias
            
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            attended.append(out)
        
        attended = torch.stack(attended).reshape(B, N, T, D)
        return attended

class PositionalEncoding(nn.Module):
    """Add positional encoding for better spatial awareness"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        B, N, T, D = x.shape
        
        # Create positional encodings
        pos_enc = torch.zeros_like(x)
        
        # Agent position encoding
        agent_pos = torch.arange(N).float().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        agent_pos = agent_pos.expand(B, N, T, D // 2)
        
        # Time position encoding
        time_pos = torch.arange(T).float().unsqueeze(0).unsqueeze(0).unsqueeze(3)
        time_pos = time_pos.expand(B, N, T, D // 2)
        
        pos_enc[:, :, :, :D//2] = torch.sin(agent_pos * 0.01)
        pos_enc[:, :, :, D//2:] = torch.cos(time_pos * 0.01)
        
        return x + pos_enc.to(x.device)

def train_improved_model():
    """Train the improved model"""
    print("🚀 Training Improved SceneDiffuser++ Model\n")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Config
    scene_config = SceneConfig()
    scene_config.batch_size = 2
    scene_config.num_agents = 32  # Smaller for better training
    scene_config.timesteps = 91
    scene_config.agent_features = 11
    scene_config.light_features = 16
    
    diffusion_config = DiffusionConfig(
        num_diffusion_steps=200,  # Fewer steps for faster training
        beta_schedule="cosine",
        prediction_type="v_prediction"
    )
    
    # Create improved model
    model = ImprovedSceneDiffuser(scene_config, diffusion_config).to(device)
    print(f"Improved model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load dataset
    from train_with_realistic_synthetic import RealisticSyntheticDataset
    from torch.utils.data import DataLoader
    
    dataset = RealisticSyntheticDataset()
    dataloader = DataLoader(dataset, batch_size=scene_config.batch_size, shuffle=True)
    
    # Training with better optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=3e-4, 
        total_steps=len(dataloader) * 20
    )
    
    print("Training improved model...")
    
    for epoch in range(20):
        epoch_losses = []
        
        for batch in dataloader:
            # Move to device and resize
            batch_device = {
                'agents': batch['agents'][:, :scene_config.num_agents].to(device),
                'lights': batch['lights'][:, :8].to(device),  # Fewer lights
            }
            
            loss = model.training_step(batch_device)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'checkpoints/improved_model_epoch{epoch+1}.pt')
            print(f"  ✓ Saved checkpoint")
    
    print("\n✅ Improved model training complete!")
    
    # Quick test
    print("\n🧪 Testing improved model...")
    
    with torch.no_grad():
        agents = torch.randn(1, 32, 91, 11).to(device)
        lights = torch.randn(1, 8, 91, 16).to(device)
        
        for t in reversed(range(0, 50, 2)):
            timesteps = torch.tensor([t]).to(device)
            pred = model.forward(agents, lights, timesteps)
            
            alpha = 1 - (t / 50)
            agents = agents - (1 - alpha) * pred['agents'] * 0.02
            lights = lights - (1 - alpha) * pred['lights'] * 0.02
        
        # Check improvements
        validity = torch.sigmoid(agents[0, :, :, 0]) > 0.7
        valid_count = validity.sum().item()
        
        # Check for collisions
        positions = agents[0, :, :, 1:3]
        collisions = 0
        total_pairs = 0
        
        for t in range(91):
            valid_t = validity[:, t]
            if valid_t.sum() > 1:
                pos_t = positions[valid_t, t]
                distances = torch.cdist(pos_t, pos_t)
                collision_pairs = ((distances < 4.0) & (distances > 0)).sum().item() // 2
                collisions += collision_pairs
                total_pairs += len(pos_t) * (len(pos_t) - 1) // 2
        
        collision_rate = collisions / max(total_pairs, 1)
        
        print(f"Test Results:")
        print(f"  Valid observations: {valid_count}")
        print(f"  Estimated collision rate: {collision_rate:.4f} ({collision_rate*100:.1f}%)")
        
        if collision_rate < 0.5:
            print("  🎉 Collision rate improved!")
        else:
            print("  🔧 Still need more training")

if __name__ == "__main__":
    train_improved_model()
