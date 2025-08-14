#!/usr/bin/env python3
"""
Train SceneDiffuser++ with realistic WOMD-format synthetic data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import sys
sys.path.append('.')

from fix_mps_training import SimpleMPSDiffusion
from core.model import SceneConfig
from core.diffusion_model import DiffusionConfig
import matplotlib.pyplot as plt
from tqdm import tqdm

class RealisticSyntheticDataset(Dataset):
    """Dataset for realistic WOMD-format synthetic data"""
    
    def __init__(self, data_path='data/womd_synthetic/synthetic_scenarios.pkl'):
        print(f"Loading realistic synthetic data from {data_path}")
        
        with open(data_path, 'rb') as f:
            self.scenarios = pickle.load(f)
        
        print(f"✓ Loaded {len(self.scenarios)} realistic scenarios")
        
        # Analyze data quality
        first_scenario = self.scenarios[0]
        agents = first_scenario['agents']
        validity = (agents[:, :, 0] > 0).sum()
        total = agents.shape[0] * agents.shape[1]
        
        print(f"  Data quality: {validity}/{total} valid observations ({validity/total:.1%})")
        print(f"  Agent features: {agents.shape[2]}")
        print(f"  Traffic lights: {first_scenario['traffic_lights'].shape}")
    
    def __len__(self):
        return len(self.scenarios)
    
    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        
        return {
            'agents': scenario['agents'],
            'lights': scenario['traffic_lights'],
            'context': {
                'roadgraph': scenario['roadgraph'],
                'scenario_id': scenario['scenario_id'],
                'sdc_track_index': scenario['sdc_track_index']
            }
        }

# Enhanced model for full WOMD features
class EnhancedMPSDiffusion(SimpleMPSDiffusion):
    """Enhanced model for full WOMD feature dimensions"""
    
    def __init__(self, scene_config, diffusion_config):
        super().__init__(scene_config, diffusion_config)
        
        # Update networks for WOMD dimensions
        self.agent_net = torch.nn.Sequential(
            torch.nn.Linear(scene_config.agent_features, 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, 512),
            torch.nn.SiLU(),
            torch.nn.Linear(512, scene_config.agent_features),
        )
        
        self.light_net = torch.nn.Sequential(
            torch.nn.Linear(scene_config.light_features, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, scene_config.light_features),
        )
        
        # Add roadgraph encoder
        self.roadgraph_encoder = torch.nn.Sequential(
            torch.nn.Linear(20, 256),  # 20 roadgraph features
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
        )
    
    def training_step(self, batch):
        """Enhanced training with roadgraph context"""
        agents = batch['agents']
        lights = batch['lights']
        
        # Process roadgraph context if available
        if 'context' in batch and 'roadgraph' in batch['context']:
            roadgraph = batch['context']['roadgraph']
            # Encode roadgraph (simplified - just take mean for now)
            roadgraph_context = self.roadgraph_encoder(roadgraph).mean(dim=1)  # [B, 256]
        
        # Standard diffusion training
        B = agents.shape[0]
        device = agents.device
        
        timesteps = torch.randint(0, self.scheduler.num_steps, (B,), device=device)
        
        noise_agents = torch.randn_like(agents)
        noise_lights = torch.randn_like(lights)
        
        noisy_agents = self.scheduler.add_noise(agents, noise_agents, timesteps)
        noisy_lights = self.scheduler.add_noise(lights, noise_lights, timesteps)
        
        pred = self.forward(noisy_agents, noisy_lights, timesteps)
        
        # Compute targets
        if self.diffusion_config.prediction_type == "v_prediction":
            target_agents = self.scheduler.get_velocity(agents, noise_agents, timesteps)
            target_lights = self.scheduler.get_velocity(lights, noise_lights, timesteps)
        else:
            target_agents = noise_agents
            target_lights = noise_lights
        
        # Enhanced loss with validity masking
        agent_validity = agents[:, :, :, 0]  # [B, N, T]
        light_validity = lights[:, :, :, 0]
        
        # Validity loss (always compute)
        validity_loss = torch.nn.functional.mse_loss(
            pred['agents'][:, :, :, 0], target_agents[:, :, :, 0]
        )
        validity_loss += torch.nn.functional.mse_loss(
            pred['lights'][:, :, :, 0], target_lights[:, :, :, 0]
        )
        
        # Feature loss (only where valid)
        agent_mask = agent_validity > 0.5
        if agent_mask.sum() > 0:
            agent_feature_loss = torch.nn.functional.mse_loss(
                pred['agents'][:, :, :, 1:][agent_mask],
                target_agents[:, :, :, 1:][agent_mask]
            )
        else:
            agent_feature_loss = torch.tensor(0.0, device=device)
        
        light_mask = light_validity > 0.5
        if light_mask.sum() > 0:
            light_feature_loss = torch.nn.functional.mse_loss(
                pred['lights'][:, :, :, 1:][light_mask],
                target_lights[:, :, :, 1:][light_mask]
            )
        else:
            light_feature_loss = torch.tensor(0.0, device=device)
        
        total_loss = validity_loss + agent_feature_loss + light_feature_loss
        return total_loss

def main():
    print("🚀 Training SceneDiffuser++ on Realistic WOMD Synthetic Data\n")
    
    # Check MPS availability
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    try:
        dataset = RealisticSyntheticDataset()
    except FileNotFoundError:
        print("❌ Realistic synthetic data not found. Please run:")
        print("python create_realistic_womd_synthetic.py")
        return
    
    # Configure for full WOMD features
    scene_config = SceneConfig()
    scene_config.batch_size = 2
    scene_config.num_agents = 128      # Full WOMD
    scene_config.timesteps = 91        # Full WOMD
    scene_config.agent_features = 11   # Full WOMD agent features
    scene_config.light_features = 16   # Full WOMD light features
    scene_config.max_roadgraph_points = 20000
    
    diffusion_config = DiffusionConfig(
        num_diffusion_steps=500,
        beta_schedule="cosine",
        prediction_type="v_prediction"
    )
    
    # Create enhanced model
    model = EnhancedMPSDiffusion(scene_config, diffusion_config).to(device)
    print(f"Enhanced model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Data loader
    dataloader = DataLoader(dataset, batch_size=scene_config.batch_size, shuffle=True)
    
    # Test one batch
    sample_batch = next(iter(dataloader))
    print(f"\nSample batch shapes:")
    print(f"  Agents: {sample_batch['agents'].shape}")
    print(f"  Lights: {sample_batch['lights'].shape}")
    print(f"  Roadgraph: {sample_batch['context']['roadgraph'].shape}")
    
    # Optimizer with better settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=3e-4, 
        weight_decay=0.01,
        betas=(0.9, 0.99)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Training
    print(f"\n🔥 Starting training on realistic data...")
    losses = []
    
    for epoch in range(15):  # More epochs for better quality
        epoch_losses = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/15")
        for batch in progress_bar:
            # Move to device
            batch_device = {
                'agents': batch['agents'].to(device),
                'lights': batch['lights'].to(device),
                'context': {
                    'roadgraph': batch['context']['roadgraph'].to(device),
                    'scenario_id': batch['context']['scenario_id']
                }
            }
            
            # Training step
            loss = model.training_step(batch_device)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'checkpoints/realistic_model_epoch{epoch+1}.pt')
            print(f"  ✓ Saved checkpoint: realistic_model_epoch{epoch+1}.pt")
    
    # Plot training progress
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training on Realistic WOMD Data')
    plt.grid(True, alpha=0.3)
    
    # Generate improved sample
    print(f"\n🎨 Generating sample with realistic model...")
    
    @torch.no_grad()
    def generate_realistic_sample(model, steps=50):
        agents = torch.randn(1, 128, 91, 11).to(device)
        lights = torch.randn(1, 16, 91, 16).to(device)
        
        for t in tqdm(reversed(range(0, steps)), desc="Generating"):
            timesteps = torch.tensor([t]).to(device)
            pred = model.forward(agents, lights, timesteps)
            
            alpha = 1 - (t / steps)
            agents = agents - (1 - alpha) * pred['agents'] * 0.01
            lights = lights - (1 - alpha) * pred['lights'] * 0.01
            
            if t > 0:
                noise_scale = (t / steps) * 0.3
                agents += torch.randn_like(agents) * noise_scale
                lights += torch.randn_like(lights) * noise_scale
        
        return agents.cpu(), lights.cpu()
    
    agents_gen, lights_gen = generate_realistic_sample(model)
    
    # Analyze generation quality
    plt.subplot(1, 2, 2)
    
    validity = torch.sigmoid(agents_gen[0, :, :, 0])
    valid_count = validity.sum(dim=1)  # Valid agents per timestep
    
    plt.plot(valid_count.numpy(), 'g-', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Valid Agents')
    plt.title('Generated Agent Count Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realistic_training_results.png')
    print(f"✓ Saved results to realistic_training_results.png")
    
    # Statistics
    total_valid = (validity > 0.5).sum().item()
    total_possible = validity.numel()
    
    print(f"\n📊 Generation Quality:")
    print(f"  Valid observations: {total_valid}/{total_possible} ({total_valid/total_possible:.1%})")
    print(f"  Max concurrent agents: {valid_count.max().item()}")
    print(f"  Average agents per timestep: {valid_count.mean().item():.1f}")
    
    print(f"\n🎉 Training complete! Model saved to checkpoints/")

if __name__ == "__main__":
    main()
