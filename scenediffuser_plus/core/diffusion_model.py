import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DiffusionConfig:
    """Configuration for diffusion model"""
    num_diffusion_steps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "cosine"  # "linear" or "cosine"
    prediction_type: str = "v_prediction"  # "epsilon", "x0", or "v_prediction"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    
class DiffusionScheduler:
    """Handles noise scheduling for diffusion"""
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.num_steps = config.num_diffusion_steps
        
        # Create beta schedule
        if config.beta_schedule == "linear":
            self.betas = torch.linspace(config.beta_start, config.beta_end, self.num_steps)
        elif config.beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(self.num_steps)
        else:
            raise ValueError(f"Unknown beta schedule: {config.beta_schedule}")
        
        # Calculate alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # Calculate other useful quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # For posterior distribution
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to data for training"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(x_start.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(x_start.device)
        
        # Handle shape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(x_start.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy = sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
        return noisy
    
    def get_velocity(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Get velocity for v-prediction parameterization"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(x_start.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(x_start.device)
        
        while len(sqrt_alpha_prod.shape) < len(x_start.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * x_start
        return velocity


class ImprovedSceneDiffuser(nn.Module):
    """SceneDiffuser++ with proper diffusion implementation"""
    
    def __init__(self, scene_config, diffusion_config: DiffusionConfig):
        super().__init__()
        self.scene_config = scene_config
        self.diffusion_config = diffusion_config
        self.scheduler = DiffusionScheduler(diffusion_config)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(scene_config.hidden_dim),
            nn.Linear(scene_config.hidden_dim, scene_config.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(scene_config.hidden_dim * 4, scene_config.hidden_dim),
        )
        
        # Improved encoder for agents
        self.agent_encoder = nn.Sequential(
            nn.Linear(scene_config.agent_features, scene_config.hidden_dim),
            nn.LayerNorm(scene_config.hidden_dim),
            nn.GELU(),
            nn.Linear(scene_config.hidden_dim, scene_config.hidden_dim),
        )
        
        # Improved encoder for traffic lights
        self.light_encoder = nn.Sequential(
            nn.Linear(scene_config.light_features, scene_config.hidden_dim),
            nn.LayerNorm(scene_config.hidden_dim),
            nn.GELU(),
            nn.Linear(scene_config.hidden_dim, scene_config.hidden_dim),
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(scene_config.hidden_dim, scene_config.num_heads)
            for _ in range(scene_config.num_layers)
        ])
        
        # Output heads
        self.agent_head = nn.Sequential(
            nn.Linear(scene_config.hidden_dim, scene_config.hidden_dim),
            nn.GELU(),
            nn.Linear(scene_config.hidden_dim, scene_config.agent_features),
        )
        
        self.light_head = nn.Sequential(
            nn.Linear(scene_config.hidden_dim, scene_config.hidden_dim),
            nn.GELU(),
            nn.Linear(scene_config.hidden_dim, scene_config.light_features),
        )
    
    def forward(self, agents: torch.Tensor, lights: torch.Tensor, 
                timesteps: torch.Tensor, context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of diffusion model"""
        B = agents.shape[0]
        
        # Get time embeddings
        t_emb = self.time_embed(timesteps)
        
        # Encode inputs
        agent_emb = self.agent_encoder(agents)  # [B, N_agents, T, D]
        light_emb = self.light_encoder(lights)  # [B, N_lights, T, D]
        
        # Combine embeddings
        # Flatten spatial and temporal dimensions for transformer
        B, N_a, T, D = agent_emb.shape
        agent_emb_flat = agent_emb.reshape(B, N_a * T, D)
        
        B, N_l, T, D = light_emb.shape
        light_emb_flat = light_emb.reshape(B, N_l * T, D)
        
        # Concatenate all tokens
        x = torch.cat([agent_emb_flat, light_emb_flat], dim=1)  # [B, N_total, D]
        
        # Add time embedding
        x = x + t_emb.unsqueeze(1)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Split back into agents and lights
        agent_out = x[:, :N_a * T].reshape(B, N_a, T, D)
        light_out = x[:, N_a * T:].reshape(B, N_l, T, D)
        
        # Apply output heads
        agent_pred = self.agent_head(agent_out)
        light_pred = self.light_head(light_out)
        
        return {
            'agents': agent_pred,
            'lights': light_pred
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Improved training step with proper diffusion"""
        agents = batch['agents']
        lights = batch['lights']
        context = batch.get('context', None)
        
        B = agents.shape[0]
        device = agents.device
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.scheduler.num_steps, (B,), device=device).long()
        
        # Sample noise
        noise_agents = torch.randn_like(agents)
        noise_lights = torch.randn_like(lights)
        
        # Add noise
        noisy_agents = self.scheduler.add_noise(agents, noise_agents, timesteps)
        noisy_lights = self.scheduler.add_noise(lights, noise_lights, timesteps)
        
        # Predict based on parameterization
        pred = self.forward(noisy_agents, noisy_lights, timesteps, context)
        
        if self.diffusion_config.prediction_type == "epsilon":
            target_agents = noise_agents
            target_lights = noise_lights
        elif self.diffusion_config.prediction_type == "v_prediction":
            target_agents = self.scheduler.get_velocity(agents, noise_agents, timesteps)
            target_lights = self.scheduler.get_velocity(lights, noise_lights, timesteps)
        else:  # x0 prediction
            target_agents = agents
            target_lights = lights
        
        # Compute loss with proper masking for validity
        agent_loss = self._masked_loss(pred['agents'], target_agents, agents[..., 0])
        light_loss = self._masked_loss(pred['lights'], target_lights, lights[..., 0])
        
        loss = agent_loss + light_loss
        return loss
    
    def _masked_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                     validity: torch.Tensor) -> torch.Tensor:
        """Compute loss with validity masking"""
        # MSE loss
        loss = F.mse_loss(pred, target, reduction='none')
        
        # Apply validity mask
        # For validity channel itself, always compute loss
        validity_loss = loss[..., 0].mean()
        
        # For other channels, only compute loss where valid
        validity_mask = validity > 0.5
        other_loss = (loss[..., 1:] * validity_mask.unsqueeze(-1)).sum() / (validity_mask.sum() + 1e-8)
        
        return validity_loss + other_loss
    
    @torch.no_grad()
    def sample(self, batch_size: int, device: torch.device, 
               context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Sample from the model using DDPM"""
        # Initialize with noise
        agents = torch.randn(batch_size, self.scene_config.num_agents, 
                           self.scene_config.timesteps, 
                           self.scene_config.agent_features, device=device)
        lights = torch.randn(batch_size, self.scene_config.num_traffic_lights,
                           self.scene_config.timesteps,
                           self.scene_config.light_features, device=device)
        
        # Denoising loop
        for t in reversed(range(self.scheduler.num_steps)):
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise/velocity
            pred = self.forward(agents, lights, timesteps, context)
            
            # Update based on prediction type
            agents = self._denoise_step(agents, pred['agents'], t)
            lights = self._denoise_step(lights, pred['lights'], t)
            
            # Apply soft clipping for validity
            agents = self._soft_clip(agents)
            lights = self._soft_clip(lights)
        
        return {'agents': agents, 'lights': lights}
    
    def _denoise_step(self, x_t: torch.Tensor, pred: torch.Tensor, t: int) -> torch.Tensor:
        """Single denoising step"""
        # Implementation depends on prediction type
        # This is simplified - full implementation would handle all cases
        if t > 0:
            noise = torch.randn_like(x_t)
            return x_t - 0.01 * pred + 0.01 * noise  # Simplified
        else:
            return x_t - 0.01 * pred
    
    def _soft_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Apply soft clipping for validity"""
        # Sigmoid on validity channel
        x_clipped = x.clone()
        x_clipped[..., 0] = torch.sigmoid(x[..., 0])
        
        # Multiply other features by validity
        validity = x_clipped[..., 0:1]
        x_clipped[..., 1:] = x[..., 1:] * validity
        
        return x_clipped


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x
