#!/usr/bin/env python3
"""Ultra-minimal training that won't crash"""

import torch
import torch.nn as nn
import numpy as np

print("Creating ultra-minimal diffusion demo...")

# Tiny model
class TinyDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
    def forward(self, x, t):
        # Simple: ignore time for now
        return self.net(x)

# Create tiny data
print("\n1. Creating tiny dataset...")
data = []
for i in range(5):
    # 4 agents, 10 timesteps, 10 features
    agents = torch.zeros(4, 10, 10)
    agents[0, :, 0] = 1  # First agent always valid
    agents[0, :, 1] = torch.linspace(-5, 5, 10)  # Simple trajectory
    data.append(agents)
print(f"   ✓ Created {len(data)} tiny scenarios")

# Train
model = TinyDiffusion()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(f"\n2. Training tiny model ({sum(p.numel() for p in model.parameters())} parameters)...")

for epoch in range(10):
    total_loss = 0
    for agents in data:
        # Add noise
        noise = torch.randn_like(agents) * 0.1
        noisy = agents + noise
        
        # Predict
        pred = model(noisy.reshape(-1, 10), torch.tensor([0.5]))
        pred = pred.reshape(agents.shape)
        
        # Loss
        loss = ((pred - agents) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % 2 == 0:
        print(f"   Epoch {epoch}: Loss = {total_loss/len(data):.4f}")

print("\n3. Testing generation...")
with torch.no_grad():
    # Start from noise
    x = torch.randn(4, 10, 10)
    
    # "Denoise" (simplified)
    for i in range(5):
        x = x - 0.1 * model(x.reshape(-1, 10), torch.tensor([0.5])).reshape(x.shape)
    
    # Check validity
    validity = torch.sigmoid(x[:, :, 0])
    print(f"   ✓ Generated sample with {(validity > 0.5).sum()} valid observations")

print("\n✅ Ultra-minimal diffusion demo complete!")
print("\nKey concepts demonstrated:")
print("- Noise addition during training")
print("- Denoising during generation")
print("- Validity handling")
print("\nNext: Run on Google Colab for full model!")
