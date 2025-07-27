#!/usr/bin/env python3
"""Demo training script"""

import torch
from torch.utils.data import DataLoader
from core.model import SceneDiffuserPlusPlus, SceneConfig, DiffusionTrainer
from core.components import WOMDDataset
from tqdm import tqdm

# Config
config = SceneConfig()
config.batch_size = 4  # Small for demo
config.num_agents = 32  # Smaller for faster training

# Model
model = SceneDiffuserPlusPlus(config)
trainer = DiffusionTrainer(model, config)

# Data
dataset = WOMDDataset("dummy_path")
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

# Train
print("Starting training demo...")
for epoch in range(3):
    total_loss = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        loss = model.training_step(batch)
        trainer.optimizer.zero_grad()
        loss.backward()
        trainer.optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

print("\n✅ Training demo complete!")
