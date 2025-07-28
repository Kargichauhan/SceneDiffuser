#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

from core.model import SceneDiffuserPlusPlus, SceneConfig, DiffusionTrainer
from core.components import WOMDDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default_config.yaml')
    parser.add_argument('--resume', type=str, help='Path to checkpoint')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    
    # Load config
    config_dict = load_config(args.config)
    
    # Create model config
    config = SceneConfig(**config_dict['model'])
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = SceneDiffuserPlusPlus(config).to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Create dummy dataset for testing
    # Replace with actual WOMD dataset
    train_dataset = WOMDDataset(config_dict['data']['data_path'], 'train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config_dict['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    # Initialize trainer
    trainer = DiffusionTrainer(model, config)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, 100):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Training step
            loss = model.training_step(batch)
            
            # Backward pass
            trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                          config_dict['training']['gradient_clip'])
            trainer.optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Save checkpoint
            if (batch_idx + 1) % 1000 == 0:
                checkpoint_path = f'checkpoints/model_epoch_{epoch}_step_{batch_idx}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'loss': epoch_loss / (batch_idx + 1),
                }, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()