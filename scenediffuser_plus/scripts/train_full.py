#!/usr/bin/env python3
"""
Full training pipeline for SceneDiffuser++
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import argparse
import yaml
import os
from datetime import datetime

from core.model import SceneDiffuserPlusPlus, SceneConfig
from core.womd_dataset import WOMDDataset

class Trainer:
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        model_cfg = SceneConfig(**self.cfg['model'])
        self.model = SceneDiffuserPlusPlus(model_cfg).to(self.device)
        
        # Setup training
        self.setup_training()
        
    def setup_training(self):
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg['training']['learning_rate'],
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg['training']['max_steps']
        )
        
        # Create data loaders
        train_dataset = WOMDDataset(
            self.cfg['data']['data_path'],
            split='train'
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        # Create experiment name
        exp_name = f"scenediffuser_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # TensorBoard
        self.writer = SummaryWriter(f'logs/{exp_name}')
        
        # Weights & Biases (optional)
        if self.cfg.get('use_wandb', False):
            wandb.init(project='scenediffuser', name=exp_name, config=self.cfg)
        
    def train(self):
        print("Starting training...")
        
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.cfg['training']['num_epochs']):
            epoch_loss = 0
            
            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                batch = self.to_device(batch)
                
                # Forward pass
                loss = self.model.training_step(batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg['training']['gradient_clip']
                )
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Logging
                epoch_loss += loss.item()
                global_step += 1
                
                if global_step % 100 == 0:
                    self.writer.add_scalar('loss/train', loss.item(), global_step)
                    self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], global_step)
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Save checkpoint
                if global_step % self.cfg['training']['checkpoint_interval'] == 0:
                    self.save_checkpoint(epoch, global_step, loss.item())
            
            # Epoch summary
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(epoch, global_step, avg_loss, is_best=True)
    
    def to_device(self, batch):
        """Move batch to device"""
        if isinstance(batch, dict):
            return {k: self.to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self.to_device(v) for v in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch
    
    def save_checkpoint(self, epoch, step, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.cfg
        }
        
        filename = 'best_model.pt' if is_best else f'checkpoint_step_{step}.pt'
        path = os.path.join('checkpoints', filename)
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default_config.yaml')
    args = parser.parse_args()
    
    trainer = Trainer(args.config)
    trainer.train()

if __name__ == "__main__":
    main()
