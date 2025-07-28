#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from core.model import SceneDiffuserPlusPlus, SceneConfig
from core.components import LongHorizonSimulator, VisualizationUtils

def test_model():
    print("Testing SceneDiffuser++ implementation...")
    
    # Create config
    config = SceneConfig()
    config.batch_size = 2  # Small batch for testing
    
    # Initialize model
    model = SceneDiffuserPlusPlus(config)
    model.eval()
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    dummy_batch = {
        'agents': torch.randn(2, 128, 91, 10),
        'lights': torch.randn(2, 32, 91, 5),
        'context': {
            'roadgraph': torch.randn(2, 1000, 7),
            'traffic_lights': torch.randn(2, 32, 5)
        }
    }
    
    with torch.no_grad():
        loss = model.training_step(dummy_batch)
    
    print(f"✓ Forward pass successful, loss: {loss.item():.4f}")
    
    # Test generation
    print("\nTesting generation...")
    simulator = LongHorizonSimulator(model, config)
    
    with torch.no_grad():
        # Short rollout for testing
        trajectory = simulator.rollout(dummy_batch, num_seconds=3.0)
    
    print(f"✓ Generated {len(trajectory)} trajectory segments")
    
    # Test metrics computation
    from core.evaluation import SimulationMetrics
    metrics_computer = SimulationMetrics()
    
    # Simple metric test
    agents = trajectory[0]['agents']
    validity = (agents[..., 0] > 0.5).float()
    avg_validity = validity.mean()
    
    print(f"✓ Average agent validity: {avg_validity:.2%}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_model()