#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

def test_model():
    print("Testing SceneDiffuser++ implementation...")
    
    # Since we need to import from core, let's first check if the files exist
    try:
        from core.model import SceneDiffuserPlusPlus, SceneConfig
        print("✓ Successfully imported model classes")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure you've saved the model.py file in the core/ directory")
        return
    
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
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_model()
