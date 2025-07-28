import torch
from torch.utils.data import Dataset
import pickle
import glob
import os

class SyntheticWOMDDataset(Dataset):
    """Dataset loader for synthetic WOMD data"""
    
    def __init__(self, data_path='data/synthetic_womd', split='train'):
        self.data_path = data_path
        self.split = split
        
        # Load all scenario files
        self.files = sorted(glob.glob(f"{data_path}/synthetic_*.pkl"))
        
        # Load scenarios
        self.scenarios = []
        for file in self.files[:5]:  # Load first 5 files for now
            with open(file, 'rb') as f:
                self.scenarios.extend(pickle.load(f))
        
        print(f"Loaded {len(self.scenarios)} scenarios")
        
        # Train/val split
        split_idx = int(0.9 * len(self.scenarios))
        if split == 'train':
            self.scenarios = self.scenarios[:split_idx]
        else:
            self.scenarios = self.scenarios[split_idx:]
    
    def __len__(self):
        return len(self.scenarios)
    
    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        
        return {
            'agents': scenario['agents'],
            'lights': scenario['traffic_lights'],
            'context': {
                'roadgraph': scenario['roadgraph'],
                'scenario_id': scenario['scenario_id']
            }
        }
