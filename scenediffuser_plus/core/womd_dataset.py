import torch
import tensorflow as tf
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import glob

class WOMDDataset(Dataset):
    """
    Waymo Open Motion Dataset loader for SceneDiffuser++
    """
    def __init__(self, data_path: str, split: str = 'train', 
                 max_agents: int = 128, 
                 max_traffic_lights: int = 32):
        self.data_path = data_path
        self.split = split
        self.max_agents = max_agents
        self.max_traffic_lights = max_traffic_lights
        
        # Find all tfrecord files
        pattern = f"{data_path}/{split}*.tfrecord*"
        self.files = sorted(glob.glob(pattern))
        
        if len(self.files) == 0:
            raise ValueError(f"No tfrecord files found at {pattern}")
        
        print(f"Found {len(self.files)} files for {split}")
        
        # Load and cache scenarios for faster access
        self.scenarios = []
        self._load_scenarios()
    
    def _load_scenarios(self):
        """Load scenarios from tfrecord files"""
        # For demo, just create dummy data
        # In production, parse actual tfrecords
        print("Loading scenarios... (using dummy data for now)")
        
        # You'll need to implement tfrecord parsing here
        # For now, create synthetic data
        for i in range(100):  # 100 dummy scenarios
            self.scenarios.append(self._create_dummy_scenario(i))
    
    def _create_dummy_scenario(self, idx):
        """Create a dummy scenario for testing"""
        # 91 timesteps (9.1 seconds at 10Hz)
        timesteps = 91
        
        # Create agents
        num_agents = np.random.randint(20, self.max_agents)
        agents = torch.zeros(self.max_agents, timesteps, 10)
        
        for i in range(num_agents):
            # Validity
            start_t = np.random.randint(0, 20)
            end_t = np.random.randint(70, timesteps)
            agents[i, start_t:end_t, 0] = 1
            
            # Simple trajectory
            agents[i, :, 1] = np.linspace(-30, 30, timesteps)  # x
            agents[i, :, 2] = np.random.uniform(-5, 5)  # y
            agents[i, :, 4] = 0  # heading
            agents[i, :, 5:8] = torch.tensor([4.5, 2.0, 1.7])  # size
        
        # Create traffic lights
        lights = torch.zeros(self.max_traffic_lights, timeste
cat > core/womd_dataset.py << 'EOF'
import torch
import tensorflow as tf
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import glob

class WOMDDataset(Dataset):
    """
    Waymo Open Motion Dataset loader for SceneDiffuser++
    """
    def __init__(self, data_path: str, split: str = 'train', 
                 max_agents: int = 128, 
                 max_traffic_lights: int = 32):
        self.data_path = data_path
        self.split = split
        self.max_agents = max_agents
        self.max_traffic_lights = max_traffic_lights
        
        # Find all tfrecord files
        pattern = f"{data_path}/{split}*.tfrecord*"
        self.files = sorted(glob.glob(pattern))
        
        if len(self.files) == 0:
            raise ValueError(f"No tfrecord files found at {pattern}")
        
        print(f"Found {len(self.files)} files for {split}")
        
        # Load and cache scenarios for faster access
        self.scenarios = []
        self._load_scenarios()
    
    def _load_scenarios(self):
        """Load scenarios from tfrecord files"""
        # For demo, just create dummy data
        # In production, parse actual tfrecords
        print("Loading scenarios... (using dummy data for now)")
        
        # You'll need to implement tfrecord parsing here
        # For now, create synthetic data
        for i in range(100):  # 100 dummy scenarios
            self.scenarios.append(self._create_dummy_scenario(i))
    
    def _create_dummy_scenario(self, idx):
        """Create a dummy scenario for testing"""
        # 91 timesteps (9.1 seconds at 10Hz)
        timesteps = 91
        
        # Create agents
        num_agents = np.random.randint(20, self.max_agents)
        agents = torch.zeros(self.max_agents, timesteps, 10)
        
        for i in range(num_agents):
            # Validity
            start_t = np.random.randint(0, 20)
            end_t = np.random.randint(70, timesteps)
            agents[i, start_t:end_t, 0] = 1
            
            # Simple trajectory
            agents[i, :, 1] = np.linspace(-30, 30, timesteps)  # x
            agents[i, :, 2] = np.random.uniform(-5, 5)  # y
            agents[i, :, 4] = 0  # heading
            agents[i, :, 5:8] = torch.tensor([4.5, 2.0, 1.7])  # size
        
        # Create traffic lights
        lights = torch.zeros(self.max_traffic_lights, timesteps, 5)
        num_lights = 8
        for i in range(num_lights):
            lights[i, :, 0] = 1  # validity
            lights[i, :, 1:3] = torch.tensor([i*10-35, i*10-35])  # position
            # Cycle through states
            for t in range(timesteps):
                phase = (t // 30) % 3
                lights[i, t, 4] = [4, 6, 5][phase]  # green, yellow, red
        
        # Create roadgraph (simplified)
        roadgraph = torch.randn(1000, 7)
        
        return {
            'agents': agents,
            'lights': lights,
            'roadgraph': roadgraph,
            'scenario_id': f'dummy_{idx}'
        }
    
    def __len__(self):
        return len(self.scenarios)
    
    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        
        # Add context
        context = {
            'roadgraph': scenario['roadgraph'],
            'scenario_id': scenario['scenario_id']
        }
        
        return {
            'agents': scenario['agents'],
            'lights': scenario['lights'],
            'context': context
        }

def parse_womd_tfrecord(tfrecord_path):
    """
    Parse actual WOMD tfrecord files
    This is a placeholder - implement based on WOMD format
    """
    # You'll need to implement this based on WOMD proto definitions
    # See: https://github.com/waymo-research/waymo-open-dataset
    pass
