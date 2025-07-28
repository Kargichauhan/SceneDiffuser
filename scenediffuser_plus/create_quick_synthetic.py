import torch
import pickle
import os

print("Creating quick synthetic data...")

# Create data directory
os.makedirs('data/synthetic_womd', exist_ok=True)

# Create a simple scenario
scenario = {
    'scenario_id': 'test_scenario_0',
    'type': 'intersection',
    'agents': torch.randn(128, 91, 10),  # Random for now
    'traffic_lights': torch.randn(32, 91, 5),
    'roadgraph': torch.randn(1000, 7),
}

# Make first 10 agents valid
for i in range(10):
    scenario['agents'][i, :, 0] = 1  # Set validity
    scenario['agents'][i, :, 1] = i * 5 - 25  # X position
    scenario['agents'][i, :, 2] = 0  # Y position
    scenario['agents'][i, :, 5:8] = torch.tensor([4.5, 2.0, 1.5])  # Size

# Save it
scenarios = [scenario]
with open('data/synthetic_womd/synthetic_00000.pkl', 'wb') as f:
    pickle.dump(scenarios, f)

print("✓ Created synthetic data at data/synthetic_womd/")
print("✓ You can now run: python test_synthetic_data.py")
