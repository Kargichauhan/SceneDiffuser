import torch
import pickle
import os
import numpy as np

print("Creating synthetic dataset...")

# Create data directory
os.makedirs('data/synthetic_womd', exist_ok=True)

scenarios = []

# Create 10 varied scenarios
for i in range(10):
    scenario = {
        'scenario_id': f'synthetic_{i}',
        'type': 'intersection',
        'agents': torch.zeros(128, 91, 10),
        'traffic_lights': torch.zeros(32, 91, 5),
        'roadgraph': torch.randn(1000, 7),
    }
    
    # Create some realistic agent trajectories
    num_agents = np.random.randint(5, 20)
    for agent_id in range(num_agents):
        # Random spawn and despawn times
        spawn_time = np.random.randint(0, 30)
        despawn_time = np.random.randint(60, 91)
        
        # Set validity
        scenario['agents'][agent_id, spawn_time:despawn_time, 0] = 1
        
        # Create simple trajectory
        start_x = np.random.uniform(-50, 50)
        start_y = np.random.uniform(-50, 50)
        vel_x = np.random.uniform(-2, 2)
        vel_y = np.random.uniform(-2, 2)
        
        for t in range(spawn_time, despawn_time):
            dt = t - spawn_time
            scenario['agents'][agent_id, t, 1] = start_x + vel_x * dt  # x
            scenario['agents'][agent_id, t, 2] = start_y + vel_y * dt  # y
            scenario['agents'][agent_id, t, 3] = 0  # z
            scenario['agents'][agent_id, t, 4] = np.arctan2(vel_y, vel_x)  # heading
            scenario['agents'][agent_id, t, 5:8] = torch.tensor([4.5, 2.0, 1.5])  # size
            scenario['agents'][agent_id, t, 8] = 1  # car type
    
    # Add some traffic lights
    for light_id in range(8):
        scenario['traffic_lights'][light_id, :, 0] = 1  # valid
        scenario['traffic_lights'][light_id, :, 1] = (light_id % 4 - 1.5) * 20  # x
        scenario['traffic_lights'][light_id, :, 2] = (light_id // 4 - 0.5) * 20  # y
        
        # Cycle through states
        for t in range(91):
            phase = (t // 30) % 3
            scenario['traffic_lights'][light_id, t, 4] = [4, 6, 5][phase]  # green, yellow, red
    
    scenarios.append(scenario)

# Save scenarios
with open('data/synthetic_womd/synthetic_00000.pkl', 'wb') as f:
    pickle.dump(scenarios, f)

print(f"✓ Created {len(scenarios)} scenarios")
print("✓ Saved to data/synthetic_womd/synthetic_00000.pkl")
