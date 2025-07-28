#!/usr/bin/env python3
"""
Create synthetic data that mimics WOMD structure for development
"""

import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class SyntheticWOMD:
    """Create synthetic traffic scenarios similar to WOMD"""
    
    def __init__(self, num_scenarios=1000, save_path='data/synthetic_womd'):
        self.num_scenarios = num_scenarios
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        
        # WOMD parameters
        self.timesteps = 91  # 9.1 seconds at 10Hz
        self.max_agents = 128
        self.max_traffic_lights = 16
        
        # Scenario types for diversity
        self.scenario_types = [
            'intersection_4way',
            'intersection_T',
            'highway_merge',
            'roundabout',
            'parking_lot',
            'urban_street'
        ]
    
    def create_dataset(self):
        """Generate full synthetic dataset"""
        print(f"Creating {self.num_scenarios} synthetic scenarios...")
        
        scenarios = []
        for i in tqdm(range(self.num_scenarios)):
            scenario_type = np.random.choice(self.scenario_types)
            scenario = self.create_scenario(i, scenario_type)
            scenarios.append(scenario)
            
            # Save individual files like WOMD
            if i % 100 == 0:
                self.save_batch(scenarios, i // 100)
                scenarios = []
        
        # Save remaining
        if scenarios:
            self.save_batch(scenarios, self.num_scenarios // 100)
        
        print(f"\n✅ Created synthetic dataset at {self.save_path}")
        self.create_dataset_stats()
    
    def create_scenario(self, idx, scenario_type):
        """Create a single realistic scenario"""
        scenario = {
            'scenario_id': f'synthetic_{idx}_{scenario_type}',
            'type': scenario_type,
            'agents': torch.zeros(self.max_agents, self.timesteps, 10),
            'traffic_lights': torch.zeros(self.max_traffic_lights, self.timesteps, 5),
            'roadgraph': self.create_roadgraph(scenario_type),
        }
        
        if scenario_type == 'intersection_4way':
            self.create_intersection_scenario(scenario)
        elif scenario_type == 'highway_merge':
            self.create_highway_scenario(scenario)
        elif scenario_type == 'urban_street':
            self.create_urban_scenario(scenario)
        else:
            self.create_intersection_scenario(scenario)  # Default
        
        return scenario
    
    def create_intersection_scenario(self, scenario):
        """Create realistic intersection traffic"""
        # Traffic flow patterns
        flows = [
            {'from': 'south', 'to': 'north', 'lane': 0},
            {'from': 'south', 'to': 'east', 'lane': -1},  # Left turn
            {'from': 'north', 'to': 'south', 'lane': 0},
            {'from': 'north', 'to': 'west', 'lane': -1},
            {'from': 'east', 'to': 'west', 'lane': 0},
            {'from': 'east', 'to': 'north', 'lane': 1},  # Right turn
            {'from': 'west', 'to': 'east', 'lane': 0},
            {'from': 'west', 'to': 'south', 'lane': 1},
        ]
        
        # Generate vehicles
        num_vehicles = np.random.randint(15, 40)
        for i in range(num_vehicles):
            flow = np.random.choice(flows)
            self.create_vehicle_trajectory(scenario['agents'], i, flow)
        
        # Add traffic lights
        self.add_intersection_lights(scenario['traffic_lights'])
        
        # Add some parked cars
        self.add_parked_vehicles(scenario['agents'], num_vehicles, 5)
    
    def create_vehicle_trajectory(self, agents, idx, flow):
        """Create realistic vehicle trajectory"""
        # Random spawn time
        spawn_time = np.random.randint(0, 30)
        
        # Vehicle properties
        vehicle_type = np.random.choice([1, 1, 1, 2, 3], p=[0.7, 0.7, 0.7, 0.2, 0.1])
        if vehicle_type == 1:  # Car
            length, width, height = 4.5, 2.0, 1.5
            max_speed = 15.0  # m/s
        elif vehicle_type == 2:  # Truck
            length, width, height = 8.0, 2.5, 3.0
            max_speed = 12.0
        else:  # Motorcycle
            length, width, height = 2.5, 1.0, 1.5
            max_speed = 20.0
        
        # Create trajectory based on flow
        if flow['from'] == 'south':
            start_pos = np.array([flow['lane'] * 3.5, -60])
            if flow['to'] == 'north':
                end_pos = np.array([flow['lane'] * 3.5, 60])
                trajectory_type = 'straight'
            else:  # Turn
                end_pos = np.array([30, flow['lane'] * 3.5])
                trajectory_type = 'left_turn'
        elif flow['from'] == 'north':
            start_pos = np.array([-flow['lane'] * 3.5, 60])
            end_pos = np.array([-flow['lane'] * 3.5, -60])
            trajectory_type = 'straight'
        else:
            # Similar logic for east/west
            start_pos = np.array([-60, flow['lane'] * 3.5])
            end_pos = np.array([60, flow['lane'] * 3.5])
            trajectory_type = 'straight'
        
        # Generate smooth trajectory
        for t in range(spawn_time, min(spawn_time + 60, self.timesteps)):
            progress = (t - spawn_time) / 60
            
            # Set validity
            agents[idx, t, 0] = 1
            
            # Position with some noise
            if trajectory_type == 'straight':
                pos = start_pos + progress * (end_pos - start_pos)
                heading = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
            else:  # Turning
                # Simplified turning trajectory
                if progress < 0.4:
                    pos = start_pos + progress * 2.5 * (np.array([0, 30]) - start_pos)
                    heading = np.pi / 2
                else:
                    turn_progress = (progress - 0.4) / 0.6
                    pos = np.array([turn_progress * 30, 0])
                    heading = 0
            
            # Add realistic noise
            pos += np.random.normal(0, 0.1, 2)
            
            agents[idx, t, 1:3] = torch.tensor(pos)
            agents[idx, t, 3] = 0  # z
            agents[idx, t, 4] = heading
            agents[idx, t, 5:8] = torch.tensor([length, width, height])
            agents[idx, t, 8] = vehicle_type
    
    def add_intersection_lights(self, traffic_lights):
        """Add realistic traffic light patterns"""
        # 4-way intersection lights
        light_positions = [
            (-10, -10), (10, -10), (10, 10), (-10, 10),  # Corner lights
            (0, -15), (15, 0), (0, 15), (-15, 0)  # Center lights
        ]
        
        cycle_length = 120  # 12 seconds full cycle
        
        for i, pos in enumerate(light_positions[:8]):  # Use first 8
            # Set position and validity
            traffic_lights[i, :, 0] = 1
            traffic_lights[i, :, 1:3] = torch.tensor(pos)
            
            # Create realistic light patterns
            phase_offset = (i % 4) * 30  # Stagger lights
            
            for t in range(self.timesteps):
                cycle_pos = (t + phase_offset) % cycle_length
                
                if cycle_pos < 50:  # Green
                    state = 4
                elif cycle_pos < 60:  # Yellow
                    state = 6
                else:  # Red
                    state = 5
                
                traffic_lights[i, t, 4] = state
    
    def create_roadgraph(self, scenario_type):
        """Create roadgraph features"""
        roadgraph = []
        
        if 'intersection' in scenario_type:
            # Create road segments
            roads = [
                # North-South road
                [(-3.5, -60), (-3.5, -20)],
                [(-3.5, -20), (-3.5, 20)],
                [(-3.5, 20), (-3.5, 60)],
                [(3.5, -60), (3.5, -20)],
                [(3.5, -20), (3.5, 20)],
                [(3.5, 20), (3.5, 60)],
                # East-West road
                [(-60, -3.5), (-20, -3.5)],
                [(-20, -3.5), (20, -3.5)],
                [(20, -3.5), (60, -3.5)],
                [(-60, 3.5), (-20, 3.5)],
                [(-20, 3.5), (20, 3.5)],
                [(20, 3.5), (60, 3.5)],
            ]
            
            for road in roads:
                for i in range(10):  # Sample points along road
                    t = i / 9
                    point = [
                        road[0][0] + t * (road[1][0] - road[0][0]),
                        road[0][1] + t * (road[1][1] - road[0][1]),
                        0,  # z
                        1,  # lane type
                        len(roads),  # polyline length
                        1,  # has speed limit
                        13.4  # 30 mph in m/s
                    ]
                    roadgraph.append(point)
        
        # Pad or truncate to 1000 features
        if len(roadgraph) > 1000:
            roadgraph = roadgraph[:1000]
        else:
            roadgraph.extend([[0]*7] * (1000 - len(roadgraph)))
        
        return torch.tensor(roadgraph, dtype=torch.float32)
    
    def add_parked_vehicles(self, agents, start_idx, count):
        """Add parked vehicles for realism"""
        for i in range(count):
            idx = start_idx + i
            if idx >= self.max_agents:
                break
            
            # Random parking spot
            x = np.random.choice([-40, -30, 30, 40])
            y = np.random.uniform(-50, 50)
            
            # Parked for entire scenario
            agents[idx, :, 0] = 1  # Valid
            agents[idx, :, 1] = x
            agents[idx, :, 2] = y
            agents[idx, :, 3] = 0
            agents[idx, :, 4] = 0 if x < 0 else np.pi  # Face road
            agents[idx, :, 5:8] = torch.tensor([4.5, 2.0, 1.5])
            agents[idx, :, 8] = 1  # Car
    
    def create_highway_scenario(self, scenario):
        """Create highway merge scenario"""
        # Main highway traffic
        for i in range(20):
            spawn_time = np.random.randint(0, 40)
            lane = np.random.choice([0, 1, 2])  # 3 lanes
            speed = np.random.uniform(25, 35)  # m/s
            
            for t in range(spawn_time, self.timesteps):
                progress = (t - spawn_time) * speed
                scenario['agents'][i, t, 0] = 1
                scenario['agents'][i, t, 1] = -100 + progress
                scenario['agents'][i, t, 2] = -5 + lane * 3.5
                scenario['agents'][i, t, 4] = 0  # Heading east
                scenario['agents'][i, t, 5:8] = torch.tensor([4.5, 2.0, 1.5])
    
    def create_urban_scenario(self, scenario):
        """Create urban street with pedestrians"""
        # Add vehicles
        for i in range(15):
            self.create_vehicle_trajectory(scenario['agents'], i, 
                                         {'from': 'west', 'to': 'east', 'lane': 0})
        
        # Add pedestrians
        for i in range(15, 25):
            spawn_time = np.random.randint(0, 60)
            crossing_point = np.random.uniform(-30, 30)
            
            for t in range(spawn_time, min(spawn_time + 30, self.timesteps)):
                progress = (t - spawn_time) / 30
                scenario['agents'][i, t, 0] = 1
                scenario['agents'][i, t, 1] = crossing_point
                scenario['agents'][i, t, 2] = -10 + progress * 20
                scenario['agents'][i, t, 4] = np.pi / 2
                scenario['agents'][i, t, 5:8] = torch.tensor([0.5, 0.5, 1.8])
                scenario['agents'][i, t, 8] = 4  # Pedestrian type
    
    def save_batch(self, scenarios, batch_idx):
        """Save scenarios in WOMD-like format"""
        filename = f"{self.save_path}/synthetic_{batch_idx:05d}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(scenarios, f)
        scenarios.clear()  # Free memory
    
    def create_dataset_stats(self):
        """Create dataset statistics"""
        stats = {
            'num_scenarios': self.num_scenarios,
            'timesteps': self.timesteps,
            'max_agents': self.max_agents,
            'max_traffic_lights': self.max_traffic_lights,
            'scenario_types': self.scenario_types
        }
        
        with open(f"{self.save_path}/dataset_stats.pkl", 'wb') as f:
            pickle.dump(stats, f)
        
        print("\nDataset Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    
    def visualize_sample(self):
        """Visualize a sample scenario"""
        # Load one scenario
        with open(f"{self.save_path}/synthetic_00000.pkl", 'rb') as f:
            scenarios = pickle.load(f)
        
        scenario = scenarios[0]
        
        # Plot at different timesteps
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        timesteps_to_plot = [10, 45, 80]
        
        for ax_idx, t in enumerate(timesteps_to_plot):
            ax = axes[ax_idx]
            
            # Draw roads (simplified)
            ax.axhline(y=0, color='gray', linewidth=40, alpha=0.3)
            ax.axvline(x=0, color='gray', linewidth=40, alpha=0.3)
            
            # Draw agents
            agents = scenario['agents']
            valid_mask = agents[:, t, 0] > 0.5
            
            for i in range(len(agents)):
                if valid_mask[i]:
                    x, y = agents[i, t, 1:3]
                    heading = agents[i, t, 4]
                    length, width = agents[i, t, 5:7]
                    
                    rect = patches.Rectangle(
                        (x - length/2, y - width/2), length, width,
                        angle=np.degrees(heading),
                        color='blue' if agents[i, t, 8] == 1 else 'green',
                        alpha=0.7
                    )
                    ax.add_patch(rect)
            
            # Draw traffic lights
            lights = scenario['traffic_lights']
            for i in range(len(lights)):
                if lights[i, t, 0] > 0.5:
                    x, y = lights[i, t, 1:3]
                    state = int(lights[i, t, 4])
                    colors = {4: 'green', 5: 'red', 6: 'yellow'}
                    
                    circle = patches.Circle(
                        (x, y), 2, 
                        color=colors.get(state, 'gray'),
                        edgecolor='black'
                    )
                    ax.add_patch(circle)
            
            ax.set_xlim(-70, 70)
            ax.set_ylim(-70, 70)
            ax.set_aspect('equal')
            ax.set_title(f't = {t/10:.1f}s')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/sample_scenario.png")
        print(f"\nSaved visualization to {self.save_path}/sample_scenario.png")
        plt.close()

def main():
    print("=== Creating Synthetic WOMD Dataset ===\n")
    
    # Create dataset
    generator = SyntheticWOMD(num_scenarios=100)  # Start with 100 for testing
    generator.create_dataset()
    generator.visualize_sample()
    
    print("\n✅ You can now use this synthetic data for development!")
    print("Update your data loader to use: data/synthetic_womd/")

if __name__ == "__main__":
    main()
