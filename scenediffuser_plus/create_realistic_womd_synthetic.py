#!/usr/bin/env python3
"""
Create highly realistic synthetic data that matches WOMD format exactly
This will let us develop the parser and model while getting real data access
"""

import torch
import numpy as np
import pickle
import os
from typing import Dict, List
import matplotlib.pyplot as plt

class RealisticWOMDGenerator:
    """Generate synthetic data that exactly matches WOMD format"""
    
    def __init__(self):
        # WOMD standard dimensions
        self.max_agents = 128
        self.max_traffic_lights = 16
        self.timesteps = 91  # 9.1 seconds at 10Hz
        self.agent_features = 11  # [valid, x, y, z, heading, vx, vy, length, width, height, type]
        self.light_features = 16
        self.roadgraph_features = 20
        self.max_roadgraph_points = 20000
        
        # Real-world parameters
        self.city_scenarios = [
            'urban_intersection',
            'highway_merge',
            'roundabout',
            'parking_lot',
            'arterial_road',
            'residential_street'
        ]
    
    def create_womd_like_dataset(self, num_scenarios=100):
        """Create dataset that matches WOMD structure exactly"""
        print(f"Creating {num_scenarios} WOMD-like scenarios...")
        
        scenarios = []
        
        for i in range(num_scenarios):
            scenario_type = np.random.choice(self.city_scenarios)
            scenario = self.create_realistic_scenario(i, scenario_type)
            scenarios.append(scenario)
            
            if i % 20 == 0:
                print(f"  Generated {i}/{num_scenarios} scenarios")
        
        # Save in WOMD-like format
        os.makedirs('data/womd_synthetic', exist_ok=True)
        with open('data/womd_synthetic/synthetic_scenarios.pkl', 'wb') as f:
            pickle.dump(scenarios, f)
        
        print(f"✅ Created {len(scenarios)} realistic scenarios")
        return scenarios
    
    def create_realistic_scenario(self, idx, scenario_type):
        """Create a single realistic scenario matching WOMD format"""
        
        # Initialize tensors with proper WOMD dimensions
        agents = torch.zeros(self.max_agents, self.timesteps, self.agent_features)
        traffic_lights = torch.zeros(self.max_traffic_lights, self.timesteps, self.light_features)
        roadgraph = self.create_realistic_roadgraph(scenario_type)
        
        if scenario_type == 'urban_intersection':
            self.create_urban_intersection(agents, traffic_lights)
        elif scenario_type == 'highway_merge':
            self.create_highway_merge(agents, traffic_lights)
        elif scenario_type == 'roundabout':
            self.create_roundabout(agents, traffic_lights)
        else:
            self.create_urban_intersection(agents, traffic_lights)  # Default
        
        return {
            'scenario_id': f'synthetic_womd_{idx:06d}_{scenario_type}',
            'agents': agents,
            'traffic_lights': traffic_lights,
            'roadgraph': roadgraph,
            'timesteps': self.timesteps,
            'sdc_track_index': 0,  # First agent is SDC
            'objects_of_interest': [0, 1, 2],
            'tracks_to_predict': [0, 1, 2, 3, 4]
        }
    
    def create_urban_intersection(self, agents, traffic_lights):
        """Create realistic urban intersection scenario"""
        
        # Define traffic flows
        flows = [
            {'from': 'south', 'to': 'north', 'lane': 0, 'speed': 12.0},  # 30 mph
            {'from': 'south', 'to': 'east', 'lane': -1, 'speed': 8.0},   # Left turn
            {'from': 'north', 'to': 'south', 'lane': 0, 'speed': 12.0},
            {'from': 'east', 'to': 'west', 'lane': 0, 'speed': 10.0},
            {'from': 'west', 'to': 'east', 'lane': 0, 'speed': 10.0},
        ]
        
        # Generate realistic vehicles
        vehicle_types = [1, 1, 1, 1, 2, 3]  # Mostly cars, some trucks/motorcycles
        vehicle_sizes = {
            1: (4.5, 2.0, 1.5),   # Car
            2: (8.0, 2.5, 3.0),   # Truck  
            3: (2.5, 1.0, 1.5),   # Motorcycle
        }
        
        num_vehicles = np.random.randint(20, 50)
        
        for i in range(min(num_vehicles, self.max_agents)):
            flow = flows[i % len(flows)]
            vehicle_type = np.random.choice(vehicle_types)
            length, width, height = vehicle_sizes[vehicle_type]
            
            # Realistic spawn timing
            spawn_time = np.random.exponential(10)  # Poisson-like arrivals
            spawn_time = int(np.clip(spawn_time, 0, 30))
            
            # Create trajectory
            trajectory = self.create_realistic_trajectory(flow, spawn_time, length, width, height, vehicle_type)
            
            # Fill agent tensor
            for t, state in enumerate(trajectory):
                if spawn_time + t >= self.timesteps:
                    break
                
                agents[i, spawn_time + t, 0] = state['valid']
                agents[i, spawn_time + t, 1] = state['x']
                agents[i, spawn_time + t, 2] = state['y']
                agents[i, spawn_time + t, 3] = state['z']
                agents[i, spawn_time + t, 4] = state['heading']
                agents[i, spawn_time + t, 5] = state['vx']
                agents[i, spawn_time + t, 6] = state['vy']
                agents[i, spawn_time + t, 7] = length
                agents[i, spawn_time + t, 8] = width
                agents[i, spawn_time + t, 9] = height
                agents[i, spawn_time + t, 10] = vehicle_type
        
        # Add realistic traffic lights
        self.create_realistic_traffic_lights(traffic_lights)
    
    def create_realistic_trajectory(self, flow, spawn_time, length, width, height, vehicle_type):
        """Create realistic vehicle trajectory with proper physics"""
        trajectory = []
        
        # Starting parameters
        if flow['from'] == 'south':
            start_x = flow['lane'] * 3.5
            start_y = -100
            if flow['to'] == 'north':
                end_x, end_y = start_x, 100
            else:  # Turn
                end_x, end_y = 100, 0
        elif flow['from'] == 'north':
            start_x = -flow['lane'] * 3.5
            start_y = 100
            end_x, end_y = start_x, -100
        elif flow['from'] == 'east':
            start_x = 100
            start_y = flow['lane'] * 3.5
            end_x, end_y = -100, start_y
        else:  # west
            start_x = -100
            start_y = -flow['lane'] * 3.5
            end_x, end_y = 100, start_y
        
        # Physics parameters
        max_speed = flow['speed']
        acceleration = 2.0  # m/s²
        deceleration = 3.0  # m/s²
        
        # Generate trajectory with realistic acceleration/deceleration
        distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        duration = int(distance / max_speed * 10) + 10  # Extra time for acc/dec
        
        current_x, current_y = start_x, start_y
        current_speed = 0.0
        
        for t in range(duration):
            # Acceleration phase
            if t < 30 and current_speed < max_speed:
                current_speed = min(current_speed + acceleration * 0.1, max_speed)
            # Deceleration phase near intersection
            elif abs(current_x) < 20 and abs(current_y) < 20:
                current_speed = max(current_speed - deceleration * 0.1, max_speed * 0.3)
            
            # Update position
            progress = t / duration
            target_x = start_x + progress * (end_x - start_x)
            target_y = start_y + progress * (end_y - start_y)
            
            # Smooth interpolation
            current_x += (target_x - current_x) * 0.1
            current_y += (target_y - current_y) * 0.1
            
            # Calculate heading and velocity
            if t > 0:
                dx = current_x - trajectory[-1]['x']
                dy = current_y - trajectory[-1]['y']
                heading = np.arctan2(dy, dx)
                vx = dx * 10  # Convert to m/s
                vy = dy * 10
            else:
                heading = np.arctan2(end_y - start_y, end_x - start_x)
                vx = np.cos(heading) * current_speed
                vy = np.sin(heading) * current_speed
            
            # Add noise for realism
            noise_scale = 0.1
            current_x += np.random.normal(0, noise_scale)
            current_y += np.random.normal(0, noise_scale)
            
            trajectory.append({
                'valid': 1.0,
                'x': current_x,
                'y': current_y,
                'z': 0.0,
                'heading': heading,
                'vx': vx,
                'vy': vy
            })
        
        return trajectory
    
    def create_realistic_traffic_lights(self, traffic_lights):
        """Create realistic traffic light patterns"""
        # Standard intersection lights
        light_positions = [
            (-15, -15), (15, -15), (15, 15), (-15, 15),  # Corner lights
            (0, -20), (20, 0), (0, 20), (-20, 0)        # Approach lights
        ]
        
        for i in range(min(len(light_positions), self.max_traffic_lights)):
            x, y = light_positions[i]
            
            # Set position (constant)
            traffic_lights[i, :, 0] = 1.0  # validity
            traffic_lights[i, :, 1] = x
            traffic_lights[i, :, 2] = y
            traffic_lights[i, :, 3] = 0.0  # z
            
            # Realistic timing: 120-second cycle (12 seconds real time)
            cycle_length = 120  # frames
            phase_offset = (i % 4) * 30  # Stagger phases
            
            for t in range(self.timesteps):
                cycle_position = (t + phase_offset) % cycle_length
                
                if cycle_position < 50:      # Green (5 seconds)
                    state = 6  # GO
                elif cycle_position < 60:   # Yellow (1 second)
                    state = 5  # CAUTION
                else:                       # Red (6 seconds)
                    state = 4  # STOP
                
                traffic_lights[i, t, 4] = state
    
    def create_realistic_roadgraph(self, scenario_type):
        """Create realistic roadgraph matching WOMD format"""
        roadgraph_points = []
        
        if scenario_type == 'urban_intersection':
            # Create intersection roads
            roads = [
                # North-South road
                {'points': [(-3.5, y) for y in range(-100, 101, 5)], 'type': 1, 'speed': 13.4},
                {'points': [(3.5, y) for y in range(-100, 101, 5)], 'type': 1, 'speed': 13.4},
                # East-West road  
                {'points': [(x, -3.5) for x in range(-100, 101, 5)], 'type': 1, 'speed': 11.2},
                {'points': [(x, 3.5) for x in range(-100, 101, 5)], 'type': 1, 'speed': 11.2},
                # Lane boundaries
                {'points': [(0, y) for y in range(-100, 101, 10)], 'type': 3, 'speed': 0},
                {'points': [(x, 0) for x in range(-100, 101, 10)], 'type': 3, 'speed': 0},
            ]
            
            for road in roads:
                for i, (x, y) in enumerate(road['points']):
                    point_features = [
                        x, y, 0.0,  # x, y, z
                        road['type'],  # lane type
                        road['speed'],  # speed limit
                        len(road['points']),  # polyline length
                        1.0,  # has speed limit
                        i / len(road['points']),  # position along polyline
                    ] + [0.0] * 12  # Pad to 20 features
                    
                    roadgraph_points.append(point_features)
        
        # Pad or truncate to max points
        if len(roadgraph_points) > self.max_roadgraph_points:
            roadgraph_points = roadgraph_points[:self.max_roadgraph_points]
        else:
            padding = self.max_roadgraph_points - len(roadgraph_points)
            roadgraph_points.extend([[0.0] * self.roadgraph_features] * padding)
        
        return torch.tensor(roadgraph_points, dtype=torch.float32)
    
    def create_highway_merge(self, agents, traffic_lights):
        """Create highway merge scenario"""
        # Highway traffic (higher speeds)
        speeds = [25, 30, 35]  # m/s (55-80 mph)
        lanes = [-5, 0, 5]  # 3 lanes
        
        for i in range(min(30, self.max_agents)):
            lane = np.random.choice(lanes)
            speed = np.random.choice(speeds)
            spawn_time = np.random.randint(0, 20)
            
            for t in range(spawn_time, min(spawn_time + 60, self.timesteps)):
                progress = (t - spawn_time) * speed
                
                agents[i, t, 0] = 1.0  # valid
                agents[i, t, 1] = -150 + progress  # x
                agents[i, t, 2] = lane  # y
                agents[i, t, 4] = 0.0  # heading (east)
                agents[i, t, 5] = speed  # vx
                agents[i, t, 7:10] = torch.tensor([4.5, 2.0, 1.5])  # size
                agents[i, t, 10] = 1  # vehicle type
    
    def create_roundabout(self, agents, traffic_lights):
        """Create roundabout scenario"""
        radius = 20
        
        for i in range(min(15, self.max_agents)):
            # Entry angle
            entry_angle = np.random.uniform(0, 2 * np.pi)
            spawn_time = np.random.randint(0, 30)
            
            for t in range(spawn_time, min(spawn_time + 50, self.timesteps)):
                # Circular motion
                angle = entry_angle + (t - spawn_time) * 0.05
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                agents[i, t, 0] = 1.0
                agents[i, t, 1] = x
                agents[i, t, 2] = y
                agents[i, t, 4] = angle + np.pi/2  # Tangent direction
                agents[i, t, 5] = -radius * 0.05 * np.sin(angle)  # vx
                agents[i, t, 6] = radius * 0.05 * np.cos(angle)   # vy
                agents[i, t, 7:10] = torch.tensor([4.5, 2.0, 1.5])
                agents[i, t, 10] = 1

def test_realistic_synthetic():
    """Test the realistic synthetic generator"""
    print("=== Testing Realistic WOMD Synthetic Generator ===\n")
    
    generator = RealisticWOMDGenerator()
    scenarios = generator.create_womd_like_dataset(num_scenarios=20)
    
    # Analyze first scenario
    scenario = scenarios[0]
    agents = scenario['agents']
    
    print(f"\nAnalyzing scenario: {scenario['scenario_id']}")
    print(f"Agents shape: {agents.shape}")
    print(f"Traffic lights shape: {scenario['traffic_lights'].shape}")
    print(f"Roadgraph shape: {scenario['roadgraph'].shape}")
    
    # Statistics
    validity = agents[:, :, 0]
    valid_observations = (validity > 0).sum().item()
    total_possible = agents.shape[0] * agents.shape[1]
    
    print(f"\nRealism metrics:")
    print(f"  Valid observations: {valid_observations}/{total_possible} ({valid_observations/total_possible:.1%})")
    
    # Check speeds
    speeds = torch.sqrt(agents[:, :, 5]**2 + agents[:, :, 6]**2)
    valid_speeds = speeds[validity > 0]
    if len(valid_speeds) > 0:
        print(f"  Speed range: {valid_speeds.min():.1f} - {valid_speeds.max():.1f} m/s")
        print(f"  Average speed: {valid_speeds.mean():.1f} m/s")
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    # Validity heatmap
    plt.subplot(1, 3, 1)
    plt.imshow(validity[:20].numpy(), aspect='auto', cmap='RdYlGn')
    plt.title('Agent Validity (First 20 agents)')
    plt.xlabel('Time')
    plt.ylabel('Agent ID')
    
    # Trajectories
    plt.subplot(1, 3, 2)
    for i in range(min(10, agents.shape[0])):
        valid_mask = validity[i] > 0
        if valid_mask.sum() > 5:
            x = agents[i, valid_mask, 1]
            y = agents[i, valid_mask, 2]
            plt.plot(x, y, '-', alpha=0.7)
    
    plt.title('Agent Trajectories')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Speed over time
    plt.subplot(1, 3, 3)
    for i in range(min(5, agents.shape[0])):
        valid_mask = validity[i] > 0
        if valid_mask.sum() > 5:
            speeds_i = speeds[i, valid_mask]
            plt.plot(speeds_i, alpha=0.7)
    
    plt.title('Speed Profiles')
    plt.xlabel('Time')
    plt.ylabel('Speed (m/s)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realistic_womd_synthetic.png')
    print(f"\n✅ Saved visualization to realistic_womd_synthetic.png")

if __name__ == "__main__":
    test_realistic_synthetic()
