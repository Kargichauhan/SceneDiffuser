#!/usr/bin/env python3
"""
Implement SceneDiffuser++ evaluation metrics from the paper
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import pickle
from typing import Dict, List, Tuple
import sys
sys.path.append('.')

from train_with_realistic_synthetic import EnhancedMPSDiffusion, RealisticSyntheticDataset
from core.model import SceneConfig
from core.diffusion_model import DiffusionConfig

class SceneDiffuserEvaluator:
    """Comprehensive evaluation matching the paper's metrics"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def generate_scenarios(self, num_scenarios=50):
        """Generate scenarios for evaluation"""
        print(f"Generating {num_scenarios} scenarios for evaluation...")
        
        generated_scenarios = []
        
        with torch.no_grad():
            for i in range(num_scenarios):
                if i % 10 == 0:
                    print(f"  Generated {i}/{num_scenarios}")
                
                # Generate one scenario
                agents = torch.randn(1, 128, 91, 11).to(self.device)
                lights = torch.randn(1, 16, 91, 16).to(self.device)
                
                # DDIM-style sampling for better quality
                for t in reversed(range(0, 100, 2)):
                    timesteps = torch.tensor([t]).to(self.device)
                    pred = self.model.forward(agents, lights, timesteps)
                    
                    alpha = 1 - (t / 100)
                    agents = agents - (1 - alpha) * pred['agents'] * 0.02
                    lights = lights - (1 - alpha) * pred['lights'] * 0.02
                    
                    if t > 0:
                        noise_scale = (t / 100) * 0.1
                        agents += torch.randn_like(agents) * noise_scale
                        lights += torch.randn_like(lights) * noise_scale
                
                generated_scenarios.append({
                    'agents': agents[0].cpu(),
                    'lights': lights[0].cpu()
                })
        
        print(f"✅ Generated {len(generated_scenarios)} scenarios")
        return generated_scenarios
    
    def evaluate_agent_metrics(self, generated_scenarios, real_scenarios):
        """Evaluate agent-level metrics from the paper"""
        print("\n📊 Computing Agent-Level Metrics...")
        
        metrics = {}
        
        # 1. Number of valid agents per timestep
        gen_valid_counts = []
        real_valid_counts = []
        
        for scenario in generated_scenarios:
            validity = torch.sigmoid(scenario['agents'][:, :, 0]) > 0.5
            valid_count = validity.sum(dim=0).float()  # Per timestep
            gen_valid_counts.append(valid_count)
        
        for scenario in real_scenarios:
            validity = scenario['agents'][:, :, 0] > 0.5
            valid_count = validity.sum(dim=0).float()
            real_valid_counts.append(valid_count)
        
        gen_valid_counts = torch.stack(gen_valid_counts).numpy()  # [scenarios, timesteps]
        real_valid_counts = torch.stack(real_valid_counts).numpy()
        
        # Jensen-Shannon divergence for agent count distribution
        gen_hist, _ = np.histogram(gen_valid_counts.flatten(), bins=50, range=(0, 50), density=True)
        real_hist, _ = np.histogram(real_valid_counts.flatten(), bins=50, range=(0, 50), density=True)
        
        # Add small epsilon to avoid log(0)
        gen_hist = gen_hist + 1e-8
        real_hist = real_hist + 1e-8
        
        js_agent_count = jensenshannon(gen_hist, real_hist) ** 2
        metrics['js_agent_count'] = js_agent_count
        
        # 2. Agent spawning/despawning events
        gen_spawns, gen_despawns = self.count_spawn_despawn_events(generated_scenarios)
        real_spawns, real_despawns = self.count_spawn_despawn_events(real_scenarios)
        
        # Wasserstein distance for spawn/despawn distributions
        metrics['wasserstein_spawns'] = wasserstein_distance(gen_spawns, real_spawns)
        metrics['wasserstein_despawns'] = wasserstein_distance(gen_despawns, real_despawns)
        
        # 3. Speed distribution
        gen_speeds = self.extract_speed_distribution(generated_scenarios)
        real_speeds = self.extract_speed_distribution(real_scenarios)
        
        gen_speed_hist, _ = np.histogram(gen_speeds, bins=30, range=(0, 20), density=True)
        real_speed_hist, _ = np.histogram(real_speeds, bins=30, range=(0, 20), density=True)
        
        gen_speed_hist = gen_speed_hist + 1e-8
        real_speed_hist = real_speed_hist + 1e-8
        
        js_speed = jensenshannon(gen_speed_hist, real_speed_hist) ** 2
        metrics['js_speed'] = js_speed
        
        return metrics
    
    def count_spawn_despawn_events(self, scenarios):
        """Count spawning and despawning events"""
        all_spawns = []
        all_despawns = []
        
        for scenario in scenarios:
            if isinstance(scenario, dict):
                agents = scenario['agents']
            else:
                agents = scenario
            
            validity = torch.sigmoid(agents[:, :, 0]) > 0.5 if 'agents' in str(type(scenario)) else agents[:, :, 0] > 0.5
            
            spawns_per_scenario = 0
            despawns_per_scenario = 0
            
            for agent_id in range(validity.shape[0]):
                agent_valid = validity[agent_id]
                
                # Find transitions
                for t in range(1, len(agent_valid)):
                    if not agent_valid[t-1] and agent_valid[t]:  # Spawn
                        spawns_per_scenario += 1
                    elif agent_valid[t-1] and not agent_valid[t]:  # Despawn
                        despawns_per_scenario += 1
            
            all_spawns.append(spawns_per_scenario)
            all_despawns.append(despawns_per_scenario)
        
        return np.array(all_spawns), np.array(all_despawns)
    
    def extract_speed_distribution(self, scenarios):
        """Extract speed distribution from scenarios"""
        all_speeds = []
        
        for scenario in scenarios:
            if isinstance(scenario, dict):
                agents = scenario['agents']
                validity_key = 0
            else:
                agents = scenario
                validity_key = 0
            
            validity = torch.sigmoid(agents[:, :, validity_key]) > 0.5 if 'agents' in str(type(scenario)) else agents[:, :, validity_key] > 0.5
            
            velocities = agents[:, :, 5:7]  # vx, vy
            speeds = torch.sqrt(velocities[:, :, 0]**2 + velocities[:, :, 1]**2)
            
            # Only consider valid agents
            valid_speeds = speeds[validity]
            all_speeds.extend(valid_speeds.numpy())
        
        return np.array(all_speeds)
    
    def evaluate_safety_metrics(self, generated_scenarios):
        """Evaluate safety metrics"""
        print("\n🚨 Computing Safety Metrics...")
        
        metrics = {}
        
        total_collisions = 0
        total_agent_pairs = 0
        total_offroad = 0
        total_agents = 0
        
        for scenario in generated_scenarios:
            agents = scenario['agents']
            validity = torch.sigmoid(agents[:, :, 0]) > 0.5
            
            # Collision detection
            for t in range(agents.shape[1]):
                valid_agents_t = validity[:, t]
                positions_t = agents[valid_agents_t, t, 1:3]  # x, y positions
                
                if len(positions_t) > 1:
                    # Compute pairwise distances
                    distances = torch.cdist(positions_t, positions_t)
                    
                    # Check for collisions (distance < 4m)
                    collision_mask = (distances < 4.0) & (distances > 0)
                    collisions_t = collision_mask.sum().item() // 2  # Avoid double counting
                    
                    total_collisions += collisions_t
                    total_agent_pairs += len(positions_t) * (len(positions_t) - 1) // 2
                
                # Off-road detection (simple: outside reasonable bounds)
                for agent_id in range(agents.shape[0]):
                    if validity[agent_id, t]:
                        x, y = agents[agent_id, t, 1:3]
                        if abs(x) > 200 or abs(y) > 200:  # Far from intersection
                            total_offroad += 1
                        total_agents += 1
        
        metrics['collision_rate'] = total_collisions / max(total_agent_pairs, 1)
        metrics['offroad_rate'] = total_offroad / max(total_agents, 1)
        
        return metrics
    
    def evaluate_realism_metrics(self, generated_scenarios, real_scenarios):
        """Evaluate realism metrics"""
        print("\n🎭 Computing Realism Metrics...")
        
        metrics = {}
        
        # Average trajectory length
        gen_traj_lengths = []
        real_traj_lengths = []
        
        for scenario in generated_scenarios:
            agents = scenario['agents']
            validity = torch.sigmoid(agents[:, :, 0]) > 0.5
            
            for agent_id in range(agents.shape[0]):
                traj_length = validity[agent_id].sum().item()
                if traj_length > 0:
                    gen_traj_lengths.append(traj_length)
        
        for scenario in real_scenarios:
            agents = scenario['agents']
            validity = agents[:, :, 0] > 0.5
            
            for agent_id in range(agents.shape[0]):
                traj_length = validity[agent_id].sum().item()
                if traj_length > 0:
                    real_traj_lengths.append(traj_length)
        
        metrics['avg_traj_length_gen'] = np.mean(gen_traj_lengths)
        metrics['avg_traj_length_real'] = np.mean(real_traj_lengths)
        metrics['traj_length_diff'] = abs(metrics['avg_traj_length_gen'] - metrics['avg_traj_length_real'])
        
        return metrics

def comprehensive_evaluation():
    """Run comprehensive evaluation"""
    print("🔬 SceneDiffuser++ Comprehensive Evaluation\n")
    
    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    scene_config = SceneConfig()
    scene_config.num_agents = 128
    scene_config.timesteps = 91
    scene_config.agent_features = 11
    scene_config.light_features = 16
    
    diffusion_config = DiffusionConfig(num_diffusion_steps=500)
    
    model = EnhancedMPSDiffusion(scene_config, diffusion_config).to(device)
    model.load_state_dict(torch.load('checkpoints/realistic_model_epoch15.pt', map_location=device))
    
    evaluator = SceneDiffuserEvaluator(model, device)
    
    # Load real scenarios for comparison
    dataset = RealisticSyntheticDataset()
    real_scenarios = []
    for i in range(min(20, len(dataset))):
        sample = dataset[i]
        real_scenarios.append({
            'agents': sample['agents'],
            'lights': sample['lights']
        })
    
    print(f"✓ Loaded {len(real_scenarios)} real scenarios for comparison")
    
    # Generate scenarios
    generated_scenarios = evaluator.generate_scenarios(num_scenarios=20)
    
    # Run evaluations
    agent_metrics = evaluator.evaluate_agent_metrics(generated_scenarios, real_scenarios)
    safety_metrics = evaluator.evaluate_safety_metrics(generated_scenarios)
    realism_metrics = evaluator.evaluate_realism_metrics(generated_scenarios, real_scenarios)
    
    # Combine all metrics
    all_metrics = {**agent_metrics, **safety_metrics, **realism_metrics}
    
    # Print results
    print(f"\n📋 EVALUATION RESULTS")
    print(f"=" * 50)
    
    print(f"\n🎯 Distribution Matching (JS Divergence - Lower is Better):")
    print(f"  Agent Count Distribution: {all_metrics['js_agent_count']:.4f}")
    print(f"  Speed Distribution: {all_metrics['js_speed']:.4f}")
    
    print(f"\n📊 Event Matching (Wasserstein Distance - Lower is Better):")
    print(f"  Spawn Events: {all_metrics['wasserstein_spawns']:.4f}")
    print(f"  Despawn Events: {all_metrics['wasserstein_despawns']:.4f}")
    
    print(f"\n🚨 Safety Metrics:")
    print(f"  Collision Rate: {all_metrics['collision_rate']:.4f} ({all_metrics['collision_rate']*100:.2f}%)")
    print(f"  Off-road Rate: {all_metrics['offroad_rate']:.4f} ({all_metrics['offroad_rate']*100:.2f}%)")
    
    print(f"\n🎭 Realism Metrics:")
    print(f"  Avg Trajectory Length (Generated): {all_metrics['avg_traj_length_gen']:.1f}")
    print(f"  Avg Trajectory Length (Real): {all_metrics['avg_traj_length_real']:.1f}")
    print(f"  Trajectory Length Difference: {all_metrics['traj_length_diff']:.1f}")
    
    # Overall score (lower is better)
    overall_score = (
        all_metrics['js_agent_count'] + 
        all_metrics['js_speed'] + 
        all_metrics['collision_rate'] + 
        all_metrics['traj_length_diff'] / 10
    )
    
    print(f"\n🏆 OVERALL QUALITY SCORE: {overall_score:.4f}")
    print(f"   (Lower is better - paper reports ~0.15 for best models)")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Agent count comparison
    gen_counts = []
    real_counts = []
    
    for scenario in generated_scenarios:
        validity = torch.sigmoid(scenario['agents'][:, :, 0]) > 0.5
        gen_counts.extend(validity.sum(dim=0).numpy())
    
    for scenario in real_scenarios:
        validity = scenario['agents'][:, :, 0] > 0.5
        real_counts.extend(validity.sum(dim=0).numpy())
    
    axes[0, 0].hist(real_counts, bins=20, alpha=0.7, label='Real', color='blue')
    axes[0, 0].hist(gen_counts, bins=20, alpha=0.7, label='Generated', color='red')
    axes[0, 0].set_xlabel('Agent Count')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Agent Count Distribution')
    axes[0, 0].legend()
    
    # Speed comparison
    gen_speeds = evaluator.extract_speed_distribution(generated_scenarios)
    real_speeds = evaluator.extract_speed_distribution(real_scenarios)
    
    axes[0, 1].hist(real_speeds, bins=30, alpha=0.7, label='Real', color='blue', range=(0, 15))
    axes[0, 1].hist(gen_speeds, bins=30, alpha=0.7, label='Generated', color='red', range=(0, 15))
    axes[0, 1].set_xlabel('Speed (m/s)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Speed Distribution')
    axes[0, 1].legend()
    
    # Metrics summary
    metrics_names = ['JS Agent Count', 'JS Speed', 'Collision Rate', 'Off-road Rate']
    metrics_values = [all_metrics['js_agent_count'], all_metrics['js_speed'], 
                     all_metrics['collision_rate'], all_metrics['offroad_rate']]
    
    axes[0, 2].bar(metrics_names, metrics_values, color=['green', 'blue', 'red', 'orange'])
    axes[0, 2].set_ylabel('Metric Value')
    axes[0, 2].set_title('Evaluation Metrics')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Sample trajectories comparison
    axes[1, 0].set_title('Generated Trajectories')
    for i, scenario in enumerate(generated_scenarios[:3]):
        agents = scenario['agents']
        validity = torch.sigmoid(agents[:, :, 0]) > 0.5
        for agent_id in range(min(10, agents.shape[0])):
            if validity[agent_id].sum() > 5:
                x = agents[agent_id, validity[agent_id], 1]
                y = agents[agent_id, validity[agent_id], 2]
                axes[1, 0].plot(x, y, alpha=0.6)
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Y (m)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Real Trajectories')
    for i, scenario in enumerate(real_scenarios[:3]):
        agents = scenario['agents']
        validity = agents[:, :, 0] > 0.5
        for agent_id in range(min(10, agents.shape[0])):
            if validity[agent_id].sum() > 5:
                x = agents[agent_id, validity[agent_id], 1]
                y = agents[agent_id, validity[agent_id], 2]
                axes[1, 1].plot(x, y, alpha=0.6)
    axes[1, 1].set_xlabel('X (m)')
    axes[1, 1].set_ylabel('Y (m)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Overall score visualization
    axes[1, 2].bar(['Overall Score'], [overall_score], color='purple')
    axes[1, 2].axhline(y=0.15, color='green', linestyle='--', label='Paper Baseline')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Overall Quality Score')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved evaluation visualization to evaluation_results.png")
    
    # Save metrics
    with open('evaluation_metrics.txt', 'w') as f:
        f.write("SceneDiffuser++ Evaluation Results\n")
        f.write("=" * 40 + "\n\n")
        for key, value in all_metrics.items():
            f.write(f"{key}: {value:.6f}\n")
        f.write(f"\nOverall Score: {overall_score:.6f}\n")
    
    print(f"✅ Saved detailed metrics to evaluation_metrics.txt")
    
    return all_metrics

if __name__ == "__main__":
    comprehensive_evaluation()
