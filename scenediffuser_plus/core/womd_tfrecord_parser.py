#!/usr/bin/env python3
"""
Parse Waymo Open Motion Dataset TFRecords
"""

import tensorflow as tf
import numpy as np
import torch
from typing import Dict, List, Tuple

# Import Waymo dataset proto
try:
    from waymo_open_dataset.protos import scenario_pb2
except ImportError:
    print("Please install waymo-open-dataset-tf-2-11-0")
    print("pip install waymo-open-dataset-tf-2-11-0")

class WOMDParser:
    """Parser for WOMD tfrecord files"""
    
    def __init__(self, max_agents=128, max_traffic_lights=32):
        self.max_agents = max_agents
        self.max_traffic_lights = max_traffic_lights
        
    def parse_scenario(self, data: bytes) -> Dict[str, torch.Tensor]:
        """Parse a single scenario from tfrecord"""
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data)
        
        # Get scenario info
        scenario_id = scenario.scenario_id
        timesteps = len(scenario.timestamps_seconds)
        
        # Initialize tensors
        agents = torch.zeros(self.max_agents, timesteps, 10)
        traffic_lights = torch.zeros(self.max_traffic_lights, timesteps, 5)
        
        # Parse tracks (agents)
        for track_idx, track in enumerate(scenario.tracks):
            if track_idx >= self.max_agents:
                break
                
            # Get track states
            for state in track.states:
                t = state.timestep
                if t < timesteps:
                    # Validity
                    agents[track_idx, t, 0] = state.valid
                    
                    # Position
                    agents[track_idx, t, 1] = state.center_x
                    agents[track_idx, t, 2] = state.center_y
                    agents[track_idx, t, 3] = state.center_z
                    
                    # Heading
                    agents[track_idx, t, 4] = state.heading
                    
                    # Size
                    agents[track_idx, t, 5] = state.length
                    agents[track_idx, t, 6] = state.width
                    agents[track_idx, t, 7] = state.height
                    
                    # Type
                    agents[track_idx, t, 8] = track.object_type
        
        # Parse dynamic map states (traffic lights)
        light_idx = 0
        for map_state in scenario.dynamic_map_states:
            for lane_state in map_state.lane_states:
                if light_idx >= self.max_traffic_lights:
                    break
                    
                # Traffic light state
                traffic_lights[light_idx, :, 0] = 1  # validity
                traffic_lights[light_idx, :, 4] = lane_state.state  # signal state
                
                # You'll need to get position from map features
                # This is simplified
                traffic_lights[light_idx, :, 1:3] = 0  # x, y position
                
                light_idx += 1
        
        # Parse map features
        roadgraph = self._parse_roadgraph(scenario)
        
        return {
            'agents': agents,
            'traffic_lights': traffic_lights,
            'roadgraph': roadgraph,
            'scenario_id': scenario_id,
            'timesteps': timesteps
        }
    
    def _parse_roadgraph(self, scenario) -> torch.Tensor:
        """Parse roadgraph polylines"""
        roadgraph_features = []
        
        for map_feature in scenario.map_features:
            for point in map_feature.polyline:
                # Extract point features
                features = [
                    point.x,
                    point.y,
                    point.z,
                    map_feature.type,  # lane type
                    len(map_feature.polyline),  # polyline length
                    1.0 if map_feature.HasField('speed_limit') else 0.0,
                    map_feature.speed_limit if map_feature.HasField('speed_limit') else 0.0
                ]
                roadgraph_features.append(features)
        
        # Limit to 1000 points
        if len(roadgraph_features) > 1000:
            roadgraph_features = roadgraph_features[:1000]
        elif len(roadgraph_features) < 1000:
            # Pad with zeros
            padding = [[0]*7] * (1000 - len(roadgraph_features))
            roadgraph_features.extend(padding)
        
        return torch.tensor(roadgraph_features, dtype=torch.float32)


def test_parser():
    """Test the parser with sample data"""
    import glob
    
    parser = WOMDParser()
    
    # Find a sample tfrecord
    tfrecords = glob.glob('data/womd_sample/**/*.tfrecord*', recursive=True)
    
    if not tfrecords:
        print("No tfrecord files found. Please download sample data first.")
        return
    
    print(f"Found {len(tfrecords)} tfrecord files")
    print(f"Testing with: {tfrecords[0]}")
    
    # Read and parse one example
    dataset = tf.data.TFRecordDataset(tfrecords[0])
    
    for data in dataset.take(1):
        result = parser.parse_scenario(data.numpy())
        
        print("\nParsed scenario:")
        print(f"  Scenario ID: {result['scenario_id']}")
        print(f"  Timesteps: {result['timesteps']}")
        print(f"  Agents shape: {result['agents'].shape}")
        print(f"  Traffic lights shape: {result['traffic_lights'].shape}")
        print(f"  Roadgraph shape: {result['roadgraph'].shape}")
        
        # Check validity
        valid_agents = (result['agents'][:, :, 0] > 0).sum()
        print(f"  Valid agent observations: {valid_agents}")

if __name__ == "__main__":
    test_parser()
