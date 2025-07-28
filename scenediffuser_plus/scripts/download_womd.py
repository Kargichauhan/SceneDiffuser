#!/usr/bin/env python3
"""
Download Waymo Open Motion Dataset
"""

import os
import subprocess
import argparse

def download_womd():
    """
    Instructions for downloading WOMD
    """
    print("=== Waymo Open Motion Dataset Download ===\n")
    
    print("1. Register at: https://waymo.com/open/data/motion/")
    print("2. Accept the license agreement")
    print("3. Download using gsutil:")
    print("\n   # Install gsutil if needed:")
    print("   pip install gsutil")
    print("\n   # Download training data:")
    print("   gsutil -m cp -r gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example/training data/womd_training/")
    print("\n   # Download validation data:")
    print("   gsutil -m cp -r gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example/validation data/womd_validation/")
    
    print("\n4. The dataset is large (~1TB), so you might want to start with a subset")
    print("\n5. For testing, download just a few files first:")
    print("   gsutil cp gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example/training/training.tfrecord-00000-of-01000 data/")

if __name__ == "__main__":
    download_womd()
