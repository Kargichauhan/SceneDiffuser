#!/usr/bin/env python3
"""
Download a small sample of WOMD for testing
"""

import os
import subprocess
import sys

def setup_gsutil():
    """Check if gsutil is installed"""
    try:
        subprocess.run(['gsutil', '--version'], check=True, capture_output=True)
        print("✓ gsutil is installed")
        return True
    except:
        print("✗ gsutil not found. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'gsutil'], check=True)
        return True

def download_sample():
    """Download a small sample of WOMD data"""
    print("\n=== Downloading WOMD Sample Data ===\n")
    
    # Create data directory
    os.makedirs('data/womd_sample', exist_ok=True)
    
    # First, you need to authenticate
    print("Step 1: Authenticate with Google Cloud")
    print("Run: gcloud auth login")
    print("(Skip if already authenticated)\n")
    
    input("Press Enter when ready to continue...")
    
    # Download just 5 files for testing
    print("\nStep 2: Downloading sample files...")
    
    sample_files = [
        "training.tfrecord-00000-of-01000",
        "training.tfrecord-00001-of-01000",
        "training.tfrecord-00002-of-01000",
        "validation.tfrecord-00000-of-00150",
        "validation.tfrecord-00001-of-00150"
    ]
    
    base_url = "gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example"
    
    for file in sample_files:
        if 'training' in file:
            src = f"{base_url}/training/{file}"
            dst = f"data/womd_sample/training/"
        else:
            src = f"{base_url}/validation/{file}"
            dst = f"data/womd_sample/validation/"
        
        os.makedirs(dst, exist_ok=True)
        
        print(f"Downloading {file}...")
        try:
            subprocess.run(['gsutil', 'cp', src, dst], check=True)
            print(f"✓ Downloaded {file}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download {file}")
            print("Make sure you have access to Waymo dataset")
            print("Register at: https://waymo.com/open/data/motion/")
            return False
    
    print("\n✓ Sample data downloaded successfully!")
    print(f"Location: {os.path.abspath('data/womd_sample')}")
    return True

def install_tf_example_parser():
    """Install dependencies for parsing tfrecords"""
    print("\nInstalling TensorFlow for tfrecord parsing...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'tensorflow>=2.13.0'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'waymo-open-dataset-tf-2-11-0'], check=True)

if __name__ == "__main__":
    if setup_gsutil():
        if download_sample():
            install_tf_example_parser()
            print("\n✅ Setup complete! You can now parse WOMD data.")
        else:
            print("\n❌ Download failed. Please check your Waymo dataset access.")
