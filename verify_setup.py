#!/usr/bin/env python3
"""
Setup verification script for Gender Bias Analysis Pipeline
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_gpu_availability():
    """Check if CUDA/GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available: {gpu_count} GPU(s)")
            print(f"   Primary GPU: {gpu_name}")
            return True
        else:
            print("⚠️  CUDA not available, will use CPU")
            return True
    except ImportError:
        print("⚠️  PyTorch not installed, cannot check GPU")
        return False

def check_required_packages():
    """Check if required packages are installed."""
    required_packages = [
        'torch',
        'torchvision', 
        'transformers',
        'datasets',
        'pandas',
        'numpy',
        'opencv-contrib-python',
        'Pillow',
        'matplotlib',
        'seaborn',
        'plotly',
        'requests',
        'tqdm',
        'python-dotenv',
        'pyyaml',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            # Handle package name variations
            if package == 'opencv-contrib-python':
                importlib.import_module('cv2')
            elif package == 'python-dotenv':
                importlib.import_module('dotenv')
            elif package == 'pyyaml':
                importlib.import_module('yaml')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt --break-system-packages")
        return False
    
    print("\n✅ All required packages installed")
    return True

def check_environment_file():
    """Check if .env file exists and has required variables."""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("❌ .env file not found")
        print("   Create .env file with your Hugging Face token")
        return False
    
    required_vars = ['HUGGINGFACE_TOKEN']
    missing_vars = []
    
    with open(env_file, 'r') as f:
        content = f.read()
    
    for var in required_vars:
        if f"{var}=" not in content or f"{var}=your_" in content:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing or unconfigured environment variables: {', '.join(missing_vars)}")
        print("   Please update your .env file with actual Hugging Face token")
        return False
    
    print("✅ Environment file configured")
    return True

def check_directory_structure():
    """Check if required directories exist."""
    required_dirs = [
        'src',
        'config', 
        'data',
        'results',
        'logs'
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if not Path(directory).exists():
            missing_dirs.append(directory)
        else:
            print(f"✅ {directory}/")
    
    if missing_dirs:
        print(f"❌ Missing directories: {', '.join(missing_dirs)}")
        return False
    
    print("✅ Directory structure complete")
    return True

def check_config_files():
    """Check if configuration files exist."""
    config_files = [
        'config/config.yaml',
        'requirements.txt',
        'main.py'
    ]
    
    missing_files = []
    
    for file_path in config_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing configuration files: {', '.join(missing_files)}")
        return False
    
    print("✅ Configuration files present")
    return True

def test_api_connections():
    """Test API connections and dataset access."""
    print("\n🔗 Testing dataset and API connections...")
    
    # Test Hugging Face
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("✅ Hugging Face connection OK")
        hf_ok = True
    except Exception as e:
        print(f"❌ Hugging Face connection failed: {str(e)}")
        hf_ok = False
    
    # Test Datasets library
    try:
        from datasets import load_dataset
        print("✅ Datasets library OK")
        datasets_ok = True
    except Exception as e:
        print(f"❌ Datasets library failed: {str(e)}")
        datasets_ok = False
    
    # Test Flickr30k dataset access (without downloading)
    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names("nlphuji/flickr30k")
        print("✅ Flickr30k dataset accessible")
        dataset_ok = True
    except Exception as e:
        print(f"⚠️  Flickr30k dataset check failed: {str(e)}")
        print("   This might be OK - dataset will be downloaded when needed")
        dataset_ok = True  # Don't fail on this
    
    return hf_ok and datasets_ok and dataset_ok

def run_verification():
    """Run complete verification."""
    print("🔍 Gender Bias Analysis Pipeline - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("GPU Availability", check_gpu_availability),
        ("Directory Structure", check_directory_structure),
        ("Configuration Files", check_config_files),
        ("Environment Variables", check_environment_file),
        ("API Connections", test_api_connections)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_function in checks:
        print(f"\n📋 {check_name}:")
        if check_function():
            passed_checks += 1
        print("-" * 40)
    
    print(f"\n📊 Verification Results: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("🎉 Setup verification successful! You can now run the pipeline.")
        print("\nNext steps:")
        print("1. python main.py --help  # View available options")
        print("2. python main.py         # Run full pipeline")
        return True
    else:
        print("⚠️  Setup verification failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("1. pip install -r requirements.txt --break-system-packages")
        print("2. Configure your .env file with actual API credentials")
        print("3. Check internet connection for downloading models")
        return False

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)