#!/usr/bin/env python3
"""Setup script for Decentralized AI Systems.

This script helps set up the environment and run initial tests.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Decentralized AI Systems...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Create virtual environment if it doesn't exist
    venv_path = Path("venv")
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        if not run_command("python -m venv venv", "Virtual environment creation"):
            sys.exit(1)
    
    # Determine activation command based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install dependencies
    print("📦 Installing dependencies...")
    if not run_command(f"{pip_cmd} install --upgrade pip", "Pip upgrade"):
        sys.exit(1)
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Dependencies installation"):
        print("⚠️  Some dependencies may have failed to install")
        print("   This is normal for optional dependencies like TensorRT, OpenVINO, etc.")
    
    # Create necessary directories
    print("📁 Creating project directories...")
    directories = [
        "data/raw",
        "data/processed", 
        "assets/models",
        "assets/evaluation",
        "outputs",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created")
    
    # Run tests
    print("🧪 Running system tests...")
    if not run_command(f"{pip_cmd} install pytest", "Pytest installation"):
        print("⚠️  Pytest installation failed, skipping tests")
    else:
        if not run_command(f"{pip_cmd} run python tests/test_system.py", "System tests"):
            print("⚠️  Some tests failed, but the system may still work")
    
    print("=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Run training: python scripts/train_decentralized.py")
    print("3. Launch demo: streamlit run demo/streamlit_demo.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
