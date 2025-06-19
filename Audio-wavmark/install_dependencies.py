#!/usr/bin/env python3
"""
WavMark Dependency Installation Script
"""

import sys
import subprocess
import importlib
import os

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 6):
        print("❌ Python 3.6 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_torch_cuda():
    """Install PyTorch with CUDA support."""
    print("Installing PyTorch with CUDA support...")
    success = install_package("torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
    if success:
        print("✅ PyTorch CUDA installed successfully")
    else:
        print("❌ PyTorch CUDA installation failed")
    return success

def install_core_dependencies():
    """Install core dependencies."""
    core_packages = [
        "numpy>=1.19.0",
        "tqdm>=4.60.0",
        "huggingface_hub>=0.10.0",
        "librosa>=0.8.0",
        "resampy>=0.3.0",
        "soundfile>=0.10.0",
        "pydub>=0.25.0"
    ]
    
    print("Installing core dependencies...")
    for package in core_packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed")
        else:
            print(f"❌ {package} installation failed")
            return False
    return True

def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("⚠️  FFmpeg not found. Audio format conversion may not work.")
    print("   Install FFmpeg from: https://ffmpeg.org/download.html")
    return False

def main():
    """Main installation function."""
    print("WavMark Dependency Installation")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install PyTorch with CUDA
    if not check_package("torch"):
        install_torch_cuda()
    else:
        print("✅ PyTorch already installed")
    
    # Install core dependencies
    if not install_core_dependencies():
        print("❌ Core dependencies installation failed")
        sys.exit(1)
    
    # Check FFmpeg
    check_ffmpeg()
    
    print("\n" + "=" * 40)
    print("✅ Installation completed!")
    print("\nNext steps:")
    print("1. Run: python test_installation.py")
    print("2. Run: python quick_test.py")
    print("\nFor help, see README.md")

if __name__ == "__main__":
    main() 