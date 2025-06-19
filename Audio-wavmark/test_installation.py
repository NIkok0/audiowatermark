#!/usr/bin/env python3
"""
WavMark Installation and Functionality Test
"""

import sys
import os
import numpy as np
import torch
import importlib

def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 6):
        print("Python 3.6 or higher is required")
        return False
    return True

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        return True
    except ImportError as e:
        print(f"{package_name} installation failed: {e}")
        return False

def check_torch_cuda():
    """Check PyTorch CUDA support."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.version.cuda}")
            return True
        else:
            print("CUDA not available (CPU-only mode)")
            return False
    except Exception as e:
        print(f"CUDA check failed: {e}")
        return False

def test_wavmark_import():
    """Test WavMark import."""
    try:
        from wavmark import load_model, encode_watermark, decode_watermark
        return True
    except ImportError as e:
        print(f"WavMark import failed: {e}")
        return False

def test_audio_processing():
    """Test audio processing functionality."""
    try:
        import librosa
        import soundfile as sf
        from quick_test import process_audio_file
        
        # Create test audio
        duration = 3.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        test_file = os.path.join(os.getcwd(), "test_audio.wav")
        sf.write(test_file, audio, sample_rate)
        
        # Test processing
        success, result = process_audio_file(test_file)
        
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
        if success and result and 'output_file' in result and os.path.exists(result['output_file']):
            os.remove(result['output_file'])
            
        return success
        
    except Exception as e:
        print(f"Audio processing test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("WavMark Installation Test")
    
    all_passed = True
    
    # Check Python version
    all_passed &= check_python_version()
    
    # Check core dependencies
    all_passed &= check_package("torch")
    all_passed &= check_package("torchaudio")
    all_passed &= check_package("numpy")
    all_passed &= check_package("librosa")
    all_passed &= check_package("soundfile")
    all_passed &= check_package("pydub")
    
    # Check CUDA support
    all_passed &= check_torch_cuda()
    
    # Test WavMark import
    all_passed &= test_wavmark_import()
    
    # Test audio processing
    all_passed &= test_audio_processing()
    
    if all_passed:
        print("\nAll tests passed! WavMark is ready to use.")
    else:
        print("\nSome tests failed. Please check the installation.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check PyTorch installation: pip install torch torchaudio")
        print("3. Install ffmpeg and add to PATH")

if __name__ == "__main__":
    main() 