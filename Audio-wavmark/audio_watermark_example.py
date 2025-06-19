#!/usr/bin/env python3
"""
Simple audio watermarking example
Usage: python audio_watermark_example.py <input_audio_file>
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

def main():
    if len(sys.argv) != 2:
        print("Usage: python audio_watermark_example.py <input_audio_file>")
        print("Supported formats: mp3, mp4, wav, flac, m4a, aac, ogg")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)
    
    from quick_test import process_audio_file
    
    success, result = process_audio_file(input_file)
    
    if success:
        print(f"\nSuccess! Watermarked audio saved to: {result['output_file']}")
        print(f"Bit Error Rate: {result['ber']:.3f}")
        print(f"SNR: {result['snr']:.1f} dB")
    else:
        print("Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 