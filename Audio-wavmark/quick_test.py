#!/usr/bin/env python3
"""
Quick test script for WavMark functionality
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import torch
import shutil
import argparse
import time  # 添加time模块导入

def setup_ffmpeg():
    """Setup ffmpeg path for audio processing."""
    try:
        # First try to find ffmpeg in PATH
        if shutil.which('ffmpeg'):
            ffmpeg_path = shutil.which('ffmpeg')
            print(f"Found ffmpeg in PATH: {ffmpeg_path}")
            return ffmpeg_path
            
        # Check common ffmpeg paths
        ffmpeg_paths = [
            os.path.join(os.path.expanduser("~"), "AppData", "Local", "ffmpegio", "ffmpeg-downloader", "ffmpeg", "bin"),
            r"C:\ffmpeg\bin",
            r"C:\Program Files\ffmpeg\bin",
            r"C:\Program Files (x86)\ffmpeg\bin"
        ]
        
        for path in ffmpeg_paths:
            ffmpeg_exe = os.path.join(path, "ffmpeg.exe")
            if os.path.exists(ffmpeg_exe):
                print(f"Found ffmpeg at: {ffmpeg_exe}")
                # Add to PATH
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
                return ffmpeg_exe
        
        print("Warning: ffmpeg not found in common locations")
        print("Audio format conversion may not work properly")
        return None
        
    except Exception as e:
        print(f"Error setting up ffmpeg: {e}")
        return None

# Setup ffmpeg before importing pydub
ffmpeg_path = setup_ffmpeg()

# Now import pydub and configure it
from pydub import AudioSegment

if ffmpeg_path:
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    ffprobe_path = ffmpeg_path.replace('ffmpeg.exe', 'ffprobe.exe')
    if os.path.exists(ffprobe_path):
        AudioSegment.ffprobe = ffprobe_path
        print(f"Found ffprobe at: {ffprobe_path}")
    else:
        print(f"Warning: ffprobe not found at {ffprobe_path}")

def check_gpu():
    """Check if GPU is available."""
    if not torch.cuda.is_available():
        print("GPU not available! This program requires GPU to run.")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def parse_custom_payload(payload_str: str) -> np.ndarray:
    """Parse custom payload from string input."""
    try:
        # Try to parse as binary string (e.g., "1010101010101010")
        if len(payload_str) == 16 and all(c in '01' for c in payload_str):
            return np.array([int(c) for c in payload_str])
        
        # Try to parse as comma or space separated values
        separators = [',', ' ']
        for sep in separators:
            if sep in payload_str:
                values = [int(x.strip()) for x in payload_str.split(sep)]
                if len(values) == 16 and all(x in [0, 1] for x in values):
                    return np.array(values)
        
        raise ValueError("Invalid payload format")
        
    except Exception as e:
        print(f"Error parsing payload: {e}")
        print("Expected format: 16 binary digits (e.g., '1010101010101010')")
        print("Or comma/space separated values (e.g., '1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0')")
        return None

def display_payload_comparison(original_payload: np.ndarray, decoded_payload: np.ndarray, 
                             file_name: str = "Audio File"):
    """Display detailed payload comparison."""
    print(f"\n{'='*60}")
    print(f"水印载荷对比 - {file_name}")
    print(f"{'='*60}")
    
    # Display original payload
    print(f"原始载荷:    {original_payload}")
    print(f"原始载荷(二进制): {''.join(map(str, original_payload))}")
    print(f"原始载荷(十六进制): {format(int(''.join(map(str, original_payload)), 2), '04x')}")
    
    # Display decoded payload
    if decoded_payload is not None:
        print(f"解码载荷:    {decoded_payload}")
        print(f"解码载荷(二进制): {''.join(map(str, decoded_payload))}")
        print(f"解码载荷(十六进制): {format(int(''.join(map(str, decoded_payload)), 2), '04x')}")
        
        # Calculate metrics
        ber = np.mean(original_payload != decoded_payload)
        correct_bits = np.sum(original_payload == decoded_payload)
        total_bits = len(original_payload)
        
        print(f"\n对比结果:")
        print(f"  正确位数: {correct_bits}/{total_bits}")
        print(f"  错误位数: {total_bits - correct_bits}/{total_bits}")
        print(f"  比特错误率(BER): {ber:.3f} ({ber*100:.1f}%)")
        
        # Visual comparison
        print(f"\n位位对比:")
        print(f"  位置:     {' '.join(f'{i:2d}' for i in range(16))}")
        print(f"  原始:     {' '.join(f'{b:2d}' for b in original_payload)}")
        print(f"  解码:     {' '.join(f'{b:2d}' for b in decoded_payload)}")
        print(f"  匹配:     {' '.join('✓' if o == d else '✗' for o, d in zip(original_payload, decoded_payload))}")
        
        # Quality assessment
        if ber == 0:
            print(f"\n✅ 完美匹配 - 水印解码成功!")
        elif ber <= 0.01:
            print(f"\n✅ 优秀质量 - 水印解码成功 (BER < 1%)")
        elif ber <= 0.05:
            print(f"\n⚠️  良好质量 - 水印解码成功 (BER < 5%)")
        elif ber <= 0.1:
            print(f"\n⚠️  一般质量 - 水印解码部分成功 (BER < 10%)")
        else:
            print(f"\n❌ 质量较差 - 水印解码失败 (BER >= 10%)")
    else:
        print(f"❌ 解码失败 - 无法提取水印载荷")
    
    print(f"{'='*60}")

def load_audio_file(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load and process audio file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']:
        raise ValueError(f"Unsupported format: {file_ext}")
    
    # Try pydub for non-WAV formats
    if file_ext != '.wav':
        try:
            temp_dir = os.path.join(os.path.dirname(file_path), 'temp_audio')
            os.makedirs(temp_dir, exist_ok=True)
            temp_wav = os.path.join(temp_dir, f"temp_{Path(file_path).stem}.wav")
            
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_channels(1).set_frame_rate(target_sr)
            audio.export(temp_wav, format="wav")
            
            audio_data, sr = sf.read(temp_wav)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return audio_data, sr
        except Exception:
            pass
    
    # Fallback to librosa
    audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio, sr

def save_audio_file(audio: np.ndarray, file_path: str, sr: int = 16000):
    """Save audio file in appropriate format."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext != '.wav':
        try:
            temp_dir = os.path.join(os.path.dirname(file_path), 'temp_audio')
            os.makedirs(temp_dir, exist_ok=True)
            temp_wav = os.path.join(temp_dir, "temp_output.wav")
            
            # Normalize audio
            audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
            audio_int16 = (audio * 32767).astype(np.int16)
            
            sf.write(temp_wav, audio_int16, sr)
            audio_segment = AudioSegment.from_wav(temp_wav)
            
            # Format mapping
            format_map = {
                '.mp3': ('mp3', 'libmp3lame'),
                '.m4a': ('ipod', 'aac'),
                '.aac': ('adts', 'aac'),
                '.ogg': ('ogg', 'libvorbis'),
                '.flac': ('flac', 'flac')
            }
            
            output_format, codec = format_map.get(file_ext, ('wav', None))
            export_params = {
                'format': output_format,
                'codec': codec,
                'parameters': ["-b:a", "192k"] if codec in ['aac', 'libmp3lame'] else None
            }
            export_params = {k: v for k, v in export_params.items() if v is not None}
            audio_segment.export(file_path, **export_params)
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"Format conversion failed: {e}")
            wav_path = os.path.splitext(file_path)[0] + '.wav'
            sf.write(wav_path, audio, sr)
    else:
        sf.write(file_path, audio, sr)

def process_audio_file(input_file: str, output_file: str = None, payload: np.ndarray = None, 
                      show_comparison: bool = True) -> tuple[bool, dict]:
    """Process single audio file with watermark."""
    try:
        print(f"\n处理文件: {input_file}")
        audio, sr = load_audio_file(input_file)
        
        if payload is None:
            # 使用时间戳设置随机种子
            seed = int(time.time() * 1000) % (2**32)
            np.random.seed(seed)
            payload = np.random.randint(0, 2, 16)
            print(f"使用随机载荷: {payload}")
        else:
            print(f"使用自定义载荷: {payload}")
        
        from wavmark import load_model, encode_watermark, decode_watermark
        model = load_model(device="cuda")
        
        print("开始编码水印...")
        watermarked_audio, encode_info = encode_watermark(
            model=model,
            signal=audio,
            payload=payload,
            min_snr=25.0,
            max_snr=35.0,
            show_progress=True
        )
        
        if output_file is None:
            output_file = str(Path(input_file).parent / f"{Path(input_file).stem}_watermarked{Path(input_file).suffix}")
        
        print(f"保存水印音频: {output_file}")
        save_audio_file(watermarked_audio, output_file, sr)
        
        print("开始解码水印...")
        decoded_payload, decode_info = decode_watermark(model=model, signal=watermarked_audio, show_progress=True)
        
        ber = np.mean(payload != decoded_payload) if decoded_payload is not None else None
        
        # Display detailed comparison
        if show_comparison:
            display_payload_comparison(payload, decoded_payload, Path(input_file).name)
        
        result = {
            'input_file': input_file,
            'output_file': output_file,
            'payload': payload,
            'decoded_payload': decoded_payload,
            'ber': ber,
            'snr': encode_info['snr']
        }
        
        print(f"处理完成! SNR: {encode_info['snr']:.1f} dB")
        return True, result
        
    except Exception as e:
        print(f"处理失败: {e}")
        return False, None

def batch_process_from_directories(input_dir: str, output_dir: str, payload: np.ndarray = None, 
                                 show_comparison: bool = True) -> tuple[bool, list]:
    """Batch process audio files from directory."""
    try:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all audio files
        audio_files = set()
        for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']:
            audio_files.update(str(f) for f in Path(input_dir).glob(f"*{ext}"))
            audio_files.update(str(f) for f in Path(input_dir).glob(f"*{ext.upper()}"))
        
        audio_files = sorted(list(audio_files))
        if not audio_files:
            print("未找到音频文件!")
            return False, []
        
        print(f"找到 {len(audio_files)} 个音频文件")
        if payload is not None:
            print(f"将使用统一载荷: {payload}")
        
        results = []
        for i, file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] 处理文件...")
            output_file = os.path.join(output_dir, f"{Path(file).stem}_watermarked{Path(file).suffix}")
            success, result = process_audio_file(file, output_file, payload, show_comparison)
            if success:
                results.append(result)
        
        return True, results
        
    except Exception as e:
        print(f"批量处理失败: {e}")
        return False, []

def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="WavMark 音频水印处理工具")
    parser.add_argument("input", nargs="?", help="输入文件或目录路径")
    parser.add_argument("output", nargs="?", help="输出目录路径")
    parser.add_argument("--payload", "-p", help="自定义16位载荷 (二进制字符串，如: 1010101010101010)")
    parser.add_argument("--no-comparison", action="store_true", help="不显示详细对比信息")
    parser.add_argument("--random", "-r", action="store_true", help="使用随机载荷 (默认)")
    
    args = parser.parse_args()
    
    # Check GPU
    check_gpu()
    
    # Parse payload
    payload = None
    if args.payload:
        payload = parse_custom_payload(args.payload)
        if payload is None:
            sys.exit(1)
    elif not args.random:
        # 使用时间戳设置随机种子
        seed = int(time.time() * 1000) % (2**32)
        np.random.seed(seed)
        payload = np.random.randint(0, 2, 16)
        print(f"使用随机载荷: {payload}")
    
    # Set input/output paths
    if args.input:
        input_path = args.input
        if args.output:
            output_path = args.output
        else:
            if os.path.isfile(input_path):
                output_path = os.path.join(os.path.dirname(input_path), "watermarked")
            else:
                output_path = os.path.join(os.path.dirname(input_path), "watermarked")
    else:
        input_path = r"C:\Users\Administrator\Desktop\audio\test_audio"
        output_path = r"C:\Users\Administrator\Desktop\audio\marked_audio"
    
    # Process files
    if os.path.isfile(input_path):
        # Single file processing
        success, result = process_audio_file(input_path, None, payload, not args.no_comparison)
        if success:
            print(f"\n✅ 文件处理成功!")
            print(f"输出文件: {result['output_file']}")
        else:
            print(f"\n❌ 文件处理失败!")
    else:
        # Batch processing
        success, results = batch_process_from_directories(input_path, output_path, payload, not args.no_comparison)
        if success:
            print(f"\n✅ 批量处理完成!")
            if results:
                print(f"成功处理 {len(results)} 个文件")
                print(f"输出目录: {output_path}")
                
                # Summary statistics
                bers = [r['ber'] for r in results if r['ber'] is not None]
                snrs = [r['snr'] for r in results if r['snr'] is not None]
                if bers:
                    print(f"平均比特错误率: {np.mean(bers):.3f}")
                if snrs:
                    print(f"平均信噪比: {np.mean(snrs):.1f} dB")
            else:
                print("没有文件被处理")
        else:
            print(f"\n❌ 批量处理失败!")

if __name__ == "__main__":
    main()

