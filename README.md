# WavMark 音频水印工具

WavMark 是一个基于深度学习的音频水印工具，可以为音频文件添加不可感知的数字水印。

## 📚 目录
1. [功能特点](#功能特点)
2. [安装要求](#安装要求)
3. [快速开始](#快速开始)
4. [使用指南](#使用指南)
5. [高级功能](#高级功能)
6. [故障排除](#故障排除)
7. [最佳实践](#最佳实践)

## 🌟 功能特点

- 🎵 支持多种音频格式：WAV, MP3, FLAC, M4A, AAC, OGG
- 🔒 高质量水印：信噪比 25-35dB，确保音频质量
- 📊 实时质量评估：显示比特错误率(BER)和信噪比(SNR)
- 🚀 GPU 加速：支持 CUDA 加速处理
- 📁 批量处理：支持目录批量处理

## 💻 安装要求

### 系统要求
- Python 3.6+
- NVIDIA GPU (支持 CUDA)
- FFmpeg (用于音频格式转换)

### 依赖安装

#### 方法1: 自动安装 (推荐)
```bash
# 运行自动安装脚本
python install_dependencies.py
```

#### 方法2: 手动安装
```bash
# 安装 PyTorch (CUDA 版本)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装核心依赖
pip install -r requirements-minimal.txt

# 或安装完整依赖 (包含开发工具)
pip install -r requirements.txt
```

#### 方法3: 使用 pip 直接安装
```bash
# 安装 PyTorch (CUDA 版本)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装音频处理库
pip install librosa soundfile pydub numpy tqdm huggingface_hub resampy

# 安装 WavMark
pip install wavmark
```

### FFmpeg 安装
Windows:
```bash
# 方法1: 使用 chocolatey
choco install ffmpeg

# 方法2: 下载并添加到 PATH
# 从 https://ffmpeg.org/download.html 下载
```

macOS:
```bash
# 使用 Homebrew
brew install ffmpeg
```

Linux:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg
```

## 🚀 快速开始

### 1. 安装测试
运行安装测试确保环境正确：
```bash
python test_installation.py
```

### 2. 单个文件处理
处理单个音频文件：
```bash
# 基本用法
python audio_watermark_example.py input.wav

# 支持多种格式
python audio_watermark_example.py music.mp3
python audio_watermark_example.py song.flac
python audio_watermark_example.py audio.m4a
```

### 3. 批量处理
处理整个目录的音频文件：
```bash
# 使用默认目录
python quick_test.py

# 指定输入和输出目录
python quick_test.py C:\Music\Input C:\Music\Output

# 相对路径
python quick_test.py ./input_audio ./output_audio
```

### 4. 图形界面
使用 Windows 批处理工具：
```bash
# 双击运行
process_audio.bat
```

## 📖 使用指南

### 支持的音频格式

| 格式 | 编码器 | 说明 |
|------|--------|------|
| WAV | PCM | 无损，推荐用于处理 |
| MP3 | LAME | 有损压缩 |
| FLAC | FLAC | 无损压缩 |
| M4A | AAC | Apple 格式 |
| AAC | AAC | 高级音频编码 |
| OGG | Vorbis | 开源格式 |

### 技术参数

- **采样率**: 16kHz (自动转换)
- **声道**: 单声道 (自动转换)
- **水印长度**: 16 位
- **信噪比范围**: 25-35dB
- **处理时间**: 约 1-3 秒/分钟音频 (GPU)

## 🔧 高级功能

### 1. 自定义载荷

您可以为音频添加自定义的 16 位水印信息：

```python
from quick_test import process_audio_file
import numpy as np

# 创建自定义载荷
custom_payload = np.array([1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0])

# 处理音频文件
success, result = process_audio_file(
    "input.wav", 
    payload=custom_payload
)

if success:
    print(f"原始载荷: {result['payload']}")
    print(f"解码载荷: {result['decoded_payload']}")
    print(f"载荷匹配: {np.array_equal(result['payload'], custom_payload)}")
```

### 2. 质量参数调整

通过修改 `config.py` 中的参数来调整水印质量：

```python
# 修改水印参数
WATERMARK_CONFIG = {
    'min_snr': 30.0,  # 提高最小信噪比
    'max_snr': 40.0   # 提高最大信噪比
}

# 修改文件设置
FILE_CONFIG = {
    'output_suffix': '_wm',  # 自定义后缀
    'backup_original': True  # 启用备份
}
```

### 3. 编程接口

直接使用 WavMark 库进行更精细的控制：

```python
import numpy as np
import soundfile as sf
from wavmark import load_model, encode_watermark, decode_watermark

# 加载模型
model = load_model(device="cuda")

# 读取音频
audio, sr = sf.read("input.wav")
if sr != 16000:
    # 重采样到 16kHz
    import librosa
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

# 创建载荷
payload = np.random.randint(0, 2, 16)

# 编码水印
watermarked_audio, encode_info = encode_watermark(
    model=model,
    signal=audio,
    payload=payload,
    min_snr=25.0,
    max_snr=35.0,
    show_progress=True
)

# 保存水印音频
sf.write("output.wav", watermarked_audio, 16000)

# 解码验证
decoded_payload, decode_info = decode_watermark(
    model=model,
    signal=watermarked_audio,
    show_progress=True
)

# 计算错误率
ber = np.mean(payload != decoded_payload)
print(f"比特错误率: {ber:.3f}")
print(f"信噪比: {encode_info['snr']:.1f} dB")
```

## ❗ 故障排除

### 1. GPU 相关问题

**问题**: `GPU not available! This program requires GPU to run.`

**解决方案**:
```bash
# 检查 CUDA 安装
nvidia-smi

# 重新安装 PyTorch CUDA 版本
pip uninstall torch torchaudio
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. FFmpeg 相关问题

**问题**: `Error setting up ffmpeg`

**解决方案**:
```bash
# Windows (使用 chocolatey)
choco install ffmpeg

# 或手动下载并添加到 PATH
# 从 https://ffmpeg.org/download.html 下载
```

### 3. 音频格式问题

**问题**: `Unsupported format: .xxx`

**解决方案**:
```bash
# 使用 FFmpeg 转换格式
ffmpeg -i input.xxx -acodec pcm_s16le -ar 16000 -ac 1 output.wav
```

### 4. 内存不足问题

**问题**: `CUDA out of memory`

**解决方案**:
- 减少批量处理的文件数量
- 使用更短的音频文件
- 关闭其他 GPU 应用程序

## 💡 最佳实践

### 1. 音频文件准备

- **推荐格式**: WAV (无损，处理最快)
- **采样率**: 16kHz (自动转换)
- **声道**: 单声道 (自动转换)
- **时长**: 建议 1-10 分钟 (平衡质量和处理时间)

### 2. 批量处理优化

```bash
# 创建处理脚本
@echo off
echo 开始批量处理...
python quick_test.py "C:\Music\Input" "C:\Music\Output"
echo 处理完成！
pause
```

### 3. 质量监控

定期检查处理结果：

```python
# 质量检查脚本
import os
import numpy as np
from quick_test import process_audio_file

def quality_check(input_file):
    success, result = process_audio_file(input_file)
    if success:
        ber = result['ber']
        snr = result['snr']
        
        if ber < 0.01:  # BER < 1%
            print(f"✅ {input_file}: 质量良好 (BER={ber:.3f}, SNR={snr:.1f}dB)")
        else:
            print(f"⚠️  {input_file}: 质量较差 (BER={ber:.3f}, SNR={snr:.1f}dB)")
    else:
        print(f"❌ {input_file}: 处理失败")

# 检查目录中的所有文件
input_dir = "C:\\Music\\Input"
for file in os.listdir(input_dir):
    if file.endswith(('.wav', '.mp3', '.flac')):
        quality_check(os.path.join(input_dir, file))
```

### 4. 性能优化

- 使用 SSD 存储提高 I/O 性能
- 确保 GPU 内存充足 (4GB+)
- 关闭不必要的后台程序
- 批量处理时使用相同格式的文件

## 📈 版本历史

- **v1.0.0**: 初始版本，支持基本音频水印功能
- **v1.1.0**: 添加批量处理和多种音频格式支持
- **v1.2.0**: 优化性能和错误处理
- **v1.3.0**: 添加配置系统和文档完善

## 📞 获取帮助

如果遇到问题，请按以下步骤操作：

1. 运行安装测试：`python test_installation.py`
2. 检查系统要求：
   - Python 3.6+
   - CUDA 支持
   - FFmpeg 安装
3. 查看错误信息和日志
4. 尝试使用推荐的音频格式和参数

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目基于 MIT 许可证开源。
