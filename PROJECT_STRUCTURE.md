# WavMark 项目结构说明

## 📁 完整项目结构

```
Audio—wavmark/
├── 📄 README.md                    # 项目主要说明文档
├── 📄 PROJECT_STRUCTURE.md        # 项目结构说明 (本文件)
    ├── 📄 config.py                   # 配置文件
    ├── 🖥️  process_audio.bat           # Windows 批处理脚本
    │
    ├── 🐍 核心脚本
    │   ├── quick_test.py              # 主要处理脚本 (批量处理)
    │   ├── audio_watermark_example.py # 简单示例脚本 (单个文件)
    │   └── test_installation.py       # 安装和功能测试
    │
    ├── 📁 src/                        # 源代码目录
    │   └── wavmark/                   # WavMark 核心库
    │       ├── __init__.py
    │       ├── models/                # 模型定义
    │       ├── utils/                 # 工具函数
    │       └── config/                # 配置管理
    │
    ├── 📁 data/                       # 数据目录
    │   └── imgs/                      # 图片资源
    │       └── structure.png          # 项目结构图
    │
    ├── 📁 examples/                   # 示例代码
    │   ├── basic_usage.py            # 基本使用示例
    │   └── advanced_usage.py         # 高级使用示例
    │
    └── 📁 __pycache__/               # Python 缓存文件 (自动生成)
        └── quick_test.cpython-39.pyc
```

## 📋 文件功能说明

### 📄 文档文件

| 文件 | 功能 | 适用人群 |
|------|------|----------|
| `README.md` | 项目概述、安装、基本使用 | 所有用户 |
| `USER_GUIDE.md` | 详细使用指南、高级功能 | 进阶用户 |
| `QUICK_START.md` | 5分钟快速上手 | 新手用户 |
| `PROJECT_STRUCTURE.md` | 项目结构说明 | 开发者 |

### 🐍 Python 脚本

| 脚本 | 功能 | 使用场景 |
|------|------|----------|
| `quick_test.py` | 核心处理脚本 | 批量处理音频文件 |
| `audio_watermark_example.py` | 简单示例 | 处理单个音频文件 |
| `test_installation.py` | 安装测试 | 验证环境配置 |
| `config.py` | 配置文件 | 自定义处理参数 |

### 🖥️ 批处理脚本

| 脚本 | 功能 | 平台 |
|------|------|------|
| `process_audio.bat` | 图形化批处理工具 | Windows |

### 📁 目录说明

| 目录 | 内容 | 用途 |
|------|------|------|
| `src/wavmark/` | 核心库代码 | 水印算法实现 |
| `data/imgs/` | 图片资源 | 文档插图 |
| `examples/` | 示例代码 | 学习参考 |
| `__pycache__/` | Python 缓存 | 自动生成 |

## 🎯 使用流程

### 用户流程
1. 阅读 `README.md`
2. 运行 `install_dependencies.py`  进行安装
3. 运行 `test_installation.py`  进行安装测试
4. 使用 `audio_watermark_example.py` 处理单个文件
5. 使用 `process_audio.bat` 进行图形化操作
6. 修改 `config.py` 自定义参数
7. 使用 `quick_test.py` 进行批量处理
8. 参考 `examples/` 中的示例代码

### 开发者流程
1. 查看 `src/wavmark/` 源码
2. 修改 `config.py` 进行实验
3. 参考 `examples/` 进行二次开发
4. 查看 `PROJECT_STRUCTURE.md` 了解架构

## 🔧 自定义配置

### 修改处理参数
编辑 `config.py` 文件：

```python
# 修改水印质量
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

### 添加新功能
1. 在 `src/wavmark/` 中添加新模块
2. 在 `examples/` 中创建示例
3. 更新相关文档

## 📊 文件大小和性能

| 文件类型 | 典型大小 | 处理时间 |
|----------|----------|----------|
| 1分钟 WAV | ~1MB | 1-2秒 |
| 1分钟 MP3 | ~1MB | 2-3秒 |
| 1分钟 FLAC | ~5MB | 1-2秒 |

## 🔍 调试和故障排除

### 日志文件
- 控制台输出：实时查看处理状态
- 错误信息：详细的错误堆栈

### 测试文件
- `test_installation.py`：完整环境测试
- 示例音频：用于功能验证

### 常见问题
1. GPU 不可用：检查 CUDA 安装
2. FFmpeg 未找到：安装 FFmpeg
3. 内存不足：减少批处理大小

## 🚀 扩展开发

### 添加新音频格式
1. 在 `AUDIO_CONFIG['supported_formats']` 中添加格式
2. 在 `quick_test.py` 中添加格式处理逻辑

### 添加新水印算法
1. 在 `src/wavmark/models/` 中添加新模型
2. 更新 `config.py` 中的参数
3. 创建相应的示例代码

### 优化性能
1. 修改 `PERFORMANCE_CONFIG` 参数
2. 调整批处理大小
3. 优化 GPU 内存使用

## 📞 支持和贡献

### 获取帮助
1. 查看文档：`README.md` 和 `USER_GUIDE.md`
2. 运行测试：`test_installation.py`
3. 检查配置：`config.py`

### 贡献代码
1. Fork 项目
2. 创建功能分支
3. 提交 Pull Request
4. 更新相关文档

## 📈 版本历史

- **v1.0.0**: 初始版本，基本水印功能
- **v1.1.0**: 添加批量处理和多种格式支持
- **v1.2.0**: 优化性能和错误处理
- **v1.3.0**: 添加配置系统和文档完善 
