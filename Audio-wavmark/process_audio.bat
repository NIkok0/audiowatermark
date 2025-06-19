@echo off
chcp 65001 >nul
echo ========================================
echo WavMark 音频水印处理工具
echo ========================================
echo.

:menu
echo 请选择操作：
echo 1. 处理单个音频文件
echo 2. 批量处理目录
echo 3. 运行安装测试
echo 4. 退出
echo.
set /p choice=请输入选择 (1-4): 

if "%choice%"=="1" goto single_file
if "%choice%"=="2" goto batch_process
if "%choice%"=="3" goto test_install
if "%choice%"=="4" goto exit
echo 无效选择，请重新输入
goto menu

:single_file
echo.
echo 单个文件处理模式
echo 请将音频文件拖拽到此处，然后按回车：
set /p input_file=
if "%input_file%"=="" (
    echo 未输入文件路径
    goto menu
)
echo 处理文件: %input_file%
python audio_watermark_example.py "%input_file%"
echo.
pause
goto menu

:batch_process
echo.
echo 批量处理模式
echo 请输入输入目录路径 (或按回车使用默认):
set /p input_dir=
if "%input_dir%"=="" (
    set input_dir=%USERPROFILE%\Desktop\audio\test_audio
    echo 使用默认目录: %input_dir%
)
echo 请输入输出目录路径 (或按回车使用默认):
set /p output_dir=
if "%output_dir%"=="" (
    set output_dir=%USERPROFILE%\Desktop\audio\marked_audio
    echo 使用默认目录: %output_dir%
)
echo.
echo 开始批量处理...
echo 输入目录: %input_dir%
echo 输出目录: %output_dir%
echo.
python quick_test.py "%input_dir%" "%output_dir%"
echo.
pause
goto menu

:test_install
echo.
echo 运行安装测试...
python test_installation.py
echo.
pause
goto menu

:exit
echo 感谢使用 WavMark！
pause 