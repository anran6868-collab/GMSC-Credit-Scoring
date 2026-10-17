@echo off
setlocal
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo 未找到项目 Python 虚拟环境，请先在项目根目录创建 .venv。
    pause
    exit /b 1
)
if not exist "app\main.py" (
    echo 未找到 Streamlit 页面入口 app\main.py。
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m src.dashboard.launcher
set "GMSC_EXIT=%errorlevel%"
if not "%GMSC_EXIT%"=="0" (
    echo GMSC 本地系统启动失败，退出码：%GMSC_EXIT%。
    pause
)
exit /b %GMSC_EXIT%
