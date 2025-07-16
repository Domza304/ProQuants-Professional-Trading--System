@echo off
title ProQuants Professional Trading System
echo.
echo ======================================================
echo   ProQuants Professional Trading System
echo   AI/ML/Neural Networks - Pure MT5 Integration
echo ======================================================
echo.

cd /d "%~dp0"

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

echo.
echo Installing/Upgrading dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo WARNING: Some dependencies may have failed to install
    echo The system will attempt to start anyway...
    echo.
    pause
)

echo.
echo Starting ProQuants Professional System...
echo.
echo Available modes:
echo [1] GUI Mode (Recommended)
echo [2] Headless Mode (Server mode)
echo [3] Exit
echo.

set /p choice="Select mode (1/2/3): "

if "%choice%"=="1" (
    echo Starting GUI Mode...
    python master_launcher.py
) else if "%choice%"=="2" (
    echo Starting Headless Mode...
    python master_launcher.py --headless
) else if "%choice%"=="3" (
    echo Exiting...
    exit /b 0
) else (
    echo Invalid choice. Starting GUI Mode by default...
    python master_launcher.py
)

echo.
echo ProQuants Professional System has stopped.
pause
