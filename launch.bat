@echo off
echo =====================================================
echo ProQuants Professional Trading System Launcher
echo =====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    pause
    exit /b 1
)

echo Installing/Upgrading dependencies...
pip install -r requirements.txt
echo.

REM Check for MetaTrader5 package specifically
python -c "import MetaTrader5" >nul 2>&1
if errorlevel 1 (
    echo Installing MetaTrader5...
    pip install MetaTrader5
)

REM Try to install TA-Lib
python -c "import talib" >nul 2>&1
if errorlevel 1 (
    echo WARNING: TA-Lib not found. Some features may be limited.
    echo You may need to install TA-Lib manually.
    echo See: https://github.com/mrjbq7/ta-lib
    echo.
)

echo Creating necessary directories...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "data\cache" mkdir data\cache

echo.
echo =====================================================
echo Launching ProQuants Professional Trading System...
echo =====================================================
echo.

REM Launch the application
python main.py

echo.
echo Application closed.
pause
