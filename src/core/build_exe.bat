@echo off
title ProQuants Professional EXE Builder
echo.
echo 🚀 ProQuants Professional EXE Builder
echo =====================================
echo By: Dominic (Domza304)
echo Repository: ProQuants-Professional-Trading--System
echo.

:: Navigate to project directory
cd /d "C:\Users\mzie_\source\vs_code_Deriv\WORKSPACE\ProQuants_Professional"

:: Install PyInstaller if not installed
echo 📦 Checking PyInstaller installation...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

:: Clean previous builds
echo 🧹 Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist __pycache__ rmdir /s /q __pycache__
del *.spec 2>nul

:: Create required directories
mkdir logs 2>nul
mkdir data 2>nul
mkdir config 2>nul

:: Build using spec file or direct command
echo 🔧 Building ProQuants Professional EXE...
if exist "ProQuants_Professional.spec" (
    echo Using spec file...
    pyinstaller ProQuants_Professional.spec
) else (
    echo Using direct PyInstaller command...
    pyinstaller --onefile --windowed --name=ProQuants_Professional ^
                --add-data=src;src ^
                --add-data=config;config ^
                --add-data=data;data ^
                --add-data=logs;logs ^
                --hidden-import=tkinter ^
                --hidden-import=numpy ^
                --hidden-import=pandas ^
                --hidden-import=MetaTrader5 ^
                --hidden-import=sqlite3 ^
                --hidden-import=threading ^
                --optimize=2 ^
                --clean ^
                complete_trading_system.py
)

:: Check if build was successful
if exist "dist\ProQuants_Professional.exe" (
    echo.
    echo ✅ BUILD SUCCESSFUL!
    echo ==================
    
    :: Get file size
    for %%I in (dist\ProQuants_Professional.exe) do set size=%%~zI
    set /a size_mb=!size!/1024/1024
    
    echo ✓ EXE Location: dist\ProQuants_Professional.exe
    echo ✓ File Size: Approximately !size_mb! MB
    echo.
    
    :: Create launcher scripts
    echo 📝 Creating launcher scripts...
    
    echo @echo off > "dist\Launch_ProQuants.bat"
    echo title ProQuants Professional >> "dist\Launch_ProQuants.bat"
    echo echo 🚀 Launching ProQuants Professional Trading System... >> "dist\Launch_ProQuants.bat"
    echo start "" "ProQuants_Professional.exe" >> "dist\Launch_ProQuants.bat"
    echo echo ✅ ProQuants Professional launched! >> "dist\Launch_ProQuants.bat"
    echo pause >> "dist\Launch_ProQuants.bat"
    
    echo ✓ Launcher script created: dist\Launch_ProQuants.bat
    echo.
    echo 🎯 READY FOR DISTRIBUTION!
    echo ==========================
    echo.
    echo Files created:
    echo - dist\ProQuants_Professional.exe
    echo - dist\Launch_ProQuants.bat
    echo.
    echo 🚀 Double-click Launch_ProQuants.bat to run!
    echo 💰 Your ML-Enhanced CREAM Strategy system is ready!
    
) else (
    echo.
    echo ❌ BUILD FAILED!
    echo ===============
    echo The EXE file was not created. Check the error messages above.
    echo.
    echo Common solutions:
    echo 1. Make sure all Python dependencies are installed
    echo 2. Check that complete_trading_system.py exists
    echo 3. Ensure you have write permissions in this directory
    echo 4. Try running as administrator
)

echo.
pause