"""
Fast ProQuants Professional EXE Builder - Optimized for Speed
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path

def fast_build():
    """Build ProQuants Professional EXE - Fast Version"""
    
    print("⚡ Fast ProQuants Professional EXE Builder")
    print("=" * 50)
    
    # Navigate to project directory
    project_dir = Path(r"C:\Users\mzie_\source\vs_code_Deriv\WORKSPACE\ProQuants_Professional")
    os.chdir(project_dir)
    
    # Check if main file exists
    if not Path("complete_trading_system.py").exists():
        print("❌ ERROR: complete_trading_system.py not found!")
        return False
    
    print("✓ Found main file")
    
    # Clean previous builds quickly
    for folder in ['build', 'dist']:
        if Path(folder).exists():
            shutil.rmtree(folder)
            print(f"✓ Cleaned {folder}")
    
    # Simple, fast PyInstaller command
    cmd = [
        'pyinstaller',
        '--onefile',                        # Single file
        '--name=ProQuants_Professional',    # EXE name
        '--clean',                          # Clean build
        '--noconfirm',                      # No prompts
        'complete_trading_system.py'        # Main file
    ]
    
    try:
        print("⚡ Building EXE (fast mode)...")
        print("Command:", ' '.join(cmd))
        
        # Run with longer timeout but show progress
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Show real-time output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Show important progress lines
                if any(keyword in output.lower() for keyword in ['collecting', 'building', 'exe']):
                    print(f"  {output.strip()}")
        
        result = process.poll()
        
        if result == 0:
            print("\n✅ Build completed successfully!")
            
            # Check if EXE exists
            exe_path = Path('dist/ProQuants_Professional.exe')
            if exe_path.exists():
                size_mb = exe_path.stat().st_size / (1024 * 1024)
                print(f"📁 Location: {exe_path.absolute()}")
                print(f"📊 Size: {size_mb:.1f} MB")
                
                # Create simple launcher
                create_simple_launcher()
                
                return True
            else:
                print("❌ EXE file not found")
                return False
        else:
            print(f"❌ Build failed with code: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False

def create_simple_launcher():
    """Create simple launcher"""
    
    launcher_content = '''@echo off
echo 🚀 Launching ProQuants Professional...
if exist "ProQuants_Professional.exe" (
    start "" "ProQuants_Professional.exe"
    echo ✅ ProQuants Professional launched!
) else (
    echo ❌ EXE not found!
)
pause'''

    with open('dist/Launch.bat', 'w') as f:
        f.write(launcher_content)
    
    print("✓ Created launcher: dist/Launch.bat")

if __name__ == "__main__":
    print("Fast ProQuants EXE Builder by Dominic")
    print("")
    
    success = fast_build()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("Files created:")
        print("  - dist/ProQuants_Professional.exe")
        print("  - dist/Launch.bat")
        print("\n🚀 Double-click Launch.bat to run!")
    else:
        print("\n❌ Build failed!")
    
    input("\nPress Enter to exit...")