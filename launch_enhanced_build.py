#!/usr/bin/env python3
"""
Enhanced GUI Build Launcher
Builds the ProQuants Ultimate GUI with real MT5 integration
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 ENHANCED PROQUANTS GUI BUILD LAUNCHER")
    print("=" * 60)
    print("Building sophisticated GUI with REAL MT5 integration...")
    print("Enhanced features: .env credentials, live authentication")
    print("=" * 60)
    
    # Change to project directory
    project_dir = Path(r"C:\Users\mzie_\source\vs_code_Deriv\WORKSPACE\ProQuants_Professional")
    os.chdir(project_dir)
    
    # Check if build script exists
    build_script = "build_ultimate_gui.py"
    if not Path(build_script).exists():
        print(f"❌ ERROR: {build_script} not found!")
        return False
    
    print(f"✅ Build script found: {build_script}")
    print("🔧 Starting enhanced build process...")
    print()
    
    try:
        # Execute the build script
        result = subprocess.run([sys.executable, build_script], 
                              capture_output=False, 
                              text=True)
        
        if result.returncode == 0:
            print("\n✅ ENHANCED GUI BUILD COMPLETED!")
            print("🔥 Your sophisticated trading interface is ready!")
            print("💰 Features real MT5 integration with .env credentials")
            return True
        else:
            print(f"\n❌ Build failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Build error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Ready to trade with enhanced GUI!")
    else:
        print("\n🔧 Please check error messages above")
    
    input("\nPress Enter to exit...")
