"""
ProQuants System Health Check
Test all dependencies and fix any issues
"""

print("=" * 60)
print("ProQuants Professional - System Health Check")
print("=" * 60)

# Test core Python modules
core_modules = ['os', 'sys', 'datetime', 'logging', 'threading', 'json']
print("\n🔍 Testing Core Python Modules:")
for module in core_modules:
    try:
        __import__(module)
        print(f"✓ {module}")
    except ImportError as e:
        print(f"✗ {module}: {e}")

# Test required packages
required_packages = [
    ('pandas', 'pip install pandas'),
    ('numpy', 'pip install numpy'),
    ('MetaTrader5', 'pip install MetaTrader5'),
    ('dotenv', 'pip install python-dotenv'),
    ('tkinter', 'Built-in with Python')
]

print("\n🔍 Testing Required Packages:")
for package, install_cmd in required_packages:
    try:
        if package == 'dotenv':
            from dotenv import load_dotenv
        else:
            __import__(package)
        print(f"✓ {package}")
    except ImportError as e:
        print(f"✗ {package}: Missing - Install with: {install_cmd}")

# Test optional AI/ML packages
ai_packages = [
    ('sklearn', 'pip install scikit-learn'),
    ('tensorflow', 'pip install tensorflow'),
    ('scipy', 'pip install scipy')
]

print("\n🔍 Testing AI/ML Packages (Optional but Recommended):")
for package, install_cmd in ai_packages:
    try:
        __import__(package)
        print(f"✓ {package}")
    except ImportError as e:
        print(f"⚠ {package}: Missing - Install with: {install_cmd}")

# Test GUI capability
print("\n🔍 Testing GUI Capability:")
try:
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()  # Hide window
    root.destroy()
    print("✓ Tkinter GUI available")
except Exception as e:
    print(f"✗ GUI test failed: {e}")

# Test file system access
print("\n🔍 Testing File System:")
import os
current_dir = os.getcwd()
print(f"✓ Current directory: {current_dir}")

required_dirs = ['src', 'src/ai', 'src/strategies', 'src/gui', 'src/data', 'logs']
for directory in required_dirs:
    if os.path.exists(directory):
        print(f"✓ {directory}")
    else:
        print(f"⚠ {directory}: Missing - Will be created")
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"  Created: {directory}")
        except Exception as e:
            print(f"  Error creating {directory}: {e}")

# Test .env file
print("\n🔍 Testing Configuration:")
if os.path.exists('.env'):
    print("✓ .env file found")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        import os
        login = os.getenv('MT5_LOGIN')
        if login:
            print(f"✓ MT5 credentials configured (Account: {login})")
        else:
            print("⚠ MT5 credentials not configured in .env")
    except Exception as e:
        print(f"⚠ .env file error: {e}")
else:
    print("⚠ .env file missing - Create with MT5 credentials")

print("\n" + "=" * 60)
print("System Health Check Complete!")
print("=" * 60)
print("\nNext Steps:")
print("1. Install any missing packages shown above")
print("2. Configure .env file with MT5 credentials if needed")
print("3. Run: python master_launcher.py")
print("=" * 60)
