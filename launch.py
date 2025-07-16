"""
ProQuants Professional Launch System
Automated startup with dependency checks and error handling
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_banner():
    """Display professional launch banner"""
    print("=" * 80)
    print("    ██████╗ ██████╗  ██████╗  ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███████╗")
    print("    ██╔══██╗██╔══██╗██╔═══██╗██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝██╔════╝")
    print("    ██████╔╝██████╔╝██║   ██║██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ███████╗")
    print("    ██╔═══╝ ██╔══██╗██║   ██║██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ╚════██║")
    print("    ██║     ██║  ██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ███████║")
    print("    ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝")
    print("")
    print("                    PROFESSIONAL TRADING SYSTEM")
    print("              AI • Machine Learning • Neural Networks")
    print("                    Pure MT5 Integration")
    print("=" * 80)
    print(f"Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def check_dependencies():
    """Check and install dependencies"""
    print("\n🔍 Checking Dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'MetaTrader5', 'python-dotenv', 
        'scikit-learn', 'tensorflow', 'scipy', 'matplotlib'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            if package == 'python-dotenv':
                import dotenv
            elif package == 'scikit-learn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Missing")
            missing.append(package)
    
    if missing:
        print(f"\n📦 Installing missing packages: {missing}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("✅ All dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            print("Please manually install: pip install " + " ".join(missing))
            return False
    else:
        print("✅ All dependencies satisfied!")
    
    return True

def check_environment():
    """Check environment setup"""
    print("\n🔧 Checking Environment Setup...")
    
    # Check .env file
    if os.path.exists('.env'):
        print("✓ .env file found")
        try:
            from dotenv import load_dotenv
            load_dotenv()
            login = os.getenv('MT5_LOGIN')
            password = os.getenv('MT5_PASSWORD')
            server = os.getenv('MT5_SERVER')
            
            if login and password and server:
                print(f"✓ MT5 credentials configured (Account: {login})")
            else:
                print("⚠ Incomplete MT5 credentials in .env file")
        except Exception as e:
            print(f"⚠ .env file error: {e}")
    else:
        print("⚠ .env file missing - Creating default...")
        create_default_env()
    
    # Check directory structure
    required_dirs = ['src', 'src/ai', 'src/strategies', 'src/gui', 'src/data', 'logs']
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}")
        else:
            print(f"⚠ Creating {directory}")
            os.makedirs(directory, exist_ok=True)
    
    return True

def create_default_env():
    """Create default .env file"""
    env_content = """# ProQuants Professional MT5 Configuration
# Update these with your actual MT5 credentials

MT5_LOGIN=31833954
MT5_PASSWORD=@Dmc65070*
MT5_SERVER=Deriv-Demo

# Optional: AI Model Configuration
AI_TRAINING_HOURS=12
NEURAL_CONFIDENCE_THRESHOLD=0.7
ML_RETRAIN_INTERVAL=24

# Risk Management
MAX_RISK_PER_TRADE=0.02
MIN_RRR=4.0
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print("✓ Default .env file created")

def test_system_components():
    """Test core system components"""
    print("\n🧪 Testing System Components...")
    
    try:
        print("Testing AI System...")
        from src.ai.unified_ai_system import UnifiedAITradingSystem
        print("✓ AI System import successful")
        
        print("Testing Trading Strategy...")
        from src.strategies.enhanced_cream_strategy import ProQuantsEnhancedStrategy
        print("✓ Trading Strategy import successful")
        
        print("Testing GUI Dashboard...")
        from src.gui.professional_dashboard import ProfessionalDashboard
        print("✓ GUI Dashboard import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def launch_system():
    """Launch the ProQuants system"""
    print("\n🚀 Launching ProQuants Professional System...")
    
    try:
        # Import master launcher
        import master_launcher
        
        print("✅ Master launcher imported successfully")
        print("\n" + "=" * 80)
        print("SYSTEM LAUNCH SUCCESSFUL!")
        print("=" * 80)
        print("Starting ProQuants Professional Trading System...")
        print("• AI/ML/Neural Networks: ACTIVE")
        print("• Pure MT5 Integration: ENABLED")
        print("• Fractal Learning: M1→H4 ACTIVE")
        print("• Goloji Bhudasi Logic: ENHANCED")
        print("=" * 80)
        
        # Create and run the master system
        master_system = master_launcher.ProQuantsMasterSystem()
        
        # Ask user for launch mode
        print("\nSelect launch mode:")
        print("1. GUI Mode (Recommended)")
        print("2. Headless Mode (Server)")
        
        while True:
            try:
                choice = input("Enter choice (1 or 2): ").strip()
                if choice == "1":
                    print("\n🖥️  Starting GUI Mode...")
                    master_system.run_gui_mode()
                    break
                elif choice == "2":
                    print("\n⚙️  Starting Headless Mode...")
                    master_system.run_headless_mode()
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except KeyboardInterrupt:
                print("\n\n🛑 Launch cancelled by user")
                break
        
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check that all dependencies are installed")
        print("2. Ensure MT5 is running and accessible")
        print("3. Verify .env file configuration")
        print("4. Check log files in logs/ directory")
        return False
    
    return True

def main():
    """Main launch sequence"""
    print_banner()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return
    
    # Step 2: Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix configuration issues.")
        return
    
    # Step 3: Test components
    if not test_system_components():
        print("\n❌ Component test failed. Please check error messages above.")
        return
    
    # Step 4: Launch system
    print("\n✅ All checks passed! Ready to launch...")
    time.sleep(2)  # Brief pause for dramatic effect
    
    launch_system()

if __name__ == "__main__":
    main()
