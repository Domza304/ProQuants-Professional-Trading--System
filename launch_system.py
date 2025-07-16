"""
Quick Launch Script for ProQuants Neural Network Enhanced Trading System
Handles initialization, dependency checking, and graceful fallbacks
"""

import sys
import os
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/launch.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'pandas', 'numpy', 'MetaTrader5', 'tkinter', 'python-dotenv'
    ]
    
    optional_packages = [
        'tensorflow', 'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'python-dotenv':
                import dotenv
            else:
                __import__(package)
            logger.info(f"✓ {package}: Available")
        except ImportError:
            missing_required.append(package)
            logger.error(f"✗ {package}: Missing (REQUIRED)")
    
    for package in optional_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package}: Available")
        except ImportError:
            missing_optional.append(package)
            logger.warning(f"⚠ {package}: Missing (OPTIONAL - Neural Networks will be disabled)")
    
    if missing_required:
        logger.error(f"Missing required packages: {missing_required}")
        logger.error("Please install with: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        logger.warning("Some optional packages are missing. Neural Networks will be disabled.")
        logger.info("To enable Neural Networks, install: pip install " + " ".join(missing_optional))
    
    return True

def check_environment():
    """Check environment setup"""
    # Check if .env file exists
    env_file = '.env'
    if os.path.exists(env_file):
        logger.info("✓ .env file found")
        
        # Check if required variables are set
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            mt5_login = os.getenv('MT5_LOGIN')
            mt5_password = os.getenv('MT5_PASSWORD')
            mt5_server = os.getenv('MT5_SERVER')
            
            if mt5_login and mt5_password and mt5_server:
                logger.info("✓ MT5 credentials configured")
            else:
                logger.warning("⚠ MT5 credentials incomplete in .env file")
                
        except Exception as e:
            logger.error(f"✗ Error reading .env file: {e}")
    else:
        logger.warning("⚠ .env file not found - MT5 integration may not work")
        logger.info("Create .env file with MT5_LOGIN, MT5_PASSWORD, MT5_SERVER")
    
    # Check directories
    required_dirs = ['logs', 'data', 'src']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            logger.info(f"✓ {dir_name}/ directory exists")
        else:
            logger.warning(f"⚠ {dir_name}/ directory missing - creating...")
            os.makedirs(dir_name, exist_ok=True)

def launch_system():
    """Launch the ProQuants system"""
    try:
        logger.info("=" * 60)
        logger.info("ProQuants Neural Network Enhanced Trading System")
        logger.info("=" * 60)
        logger.info(f"Launch time: {datetime.now()}")
        
        # Check dependencies
        logger.info("\n--- Checking Dependencies ---")
        if not check_dependencies():
            logger.error("Dependency check failed. Cannot continue.")
            return False
        
        # Check environment
        logger.info("\n--- Checking Environment ---")
        check_environment()
        
        # Import and initialize system
        logger.info("\n--- Initializing System ---")
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from main import ProQuantsProfessionalSystem
        
        system = ProQuantsProfessionalSystem()
        
        if system.initialize():
            logger.info("✓ System initialized successfully!")
            logger.info(f"Neural Network Status: {system.neural_network_status}")
            
            # Start the GUI
            logger.info("\n--- Starting Professional Dashboard ---")
            if system.dashboard:
                logger.info("Launching GUI...")
                system.dashboard.root.mainloop()
            else:
                logger.error("Dashboard not available")
                return False
                
        else:
            logger.error("System initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"Launch failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main entry point"""
    try:
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        success = launch_system()
        
        if success:
            logger.info("ProQuants system closed successfully")
        else:
            logger.error("ProQuants system encountered errors")
            
        return success
        
    except KeyboardInterrupt:
        logger.info("System shutdown requested by user")
        return True
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    # Keep console open on Windows
    if sys.platform.startswith('win'):
        input("\nPress Enter to exit...")
    
    sys.exit(0 if success else 1)
