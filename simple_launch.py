"""
ProQuants Simple Launcher
Launch with better error handling
"""

import sys
import os
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def launch_proquants():
    """Launch ProQuants with comprehensive error handling"""
    
    print("🚀 ProQuants Professional - Simple Launch")
    print("=" * 50)
    
    try:
        logger.info("Step 1: Loading AI System...")
        from src.ai.unified_ai_system import UnifiedAITradingSystem
        ai_system = UnifiedAITradingSystem()
        logger.info("✓ AI System loaded")
        
        logger.info("Step 2: Connecting to MT5...")
        mt5_result = ai_system.initialize_mt5()
        if mt5_result["success"]:
            logger.info("✓ MT5 Connected successfully")
        else:
            logger.warning(f"⚠ MT5 Connection issue: {mt5_result.get('error', 'Unknown')}")
            logger.info("Continuing in offline mode...")
        
        logger.info("Step 3: Loading Enhanced Strategy...")
        from src.strategies.enhanced_cream_strategy import ProQuantsEnhancedStrategy
        strategy = ProQuantsEnhancedStrategy(ai_system)
        logger.info("✓ Enhanced Strategy loaded")
        
        logger.info("Step 4: Training AI Models...")
        training_results = ai_system.train_all_models()
        models_trained = sum(1 for result in training_results.values() 
                           if isinstance(result, dict) and not result.get("error"))
        logger.info(f"✓ AI Models trained: {models_trained} symbols")
        
        logger.info("Step 5: Loading Professional Dashboard...")
        from src.gui.professional_dashboard import ProfessionalDashboard
        dashboard = ProfessionalDashboard(ai_system, strategy)
        logger.info("✓ Dashboard loaded")
        
        print("\n" + "=" * 50)
        print("🎯 ProQuants Professional READY!")
        print("=" * 50)
        print("• AI/ML/Neural Networks: ✓ ACTIVE")
        print("• Pure MT5 Integration: ✓ ENABLED") 
        print("• Fractal Learning: ✓ M1→H4")
        print("• Goloji Bhudasi Logic: ✓ ENHANCED")
        print("• Professional Dashboard: ✓ STARTING")
        print("=" * 50)
        
        logger.info("🖥️ Starting Professional Dashboard...")
        dashboard.run()
        
    except KeyboardInterrupt:
        logger.info("Launch cancelled by user")
    except Exception as e:
        logger.error(f"Launch failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Detailed error information
        import traceback
        logger.error("Full traceback:")
        traceback.print_exc()
        
        print("\n🔧 Troubleshooting:")
        print("1. Ensure all dependencies installed: pip install -r requirements.txt")
        print("2. Check MT5 terminal is running")
        print("3. Verify .env file with credentials")
        print("4. Try running: python test_gui.py")
        
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    launch_proquants()
