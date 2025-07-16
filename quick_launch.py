"""
Quick ProQuants Launcher
Direct system startup
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 60)
print("🚀 ProQuants Professional - Quick Launch")
print("=" * 60)

try:
    print("Loading AI System...")
    from src.ai.unified_ai_system import UnifiedAITradingSystem
    ai_system = UnifiedAITradingSystem()
    
    print("Loading Enhanced Strategy...")
    from src.strategies.enhanced_cream_strategy import ProQuantsEnhancedStrategy
    strategy = ProQuantsEnhancedStrategy(ai_system)
    
    print("Loading Professional Dashboard...")
    from src.gui.professional_dashboard import ProfessionalDashboard
    dashboard = ProfessionalDashboard(ai_system, strategy)
    
    print("✅ All components loaded successfully!")
    print("🖥️  Starting Professional Dashboard...")
    print("=" * 60)
    
    # Run the dashboard
    dashboard.run()
    
except Exception as e:
    print(f"❌ Launch error: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
    print("2. Check .env file exists with MT5 credentials")
    print("3. Verify file structure is intact")
    input("Press Enter to exit...")
