#!/usr/bin/env python3
"""
ProQuants Professional - Simple System Launcher
Focus on MT5 integration with clean dashboard launch
"""

import os
import sys
from dotenv import load_dotenv

def main():
    """Launch ProQuants Professional with MT5 focus"""
    print("🚀 ProQuants Professional - System Launch")
    print("=" * 60)
    
    # Load environment variables
    print("📁 Loading configuration...")
    load_dotenv()
    
    # Get MT5 credentials from .env
    mt5_login = os.getenv('MT5_LOGIN', '31833954')
    mt5_password = os.getenv('MT5_PASSWORD', '@Dmc65070*')
    mt5_server = os.getenv('MT5_SERVER', 'Deriv-Demo')
    
    print(f"✓ MT5 Account: {mt5_login}")
    print(f"✓ MT5 Server: {mt5_server}")
    print(f"✓ Credentials: Loaded")
    print()
    
    # Test dashboard import
    print("📦 Testing system components...")
    try:
        from src.gui.professional_dashboard import ProfessionalDashboard
        print("✓ Professional Dashboard: READY")
    except ImportError as e:
        print(f"❌ Dashboard import failed: {e}")
        return False
    
    # Optional AI system
    ai_system = None
    try:
        from src.ai.unified_ai_system import UnifiedAISystem
        print("✓ AI System: AVAILABLE")
        # Don't initialize yet to avoid complexity
    except ImportError:
        print("⚠️  AI System: Not available (will run in demo mode)")
    
    # Optional strategy
    strategy = None
    try:
        from src.strategies.enhanced_cream_strategy import EnhancedCreamStrategy
        print("✓ CREAM Strategy: AVAILABLE")
        # Don't initialize yet to avoid complexity
    except ImportError:
        print("⚠️  Strategy: Not available (will run in basic mode)")
    
    print()
    print("🎯 Launching Professional Dashboard...")
    print("💡 MT5 Credentials will be used when connecting")
    print("📊 Focus: V75, V25, V75(1s) synthetic indices")
    print()
    
    try:
        # Create and launch dashboard
        dashboard = ProfessionalDashboard(ai_system=ai_system, trading_strategy=strategy)
        
        print("✅ LAUNCH SUCCESSFUL!")
        print("🎉 ProQuants Professional is now running!")
        print()
        print("📋 QUICK START:")
        print("  1. Click 'START TRADING' to begin monitoring")
        print("  2. Use console commands (type 'help')")
        print("  3. Monitor Deriv synthetic indices")
        print("  4. Watch for trading signals")
        print()
        print("🔧 MT5 Connection Details:")
        print(f"  • Account: {mt5_login}")
        print(f"  • Server: {mt5_server}")
        print(f"  • Status: Ready to connect")
        print()
        
        # Start the GUI
        dashboard.run()
        
        return True
        
    except Exception as e:
        print(f"❌ LAUNCH FAILED: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Check if all dependencies are installed")
        print("  2. Verify .env file contains correct MT5 credentials")
        print("  3. Ensure tkinter is available")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting ProQuants Professional...")
    success = main()
    if success:
        print("System shutdown complete.")
    else:
        print("Launch failed. Please check the errors above.")
        sys.exit(1)
