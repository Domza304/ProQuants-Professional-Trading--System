"""
Test ProQuants Import Issues
"""

print("Testing ProQuants imports...")

try:
    print("1. Testing basic imports...")
    import pandas as pd
    import numpy as np
    print("✓ Basic imports OK")
except Exception as e:
    print(f"✗ Basic imports failed: {e}")

try:
    print("2. Testing AI system...")
    from src.ai.unified_ai_system import UnifiedAITradingSystem
    print("✓ AI system import OK")
except Exception as e:
    print(f"✗ AI system import failed: {e}")

try:
    print("3. Testing strategy...")
    from src.strategies.enhanced_cream_strategy import ProQuantsEnhancedStrategy
    print("✓ Strategy import OK")
except Exception as e:
    print(f"✗ Strategy import failed: {e}")

try:
    print("4. Testing GUI...")
    from src.gui.professional_dashboard import ProfessionalDashboard
    print("✓ GUI import OK")
except Exception as e:
    print(f"✗ GUI import failed: {e}")

print("Test complete!")
