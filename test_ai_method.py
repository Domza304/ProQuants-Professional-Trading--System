import sys
import os
sys.path.append('c:/Users/mzie_/source/vs_code_Deriv/WORKSPACE/ProQuants_Professional')

try:
    from src.ai.unified_ai_system import UnifiedAITradingSystem
    print("✓ AI system imported successfully")
    
    ai = UnifiedAITradingSystem()
    print("✓ AI system instantiated")
    
    # Check for the method
    if hasattr(ai, 'get_ai_training_data'):
        print("✓ get_ai_training_data method EXISTS")
    else:
        print("✗ get_ai_training_data method MISSING")
        
    # List all methods containing 'get_ai' or 'training'
    methods = [m for m in dir(ai) if 'get_ai' in m or 'training' in m]
    print(f"Related methods: {methods}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
