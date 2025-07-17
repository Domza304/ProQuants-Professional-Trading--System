#!/usr/bin/env python3
"""
🎼 SYMPHONIC MANAGER TEST 🎼
Quick test to verify all required methods are available
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from clean_12_panel_dashboard import SymphonicMT5Manager
    
    print("🎼" + "="*60 + "🎼")
    print("🎼 SYMPHONIC MT5 MANAGER METHOD TEST 🎼")
    print("🎼" + "="*60 + "🎼")
    
    # Create manager instance
    manager = SymphonicMT5Manager()
    print("✅ SymphonicMT5Manager instantiated successfully")
    
    # Check for required methods
    required_methods = [
        'connect_to_symphony',
        'get_account_symphony', 
        'get_account_info',
        'connect',
        'get_symphonic_positions',
        'execute_symphonic_order',
        'disconnect'
    ]
    
    print("\n🔍 Checking required methods:")
    for method in required_methods:
        if hasattr(manager, method):
            print(f"✅ {method}")
        else:
            print(f"❌ {method} - MISSING!")
    
    # Check connected attribute
    if hasattr(manager, 'connected'):
        print(f"✅ connected attribute: {manager.connected}")
    else:
        print("❌ connected attribute - MISSING!")
    
    print("\n🎼" + "="*60 + "🎼")
    print("🎼 METHOD TEST COMPLETE 🎼")
    print("🎼" + "="*60 + "🎼")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
