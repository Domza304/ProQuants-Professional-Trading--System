"""
Test GUI Dashboard Directly
Simple standalone test
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_gui():
    print("Testing GUI components...")
    
    try:
        print("1. Testing tkinter...")
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.destroy()
        print("   ✓ Tkinter available")
        
        print("2. Testing dashboard import...")
        from src.gui.professional_dashboard import ProfessionalDashboard
        print("   ✓ Dashboard import successful")
        
        print("3. Creating dashboard...")
        dashboard = ProfessionalDashboard()
        print("   ✓ Dashboard created")
        
        print("4. Starting GUI...")
        print("   📊 Professional Dashboard should open now...")
        dashboard.run()
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gui()
