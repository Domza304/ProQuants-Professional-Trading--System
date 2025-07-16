#!/usr/bin/env python3
"""
ProQuants Professional - GUI Diagnostics
Complete system for testing and diagnosing GUI issues
"""

import sys
import os
import traceback
from datetime import datetime

def print_banner():
    print("\n" + "="*60)
    print("   ProQuants Professional - GUI Diagnostics")
    print("   Testing GUI Components & Dependencies")
    print("="*60)

def test_python_environment():
    """Test Python environment"""
    print(f"\n🐍 Python Environment:")
    print(f"   Version: {sys.version}")
    print(f"   Platform: {sys.platform}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Python path: {sys.executable}")

def test_basic_imports():
    """Test basic imports"""
    print(f"\n📦 Testing Basic Imports:")
    
    imports_to_test = [
        'os', 'sys', 'datetime', 'json', 'logging',
        'numpy', 'pandas', 'matplotlib', 'tkinter'
    ]
    
    for module_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"   ✓ {module_name}")
        except ImportError as e:
            print(f"   ✗ {module_name}: {e}")

def test_tkinter_basic():
    """Test basic tkinter functionality"""
    print(f"\n🖥️  Testing Tkinter Basic:")
    
    try:
        import tkinter as tk
        print("   ✓ Tkinter import successful")
        
        # Test root creation
        root = tk.Tk()
        print("   ✓ Root window created")
        
        # Test basic widgets
        label = tk.Label(root, text="Test")
        print("   ✓ Label widget created")
        
        button = tk.Button(root, text="Test")
        print("   ✓ Button widget created")
        
        frame = tk.Frame(root)
        print("   ✓ Frame widget created")
        
        # Clean up
        root.destroy()
        print("   ✓ Window destroyed successfully")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Tkinter error: {e}")
        traceback.print_exc()
        return False

def test_tkinter_display():
    """Test tkinter display functionality"""
    print(f"\n🖼️  Testing Tkinter Display:")
    
    try:
        import tkinter as tk
        
        # Create test window
        root = tk.Tk()
        root.title("ProQuants GUI Test")
        root.geometry("400x300")
        root.configure(bg='#2b2b2b')
        
        # Add test content
        main_frame = tk.Frame(root, bg='#2b2b2b')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        title_label = tk.Label(
            main_frame, 
            text="ProQuants Professional", 
            font=('Arial', 16, 'bold'),
            bg='#2b2b2b', 
            fg='#00ff88'
        )
        title_label.pack(pady=10)
        
        status_label = tk.Label(
            main_frame, 
            text="GUI Test - If you see this, GUI is working!", 
            font=('Arial', 12),
            bg='#2b2b2b', 
            fg='white'
        )
        status_label.pack(pady=10)
        
        close_button = tk.Button(
            main_frame, 
            text="Close Test", 
            command=root.destroy,
            bg='#4CAF50', 
            fg='white',
            font=('Arial', 10, 'bold')
        )
        close_button.pack(pady=20)
        
        print("   ✓ Test window created with widgets")
        print("   ℹ️  Displaying test window for 5 seconds...")
        
        # Auto-close after 5 seconds
        root.after(5000, root.destroy)
        
        # Show window
        root.mainloop()
        
        print("   ✓ Display test completed")
        return True
        
    except Exception as e:
        print(f"   ✗ Display test error: {e}")
        traceback.print_exc()
        return False

def test_professional_dashboard_import():
    """Test professional dashboard import"""
    print(f"\n📊 Testing Professional Dashboard Import:")
    
    try:
        # Add src path
        src_path = os.path.join(os.getcwd(), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from gui.professional_dashboard import ProfessionalDashboard
        print("   ✓ Professional dashboard imported successfully")
        return True
        
    except Exception as e:
        print(f"   ✗ Dashboard import error: {e}")
        traceback.print_exc()
        return False

def test_professional_dashboard_creation():
    """Test professional dashboard creation"""
    print(f"\n🏗️  Testing Professional Dashboard Creation:")
    
    try:
        # Add src path
        src_path = os.path.join(os.getcwd(), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        from gui.professional_dashboard import ProfessionalDashboard
        
        print("   ℹ️  Creating dashboard instance...")
        dashboard = ProfessionalDashboard()
        print("   ✓ Dashboard instance created")
        
        print("   ℹ️  Testing dashboard for 3 seconds...")
        dashboard.root.after(3000, dashboard.root.destroy)
        dashboard.run()
        
        print("   ✓ Dashboard creation test completed")
        return True
        
    except Exception as e:
        print(f"   ✗ Dashboard creation error: {e}")
        traceback.print_exc()
        return False

def run_complete_gui_diagnostics():
    """Run complete GUI diagnostics"""
    print_banner()
    
    results = {}
    
    # Test 1: Python Environment
    test_python_environment()
    
    # Test 2: Basic Imports
    test_basic_imports()
    
    # Test 3: Tkinter Basic
    results['tkinter_basic'] = test_tkinter_basic()
    
    # Test 4: Tkinter Display (only if basic works)
    if results['tkinter_basic']:
        results['tkinter_display'] = test_tkinter_display()
    else:
        results['tkinter_display'] = False
        print("   ⚠️  Skipping display test due to basic tkinter failure")
    
    # Test 5: Dashboard Import
    results['dashboard_import'] = test_professional_dashboard_import()
    
    # Test 6: Dashboard Creation (only if import works)
    if results['dashboard_import'] and results['tkinter_basic']:
        results['dashboard_creation'] = test_professional_dashboard_creation()
    else:
        results['dashboard_creation'] = False
        print("   ⚠️  Skipping dashboard creation test due to previous failures")
    
    # Summary
    print(f"\n" + "="*60)
    print("   DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    all_passed = all(results.values())
    overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
    print(f"\n   Overall Status: {overall_status}")
    
    if all_passed:
        print(f"\n   🎉 GUI system is fully functional!")
        print(f"   Ready to launch ProQuants Professional")
    else:
        print(f"\n   🔧 GUI issues detected - see errors above")
        print(f"   Fix these issues before launching the full system")
    
    print("="*60 + "\n")
    
    return all_passed

if __name__ == "__main__":
    try:
        run_complete_gui_diagnostics()
    except KeyboardInterrupt:
        print("\n\n⚠️  Diagnostics interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error during diagnostics: {e}")
        traceback.print_exc()
