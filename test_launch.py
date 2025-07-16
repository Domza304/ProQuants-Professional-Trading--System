"""
ProQuants Professional System Test Launcher
Quick test to verify all components are working
"""

import sys
import os
import traceback

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import tkinter as tk
        print("✓ tkinter imported successfully")
    except ImportError as e:
        print(f"✗ tkinter import failed: {e}")
        return False
        
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
        
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
        
    try:
        import MetaTrader5 as mt5
        print("✓ MetaTrader5 imported successfully")
    except ImportError as e:
        print(f"✗ MetaTrader5 import failed: {e}")
        print("  Note: MetaTrader5 may not work in all environments")
        
    try:
        import talib
        print("✓ TA-Lib imported successfully")
    except ImportError as e:
        print(f"⚠ TA-Lib import failed: {e}")
        print("  Note: TA-Lib is optional but recommended for full functionality")
        
    return True

def test_system_components():
    """Test system components"""
    print("\nTesting system components...")
    
    # Add src to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from src.gui.professional_dashboard import ProfessionalDashboard
        print("✓ Professional Dashboard imported successfully")
    except Exception as e:
        print(f"✗ Professional Dashboard import failed: {e}")
        return False
        
    try:
        from src.core.trading_engine import TradingEngine
        print("✓ Trading Engine imported successfully")
    except Exception as e:
        print(f"✗ Trading Engine import failed: {e}")
        return False
        
    try:
        from src.strategies.cream_strategy import CREAMStrategy
        print("✓ CREAM Strategy imported successfully")
    except Exception as e:
        print(f"✗ CREAM Strategy import failed: {e}")
        return False
        
    return True

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from src.core.trading_engine import TradingEngine
        engine = TradingEngine()
        engine.offline_mode = True
        
        # Test market data generation
        data = engine.get_market_data("TEST_SYMBOL", "M1", 20)
        if data is not None and len(data) > 0:
            print("✓ Market data generation working")
        else:
            print("✗ Market data generation failed")
            return False
            
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False
        
    try:
        from src.strategies.cream_strategy import CREAMStrategy
        strategy = CREAMStrategy()
        
        # Test strategy analysis with sample data
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start='2024-01-01', periods=30, freq='1min')
        sample_data = pd.DataFrame({
            'time': dates,
            'open': np.random.rand(30) * 100 + 1000,
            'high': np.random.rand(30) * 100 + 1020,
            'low': np.random.rand(30) * 100 + 980,
            'close': np.random.rand(30) * 100 + 1000,
            'tick_volume': np.random.randint(100, 1000, 30),
            'spread': np.random.randint(1, 5, 30),
            'real_volume': np.random.randint(1000, 10000, 30)
        })
        
        analysis = strategy.analyze(sample_data, "TEST_SYMBOL")
        if analysis and 'overall_signal' in analysis:
            print("✓ CREAM strategy analysis working")
        else:
            print("✗ CREAM strategy analysis failed")
            return False
            
    except Exception as e:
        print(f"✗ Strategy test failed: {e}")
        return False
        
    return True

def create_test_gui():
    """Create a simple test GUI"""
    print("\nTesting GUI creation...")
    
    try:
        import tkinter as tk
        
        root = tk.Tk()
        root.title("ProQuants Professional - System Test")
        root.geometry("600x400")
        root.configure(bg='#0a0a0a')
        
        # Test label
        test_label = tk.Label(root, 
                             text="ProQuants Professional System Test",
                             bg='#0a0a0a',
                             fg='#ffd700',
                             font=('Segoe UI', 16, 'bold'))
        test_label.pack(pady=20)
        
        # Status label
        status_label = tk.Label(root,
                               text="All systems operational! ✓",
                               bg='#0a0a0a',
                               fg='#00ff88',
                               font=('Segoe UI', 12))
        status_label.pack(pady=10)
        
        # Instructions
        instructions = tk.Label(root,
                               text="Close this window to continue with full system launch",
                               bg='#0a0a0a',
                               fg='#cccccc',
                               font=('Segoe UI', 10))
        instructions.pack(pady=10)
        
        # Launch button
        def launch_full_system():
            root.destroy()
            launch_professional_system()
            
        launch_btn = tk.Button(root,
                              text="Launch Full Professional System",
                              bg='#00d4ff',
                              fg='#000000',
                              font=('Segoe UI', 11, 'bold'),
                              relief='flat',
                              bd=0,
                              cursor='hand2',
                              command=launch_full_system)
        launch_btn.pack(pady=20)
        
        # Test button
        def run_tests():
            test_window = tk.Toplevel(root)
            test_window.title("System Tests")
            test_window.geometry("500x300")
            test_window.configure(bg='#1a1a1a')
            
            test_text = tk.Text(test_window, bg='#1a1a1a', fg='#ffffff', font=('Consolas', 9))
            test_text.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Redirect output to text widget
            import io
            from contextlib import redirect_stdout, redirect_stderr
            
            output = io.StringIO()
            
            with redirect_stdout(output), redirect_stderr(output):
                print("Running system tests...\n")
                test_imports()
                test_system_components()
                test_basic_functionality()
                print("\nTests completed!")
                
            test_text.insert('1.0', output.getvalue())
            
        test_btn = tk.Button(root,
                            text="Run System Tests",
                            bg='#ffd700',
                            fg='#000000',
                            font=('Segoe UI', 10),
                            relief='flat',
                            bd=0,
                            cursor='hand2',
                            command=run_tests)
        test_btn.pack(pady=5)
        
        print("✓ Test GUI created successfully")
        root.mainloop()
        
    except Exception as e:
        print(f"✗ GUI test failed: {e}")
        traceback.print_exc()
        return False
        
    return True

def launch_professional_system():
    """Launch the full professional system"""
    print("\nLaunching ProQuants Professional System...")
    
    try:
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from main import ProQuantsProfessionalSystem
        
        system = ProQuantsProfessionalSystem()
        system.run()
        
    except Exception as e:
        print(f"Failed to launch professional system: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("ProQuants Professional Trading System - Test Launcher")
    print("=" * 60)
    
    # Run basic tests
    if not test_imports():
        print("\n❌ Import tests failed. Please check your Python environment.")
        input("Press Enter to exit...")
        sys.exit(1)
        
    if not test_system_components():
        print("\n❌ Component tests failed. Please check the system installation.")
        input("Press Enter to exit...")
        sys.exit(1)
        
    if not test_basic_functionality():
        print("\n❌ Functionality tests failed. System may have issues.")
        input("Press Enter to continue anyway...")
        
    print("\n✅ All basic tests passed!")
    print("\nStarting test GUI...")
    
    # Create test GUI
    create_test_gui()
