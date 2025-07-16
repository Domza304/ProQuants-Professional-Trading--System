"""
Test script for ML-Enhanced ProQuants Trading System
Verifies that all components work together correctly
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project path
project_root = os.path.dirname(__file__)
sys.path.append(os.path.join(project_root, 'src'))

def test_ml_system():
    """Test the ML-enhanced trading system"""
    print("🧪 Testing ProQuants ML-Enhanced Trading System...")
    
    try:
        # Add src to path
        import sys
        import os
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        sys.path.insert(0, src_path)
        
        # Test CREAM strategy import
        from strategies.cream_strategy import CREAMStrategy
        print("✅ CREAM Strategy imported successfully")
        
        # Create strategy instance
        strategy = CREAMStrategy()
        print("✅ Strategy instance created")
        
        # Create sample market data
        dates = pd.date_range(start='2025-01-01', periods=100, freq='5min')
        np.random.seed(42)
        
        # Generate realistic price data for Volatility 75 Index
        base_price = 500.0
        price_changes = np.random.normal(0, 0.002, 100)  # 0.2% volatility
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLC data
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'close': prices,
            'tick_volume': np.random.randint(50, 200, 100)
        })
        
        print("✅ Sample market data created")
        
        # Test CREAM analysis
        analysis = strategy.analyze_cream("Volatility 75 Index", sample_data)
        print("✅ CREAM analysis completed")
        
        # Print analysis results
        print(f"\n📊 Analysis Results:")
        print(f"Symbol: {analysis['symbol']}")
        print(f"Primary Signal: {analysis['primary_signal']}")
        print(f"Signal Strength: {analysis['signal_strength']:.1%}")
        print(f"Recommended Action: {analysis['recommended_action']}")
        print(f"Analysis Quality: {analysis['analysis_quality']}")
        print(f"ML Enhanced: {analysis.get('ml_enhanced', False)}")
        print(f"Manipulation Detected: {analysis.get('manipulation_detected', False)}")
        
        if analysis.get('active_setup'):
            setup = analysis['active_setup']
            print(f"\n🎯 Active Setup:")
            print(f"Direction: {setup['direction']}")
            print(f"Entry Price: {setup['entry_price']:.5f}")
            print(f"Stop Loss: {setup['stop_loss']:.5f}")
            print(f"Take Profit: {setup['take_profit']:.5f}")
            print(f"Risk/Reward Ratio: {setup['risk_reward_ratio']:.1f}")
            print(f"Setup Quality: {setup['setup_quality']}")
        
        # Test ML functionality if available
        if hasattr(strategy, 'ml_levels') and strategy.ml_levels is not None:
            print("\n🤖 ML System Status:")
            print("✅ ML module loaded successfully")
            
            # Test adaptive levels
            adaptive_info = strategy.get_adaptive_fibonacci_levels(sample_data)
            print(f"✅ Adaptive levels calculated: {adaptive_info['source']}")
            
            # Test manipulation detection
            if 'manipulation_detected' in adaptive_info:
                print(f"✅ Manipulation detection: {'DETECTED' if adaptive_info['manipulation_detected'] else 'CLEAR'}")
        else:
            print("\n⚠️ ML module not available - using static levels")
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard():
    """Test the professional dashboard"""
    print("\n🖥️ Testing Professional Dashboard...")
    
    try:
        from gui.professional_dashboard import ProfessionalDashboard
        print("✅ Dashboard imported successfully")
        print("✅ Dashboard ready to launch")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 ProQuants ML-Enhanced Trading System Test Suite")
    print("=" * 60)
    
    # Test ML trading system
    ml_test_passed = test_ml_system()
    
    # Test dashboard
    dashboard_test_passed = test_dashboard()
    
    print("\n" + "=" * 60)
    if ml_test_passed and dashboard_test_passed:
        print("🎉 ALL TESTS PASSED - System ready for trading!")
        print("\nTo start the system:")
        print("1. Run: python main.py")
        print("2. Click 'START TRADING' in the dashboard")
        print("3. Monitor ML adaptive levels and manipulation alerts")
    else:
        print("❌ Some tests failed - check error messages above")
    
    print("\n📚 For more information, see ML_ENHANCEMENT_GUIDE.md")
