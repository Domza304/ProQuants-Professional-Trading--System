"""
Test Script for ProQuants Neural Network Enhanced Trading System
Tests all components including MT5 integration, neural networks, and mathematical certainty
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_mt5_integration():
    """Test MT5 integration with proper Deriv symbols"""
    try:
        logger.info("Testing MT5 Integration...")
        
        from src.data.deriv_mt5_manager import DerivMT5Manager
        
        mt5_manager = DerivMT5Manager()
        
        # Test MT5 connection
        if mt5_manager.initialize_mt5():
            logger.info("✓ MT5 Connection successful")
            
            # Test symbol verification
            test_symbols = ['Volatility 75 Index', 'Volatility 25 Index', 'Volatility 10 Index']
            for symbol in test_symbols:
                is_valid = mt5_manager.verify_deriv_symbol(symbol)
                logger.info(f"✓ Symbol '{symbol}' verification: {'VALID' if is_valid else 'INVALID'}")
            
            # Test data collection
            test_symbol = 'Volatility 75 Index'
            data = mt5_manager.get_historical_data(test_symbol, hours=2)
            
            if data is not None and not data.empty:
                logger.info(f"✓ Historical data collected: {len(data)} data points")
                logger.info(f"  - Time range: {data.index[0]} to {data.index[-1]}")
                logger.info(f"  - Columns: {list(data.columns)}")
                
                # Test neural network dataset preparation
                nn_dataset = mt5_manager.prepare_neural_network_dataset(test_symbol, hours=12)
                if nn_dataset and nn_dataset.get('neural_network_ready'):
                    logger.info(f"✓ Neural network dataset ready: {nn_dataset['data_points']} points")
                else:
                    logger.warning("⚠ Neural network dataset not ready (insufficient data)")
                
                return True
            else:
                logger.error("✗ Failed to collect historical data")
                return False
        else:
            logger.error("✗ MT5 connection failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ MT5 integration test failed: {e}")
        return False

def test_neural_networks():
    """Test neural network system"""
    try:
        logger.info("Testing Neural Network System...")
        
        from src.ml.neural_networks import DerivNeuralNetworkSystem
        
        nn_system = DerivNeuralNetworkSystem()
        
        # Test system initialization
        logger.info(f"✓ Neural Network System initialized")
        logger.info(f"  - Available models: {list(nn_system.models.keys())}")
        logger.info(f"  - Sequence length: {nn_system.config['sequence_length']}")
        logger.info(f"  - Training epochs: {nn_system.config['training_epochs']}")
        
        # Create sample data for testing
        sample_data = create_sample_market_data()
        test_symbol = 'Volatility 75 Index'
        
        # Test data preparation
        prepared_data = nn_system.prepare_neural_network_data(sample_data, test_symbol)
        
        if prepared_data and prepared_data['data_points'] >= 100:
            logger.info(f"✓ Data preparation successful: {prepared_data['data_points']} points")
            
            # Test model training
            training_result = nn_system.train_models_for_symbol(test_symbol, prepared_data)
            
            if training_result['status'] == 'success':
                logger.info("✓ Neural network training successful")
                logger.info(f"  - Trained models: {list(training_result['trained_models'])}")
                
                # Test prediction
                current_features = extract_sample_features(sample_data)
                predictions = nn_system.predict_with_mathematical_certainty(test_symbol, current_features)
                
                if predictions['status'] == 'success':
                    logger.info("✓ Neural network predictions successful")
                    logger.info(f"  - Overall certainty: {predictions['overall_mathematical_certainty']:.3f}")
                    logger.info(f"  - Certainty level: {predictions['certainty_level']}")
                    logger.info(f"  - Price direction: {predictions['predictions']['price_direction']['prediction']}")
                    logger.info(f"  - Volatility regime: {predictions['predictions']['volatility_regime']['prediction']}")
                    
                    return True
                else:
                    logger.error(f"✗ Neural network prediction failed: {predictions.get('error', 'Unknown error')}")
                    return False
            else:
                logger.warning(f"⚠ Neural network training failed: {training_result.get('error', 'Unknown error')}")
                return False
        else:
            logger.warning("⚠ Insufficient data for neural network testing")
            return False
            
    except Exception as e:
        logger.error(f"✗ Neural network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cream_strategy():
    """Test CREAM strategy with neural network integration"""
    try:
        logger.info("Testing CREAM Strategy with Neural Networks...")
        
        from src.strategies.cream_strategy import CreamStrategy
        
        strategy = CreamStrategy()
        
        # Check neural network integration
        has_neural_networks = hasattr(strategy, 'neural_network') and strategy.neural_network is not None
        logger.info(f"✓ CREAM Strategy initialized")
        logger.info(f"  - Neural networks available: {'YES' if has_neural_networks else 'NO'}")
        
        # Create sample market data
        sample_data = create_sample_market_data()
        test_symbol = 'Volatility 75 Index'
        
        # Test traditional analysis
        fibonacci_levels = strategy.calculate_enhanced_fibonacci_levels(sample_data)
        logger.info(f"✓ Fibonacci levels calculated: {len(fibonacci_levels)} levels")
        
        # Test BOS detection
        bos_signal = strategy.detect_break_of_structure(sample_data)
        logger.info(f"✓ Break of Structure detection: {bos_signal}")
        
        # Test neural network analysis (if available)
        if has_neural_networks:
            nn_analysis = strategy.get_neural_network_analysis(test_symbol, sample_data)
            logger.info(f"✓ Neural network analysis: {nn_analysis.get('status', 'Unknown')}")
            
            if nn_analysis.get('status') == 'success':
                logger.info(f"  - Mathematical certainty: {nn_analysis['mathematical_certainty']}")
                logger.info(f"  - Recommendation: {nn_analysis['recommendation']['recommendation']}")
            elif nn_analysis.get('status') == 'neural_networks_not_available':
                logger.info("  - Neural networks not available (expected for testing)")
            else:
                logger.info(f"  - Status: {nn_analysis.get('status', 'Unknown')}")
        
        # Test complete signal generation
        trading_signal = strategy.generate_signal(sample_data, test_symbol)
        logger.info(f"✓ Trading signal generated: {trading_signal['action']}")
        logger.info(f"  - Confidence: {trading_signal['confidence']:.3f}")
        logger.info(f"  - Risk/Reward: {trading_signal.get('risk_reward_ratio', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ CREAM strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_integration():
    """Test complete system integration"""
    try:
        logger.info("Testing Complete System Integration...")
        
        from main import ProQuantsProfessionalSystem
        
        system = ProQuantsProfessionalSystem()
        
        # Test initialization
        if system.initialize():
            logger.info("✓ System initialization successful")
            
            # Check component availability
            logger.info(f"  - Dashboard: {'Available' if system.dashboard else 'Not Available'}")
            logger.info(f"  - Trading Engine: {'Available' if system.trading_engine else 'Not Available'}")
            logger.info(f"  - CREAM Strategy: {'Available' if system.cream_strategy else 'Not Available'}")
            logger.info(f"  - Neural Network Status: {system.neural_network_status}")
            
            # Test system status
            time.sleep(2)  # Allow update thread to run
            
            logger.info("✓ System integration test successful")
            return True
        else:
            logger.error("✗ System initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"✗ System integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_market_data():
    """Create sample market data for testing"""
    # Generate 1000 data points (about 83 hours of 5-minute data)
    dates = pd.date_range(start=datetime.now() - timedelta(days=4), periods=1000, freq='5T')
    
    # Simulate realistic price movement
    np.random.seed(42)  # For reproducible results
    price_base = 1000.0
    returns = np.random.normal(0, 0.002, 1000)  # 0.2% volatility
    
    # Add some trend and volatility clustering
    for i in range(1, len(returns)):
        if np.random.random() < 0.1:  # 10% chance of volatility spike
            returns[i] *= 3
        returns[i] += returns[i-1] * 0.05  # Small momentum effect
    
    prices = price_base * np.exp(np.cumsum(returns))
    
    # Create OHLC data
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    
    # Generate OHLC from close prices
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
    
    # High and Low with realistic spreads
    spread = np.random.uniform(0.001, 0.005, len(data))
    data['high'] = np.maximum(data['open'], data['close']) * (1 + spread)
    data['low'] = np.minimum(data['open'], data['close']) * (1 - spread)
    
    # Add volume
    data['tick_volume'] = np.random.randint(50, 500, len(data))
    
    return data

def extract_sample_features(market_data):
    """Extract features for neural network testing"""
    latest = market_data.iloc[-1]
    
    # Basic features
    features = [
        latest['open'], latest['high'], latest['low'], latest['close'],
        latest['tick_volume'],  # volume
        0.001,  # returns (placeholder)
        0.001,  # log_returns (placeholder)
        0.02,   # volatility (placeholder)
        0.01,   # hl_ratio
        0.001,  # oc_ratio
        0.005,  # upper_shadow
        0.005,  # lower_shadow
        1.0,    # volume_ratio
        50.0,   # rsi
        0.5,    # bb_position
        0.001,  # price_to_ma_5
        0.002,  # price_to_ma_10
        0.003,  # price_to_ma_20
        0.004,  # price_to_ma_50
        12,     # hour
        1,      # day_of_week
        1,      # is_london_session
        0       # is_ny_session
    ]
    
    return np.array(features)

def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("ProQuants Neural Network Enhanced Trading System Tests")
    logger.info("=" * 60)
    
    tests = [
        ("MT5 Integration", test_mt5_integration),
        ("Neural Networks", test_neural_networks),
        ("CREAM Strategy", test_cream_strategy),
        ("System Integration", test_system_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
            logger.info(f"✓ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"✗ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:<20}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Neural Network Enhanced Trading System is ready!")
    else:
        logger.warning(f"⚠ {total - passed} test(s) failed. Please review the results above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
