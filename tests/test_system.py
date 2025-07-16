"""
Test suite for ProQuants Professional Trading System
"""

import unittest
import sys
import os

# Add src to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.strategies.cream_strategy import CREAMStrategy
from src.core.trading_engine import TradingEngine
from src.data.market_data_manager import MarketDataManager
import pandas as pd
import numpy as np

class TestCREAMStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = CREAMStrategy()
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
        np.random.seed(42)
        prices = 1000 + np.cumsum(np.random.randn(100) * 0.1)
        
        self.sample_data = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': prices + np.random.rand(100) * 0.5,
            'low': prices - np.random.rand(100) * 0.5,
            'close': prices,
            'tick_volume': np.random.randint(100, 1000, 100),
            'spread': np.random.randint(1, 5, 100),
            'real_volume': np.random.randint(1000, 10000, 100)
        })
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        self.sample_data['high'] = np.maximum.reduce([
            self.sample_data['open'], 
            self.sample_data['close'], 
            self.sample_data['high']
        ])
        self.sample_data['low'] = np.minimum.reduce([
            self.sample_data['open'], 
            self.sample_data['close'], 
            self.sample_data['low']
        ])
        
    def test_analyze_returns_dict(self):
        """Test that analyze returns a dictionary"""
        result = self.strategy.analyze(self.sample_data, "TEST_SYMBOL")
        self.assertIsInstance(result, dict)
        
    def test_analyze_required_keys(self):
        """Test that analyze returns required keys"""
        result = self.strategy.analyze(self.sample_data, "TEST_SYMBOL")
        required_keys = ['symbol', 'timestamp', 'clean', 'range', 'easy', 'accuracy', 'momentum', 'overall_signal']
        for key in required_keys:
            self.assertIn(key, result)
            
    def test_analyze_insufficient_data(self):
        """Test analyze with insufficient data"""
        small_data = self.sample_data.head(5)
        result = self.strategy.analyze(small_data, "TEST_SYMBOL")
        self.assertEqual(result['overall_signal'], 'INSUFFICIENT_DATA')
        
    def test_clean_analysis(self):
        """Test clean component analysis"""
        clean_result = self.strategy.analyze_clean(self.sample_data)
        self.assertIsInstance(clean_result, dict)
        self.assertIn('signal', clean_result)
        self.assertIn('trend_direction', clean_result)
        
    def test_range_analysis(self):
        """Test range component analysis"""
        range_result = self.strategy.analyze_range(self.sample_data)
        self.assertIsInstance(range_result, dict)
        self.assertIn('signal', range_result)
        self.assertIn('range_state', range_result)
        
    def test_easy_analysis(self):
        """Test easy component analysis"""
        easy_result = self.strategy.analyze_easy(self.sample_data)
        self.assertIsInstance(easy_result, dict)
        self.assertIn('signal', easy_result)
        self.assertIn('entry_signal', easy_result)
        
    def test_accuracy_analysis(self):
        """Test accuracy component analysis"""
        accuracy_result = self.strategy.analyze_accuracy(self.sample_data)
        self.assertIsInstance(accuracy_result, dict)
        self.assertIn('signal', accuracy_result)
        self.assertIn('confidence_level', accuracy_result)
        
    def test_momentum_analysis(self):
        """Test momentum component analysis"""
        momentum_result = self.strategy.analyze_momentum(self.sample_data)
        self.assertIsInstance(momentum_result, dict)
        self.assertIn('signal', momentum_result)
        self.assertIn('direction', momentum_result)

class TestTradingEngine(unittest.TestCase):
    def setUp(self):
        # Initialize engine in offline mode for testing
        self.engine = TradingEngine()
        self.engine.offline_mode = True
        
    def test_engine_initialization(self):
        """Test trading engine initialization"""
        self.assertIsNotNone(self.engine)
        self.assertIsInstance(self.engine.config, dict)
        
    def test_get_market_data_offline(self):
        """Test getting market data in offline mode"""
        data = self.engine.get_market_data("TEST_SYMBOL", "M1", 50)
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        
    def test_symbol_info_offline(self):
        """Test getting symbol info in offline mode"""
        info = self.engine.get_symbol_info("TEST_SYMBOL")
        self.assertIsNotNone(info)
        self.assertIsInstance(info, dict)
        self.assertIn('bid', info)
        self.assertIn('ask', info)
        
    def test_simulate_order(self):
        """Test order simulation"""
        result = self.engine.simulate_order("TEST_SYMBOL", "BUY", 0.01, 1000.0)
        self.assertIsInstance(result, dict)
        self.assertIn('retcode', result)
        self.assertIn('volume', result)
        
    def test_toggle_offline_mode(self):
        """Test offline mode toggle"""
        original_mode = self.engine.offline_mode
        self.engine.toggle_offline_mode()
        self.assertNotEqual(original_mode, self.engine.offline_mode)

class TestMarketDataManager(unittest.TestCase):
    def setUp(self):
        self.manager = MarketDataManager("test_data")
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='1min')
        self.sample_df = pd.DataFrame({
            'time': dates,
            'open': np.random.rand(50) * 100 + 1000,
            'high': np.random.rand(50) * 100 + 1050,
            'low': np.random.rand(50) * 100 + 950,
            'close': np.random.rand(50) * 100 + 1000,
            'volume': np.random.randint(100, 1000, 50)
        })
        
    def tearDown(self):
        # Clean up test data
        import shutil
        if os.path.exists("test_data"):
            shutil.rmtree("test_data")
            
    def test_store_and_retrieve_data(self):
        """Test storing and retrieving market data"""
        self.manager.store_market_data("TEST_SYMBOL", "M1", self.sample_df)
        retrieved = self.manager.get_cached_data("TEST_SYMBOL", "M1")
        
        self.assertIsNotNone(retrieved)
        self.assertIsInstance(retrieved, pd.DataFrame)
        self.assertEqual(len(retrieved), len(self.sample_df))
        
    def test_cache_stats(self):
        """Test cache statistics"""
        self.manager.store_market_data("TEST_SYMBOL", "M1", self.sample_df)
        stats = self.manager.get_cache_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_entries', stats)
        self.assertGreater(stats['total_entries'], 0)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestCREAMStrategy))
    test_suite.addTest(unittest.makeSuite(TestTradingEngine))
    test_suite.addTest(unittest.makeSuite(TestMarketDataManager))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
