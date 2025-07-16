"""
ProQuants CREAM Strategy Implementation - ML-Enhanced Goloji Bhudasi Trading Logic
Original CREAM + BOS (Break of Structure) + Fibonacci Integration + Machine Learning
Based on the proven Goloji_Bhudasi_Live_Engine backbone with adaptive optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import sys
import os

# Add ML module to path
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
    from adaptive_levels import AdaptiveLevelsML
    from neural_networks import DerivNeuralNetworkSystem
    from deriv_mt5_manager import DerivMT5Manager
    ML_AVAILABLE = True
    NEURAL_NETWORKS_AVAILABLE = True
except ImportError as e:
    # Fallback if ML modules not available
    ML_AVAILABLE = False
    NEURAL_NETWORKS_AVAILABLE = False
    AdaptiveLevelsML = None
    DerivNeuralNetworkSystem = None
    DerivMT5Manager = None

class CREAMStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or self.get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Strategy state
        self.signals = {}
        self.last_analysis = {}
        
        # Initialize ML-Enhanced Adaptive Levels System
        if ML_AVAILABLE:
            self.ml_levels = AdaptiveLevelsML()
            
            # Load previous learning data if available
            learning_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'ml_learning_data.json')
            load_result = self.ml_levels.load_learning_data(learning_data_path)
            if load_result["status"] == "success":
                self.logger.info(f"ML Learning: Loaded {load_result['trades_loaded']} historical trades")
        else:
            self.ml_levels = None
            self.logger.warning("ML module not available - using static fibonacci levels")
        
        # Initialize Neural Network System for Mathematical Certainty
        if NEURAL_NETWORKS_AVAILABLE:
            self.neural_network = DerivNeuralNetworkSystem()
            self.logger.info("Neural Network System initialized for mathematical precision")
        else:
            self.neural_network = None
            self.logger.warning("Neural Networks not available - using traditional analysis")
        
        # Initialize MT5 Data Manager for proper Deriv symbol handling
        if ML_AVAILABLE:  # Same availability check as other modules
            self.mt5_manager = DerivMT5Manager()
            if self.mt5_manager.initialize_mt5():
                self.logger.info("MT5 connection established with Deriv synthetic indices")
                # Update our symbol list with proper MT5 symbol names
                self.TRADING_SYMBOLS = list(self.mt5_manager.DERIV_SYMBOLS.values())
            else:
                self.logger.warning("MT5 connection failed - using offline mode")
                self.mt5_manager = None
        else:
            self.mt5_manager = None
        
        # Base Goloji Bhudasi Fibonacci Levels (Reference Standards)
        self.BASE_FIB_LEVELS = {
            # BOS Detection Levels
            "BOS_TEST":        1.15,    # Breakout Test Level
            "SL2":             1.05,    # Secondary Stop-Loss  
            "ONE_HUNDRED":     1.00,    # 100% Retracement (Full Rollback)
            "LR":              0.88,    # Liquidity Run Entry
            "SL1":             0.76,    # Primary Stop-Loss
            "ME":              0.685,   # Main Entry Point
            "ZERO":            0.00,    # Base Price Level
            "TP":             -0.15,    # Primary Take-Profit
            "MINUS_35":       -0.35,    # New Extended TP
            "MINUS_62":       -0.62     # Extreme Profit Target
        }
        
        # Multi-Instrument Configuration for Deriv Synthetic Indices
        self.DERIV_SYMBOLS = {
            "Volatility 75 Index": {
                "instrument_type": "synthetic_index",
                "volatility_profile": "high",
                "typical_daily_range": 0.08,  # 8% typical daily range
                "min_data_hours": 12
            },
            "Volatility 25 Index": {
                "instrument_type": "synthetic_index", 
                "volatility_profile": "medium",
                "typical_daily_range": 0.06,  # 6% typical daily range
                "min_data_hours": 12
            },
            "Volatility 75 (1s) Index": {
                "instrument_type": "synthetic_index",
                "volatility_profile": "very_high", 
                "typical_daily_range": 0.12,  # 12% typical daily range
                "min_data_hours": 12
            }
        }
        self.MIN_RRR = 4.0  # Minimum 1:4 Risk-to-Reward ratio for all Deriv synthetics
        
        # ML Enhancement tracking
        self.ml_predictions = {}
        self.manipulation_alerts = []
        
    def get_default_config(self) -> Dict:
        """Get default CREAM strategy configuration based on Goloji Bhudasi logic"""
        return {
            "lookback_periods": 12,  # Reduced for faster signals
            "signal_threshold": 0.7,
            "swing_lookback": 5,
            "min_swing_size": 2,
            "min_impulse_ratio": 0.3,
            "max_bos_count": 3,
            "bos_lookback": 30,
            "trend_lookback": 8,
            "min_distance": 0.002,
            "consolidation_threshold": 0.001,
            "min_trend_swing": 0.002
        }
        
    def find_last_two_swings_advanced(self, df, direction="bull", swing_lookback=5, min_swing_size=2):
        """
        Finds the last two significant swing highs (bear) or swing lows (bull) in the dataframe.
        Uses a rolling window to detect local extrema and filters by minimum swing size.
        Returns (anchor1, anchor2) as (price, index).
        """
        if direction == "bull":
            lows = df['low']
            swing_lows = []
            for i in range(swing_lookback, len(lows) - swing_lookback):
                is_swing_low = all(lows.iloc[i] <= lows.iloc[j] for j in range(i-swing_lookback, i+swing_lookback+1) if j != i)
                if is_swing_low and (not swing_lows or abs(lows.iloc[i] - swing_lows[-1][0]) > min_swing_size * 0.0001):
                    swing_lows.append((lows.iloc[i], i))
                    
            if len(swing_lows) < 2:
                return None
                
            anchor2 = swing_lows[-1]
            anchor1 = swing_lows[-2]
            return (anchor1, anchor2)
        else:
            highs = df['high']
            swing_highs = []
            for i in range(swing_lookback, len(highs) - swing_lookback):
                is_swing_high = all(highs.iloc[i] >= highs.iloc[j] for j in range(i-swing_lookback, i+swing_lookback+1) if j != i)
                if is_swing_high and (not swing_highs or abs(highs.iloc[i] - swing_highs[-1][0]) > min_swing_size * 0.0001):
                    swing_highs.append((highs.iloc[i], i))
                    
            if len(swing_highs) < 2:
                return None
                
            anchor2 = swing_highs[-1]
            anchor1 = swing_highs[-2]
            return (anchor1, anchor2)

    def is_choppy(self, df, lookback=8, min_distance=0.002):
        """Check if market is choppy/ranging"""
        closes = df['close'][-lookback:]
        alternations = np.sum(np.diff(np.sign(closes.diff().fillna(0))) != 0)
        recent_high = df['high'][-lookback:].max()
        recent_low = df['low'][-lookback:].min()
        return alternations > (lookback // 2) or (recent_high - recent_low) < min_distance

    def is_consolidating(self, df, lookback=12, threshold=0.001):
        """Check if market is consolidating"""
        recent_high = df['high'][-lookback:].max()
        recent_low = df['low'][-lookback:].min()
        return (recent_high - recent_low) < threshold

    def get_trend_structure(self, df, lookback=8, min_swing=0.002):
        """Determine trend structure: strong_bull, strong_bear, weak_trend, or none"""
        highs = df['high'][-lookback:]
        lows = df['low'][-lookback:]
        
        hh = sum(highs.iloc[i] > highs.iloc[i-1] for i in range(1, len(highs)))
        hl = sum(lows.iloc[i] > lows.iloc[i-1] for i in range(1, len(lows)))
        ll = sum(lows.iloc[i] < lows.iloc[i-1] for i in range(1, len(lows)))
        lh = sum(highs.iloc[i] < highs.iloc[i-1] for i in range(1, len(highs)))
        
        swing_size = highs.max() - lows.min()
        
        if hh >= 3 and hl >= 3 and swing_size > min_swing:
            return "strong_bull"
        elif ll >= 3 and lh >= 3 and swing_size > min_swing:
            return "strong_bear"
        elif (hh >= 2 and hl >= 2 and swing_size > min_swing/2) or (ll >= 2 and lh >= 2 and swing_size > min_swing/2):
            return "weak_trend"
        else:
            return "none"

    def is_clean_trend(self, df, lookback=8):
        """Check if trend is clean (minimal alternations)"""
        highs = df['high'][-lookback:]
        lows = df['low'][-lookback:]
        alternations = sum((highs.iloc[i] > highs.iloc[i-1]) != (lows.iloc[i] > lows.iloc[i-1]) for i in range(1, len(highs)))
        tight_range = (highs.max() - lows.min()) < 0.001
        return alternations < 2 and not tight_range

    def impulse_length_and_momentum(self, df, idx1, idx2, direction="bear"):
        """
        Returns the impulse length (number of candles) and momentum (price change per candle)
        between two swing points in the current TF.
        """
        if idx2 <= idx1:
            return 0, 0
            
        closes = df['close'].iloc[idx1:idx2+1]
        if direction == "bear":
            price_change = closes.iloc[0] - closes.iloc[-1]
        else:
            price_change = closes.iloc[-1] - closes.iloc[0]
            
        length = idx2 - idx1
        momentum = price_change / length if length > 0 else 0
        return length, momentum

    def detect_bos(self, df, direction):
        """
        Detects Break of Structure (BOS) using last two significant swings.
        BOS is confirmed only if price touches/surpasses BOS_TEST FIB level (1.15).
        Returns (bos_detected, (swing1_idx, swing2_idx))
        """
        BOS_TEST_FIB = self.BASE_FIB_LEVELS["BOS_TEST"]
        
        if direction == "strong_bull":
            anchors = self.find_last_two_swings_advanced(df, direction="bull", swing_lookback=5, min_swing_size=2)
            if anchors is None or anchors[0] is None or anchors[1] is None:
                return False, None
                
            (swing1, idx1), (swing2, idx2) = anchors
            fib_start = df['low'].iloc[idx1]
            fib_end = df['low'].iloc[idx2]
            bos_test_level = fib_start + (fib_end - fib_start) * BOS_TEST_FIB
            
            # BOS confirmed if any candle's high or close >= bos_test_level
            bos = (df['high'] >= bos_test_level).any() or (df['close'] >= bos_test_level).any()
            return bos, (idx1, idx2)
            
        elif direction == "strong_bear":
            anchors = self.find_last_two_swings_advanced(df, direction="bear", swing_lookback=5, min_swing_size=2)
            if anchors is None or anchors[0] is None or anchors[1] is None:
                return False, None
                
            (swing1, idx1), (swing2, idx2) = anchors
            fib_start = df['high'].iloc[idx1]
            fib_end = df['high'].iloc[idx2]
            bos_test_level = fib_start - (fib_start - fib_end) * BOS_TEST_FIB
            
            # BOS confirmed if any candle's low or close <= bos_test_level
            bos = (df['low'] <= bos_test_level).any() or (df['close'] <= bos_test_level).any()
            return bos, (idx1, idx2)
            
        return False, None

    def calculate_fibonacci_levels(self, high_price, low_price):
        """Calculate Fibonacci retracement and extension levels"""
        price_range = high_price - low_price
        levels = {}
        
        for name, ratio in self.BASE_FIB_LEVELS.items():
            if ratio >= 0:
                levels[name] = low_price + (price_range * ratio)
            else:
                # Extension levels (negative ratios)
                levels[name] = high_price + (price_range * abs(ratio))
                
        return levels

    def get_trade_setup(self, df, direction):
        """
        Generate complete trade setup based on Goloji Bhudasi logic
        Returns entry, stop loss, take profit levels with minimum 1:4 RRR
        """
        if direction not in ["strong_bull", "strong_bear"]:
            return None
            
        # Detect BOS first
        bos_detected, swing_indices = self.detect_bos(df, direction)
        if not bos_detected or swing_indices is None:
            return None
            
        idx1, idx2 = swing_indices
        current_price = df['close'].iloc[-1]
        
        if direction == "strong_bull":
            # For bullish BOS, look for pullback to fibonacci levels
            swing_high = df['high'].iloc[max(idx1, idx2)]
            swing_low = df['low'].iloc[min(idx1, idx2)]
            
            fib_levels = self.calculate_fibonacci_levels(swing_high, swing_low)
            
            # Entry at fibonacci pullback levels
            entry_level = fib_levels.get("FIB_618", current_price)
            stop_loss = swing_low * 0.995  # Just below swing low
            
            # Take profit with minimum 1:4 RRR
            risk = abs(entry_level - stop_loss)
            min_tp = entry_level + (risk * self.MIN_RRR)
            take_profit = max(min_tp, fib_levels.get("TP_MIN", min_tp))
            
        else:  # strong_bear
            # For bearish BOS, look for pullback to fibonacci levels
            swing_high = df['high'].iloc[max(idx1, idx2)]
            swing_low = df['low'].iloc[min(idx1, idx2)]
            
            fib_levels = self.calculate_fibonacci_levels(swing_high, swing_low)
            
            # Entry at fibonacci pullback levels
            entry_level = fib_levels.get("FIB_618", current_price)
            stop_loss = swing_high * 1.005  # Just above swing high
            
            # Take profit with minimum 1:4 RRR
            risk = abs(stop_loss - entry_level)
            min_tp = entry_level - (risk * self.MIN_RRR)
            take_profit = min(min_tp, swing_low + fib_levels.get("TP_MIN", 0))
            
        # Calculate RRR
        actual_risk = abs(entry_level - stop_loss)
        actual_reward = abs(take_profit - entry_level)
        rrr = actual_reward / actual_risk if actual_risk > 0 else 0
        
        # Only return setup if RRR meets minimum requirement
        if rrr >= self.MIN_RRR:
            return {
                "direction": direction,
                "entry_price": entry_level,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": rrr,
                "fibonacci_levels": fib_levels,
                "swing_indices": swing_indices,
                "setup_quality": "HIGH" if rrr >= 5.0 else "GOOD"
            }
            
        return None

    def analyze_cream(self, symbol: str, data: pd.DataFrame) -> Dict:
        """
        Perform complete CREAM analysis using ML-Enhanced Goloji Bhudasi logic
        Focuses on BOS detection, adaptive fibonacci levels, manipulation protection, and high RRR setups
        """
        if data is None or len(data) < self.config["lookback_periods"]:
            return self.get_empty_analysis(symbol)
            
        try:
            # Get ML-enhanced adaptive fibonacci levels
            adaptive_levels_info = self.get_adaptive_fibonacci_levels(data)
            
            # Determine market structure
            trend_structure = self.get_trend_structure(data)
            is_clean_trend_result = self.is_clean_trend(data)
            is_choppy_result = self.is_choppy(data)
            is_consolidating_result = self.is_consolidating(data)
            
            # Look for BOS in both directions
            bull_bos, bull_swings = self.detect_bos(data, "strong_bull")
            bear_bos, bear_swings = self.detect_bos(data, "strong_bear")
            
            # Generate ML-enhanced trade setups
            bull_setup = self.get_ml_enhanced_trade_setup(data, "strong_bull") if bull_bos else None
            bear_setup = self.get_ml_enhanced_trade_setup(data, "strong_bear") if bear_bos else None
            
            # Determine primary signal with ML considerations
            if bull_setup and bull_setup["risk_reward_ratio"] >= self.MIN_RRR:
                primary_signal = "BULLISH_BOS_ML"
                signal_strength = min(1.0, bull_setup["risk_reward_ratio"] / 6.0)
                # Boost signal strength if high ML confidence
                if bull_setup.get("ml_info", {}).get("model_confidence"):
                    avg_confidence = np.mean(list(bull_setup["ml_info"]["model_confidence"].values()))
                    signal_strength *= (1 + avg_confidence * 0.2)  # Up to 20% boost
                recommended_action = "BUY_SETUP_ML"
                active_setup = bull_setup
            elif bear_setup and bear_setup["risk_reward_ratio"] >= self.MIN_RRR:
                primary_signal = "BEARISH_BOS_ML"
                signal_strength = min(1.0, bear_setup["risk_reward_ratio"] / 6.0)
                # Boost signal strength if high ML confidence
                if bear_setup.get("ml_info", {}).get("model_confidence"):
                    avg_confidence = np.mean(list(bear_setup["ml_info"]["model_confidence"].values()))
                    signal_strength *= (1 + avg_confidence * 0.2)  # Up to 20% boost
                recommended_action = "SELL_SETUP_ML"
                active_setup = bear_setup
            else:
                primary_signal = "NO_ML_SETUP"
                signal_strength = 0.0
                recommended_action = "WAIT_ML"
                active_setup = None
                
            # Create comprehensive analysis
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "trend_structure": trend_structure,
                "is_clean_trend": is_clean_trend_result,
                "is_choppy": is_choppy_result,
                "is_consolidating": is_consolidating_result,
                "bull_bos_detected": bull_bos,
                "bear_bos_detected": bear_bos,
                "bull_setup": bull_setup,
                "bear_setup": bear_setup,
                "primary_signal": primary_signal,
                "signal_strength": signal_strength,
                "recommended_action": recommended_action,
                "active_setup": active_setup,
                "current_price": data['close'].iloc[-1],
                "analysis_quality": "HIGH_ML" if active_setup else "LOW",
                
                # ML-specific information
                "ml_enhanced": ML_AVAILABLE,
                "adaptive_levels": adaptive_levels_info,
                "manipulation_detected": adaptive_levels_info.get("manipulation_detected", False),
                "manipulation_type": adaptive_levels_info.get("manipulation_type"),
                "manipulation_confidence": adaptive_levels_info.get("manipulation_confidence", 0.0),
                "ml_model_status": {
                    "models_trained": adaptive_levels_info.get("models_trained", False),
                    "trade_count": adaptive_levels_info.get("trade_count", 0),
                    "last_training": adaptive_levels_info.get("last_training")
                }
            }
            
            # Store for future reference
            self.last_analysis[symbol] = analysis
            
            # Log important ML events
            if analysis["manipulation_detected"]:
                self.logger.warning(f"MANIPULATION ALERT for {symbol}: {analysis['manipulation_type']} "
                                 f"(Confidence: {analysis['manipulation_confidence']:.1%})")
            
            if active_setup and active_setup.get("setup_quality", "").endswith("_ML"):
                self.logger.info(f"ML-Enhanced setup for {symbol}: {active_setup['setup_quality']} "
                               f"RRR: {active_setup['risk_reward_ratio']:.1f}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"ML-Enhanced CREAM analysis failed for {symbol}: {e}")
            return self.get_empty_analysis(symbol)
    def get_empty_analysis(self, symbol: str) -> Dict:
        """Return empty analysis structure for error cases"""
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "trend_structure": "UNKNOWN",
            "is_clean_trend": False,
            "is_choppy": True,
            "is_consolidating": True,
            "bull_bos_detected": False,
            "bear_bos_detected": False,
            "bull_setup": None,
            "bear_setup": None,
            "primary_signal": "INSUFFICIENT_DATA",
            "signal_strength": 0.0,
            "recommended_action": "WAIT",
            "active_setup": None,
            "current_price": 0.0,
            "analysis_quality": "LOW"
        }
        
    def get_last_analysis(self, symbol: str) -> Optional[Dict]:
        """Get the last analysis for a symbol"""
        return self.last_analysis.get(symbol)
        
    def get_signal_summary(self, symbol: str) -> str:
        """Get a human-readable signal summary"""
        analysis = self.get_last_analysis(symbol)
        if not analysis:
            return "No analysis available"
            
        setup_info = ""
        if analysis.get("active_setup"):
            setup = analysis["active_setup"]
            setup_info = f" | RRR: {setup['risk_reward_ratio']:.1f} | Entry: {setup['entry_price']:.5f}"
            
        return f"{analysis['primary_signal']} ({analysis['signal_strength']:.1%}) - {analysis['recommended_action']}{setup_info}"

    def should_enter_trade(self, symbol: str, current_price: float) -> Dict:
        """
        Determine if we should enter a trade based on current analysis and price action
        """
        analysis = self.get_last_analysis(symbol)
        if not analysis or not analysis.get("active_setup"):
            return {"should_enter": False, "reason": "No active setup"}
            
        setup = analysis["active_setup"]
        direction = setup["direction"]
        entry_price = setup["entry_price"]
        
        # Check if current price is near entry level (within 0.1% tolerance)
        price_tolerance = entry_price * 0.001
        
        if direction == "strong_bull":
            # For bullish setup, enter when price pulls back to fibonacci level
            if abs(current_price - entry_price) <= price_tolerance:
                return {
                    "should_enter": True,
                    "direction": "BUY",
                    "entry_price": current_price,
                    "stop_loss": setup["stop_loss"],
                    "take_profit": setup["take_profit"],
                    "reason": "Fibonacci pullback entry - Bullish BOS confirmed"
                }
        else:  # strong_bear
            # For bearish setup, enter when price pulls back to fibonacci level
            if abs(current_price - entry_price) <= price_tolerance:
                return {
                    "should_enter": True,
                    "direction": "SELL",
                    "entry_price": current_price,
                    "stop_loss": setup["stop_loss"],
                    "take_profit": setup["take_profit"],
                    "reason": "Fibonacci pullback entry - Bearish BOS confirmed"
                }
                
        return {"should_enter": False, "reason": "Price not at entry level"}

    def validate_setup_quality(self, setup: Dict) -> str:
        """
        Validate the quality of a trade setup based on Goloji Bhudasi criteria
        """
        if not setup:
            return "NO_SETUP"
            
        rrr = setup.get("risk_reward_ratio", 0)
        
        if rrr >= 6.0:
            return "EXCELLENT"
        elif rrr >= 5.0:
            return "VERY_GOOD"
        elif rrr >= 4.0:
            return "GOOD"
        elif rrr >= 3.0:
            return "ACCEPTABLE"
        else:
            return "POOR"
        
    def get_adaptive_fibonacci_levels(self, symbol: str, market_data: pd.DataFrame) -> Dict:
        """
        Get ML-optimized fibonacci levels for specific Deriv synthetic instrument
        Each instrument learns independently due to unique behavioral patterns
        """
        if not ML_AVAILABLE or self.ml_levels is None:
            # Fallback to base levels
            return {
                "levels": self.BASE_FIB_LEVELS,
                "source": "static_base_levels",
                "manipulation_detected": False,
                "confidence": 0.0,
                "instrument": symbol
            }
        
        # Validate instrument is a supported Deriv synthetic
        if symbol not in self.DERIV_SYMBOLS:
            self.logger.warning(f"Unknown Deriv instrument: {symbol}")
            return {
                "levels": self.BASE_FIB_LEVELS,
                "source": "unsupported_instrument",
                "manipulation_detected": False,
                "confidence": 0.0,
                "instrument": symbol
            }
        
        try:
            # Check minimum data requirement (12 hours for Deriv synthetics)
            min_hours = self.DERIV_SYMBOLS[symbol]["min_data_hours"]
            required_candles = min_hours * 12  # Assuming 5-min candles
            
            if len(market_data) < required_candles:
                self.logger.info(f"Insufficient data for {symbol}: {len(market_data)}/{required_candles} candles")
                return {
                    "levels": self.BASE_FIB_LEVELS,
                    "source": "insufficient_data",
                    "manipulation_detected": False,
                    "confidence": 0.0,
                    "instrument": symbol,
                    "data_hours": len(market_data) / 12,
                    "required_hours": min_hours
                }
            
            # Get instrument-specific adaptive levels from ML system
            ml_result = self.ml_levels.get_instrument_adaptive_levels(symbol, market_data)
            
            # Use safety-adjusted levels if manipulation detected
            active_levels = ml_result.get("safety_adjusted_levels", ml_result.get("adaptive_levels", self.BASE_FIB_LEVELS))
            
            result = {
                "levels": active_levels,
                "base_levels": ml_result.get("base_levels", self.BASE_FIB_LEVELS),
                "source": "ml_adaptive_deriv",
                "instrument": symbol,
                "manipulation_detected": ml_result.get("manipulation_check", {}).get("manipulation_detected", False),
                "manipulation_type": ml_result.get("manipulation_check", {}).get("manipulation_type"),
                "manipulation_confidence": ml_result.get("manipulation_check", {}).get("confidence", 0.0),
                "model_confidence": ml_result.get("confidence_scores", {}),
                "models_trained": ml_result.get("models_trained", False),
                "trade_count": ml_result.get("trade_count", 0),
                "last_training": ml_result.get("last_training"),
                "data_hours": len(market_data) / 12,
                "volatility_profile": self.DERIV_SYMBOLS[symbol]["volatility_profile"]
            }
            
            # Log manipulation alerts for Deriv synthetics
            if result["manipulation_detected"]:
                self.logger.warning(f"DERIV MANIPULATION DETECTED on {symbol}: {result['manipulation_type']} "
                                 f"(Confidence: {result['manipulation_confidence']:.1%})")
                self.manipulation_alerts.append({
                    "timestamp": datetime.now(),
                    "instrument": symbol,
                    "type": result["manipulation_type"],
                    "confidence": result["manipulation_confidence"]
                })
                
                # Keep only recent alerts (per instrument)
                if len(self.manipulation_alerts) > 100:
                    self.manipulation_alerts = self.manipulation_alerts[-100:]
            
            return result
            
        except Exception as e:
            self.logger.error(f"ML adaptive levels failed for {symbol}: {e}")
            return {
                "levels": self.BASE_FIB_LEVELS,
                "source": "fallback_static",
                "manipulation_detected": False,
                "error": str(e),
                "instrument": symbol
            }

    def calculate_ml_fibonacci_levels(self, high_price: float, low_price: float, symbol: str, market_data: pd.DataFrame) -> Dict:
        """
        Calculate Fibonacci retracement and extension levels using ML-optimized ratios
        Specific to Deriv synthetic instruments with independent learning
        """
        # Get instrument-specific adaptive levels
        adaptive_result = self.get_adaptive_fibonacci_levels(symbol, market_data)
        fib_ratios = adaptive_result["levels"]
        
        price_range = high_price - low_price
        levels = {}
        
        for name, ratio in fib_ratios.items():
            if ratio >= 0:
                levels[name] = low_price + (price_range * ratio)
            else:
                # Extension levels (negative ratios)
                levels[name] = high_price + (price_range * abs(ratio))
                
        # Add Deriv-specific metadata
        levels_with_meta = {
            "fibonacci_levels": levels,
            "price_range": price_range,
            "high_price": high_price,
            "low_price": low_price,
            "adaptive_info": adaptive_result,
            "instrument": symbol,
            "volatility_profile": self.DERIV_SYMBOLS.get(symbol, {}).get("volatility_profile", "unknown"),
            "data_quality": "sufficient" if adaptive_result.get("data_hours", 0) >= 12 else "insufficient"
        }
        
        return levels_with_meta

    def get_ml_enhanced_trade_setup(self, df: pd.DataFrame, direction: str, symbol: str) -> Optional[Dict]:
        """
        Generate ML-enhanced trade setup with Deriv synthetic-specific adaptive levels
        Each instrument uses its own learned patterns and manipulation protection
        """
        if direction not in ["strong_bull", "strong_bear"]:
            return None
            
        if symbol not in self.DERIV_SYMBOLS:
            self.logger.warning(f"Unsupported Deriv instrument: {symbol}")
            return None
            
        # Detect BOS first
        bos_detected, swing_indices = self.detect_bos(df, direction)
        if not bos_detected or swing_indices is None:
            return None
            
        idx1, idx2 = swing_indices
        current_price = df['close'].iloc[-1]
        
        # Get swing points
        if direction == "strong_bull":
            swing_high = df['high'].iloc[max(idx1, idx2)]
            swing_low = df['low'].iloc[min(idx1, idx2)]
        else:  # strong_bear
            swing_high = df['high'].iloc[max(idx1, idx2)]
            swing_low = df['low'].iloc[min(idx1, idx2)]
            
        # Calculate Deriv-specific ML-enhanced fibonacci levels
        ml_fib_result = self.calculate_ml_fibonacci_levels(swing_high, swing_low, symbol, df)
        fib_levels = ml_fib_result["fibonacci_levels"]
        adaptive_info = ml_fib_result["adaptive_info"]
        
        # Check data quality for Deriv synthetics
        if ml_fib_result["data_quality"] == "insufficient":
            self.logger.warning(f"Insufficient data for {symbol}: {adaptive_info.get('data_hours', 0):.1f}/12 hours")
            # Still proceed but with lower confidence
            
        # Generate trade setup with Deriv-adaptive levels
        if direction == "strong_bull":
            # Entry at ML-optimized fibonacci pullback levels for this specific Deriv instrument
            me_level = fib_levels.get("ME", current_price)
            lr_level = fib_levels.get("LR", current_price)
            
            # Deriv-specific entry logic (volatility-adjusted)
            volatility_profile = self.DERIV_SYMBOLS[symbol]["volatility_profile"]
            if volatility_profile == "very_high":
                # For V75(1s), prefer deeper pullbacks
                entry_level = lr_level if abs(current_price - lr_level) < abs(current_price - me_level) else me_level
            else:
                # For V75, V25, use standard ME level preference
                entry_level = me_level if abs(current_price - me_level) < abs(current_price - lr_level) else lr_level
                
            # ML-optimized stop loss for Deriv synthetics
            sl1_level = fib_levels.get("SL1", swing_low * 0.995)
            
            # Deriv-specific SL adjustment based on manipulation detection
            if adaptive_info.get("manipulation_detected", False):
                manipulation_type = adaptive_info.get("manipulation_type", "")
                if "WICK_HUNTING" in manipulation_type:
                    # Move SL further away during wick hunting
                    stop_loss = min(sl1_level * 0.998, swing_low * 0.993)
                else:
                    stop_loss = min(sl1_level, swing_low * 0.998)
            else:
                stop_loss = min(sl1_level, swing_low * 0.998)
                
            # Take profit with Deriv-specific ML optimization
            tp_primary = fib_levels.get("TP", entry_level * 1.04)
            tp_extended = fib_levels.get("MINUS_35", entry_level * 1.06)
            
            # Use extended TP if high confidence and sufficient data
            model_confidence = adaptive_info.get("model_confidence", {})
            avg_confidence = np.mean(list(model_confidence.values())) if model_confidence else 0.0
            data_hours = adaptive_info.get("data_hours", 0)
            
            if avg_confidence > 0.7 and data_hours >= 12:
                take_profit = tp_extended
            else:
                take_profit = tp_primary
                
        else:  # strong_bear
            # Entry at ML-optimized fibonacci pullback levels
            me_level = fib_levels.get("ME", current_price)
            lr_level = fib_levels.get("LR", current_price)
            
            # Deriv-specific entry logic
            volatility_profile = self.DERIV_SYMBOLS[symbol]["volatility_profile"]
            if volatility_profile == "very_high":
                entry_level = lr_level if abs(current_price - lr_level) < abs(current_price - me_level) else me_level
            else:
                entry_level = me_level if abs(current_price - me_level) < abs(current_price - lr_level) else lr_level
                
            # ML-optimized stop loss
            sl1_level = fib_levels.get("SL1", swing_high * 1.005)
            
            # Deriv-specific SL adjustment
            if adaptive_info.get("manipulation_detected", False):
                manipulation_type = adaptive_info.get("manipulation_type", "")
                if "WICK_HUNTING" in manipulation_type:
                    stop_loss = max(sl1_level * 1.002, swing_high * 1.007)
                else:
                    stop_loss = max(sl1_level, swing_high * 1.002)
            else:
                stop_loss = max(sl1_level, swing_high * 1.002)
                
            # Take profit optimization
            tp_primary = fib_levels.get("TP", entry_level * 0.96)
            tp_extended = fib_levels.get("MINUS_35", entry_level * 0.94)
            
            model_confidence = adaptive_info.get("model_confidence", {})
            avg_confidence = np.mean(list(model_confidence.values())) if model_confidence else 0.0
            data_hours = adaptive_info.get("data_hours", 0)
            
            if avg_confidence > 0.7 and data_hours >= 12:
                take_profit = tp_extended
            else:
                take_profit = tp_primary
                
        # Calculate RRR with Deriv-specific enhancements
        actual_risk = abs(entry_level - stop_loss)
        actual_reward = abs(take_profit - entry_level)
        rrr = actual_reward / actual_risk if actual_risk > 0 else 0
        
        # Apply Deriv-specific RRR requirements
        min_rrr = self.MIN_RRR
        if adaptive_info.get("manipulation_detected", False):
            min_rrr *= 1.2  # Require 20% higher RRR during manipulation
        
        # Adjust RRR requirement based on data quality
        if ml_fib_result["data_quality"] == "insufficient":
            min_rrr *= 1.1  # Require 10% higher RRR with insufficient data
            
        # Only return setup if RRR meets Deriv-adjusted minimum requirement
        if rrr >= min_rrr:
            setup = {
                "direction": direction,
                "entry_price": entry_level,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": rrr,
                "fibonacci_levels": fib_levels,
                "swing_indices": swing_indices,
                "setup_quality": self._assess_deriv_setup_quality(rrr, adaptive_info, symbol),
                "instrument": symbol,
                "ml_info": {
                    "adaptive_levels_used": True,
                    "manipulation_protection": adaptive_info.get("manipulation_detected", False),
                    "manipulation_type": adaptive_info.get("manipulation_type"),
                    "model_confidence": adaptive_info.get("model_confidence", {}),
                    "source": adaptive_info.get("source", "unknown"),
                    "data_hours": adaptive_info.get("data_hours", 0),
                    "data_quality": ml_fib_result["data_quality"],
                    "volatility_profile": self.DERIV_SYMBOLS[symbol]["volatility_profile"]
                }
            }
            
            return setup
            
        return None

    def _assess_deriv_setup_quality(self, rrr: float, adaptive_info: Dict, symbol: str) -> str:
        """Assess setup quality considering Deriv-specific ML factors"""
        base_quality = "GOOD" if rrr >= 4.0 else "POOR"
        
        # Get ML confidence
        model_confidence = adaptive_info.get("model_confidence", {})
        avg_confidence = np.mean(list(model_confidence.values())) if model_confidence else 0.0
        
        # Get data quality
        data_hours = adaptive_info.get("data_hours", 0)
        data_quality = "HIGH" if data_hours >= 12 else "LOW"
        
        # Get volatility profile
        volatility_profile = self.DERIV_SYMBOLS.get(symbol, {}).get("volatility_profile", "unknown")
        
        # Assess quality with Deriv-specific criteria
        if rrr >= 6.0 and avg_confidence > 0.8 and data_quality == "HIGH":
            return f"EXCELLENT_DERIV_{volatility_profile.upper()}"
        elif rrr >= 5.0 and avg_confidence > 0.7 and data_quality == "HIGH":
            return f"VERY_GOOD_DERIV_{volatility_profile.upper()}"
        elif rrr >= 4.0 and avg_confidence > 0.6:
            return f"GOOD_DERIV_{volatility_profile.upper()}"
        elif adaptive_info.get("manipulation_detected", False):
            return f"{base_quality}_DERIV_PROTECTED"
        elif data_quality == "LOW":
            return f"{base_quality}_DERIV_LIMITED_DATA"
        else:
            return f"{base_quality}_DERIV"

    def record_trade_result(self, setup: Dict, outcome: Dict):
        """Record trade result for ML learning"""
        if ML_AVAILABLE and self.ml_levels is not None:
            try:
                self.ml_levels.record_trade_outcome(setup, outcome)
                
                # Save learning data periodically
                if len(self.ml_levels.historical_trades) % 10 == 0:
                    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
                    if not os.path.exists(data_dir):
                        os.makedirs(data_dir)
                    
                    learning_data_path = os.path.join(data_dir, 'ml_learning_data.json')
                    save_result = self.ml_levels.save_learning_data(learning_data_path)
                    
                    if save_result["status"] == "success":
                        self.logger.info(f"ML Learning: Saved trade data ({len(self.ml_levels.historical_trades)} trades)")
                        
            except Exception as e:
                self.logger.error(f"Failed to record trade result for ML: {e}")

    def retrain_ml_models(self, market_data: pd.DataFrame) -> Dict:
        """Retrain ML models with latest market data"""
        if not ML_AVAILABLE or self.ml_levels is None:
            return {"status": "ml_not_available"}
            
        try:
            result = self.ml_levels.retrain_models(market_data)
            
            if result["status"] == "success":
                self.logger.info(f"ML Models retrained successfully with {result['training_samples']} samples")
                self.logger.info(f"Model performance - Train: {result['model_performance']['train_score']:.3f}, "
                               f"Test: {result['model_performance']['test_score']:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"ML model retraining failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_manipulation_report(self) -> Dict:
        """Get recent manipulation detection report"""
        if not self.manipulation_alerts:
            return {"manipulation_events": 0, "recent_alerts": []}
            
        # Count recent events (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_alerts = [alert for alert in self.manipulation_alerts 
                        if alert["timestamp"] > recent_cutoff]
        
        return {
            "manipulation_events": len(recent_alerts),
            "recent_alerts": recent_alerts[-10:],  # Last 10 alerts
            "total_recorded": len(self.manipulation_alerts)
        }
    
    def get_neural_network_analysis(self, symbol: str, market_data: pd.DataFrame) -> Dict:
        """
        Get neural network analysis with mathematical certainty for Deriv instruments
        """
        if not NEURAL_NETWORKS_AVAILABLE or self.neural_network is None:
            return {
                'status': 'neural_networks_not_available',
                'mathematical_certainty': 'UNAVAILABLE',
                'recommendation': 'USE_TRADITIONAL_ANALYSIS'
            }
        
        try:
            # Prepare neural network dataset
            if self.mt5_manager:
                nn_dataset = self.mt5_manager.prepare_neural_network_dataset(symbol, hours=24)
            else:
                # Fallback to provided market data
                nn_dataset = self._prepare_fallback_nn_dataset(market_data, symbol)
            
            if not nn_dataset or not nn_dataset.get('neural_network_ready', False):
                return {
                    'status': 'insufficient_data',
                    'mathematical_certainty': 'INSUFFICIENT_DATA',
                    'recommendation': 'COLLECT_MORE_DATA',
                    'data_points': nn_dataset.get('data_points', 0) if nn_dataset else 0,
                    'minimum_required': self.neural_network.config['sequence_length'] * 2
                }
            
            # Check if models are trained for this symbol
            if symbol not in self.neural_network.models['price_direction']:
                # Train models if we have enough data
                prepared_data = self.neural_network.prepare_neural_network_data(
                    nn_dataset['data'], symbol
                )
                
                if prepared_data and prepared_data['data_points'] >= 100:
                    training_result = self.neural_network.train_models_for_symbol(symbol, prepared_data)
                    if training_result['status'] != 'success':
                        return {
                            'status': 'training_failed',
                            'mathematical_certainty': 'TRAINING_FAILED',
                            'error': training_result.get('error', 'Unknown training error')
                        }
                else:
                    return {
                        'status': 'insufficient_training_data',
                        'mathematical_certainty': 'INSUFFICIENT_TRAINING_DATA',
                        'recommendation': 'WAIT_FOR_MORE_DATA'
                    }
            
            # Get current market features for prediction
            current_features = self._extract_current_features(market_data)
            if current_features is None:
                return {
                    'status': 'feature_extraction_failed',
                    'mathematical_certainty': 'FEATURE_ERROR'
                }
            
            # Generate neural network predictions with mathematical certainty
            predictions = self.neural_network.predict_with_mathematical_certainty(
                symbol, current_features
            )
            
            if predictions['status'] != 'success':
                return predictions
            
            # Enhanced analysis with scientific principles
            enhanced_analysis = self._enhance_predictions_with_science(predictions, market_data)
            
            return {
                'status': 'success',
                'symbol': symbol,
                'neural_network_predictions': predictions,
                'mathematical_certainty': predictions['overall_mathematical_certainty'],
                'certainty_level': predictions['certainty_level'],
                'scientific_analysis': enhanced_analysis,
                'recommendation': self._generate_scientific_recommendation(predictions, enhanced_analysis),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Neural network analysis failed for {symbol}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'mathematical_certainty': 'ERROR'
            }

    def _prepare_fallback_nn_dataset(self, market_data: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Prepare neural network dataset from provided market data"""
        try:
            if len(market_data) < 720:  # Minimum 60 hours of 5-minute data
                return None
                
            # Add basic technical indicators
            data = market_data.copy()
            
            # Calculate returns and volatility
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            data['volatility'] = data['returns'].rolling(20).std()
            
            # OHLC relationships
            data['hl_ratio'] = (data['high'] - data['low']) / data['close']
            data['oc_ratio'] = (data['open'] - data['close']) / data['close']
            data['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['close']
            data['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['close']
            
            # Volume features (if available)
            if 'tick_volume' in data.columns:
                data['volume_ma'] = data['tick_volume'].rolling(20).mean()
                data['volume_ratio'] = data['tick_volume'] / data['volume_ma']
            else:
                data['volume_ratio'] = 1.0
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['bb_middle'] = data['close'].rolling(20).mean()
            bb_std = data['close'].rolling(20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
            data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # Time features
            data['hour'] = data.index.hour if hasattr(data.index, 'hour') else 12
            data['day_of_week'] = data.index.dayofweek if hasattr(data.index, 'dayofweek') else 1
            data['is_london_session'] = ((data['hour'] >= 8) & (data['hour'] <= 16)).astype(int)
            data['is_ny_session'] = ((data['hour'] >= 13) & (data['hour'] <= 21)).astype(int)
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                data[f'ma_{period}'] = data['close'].rolling(period).mean()
                data[f'price_to_ma_{period}'] = data['close'] / data[f'ma_{period}'] - 1
            
            # Clean data
            data = data.dropna()
            
            return {
                'symbol': symbol,
                'data': data,
                'data_points': len(data),
                'neural_network_ready': len(data) >= 720,
                'timeframe': '5M',
                'hours_covered': len(data) / 12  # Assuming 5-minute bars
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing fallback NN dataset for {symbol}: {e}")
            return None

    def _extract_current_features(self, market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract current market features for neural network prediction"""
        try:
            if len(market_data) < 10:
                return None
                
            # Get the latest data point
            latest = market_data.iloc[-1]
            
            # Extract features (same as used in training)
            features = []
            
            # OHLC
            features.extend([latest['open'], latest['high'], latest['low'], latest['close']])
            
            # Volume
            if 'tick_volume' in latest:
                features.append(latest['tick_volume'])
            else:
                features.append(100)  # Default volume
            
            # Technical indicators (calculate from recent data)
            recent_data = market_data.tail(50)  # Last 50 points for calculations
            
            # Returns and volatility
            returns = recent_data['close'].pct_change().iloc[-1]
            log_returns = np.log(recent_data['close'].iloc[-1] / recent_data['close'].iloc[-2])
            volatility = recent_data['close'].pct_change().std()
            
            features.extend([returns, log_returns, volatility])
            
            # OHLC ratios
            hl_ratio = (latest['high'] - latest['low']) / latest['close']
            oc_ratio = (latest['open'] - latest['close']) / latest['close']
            upper_shadow = (latest['high'] - max(latest['open'], latest['close'])) / latest['close']
            lower_shadow = (min(latest['open'], latest['close']) - latest['low']) / latest['close']
            
            features.extend([hl_ratio, oc_ratio, upper_shadow, lower_shadow])
            
            # Volume ratio
            if 'tick_volume' in latest:
                volume_ma = recent_data['tick_volume'].mean()
                volume_ratio = latest['tick_volume'] / volume_ma if volume_ma > 0 else 1.0
            else:
                volume_ratio = 1.0
            features.append(volume_ratio)
            
            # RSI
            delta = recent_data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + gain/loss)) if loss > 0 else 50
            features.append(rsi)
            
            # Bollinger Bands position
            bb_middle = recent_data['close'].rolling(20).mean().iloc[-1]
            bb_std = recent_data['close'].rolling(20).std().iloc[-1]
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_position = (latest['close'] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            features.append(bb_position)
            
            # Moving average ratios
            for period in [5, 10, 20, 50]:
                ma = recent_data['close'].rolling(period).mean().iloc[-1]
                price_to_ma = (latest['close'] / ma - 1) if ma > 0 else 0
                features.append(price_to_ma)
            
            # Time features
            current_hour = datetime.now().hour
            current_dow = datetime.now().weekday()
            is_london = 1 if 8 <= current_hour <= 16 else 0
            is_ny = 1 if 13 <= current_hour <= 21 else 0
            
            features.extend([current_hour, current_dow, is_london, is_ny])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Error extracting current features: {e}")
            return None

    def _enhance_predictions_with_science(self, predictions: Dict, market_data: pd.DataFrame) -> Dict:
        """
        Enhance neural network predictions with scientific principles and mathematical analysis
        """
        try:
            # Extract prediction data
            direction_pred = predictions['predictions']['price_direction']
            volatility_pred = predictions['predictions']['volatility_regime']
            timing_pred = predictions['predictions']['entry_timing']
            manipulation_pred = predictions['predictions']['manipulation_detection']
            
            # Scientific analysis
            scientific_analysis = {
                'probability_theory': self._apply_probability_theory(direction_pred),
                'statistical_significance': self._calculate_statistical_significance(predictions),
                'market_efficiency': self._assess_market_efficiency(market_data, volatility_pred),
                'risk_mathematics': self._calculate_mathematical_risk(predictions, market_data),
                'information_theory': self._apply_information_theory(predictions),
                'fractal_analysis': self._perform_fractal_analysis(market_data),
                'chaos_theory': self._apply_chaos_theory(market_data, volatility_pred)
            }
            
            return scientific_analysis
            
        except Exception as e:
            self.logger.error(f"Error enhancing predictions with science: {e}")
            return {'error': str(e)}

    def _apply_probability_theory(self, direction_pred: Dict) -> Dict:
        """Apply probability theory to direction predictions"""
        probs = direction_pred['probabilities']
        
        # Calculate entropy (measure of uncertainty)
        entropy = -sum(p * np.log2(p + 1e-10) for p in probs.values())
        max_entropy = np.log2(len(probs))
        normalized_entropy = entropy / max_entropy
        
        # Calculate confidence interval
        max_prob = max(probs.values())
        confidence_95 = max_prob * 1.96  # Approximate 95% confidence interval
        
        return {
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'uncertainty_level': 'HIGH' if normalized_entropy > 0.8 else 'MEDIUM' if normalized_entropy > 0.5 else 'LOW',
            'confidence_interval_95': min(confidence_95, 1.0),
            'statistical_edge': max_prob - (1/3)  # Edge over random chance (1/3 for 3 outcomes)
        }

    def _calculate_statistical_significance(self, predictions: Dict) -> Dict:
        """Calculate statistical significance of predictions"""
        certainty = predictions['overall_mathematical_certainty']
        
        # Calculate p-value approximation
        p_value = 1 - certainty
        
        # Determine significance level
        if p_value < 0.01:
            significance = 'HIGHLY_SIGNIFICANT'
        elif p_value < 0.05:
            significance = 'SIGNIFICANT'
        elif p_value < 0.1:
            significance = 'MARGINALLY_SIGNIFICANT'
        else:
            significance = 'NOT_SIGNIFICANT'
        
        return {
            'p_value_approximation': p_value,
            'significance_level': significance,
            'confidence_level': certainty,
            'degrees_of_freedom': 3,  # Approximate for our model
            'statistical_power': certainty * 0.8  # Conservative estimate
        }

    def _assess_market_efficiency(self, market_data: pd.DataFrame, volatility_pred: Dict) -> Dict:
        """Assess market efficiency based on volatility and price patterns"""
        try:
            if len(market_data) < 100:
                return {'efficiency': 'UNKNOWN', 'reason': 'insufficient_data'}
            
            # Calculate autocorrelation of returns (efficient market should have low autocorrelation)
            returns = market_data['close'].pct_change().dropna()
            if len(returns) < 10:
                return {'efficiency': 'UNKNOWN', 'reason': 'insufficient_returns'}
            
            # Lag-1 autocorrelation
            autocorr_1 = returns.autocorr(lag=1)
            
            # Volatility clustering (GARCH effects)
            squared_returns = returns ** 2
            vol_clustering = squared_returns.autocorr(lag=1)
            
            # Hurst exponent approximation
            returns_array = returns.values
            lags = range(2, min(20, len(returns_array)//4))
            variability = [np.var(np.diff(returns_array, n=lag)) for lag in lags]
            if len(variability) > 1:
                hurst = np.polyfit(np.log(lags), np.log(variability), 1)[0] / 2
            else:
                hurst = 0.5
            
            # Efficiency assessment
            if abs(autocorr_1) < 0.05 and hurst < 0.55:
                efficiency = 'HIGHLY_EFFICIENT'
            elif abs(autocorr_1) < 0.1 and hurst < 0.6:
                efficiency = 'EFFICIENT'
            elif abs(autocorr_1) < 0.2 and hurst < 0.7:
                efficiency = 'SEMI_EFFICIENT'
            else:
                efficiency = 'INEFFICIENT'
            
            return {
                'efficiency': efficiency,
                'autocorrelation_lag1': autocorr_1,
                'volatility_clustering': vol_clustering,
                'hurst_exponent': hurst,
                'predicted_volatility_regime': volatility_pred['prediction'],
                'market_predictability': 'HIGH' if efficiency == 'INEFFICIENT' else 'LOW'
            }
            
        except Exception as e:
            return {'efficiency': 'ERROR', 'error': str(e)}

    def _calculate_mathematical_risk(self, predictions: Dict, market_data: pd.DataFrame) -> Dict:
        """Calculate mathematical risk metrics"""
        try:
            # Extract volatility prediction
            vol_pred = predictions['predictions']['volatility_regime']
            current_price = market_data['close'].iloc[-1]
            
            # Historical volatility
            returns = market_data['close'].pct_change().dropna()
            if len(returns) < 10:
                return {'risk': 'UNKNOWN', 'reason': 'insufficient_data'}
            
            hist_vol = returns.std() * np.sqrt(252 * 24 * 12)  # Annualized volatility
            
            # Value at Risk (VaR) calculation
            var_95 = np.percentile(returns, 5) * current_price
            var_99 = np.percentile(returns, 1) * current_price
            
            # Expected Shortfall (Conditional VaR)
            cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * current_price
            
            # Risk level based on volatility prediction
            vol_multipliers = {'LOW': 0.8, 'MEDIUM': 1.0, 'HIGH': 1.5, 'EXTREME': 2.0}
            risk_multiplier = vol_multipliers.get(vol_pred['prediction'], 1.0)
            
            adjusted_var_95 = var_95 * risk_multiplier
            adjusted_var_99 = var_99 * risk_multiplier
            
            return {
                'historical_volatility_annualized': hist_vol,
                'value_at_risk_95': var_95,
                'value_at_risk_99': var_99,
                'conditional_var_95': cvar_95,
                'predicted_volatility_regime': vol_pred['prediction'],
                'risk_adjustment_multiplier': risk_multiplier,
                'adjusted_var_95': adjusted_var_95,
                'adjusted_var_99': adjusted_var_99,
                'risk_level': 'HIGH' if risk_multiplier > 1.2 else 'MEDIUM' if risk_multiplier > 0.9 else 'LOW'
            }
            
        except Exception as e:
            return {'risk': 'ERROR', 'error': str(e)}

    def _apply_information_theory(self, predictions: Dict) -> Dict:
        """Apply information theory to assess prediction quality"""
        try:
            # Calculate information content (surprisal) of predictions
            direction_probs = list(predictions['predictions']['price_direction']['probabilities'].values())
            vol_probs = list(predictions['predictions']['volatility_regime']['probabilities'].values())
            
            # Information content (bits)
            direction_info = -np.log2(max(direction_probs) + 1e-10)
            vol_info = -np.log2(max(vol_probs) + 1e-10)
            
            # Total information
            total_info = direction_info + vol_info
            
            # Information gain over random
            random_info_direction = -np.log2(1/3)  # 3 possible directions
            random_info_vol = -np.log2(1/4)  # 4 volatility regimes
            
            info_gain = (random_info_direction - direction_info) + (random_info_vol - vol_info)
            
            return {
                'direction_information_content': direction_info,
                'volatility_information_content': vol_info,
                'total_information_content': total_info,
                'information_gain_over_random': info_gain,
                'prediction_quality': 'HIGH' if info_gain > 1.0 else 'MEDIUM' if info_gain > 0.5 else 'LOW'
            }
            
        except Exception as e:
            return {'error': str(e)}

    def _perform_fractal_analysis(self, market_data: pd.DataFrame) -> Dict:
        """Perform fractal analysis on market data"""
        try:
            if len(market_data) < 100:
                return {'fractal_dimension': 'UNKNOWN', 'reason': 'insufficient_data'}
            
            prices = market_data['close'].values
            
            # Calculate fractal dimension using box-counting method (simplified)
            scales = np.logspace(0, 2, 10).astype(int)
            counts = []
            
            for scale in scales:
                if scale >= len(prices):
                    continue
                # Simplified box counting
                normalized_prices = (prices - prices.min()) / (prices.max() - prices.min())
                boxes = int(1.0 / scale) if scale > 0 else 1
                count = len(np.unique(np.floor(normalized_prices * boxes)))
                counts.append(count)
            
            if len(counts) > 2:
                # Fit power law
                log_scales = np.log(scales[:len(counts)])
                log_counts = np.log(counts)
                fractal_dim = -np.polyfit(log_scales, log_counts, 1)[0]
                
                # Interpret fractal dimension
                if fractal_dim < 1.3:
                    complexity = 'LOW'
                elif fractal_dim < 1.6:
                    complexity = 'MEDIUM'
                else:
                    complexity = 'HIGH'
                    
                return {
                    'fractal_dimension': fractal_dim,
                    'market_complexity': complexity,
                    'predictability': 'HIGH' if complexity == 'LOW' else 'MEDIUM' if complexity == 'MEDIUM' else 'LOW'
                }
            else:
                return {'fractal_dimension': 'UNKNOWN', 'reason': 'calculation_failed'}
                
        except Exception as e:
            return {'fractal_dimension': 'ERROR', 'error': str(e)}

    def _apply_chaos_theory(self, market_data: pd.DataFrame, volatility_pred: Dict) -> Dict:
        """Apply chaos theory principles to market analysis"""
        try:
            if len(market_data) < 50:
                return {'chaos_analysis': 'UNKNOWN', 'reason': 'insufficient_data'}
            
            returns = market_data['close'].pct_change().dropna().values
            
            if len(returns) < 10:
                return {'chaos_analysis': 'UNKNOWN', 'reason': 'insufficient_returns'}
            
            # Lyapunov exponent approximation
            # This is a simplified version - full calculation would require more sophisticated methods
            
            # Calculate correlation dimension (simplified)
            embedded_dim = min(5, len(returns) // 10)
            if embedded_dim < 2:
                return {'chaos_analysis': 'UNKNOWN', 'reason': 'insufficient_data_for_embedding'}
            
            # Phase space reconstruction (simplified)
            embedded = np.array([returns[i:i+embedded_dim] for i in range(len(returns)-embedded_dim+1)])
            
            # Calculate correlation sum for different radii
            radii = np.logspace(-4, -1, 10)
            correlations = []
            
            for r in radii:
                distances = np.sqrt(np.sum((embedded[:, None] - embedded[None, :]) ** 2, axis=2))
                correlation = np.mean(distances < r)
                correlations.append(correlation + 1e-10)  # Avoid log(0)
            
            # Estimate correlation dimension
            log_r = np.log(radii)
            log_c = np.log(correlations)
            
            # Find linear region (middle part of the curve)
            start_idx = len(log_r) // 4
            end_idx = 3 * len(log_r) // 4
            
            if end_idx > start_idx:
                correlation_dim = np.polyfit(log_r[start_idx:end_idx], log_c[start_idx:end_idx], 1)[0]
            else:
                correlation_dim = 2.0  # Default
            
            # Interpret results
            if correlation_dim < 2.5:
                system_type = 'DETERMINISTIC_CHAOS'
                predictability = 'MEDIUM'
            elif correlation_dim < 5.0:
                system_type = 'STOCHASTIC_WITH_STRUCTURE'
                predictability = 'LOW'
            else:
                system_type = 'RANDOM'
                predictability = 'VERY_LOW'
            
            # Assess attractor stability
            vol_regime = volatility_pred['prediction']
            if vol_regime in ['LOW', 'MEDIUM']:
                attractor_stability = 'STABLE'
            else:
                attractor_stability = 'UNSTABLE'
            
            return {
                'correlation_dimension': correlation_dim,
                'system_type': system_type,
                'predictability': predictability,
                'attractor_stability': attractor_stability,
                'volatility_regime': vol_regime,
                'chaos_level': 'HIGH' if correlation_dim < 3 else 'MEDIUM' if correlation_dim < 6 else 'LOW'
            }
            
        except Exception as e:
            return {'chaos_analysis': 'ERROR', 'error': str(e)}

    def _generate_scientific_recommendation(self, predictions: Dict, scientific_analysis: Dict) -> Dict:
        """
        Generate trading recommendation based on scientific analysis and mathematical certainty
        """
        try:
            # Extract key metrics
            overall_certainty = predictions['overall_mathematical_certainty']
            direction = predictions['predictions']['price_direction']['prediction']
            vol_regime = predictions['predictions']['volatility_regime']['prediction']
            timing_score = predictions['predictions']['entry_timing']['score']
            manipulation_detected = predictions['predictions']['manipulation_detection']['manipulation_detected']
            
            # Scientific factors
            prob_analysis = scientific_analysis.get('probability_theory', {})
            stat_significance = scientific_analysis.get('statistical_significance', {})
            market_efficiency = scientific_analysis.get('market_efficiency', {})
            risk_analysis = scientific_analysis.get('risk_mathematics', {})
            
            # Calculate composite recommendation score
            base_score = overall_certainty
            
            # Adjust for statistical significance
            if stat_significance.get('significance_level') == 'HIGHLY_SIGNIFICANT':
                base_score *= 1.2
            elif stat_significance.get('significance_level') == 'SIGNIFICANT':
                base_score *= 1.1
            elif stat_significance.get('significance_level') == 'NOT_SIGNIFICANT':
                base_score *= 0.7
            
            # Adjust for market efficiency
            efficiency = market_efficiency.get('efficiency', 'UNKNOWN')
            if efficiency in ['INEFFICIENT', 'SEMI_EFFICIENT']:
                base_score *= 1.1  # More predictable market
            elif efficiency == 'HIGHLY_EFFICIENT':
                base_score *= 0.9  # Less predictable market
            
            # Adjust for risk level
            risk_level = risk_analysis.get('risk_level', 'MEDIUM')
            if risk_level == 'HIGH':
                base_score *= 0.8  # Reduce confidence in high-risk environment
            elif risk_level == 'LOW':
                base_score *= 1.1  # Increase confidence in low-risk environment
            
            # Manipulation penalty
            if manipulation_detected:
                base_score *= 0.6
            
            # Volatility adjustment
            vol_adjustments = {'LOW': 1.1, 'MEDIUM': 1.0, 'HIGH': 0.9, 'EXTREME': 0.7}
            base_score *= vol_adjustments.get(vol_regime, 1.0)
            
            # Generate final recommendation
            final_score = min(base_score, 1.0)
            
            if final_score >= 0.85 and timing_score > 0.7:
                if direction == 'UP':
                    recommendation = 'STRONG_BUY'
                elif direction == 'DOWN':
                    recommendation = 'STRONG_SELL'
                else:
                    recommendation = 'HOLD'
            elif final_score >= 0.7 and timing_score > 0.6:
                if direction == 'UP':
                    recommendation = 'BUY'
                elif direction == 'DOWN':
                    recommendation = 'SELL'
                else:
                    recommendation = 'HOLD'
            elif final_score >= 0.6:
                if direction in ['UP', 'DOWN']:
                    recommendation = 'WEAK_SIGNAL'
                else:
                    recommendation = 'HOLD'
            else:
                recommendation = 'NO_TRADE'
            
            return {
                'recommendation': recommendation,
                'confidence_score': final_score,
                'mathematical_certainty': predictions['certainty_level'],
                'direction_prediction': direction,
                'volatility_regime': vol_regime,
                'timing_score': timing_score,
                'manipulation_detected': manipulation_detected,
                'risk_level': risk_level,
                'statistical_significance': stat_significance.get('significance_level', 'UNKNOWN'),
                'market_efficiency': efficiency,
                'scientific_factors': {
                    'probability_entropy': prob_analysis.get('normalized_entropy', 0),
                    'statistical_edge': prob_analysis.get('statistical_edge', 0),
                    'information_gain': scientific_analysis.get('information_theory', {}).get('information_gain_over_random', 0),
                    'fractal_complexity': scientific_analysis.get('fractal_analysis', {}).get('market_complexity', 'UNKNOWN'),
                    'chaos_predictability': scientific_analysis.get('chaos_theory', {}).get('predictability', 'UNKNOWN')
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating scientific recommendation: {e}")
            return {
                'recommendation': 'ERROR',
                'error': str(e),
                'confidence_score': 0.0
            }
