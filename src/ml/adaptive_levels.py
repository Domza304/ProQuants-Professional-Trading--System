"""
ProQuants ML-Enhanced Adaptive Trading Levels for Deriv Synthetic Indices
Dynamic Fibonacci and Stop Loss Optimization using Machine Learning
Specialized for Volatility 75, Volatility 25, and Volatility 75 (1s) indices
Each instrument learns independently due to unique behavioral patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import os
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AdaptiveLevelsML:
    def __init__(self):
        self.base_levels = {
            # Standard Goloji Bhudasi Reference Levels for Deriv Synthetics
            "BOS_TEST": 1.15,
            "SL2": 1.05,
            "ONE_HUNDRED": 1.00,
            "LR": 0.88,
            "SL1": 0.76,
            "ME": 0.685,  # Main Entry Point
            "ZERO": 0.00,
            "TP": -0.15,
            "MINUS_35": -0.35,
            "MINUS_62": -0.62
        }
        
        # Deriv Synthetic Indices Configuration
        self.DERIV_INSTRUMENTS = {
            "Volatility 75 Index": {
                "tick_size": 0.00001,
                "typical_spread": 2,  # pips
                "volatility_profile": "high",
                "manipulation_patterns": ["spread_widening", "wick_hunting"],
                "min_data_hours": 12
            },
            "Volatility 25 Index": {
                "tick_size": 0.00001,
                "typical_spread": 1.5,  # pips
                "volatility_profile": "medium",
                "manipulation_patterns": ["stop_hunting", "gap_creation"],
                "min_data_hours": 12
            },
            "Volatility 75 (1s) Index": {
                "tick_size": 0.00001,
                "typical_spread": 3,  # pips
                "volatility_profile": "very_high",
                "manipulation_patterns": ["rapid_reversals", "volume_spikes"],
                "min_data_hours": 12
            }
        }
        
        # Independent ML Models for each Deriv instrument
        self.instrument_models = {}
        self.instrument_scalers = {}
        self.manipulation_detectors = {}
        
        # Initialize models for each Deriv synthetic
        for instrument in self.DERIV_INSTRUMENTS.keys():
            self.instrument_models[instrument] = {
                "entry_optimizer": RandomForestRegressor(n_estimators=150, random_state=42),
                "sl_optimizer": RandomForestRegressor(n_estimators=150, random_state=42),
                "tp_optimizer": RandomForestRegressor(n_estimators=150, random_state=42)
            }
            self.instrument_scalers[instrument] = StandardScaler()
            self.manipulation_detectors[instrument] = IsolationForest(
                contamination=0.15,  # Higher sensitivity for Deriv synthetics
                random_state=42
            )
        
        # Learning data storage (separate for each instrument)
        self.instrument_trades = {instrument: [] for instrument in self.DERIV_INSTRUMENTS.keys()}
        self.manipulation_events = {instrument: [] for instrument in self.DERIV_INSTRUMENTS.keys()}
        
        # Adaptive levels (updated by ML, per instrument)
        self.adaptive_levels = {instrument: self.base_levels.copy() for instrument in self.DERIV_INSTRUMENTS.keys()}
        self.confidence_scores = {instrument: {} for instrument in self.DERIV_INSTRUMENTS.keys()}
        
        # Model training status (per instrument)
        self.models_trained = {instrument: False for instrument in self.DERIV_INSTRUMENTS.keys()}
        self.last_training = {instrument: None for instrument in self.DERIV_INSTRUMENTS.keys()}
        
        # Minimum data requirements for Deriv synthetics
        self.MIN_HOURS_DATA = 12  # 12 hours minimum
        self.MIN_TRADES_FOR_TRAINING = 30  # Reduced for faster learning on synthetics
        
    def detect_deriv_manipulation(self, instrument: str, market_data: pd.DataFrame) -> Dict:
        """
        Detect manipulation patterns specific to Deriv synthetic indices
        Each instrument has unique manipulation characteristics
        """
        if instrument not in self.DERIV_INSTRUMENTS:
            return {"manipulation_detected": False, "confidence": 0.0, "error": "Unknown instrument"}
            
        try:
            # Check minimum data requirement (12 hours)
            if len(market_data) < self.MIN_HOURS_DATA * 12:  # Assuming 5-min candles
                return {
                    "manipulation_detected": False, 
                    "confidence": 0.0,
                    "insufficient_data": True,
                    "required_hours": self.MIN_HOURS_DATA,
                    "current_hours": len(market_data) / 12
                }
            
            # Extract Deriv-specific manipulation indicators
            features = self._extract_deriv_manipulation_features(instrument, market_data)
            
            if len(features) < 10:
                return {"manipulation_detected": False, "confidence": 0.0}
                
            # Use instrument-specific manipulation detector
            detector = self.manipulation_detectors[instrument]
            scaler = self.instrument_scalers[instrument]
            
            # Fit scaler if not already fitted
            if not hasattr(scaler, 'mean_'):
                # Need more data to fit scaler
                dummy_data = np.random.normal(0, 1, (50, len(features)))
                scaler.fit(dummy_data)
            
            features_scaled = scaler.transform([features])
            anomaly_score = detector.decision_function(features_scaled)[0]
            is_manipulation = detector.predict(features_scaled)[0] == -1
            
            # Calculate confidence (normalized anomaly score)
            confidence = min(1.0, abs(anomaly_score) / 2.0)
            
            # Classify manipulation type specific to Deriv synthetics
            manipulation_type = self._classify_deriv_manipulation_type(instrument, features) if is_manipulation else None
            
            return {
                "instrument": instrument,
                "manipulation_detected": is_manipulation,
                "confidence": confidence,
                "anomaly_score": anomaly_score,
                "manipulation_type": manipulation_type,
                "timestamp": datetime.now(),
                "data_hours": len(market_data) / 12
            }
            
        except Exception as e:
            return {"manipulation_detected": False, "confidence": 0.0, "error": str(e)}
    
    def _extract_deriv_manipulation_features(self, instrument: str, data: pd.DataFrame) -> List[float]:
        """Extract manipulation features specific to Deriv synthetic indices"""
        if len(data) < 24:  # Need at least 2 hours of 5-min data
            return []
            
        features = []
        instrument_config = self.DERIV_INSTRUMENTS[instrument]
        
        # 1. Spread Analysis (critical for Deriv synthetics)
        if 'bid' in data.columns and 'ask' in data.columns:
            spreads = data['ask'] - data['bid']
            typical_spread = instrument_config["typical_spread"] * instrument_config["tick_size"] * 10
            
            avg_spread = spreads.mean()
            spread_volatility = spreads.std()
            spread_spikes = (spreads > typical_spread * 2).sum()
            spread_ratio = avg_spread / typical_spread if typical_spread > 0 else 1
            
            features.extend([avg_spread, spread_volatility, spread_spikes, spread_ratio])
        else:
            # Estimate spread from OHLC
            estimated_spreads = (data['high'] - data['low']) * 0.1  # Rough estimate
            features.extend([estimated_spreads.mean(), estimated_spreads.std(), 0, 1.0])
            
        # 2. Volatility Index Specific Patterns
        price_changes = data['close'].pct_change().dropna()
        
        # Extreme volatility spikes (common in V75, V25)
        volatility_threshold = {
            "Volatility 75 Index": 0.02,      # 2% moves
            "Volatility 25 Index": 0.015,     # 1.5% moves  
            "Volatility 75 (1s) Index": 0.025 # 2.5% moves
        }
        
        threshold = volatility_threshold.get(instrument, 0.02)
        extreme_moves = (abs(price_changes) > threshold).sum()
        volatility_clustering = self._detect_volatility_clustering(price_changes)
        
        features.extend([extreme_moves, volatility_clustering])
        
        # 3. Synthetic Index Manipulation Patterns
        
        # Wick hunting (stop loss hunting via large wicks)
        upper_wicks = data['high'] - np.maximum(data['open'], data['close'])
        lower_wicks = np.minimum(data['open'], data['close']) - data['low']
        body_sizes = abs(data['close'] - data['open'])
        
        # Large wicks relative to body (manipulation indicator)
        large_upper_wicks = (upper_wicks > body_sizes * 2).sum()
        large_lower_wicks = (lower_wicks > body_sizes * 2).sum()
        
        features.extend([large_upper_wicks, large_lower_wicks])
        
        # 4. Volume/Tick Analysis for Deriv synthetics
        if 'tick_volume' in data.columns:
            tick_volume = data['tick_volume']
            avg_volume = tick_volume.mean()
            volume_spikes = (tick_volume > avg_volume * 3).sum()
            volume_drought = (tick_volume < avg_volume * 0.3).sum()
            
            features.extend([volume_spikes, volume_drought])
        else:
            features.extend([0, 0])
            
        # 5. Time-based Deriv Patterns
        if 'timestamp' in data.columns or hasattr(data.index, 'hour'):
            try:
                if 'timestamp' in data.columns:
                    timestamps = pd.to_datetime(data['timestamp'])
                else:
                    timestamps = data.index
                    
                hours = timestamps.hour
                
                # Deriv manipulation often occurs during:
                # - London/NY overlap (12:00-16:00 GMT)
                # - Asian session (00:00-08:00 GMT)
                overlap_hours = ((hours >= 12) & (hours <= 16)).sum()
                asian_hours = ((hours >= 0) & (hours <= 8)).sum()
                
                features.extend([overlap_hours, asian_hours])
            except:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
            
        # 6. Deriv-specific Gap Analysis
        # Gaps between candles (rare but significant when they occur)
        gaps = abs(data['open'] - data['close'].shift(1)).fillna(0)
        significant_gaps = (gaps > data['close'] * 0.001).sum()  # 0.1% gaps
        
        features.append(significant_gaps)
        
        # 7. Price Level Rejection Patterns
        # Deriv often shows rejection at psychological levels
        price_levels = data['close']
        round_number_tests = 0
        
        for price in price_levels.tail(20):  # Last 20 candles
            # Check if price tested round numbers (e.g., 500.00000, 501.00000)
            rounded = round(price, 2)
            if abs(price - rounded) < 0.00005:  # Very close to round number
                round_number_tests += 1
                
        features.append(round_number_tests)
        
        return features
    
    def _detect_volatility_clustering(self, price_changes: pd.Series) -> float:
        """Detect volatility clustering patterns in Deriv synthetics"""
        if len(price_changes) < 10:
            return 0.0
            
        # Calculate rolling volatility
        rolling_vol = price_changes.rolling(window=5).std()
        
        # Detect periods of high/low volatility clustering
        high_vol_periods = (rolling_vol > rolling_vol.mean() + rolling_vol.std()).sum()
        total_periods = len(rolling_vol.dropna())
        
        return high_vol_periods / total_periods if total_periods > 0 else 0.0
    
    def _classify_deriv_manipulation_type(self, instrument: str, features: List[float]) -> str:
        """Classify manipulation type specific to Deriv synthetic indices"""
        if len(features) < 12:
            return "UNKNOWN"
            
        # Extract key features
        avg_spread, spread_volatility, spread_spikes, spread_ratio = features[0:4]
        extreme_moves, volatility_clustering = features[4:6]
        large_upper_wicks, large_lower_wicks = features[6:8]
        volume_spikes, volume_drought = features[8:10] if len(features) > 9 else [0, 0]
        
        # Deriv-specific manipulation patterns
        if spread_spikes > 5 or spread_ratio > 2.0:
            return "DERIV_SPREAD_MANIPULATION"
        elif large_upper_wicks > 3 or large_lower_wicks > 3:
            return "DERIV_WICK_HUNTING"
        elif extreme_moves > 5 and volatility_clustering > 0.3:
            return "DERIV_VOLATILITY_SPIKE"
        elif volume_spikes > 5:
            return "DERIV_VOLUME_MANIPULATION"
        elif volume_drought > 10:
            return "DERIV_LIQUIDITY_DROUGHT"
        else:
            return "DERIV_GENERAL_MANIPULATION"
    
    def _extract_manipulation_features(self, data: pd.DataFrame) -> List[float]:
        """Extract features that indicate potential broker manipulation"""
        if len(data) < 10:
            return []
            
        # Calculate manipulation indicators
        features = []
        
        # 1. Spread Analysis
        if 'bid' in data.columns and 'ask' in data.columns:
            spreads = data['ask'] - data['bid']
            avg_spread = spreads.mean()
            spread_volatility = spreads.std()
            max_spread = spreads.max()
            spread_spikes = (spreads > avg_spread + 2 * spread_volatility).sum()
            
            features.extend([avg_spread, spread_volatility, max_spread, spread_spikes])
        else:
            features.extend([0, 0, 0, 0])
            
        # 2. Price Movement Anomalies
        price_changes = data['close'].pct_change().dropna()
        extreme_moves = (abs(price_changes) > 3 * price_changes.std()).sum()
        gap_moves = abs(data['open'] - data['close'].shift(1)).fillna(0)
        avg_gap = gap_moves.mean()
        
        features.extend([extreme_moves, avg_gap])
        
        # 3. Volume/Tick Analysis (if available)
        if 'tick_volume' in data.columns:
            volume_spikes = (data['tick_volume'] > data['tick_volume'].mean() + 2 * data['tick_volume'].std()).sum()
            features.append(volume_spikes)
        else:
            features.append(0)
            
        # 4. Wick Analysis (potential stop hunting)
        upper_wicks = data['high'] - np.maximum(data['open'], data['close'])
        lower_wicks = np.minimum(data['open'], data['close']) - data['low']
        avg_upper_wick = upper_wicks.mean()
        avg_lower_wick = lower_wicks.mean()
        large_wicks = ((upper_wicks > 2 * avg_upper_wick) | (lower_wicks > 2 * avg_lower_wick)).sum()
        
        features.extend([avg_upper_wick, avg_lower_wick, large_wicks])
        
        # 5. Time-based patterns (manipulation often occurs at specific times)
        if 'timestamp' in data.columns or data.index.name == 'timestamp':
            try:
                timestamps = pd.to_datetime(data.index if data.index.name == 'timestamp' else data['timestamp'])
                hours = timestamps.hour
                news_hours = ((hours >= 8) & (hours <= 10)).sum()  # Common news release times
                session_overlaps = ((hours >= 7) & (hours <= 9)).sum()  # Session overlaps
                features.extend([news_hours, session_overlaps])
            except:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
            
        return features
    
    def _classify_manipulation_type(self, features: List[float]) -> str:
        """Classify the type of manipulation detected"""
        if len(features) < 9:
            return "UNKNOWN"
            
        avg_spread, spread_volatility, max_spread, spread_spikes = features[0:4]
        extreme_moves, avg_gap = features[4:6]
        volume_spikes = features[6]
        avg_upper_wick, avg_lower_wick, large_wicks = features[7:10]
        
        # Classification logic
        if spread_spikes > 5 or max_spread > avg_spread * 3:
            return "SPREAD_MANIPULATION"
        elif large_wicks > 3:
            return "STOP_HUNTING"
        elif extreme_moves > 2:
            return "PRICE_SPIKING"
        elif avg_gap > avg_spread * 2:
            return "GAP_MANIPULATION"
        else:
            return "GENERAL_MANIPULATION"
    
    def optimize_fibonacci_levels(self, historical_data: pd.DataFrame, trade_outcomes: List[Dict]) -> Dict:
        """
        Use ML to optimize fibonacci levels based on recent performance
        """
        try:
            if len(trade_outcomes) < self.min_samples_for_training:
                return {"status": "insufficient_data", "levels": self.base_levels}
                
            # Prepare training data
            X, y = self._prepare_fibonacci_training_data(historical_data, trade_outcomes)
            
            if len(X) < 10:
                return {"status": "insufficient_features", "levels": self.base_levels}
                
            # Train the model
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.entry_optimizer.fit(X_train, y_train)
            
            # Calculate model performance
            train_score = self.entry_optimizer.score(X_train, y_train)
            test_score = self.entry_optimizer.score(X_test, y_test)
            
            # Generate optimized levels
            optimized_levels = self._generate_optimized_levels(historical_data)
            
            # Update adaptive levels with confidence weighting
            for level_name, base_value in self.base_levels.items():
                if level_name in optimized_levels:
                    optimal_value = optimized_levels[level_name]
                    confidence = min(test_score, 0.9)  # Cap confidence at 90%
                    
                    # Weighted average: base_value * (1-confidence) + optimal_value * confidence
                    self.adaptive_levels[level_name] = base_value * (1 - confidence) + optimal_value * confidence
                    self.confidence_scores[level_name] = confidence
                    
            self.models_trained = True
            self.last_training = datetime.now()
            
            return {
                "status": "success",
                "levels": self.adaptive_levels,
                "confidence_scores": self.confidence_scores,
                "model_performance": {"train_score": train_score, "test_score": test_score},
                "training_samples": len(trade_outcomes)
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e), "levels": self.base_levels}
    
    def _prepare_fibonacci_training_data(self, market_data: pd.DataFrame, trade_outcomes: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for fibonacci level optimization"""
        X = []  # Features
        y = []  # Success rate at each level
        
        for trade in trade_outcomes:
            try:
                # Extract features for this trade
                features = self._extract_trade_features(market_data, trade)
                if features and 'success_rate' in trade:
                    X.append(features)
                    y.append(trade['success_rate'])
            except Exception:
                continue
                
        return np.array(X), np.array(y)
    
    def _extract_trade_features(self, market_data: pd.DataFrame, trade: Dict) -> Optional[List[float]]:
        """Extract relevant features for a trade"""
        try:
            features = []
            
            # Market volatility
            price_changes = market_data['close'].pct_change().dropna()
            volatility = price_changes.std()
            features.append(volatility)
            
            # Trend strength
            ma_short = market_data['close'].rolling(5).mean()
            ma_long = market_data['close'].rolling(20).mean()
            trend_strength = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1]
            features.append(trend_strength)
            
            # Range analysis
            recent_high = market_data['high'].tail(10).max()
            recent_low = market_data['low'].tail(10).min()
            range_size = (recent_high - recent_low) / market_data['close'].iloc[-1]
            features.append(range_size)
            
            # Time of day effect
            if 'timestamp' in trade:
                hour = pd.to_datetime(trade['timestamp']).hour
                features.append(hour)
            else:
                features.append(12)  # Default noon
                
            # Trade-specific features
            if 'entry_level' in trade:
                features.append(trade['entry_level'])
            if 'stop_loss_level' in trade:
                features.append(trade['stop_loss_level'])
                
            return features
            
        except Exception:
            return None
    
    def _generate_optimized_levels(self, market_data: pd.DataFrame) -> Dict:
        """Generate optimized fibonacci levels using trained model"""
        optimized = {}
        
        try:
            # Current market features
            current_features = self._extract_current_market_features(market_data)
            
            if current_features and self.models_trained:
                # Predict optimal adjustments for each level
                for level_name, base_value in self.base_levels.items():
                    # Add level-specific features
                    level_features = current_features + [base_value]
                    
                    try:
                        prediction = self.entry_optimizer.predict([level_features])[0]
                        # Apply bounds to prevent extreme adjustments
                        adjustment = np.clip(prediction, -0.1, 0.1)  # Max 10% adjustment
                        optimized[level_name] = base_value + adjustment
                    except Exception:
                        optimized[level_name] = base_value
                        
        except Exception:
            pass
            
        return optimized
    
    def _extract_current_market_features(self, market_data: pd.DataFrame) -> Optional[List[float]]:
        """Extract current market features for prediction"""
        try:
            features = []
            
            # Current volatility
            price_changes = market_data['close'].pct_change().dropna()
            current_volatility = price_changes.tail(20).std()
            features.append(current_volatility)
            
            # Current trend
            ma_short = market_data['close'].rolling(5).mean().iloc[-1]
            ma_long = market_data['close'].rolling(20).mean().iloc[-1]
            trend_strength = (ma_short - ma_long) / ma_long
            features.append(trend_strength)
            
            # Current range
            recent_high = market_data['high'].tail(10).max()
            recent_low = market_data['low'].tail(10).min()
            range_size = (recent_high - recent_low) / market_data['close'].iloc[-1]
            features.append(range_size)
            
            # Time features
            current_hour = datetime.now().hour
            features.append(current_hour)
            
            return features
            
        except Exception:
            return None
    
    def record_trade_outcome(self, trade_setup: Dict, outcome: Dict):
        """Record trade outcome for ML learning"""
        trade_record = {
            "timestamp": datetime.now(),
            "setup": trade_setup,
            "outcome": outcome,
            "success_rate": 1.0 if outcome.get("profitable", False) else 0.0,
            "entry_level": trade_setup.get("entry_price", 0),
            "stop_loss_level": trade_setup.get("stop_loss", 0),
            "take_profit_level": trade_setup.get("take_profit", 0),
            "actual_exit": outcome.get("exit_price", 0),
            "profit_loss": outcome.get("profit_loss", 0)
        }
        
        self.historical_trades.append(trade_record)
        
        # Keep only recent trades (last 1000)
        if len(self.historical_trades) > 1000:
            self.historical_trades = self.historical_trades[-1000:]
    
    def get_adaptive_levels(self, market_data: pd.DataFrame) -> Dict:
        """Get current adaptive levels with manipulation detection"""
        result = {
            "base_levels": self.base_levels,
            "adaptive_levels": self.adaptive_levels,
            "confidence_scores": self.confidence_scores,
            "manipulation_check": self.detect_broker_manipulation(market_data),
            "last_training": self.last_training,
            "models_trained": self.models_trained,
            "trade_count": len(self.historical_trades)
        }
        
        # If manipulation detected, apply safety adjustments
        if result["manipulation_check"]["manipulation_detected"]:
            result["safety_adjusted_levels"] = self._apply_manipulation_adjustments(
                self.adaptive_levels, 
                result["manipulation_check"]
            )
        else:
            result["safety_adjusted_levels"] = self.adaptive_levels
            
        return result
    
    def _apply_manipulation_adjustments(self, levels: Dict, manipulation_info: Dict) -> Dict:
        """Apply safety adjustments when manipulation is detected"""
        adjusted = levels.copy()
        manipulation_type = manipulation_info.get("manipulation_type", "GENERAL_MANIPULATION")
        confidence = manipulation_info.get("confidence", 0.5)
        
        # Adjust levels based on manipulation type
        if manipulation_type == "STOP_HUNTING":
            # Move stop losses further away
            adjusted["SL1"] = levels["SL1"] * (1 - 0.05 * confidence)  # Move SL further
            adjusted["SL2"] = levels["SL2"] * (1 + 0.03 * confidence)
            
        elif manipulation_type == "SPREAD_MANIPULATION":
            # Adjust entry levels to account for wider spreads
            adjusted["ME"] = levels["ME"] * (1 + 0.02 * confidence)
            adjusted["LR"] = levels["LR"] * (1 + 0.02 * confidence)
            
        elif manipulation_type == "PRICE_SPIKING":
            # Use more conservative entry and exit levels
            for level_name in ["ME", "LR"]:
                if level_name in adjusted:
                    adjusted[level_name] = levels[level_name] * (1 + 0.03 * confidence)
                    
        return adjusted
    
    def retrain_models(self, market_data: pd.DataFrame) -> Dict:
        """Retrain ML models with latest data"""
        if len(self.historical_trades) >= self.min_samples_for_training:
            return self.optimize_fibonacci_levels(market_data, self.historical_trades)
        else:
            return {
                "status": "insufficient_data",
                "required_samples": self.min_samples_for_training,
                "current_samples": len(self.historical_trades)
            }
    
    def save_learning_data(self, filepath: str):
        """Save learning data to file"""
        try:
            data = {
                "historical_trades": self.historical_trades,
                "adaptive_levels": self.adaptive_levels,
                "confidence_scores": self.confidence_scores,
                "last_training": self.last_training.isoformat() if self.last_training else None,
                "models_trained": self.models_trained
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, default=str, indent=2)
                
            return {"status": "success", "filepath": filepath}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def load_learning_data(self, filepath: str):
        """Load learning data from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                self.historical_trades = data.get("historical_trades", [])
                self.adaptive_levels = data.get("adaptive_levels", self.base_levels)
                self.confidence_scores = data.get("confidence_scores", {})
                self.models_trained = data.get("models_trained", False)
                
                if data.get("last_training"):
                    self.last_training = datetime.fromisoformat(data["last_training"])
                    
                return {"status": "success", "trades_loaded": len(self.historical_trades)}
            else:
                return {"status": "file_not_found"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
