"""
ProQuants Enhanced Trading Strategy
Integrates Goloji Bhudasi logic with AI/ML/Neural Networks
Pure MT5 data - No conflicts - Mathematical certainty
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json

class ProQuantsEnhancedStrategy:
    def __init__(self, ai_system):
        self.ai_system = ai_system
        self.logger = logging.getLogger(__name__)
        
        # Goloji Bhudasi Fibonacci Levels (Original proven logic)
        self.fibonacci_levels = {
            'ME': 0.685,     # Market Entry
            'SL1': 0.76,     # Stop Loss 1  
            'SL2': 0.85,     # Stop Loss 2
            'TP1': 0.50,     # Take Profit 1
            'TP2': 0.38,     # Take Profit 2
            'TP3': 0.23,     # Take Profit 3
            'TP4': 0.00      # Take Profit 4
        }
        
        # Risk Management (Enhanced with AI)
        self.min_rrr = 4.0  # Minimum 1:4 Risk-Reward Ratio
        self.max_risk_per_trade = 0.02  # 2% max risk
        self.confidence_threshold = 0.7  # AI confidence threshold
        
        # BOS Detection parameters
        self.bos_confirmation_bars = 3
        self.structure_timeframes = ['M5', 'M15', 'H1']
        
        # AI Enhancement settings
        self.ai_weight = 0.6  # 60% AI, 40% traditional analysis
        self.neural_confidence_min = 0.65
        self.manipulation_filter = True
        
        # Performance tracking
        self.trade_history = []
        self.ai_performance_stats = {}
        
    def analyze_market_structure(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Analyze market structure with BOS detection
        Enhanced with AI pattern recognition
        """
        try:
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "bos_detected": False,
                "bos_direction": None,
                "structure_levels": {},
                "ai_enhancement": {},
                "trade_setup": None
            }
            
            if len(df) < 50:
                return analysis
            
            # Get AI predictions
            ai_predictions = self.ai_system.get_ai_predictions(symbol, df)
            analysis["ai_enhancement"] = ai_predictions
            
            # Check for manipulation before proceeding
            if ai_predictions.get("manipulation_detected", False):
                analysis["manipulation_warning"] = True
                if self.manipulation_filter:
                    self.logger.warning(f"Manipulation detected for {symbol} - Filtering trade")
                    return analysis
            
            # Calculate swing highs and lows
            swing_highs = self._find_swing_highs(df)
            swing_lows = self._find_swing_lows(df)
            
            # Detect BOS (Break of Structure)
            bos_result = self._detect_bos(df, swing_highs, swing_lows)
            analysis.update(bos_result)
            
            # Calculate structure levels if BOS detected
            if analysis["bos_detected"]:
                structure_levels = self._calculate_structure_levels(df, bos_result)
                analysis["structure_levels"] = structure_levels
                
                # AI-Enhanced level validation
                validated_levels = self._ai_validate_levels(structure_levels, ai_predictions)
                analysis["ai_validated_levels"] = validated_levels
                
                # Generate trade setup
                trade_setup = self._generate_trade_setup(
                    df, analysis, ai_predictions, validated_levels
                )
                analysis["trade_setup"] = trade_setup
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Market structure analysis failed for {symbol}: {e}")
            return analysis
    
    def _find_swing_highs(self, df: pd.DataFrame, window: int = 5) -> List[Tuple]:
        """Find swing highs in price data"""
        highs = []
        for i in range(window, len(df) - window):
            current_high = df['high'].iloc[i]
            is_swing_high = True
            
            # Check if current high is higher than surrounding bars
            for j in range(i - window, i + window + 1):
                if j != i and df['high'].iloc[j] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                highs.append((i, current_high, df.index[i]))
        
        return highs[-10:]  # Keep last 10 swing highs
    
    def _find_swing_lows(self, df: pd.DataFrame, window: int = 5) -> List[Tuple]:
        """Find swing lows in price data"""
        lows = []
        for i in range(window, len(df) - window):
            current_low = df['low'].iloc[i]
            is_swing_low = True
            
            # Check if current low is lower than surrounding bars
            for j in range(i - window, i + window + 1):
                if j != i and df['low'].iloc[j] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                lows.append((i, current_low, df.index[i]))
        
        return lows[-10:]  # Keep last 10 swing lows
    
    def _detect_bos(self, df: pd.DataFrame, swing_highs: List, swing_lows: List) -> Dict:
        """
        Detect Break of Structure (BOS)
        Enhanced with AI confirmation
        """
        bos_result = {
            "bos_detected": False,
            "bos_direction": None,
            "bos_level": None,
            "bos_strength": 0,
            "confirmation_bars": 0
        }
        
        try:
            current_price = df['close'].iloc[-1]
            
            # Check for bullish BOS (break above recent swing high)
            if swing_highs:
                recent_high = max(swing_highs, key=lambda x: x[2])  # Most recent high
                if current_price > recent_high[1]:
                    # Count confirmation bars
                    confirmation = 0
                    for i in range(len(df) - self.bos_confirmation_bars, len(df)):
                        if df['close'].iloc[i] > recent_high[1]:
                            confirmation += 1
                    
                    if confirmation >= self.bos_confirmation_bars:
                        bos_result.update({
                            "bos_detected": True,
                            "bos_direction": "BULLISH",
                            "bos_level": recent_high[1],
                            "bos_strength": (current_price - recent_high[1]) / recent_high[1],
                            "confirmation_bars": confirmation
                        })
            
            # Check for bearish BOS (break below recent swing low)
            if swing_lows and not bos_result["bos_detected"]:
                recent_low = min(swing_lows, key=lambda x: x[2])  # Most recent low
                if current_price < recent_low[1]:
                    # Count confirmation bars
                    confirmation = 0
                    for i in range(len(df) - self.bos_confirmation_bars, len(df)):
                        if df['close'].iloc[i] < recent_low[1]:
                            confirmation += 1
                    
                    if confirmation >= self.bos_confirmation_bars:
                        bos_result.update({
                            "bos_detected": True,
                            "bos_direction": "BEARISH",
                            "bos_level": recent_low[1],
                            "bos_strength": (recent_low[1] - current_price) / recent_low[1],
                            "confirmation_bars": confirmation
                        })
            
            return bos_result
            
        except Exception as e:
            self.logger.error(f"BOS detection failed: {e}")
            return bos_result
    
    def _calculate_structure_levels(self, df: pd.DataFrame, bos_result: Dict) -> Dict:
        """
        Calculate trading levels based on market structure
        Using original Goloji Bhudasi fibonacci levels
        """
        levels = {}
        
        try:
            current_price = df['close'].iloc[-1]
            bos_level = bos_result["bos_level"]
            direction = bos_result["bos_direction"]
            
            if direction == "BULLISH":
                # For bullish BOS, calculate levels from recent low to BOS level
                recent_low = df['low'].rolling(20).min().iloc[-1]
                price_range = bos_level - recent_low
                
                # Fibonacci-based levels
                levels = {
                    'entry_zone': recent_low + (price_range * self.fibonacci_levels['ME']),
                    'stop_loss_1': recent_low + (price_range * self.fibonacci_levels['SL1']),
                    'stop_loss_2': recent_low + (price_range * self.fibonacci_levels['SL2']),
                    'take_profit_1': recent_low + (price_range * self.fibonacci_levels['TP1']),
                    'take_profit_2': recent_low + (price_range * self.fibonacci_levels['TP2']),
                    'take_profit_3': recent_low + (price_range * self.fibonacci_levels['TP3']),
                    'take_profit_4': recent_low + (price_range * self.fibonacci_levels['TP4']),
                    'range_low': recent_low,
                    'range_high': bos_level
                }
                
            elif direction == "BEARISH":
                # For bearish BOS, calculate levels from recent high to BOS level
                recent_high = df['high'].rolling(20).max().iloc[-1]
                price_range = recent_high - bos_level
                
                # Fibonacci-based levels (inverted for bearish)
                levels = {
                    'entry_zone': recent_high - (price_range * self.fibonacci_levels['ME']),
                    'stop_loss_1': recent_high - (price_range * self.fibonacci_levels['SL1']),
                    'stop_loss_2': recent_high - (price_range * self.fibonacci_levels['SL2']),
                    'take_profit_1': recent_high - (price_range * self.fibonacci_levels['TP1']),
                    'take_profit_2': recent_high - (price_range * self.fibonacci_levels['TP2']),
                    'take_profit_3': recent_high - (price_range * self.fibonacci_levels['TP3']),
                    'take_profit_4': recent_high - (price_range * self.fibonacci_levels['TP4']),
                    'range_low': bos_level,
                    'range_high': recent_high
                }
            
            # Add risk-reward calculations
            if levels:
                entry = levels['entry_zone']
                stop_loss = levels['stop_loss_1']
                take_profit = levels['take_profit_1']
                
                risk = abs(entry - stop_loss)
                reward = abs(take_profit - entry)
                rrr = reward / risk if risk > 0 else 0
                
                levels.update({
                    'risk_amount': risk,
                    'reward_amount': reward,
                    'risk_reward_ratio': rrr,
                    'meets_rrr_criteria': rrr >= self.min_rrr
                })
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Level calculation failed: {e}")
            return {}
    
    def _ai_validate_levels(self, structure_levels: Dict, ai_predictions: Dict) -> Dict:
        """
        Validate structure levels using AI predictions
        Enhance accuracy with machine learning
        """
        validation = {
            "ai_confidence": ai_predictions.get("confidence", 0),
            "neural_support": False,
            "ml_support": False,
            "volatility_adjusted": False,
            "final_score": 0
        }
        
        try:
            # Neural network validation
            neural_pred = ai_predictions.get("neural_prediction")
            if neural_pred:
                neural_confidence = neural_pred.get("confidence_score", 0)
                if neural_confidence >= self.neural_confidence_min:
                    validation["neural_support"] = True
                    validation["neural_confidence"] = neural_confidence
            
            # ML model validation
            ml_preds = ai_predictions.get("ml_predictions", {})
            if ml_preds:
                price_direction = ml_preds.get("price_direction", 0.5)
                if abs(price_direction - 0.5) > 0.2:  # Significant deviation from neutral
                    validation["ml_support"] = True
                    validation["ml_confidence"] = abs(price_direction - 0.5) * 2
            
            # Volatility adjustment
            predicted_volatility = None
            if neural_pred:
                predicted_volatility = neural_pred.get("volatility_forecast")
            elif ml_preds:
                predicted_volatility = ml_preds.get("volatility")
            
            if predicted_volatility:
                # Adjust levels based on predicted volatility
                volatility_multiplier = max(0.5, min(2.0, predicted_volatility * 10))
                validation["volatility_multiplier"] = volatility_multiplier
                validation["volatility_adjusted"] = True
            
            # Calculate final validation score
            score_components = []
            if validation["neural_support"]:
                score_components.append(validation["neural_confidence"] * 0.6)
            if validation["ml_support"]:
                score_components.append(validation["ml_confidence"] * 0.4)
            
            validation["final_score"] = np.mean(score_components) if score_components else 0
            
            return validation
            
        except Exception as e:
            self.logger.error(f"AI validation failed: {e}")
            return validation
    
    def _generate_trade_setup(self, df: pd.DataFrame, analysis: Dict, 
                            ai_predictions: Dict, validated_levels: Dict) -> Optional[Dict]:
        """
        Generate complete trade setup with AI enhancement
        Mathematical certainty approach
        """
        try:
            structure_levels = analysis["structure_levels"]
            bos_direction = analysis["bos_direction"]
            
            # Check if setup meets all criteria
            if not structure_levels.get("meets_rrr_criteria", False):
                return None
            
            # Check AI confidence
            ai_confidence = validated_levels.get("final_score", 0)
            if ai_confidence < self.confidence_threshold:
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Generate trade setup
            trade_setup = {
                "symbol": analysis["symbol"],
                "timestamp": datetime.now(),
                "direction": bos_direction,
                "strategy": "ProQuants_Enhanced_Goloji_Bhudasi",
                "ai_enhanced": True,
                
                # Entry details
                "entry_price": structure_levels["entry_zone"],
                "entry_valid": self._is_price_near_level(current_price, structure_levels["entry_zone"], 0.001),
                
                # Risk management
                "stop_loss": structure_levels["stop_loss_1"],
                "stop_loss_backup": structure_levels["stop_loss_2"],
                
                # Take profits (scaled)
                "take_profit_1": structure_levels["take_profit_1"],
                "take_profit_2": structure_levels["take_profit_2"],
                "take_profit_3": structure_levels["take_profit_3"],
                "take_profit_4": structure_levels["take_profit_4"],
                
                # Risk-Reward
                "risk_reward_ratio": structure_levels["risk_reward_ratio"],
                "risk_amount": structure_levels["risk_amount"],
                "reward_amount": structure_levels["reward_amount"],
                
                # AI Enhancement
                "ai_confidence": ai_confidence,
                "neural_support": validated_levels.get("neural_support", False),
                "ml_support": validated_levels.get("ml_support", False),
                "manipulation_risk": ai_predictions.get("manipulation_detected", False),
                
                # Position sizing (AI-adjusted)
                "position_size": self._calculate_position_size(
                    structure_levels["risk_amount"], ai_confidence
                ),
                
                # Validity
                "setup_valid": True,
                "expiry_time": datetime.now() + timedelta(hours=4),  # 4-hour validity
                
                # Performance tracking
                "setup_id": f"{analysis['symbol']}_{int(datetime.now().timestamp())}"
            }
            
            # Add volatility adjustments if available
            if validated_levels.get("volatility_adjusted"):
                vol_multiplier = validated_levels["volatility_multiplier"]
                trade_setup["volatility_adjusted"] = True
                trade_setup["volatility_multiplier"] = vol_multiplier
                
                # Adjust stop loss for high volatility
                if vol_multiplier > 1.5:
                    trade_setup["stop_loss_adjusted"] = True
                    risk_adjustment = structure_levels["risk_amount"] * (vol_multiplier - 1) * 0.5
                    
                    if bos_direction == "BULLISH":
                        trade_setup["stop_loss"] -= risk_adjustment
                    else:
                        trade_setup["stop_loss"] += risk_adjustment
            
            self.logger.info(f"Trade setup generated for {analysis['symbol']}: "
                           f"{bos_direction} with {ai_confidence:.1%} AI confidence")
            
            return trade_setup
            
        except Exception as e:
            self.logger.error(f"Trade setup generation failed: {e}")
            return None
    
    def _is_price_near_level(self, current_price: float, target_level: float, 
                           tolerance: float = 0.001) -> bool:
        """Check if current price is near target level within tolerance"""
        return abs(current_price - target_level) / target_level <= tolerance
    
    def _calculate_position_size(self, risk_amount: float, ai_confidence: float) -> float:
        """
        Calculate position size based on risk and AI confidence
        Enhanced risk management
        """
        try:
            # Base position size (2% max risk)
            account_balance = self.ai_system.account_info.get("balance", 10000)
            max_risk_amount = account_balance * self.max_risk_per_trade
            
            # AI confidence adjustment
            confidence_multiplier = 0.5 + (ai_confidence * 0.5)  # 0.5 to 1.0
            adjusted_risk = max_risk_amount * confidence_multiplier
            
            # Calculate position size
            position_size = adjusted_risk / risk_amount if risk_amount > 0 else 0
            
            # Ensure minimum and maximum limits
            min_position = 0.01  # Minimum lot size
            max_position = account_balance * 0.1 / 1000  # Max 10% account per trade
            
            return max(min_position, min(position_size, max_position))
            
        except Exception as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return 0.01  # Default minimum
    
    def get_trading_signals(self, symbols: List[str] = None) -> Dict:
        """
        Get trading signals for all or specified symbols
        Main entry point for signal generation
        """
        symbols = symbols or self.ai_system.SYMBOLS
        signals = {}
        
        for symbol in symbols:
            try:
                # Get market data
                df = self.ai_system.get_ai_training_data(symbol, hours=24)  # Last 24 hours
                if df is None:
                    signals[symbol] = {"error": "No market data available"}
                    continue
                
                # Analyze market structure
                analysis = self.analyze_market_structure(df, symbol)
                signals[symbol] = analysis
                
                # Log significant signals
                if analysis.get("trade_setup"):
                    setup = analysis["trade_setup"]
                    self.logger.info(f"TRADE SIGNAL: {symbol} {setup['direction']} "
                                   f"RRR: {setup['risk_reward_ratio']:.1f} "
                                   f"AI: {setup['ai_confidence']:.1%}")
                
            except Exception as e:
                self.logger.error(f"Signal generation failed for {symbol}: {e}")
                signals[symbol] = {"error": str(e)}
        
        return signals
    
    def validate_trade_setup(self, trade_setup: Dict) -> Dict:
        """
        Final validation before trade execution
        Multiple layers of verification
        """
        validation = {
            "valid": False,
            "checks": {},
            "issues": [],
            "confidence": 0
        }
        
        try:
            # Check basic requirements
            validation["checks"]["has_entry"] = trade_setup.get("entry_price") is not None
            validation["checks"]["has_stop_loss"] = trade_setup.get("stop_loss") is not None
            validation["checks"]["has_take_profit"] = trade_setup.get("take_profit_1") is not None
            
            # Check risk-reward ratio
            rrr = trade_setup.get("risk_reward_ratio", 0)
            validation["checks"]["rrr_acceptable"] = rrr >= self.min_rrr
            
            # Check AI confidence
            ai_confidence = trade_setup.get("ai_confidence", 0)
            validation["checks"]["ai_confidence_ok"] = ai_confidence >= self.confidence_threshold
            
            # Check manipulation risk
            manipulation_risk = trade_setup.get("manipulation_risk", False)
            validation["checks"]["manipulation_safe"] = not manipulation_risk
            
            # Check expiry
            expiry = trade_setup.get("expiry_time")
            if expiry:
                validation["checks"]["not_expired"] = datetime.now() < expiry
            else:
                validation["checks"]["not_expired"] = True
            
            # Collect issues
            for check, passed in validation["checks"].items():
                if not passed:
                    validation["issues"].append(check)
            
            # Calculate overall validation
            validation["valid"] = len(validation["issues"]) == 0
            validation["confidence"] = ai_confidence if validation["valid"] else 0
            
            return validation
            
        except Exception as e:
            validation["issues"].append(f"Validation error: {e}")
            return validation
    
    def get_performance_summary(self) -> Dict:
        """Get strategy performance summary"""
        return {
            "total_signals": len(self.trade_history),
            "ai_enhanced_signals": sum(1 for trade in self.trade_history if trade.get("ai_enhanced", False)),
            "average_ai_confidence": np.mean([trade.get("ai_confidence", 0) for trade in self.trade_history]) if self.trade_history else 0,
            "symbols_active": len(self.ai_system.SYMBOLS),
            "last_signal_time": max([trade.get("timestamp") for trade in self.trade_history]) if self.trade_history else None
        }
