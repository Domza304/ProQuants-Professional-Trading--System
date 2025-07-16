"""
ProQuants Unified AI System - MT5 Data Integration
Combines Neural Networks, Machine Learning, and AI for maximum trading efficiency
Pure MT5 API - No Deriv API conflicts
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv

# AI/ML Imports
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    import pickle
    ML_AVAILABLE = True
    print("✓ Scikit-learn available for machine learning")
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠ Scikit-learn not available: {e}")
    print("  Install with: pip install scikit-learn")
    # Create dummy classes for fallback
    class RandomForestRegressor: pass
    class IsolationForest: pass
    class StandardScaler: pass
    class MinMaxScaler: pass

# Neural Network Implementation (TensorFlow/Keras fallback)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    NEURAL_AVAILABLE = True
    print("✓ TensorFlow/Keras available for neural networks")
except ImportError as e:
    NEURAL_AVAILABLE = False
    print(f"⚠ TensorFlow not available: {e}")
    print("  Install with: pip install tensorflow")
    # Create dummy classes for fallback
    class tf:
        class keras:
            class Sequential: pass
            class layers:
                class Dense: pass
                class Dropout: pass
            class utils:
                class to_categorical: pass
    keras = tf.keras
    layers = tf.keras.layers

class UnifiedAITradingSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load MT5 credentials from .env
        load_dotenv()
        self.login = int(os.getenv('MT5_LOGIN', '31833954'))
        self.password = os.getenv('MT5_PASSWORD', '@Dmc65070*')
        self.server = os.getenv('MT5_SERVER', 'Deriv-Demo')
        
        # MT5 Connection
        self.connected = False
        self.account_info = {}
        
        # Deriv Synthetic Indices (MT5 Symbol Names)
        self.SYMBOLS = [
            "Volatility 75 Index",
            "Volatility 25 Index", 
            "Volatility 75 (1s) Index"
        ]
        
        # AI System Components
        self.neural_networks = {}  # One per symbol
        self.ml_models = {}        # Classical ML models
        self.ai_scalers = {}       # Data scalers
        self.prediction_cache = {} # Cached predictions
        
        # Fractal learning - Start from lowest TF going up (more data, better patterns)
        self.FRACTAL_TIMEFRAMES = [
            ('M1', mt5.TIMEFRAME_M1, 1440),    # 1 min - 24 hours of data
            ('M5', mt5.TIMEFRAME_M5, 288),     # 5 min - 24 hours of data  
            ('M15', mt5.TIMEFRAME_M15, 96),    # 15 min - 24 hours of data
            ('H1', mt5.TIMEFRAME_H1, 24),     # 1 hour - 24 hours of data
            ('H4', mt5.TIMEFRAME_H4, 6)       # 4 hour - 24 hours of data
        ]
        
        # Data requirements (minimum 12 hours for neural training)
        self.MIN_DATA_HOURS = 12
        self.MIN_BARS_M1 = self.MIN_DATA_HOURS * 60   # 720 bars M1
        self.MIN_BARS_M5 = self.MIN_DATA_HOURS * 12   # 144 bars M5
        
        # AI Model configurations
        self.neural_config = {
            'input_features': 20,
            'hidden_layers': [64, 32, 16],
            'output_features': 3,  # Price direction, volatility, confidence
            'epochs': 100,
            'batch_size': 32
        }
        
        # Initialize AI components
        self._initialize_ai_system()
        
    def _initialize_ai_system(self):
        """Initialize all AI components"""
        self.logger.info("Initializing Unified AI Trading System...")
        
        # Suppress TensorFlow info messages for cleaner output
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        for symbol in self.SYMBOLS:
            # Initialize per-symbol AI models
            self.neural_networks[symbol] = None
            self.ml_models[symbol] = {
                'price_predictor': None,
                'volatility_predictor': None,
                'manipulation_detector': None
            }
            self.ai_scalers[symbol] = {
                'feature_scaler': StandardScaler(),
                'target_scaler': MinMaxScaler()
            }
            self.prediction_cache[symbol] = {
                'timestamp': None,
                'predictions': None,
                'confidence': 0.0
            }
    
    def initialize_mt5(self) -> Dict:
        """Initialize MT5 connection with comprehensive error handling"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                error = mt5.last_error()
                return {"success": False, "error": f"MT5 init failed: {error}"}
            
            # Login to account
            if not mt5.login(self.login, password=self.password, server=self.server):
                error = mt5.last_error()
                mt5.shutdown()
                return {"success": False, "error": f"Login failed: {error}"}
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                return {"success": False, "error": "Failed to get account info"}
            
            self.account_info = account_info._asdict()
            self.connected = True
            
            # Validate symbol availability
            available_symbols = self._validate_symbols()
            
            self.logger.info(f"MT5 Connected: Account {self.login} on {self.server}")
            self.logger.info(f"AI System Ready: Neural={NEURAL_AVAILABLE}, ML={ML_AVAILABLE}")
            
            return {
                "success": True,
                "account_info": self.account_info,
                "available_symbols": available_symbols,
                "ai_status": {
                    "neural_networks": NEURAL_AVAILABLE,
                    "machine_learning": ML_AVAILABLE,
                    "symbols_ready": len(available_symbols)
                }
            }
            
        except Exception as e:
            self.logger.error(f"MT5 initialization failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _validate_symbols(self) -> List[str]:
        """Validate that Deriv symbols are available in MT5"""
        available = []
        try:
            symbols = mt5.symbols_get()
            if symbols:
                symbol_names = [s.name for s in symbols]
                for symbol in self.SYMBOLS:
                    if symbol in symbol_names:
                        available.append(symbol)
                        # Enable symbol for trading
                        mt5.symbol_select(symbol, True)
                    else:
                        self.logger.warning(f"Symbol not available: {symbol}")
        except Exception as e:
            self.logger.error(f"Symbol validation failed: {e}")
        
        return available
    
    def get_fractal_training_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get fractal training data from lowest TF going up
        Provides maximum data density for ML learning
        """
        fractal_data = {}
        
        try:
            if not self.connected:
                init_result = self.initialize_mt5()
                if not init_result["success"]:
                    return fractal_data
            
            self.logger.info(f"Collecting fractal data for {symbol} from M1 upwards...")
            
            # Collect data from each timeframe (M1 -> H4)
            for tf_name, tf_constant, bars_needed in self.FRACTAL_TIMEFRAMES:
                try:
                    self.logger.info(f"Getting {tf_name} data: {bars_needed} bars")
                    
                    # Get rates from MT5
                    rates = mt5.copy_rates_from_pos(symbol, tf_constant, 0, bars_needed)
                    
                    if rates is None or len(rates) < 10:
                        self.logger.warning(f"Insufficient {tf_name} data for {symbol}")
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    
                    # Add timeframe-specific features
                    df = self._add_fractal_features(df, tf_name, symbol)
                    
                    fractal_data[tf_name] = df
                    self.logger.info(f"✓ {tf_name}: {len(df)} bars collected for {symbol}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to get {tf_name} data for {symbol}: {e}")
                    continue
            
            return fractal_data
            
        except Exception as e:
            self.logger.error(f"Fractal data collection failed for {symbol}: {e}")
            return fractal_data
    
    def _add_fractal_features(self, df: pd.DataFrame, timeframe: str, symbol: str) -> pd.DataFrame:
        """Add timeframe-specific fractal features"""
        try:
            # Base features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(20).std()
            
            # Timeframe-specific adjustments
            if timeframe == 'M1':
                # High frequency features for M1
                df['micro_trend'] = df['close'].rolling(5).mean()
                df['noise_ratio'] = (df['high'] - df['low']) / df['close']
                df['tick_momentum'] = df['close'].diff().rolling(10).sum()
                
            elif timeframe == 'M5':
                # Medium frequency features
                df['short_trend'] = df['close'].rolling(12).mean()  # 1 hour
                df['momentum_5m'] = df['close'] / df['close'].shift(12) - 1
                
            elif timeframe == 'M15':
                # Swing features
                df['swing_high'] = df['high'].rolling(8).max()
                df['swing_low'] = df['low'].rolling(8).min()
                df['range_position'] = (df['close'] - df['swing_low']) / (df['swing_high'] - df['swing_low'])
                
            elif timeframe == 'H1':
                # Trend features
                df['daily_trend'] = df['close'].rolling(24).mean()
                df['trend_strength'] = abs(df['close'] - df['daily_trend']) / df['daily_trend']
                
            elif timeframe == 'H4':
                # Structure features
                df['weekly_high'] = df['high'].rolling(42).max()  # ~1 week
                df['weekly_low'] = df['low'].rolling(42).min()
                df['structure_level'] = (df['close'] - df['weekly_low']) / (df['weekly_high'] - df['weekly_low'])
            
            # Fractal relationship features
            df[f'{timeframe}_fractal_pos'] = df['close'] / df['close'].rolling(20).mean() - 1
            df[f'{timeframe}_volatility_regime'] = (df['volatility'] > df['volatility'].rolling(50).mean()).astype(int)
            
            # Clean data
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            df.replace([np.inf, -np.inf], 0, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to add fractal features for {timeframe}: {e}")
            return df
        """
        Get market data specifically formatted for AI/ML training
        Minimum 12 hours of data for neural network training
        """
        try:
            if not self.connected:
                init_result = self.initialize_mt5()
                if not init_result["success"]:
                    return None
            
            # Calculate required bars (default: minimum 12 hours)
            hours = hours or self.MIN_DATA_HOURS
            bars_needed = hours * 12  # M5 bars per hour
            
            # Get rates from MT5
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars_needed)
            
            if rates is None or len(rates) < self.MIN_BARS_M5:
                self.logger.warning(f"Insufficient data for {symbol}: {len(rates) if rates else 0} bars")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Add AI-specific features
            df = self._add_ai_features(df, symbol)
            
            # Validate data quality for AI training
            quality = self._validate_ai_data_quality(df, symbol)
            if quality['status'] != 'GOOD':
                self.logger.warning(f"Data quality issues for {symbol}: {quality['issues']}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get AI training data for {symbol}: {e}")
            return None
    
    def _add_ai_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add comprehensive features for AI/ML models"""
        try:
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(20).std()
            df['price_momentum'] = df['close'] / df['close'].shift(10) - 1
            
            # Technical indicators (pure numpy/pandas)
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_vs_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
            
            # Volatility indicators
            df['atr'] = self._calculate_atr(df, 14)
            df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume-based features
            if 'tick_volume' in df.columns:
                df['volume_sma'] = df['tick_volume'].rolling(20).mean()
                df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
                df['price_volume'] = df['returns'] * df['volume_ratio']
            
            # Time-based features for neural networks
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_market_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
            
            # Candlestick patterns
            df['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'])).fillna(0)
            df['hammer'] = ((df['close'] > df['open']) & 
                           ((df['open'] - df['low']) > 2 * (df['close'] - df['open']))).astype(int)
            
            # Manipulation detection features
            df['spread_proxy'] = (df['high'] - df['low']) / df['close']
            df['gap'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            df['price_spike'] = (abs(df['returns']) > df['volatility'] * 3).astype(int)
            
            # Lag features for neural networks
            for lag in [1, 2, 3, 5, 10]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
                df[f'volatility_lag_{lag}'] = df['volatility'].shift(lag)
            
            # Forward-looking targets (for training)
            df['future_return_1'] = df['returns'].shift(-1)
            df['future_return_5'] = df['close'].shift(-5) / df['close'] - 1
            df['future_volatility'] = df['returns'].shift(-5).rolling(5).std()
            
            # Clean data
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            
            # Remove infinite values
            df.replace([np.inf, -np.inf], 0, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to add AI features: {e}")
            return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2):
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(period).mean()
        rolling_std = df['close'].rolling(period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        return upper, lower
    
    def _validate_ai_data_quality(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Validate data quality for AI training"""
        try:
            issues = []
            
            # Check minimum data requirement
            if len(df) < self.MIN_BARS_M5:
                issues.append(f"Insufficient data: {len(df)} bars (need {self.MIN_BARS_M5})")
            
            # Check for missing values
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if missing_pct > 0.05:  # 5% threshold
                issues.append(f"Too many missing values: {missing_pct:.1%}")
            
            # Check for data staleness
            last_update = df.index.max()
            staleness = datetime.now() - last_update.to_pydatetime()
            if staleness > timedelta(minutes=30):
                issues.append(f"Stale data: {staleness}")
            
            # Check for price anomalies
            extreme_returns = (abs(df['returns']) > 0.1).sum()  # >10% moves
            if extreme_returns > len(df) * 0.001:  # More than 0.1% of data
                issues.append(f"Unusual volatility: {extreme_returns} extreme moves")
            
            status = "GOOD" if not issues else "POOR" if len(issues) > 2 else "FAIR"
            
            return {
                "symbol": symbol,
                "status": status,
                "total_bars": len(df),
                "data_span": f"{(df.index.max() - df.index.min()).total_seconds() / 3600:.1f} hours",
                "issues": issues
            }
            
        except Exception as e:
            return {"symbol": symbol, "status": "ERROR", "error": str(e)}
    
    def train_fractal_neural_network(self, symbol: str) -> Dict:
        """Train neural network using fractal data from all timeframes"""
        if not NEURAL_AVAILABLE:
            return {"success": False, "error": "TensorFlow not available"}
        
        try:
            self.logger.info(f"Training fractal neural network for {symbol}...")
            
            # Get fractal data from all timeframes
            fractal_data = self.get_fractal_training_data(symbol)
            
            if not fractal_data:
                return {"success": False, "error": "No fractal data available"}
            
            # Combine fractal features from all timeframes
            combined_features = []
            combined_targets = []
            
            # Start with M1 as base (most data)
            if 'M1' not in fractal_data:
                return {"success": False, "error": "M1 data required for fractal learning"}
            
            base_df = fractal_data['M1'].copy()
            
            # Add features from higher timeframes (fractal approach)
            for tf_name in ['M5', 'M15', 'H1', 'H4']:
                if tf_name in fractal_data:
                    tf_df = fractal_data[tf_name]
                    # Resample to M1 timeframe for alignment
                    tf_resampled = self._resample_to_base_timeframe(tf_df, tf_name, base_df)
                    # Add as additional features
                    for col in tf_resampled.columns:
                        if col not in ['open', 'high', 'low', 'close', 'tick_volume']:
                            base_df[f'{tf_name}_{col}'] = tf_resampled[col]
            
            # Prepare features for ML
            feature_cols = [col for col in base_df.columns if not col.startswith('future_') and 
                          col not in ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
            
            # Create targets (future price movements)
            base_df['future_return_1'] = base_df['returns'].shift(-1)
            base_df['future_return_5'] = base_df['close'].shift(-5) / base_df['close'] - 1
            base_df['future_volatility'] = base_df['returns'].shift(-5).rolling(5).std()
            
            X = base_df[feature_cols].values
            y = base_df[['future_return_1', 'future_return_5', 'future_volatility']].values
            
            # Remove rows with NaN targets
            valid_rows = ~np.isnan(y).any(axis=1) & ~np.isnan(X).any(axis=1)
            X = X[valid_rows]
            y = y[valid_rows]
            
            if len(X) < 200:  # Need more data for fractal learning
                return {"success": False, "error": "Insufficient fractal training data"}
            
            # Scale features
            scaler = self.ai_scalers[symbol]['feature_scaler']
            X_scaled = scaler.fit_transform(X)
            
            # Scale targets
            target_scaler = self.ai_scalers[symbol]['target_scaler']
            y_scaled = target_scaler.fit_transform(y)
            
            # Split data (80/20)
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
            
            # Build enhanced fractal neural network
            model = keras.Sequential([
                # Input layer
                layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
                layers.Dropout(0.3),
                
                # Fractal pattern recognition layers
                layers.Dense(96, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                
                # Multi-timeframe integration layer
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.1),
                
                # Output layer (3 predictions)
                layers.Dense(3, activation='linear')
            ])
            
            # Compile with advanced optimizer
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae', 'mape']
            )
            
            # Train with early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                epochs=150,  # More epochs for fractal learning
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate model
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            val_mae = history.history['val_mae'][-1]
            
            # Store fractal model
            self.neural_networks[symbol] = {
                'model': model,
                'feature_columns': feature_cols,
                'training_date': datetime.now(),
                'fractal_trained': True,
                'timeframes_used': list(fractal_data.keys()),
                'performance': {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'training_samples': len(X_train),
                    'total_features': len(feature_cols)
                }
            }
            
            self.logger.info(f"✓ Fractal neural network trained for {symbol}")
            self.logger.info(f"  Timeframes: {list(fractal_data.keys())}")
            self.logger.info(f"  Features: {len(feature_cols)}")
            self.logger.info(f"  Val Loss: {val_loss:.6f}")
            self.logger.info(f"  Val MAE: {val_mae:.6f}")
            
            return {
                "success": True,
                "fractal_learning": True,
                "timeframes": list(fractal_data.keys()),
                "performance": self.neural_networks[symbol]['performance'],
                "feature_count": len(feature_cols),
                "training_samples": len(X_train)
            }
            
        except Exception as e:
            self.logger.error(f"Fractal neural network training failed for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    def _resample_to_base_timeframe(self, df: pd.DataFrame, source_tf: str, base_df: pd.DataFrame) -> pd.DataFrame:
        """Resample higher timeframe data to base timeframe (M1)"""
        try:
            # Forward fill to match base timeframe index
            resampled = df.reindex(base_df.index, method='ffill')
            return resampled.fillna(0)
        except Exception as e:
            self.logger.error(f"Resampling failed for {source_tf}: {e}")
            return pd.DataFrame(index=base_df.index)
        """Train neural network for price prediction"""
        if not NEURAL_AVAILABLE:
            return {"success": False, "error": "TensorFlow not available"}
        
        try:
            # Prepare features and targets
            feature_cols = [col for col in df.columns if not col.startswith('future_') and 
                          col not in ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
            
            X = df[feature_cols].values
            y = df[['future_return_1', 'future_return_5', 'future_volatility']].values
            
            # Remove rows with NaN targets
            valid_rows = ~np.isnan(y).any(axis=1)
            X = X[valid_rows]
            y = y[valid_rows]
            
            if len(X) < 100:  # Minimum samples for training
                return {"success": False, "error": "Insufficient training data"}
            
            # Scale features
            scaler = self.ai_scalers[symbol]['feature_scaler']
            X_scaled = scaler.fit_transform(X)
            
            # Scale targets
            target_scaler = self.ai_scalers[symbol]['target_scaler']
            y_scaled = target_scaler.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )
            
            # Build neural network
            model = keras.Sequential([
                layers.Dense(self.neural_config['hidden_layers'][0], activation='relu', input_shape=(X.shape[1],)),
                layers.Dropout(0.2),
                layers.Dense(self.neural_config['hidden_layers'][1], activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(self.neural_config['hidden_layers'][2], activation='relu'),
                layers.Dense(3, activation='linear')  # Price direction, volatility, confidence
            ])
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=self.neural_config['epochs'],
                batch_size=self.neural_config['batch_size'],
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Evaluate model
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            
            # Store model
            self.neural_networks[symbol] = {
                'model': model,
                'feature_columns': feature_cols,
                'training_date': datetime.now(),
                'performance': {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'training_samples': len(X_train)
                }
            }
            
            self.logger.info(f"Neural network trained for {symbol}: Val Loss = {val_loss:.6f}")
            
            return {
                "success": True,
                "performance": self.neural_networks[symbol]['performance'],
                "feature_count": len(feature_cols),
                "training_samples": len(X_train)
            }
            
        except Exception as e:
            self.logger.error(f"Neural network training failed for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    def train_ml_models(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Train classical ML models for different aspects"""
        if not ML_AVAILABLE:
            return {"success": False, "error": "Scikit-learn not available"}
        
        try:
            results = {}
            
            # Prepare base features
            feature_cols = [col for col in df.columns if not col.startswith('future_') and 
                          col not in ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
            
            X = df[feature_cols].fillna(0)
            
            # 1. Price Direction Predictor
            y_direction = (df['future_return_1'] > 0).astype(int)
            valid_direction = ~y_direction.isna()
            
            if valid_direction.sum() > 50:
                X_dir = X[valid_direction]
                y_dir = y_direction[valid_direction]
                
                direction_model = RandomForestRegressor(n_estimators=100, random_state=42)
                direction_model.fit(X_dir, y_dir)
                direction_score = direction_model.score(X_dir, y_dir)
                
                self.ml_models[symbol]['price_predictor'] = direction_model
                results['price_predictor'] = {"score": direction_score, "samples": len(X_dir)}
            
            # 2. Volatility Predictor
            y_volatility = df['future_volatility']
            valid_volatility = ~y_volatility.isna()
            
            if valid_volatility.sum() > 50:
                X_vol = X[valid_volatility]
                y_vol = y_volatility[valid_volatility]
                
                volatility_model = RandomForestRegressor(n_estimators=100, random_state=42)
                volatility_model.fit(X_vol, y_vol)
                volatility_score = volatility_model.score(X_vol, y_vol)
                
                self.ml_models[symbol]['volatility_predictor'] = volatility_model
                results['volatility_predictor'] = {"score": volatility_score, "samples": len(X_vol)}
            
            # 3. Manipulation Detector
            manipulation_features = ['spread_proxy', 'gap', 'price_spike', 'volume_ratio']
            available_features = [f for f in manipulation_features if f in X.columns]
            
            if available_features:
                X_manip = X[available_features]
                
                manipulation_detector = IsolationForest(contamination=0.1, random_state=42)
                manipulation_detector.fit(X_manip)
                
                self.ml_models[symbol]['manipulation_detector'] = manipulation_detector
                results['manipulation_detector'] = {"features": len(available_features)}
            
            self.logger.info(f"ML models trained for {symbol}: {len(results)} models")
            
            return {"success": True, "models": results}
            
        except Exception as e:
            self.logger.error(f"ML training failed for {symbol}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_ai_predictions(self, symbol: str, current_data: pd.DataFrame) -> Dict:
        """Get comprehensive AI predictions for trading decisions"""
        try:
            predictions = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "neural_prediction": None,
                "ml_predictions": {},
                "combined_signal": "NEUTRAL",
                "confidence": 0.0,
                "manipulation_detected": False
            }
            
            # Prepare current features
            if symbol not in self.neural_networks or self.neural_networks[symbol] is None:
                return predictions
            
            nn_info = self.neural_networks[symbol]
            feature_cols = nn_info['feature_columns']
            
            # Get latest data point features
            current_features = current_data[feature_cols].iloc[-1:].fillna(0)
            
            # Neural Network Prediction
            if NEURAL_AVAILABLE and nn_info['model'] is not None:
                scaler = self.ai_scalers[symbol]['feature_scaler']
                target_scaler = self.ai_scalers[symbol]['target_scaler']
                
                X_scaled = scaler.transform(current_features.values)
                nn_pred_scaled = nn_info['model'].predict(X_scaled)
                nn_pred = target_scaler.inverse_transform(nn_pred_scaled)[0]
                
                predictions["neural_prediction"] = {
                    "price_direction": nn_pred[0],
                    "volatility_forecast": nn_pred[1],
                    "confidence_score": abs(nn_pred[2])
                }
            
            # ML Model Predictions
            if ML_AVAILABLE:
                ml_models = self.ml_models[symbol]
                
                # Price direction
                if ml_models['price_predictor'] is not None:
                    price_prob = ml_models['price_predictor'].predict(current_features)[0]
                    predictions["ml_predictions"]["price_direction"] = price_prob
                
                # Volatility
                if ml_models['volatility_predictor'] is not None:
                    vol_pred = ml_models['volatility_predictor'].predict(current_features)[0]
                    predictions["ml_predictions"]["volatility"] = vol_pred
                
                # Manipulation detection
                if ml_models['manipulation_detector'] is not None:
                    manip_features = ['spread_proxy', 'gap', 'price_spike', 'volume_ratio']
                    available_features = [f for f in manip_features if f in current_features.columns]
                    
                    if available_features:
                        manip_data = current_features[available_features]
                        anomaly_score = ml_models['manipulation_detector'].decision_function(manip_data)[0]
                        is_manipulation = ml_models['manipulation_detector'].predict(manip_data)[0] == -1
                        
                        predictions["manipulation_detected"] = is_manipulation
                        predictions["manipulation_score"] = anomaly_score
            
            # Combine predictions for final signal
            predictions = self._combine_ai_signals(predictions)
            
            # Cache predictions
            self.prediction_cache[symbol] = predictions
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"AI prediction failed for {symbol}: {e}")
            return predictions
    
    def _combine_ai_signals(self, predictions: Dict) -> Dict:
        """Combine neural network and ML predictions into unified signal"""
        try:
            signals = []
            weights = []
            
            # Neural network signal
            if predictions["neural_prediction"]:
                nn_signal = predictions["neural_prediction"]["price_direction"]
                nn_confidence = predictions["neural_prediction"]["confidence_score"]
                signals.append(nn_signal)
                weights.append(nn_confidence * 0.6)  # 60% weight for neural network
            
            # ML price direction signal
            if "price_direction" in predictions["ml_predictions"]:
                ml_signal = predictions["ml_predictions"]["price_direction"] * 2 - 1  # Convert 0-1 to -1,1
                signals.append(ml_signal)
                weights.append(0.4)  # 40% weight for ML
            
            # Calculate combined signal
            if signals and weights:
                combined_signal = np.average(signals, weights=weights)
                confidence = np.mean(weights)
                
                # Determine signal direction
                if combined_signal > 0.3:
                    signal_direction = "BULLISH"
                elif combined_signal < -0.3:
                    signal_direction = "BEARISH"
                else:
                    signal_direction = "NEUTRAL"
                
                # Adjust for manipulation
                if predictions["manipulation_detected"]:
                    confidence *= 0.5  # Reduce confidence during manipulation
                    signal_direction += "_CAUTION"
                
                predictions["combined_signal"] = signal_direction
                predictions["confidence"] = min(confidence, 1.0)
                predictions["signal_strength"] = abs(combined_signal)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Signal combination failed: {e}")
            return predictions
    
    def train_all_models(self) -> Dict:
        """Train all AI models for all symbols using fractal approach"""
        results = {}
        
        for symbol in self.SYMBOLS:
            self.logger.info(f"Training fractal AI models for {symbol}...")
            
            symbol_results = {}
            
            # Train fractal neural network (M1 -> H4)
            if NEURAL_AVAILABLE:
                nn_result = self.train_fractal_neural_network(symbol)
                symbol_results["fractal_neural_network"] = nn_result
                
                if nn_result["success"]:
                    self.logger.info(f"✓ Fractal NN trained for {symbol}: {nn_result['timeframes']}")
                else:
                    self.logger.warning(f"✗ Fractal NN failed for {symbol}: {nn_result.get('error', 'Unknown')}")
            
            # Train ML models with fractal data
            if ML_AVAILABLE:
                fractal_data = self.get_fractal_training_data(symbol)
                if fractal_data and 'M5' in fractal_data:  # Use M5 for ML training
                    ml_result = self.train_ml_models(symbol, fractal_data['M5'])
                    symbol_results["ml_models"] = ml_result
                else:
                    symbol_results["ml_models"] = {"success": False, "error": "No M5 fractal data"}
            
            results[symbol] = symbol_results
        
        # Log summary
        successful_symbols = sum(1 for r in results.values() 
                               if r.get("fractal_neural_network", {}).get("success", False))
        self.logger.info(f"✓ Fractal training complete: {successful_symbols}/{len(self.SYMBOLS)} symbols successful")
        
        return results
    
    def get_system_status(self) -> Dict:
        """Get comprehensive AI system status"""
        return {
            "mt5_connected": self.connected,
            "ai_capabilities": {
                "neural_networks": NEURAL_AVAILABLE,
                "machine_learning": ML_AVAILABLE,
                "models_trained": sum(1 for nn in self.neural_networks.values() if nn is not None)
            },
            "symbols": {
                symbol: {
                    "neural_trained": self.neural_networks[symbol] is not None,
                    "ml_trained": any(model is not None for model in self.ml_models[symbol].values()),
                    "last_prediction": self.prediction_cache[symbol]["timestamp"] if self.prediction_cache[symbol]["timestamp"] else None
                }
                for symbol in self.SYMBOLS
            },
            "data_quality": [
                self._validate_ai_data_quality(self.get_ai_training_data(symbol), symbol)
                for symbol in self.SYMBOLS
                if self.get_ai_training_data(symbol) is not None
            ]
        }
    
    def shutdown(self):
        """Shutdown MT5 and cleanup AI resources"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            
        # Clear AI models to free memory
        for symbol in self.SYMBOLS:
            if self.neural_networks[symbol] and NEURAL_AVAILABLE:
                del self.neural_networks[symbol]['model']
            self.neural_networks[symbol] = None
        
        self.logger.info("Unified AI Trading System shutdown complete")
    
    def get_ai_training_data(self, symbol: str, hours: int = None) -> Optional[pd.DataFrame]:
        """
        Get market data specifically formatted for AI/ML training
        Minimum 12 hours of data for neural network training
        """
        try:
            if not self.connected:
                init_result = self.initialize_mt5()
                if not init_result["success"]:
                    return None
                    
            # Calculate required bars (default: minimum 12 hours)
            hours = hours or self.MIN_DATA_HOURS
            bars_needed = hours * 12  # M5 bars per hour
            
            # Get rates from MT5
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars_needed)
            
            if rates is None or len(rates) < self.MIN_BARS_M5:
                self.logger.warning(f"Insufficient data for {symbol}: {len(rates) if rates else 0} bars")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Add AI-specific features
            df = self._add_ai_features(df, symbol)
            
            # Validate data quality for AI training
            quality = self._validate_ai_data_quality(df, symbol)
            if quality['status'] != 'GOOD':
                self.logger.warning(f"Data quality issues for {symbol}: {quality['issues']}")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get AI training data for {symbol}: {e}")
            return None
    
    def _add_ai_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add comprehensive features for AI/ML models"""
        try:
            # Price-based features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(20).std()
            df['price_momentum'] = df['close'] / df['close'].shift(10) - 1
            
            # Technical indicators (pure numpy/pandas)
            for period in [5, 10, 20, 50]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_vs_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
            
            # Volatility indicators
            df['atr'] = self._calculate_atr(df, 14)
            df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume-based features
            if 'tick_volume' in df.columns:
                df['volume_sma'] = df['tick_volume'].rolling(20).mean()
                df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
                df['price_volume'] = df['returns'] * df['volume_ratio']
            
            # Time-based features for neural networks
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['is_market_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
            
            # Candlestick patterns
            df['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'])).fillna(0)
            df['hammer'] = ((df['close'] > df['open']) & 
                           ((df['open'] - df['low']) > 2 * (df['close'] - df['open']))).astype(int)
            
            # Manipulation detection features
            df['spread_proxy'] = (df['high'] - df['low']) / df['close']
            df['gap'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            df['price_spike'] = (abs(df['returns']) > df['volatility'] * 3).astype(int)
            
            # Lag features for neural networks
            for lag in [1, 2, 3, 5, 10]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
                df[f'volatility_lag_{lag}'] = df['volatility'].shift(lag)
            
            # Forward-looking targets (for training)
            df['future_return_1'] = df['returns'].shift(-1)
            df['future_return_5'] = df['close'].shift(-5) / df['close'] - 1
            df['future_volatility'] = df['returns'].shift(-5).rolling(5).std()
            
            # Clean data
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            
            # Remove infinite values
            df.replace([np.inf, -np.inf], 0, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to add AI features: {e}")
            return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(period).mean()
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2):
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(period).mean()
        rolling_std = df['close'].rolling(period).std()
        upper = sma + (rolling_std * std)
        lower = sma - (rolling_std * std)
        return upper, lower
    
    def _validate_ai_data_quality(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Validate data quality for AI training"""
        try:
            issues = []
            
            # Check minimum data requirement
            if len(df) < self.MIN_BARS_M5:
                issues.append(f"Insufficient data: {len(df)} bars (need {self.MIN_BARS_M5})")
            
            # Check for missing values
            missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if missing_pct > 0.05:  # 5% threshold
                issues.append(f"Too many missing values: {missing_pct:.1%}")
            
            # Check for data staleness
            last_update = df.index.max()
            staleness = datetime.now() - last_update.to_pydatetime()
            if staleness.total_seconds() > 3600:  # 1 hour
                issues.append(f"Data is stale: {staleness}")
            
            return {
                'status': 'GOOD' if not issues else 'WARNING',
                'issues': issues,
                'data_points': len(df),
                'last_update': last_update
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'issues': [f"Validation failed: {e}"],
                'data_points': 0,
                'last_update': None
            }