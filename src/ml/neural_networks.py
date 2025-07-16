"""
ProQuants Neural Network Enhanced Trading System
Deep Learning for Mathematical Certainty and Scientific Precision
Specialized for Deriv Synthetic Indices Pattern Recognition
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import pickle
import os

try:
    # TensorFlow/Keras for deep learning
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    NEURAL_NETWORKS_AVAILABLE = True
except ImportError:
    NEURAL_NETWORKS_AVAILABLE = False
    tf = None

class DerivNeuralNetworkSystem:
    """
    Advanced Neural Network System for Deriv Synthetic Indices
    Uses multiple neural network architectures for mathematical precision
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if not NEURAL_NETWORKS_AVAILABLE:
            self.logger.warning("TensorFlow not available - Neural Networks disabled")
            return
            
        # Neural Network Models for each Deriv instrument
        self.models = {
            'price_direction': {},      # LSTM for price direction prediction
            'volatility_regime': {},    # CNN for volatility regime classification
            'entry_timing': {},         # GRU for optimal entry timing
            'risk_assessment': {},      # Dense NN for risk evaluation
            'manipulation_detection': {} # Autoencoder for anomaly detection
        }
        
        # Scalers for data normalization
        self.scalers = {}
        
        # Model training history
        self.training_history = {}
        
        # Neural network configuration
        self.config = {
            'sequence_length': 60,      # 5 hours of 5-minute data
            'prediction_horizon': 12,   # Predict 1 hour ahead
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'learning_rate': 0.001,
            'dropout_rate': 0.3
        }
        
        # Model performance metrics
        self.performance_metrics = {}
        
        # Mathematical certainty thresholds
        self.certainty_thresholds = {
            'high_certainty': 0.85,     # 85% model confidence
            'medium_certainty': 0.70,   # 70% model confidence
            'low_certainty': 0.55       # 55% model confidence
        }
        
    def create_price_direction_model(self, input_shape: Tuple) -> Model:
        """
        Create LSTM model for price direction prediction with mathematical precision
        """
        model = Sequential([
            # LSTM layers for temporal pattern recognition
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            # Dense layers for prediction
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            
            # Output layer: 3 classes (UP, DOWN, SIDEWAYS)
            Dense(3, activation='softmax', name='direction_output'),
            
            # Additional output for confidence score
            Dense(1, activation='sigmoid', name='confidence_output')
        ])
        
        # Compile with custom loss function for mathematical certainty
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss={
                'direction_output': 'categorical_crossentropy',
                'confidence_output': 'binary_crossentropy'
            },
            loss_weights={
                'direction_output': 0.8,
                'confidence_output': 0.2
            },
            metrics=['accuracy']
        )
        
        return model
    
    def create_volatility_regime_model(self, input_shape: Tuple) -> Model:
        """
        Create CNN model for volatility regime classification
        """
        model = Sequential([
            # Convolutional layers for pattern detection
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(self.config['dropout_rate']),
            
            Conv1D(128, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(self.config['dropout_rate']),
            
            Conv1D(64, 3, activation='relu'),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            # Flatten and dense layers
            Flatten(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            
            # Output: 4 volatility regimes (LOW, MEDIUM, HIGH, EXTREME)
            Dense(4, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_entry_timing_model(self, input_shape: Tuple) -> Model:
        """
        Create GRU model for optimal entry timing prediction
        """
        model = Sequential([
            # GRU layers for temporal modeling
            GRU(96, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            GRU(48, return_sequences=True),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            GRU(24, return_sequences=False),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            # Dense layers
            Dense(48, activation='relu'),
            BatchNormalization(),
            Dropout(self.config['dropout_rate']),
            
            Dense(24, activation='relu'),
            Dense(12, activation='relu'),
            
            # Output: Entry timing score (0-1)
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_manipulation_detection_model(self, input_shape: Tuple) -> Model:
        """
        Create Autoencoder for manipulation pattern detection
        """
        # Encoder
        input_layer = layers.Input(shape=input_shape)
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(16, activation='relu')(encoded)  # Bottleneck
        
        # Decoder
        decoded = layers.Dense(32, activation='relu')(encoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dense(input_shape[0], activation='linear')(decoded)
        
        # Autoencoder model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder
    
    def prepare_neural_network_data(self, data: pd.DataFrame, symbol: str) -> Dict:
        """
        Prepare data for neural network training with scientific precision
        """
        try:
            # Create feature matrix
            features = [
                'open', 'high', 'low', 'close', 'tick_volume',
                'returns', 'log_returns', 'volatility',
                'hl_ratio', 'oc_ratio', 'upper_shadow', 'lower_shadow',
                'volume_ratio', 'rsi', 'bb_position'
            ]
            
            # Add moving average features
            for period in [5, 10, 20, 50]:
                if f'price_to_ma_{period}' in data.columns:
                    features.append(f'price_to_ma_{period}')
            
            # Add time features
            time_features = ['hour', 'day_of_week', 'is_london_session', 'is_ny_session']
            features.extend([f for f in time_features if f in data.columns])
            
            # Select and clean features
            feature_data = data[features].dropna()
            
            if len(feature_data) < self.config['sequence_length'] * 2:
                return None
                
            # Normalize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_data)
            
            # Store scaler for later use
            self.scalers[symbol] = scaler
            
            # Create sequences for LSTM/GRU models
            X_sequences = []
            y_direction = []
            y_volatility = []
            y_entry_timing = []
            
            for i in range(self.config['sequence_length'], len(scaled_features) - self.config['prediction_horizon']):
                # Input sequence
                X_sequences.append(scaled_features[i-self.config['sequence_length']:i])
                
                # Price direction target (future price movement)
                current_price = feature_data.iloc[i]['close']
                future_price = feature_data.iloc[i + self.config['prediction_horizon']]['close']
                price_change = (future_price - current_price) / current_price
                
                # Classify direction (UP: >0.5%, DOWN: <-0.5%, SIDEWAYS: between)
                if price_change > 0.005:
                    y_direction.append([1, 0, 0])  # UP
                elif price_change < -0.005:
                    y_direction.append([0, 1, 0])  # DOWN
                else:
                    y_direction.append([0, 0, 1])  # SIDEWAYS
                
                # Volatility regime target
                future_volatility = feature_data.iloc[i:i + self.config['prediction_horizon']]['volatility'].mean()
                vol_percentile = np.percentile(feature_data['volatility'].dropna(), [25, 50, 75])
                
                if future_volatility <= vol_percentile[0]:
                    y_volatility.append([1, 0, 0, 0])  # LOW
                elif future_volatility <= vol_percentile[1]:
                    y_volatility.append([0, 1, 0, 0])  # MEDIUM
                elif future_volatility <= vol_percentile[2]:
                    y_volatility.append([0, 0, 1, 0])  # HIGH
                else:
                    y_volatility.append([0, 0, 0, 1])  # EXTREME
                
                # Entry timing target (1 if profitable entry, 0 otherwise)
                entry_profitable = 1 if abs(price_change) > 0.01 else 0  # 1% minimum move
                y_entry_timing.append(entry_profitable)
            
            return {
                'X_sequences': np.array(X_sequences),
                'y_direction': np.array(y_direction),
                'y_volatility': np.array(y_volatility),
                'y_entry_timing': np.array(y_entry_timing),
                'feature_names': features,
                'scaler': scaler,
                'data_points': len(X_sequences),
                'sequence_length': self.config['sequence_length'],
                'features_count': len(features)
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing neural network data for {symbol}: {e}")
            return None
    
    def train_models_for_symbol(self, symbol: str, prepared_data: Dict) -> Dict:
        """
        Train all neural network models for a specific Deriv symbol
        """
        if not NEURAL_NETWORKS_AVAILABLE:
            return {'status': 'neural_networks_not_available'}
            
        try:
            results = {}
            X = prepared_data['X_sequences']
            
            # Set up callbacks
            early_stopping = callbacks.EarlyStopping(
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
            
            # Train Price Direction Model
            self.logger.info(f"Training price direction model for {symbol}...")
            direction_model = self.create_price_direction_model((X.shape[1], X.shape[2]))
            
            direction_history = direction_model.fit(
                X, prepared_data['y_direction'],
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_split=self.config['validation_split'],
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            self.models['price_direction'][symbol] = direction_model
            results['price_direction'] = {
                'final_accuracy': direction_history.history['val_accuracy'][-1],
                'epochs_trained': len(direction_history.history['loss'])
            }
            
            # Train Volatility Regime Model
            self.logger.info(f"Training volatility regime model for {symbol}...")
            volatility_model = self.create_volatility_regime_model((X.shape[1], X.shape[2]))
            
            volatility_history = volatility_model.fit(
                X, prepared_data['y_volatility'],
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_split=self.config['validation_split'],
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            self.models['volatility_regime'][symbol] = volatility_model
            results['volatility_regime'] = {
                'final_accuracy': volatility_history.history['val_accuracy'][-1],
                'epochs_trained': len(volatility_history.history['loss'])
            }
            
            # Train Entry Timing Model
            self.logger.info(f"Training entry timing model for {symbol}...")
            timing_model = self.create_entry_timing_model((X.shape[1], X.shape[2]))
            
            timing_history = timing_model.fit(
                X, prepared_data['y_entry_timing'],
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_split=self.config['validation_split'],
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            self.models['entry_timing'][symbol] = timing_model
            results['entry_timing'] = {
                'final_accuracy': timing_history.history['val_accuracy'][-1],
                'epochs_trained': len(timing_history.history['loss'])
            }
            
            # Train Manipulation Detection Model (Autoencoder)
            self.logger.info(f"Training manipulation detection model for {symbol}...")
            manipulation_model = self.create_manipulation_detection_model((X.shape[2],))
            
            # Reshape data for autoencoder (use last timestep only)
            X_manipulation = X[:, -1, :]
            
            manipulation_history = manipulation_model.fit(
                X_manipulation, X_manipulation,
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_split=self.config['validation_split'],
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            self.models['manipulation_detection'][symbol] = manipulation_model
            results['manipulation_detection'] = {
                'final_loss': manipulation_history.history['val_loss'][-1],
                'epochs_trained': len(manipulation_history.history['loss'])
            }
            
            # Store training history
            self.training_history[symbol] = {
                'price_direction': direction_history.history,
                'volatility_regime': volatility_history.history,
                'entry_timing': timing_history.history,
                'manipulation_detection': manipulation_history.history,
                'training_timestamp': datetime.now()
            }
            
            self.logger.info(f"Neural network training completed for {symbol}")
            return {
                'status': 'success',
                'symbol': symbol,
                'models_trained': len(results),
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Error training neural networks for {symbol}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def predict_with_mathematical_certainty(self, symbol: str, current_data: np.ndarray) -> Dict:
        """
        Generate predictions with mathematical certainty scores
        """
        if not NEURAL_NETWORKS_AVAILABLE or symbol not in self.models['price_direction']:
            return {'status': 'models_not_available'}
            
        try:
            predictions = {}
            
            # Normalize current data
            if symbol in self.scalers:
                normalized_data = self.scalers[symbol].transform(current_data.reshape(1, -1))
                sequence_data = normalized_data.reshape(1, 1, -1)
            else:
                return {'status': 'scaler_not_available'}
            
            # Price Direction Prediction
            direction_pred = self.models['price_direction'][symbol].predict(sequence_data, verbose=0)
            direction_probs = direction_pred[0]
            direction_confidence = np.max(direction_probs)
            direction_class = np.argmax(direction_probs)
            direction_labels = ['UP', 'DOWN', 'SIDEWAYS']
            
            predictions['price_direction'] = {
                'prediction': direction_labels[direction_class],
                'confidence': float(direction_confidence),
                'probabilities': {
                    'UP': float(direction_probs[0]),
                    'DOWN': float(direction_probs[1]),
                    'SIDEWAYS': float(direction_probs[2])
                },
                'mathematical_certainty': self._calculate_certainty_level(direction_confidence)
            }
            
            # Volatility Regime Prediction
            volatility_pred = self.models['volatility_regime'][symbol].predict(sequence_data, verbose=0)
            volatility_probs = volatility_pred[0]
            volatility_confidence = np.max(volatility_probs)
            volatility_class = np.argmax(volatility_probs)
            volatility_labels = ['LOW', 'MEDIUM', 'HIGH', 'EXTREME']
            
            predictions['volatility_regime'] = {
                'prediction': volatility_labels[volatility_class],
                'confidence': float(volatility_confidence),
                'probabilities': {
                    'LOW': float(volatility_probs[0]),
                    'MEDIUM': float(volatility_probs[1]),
                    'HIGH': float(volatility_probs[2]),
                    'EXTREME': float(volatility_probs[3])
                },
                'mathematical_certainty': self._calculate_certainty_level(volatility_confidence)
            }
            
            # Entry Timing Prediction
            timing_pred = self.models['entry_timing'][symbol].predict(sequence_data, verbose=0)
            timing_score = float(timing_pred[0][0])
            
            predictions['entry_timing'] = {
                'score': timing_score,
                'recommendation': 'ENTER' if timing_score > 0.7 else 'WAIT',
                'confidence': timing_score,
                'mathematical_certainty': self._calculate_certainty_level(timing_score)
            }
            
            # Manipulation Detection
            manipulation_pred = self.models['manipulation_detection'][symbol].predict(
                current_data.reshape(1, -1), verbose=0
            )
            reconstruction_error = np.mean(np.square(current_data - manipulation_pred[0]))
            manipulation_threshold = 0.1  # Adjust based on training data
            manipulation_detected = reconstruction_error > manipulation_threshold
            
            predictions['manipulation_detection'] = {
                'manipulation_detected': bool(manipulation_detected),
                'reconstruction_error': float(reconstruction_error),
                'threshold': manipulation_threshold,
                'confidence': float(1.0 - min(reconstruction_error / manipulation_threshold, 1.0))
            }
            
            # Overall Mathematical Certainty
            certainty_scores = [
                predictions['price_direction']['confidence'],
                predictions['volatility_regime']['confidence'],
                predictions['entry_timing']['confidence']
            ]
            
            overall_certainty = np.mean(certainty_scores)
            
            return {
                'status': 'success',
                'symbol': symbol,
                'timestamp': datetime.now(),
                'predictions': predictions,
                'overall_mathematical_certainty': float(overall_certainty),
                'certainty_level': self._calculate_certainty_level(overall_certainty),
                'neural_network_consensus': self._calculate_consensus(predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating neural network predictions for {symbol}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_certainty_level(self, confidence: float) -> str:
        """Calculate mathematical certainty level"""
        if confidence >= self.certainty_thresholds['high_certainty']:
            return 'HIGH_CERTAINTY'
        elif confidence >= self.certainty_thresholds['medium_certainty']:
            return 'MEDIUM_CERTAINTY'
        elif confidence >= self.certainty_thresholds['low_certainty']:
            return 'LOW_CERTAINTY'
        else:
            return 'UNCERTAIN'
    
    def _calculate_consensus(self, predictions: Dict) -> Dict:
        """Calculate neural network consensus for trading decisions"""
        # Extract key signals
        direction = predictions['price_direction']['prediction']
        volatility = predictions['volatility_regime']['prediction']
        timing = predictions['entry_timing']['recommendation']
        manipulation = predictions['manipulation_detection']['manipulation_detected']
        
        # Calculate consensus score
        consensus_factors = []
        
        # Direction confidence
        consensus_factors.append(predictions['price_direction']['confidence'])
        
        # Volatility appropriateness (prefer medium volatility)
        vol_score = 0.8 if volatility == 'MEDIUM' else 0.6 if volatility in ['LOW', 'HIGH'] else 0.3
        consensus_factors.append(vol_score)
        
        # Timing appropriateness
        timing_score = 0.9 if timing == 'ENTER' else 0.1
        consensus_factors.append(timing_score)
        
        # Manipulation penalty
        manipulation_penalty = 0.5 if manipulation else 1.0
        
        overall_consensus = np.mean(consensus_factors) * manipulation_penalty
        
        return {
            'consensus_score': float(overall_consensus),
            'recommendation': 'STRONG_BUY' if direction == 'UP' and overall_consensus > 0.8 else
                           'STRONG_SELL' if direction == 'DOWN' and overall_consensus > 0.8 else
                           'WEAK_BUY' if direction == 'UP' and overall_consensus > 0.6 else
                           'WEAK_SELL' if direction == 'DOWN' and overall_consensus > 0.6 else
                           'HOLD',
            'factors': {
                'direction_confidence': predictions['price_direction']['confidence'],
                'volatility_score': vol_score,
                'timing_score': timing_score,
                'manipulation_penalty': manipulation_penalty
            }
        }
    
    def save_models(self, symbol: str, base_path: str):
        """Save trained models for a symbol"""
        if not NEURAL_NETWORKS_AVAILABLE:
            return False
            
        try:
            symbol_dir = os.path.join(base_path, 'neural_networks', symbol)
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Save each model type
            for model_type, models in self.models.items():
                if symbol in models:
                    model_path = os.path.join(symbol_dir, f'{model_type}_model.h5')
                    models[symbol].save(model_path)
            
            # Save scaler
            if symbol in self.scalers:
                scaler_path = os.path.join(symbol_dir, 'scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[symbol], f)
            
            # Save training history
            if symbol in self.training_history:
                history_path = os.path.join(symbol_dir, 'training_history.pkl')
                with open(history_path, 'wb') as f:
                    pickle.dump(self.training_history[symbol], f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving models for {symbol}: {e}")
            return False
    
    def load_models(self, symbol: str, base_path: str):
        """Load trained models for a symbol"""
        if not NEURAL_NETWORKS_AVAILABLE:
            return False
            
        try:
            symbol_dir = os.path.join(base_path, 'neural_networks', symbol)
            
            if not os.path.exists(symbol_dir):
                return False
            
            # Load each model type
            for model_type in self.models.keys():
                model_path = os.path.join(symbol_dir, f'{model_type}_model.h5')
                if os.path.exists(model_path):
                    self.models[model_type][symbol] = keras.models.load_model(model_path)
            
            # Load scaler
            scaler_path = os.path.join(symbol_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers[symbol] = pickle.load(f)
            
            # Load training history
            history_path = os.path.join(symbol_dir, 'training_history.pkl')
            if os.path.exists(history_path):
                with open(history_path, 'rb') as f:
                    self.training_history[symbol] = pickle.load(f)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models for {symbol}: {e}")
            return False
    
    def get_model_performance_report(self, symbol: str) -> Dict:
        """Generate comprehensive model performance report"""
        if symbol not in self.training_history:
            return {'status': 'no_training_history'}
        
        history = self.training_history[symbol]
        
        report = {
            'symbol': symbol,
            'training_timestamp': history['training_timestamp'],
            'models_performance': {}
        }
        
        for model_type, model_history in history.items():
            if model_type == 'training_timestamp':
                continue
                
            if 'val_accuracy' in model_history:
                final_accuracy = model_history['val_accuracy'][-1]
                best_accuracy = max(model_history['val_accuracy'])
                epochs_trained = len(model_history['val_accuracy'])
                
                report['models_performance'][model_type] = {
                    'final_accuracy': final_accuracy,
                    'best_accuracy': best_accuracy,
                    'epochs_trained': epochs_trained,
                    'mathematical_certainty': self._calculate_certainty_level(final_accuracy)
                }
        
        return report
