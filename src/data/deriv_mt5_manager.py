"""
ProQuants MT5 Data Manager - Deriv Synthetic Indices Integration
Proper symbol handling and real-time data collection for neural network training
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import logging
from typing import Dict, List, Optional, Tuple
import time

class DerivMT5Manager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        
        # MT5 connection details from .env
        self.mt5_login = int(os.getenv('MT5_LOGIN', '31833954'))
        self.mt5_password = os.getenv('MT5_PASSWORD', '@Dmc65070*')
        self.mt5_server = os.getenv('MT5_SERVER', 'Deriv-Demo')
        
        # Proper Deriv synthetic indices symbols (MT5 format)
        self.DERIV_SYMBOLS = {
            "Volatility 75 Index": "Volatility 75 Index",
            "Volatility 25 Index": "Volatility 25 Index", 
            "Volatility 75 (1s) Index": "Volatility 75 (1s) Index",
            "Volatility 100 Index": "Volatility 100 Index",
            "Volatility 10 Index": "Volatility 10 Index"
        }
        
        # Neural network training data requirements (minimum 12 hours)
        self.MIN_DATA_HOURS = 12
        self.MIN_DATA_POINTS = self.MIN_DATA_HOURS * 60 * 12  # 5-minute bars
        
        self.connected = False
        self.account_info = {}
        
    def initialize_mt5(self) -> bool:
        """Initialize MT5 connection with proper error handling"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login to account
            if not mt5.login(self.mt5_login, password=self.mt5_password, server=self.mt5_server):
                self.logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            
            # Get account info
            self.account_info = mt5.account_info()._asdict() if mt5.account_info() else {}
            
            self.connected = True
            self.logger.info(f"MT5 connected successfully - Account: {self.mt5_login}")
            
            # Verify Deriv symbols availability
            self._verify_deriv_symbols()
            
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 initialization error: {e}")
            return False
    
    def _verify_deriv_symbols(self):
        """Verify and get correct Deriv symbol names from MT5"""
        available_symbols = []
        
        # Get all symbols
        symbols = mt5.symbols_get()
        if symbols:
            symbol_names = [symbol.name for symbol in symbols]
            
            # Find Deriv volatility indices
            deriv_patterns = ["Volatility", "VIX", "V75", "V25", "V100", "V10"]
            
            for symbol_name in symbol_names:
                for pattern in deriv_patterns:
                    if pattern.lower() in symbol_name.lower():
                        available_symbols.append(symbol_name)
                        
            # Update our symbol list with actual MT5 names
            if available_symbols:
                self.logger.info(f"Found Deriv symbols: {available_symbols}")
                # Update DERIV_SYMBOLS with actual names from MT5
                for i, symbol in enumerate(available_symbols[:5]):  # Take first 5
                    key = f"Deriv_Symbol_{i+1}"
                    self.DERIV_SYMBOLS[key] = symbol
            else:
                self.logger.warning("No Deriv symbols found - using default names")
                
    def get_historical_data(self, symbol: str, timeframe: int = mt5.TIMEFRAME_M5, 
                          hours: int = None) -> Optional[pd.DataFrame]:
        """
        Get historical data for neural network training (minimum 12 hours)
        """
        if not self.connected:
            if not self.initialize_mt5():
                return None
                
        try:
            # Use minimum 12 hours or specified hours
            data_hours = max(hours or self.MIN_DATA_HOURS, self.MIN_DATA_HOURS)
            
            # Calculate date range
            utc_to = datetime.now()
            utc_from = utc_to - timedelta(hours=data_hours)
            
            # Get rates from MT5
            rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"No data received for {symbol}: {mt5.last_error()}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Check if we have enough data for neural network training
            if len(df) < self.MIN_DATA_POINTS:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df)} points "
                                 f"(need {self.MIN_DATA_POINTS} for proper neural network training)")
            
            self.logger.info(f"Retrieved {len(df)} data points for {symbol} "
                           f"({data_hours}h period)")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def get_real_time_data(self, symbol: str) -> Optional[Dict]:
        """Get real-time tick data for live trading"""
        if not self.connected:
            if not self.initialize_mt5():
                return None
                
        try:
            # Get current tick
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
                
            return {
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'time': datetime.fromtimestamp(tick.time),
                'volume': tick.volume if hasattr(tick, 'volume') else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting real-time data for {symbol}: {e}")
            return None
    
    def get_all_deriv_data(self, hours: int = None) -> Dict[str, pd.DataFrame]:
        """Get historical data for all Deriv instruments"""
        all_data = {}
        
        for display_name, mt5_symbol in self.DERIV_SYMBOLS.items():
            self.logger.info(f"Fetching data for {display_name} ({mt5_symbol})...")
            
            data = self.get_historical_data(mt5_symbol, hours=hours)
            if data is not None:
                all_data[mt5_symbol] = data
            else:
                self.logger.warning(f"Failed to get data for {mt5_symbol}")
                
        return all_data
    
    def get_account_info(self) -> Dict:
        """Get current account information"""
        if not self.connected:
            if not self.initialize_mt5():
                return {}
                
        try:
            account = mt5.account_info()
            if account:
                return {
                    'login': account.login,
                    'server': account.server,
                    'name': account.name,
                    'company': account.company,
                    'currency': account.currency,
                    'balance': account.balance,
                    'equity': account.equity,
                    'margin': account.margin,
                    'free_margin': account.margin_free,
                    'margin_level': account.margin_level if account.margin > 0 else 0,
                    'profit': account.profit
                }
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_open_positions(self) -> List[Dict]:
        """Get current open positions"""
        if not self.connected:
            return []
            
        try:
            positions = mt5.positions_get()
            if positions:
                return [
                    {
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'type': 'BUY' if pos.type == 0 else 'SELL',
                        'volume': pos.volume,
                        'price_open': pos.price_open,
                        'price_current': pos.price_current,
                        'profit': pos.profit,
                        'time': datetime.fromtimestamp(pos.time)
                    }
                    for pos in positions
                ]
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists and is tradeable"""
        if not self.connected:
            return False
            
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False
                
            # Check if symbol is visible and tradeable
            if not symbol_info.visible:
                # Try to enable symbol
                if not mt5.symbol_select(symbol, True):
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed symbol information"""
        if not self.connected:
            return None
            
        try:
            info = mt5.symbol_info(symbol)
            if info:
                return {
                    'name': info.name,
                    'description': info.description,
                    'currency_base': info.currency_base,
                    'currency_profit': info.currency_profit,
                    'point': info.point,
                    'digits': info.digits,
                    'spread': info.spread,
                    'trade_contract_size': info.trade_contract_size,
                    'minimum_volume': info.volume_min,
                    'maximum_volume': info.volume_max,
                    'volume_step': info.volume_step,
                    'margin_initial': info.margin_initial,
                    'session_deals': info.session_deals,
                    'session_buy_orders': info.session_buy_orders,
                    'session_sell_orders': info.session_sell_orders
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {e}")
            return None
    
    def prepare_neural_network_dataset(self, symbol: str, hours: int = 24) -> Optional[Dict]:
        """
        Prepare comprehensive dataset for neural network training
        Includes technical indicators and market microstructure data
        """
        data = self.get_historical_data(symbol, hours=hours)
        if data is None or len(data) < self.MIN_DATA_POINTS:
            return None
            
        try:
            # Calculate technical features for neural network
            dataset = data.copy()
            
            # Price-based features
            dataset['returns'] = dataset['close'].pct_change()
            dataset['log_returns'] = np.log(dataset['close'] / dataset['close'].shift(1))
            dataset['volatility'] = dataset['returns'].rolling(20).std()
            
            # OHLC relationships
            dataset['hl_ratio'] = (dataset['high'] - dataset['low']) / dataset['close']
            dataset['oc_ratio'] = (dataset['open'] - dataset['close']) / dataset['close']
            dataset['upper_shadow'] = (dataset['high'] - np.maximum(dataset['open'], dataset['close'])) / dataset['close']
            dataset['lower_shadow'] = (np.minimum(dataset['open'], dataset['close']) - dataset['low']) / dataset['close']
            
            # Volume features
            dataset['volume_ma'] = dataset['tick_volume'].rolling(20).mean()
            dataset['volume_ratio'] = dataset['tick_volume'] / dataset['volume_ma']
            
            # Moving averages (multiple timeframes)
            for period in [5, 10, 20, 50]:
                dataset[f'ma_{period}'] = dataset['close'].rolling(period).mean()
                dataset[f'price_to_ma_{period}'] = dataset['close'] / dataset[f'ma_{period}'] - 1
            
            # RSI calculation
            delta = dataset['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            dataset['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            dataset['bb_middle'] = dataset['close'].rolling(bb_period).mean()
            bb_std_dev = dataset['close'].rolling(bb_period).std()
            dataset['bb_upper'] = dataset['bb_middle'] + (bb_std_dev * bb_std)
            dataset['bb_lower'] = dataset['bb_middle'] - (bb_std_dev * bb_std)
            dataset['bb_position'] = (dataset['close'] - dataset['bb_lower']) / (dataset['bb_upper'] - dataset['bb_lower'])
            
            # Time-based features
            dataset['hour'] = dataset.index.hour
            dataset['day_of_week'] = dataset.index.dayofweek
            dataset['is_london_session'] = ((dataset.index.hour >= 8) & (dataset.index.hour <= 16)).astype(int)
            dataset['is_ny_session'] = ((dataset.index.hour >= 13) & (dataset.index.hour <= 21)).astype(int)
            
            # Remove NaN values
            dataset = dataset.dropna()
            
            return {
                'symbol': symbol,
                'data': dataset,
                'features_count': len(dataset.columns),
                'data_points': len(dataset),
                'data_quality': 'EXCELLENT' if len(dataset) >= self.MIN_DATA_POINTS else 'INSUFFICIENT',
                'timeframe': '5M',
                'hours_covered': hours,
                'neural_network_ready': len(dataset) >= self.MIN_DATA_POINTS
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing neural network dataset for {symbol}: {e}")
            return None
    
    def shutdown(self):
        """Properly shutdown MT5 connection"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            self.logger.info("MT5 connection closed")
    
    def __del__(self):
        """Ensure proper cleanup"""
        self.shutdown()
