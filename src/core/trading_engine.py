"""
ProQuants Core Trading Engine
Professional MT5 integration with advanced error handling
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import json
import os
from typing import Dict, List, Optional, Tuple
import logging

class TradingEngine:
    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.connected = False
        self.account_info = {}
        self.positions = []
        self.market_data = {}
        self.offline_mode = False
        self.cached_data = {}
        
        # Setup logging
        self.setup_logging()
        
        # Initialize MT5 connection
        self.initialize_mt5()
        
    def setup_logging(self):
        """Setup professional logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading_engine.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_path: str) -> Dict:
        """Load trading configuration"""
        default_config = {
            "mt5": {
                "login": 31833954,
                "password": "@Dmc65070*",
                "server": "Deriv-Demo"
            },
            "symbols": ["Volatility 75 Index", "Volatility 25 Index", "Volatility 75 (1s) Index"],
            "risk_management": {
                "max_risk_per_trade": 0.02,  # 2%
                "max_daily_loss": 0.10,      # 10%
                "max_positions": 5
            },
            "cream_strategy": {
                "timeframe": "M1",
                "lookback_periods": 20,
                "signal_threshold": 0.7
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}")
                
        return default_config
        
    def initialize_mt5(self) -> bool:
        """Initialize MetaTrader5 connection"""
        try:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return False
                
            # Login to account
            login_result = mt5.login(
                login=self.config["mt5"]["login"],
                password=self.config["mt5"]["password"],
                server=self.config["mt5"]["server"]
            )
            
            if login_result:
                self.connected = True
                self.account_info = mt5.account_info()._asdict()
                self.logger.info(f"Connected to MT5 account: {self.account_info['login']}")
                return True
            else:
                self.logger.error("MT5 login failed")
                return False
                
        except Exception as e:
            self.logger.error(f"MT5 initialization error: {e}")
            self.offline_mode = True
            return False
            
    def get_account_info(self) -> Dict:
        """Get current account information"""
        if self.connected and not self.offline_mode:
            try:
                account = mt5.account_info()
                if account:
                    self.account_info = account._asdict()
                    return self.account_info
            except Exception as e:
                self.logger.error(f"Failed to get account info: {e}")
                
        return self.account_info
        
    def get_positions(self) -> List[Dict]:
        """Get current open positions"""
        if self.connected and not self.offline_mode:
            try:
                positions = mt5.positions_get()
                if positions:
                    self.positions = [pos._asdict() for pos in positions]
                    return self.positions
            except Exception as e:
                self.logger.error(f"Failed to get positions: {e}")
                
        return self.positions
        
    def get_market_data(self, symbol: str, timeframe: str = "M1", bars: int = 100) -> Optional[pd.DataFrame]:
        """Get market data for a symbol"""
        if self.offline_mode:
            return self.get_cached_market_data(symbol, timeframe, bars)
            
        try:
            # Convert timeframe
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
            
            timeframe_mt5 = tf_map.get(timeframe, mt5.TIMEFRAME_M1)
            
            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, bars)
            
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Cache the data
                cache_key = f"{symbol}_{timeframe}_{bars}"
                self.cached_data[cache_key] = df.copy()
                
                return df
            else:
                self.logger.warning(f"No data received for {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            return self.get_cached_market_data(symbol, timeframe, bars)
            
    def get_cached_market_data(self, symbol: str, timeframe: str = "M1", bars: int = 100) -> Optional[pd.DataFrame]:
        """Get cached market data for offline mode"""
        cache_key = f"{symbol}_{timeframe}_{bars}"
        
        if cache_key in self.cached_data:
            return self.cached_data[cache_key].copy()
            
        # Generate synthetic data for demonstration
        self.logger.info(f"Generating synthetic data for {symbol} (offline mode)")
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(minutes=bars),
            periods=bars,
            freq='1min'
        )
        
        # Simple random walk for demonstration
        np.random.seed(42)  # For reproducible results
        base_price = 1000.0
        returns = np.random.normal(0, 0.001, bars)
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.001, bars))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.001, bars))),
            'close': prices,
            'tick_volume': np.random.randint(100, 1000, bars),
            'spread': np.random.randint(1, 5, bars),
            'real_volume': np.random.randint(1000, 10000, bars)
        })
        
        # Ensure high >= open, close and low <= open, close
        df['high'] = np.maximum.reduce([df['open'], df['close'], df['high']])
        df['low'] = np.minimum.reduce([df['open'], df['close'], df['low']])
        
        self.cached_data[cache_key] = df.copy()
        return df
        
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information"""
        if self.connected and not self.offline_mode:
            try:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info:
                    return symbol_info._asdict()
            except Exception as e:
                self.logger.error(f"Failed to get symbol info for {symbol}: {e}")
                
        # Return default symbol info for offline mode
        return {
            'name': symbol,
            'bid': 1000.0,
            'ask': 1000.5,
            'spread': 5,
            'digits': 5,
            'point': 0.00001,
            'trade_mode': 1,
            'min_lot': 0.01,
            'max_lot': 100.0,
            'lot_step': 0.01
        }
        
    def place_order(self, symbol: str, order_type: str, volume: float, price: float = None, 
                   sl: float = None, tp: float = None, comment: str = "") -> Dict:
        """Place a trading order"""
        if self.offline_mode:
            return self.simulate_order(symbol, order_type, volume, price, sl, tp, comment)
            
        try:
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return {"retcode": 1, "comment": "Symbol not found"}
                
            # Prepare order request
            order_type_map = {
                "BUY": mt5.ORDER_TYPE_BUY,
                "SELL": mt5.ORDER_TYPE_SELL,
                "BUY_LIMIT": mt5.ORDER_TYPE_BUY_LIMIT,
                "SELL_LIMIT": mt5.ORDER_TYPE_SELL_LIMIT,
                "BUY_STOP": mt5.ORDER_TYPE_BUY_STOP,
                "SELL_STOP": mt5.ORDER_TYPE_SELL_STOP
            }
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type_map.get(order_type, mt5.ORDER_TYPE_BUY),
                "price": price if price else (symbol_info['ask'] if 'BUY' in order_type else symbol_info['bid']),
                "deviation": 20,
                "magic": 12345,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            # Add SL and TP if provided
            if sl:
                request["sl"] = sl
            if tp:
                request["tp"] = tp
                
            # Send order
            result = mt5.order_send(request)
            if result:
                return result._asdict()
            else:
                return {"retcode": 1, "comment": "Order failed"}
                
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return {"retcode": 1, "comment": str(e)}
            
    def simulate_order(self, symbol: str, order_type: str, volume: float, price: float = None,
                      sl: float = None, tp: float = None, comment: str = "") -> Dict:
        """Simulate order placement for offline mode"""
        self.logger.info(f"SIMULATED ORDER: {order_type} {volume} {symbol} at {price}")
        
        return {
            "retcode": mt5.TRADE_RETCODE_DONE,
            "deal": 12345,
            "order": 67890,
            "volume": volume,
            "price": price if price else 1000.0,
            "bid": 999.95,
            "ask": 1000.05,
            "comment": f"SIMULATED: {comment}",
            "request_id": int(time.time()),
            "retcode_external": 0
        }
        
    def close_position(self, ticket: int) -> Dict:
        """Close an open position"""
        if self.offline_mode:
            self.logger.info(f"SIMULATED: Closing position {ticket}")
            return {"retcode": mt5.TRADE_RETCODE_DONE, "comment": "Position closed (simulated)"}
            
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {"retcode": 1, "comment": "Position not found"}
                
            position = position[0]
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
                "deviation": 20,
                "magic": 12345,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            result = mt5.order_send(request)
            if result:
                return result._asdict()
            else:
                return {"retcode": 1, "comment": "Close failed"}
                
        except Exception as e:
            self.logger.error(f"Failed to close position {ticket}: {e}")
            return {"retcode": 1, "comment": str(e)}
            
    def toggle_offline_mode(self):
        """Toggle between online and offline mode"""
        self.offline_mode = not self.offline_mode
        mode = "OFFLINE" if self.offline_mode else "ONLINE"
        self.logger.info(f"Trading engine switched to {mode} mode")
        
    def is_market_open(self, symbol: str) -> bool:
        """Check if market is open for a symbol"""
        if self.offline_mode:
            return True  # Always open in offline mode
            
        try:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                return symbol_info.trade_mode != 0
        except Exception as e:
            self.logger.error(f"Failed to check market status for {symbol}: {e}")
            
        return True  # Default to open
        
    def get_last_error(self) -> str:
        """Get the last MT5 error"""
        if self.connected and not self.offline_mode:
            try:
                return mt5.last_error()
            except:
                pass
        return "Offline mode or connection error"
        
    def shutdown(self):
        """Shutdown the trading engine"""
        if self.connected:
            try:
                mt5.shutdown()
                self.logger.info("MT5 connection closed")
            except Exception as e:
                self.logger.error(f"Error shutting down MT5: {e}")
                
        self.connected = False
