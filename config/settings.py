"""
Configuration file for ProQuants Professional Trading System
"""

# MetaTrader5 Connection Settings
MT5_CONFIG = {
    "login": 31833954,
    "password": "@Dmc65070*",
    "server": "Deriv-Demo"
}

# Trading Symbols
TRADING_SYMBOLS = [
    "Volatility 75 Index",
    "Volatility 25 Index", 
    "Volatility 75 (1s) Index"
]

# Risk Management Settings
RISK_MANAGEMENT = {
    "max_risk_per_trade": 0.02,      # 2% per trade
    "max_daily_loss": 0.10,          # 10% daily loss limit
    "max_positions": 5,              # Maximum concurrent positions
    "stop_loss_pips": 50,            # Default stop loss in pips
    "take_profit_pips": 100,         # Default take profit in pips
    "trailing_stop": True,           # Enable trailing stop
    "trailing_stop_distance": 30     # Trailing stop distance in pips
}

# CREAM Strategy Configuration
CREAM_STRATEGY = {
    "timeframe": "M1",
    "lookback_periods": 20,
    "signal_threshold": 0.7,
    
    # Clean component
    "clean_ma_period": 14,
    
    # Range component  
    "range_bb_period": 20,
    "range_bb_std": 2.0,
    
    # Easy component
    "easy_rsi_period": 14,
    "easy_rsi_oversold": 30,
    "easy_rsi_overbought": 70,
    
    # Accuracy component
    "accuracy_success_rate": 0.65,
    
    # Momentum component
    "momentum_period": 10,
    "volume_threshold": 1.2
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    "window_size": "1920x1080",
    "theme": "dark",
    "update_interval": 1000,  # milliseconds
    "max_console_lines": 1000,
    "auto_scroll": True,
    "professional_mode": True
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": "logs/proquants.log",
    "max_file_size": "10MB",
    "backup_count": 5
}

# Data Storage
DATA_CONFIG = {
    "cache_enabled": True,
    "cache_directory": "data/cache",
    "historical_data_days": 30,
    "offline_data_file": "data/offline_market_data.json"
}

# Performance Settings
PERFORMANCE_CONFIG = {
    "max_threads": 4,
    "data_update_interval": 10,  # seconds
    "analysis_update_interval": 5,  # seconds
    "ui_refresh_rate": 60  # FPS
}
