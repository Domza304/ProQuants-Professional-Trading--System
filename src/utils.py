"""
Utility functions for ProQuants Professional Trading System
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

def setup_logging(log_level: str = "INFO", log_file: str = "logs/proquants.log") -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_file: str = "config/settings.py") -> Dict[str, Any]:
    """Load configuration from settings file"""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("settings", config_file)
        settings = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(settings)
        
        config = {}
        for attr in dir(settings):
            if not attr.startswith('_'):
                config[attr] = getattr(settings, attr)
                
        return config
        
    except Exception as e:
        print(f"Failed to load config: {e}")
        return {}

def save_json(data: Dict, filename: str):
    """Save data to JSON file"""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        print(f"Failed to save JSON to {filename}: {e}")

def load_json(filename: str) -> Optional[Dict]:
    """Load data from JSON file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Failed to load JSON from {filename}: {e}")
    return None

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount"""
    if currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def format_percentage(value: float) -> str:
    """Format percentage value"""
    return f"{value:.2%}"

def format_timestamp(timestamp: datetime = None) -> str:
    """Format timestamp for display"""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def validate_symbol(symbol: str) -> bool:
    """Validate trading symbol format"""
    # Basic validation for Deriv symbols
    valid_symbols = [
        "Volatility 75 Index",
        "Volatility 25 Index", 
        "Volatility 75 (1s) Index",
        "Volatility 10 Index",
        "Volatility 10 (1s) Index",
        "Volatility 50 Index",
        "Volatility 100 Index",
        "Volatility 100 (1s) Index",
        "Volatility 200 (1s) Index"
    ]
    return symbol in valid_symbols

def calculate_position_size(account_balance: float, risk_percent: float, 
                          stop_loss_pips: int, pip_value: float) -> float:
    """Calculate position size based on risk management"""
    try:
        risk_amount = account_balance * (risk_percent / 100)
        position_size = risk_amount / (stop_loss_pips * pip_value)
        return round(position_size, 2)
    except:
        return 0.01  # Default minimum

def get_system_info() -> Dict[str, str]:
    """Get system information"""
    import platform
    import sys
    
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "timestamp": format_timestamp()
    }

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available"""
    dependencies = {
        "pandas": False,
        "numpy": False,
        "MetaTrader5": False,
        "talib": False,
        "tkinter": False
    }
    
    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
            
    return dependencies

def create_backup(source_file: str, backup_dir: str = "backups"):
    """Create backup of important files"""
    try:
        import shutil
        
        os.makedirs(backup_dir, exist_ok=True)
        
        if os.path.exists(source_file):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(source_file)
            backup_name = f"{filename}_{timestamp}"
            backup_path = os.path.join(backup_dir, backup_name)
            
            shutil.copy2(source_file, backup_path)
            return backup_path
            
    except Exception as e:
        print(f"Backup failed: {e}")
        
    return None

class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.metrics = {}
        
    def record_metric(self, name: str, value: float):
        """Record a performance metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({
            "timestamp": datetime.now(),
            "value": value
        })
        
    def get_average(self, name: str, last_n: int = 10) -> float:
        """Get average of last N metrics"""
        if name in self.metrics and self.metrics[name]:
            recent = self.metrics[name][-last_n:]
            return sum(m["value"] for m in recent) / len(recent)
        return 0.0
        
    def get_uptime(self) -> str:
        """Get system uptime"""
        uptime = datetime.now() - self.start_time
        hours, remainder = divmod(uptime.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
    def get_summary(self) -> Dict:
        """Get performance summary"""
        return {
            "uptime": self.get_uptime(),
            "start_time": format_timestamp(self.start_time),
            "metrics_count": len(self.metrics),
            "total_recordings": sum(len(v) for v in self.metrics.values())
        }
