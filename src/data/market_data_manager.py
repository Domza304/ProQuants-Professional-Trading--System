"""
Data Manager for ProQuants Professional Trading System
Handles market data caching, storage, and retrieval
"""

import json
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

class MarketDataManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.cache_file = os.path.join(data_dir, "market_cache.json")
        self.offline_file = os.path.join(data_dir, "offline_data.json")
        self.logger = logging.getLogger(__name__)
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing cache
        self.cache = self.load_cache()
        
    def load_cache(self) -> Dict:
        """Load market data cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load cache: {e}")
        return {}
        
    def save_cache(self):
        """Save market data cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
            
    def store_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """Store market data in cache"""
        try:
            cache_key = f"{symbol}_{timeframe}"
            
            # Convert DataFrame to dict for JSON storage
            data_dict = {
                'timestamp': datetime.now().isoformat(),
                'data': data.to_dict('records'),
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            self.cache[cache_key] = data_dict
            self.save_cache()
            
        except Exception as e:
            self.logger.error(f"Failed to store market data: {e}")
            
    def get_cached_data(self, symbol: str, timeframe: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """Retrieve cached market data"""
        try:
            cache_key = f"{symbol}_{timeframe}"
            
            if cache_key in self.cache:
                cached_item = self.cache[cache_key]
                
                # Check if data is not too old
                cached_time = datetime.fromisoformat(cached_item['timestamp'])
                age = datetime.now() - cached_time
                
                if age.total_seconds() / 3600 < max_age_hours:
                    # Convert back to DataFrame
                    df = pd.DataFrame(cached_item['data'])
                    if 'time' in df.columns:
                        df['time'] = pd.to_datetime(df['time'])
                    return df
                    
        except Exception as e:
            self.logger.error(f"Failed to get cached data: {e}")
            
        return None
        
    def cleanup_old_cache(self, max_age_days: int = 7):
        """Remove old cache entries"""
        try:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            
            keys_to_remove = []
            for key, value in self.cache.items():
                try:
                    cached_time = datetime.fromisoformat(value['timestamp'])
                    if cached_time < cutoff_time:
                        keys_to_remove.append(key)
                except:
                    keys_to_remove.append(key)  # Remove invalid entries
                    
            for key in keys_to_remove:
                del self.cache[key]
                
            if keys_to_remove:
                self.save_cache()
                self.logger.info(f"Cleaned up {len(keys_to_remove)} old cache entries")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup cache: {e}")
            
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        stats = {
            'total_entries': len(self.cache),
            'symbols': set(),
            'timeframes': set(),
            'oldest_entry': None,
            'newest_entry': None
        }
        
        timestamps = []
        for value in self.cache.values():
            try:
                symbols = value.get('symbol', '')
                timeframe = value.get('timeframe', '')
                timestamp = value.get('timestamp', '')
                
                if symbols:
                    stats['symbols'].add(symbols)
                if timeframe:
                    stats['timeframes'].add(timeframe)
                if timestamp:
                    timestamps.append(datetime.fromisoformat(timestamp))
                    
            except:
                continue
                
        if timestamps:
            stats['oldest_entry'] = min(timestamps)
            stats['newest_entry'] = max(timestamps)
            
        stats['symbols'] = list(stats['symbols'])
        stats['timeframes'] = list(stats['timeframes'])
        
        return stats
