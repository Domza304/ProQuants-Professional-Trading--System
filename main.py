"""
ProQuants Professional Trading System - Main Application
Enhanced with Neural Networks for Mathematical Certainty
Integrates the professional dashboard with trading engine and CREAM strategy
"""

import sys
import os
import threading
import time
import traceback
from datetime import datetime
from typing import Dict
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.gui.professional_dashboard import ProfessionalDashboard
from src.core.trading_engine import TradingEngine
from src.strategies.cream_strategy import CreamStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ProQuantsProfessionalSystem:
    def __init__(self):
        self.dashboard = None
        self.trading_engine = None
        self.cream_strategy = None
        self.running = False
        self.update_thread = None
        self.neural_network_status = "INITIALIZING"
        
    def initialize(self):
        """Initialize all system components with neural network enhancement"""
        try:
            logger.info("Initializing ProQuants Professional System with Neural Networks...")
            
            # Create logs directory if it doesn't exist
            os.makedirs('logs', exist_ok=True)
            os.makedirs('data', exist_ok=True)
            
            # Initialize components
            logger.info("Creating Professional Dashboard...")
            self.dashboard = ProfessionalDashboard()
            
            logger.info("Initializing Trading Engine...")
            self.trading_engine = TradingEngine()
            
            logger.info("Loading CREAM Strategy with Neural Networks...")
            self.cream_strategy = CreamStrategy()
            
            # Check neural network availability
            self.check_neural_network_status()
            
            # Connect dashboard to trading engine
            self.connect_dashboard_to_engine()
            
            # Start data update thread
            self.start_update_thread()
            
            logger.info("ProQuants Professional System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            traceback.print_exc()
            return False
    
    def check_neural_network_status(self):
        """Check neural network availability and status"""
        try:
            # Check if neural networks are available
            neural_available = hasattr(self.cream_strategy, 'neural_network') and self.cream_strategy.neural_network is not None
            
            if neural_available:
                self.neural_network_status = "AVAILABLE"
                logger.info("Neural Networks: AVAILABLE - Mathematical certainty enabled")
            else:
                self.neural_network_status = "UNAVAILABLE"
                logger.warning("Neural Networks: UNAVAILABLE - Running with traditional analysis")
                
            # Update dashboard status if available
            if self.dashboard:
                self.dashboard.update_neural_network_status(self.neural_network_status)
                
        except Exception as e:
            logger.error(f"Error checking neural network status: {e}")
            self.neural_network_status = "ERROR"
            
    def connect_dashboard_to_engine(self):
        """Connect dashboard controls to trading engine functions"""
        # Override dashboard methods to use trading engine
        original_start = self.dashboard.start_trading
        original_stop = self.dashboard.stop_trading
        original_toggle_offline = self.dashboard.toggle_offline_mode
        
        def enhanced_start_trading():
            """Enhanced start trading with engine integration"""
            try:
                if not self.dashboard.trading_active:
                    self.dashboard.trading_active = True
                    self.running = True
                    
                    # Update UI
                    self.dashboard.log_to_console("Initializing trading engine...", "SYSTEM")
                    self.dashboard.connection_status.config(
                        text="● CONNECTING...", 
                        fg=self.dashboard.colors['accent_gold']
                    )
                    
                    # Connect to MT5
                    if self.trading_engine.connected or self.trading_engine.offline_mode:
                        self.dashboard.connection_status.config(
                            text="● CONNECTED", 
                            fg=self.dashboard.colors['accent_green']
                        )
                        self.dashboard.log_to_console("Trading engine connected successfully", "SUCCESS")
                    else:
                        self.dashboard.connection_status.config(
                            text="● OFFLINE MODE", 
                            fg=self.dashboard.colors['accent_gold']
                        )
                        self.dashboard.log_to_console("Running in offline mode", "WARNING")
                        
                    # Update button states
                    self.dashboard.start_btn.config(state='disabled')
                    self.dashboard.stop_btn.config(state='normal')
                    
                    # Start enhanced trading loop
                    threading.Thread(target=self.enhanced_trading_loop, daemon=True).start()
                    
            except Exception as e:
                self.dashboard.log_to_console(f"Failed to start trading: {e}", "ERROR")
                
        def enhanced_stop_trading():
            """Enhanced stop trading with engine cleanup"""
            try:
                if self.dashboard.trading_active:
                    self.dashboard.trading_active = False
                    self.running = False
                    
                    self.dashboard.log_to_console("Stopping trading engine...", "SYSTEM")
                    self.dashboard.connection_status.config(
                        text="● DISCONNECTED", 
                        fg=self.dashboard.colors['accent_red']
                    )
                    
                    # Update button states
                    self.dashboard.start_btn.config(state='normal')
                    self.dashboard.stop_btn.config(state='disabled')
                    
                    self.dashboard.log_to_console("Trading engine stopped", "WARNING")
                    
            except Exception as e:
                self.dashboard.log_to_console(f"Error stopping trading: {e}", "ERROR")
                
        def enhanced_toggle_offline():
            """Enhanced offline mode toggle"""
            try:
                self.trading_engine.toggle_offline_mode()
                self.dashboard.offline_mode = self.trading_engine.offline_mode
                
                mode_text = "ENABLED" if self.dashboard.offline_mode else "DISABLED"
                self.dashboard.log_to_console(f"Offline mode {mode_text}", "SYSTEM")
                
                if self.dashboard.offline_mode:
                    self.dashboard.offline_btn.config(
                        bg=self.dashboard.colors['accent_red'], 
                        text="ONLINE MODE"
                    )
                    self.dashboard.connection_status.config(
                        text="● OFFLINE MODE", 
                        fg=self.dashboard.colors['accent_gold']
                    )
                else:
                    self.dashboard.offline_btn.config(
                        bg=self.dashboard.colors['accent_gold'], 
                        text="OFFLINE MODE"
                    )
                    if self.trading_engine.connected:
                        self.dashboard.connection_status.config(
                            text="● CONNECTED", 
                            fg=self.dashboard.colors['accent_green']
                        )
                        
            except Exception as e:
                self.dashboard.log_to_console(f"Error toggling offline mode: {e}", "ERROR")
                
        # Replace dashboard methods
        self.dashboard.start_trading = enhanced_start_trading
        self.dashboard.stop_trading = enhanced_stop_trading
        self.dashboard.toggle_offline_mode = enhanced_toggle_offline
        
    def enhanced_trading_loop(self):
        """Enhanced trading loop with CREAM strategy integration"""
        symbols = ["Volatility 75 Index", "Volatility 25 Index", "Volatility 75 (1s) Index"]
        
        while self.running and self.dashboard.trading_active:
            try:
                self.dashboard.log_to_console("Running CREAM strategy analysis...", "TRADE")
                
                for symbol in symbols:
                    if not self.running:
                        break
                        
                    # Get market data
                    market_data = self.trading_engine.get_market_data(symbol, "M1", 100)
                    
                    if market_data is not None:
                        # Run CREAM analysis
                        analysis = self.cream_strategy.analyze(market_data, symbol)
                        
                        # Update dashboard with analysis results
                        self.update_dashboard_with_analysis(symbol, analysis)
                        
                        # Log significant signals
                        if analysis['recommended_action'] != 'WAIT':
                            signal_msg = f"{symbol}: {analysis['overall_signal']} - {analysis['recommended_action']}"
                            self.dashboard.log_to_console(signal_msg, "TRADE")
                            
                        # Auto-trading logic (if enabled)
                        if self.dashboard.auto_trade_var.get():
                            self.execute_auto_trade(symbol, analysis)
                            
                    else:
                        self.dashboard.log_to_console(f"No data available for {symbol}", "WARNING")
                        
                # Update account information
                self.update_account_info()
                
                # Update positions
                self.update_positions()
                
                # Wait before next iteration
                time.sleep(10)  # 10-second intervals
                
            except Exception as e:
                self.dashboard.log_to_console(f"Trading loop error: {e}", "ERROR")
                time.sleep(5)
                
        self.dashboard.log_to_console("Enhanced trading loop terminated", "SYSTEM")
        
    def update_dashboard_with_analysis(self, symbol: str, analysis: Dict):
        """Update dashboard with CREAM analysis results"""
        try:
            # Update strategy panel
            cream_components = ['clean', 'range', 'easy', 'accuracy', 'momentum']
            
            for component in cream_components:
                if component in analysis and component in self.dashboard.strategy_labels:
                    component_data = analysis[component]
                    signal = component_data.get('signal', 'UNKNOWN')
                    
                    # Determine color based on signal
                    if 'STRONG' in signal or signal in ['CLEAN', 'EASY', 'HIGH']:
                        color = self.dashboard.colors['accent_green']
                    elif 'MODERATE' in signal or signal in ['MODERATELY_CLEAN', 'MODERATE', 'MEDIUM']:
                        color = self.dashboard.colors['accent_gold']
                    elif 'WEAK' in signal or signal in ['NOISY', 'DIFFICULT', 'LOW']:
                        color = self.dashboard.colors['accent_red']
                    else:
                        color = self.dashboard.colors['text_muted']
                        
                    # Update label
                    label_key = f"{component.title()} Signal:"
                    if label_key in self.dashboard.strategy_labels:
                        self.dashboard.strategy_labels[label_key].config(text=signal, fg=color)
                        
            # Update market data if available
            self.update_market_data(symbol, analysis)
            
        except Exception as e:
            self.dashboard.log_to_console(f"Error updating dashboard analysis: {e}", "ERROR")
            
    def update_market_data(self, symbol: str, analysis: Dict):
        """Update market watch with current symbol data"""
        try:
            # Get current symbol info
            symbol_info = self.trading_engine.get_symbol_info(symbol)
            
            if symbol_info:
                bid = symbol_info['bid']
                ask = symbol_info['ask']
                spread = symbol_info['spread']
                
                # Calculate change (placeholder for now)
                change = "0.00%"
                
                # Find and update the row in market tree
                for item in self.dashboard.market_tree.get_children():
                    values = self.dashboard.market_tree.item(item, 'values')
                    if values[0] == symbol:
                        self.dashboard.market_tree.item(item, values=(
                            symbol, f"{bid:.5f}", f"{ask:.5f}", str(spread), change
                        ))
                        break
                        
        except Exception as e:
            self.dashboard.log_to_console(f"Error updating market data: {e}", "ERROR")
            
    def update_account_info(self):
        """Update account information display"""
        try:
            account = self.trading_engine.get_account_info()
            
            if account:
                # Update account labels
                updates = {
                    "Balance:": f"${account.get('balance', 0.0):.2f}",
                    "Equity:": f"${account.get('equity', 0.0):.2f}",
                    "Margin:": f"${account.get('margin', 0.0):.2f}",
                    "Free Margin:": f"${account.get('margin_free', 0.0):.2f}",
                    "Margin Level:": f"{account.get('margin_level', 0.0):.2f}%",
                    "Profit/Loss:": f"${account.get('profit', 0.0):.2f}"
                }
                
                for label, value in updates.items():
                    if label in self.dashboard.account_labels:
                        self.dashboard.account_labels[label].config(text=value)
                        
        except Exception as e:
            self.dashboard.log_to_console(f"Error updating account info: {e}", "ERROR")
            
    def update_positions(self):
        """Update positions display"""
        try:
            positions = self.trading_engine.get_positions()
            
            # Clear existing positions
            for item in self.dashboard.positions_tree.get_children():
                self.dashboard.positions_tree.delete(item)
                
            # Add current positions
            for pos in positions:
                self.dashboard.positions_tree.insert('', 'end', values=(
                    pos.get('symbol', 'Unknown'),
                    'BUY' if pos.get('type', 0) == 0 else 'SELL',
                    f"{pos.get('volume', 0.0):.2f}",
                    f"{pos.get('price_open', 0.0):.5f}",
                    f"{pos.get('price_current', 0.0):.5f}",
                    f"${pos.get('profit', 0.0):.2f}"
                ))
                
        except Exception as e:
            self.dashboard.log_to_console(f"Error updating positions: {e}", "ERROR")
            
    def execute_auto_trade(self, symbol: str, analysis: Dict):
        """Execute automatic trades based on CREAM analysis"""
        try:
            if not self.dashboard.risk_mgmt_var.get():
                return  # Risk management must be enabled for auto-trading
                
            action = analysis['recommended_action']
            signal_strength = analysis['signal_strength']
            
            if action == 'TRADE' and signal_strength > 0.8:
                # Determine trade direction based on momentum
                momentum = analysis.get('momentum', {})
                direction = momentum.get('direction', 'NEUTRAL')
                
                if direction == 'BULLISH':
                    self.dashboard.log_to_console(f"AUTO-TRADE: Placing BUY order for {symbol}", "TRADE")
                    # Placeholder for actual order placement
                    # result = self.trading_engine.place_order(symbol, "BUY", 0.01)
                elif direction == 'BEARISH':
                    self.dashboard.log_to_console(f"AUTO-TRADE: Placing SELL order for {symbol}", "TRADE")
                    # Placeholder for actual order placement
                    # result = self.trading_engine.place_order(symbol, "SELL", 0.01)
                    
        except Exception as e:
            self.dashboard.log_to_console(f"Auto-trade error: {e}", "ERROR")
            
    def start_update_thread(self):
        """Start background update thread"""
        self.update_thread = threading.Thread(target=self.background_updates, daemon=True)
        self.update_thread.start()
        
    def background_updates(self):
        """Background thread for continuous updates"""
        while True:
            try:
                if self.dashboard and hasattr(self.dashboard, 'root'):
                    # Perform background updates here
                    time.sleep(1)
                else:
                    break
            except Exception as e:
                print(f"Background update error: {e}")
                time.sleep(5)
                
    def run(self):
        """Run the professional trading system"""
        if self.initialize():
            self.dashboard.log_to_console("ProQuants Professional System Ready", "SUCCESS")
            self.dashboard.run()
        else:
            print("Failed to initialize ProQuants Professional System")
            
    def shutdown(self):
        """Shutdown the system gracefully"""
        try:
            self.running = False
            if self.trading_engine:
                self.trading_engine.shutdown()
            print("ProQuants Professional System shutdown complete")
        except Exception as e:
            print(f"Shutdown error: {e}")

if __name__ == "__main__":
    # Create and run the professional system
    system = ProQuantsProfessionalSystem()
    
    try:
        system.run()
    except KeyboardInterrupt:
        print("\nShutdown requested by user...")
    except Exception as e:
        print(f"System error: {e}")
        traceback.print_exc()
    finally:
        system.shutdown()
