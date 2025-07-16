# ProQuants Professional Trading System
## Unified AI/ML/Neural Network Platform - Pure MT5 Integration

### 🚀 Overview
ProQuants Professional is a mathematically-driven trading system that combines:
- **Original Goloji Bhudasi Trading Logic** (Proven fibonacci-based strategy)
- **Advanced Neural Networks** (Deep learning price prediction)
- **Machine Learning Models** (Pattern recognition & risk assessment)
- **Pure MT5 API Integration** (No data source conflicts)

### ✨ Key Features

#### 🧠 Artificial Intelligence Stack
- **TensorFlow/Keras Neural Networks**: LSTM models for price prediction
- **Scikit-learn ML Models**: RandomForest for pattern classification
- **Isolation Forest**: Market manipulation detection
- **Ensemble Learning**: Combined AI predictions for maximum accuracy

#### 📊 Trading Strategy
- **BOS Detection**: Break of Structure analysis
- **Fibonacci Levels**: Original Goloji Bhudasi levels (ME: 0.685, SL1: 0.76, etc.)
- **Risk Management**: Minimum 1:4 RRR, 2% max risk per trade
- **AI Enhancement**: 60% AI weight, 40% traditional analysis

#### 🔗 MT5 Integration
- **Pure MT5 API**: No Deriv API conflicts
- **Real-time Data**: 12+ hour minimum for neural training
- **Symbol Support**: V75, V25, V75(1s) indices
- **Account Integration**: .env credential management

### 🏗️ System Architecture

```
ProQuants Professional/
├── master_launcher.py          # Main system launcher
├── launch_proquants.bat        # Windows startup script
├── requirements.txt            # All dependencies
├── .env                        # MT5 credentials
├── src/
│   ├── ai/
│   │   └── unified_ai_system.py    # Core AI/ML/NN system
│   ├── strategies/
│   │   └── enhanced_cream_strategy.py  # Trading strategy
│   ├── gui/
│   │   └── professional_dashboard.py   # Professional UI
│   └── data/
│       └── deriv_mt5_manager.py        # MT5 data manager
└── logs/                       # System logs
```

### 🛠️ Installation & Setup

#### Prerequisites
- Python 3.8+
- MetaTrader 5 Terminal
- Valid Deriv account

#### Quick Start
1. **Clone/Download** the ProQuants Professional folder
2. **Configure MT5 credentials** in `.env` file:
   ```
   MT5_LOGIN=31833954
   MT5_PASSWORD=@Dmc65070*
   MT5_SERVER=Deriv-Demo
   ```
3. **Run the system**:
   ```bash
   # Windows
   launch_proquants.bat
   
   # Manual
   python master_launcher.py
   ```

#### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Start system
python master_launcher.py          # GUI mode
python master_launcher.py --headless  # Server mode
```

### 💡 Usage Modes

#### GUI Mode (Recommended)
- Professional dashboard interface
- Real-time charts and signals
- Interactive controls
- Performance monitoring

#### Headless Mode
- Server deployment
- Background processing
- Log-based monitoring
- API integration ready

### 🔍 AI Model Training

The system automatically trains AI models on startup:

#### Neural Network Training
- **Minimum Data**: 12 hours per symbol
- **Features**: 20+ technical indicators
- **Architecture**: LSTM with dropout layers
- **Targets**: Price direction, volatility, confidence

#### ML Model Training
- **Price Predictor**: RandomForest classifier
- **Volatility Forecaster**: Regression model
- **Manipulation Detector**: Isolation Forest

### 📈 Trading Signals

#### Signal Generation Process
1. **Market Structure Analysis**: BOS detection with swing points
2. **AI Prediction**: Neural network + ML ensemble
3. **Level Calculation**: Fibonacci-based entry/exit levels
4. **Validation**: Multi-layer verification system
5. **Risk Assessment**: AI-adjusted position sizing

#### Signal Components
```json
{
  "symbol": "Volatility 75 Index",
  "direction": "BULLISH",
  "entry_price": 1.23456,
  "stop_loss": 1.23000,
  "take_profit_1": 1.24000,
  "risk_reward_ratio": 4.2,
  "ai_confidence": 0.87,
  "neural_support": true,
  "manipulation_detected": false
}
```

### 🎯 Risk Management

#### Position Sizing
- **Maximum Risk**: 2% per trade
- **AI Confidence Adjustment**: 0.5x to 1.0x multiplier
- **Volatility Scaling**: Dynamic adjustment
- **Account Protection**: Maximum 10% total exposure

#### Stop Loss Management
- **Primary SL**: Fibonacci 0.76 level
- **Backup SL**: Fibonacci 0.85 level
- **Volatility Adjustment**: AI-based modifications
- **Trailing Options**: Manual implementation

### 📊 Performance Monitoring

#### Real-time Metrics
- **System Uptime**: Continuous monitoring
- **Signals Generated**: Per hour/day statistics
- **AI Accuracy**: Prediction success rates
- **Trade Performance**: Win/loss tracking

#### Logging System
- **Debug Logs**: Detailed system operations
- **Trade Logs**: Signal generation and validation
- **Error Logs**: Exception handling and recovery
- **Performance Logs**: System health metrics

### 🔧 Configuration

#### AI Model Parameters
```python
neural_config = {
    'input_features': 20,
    'hidden_layers': [64, 32, 16],
    'epochs': 100,
    'batch_size': 32
}
```

#### Trading Parameters
```python
fibonacci_levels = {
    'ME': 0.685,    # Market Entry
    'SL1': 0.76,    # Stop Loss 1
    'TP1': 0.50,    # Take Profit 1
    # ... etc
}
```

### 🚨 Troubleshooting

#### Common Issues
1. **MT5 Connection Failed**
   - Check credentials in .env file
   - Ensure MT5 terminal is running
   - Verify server connection

2. **AI Models Not Training**
   - Check data availability (12+ hours)
   - Verify symbol names in MT5
   - Review log files for errors

3. **GUI Not Starting**
   - Install tkinter: `pip install tk`
   - Check display settings
   - Try headless mode

#### Support Resources
- **Log Files**: `logs/proquants_YYYYMMDD.log`
- **System Status**: Available in GUI dashboard
- **Debug Mode**: Set logging level to DEBUG

### 🔄 Updates & Maintenance

#### Regular Maintenance
- **Model Retraining**: Automatic on startup
- **Data Validation**: Continuous quality checks
- **Performance Review**: Weekly analysis
- **Dependency Updates**: Monthly reviews

#### Version History
- **v1.0**: Initial unified system
- **v1.1**: Enhanced neural networks
- **v1.2**: Pure MT5 integration (current)

### 📝 License & Disclaimer

This software is for educational and research purposes. Trading involves significant risk of loss. Past performance does not guarantee future results. Use at your own risk.

### 🤝 Contributing

ProQuants Professional is designed for mathematical certainty and scientific trading principles. All enhancements must maintain:
- Mathematical rigor
- Scientific validation
- Risk management focus
- Pure MT5 integration

---

**ProQuants Professional**: Where AI meets proven trading logic for mathematical certainty in the markets.
