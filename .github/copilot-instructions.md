# Copilot Instructions for ProQuants Professional Trading System

## Project Overview
This is a professional-grade trading system featuring a sophisticated dashboard, MetaTrader5 integration, and advanced CREAM strategy implementation. The system is designed for expert traders requiring institutional-level tools with offline capabilities.

## Architecture & Code Organization

### Core Components
- **Dashboard** (`src/gui/professional_dashboard.py`): Professional tkinter-based GUI with dark theme
- **Trading Engine** (`src/core/trading_engine.py`): MT5 integration with offline mode support  
- **CREAM Strategy** (`src/strategies/cream_strategy.py`): Advanced technical analysis strategy
- **Main Application** (`main.py`): System launcher and component integration

### Key Design Patterns
- **Professional UI**: Dark theme with gold accents, multi-panel layout, console interface
- **Offline-First**: System operates with or without internet/MT5 connection
- **Real-Time Updates**: Background threads for live data and analysis
- **Error Resilience**: Comprehensive exception handling and fallback mechanisms

## Coding Standards & Best Practices

### Code Style
- Use descriptive variable and function names
- Follow PEP 8 conventions
- Include comprehensive docstrings for all classes and methods
- Use type hints where appropriate
- Maintain professional commenting standards

### Error Handling
- Always include try-catch blocks for external API calls (MT5)
- Provide meaningful error messages in the console
- Implement graceful degradation for offline scenarios
- Log errors appropriately with context

### Threading & Concurrency
- Use daemon threads for background operations
- Implement proper thread safety for shared resources
- Always check running state in loops
- Gracefully handle thread termination

## Domain-Specific Knowledge

### Trading System Context
- **Real-Time Focus**: This is a live trading system requiring minimal latency
- **Risk Management**: Always consider position sizing and risk limits
- **Market Hours**: Handle different market sessions and trading windows
- **Data Quality**: Validate market data before analysis

### CREAM Strategy Components
- **Clean**: Trend analysis using moving averages and ADX
- **Range**: Bollinger Bands and volatility analysis
- **Easy**: RSI and volume confirmation for entry/exit
- **Accuracy**: Historical performance and reliability metrics
- **Momentum**: MACD and momentum oscillators

### MetaTrader5 Integration
- **Account Types**: Support both demo and live accounts
- **Symbol Handling**: Work with Deriv volatility indices
- **Order Management**: Handle different order types and execution modes
- **Connection States**: Manage online/offline transitions seamlessly

## UI/UX Guidelines

### Professional Dashboard Design
- **Color Scheme**: Dark background (#0a0a0a) with accent colors (blue, green, red, gold)
- **Layout**: Multi-panel design with account, positions, market data, and console
- **Typography**: Use Segoe UI for labels, Consolas for console/data
- **Responsiveness**: Handle window resizing and different screen sizes

### Console Interface
- **Command System**: Implement intuitive commands (status, clear, help)
- **Message Types**: Use color coding for different log levels
- **Timestamps**: Include timestamps for all console messages
- **Auto-Scroll**: Keep latest messages visible

## Technical Specifications

### Dependencies & Libraries
- **MetaTrader5**: Primary trading platform integration
- **pandas/numpy**: Data manipulation and analysis
- **TA-Lib**: Technical analysis indicators
- **tkinter**: GUI framework with professional styling
- **threading**: Background operations and real-time updates

### Performance Considerations
- **Update Intervals**: Balance real-time updates with system performance
- **Memory Management**: Limit data cache size and clean up resources
- **CPU Usage**: Optimize analysis loops and background processes
- **UI Responsiveness**: Keep GUI responsive during heavy computations

## Common Development Scenarios

### Adding New Features
1. Follow the existing architectural patterns
2. Implement with offline mode compatibility
3. Add appropriate error handling and logging
4. Update the dashboard interface if needed
5. Include configuration options where applicable

### Debugging Trading Issues
1. Check MT5 connection status first
2. Verify account credentials and permissions
3. Test with offline mode to isolate issues
4. Monitor console output for error details
5. Validate market data quality and availability

### Enhancing the CREAM Strategy
1. Maintain the five-component structure
2. Use established technical analysis principles
3. Include backtesting and validation metrics
4. Consider market regime changes
5. Implement proper signal weighting

## Integration Points

### External Systems
- **MetaTrader5**: Primary data source and execution platform
- **Market Data**: Real-time price feeds and historical data
- **File System**: Configuration, logs, and data persistence
- **Threading**: Background data updates and analysis

### Internal Components
- Dashboard ↔ Trading Engine: Control commands and status updates
- Trading Engine ↔ CREAM Strategy: Market data and analysis results
- All Components ↔ Configuration: Settings and parameters
- All Components ↔ Logging: Error reporting and system monitoring

## Security & Risk Considerations

### Account Security
- Protect MT5 credentials appropriately
- Use environment variables for sensitive data
- Implement proper session management
- Log security-relevant events

### Trading Risk
- Validate all trading parameters before execution
- Implement position size limits and daily loss limits
- Provide clear risk warnings in the interface
- Monitor system performance and stability

## Maintenance & Support

### Code Maintenance
- Keep dependencies updated regularly
- Monitor performance metrics and optimize as needed
- Maintain comprehensive logging for debugging
- Document any architectural changes

### User Support
- Provide clear error messages and resolution steps
- Include helpful console commands and status information
- Maintain up-to-date documentation
- Consider user experience in all interface changes

## Future Development

### Potential Enhancements
- Additional technical analysis indicators
- Machine learning integration for signal prediction
- Multi-timeframe analysis capabilities
- Advanced risk management features
- Portfolio management and reporting

### Scalability Considerations
- Design for multiple trading accounts
- Consider distributed processing for heavy analysis
- Plan for additional trading platforms beyond MT5
- Prepare for cloud deployment scenarios

Remember: This is a professional trading system where reliability, accuracy, and user experience are paramount. Every code change should be thoroughly tested and consider the real-time nature of financial markets.
