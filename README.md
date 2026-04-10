# Alpaca-Powered Strategy Backtester

A robust, event-driven backtesting engine built in Python. This tool integrates directly with the Alpaca Historical Data API to test technical strategies on stocks and cryptocurrencies using realistic parameters like commissions, trailing stops, and take-profits.

## 🛠️ Key Features
* **Multi-Asset Support:** Seamlessly fetch and test data for both Stocks and Crypto (BTC/USD, AAPL, etc.).
* **Diverse Strategy Library:** Includes built-in strategies for:
    * SMA Crossover & MACD
    * RSI & Bollinger Bands
    * Price Action (Pin Bar & Engulfing patterns)
    * **Confluence Strategy:** A weighted voting system combining all indicators.
* **Risk Management:** Built-in logic for trailing stop-losses and percentage-based take-profit levels.
* **Performance Analytics:** Comprehensive reporting on Sharpe Ratio, Win Rate, Profit Factor, and Max Drawdown.
* **Visual Reports:** Dual-chart output showing equity curves vs. Buy & Hold benchmarks and underwater drawdown plots.

## 🚀 Getting Started

### 1. Prerequisites
You will need an Alpaca account to fetch live historical data. Set your API keys in your environment variables:
```bash
export APCA_API_KEY_ID='your_api_key'
export APCA_API_SECRET_KEY='your_secret_key'
