# Binance Spot Demo Trading Bot

An automated algorithmic trading bot designed for the **Binance Spot Demo Trading** environment (`https://demo-api.binance.com`).

---

## Features

- **Direct Binance Demo API Integration**: Authenticates against `demo-api.binance.com` using HMAC-SHA256 with automatic server clock synchronization.
- **Multiple Strategies**:
  - `rsi_macd`: Combines Relative Strength Index (RSI) oversold/overbought conditions with MACD signal line crossovers.
  - `sma_cross`: Fast/Slow Moving Average crossover (Golden Cross / Death Cross).
- **Risk Management**:
  - Configurable Stop-Loss (`--stop-loss`) and Take-Profit (`--take-profit`).
  - Strict position sizing and capital preservation.
  - Automatic lot size and price filter precision calculation to comply with Binance exchange rules.
- **Real-Time Account Inspector**: Check live demo balances, prices, and open orders with a single command.

---

## Project Structure

```
binance trading/
├── .env                  # Live credentials and configuration (gitignored)
├── .env.example          # Sample environment configuration template
├── .gitignore            # Protects credentials, venvs, and logs
├── check_account.py      # Quick account balance & connectivity checker
├── client.py             # Binance Spot Demo REST API client
├── config.py             # Config & environment settings loader
├── risk_manager.py       # Position sizing, Stop-Loss, and Take-Profit logic
├── bot.py                # Trading engine, tick loop, and order execution
├── run_bot.py            # CLI entry point to launch the bot
└── strategies/
    ├── base.py           # Abstract strategy interface
    ├── rsi_macd.py       # RSI + MACD momentum strategy
    └── sma_cross.py      # SMA Crossover strategy
```

---

## Quick Start

### 1. Check Demo Account Balance & Live Prices
```bash
python3 check_account.py
```

### 2. Run the Bot (Live Automated Trading on Demo Account)

**Run RSI + MACD on BTC/USDT:**
```bash
python3 run_bot.py --symbol BTCUSDT --strategy rsi_macd --amount 50 --poll 10
```

**Run SMA Crossover on ETH/USDT:**
```bash
python3 run_bot.py --symbol ETHUSDT --strategy sma_cross --amount 50 --timeframe 1m
```

**Custom Risk Profile (e.g. 1% Stop Loss, 2.5% Take Profit):**
```bash
python3 run_bot.py --symbol BTCUSDT --stop-loss 1.0 --take-profit 2.5 --amount 100
```

---

## Configuration (`.env`)

```ini
BINANCE_API_KEY=your_demo_api_key
BINANCE_API_SECRET=your_demo_api_secret
BINANCE_BASE_URL=https://demo-api.binance.com
BINANCE_WS_URL=wss://demo-stream.binance.com/ws
DEFAULT_SYMBOL=BTCUSDT
DEFAULT_TIMEFRAME=1m
DEFAULT_TRADE_AMOUNT_USDT=50.0
```
