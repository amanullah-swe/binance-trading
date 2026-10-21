#!/usr/bin/env python3
"""
CLI Runner for Binance Spot Demo Trading Bot.
Usage:
  python3 run_bot.py --symbol BTCUSDT --strategy rsi_macd --amount 50
"""
import argparse
from bot import BinanceTradingBot
from strategies.rsi_macd import RsiMacdStrategy
from strategies.sma_cross import SmaCrossoverStrategy
from config import config

def parse_args():
    parser = argparse.ArgumentParser(description="Binance Spot Demo Trading Bot")
    parser.add_argument("--symbol", type=str, default=config.default_symbol, help="Trading pair symbol (default: BTCUSDT)")
    parser.add_argument("--strategy", type=str, default="rsi_macd", choices=["rsi_macd", "sma_cross"], help="Strategy to run")
    parser.add_argument("--amount", type=float, default=config.default_trade_amount_usdt, help="Trade amount in USDT (default: 50.0)")
    parser.add_argument("--timeframe", type=str, default=config.default_timeframe, help="Candle timeframe (1m, 3m, 5m, 15m, 1h)")
    parser.add_argument("--poll", type=int, default=10, help="Polling interval in seconds (default: 10)")
    parser.add_argument("--stop-loss", type=float, default=1.5, help="Stop loss percentage (default: 1.5%%)")
    parser.add_argument("--take-profit", type=float, default=3.0, help="Take profit percentage (default: 3.0%%)")
    parser.add_argument("--ticks", type=int, default=None, help="Run for N ticks and exit (useful for testing)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Instantiate Strategy
    if args.strategy == "rsi_macd":
        strat = RsiMacdStrategy()
    elif args.strategy == "sma_cross":
        strat = SmaCrossoverStrategy(fast_period=9, slow_period=21)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    print("\n" + "=" * 65)
    print("      BINANCE SPOT DEMO TRADING BOT STARTING")
    print("=" * 65)
    print(f"  Symbol:          {args.symbol.upper()}")
    print(f"  Strategy:        {strat.name}")
    print(f"  Timeframe:       {args.timeframe}")
    print(f"  Order Size:      ${args.amount:.2f} USDT")
    print(f"  Risk Profile:    Stop-Loss: -{args.stop_loss}% | Take-Profit: +{args.take_profit}%")
    print(f"  Poll Interval:   {args.poll}s")
    print("=" * 65 + "\n")

    bot = BinanceTradingBot(
        symbol=args.symbol,
        strategy=strat,
        trade_amount_usdt=args.amount,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        timeframe=args.timeframe
    )

    bot.run(poll_interval_seconds=args.poll, max_ticks=args.ticks)

if __name__ == "__main__":
    main()
