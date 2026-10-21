#!/usr/bin/env python3
"""
Live / Demo Execution Runner for DCA and Value Averaging Strategies.
Trades automatically on Binance Spot Demo Mode according to your schedule.
Maintains live state in data/bot_state.json for autonomous monitoring.
"""

import os
import sys
import time
import json
import math
import argparse
import logging
import datetime
from pathlib import Path
from client import BinanceDemoClient
from strategies.dca_va import DcaValueAveragingStrategy
from dca_analytics import fetch_fear_and_greed

BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

STATE_FILE = DATA_DIR / "bot_state.json"
LOG_FILE = LOGS_DIR / "dca_bot.log"

# Setup dual logging (stdout + file)
logger = logging.getLogger("DcaBot")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

c_handler = logging.StreamHandler(sys.stdout)
c_handler.setFormatter(formatter)
logger.addHandler(c_handler)

f_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
f_handler.setFormatter(formatter)
logger.addHandler(f_handler)

def parse_args():
    parser = argparse.ArgumentParser(description="Binance Spot Demo DCA & Value Averaging Bot")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--mode", type=str, default="dynamic_fng", choices=["pure_dca", "dynamic_fng", "value_averaging"], help="DCA Mode")
    parser.add_argument("--amount", type=float, default=50.0, help="Base buy amount in USDT (default: 50.0)")
    parser.add_argument("--interval", type=int, default=60, help="Interval between buys in seconds (default: 60s)")
    parser.add_argument("--fng-mult", type=float, default=2.0, help="Multiplier when Fear & Greed < 25 (default: 2.0)")
    parser.add_argument("--ticks", type=int, default=None, help="Stop after N execution checks")
    return parser.parse_args()

def save_state(state: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save state: {e}")

def main():
    args = parse_args()
    client = BinanceDemoClient()
    
    logger.info("=" * 65)
    logger.info("      BINANCE DEMO DCA & VALUE AVERAGING BOT STARTING")
    logger.info("=" * 65)
    logger.info(f"  PID:             {os.getpid()}")
    logger.info(f"  Symbol:          {args.symbol.upper()}")
    logger.info(f"  Strategy Mode:   {args.mode.upper()}")
    logger.info(f"  Base Amount:     ${args.amount:.2f} USDT")
    logger.info(f"  Interval:        {args.interval}s")
    logger.info(f"  F&G Multiplier:  {args.fng_mult}x (active when F&G < 25)")
    logger.info(f"  Log File:        {LOG_FILE}")
    logger.info(f"  State File:      {STATE_FILE}")
    logger.info("=" * 65)

    strat = DcaValueAveragingStrategy(
        mode=args.mode,
        base_order_usdt=args.amount,
        interval_seconds=args.interval,
        fng_multiplier=args.fng_mult,
        target_increment_usdt=args.amount
    )

    # Get symbol stepSize and minNotional
    info = client.get_exchange_info(args.symbol)
    s_info = info["symbols"][0]
    base_asset = s_info["baseAsset"]
    quote_asset = s_info["quoteAsset"]
    step_size = 0.00001
    min_notional = 5.0
    for f in s_info.get("filters", []):
        if f["filterType"] == "LOT_SIZE":
            step_size = float(f["stepSize"])
        elif f["filterType"] in ("NOTIONAL", "MIN_NOTIONAL"):
            min_notional = float(f.get("minNotional", 5.0))

    precision = max(0, int(round(-math.log10(step_size)))) if step_size > 0 else 5

    # Load existing state if available
    trades_history = []
    total_invested = 0.0
    total_coins_bought = 0.0
    
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                prev_state = json.load(f)
                if prev_state.get("symbol") == args.symbol:
                    trades_history = prev_state.get("trades", [])
                    total_invested = prev_state.get("total_invested_usdt", 0.0)
                    total_coins_bought = prev_state.get("total_coins_bought", 0.0)
        except Exception:
            pass

    ticks = 0
    while True:
        try:
            price = client.get_ticker_price(args.symbol)
            balances = client.get_balances(non_zero_only=True)
            base_bal = balances.get(base_asset, {}).get("free", 0.0)
            quote_bal = balances.get(quote_asset, {}).get("free", 0.0)

            position = {"qty": base_bal, "price": price}

            # Fetch latest Fear & Greed
            fng_map = fetch_fear_and_greed(limit=1)
            latest_fng = list(fng_map.values())[0]["value"] if fng_map else 50
            fng_class = list(fng_map.values())[0]["classification"] if fng_map else "Neutral"

            eval_res = strat.evaluate(candles=[], current_price=price, position=position, current_fng=latest_fng)
            signal = eval_res.get("signal", "HOLD")
            reason = eval_res.get("reason", "")
            
            # Cost basis & PnL calculation
            avg_cost = (total_invested / total_coins_bought) if total_coins_bought > 0 else 0.0
            curr_val = base_bal * price
            pnl_usd = (curr_val - total_invested) if total_invested > 0 else 0.0
            pnl_pct = (pnl_usd / total_invested * 100.0) if total_invested > 0 else 0.0

            if signal == "BUY":
                buy_usdt = eval_res.get("buy_amount_usdt", args.amount)
                if quote_bal < buy_usdt:
                    logger.warning(f"Insufficient {quote_asset} balance (${quote_bal:.2f}) for order of ${buy_usdt:.2f}")
                else:
                    raw_qty = buy_usdt / price
                    order_qty = round(math.floor(raw_qty / step_size) * step_size, precision)

                    if order_qty * price >= min_notional:
                        logger.info(f">>> EXECUTING BUY: {order_qty} {base_asset} (~${buy_usdt:.2f} {quote_asset}) | Reason: {reason}")
                        order = client.create_order(
                            symbol=args.symbol,
                            side="BUY",
                            order_type="MARKET",
                            quantity=order_qty
                        )
                        order_id = order.get("orderId")
                        status = order.get("status")
                        executed_qty = float(order.get("executedQty", order_qty))
                        cummulative_quote = float(order.get("cummulativeQuoteQty", buy_usdt))
                        fill_price = cummulative_quote / executed_qty if executed_qty > 0 else price

                        logger.info(f"[SUCCESS] Order filled: ID {order_id} | Status: {status} | Filled {executed_qty} @ ${fill_price:,.2f}")
                        
                        trade_entry = {
                            "order_id": order_id,
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "side": "BUY",
                            "symbol": args.symbol,
                            "qty": executed_qty,
                            "price": fill_price,
                            "cost_usdt": cummulative_quote,
                            "fng_value": latest_fng,
                            "reason": reason
                        }
                        trades_history.append(trade_entry)
                        total_invested += cummulative_quote
                        total_coins_bought += executed_qty
                        
                        # Recalculate cost basis
                        avg_cost = total_invested / total_coins_bought
                        base_bal += executed_qty
                        quote_bal -= cummulative_quote
                    else:
                        logger.warning(f"Order notional below minimum ${min_notional:.2f}")
            else:
                # Log periodically
                if ticks % 3 == 0:
                    logger.info(f"[{args.symbol} @ ${price:,.2f} | F&G: {latest_fng}] Holdings: {base_bal:.5f} {base_asset} (~${curr_val:,.2f}) | {reason}")

            # Update live state file for the monitoring agent
            state_data = {
                "status": "RUNNING",
                "pid": os.getpid(),
                "symbol": args.symbol,
                "mode": args.mode,
                "base_amount_usdt": args.amount,
                "interval_seconds": args.interval,
                "current_price": price,
                "current_fng": latest_fng,
                "fng_classification": fng_class,
                "base_asset": base_asset,
                "quote_asset": quote_asset,
                "base_balance": round(base_bal, 6),
                "quote_balance": round(quote_bal, 2),
                "total_invested_usdt": round(total_invested, 2),
                "total_coins_bought": round(total_coins_bought, 6),
                "avg_cost_basis": round(avg_cost, 2),
                "current_portfolio_value_usdt": round(curr_val, 2),
                "unrealized_pnl_usdt": round(pnl_usd, 2),
                "unrealized_pnl_pct": round(pnl_pct, 2),
                "total_trades": len(trades_history),
                "last_heartbeat": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "trades": trades_history[-10:] # Keep last 10
            }
            save_state(state_data)

        except Exception as e:
            logger.error(f"Error in DCA loop: {e}")
            if STATE_FILE.exists():
                try:
                    with open(STATE_FILE, "r") as f:
                        curr_state = json.load(f)
                    curr_state["last_error"] = str(e)
                    save_state(curr_state)
                except Exception:
                    pass

        ticks += 1
        if args.ticks and ticks >= args.ticks:
            logger.info("Reached target ticks. Exiting.")
            break
        time.sleep(10)

if __name__ == "__main__":
    main()
