#!/usr/bin/env python3
"""
Emergency & Manual Liquidation Utility.
Sells 100% of all open non-stablecoin crypto holdings at MARKET price on Binance Demo.
Updates state files and trade journal.
"""

import math
import json
import logging
import datetime
from pathlib import Path
from client import BinanceDemoClient
from learning_engine import TradeLearningEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SellAll")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "auto_trader_state.json"
DCA_STATE_FILE = DATA_DIR / "bot_state.json"

STABLECOINS = {"USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI"}

def sell_all_holdings():
    client = BinanceDemoClient()
    learner = TradeLearningEngine()
    
    balances = client.get_balances(non_zero_only=True)
    logger.info(f"Current Raw Balances: {balances}")

    crypto_holdings = {asset: bal for asset, bal in balances.items() if asset not in STABLECOINS and bal["free"] > 0}
    
    if not crypto_holdings:
        logger.info("No crypto holdings found to sell. Account is 100% cash/stablecoins.")
        return

    for asset, bal in crypto_holdings.items():
        symbol = f"{asset}USDT"
        free_qty = bal["free"]
        logger.info(f"Liquidating {free_qty} {asset} via {symbol}...")

        try:
            info = client.get_exchange_info(symbol)
            s_info = info["symbols"][0]
            step_size = 0.00001
            min_qty = 0.00001
            min_notional = 5.0
            
            for f in s_info.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    step_size = float(f["stepSize"])
                    min_qty = float(f["minQty"])
                elif f["filterType"] in ("NOTIONAL", "MIN_NOTIONAL"):
                    min_notional = float(f.get("minNotional", 5.0))

            precision = max(0, int(round(-math.log10(step_size)))) if step_size > 0 else 5
            sell_qty = round(math.floor(free_qty / step_size) * step_size, precision)
            ticker_price = client.get_ticker_price(symbol)
            est_notional = sell_qty * ticker_price

            if sell_qty < min_qty or est_notional < min_notional:
                logger.warning(f"Quantity {sell_qty} {asset} (~${est_notional:.2f}) is below minimum notional (${min_notional}). Skipping.")
                continue

            logger.info(f">>> EXECUTING MARKET SELL: {sell_qty} {asset} on {symbol} (est. value: ${est_notional:.2f} USDT)")
            order = client.create_order(
                symbol=symbol,
                side="SELL",
                order_type="MARKET",
                quantity=sell_qty
            )
            order_id = order.get("orderId")
            status = order.get("status")
            executed_qty = float(order.get("executedQty", sell_qty))
            cummulative_quote = float(order.get("cummulativeQuoteQty", est_notional))
            fill_price = cummulative_quote / executed_qty if executed_qty > 0 else ticker_price

            logger.info(f"[SUCCESS] Order #{order_id} FILLED: Sold {executed_qty} {asset} @ ${fill_price:,.2f} -> Received ${cummulative_quote:.2f} USDT")

            # Check if there was an active position in state file to record in learner
            entry_price = fill_price
            cost_usdt = cummulative_quote
            entry_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            if STATE_FILE.exists():
                try:
                    with open(STATE_FILE, "r") as f:
                        s_data = json.load(f)
                    pos = s_data.get("active_position")
                    if pos:
                        entry_price = pos.get("entry_price", fill_price)
                        cost_usdt = pos.get("cost_usdt", cummulative_quote)
                        entry_time = pos.get("entry_time", entry_time)
                except Exception:
                    pass

            learner.record_completed_trade(
                symbol=symbol,
                entry_price=entry_price,
                exit_price=fill_price,
                qty=executed_qty,
                cost_usdt=cost_usdt,
                revenue_usdt=cummulative_quote,
                entry_time=entry_time,
                exit_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                exit_reason="MANUAL_COMMAND_SELL_ALL",
                entry_indicators={},
                exit_indicators={"exit_price": fill_price}
            )

            # Clear position from auto_trader_state.json
            if STATE_FILE.exists():
                try:
                    with open(STATE_FILE, "r") as f:
                        s_data = json.load(f)
                    s_data["active_position"] = None
                    with open(STATE_FILE, "w") as f:
                        json.dump(s_data, f, indent=2)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Failed to sell {asset}: {e}")

    # Final balance check
    updated_balances = client.get_balances(non_zero_only=True)
    logger.info("=" * 60)
    logger.info("           LIQUIDATION COMPLETE: UPDATED BALANCES")
    logger.info("=" * 60)
    for a, b in updated_balances.items():
        logger.info(f"  • {a:8s} Free: {b['free']:>12.4f} | Total: {b['total']:>12.4f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    sell_all_holdings()
